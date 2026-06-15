# cem.py
import os
import sys
sys.path.append('/home/bbe9928/thesis_work/hirid_jepa')

import torch
import numpy as np
import json
import h5py, hdf5plugin
from torch.utils.data import DataLoader

import config as config
from data.constants import (MEASUREMENT_IDX, TREATMENT_IDX, DEMOGRAPHIC_IDX,
                             DATETIME_IDX, CEM_TREATMENT_LOCAL_IDX, CEM_TREATMENT_NAMES, CONTINUOUS_CEM, BINARY_CEM)
from models.gru_predictor import GRUPredictor
from models.gru_classifier import GRUClassifier

def to_device(data, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()}
def load_patient_data(h5_path, split, patient_idx, device=None):
    f = h5py.File(h5_path, 'r')
    windows = f['windows'][split][:]
    start, end, pid = windows[patient_idx]

    stay_length = end - start
    t = start + min(config.CEM_START_STEP, stay_length - config.CONTEXT_STEPS)
    t = max(start, t)

    context      = f['data'][split][t : t + config.CONTEXT_STEPS]
    context_mask = f['mask'][split][t : t + config.CONTEXT_STEPS]
    delta_t_arr  = f['delta_t'][split][t : t + config.CONTEXT_STEPS] \
                   if 'delta_t' in f else None
    all_treatments = f['data'][split][start:end][:, TREATMENT_IDX]
    all_treatments_mask = f['mask'][split][start:end][:, TREATMENT_IDX]
    print(f"t={t}, context_steps={config.CONTEXT_STEPS}, end={end}, stay_length={stay_length}")

    f.close()

    m_cols = MEASUREMENT_IDX
    t_cols = TREATMENT_IDX

    data = {
        'measurements': torch.tensor(context[:, m_cols],
                                     dtype=torch.float32).unsqueeze(0),
        'treatments':   torch.tensor(context[:, t_cols],
                                     dtype=torch.float32).unsqueeze(0),
        'datetime':     torch.tensor(context[:, DATETIME_IDX],
                                     dtype=torch.float32).unsqueeze(0),
        'demographics': torch.tensor(context[0, DEMOGRAPHIC_IDX],
                                     dtype=torch.float32).unsqueeze(0),
        'context_mask': torch.tensor(context_mask[:, m_cols],
                                     dtype=torch.float32).unsqueeze(0),
        'delta_t':      torch.tensor(
                            delta_t_arr[:, m_cols] / config.CONTEXT_STEPS
                            if delta_t_arr is not None
                            else np.zeros((config.CONTEXT_STEPS, len(m_cols)),
                                          dtype=np.float32),
                            dtype=torch.float32).unsqueeze(0),
        'pid':          int(pid),
        'stay_length':   int(stay_length),
        'all_treatments': torch.tensor(all_treatments,
                                     dtype=torch.float32).unsqueeze(0),
        'all_treatments_mask': torch.tensor(all_treatments_mask, dtype=torch.float32).unsqueeze(0),
        'current_t' : t  - start                                   
    }
    if device is not None:
        data = to_device(data, device)

    return data

def load_predictor(device):
    checkpoint = torch.load(config.get_checkpoint_path(config.BEST_CHECKPOINT), map_location=device)
    checkpoint_config        = checkpoint['config']
    state_dict = checkpoint['model_state_dict']

    predictor = GRUPredictor(
        hidden_dim=checkpoint_config['hidden_dim'],
        num_layers=checkpoint_config['num_layers'],
        dropout=checkpoint_config['dropout'],
        target_steps=checkpoint_config['target_steps'],
        encoder_dim=checkpoint_config['encoder_dim'],
        n_measurements=checkpoint_config['n_measurements'],
        n_treatments=checkpoint_config['n_treatments'],
        use_context_mask=checkpoint_config.get('uses_context_mask', False),
        use_delta_t=checkpoint_config.get('uses_delta_t', False)
    ).to(device)
    predictor.load_state_dict(state_dict)
    predictor.eval()
    return predictor, checkpoint_config

def load_classifier(device):
    checkpoint = torch.load(config.get_checkpoint_path(config.BEST_CLASSIFIER_CHECKPOINT), map_location=device)
    classifier = GRUClassifier(
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        num_layers=config.CLASSIFIER_NUM_LAYERS,
        dropout=config.CLASSIFIER_DROPOUT,
        n_measurements=len(MEASUREMENT_IDX),
        use_context_mask=config.CLASSIFIER_USE_CONTEXT_MASK,
        use_delta_t=config.CLASSIFIER_USE_DELTA_T
    ).to(device)
    classifier.load_state_dict(checkpoint)
    classifier.eval()
    return classifier

def create_treatments_vector(theta, device, original_treatments):
    treatments = original_treatments.clone()
    
    for cem_i, local_idx in enumerate(CEM_TREATMENT_LOCAL_IDX):
        if local_idx in CONTINUOUS_CEM:
            treatments[0, :, local_idx] = theta[cem_i]
        elif local_idx in BINARY_CEM:
            treatments[0, 0,  local_idx] = theta[cem_i]
            treatments[0, 1:, local_idx] = 0.0
    return treatments.to(device)

def treatment_diff(cem_treatments, original_treatments, original_treatment_mask):
    per_variable_diff = []
    cem_means  = []
    orig_means = []
    for cem_i, local_idx in enumerate(CEM_TREATMENT_LOCAL_IDX):
        if local_idx in CONTINUOUS_CEM:
            cem_mean = cem_treatments[0, :, local_idx].mean().item()
        else:
            cem_mean = cem_treatments[0, 0, local_idx].item()

        mask_col = original_treatment_mask[0, :, local_idx]
        n_obs = mask_col.sum().item()
        if n_obs > 0:
            orig_mean = (original_treatments[0, :, local_idx] * mask_col).sum().item() / n_obs
        else:
            orig_mean = 0.0

        cem_means.append(cem_mean)
        orig_means.append(orig_mean)
        per_variable_diff.append(abs(cem_mean - orig_mean))

    per_variable_diff = np.array(per_variable_diff)
    summary_metric    = per_variable_diff.mean()
    return per_variable_diff, summary_metric, np.array(cem_means), np.array(orig_means)

def evaluate_policy(theta, classifier, predictor, predictor_config, data, device):
    use_context_mask = predictor_config.get('uses_context_mask', False)
    use_delta_t      = predictor_config.get('uses_delta_t', False)

    measurements = data['measurements'].to(device)
    treatments   = data['treatments'].to(device)
    datetime     = data['datetime'].to(device)
    demographics = data['demographics'].to(device)
    context_mask = data['context_mask'].to(device)
    delta_t      = data['delta_t'].to(device)

    treatments = create_treatments_vector(theta, device, treatments)
    with torch.no_grad():
        new_state = predictor(measurements, treatments, datetime, demographics,context_mask if use_context_mask else None,delta_t if use_delta_t else None)
        mortality_before = classifier(measurements, datetime, demographics, context_mask if config.CLASSIFIER_USE_CONTEXT_MASK else None,delta_t if config.CLASSIFIER_USE_DELTA_T else None)
        datetime_predicted = (datetime + (config.TARGET_STEPS / data['stay_length'])).clamp(0.0,1.0)
        context_mask_predicted = torch.ones_like(new_state)
        delta_t_predicted = torch.zeros_like(new_state) 
        mortality_predicted = classifier(new_state, datetime_predicted, demographics, context_mask_predicted if config.CLASSIFIER_USE_CONTEXT_MASK else None,delta_t_predicted if config.CLASSIFIER_USE_DELTA_T else None)

    reward = mortality_before - mortality_predicted

    return reward, mortality_predicted, mortality_before,new_state, datetime_predicted, context_mask_predicted, delta_t_predicted, treatments

#https://sesen.ai/blog/cross-entropy-method-evolution-style-rl
def cem(patient_i, classifier, predictor, predictor_config, device):
    data = load_patient_data(config.CEM_DATASET, split='test',
                             patient_idx=patient_i, device=device)

    measurements = data['measurements'].to(device)
    treatments   = data['treatments'].to(device)
    datetime     = data['datetime'].to(device)
    demographics = data['demographics'].to(device)
    context_mask = data['context_mask'].to(device)
    delta_t      = data['delta_t'].to(device)
    use_context_mask = predictor_config.get('uses_context_mask', False)
    use_delta_t      = predictor_config.get('uses_delta_t', False)

    #predict one state ahead.
    with torch.no_grad():
        initial_mortality = classifier(measurements, datetime, demographics, context_mask if config.CLASSIFIER_USE_CONTEXT_MASK else None,delta_t if config.CLASSIFIER_USE_DELTA_T else None).item()
        state_predicted = predictor(measurements, treatments, datetime, demographics,context_mask if use_context_mask else None,delta_t if use_delta_t else None)
        datetime_predicted = (datetime + (config.TARGET_STEPS / data['stay_length'])).clamp(0.0,1.0)
        context_mask_predicted = torch.ones_like(state_predicted)
        delta_t_predicted = torch.zeros_like(state_predicted)

    # Update data
    next_t   = data['current_t'] + config.TARGET_STEPS
    stay_len = data['stay_length']
    if next_t + config.CONTEXT_STEPS <= stay_len:
        next_treatments    = data['all_treatments'][
            0, next_t:next_t + config.CONTEXT_STEPS, :]
        data['treatments'] = next_treatments.unsqueeze(0).to(device)
        data['current_t']  = next_t
        data['measurements'] = state_predicted
        data['datetime']     = datetime_predicted
        data['context_mask'] = context_mask_predicted
        data['delta_t']      = delta_t_predicted
    else:
        print(f"\n  Reached end of stay at step 0, before optimizing any treatments, stopping early")
        pass


    print(f"\n{'='*60}")
    print(f"Patient {data['pid']} (index {patient_i})")
    print(f"Stay length: {data['stay_length']} timesteps "
          f"({data['stay_length']*5/60:.1f}h)")
    print(f"Initial mortality risk: {initial_mortality:.4f}")
    print(f"CEM: {config.CEM_NUM_STEPS} steps x {config.TARGET_STEPS} timesteps "
          f"= {config.CEM_NUM_STEPS * config.TARGET_STEPS * 5 / 60:.1f}h lookahead")
    print(f"{'='*60}")

    n_elite  = int(np.round(config.CEM_ELITE_FRAC * config.CEM_BATCH_SIZE))
    n_params = len(CEM_TREATMENT_LOCAL_IDX)
    rewards_total         = np.zeros(config.CEM_NUM_STEPS)
    all_per_variable_diff = np.zeros(len(CEM_TREATMENT_LOCAL_IDX))
    all_cem_means  = np.zeros(len(CEM_TREATMENT_LOCAL_IDX))
    all_orig_means = np.zeros(len(CEM_TREATMENT_LOCAL_IDX))
    steps_executed = 0
    mortality_predicted = None

    for step in range(config.CEM_NUM_STEPS):
        print(f"\n--- Step {step+1}/{config.CEM_NUM_STEPS} "
              f"(t={data['current_t']} -> "
              f"t={data['current_t'] + config.TARGET_STEPS}) ---")

        theta_mean  = np.zeros(n_params)
        theta_stdev = np.ones(n_params) * config.CEM_INIT_STDEV

        for iter in range(config.CEM_NUM_ITER):
            noise_multiplier = max(1.0 - iter / float(config.CEM_STDEV_DECAY_TIME), 0)
            sample_std = np.sqrt(theta_stdev + np.square(config.CEM_EXTRA_STDEV) * noise_multiplier)
            thetas  = theta_mean + sample_std * np.random.randn(config.CEM_BATCH_SIZE, n_params)
            rewards = np.array([evaluate_policy(th, classifier, predictor,predictor_config, data, device)[0].item()for th in thetas])

            elite_inds   = rewards.argsort()[-n_elite:]
            elite_thetas = thetas[elite_inds]
            theta_mean   = elite_thetas.mean(axis=0)
            theta_stdev  = elite_thetas.var(axis=0)

            if iter % 10 == 0 or iter == config.CEM_NUM_ITER - 1:
                print(f"  Iter {iter+1:3d}/{config.CEM_NUM_ITER} | "
                      f"Mean: {rewards.mean():.4f} | "
                      f"Max: {rewards.max():.4f} | "
                      f"Std: {rewards.std():.4f} | "
                      f"theta_mean: {theta_mean.mean():.3f} | "
                      f"theta_stdev: {theta_stdev.mean():.3f}")

        # Execute best action
        (reward, mortality_predicted, mortality_before,
         state_predicted, datetime_predicted,
         context_mask_predicted, delta_t_predicted,
         cem_treatments) = evaluate_policy(
            theta_mean, classifier, predictor, predictor_config, data, device)

        #compute difference to actual treatment
        current_mask = data['all_treatments_mask'][0, data['current_t']:data['current_t'] + config.CONTEXT_STEPS, :].unsqueeze(0).to(device)
        per_var_diff, step_diff_metric, step_cem_means, step_orig_means = treatment_diff(cem_treatments, data['treatments'].to(device), current_mask)
        all_per_variable_diff += per_var_diff
        all_cem_means  += step_cem_means
        all_orig_means += step_orig_means
        steps_executed += 1
        rewards_total[step] = reward.item()

        # Update data
        data['measurements'] = state_predicted
        data['datetime']     = datetime_predicted
        data['context_mask'] = context_mask_predicted
        data['delta_t']      = delta_t_predicted

        next_t   = data['current_t'] + config.TARGET_STEPS
        stay_len = data['stay_length']
        if next_t + config.CONTEXT_STEPS <= stay_len:
            next_treatments    = data['all_treatments'][
                0, next_t:next_t + config.CONTEXT_STEPS, :]
            data['treatments'] = next_treatments.unsqueeze(0).to(device)
            data['current_t']  = next_t
        else:
            print(f"\n  Reached end of stay at step {step+1}, stopping early")
            break


        print(f"\n  >> Step {step+1} executed:")
        print(f"     mortality_before:    {mortality_before.item():.4f}")
        print(f"     mortality_predicted: {mortality_predicted.item():.4f}")
        print(f"     reward:              {reward.item():+.4f}")
        print(f"     treatment_diff:      {step_diff_metric:.4f}")
        print(f"     per-variable diff:")
        for name, d in zip(CEM_TREATMENT_NAMES, per_var_diff):
            print(f"       {name:25s}: {d:.4f}")

    # Load actual measured state at final timestep
    f = h5py.File(config.CEM_DATASET, 'r')
    windows       = f['windows']['test'][:]
    start, end, _ = windows[patient_i]
    final_t       = data['current_t']
    actual_final_mortality = None
    if final_t + config.CONTEXT_STEPS <= (end - start):
        actual_context = f['data']['test'][
            start + final_t : start + final_t + config.CONTEXT_STEPS]
        actual_mask    = f['mask']['test'][
            start + final_t : start + final_t + config.CONTEXT_STEPS]
        m_cols  = MEASUREMENT_IDX
        act_meas  = torch.tensor(actual_context[:, m_cols],
                                  dtype=torch.float32).unsqueeze(0).to(device)
        act_cmask = torch.tensor(actual_mask[:, m_cols],
                                  dtype=torch.float32).unsqueeze(0).to(device)
        act_dt    = torch.tensor(actual_context[:, DATETIME_IDX],
                                  dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            actual_final_mortality = classifier(
                act_meas, act_dt, data['demographics'].to(device),
                act_cmask if config.CLASSIFIER_USE_CONTEXT_MASK else None,
                None
            ).item()
    f.close()

    avg_per_var_diff = all_per_variable_diff / steps_executed
    cem_means        = all_cem_means  / steps_executed
    orig_means       = all_orig_means / steps_executed

    print(f"\n{'='*60}")
    print(f"Patient {data['pid']} — CEM complete")
    print(f"Initial mortality (t=0):           {initial_mortality:.4f}")
    print(f"Final predicted mortality (CEM):   "
          f"{mortality_predicted.item():.4f}")
    if actual_final_mortality is not None:
        print(f"Actual measured mortality (final): {actual_final_mortality:.4f}")
        print(f"CEM vs actual gap:                 "
              f"{mortality_predicted.item() - actual_final_mortality:+.4f}")
    print(f"Total improvement (init->final):   "
          f"{initial_mortality - mortality_predicted.item():+.4f} "
          f"({(initial_mortality - mortality_predicted.item()) / max(initial_mortality, 1e-8) * 100:+.1f}%)")
    print(f"Per-step rewards:                  {rewards_total.round(4)}")
    print(f"Total reward:                      {rewards_total.sum():.4f}")

    print(f"\nTreatment comparison (mean scaled values over CEM horizon):")
    print(f"  {'Variable':25s} {'Actual':>10} {'CEM':>10} {'Abs Diff':>10}")
    print(f"  {'-'*55}")
    for name, orig, cem_val, diff in zip(
            CEM_TREATMENT_NAMES, orig_means, cem_means, avg_per_var_diff):
        print(f"  {name:25s} {orig:10.4f} {cem_val:10.4f} {diff:10.4f}")
    print(f"  {'-'*55}")
    print(f"  {'Overall diff metric':25s} {'':>10} {'':>10} "
          f"{avg_per_var_diff.mean():10.4f}")
    print(f"{'='*60}\n")
    
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor, predictor_config = load_predictor(device=device)
    classifier = load_classifier(device=device)

    NU_PATIENTS = 1
    for patient in range(NU_PATIENTS):
        cem(23, classifier=classifier, predictor=predictor, predictor_config=predictor_config, device=device)
        