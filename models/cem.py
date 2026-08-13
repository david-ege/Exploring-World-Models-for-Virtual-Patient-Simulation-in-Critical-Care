# cem.py
import os
import sys
sys.path.append('/home/bbe9928/thesis_work/hirid_jepa')

import joblib
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
from data.sofa import compute_sofa

with open('data/cem_treatment_percentiles.json') as percentile_json:
    PERCENTILES = json.load(percentile_json)

def to_device(data, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()}

def load_patient_data(h5_path, split, patient_idx, context_steps, device=None):
    f = h5py.File(h5_path, 'r')
    windows = f['windows'][split][:]
    start, end, pid = windows[patient_idx]

    stay_length = end - start
    t = start + min(config.CEM_START_STEP, stay_length - context_steps)
    t = max(start, t)

    context      = f['data'][split][t : t + context_steps]
    context_mask = f['mask'][split][t : t + context_steps]
    delta_t_arr  = f['delta_t'][split][t : t + context_steps] if 'delta_t' in f else None

    all_data = f['data'][split][start:end]
    all_mask = f['mask'][split][start:end]
    f.close()

    m_cols = MEASUREMENT_IDX
    t_cols = TREATMENT_IDX
    current_t = t - start

    data = {
        'measurements': torch.tensor(context[:, m_cols], dtype=torch.float32).unsqueeze(0),
        'treatments':   torch.tensor(context[:, t_cols], dtype=torch.float32).unsqueeze(0),
        'treatments_mask': torch.tensor(context_mask[:, t_cols], dtype=torch.float32).unsqueeze(0),
        'datetime':     torch.tensor(context[:, DATETIME_IDX], dtype=torch.float32).unsqueeze(0),
        'demographics': torch.tensor(context[0, DEMOGRAPHIC_IDX], dtype=torch.float32).unsqueeze(0),
        'context_mask': torch.tensor(context_mask[:, m_cols], dtype=torch.float32).unsqueeze(0),
        'delta_t': torch.tensor(
            delta_t_arr[:, m_cols] / context_steps if delta_t_arr is not None
            else np.zeros((context_steps, len(m_cols)), dtype=np.float32),
            dtype=torch.float32).unsqueeze(0),
        'pid': int(pid),
        'stay_length': int(stay_length),
        'all_treatments':      torch.tensor(all_data[:, t_cols], dtype=torch.float32).unsqueeze(0),
        'all_treatments_mask': torch.tensor(all_mask[:, t_cols], dtype=torch.float32).unsqueeze(0),
        'all_datetime':        torch.tensor(all_data[:, DATETIME_IDX], dtype=torch.float32).unsqueeze(0),
        'current_t': current_t,
        'start': start,
    }
    if device is not None:
        data = to_device(data, device)
    return data

def load_predictor(device, path=None):
    if path is None:
        path = config.get_checkpoint_path(config.BEST_CHECKPOINT)
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint['config']
    state_dict = checkpoint['model_state_dict']

    predictor = GRUPredictor(
        hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'], dropout=cfg['dropout'],
        target_steps=cfg['target_steps'], n_measurements=cfg['n_measurements'],
        n_treatments=cfg['n_treatments'],
        use_context_mask=cfg.get('uses_context_mask', False),
        use_treatment_mask=cfg.get('uses_treatment_mask', False),
        use_delta_t=cfg.get('uses_delta_t', False),
        use_pat_summary=cfg.get('uses_pat_summary', False),
    ).to(device)
    predictor.load_state_dict(state_dict)
    predictor.eval()
    return predictor, cfg

def summarize(arr, mask):
    obs_mean = np.where(mask==1, arr, 0.0).sum(axis=0) / np.maximum(mask.sum(axis=0), 1)
    obs_std  = np.sqrt(
        np.where(mask==1, (arr - obs_mean)**2, 0.0).sum(axis=0) /
        np.maximum(mask.sum(axis=0)-1, 1)
    )
    obs_rate = mask.mean(axis=0)
    return np.concatenate([obs_mean, obs_std, obs_rate])

def load_classifier(device):
    path = config.get_checkpoint_path(config.BEST_CLASSIFIER_CHECKPOINT)
    if path.endswith('.pkl'):
        lr_model = joblib.load(path)
        def classifier(measurements, datetime, demographics, context_mask=None, delta_t=None):
            pred_np = measurements.squeeze(0).cpu().numpy()
            mask_np = np.ones_like(pred_np) if context_mask is None else context_mask.squeeze(0).cpu().numpy()
            demo_np = demographics.squeeze(0).cpu().numpy()
            feat = np.concatenate([summarize(pred_np, mask_np), demo_np])
            feat = np.nan_to_num(feat, nan=0.0).reshape(1, -1)
            prob = float(lr_model.predict_proba(feat)[0, 1])
            return torch.tensor(prob, dtype=torch.float32, device=device)
        return classifier
    else:
        checkpoint = torch.load(path, map_location=device)
        model = GRUClassifier(
            hidden_dim=config.CLASSIFIER_HIDDEN_DIM, num_layers=config.CLASSIFIER_NUM_LAYERS,
            dropout=config.CLASSIFIER_DROPOUT, n_measurements=len(MEASUREMENT_IDX),
            use_context_mask=config.CLASSIFIER_USE_CONTEXT_MASK, use_delta_t=config.CLASSIFIER_USE_DELTA_T
        ).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

def create_future_treatments_vector(theta, device, template, template_mask):
    """
    template / template_mask: (1, target_steps, n_treatments) — REAL recorded treatments
    for the window being planned. Only the CEM-controlled channels get overwritten;
    every other drug is left exactly as it actually was administered.
    """
    treatments      = template.clone()
    treatment_mask  = template_mask.clone()
    n_cem           = len(CEM_TREATMENT_LOCAL_IDX)
    no_treat_option = config.CEM_NO_TREAT_OPTION_ENABLED
    treatments_flat = np.zeros(n_cem)

    for cem_i, local_idx in enumerate(CEM_TREATMENT_LOCAL_IDX):
        p1  = PERCENTILES[CEM_TREATMENT_NAMES[cem_i]]['p1']
        p99 = PERCENTILES[CEM_TREATMENT_NAMES[cem_i]]['p99']

        if no_treat_option:
            gate     = theta[cem_i]
            dose_raw = theta[n_cem + cem_i]
            val      = float(np.clip(dose_raw, p1, p99)) if gate > 0 else 0.0
        else:
            val      = float(np.clip(theta[cem_i], p1, p99))

        treatments_flat[cem_i] = val

        if local_idx in CONTINUOUS_CEM:
            treatments[0, :, local_idx]     = val
            treatment_mask[0, :, local_idx] = 1.0 if val != 0.0 else 0.0
        elif local_idx in BINARY_CEM:
            treatments[0, 0,  local_idx]    = val
            treatments[0, 1:, local_idx]    = 0.0
            treatment_mask[0, 0, local_idx] = 1.0 if val != 0.0 else 0.0
            treatment_mask[0, 1:, local_idx] = 0.0

    return treatments.to(device), treatment_mask.to(device), treatments_flat

def treatment_diff(cem_treatments, real_future_treatments, real_future_treatment_mask):
    per_variable_diff, cem_means, orig_means = [], [], []
    for cem_i, local_idx in enumerate(CEM_TREATMENT_LOCAL_IDX):
        if local_idx in CONTINUOUS_CEM:
            cem_mean = cem_treatments[0, :, local_idx].mean().item()
        else:
            cem_mean = cem_treatments[0, 0, local_idx].item()

        mask_col = real_future_treatment_mask[0, :, local_idx]
        n_obs = mask_col.sum().item()
        orig_mean = (real_future_treatments[0, :, local_idx] * mask_col).sum().item() / n_obs if n_obs > 0 else 0.0

        cem_means.append(cem_mean)
        orig_means.append(orig_mean)
        per_variable_diff.append(abs(cem_mean - orig_mean))

    per_variable_diff = np.array(per_variable_diff)
    return per_variable_diff, per_variable_diff.mean(), np.array(cem_means), np.array(orig_means)

def evaluate_policy(theta, classifier, predictor, predictor_config, data, device):
    use_context_mask   = predictor_config.get('uses_context_mask', False)
    use_treatment_mask = predictor_config.get('uses_treatment_mask', False)
    use_delta_t          = predictor_config.get('uses_delta_t', False)

    measurements = data['measurements'].to(device)
    treatments   = data['treatments'].to(device)
    datetime     = data['datetime'].to(device)
    demographics = data['demographics'].to(device)
    context_mask = data['context_mask'].to(device)
    treatment_mask = data['treatments_mask'].to(device)
    delta_t      = data['delta_t'].to(device)

    real_future_treatments     = data['real_future_treatments'].to(device)
    real_future_treatment_mask = data['real_future_treatment_mask'].to(device)
    future_datetime             = data['future_datetime'].to(device)

    future_treatments, future_treatment_mask, treatments_flat = create_future_treatments_vector(
        theta, device, real_future_treatments, real_future_treatment_mask)

    with torch.no_grad():
        new_state = predictor(
            measurements, treatments, datetime, demographics,
            future_treatments=future_treatments, future_datetime=future_datetime,
            context_mask=context_mask if use_context_mask else None,
            treatment_mask=treatment_mask if use_treatment_mask else None,
            delta_t=delta_t if use_delta_t else None)

        mortality_before = classifier(
            measurements, datetime, demographics,
            context_mask if config.CLASSIFIER_USE_CONTEXT_MASK else None,
            delta_t      if config.CLASSIFIER_USE_DELTA_T      else None)

        context_mask_predicted = torch.ones_like(new_state)
        delta_t_predicted      = torch.zeros_like(new_state)

        mortality_predicted = classifier(
            new_state, future_datetime, demographics,
            context_mask_predicted if config.CLASSIFIER_USE_CONTEXT_MASK else None,
            delta_t_predicted      if config.CLASSIFIER_USE_DELTA_T      else None)

    real_meas_np       = measurements.squeeze(0).cpu().numpy()
    real_mask_np        = context_mask.squeeze(0).cpu().numpy()
    real_treat_np        = treatments.squeeze(0).cpu().numpy()
    real_treat_mask_np   = treatment_mask.squeeze(0).cpu().numpy()
    pred_np              = new_state.squeeze(0).cpu().numpy()
    pred_mask_np          = np.ones_like(pred_np)
    future_treat_np       = future_treatments.squeeze(0).cpu().numpy()
    future_treat_mask_np  = future_treatment_mask.squeeze(0).cpu().numpy()

    sofa_before = compute_sofa(real_meas_np, real_treat_np, real_mask_np, real_treat_mask_np,
                                verbose=False)['total'] or 0
    sofa_predicted = compute_sofa(pred_np, future_treat_np, pred_mask_np, future_treat_mask_np,
                                   verbose=False)['total'] or 0

    sofa_improvement    = (sofa_before - sofa_predicted) / 24.0
    treatment_sq_dist_0 = np.mean(treatments_flat ** 2)

    reward = (mortality_before - mortality_predicted) \
           + config.CEM_GAMMA_SOFA * sofa_improvement \
           - config.CEM_GAMMA_TREATMENT_SIZE * treatment_sq_dist_0

    return (reward, mortality_predicted, mortality_before,
            new_state, future_datetime, context_mask_predicted, delta_t_predicted,
            future_treatments, future_treatment_mask,
            sofa_before, sofa_predicted)

def _advance_context(data_dict, target_steps, context_steps, new_state, context_mask_predicted,
                      delta_t_predicted, chosen_treatments, chosen_treatment_mask, new_datetime):
    """Slide the encoder's context window forward by target_steps, keeping it at a
    constant context_steps length — fixes the silent shrink-to-target_steps bug."""
    if context_steps > target_steps:
        data_dict['measurements']    = torch.cat([data_dict['measurements'][:, target_steps:], new_state], dim=1)
        data_dict['context_mask']    = torch.cat([data_dict['context_mask'][:, target_steps:], context_mask_predicted], dim=1)
        data_dict['delta_t']         = torch.cat([data_dict['delta_t'][:, target_steps:], delta_t_predicted], dim=1)
        data_dict['treatments']      = torch.cat([data_dict['treatments'][:, target_steps:], chosen_treatments], dim=1)
        data_dict['treatments_mask'] = torch.cat([data_dict['treatments_mask'][:, target_steps:], chosen_treatment_mask], dim=1)
        data_dict['datetime']        = torch.cat([data_dict['datetime'][:, target_steps:], new_datetime], dim=1)
    elif context_steps == target_steps:
        data_dict['measurements']    = new_state
        data_dict['context_mask']    = context_mask_predicted
        data_dict['delta_t']         = delta_t_predicted
        data_dict['treatments']      = chosen_treatments
        data_dict['treatments_mask'] = chosen_treatment_mask
        data_dict['datetime']        = new_datetime
    else:
        data_dict['measurements']    = new_state[:, -context_steps:]
        data_dict['context_mask']    = context_mask_predicted[:, -context_steps:]
        data_dict['delta_t']         = delta_t_predicted[:, -context_steps:]
        data_dict['treatments']      = chosen_treatments[:, -context_steps:]
        data_dict['treatments_mask'] = chosen_treatment_mask[:, -context_steps:]
        data_dict['datetime']        = new_datetime[:, -context_steps:]
    return data_dict

def simulate_baseline(initial_data, predictor, predictor_config, classifier, device):
    context_steps = predictor_config.get('context_steps', config.CONTEXT_STEPS)
    target_steps  = predictor_config['target_steps']
    use_context_mask   = predictor_config.get('uses_context_mask', False)
    use_treatment_mask = predictor_config.get('uses_treatment_mask', False)
    use_delta_t          = predictor_config.get('uses_delta_t', False)

    sim_data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in initial_data.items()}

    with torch.no_grad():
        for _ in range(config.CEM_NUM_STEPS):
            real_future_treatments     = sim_data['real_future_treatments'].to(device)
            real_future_treatment_mask = sim_data['real_future_treatment_mask'].to(device)
            future_datetime              = sim_data['future_datetime'].to(device)

            sim_state = predictor(
                sim_data['measurements'].to(device), sim_data['treatments'].to(device),
                sim_data['datetime'].to(device), sim_data['demographics'].to(device),
                future_treatments=real_future_treatments, future_datetime=future_datetime,
                context_mask=sim_data['context_mask'].to(device) if use_context_mask else None,
                treatment_mask=sim_data['treatments_mask'].to(device) if use_treatment_mask else None,
                delta_t=sim_data['delta_t'].to(device) if use_delta_t else None)

            context_mask_predicted = torch.ones_like(sim_state)
            delta_t_predicted      = torch.zeros_like(sim_state)

            sim_data = _advance_context(sim_data, target_steps, context_steps, sim_state,
                                         context_mask_predicted, delta_t_predicted,
                                         real_future_treatments, real_future_treatment_mask, future_datetime)

            sim_data['current_t'] = sim_data['current_t'] + target_steps
            next_t = sim_data['current_t'] + target_steps
            if next_t <= sim_data['stay_length']:
                sim_data['real_future_treatments']     = sim_data['all_treatments'][:, sim_data['current_t']:next_t, :]
                sim_data['real_future_treatment_mask'] = sim_data['all_treatments_mask'][:, sim_data['current_t']:next_t, :]
                sim_data['future_datetime']              = sim_data['all_datetime'][:, sim_data['current_t']:next_t]
            else:
                break

        sim_mortality = classifier(
            sim_state, future_datetime, sim_data['demographics'].to(device),
            torch.ones_like(sim_state) if config.CLASSIFIER_USE_CONTEXT_MASK else None, None
        ).item()

    sim_meas_np       = sim_state.squeeze(0).cpu().numpy()
    sim_treat_np       = real_future_treatments.squeeze(0).cpu().numpy()
    sim_mask_np         = np.ones_like(sim_meas_np)
    sim_treat_mask_np   = real_future_treatment_mask.squeeze(0).cpu().numpy()

    baseline_sofa = compute_sofa(sim_meas_np, sim_treat_np, sim_mask_np, sim_treat_mask_np, verbose=False)['total'] or 0
    return sim_mortality, baseline_sofa

def cem(patient_i, classifier, predictor, predictor_config, device, verbose=True):
    context_steps = predictor_config.get('context_steps', config.CONTEXT_STEPS)
    target_steps  = predictor_config['target_steps']

    data = load_patient_data(config.CEM_DATASET, split='test', patient_idx=patient_i,
                              context_steps=context_steps, device=device)

    # No warm-up step: propose treatments for the immediately-next window directly.
    next_t = data['current_t'] + target_steps
    data['real_future_treatments']     = data['all_treatments'][:, data['current_t']:next_t, :]
    data['real_future_treatment_mask'] = data['all_treatments_mask'][:, data['current_t']:next_t, :]
    data['future_datetime']              = data['all_datetime'][:, data['current_t']:next_t]

    with torch.no_grad():
        initial_mortality = classifier(
            data['measurements'], data['datetime'], data['demographics'],
            data['context_mask'] if config.CLASSIFIER_USE_CONTEXT_MASK else None,
            data['delta_t']      if config.CLASSIFIER_USE_DELTA_T      else None
        ).item()

    initial_cem_data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Patient {data['pid']} (index {patient_i})")
        print(f"Stay length: {data['stay_length']} timesteps ({data['stay_length']*5/60:.1f}h)")
        print(f"Initial mortality risk: {initial_mortality:.4f}")
        print(f"CEM: {config.CEM_NUM_STEPS} steps x {target_steps} timesteps "
              f"= {config.CEM_NUM_STEPS * target_steps * 5 / 60:.1f}h lookahead")
        print(f"{'='*60}")

    n_elite  = int(np.round(config.CEM_ELITE_FRAC * config.CEM_BATCH_SIZE))
    n_params = 2 * len(CEM_TREATMENT_LOCAL_IDX) if config.CEM_NO_TREAT_OPTION_ENABLED else len(CEM_TREATMENT_LOCAL_IDX)
    rewards_total = np.zeros(config.CEM_NUM_STEPS)
    all_per_variable_diff, all_cem_means, all_orig_means = [], [], []
    steps_executed = 0
    mortality_predicted = None
    sofa_before_first = sofa_predicted_last = None

    for step in range(config.CEM_NUM_STEPS):
        if verbose:
            print(f"\n--- Step {step+1}/{config.CEM_NUM_STEPS} "
                  f"(t={data['current_t']} -> t={data['current_t']+target_steps}) ---")

        theta_mean  = np.zeros(n_params)
        theta_stdev = np.ones(n_params) * config.CEM_INIT_STDEV

        for it in range(config.CEM_NUM_ITER):
            noise_multiplier = max(1.0 - it / float(config.CEM_STDEV_DECAY_TIME), 0)
            sample_std = np.sqrt(theta_stdev + np.square(config.CEM_EXTRA_STDEV) * noise_multiplier)
            thetas  = theta_mean + sample_std * np.random.randn(config.CEM_BATCH_SIZE, n_params)
            rewards = np.array([evaluate_policy(th, classifier, predictor, predictor_config, data, device)[0].item()
                                for th in thetas])
            elite_inds   = rewards.argsort()[-n_elite:]
            elite_thetas = thetas[elite_inds]
            theta_mean   = elite_thetas.mean(axis=0)
            theta_stdev  = elite_thetas.var(axis=0)
            if verbose and (it % 10 == 0 or it == config.CEM_NUM_ITER - 1):
                print(f"  Iter {it+1:3d}/{config.CEM_NUM_ITER} | Mean: {rewards.mean():.4f} | "
                      f"Max: {rewards.max():.4f} | Std: {rewards.std():.4f}")

        (reward, mortality_predicted, mortality_before, new_state, new_datetime,
         context_mask_predicted, delta_t_predicted, chosen_treatments, chosen_treatment_mask,
         sofa_before_step, sofa_predicted_step) = evaluate_policy(
            theta_mean, classifier, predictor, predictor_config, data, device)

        if step == 0:
            sofa_before_first = sofa_before_step
        sofa_predicted_last = sofa_predicted_step
        rewards_total[step] = reward.item()

        per_var_diff, step_diff_metric, step_cem_means, step_orig_means = treatment_diff(
            chosen_treatments, data['real_future_treatments'].to(device), data['real_future_treatment_mask'].to(device))
        all_per_variable_diff.append(per_var_diff.copy())
        all_cem_means.append(step_cem_means.copy())
        all_orig_means.append(step_orig_means.copy())
        steps_executed += 1

        data = _advance_context(data, target_steps, context_steps, new_state, context_mask_predicted,
                                 delta_t_predicted, chosen_treatments, chosen_treatment_mask, new_datetime)
        data['current_t'] = data['current_t'] + target_steps
        next_t = data['current_t'] + target_steps
        if next_t <= data['stay_length']:
            data['real_future_treatments']     = data['all_treatments'][:, data['current_t']:next_t, :]
            data['real_future_treatment_mask'] = data['all_treatments_mask'][:, data['current_t']:next_t, :]
            data['future_datetime']              = data['all_datetime'][:, data['current_t']:next_t]
        else:
            if verbose:
                print(f"\n  Reached end of stay at step {step+1}, stopping early")
            break

        if verbose:
            print(f"  mortality_before: {mortality_before.item():.4f} | "
                  f"mortality_predicted: {mortality_predicted.item():.4f} | "
                  f"reward: {reward.item():+.4f} | treatment_diff: {step_diff_metric:.4f}")

    cem_means_per_step  = np.array(all_cem_means)
    orig_means_per_step = np.array(all_orig_means)
    avg_per_var_diff    = np.array(all_per_variable_diff).mean(axis=0)

    f = h5py.File(config.CEM_DATASET, 'r')
    windows = f['windows']['test'][:]
    start, end, _ = windows[patient_i]
    final_t = data['current_t']
    actual_final_mortality = None
    if final_t + context_steps <= (end - start):
        actual_context = f['data']['test'][start + final_t : start + final_t + context_steps]
        actual_mask    = f['mask']['test'][start + final_t : start + final_t + context_steps]
        m_cols = MEASUREMENT_IDX
        act_meas  = torch.tensor(actual_context[:, m_cols], dtype=torch.float32).unsqueeze(0).to(device)
        act_cmask = torch.tensor(actual_mask[:, m_cols], dtype=torch.float32).unsqueeze(0).to(device)
        act_dt    = torch.tensor(actual_context[:, DATETIME_IDX], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            actual_final_mortality = classifier(
                act_meas, act_dt, data['demographics'].to(device),
                act_cmask if config.CLASSIFIER_USE_CONTEXT_MASK else None, None
            ).item()
    f.close()

    baseline_simulated_mortality, baseline_simulated_sofa = simulate_baseline(
        initial_cem_data, predictor, predictor_config, classifier, device)

    if verbose:
        print(f"\n{'='*60}\nPatient {data['pid']} — CEM complete")
        print(f"Initial mortality (t=0): {initial_mortality:.4f}")
        print(f"Final predicted mortality (CEM): {mortality_predicted.item():.4f}")
        if actual_final_mortality is not None:
            print(f"Actual measured mortality (final): {actual_final_mortality:.4f}")
        print(f"Simulated Mortality Using Baseline Treatments: {baseline_simulated_mortality:.4f}")
        print(f"{'='*60}\n")

    return {
        'patient_id': data['pid'],
        'initial_mortality': initial_mortality,
        'cem_mortality': mortality_predicted.item() if mortality_predicted is not None else None,
        'baseline_mortality': baseline_simulated_mortality,
        'actual_mortality': actual_final_mortality,
        'improvement': baseline_simulated_mortality - mortality_predicted.item() if mortality_predicted is not None else None,
        'sofa_initial_real': sofa_before_first,
        'sofa_final_predicted': sofa_predicted_last,
        'sofa_improvement': (sofa_before_first - sofa_predicted_last) if sofa_before_first is not None and sofa_predicted_last is not None else None,
        'baseline_sofa': baseline_simulated_sofa,
        'sofa_improvement_vs_baseline': (baseline_simulated_sofa - sofa_predicted_last) if sofa_predicted_last is not None else None,
        'cem_means': cem_means_per_step, 'orig_means': orig_means_per_step,
        'treatment_diff': avg_per_var_diff, 'total_reward': rewards_total.sum(),
        'steps_executed': steps_executed,
    }

def run_cem_evaluation(n_patients, classifier, predictor, predictor_config, device, config_tag='default', split='test'):
    f = h5py.File(config.CEM_DATASET, 'r')
    n_available = f['windows'][split].shape[0]
    f.close()
    patient_indices = list(range(min(n_patients, n_available)))
    all_results = []
    for i, patient_i in enumerate(patient_indices):
        print(f"\n[{i+1}/{len(patient_indices)}] Running CEM for patient index {patient_i}")
        try:
            result = cem(patient_i, classifier, predictor, predictor_config, device, verbose=False)
            result['config_tag'] = config_tag
            all_results.append(result)
        except Exception as e:
            print(f"  Skipped patient {patient_i}: {e}")

    return {
        'config_tag': config_tag, 'n_patients': len(all_results), 'results': all_results,
        'improvement':        np.array([r['improvement'] for r in all_results if r['improvement'] is not None]),
        'cem_mortality':      np.array([r['cem_mortality'] for r in all_results if r['cem_mortality'] is not None]),
        'baseline_mortality': np.array([r['baseline_mortality'] for r in all_results if r['baseline_mortality'] is not None]),
        'actual_mortality':   np.array([r['actual_mortality'] for r in all_results if r['actual_mortality'] is not None]),
        'sofa_improvement':   np.array([r['sofa_improvement'] for r in all_results if r['sofa_improvement'] is not None]),
        'cem_means':          np.array([r['cem_means'] for r in all_results]),
        'orig_means':         np.array([r['orig_means'] for r in all_results]),
        'treatment_diff':     np.array([r['treatment_diff'] for r in all_results]),
        'total_reward':       np.array([r['total_reward'] for r in all_results]),
    }

def save_evaluation(eval_results, path):
    serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in eval_results.items() if k != 'results'}
    serializable['results'] = [
        {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in r.items()}
        for r in eval_results['results']
    ]
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)

def load_evaluation(path):
    with open(path) as f:
        data = json.load(f)
    for k in ['improvement', 'cem_mortality', 'baseline_mortality', 'actual_mortality',
              'sofa_improvement', 'cem_means', 'orig_means', 'treatment_diff', 'total_reward']:
        if k in data:
            data[k] = np.array(data[k])
    return data

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor, predictor_config = load_predictor(device=device)
    classifier = load_classifier(device=device)
    cem(24, classifier=classifier, predictor=predictor, predictor_config=predictor_config, device=device)