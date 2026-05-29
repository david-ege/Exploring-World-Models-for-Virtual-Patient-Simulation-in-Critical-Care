# evaluate.py
import sys
sys.path.append('/home/bbe9928/thesis_work/hirid_jepa')

import os
import glob
import argparse
import torch
import numpy as np
import h5py
import hdf5plugin
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, mean_absolute_error

import config
from data.dataset import HiRIDDataset
from data.constants import MEASUREMENT_IDX, N_MEASUREMENTS, N_TREATMENTS
from models.gru_predictor import GRUPredictor


def get_checkpoint(checkpoint_dir, name=None):
    if name is not None:
        return os.path.join(checkpoint_dir, name)
    files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(files, key=os.path.getmtime)


def evaluate(checkpoint_name=None):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

    checkpoint_path = get_checkpoint(config.CHECKPOINT_DIR, checkpoint_name)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    #trying to make evaluation pipeline compatible with old models
    if 'config' in checkpoint:
        cfg        = checkpoint['config']
        state_dict = checkpoint['model_state_dict']
    else:
        print("Warning: old checkpoint format, using config.py values")
        state_dict = checkpoint
        n_m = len(config.MEASUREMENT_SUBSET) if config.MEASUREMENT_SUBSET else N_MEASUREMENTS
        n_t = len(config.TREATMENT_SUBSET)   if config.TREATMENT_SUBSET   else N_TREATMENTS
        cfg = {
            'hidden_dim':         config.HIDDEN_DIM,
            'num_layers':         config.NUM_LAYERS,
            'dropout':            config.DROPOUT,
            'target_steps':       config.TARGET_STEPS,
            'encoder_dim':        config.ENCODER_DIM,
            'n_measurements':     n_m,
            'n_treatments':       n_t,
            'measurement_subset': config.MEASUREMENT_SUBSET,
            'treatment_subset':   config.TREATMENT_SUBSET,
        }

    test_dataset = HiRIDDataset(
        config.DATA_PATH, 'test',
        config.CONTEXT_STEPS, cfg['target_steps'],
        measurement_subset=cfg['measurement_subset'],
        treatment_subset=cfg['treatment_subset']
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=4)

    model = GRUPredictor(
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        target_steps=cfg['target_steps'],
        encoder_dim=cfg['encoder_dim'],
        n_measurements=cfg['n_measurements'],
        n_treatments=cfg['n_treatments']
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_targets, all_last, all_masks = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            measurements = batch['measurements'].to(device)
            treatments   = batch['treatments'].to(device)
            datetime     = batch['datetime'].to(device)
            demographics = batch['demographics'].to(device)
            target       = batch['target'].to(device)
            target_mask  = batch['target_mask'].to(device)
            context_mask = batch['context_mask'].to(device)
            delta_t      = batch['delta_t'].to(device)

            uses_context_mask = cfg.get('uses_context_mask', False)
            uses_delta_t      = cfg.get('uses_delta_t', False)

            if uses_delta_t:
                pred = model(measurements, treatments, datetime, demographics,
                            context_mask, delta_t)
            elif uses_context_mask:
                pred = model(measurements, treatments, datetime, demographics,
                            context_mask)
            else:
                pred = model(measurements, treatments, datetime, demographics)

            last_step = measurements[:, -1:, :].repeat(1, cfg['target_steps'], 1)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_last.append(last_step.cpu().numpy())
            all_masks.append(target_mask.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    last    = np.concatenate(all_last,    axis=0)
    masks   = np.concatenate(all_masks,   axis=0)

    # Flatten: (N, target_steps, n_vars) -> (N*target_steps, n_vars)
    preds_flat   = preds.reshape(-1,   preds.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])
    last_flat    = last.reshape(-1,    last.shape[-1])
    masks_flat   = masks.reshape(-1,   masks.shape[-1])

    # Load column names
    f = h5py.File(config.DATA_PATH, 'r')
    all_columns = [c.decode('utf-8') for c in f['columns'][:]]
    f.close()

    active_subset      = cfg['measurement_subset'] if cfg['measurement_subset'] is not None else list(range(len(MEASUREMENT_IDX)))
    active_global_idx  = [MEASUREMENT_IDX[i] for i in active_subset]
    measurement_names  = [all_columns[i] for i in active_global_idx]

    # === Per-variable MAE on OBSERVED timesteps only (supervisor requirement) ===
    print("\n=== Per-Variable MAE (observed timesteps only) ===")
    model_maes = []
    pers_maes  = []
    names      = []
    obs_counts = []

    for i, name in enumerate(measurement_names):
        obs_idx = masks_flat[:, i] == 1
        n_obs   = obs_idx.sum()


        m_mae = mean_absolute_error(targets_flat[obs_idx, i], preds_flat[obs_idx, i])
        p_mae = mean_absolute_error(targets_flat[obs_idx, i], last_flat[obs_idx, i])

        if n_obs < 10 or p_mae < 1e-6:  # skip variables with too few observations
            continue

        model_maes.append(m_mae)
        pers_maes.append(p_mae)
        names.append(name)
        obs_counts.append(n_obs)

    model_maes = np.array(model_maes)
    pers_maes  = np.array(pers_maes)
    improvement = np.full(len(model_maes), np.nan)
    valid = pers_maes > 1e-8
    improvement[valid] = (1 - model_maes[valid] / pers_maes[valid]) * 100
    ranked = np.argsort(np.where(np.isnan(improvement), -np.inf, improvement))[::-1]

    print(f"Variables evaluated: {len(names)} (skipped variables with <10 observations)")
    print(f"Overall MAE  — model: {model_maes.mean():.4f}, persistence: {pers_maes.mean():.4f}")
    print(f"Mean improvement: {np.nanmean(improvement):.1f}%")

    print("\nTop 10 best:")
    for i in ranked[:10]:
        print(f"  {names[i]:35s} model={model_maes[i]:.4f}  pers={pers_maes[i]:.4f}  "
              f"imp={improvement[i]:.1f}%  n={obs_counts[i]}")

    print("\nTop 10 worst:")
    for i in ranked[-10:]:
        print(f"  {names[i]:35s} model={model_maes[i]:.4f}  pers={pers_maes[i]:.4f}  "
              f"imp={improvement[i]:.1f}%  n={obs_counts[i]}")

    # === Overall summary ===
    print(f"\n=== Summary ===")
    print(f"Variables beating persistence: {(improvement > 0).sum()} / {len(names)}")
    print(f"Best variable:  {names[ranked[0]]} ({improvement[ranked[0]]:.1f}% improvement)")
    print(f"Worst variable: {names[ranked[-1]]} ({improvement[ranked[-1]]:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.checkpoint)