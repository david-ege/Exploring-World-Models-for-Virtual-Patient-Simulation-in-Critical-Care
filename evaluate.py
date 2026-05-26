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
from data.constants import MEASUREMENT_IDX, BINARY_MEASUREMENT_IDX, CONTINUOUS_MEASUREMENT_IDX, FREQUENT_MEASUREMENT_IDX, N_MEASUREMENTS, N_TREATMENTS
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

    # Handle old and new checkpoint formats
    if 'config' in checkpoint:
        cfg = checkpoint['config']
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
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

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

    all_preds, all_targets, all_last = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            measurements = batch['measurements'].to(device)
            treatments   = batch['treatments'].to(device)
            datetime     = batch['datetime'].to(device)
            demographics = batch['demographics'].to(device)
            target       = batch['target'].to(device)

            pred = model(measurements, treatments, datetime, demographics)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            last_step = measurements[:, -1:, :].repeat(1, cfg['target_steps'], 1)
            all_last.append(last_step.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    last    = np.concatenate(all_last,    axis=0)

    preds_flat   = preds.reshape(-1, preds.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])
    last_flat    = last.reshape(-1, last.shape[-1])

    # Build active measurement names and index maps
    f = h5py.File(config.DATA_PATH, 'r')
    all_columns = [c.decode('utf-8') for c in f['data']['columns'][:]]
    f.close()

    active_subset = cfg['measurement_subset'] if cfg['measurement_subset'] is not None else list(range(len(MEASUREMENT_IDX)))
    active_global_idx  = [MEASUREMENT_IDX[i] for i in active_subset]
    measurement_names  = [all_columns[i] for i in active_global_idx]
    active_continuous  = [local for local, orig in enumerate(active_subset) if orig in CONTINUOUS_MEASUREMENT_IDX]
    active_binary      = [local for local, orig in enumerate(active_subset) if orig in BINARY_MEASUREMENT_IDX]
    active_frequent    = [local for local, orig in enumerate(active_subset) if orig in FREQUENT_MEASUREMENT_IDX]

    # === Continuous MAE ===
    print("\n=== Continuous Variables (MAE) ===")
    cont_model_maes, cont_pers_maes, cont_names = [], [], []
    for i in active_continuous:
        cont_model_maes.append(mean_absolute_error(targets_flat[:, i], preds_flat[:, i]))
        cont_pers_maes.append(mean_absolute_error(targets_flat[:, i], last_flat[:, i]))
        cont_names.append(measurement_names[i])

    pers_arr  = np.array(cont_pers_maes)
    model_arr = np.array(cont_model_maes)
    cont_improvement = np.full(len(pers_arr), np.nan)
    valid = pers_arr > 1e-8
    cont_improvement[valid] = (1 - model_arr[valid] / pers_arr[valid]) * 100
    ranked = np.argsort(np.where(np.isnan(cont_improvement), -np.inf, cont_improvement))[::-1]

    print(f"Overall MAE  — model: {np.mean(cont_model_maes):.4f}, persistence: {np.mean(cont_pers_maes):.4f}")
    print(f"Mean improvement: {np.nanmean(cont_improvement):.1f}%")
    print("\nTop 5 best:")
    for i in ranked[:5]:
        print(f"  {cont_names[i]:40s} model={cont_model_maes[i]:.4f}  pers={cont_pers_maes[i]:.4f}  imp={cont_improvement[i]:.1f}%")
    print("Top 5 worst:")
    for i in ranked[-5:]:
        print(f"  {cont_names[i]:40s} model={cont_model_maes[i]:.4f}  pers={cont_pers_maes[i]:.4f}  imp={cont_improvement[i]:.1f}%")

    # === Binary AUROC ===
    print("\n=== Binary Variables (AUROC) ===")
    bin_model_aurocs, bin_pers_aurocs, bin_names = [], [], []
    for i in active_binary:
        t = targets_flat[:, i]
        if len(np.unique(t)) < 2:
            continue
        try:
            bin_model_aurocs.append(roc_auc_score(t, preds_flat[:, i]))
            bin_pers_aurocs.append(roc_auc_score(t, last_flat[:, i]))
            bin_names.append(measurement_names[i])
        except Exception:
            continue

    if bin_model_aurocs:
        print(f"Overall AUROC — model: {np.mean(bin_model_aurocs):.4f}, persistence: {np.mean(bin_pers_aurocs):.4f}")
        ranked_bin = np.argsort(bin_model_aurocs)[::-1]
        print("Top 5 best AUROC:")
        for i in ranked_bin[:5]:
            print(f"  {bin_names[i]:40s} model={bin_model_aurocs[i]:.4f}  pers={bin_pers_aurocs[i]:.4f}")

    # === Frequent Variables ===
    print("\n=== Frequent Variables (MAE) ===")
    freq_model_maes, freq_pers_maes, freq_names = [], [], []
    for local_idx in active_frequent:
        if measurement_names[local_idx] in cont_names:
            ci = cont_names.index(measurement_names[local_idx])
            freq_model_maes.append(cont_model_maes[ci])
            freq_pers_maes.append(cont_pers_maes[ci])
            freq_names.append(measurement_names[local_idx])

    if freq_model_maes:
        freq_pers_arr  = np.array(freq_pers_maes)
        freq_model_arr = np.array(freq_model_maes)
        freq_improvement = np.full(len(freq_pers_arr), np.nan)
        valid = freq_pers_arr > 1e-8
        freq_improvement[valid] = (1 - freq_model_arr[valid] / freq_pers_arr[valid]) * 100
        ranked_freq = np.argsort(np.where(np.isnan(freq_improvement), -np.inf, freq_improvement))[::-1]

        print(f"Overall MAE  — model: {np.mean(freq_model_maes):.4f}, persistence: {np.mean(freq_pers_maes):.4f}")
        print(f"Mean improvement: {np.nanmean(freq_improvement):.1f}%")
        print("Top 5 best:")
        for i in ranked_freq[:5]:
            print(f"  {freq_names[i]:40s} model={freq_model_maes[i]:.4f}  pers={freq_pers_maes[i]:.4f}  imp={freq_improvement[i]:.1f}%")
        print("Top 5 worst:")
        for i in ranked_freq[-5:]:
            print(f"  {freq_names[i]:40s} model={freq_model_maes[i]:.4f}  pers={freq_pers_maes[i]:.4f}  imp={freq_improvement[i]:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.checkpoint)