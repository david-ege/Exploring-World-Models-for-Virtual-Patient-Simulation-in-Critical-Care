# data/compute_patient_summary.py — corrected accumulation precision
import sys
sys.path.append('/home/bbe9928/thesis_work/hirid_jepa')

import h5py, hdf5plugin
import numpy as np
from data.constants import MEASUREMENT_IDX, TREATMENT_IDX
import config

def precompute_pat_summary(h5_path, splits=('train', 'val', 'test')):
    f = h5py.File(h5_path, 'r+')
    summary_cols = MEASUREMENT_IDX + TREATMENT_IDX
    n_cols = len(summary_cols)

    for split in splits:
        data    = f['data'][split][:]
        mask    = f['mask'][split][:] if 'mask' in f else np.ones_like(data)
        windows = f['windows'][split][:]
        n_rows  = data.shape[0]

        vals = data[:, summary_cols].astype(np.float64)   # float64 accumulation, not float32
        obs  = mask[:, summary_cols].astype(np.float64)
        masked_vals = vals * obs

        baseline_idx = np.zeros(n_rows, dtype=np.int64)
        for start, end, pid in windows:
            baseline_idx[start:end] = start

        zero_row = np.zeros((1, n_cols), dtype=np.float64)
        cumsum_val    = np.concatenate([zero_row, np.cumsum(masked_vals,    axis=0)], axis=0)
        cumsum_val_sq = np.concatenate([zero_row, np.cumsum(masked_vals**2, axis=0)], axis=0)
        cumsum_obs    = np.concatenate([zero_row, np.cumsum(obs,            axis=0)], axis=0)

        row_idx = np.arange(n_rows)
        n_obs   = cumsum_obs[row_idx]    - cumsum_obs[baseline_idx]
        sum_val = cumsum_val[row_idx]    - cumsum_val[baseline_idx]
        sum_sq  = cumsum_val_sq[row_idx] - cumsum_val_sq[baseline_idx]

        denom    = np.maximum(n_obs, 1)
        mean     = sum_val / denom
        variance = np.clip(sum_sq / denom - mean**2, 0.0, None)
        std      = np.sqrt(variance)
        elapsed  = np.maximum(row_idx - baseline_idx, 1).astype(np.float64)[:, None]
        obs_rate = n_obs / elapsed

        pat_summary = np.concatenate([mean, std, obs_rate], axis=1).astype(np.float32)  # downcast only here, at the end

        group = f['pat_summary'] if 'pat_summary' in f else f.create_group('pat_summary')
        if split in group:
            del group[split]
        group.create_dataset(split, data=pat_summary, **hdf5plugin.Blosc())
        print(f"{split}: wrote pat_summary {pat_summary.shape}")

    f.close()

if __name__ == '__main__':
    precompute_pat_summary(config.DATA_PATH)