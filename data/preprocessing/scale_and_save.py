# data/scale_and_save.py
import sys
sys.path.append('/home/bbe9928/thesis_work/hirid_jepa')

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os
import json
import h5py

common_stage_path = '/home/bbe9928/HIRID-ICU-Benchmark/real_data_wdir/common_stage'
split_path        = '/home/bbe9928/HIRID-ICU-Benchmark/preprocessing/resources/split.tsv'
stats_path        = '/home/bbe9928/thesis_work/hirid_jepa/data/scaling_stats.json'
output_path       = '/home/bbe9928/thesis_work/hirid_jepa/data/common_stage_scaled.h5'

files = sorted(os.listdir(common_stage_path))

# Load scaling stats
with open(stats_path, 'r') as f:
    stats = json.load(f)

sample          = pq.read_table(f"{common_stage_path}/{files[0]}").to_pandas()
all_columns     = sample.columns.tolist()
feature_columns = [c for c in all_columns if c != 'patientid']
mean = np.array(stats['mean'], dtype=np.float32)
std  = np.array(stats['std'],  dtype=np.float32)
vmin = np.array(stats['min'],  dtype=np.float32)
vmax = np.array(stats['max'],  dtype=np.float32)

# Identify datetime column index for min-max scaling
# datetime is not in feature_columns (we excluded it) — handle separately
# admissiontime is in feature_columns — min-max scale it
MINMAX_COLS = ['admissiontime', 'datetime']
minmax_idx  = [i for i, c in enumerate(feature_columns) if c in MINMAX_COLS]
standard_idx = [i for i in range(len(feature_columns)) if i not in minmax_idx]
print(f"Standard scaled: {len(standard_idx)} variables")
print(f"Min-max scaled:  {len(minmax_idx)} variables: {[feature_columns[i] for i in minmax_idx]}")

# Load split
split_df = pd.read_csv(split_path, sep='\t')
splits = {
    'train': set(split_df[split_df['split'] == 'train']['patientid'].tolist()),
    'val':   set(split_df[split_df['split'] == 'val']['patientid'].tolist()),
    'test':  set(split_df[split_df['split'] == 'test']['patientid'].tolist()),
}


# We'll collect data per split then write to HDF5
split_data    = {'train': [], 'val': [], 'test': []}
split_masks   = {'train': [], 'val': [], 'test': []}
split_windows = {'train': [], 'val': [], 'test': []}
split_counts  = {'train': 0,  'val': 0,  'test': 0}

def scale(values, mean, std, vmin, vmax, minmax_idx, standard_idx):
    scaled = np.zeros_like(values)
    # Standard scale
    scaled[:, standard_idx] = (values[:, standard_idx] - mean[standard_idx]) / std[standard_idx]
    # Min-max scale
    for i in minmax_idx:
        rng = vmax[i] - vmin[i]
        if rng > 1e-8:
            scaled[:, i] = (values[:, i] - vmin[i]) / rng
        else:
            scaled[:, i] = 0.0
    return scaled

for file_idx, fname in enumerate(files):
    df = pq.read_table(f"{common_stage_path}/{fname}").to_pandas()

    # Encode sex
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'M': 1.0, 'F': 0.0})

    for split_name, pids in splits.items():
        split_df_part = df[df['patientid'].isin(pids)].copy()
        if len(split_df_part) == 0:
            continue

        for pid, patient_df in split_df_part.groupby('patientid'):
            patient_df = patient_df.sort_values('datetime')
            values     = patient_df[feature_columns].values.astype(np.float32)
            mask       = (~np.isnan(values)).astype(np.float32)

            # Scale — only observed values, NaN → 0 after scaling
            scaled        = scale(values.copy(), mean, std, vmin, vmax, minmax_idx, standard_idx)
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
            scaled[mask == 0] = 0.0  # zero out unobserved after scaling
            nan_after_scale = np.isnan(scaled) | np.isinf(scaled)
            mask[nan_after_scale] = 0.0
            scaled[nan_after_scale] = 0.0

            n_rows  = len(patient_df)
            current = split_counts[split_name]

            split_data[split_name].append(scaled)
            split_masks[split_name].append(mask)
            split_windows[split_name].append([current, current + n_rows, pid])
            split_counts[split_name] += n_rows

    if (file_idx + 1) % 50 == 0:
        print(f"  {file_idx+1}/{len(files)} files processed")

print("All files processed — writing HDF5...")

with h5py.File(output_path, 'w') as out:
    # Save column names
    out.create_dataset('columns', data=np.array([c.encode('utf-8') for c in feature_columns]))

    for split_name in ['train', 'val', 'test']:
        if not split_data[split_name]:
            continue
        data_arr    = np.concatenate(split_data[split_name],  axis=0).astype(np.float32)
        mask_arr    = np.concatenate(split_masks[split_name],  axis=0).astype(np.float32)
        windows_arr = np.array(split_windows[split_name],      dtype=np.int64)

        out.create_dataset(f'data/{split_name}',    data=data_arr,    compression='lzf')
        out.create_dataset(f'mask/{split_name}',    data=mask_arr,    compression='lzf')
        out.create_dataset(f'windows/{split_name}', data=windows_arr)

        print(f"{split_name}: {len(split_windows[split_name])} patients, "
              f"{len(data_arr)} timesteps, data shape={data_arr.shape}")

    # Save scaling stats as metadata
    out.attrs['mean'] = mean
    out.attrs['std']  = std
    out.attrs['min']  = vmin
    out.attrs['max']  = vmax

print(f"\nSaved to {output_path}")