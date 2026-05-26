import sys
sys.path.append('/home/bbe9928/thesis_work/hirid_jepa')

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os
import json

common_stage_path = '/home/bbe9928/HIRID-ICU-Benchmark/real_data_wdir/common_stage'
split_path        = '/home/bbe9928/HIRID-ICU-Benchmark/preprocessing/resources/split.tsv'
output_path       = '/home/bbe9928/thesis_work/hirid_jepa/data/scaling_stats.json'

split_df   = pd.read_csv(split_path, sep='\t')
train_pids = set(split_df[split_df['split'] == 'train']['patientid'].tolist())
print(f"Train patients: {len(train_pids)}")

files = sorted(os.listdir(common_stage_path))

# Load one file to get column names
sample         = pq.read_table(f"{common_stage_path}/{files[0]}").to_pandas()
all_columns    = sample.columns.tolist()
feature_columns = [c for c in all_columns if c != 'patientid']
print(f"Feature columns: {len(feature_columns)}")

batch_means = []
batch_stds  = []
batch_ns    = []
batch_mins  = []
batch_maxs  = []

for i, fname in enumerate(files):
    df       = pq.read_table(f"{common_stage_path}/{fname}").to_pandas()
    train_df = df[df['patientid'].isin(train_pids)].copy()
    if len(train_df) == 0:
        continue

    if 'sex' in train_df.columns:
        train_df['sex'] = train_df['sex'].map({'M': 1.0, 'F': 0.0})

    values = train_df[feature_columns].values.astype(np.float64)
    n      = (~np.isnan(values)).sum(axis=0)

    batch_ns.append(n)
    batch_means.append(np.nanmean(values, axis=0))
    batch_stds.append(np.nanstd(values,  axis=0))

    with np.errstate(all='ignore'):
        batch_mins.append(np.nanmin(values, axis=0))
        batch_maxs.append(np.nanmax(values, axis=0))

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(files)} batches done")

print("All batches processed — combining statistics...")

batch_ns    = np.array(batch_ns,    dtype=np.float64)
batch_means = np.array(batch_means, dtype=np.float64)
batch_stds  = np.array(batch_stds,  dtype=np.float64)
batch_mins  = np.array(batch_mins,  dtype=np.float64)
batch_maxs  = np.array(batch_maxs,  dtype=np.float64)

total_n     = batch_ns.sum(axis=0)
global_mean = (batch_ns * batch_means).sum(axis=0) / np.maximum(total_n, 1)
global_min  = np.nanmin(batch_mins, axis=0)
global_max  = np.nanmax(batch_maxs, axis=0)

# Combine variances using parallel variance formula
batch_vars  = batch_stds ** 2
global_var  = np.zeros(len(feature_columns))
for b in range(len(batch_ns)):
    global_var += batch_ns[b] * (batch_vars[b] + (batch_means[b] - global_mean) ** 2)
global_var /= np.maximum(total_n - 1, 1)
global_std  = np.sqrt(global_var)
global_std[global_std < 1e-8] = 1.0  # avoid division by zero for constant variables

scaling_stats = {
    'columns':   feature_columns,
    'mean':      global_mean.tolist(),
    'std':       global_std.tolist(),
    'min':       global_min.tolist(),
    'max':       global_max.tolist(),
    'obs_count': total_n.tolist(),
}

with open(output_path, 'w') as f:
    json.dump(scaling_stats, f, indent=2)

print(f"\nSaved scaling stats to {output_path}")
print(f"\n{'Variable':30s} {'mean':>10} {'std':>10} {'obs_rate':>10}")
print("-" * 65)
total_timesteps = total_n.max()
for i, col in enumerate(feature_columns[:20]):
    print(f"{col:30s} {global_mean[i]:10.3f} {global_std[i]:10.3f} {total_n[i]/total_timesteps:10.3f}")