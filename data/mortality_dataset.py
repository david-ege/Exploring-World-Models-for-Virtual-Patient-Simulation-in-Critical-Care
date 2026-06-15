# data/mortality_dataset.py
import sys
sys.path.append('/home/bbe9928/thesis_work/hirid_jepa')

import torch
import numpy as np
import json
import h5py, hdf5plugin
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

from data.constants import MEASUREMENT_IDX, TREATMENT_IDX, DEMOGRAPHIC_IDX, DATETIME_IDX

STEPS_24H = 288

class MortalityDataset(Dataset):
    def __init__(self, h5_path, split, context_steps, target_steps):

        with open('/home/bbe9928/thesis_work/hirid_jepa/data/mortality_labels.json') as f:
            mortality_labels = json.load(f)

        self.context_steps = context_steps
        self.target_steps  = target_steps
        labels_d = {int(k): v for k, v in mortality_labels[split].items()}

        f = h5py.File(h5_path, 'r')
        self.data    = f['data'][split][:]
        self.mask    = f['mask'][split][:]
        self.delta_t = f['delta_t'][split][:] if 'delta_t' in f else None
        self.windows = f['windows'][split][:]
        f.close()

        self.samples = []
        for start, end, pid in self.windows:
            pid = int(pid)
            if pid not in labels_d:
                continue
            stay_length = end - start
            if stay_length < context_steps + target_steps:
                continue
            t = start + min(STEPS_24H - context_steps, stay_length - context_steps - target_steps)
            t = max(start, t)
            self.samples.append((t, pid, labels_d[pid]))

        print(f"{split}: {len(self.samples)} patients")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t, pid, label = self.samples[idx]
        context      = self.data[t : t + self.context_steps]
        context_mask = self.mask[t : t + self.context_steps]

        m_cols = MEASUREMENT_IDX
        t_cols = TREATMENT_IDX

        if self.delta_t is not None:
            delta_t = self.delta_t[t : t + self.context_steps][:, m_cols] / self.context_steps
        else:
            delta_t = np.zeros((self.context_steps, len(m_cols)), dtype=np.float32)

        return {
            'measurements': torch.tensor(context[:, m_cols],          dtype=torch.float32),
            'treatments':   torch.tensor(context[:, t_cols],          dtype=torch.float32),
            'datetime':     torch.tensor(context[:, DATETIME_IDX],    dtype=torch.float32),
            'demographics': torch.tensor(context[0, DEMOGRAPHIC_IDX], dtype=torch.float32),
            'context_mask': torch.tensor(context_mask[:, m_cols],     dtype=torch.float32),
            'delta_t':      torch.tensor(delta_t,                     dtype=torch.float32),
            'label':        torch.tensor(label,                       dtype=torch.float32),
        }
