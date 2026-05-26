# data/dataset.py

import torch
import numpy as np
import h5py
import hdf5plugin
from torch.utils.data import Dataset
from data.constants import DATETIME_IDX, DEMOGRAPHIC_IDX, TREATMENT_IDX, MEASUREMENT_IDX


class HiRIDDataset(Dataset):
    def __init__(self, h5_path, split, context_steps=24, target_steps=12,
                 measurement_subset=None, treatment_subset=None):
        self.context_steps = context_steps
        self.target_steps  = target_steps
        
        # Which indices to use
        self.m_idx = measurement_subset if measurement_subset is not None else list(range(len(MEASUREMENT_IDX)))
        self.t_idx = treatment_subset   if treatment_subset   is not None else list(range(len(TREATMENT_IDX)))

        f = h5py.File(h5_path, 'r')
        self.data    = f['data'][split][:]
        self.windows = f['patient_windows'][split][:]
        f.close()

        self.samples = []
        for start, end, pid in self.windows:
            for t in range(start, end - context_steps - target_steps + 1):
                self.samples.append(t)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        t       = self.samples[idx]
        context = self.data[t : t + self.context_steps]
        target  = self.data[t + self.context_steps : t + self.context_steps + self.target_steps]

        # Apply subset selection after slicing to measurement/treatment columns
        m_cols = [MEASUREMENT_IDX[i] for i in self.m_idx]
        t_cols = [TREATMENT_IDX[i]   for i in self.t_idx]

        return {
            'demographics': torch.tensor(context[0, DEMOGRAPHIC_IDX],  dtype=torch.float32),
            'measurements': torch.tensor(context[:, m_cols],            dtype=torch.float32),
            'treatments':   torch.tensor(context[:, t_cols],            dtype=torch.float32),
            'datetime':     torch.tensor(context[:, DATETIME_IDX],      dtype=torch.float32),
            'target':       torch.tensor(target[:,  m_cols],            dtype=torch.float32),
        }