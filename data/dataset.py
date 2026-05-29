# data/dataset.py
import torch
import numpy as np
import h5py
import hdf5plugin
from torch.utils.data import Dataset
from data.constants import DATETIME_IDX, DEMOGRAPHIC_IDX, TREATMENT_IDX, MEASUREMENT_IDX


def compute_delta_t(mask_window):
    T, V = mask_window.shape
    delta = np.zeros((T, V), dtype=np.float32)

    for v in range(V):
        last_obs = -1
        for t in range(T):
            if mask_window[t, v] == 1:
                last_obs = t
                delta[t, v] = 0.0
            else:
                delta[t, v] = (t - last_obs) if last_obs >= 0 else (t + 1)

    return delta / T

class HiRIDDataset(Dataset):
    def __init__(self, h5_path, split, context_steps=36, target_steps=12,
                 measurement_subset=None, treatment_subset=None):
        self.context_steps = context_steps
        self.target_steps  = target_steps
        self.m_idx = measurement_subset if measurement_subset is not None else list(range(len(MEASUREMENT_IDX)))
        self.t_idx = treatment_subset   if treatment_subset   is not None else list(range(len(TREATMENT_IDX)))

        f = h5py.File(h5_path, 'r')
        self.data    = f['data'][split][:]
        self.windows = f['windows'][split][:]
        self.mask    = f['mask'][split][:] if 'mask' in f else np.ones_like(self.data)
        self.delta_t_full = f['delta_t'][split][:] if 'delta_t' in f else None
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
        context_mask = self.mask[t : t + self.context_steps]
        target_mask  = self.mask[t + self.context_steps : t + self.context_steps + self.target_steps]

        m_cols = [MEASUREMENT_IDX[i] for i in self.m_idx]
        t_cols = [TREATMENT_IDX[i]   for i in self.t_idx]

        context_mask_m = context_mask[:, m_cols]
        if self.delta_t_full is not None:
            delta_t = self.delta_t_full[t : t + self.context_steps][:, m_cols] / self.context_steps
        else:
            delta_t = compute_delta_t(context_mask_m)

        return {
            'demographics': torch.tensor(context[0, DEMOGRAPHIC_IDX],  dtype=torch.float32),
            'measurements': torch.tensor(context[:, m_cols],            dtype=torch.float32),
            'treatments':   torch.tensor(context[:, t_cols],            dtype=torch.float32),
            'datetime':     torch.tensor(context[:, DATETIME_IDX],      dtype=torch.float32),
            'context_mask': torch.tensor(context_mask_m,                dtype=torch.float32),
            'delta_t':      torch.tensor(delta_t,                       dtype=torch.float32),
            'target':       torch.tensor(target[:, m_cols],             dtype=torch.float32),
            'target_mask':  torch.tensor(target_mask[:, m_cols],        dtype=torch.float32),
        }