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
                use_delta_t = True, include_prev_window=False):
        self.context_steps = context_steps
        self.target_steps  = target_steps
        self.m_idx =  list(range(len(MEASUREMENT_IDX)))
        self.t_idx =  list(range(len(TREATMENT_IDX)))
        self.use_delta_t = use_delta_t
        self.include_prev_window = include_prev_window

        f = h5py.File(h5_path, 'r')
        self.data    = f['data'][split][:]
        self.windows = f['windows'][split][:]
        self.mask    = f['mask'][split][:] if 'mask' in f else np.ones_like(self.data)
        self.delta_t_full = f['delta_t'][split][:] if ('delta_t' in f and use_delta_t) else None
        f.close()

        self.samples = []
        for start, end, pid in self.windows:
            for t in range(start, end - context_steps - target_steps + 1):
                self.samples.append(t)

        if include_prev_window:
            # Map each timestep to its patient's start row
            self.timestep_to_patient_start = {}
            for start, end, pid in self.windows:
                for t in range(start, end - context_steps - target_steps + 1):
                    self.timestep_to_patient_start[t] = start

    def __len__(self):
        return len(self.samples)

    def _get_patient_start(self, t):
        return self.timestep_to_patient_start[t]

    def __getitem__(self, idx):
        t       = self.samples[idx]
        context = self.data[t : t + self.context_steps]
        target  = self.data[t + self.context_steps : t + self.context_steps + self.target_steps]
        context_mask = self.mask[t : t + self.context_steps]
        target_mask  = self.mask[t + self.context_steps : t + self.context_steps + self.target_steps]

        m_cols = [MEASUREMENT_IDX[i] for i in self.m_idx]
        t_cols = [TREATMENT_IDX[i]   for i in self.t_idx]

        context_mask_m = context_mask[:, m_cols]

        if self.use_delta_t:
            if self.delta_t_full is not None:
                delta_t = self.delta_t_full[t : t + self.context_steps][:, m_cols] / self.context_steps
            else:
                delta_t = compute_delta_t(context_mask_m)
        else:
            delta_t = np.zeros((self.context_steps, len(m_cols)), dtype=np.float32)

        out = {
            'demographics': torch.tensor(context[0, DEMOGRAPHIC_IDX],  dtype=torch.float32),
            'measurements': torch.tensor(context[:, m_cols],            dtype=torch.float32),
            'treatments':   torch.tensor(context[:, t_cols],            dtype=torch.float32),
            'datetime':     torch.tensor(context[:, DATETIME_IDX],      dtype=torch.float32),
            'context_mask': torch.tensor(context_mask_m,                dtype=torch.float32),
            'delta_t':      torch.tensor(delta_t,                       dtype=torch.float32),
            'target':       torch.tensor(target[:, m_cols],             dtype=torch.float32),
            'target_mask':  torch.tensor(target_mask[:, m_cols],        dtype=torch.float32),
        }
        if self.include_prev_window:
        # Find the start of this patient's stay
        # t is an absolute row index, need to find the patient's start
        # to know if a previous window exists
            patient_start = self._get_patient_start(t)
            t_prev = t - self.target_steps
        
            if t_prev >= patient_start:
                prev_context      = self.data[t_prev : t_prev + self.context_steps]
                prev_context_mask = self.mask[t_prev : t_prev + self.context_steps]
                
                out['prev_measurements'] = torch.tensor(
                    prev_context[:, m_cols], dtype=torch.float32)
                out['prev_context_mask'] = torch.tensor(
                    prev_context_mask[:, m_cols], dtype=torch.float32)
                out['prev_treatments']   = torch.tensor(
                    prev_context[:, t_cols], dtype=torch.float32)
                out['prev_datetime']     = torch.tensor(
                    prev_context[:, DATETIME_IDX], dtype=torch.float32)
                out['has_prev']          = torch.tensor(True)
            else:
                # First window of stay — no previous window available
                out['prev_measurements'] = torch.zeros_like(out['measurements'])
                out['prev_context_mask'] = torch.zeros_like(out['context_mask'])
                out['prev_treatments']   = torch.zeros_like(out['treatments'])
                out['prev_datetime']     = torch.zeros_like(out['datetime'])
                out['has_prev']          = torch.tensor(False)

        return out