# mortality_probe.py
import sys
sys.path.append('/home/bbe9928/thesis_work/hirid_jepa')

import torch
import numpy as np
import json
import h5py
import hdf5plugin
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import glob
import os

import thesis_work.hirid_jepa.config as config
from data.constants import MEASUREMENT_IDX, TREATMENT_IDX, DEMOGRAPHIC_IDX, DATETIME_IDX
from models.gru_predictor import GRUPredictor

# ── Load mortality labels ────────────────────────────────────────────────────
with open('/home/bbe9928/thesis_work/hirid_jepa/data/mortality_labels.json') as f:
    mortality_labels = json.load(f)

# ── Dataset that returns one hidden state per window ────────────────────────
class MortalityProbeDataset(Dataset):
    def __init__(self, h5_path, split, context_steps, mortality_labels):
        self.context_steps    = context_steps
        # When loading JSON, keys are always strings
        self.mortality_labels = {int(k): v for k, v in mortality_labels[split].items()}

        f = h5py.File(h5_path, 'r')
        self.data    = f['data'][split][:]
        self.mask    = f['mask'][split][:]
        self.delta_t = f['delta_t'][split][:] if 'delta_t' in f else None
        self.windows = f['windows'][split][:]
        f.close()

        # Steps per hour at 5-min resolution = 12
        # 24 hours = 288 steps
        self.samples = []
        STEPS_24H = 288

        for start, end, pid in self.windows:
            pid = int(pid)
            if pid not in self.mortality_labels:
                continue
            stay_length = end - start
            if stay_length < context_steps:
                continue

            t = start + min(STEPS_24H - context_steps, stay_length - context_steps)
            t = max(start, t)

            self.samples.append((t, pid, self.mortality_labels[pid]))

        print(f"{split}: {len(self.samples)} patients with mortality label")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start, pid, label = self.samples[idx]
        t = start  # use start of stay as context

        context      = self.data[t : t + self.context_steps]
        context_mask = self.mask[t : t + self.context_steps]

        m_cols = MEASUREMENT_IDX
        t_cols = TREATMENT_IDX

        if self.delta_t is not None:
            delta_t = self.delta_t[t : t + self.context_steps][:, m_cols] / self.context_steps
        else:
            delta_t = np.zeros((self.context_steps, len(m_cols)), dtype=np.float32)

        return {
            'measurements': torch.tensor(context[:, m_cols],       dtype=torch.float32),
            'treatments':   torch.tensor(context[:, t_cols],       dtype=torch.float32),
            'datetime':     torch.tensor(context[:, DATETIME_IDX], dtype=torch.float32),
            'demographics': torch.tensor(context[0, DEMOGRAPHIC_IDX], dtype=torch.float32),
            'context_mask': torch.tensor(context_mask[:, m_cols],  dtype=torch.float32),
            'delta_t':      torch.tensor(delta_t,                  dtype=torch.float32),
            'label':        torch.tensor(label,                    dtype=torch.float32),
            'pid':          pid,
        }


def extract_hidden_states(model, dataloader, device, use_context_mask, use_delta_t):
    model.eval()
    all_hidden = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            measurements = batch['measurements'].to(device)
            treatments   = batch['treatments'].to(device)
            datetime     = batch['datetime'].to(device)
            demographics = batch['demographics'].to(device)
            context_mask = batch['context_mask'].to(device)
            delta_t      = batch['delta_t'].to(device)

            # Run GRU up to hidden state — don't use output_proj
            parts = [measurements, treatments, datetime]
            if use_context_mask:
                parts.append(context_mask)
            if use_delta_t:
                parts.append(delta_t)
            x  = torch.cat(parts, dim=-1)
            h0 = model.demo_proj(demographics).unsqueeze(0).repeat(model.num_layers, 1, 1)
            _, h_n = model.gru(x, h0)
            last_hidden = h_n[-1]  # (B, hidden_dim)

            all_hidden.append(last_hidden.cpu().numpy())
            all_labels.append(batch['label'].numpy())

    return np.concatenate(all_hidden), np.concatenate(all_labels)


def run_mortality_probe(checkpoint_name=None):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    if checkpoint_name is None:
        files = glob.glob(os.path.join(config.CHECKPOINT_DIR, '*.pt'))
        checkpoint_path = max(files, key=os.path.getmtime)
    else:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, checkpoint_name)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg        = checkpoint['config']
    state_dict = checkpoint['model_state_dict']

    use_context_mask = cfg.get('uses_context_mask', False)
    use_delta_t      = cfg.get('uses_delta_t', False)

    model = GRUPredictor(
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        target_steps=cfg['target_steps'],
        encoder_dim=cfg['encoder_dim'],
        n_measurements=cfg['n_measurements'],
        n_treatments=cfg['n_treatments'],
        use_context_mask=use_context_mask,
        use_delta_t=use_delta_t
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded — hidden_dim={cfg['hidden_dim']}, "
          f"use_mask={use_context_mask}, use_delta_t={use_delta_t}")

    # Build datasets
    train_ds = MortalityProbeDataset(config.DATA_PATH, 'train',
                                     config.CONTEXT_STEPS, mortality_labels)
    val_ds   = MortalityProbeDataset(config.DATA_PATH, 'val',
                                     config.CONTEXT_STEPS, mortality_labels)
    test_ds  = MortalityProbeDataset(config.DATA_PATH, 'test',
                                     config.CONTEXT_STEPS, mortality_labels)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=4)

    # Extract hidden states
    print("\nExtracting hidden states...")
    X_train, y_train = extract_hidden_states(model, train_loader, device,
                                              use_context_mask, use_delta_t)
    X_val,   y_val   = extract_hidden_states(model, val_loader,   device,
                                              use_context_mask, use_delta_t)
    X_test,  y_test  = extract_hidden_states(model, test_loader,  device,
                                              use_context_mask, use_delta_t)

    print(f"Train hidden states: {X_train.shape}, mortality rate: {y_train.mean():.3f}")
    print(f"Val hidden states:   {X_val.shape},   mortality rate: {y_val.mean():.3f}")
    print(f"Test hidden states:  {X_test.shape},  mortality rate: {y_test.mean():.3f}")

    # Train logistic regression probe
    print("\nTraining logistic regression probe...")
    probe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'))
    ])
    probe.fit(X_train, y_train)

    # Evaluate
    val_auroc  = roc_auc_score(y_val,  probe.predict_proba(X_val)[:, 1])
    test_auroc = roc_auc_score(y_test, probe.predict_proba(X_test)[:, 1])

    print(f"\n=== Mortality Probe Results ===")
    print(f"Val  AUROC: {val_auroc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"\nTest classification report:")
    print(classification_report(y_test, probe.predict(X_test),
                                target_names=['survived', 'died']))

    # Save results
    results = {
        'checkpoint': checkpoint_path,
        'val_auroc':  val_auroc,
        'test_auroc': test_auroc,
        'n_train':    len(y_train),
        'n_val':      len(y_val),
        'n_test':     len(y_test),
        'mortality_rate_train': float(y_train.mean()),
        'mortality_rate_test':  float(y_test.mean()),
    }
    results_path = os.path.join(config.RESULTS_DIR,
                                'mortality_probe_results.json')
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    run_mortality_probe(args.checkpoint)