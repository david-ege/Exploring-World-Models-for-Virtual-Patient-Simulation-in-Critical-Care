# train.py
import sys
sys.path.append('/home/bbe9928/thesis_work/hirid_jepa')

import os
import argparse
from datetime import datetime as dt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data.dataset import HiRIDDataset
from data.constants import N_MEASUREMENTS, N_TREATMENTS
from models.gru_predictor import GRUPredictor
from evaluate import evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context',  type=int,   default=None)
    parser.add_argument('--target',   type=int,   default=None)
    parser.add_argument('--hidden',   type=int,   default=None)
    parser.add_argument('--layers',   type=int,   default=None)
    parser.add_argument('--dropout',  type=float, default=None)
    parser.add_argument('--lr',       type=float, default=None)
    parser.add_argument('--batch',    type=int,   default=None)
    parser.add_argument('--epochs',   type=int,   default=None)
    return parser.parse_args()

def masked_mse(pred, target, mask):
    """MSE only on observed target values."""
    loss = (pred - target) ** 2 * mask
    n    = mask.sum().clamp(min=1)
    return loss.sum() / n

def get_lr(epoch):
    if epoch < 5:
        return (epoch + 1) / 5
    return 1.0

def train(override_cfg={}):
    args = parse_args()

    context_steps = override_cfg.get('context_steps', args.context  or config.CONTEXT_STEPS)
    target_steps  = override_cfg.get('target_steps',  args.target   or config.TARGET_STEPS)
    hidden_dim    = override_cfg.get('hidden_dim',    args.hidden   or config.HIDDEN_DIM)
    num_layers    = override_cfg.get('num_layers',    args.layers   or config.NUM_LAYERS)
    dropout       = override_cfg.get('dropout',       args.dropout  or config.DROPOUT)
    learning_rate = override_cfg.get('learning_rate', args.lr       or config.LEARNING_RATE)
    batch_size    = override_cfg.get('batch_size',    args.batch    or config.BATCH_SIZE)
    num_epochs    = override_cfg.get('num_epochs',    args.epochs   or config.NUM_EPOCHS)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Config: context={context_steps}, target={target_steps}, hidden={hidden_dim}, "
          f"layers={num_layers}, dropout={dropout}, lr={learning_rate}, batch={batch_size}")

    date_str        = dt.now().strftime("%d_%m_%H-%M")
    checkpoint_name = f"gru_ctx{context_steps}_tgt{target_steps}_h{hidden_dim}_l{num_layers}_{date_str}.pt"

    train_dataset = HiRIDDataset(config.DATA_PATH, 'train', context_steps, target_steps,
                                 measurement_subset=config.MEASUREMENT_SUBSET,
                                 treatment_subset=config.TREATMENT_SUBSET)
    val_dataset   = HiRIDDataset(config.DATA_PATH, 'val',   context_steps, target_steps,
                                 measurement_subset=config.MEASUREMENT_SUBSET,
                                 treatment_subset=config.TREATMENT_SUBSET)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    n_measurements = len(config.MEASUREMENT_SUBSET) if config.MEASUREMENT_SUBSET else N_MEASUREMENTS
    n_treatments   = len(config.TREATMENT_SUBSET)   if config.TREATMENT_SUBSET   else N_TREATMENTS

    model = GRUPredictor(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        target_steps=target_steps,
        encoder_dim=config.ENCODER_DIM,
        n_measurements=n_measurements,
        n_treatments=n_treatments
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    best_val_loss             = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            measurements = batch['measurements'].to(device)
            treatments   = batch['treatments'].to(device)
            datetime     = batch['datetime'].to(device)
            demographics = batch['demographics'].to(device)
            target       = batch['target'].to(device)
            target_mask  = batch['target_mask'].to(device)

            optimizer.zero_grad()
            pred = model(measurements, treatments, datetime, demographics)
            loss = masked_mse(pred, target, target_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                measurements = batch['measurements'].to(device)
                treatments   = batch['treatments'].to(device)
                datetime     = batch['datetime'].to(device)
                demographics = batch['demographics'].to(device)
                target       = batch['target'].to(device)
                target_mask  = batch['target_mask'].to(device)

                pred      = model(measurements, treatments, datetime, demographics)
                val_loss += masked_mse(pred, target, target_mask).item()

        val_loss /= len(val_loader)
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} — train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss              = val_loss
            epochs_without_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'hidden_dim':         hidden_dim,
                    'num_layers':         num_layers,
                    'dropout':            dropout,
                    'target_steps':       target_steps,
                    'encoder_dim':        config.ENCODER_DIM,
                    'n_measurements':     n_measurements,
                    'n_treatments':       n_treatments,
                    'measurement_subset': config.MEASUREMENT_SUBSET,
                    'treatment_subset':   config.TREATMENT_SUBSET,
                }
            }, f"{config.CHECKPOINT_DIR}/{checkpoint_name}")
            print(f" Saved: {checkpoint_name}")
        else:
            epochs_without_improvement += 1
            print(f" No improvement ({epochs_without_improvement}/{config.PATIENCE})")
            if epochs_without_improvement >= config.PATIENCE:
                print(f" Early stopping at epoch {epoch+1}")
                break

    print(f"\n--- Training complete. Best val loss: {best_val_loss:.4f} ---")
    print(f"--- Running evaluation on: {checkpoint_name} ---\n")
    evaluate(checkpoint_name)
    return best_val_loss

if __name__ == '__main__':
    train()