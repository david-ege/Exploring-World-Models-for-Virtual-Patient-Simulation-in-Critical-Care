# train_mortality_gru.py
import sys
sys.path.append('/home/bbe9928/thesis_work/hirid_jepa')

import torch
import torch.nn as nn
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from datetime import datetime as dt

import config
from data import mortality_dataset
from models.gru_classifier import GRUClassifier
from data.constants import N_MEASUREMENTS, N_TREATMENTS

def train_gru_classifier():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

    # Load datasets ONCE
    print("Loading datasets...")
    train_ds = mortality_dataset.MortalityDataset(config.DATA_PATH, 'train',
                                config.CONTEXT_STEPS, config.TARGET_STEPS)
    val_ds   = mortality_dataset.MortalityDataset(config.DATA_PATH, 'val',
                                config.CONTEXT_STEPS, config.TARGET_STEPS)
    test_ds  = mortality_dataset.MortalityDataset(config.DATA_PATH, 'test',
                                config.CONTEXT_STEPS, config.TARGET_STEPS)

    train_loader = DataLoader(train_ds, batch_size=config.CLASSIFIER_BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Grid search params
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', action='store_true',
                        help='Run grid search over hidden_dim and dropout')
    args = parser.parse_args()

    if args.grid:
        grid = [
            {'hidden_dim': 64,  'dropout': 0.2},
            {'hidden_dim': 64,  'dropout': 0.3},
            {'hidden_dim': 32, 'dropout': 0.2},
            {'hidden_dim': 32, 'dropout': 0.3},
        ]
    else:
        grid = [{'hidden_dim': config.CLASSIFIER_HIDDEN_DIM,
                 'dropout':    config.CLASSIFIER_DROPOUT}]

    best_overall_auroc = 0.0
    best_overall_cfg   = None
    best_overall_state = None
    results_all        = []

    for cfg_run in grid:
        hidden_dim = cfg_run['hidden_dim']
        dropout    = cfg_run['dropout']
        print(f"\n=== hidden={hidden_dim} dropout={dropout} ===")

        model = GRUClassifier(
            hidden_dim=hidden_dim,
            num_layers=config.CLASSIFIER_NUM_LAYERS,
            dropout=dropout,
            n_measurements=N_MEASUREMENTS,
            use_context_mask=config.CLASSIFIER_USE_CONTEXT_MASK,
            use_delta_t=config.CLASSIFIER_USE_DELTA_T
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.CLASSIFIER_LEARNING_RATE,
                                     weight_decay=config.CLASSIFIER_WEIGHT_DECAY)
        criterion = nn.BCELoss()

        best_val_auroc = 0.0
        best_state     = None
        epochs_no_imp  = 0

        for epoch in range(config.CLASSIFIER_NUM_EPOCHS):
            # Training
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                measurements = batch['measurements'].to(device)
                datetime     = batch['datetime'].to(device)
                demographics = batch['demographics'].to(device)
                context_mask = batch['context_mask'].to(device)
                delta_t      = batch['delta_t'].to(device)
                labels       = batch['label'].to(device)

                optimizer.zero_grad()
                pred = model(measurements, datetime, demographics,
                            context_mask if config.CLASSIFIER_USE_CONTEXT_MASK else None,
                            delta_t      if config.CLASSIFIER_USE_DELTA_T      else None)
                loss = criterion(pred, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLASSIFIER_GRAD_CLIP)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    measurements = batch['measurements'].to(device)
                    datetime     = batch['datetime'].to(device)
                    demographics = batch['demographics'].to(device)
                    context_mask = batch['context_mask'].to(device)
                    delta_t      = batch['delta_t'].to(device)
                    pred = model(measurements, datetime, demographics,
                                context_mask if config.CLASSIFIER_USE_CONTEXT_MASK else None,
                                delta_t      if config.CLASSIFIER_USE_DELTA_T      else None)
                    all_preds.append(pred.cpu().numpy())
                    all_labels.append(batch['label'].numpy())

            val_auroc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_preds))
            print(f"  Epoch {epoch+1}/{config.CLASSIFIER_NUM_EPOCHS} "
                f"— loss: {train_loss:.4f}, val AUROC: {val_auroc:.4f}")

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_imp  = 0
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= config.CLASSIFIER_PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # After epoch loop — test evaluation for this config
        model.load_state_dict(best_state)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                measurements = batch['measurements'].to(device)
                datetime     = batch['datetime'].to(device)
                demographics = batch['demographics'].to(device)
                context_mask = batch['context_mask'].to(device)
                delta_t      = batch['delta_t'].to(device)
                pred = model(measurements, datetime, demographics,
                            context_mask if config.CLASSIFIER_USE_CONTEXT_MASK else None,
                            delta_t      if config.CLASSIFIER_USE_DELTA_T      else None)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch['label'].numpy())
        run_test_auroc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_preds))
        results_all.append({**cfg_run, 'val_auroc': best_val_auroc, 'test_auroc': run_test_auroc})

        if best_val_auroc > best_overall_auroc:
            best_overall_auroc = best_val_auroc
            best_overall_cfg   = cfg_run
            best_overall_state = best_state

    # Evaluate best model on test set
    print(f"\n=== Best config: {best_overall_cfg} ===")
    print(f"Best val AUROC: {best_overall_auroc:.4f}")

    model = GRUClassifier(
        hidden_dim=best_overall_cfg['hidden_dim'],
        num_layers=config.CLASSIFIER_NUM_LAYERS,
        dropout=best_overall_cfg['dropout'],
        n_measurements=N_MEASUREMENTS,
        use_context_mask=config.CLASSIFIER_USE_CONTEXT_MASK,
        use_delta_t=config.CLASSIFIER_USE_DELTA_T
    ).to(device)
    model.load_state_dict(best_overall_state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            measurements = batch['measurements'].to(device)
            treatments   = batch['treatments'].to(device)
            datetime     = batch['datetime'].to(device)
            demographics = batch['demographics'].to(device)
            context_mask = batch['context_mask'].to(device)
            delta_t      = batch['delta_t'].to(device)
            pred = model(measurements, datetime, demographics,
             context_mask if config.CLASSIFIER_USE_CONTEXT_MASK else None,
             delta_t      if config.CLASSIFIER_USE_DELTA_T      else None)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch['label'].numpy())

    test_auroc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_preds))
    print(f"Test AUROC: {test_auroc:.4f}")

    # Print grid search summary
    print("\n=== Grid Search Summary ===")
    for r in sorted(results_all, key=lambda x: x['val_auroc'], reverse=True):
        print(f"  hidden={r['hidden_dim']} dropout={r['dropout']} "
            f"val={r['val_auroc']:.4f} test={r['test_auroc']:.4f}")

    # Save best model
    date_str        = dt.now().strftime("%d_%m_%H-%M")
    checkpoint_name = (f"gru_classifier_h{best_overall_cfg['hidden_dim']}"
                       f"_do{best_overall_cfg['dropout']}_{date_str}.pt")
    torch.save(best_overall_state,
               os.path.join(config.CHECKPOINT_DIR, checkpoint_name))
    print(f"Saved: {checkpoint_name}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, 'gru_classifier_results.json'), 'w') as f:
        json.dump({
            'best_config':    best_overall_cfg,
            'best_val_auroc': best_overall_auroc,
            'test_auroc':     test_auroc,
            'grid_results':   results_all,
        }, f, indent=2)

if __name__ == '__main__':
    train_gru_classifier()