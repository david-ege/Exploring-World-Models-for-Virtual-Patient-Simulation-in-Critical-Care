# train_mortality_gru.py
import sys
sys.path.append('/home/bbe9928/thesis_work/hirid_jepa')

import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import os
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from datetime import datetime as dt

import config
from data import mortality_dataset
from models.gru_classifier import GRUClassifier
from models.gru_predictor import GRUPredictor
from data.constants import N_MEASUREMENTS, N_TREATMENTS, MEASUREMENT_IDX


def load_predictor(device):
    checkpoint = torch.load(config.get_checkpoint_path(), map_location=device)
    cfg        = checkpoint['config']
    model = GRUPredictor(
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        target_steps=cfg['target_steps'],
        encoder_dim=cfg['encoder_dim'],
        n_measurements=cfg['n_measurements'],
        n_treatments=cfg['n_treatments'],
        use_context_mask=cfg.get('uses_context_mask', False),
        use_delta_t=cfg.get('uses_delta_t', False)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, cfg


def get_classifier_input(batch, data_mode, predictor, pred_cfg, device):
    """
    Returns (measurements, context_mask, datetime, demographics, delta_t)
    """
    use_context_mask = pred_cfg.get('uses_context_mask', False) if pred_cfg else False
    use_delta_t      = pred_cfg.get('uses_delta_t', False)      if pred_cfg else False

    datetime     = batch['datetime'].to(device)
    demographics = batch['demographics'].to(device)
    delta_t      = batch['delta_t'].to(device)

    if data_mode == 'real':
        return (batch['measurements'].to(device),
                batch['context_mask'].to(device),
                datetime, demographics, delta_t)

    with torch.no_grad():
        predicted = predictor(
            batch['prev_measurements'].to(device),
            batch['prev_treatments'].to(device),
            batch['prev_datetime'].to(device),
            demographics,
            batch['prev_context_mask'].to(device) if use_context_mask else None,
            batch['prev_delta_t'].to(device)      if use_delta_t      else None
        )
    pred_mask = torch.ones_like(predicted)

    if data_mode == 'predicted':
        return predicted, pred_mask, datetime, demographics, delta_t

    # 'both' — double everything along batch dimension
    return (torch.cat([batch['measurements'].to(device), predicted],        dim=0),
            torch.cat([batch['context_mask'].to(device), pred_mask],        dim=0),
            torch.cat([datetime,     datetime],     dim=0),
            torch.cat([demographics, demographics], dim=0),
            torch.cat([delta_t,      delta_t],      dim=0))

def train_gru_classifier():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--data_mode', type=str, default='real',
                        choices=['real', 'predicted', 'both'],
                        help='Train on real, predicted, or both')
    args = parser.parse_args()

    need_prev = args.data_mode in ['predicted', 'both']

    # Load predictor if needed
    predictor = pred_cfg = None
    if need_prev:
        print(f"Loading predictor for data_mode='{args.data_mode}'...")
        predictor, pred_cfg = load_predictor(device)

    print("Loading datasets...")
    train_ds = mortality_dataset.MortalityDataset(
        config.DATA_PATH, 'train', config.CONTEXT_STEPS, config.TARGET_STEPS,
        include_prev_window=need_prev)
    val_ds   = mortality_dataset.MortalityDataset(
        config.DATA_PATH, 'val',   config.CONTEXT_STEPS, config.TARGET_STEPS,
        include_prev_window=need_prev)
    test_ds  = mortality_dataset.MortalityDataset(
        config.DATA_PATH, 'test',  config.CONTEXT_STEPS, config.TARGET_STEPS,
        include_prev_window=need_prev)

    train_loader = DataLoader(train_ds, batch_size=config.CLASSIFIER_BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128,
                              shuffle=False, num_workers=4, pin_memory=True)

    if args.grid:
        grid = [
            {'hidden_dim': 64, 'dropout': 0.2},
            {'hidden_dim': 64, 'dropout': 0.3},
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
        print(f"\n=== hidden={hidden_dim} dropout={dropout} "
              f"data_mode={args.data_mode} ===")

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
                measurements, context_mask, datetime, demographics, delta_t = get_classifier_input(
                    batch, args.data_mode, predictor, pred_cfg, device)

                labels       = batch['label'].to(device)

                # For 'both' mode labels are duplicated to match doubled batch
                if args.data_mode == 'both':
                    labels = labels.repeat(2)

                optimizer.zero_grad()
                pred = model(measurements, datetime, demographics,
                             context_mask if config.CLASSIFIER_USE_CONTEXT_MASK else None,
                             delta_t      if config.CLASSIFIER_USE_DELTA_T      else None)
                loss = criterion(pred, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config.CLASSIFIER_GRAD_CLIP)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation — always on predicted states to measure real use-case performance
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    measurements, context_mask, datetime, demographics, delta_t = get_classifier_input(
                        batch, args.data_mode, predictor, pred_cfg, device)

                    pred = model(measurements, datetime, demographics,
                                 context_mask if config.CLASSIFIER_USE_CONTEXT_MASK else None,
                                 delta_t      if config.CLASSIFIER_USE_DELTA_T      else None)

                    # For 'both', take only second half (predicted) for val metric
                    if args.data_mode == 'both':
                        b = pred.shape[0] // 2
                        pred = pred[b:]

                    all_preds.append(pred.cpu().numpy())
                    all_labels.append(batch['label'].numpy())

            val_auroc = roc_auc_score(np.concatenate(all_labels),
                                      np.concatenate(all_preds))
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

        # Test evaluation
        model.load_state_dict(best_state)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                measurements, context_mask, datetime, demographics, delta_t = get_classifier_input(
                    batch, args.data_mode, predictor, pred_cfg, device)

                pred = model(measurements, datetime, demographics,
                             context_mask if config.CLASSIFIER_USE_CONTEXT_MASK else None,
                             delta_t      if config.CLASSIFIER_USE_DELTA_T      else None)

                if args.data_mode == 'both':
                    b = pred.shape[0] // 2
                    pred = pred[b:]

                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch['label'].numpy())

        run_test_auroc = roc_auc_score(np.concatenate(all_labels),
                                       np.concatenate(all_preds))
        results_all.append({**cfg_run,
                            'val_auroc':  best_val_auroc,
                            'test_auroc': run_test_auroc})

        if best_val_auroc > best_overall_auroc:
            best_overall_auroc = best_val_auroc
            best_overall_cfg   = cfg_run
            best_overall_state = best_state

    print(f"\n=== Best config: {best_overall_cfg} ===")
    print(f"Best val AUROC: {best_overall_auroc:.4f}")
    print(f"Test AUROC:     {run_test_auroc:.4f}")
    print("\n=== Grid Search Summary ===")
    for r in sorted(results_all, key=lambda x: x['val_auroc'], reverse=True):
        print(f"  hidden={r['hidden_dim']} dropout={r['dropout']} "
              f"val={r['val_auroc']:.4f} test={r['test_auroc']:.4f}")

    # Checkpoint name includes data_mode
    mode_tag  = {'real': 'real_data',
                 'predicted': 'predicted_data',
                 'both': 'real_and_predicted_data'}[args.data_mode]
    date_str  = dt.now().strftime("%d_%m_%H-%M")
    ckpt_name = (f"gru_classifier_h{best_overall_cfg['hidden_dim']}"
                 f"_do{best_overall_cfg['dropout']}"
                 f"_{mode_tag}_{date_str}.pt")

    torch.save(best_overall_state,
               os.path.join(config.CHECKPOINT_DIR, ckpt_name))
    print(f"Saved: {ckpt_name}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR,
                           f'gru_classifier_{mode_tag}_results.json'), 'w') as f:
        json.dump({
            'data_mode':      args.data_mode,
            'best_config':    best_overall_cfg,
            'best_val_auroc': best_overall_auroc,
            'test_auroc':     run_test_auroc,
            'grid_results':   results_all,
        }, f, indent=2)


if __name__ == '__main__':
    train_gru_classifier()

#python train_classifier.py --grid --data_mode real
#python train_classifier.py --data_mode predicted
#python train_classifier.py --data_mode both