# world_model_mortality.py
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
import glob, os

import config
from data.constants import MEASUREMENT_IDX, TREATMENT_IDX, DEMOGRAPHIC_IDX, DATETIME_IDX
from data import mortality_dataset
from models.gru_predictor import GRUPredictor

with open('/home/bbe9928/thesis_work/hirid_jepa/data/mortality_labels.json') as f:
    mortality_labels = json.load(f)


def summarize(arr, mask):
    """Compute mean, std, obs_rate per variable from a (T, V) array."""
    obs_mean = np.where(mask == 1, arr, 0.0).sum(axis=0) / np.maximum(mask.sum(axis=0), 1)
    obs_std  = np.sqrt(
        np.where(mask == 1, (arr - obs_mean) ** 2, 0.0).sum(axis=0) /
        np.maximum(mask.sum(axis=0) - 1, 1)
    )
    obs_rate = mask.mean(axis=0)
    return np.concatenate([obs_mean, obs_std, obs_rate])


def run(checkpoint_name):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg        = checkpoint['config']
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded: {checkpoint_name}")

    results = {}
    for split in ['train', 'val', 'test']:
        ds     = mortality_dataset.MortalityDataset(config.DATA_PATH, split,
                                  config.CONTEXT_STEPS, cfg['target_steps'])
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)

        context_features = []
        predicted_features = []
        labels = []

        with torch.no_grad():
            for batch in loader:
                measurements = batch['measurements'].to(device)
                treatments   = batch['treatments'].to(device)
                datetime     = batch['datetime'].to(device)
                demographics = batch['demographics'].to(device)
                context_mask = batch['context_mask'].to(device)
                delta_t      = batch['delta_t'].to(device)

                # World model prediction
                pred = model(measurements, treatments, datetime, demographics,
                             context_mask if use_context_mask else None,
                             delta_t      if use_delta_t      else None)
                # pred shape: (B, target_steps, n_measurements)

                # Context features — summarize real observed context
                m_np   = measurements.cpu().numpy()  # (B, T, V)
                mk_np  = context_mask.cpu().numpy()
                pred_np = pred.cpu().numpy()          # (B, target_steps, V)

                for b in range(m_np.shape[0]):
                    demo_np = batch['demographics'].cpu().numpy()[b]  # (N_DEMOGRAPHICS,)
                    
                    # Context features: measurement summary + demographics
                    context_feat = np.concatenate([summarize(m_np[b], mk_np[b]), demo_np])
                    context_features.append(context_feat)
                    
                    # Predicted features: predicted state summary + demographics
                    pred_mask = np.ones((pred_np.shape[1], pred_np.shape[2]))
                    pred_feat = np.concatenate([summarize(pred_np[b], pred_mask), demo_np])
                    predicted_features.append(pred_feat)

                labels.append(batch['label'].numpy())

        results[split] = {
            'X_context':   np.array(context_features,   dtype=np.float32),
            'X_predicted': np.array(predicted_features, dtype=np.float32),
            'y':           np.concatenate(labels),
        }
        print(f"{split}: {len(results[split]['y'])} patients, "
              f"mortality={results[split]['y'].mean():.3f}")

    # Replace NaN
    for split in results:
        results[split]['X_context']   = np.nan_to_num(results[split]['X_context'],   nan=0.0)
        results[split]['X_predicted'] = np.nan_to_num(results[split]['X_predicted'], nan=0.0)

    print("\n=== LR on Context (real observed data) ===")
    for C in [0.001, 0.01, 0.1, 1, 10]:
        probe = Pipeline([('scaler', StandardScaler()),
                        ('clf', LogisticRegression(max_iter=1000, C=C,
                                                    class_weight='balanced'))])
        probe.fit(results['train']['X_context'], results['train']['y'])
        val_auroc  = roc_auc_score(results['val']['y'],
                                probe.predict_proba(results['val']['X_context'])[:, 1])
        test_auroc = roc_auc_score(results['test']['y'],
                                probe.predict_proba(results['test']['X_context'])[:, 1])
        print(f"  C={C}: val AUROC={val_auroc:.4f}, test AUROC={test_auroc:.4f}")

    best_C = 0.01  # update after seeing results
    probe_ctx = Pipeline([('scaler', StandardScaler()),
                          ('clf', LogisticRegression(max_iter=1000, C=best_C,
                                                     class_weight='balanced'))])
    probe_ctx.fit(results['train']['X_context'], results['train']['y'])
    val_auroc_ctx  = roc_auc_score(results['val']['y'],
                                    probe_ctx.predict_proba(results['val']['X_context'])[:, 1])
    test_auroc_ctx = roc_auc_score(results['test']['y'],
                                    probe_ctx.predict_proba(results['test']['X_context'])[:, 1])
    print(f"\nBest C={best_C} — Val AUROC: {val_auroc_ctx:.4f}, Test AUROC: {test_auroc_ctx:.4f}")
    print(classification_report(results['test']['y'],
                                 probe_ctx.predict(results['test']['X_context']),
                                 target_names=['survived', 'died']))

    print("\n=== LR on Predicted Future States (world model output) ===")
    for C in [0.001, 0.01, 0.1, 1, 10]:
        probe = Pipeline([('scaler', StandardScaler()),
                          ('clf', LogisticRegression(max_iter=1000, C=C,
                                                     class_weight='balanced'))])
        probe.fit(results['train']['X_predicted'], results['train']['y'])
        val_auroc = roc_auc_score(results['val']['y'],
                                   probe.predict_proba(results['val']['X_predicted'])[:, 1])
        test_auroc = roc_auc_score(results['test']['y'],
                               probe.predict_proba(results['test']['X_predicted'])[:, 1])
        print(f"  C={C}: val AUROC={val_auroc:.4f}, test AUROC={test_auroc:.4f}")

    best_C_pred = 1  # update after seeing results
    probe_pred = Pipeline([('scaler', StandardScaler()),
                           ('clf', LogisticRegression(max_iter=1000, C=best_C_pred,
                                                      class_weight='balanced'))])
    probe_pred.fit(results['train']['X_predicted'], results['train']['y'])
    val_auroc_pred  = roc_auc_score(results['val']['y'],
                                     probe_pred.predict_proba(results['val']['X_predicted'])[:, 1])
    test_auroc_pred = roc_auc_score(results['test']['y'],
                                     probe_pred.predict_proba(results['test']['X_predicted'])[:, 1])
    print(f"\nBest C={best_C_pred} — Val AUROC: {val_auroc_pred:.4f}, Test AUROC: {test_auroc_pred:.4f}")
    print(classification_report(results['test']['y'],
                                 probe_pred.predict(results['test']['X_predicted']),
                                 target_names=['survived', 'died']))

    print("\n=== Summary ===")
    print(f"Context  LR: Val={val_auroc_ctx:.4f}, Test={test_auroc_ctx:.4f}")
    print(f"Predicted LR: Val={val_auroc_pred:.4f}, Test={test_auroc_pred:.4f}")
    print(f"Gap (context - predicted): {val_auroc_ctx - val_auroc_pred:.4f}")

    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, 'world_model_mortality.json'), 'w') as f:
        json.dump({
            'checkpoint': checkpoint_name,
            'context_val_auroc':   val_auroc_ctx,
            'context_test_auroc':  test_auroc_ctx,
            'predicted_val_auroc': val_auroc_pred,
            'predicted_test_auroc': test_auroc_pred,
            'gap': test_auroc_ctx - test_auroc_pred,
        }, f, indent=2)
    print(f"Results saved to {config.RESULTS_DIR}/world_model_mortality.json")


if __name__ == '__main__':
    run(config.BEST_CHECKPOINT)