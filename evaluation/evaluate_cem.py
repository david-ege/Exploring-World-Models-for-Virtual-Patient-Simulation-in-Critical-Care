import sys
sys.path.append('/home/bbe9928/thesis_work/hirid_jepa')

import argparse
import torch
import numpy as np

import config
from models.cem import load_predictor, load_classifier, run_cem_evaluation, save_evaluation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_patients',  type=int,   default=100)
    parser.add_argument('--config_tag',  type=str,   default='baseline_config')
    parser.add_argument('--split',       type=str,   default='test')
    return parser.parse_args()

if __name__ == '__main__':
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Running CEM evaluation: {args.n_patients} patients, tag='{args.config_tag}'")

    predictor, predictor_config = load_predictor(device)
    classifier                  = load_classifier(device)

    results = run_cem_evaluation(
        n_patients=args.n_patients,
        classifier=classifier,
        predictor=predictor,
        predictor_config=predictor_config,
        device=device,
        config_tag=args.config_tag,
        split=args.split
    )

    path = f"{config.RESULTS_DIR}/cem_eval_{results['config_tag']}.json"
    save_evaluation(results, path)

    print(f"\nDone: {results['n_patients']} patients — saved to {path}")
    print(f"Mean improvement: {results['improvement'].mean():.4f} ± {results['improvement'].std():.4f}")
    print(f"Patients improved: {(results['improvement'] > 0).sum()}/{len(results['improvement'])}")