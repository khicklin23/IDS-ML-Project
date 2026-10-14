"""Run a trusted local binary candidate on parquet telemetry without labels."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from .extended_inference import load_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--telemetry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', choices=('0.01', '0.02', '0.05'), default='0.01')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve existing outputs; choose a new output path.')
    torch.set_num_threads(args.threads)
    model = load_predictor(args.candidate)
    with threadpool_limits(limits=args.threads):
        prediction = model.predict(pd.read_parquet(args.telemetry))
    threshold = model.manifest['thresholds'][args.budget]
    result = {'artifact_version': model.manifest['artifact_version'], 'unit': 'whole observation window',
              'candidate': str(args.candidate), 'validation_budget': args.budget, 'threshold': threshold,
              'predictions': [{'session_id': str(identity), 'suspiciousness': float(score), 'flagged': bool(score >= threshold)}
                              for identity, score in zip(prediction['session_ids'], prediction['scores'])]}
    args.output.write_text(json.dumps(result, indent=2))
    print(f'Wrote {len(result["predictions"])} window predictions to {args.output}')


if __name__ == '__main__': main()
