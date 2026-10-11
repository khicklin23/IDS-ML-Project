"""Paired timing-order benchmark, separate from the six-family showcase.

A benign lookalike and a beacon-like session share exactly the same flow
feature multiset, first/last flow, and length. Only interior ordering differs.
This controlled capability experiment is not a realism/production benchmark.
The existing Beta 2 MLP and Sona GRU architectures are retrained on two
classes; the deployed weights/artifacts are not changed or evaluated here.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ids_ml.features import build_preprocessor, transform_features, FEATURE_COLUMNS
from ids_ml.models.evaluate import predict_class_proba
from ids_ml.models.sequence import SequenceTrainConfig, train_sequence_model, predict_sequence_proba
from ids_ml.models.train import TrainConfig, train_multiclass_model


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    seeds: tuple[int, ...] = (11, 29, 47, 71, 101)
    train_pairs: int = 1200
    val_pairs: int = 400
    test_pairs: int = 1000
    shifted_pairs: int = 1000
    min_length: int = 32
    max_length: int = 64
    max_epochs: int = 50
    patience: int = 8
    batch_size: int = 128
    threads: int = 2
    false_alarm_target: float = 0.01


def generate_pairs(seed: int, pairs: int, prefix: str, *, shifted: bool = False,
                   min_length: int = 32, max_length: int = 64,
                   pace=None, contrast=None, blocks=None, jitter=None) -> tuple[pd.DataFrame, dict]:
    if pairs < 1 or min_length < 8 or max_length < min_length:
        raise ValueError('Positive pairs and valid sequence lengths >=8 required.')
    rng = np.random.default_rng(seed)
    records = []
    for pair in range(pairs):
        n = int(rng.integers(min_length, max_length + 1))
        base = rng.uniform(*(pace or ((30, 70) if shifted else (5, 25))))
        ratio = rng.uniform(*(contrast or ((1.4, 2) if shifted else (2, 3))))
        block = int(rng.choice(blocks or ([1, 2, 3] if shifted else [1, 2])))
        phase = int(rng.integers(0, block * 2))
        levels = ((np.arange(n) + phase) // block) % 2
        gaps = base * np.where(levels, ratio, 1.) * rng.lognormal(0, (.16 if shifted else .08) if jitter is None else jitter, n)
        gaps[0] = 0
        sent = rng.integers(2, 10, n)
        recv = rng.integers(1, 8, n)
        # These nuisance fields have identical distributions and multisets in each pair.
        frame = pd.DataFrame({'duration_s': rng.lognormal(-1., .5, n),
                              'src_port': rng.integers(30000, 60000, n), 'dst_port': 443,
                              'protocol': 'tcp', 'service': 'https', 'outcome': 'established',
                              'packets_sent': sent, 'packets_recv': recv,
                              'bytes_sent': sent * rng.integers(60, 300, n),
                              'bytes_recv': recv * rng.integers(80, 500, n), 'prior_gap_s': gaps})
        order = np.arange(n)
        order[1:-1] = rng.permutation(order[1:-1])
        for label, indices in [(0, order), (1, np.arange(n))]:
            sample = frame.iloc[indices].copy()
            sample['event_time'] = np.cumsum(sample.prior_gap_s.to_numpy())
            sample['pair_id'] = f'{prefix}-pair-{pair}'
            sample['session_id'] = f'{prefix}-pair-{pair}-label-{label}'
            sample['scenario'] = 'beaconing' if label else 'benign'
            sample['target'] = label
            sample['flow_id'] = [f'{prefix}-{pair}-{label}-{i}' for i in range(n)]
            records.append(sample)
    df = pd.concat(records, ignore_index=True)
    return df, {'seed': seed, 'pairs': pairs, 'sessions': pairs * 2, 'flows': len(df),
                'shifted': shifted, 'length_range': [min_length, max_length],
                'note': 'Matched feature multisets and endpoints; attack-like cyclic timing vs shuffled benign lookalike.'}


def pack(df, preprocessor):
    features = transform_features(preprocessor, df)
    groups = list(df.groupby('session_id', sort=False).indices.values())
    lengths = np.array([len(indices) for indices in groups], dtype=np.int64)
    X = np.zeros((len(groups), lengths.max(), features.shape[1]), dtype=np.float32)
    y = np.zeros(len(groups), dtype=np.int64)
    for i, indices in enumerate(groups):
        X[i, :len(indices)] = features[indices]
        y[i] = int(df.iloc[indices[0]].target)
    return X, lengths, y, groups


def shuffled(X, lengths, seed):
    out = X.copy()
    rng = np.random.default_rng(seed)
    for i, length in enumerate(lengths):
        # Preserve endpoints to prevent a last-flow shortcut explaining the drop.
        out[i, 1:length - 1] = out[i, rng.permutation(np.arange(1, length - 1))]
    return out


def summary_features(df, groups):
    rows = []
    for indices in groups:
        gap = df.iloc[indices].prior_gap_s.to_numpy()[1:]
        scale = max(gap.std(), 1e-6)
        centered = (gap - gap.mean()) / scale
        rows.append([len(gap), gap.mean(), gap.std(), gap.min(), gap.max(),
                     *np.quantile(gap, [.1, .5, .9]),
                     *[float(np.mean(centered[:-lag] * centered[lag:])) for lag in (1, 2, 3, 4)],
                     float(np.mean(np.abs(np.diff(gap))) / max(gap.mean(), 1e-6))])
    return np.asarray(rows, dtype=np.float32)


def threshold_for_false_alarms(scores, y, target):
    if not 0 <= target < 1 or not np.any(y == 0):
        raise ValueError('Need benign validation examples and a target in [0,1).')
    benign = np.sort(np.asarray(scores)[y == 0])[::-1]
    permitted = int(np.floor(target * len(benign)))
    # Strictly above the next rejected score: ties cannot exceed the budget.
    return float(np.nextafter(benign[permitted], np.inf))


def score_metrics(scores, y, threshold, predictions=None):
    guesses = (scores >= .5).astype(int) if predictions is None else predictions
    tn, fp, fn, tp = confusion_matrix(y, guesses, labels=[0, 1]).ravel()
    qtn, qfp, qfn, qtp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel()
    return {'sessions': len(y), 'accuracy': float(np.mean(guesses == y)),
            'macro_f1': float(f1_score(y, guesses, average='macro', zero_division=0)),
            'roc_auc': float(roc_auc_score(y, scores)), 'average_precision': float(average_precision_score(y, scores)),
            'default_confusion': {k: int(v) for k, v in zip(('tn','fp','fn','tp'), (tn,fp,fn,tp))},
            'threshold_from_validation': threshold,
            'at_validation_false_alarm_target': {'recall': float(qtp / (qtp + qfn)),
                                                'false_positive_rate': float(qfp / (qfp + qtn)),
                                                'tp': int(qtp), 'fp': int(qfp), 'fn': int(qfn), 'tn': int(qtn)}}


def beta_scores(model, df, preprocessor, groups):
    probs = predict_class_proba(model, transform_features(preprocessor, df))[:, 1]
    scores = np.array([np.mean(probs[indices], dtype=np.float64) for indices in groups], dtype=np.float64)
    # Existing Beta2 session prediction: majority vote; tie resolves to benign here.
    votes = np.array([np.mean(probs[indices] >= .5) > .5 for indices in groups], dtype=int)
    return scores, votes


def sequence_scores(model, X, lengths, batch=256):
    return np.concatenate([predict_sequence_proba(model, X[i:i+batch], lengths[i:i+batch])[:, 1]
                           for i in range(0, len(X), batch)]).astype(np.float64)


def run_seed(config, seed, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    datasets, manifests = {}, {}
    for offset, (split, pairs) in enumerate([('train', config.train_pairs), ('val', config.val_pairs),
                                          ('test', config.test_pairs), ('shifted', config.shifted_pairs)]):
        datasets[split], manifests[split] = generate_pairs(seed + offset * 10000, pairs, f'seed-{seed}-{split}',
                                                         shifted=split == 'shifted', min_length=config.min_length,
                                                         max_length=config.max_length)
        datasets[split].to_parquet(output / f'{split}.parquet', index=False)
    preprocessor = build_preprocessor(datasets['train'])
    packed = {name: pack(df, preprocessor) for name, df in datasets.items()}
    X, lengths, y, groups = packed['train']
    V, vl, vy, vg = packed['val']
    histories, models = {}, {}
    print(f'seed {seed}: {sum(m["flows"] for m in manifests.values()):,} flows, training Beta2', flush=True)
    train_df, val_df = datasets['train'], datasets['val']
    models['beta2'], histories['beta2'] = train_multiclass_model(
        transform_features(preprocessor, train_df), train_df.target.to_numpy(),
        transform_features(preprocessor, val_df), val_df.target.to_numpy(),
        TrainConfig(seed=seed, max_epochs=config.max_epochs, patience=config.patience, batch_size=512), 2)
    def progress(model_name):
        return lambda row: print(f'seed {seed} {model_name}: epoch {row["epoch"]+1}, val F1 {row["val_f1"]:.3f}', flush=True) if row['epoch'] % 5 == 0 else None
    seq_config = SequenceTrainConfig(seed=seed, max_epochs=config.max_epochs, patience=config.patience,
                                    batch_size=config.batch_size)
    print(f'seed {seed}: training Sona', flush=True)
    models['sona'], histories['sona'] = train_sequence_model(X, lengths, y, V, vl, vy, seq_config, 2, on_epoch=progress('sona'))
    print(f'seed {seed}: training shuffled-order control', flush=True)
    models['sona_shuffled_train'], histories['sona_shuffled_train'] = train_sequence_model(
        shuffled(X, lengths, seed + 50000), lengths, y, shuffled(V, vl, seed + 60000), vl, vy,
        seq_config, 2, on_epoch=progress('shuffled'))
    summary_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    summary_model.fit(summary_features(train_df, groups), y)
    last_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    last_model.fit(X[np.arange(len(X)), lengths - 1], y)
    def scores_for(split):
        xx, ll, yy, gg = packed[split]
        df = datasets[split]
        beta, vote = beta_scores(models['beta2'], df, preprocessor, gg)
        return yy, {'beta2': (beta, vote), 'sona': (sequence_scores(models['sona'], xx, ll), None),
                    'sona_shuffled_test': (sequence_scores(models['sona'], shuffled(xx, ll, seed + 70000), ll), None),
                    'sona_shuffled_train': (sequence_scores(models['sona_shuffled_train'], shuffled(xx, ll, seed + 80000), ll), None),
                    'timing_summary_logistic': (summary_model.predict_proba(summary_features(df, gg))[:, 1], None),
                    'last_flow_logistic': (last_model.predict_proba(xx[np.arange(len(xx)), ll - 1])[:, 1], None)}
    val_y, val_scores = scores_for('val')
    thresholds = {name: threshold_for_false_alarms(scores, val_y, config.false_alarm_target)
                  for name, (scores, _) in val_scores.items()}
    results = {}
    for split in ('test', 'shifted'):
        yy, scores = scores_for(split)
        results[split] = {name: score_metrics(value, yy, thresholds[name], pred)
                          for name, (value, pred) in scores.items()}
        np.savez_compressed(output / f'{split}_predictions.npz', y=yy,
                            **{name: value for name, (value, _) in scores.items()})
    for name, model in models.items():
        torch.save(model.state_dict(), output / f'{name}.pt')
    (output / 'preprocessor.pkl').write_bytes(pickle.dumps(preprocessor))
    (output / 'summary_baseline.pkl').write_bytes(pickle.dumps(summary_model))
    record = {'seed': seed, 'manifests': manifests, 'histories': histories, 'results': results,
              'elapsed_seconds': time.perf_counter() - started, 'encoded_features': X.shape[-1],
              'classes': ['benign', 'beaconing'], 'config': dataclasses.asdict(config)}
    (output / 'results.json').write_text(json.dumps(record, indent=2))
    print(f'seed {seed} done: test F1 Beta2={results["test"]["beta2"]["macro_f1"]:.3f}, Sona={results["test"]["sona"]["macro_f1"]:.3f}', flush=True)
    return record


def run_benchmark(config, output):
    if output.exists():
        raise ValueError('Use a new output directory to preserve previous experiments.')
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError('Seeds must be unique.')
    output.mkdir(parents=True)
    torch.set_num_threads(config.threads)
    records = []
    for seed in config.seeds:
        records.append(run_seed(config, seed, output / f'seed-{seed}'))
        (output / 'checkpoint.json').write_text(json.dumps(records, indent=2))
    summary = {}
    for split in ('test', 'shifted'):
        summary[split] = {}
        for name in records[0]['results'][split]:
            rows = [r['results'][split][name] for r in records]
            summary[split][name] = {key: {'mean': float(np.mean([r[key] for r in rows])),
                                         'min': float(min(r[key] for r in rows)), 'max': float(max(r[key] for r in rows)),
                                         'std': float(np.std([r[key] for r in rows], ddof=1)) if len(rows)>1 else 0.}
                                    for key in ('accuracy','macro_f1','roc_auc','average_precision')}
            for key in ('recall','false_positive_rate'):
                values = [r['at_validation_false_alarm_target'][key] for r in rows]
                summary[split][name]['at_target_' + key] = {'mean': float(np.mean(values)), 'min': min(values), 'max': max(values)}
    report = {'benchmark_version': '1.0.0', 'generated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'config': dataclasses.asdict(config), 'summary': summary, 'runs': records,
              'limitations': ['Purpose-built timing-order discrimination, not general IDS or attacker intent.',
                              'Fresh two-class training using existing architectures, not the deployed six-class checkpoints.',
                              'Matched pairs stay within one split; paired rows must never be split independently.',
                              'False-alarm thresholds are selected on validation; test rates may differ from the target.',
                              'Timing-summary baseline tests whether simple feature engineering also solves the task.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(BenchmarkConfig().seeds))
    parser.add_argument('--train-pairs', type=int, default=1200)
    parser.add_argument('--val-pairs', type=int, default=400)
    parser.add_argument('--test-pairs', type=int, default=1000)
    parser.add_argument('--shifted-pairs', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=8,
                        help='Epochs without validation-loss improvement before early stopping.')
    args = parser.parse_args()
    config = BenchmarkConfig(seeds=tuple(args.seeds), train_pairs=args.train_pairs, val_pairs=args.val_pairs,
                             test_pairs=args.test_pairs, shifted_pairs=args.shifted_pairs,
                             max_epochs=args.epochs, patience=args.patience)
    run_benchmark(config, args.out_dir)


if __name__ == '__main__':
    main()
