"""Fresh-data temporal robustness study; never replaces public model artifacts.

Session-relative timing uses complete closed sessions, not future-free streaming.
Training includes more repetition widths and contrast, and regular benign polling.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ids_ml.features import build_preprocessor, transform_features
from ids_ml.models.sequence import SequenceTrainConfig, train_sequence_model
from ids_ml.models.train import TrainConfig, train_multiclass_model
from .temporal_patterns import (generate_pairs, pack, shuffled, sequence_scores, beta_scores,
    summary_features, threshold_for_false_alarms, score_metrics)

@dataclasses.dataclass(frozen=True)
class RobustnessConfig:
    seeds: tuple[int, ...] = (211, 239, 271)
    train_pairs: int = 1200
    val_pairs: int = 400
    test_pairs: int = 600
    polling_sessions: int = 300
    max_epochs: int = 80
    min_epochs: int = 45
    patience: int = 10

SUITES = {
    'familiar': {},
    'pace_only': {'pace': (30, 70)},
    'jitter_only': {'jitter': .25},
    'width_only': {'blocks': [5]},
    'contrast_only': {'contrast': (1.15, 1.4)},
    'combined': {'shifted': True},
}
TRAIN_RECIPE = {'blocks': [1, 2, 3, 4], 'contrast': (1.4, 3.5), 'jitter': .12}


def relative_view(df):
    """Pure session transform. No labels, fitted test statistics, or absolute pace.

    Sorting before summation makes the scale exactly permutation-invariant.
    Closed-session context is available to both the MLP and GRU in this study.
    """
    out = df.copy()
    gaps = out.prior_gap_s.to_numpy(copy=True)
    for indices in out.groupby('session_id', sort=False).indices.values():
        positive = np.sort(gaps[indices][gaps[indices] > 0])
        scale = float(np.mean(positive)) if len(positive) else 1.
        gaps[indices] /= max(scale, 1e-6)
    out['prior_gap_s'] = gaps
    return out


def polling(seed, sessions, prefix, pace=(5, 25)):
    base, _ = generate_pairs(seed, sessions, prefix)
    df = base[base.target == 0].copy().reset_index(drop=True)
    rng = np.random.default_rng(seed + 999)
    gaps = df.prior_gap_s.to_numpy(copy=True)
    for indices in df.groupby('session_id', sort=False).indices.values():
        gaps[indices] = rng.uniform(*pace) * rng.lognormal(0, .04, len(indices))
        gaps[indices[0]] = 0
        df.loc[indices, 'event_time'] = np.cumsum(gaps[indices])
    df['prior_gap_s'] = gaps
    df['scenario'] = 'benign_scheduled_polling'
    return df


def run_seed(config, seed, output):
    output.mkdir(parents=True)
    started = time.perf_counter()
    datasets = {}
    for offset, (name, count) in enumerate([('train', config.train_pairs), ('val', config.val_pairs)]):
        paired, _ = generate_pairs(seed + offset * 10000, count, f'{seed}-{name}', **TRAIN_RECIPE)
        benign = polling(seed + offset * 10000 + 5000, config.polling_sessions if name == 'train' else 100,
                         f'{seed}-{name}-polling')
        datasets[name] = pd.concat([paired, benign], ignore_index=True)
    for offset, (name, recipe) in enumerate(SUITES.items(), start=2):
        datasets[name], _ = generate_pairs(seed + offset * 10000, config.test_pairs, f'{seed}-{name}', **recipe)
    datasets['polling'] = polling(seed + 100000, config.polling_sessions, f'{seed}-polling')
    datasets['polling_slow'] = polling(seed + 110000, config.polling_sessions, f'{seed}-polling-slow', (30, 70))
    for name, df in datasets.items():
        df.to_parquet(output / f'{name}.parquet', index=False)
    views = {'raw': datasets, 'relative': {name: relative_view(df) for name, df in datasets.items()}}
    processors = {name: build_preprocessor(data['train']) for name, data in views.items()}
    packs = {view: {name: pack(df, processors[view]) for name, df in data.items()} for view, data in views.items()}
    models, histories = {}, {}
    seq_config = SequenceTrainConfig(seed=seed, max_epochs=config.max_epochs, min_epochs=config.min_epochs,
                                    patience=config.patience, batch_size=128)
    def callback(name):
        def emit(row):
            if row['epoch'] % 10 == 0:
                print(f'seed {seed} {name}: epoch {row["epoch"]+1}, validation F1 {row["val_f1"]:.3f}', flush=True)
        return emit
    for name, view in [('sona_raw', 'raw'), ('sona_relative', 'relative'), ('sona_relative_shuffled_train', 'relative')]:
        X, lengths, y, _ = packs[view]['train']
        V, vl, vy, _ = packs[view]['val']
        if 'shuffled_train' in name:
            X, V = shuffled(X, lengths, seed + 500), shuffled(V, vl, seed + 600)
        print(f'seed {seed}: training {name}', flush=True)
        models[name], histories[name] = train_sequence_model(X, lengths, y, V, vl, vy, seq_config, 2, on_epoch=callback(name))
    train, val = views['relative']['train'], views['relative']['val']
    models['beta2_relative'], histories['beta2_relative'] = train_multiclass_model(
        transform_features(processors['relative'], train), train.target.to_numpy(),
        transform_features(processors['relative'], val), val.target.to_numpy(),
        TrainConfig(seed=seed, max_epochs=60, patience=10, batch_size=512), 2)
    summary = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    summary.fit(summary_features(datasets['train'], packs['raw']['train'][3]), packs['raw']['train'][2])
    def scores_for(split):
        X, lengths, y, groups = packs['relative'][split]
        R, rl, _, rg = packs['raw'][split]
        beta, vote = beta_scores(models['beta2_relative'], views['relative'][split], processors['relative'], groups)
        scores = {'beta2_relative': (beta, vote), 'sona_raw': (sequence_scores(models['sona_raw'], R, rl), None),
                  'sona_relative': (sequence_scores(models['sona_relative'], X, lengths), None),
                  'sona_relative_shuffled_test': (sequence_scores(models['sona_relative'], shuffled(X, lengths, seed+700), lengths), None),
                  'sona_relative_shuffled_train': (sequence_scores(models['sona_relative_shuffled_train'], shuffled(X, lengths, seed+800), lengths), None),
                  'timing_summary': (summary.predict_proba(summary_features(datasets[split], rg))[:,1], None)}
        return y, scores
    vy, vs = scores_for('val')
    thresholds = {name: threshold_for_false_alarms(value, vy, .01) for name, (value, _) in vs.items()}
    results = {}
    for split in list(SUITES) + ['polling', 'polling_slow']:
        y, scores = scores_for(split)
        results[split] = {}
        for name, (value, guesses) in scores.items():
            if split.startswith('polling'):
                predictions = value >= .5 if guesses is None else guesses
                results[split][name] = {'sessions': len(y), 'false_positive_rate': float(np.mean(predictions)),
                    'false_positive_rate_at_validation_threshold': float(np.mean(value >= thresholds[name]))}
            else:
                results[split][name] = score_metrics(value, y, thresholds[name], guesses)
        np.savez_compressed(output/f'{split}_predictions.npz', y=y, **{k:v[0] for k,v in scores.items()})
        print(f'seed {seed} evaluated {split}', flush=True)
    for name, model in models.items():
        torch.save(model.state_dict(), output/f'{name}.pt')
    for view, processor in processors.items():
        (output/f'{view}_preprocessor.pkl').write_bytes(pickle.dumps(processor))
    (output/'summary.pkl').write_bytes(pickle.dumps(summary))
    record = {'seed': seed, 'results': results, 'histories': histories,
              'counts': {name: {'flows':len(df),'sessions':df.session_id.nunique()} for name,df in datasets.items()},
              'elapsed_seconds': time.perf_counter()-started}
    (output/'results.json').write_text(json.dumps(record, indent=2))
    return record


def run(config, output):
    if output.exists():
        raise ValueError('Choose a new output directory; prior results are preserved.')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    records=[]
    for seed in config.seeds:
        records.append(run_seed(config, seed, output/f'seed-{seed}'))
        (output/'checkpoint.json').write_text(json.dumps(records,indent=2))
    report={'benchmark_version':'2.0.0','config':dataclasses.asdict(config),'train_recipe':TRAIN_RECIPE,
            'suites':SUITES,'runs':records,'limitations':[
                'Fresh two-class architecture comparison, not deployed six-class checkpoints.',
                'Relative timing requires a completed session; not streaming inference.',
                'Benign scheduled polling is synthetic and not proof of benign/attack intent.',
                'Broader training changes widths/contrast/jitter; raw versus relative isolates normalization on that shared recipe.']}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    lines=['# Temporal robustness fixes','', 'Both models are freshly trained two-class architectures. Public artifacts are unchanged.', '',
           '| Test | Beta 2 (relative) accuracy | Sona raw accuracy | Sona relative accuracy | Shuffled relative accuracy | Timing summary accuracy |',
           '|---|---:|---:|---:|---:|---:|']
    for split in SUITES:
        values=[np.mean([r['results'][split][name]['accuracy'] for r in records]) for name in
                ['beta2_relative','sona_raw','sona_relative','sona_relative_shuffled_test','timing_summary']]
        lines.append('| '+split+' | '+' | '.join(f'{v:.2%}' for v in values)+' |')
    lines+=['','| Benign repeating traffic | Sona raw false alarms | Sona relative false alarms | Relative at validation-selected threshold |','|---|---:|---:|---:|']
    for split in ('polling','polling_slow'):
        values=[np.mean([r['results'][split][name][key] for r in records]) for name,key in
                [('sona_raw','false_positive_rate'),('sona_relative','false_positive_rate'),('sona_relative','false_positive_rate_at_validation_threshold')]]
        lines.append('| '+split+' | '+' | '.join(f'{v:.2%}' for v in values)+' |')
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[211,239,271])
    args=parser.parse_args()
    run(RobustnessConfig(seeds=tuple(args.seeds)),args.out_dir)

if __name__ == '__main__':
    main()
