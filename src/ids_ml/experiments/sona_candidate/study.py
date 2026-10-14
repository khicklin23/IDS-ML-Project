"""Frozen matched confirmation study with restartable per-candidate checkpoints."""
import argparse
import dataclasses
import json
import pickle
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from ids_ml.experiments.scan_generalization_features import generate_partition, generate_condition
from ids_ml.experiments.mixed_scan import summaries, digest
from ids_ml.experiments.temporal_patterns import threshold_for_false_alarms, score_metrics, shuffled
from .features import CandidatePreprocessor, pack
from .models import train, probabilities
from .inference import CandidatePredictor


@dataclasses.dataclass(frozen=True)
class Config:
    seeds: tuple = (1301, 1327, 1361)
    train_windows: int = 3200
    val_windows: int = 3200
    test_windows: int = 1200
    epochs: int = 60
    min_epochs: int = 20
    patience: int = 12
    batch_size: int = 128
    threads: int = 2
    max_length: int = 160
    budgets: tuple = (.01, .02, .05)


SPECS = {
    'raw_mixed_gru': ('mixed', 'raw', 'gru'),
    'relative_ordinary_gru': ('ordinary', 'relative', 'gru'),
    'relative_mixed_gru': ('mixed', 'relative', 'gru'),
    'relative_ordinary_pool': ('ordinary', 'relative', 'pool'),
    'relative_mixed_pool': ('mixed', 'relative', 'pool'),
    'relative_mixed_hierarchical': ('mixed', 'relative', 'hierarchical'),
    'summary_ordinary': ('ordinary', 'raw', 'summary'),
    'summary_mixed': ('mixed', 'raw', 'summary'),
}
TESTS = ('familiar', 'duration', 'density', 'combined', 'inventory', 'intermediate', 'longer')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def targets(frame):
    grouped = frame.groupby('session_id', sort=False)
    if not grouped.window_target.nunique().eq(1).all():
        raise ValueError('Inconsistent target within a window.')
    return grouped.window_target.first().to_numpy(dtype=np.int64)


def generate_validation(seed, count, prefix):
    if count % 40: raise ValueError('Validation count must be divisible by 40.')
    return pd.concat([generate_condition(seed+i*1000, count//5, f'{prefix}-{name}', name)
                      for i, name in enumerate(('test', 'duration', 'density', 'combined', 'inventory'))], ignore_index=True)


def generate_test(seed, count, prefix, name):
    frame = generate_condition(seed, count, prefix, 'test' if name in ('familiar', 'intermediate', 'longer') else name)
    if name in ('intermediate', 'longer'):
        factor = 1.5 if name == 'intermediate' else 3.
        frame['event_time'] *= factor
        frame['observation_start_s'] *= factor
        frame['observation_duration_s'] *= factor
        frame['prior_gap_s'] = frame.groupby('session_id', sort=False).event_time.diff().fillna(0.)
    return frame


def metrics(scores, frame, thresholds):
    y = targets(frame)
    behaviors = frame.groupby('session_id', sort=False).behavior.first().to_numpy()
    result = {}
    for budget, threshold in thresholds.items():
        m = score_metrics(scores, y, threshold)
        m['by_behavior'] = {str(kind): {'windows': int(sum(behaviors == kind)),
            'flagged': int(sum((behaviors == kind) & (scores >= threshold))),
            'flag_rate': float(np.mean(scores[behaviors == kind] >= threshold))} for kind in sorted(set(behaviors))}
        result[budget] = m
    return result


def efficiency(model, x, lengths, threads):
    """Model-only warm latency, same CPU/batch/sequence workload per candidate."""
    model.eval()
    records = {}
    with torch.inference_mode():
        for size in (1, min(128, len(x))):
            xb, lb = torch.as_tensor(x[:size]), torch.as_tensor(lengths[:size])
            for _ in range(3): model(xb, lb)
            timings = []
            for _ in range(15):
                start = time.perf_counter(); model(xb, lb); timings.append(time.perf_counter()-start)
            median = float(np.median(timings))
            records[str(size)] = {'median_ms': median*1000, 'p95_ms': float(np.quantile(timings, .95))*1000,
                                 'windows_per_second': size/median}
    return {'model_only_warm': records, 'threads': threads, 'input_shape': list(x.shape),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'limitation': 'In-process CPU timing may contend with another agent; independent benchmarking required.'}


def source_hashes():
    root = Path(__file__).parent
    shared = root.parent
    files = ['__init__.py', 'features.py', 'models.py', 'inference.py', 'study.py', 'audit.py']
    return {str(p): digest(p) for p in [*[root/f for f in files], shared/'mixed_scan.py', shared/'scan_context.py', shared/'scan_generalization_features.py']}


def select_global(records, output):
    names = [n for n in SPECS if SPECS[n][2] != 'summary']
    def ranking(name):
        recall = np.mean([r['results']['val'][name]['0.01']['at_validation_false_alarm_target']['recall'] for r in records])
        parameters = np.mean([CandidatePredictor.load(output/f'seed-{r["seed"]}'/name).manifest['efficiency']['parameter_count'] for r in records])
        return -recall, parameters, name
    return sorted(names, key=ranking)[0]


def run_seed(config, seed, output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frames = {}
    for name, count, condition, offset in [('train_ordinary', config.train_windows, 'test', 0),
            ('train_mixed', config.train_windows, 'mixture', 10000), ('val', config.val_windows, 'validation', 20000)]:
        path = output/f'{name}.parquet'
        if not path.exists():
            prefix = f'candidate-{seed}-{name}'
            f = generate_validation(seed+offset, count, prefix) if condition == 'validation' else generate_partition(seed+offset, count, prefix, condition)
            f.to_parquet(path, index=False)
        frames[name] = pd.read_parquet(path)
    caches, models, records = {}, {}, {}
    for name, (regime, mode, architecture) in SPECS.items():
        directory = output/name
        if (directory/'manifest.json').exists():
            predictor = CandidatePredictor.load(directory)
            records[name] = json.loads((directory/'validation.json').read_text())
            models[name] = predictor
            print(f'seed {seed}: resumed {name}', flush=True)
            continue
        directory.mkdir(exist_ok=True)
        training_frame = frames[f'train_{regime}']
        if (regime, mode) not in caches:
            processor = CandidatePreprocessor.fit(training_frame, mode)
            tr = pack(training_frame, processor, config.max_length)
            va = pack(frames['val'], processor, config.max_length)
            caches[regime, mode] = processor, tr, va
        processor, tr, va = caches[regime, mode]
        print(f'seed {seed}: training {name} on {len(tr[0])} windows', flush=True)
        if architecture == 'summary':
            values, _ = summaries(training_frame, tr[3])
            model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, l2_regularization=1., random_state=seed)
            with threadpool_limits(limits=config.threads):
                model.fit(values, targets(training_frame))
                vv, _ = summaries(frames['val'], va[3]); scores = model.predict_proba(vv)[:, 1]
            (directory/'model.pkl').write_bytes(pickle.dumps(model))
            history, performance, files = [], {}, ['model.pkl']
        else:
            model, history = train(architecture, (*tr[:2], targets(training_frame)), (*va[:2], targets(frames['val'])), config, seed)
            scores = probabilities(model, *va[:2])[:, 1].astype(np.float64)
            torch.save(model.state_dict(), directory/'model.pt')
            (directory/'preprocessor.pkl').write_bytes(pickle.dumps(processor))
            performance = efficiency(model, *va[:2], config.threads)
            files = ['model.pt', 'preprocessor.pkl']
        thresholds = {f'{b:.2f}': threshold_for_false_alarms(scores, targets(frames['val']), b) for b in config.budgets}
        validation = metrics(scores, frames['val'], thresholds)
        write_json(directory/'history.json', history)
        write_json(directory/'validation.json', validation)
        np.save(directory/'validation_scores.npy', scores)
        manifest = {'artifact_version': 'sona-candidate-1.0', 'architecture': architecture, 'representation': mode,
            'training_regime': regime, 'seed': seed, 'classes': ['benign', 'suspicious'], 'input_dim': tr[0].shape[-1],
            'max_length': config.max_length, 'thresholds': thresholds, 'feature_names': processor.get_feature_names_out().tolist(),
            'training_sha256': digest(output/f'train_{regime}.parquet'), 'validation_sha256': digest(output/'val.parquet'),
            'sha256': {f: digest(directory/f) for f in files}, 'efficiency': performance,
            'checkpoint_bytes': (directory/files[0]).stat().st_size}
        write_json(directory/'manifest.json', manifest)
        models[name] = CandidatePredictor.load(directory)
        records[name] = validation
    # Choice is frozen on validation before opening/generating any test partitions.
    names = [n for n in SPECS if SPECS[n][2] != 'summary']
    selected = sorted(names, key=lambda n: (-records[n]['0.01']['at_validation_false_alarm_target']['recall'],
        models[n].manifest['efficiency']['parameter_count'], n))[0]
    write_json(output/'selection.json', {'selected': selected, 'basis': 'validation recall at 1% budget, parameter-count tie break',
                                        'validation_results': records})
    results, counts, hashes = {'val': records}, {k: len(f) for k, f in frames.items()}, {k: digest(output/f'{k}.parquet') for k in frames}
    for i, split in enumerate(TESTS):
        path = output/f'{split}.parquet'
        if not path.exists(): generate_test(seed+30000+i*10000, config.test_windows, f'candidate-{seed}-{split}', split).to_parquet(path, index=False)
        frame = pd.read_parquet(path)
        predictions = {'y': targets(frame)}
        results[split] = {}
        for name, predictor in models.items():
            prediction = predictor.predict(frame)
            scores = prediction['scores']
            predictions[name] = scores
            predictions['session_ids'] = prediction['session_ids']
            results[split][name] = metrics(scores, frame, predictor.manifest['thresholds'])
        np.savez_compressed(output/f'{split}_predictions.npz', **predictions)
        counts[split], hashes[split] = len(frame), digest(path)
        print(f'seed {seed}: evaluated {split}', flush=True)
    # Input-order perturbation diagnostic, without recalibrating thresholds.
    diagnostic = {}
    frame = pd.read_parquet(output/'familiar.parquet')
    for name, predictor in models.items():
        if predictor.processor is None: continue
        x, lengths, _, _ = pack(frame, predictor.processor, config.max_length)
        scores = probabilities(predictor.model, shuffled(x, lengths, seed+999), lengths)[:, 1].astype(np.float64)
        diagnostic[name] = metrics(scores, frame, predictor.manifest['thresholds'])
    record = {'seed': seed, 'selected': selected, 'results': results, 'shuffled_inputs': diagnostic,
              'flow_counts': counts, 'source_sha256': hashes, 'elapsed_seconds': time.perf_counter()-started}
    write_json(output/'results.json', record)
    return record


def aggregate(records):
    summary = {}
    for split in TESTS:
        summary[split] = {}
        for name in SPECS:
            summary[split][name] = {}
            for budget in ('0.01', '0.02', '0.05'):
                row = {}
                for metric in ('recall', 'false_positive_rate'):
                    values = [r['results'][split][name][budget]['at_validation_false_alarm_target'][metric] for r in records]
                    row[metric] = {'mean': float(np.mean(values)), 'min': float(min(values)), 'max': float(max(values))}
                summary[split][name][budget] = row
    return summary


def run(config, output):
    if config.train_windows % 8 or config.val_windows % 40 or config.test_windows < 4 or config.test_windows % 2:
        raise ValueError('Use valid balanced train/validation/test counts.')
    if len(set(config.seeds)) != len(config.seeds): raise ValueError('Unique seeds required.')
    if not 1 <= config.min_epochs <= config.epochs: raise ValueError('Invalid epoch bounds.')
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(json.dumps(dataclasses.asdict(config)))
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text()) != protocol: raise ValueError('Resume configuration changed.')
        if json.loads((output/'implementation_sha256.json').read_text()) != source_hashes(): raise ValueError('Resume implementation changed.')
    else:
        write_json(output/'protocol.json', protocol)
        write_json(output/'implementation_sha256.json', source_hashes())
        write_json(output/'environment.json', {'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform()})
    torch.set_num_threads(config.threads)
    records = []
    with threadpool_limits(limits=config.threads):
        for seed in config.seeds:
            path = output/f'seed-{seed}'
            record = json.loads((path/'results.json').read_text()) if (path/'results.json').exists() else run_seed(config, seed, path)
            records.append(record)
            write_json(output/'checkpoint.json', records)
    selected = select_global(records, output)
    report = {'version': 'sona-confirmation-1.0', 'config': protocol, 'runs': records, 'summary': aggregate(records),
              'selected_global': selected, 'selection_basis': 'Mean validation strict-budget recall; lower parameter count breaks ties.'}
    write_json(output/'report.json', report)
    write_json(output/'finalists.json', {str(r['seed']): str((output/f'seed-{r["seed"]}'/selected).resolve()) for r in records})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(Config().seeds))
    for flag, default in [('train-windows', 3200), ('val-windows', 3200), ('test-windows', 1200), ('epochs', 60), ('min-epochs', 20)]:
        parser.add_argument('--'+flag, type=int, default=default)
    args = parser.parse_args()
    run(Config(seeds=tuple(args.seeds), train_windows=args.train_windows, val_windows=args.val_windows,
               test_windows=args.test_windows, epochs=args.epochs, min_epochs=args.min_epochs), args.out_dir)


if __name__ == '__main__': main()
