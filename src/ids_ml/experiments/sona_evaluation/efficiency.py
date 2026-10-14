"""Reproducible CPU efficiency measurements on identical inputs and settings.

    python -m ids_ml.experiments.sona_evaluation.efficiency --out FILE.json --replicate-dir REP
        [--candidate MODULE:FUNCTION] [--threads 1 2] [--lengths 64 256 1024] [--batches 1 32 256]

Each (adapter, threads) runs in a fresh subprocess with thread environment
variables set before torch is imported. Benchmark windows are synthetic
development fixtures of exact lengths, written once and reused for every
adapter. Preprocessing (prepare) and model (score_prepared) are timed
separately. Load average is recorded around every measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

T_START = time.perf_counter()  # cold timing begins at interpreter module load

WARMUP, REPEATS, COLD_REPEATS = 3, 20, 5
REFERENCE = {'batch': 32, 'length': 256, 'threads': 2}
LOAD_FLAG = 1.0
RSS_FLOOR_KIB = 16 * 1024  # increments below this are measurement noise for ratios


def adapter_specs(rep_dir, candidate):
    specs = {name: {'kind': 'saved', 'name': name} for name in ('beta1_saved', 'beta2_saved', 'sona_public_saved')}
    for name in ('beta2_counterpart', 'sona_counterpart', 'pooled_mlp', 'summary_boosting'):
        if (Path(rep_dir) / 'models' / name / 'DONE').exists():
            specs[name] = {'kind': 'retrained', 'directory': str(Path(rep_dir) / 'models' / name)}
    if candidate and (Path(rep_dir) / 'models' / 'candidate_fit' / 'SAVED_BY_ADAPTER').exists():
        specs['candidate_fit'] = {'kind': 'candidate', 'spec': candidate,
                                  'directory': str(Path(rep_dir) / 'models' / 'candidate_fit')}
    return specs


def build_inputs(directory, lengths, batches, seed):
    """Windows of exact length, built by joining development windows end to end."""
    import numpy as np
    import pandas as pd
    from . import cases
    from .protocol import INFERENCE_COLUMNS
    need = max(batches)
    pool, _ = cases.generate_split(seed, 'efficiency_pool', windows=max(16, need // 2 * 2), condition='long_sparse')
    pool = cases.inference_frame(pool)
    parts = [g for _, g in pool.groupby('session_id', sort=False)]
    rng = np.random.default_rng(seed)
    for length in lengths:
        windows = []
        for w in range(need):
            rows, total, clock = [], 0, 0.
            while total < length:
                piece = parts[int(rng.integers(len(parts)))].copy()
                piece['event_time'] = clock + (piece.event_time - piece.event_time.iloc[0]) + (30. if total else 0.)
                rows.append(piece)
                total += len(piece)
                clock = float(piece.event_time.iloc[-1])
            window = pd.concat(rows, ignore_index=True).iloc[:length].copy()
            window['session_id'] = f'eff-{length}-{w:04d}'
            window['flow_id'] = [f'eff-{length}-{w:04d}-{i:05d}' for i in range(length)]
            window['prior_gap_s'] = np.r_[0., np.diff(window.event_time.to_numpy())]
            windows.append(window)
        frame = pd.concat(windows, ignore_index=True).loc[:, list(INFERENCE_COLUMNS)]
        frame.to_parquet(Path(directory) / f'length-{length}.parquet', index=False)


def _load(spec):
    from . import adapters as A
    if spec['kind'] == 'saved':
        adapter = A.SAVED[spec['name']]()
    elif spec['kind'] == 'retrained':
        adapter = A.load_retrained(spec['directory'])
    else:
        adapter = type(A.load_candidate(spec['spec'], {'mode': 'fit'})).load(spec['directory'])
    return adapter


def _artifact_bytes(adapter, spec):
    directory = spec.get('directory') or str(getattr(adapter, 'directory', ''))
    if not directory:
        return None
    files = [p for p in Path(directory).iterdir() if p.is_file() and p.suffix in ('.pt', '.pkl', '.json')]
    return {'total': sum(p.stat().st_size for p in files), 'files': {p.name: p.stat().st_size for p in files}}


def _status_kib(field):
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(field + ':'):
            return int(line.split()[1])
    return None


def _reset_peak():
    """Linux: writing 5 to clear_refs resets VmHWM, isolating the peak of what follows."""
    try:
        Path('/proc/self/clear_refs').write_text('5')
        return True
    except OSError:
        return False


def child(args):
    """Runs inside a fresh process: import, load, measure, print JSON."""
    import resource
    import numpy as np
    import pandas as pd
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args['threads'])
    imported = time.perf_counter()
    spec = args['spec']
    adapter = _load(spec)
    loaded = time.perf_counter()
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    results = {'import_seconds': imported - T_START, 'load_seconds': loaded - imported,
               'rss_after_load_kib': rss_after_load, 'configs': []}
    modules = adapter.modules() if hasattr(adapter, 'modules') else []
    results['parameter_count'] = int(sum(p.numel() for m in modules for p in m.parameters())) if modules else adapter.describe().get('parameter_count')
    results['artifact_bytes'] = _artifact_bytes(adapter, spec)
    configs = sorted(args['configs'], key=lambda c: c['batch'] * c['length'])
    with threadpool_limits(limits=args['threads']):
        for config in configs:
            frame = pd.read_parquet(Path(args['inputs']) / f'length-{config["length"]}.parquet')
            ids = frame.session_id.unique()[:config['batch']]
            batch = frame[frame.session_id.isin(set(ids))].reset_index(drop=True)
            load_before = os.getloadavg()[0]
            reset = _reset_peak()
            rss_before = _status_kib('VmRSS')
            first = None
            prep, model = [], []
            for i in range(WARMUP + REPEATS):
                t0 = time.perf_counter()
                prepared = adapter.prepare(batch)
                t1 = time.perf_counter()
                scores = adapter.score_prepared(prepared)
                t2 = time.perf_counter()
                if first is None:
                    first = {'prepare': t1 - t0, 'model': t2 - t1, 'since_process_start': t2 - T_START}
                if i >= WARMUP:
                    prep.append(t1 - t0)
                    model.append(t2 - t1)
            assert len(scores) == config['batch'] and np.isfinite(scores).all()
            total = np.array(prep) + np.array(model)
            results['configs'].append({**config, 'first_call': first,
                'prepare_median_s': float(np.median(prep)), 'prepare_p90_s': float(np.quantile(prep, .9)),
                'model_median_s': float(np.median(model)), 'model_p90_s': float(np.quantile(model, .9)),
                'total_median_s': float(np.median(total)), 'total_p90_s': float(np.quantile(total, .9)),
                'windows_per_s': config['batch'] / float(np.median(total)),
                'flows_per_s': config['batch'] * config['length'] / float(np.median(total)),
                'model_windows_per_s': config['batch'] / float(np.median(model)),
                'peak_rss_kib_cumulative': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'inference_peak_increment_kib': (_status_kib('VmHWM') - rss_before) if reset else None,
                'rss_before_kib': rss_before,
                'load_avg_before': load_before, 'load_avg_after': os.getloadavg()[0]})
    print('RESULT ' + json.dumps(results))


def run_child(spec, threads, configs, inputs, cpus):
    env = {**os.environ, 'OMP_NUM_THREADS': str(threads), 'MKL_NUM_THREADS': str(threads),
           'OPENBLAS_NUM_THREADS': str(threads), 'PYTHONWARNINGS': 'ignore'}
    payload = json.dumps({'spec': spec, 'threads': threads, 'configs': configs, 'inputs': str(inputs)})
    command = [sys.executable, '-m', 'ids_ml.experiments.sona_evaluation.efficiency', '--child', payload]
    if cpus:
        command = ['taskset', '-c', cpus] + command
    done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    line = [l for l in done.stdout.splitlines() if l.startswith('RESULT ')]
    if done.returncode or not line:
        raise RuntimeError(f'efficiency child failed for {spec}: {done.stderr[-2000:]}')
    return json.loads(line[-1][7:])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--child', default=None)
    p.add_argument('--out', type=Path)
    p.add_argument('--replicate-dir', type=Path)
    p.add_argument('--candidate', default=None)
    p.add_argument('--threads', type=int, nargs='+', default=[1, 2])
    p.add_argument('--lengths', type=int, nargs='+', default=[64, 256, 1024])
    p.add_argument('--batches', type=int, nargs='+', default=[1, 32, 256])
    p.add_argument('--cpus', default='8-11', help='taskset CPU list for children (empty to disable)')
    p.add_argument('--only', nargs='*', default=None)
    a = p.parse_args(argv)
    if a.child:
        child(json.loads(a.child))
        return
    import platform
    import torch
    from .audit import cpu_model
    from .protocol import DEV_SEED
    inputs = a.out.parent / (a.out.stem + '_inputs')
    inputs.mkdir(parents=True, exist_ok=True)
    if not all((inputs / f'length-{L}.parquet').exists() for L in a.lengths):
        build_inputs(inputs, a.lengths, a.batches, DEV_SEED)
    specs = adapter_specs(a.replicate_dir, a.candidate)
    if a.only:
        specs = {k: v for k, v in specs.items() if k in a.only}
    configs = [{'batch': b, 'length': L} for L in a.lengths for b in a.batches]
    previous = json.loads(a.out.read_text()) if a.out.exists() else {}
    report = {'machine': {'cpu': cpu_model(), 'logical_cores': os.cpu_count(), 'python': platform.python_version(),
                          'torch': torch.__version__, 'platform': platform.platform(), 'taskset_cpus': a.cpus},
              'method': {'warmup': WARMUP, 'repeats': REPEATS, 'cold_repeats': COLD_REPEATS, 'reference': REFERENCE,
                         'load_flag_threshold': LOAD_FLAG,
                         'peak_rss_note': 'inference_peak_increment_kib: VmHWM after resetting it via /proc/self/clear_refs immediately before the warmup and timed calls of one configuration, minus VmRSS at that moment (input batch already loaded). Reference ratios use dedicated cold processes.'},
              'adapters': previous.get('adapters', {})}
    for name, spec in specs.items():
        for threads in a.threads:
            key = f'{name}@{threads}'
            if key in report['adapters']:
                continue
            print(f'measuring {key}', flush=True)
            warm = run_child(spec, threads, configs, inputs, a.cpus)
            # Cold starts (fresh process, import, load, first call) at the reference thread count only.
            cold = [run_child(spec, threads, [{'batch': REFERENCE['batch'], 'length': REFERENCE['length']}], inputs, a.cpus)
                    for _ in range(COLD_REPEATS if threads == REFERENCE['threads'] else 0)]
            report['adapters'][key] = {'spec': spec, 'threads': threads, 'warm': warm,
                                       'cold': [{'import_s': c['import_seconds'], 'load_s': c['load_seconds'],
                                                 'first_call': c['configs'][0]['first_call'],
                                                 'reference_peak_rss_kib': c['configs'][0]['peak_rss_kib_cumulative'],
                                                 'inference_peak_increment_kib': c['configs'][0]['inference_peak_increment_kib'],
                                                 'rss_after_load_kib': c['rss_after_load_kib']} for c in cold]}
            a.out.write_text(json.dumps(report, indent=2))
    report['ratios'] = ratios(report)
    report['flags'] = [f'{k} L{c["length"]} B{c["batch"]}: load {c["load_avg_before"]:.2f}->{c["load_avg_after"]:.2f}'
                       for k, v in report['adapters'].items() for c in v['warm']['configs']
                       if max(c['load_avg_before'], c['load_avg_after']) > LOAD_FLAG]
    a.out.write_text(json.dumps(report, indent=2))
    print(f'wrote {a.out}')


def reference_numbers(entry):
    import numpy as np
    config = next(c for c in entry['warm']['configs'] if c['batch'] == REFERENCE['batch'] and c['length'] == REFERENCE['length'])
    # Peak RSS reached during warmup and timed calls (VmHWM reset just before), minus RSS before them.
    peaks = [c['inference_peak_increment_kib'] for c in entry['cold']] or [config['inference_peak_increment_kib']]
    return {'total_median_s': config['total_median_s'], 'model_median_s': config['model_median_s'],
            'peak_rss_increment_kib': float(np.median(peaks))}


def ratios(report):
    """G4 ratios versus the public Sona checkpoint at the reference configuration."""
    base_key = f'sona_public_saved@{REFERENCE["threads"]}'
    if base_key not in report['adapters']:
        return {}
    base = reference_numbers(report['adapters'][base_key])
    out = {}
    for key, entry in report['adapters'].items():
        name, threads = key.rsplit('@', 1)
        if int(threads) != REFERENCE['threads']:
            continue
        mine = reference_numbers(entry)
        out[name] = {'latency': mine['total_median_s'] / base['total_median_s'],
                     'model_latency': mine['model_median_s'] / base['model_median_s'],
                     'peak_rss_increment': max(mine['peak_rss_increment_kib'], RSS_FLOOR_KIB) / max(base['peak_rss_increment_kib'], RSS_FLOOR_KIB),
                     'reference': mine, 'baseline': base}
    return out


if __name__ == '__main__':
    main()
