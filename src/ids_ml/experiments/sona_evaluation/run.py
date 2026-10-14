"""Resumable runner: generate -> fit -> freeze thresholds -> score -> audit.

    python -m ids_ml.experiments.sona_evaluation.run --out-dir DIR --mode smoke|baseline|final
        [--candidate MODULE:FUNCTION] [--candidate-arg key=value ...] [--candidate-mode fit|saved|both]

Every stage writes its outputs before the next starts; reruns skip completed
stages after checking the protocol and suite-source hashes. Final mode draws
sealed seeds from the OS only after hashing the candidate source/artifacts.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import platform
import secrets
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import adapters as A
from . import audit, cases
from .metrics import at_threshold, budget_key, ranking, subtype_rates, threshold_at_budget, triple_metrics
from .protocol import (BASELINE_SEEDS, BUDGETS, CHAIN_SPLITS, CONDITIONS, DEV_SEED, SMOKE_RECIPE, SMOKE_SIZES,
                       TRUNCATIONS, USED_SEEDS, Sizes, TrainRecipe, protocol_record, record_hash,
                       suite_source_hashes)

CONTROL_SOURCES = ('familiar', 'chain_familiar')


def log(message):
    print(f'[{dt.datetime.now().strftime("%H:%M:%S")}] {message}', flush=True)


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=float))
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- stages

def generate(rep_dir, seed, sizes, namespace):
    data = rep_dir / 'data'
    done = data / 'hashes.json'
    if done.exists():
        return read_json(done)
    if data.exists():
        shutil.rmtree(data)  # partial generation is regenerated deterministically
    data.mkdir(parents=True)
    hashes = {}
    for split, kwargs in cases.split_plan(sizes):
        frame, key = cases.generate_split(seed, split, namespace=f'{namespace}-{split}', **kwargs)
        frame.to_parquet(data / f'{split}.parquet', index=False)
        key.to_parquet(data / f'{split}_key.parquet', index=False)
        hashes[split] = {'flows': A.file_hash(data / f'{split}.parquet'),
                         'key': A.file_hash(data / f'{split}_key.parquet'),
                         'windows': len(key), 'flow_rows': len(frame)}
        log(f'seed {seed} generated {split}: {len(key):,} windows, {len(frame):,} flows')
    write_json(done, hashes)
    return hashes


def load_split(rep_dir, split):
    return (pd.read_parquet(rep_dir / 'data' / f'{split}.parquet'),
            pd.read_parquet(rep_dir / 'data' / f'{split}_key.parquet'))


def fit_models(rep_dir, seed, recipe, model_names, candidate):
    train, train_key = load_split(rep_dir, 'train')
    val, _ = load_split(rep_dir, 'val')
    fitted = {}
    for name in model_names:
        directory = rep_dir / 'models' / name
        if (directory / 'DONE').exists():
            continue
        started = time.perf_counter()
        if name == 'candidate_fit':
            adapter = candidate()
        else:
            adapter = A.RETRAINED[name](recipe)
        subset = train
        if name in A.TRAIN_FILTERS:
            keep = set(train_key.session_id[A.TRAIN_FILTERS[name](train_key)])
            subset = train[train.session_id.isin(keep)].reset_index(drop=True)
        log(f'seed {seed} fitting {name} on {subset.session_id.nunique():,} windows')
        adapter.fit(subset, val, seed)
        elapsed = time.perf_counter() - started
        if directory.exists():
            shutil.rmtree(directory)
        if hasattr(adapter, 'save'):
            adapter.save(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if name == 'candidate_fit' and hasattr(adapter, 'save') and hasattr(type(adapter), 'load'):
            (directory / 'SAVED_BY_ADAPTER').write_text('ok')
        write_json(directory / 'fit.json', {'fit_seconds': elapsed, 'train_windows': int(subset.session_id.nunique()),
                                            'describe': _describe(adapter)})
        (directory / 'DONE').write_text('ok')
        fitted[name] = adapter
    return fitted


def _describe(adapter):
    d = adapter.describe()
    return json.loads(json.dumps(d, default=str))


def load_models(rep_dir, model_names, saved_names, fitted, candidate, candidate_saved):
    """Score with models reloaded from disk, so scoring itself exercises persistence."""
    models = {}
    for name in model_names:
        directory = rep_dir / 'models' / name
        if name != 'candidate_fit':
            models[name] = A.load_retrained(directory)
            continue
        if (directory / 'SAVED_BY_ADAPTER').exists():
            models[name] = type(fitted.get(name) or candidate()).load(directory)
        elif name in fitted:
            models[name] = fitted[name]  # no save(): in-process only, flagged by the audit
        else:
            raise RuntimeError('Candidate has no save()/load(); remove its DONE marker to refit.')
    for name in saved_names:
        models[name] = A.SAVED[name]()
    if candidate_saved is not None:
        models['candidate_saved'] = candidate_saved()
    return models


def control_frames(frame):
    yield 'shuffled', cases.shuffle_interior(frame, 9001)
    for k in TRUNCATIONS:
        yield f'trunc{k}', cases.truncate_last(frame, k)


def evaluation_sets():
    sets = [(c, c, None) for c in CONDITIONS] + [(c, c, None) for c in CHAIN_SPLITS]
    for source in CONTROL_SOURCES:
        for control in ('shuffled', *[f'trunc{k}' for k in TRUNCATIONS]):
            sets.append((f'{source}:{control}', source, control))
    return sets


def score_all(rep_dir, models, seed):
    scores_dir = rep_dir / 'scores'
    scores_dir.mkdir(exist_ok=True)
    # 1. Validation scores and frozen thresholds, written before any test scoring.
    thresholds_path = rep_dir / 'thresholds.json'
    val, val_key = load_split(rep_dir, 'val')
    val_ids = val_key.session_id.tolist()
    val_inf = cases.inference_frame(val)
    if thresholds_path.exists():
        frozen = read_json(thresholds_path)
    else:
        frozen = {'thresholds': {}, 'written_at': None}
    val_path = scores_dir / 'val.npz'
    val_scores = dict(np.load(val_path)) if val_path.exists() else {}
    for name, adapter in models.items():
        if name in frozen['thresholds']:
            continue
        val_scores[name] = A.score_frame(adapter, val_inf, val_ids)
        frozen['thresholds'][name] = {budget_key(b): threshold_at_budget(val_scores[name], val_key.target.to_numpy(), b)
                                      for b in BUDGETS}
        np.savez_compressed(val_path, session_id=np.array(val_ids), **{k: v for k, v in val_scores.items() if k != 'session_id'})
    if frozen.get('sha256_of_thresholds') != record_hash(frozen['thresholds']):
        # Only new models change this file; existing thresholds are never recomputed.
        frozen['written_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        frozen['sha256_of_thresholds'] = record_hash(frozen['thresholds'])
        write_json(thresholds_path, frozen)
    log(f'seed {seed}: thresholds frozen for {len(frozen["thresholds"])} models')
    # 2. Test, chain and control scoring with the frozen thresholds.
    for set_name, source, control in evaluation_sets():
        path = scores_dir / f'{set_name.replace(":", "__")}.npz'
        existing = dict(np.load(path)) if path.exists() else {}
        missing = [m for m in models if m not in existing]
        if not missing:
            continue
        frame, key = load_split(rep_dir, source)
        if control:
            frame = dict(control_frames(frame))[control]
        inf = cases.inference_frame(frame)
        ids = key.session_id.tolist()
        for name in missing:
            existing[name] = A.score_frame(models[name], inf, ids)
        existing['session_id'] = np.array(ids)
        np.savez_compressed(path, **existing)
        log(f'seed {seed}: scored {set_name} for {len(missing)} models')
    return frozen


def compute_metrics(rep_dir, model_names):
    frozen = read_json(rep_dir / 'thresholds.json')
    if record_hash(frozen['thresholds']) != frozen['sha256_of_thresholds']:
        raise RuntimeError('Frozen thresholds changed after writing.')
    results = {}
    for set_name, source, control in [('val', 'val', None), *evaluation_sets()]:
        _, key = load_split(rep_dir, source)
        frame = None
        with np.load(rep_dir / 'scores' / f'{set_name.replace(":", "__")}.npz') as saved:
            assert saved['session_id'].tolist() == key.session_id.tolist()
            scores = {m: saved[m] for m in model_names}
        y = key.target.to_numpy()
        attack_counts = None
        if source not in CHAIN_SPLITS:
            frame, _ = load_split(rep_dir, source)
            per = frame.groupby('session_id', sort=False).attack_flow.agg(['size', 'sum']).loc[key.session_id]
            flow_counts, attack_counts = per['size'].to_numpy(), per['sum'].to_numpy()
        results[set_name] = {}
        for m in model_names:
            entry = {'ranking': ranking(scores[m], y), 'budgets': {}}
            for b, th in frozen['thresholds'][m].items():
                entry['budgets'][b] = at_threshold(scores[m], y, th, *((flow_counts, attack_counts) if attack_counts is not None else ()))
                if source in CONDITIONS and control is None:
                    entry['budgets'][b]['subtypes'] = subtype_rates(scores[m], key, th)
            if source in CHAIN_SPLITS or (key.triple_id != '').any():
                if (key.triple_id != '').any():
                    entry['triples'] = triple_metrics(scores[m], key)[0]
            results[set_name][m] = entry
    write_json(rep_dir / 'metrics.json', results)
    return results


# --------------------------------------------------------------------------- orchestration

def replicate(out, seed, sizes, recipe, model_names, saved_names, candidate, candidate_saved, namespace, hashes):
    rep_dir = out / f'replicate-{seed}'
    rep_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(recipe.threads)
    started = time.perf_counter()
    data_hashes = generate(rep_dir, seed, sizes, namespace)
    fitted = fit_models(rep_dir, seed, recipe, model_names, candidate)
    models = load_models(rep_dir, model_names, saved_names, fitted, candidate, candidate_saved)
    score_all(rep_dir, models, seed)
    names = list(models)
    metrics = compute_metrics(rep_dir, names)
    report = audit.audit_replicate(rep_dir, models, data_hashes, hashes, candidate, candidate_saved)
    write_json(rep_dir / 'audit.json', report)
    write_json(rep_dir / 'replicate.json', {'seed': seed, 'models': names, 'elapsed_seconds': time.perf_counter() - started,
                                            'audit_status': report['status']})
    log(f'seed {seed}: replicate complete, audit {report["status"]}')
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--mode', choices=('smoke', 'baseline', 'final'), required=True)
    p.add_argument('--candidate', default=None)
    p.add_argument('--candidate-arg', action='append', default=[])
    p.add_argument('--candidate-mode', choices=('fit', 'saved', 'both'), default='fit')
    p.add_argument('--replicates', type=int, default=3)
    p.add_argument('--skip-models', nargs='*', default=[])
    a = p.parse_args(argv)
    sizes, recipe = (SMOKE_SIZES, SMOKE_RECIPE) if a.mode == 'smoke' else (Sizes(), TrainRecipe())
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    record = protocol_record(sizes, recipe)
    source_hashes = suite_source_hashes()
    stamp = {'protocol': record, 'protocol_sha256': record_hash(record), 'suite_source_sha256': source_hashes}
    stamp_path = out / 'protocol.json'
    if stamp_path.exists():
        previous = read_json(stamp_path)
        if previous['protocol_sha256'] != stamp['protocol_sha256'] or previous['suite_source_sha256'] != source_hashes:
            raise SystemExit('Protocol or suite source changed since this run started; use a new --out-dir.')
    else:
        write_json(stamp_path, stamp)
    kwargs = dict(item.split('=', 1) for item in a.candidate_arg)
    candidate = candidate_saved = None
    if a.candidate:
        if a.candidate_mode in ('fit', 'both'):
            candidate = lambda: A.load_candidate(a.candidate, {**kwargs, 'mode': 'fit'} if 'mode' not in kwargs else kwargs)
        if a.candidate_mode in ('saved', 'both'):
            candidate_saved = lambda: A.load_candidate(a.candidate, {**kwargs, 'mode': 'saved'} if 'mode' not in kwargs else kwargs)
    model_names = [m for m in A.RETRAINED if m not in a.skip_models] + (['candidate_fit'] if candidate else [])
    saved_names = [m for m in A.SAVED if m not in a.skip_models]
    public_before = audit.public_artifact_hashes()
    sealing_path = out / 'sealing.json'
    if a.mode == 'final':
        if not a.candidate:
            raise SystemExit('Final mode requires a candidate.')
        if sealing_path.exists():
            sealing = read_json(sealing_path)
        else:
            freeze = audit.candidate_freeze(candidate or candidate_saved)
            freeze['frozen_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_json(sealing_path, {'candidate_freeze': freeze, 'seeds': None})
            seeds = []
            while len(seeds) < a.replicates:
                s = 100000 + secrets.randbelow(2 ** 30)
                if s not in USED_SEEDS and s not in BASELINE_SEEDS and s not in seeds and s != DEV_SEED:
                    seeds.append(s)
            sealing = {'candidate_freeze': freeze, 'seeds': seeds,
                       'seeds_drawn_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                       'note': 'Seeds drawn from the OS after the candidate freeze was written.'}
            write_json(sealing_path, sealing)
        seeds, namespace = sealing['seeds'], 'sealed'
    elif a.mode == 'baseline':
        seeds, namespace = list(BASELINE_SEEDS[:a.replicates]), 'baseline'
    else:
        seeds, namespace = [DEV_SEED], 'smoke'
    env = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
           'pandas': pd.__version__, 'platform': platform.platform(), 'cpu': audit.cpu_model(),
           'logical_cores': os.cpu_count(), 'threads': recipe.threads}
    write_json(out / 'run.json', {'mode': a.mode, 'seeds': seeds, 'models': model_names + saved_names,
                                  'candidate': a.candidate, 'candidate_args': kwargs, 'candidate_mode': a.candidate_mode,
                                  'environment': env, 'started_or_resumed_at': dt.datetime.now(dt.timezone.utc).isoformat()})
    for seed in seeds:
        replicate(out, seed, sizes, recipe, model_names, saved_names, candidate, candidate_saved,
                  f'{namespace}-{seed}', {'suite': source_hashes, 'public_before': public_before})
    after = audit.public_artifact_hashes()
    final = {'public_artifacts_unchanged': after == public_before, 'suite_source_unchanged': suite_source_hashes() == source_hashes}
    if a.mode == 'final':
        final['candidate_unchanged'] = audit.candidate_freeze(candidate or candidate_saved)['hashes'] == read_json(sealing_path)['candidate_freeze']['hashes']
    write_json(out / 'run_audit.json', final)
    log(f'run complete: {final}')


if __name__ == '__main__':
    main()
