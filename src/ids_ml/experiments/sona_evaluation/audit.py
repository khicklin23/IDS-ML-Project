"""Independent audit of one replicate: data, identities, bounds, labels, fitting, replay."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ids_ml.features import build_preprocessor
from . import adapters as A
from . import cases
from .protocol import INFERENCE_COLUMNS, record_hash, suite_source_hashes

ROOT = A.ROOT
PUBLIC_FILES = ('data/generated/showcase_v4/artifact_sequence/model.pt',
                'data/generated/showcase_v4/artifact_sequence/preprocessor.pkl',
                'data/generated/showcase_v4/artifact_sequence/metadata.json',
                'data/generated/showcase_v4/report.json',
                'data/generated/model_versions/v1/model.pt', 'data/generated/model_versions/v1/preprocessor.pkl',
                'data/generated/model_versions/v2/model.pt', 'data/generated/model_versions/v2/preprocessor.pkl',
                'ui/report.json', 'ui/research.json', 'ui/versions.json', 'CURRENT_RELEASE.txt')
REPLAY_SETS = ('familiar', 'chain_familiar')
REPLAY_TOLERANCE = 1e-6


def cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return 'unknown'


def public_artifact_hashes():
    """SHA-256 of public artifacts plus a size/mtime snapshot of earlier generated studies."""
    out = {p: A.file_hash(ROOT / p) for p in PUBLIC_FILES if (ROOT / p).exists()}
    for release in sorted(ROOT.glob('Release_*.html')):
        out[release.name] = A.file_hash(release)
    generated = ROOT / 'data/generated'
    for path in sorted(generated.rglob('*')):
        top = path.relative_to(generated).parts[0]
        if path.is_file() and not top.startswith(('sona_evaluation', 'sona_candidate')):
            st = path.stat()
            out[f'stat:{path.relative_to(ROOT)}'] = f'{st.st_size}:{int(st.st_mtime)}'
    return out


def preserved(before, after):
    """Earlier files unchanged and present; new files elsewhere are allowed."""
    changed = sorted(k for k, v in before.items() if after.get(k) != v)
    return {'passed': not changed, 'changed_or_missing': changed[:20], 'checked': len(before)}


def candidate_freeze(factory):
    """Hash candidate sources, its declared artifacts and its document before sealing."""
    hashes = dict(A.candidate_source_hashes())
    doc = ROOT / 'docs/SONA_CANDIDATE.md'
    if doc.exists():
        hashes['docs/SONA_CANDIDATE.md'] = A.file_hash(doc)
    describe = {}
    if factory is not None:
        adapter = factory()
        describe = json.loads(json.dumps(adapter.describe(), default=str))
        for path in describe.get('artifact_paths', []):
            if Path(path).is_file():
                hashes[str(path)] = A.file_hash(path)
    return {'hashes': hashes, 'describe': describe}


def _identity_check(rep_dir, splits):
    seen = {c: {} for c in ('session_id', 'flow_id', 'src_host_id')}
    overlaps = []
    for split in splits:
        frame = pd.read_parquet(rep_dir / 'data' / f'{split}.parquet', columns=list(seen))
        for column, owners in seen.items():
            values = set(frame[column])
            for other, other_values in owners.items():
                if values & other_values:
                    overlaps.append(f'{column}:{split}&{other}')
            owners[split] = values
    return {'passed': not overlaps, 'overlaps': overlaps}


def _bounds_check(rep_dir, splits):
    bad = []
    for split in splits:
        frame = pd.read_parquet(rep_dir / 'data' / f'{split}.parquet', columns=['session_id', 'event_time', 'prior_gap_s'])
        key = pd.read_parquet(rep_dir / 'data' / f'{split}_key.parquet').set_index('session_id')
        start = frame.session_id.map(key.observation_start_s)
        rel = frame.event_time - start
        duration = frame.session_id.map(key.duration_s)
        if (rel < -1e-6).any() or (rel >= duration).any():
            bad.append(f'{split}:outside_capture')
        if (frame.prior_gap_s < 0).any() or not np.isfinite(frame.prior_gap_s).all():
            bad.append(f'{split}:invalid_gap')
        if (frame.groupby('session_id', sort=False).event_time.diff().fillna(0) < -1e-9).any():
            bad.append(f'{split}:not_chronological')
    return {'passed': not bad, 'violations': bad}


def _label_exclusion_check(models, sample):
    """Every adapter must refuse frames with extra (label) columns."""
    refused = {}
    leaky = sample.assign(window_target=1)
    for name, adapter in models.items():
        try:
            A.score_frame(adapter, leaky, sample.session_id.unique().tolist())
            refused[name] = False
        except ValueError:
            refused[name] = True
    return {'passed': all(refused.values()) and tuple(cases.inference_frame(sample).columns) == INFERENCE_COLUMNS,
            'refused_label_columns': refused}


def _train_only_check(rep_dir, models):
    train, key = cases_load(rep_dir, 'train')
    results = {}
    for name, adapter in models.items():
        processor = getattr(adapter, 'processor', None)
        if processor is None or adapter.provenance != 'retrained' or name.startswith('candidate'):
            continue
        subset = train
        if name in A.TRAIN_FILTERS:
            keep = set(key.session_id[A.TRAIN_FILTERS[name](key)])
            subset = train[train.session_id.isin(keep)]
        reference = build_preprocessor(subset.loc[:, list(INFERENCE_COLUMNS)])
        a, b = processor.named_transformers_['numeric'], reference.named_transformers_['numeric']
        same = (np.allclose(a.mean_, b.mean_, rtol=0, atol=1e-9) and np.allclose(a.scale_, b.scale_, rtol=0, atol=1e-9)
                and [list(c) for c in processor.named_transformers_['categorical'].categories_]
                == [list(c) for c in reference.named_transformers_['categorical'].categories_])
        results[name] = bool(same)
    return {'passed': all(results.values()), 'models': results}


def cases_load(rep_dir, split):
    return (pd.read_parquet(rep_dir / 'data' / f'{split}.parquet'),
            pd.read_parquet(rep_dir / 'data' / f'{split}_key.parquet'))


def _replay_check(rep_dir, models, candidate, candidate_saved):
    out = {}
    for split in REPLAY_SETS:
        frame, key = cases_load(rep_dir, split)
        inf = cases.inference_frame(frame)
        ids = key.session_id.tolist()
        with np.load(rep_dir / 'scores' / f'{split}.npz') as saved:
            for name in models:
                if name in A.SAVED:
                    fresh = A.SAVED[name]()
                elif name == 'candidate_saved':
                    fresh = candidate_saved()
                elif name == 'candidate_fit':
                    directory = rep_dir / 'models' / name
                    if not (directory / 'SAVED_BY_ADAPTER').exists():
                        out[f'{split}:{name}'] = {'passed': None, 'note': 'no save/load; replay not possible'}
                        continue
                    fresh = type(candidate()).load(directory)
                else:
                    fresh = A.load_retrained(rep_dir / 'models' / name)
                diff = float(np.max(np.abs(A.score_frame(fresh, inf, ids) - saved[name])))
                out[f'{split}:{name}'] = {'passed': diff <= REPLAY_TOLERANCE, 'max_abs_diff': diff}
    return {'passed': all(v['passed'] is not False for v in out.values()), 'checks': out}


def prefix_causality(adapter, flows, windows=40):
    """Per-flow features of a prefix must equal the same rows computed on the full window."""
    ids = flows.session_id.unique()[:windows]
    sub = flows[flows.session_id.isin(ids)].reset_index(drop=True)
    full = adapter.flow_features(sub)
    keep = np.zeros(len(sub), bool)
    for _, positions in sub.groupby('session_id', sort=False).indices.items():
        keep[positions[:max(1, len(positions) // 2)]] = True
    prefix = adapter.flow_features(sub[keep].reset_index(drop=True))
    return bool(np.allclose(full[keep], prefix, rtol=0, atol=1e-9))


def _causality_check(rep_dir, models):
    frame, _ = cases_load(rep_dir, 'familiar')
    inf = cases.inference_frame(frame)
    results = {}
    for name, adapter in models.items():
        if getattr(adapter, 'causal_features', False):
            if hasattr(adapter, 'flow_features'):
                results[name] = prefix_causality(adapter, inf)
            else:
                results[name] = None  # claim without a verifiable interface
    return {'passed': all(v is not False for v in results.values()), 'models': results,
            'note': 'None means a causal claim without flow_features(); recorded as unverified.'}


def _threshold_check(rep_dir):
    frozen = json.loads((rep_dir / 'thresholds.json').read_text())
    intact = record_hash(frozen['thresholds']) == frozen['sha256_of_thresholds']
    written = (rep_dir / 'thresholds.json').stat().st_mtime
    before_tests = all(p.stat().st_mtime >= written - 1 for p in (rep_dir / 'scores').glob('*.npz') if p.name != 'val.npz')
    return {'passed': intact and before_tests, 'hash_intact': intact, 'test_scores_not_older_than_thresholds': before_tests}


def audit_replicate(rep_dir, models, data_hashes, hashes, candidate=None, candidate_saved=None):
    splits = list(data_hashes)
    rehash = {s: {'flows': A.file_hash(rep_dir / 'data' / f'{s}.parquet'),
                  'key': A.file_hash(rep_dir / 'data' / f'{s}_key.parquet')} for s in splits}
    sample = cases.inference_frame(cases_load(rep_dir, 'val')[0]).head(500)
    sample = sample[sample.session_id.isin(sample.session_id.unique()[:-1])]
    checks = {
        'data_hashes': {'passed': all(rehash[s]['flows'] == data_hashes[s]['flows'] and rehash[s]['key'] == data_hashes[s]['key'] for s in splits)},
        'disjoint_identities': _identity_check(rep_dir, splits),
        'capture_bounds_and_order': _bounds_check(rep_dir, splits),
        'label_exclusion': _label_exclusion_check(models, sample),
        'train_only_preprocessing': _train_only_check(rep_dir, models),
        'frozen_thresholds': _threshold_check(rep_dir),
        'prediction_replay': _replay_check(rep_dir, models, candidate, candidate_saved),
        'prefix_causality': _causality_check(rep_dir, models),
        'suite_source_unchanged': {'passed': suite_source_hashes() == hashes['suite']},
        'public_artifacts_preserved': preserved(hashes['public_before'], public_artifact_hashes()),
    }
    return {'status': 'passed' if all(c['passed'] for c in checks.values()) else 'failed', 'checks': checks}
