"""Adapter contract plus saved-checkpoint and retrained reference adapters.

Contract (see docs/SONA_EVALUATION.md section 3):

    name, provenance ('saved_checkpoint' | 'retrained'), causal_features (bool claim)
    fit(train, val, seed)        optional; train/val carry window_target and attack_flow
    prepare(flows)               preprocessing only; flows hold INFERENCE_COLUMNS only
    score_prepared(prepared)     model only; one finite score per window, higher = suspicious
    window_order(prepared)       window ids aligned with the scores
    describe()                   JSON-able: parameter_count, artifact_paths, notes
    save(dir) / load(dir)        retrained adapters, for replay audits
    flow_features(flows)         optional; per-flow features for prefix-causality audits
    modules()                    optional; torch modules for verified parameter counts

A candidate exposes build_evaluation_adapter(**kwargs) in a module under
ids_ml.experiments.sona_candidate; `load_candidate` imports it read-only.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from ids_ml.data.schema import FEATURE_COLUMNS
from ids_ml.features import build_preprocessor, transform_features
from ids_ml.models.classifier import MLPClassifier
from ids_ml.models.sequence import SequenceClassifier, SequenceTrainConfig, train_sequence_model
from ids_ml.models.train import TrainConfig, train_multiclass_model
from ids_ml.experiments.mixed_scan import PooledMLP, train_pool
from .protocol import INFERENCE_COLUMNS

ROOT = Path(__file__).resolve().parents[4]
SAVED_PATHS = {
    'beta1_saved': ROOT / 'data/generated/model_versions/v1',
    'beta2_saved': ROOT / 'data/generated/model_versions/v2',
    'sona_public_saved': ROOT / 'data/generated/showcase_v4/artifact_sequence',
}
REQUIRED = ('name', 'provenance', 'prepare', 'score_prepared', 'window_order', 'describe')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_inference_frame(flows):
    if tuple(flows.columns) != INFERENCE_COLUMNS:
        raise ValueError(f'Adapter input must have exactly {INFERENCE_COLUMNS}.')


def groups_of(flows):
    groups = flows.groupby('session_id', sort=False).indices
    return list(groups.keys()), list(groups.values())


def parameter_count(modules):
    return int(sum(p.numel() for m in modules for p in m.parameters()))


def _interior_permutation(window_id, length):
    """Deterministic per-window interior shuffle keyed only by the opaque id."""
    seed = int(hashlib.sha256(str(window_id).encode()).hexdigest()[:8], 16)
    order = np.arange(length)
    if length > 3:
        order[1:-1] = order[1:-1][np.random.default_rng(seed).permutation(length - 2)]
    return order


class EvaluationAdapter:
    name = 'adapter'
    provenance = 'retrained'
    causal_features = False

    def window_order(self, prepared):
        return list(prepared['ids'])

    def modules(self):
        return []

    def describe(self):
        return {'name': self.name, 'provenance': self.provenance,
                'parameter_count': parameter_count(self.modules()),
                'parameter_count_source': 'verified_modules', 'artifact_paths': [], 'notes': ''}


# --------------------------------------------------------------------------- sequence models

class _SequenceBase(EvaluationAdapter):
    """Absolute-timing per-flow preprocessing (public Sona's mode) + window tensors."""
    causal_features = True
    batch = 256
    shuffle_inputs = False

    def flow_features(self, flows):
        return transform_features(self.processor, flows)

    def prepare(self, flows):
        check_inference_frame(flows)
        features = transform_features(self.processor, flows)
        ids, groups = groups_of(flows)
        windows = [features[g] for g in groups]
        if self.shuffle_inputs:
            windows = [w[_interior_permutation(i, len(w))] for i, w in zip(ids, windows)]
        return {'ids': ids, 'windows': windows}

    def _forward(self, x, lengths):
        return torch.softmax(self.model(x, lengths), dim=-1)

    def score_prepared(self, prepared):
        windows = prepared['windows']
        lengths = np.array([len(w) for w in windows])
        order = np.argsort(lengths, kind='stable')
        scores = np.empty(len(windows))
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(order), self.batch):
                idx = order[start:start + self.batch]
                cap = int(lengths[idx].max())
                x = np.zeros((len(idx), cap, windows[0].shape[1]), dtype=np.float32)
                for j, i in enumerate(idx):
                    x[j, :lengths[i]] = windows[i]
                probs = self._forward(torch.from_numpy(x), torch.from_numpy(lengths[idx]))
                scores[idx] = self._score(probs).numpy()
        return scores

    def _score(self, probs):
        return probs[:, 1]

    def modules(self):
        return [self.model]


def pack(prepared, cap=None, shuffle=False):
    windows = prepared['windows']
    lengths = np.array([len(w) for w in windows], dtype=np.int64)
    cap = cap or int(lengths.max())
    X = np.zeros((len(windows), cap, windows[0].shape[1]), dtype=np.float32)
    for i, w in enumerate(windows):
        X[i, :len(w)] = w
    return X, lengths


class SequenceCounterpart(_SequenceBase):
    """Public Sona architecture retrained on the suite: GRU(32) + linear, window targets."""

    def __init__(self, name='sona_counterpart', recipe=None, shuffle_inputs=False):
        self.name, self.recipe, self.shuffle_inputs = name, recipe, shuffle_inputs
        self.processor = self.model = None
        self.history = []

    def fit(self, train, val, seed):
        self.processor = build_preprocessor(train.loc[:, list(INFERENCE_COLUMNS)])
        packs = []
        for frame in (train, val):
            prepared = self.prepare(frame.loc[:, list(INFERENCE_COLUMNS)])
            target = frame.groupby('session_id', sort=False).window_target.first()
            packs.append((*pack(prepared), target.loc[prepared['ids']].to_numpy(dtype=np.int64)))
        r = self.recipe
        config = SequenceTrainConfig(seed=seed, hidden_dim=r.hidden_dim, max_epochs=r.epochs,
                                     min_epochs=r.min_epochs, patience=r.patience,
                                     batch_size=r.batch_size, class_weight=False)
        self.model, self.history = train_sequence_model(*packs[0], *packs[1], config, 2)
        self.config = dataclasses.asdict(config)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / 'model.pt')
        (directory / 'preprocessor.pkl').write_bytes(pickle.dumps(self.processor))
        meta = {'kind': 'sequence', 'name': self.name, 'shuffle_inputs': self.shuffle_inputs,
                'input_dim': self.model.gru.input_size, 'hidden_dim': self.model.gru.hidden_size,
                'config': self.config, 'history': self.history}
        (directory / 'metadata.json').write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / 'metadata.json').read_text())
        self = cls(meta['name'], None, meta['shuffle_inputs'])
        self.processor = pickle.loads((directory / 'preprocessor.pkl').read_bytes())
        self.model = SequenceClassifier(meta['input_dim'], 2, meta['hidden_dim'])
        self.model.load_state_dict(torch.load(directory / 'model.pt', weights_only=True))
        self.config, self.history, self.directory = meta['config'], meta['history'], directory
        return self

    def describe(self):
        d = super().describe()
        d['notes'] = ('GRU hidden 32, absolute timing, retrained on suite windows'
                      + ('; interior-shuffled train and inference control' if self.shuffle_inputs else ''))
        d['artifact_paths'] = [str(p) for p in sorted(getattr(self, 'directory', Path('/nonexistent')).glob('*'))]
        return d


class SavedSequence(_SequenceBase):
    """Public Sona six-class GRU checkpoint, zero-shot: score = 1 - P(benign)."""
    provenance = 'saved_checkpoint'

    def __init__(self, directory=SAVED_PATHS['sona_public_saved']):
        self.name, self.directory = 'sona_public_saved', Path(directory)
        meta = json.loads((self.directory / 'metadata.json').read_text())
        self.class_names = meta['class_names']
        self.processor = pickle.loads((self.directory / 'preprocessor.pkl').read_bytes())
        state = torch.load(self.directory / 'model.pt', weights_only=True)
        self.model = SequenceClassifier(state['gru.weight_ih_l0'].shape[1], len(self.class_names), meta['hidden_dim'])
        self.model.load_state_dict(state)

    def _score(self, probs):
        return 1. - probs[:, self.class_names.index('benign')]

    def describe(self):
        d = super().describe()
        d['artifact_paths'] = [str(p) for p in sorted(self.directory.glob('*'))]
        d['notes'] = 'Public six-class Sona checkpoint (showcase_v4); not trained on suite windows'
        return d


# --------------------------------------------------------------------------- flow-level models

class FlowVote(EvaluationAdapter):
    """Beta 1/2 MLP on flows; window score = fraction of flows predicted attack.

    Historical Beta sessions used a majority vote; the fraction with a
    validation-frozen threshold is the same order-invariant aggregate, made
    comparable at fixed false-alarm budgets.
    """
    causal_features = True

    def __init__(self, name='beta2_counterpart', recipe=None):
        self.name, self.recipe = name, recipe
        self.processor = self.model = None
        self.history = []

    def flow_features(self, flows):
        return transform_features(self.processor, flows)

    def fit(self, train, val, seed):
        self.processor = build_preprocessor(train.loc[:, list(INFERENCE_COLUMNS)])
        config = TrainConfig(seed=seed, max_epochs=self.recipe.epochs, patience=self.recipe.patience,
                             batch_size=self.recipe.flow_batch_size)
        self.model, self.history = train_multiclass_model(
            transform_features(self.processor, train), train.attack_flow.to_numpy(dtype=np.int64),
            transform_features(self.processor, val), val.attack_flow.to_numpy(dtype=np.int64), config, 2)
        self.config = dataclasses.asdict(config)

    def prepare(self, flows):
        check_inference_frame(flows)
        ids, groups = groups_of(flows)
        return {'ids': ids, 'groups': groups, 'features': transform_features(self.processor, flows)}

    def _attack_flags(self, features):
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(torch.from_numpy(features)), dim=-1).argmax(dim=-1).numpy() != 0

    def score_prepared(self, prepared):
        flags = self._attack_flags(prepared['features'])
        return np.array([flags[g].mean() for g in prepared['groups']], dtype=np.float64)

    def modules(self):
        return [self.model]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / 'model.pt')
        (directory / 'preprocessor.pkl').write_bytes(pickle.dumps(self.processor))
        (directory / 'metadata.json').write_text(json.dumps(
            {'kind': 'flow_vote', 'name': self.name, 'input_dim': self.model.net[0].in_features,
             'config': self.config, 'history': self.history}, indent=2))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / 'metadata.json').read_text())
        self = cls(meta['name'])
        self.processor = pickle.loads((directory / 'preprocessor.pkl').read_bytes())
        c = meta['config']
        self.model = MLPClassifier(meta['input_dim'], c['hidden_dim'], c['dropout'], output_dim=2)
        self.model.load_state_dict(torch.load(directory / 'model.pt', weights_only=True))
        self.config, self.history, self.directory = c, meta['history'], directory
        return self

    def describe(self):
        d = super().describe()
        d['notes'] = 'Beta MLP 32-16 on flows (component labels), window vote fraction'
        d['artifact_paths'] = [str(p) for p in sorted(getattr(self, 'directory', Path('/nonexistent')).glob('*'))]
        return d


class SavedFlowVote(FlowVote):
    """Saved six-class Beta checkpoint, zero-shot: fraction of flows not predicted benign."""
    provenance = 'saved_checkpoint'

    def __init__(self, key):
        from ids_ml.models.train import load_artifact
        self.name, self.directory = key, SAVED_PATHS[key]
        self.model, self.processor, meta = load_artifact(self.directory)
        self.class_names = meta['class_names']
        assert self.class_names[0] == 'benign'

    def describe(self):
        d = EvaluationAdapter.describe(self)
        d['artifact_paths'] = [str(p) for p in sorted(self.directory.glob('*'))]
        d['notes'] = 'Saved six-class Beta checkpoint; not trained on suite windows'
        return d


class Pooled(EvaluationAdapter):
    """Order-invariant neural control: per-flow MLP logits mean-pooled per window."""
    causal_features = True

    def __init__(self, name='pooled_mlp', recipe=None):
        self.name, self.recipe = name, recipe

    def fit(self, train, val, seed):
        self.processor = build_preprocessor(train.loc[:, list(INFERENCE_COLUMNS)])
        packs = []
        for frame in (train, val):
            prepared = self.prepare(frame.loc[:, list(INFERENCE_COLUMNS)])
            target = frame.groupby('session_id', sort=False).window_target.first()
            packs.append((*pack(prepared), target.loc[prepared['ids']].to_numpy(dtype=np.int64)))
        self.model, self.history = train_pool(*packs[0], *packs[1], self.recipe, seed)
        self.config = dataclasses.asdict(self.recipe)

    flow_features = _SequenceBase.flow_features
    prepare = _SequenceBase.prepare
    shuffle_inputs = False

    def score_prepared(self, prepared):
        X, lengths = pack(prepared)
        self.model.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 256):
                logits = self.model(torch.from_numpy(X[i:i + 256]), torch.from_numpy(lengths[i:i + 256]))
                out.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
        return np.concatenate(out).astype(np.float64)

    def modules(self):
        return [self.model]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / 'model.pt')
        (directory / 'preprocessor.pkl').write_bytes(pickle.dumps(self.processor))
        (directory / 'metadata.json').write_text(json.dumps(
            {'kind': 'pooled', 'name': self.name, 'input_dim': self.model.flow.net[0].in_features,
             'config': self.config, 'history': self.history}, indent=2))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / 'metadata.json').read_text())
        self = cls(meta['name'])
        self.processor = pickle.loads((directory / 'preprocessor.pkl').read_bytes())
        self.model = PooledMLP(meta['input_dim'])
        self.model.load_state_dict(torch.load(directory / 'model.pt', weights_only=True))
        self.config, self.history, self.directory = meta['config'], meta['history'], directory
        return self


# --------------------------------------------------------------------------- engineered baselines

SUMMARY_NAMES = ('flows', 'span_s', 'rate_per_min', 'failed_fraction', 'reset_fraction', 'unique_ports',
                 'unique_failed_ports', 'max_flows_one_port', 'max_failed_one_port',
                 'longest_same_port_failure_run', 'ascending_fraction', 'periodic_count', 'periodic_cv',
                 'periodic_bytes_cv', 'periodic_port_is_web', 'max_bytes_sent', 'max_bytes_position',
                 'failure_time_centroid', 'mean_duration', 'mean_packets', 'mean_bytes_sent', 'udp_fraction')


def window_summaries(flows):
    """Engineered window features from allowed inputs only, including generic order features."""
    ids, groups = groups_of(flows)
    port = flows.dst_port.to_numpy()
    failed = flows.outcome.isin(['rejected', 'timeout']).to_numpy()
    reset = (flows.outcome == 'reset').to_numpy()
    time_s = flows.event_time.to_numpy(dtype=float)
    bytes_sent = flows.bytes_sent.to_numpy(dtype=float)
    rows = []
    for g in groups:
        p, f, t, b = port[g], failed[g], time_s[g] - time_s[g][0], bytes_sent[g]
        span = max(float(t[-1]), 1.)
        uniq, counts = np.unique(p, return_counts=True)
        failed_ports = p[f]
        fu, fc = np.unique(failed_ports, return_counts=True) if len(failed_ports) else ([], [0])
        run = best = 0
        for i in range(len(p)):
            run = run + 1 if f[i] and i and f[i - 1] and p[i] == p[i - 1] else int(f[i])
            best = max(best, run)
        # Ascending order among consecutive flows to distinct non-background-like ports.
        distinct = p[np.r_[True, p[1:] != p[:-1]]]
        asc = float(np.mean(np.diff(distinct) > 0)) if len(distinct) > 2 else 0.
        top = uniq[np.argmax(counts)]
        tt, tb = t[p == top], b[p == top]
        gaps = np.diff(tt)
        cv = float(np.std(gaps) / max(np.mean(gaps), 1e-6)) if len(gaps) > 1 else 5.
        bcv = float(np.std(tb) / max(np.mean(tb), 1e-6)) if len(tb) > 1 else 5.
        rows.append([len(g), span, len(g) / span * 60, f.mean(), reset[g].mean(), len(uniq), len(fu),
                     counts.max(), max(fc), best, asc, len(tt), cv, bcv, float(top in (80, 443, 53)),
                     b.max(), float(np.argmax(b)) / max(len(g) - 1, 1),
                     float(np.mean(t[f]) / span) if f.any() else -1., flows.duration_s.to_numpy()[g].mean(),
                     flows.packets_sent.to_numpy()[g].mean(), b.mean(), (flows.protocol.to_numpy()[g] == 'udp').mean()])
    return ids, np.asarray(rows, dtype=np.float64)


class SummaryBoosting(EvaluationAdapter):
    name = 'summary_boosting'
    causal_features = False  # whole-window summaries, by design

    def __init__(self, name='summary_boosting', recipe=None):
        self.name, self.recipe = name, recipe

    def fit(self, train, val, seed):
        ids, S = window_summaries(train.loc[:, list(INFERENCE_COLUMNS)])
        y = train.groupby('session_id', sort=False).window_target.first().loc[ids].to_numpy()
        self.model = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, l2_regularization=1.,
                                                    random_state=seed)
        with threadpool_limits(limits=self.recipe.threads):
            self.model.fit(S, y)

    def prepare(self, flows):
        check_inference_frame(flows)
        ids, S = window_summaries(flows)
        return {'ids': ids, 'summaries': S}

    def score_prepared(self, prepared):
        # No per-call threadpool_limits: it re-inspects native libraries on every
        # call (~0.25 s). Predictions do not depend on thread count; callers set
        # thread limits for the process (runner/efficiency harness).
        return self.model.predict_proba(prepared['summaries'])[:, 1].astype(np.float64)

    def describe(self):
        return {**super().describe(), 'parameter_count': None, 'parameter_count_source': 'not_neural',
                'notes': f'HistGradientBoosting on {len(SUMMARY_NAMES)} engineered summaries incl. order features'}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'model.pkl').write_bytes(pickle.dumps(self.model))
        (directory / 'metadata.json').write_text(json.dumps({'kind': 'summary', 'name': self.name}))

    @classmethod
    def load(cls, directory):
        self = cls(json.loads((Path(directory) / 'metadata.json').read_text())['name'])
        self.model = pickle.loads((Path(directory) / 'model.pkl').read_bytes())
        self.directory = Path(directory)
        return self


class TemporalRule(EvaluationAdapter):
    """Deterministic comparison heuristic. Never a primary detector."""
    name, provenance = 'temporal_rule', 'rule'

    def fit(self, train, val, seed):
        return None

    def prepare(self, flows):
        check_inference_frame(flows)
        ids, S = window_summaries(flows)
        return {'ids': ids, 'summaries': S}

    def score_prepared(self, prepared):
        S = prepared['summaries']
        col = {n: i for i, n in enumerate(SUMMARY_NAMES)}
        periodic = (S[:, col['periodic_cv']] < .35) & (S[:, col['periodic_port_is_web']] == 0) & (S[:, col['periodic_count']] >= 4)
        return np.maximum.reduce([S[:, col['unique_failed_ports']] / 10, S[:, col['longest_same_port_failure_run']] / 10,
                                  periodic * S[:, col['periodic_count']] / 10]).astype(np.float64)

    def describe(self):
        return {**super().describe(), 'parameter_count': 0, 'notes': 'max(unique failed ports, same-port failure run, uncommon-port periodic count)/10'}

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / 'metadata.json').write_text(json.dumps({'kind': 'rule', 'name': self.name}))

    @classmethod
    def load(cls, directory):
        return cls()


# --------------------------------------------------------------------------- registry

RETRAINED = {
    'beta1_counterpart': lambda r: FlowVote('beta1_counterpart', r),
    'beta2_counterpart': lambda r: FlowVote('beta2_counterpart', r),
    'sona_counterpart': lambda r: SequenceCounterpart('sona_counterpart', r),
    'sona_shuffled_trained': lambda r: SequenceCounterpart('sona_shuffled_trained', r, shuffle_inputs=True),
    'pooled_mlp': lambda r: Pooled('pooled_mlp', r),
    'summary_boosting': lambda r: SummaryBoosting('summary_boosting', r),
    'temporal_rule': lambda r: TemporalRule(),
}
SAVED = {
    'beta1_saved': lambda: SavedFlowVote('beta1_saved'),
    'beta2_saved': lambda: SavedFlowVote('beta2_saved'),
    'sona_public_saved': lambda: SavedSequence(),
}
LOADERS = {'sequence': SequenceCounterpart, 'flow_vote': FlowVote, 'pooled': Pooled,
           'summary': SummaryBoosting, 'rule': TemporalRule}

# Training-data restriction per recipe (Beta 1: narrower ordinary training).
TRAIN_FILTERS = {'beta1_counterpart': lambda key: key.pace != 'slow'}


def load_retrained(directory):
    kind = json.loads((Path(directory) / 'metadata.json').read_text())['kind']
    return LOADERS[kind].load(directory)


def load_candidate(spec, kwargs=None):
    """Import `module:function` read-only and validate the returned adapter."""
    module_name, _, function = spec.partition(':')
    if not module_name.startswith('ids_ml.experiments.sona_candidate'):
        raise ValueError('Candidate adapters must live under ids_ml.experiments.sona_candidate.')
    adapter = getattr(importlib.import_module(module_name), function or 'build_evaluation_adapter')(**(kwargs or {}))
    missing = [a for a in REQUIRED if not hasattr(adapter, a)]
    if missing:
        raise TypeError(f'Candidate adapter lacks {missing}.')
    if adapter.provenance not in ('saved_checkpoint', 'retrained'):
        raise ValueError('provenance must be saved_checkpoint or retrained.')
    return adapter


def candidate_source_hashes():
    base = ROOT / 'src/ids_ml/experiments/sona_candidate'
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(base.rglob('*.py'))} if base.exists() else {}


def score_frame(adapter, flows, window_ids):
    """Contract-enforcing scoring: label-free input, aligned finite output."""
    check_inference_frame(flows)
    prepared = adapter.prepare(flows)
    scores = np.asarray(adapter.score_prepared(prepared), dtype=np.float64)
    order = list(adapter.window_order(prepared))
    if len(order) != len(scores) or set(order) != set(window_ids) or len(set(order)) != len(order):
        raise ValueError(f'{adapter.name}: scores not aligned with windows.')
    if not np.isfinite(scores).all():
        raise ValueError(f'{adapter.name}: non-finite scores.')
    return pd.Series(scores, index=order).loc[list(window_ids)].to_numpy()
