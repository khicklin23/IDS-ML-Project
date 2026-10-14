"""Adapter contract, structural order-invariance, replay and causality auditing."""
import numpy as np
import pytest
import torch

from ids_ml.experiments.sona_evaluation import adapters as A
from ids_ml.experiments.sona_evaluation import audit, cases
from ids_ml.experiments.sona_evaluation.protocol import SMOKE_RECIPE


@pytest.fixture(scope='module')
def data():
    torch.set_num_threads(1)
    train, _ = cases.generate_split(52001, 'train', windows=80, triples=12)
    val, _ = cases.generate_split(52001, 'val', windows=80, triples=12)
    chains, key = cases.generate_split(52001, 'chains', triples=10, gaps=(0, 8))
    return train, val, chains, key


@pytest.fixture(scope='module')
def fitted(data):
    train, val, _, _ = data
    out = {}
    for name in ('beta2_counterpart', 'pooled_mlp', 'sona_counterpart', 'summary_boosting'):
        adapter = A.RETRAINED[name](SMOKE_RECIPE)
        adapter.fit(train, val, 3)
        out[name] = adapter
    return out


def _scores(adapter, frame, key):
    return A.score_frame(adapter, cases.inference_frame(frame), key.session_id.tolist())


@pytest.mark.parametrize('name', ['beta2_counterpart', 'pooled_mlp'])
def test_order_invariant_models_cannot_separate_attack_from_swap(data, fitted, name):
    _, _, chains, key = data
    scores = _scores(fitted[name], chains, key)
    wide = key.assign(score=scores).pivot(index='triple_id', columns='role', values='score')
    np.testing.assert_allclose(wide.attack, wide.swap, atol=1e-6)


def test_sequence_model_reads_order(data, fitted):
    _, _, chains, key = data
    scores = _scores(fitted['sona_counterpart'], chains, key)
    wide = key.assign(score=scores).pivot(index='triple_id', columns='role', values='score')
    assert np.abs(wide.attack - wide.swap).max() > 1e-6


def test_scoring_rejects_label_columns_and_misaligned_output(data, fitted):
    _, _, chains, key = data
    leaky = cases.inference_frame(chains).assign(window_target=chains.window_target)
    with pytest.raises(ValueError):
        A.score_frame(fitted['sona_counterpart'], leaky, key.session_id.tolist())

    class Broken(A.EvaluationAdapter):
        name = 'broken'

        def prepare(self, flows):
            return {'ids': flows.session_id.unique()[:-1].tolist()}

        def score_prepared(self, prepared):
            return np.zeros(len(prepared['ids']))
    with pytest.raises(ValueError):
        A.score_frame(Broken(), cases.inference_frame(chains), key.session_id.tolist())


def test_save_load_replay_is_exact(tmp_path, data, fitted):
    _, _, chains, key = data
    for name, adapter in fitted.items():
        adapter.save(tmp_path / name)
        again = A.load_retrained(tmp_path / name)
        np.testing.assert_array_equal(_scores(adapter, chains, key), _scores(again, chains, key))


def test_saved_checkpoints_score_without_modifying_public_artifacts(data):
    _, _, chains, key = data
    before = {p: A.file_hash(A.ROOT / p) for p in audit.PUBLIC_FILES if (A.ROOT / p).exists()}
    if not before:
        pytest.skip('public artifacts absent in this clone')
    for name, factory in A.SAVED.items():
        if not A.SAVED_PATHS[name].exists():
            continue
        scores = _scores(factory(), chains, key)
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    assert before == {p: A.file_hash(A.ROOT / p) for p in before}


def test_prefix_causality_audit_catches_window_level_leakage(data, fitted):
    _, _, chains, _ = data
    flows = cases.inference_frame(chains)
    assert audit.prefix_causality(fitted['sona_counterpart'], flows)

    class Peeking:
        def flow_features(self, frame):
            # Uses the whole window's mean gap: future observations change early rows.
            return frame.groupby('session_id').prior_gap_s.transform('mean').to_numpy()[:, None]
    assert not audit.prefix_causality(Peeking(), flows)


def test_candidate_loader_is_restricted_to_candidate_package():
    with pytest.raises(ValueError):
        A.load_candidate('ids_ml.experiments.sona_evaluation.adapters:SequenceCounterpart')


def test_shuffled_control_permutation_depends_only_on_window_id():
    a = A._interior_permutation('wabc', 12)
    assert (a == A._interior_permutation('wabc', 12)).all()
    assert a[0] == 0 and a[-1] == 11 and sorted(a) == list(range(12))
