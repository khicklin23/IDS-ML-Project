"""Case-generator guarantees the evaluation's conclusions depend on."""
import numpy as np
import pandas as pd
import pytest

from ids_ml.data.schema import FEATURE_COLUMNS, validate
from ids_ml.experiments.sona_evaluation import cases
from ids_ml.experiments.sona_evaluation.protocol import CONDITIONS, INFERENCE_COLUMNS, SMOKE_SIZES

FEATURES = list(FEATURE_COLUMNS)


@pytest.fixture(scope='module')
def chains():
    return cases.generate_split(52001, 'test-chains', triples=30, gaps=(0, 4, 16, 64))


def _windows(frame, key):
    groups = cases.window_groups(frame)
    return {sid: frame.iloc[groups[sid]] for sid in key.session_id}


def test_generation_is_deterministic_and_streams_differ():
    a, ka = cases.generate_split(52001, 'familiar', windows=20)
    b, kb = cases.generate_split(52001, 'familiar', windows=20)
    c, _ = cases.generate_split(52001, 'other', windows=20)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ka, kb)
    assert not set(a.session_id) & set(c.session_id)


@pytest.mark.parametrize('condition', CONDITIONS)
def test_conditions_are_valid_balanced_and_inside_capture(condition):
    frame, key = cases.generate_split(52001, condition, windows=40, condition=condition)
    validate(frame)
    assert key.target.sum() == 20
    start = frame.session_id.map(key.set_index('session_id').observation_start_s)
    rel = frame.event_time - start
    assert (rel >= 0).all() and (rel < frame.session_id.map(key.set_index('session_id').duration_s)).all()
    assert frame.groupby('session_id').window_target.nunique().eq(1).all()
    # Window target is 1 exactly when some flow belongs to an attack component.
    assert (frame.groupby('session_id').attack_flow.max() == frame.groupby('session_id').window_target.first()).all()


def test_gaps_match_successive_timestamps_including_controls():
    frame, _ = cases.generate_split(52001, 'noisy_timing', windows=20, condition='noisy_timing', triples=4)
    for variant in (frame, cases.shuffle_interior(frame, 3), cases.truncate_last(frame, 8)):
        for _, w in variant.groupby('session_id', sort=False):
            expected = np.r_[0., np.diff(w.event_time.to_numpy())]
            np.testing.assert_allclose(w.prior_gap_s.to_numpy(), expected, atol=1e-6, rtol=1e-7)


def test_swap_has_identical_measurements_and_endpoints(chains):
    frame, key = chains
    windows = _windows(frame, key)
    for _, triple in key.groupby('triple_id'):
        roles = dict(zip(triple.role, triple.session_id))
        attack, swap = windows[roles['attack']], windows[roles['swap']]
        sort = lambda w: w[FEATURES].sort_values(FEATURES).reset_index(drop=True)
        pd.testing.assert_frame_equal(sort(attack), sort(swap))
        pd.testing.assert_frame_equal(attack[FEATURES].iloc[[0, -1]].reset_index(drop=True),
                                      swap[FEATURES].iloc[[0, -1]].reset_index(drop=True))
        assert not attack[FEATURES].reset_index(drop=True).equals(swap[FEATURES].reset_index(drop=True))


def test_innocent_shares_the_attack_suffix_after_the_early_block(chains):
    frame, key = chains
    windows = _windows(frame, key)
    for _, triple in key.groupby('triple_id'):
        roles = dict(zip(triple.role, triple.session_id))
        attack, innocent = windows[roles['attack']], windows[roles['innocent']]
        last_fail = np.flatnonzero(attack.component.to_numpy() == 'chain_fail').max()
        tail = len(attack) - last_fail - 1
        assert tail >= triple.gap_distractors.iloc[0] + 2  # distractors, login, transfer, post-background
        pd.testing.assert_frame_equal(attack[FEATURES].iloc[-tail:].reset_index(drop=True),
                                      innocent[FEATURES].iloc[-tail:].reset_index(drop=True))
        assert innocent.attack_flow.max() == 0 and windows[roles['swap']].attack_flow.max() == 0


def test_gap_distractors_and_decoys(chains):
    frame, key = chains
    attacks = key[key.role == 'attack']
    windows = _windows(frame, key)
    for _, row in attacks.iterrows():
        w = windows[row.session_id]
        fail = np.flatnonzero(w.component.to_numpy() == 'chain_fail')
        login = np.flatnonzero(w.component.to_numpy() == 'chain_login')[0]
        assert login - fail.max() - 1 == row.gap_distractors
    assert (key.decoys[key.gap_distractors < 4] == 0).all()
    assert key.decoys.max() >= 2


def test_inference_frame_is_label_free_and_ids_are_opaque(chains):
    frame, key = chains
    view = cases.inference_frame(frame)
    assert tuple(view.columns) == INFERENCE_COLUMNS
    for forbidden in ('window_target', 'attack_flow', 'scenario', 'label', 'component', 'run_id', 'src_host_id'):
        assert forbidden not in view
    assert all(sid.startswith('w') and len(sid) == 17 for sid in key.session_id)
    assert not any(token in ''.join(key.session_id) for token in ('attack', 'swap', 'chain', 'scan'))


def test_shuffle_keeps_multiset_and_endpoints_truncate_keeps_suffix(chains):
    frame, _ = chains
    shuffled = cases.shuffle_interior(frame, 7)
    for (sid, a), (_, b) in zip(frame.groupby('session_id', sort=False), shuffled.groupby('session_id', sort=False)):
        cols = [c for c in FEATURES if c != 'prior_gap_s']
        pd.testing.assert_frame_equal(a[cols].sort_values(cols).reset_index(drop=True),
                                      b[cols].sort_values(cols).reset_index(drop=True))
        assert a.flow_id.iloc[0] == b.flow_id.iloc[0] and a.flow_id.iloc[-1] == b.flow_id.iloc[-1]
    truncated = cases.truncate_last(frame, 5)
    assert truncated.groupby('session_id').size().max() == 5
    last = frame.groupby('session_id').flow_id.last()
    kept = truncated.groupby('session_id').flow_id.last()
    assert (kept == last.loc[kept.index]).all()


def test_lookalike_distinguishing_observables_are_as_documented():
    frame, key = cases.generate_split(52001, 'lookalike_heavy', windows=200, condition='lookalike_heavy')
    failed = frame.outcome.isin(['rejected', 'timeout'])
    scan = failed[frame.component == 'scan'].mean()
    inventory = failed[frame.component == 'inventory'].mean()
    assert scan > .5 > inventory
    for _, w in frame[frame.component == 'inventory'].groupby('session_id'):
        assert (np.diff(w.dst_port.to_numpy()) > 0).all()  # systematic ascending sweep
    beacon_ports = set(frame.dst_port[frame.component == 'beacon'])
    polling_ports = set(frame.dst_port[frame.component == 'polling'])
    assert polling_ports <= {80, 443} and not beacon_ports & {80, 443}


def test_split_plan_covers_every_partition():
    names = [name for name, _ in cases.split_plan(SMOKE_SIZES)]
    assert names[:2] == ['train', 'val'] and set(CONDITIONS) <= set(names)
    assert {'chain_familiar', 'chain_long'} <= set(names)
