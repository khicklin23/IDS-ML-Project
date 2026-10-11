"""Validate the experimental controls, not a predetermined winning score."""
import numpy as np
import pandas as pd
import pytest
from ids_ml.data.schema import FEATURE_COLUMNS
from ids_ml.features import build_preprocessor
from ids_ml.experiments.temporal_patterns import generate_pairs, pack, shuffled, threshold_for_false_alarms, score_metrics


def test_paired_flow_features_are_identical_but_order_differs():
    df, manifest = generate_pairs(42, 12, 'test', min_length=16, max_length=24)
    assert manifest['sessions'] == 24
    for _, pair in df.groupby('pair_id'):
        benign, attack = [pair[pair.target == label] for label in (0, 1)]
        columns = list(FEATURE_COLUMNS)
        pd.testing.assert_frame_equal(benign[columns].sort_values(columns).reset_index(drop=True),
                                      attack[columns].sort_values(columns).reset_index(drop=True))
        np.testing.assert_array_equal(benign[columns].iloc[[0,-1]].to_numpy(), attack[columns].iloc[[0,-1]].to_numpy())
        assert not np.array_equal(benign.prior_gap_s, attack.prior_gap_s)
        for frame in (benign, attack):
            np.testing.assert_allclose(np.diff(frame.event_time), frame.prior_gap_s.iloc[1:])
            assert frame.event_time.is_monotonic_increasing
            assert (frame.bytes_sent >= frame.packets_sent).all()
            assert len(set(frame.scenario)) == 1


def test_generation_and_split_identifiers_are_reproducible_and_disjoint():
    a, _ = generate_pairs(11, 8, 'train')
    repeat, _ = generate_pairs(11, 8, 'train')
    b, _ = generate_pairs(10011, 8, 'val')
    pd.testing.assert_frame_equal(a, repeat)
    assert not set(a.flow_id) & set(b.flow_id)
    assert not set(a.pair_id) & set(b.pair_id)
    assert not set(a.session_id) & set(b.session_id)


def test_padding_and_shuffle_do_not_change_endpoints_or_feature_multisets():
    df, _ = generate_pairs(11, 8, 'train')
    X, lengths, y, _ = pack(df, build_preprocessor(df))
    control = shuffled(X, lengths, 51)
    assert set(y) == {0, 1}
    for i, n in enumerate(lengths):
        np.testing.assert_array_equal(X[i, [0,n-1]], control[i, [0,n-1]])
        np.testing.assert_array_equal(X[i,n:], control[i,n:])
        assert np.all(X[i,n:] == 0)
        assert sorted(map(tuple, X[i,:n])) == sorted(map(tuple, control[i,:n]))
    assert not np.array_equal(X, control)


def test_validation_threshold_handles_ties_without_exceeding_budget():
    y = np.array([0] * 100 + [1] * 100)
    flat = np.full(200, .5)
    cutoff = threshold_for_false_alarms(flat, y, .01)
    metrics = score_metrics(flat, y, cutoff)
    assert cutoff > .5
    assert metrics['at_validation_false_alarm_target']['false_positive_rate'] == 0
    assert metrics['at_validation_false_alarm_target']['recall'] == 0
    perfect = np.r_[np.zeros(100), np.ones(100)]
    assert score_metrics(perfect, y, threshold_for_false_alarms(perfect, y, .01))['roc_auc'] == 1
    with pytest.raises(ValueError):
        threshold_for_false_alarms(flat, y, 1)


def test_shift_does_not_break_the_matched_pair_control():
    df, _ = generate_pairs(42, 8, 'shift', shifted=True)
    columns=list(FEATURE_COLUMNS)
    for _, pair in df.groupby('pair_id'):
        a,b=[pair[pair.target==label][columns].sort_values(columns).reset_index(drop=True) for label in (0,1)]
        pd.testing.assert_frame_equal(a,b)
