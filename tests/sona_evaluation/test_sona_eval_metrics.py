import numpy as np
import pandas as pd
import pytest

from ids_ml.experiments.sona_evaluation import metrics
from ids_ml.experiments.sona_evaluation.summarize import paired_triples


@pytest.mark.parametrize('budget', [.01, .02, .05])
def test_threshold_never_exceeds_validation_budget_even_with_ties(budget):
    rng = np.random.default_rng(1)
    scores = np.round(rng.random(1000), 2)  # many ties
    y = rng.integers(0, 2, 1000)
    threshold = metrics.threshold_at_budget(scores, y, budget)
    benign = scores[y == 0]
    assert np.sum(benign >= threshold) <= np.floor(budget * len(benign))


def test_at_threshold_counts_and_flow_units():
    scores = np.array([.9, .8, .1, .7, .2])
    y = np.array([1, 1, 1, 0, 0])
    m = metrics.at_threshold(scores, y, .75, flow_counts=np.array([10, 5, 5, 20, 4]),
                             attack_flow_counts=np.array([2, 1, 3, 0, 0]))
    assert (m['tp'], m['fp'], m['fn'], m['tn']) == (2, 0, 1, 2)
    assert m['recall'] == pytest.approx(2 / 3) and m['false_positive_rate'] == 0
    assert m['flow_attack_coverage'] == pytest.approx(3 / 6)
    assert m['flow_weighted_false_alarm'] == 0


def _key(n):
    rows = []
    for t in range(n):
        for role in ('attack', 'swap', 'innocent'):
            rows.append({'triple_id': f't{t}', 'role': role, 'gap_distractors': 8 if t % 2 else 64,
                         'decoys': 0, 'target': int(role == 'attack')})
    return pd.DataFrame(rows)


def test_order_invariant_scores_are_exactly_chance_on_swaps():
    key = _key(50)
    rng = np.random.default_rng(0)
    scores = np.zeros(len(key))
    for t in range(50):
        shared = rng.random()
        scores[3 * t:3 * t + 2] = shared            # attack and swap identical
        scores[3 * t + 2] = shared - .5               # innocent lower
    summary, _, _ = metrics.triple_metrics(scores, key)
    assert summary['all']['attack>swap'] == .5
    assert summary['all']['attack>innocent'] == 1.
    assert summary['all']['triple_solved'] == .5
    assert set(summary) >= {'gap=8', 'gap=64', 'decoys', 'no_decoys'}


def test_summation_noise_counts_as_a_tie_after_rounding():
    key = _key(40)
    scores = np.tile([.6, .6, .1], 40) + np.tile([1e-9, -1e-9, 0], 40)
    strict = metrics.triple_metrics(scores, key)[0]['all']['attack>swap']
    tolerant = paired_triples(scores, key)[0]['all']['attack>swap']
    assert strict == 1. and tolerant == .5


def test_bootstrap_difference_contains_point_and_zero_for_identical_models():
    y = np.r_[np.ones(100), np.zeros(100)].astype(int)
    rng = np.random.default_rng(3)
    a = rng.random(200) > .4
    result = metrics.bootstrap_difference([(y, a, a)], metrics.recall_difference, 200, 1)
    assert result['difference'] == 0 and result['ci95'] == [0., 0.]
    b = a.copy(); b[:100] = False
    result = metrics.bootstrap_difference([(y, a, b)], metrics.recall_difference, 300, 1)
    assert result['ci95'][0] <= result['difference'] <= result['ci95'][1] and result['ci95'][0] > 0
