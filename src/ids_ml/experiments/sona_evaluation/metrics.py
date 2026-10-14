"""Frozen-threshold metrics, subtype breakdowns, paired triples and bootstrap."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def budget_key(budget):
    return f'{budget:.2f}'


def threshold_at_budget(scores, y, budget):
    """Smallest threshold whose validation benign flag count is <= floor(budget * benign).

    Flags are `score >= threshold`. Placed strictly above the first rejected
    benign score, so ties can never exceed the budget on validation.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('Scores must be finite.')
    benign = np.sort(scores[np.asarray(y) == 0])[::-1]
    if not len(benign) or not 0 <= budget < 1:
        raise ValueError('Need benign validation windows and a budget in [0, 1).')
    permitted = int(np.floor(budget * len(benign)))
    return float(np.nextafter(benign[permitted], np.inf))


def at_threshold(scores, y, threshold, flow_counts=None, attack_flow_counts=None):
    scores, y = np.asarray(scores, dtype=np.float64), np.asarray(y)
    flag = scores >= threshold
    tp, fp = int(np.sum(flag & (y == 1))), int(np.sum(flag & (y == 0)))
    fn, tn = int(np.sum(~flag & (y == 1))), int(np.sum(~flag & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else float('nan')
    out = {'threshold': float(threshold), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
           'recall': recall, 'false_positive_rate': fp / (fp + tn) if fp + tn else float('nan'),
           'precision': precision,
           'f1': 2 * precision * recall / (precision + recall) if tp else 0.}
    if flow_counts is not None:
        # Flow units: attack-component flows inside flagged windows, and benign
        # windows' flows inheriting a false flag. Not independent flow decisions.
        fc, afc = np.asarray(flow_counts), np.asarray(attack_flow_counts)
        out['flow_attack_coverage'] = float(afc[flag].sum() / afc.sum()) if afc.sum() else float('nan')
        benign_flows = fc[y == 0].sum()
        out['flow_weighted_false_alarm'] = float(fc[flag & (y == 0)].sum() / benign_flows) if benign_flows else float('nan')
    return out


def ranking(scores, y):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return {'roc_auc': float('nan'), 'average_precision': float('nan')}
    return {'roc_auc': float(roc_auc_score(y, scores)), 'average_precision': float(average_precision_score(y, scores))}


def subtype_rates(scores, key, threshold):
    """Flag rate per attack family/pace and per benign lookalike."""
    flag = np.asarray(scores) >= threshold
    out = {}
    t = key.target.to_numpy()
    for column in ('attack_kind', 'pace'):
        for value in sorted(set(key[column][t == 1]) - {''}):
            mask = (t == 1) & (key[column].to_numpy() == value)
            out[f'attack:{column}={value}'] = {'windows': int(mask.sum()), 'flag_rate': float(flag[mask].mean())}
    look = key.lookalike.to_numpy()
    for value in sorted(set(look[t == 0])):
        mask = (t == 0) & (look == value)
        out[f'benign:lookalike={value or "none"}'] = {'windows': int(mask.sum()), 'flag_rate': float(flag[mask].mean())}
    for value in sorted(set(look[t == 1]) - {''}):
        mask = (t == 1) & (look == value)
        out[f'attack:with_lookalike={value}'] = {'windows': int(mask.sum()), 'flag_rate': float(flag[mask].mean())}
    return out


def triple_metrics(scores, key):
    """Threshold-free paired comparisons within chain triples; ties count one half."""
    frame = key.assign(score=np.asarray(scores, dtype=np.float64))
    frame = frame[frame.triple_id != '']
    wide = frame.pivot(index='triple_id', columns='role', values='score')
    gaps = frame.groupby('triple_id').gap_distractors.first().reindex(wide.index)
    decoys = frame.groupby('triple_id').decoys.first().reindex(wide.index)

    def beat(a, b):
        return np.where(a > b, 1., np.where(a == b, .5, 0.))
    swap = beat(wide.attack.to_numpy(), wide.swap.to_numpy())
    innocent = beat(wide.attack.to_numpy(), wide.innocent.to_numpy())
    solved = swap * innocent  # 1 only if both strict; ties give partial credit
    per = {'attack>swap': swap, 'attack>innocent': innocent, 'triple_solved': solved}

    def summarize(mask):
        return {name: float(v[mask].mean()) for name, v in per.items()} | {'triples': int(mask.sum())}
    out = {'all': summarize(np.ones(len(wide), bool))}
    for g in sorted(gaps.unique()):
        out[f'gap={int(g)}'] = summarize(gaps.to_numpy() == g)
    out['decoys'] = summarize(decoys.to_numpy() > 0)
    out['no_decoys'] = summarize(decoys.to_numpy() == 0)
    return out, {name: v for name, v in per.items()}, list(wide.index)


def bootstrap_difference(per_replicate, statistic, resamples, seed):
    """Paired bootstrap of mean-over-replicates statistic(a) - statistic(b).

    per_replicate: list of tuples of aligned arrays for one replicate.
    statistic(*arrays, idx) -> difference for resampled indices. Resampling
    is within replicate (windows or triples), preserving pairing.
    """
    rng = np.random.default_rng(seed)
    point = float(np.mean([statistic(*arrays, np.arange(len(arrays[0]))) for arrays in per_replicate]))
    draws = np.empty(resamples)
    for i in range(resamples):
        draws[i] = np.mean([statistic(*arrays, rng.integers(0, len(arrays[0]), len(arrays[0])))
                            for arrays in per_replicate])
    lo, hi = np.quantile(draws, [.025, .975])
    return {'difference': point, 'ci95': [float(lo), float(hi)]}


def recall_difference(y, flag_a, flag_b, idx):
    yy = y[idx]
    pos = yy == 1
    return float(flag_a[idx][pos].mean() - flag_b[idx][pos].mean()) if pos.any() else 0.


def fpr_difference(y, flag_a, flag_b, idx):
    yy = y[idx]
    neg = yy == 0
    return float(flag_a[idx][neg].mean() - flag_b[idx][neg].mean()) if neg.any() else 0.


def mean_difference(a, b, idx):
    return float(a[idx].mean() - b[idx].mean())


def per_seed_summary(values):
    values = [v for v in values if v == v]
    if not values:
        return {'mean': float('nan'), 'min': float('nan'), 'max': float('nan'), 'n': 0}
    return {'mean': float(np.mean(values)), 'min': float(min(values)), 'max': float(max(values)), 'n': len(values)}
