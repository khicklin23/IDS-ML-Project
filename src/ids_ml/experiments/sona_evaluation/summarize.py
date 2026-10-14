"""Aggregate audited replicates, compare models pairwise, and apply the frozen gates.

    python -m ids_ml.experiments.sona_evaluation.summarize RUN_DIR
        [--candidate candidate_fit] [--reference sona_counterpart] [--efficiency EFFICIENCY.json]

Refuses to summarize any replicate whose audit did not pass. Writes
summary.json and RESULTS.md inside RUN_DIR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (bootstrap_difference, budget_key, fpr_difference, per_seed_summary, recall_difference,
                      triple_metrics)
from .protocol import CHAIN_SPLITS, CONDITIONS, PRIMARY_BUDGET, Gates

PRIMARY = budget_key(PRIMARY_BUDGET)
CONTROL_SETS = ('familiar:shuffled', 'familiar:trunc8', 'familiar:trunc32',
                'chain_familiar:shuffled', 'chain_familiar:trunc8', 'chain_familiar:trunc32')
ORDER_INVARIANT = ('pooled_mlp', 'beta2_counterpart', 'beta1_counterpart')


def load_run(run_dir):
    reps = []
    for rep in sorted(Path(run_dir).glob('replicate-*')):
        if not (rep / 'replicate.json').exists():
            continue
        audit = json.loads((rep / 'audit.json').read_text())
        if audit['status'] != 'passed':
            raise SystemExit(f'{rep.name}: audit {audit["status"]}; refusing to summarize.')
        reps.append({'dir': rep, 'seed': int(rep.name.split('-')[1]),
                     'metrics': json.loads((rep / 'metrics.json').read_text()),
                     'thresholds': json.loads((rep / 'thresholds.json').read_text())['thresholds']})
    if not reps:
        raise SystemExit('No completed replicates.')
    return reps


def scores_for(rep, set_name):
    with np.load(rep['dir'] / 'scores' / f'{set_name.replace(":", "__")}.npz') as saved:
        return {k: saved[k] for k in saved.files}


def key_for(rep, source):
    return pd.read_parquet(rep['dir'] / 'data' / f'{source}_key.parquet')


def models_of(reps):
    return [m for m in reps[0]['metrics']['familiar'] if all(m in r['metrics']['familiar'] for r in reps)]


def seed_table(reps, set_name, model, budget, field):
    return per_seed_summary([r['metrics'][set_name][model]['budgets'][budget][field] for r in reps])


def detection_tables(reps, models):
    out = {}
    for condition in CONDITIONS:
        out[condition] = {}
        for m in models:
            row = {b: {f: seed_table(reps, condition, m, b, f)
                       for f in ('recall', 'false_positive_rate', 'precision', 'f1', 'flow_attack_coverage',
                                 'flow_weighted_false_alarm')}
                   for b in reps[0]['metrics'][condition][m]['budgets']}
            row['roc_auc'] = per_seed_summary([r['metrics'][condition][m]['ranking']['roc_auc'] for r in reps])
            row['average_precision'] = per_seed_summary([r['metrics'][condition][m]['ranking']['average_precision'] for r in reps])
            out[condition][m] = row
    return out


def subtype_tables(reps, models):
    out = {}
    for condition in CONDITIONS:
        out[condition] = {}
        for m in models:
            names = reps[0]['metrics'][condition][m]['budgets'][PRIMARY]['subtypes']
            out[condition][m] = {n: per_seed_summary([r['metrics'][condition][m]['budgets'][PRIMARY]['subtypes'][n]['flag_rate']
                                                      for r in reps if n in r['metrics'][condition][m]['budgets'][PRIMARY]['subtypes']])
                                 | {'windows': sum(r['metrics'][condition][m]['budgets'][PRIMARY]['subtypes'].get(n, {}).get('windows', 0) for r in reps)}
                                 for n in names}
    return out


TIE_DECIMALS = 6  # amendment 1: floating-point summation noise below 1e-6 is a tie


def paired_triples(scores, key):
    return triple_metrics(np.round(np.asarray(scores, dtype=np.float64), TIE_DECIMALS), key)


def memory_tables(reps, models):
    out = {}
    for set_name in (*CHAIN_SPLITS, 'chain_familiar:shuffled', 'chain_familiar:trunc8', 'chain_familiar:trunc32'):
        out[set_name] = {}
        source = set_name.split(':')[0]
        cached = [(scores_for(r, set_name), key_for(r, source)) for r in reps]
        for m in models:
            entries = [paired_triples(s[m], k)[0] for s, k in cached]
            out[set_name][m] = {group: {k: per_seed_summary([e[group][k] for e in entries if group in e])
                                        for k in ('attack>swap', 'attack>innocent', 'triple_solved')}
                                for group in entries[0]}
    return out


def control_tables(reps, models):
    out = {}
    for set_name in ('familiar', 'familiar:shuffled', 'familiar:trunc8', 'familiar:trunc32'):
        out[set_name] = {m: {'recall': seed_table(reps, set_name, m, PRIMARY, 'recall'),
                             'false_positive_rate': seed_table(reps, set_name, m, PRIMARY, 'false_positive_rate'),
                             'roc_auc': per_seed_summary([r['metrics'][set_name][m]['ranking']['roc_auc'] for r in reps])}
                         for m in models}
    return out


def sensitivity_nonchain(reps, models):
    """POST HOC, not a gate: thresholds from non-chain validation benigns only.

    Validation contains benign chain swaps whose flows match attack chains;
    order-invariant models therefore get inflated thresholds under the frozen
    protocol. This view separates that coupling from ordinary detection.
    """
    from .metrics import threshold_at_budget
    out = {}
    for m in models:
        out[m] = {c: [] for c in CONDITIONS}
        for r in reps:
            key = key_for(r, 'val')
            keep = (key.condition != 'chain').to_numpy()
            th = threshold_at_budget(scores_for(r, 'val')[m][keep], key.target.to_numpy()[keep], PRIMARY_BUDGET)
            for c in CONDITIONS:
                y, sc = key_for(r, c).target.to_numpy(), scores_for(r, c)[m]
                out[m][c].append((float((sc[y == 1] >= th).mean()), float((sc[y == 0] >= th).mean())))
        out[m] = {c: {'recall': per_seed_summary([v[0] for v in vals]), 'false_positive_rate': per_seed_summary([v[1] for v in vals])}
                  for c, vals in out[m].items()}
    return out


def _flags(rep, set_name, model, budget=PRIMARY):
    scores = scores_for(rep, set_name)
    return scores[model] >= rep['thresholds'][model][budget]


def compare(reps, a, b, gates):
    """Paired a-minus-b comparisons on identical windows, bootstrap within replicate."""
    out = {'conditions': {}}
    for condition in CONDITIONS:
        per = []
        for r in reps:
            y = key_for(r, condition).target.to_numpy()
            per.append((y, _flags(r, condition, a), _flags(r, condition, b)))
        seed = gates.bootstrap_seed + CONDITIONS.index(condition)
        out['conditions'][condition] = {
            'recall': bootstrap_difference(per, recall_difference, gates.bootstrap_resamples, seed),
            'false_positive_rate': bootstrap_difference(per, fpr_difference, gates.bootstrap_resamples, seed + 100)}

    # Macro mean over conditions: resample each condition's windows independently.
    rng = np.random.default_rng(gates.bootstrap_seed + 999)
    cached = {c: [(key_for(r, c).target.to_numpy(), _flags(r, c, a), _flags(r, c, b)) for r in reps] for c in CONDITIONS}

    def macro(index_fn):
        return float(np.mean([np.mean([recall_difference(y, fa, fb, index_fn(len(y))) for y, fa, fb in cached[c]])
                              for c in CONDITIONS]))
    point = macro(np.arange)
    draws = [macro(lambda n: rng.integers(0, n, n)) for _ in range(gates.bootstrap_resamples)]
    out['macro_recall'] = {'difference': point, 'ci95': [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}

    # Memory: paired triple comparisons.
    for set_name in CHAIN_SPLITS:
        per = []
        for r in reps:
            key = key_for(r, set_name)
            scores = scores_for(r, set_name)
            sa, sb = paired_triples(scores[a], key)[1]['triple_solved'], paired_triples(scores[b], key)[1]['triple_solved']
            per.append((sa, sb))
        out[f'{set_name}:triple_solved'] = bootstrap_difference(
            per, lambda x, y, idx: float(x[idx].mean() - y[idx].mean()), gates.bootstrap_resamples, gates.bootstrap_seed + 500)
    return out


def gate_decision(candidate, reference, tables, comparison, memory, efficiency, n_reps, gates, audits_passed):
    g = {}
    g['G0'] = {'passed': bool(audits_passed), 'detail': 'all replicate audits passed' if audits_passed else 'audit failure'}
    macro = comparison['macro_recall']
    floors = {c: v['recall']['ci95'][1] for c, v in comparison['conditions'].items()}
    worst = min(floors, key=floors.get)
    g['G1'] = {'passed': macro['ci95'][0] > gates.g1_macro_lower_bound and floors[worst] >= gates.g1_condition_floor,
               'failed_hard': macro['ci95'][1] < 0 or floors[worst] < gates.g1_condition_floor,
               'macro_recall_difference': macro, 'worst_condition_upper_bound': {worst: floors[worst]}}
    g2 = {c: comparison['conditions'][c]['false_positive_rate'] for c in gates.g2_conditions}
    g['G2'] = {'passed': all(v['ci95'][0] <= gates.g2_fpr_margin for v in g2.values()), 'fpr_difference': g2}
    fam = memory['chain_familiar'][candidate]['all']
    unseen = memory['chain_long'][candidate]['all']
    shuffled = memory['chain_familiar:shuffled'][candidate]['all']
    pooled = comparison.get('pooled_vs_candidate')
    g3 = {'familiar': {k: fam[k]['mean'] for k in fam}, 'unseen_triple_solved': unseen['triple_solved']['mean'],
          'shuffled_attack>swap': shuffled['attack>swap']['mean'], 'vs_pooled_mlp': pooled}
    g3['passed'] = (all(fam[k]['mean'] >= gates.g3_familiar_min for k in ('attack>swap', 'attack>innocent', 'triple_solved'))
                    and unseen['triple_solved']['mean'] >= gates.g3_unseen_triple_min
                    and shuffled['attack>swap']['mean'] <= gates.g3_shuffled_swap_max
                    and pooled is not None and pooled['ci95'][0] > 0)
    g['G3'] = g3
    if efficiency is None:
        g['G4'] = {'passed': None, 'detail': 'efficiency measurements not supplied'}
    else:
        ratio = efficiency.get('ratios', {}).get(candidate)
        g['G4'] = {'ratios_vs_sona_public_saved': ratio,
                   'passed': bool(ratio and ratio['latency'] <= gates.g4_latency_ratio and ratio['peak_rss_increment'] <= gates.g4_memory_ratio)
                   or macro['difference'] >= gates.g4_override_gain}
    if not g['G0']['passed'] or g['G1']['failed_hard'] or not g['G2']['passed'] or (g['G1']['passed'] and g['G4']['passed'] is False):
        verdict = 'FAIL'
    elif all(g[k]['passed'] for k in ('G0', 'G1', 'G2', 'G3', 'G4')) and n_reps >= gates.min_final_replicates:
        verdict = 'PASS'
    else:
        verdict = 'INSUFFICIENT EVIDENCE'
    return {'candidate': candidate, 'reference': reference, 'replicates': n_reps, 'gates': g, 'verdict': verdict}


# --------------------------------------------------------------------------- markdown

def pct(x):
    return '–' if x is None or x != x else f'{100 * x:.1f}'


def rng_pct(s):
    return f'{pct(s["mean"])} ({pct(s["min"])}–{pct(s["max"])})'


def write_markdown(path, summary, models):
    lines = [f'# Sona evaluation results: {summary["run"]}', '',
             f'Mode `{summary["mode"]}`, protocol `{summary["protocol_sha256"][:12]}`, replicates {summary["seeds"]}. '
             'Values are means across replicates with (min–max). Window units unless stated. '
             f'Primary budget {PRIMARY} validation false-alarm rate; thresholds frozen on validation.', '']
    det = summary['detection']
    lines += ['## Recall / actual false-positive rate at the primary budget (%)', '',
              '| Condition | ' + ' | '.join(models) + ' |', '|---|' + '---|' * len(models)]
    for c in CONDITIONS:
        lines.append(f'| {c} | ' + ' | '.join(f'{pct(det[c][m][PRIMARY]["recall"]["mean"])} / {pct(det[c][m][PRIMARY]["false_positive_rate"]["mean"])}'
                                         for m in models) + ' |')
    lines += ['', '## ROC AUC (threshold-free ranking), mean (min–max)', '',
              '| Condition | ' + ' | '.join(models) + ' |', '|---|' + '---|' * len(models)]
    for c in CONDITIONS:
        lines.append(f'| {c} | ' + ' | '.join(f'{det[c][m]["roc_auc"]["mean"]:.3f} ({det[c][m]["roc_auc"]["min"]:.3f}–{det[c][m]["roc_auc"]["max"]:.3f})' for m in models) + ' |')
    sens = summary['post_hoc_nonchain_thresholds']
    lines += ['', '## POST HOC sensitivity (not a gate): thresholds from non-chain validation benigns, recall / FPR at 2% (%)', '',
              '| Condition | ' + ' | '.join(models) + ' |', '|---|' + '---|' * len(models)]
    for c in CONDITIONS:
        lines.append(f'| {c} | ' + ' | '.join(f'{pct(sens[m][c]["recall"]["mean"])} / {pct(sens[m][c]["false_positive_rate"]["mean"])}' for m in models) + ' |')
    lines += ['', '## Recall per budget, per-seed range (%)', '']
    for b in ('0.01', '0.02', '0.05'):
        lines += [f'### Budget {b}', '', '| Condition | ' + ' | '.join(models) + ' |', '|---|' + '---|' * len(models)]
        for c in CONDITIONS:
            lines.append(f'| {c} | ' + ' | '.join(f'{rng_pct(det[c][m][b]["recall"])} / FPR {pct(det[c][m][b]["false_positive_rate"]["mean"])}' for m in models) + ' |')
        lines.append('')
    lines += ['## Precision, F1, ROC AUC and flow units on `familiar` at the primary budget', '',
              '| Model | Precision | F1 | ROC AUC | Attack-flow coverage | Flow-weighted false alarm |', '|---|---|---|---|---|---|']
    for m in models:
        r = det['familiar'][m]
        lines.append(f'| {m} | {pct(r[PRIMARY]["precision"]["mean"])} | {pct(r[PRIMARY]["f1"]["mean"])} | {r["roc_auc"]["mean"]:.3f} | '
                     f'{pct(r[PRIMARY]["flow_attack_coverage"]["mean"])} | {pct(r[PRIMARY]["flow_weighted_false_alarm"]["mean"])} |')
    lines += ['', '## Memory: chain triples (paired, threshold-free, %)', '',
              '| Set | Model | attack>swap | attack>innocent | triple solved |', '|---|---|---|---|---|']
    for set_name, table in summary['memory'].items():
        for m in models:
            a = table[m]['all']
            lines.append(f'| {set_name} | {m} | {rng_pct(a["attack>swap"])} | {rng_pct(a["attack>innocent"])} | {rng_pct(a["triple_solved"])} |')
    lines += ['', '### Triple solved by distractor gap (%)', '']
    for set_name in CHAIN_SPLITS:
        groups = [g for g in summary['memory'][set_name][models[0]] if g.startswith('gap=')] + ['decoys', 'no_decoys']
        lines += [f'`{set_name}`', '', '| Model | ' + ' | '.join(groups) + ' |', '|---|' + '---|' * len(groups)]
        for m in models:
            lines.append(f'| {m} | ' + ' | '.join(pct(summary['memory'][set_name][m][g]['triple_solved']['mean']) for g in groups) + ' |')
        lines.append('')
    lines += ['## Order and history controls on `familiar` (recall / FPR at primary budget, ROC AUC)', '',
              '| Model | ' + ' | '.join(summary['controls']) + ' |', '|---|' + '---|' * len(summary['controls'])]
    for m in models:
        lines.append(f'| {m} | ' + ' | '.join(f'{pct(summary["controls"][s][m]["recall"]["mean"])} / {pct(summary["controls"][s][m]["false_positive_rate"]["mean"])}, {summary["controls"][s][m]["roc_auc"]["mean"]:.3f}'
                                         for s in summary['controls']) + ' |')
    lines += ['', '## Subtype flag rates at the primary budget (`familiar`, `slower`, `lookalike_heavy`, %)', '']
    for c in ('familiar', 'slower', 'lookalike_heavy', 'mixed'):
        names = list(summary['subtypes'][c][models[0]])
        lines += [f'`{c}`', '', '| Subtype | ' + ' | '.join(models) + ' |', '|---|' + '---|' * len(models)]
        for n in names:
            lines.append(f'| {n} | ' + ' | '.join(pct(summary['subtypes'][c][m].get(n, {}).get('mean')) for m in models) + ' |')
        lines.append('')
    for title, comp in summary['comparisons'].items():
        lines += [f'## Paired comparison: {title}', '', f'Macro recall difference {pct(comp["macro_recall"]["difference"])} points, '
                  f'95% CI [{pct(comp["macro_recall"]["ci95"][0])}, {pct(comp["macro_recall"]["ci95"][1])}].', '',
                  '| Condition | Recall diff (CI) | FPR diff (CI) | Verdict |', '|---|---|---|---|']
        for c, v in comp['conditions'].items():
            lo, hi = v['recall']['ci95']
            flo, fhi = v['false_positive_rate']['ci95']
            # Recall and actual false alarms judged together (a recall gain bought with
            # more false alarms is a tradeoff, not an improvement).
            if lo > 0:
                verdict = 'higher recall but more false alarms (tradeoff)' if flo > 0 else 'better'
            elif hi < 0:
                verdict = 'worse recall' + (', fewer false alarms' if fhi < 0 else '')
            elif flo > 0:
                verdict = 'more false alarms'
            elif fhi < 0:
                verdict = 'fewer false alarms'
            else:
                verdict = 'no demonstrated difference'
            lines.append(f'| {c} | {pct(v["recall"]["difference"])} [{pct(lo)}, {pct(hi)}] | '
                         f'{pct(v["false_positive_rate"]["difference"])} [{pct(v["false_positive_rate"]["ci95"][0])}, {pct(v["false_positive_rate"]["ci95"][1])}] | {verdict} |')
        for s in CHAIN_SPLITS:
            v = comp[f'{s}:triple_solved']
            lines.append(f'| {s} triple solved | {pct(v["difference"])} [{pct(v["ci95"][0])}, {pct(v["ci95"][1])}] | | |')
        lines.append('')
    if summary.get('decision'):
        d = summary['decision']
        lines += [f'## Gate decision: {d["candidate"]} vs {d["reference"]}: **{d["verdict"]}**', '', '```json',
                  json.dumps(d['gates'], indent=2, default=float), '```', '']
    Path(path).write_text('\n'.join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', type=Path)
    p.add_argument('--candidate', default=None)
    p.add_argument('--reference', default='sona_counterpart')
    p.add_argument('--compare', nargs=2, action='append', default=[], metavar=('A', 'B'))
    p.add_argument('--efficiency', type=Path, default=None)
    a = p.parse_args(argv)
    gates = Gates()
    reps = load_run(a.run_dir)
    run = json.loads((a.run_dir / 'run.json').read_text())
    protocol = json.loads((a.run_dir / 'protocol.json').read_text())
    models = models_of(reps)
    summary = {'run': str(a.run_dir), 'mode': run['mode'], 'seeds': [r['seed'] for r in reps],
               'protocol_sha256': protocol['protocol_sha256'], 'models': models,
               'detection': detection_tables(reps, models), 'subtypes': subtype_tables(reps, models),
               'memory': memory_tables(reps, models), 'controls': control_tables(reps, models), 'comparisons': {},
               'post_hoc_nonchain_thresholds': sensitivity_nonchain(reps, models)}
    pairs = list(a.compare)
    if a.candidate:
        pairs = [(a.candidate, a.reference), (a.candidate, 'beta1_counterpart'), (a.candidate, 'beta2_counterpart'),
                 (a.candidate, 'sona_public_saved')] + pairs
    for x, y in pairs:
        summary['comparisons'][f'{x} - {y}'] = compare(reps, x, y, gates)
    if a.candidate:
        comp = summary['comparisons'][f'{a.candidate} - {a.reference}']
        comp['pooled_vs_candidate'] = compare(reps, a.candidate, 'pooled_mlp', gates)['chain_familiar:triple_solved']
        efficiency = json.loads(a.efficiency.read_text()) if a.efficiency else None
        run_audit_path = a.run_dir / 'run_audit.json'
        run_audit = json.loads(run_audit_path.read_text()) if run_audit_path.exists() else {}
        summary['run_audit'] = run_audit
        # Replicate audits already passed (load_run refuses otherwise); add run-level preservation checks.
        # Amendment 5: the freeze also hashes the candidate's document; an edit to that
        # document alone (e.g. appended results) is reported but is not a code/artifact change.
        sealing_path = a.run_dir / 'sealing.json'
        if sealing_path.exists():
            from .adapters import ROOT, file_hash
            frozen = json.loads(sealing_path.read_text())['candidate_freeze']['hashes']
            now = {k: (file_hash(ROOT / k) if (ROOT / k).exists() else (file_hash(k) if Path(k).exists() else None)) for k in frozen}
            changed = sorted(k for k in frozen if now[k] != frozen[k])
            code_changed = [k for k in changed if not k.endswith('SONA_CANDIDATE.md')]
            run_audit['candidate_files_changed_since_freeze'] = changed
            run_audit['candidate_code_and_artifacts_unchanged'] = not code_changed
        checks = {k: v for k, v in run_audit.items() if k not in ('candidate_unchanged', 'candidate_files_changed_since_freeze')}
        audits_passed = bool(run_audit) and all(checks.values())
        summary['decision'] = gate_decision(a.candidate, a.reference, summary['detection'], comp, summary['memory'],
                                            efficiency, len(reps), gates, audits_passed)
        if run['mode'] != 'final':
            summary['decision']['verdict'] += ' (non-final run: not an acceptance decision)'
    (a.run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    write_markdown(a.run_dir / 'RESULTS.md', summary, models)
    print(f'wrote {a.run_dir / "summary.json"} and RESULTS.md' + (f'; verdict {summary["decision"]["verdict"]}' if a.candidate else ''))


if __name__ == '__main__':
    main()
