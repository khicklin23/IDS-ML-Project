"""Audited all-arm results, frozen gates and a reproducible candidate decision."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .study import TESTS, SPECS, write_json
from .summarize import gates


def report(study, controls, extension, output):
    for directory in (study, controls, extension):
        if json.loads((directory/'audit.json').read_text())['status'] != 'passed': raise ValueError('All audits required.')
    base = json.loads((study/'report.json').read_text())
    pooled = json.loads((controls/'report.json').read_text())
    extended = json.loads((extension/'report.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for run in base['runs']:
        records.extend({'seed': run['seed'], 'model': name, 'results': {s: run['results'][s][name] for s in ('val', *TESTS)}} for name in SPECS)
    records.extend({'seed': r['seed'], 'model': f'relative_{r["regime"]}_pool_matched', 'results': r['results']} for r in pooled['runs'])
    records.extend({'seed': r['seed'], 'model': r['architecture'], 'results': r['results']} for r in extended['runs'])
    rows = []
    for r in records:
        for split, result in r['results'].items():
            for budget, metric in result.items():
                p = metric['at_validation_false_alarm_target']
                precision = p['tp']/max(p['tp']+p['fp'], 1)
                f1 = 2*p['tp']/max(2*p['tp']+p['fp']+p['fn'], 1)
                rows.append({'seed': r['seed'], 'model': r['model'], 'split': split, 'budget': budget,
                             'recall': p['recall'], 'actual_fpr': p['false_positive_rate'], 'precision': precision, 'f1': f1,
                             'inventory_fpr': metric['by_behavior'].get('inventory', {}).get('flag_rate', ''),
                             'threshold': metric['threshold_from_validation']})
    with (output/'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries = {}
    names = sorted(set(r['model'] for r in records))
    for split in TESTS:
        summaries[split] = {}
        for name in names:
            summaries[split][name] = {}
            for budget in ('0.01', '0.02', '0.05'):
                subset = [r for r in rows if r['model'] == name and r['split'] == split and r['budget'] == budget]
                summaries[split][name][budget] = {k: {'mean': float(np.mean([r[k] for r in subset])),
                    'min': float(min(r[k] for r in subset)), 'max': float(max(r[k] for r in subset))} for k in ('recall', 'actual_fpr', 'precision', 'f1')}
    retrieval_details = []
    for r in extended['runs']:
        if r['architecture'] != 'retrieval': continue
        original = next(b for b in base['runs'] if b['seed'] == r['seed'])
        successes = 0
        for split in ('familiar', 'intermediate', 'longer'):
            candidate = r['results'][split]['0.01']['at_validation_false_alarm_target']
            control = original['results'][split]['relative_mixed_gru']['0.01']['at_validation_false_alarm_target']
            successes += int(candidate['recall'] >= control['recall']+.05-1e-12 and candidate['false_positive_rate'] <= .02
                and abs(candidate['false_positive_rate']-control['false_positive_rate']) <= .005+1e-12)
        inv = r['results']['inventory']['0.01']
        safe = inv['at_validation_false_alarm_target']['false_positive_rate'] <= .02 and inv['by_behavior'].get('inventory', {'flag_rate': 1.})['flag_rate'] <= .10
        retrieval_details.append({'seed': r['seed'], 'gain_cells': successes, 'inventory_safe': safe, 'passed': successes >= 2 and safe})
    original_decision = gates(base)
    retrieval_passed = len(retrieval_details) >= 3 and sum(r['passed'] for r in retrieval_details) >= 2
    selected = extended['selection']['selected']
    selected_gate = retrieval_passed if selected == 'retrieval' else (original_decision['confirmation_passed'] if selected == 'relative_mixed_gru' else False)
    result = {'selection': extended['selection'], 'summary': summaries,
              'original_gates': original_decision, 'retrieval_gate': {'passed': retrieval_passed, 'details': retrieval_details},
              'selected_binary_gate_passed': selected_gate, 'public_promotion': False,
              'independent_acceptance': 'pending separate suite',
              'six_category_next_step': 'Run separate six-category regression if independent acceptance also supports candidate.' if selected_gate else 'Deferred: selected candidate did not meet frozen binary quality gate.',
              'fresh_flows': sum(sum(r['flow_counts'].values()) for r in base['runs']),
              'fresh_windows': len(base['runs'])*(2*base['config']['train_windows']+base['config']['val_windows']+len(TESTS)*base['config']['test_windows']),
              'full_trained_models': len(records), 'csv_rows': len(rows)}
    write_json(output/'summary.json', result)
    lines = ['# Final experimental Sona candidate report', '', f'Validation-selected candidate: **{selected}**.', '',
             f'Frozen binary quality gate passed: **{selected_gate}**. Public promotion: **false**.', '',
             f'{result["fresh_flows"]:,} fresh flows, {result["fresh_windows"]:,} windows, {len(records)} full trained models.', '',
             'Strict 1% validation-budget means below; actual false alarms vary. All seeds and 1/2/5% budgets are retained in per_seed.csv.', '',
             '| Condition | Candidate | Recall | Actual false alarms |', '| --- | --- | ---: | ---: |']
    for split in TESTS:
        for name in names:
            p = summaries[split][name]['0.01']
            lines.append(f'| {split} | {name} | {100*p["recall"]["mean"]:.2f}% | {100*p["actual_fpr"]["mean"]:.2f}% |')
    lines += ['', 'Selection used mean validation recall only, never these test results.',
              'The matched pooling control has nearly the same parameter count as GRU32. GRU40 matches the retrieval model size.',
              'Causal input preparation, saved prediction replay and frozen thresholds were audited.',
              'Two learned retrieval hops are a computational mechanism, not proof of human reasoning or production IDS accuracy.',
              'Legitimate inventory overlaps scanning; unresolved intent ambiguity cannot be fixed by claiming a deeper network.',
              result['six_category_next_step'], 'Independent chain-memory and efficiency acceptance remains separate.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('study', 'controls', 'extension', 'output'): p.add_argument(arg, type=Path)
    a = p.parse_args(); print(json.dumps(report(a.study, a.controls, a.extension, a.output)['selection'], indent=2))
