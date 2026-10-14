"""Retain all results and apply the predeclared research gates."""
import argparse
import csv
import json
from pathlib import Path
from .study import SPECS, TESTS, write_json


def point(record, split, model):
    return record['results'][split][model]['0.01']['at_validation_false_alarm_target']


def gates(report):
    confirmations, hierarchy, details = [], [], []
    for record in report['runs']:
        memory_cells, hierarchical_cells = 0, 0
        for split in ('familiar', 'intermediate', 'longer'):
            relative = point(record, split, 'relative_mixed_gru')
            controls = [point(record, split, name) for name in ('relative_ordinary_gru', 'relative_mixed_pool')]
            good = relative['false_positive_rate'] <= .02 and all(
                relative['recall']-c['recall'] >= .05-1e-12 and abs(relative['false_positive_rate']-c['false_positive_rate']) <= .005+1e-12 for c in controls)
            memory_cells += int(good)
            hi = point(record, split, 'relative_mixed_hierarchical')
            # Timing comparison is reported separately; this gate tests quality only.
            hierarchical_cells += int(hi['false_positive_rate'] <= .02 and hi['recall'] >= relative['recall']-.02)
        inv = record['results']['inventory']['relative_mixed_gru']['0.01']
        safe = inv['at_validation_false_alarm_target']['false_positive_rate'] <= .02 and inv['by_behavior'].get('inventory', {'flag_rate': 1.})['flag_rate'] <= .10
        h_inv = record['results']['inventory']['relative_mixed_hierarchical']['0.01']
        h_safe = h_inv['at_validation_false_alarm_target']['false_positive_rate'] <= .02 and h_inv['by_behavior'].get('inventory', {'flag_rate': 1.})['flag_rate'] <= .10
        confirmations.append(memory_cells >= 2 and safe)
        hierarchy.append(hierarchical_cells >= 2 and h_safe)
        details.append({'seed': record['seed'], 'memory_gain_cells': memory_cells, 'inventory_gate': safe,
                        'hierarchical_quality_cells': hierarchical_cells, 'hierarchical_inventory_gate': h_safe})
    return {'confirmation_passed': len(confirmations) >= 3 and sum(confirmations) >= 2,
            'hierarchical_quality_passed': len(hierarchy) >= 3 and sum(hierarchy) >= 2,
            'hierarchical_efficiency_requires_independent_measurement': True,
            'details': details, 'public_promotion': False,
            'reason': 'Binary synthetic evidence alone cannot establish six-category or independent acceptance.'}


def summarize(directory):
    report = json.loads((directory/'report.json').read_text())
    audit = json.loads((directory/'audit.json').read_text())
    if audit['status'] != 'passed' or len(audit['runs']) != len(report['runs']): raise ValueError('Complete audit required.')
    rows = []
    for record in report['runs']:
        for split in ('val', *TESTS):
            for name in SPECS:
                for budget, metric in record['results'][split][name].items():
                    p = metric['at_validation_false_alarm_target']
                    rows.append({'seed': record['seed'], 'split': split, 'model': name, 'budget': budget,
                                 'recall': p['recall'], 'actual_fpr': p['false_positive_rate'],
                                 'inventory_fpr': metric['by_behavior'].get('inventory', {}).get('flag_rate', ''),
                                 'threshold': metric['threshold_from_validation'], 'tp': p['tp'], 'fp': p['fp'], 'fn': p['fn'], 'tn': p['tn']})
    with (directory/'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    decision = gates(report)
    write_json(directory/'decision.json', decision)
    text = ['# Sona candidate confirmation results', '', f'Validation-selected architecture: **{report["selected_global"]}**.', '',
            f'Confirmation gate passed: **{decision["confirmation_passed"]}**. Public promotion: **false**.', '',
            'All percentages below are strict-budget equal-weight seed means. Actual test false alarms are not fixed by the validation budget.', '',
            '| Test | Model | Recall | Actual false alarms |', '| --- | --- | ---: | ---: |']
    for split in TESTS:
        for name in SPECS:
            point = report['summary'][split][name]['0.01']
            text.append(f'| {split} | {name} | {100*point["recall"]["mean"]:.2f}% | {100*point["false_positive_rate"]["mean"]:.2f}% |')
    text += ['', 'Full per-seed 1%, 2%, and 5% budgets: `per_seed.csv`. Gate details: `decision.json`.',
             'Negative outcomes are preserved. Generated intent may be ambiguous between scan and benign inventory.',
             'Pooling on causal relative features already has timing context. Shuffling encoded inputs measures model sensitivity, not a physically regenerated intervention.',
             'Saved artifacts are binary-window candidates, not the public Sona checkpoint or production IDS evidence.']
    (directory/'RESULTS.md').write_text('\n'.join(text)+'\n')
    return decision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory', type=Path)
    print(json.dumps(summarize(parser.parse_args().directory), indent=2))
