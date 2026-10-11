"""Write readable tables for a completed temporal benchmark, without rerunning training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

NAMES = {
    'beta2': 'Beta 2 architecture',
    'sona': 'Sona architecture',
    'sona_shuffled_test': 'Sona: test order shuffled',
    'sona_shuffled_train': 'Sona: trained and tested with shuffled order',
    'timing_summary_logistic': 'Simple timing-summary classifier',
    'last_flow_logistic': 'Last-flow classifier',
}


def summarize(directory: Path) -> Path:
    report = json.loads((directory / 'report.json').read_text())
    records = report['runs']
    total_flows = sum(m['flows'] for run in records for m in run['manifests'].values())
    total_sessions = sum(m['sessions'] for run in records for m in run['manifests'].values())
    lines = ['# Separate temporal-pattern benchmark', '',
             f'{len(records)} independently generated and trained runs; {total_flows:,} flows in '
             f'{total_sessions:,} sessions, including training, validation, and testing.', '',
             'A flow is one network conversation. A session is an ordered group of flows. '
             'Every attack-like session has a benign partner with the same flow measurements, '
             'length, and endpoints. The order of the interior flows is the only difference.', '',
             '**These are freshly trained two-class versions of the existing architectures. '
             'The public six-class model weights and showcase results are unchanged.**', '',
             'Accuracy is the percentage of sessions classified correctly. Macro F1 balances '
             'missed attacks and false alarms for both classes. AUC measures how well scores '
             'rank attack-like sessions above benign ones: 50% is chance; 100% is perfect.', '']
    for split, title in [('test', 'Unseen sessions with familiar timing ranges'),
                         ('shifted', 'Unseen, slower and noisier timing patterns')]:
        lines.extend([f'## {title}', '',
                      '| Model / control | Accuracy | Macro F1 | AUC | Attack recall at validation-selected threshold | Actual benign false alarms |',
                      '|---|---:|---:|---:|---:|---:|'])
        for name, row in report['summary'][split].items():
            values = [row[key]['mean'] for key in ('accuracy', 'macro_f1', 'roc_auc', 'at_target_recall', 'at_target_false_positive_rate')]
            lines.append('| ' + NAMES[name] + ' | ' + ' | '.join(f'{v:.2%}' for v in values) + ' |')
        lines.extend(['', 'Each value is the average across runs. Thresholds were selected using '
                      'validation data to allow at most 1% false alarms there. The last column '
                      'shows the actual rate on unseen benign sessions; it is not guaranteed to be 1%.', '',
                      '| Seed | Beta 2 F1 | Sona F1 | Sona advantage, percentage points |',
                      '|---|---:|---:|---:|'])
        for run in records:
            b = run['results'][split]['beta2']['macro_f1']
            s = run['results'][split]['sona']['macro_f1']
            lines.append(f'| {run["seed"]} | {b:.2%} | {s:.2%} | {(s-b)*100:+.2f} |')
        lines.append('')
    lines.extend(['## How to interpret the controls', '',
                  '- Beta 2 sees each flow separately. Identical flow collections cannot tell it which order occurred.',
                  '- Shuffling the order after training tests whether Sona relies on the sequence.',
                  '- Training with shuffled order removes that information from both learning and testing.',
                  '- The last-flow classifier tests whether the endpoint alone explains a result.',
                  '- The timing-summary classifier uses explicit calculations of repeated timing. If it succeeds, a neural network is not uniquely necessary.', '',
                  'This deliberately controlled test demonstrates timing-order discrimination, not '
                  'real-world intrusion detection or malicious intent. Scheduled benign software '
                  'can also produce repeating traffic. Test those lookalikes before making broader claims.', '',
                  'Raw split data, checkpoints, preprocessing, training histories, and prediction scores '
                  'are saved in each seed directory. `report.json` contains aggregate results; '
                  '`per_seed_metrics.csv` is suitable for a spreadsheet.', ''])
    destination = directory / 'RESULTS.md'
    destination.write_text('\n'.join(lines))
    with (directory / 'per_seed_metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seed', 'split', 'model', 'sessions', 'accuracy', 'macro_f1', 'roc_auc',
                         'average_precision', 'validation_threshold', 'recall', 'false_positive_rate'])
        for run in records:
            for split, models in run['results'].items():
                for name, row in models.items():
                    writer.writerow([run['seed'], split, name, row['sessions'], row['accuracy'],
                                     row['macro_f1'], row['roc_auc'], row['average_precision'],
                                     row['threshold_from_validation'],
                                     row['at_validation_false_alarm_target']['recall'],
                                     row['at_validation_false_alarm_target']['false_positive_rate']])
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(summarize(args.directory))


if __name__ == '__main__':
    main()
