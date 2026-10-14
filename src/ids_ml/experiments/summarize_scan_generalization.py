"""Audit-gated complete tables for the factorial generalization experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .mixed_scan_generalization import MODEL_NAMES
from .scan_generalization_features import CONDITIONS


def summarize(directory):
    report = json.loads((directory / 'report.json').read_text())
    audited = json.loads((directory / 'audit.json').read_text())
    if audited['status'] != 'passed':
        raise ValueError('Independent audit must pass before summarizing.')
    if [r['seed'] for r in audited['runs']] != list(report['config']['seeds']):
        raise ValueError('Audit does not cover all seeds.')
    rows = []
    for run in report['runs']:
        for condition, candidates in run['results'].items():
            for candidate, budgets in candidates.items():
                for budget, metrics in budgets.items():
                    operating = metrics['at_validation_false_alarm_target']
                    rows.append({'seed': run['seed'], 'condition': condition,
                                 'candidate': candidate, 'validation_budget': budget,
                                 'threshold': run['thresholds'][candidate][budget],
                                 'recall': operating['recall'],
                                 'false_positive_rate': operating['false_positive_rate']})
    with (directory / 'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    calibration = []
    for run in report['runs']:
        folder = directory / f"seed-{run['seed']}"
        frame = pd.read_parquet(folder / 'val.parquet')
        cells = frame.groupby('session_id', sort=False).condition.first().to_numpy()
        saved = np.load(folder / 'val_predictions.npz', allow_pickle=False)
        for condition in sorted(set(cells)):
            benign = (cells == condition) & (saved['y'] == 0)
            for candidate in MODEL_NAMES:
                for budget, threshold in run['thresholds'][candidate].items():
                    calibration.append({'seed': run['seed'], 'condition': condition,
                        'candidate': candidate, 'validation_budget': budget,
                        'benign_windows': int(benign.sum()),
                        'false_positive_rate': float(np.mean(saved[candidate][benign] >= threshold))})
    with (directory / 'validation_cells.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration[0]))
        writer.writeheader()
        writer.writerows(calibration)
    lines = ['# Generalization study results', '',
             'All numbers are from frozen validation-selected thresholds. Actual test false alarms may exceed the validation budget.', '',
             f"Total flows: {sum(sum(r['flow_counts'].values()) for r in report['runs']):,}.", '',
             'Recall and false-alarm rates below are equal-weight seed means, with seed ranges in parentheses. All candidates and budgets are retained.', '']
    for condition in CONDITIONS:
        for budget in report['summary'][condition][MODEL_NAMES[0]]:
            lines.extend([f'## {condition}: validation budget {float(budget):.0%}', '',
                          '| Candidate | Recall | Actual false-alarm rate |',
                          '| --- | ---: | ---: |'])
            for candidate in MODEL_NAMES:
                metrics = report['summary'][condition][candidate][budget]
                def fmt(key):
                    m = metrics[key]
                    return f"{m['mean']:.2%} ({m['min']:.2%} to {m['max']:.2%})"
                lines.append(f"| {candidate} | {fmt('recall')} | {fmt('false_positive_rate')} |")
            lines.append('')
    lines.extend(['## Limits', '', *['- ' + text for text in report['limitations']], '',
                  'validation_cells.csv reports the actual benign flag rate in each calibration cell; the budget constrains the mixture, not each cell separately.', '',
                  'Training covers all four duration/volume cells. This is fresh-sample generalization within those cells, not an unseen-duration experiment. No ordinary-only training control isolates augmentation. Public Sona is unchanged.'])
    (directory / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)


if __name__ == '__main__':
    main()
