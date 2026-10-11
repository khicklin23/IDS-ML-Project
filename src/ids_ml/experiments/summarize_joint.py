"""Summarize complete six-class candidates without selecting a winning seed."""
import argparse
import csv
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np


def summarize(directories, output):
    rows = []
    runs = []
    for directory in directories:
        report = json.loads((directory / 'report.json').read_text())
        benign = report['metadata']['class_names'].index('benign')
        audit = json.loads((directory / 'audit.json').read_text())
        assert audit['status'] == 'passed'
        config = report['metadata']['train_config']
        phase = ('expanded pace' if report.get('pace_augmentation') else
                 'gentler weighting' if config.get('class_weight_power', 1.) == .5 else 'initial joint')
        operating_path = directory/'alert_operating_point.json'
        extra_path = directory/'extra_stress/report.json'
        extra = json.loads(extra_path.read_text()) if extra_path.exists() else None
        if extra:
            assert extra['audit_status']=='passed'
        runs.append({'directory': str(directory), 'seed': report['seed'], 'phase': phase,
                     'flows': audit['flows'] + (sum(c['flows'] for c in extra['counts'].values()) if extra else 0),
                     'sessions': audit['sessions'] + (sum(c['sessions'] for c in extra['counts'].values()) if extra else 0),
                     'training_config': config, 'audit': audit,
                     'training_recipe': report['training_recipe'],
                     'polling_training_paces': report.get('polling_training_paces', [(5,25)]),
                     'alert_operating_point': json.loads(operating_path.read_text())
                         if operating_path.exists() else None, 'extra_stress':extra,
                     'pace_counterfactual': json.loads((directory/'extra_stress/pace_counterfactual.json').read_text())
                         if (directory/'extra_stress/pace_counterfactual.json').exists() else None})
        results_by_split={**report['results'], **(extra['results'] if extra else {})}
        for split, results in results_by_split.items():
            score_directory=directory/'extra_stress' if extra and split in extra['results'] else directory
            with np.load(score_directory / f'{split}_predictions.npz') as saved:
                n = len(saved['y'])
                for model in ('beta2', 'sona_hybrid'):
                    predictions = saved['beta2' if model == 'beta2' else 'sona']
                    correct = int(np.sum(predictions == saved['y']))
                    truth_attack = saved['y'] != benign
                    flagged = predictions != benign
                    tp = int(np.sum(flagged & truth_attack))
                    fp = int(np.sum(flagged & ~truth_attack))
                    fn = int(np.sum(~flagged & truth_attack))
                    assert np.isclose(correct/n, results[model]['accuracy'])
                    rows.append({'phase': phase, 'seed': report['seed'], 'test': split,
                                 'model': model, 'sessions': n, 'correct': correct,
                                 'accuracy': correct/n,
                                 'macro_f1': results[model]['macro_f1'],
                                 'detection_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                                 'attack_recall': tp/(tp+fn) if tp+fn else '',
                                 'false_alert_rate': fp/int((~truth_attack).sum())
                                     if (~truth_attack).any() else '',
                                 'false_alerts': n-correct if split in ('polling','unseen_polling') else '',
                                 'shuffled_accuracy': results.get('sona_shuffled', {}).get('accuracy', '')
                                     if model == 'sona_hybrid' else ''})
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'per_seed.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    aggregate = []
    for phase, test, model in sorted({(r['phase'], r['test'], r['model']) for r in rows}):
        selected = [r for r in rows if (r['phase'], r['test'], r['model']) == (phase,test,model)]
        accuracies = [r['accuracy'] for r in selected]
        aggregate.append({'phase': phase, 'test': test, 'model': model,
                          'runs': len(selected), 'mean_accuracy': float(np.mean(accuracies)),
                          'min_accuracy': min(accuracies), 'max_accuracy': max(accuracies),
                          'mean_macro_f1': float(np.mean([r['macro_f1'] for r in selected])),
                          'mean_detection_f1':float(np.mean([r['detection_f1'] for r in selected])),
                          'sessions': sum(r['sessions'] for r in selected),
                          'correct': sum(r['correct'] for r in selected)})
    summary = {'runs': runs, 'aggregate': aggregate,
               'flows': sum(r['flows'] for r in runs),
               'environment': {'python': platform.python_version(),
                   **{p:version(p) for p in ('numpy','pandas','scikit-learn','torch','pyarrow')}},
               'limitations': ['All outcomes retained; no best-seed selection.',
                   'Recipe phases also use different seeds; this is not a causal weighting ablation.',
                   'Macro F1 is meaningful for ordinary six-class tests; timing accuracy is reported separately.',
                   'False-alert rate is measured on explicitly benign polling, not a live network.',
                   'Expanded-pace training includes 5–120 second paired gaps and 5–600 second polling.',
                   'Extreme-pace paired tests use 150–300 seconds, beyond paired training pace.',
                   'All relative features require completed sessions.']}
    source_root = Path(__file__).resolve().parents[1]
    summary['source_hashes_at_summary'] = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [source_root/'features.py', source_root/'models/sequence.py',
                     source_root/'models/train.py', source_root/'experiments/joint_temporal.py',
                     source_root/'experiments/temporal_patterns.py',
                     source_root/'experiments/temporal_robustness.py']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    text = ['# Six-class temporal candidate results', '',
            'Accuracy means the percentage of sessions classified correctly. Each timing pair has',
            'the same individual measurements in a different order. Beta 2 sees each flow separately;',
            'Sona sees the ordered session. All runs are shown, including unsuccessful candidates.', '',
            f"Total generated flows across candidates: **{summary['flows']:,}**.", '']
    for phase in dict.fromkeys(r['phase'] for r in runs):
        text += [f'## {phase.capitalize()}', '',
                 '| Test | Beta 2 accuracy | Sona accuracy | Sona range |',
                 '|---|---:|---:|---:|']
        selected = [r for r in aggregate if r['phase'] == phase]
        labels = {'normal':'Standard traffic', 'stealth':'Less obvious attacks',
                  'timing':'Repeated timing', 'slow_timing':'Slower, noisier cycles',
                  'extreme_pace':'Much slower cycles: new paired pace',
                  'extra_noise':'Heavier timing noise',
                  'known_long_cycles':'Long cycles included in training',
                  'unseen_long_cycles':'Longer cycles absent from training',
                  'subtle_timing':'Small timing differences',
                  'polling':'Harmless slow polling: correctly ignored',
                  'unseen_polling':'Harmless polling beyond training pace: correctly ignored'}
        for split, label in labels.items():
            sona = next((r for r in selected if r['test']==split and r['model']=='sona_hybrid'), None)
            if sona is None: continue
            beta = next(r for r in selected if r['test']==sona['test'] and r['model']=='beta2')
            text.append(f"| {label} | {beta['mean_accuracy']:.2%} | "
                        f"{sona['mean_accuracy']:.2%} | {sona['min_accuracy']:.2%}–{sona['max_accuracy']:.2%} |")
        poll = next(r for r in selected if r['test']=='polling' and r['model']=='sona_hybrid')
        text += ['', f"Sona falsely flagged **{poll['sessions']-poll['correct']} of {poll['sessions']}** "
                 f"benign polling sessions ({1-poll['correct']/poll['sessions']:.2%}).", '']
        unseen=next((r for r in selected if r['test']=='unseen_polling' and r['model']=='sona_hybrid'),None)
        if unseen:
            text += [f"Beyond training pace, Sona falsely flagged **{unseen['sessions']-unseen['correct']} "
                     f"of {unseen['sessions']}** benign polling sessions "
                     f"({1-unseen['correct']/unseen['sessions']:.2%}).", '']
        shuffle = [r['shuffled_accuracy'] for r in rows if r['phase']==phase and
                   r['test']=='timing' and r['model']=='sona_hybrid']
        text += [f"Shuffling familiar timing at inference: mean accuracy **{np.mean(shuffle):.2%}**.", '']
    text += ['## Scope', '', *summary['limitations'], '',
             'Standard and less-obvious traffic cover all six classes. Timing tests contain benign and beaconing only.',
             'For polling, higher accuracy means fewer false alerts; every polling session is benign.',
             'Use per_seed.csv for each seed and summary.json for configuration, environment, and audit results.',
             'Public dashboard/model files have not been replaced by these experiments.', '']
    (output / 'RESULTS.md').write_text('\n'.join(text))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directories, args.out_dir)
    print(json.dumps({'flows':result['flows'], 'runs':len(result['runs'])}))
