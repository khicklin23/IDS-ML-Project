"""Publish all frozen mixed-scan results without selecting favorable seeds."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


LABELS={'beta2_vote':'Connection voting','beta2_mean':'Connection mean score',
        'pooled_mlp':'Mean-pooled neural model','sona':'Sona-style GRU',
        'sona_shuffled_test':'Sona with shuffled observations','shuffled_train':'Shuffled-trained GRU',
        'summary_boosting':'Summary boosting','temporal_rule':'Temporal rule'}


def summarize(directory):
    report=json.loads((directory/'report.json').read_text())
    audit=json.loads((directory/'audit.json').read_text())
    if audit['status']!='passed':raise ValueError('An independently audited study is required.')
    rows=[]
    for run in report['runs']:
        for split,models in run['results'].items():
            for model,result in models.items():
                m=result['at_validation_false_alarm_target']
                rows.append({'seed':run['seed'],'test':split,'model':model,
                             'threshold':result['threshold_from_validation'],**m,
                             'roc_auc':result['roc_auc'],'average_precision':result['average_precision']})
    with (directory/'per_seed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def percent(value):return f'{value*100:.2f}%'
    lines=['# Mixed-traffic gradual-scan results','',
        f'Seeds: {", ".join(map(str,report["config"]["seeds"]))}. Independent audit: passed.',
        f'Total generated flows: {sum(sum(r["flow_counts"].values()) for r in report["runs"]):,}.',
        f'Observation windows: {len(report["runs"])*(report["config"]["train_windows"]+report["config"]["val_windows"]+3*report["config"]["test_windows"]):,}.','',
        'Each model has its own validation-selected threshold allowing at most 1% benign flags. Thresholds are frozen for all tests. Numbers below are means across seeds; parentheses show minimum to maximum. No public checkpoint or UI data was replaced.','']
    for split,title in [('test','Familiar mixed traffic'),('shifted','Longer windows and changed background'),('lookalikes','More legitimate inventory activity')]:
        lines += [f'## {title}','','| Model | Attack recall | Actual false-positive rate |','| --- | ---: | ---: |']
        for model,label in LABELS.items():
            m=report['summary'][split][model]
            cells=[f'{percent(m[k]["mean"])} ({percent(m[k]["min"])} to {percent(m[k]["max"])})' for k in ('recall','false_positive_rate')]
            lines.append(f'| {label} | {cells[0]} | {cells[1]} |')
        lines.append('')
    lines += ['## Harmless lookalikes','','False flags by benign behavior, pooled across seeds. These rates use each seed’s frozen threshold.','',
              '| Test | Model | Behavior | Benign windows | False flags | Rate |','| --- | --- | --- | ---: | ---: | ---: |']
    for split in ('test','shifted','lookalikes'):
        for model in LABELS:
            kinds=sorted(set().union(*(r['results'][split][model]['by_behavior'] for r in report['runs']))-{'gradual_scan'})
            for kind in kinds:
                cells=[r['results'][split][model]['by_behavior'].get(kind,{'windows':0,'flag_rate':0.}) for r in report['runs']]
                total=sum(c['windows'] for c in cells)
                flags=sum(round(c['windows']*c['flag_rate']) for c in cells)
                lines.append(f'| {split} | {LABELS[model]} | {kind} | {total} | {flags} | {percent(flags/total) if total else "N/A"} |')
    lines += ['','## Interpretation boundaries','',
        'Recall must be considered with actual test false alarms. Higher recall at a substantially higher false-alarm rate does not establish superiority. A 1% validation budget is not a guaranteed test operating point.','',
        'Order controls keep the same flow-vector multiset, including the gap feature attached to each vector. They test dependence on representation order; they are not realistic regenerated traffic. Similar ordered and shuffled performance weakens a claim that useful sequence order drives the result.','',
        'Connection voting uses repeated window labels, which are weak supervision for mixed normal/attack flows. The directly trained mean-pooling model is the stronger order-independent neural control. Summary boosting and the temporal rule can use time/port diversity without recurrent memory.','',
        'Balanced, generator-labeled complete windows do not prove malicious intent or operational alert precision. Legitimate inventory can resemble scanning. Three seeds show variability, not definitive statistical or real-world superiority. Original datasets, predictions, model weights, threshold selection, and independent audits remain available alongside this summary.']
    (directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report['summary']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    summarize(args.directory)
    print(f'Wrote {args.directory}/RESULTS.md and per_seed.csv')


if __name__=='__main__':main()
