"""Audited context-study tables retaining every candidate and operating point."""
import argparse
import csv
import json
from pathlib import Path

LABELS={'raw_sona':'Raw GRU','pace_sona':'Pace-aware GRU','context_sona':'Context GRU',
        'context_pool':'Context neural pooling','context_vote':'Context connection voting',
        'context_mean':'Context connection mean','summary_boosting':'Original summary boosting',
        'context_summary':'Context summary boosting','context_shuffled':'Context GRU shuffled',
        'temporal_rule':'Temporal rule'}


def summarize(path):
    r=json.loads((path/'report.json').read_text());a=json.loads((path/'audit.json').read_text())
    if a['status']!='passed':raise ValueError('Independent audit must pass first.')
    rows=[]
    for run in r['runs']:
        for split,models in run['results'].items():
            for model,budgets in models.items():
                for budget,m in budgets.items():
                    rows.append(dict(seed=run['seed'],test=split,model=model,validation_budget=budget,
                        threshold=m['threshold_from_validation'],**m['at_validation_false_alarm_target'],
                        roc_auc=m['roc_auc'],average_precision=m['average_precision']))
    with (path/'per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    percent=lambda x:f'{100*x:.2f}%'
    lines=['# Context and pace results','',
           f'Seeds: {r["config"]["seeds"]}. All candidates independently audited.',
           f'Fresh flows: {sum(sum(x["flow_counts"].values()) for x in r["runs"]):,}.',
           'All thresholds are frozen from validation; columns show actual test outcomes. No public model was replaced.','']
    for split,title in [('test','Familiar mixed traffic'),('shifted','Longer windows / changed background'),('lookalikes','More legitimate inventory')]:
        for budget in ('0.01','0.02','0.05'):
            lines += [f'## {title}, {percent(float(budget))} validation budget','',
                      '| Candidate | Scan recall | Actual false-positive rate | Recall range across seeds |',
                      '| --- | ---: | ---: | --- |']
            for model,label in LABELS.items():
                m=r['summary'][split][model][budget]
                lines.append(f'| {label} | {percent(m["recall"]["mean"])} | {percent(m["false_positive_rate"]["mean"])} | {percent(m["recall"]["min"])} to {percent(m["recall"]["max"])} |')
            lines.append('')
    lines += ['## Parameter counts','', '| Candidate | Parameters per seed |','| --- | --- |']
    for model in ('raw_sona','pace_sona','context_sona','context_pool','context_vote'):
        lines.append(f'| {LABELS[model]} | '+', '.join(str(x['parameter_counts'][model]) for x in r['runs'])+' |')
    lines += ['','## Interpretation boundaries','',
        'Improvement over the raw GRU measures the combined effect of additional observed context and its representation. Context pooling and voting already receive deterministic history features; their networks are not recurrent, but their inputs are not history-free.','',
        'Higher recall at a higher actual test false-positive rate is a tradeoff, not a same-error-rate victory. Raising validation budgets from 1% to 5% does not create a better model. Every operating point, seed and condition is retained.','',
        'Context is derived causally, but classification still consumes complete windows. No online delay, production precision, malicious-intent identification or public checkpoint promotion is established.']
    (path/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    a=p.parse_args();summarize(a.directory);print('Wrote RESULTS.md and per_seed.csv.')


if __name__=='__main__':main()
