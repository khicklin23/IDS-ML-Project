"""Aggregate every seed from the matched four-model follow-up."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(root, original_summary):
    prior = json.loads(original_summary.read_text())
    seeds = [701, 733, 769]
    assert prior['seeds'] == seeds
    records = [json.loads((root/f'seed_{seed}/report.json').read_text()) for seed in seeds]
    tests = list(records[0]['results'])
    assert all(list(record['results']) == tests and record['audit']['status'] == 'passed'
               and record['audit']['reload_checks_all_four_models'] == len(tests) for record in records)
    rows = []
    aggregate = []
    for test in tests:
        for version in ('v1', 'v2', 'v3', 'v4'):
            metrics = [record['results'][test][version] for record in records]
            for seed, metric in zip(seeds, metrics):
                rows.append({'seed':seed, 'version':version, 'test':test, 'sessions':metric['n'],
                             'accuracy':metric['accuracy'], 'macro_f1':metric['macro_f1'],
                             'shuffled_accuracy':next(r for r in records if r['seed']==seed)['results'][test].get('v3_shuffled',{}).get('accuracy','') if version=='v3' else ''})
            scores = [metric['accuracy'] for metric in metrics]
            summary = {'version':version, 'test':test, 'mean':float(np.mean(scores)),
                       'min':min(scores), 'max':max(scores), 'sessions':sum(m['n'] for m in metrics)}
            if version in ('v2', 'v4'):
                reference = next(a for a in prior['aggregate'] if (a['mode'],a['test'],a['model']) ==
                                 ('relative',test,'beta2' if version=='v2' else 'sona_hybrid'))
                assert summary['mean'] == reference['mean_accuracy']
                assert summary['min'] == reference['min_accuracy'] and summary['max'] == reference['max_accuracy']
                assert summary['sessions'] == reference['sessions']
            aggregate.append(summary)
    totals = {}
    for version in ('v1','v2','v3','v4'):
        per_seed = [float(np.mean([r['results'][test][version]['accuracy'] for test in ('normal','stealth','timing')])) for r in records]
        totals[version] = {'mean':float(np.mean(per_seed)), 'min':min(per_seed), 'max':max(per_seed)}
    result = {'schema':1, 'seeds':seeds, 'tests':tests, 'aggregate':aggregate, 'three_test_scores':totals,
              'shared_unique_flows':prior['unique_flows'], 'shared_unique_sessions':prior['unique_sessions'],
              'source_study':str(original_summary), 'records':records,
              'audit_status':'passed', 'summary_policy':'Mean accuracy per seed, then equal average of ordinary, slowed-down and timing means.',
              'scope':'Retrained candidates on known saved benchmark; not original public weights. Speed not measured.'}
    (root/'summary.json').write_text(json.dumps(result, indent=2))
    with (root/'per_seed.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Four models on identical test sessions', '',
          'All four models receive identical temporal training examples. Beta1/Beta3 exclude the ordinary stealth training run; Beta2/Sona include it.',
          'The shared benchmark contains 2,171,844 unique flows; no new data was generated. Every saved-model reload and source hash audit passed.', '',
          '| Test | Beta 1 | Beta 2 | Beta 3 | Sona candidate |','|---|---:|---:|---:|---:|']
    for test in tests:
        values=[next(a for a in aggregate if (a['version'],a['test'])==(version,test)) for version in ('v1','v2','v3','v4')]
        text.append('| '+test+' | '+' | '.join(f'{v["mean"]:.2%} ({v["min"]:.2%}–{v["max"]:.2%})' for v in values)+' |')
    text+=['', '| Equal-weight three-test score | Beta 1 | Beta 2 | Beta 3 | Sona candidate |',
           '|---|---:|---:|---:|---:|', '| Ordinary + stealth + timing | '+' | '.join(f'{totals[v]["mean"]:.2%}' for v in ('v1','v2','v3','v4'))+' |', '',
           'The three-test score is a descriptive average, not pooled accuracy or a production IDS rating. Do not compare it with the older two-test score.',
           'Source checkpoints and preprocessing remain frozen; new Beta1/Beta3 counterparts are fit only on their training subsets.',
           'Temporal conditional tests have only two true classes; accuracy is meaningful there while six-class macro F1 includes absent-class zeros.',
           'Beta3 shuffle controls and every seed remain in per_seed.csv. No claim of faster inference or universal superiority.','']
    (root/'RESULTS.md').write_text('\n'.join(text))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--original-summary',type=Path,default=Path('data/generated/sona_pace_noise_v1/summary.json'))
    args=parser.parse_args();print(json.dumps(summarize(args.root,args.original_summary)['three_test_scores'],indent=2))
