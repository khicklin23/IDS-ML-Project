"""All-seed matched timing/noise results and predeclared candidate eligibility."""
import argparse
import csv
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(root):
    seeds=(701,733,769)
    records=[];rows=[]
    for seed in seeds:
        directory=root/f'seed_{seed}'
        record=json.loads((directory/'study.json').read_text())
        assert record['matched_raw_datasets']=='passed'
        relative=json.loads((directory/'relative_study.json').read_text())
        record['variants']['relative']=relative
        counts=json.loads((directory/'hybrid/report.json').read_text())['counts']
        for name in counts:
            pd.testing.assert_frame_equal(pd.read_parquet(directory/'hybrid'/f'{name}.parquet'),
                                          pd.read_parquet(directory/'relative'/f'{name}.parquet'))
        for name in relative['extra']['counts']:
            pd.testing.assert_frame_equal(pd.read_parquet(directory/'hybrid/extra_stress'/f'{name}.parquet'),
                                          pd.read_parquet(directory/'relative/extra_stress'/f'{name}.parquet'))
        records.append(record)
        for mode,variant in record['variants'].items():
            audit=json.loads((directory/mode/'audit.json').read_text())
            assert audit['status']=='passed' and variant['extra']['audit_status']=='passed'
            for split,metrics in {**variant['results'],**variant['extra']['results']}.items():
                for model in ('beta2','sona_hybrid'):
                    score=metrics[model]
                    alert=variant['alerts']['tests'][split] if model=='sona_hybrid' else None
                    rows.append({'seed':seed,'mode':mode,'test':split,'model':model,
                        'sessions':score['n'],'accuracy':score['accuracy'],'macro_f1':score['macro_f1'],
                        'native_false_alert_rate':1-score['accuracy'] if 'polling' in split else '',
                        'alert_false_rate':alert['false_alert_rate'] if alert else '',
                        'alert_attack_recall':alert['attack_recall'] if alert else '',
                        'shuffled_accuracy':metrics.get('sona_shuffled',{}).get('accuracy','') if model=='sona_hybrid' else ''})
                baseline=variant['timing_summary']['results'].get(split)
                if baseline:
                    rows.append({'seed':seed,'mode':mode,'test':split,'model':'timing_summary',
                        'sessions':baseline['sessions'],'accuracy':baseline['accuracy'],'macro_f1':'',
                        'native_false_alert_rate':baseline['false_alert_rate'],
                        'alert_false_rate':baseline['validation_threshold_false_alert_rate'],
                        'alert_attack_recall':baseline['validation_threshold_attack_recall'],
                        'shuffled_accuracy':''})
    aggregate=[]
    for mode,test,model in sorted({(r['mode'],r['test'],r['model']) for r in rows}):
        selected=[r for r in rows if (r['mode'],r['test'],r['model'])==(mode,test,model)]
        values=[r['accuracy'] for r in selected]
        aggregate.append({'mode':mode,'test':test,'model':model,'mean_accuracy':float(np.mean(values)),
                          'min_accuracy':min(values),'max_accuracy':max(values),
                          'sessions':sum(r['sessions'] for r in selected)})
    gates={}
    for mode in ('hybrid','hybrid_log','relative'):
        def native(split):return [r for r in rows if r['mode']==mode and r['test']==split and r['model']=='sona_hybrid']
        def mean(split):return float(np.mean([r['accuracy'] for r in native(split)]))
        conditions={
            'standard_and_stealth_each_macro_f1_at_least_95pct':all(r['macro_f1']>=.95 for split in ('normal','stealth') for r in native(split)),
            'familiar_mean_95pct_each_90pct':mean('timing')>=.95 and all(r['accuracy']>=.90 for r in native('timing')),
            'noise_mean_80pct_each_70pct':mean('extra_noise')>=.80 and all(r['accuracy']>=.70 for r in native('extra_noise')),
            'benign_polling_each_false_flag_rate_at_most_1pct':all(r['native_false_alert_rate']<=.01+1e-12 for split in ('polling','unseen_polling','extreme_polling') for r in native(split)),
            'subtle_and_extreme_pace_mean_at_least_70pct':all(mean(split)>=.70 for split in ('subtle_timing','extreme_pace'))}
        gates[mode]={'conditions':conditions,'eligible':all(conditions.values())}
    # The same raw dataset is generated for both variants: count once per seed.
    unique_flows=0;unique_sessions=0
    for seed in seeds:
        report=json.loads((root/f'seed_{seed}/hybrid/report.json').read_text())
        extra=records[list(seeds).index(seed)]['variants']['hybrid']['extra']
        unique_flows+=sum(c['flows'] for c in report['counts'].values())+sum(c['flows'] for c in extra['counts'].values())
        unique_sessions+=sum(c['sessions'] for c in report['counts'].values())+sum(c['sessions'] for c in extra['counts'].values())
    result={'seeds':list(seeds),'unique_flows':unique_flows,'unique_sessions':unique_sessions,
            'candidate_gate':gates,'aggregate':aggregate,'records':records,
            'selection':'Seed701 preselected if a mode qualifies. Gate is synthetic showcase eligibility, not production validity.',
            'limitations':['Six-class normal/stealth tests are smaller than conditional timing tests.',
                'Timing summary sees engineered sequence statistics and only binary temporal labels.',
                'Jitter .25 is included in the new training noise range; .45 is beyond it.',
                'Benign650–1000 and1500–3000second base polling paces lie beyond the configured training upper pace600seconds.',
                'All neural variants require closed complete sessions.']}
    result['selected_mode']=next((mode for mode in ('hybrid_log','relative','hybrid') if gates[mode]['eligible']),None)
    result['selected_seed']=701 if result['selected_mode'] else None
    result['environment']={'python':platform.python_version(),
                          **{name:version(name) for name in ('numpy','pandas','scikit-learn','torch','pyarrow')}}
    source=Path(__file__).resolve().parents[1]
    result['source_hashes_at_summary']={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [source/'features.py',source/'models/sequence.py',source/'models/train.py',
                  source/'experiments/joint_temporal.py',source/'experiments/range_study.py',
                  source/'experiments/extra_stress_joint.py']}
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    with (root/'per_seed.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Matched pace/noise study','',f"Three seeds, **{unique_flows:,} unique generated flows**, nine neural candidates.",
          'All preparations use identical raw data and hyperparameters. Hybrid keeps raw and relative gaps; hybrid_log compresses the raw gap; relative discards absolute pace.',
          'Every variant receives the expanded noise recipe, so this compares gap preparation, not the effect of noise augmentation alone.','']
    labels={'normal':'Standard traffic','stealth':'Less obvious attacks','timing':'Repeated timing',
            'slow_timing':'Slower, noisier cycles','extra_noise':'Heavier noise included in new training range',
            'extreme_noise':'Noise beyond training range','known_long_cycles':'Known long cycles',
            'unseen_long_cycles':'New cycle lengths','subtle_timing':'Small timing differences',
            'extreme_pace':'Much slower paired cycles','polling':'Harmless slow polling',
            'unseen_polling':'Harmless polling:650–1000seconds','extreme_polling':'Harmless polling:1500–3000seconds'}
    for mode in ('hybrid','hybrid_log','relative'):
        text += [f'## {mode}', '', '| Test | Beta 2 accuracy | Sona accuracy | Sona range | Timing-summary accuracy |',
                 '|---|---:|---:|---:|---:|']
        for split,label in labels.items():
            def get(model):return next((a for a in aggregate if (a['mode'],a['test'],a['model'])==(mode,split,model)),None)
            beta,sona,baseline=get('beta2'),get('sona_hybrid'),get('timing_summary')
            text.append(f"| {label} | {beta['mean_accuracy']:.2%} | {sona['mean_accuracy']:.2%} | "
                        f"{sona['min_accuracy']:.2%}–{sona['max_accuracy']:.2%} | "+(f"{baseline['mean_accuracy']:.2%}" if baseline else 'Not a six-class model')+' |')
        text+=['',f"Predeclared eligibility gate: **{'passed' if gates[mode]['eligible'] else 'not passed'}**.",
               *[f"- {name}: {'passed' if value else 'failed'}" for name,value in gates[mode]['conditions'].items()],'']
    text+=['## Interpretation','',
           'Polling accuracy is the percentage correctly ignored; lower accuracy means more false alerts.',
           'Threshold-selected attack recall and false-alert rates are separate columns in per_seed.csv. Thresholds use validation only.',
           'Shuffled sequence controls and every seed are retained. No best-test-seed selection.',
           'The public UI Overview scores and model artifacts remain separate from these candidates.','']
    text += [f"Predeclared selection: **{result['selected_mode'] or 'no eligible candidate'}**. Public model is unchanged until a consistent export is prepared.", '']
    (root/'RESULTS.md').write_text('\n'.join(text))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();print(json.dumps(summarize(args.root)['candidate_gate'],indent=2))
