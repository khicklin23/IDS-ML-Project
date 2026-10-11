"""Matched raw/log timing experiments on fresh six-class data.

Each seed runs both preprocessing variants; every raw split must match exactly.
No dashboard artifacts are replaced by this study.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .joint_temporal import run
from .audit_joint import audit
from .extra_stress_joint import measure as extra
from .operating_point_joint import measure as operating
from .temporal_patterns import summary_features, threshold_for_false_alarms


def timing_summary(directory):
    """Binary timing-summary baseline; deliberately excludes ordinary six-class tests."""
    def prepare(df):
        df=df.sort_values('event_time',kind='stable').reset_index(drop=True)
        groups=list(df.groupby('session_id',sort=False).indices.values())
        return summary_features(df,groups),np.array([int(df.iloc[g[0]].target) for g in groups])
    train=pd.read_parquet(directory/'train.parquet')
    val=pd.read_parquet(directory/'val.parquet')
    train=train[train.pair_id.notna()]
    val=val[val.pair_id.notna()]
    X,y=prepare(train);V,vy=prepare(val)
    estimator=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
    estimator.fit(X,y)
    threshold=threshold_for_false_alarms(estimator.predict_proba(V)[:,1],vy,.01)
    encoded=pickle.dumps(estimator)
    (directory/'timing_summary.pkl').write_bytes(encoded)
    reloaded=pickle.loads(encoded)
    report=json.loads((directory/'report.json').read_text())
    extras=json.loads((directory/'extra_stress/report.json').read_text())
    results={}
    for split in list(report['results'])+list(extras['results']):
        if split in ('normal','stealth'):continue
        parent=directory/'extra_stress' if split in extras['results'] else directory
        T,ty=prepare(pd.read_parquet(parent/f'{split}.parquet'))
        scores=estimator.predict_proba(T)[:,1]
        np.testing.assert_array_equal(scores,reloaded.predict_proba(T)[:,1])
        guesses=scores>=.5;flags=scores>=threshold
        results[split]={'sessions':len(ty),'accuracy':float((guesses==ty).mean()),
            'false_alert_rate':float(guesses[ty==0].mean()),
            'attack_recall':float(guesses[ty==1].mean()) if (ty==1).any() else None,
            'validation_threshold_false_alert_rate':float(flags[ty==0].mean()),
            'validation_threshold_attack_recall':float(flags[ty==1].mean()) if (ty==1).any() else None}
    result={'threshold':threshold,'results':results,'scope':'Binary temporal subset only; not a six-class competitor.',
            'features':'13 gap summaries: size, mean/std/extrema/quantiles, lag1–4 correlations, relative adjacent change.',
            'audit':'Persisted classifier reproduces every score exactly. Train fit; validation-only threshold.'}
    (directory/'timing_summary.json').write_text(json.dumps(result,indent=2))
    return result


def study(output,seed):
    output.mkdir(parents=True,exist_ok=False)
    variants={}
    for mode in ('hybrid','hybrid_log'):
        directory=output/mode
        result=run(directory,seed=seed,max_epochs=200,class_weight_power=.5,
                   pace_augmentation=True,timing_mode=mode,noise_augmentation=True)
        print(json.dumps({'seed':seed,'mode':mode,'audit':audit(directory)}),flush=True)
        supplemental=extra(directory,stronger=True)
        alerts=operating(directory)
        baseline=timing_summary(directory)
        variants[mode]={'metadata':result['metadata'],'results':result['results'],
                        'extra':supplemental,'alerts':alerts,'timing_summary':baseline}
        print(json.dumps({'seed':seed,'mode':mode,'tests':{name:values['sona_hybrid']['accuracy']
              for name,values in {**result['results'],**supplemental['results']}.items()}}),flush=True)
    for name in json.loads((output/'hybrid/report.json').read_text())['counts']:
        pd.testing.assert_frame_equal(pd.read_parquet(output/'hybrid'/f'{name}.parquet'),
                                      pd.read_parquet(output/'hybrid_log'/f'{name}.parquet'))
    for name in variants['hybrid']['extra']['counts']:
        pd.testing.assert_frame_equal(pd.read_parquet(output/'hybrid/extra_stress'/f'{name}.parquet'),
                                      pd.read_parquet(output/'hybrid_log/extra_stress'/f'{name}.parquet'))
    record={'seed':seed,'variants':variants,'matched_raw_datasets':'passed',
            'selection':'Checkpoints selected by validation loss; no test-based seed selection.'}
    (output/'study.json').write_text(json.dumps(record,indent=2))
    return record


def relative_study(output,seed):
    """Prospective fallback: discard absolute pace, keep all other flow features."""
    directory=output/'relative'
    result=run(directory,seed=seed,max_epochs=200,class_weight_power=.5,
               pace_augmentation=True,timing_mode='relative',noise_augmentation=True)
    print(json.dumps({'seed':seed,'mode':'relative','audit':audit(directory)}),flush=True)
    supplemental=extra(directory,stronger=True)
    alerts=operating(directory)
    baseline=timing_summary(directory)
    record={'seed':seed,'metadata':result['metadata'],'results':result['results'],
            'extra':supplemental,'alerts':alerts,'timing_summary':baseline}
    (output/'relative_study.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({'seed':seed,'mode':'relative','tests':{name:values['sona_hybrid']['accuracy']
          for name,values in {**result['results'],**supplemental['results']}.items()}}),flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--only-relative',action='store_true')
    args=parser.parse_args()
    (relative_study if args.only_relative else study)(args.out_dir,args.seed)
