"""Reload mixed-scan artifacts and independently reproduce frozen evaluations."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from ids_ml.data.schema import validate
from ids_ml.features import transform_features, build_preprocessor
from ids_ml.models.classifier import MLPClassifier
from ids_ml.models.sequence import SequenceClassifier
from ids_ml.models.evaluate import predict_class_proba
from .mixed_scan import PooledMLP, pack_windows, summaries, digest
from .temporal_patterns import shuffled, sequence_scores, threshold_for_false_alarms, score_metrics


def audit_seed(path):
    record=json.loads((path/'results.json').read_text())
    config=record['config'];seed=record['seed']
    torch.set_num_threads(config['threads'])
    frames={name:pd.read_parquet(path/f'{name}.parquet') for name in record['source_sha256']}
    for i,(name,frame) in enumerate(frames.items()):
        validate(frame)
        assert digest(path/f'{name}.parquet')==record['source_sha256'][name]
        assert len(frame)==record['flow_counts'][name]
        for other in list(frames)[:i]:
            for key in ('flow_id','session_id','src_host_id'):
                assert not set(frame[key])&set(frames[other][key])
        for _,w in frame.groupby('session_id',sort=False):
            assert w.event_time.is_monotonic_increasing
            np.testing.assert_allclose(np.diff(w.event_time),w.prior_gap_s.iloc[1:])
            assert bool(w.window_target.iloc[0])==bool((w.scenario=='port_scan').any())
    processor=pickle.loads((path/'preprocessor.pkl').read_bytes())
    fresh_processor=build_preprocessor(frames['train'])
    np.testing.assert_array_equal(transform_features(processor,frames['val']),
                                  transform_features(fresh_processor,frames['val']))
    d=record['encoded_features']
    models={'beta2_vote':MLPClassifier(d,output_dim=2),'pooled_mlp':PooledMLP(d),
            'sona':SequenceClassifier(d,2),'shuffled_train':SequenceClassifier(d,2)}
    for name,model in models.items():model.load_state_dict(torch.load(path/f'{name}.pt',weights_only=True))
    summary=pickle.loads((path/'summary_model.pkl').read_bytes())
    checks=0
    for name in ('val','test','shifted','lookalikes'):
        frame=frames[name];xx,ll,yy,gg=pack_windows(frame,processor,config['max_length'])
        saved=np.load(path/f'{name}_predictions.npz',allow_pickle=False)
        np.testing.assert_array_equal(saved['y'],yy)
        np.testing.assert_array_equal(saved['window_ids'],frame.groupby('session_id',sort=False).session_id.first().to_numpy(dtype=str))
        flow=predict_class_proba(models['beta2_vote'],transform_features(processor,frame))[:,1]
        s,rule=summaries(frame,gg)
        with threadpool_limits(limits=config['threads']):summary_scores=summary.predict_proba(s)[:,1]
        values={'beta2_vote':np.array([np.mean(flow[g]>=.5) for g in gg]),
                'beta2_mean':np.array([flow[g].mean(dtype=np.float64) for g in gg]),
                'pooled_mlp':sequence_scores(models['pooled_mlp'],xx,ll),
                'sona':sequence_scores(models['sona'],xx,ll),
                'sona_shuffled_test':sequence_scores(models['sona'],shuffled(xx,ll,seed+70000),ll),
                'shuffled_train':sequence_scores(models['shuffled_train'],shuffled(xx,ll,seed+80000),ll),
                'summary_boosting':summary_scores,'temporal_rule':rule}
        assert set(values)==set(record['results'][name])
        for model,score in values.items():
            np.testing.assert_allclose(score,saved[model],rtol=0,atol=1e-7)
            threshold=record['thresholds'][model]
            if name=='val':
                assert threshold==threshold_for_false_alarms(score,yy,config['false_alarm_target'])
            computed=score_metrics(score,yy,threshold)
            original=record['results'][name][model]
            assert original['threshold_from_validation']==threshold
            assert computed['at_validation_false_alarm_target']==original['at_validation_false_alarm_target']
            if name=='val':assert computed['at_validation_false_alarm_target']['false_positive_rate']<=config['false_alarm_target']
            for key in ('roc_auc','average_precision'):
                assert abs(computed[key]-original[key])<1e-12
            kinds=frame.groupby('session_id',sort=False).behavior.first().to_numpy()
            for kind,metrics in original['by_behavior'].items():
                assert metrics['windows']==np.sum(kinds==kind)
                assert metrics['flag_rate']==np.mean(score[kinds==kind]>=threshold)
            checks+=1
    result={'status':'passed','seed':seed,'reloaded_model_split_checks':checks,
            'source_hashes_verified':True,'identities_verified':True,
            'thresholds_recomputed_from_validation':True,'metrics_reproduced':True}
    (path/'independent_audit.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    report=json.loads((args.directory/'report.json').read_text())
    results=[]
    for record in report['runs']:
        result=audit_seed(args.directory/f'seed-{record["seed"]}')
        print(json.dumps(result),flush=True);results.append(result)
    for split,models in report['summary'].items():
        for model,metrics in models.items():
            for key,values in metrics.items():
                actual=[r['results'][split][model]['at_validation_false_alarm_target'][key] for r in report['runs']]
                assert values=={'mean':float(np.mean(actual)),'min':float(min(actual)),'max':float(max(actual))}
    (args.directory/'audit.json').write_text(json.dumps({'status':'passed','runs':results},indent=2))


if __name__=='__main__':main()
