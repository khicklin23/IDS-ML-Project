"""Independent replay audit for all context candidates, conditions and budgets."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ids_ml.data.schema import validate
from ids_ml.models.classifier import MLPClassifier
from ids_ml.models.sequence import SequenceClassifier
from .mixed_scan import PooledMLP, digest
from .mixed_scan_generalization import score_models, evaluate_scores, aggregate, budget_key
from .scan_context import pack_context
from .scan_generalization_features import prepare, build_processor, CONDITIONS
from .temporal_patterns import threshold_for_false_alarms


def audit_seed(path):
    r=json.loads((path/'results.json').read_text());c=r['config'];seed=r['seed']
    torch.set_num_threads(c['threads'])
    frames={k:pd.read_parquet(path/f'{k}.parquet') for k in r['source_sha256']}
    keys={}
    for split,frame in frames.items():
        validate(frame);assert digest(path/f'{split}.parquet')==r['source_sha256'][split]
        assert len(frame)==r['flow_counts'][split]
        assert (frame.event_time>=frame.observation_start_s).all()
        assert (frame.event_time<frame.observation_start_s+frame.observation_duration_s).all()
        keys[split]={k:set(frame[k]) for k in ('flow_id','session_id','src_host_id')}
        for other in keys:
            if other!=split:
                for k in keys[split]:assert not keys[split][k]&keys[other][k]
        target=frame.groupby('session_id',sort=False).window_target.first()
        truth=frame.scenario.eq('port_scan').groupby(frame.session_id,sort=False).any()
        np.testing.assert_array_equal(target.astype(bool),truth)
    prepared={k:prepare(f) for k,f in frames.items()}
    processors={mode:pickle.loads((path/f'{mode}_preprocessor.pkl').read_bytes()) for mode in ('raw','relative','small')}
    for mode,p in processors.items():
        fresh=build_processor(frames['train'],mode,prepared['train'])
        np.testing.assert_array_equal(p.transform(frames['val'],prepared['val']),fresh.transform(frames['val'],prepared['val']))
    dims=r['dimensions']
    models={'raw_sona':SequenceClassifier(dims['raw'],2),
            'relative_sona':SequenceClassifier(dims['relative'],2),
            'small_sona':SequenceClassifier(dims['small'],2),
            'small_pool':PooledMLP(dims['small'])}
    for name,model in models.items():
        model.load_state_dict(torch.load(path/f'{name}.pt',weights_only=True))
        assert sum(p.numel() for p in model.parameters())==r['parameter_counts'][name]
    boosters={k:pickle.loads((path/f'{k}.pkl').read_bytes()) for k in ('summary_boosting',)}
    score_checks,operating_checks=0,0
    for split in ('val',*CONDITIONS):
        frame=frames[split]
        packs={mode:pack_context(frame,p,c['max_length'],prepared[split]) for mode,p in processors.items()}
        values=score_models(frame,prepared[split],packs,processors,models,boosters,seed,c['threads'])
        saved=np.load(path/f'{split}_predictions.npz',allow_pickle=False)
        y=packs['small'][2];np.testing.assert_array_equal(saved['y'],y)
        np.testing.assert_array_equal(saved['window_ids'],frame.groupby('session_id',sort=False).session_id.first().to_numpy(dtype=str))
        kinds=frame.groupby('session_id',sort=False).behavior.first().to_numpy()
        for model,value in values.items():
            np.testing.assert_allclose(value,saved[model],rtol=0,atol=1e-7);score_checks+=1
            if split=='val':
                for b in c['budgets']:
                    assert r['thresholds'][model][budget_key(b)]==threshold_for_false_alarms(value,y,b)
        computed=evaluate_scores(values,y,kinds,r['thresholds'])
        assert computed==r['results'][split]
        for model,budgets in computed.items():
            for b,m in budgets.items():
                assert m['threshold_from_validation']==r['thresholds'][model][b]
                if split=='val':assert m['at_validation_false_alarm_target']['false_positive_rate']<=float(b)
                operating_checks+=1
    result={'status':'passed','seed':seed,'prediction_checks':score_checks,'operating_point_checks':operating_checks,
            'source_hashes_and_identity_verified':True,'train_only_preprocessing_verified':True,
            'thresholds_validation_only':True,'all_metrics_reproduced':True}
    (path/'independent_audit.json').write_text(json.dumps(result,indent=2))
    return result


def audit(directory):
    r=json.loads((directory/'report.json').read_text())
    assert r['config']==json.loads((directory/'protocol.json').read_text())
    results=[]
    for record in r['runs']:
        path=directory/f'seed-{record["seed"]}'
        assert record==json.loads((path/'results.json').read_text())
        result=audit_seed(path);results.append(result);print(json.dumps(result),flush=True)
    assert aggregate(r['runs'])==r['summary']
    result={'status':'passed','runs':results}
    (directory/'audit.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    a=p.parse_args();audit(a.directory)


if __name__=='__main__':main()
