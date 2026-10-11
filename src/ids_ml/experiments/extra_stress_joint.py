"""Fresh post-training noise and unseen benign-pace tests; no model refitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ids_ml.data.schema import FEATURE_COLUMNS
from ids_ml.features import load_preprocessor, transform_features, relative_timing_frame
from ids_ml.models.train import load_artifact
from ids_ml.models.evaluate import predict_class_proba, multiclass_metrics_from_predictions
from ids_ml.models.sequence import SequenceClassifier
from ids_ml.models.sessions import build_session_sequences
from .joint_temporal import schema_frame, sequence_probs
from .temporal_patterns import generate_pairs, shuffled
from .temporal_robustness import polling


def measure(directory, *, stronger=False):
    torch.set_num_threads(2)
    output=directory/'extra_stress'
    output.mkdir(exist_ok=False)
    report=json.loads((directory/'report.json').read_text())
    seed=report['seed']
    meta=report['metadata']
    classes=meta['class_names']
    processor=load_preprocessor(directory/'sona_preprocessor.pkl')
    model=SequenceClassifier(meta['input_dim'],len(classes),meta['hidden_dim'])
    model.load_state_dict(torch.load(directory/'sona.pt',weights_only=True))
    beta,beta_processor,_=load_artifact(directory/'beta2')
    noisy,_=generate_pairs(seed+150000,600,f'{seed}-extra-noise',jitter=.25)
    long_polling=polling(seed+160000,300,f'{seed}-extra-polling',(650,1000))
    results={};counts={}
    tests=[('extra_noise',noisy),('unseen_polling',long_polling)]
    if stronger:
        extreme,_=generate_pairs(seed+180000,600,f'{seed}-extreme-noise',jitter=.45)
        tests.extend([('extreme_noise',extreme),
                      ('extreme_polling',polling(seed+190000,300,f'{seed}-extreme-polling',(1500,3000)))])
    for name,frame in tests:
        df=schema_frame(frame,seed,name)
        assert df.groupby('session_id').size().max() <= meta['max_seq_len']
        # New split identities, including extra noisy pairs, are independent of every original split.
        for split in report['counts']:
            prior=pd.read_parquet(directory/f'{split}.parquet',columns=['session_id'])
            assert not set(df.session_id) & set(prior.session_id)
        for view in (df,relative_timing_frame(df)):
            hashes=pd.util.hash_pandas_object(view[list(FEATURE_COLUMNS)],index=False).to_numpy()
            for indices in view.groupby('pair_id').indices.values():
                labels=view.iloc[indices].target.to_numpy()
                if np.all(labels==0):continue
                a,b=indices[labels==0],indices[labels==1]
                np.testing.assert_array_equal(np.sort(hashes[a]),np.sort(hashes[b]))
                np.testing.assert_array_equal(hashes[a[[0,-1]]],hashes[b[[0,-1]]])
        df.to_parquet(output/f'{name}.parquet',index=False)
        X,lengths,y,ids=build_session_sequences(df,processor,classes,max_len=meta['max_seq_len'])
        flow_preds=predict_class_proba(beta,transform_features(beta_processor,df)).argmax(axis=1)
        groups=df.groupby('session_id',sort=False).indices
        guesses=np.array([np.bincount(flow_preds[groups[sid]],minlength=len(classes)).argmax() for sid in ids])
        if name in ('extra_noise','extreme_noise'):
            mapping=dict(zip(ids,guesses))
            for _,pair in df.groupby('pair_id'):
                a,b=pair.session_id.unique()
                assert mapping[a]==mapping[b]
        probs=sequence_probs(model,X,lengths)
        results[name]={'beta2':multiclass_metrics_from_predictions(guesses,y,classes),
                       'sona_hybrid':multiclass_metrics_from_predictions(probs.argmax(axis=1),y,classes)}
        if name in ('extra_noise','extreme_noise'):
            results[name]['sona_shuffled']=multiclass_metrics_from_predictions(
                sequence_probs(model,shuffled(X,lengths,seed+170000),lengths).argmax(axis=1),y,classes)
        np.savez_compressed(output/f'{name}_predictions.npz',y=y,session_ids=np.array(ids),
                            beta2=guesses,sona=probs.argmax(axis=1),sona_probs=probs)
        counts[name]={'flows':len(df),'sessions':len(ids)}
    result={'seed':seed,'results':results,'counts':counts,'audit_status':'passed',
            'recipe':{'extra_noise':{'jitter':.25,'pairs':600,'seed':seed+150000},
                      'unseen_polling':{'pace':[650,1000],'sessions':300,'seed':seed+160000}},
            'note':'Frozen parent artifacts; no refitting or original score changes. Fresh noise and polling pace beyond training.'}
    if stronger:
        result['recipe'].update({'extreme_noise':{'jitter':.45,'pairs':600,'seed':seed+180000},
                                 'extreme_polling':{'pace':[1500,3000],'sessions':300,'seed':seed+190000}})
    (output/'report.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(json.dumps(measure(parser.parse_args().directory),indent=2))
