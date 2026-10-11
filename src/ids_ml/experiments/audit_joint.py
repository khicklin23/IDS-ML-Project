"""Audit schema-valid six-class candidate splits and saved neural inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ids_ml.data.schema import FEATURE_COLUMNS, validate
from ids_ml.features import load_preprocessor, relative_timing_frame, transform_features
from ids_ml.models.sequence import SequenceClassifier, predict_sequence_proba
from ids_ml.models.sessions import build_session_sequences
from ids_ml.models.train import load_artifact
from ids_ml.models.evaluate import predict_class_proba


def audit(directory):
    torch.set_num_threads(2)
    report=json.loads((directory/'report.json').read_text())
    meta=json.loads((directory/'sona_metadata.json').read_text())
    processor=load_preprocessor(directory/'sona_preprocessor.pkl')
    model=SequenceClassifier(meta['input_dim'],len(meta['class_names']),meta['hidden_dim'])
    model.load_state_dict(torch.load(directory/'sona.pt',weights_only=True))
    seen_flows,seen_sessions,seen_pairs=set(),set(),set()
    totals={'flows':0,'sessions':0,'paired_groups':0,'reload_checks':0}
    for split,counts in report['counts'].items():
        df=pd.read_parquet(directory/f'{split}.parquet')
        validate(df)
        assert len(df)==counts['flows'] and df.session_id.nunique()==counts['sessions']
        # Independent runs can reuse names; canonical run/session identities prevent collisions.
        flows=set(zip(df.run_id.astype(str),df.flow_id.astype(str)))
        sessions=set(zip(df.run_id.astype(str),df.session_id.astype(str).str.split('::').str[-1]))
        assert not flows & seen_flows, (split,'flow overlap')
        assert not sessions & seen_sessions, (split,'session overlap')
        seen_flows.update(flows);seen_sessions.update(sessions)
        if 'pair_id' in df:
            paired=df[df.pair_id.notna()]
            ids=set(paired.pair_id)
            assert not ids & seen_pairs
            seen_pairs.update(ids)
            raw=pd.util.hash_pandas_object(df[list(FEATURE_COLUMNS)],index=False).to_numpy()
            normalized=pd.util.hash_pandas_object(relative_timing_frame(df)[list(FEATURE_COLUMNS)],index=False).to_numpy()
            for indices in df.groupby('pair_id',sort=False).indices.values():
                target=df.iloc[indices].target.to_numpy()
                if np.all(target==0):continue
                a,b=indices[target==0],indices[target==1]
                assert len(a)==len(b)>0
                for group in (a,b):
                    session=df.iloc[group].sort_values('event_time',kind='stable')
                    np.testing.assert_allclose(np.diff(session.event_time.to_numpy()),
                                               session.prior_gap_s.to_numpy()[1:],atol=1e-8,rtol=1e-8)
                    assert session.prior_gap_s.iloc[0]==0
                for hashes in (raw,normalized):
                    np.testing.assert_array_equal(np.sort(hashes[a]),np.sort(hashes[b]))
                    np.testing.assert_array_equal(hashes[a[[0,-1]]],hashes[b[[0,-1]]])
                totals['paired_groups']+=1
        if split not in ('train','val'):
            X,lengths,y,ids=build_session_sequences(df,processor,meta['class_names'],max_len=meta['max_seq_len'])
            with np.load(directory/f'{split}_predictions.npz') as scores:
                np.testing.assert_array_equal(y,scores['y'])
                np.testing.assert_array_equal(np.array(ids),scores['session_ids'])
                # Same evaluation batch size as the experiment: saved weights + persisted
                # preprocessing must reproduce actual inference, not just tensor shapes.
                probs=predict_sequence_proba(model,X[:128],lengths[:128])
                np.testing.assert_allclose(probs,scores['sona_probs'][:128],atol=1e-6,rtol=1e-5)
                np.testing.assert_array_equal(probs.argmax(axis=1),scores['sona'][:128])
                if split not in ('normal','stealth','polling'):
                    predictions=dict(zip(ids,scores['beta2']))
                    for _,pair in df.groupby('pair_id',sort=False):
                        sessions_in_pair=list(pair.session_id.unique())
                        assert predictions[sessions_in_pair[0]]==predictions[sessions_in_pair[1]]
            totals['reload_checks']+=1
        totals['flows']+=len(df);totals['sessions']+=counts['sessions']
    beta,beta_processor,beta_meta=load_artifact(directory/'beta2')
    sample=pd.read_parquet(directory/'normal.parquet')
    prediction=predict_class_proba(beta,transform_features(beta_processor,sample))
    assert prediction.shape==(len(sample),len(meta['class_names']))
    assert np.isfinite(prediction).all()
    totals['status']='passed'
    (directory/'audit.json').write_text(json.dumps(totals,indent=2))
    return totals

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(json.dumps(audit(parser.parse_args().directory),indent=2))
