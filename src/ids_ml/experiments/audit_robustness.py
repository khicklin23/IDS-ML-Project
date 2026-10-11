"""Verify paired controls, closed-session normalization, and split isolation on disk."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ids_ml.data.schema import FEATURE_COLUMNS
from .temporal_robustness import relative_view


def audit(directory):
    report = json.loads((directory/'report.json').read_text())
    totals = {'flows':0,'sessions':0,'matched_pairs':0,'unpaired_polling_sessions':0,'seeds':[]}
    for record in report['runs']:
        seen = {c:set() for c in ('flow_id','session_id','pair_id')}
        for split, counts in record['counts'].items():
            root = directory/f'seed-{record["seed"]}'
            df = pd.read_parquet(root/f'{split}.parquet')
            assert len(df) == counts['flows'] and df.session_id.nunique() == counts['sessions']
            assert df.flow_id.is_unique
            for column in seen:
                ids=set(df[column])
                assert not ids & seen[column], (record['seed'],split,column,'overlap')
                seen[column].update(ids)
            raw = pd.util.hash_pandas_object(df[list(FEATURE_COLUMNS)],index=False).to_numpy()
            normalized = pd.util.hash_pandas_object(relative_view(df)[list(FEATURE_COLUMNS)],index=False).to_numpy()
            targets=df.target.to_numpy()
            for indices in df.groupby('pair_id',sort=False).indices.values():
                labels=targets[indices]
                if np.all(labels==0):
                    assert set(df.iloc[indices].scenario)=={'benign_scheduled_polling'}
                    totals['unpaired_polling_sessions']+=1
                    continue
                a,b=indices[labels==0],indices[labels==1]
                assert len(a)==len(b)>0
                for hashes in (raw,normalized):
                    np.testing.assert_array_equal(np.sort(hashes[a]),np.sort(hashes[b]))
                    np.testing.assert_array_equal(hashes[a[[0,-1]]],hashes[b[[0,-1]]])
                totals['matched_pairs']+=1
            for indices in df.groupby('session_id',sort=False).indices.values():
                s=df.iloc[indices]
                np.testing.assert_allclose(np.diff(s.event_time),s.prior_gap_s.to_numpy()[1:])
            if split not in ('train','val','polling','polling_slow'):
                with np.load(root/f'{split}_predictions.npz') as scores:
                    np.testing.assert_array_equal(scores['y'],np.tile([0,1],counts['sessions']//2))
                    np.testing.assert_array_equal(scores['beta2_relative'][::2],scores['beta2_relative'][1::2])
            totals['flows']+=len(df)
            totals['sessions']+=counts['sessions']
        totals['seeds'].append(record['seed'])
    totals['status']='passed'
    (directory/'audit.json').write_text(json.dumps(totals,indent=2))
    return totals

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(json.dumps(audit(parser.parse_args().directory),indent=2))
