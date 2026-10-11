"""Persisted logarithmic timing must agree between training and inference."""
import numpy as np
import pandas as pd

from ids_ml.experiments.temporal_patterns import generate_pairs
from ids_ml.features import build_preprocessor, transform_features, save_preprocessor, load_preprocessor


def test_log_hybrid_compresses_absolute_gap_and_preserves_relative_context(tmp_path):
    frame,_=generate_pairs(91,8,'log-fit')
    original=frame.copy(deep=True)
    processor=build_preprocessor(frame,timing_mode='hybrid_log')
    names=processor.get_feature_names_out().tolist()
    absolute=names.index('numeric__prior_gap_s')
    relative=names.index('numeric__session_relative_gap')
    scaler=processor.encoder.named_transformers_['numeric']
    numeric=list(processor.encoder.transformers_[0][2])
    idx=numeric.index('prior_gap_s')
    assert scaler.mean_[idx]==np.log1p(frame.prior_gap_s).mean()
    expected=(np.log1p(frame.prior_gap_s)-scaler.mean_[idx])/scaler.scale_[idx]
    A=transform_features(processor,frame)
    np.testing.assert_allclose(A[:,absolute],expected,atol=1e-6)
    slow=frame.copy()
    slow['prior_gap_s']*=1000
    B=transform_features(processor,slow)
    np.testing.assert_allclose(A[:,relative],B[:,relative],atol=1e-6)
    np.testing.assert_allclose(np.delete(A,absolute,axis=1),np.delete(B,absolute,axis=1),atol=1e-6)
    assert np.isfinite(B).all()
    assert not np.allclose(A[:,absolute],B[:,absolute])
    pd.testing.assert_frame_equal(frame,original)
    save_preprocessor(processor,tmp_path/'log.pkl')
    loaded=load_preprocessor(tmp_path/'log.pkl')
    assert loaded.timing_mode=='hybrid_log'
    np.testing.assert_array_equal(transform_features(loaded,slow),B)


def test_log_transform_preserves_zero_and_pair_equality():
    frame,_=generate_pairs(93,8,'log-pairs')
    processor=build_preprocessor(frame,timing_mode='hybrid_log')
    features=transform_features(processor,frame)
    for indices in frame.groupby('pair_id',sort=False).indices.values():
        y=frame.iloc[indices].target.to_numpy()
        for side in (0,1):
            assert frame.iloc[indices[y==side]].prior_gap_s.iloc[0]==0
        a,b=features[indices[y==0]],features[indices[y==1]]
        np.testing.assert_array_equal(a[[0,-1]],b[[0,-1]])
        # Sorted hashes avoid accepting a timing-induced per-flow difference.
        ah=pd.util.hash_pandas_object(pd.DataFrame(a),index=False)
        bh=pd.util.hash_pandas_object(pd.DataFrame(b),index=False)
        np.testing.assert_array_equal(np.sort(ah),np.sort(bh))
