import numpy as np
import pandas as pd
import pytest
from ids_ml.experiments.scan_generalization_features import generate_partition, generate_condition, prepare, build_processor, CONDITIONS
from ids_ml.experiments.scan_context import PACE_COLUMNS


def test_independent_conditions_and_balanced_training_cells():
    low = generate_condition(1103,16,'same','test')
    longer = generate_condition(1103,16,'same','duration')
    pd.testing.assert_series_equal(low.dst_port,longer.dst_port)
    np.testing.assert_allclose(longer.event_time,2*low.event_time)
    high = generate_condition(1103,16,'high','density')
    assert high.observation_duration_s.eq(1800).all()
    assert len(high)>len(low)
    mixed=generate_partition(1103,32,'train','mixture')
    for _,cell in mixed.groupby('condition'):
        targets=cell.groupby('session_id').window_target.first()
        assert len(targets)==8 and targets.sum()==4
    assert (mixed.event_time>=mixed.observation_start_s).all()
    assert (mixed.event_time<mixed.observation_start_s+mixed.observation_duration_s).all()
    with pytest.raises(ValueError):generate_partition(1103,30,'bad','mixture')


def test_relative_timing_is_scale_invariant_causal_and_label_blind():
    f=generate_condition(1103,8,'a','test');a=prepare(f)
    scaled=f.copy();scaled.event_time*=3;scaled.prior_gap_s*=3
    np.testing.assert_allclose(prepare(scaled)[list(PACE_COLUMNS)],a[list(PACE_COLUMNS)],atol=1e-10)
    w=f.groupby('session_id',sort=False).get_group(f.session_id.iloc[0]).iloc[:15]
    pd.testing.assert_frame_equal(prepare(w.reset_index(drop=True)),a.loc[w.index].reset_index(drop=True))
    changed=f.copy();changed.window_target=1-changed.window_target
    for c in ('label','scenario','behavior'):changed[c]='hidden'
    pd.testing.assert_frame_equal(prepare(changed),a)
    p=build_processor(f,'relative',a)
    assert not any('prior_gap_s' in n for n in p.get_feature_names_out())
    assert not any('window_target' in n for n in p.get_feature_names_out())
    means=p.encoder.named_transformers_['numeric'].mean_.copy()
    p.transform(scaled,prepare(scaled))
    np.testing.assert_array_equal(means,p.encoder.named_transformers_['numeric'].mean_)


def test_full_artifacts_replay_and_each_condition_budget(tmp_path):
    from ids_ml.experiments.mixed_scan_generalization import Config,run
    from ids_ml.experiments.audit_scan_generalization import audit
    result=run(Config(seeds=(1103,),train_windows=32,val_windows=32,test_windows=8,epochs=2,min_epochs=1),tmp_path/'study')
    checked=audit(tmp_path/'study')
    assert checked['runs'][0]['prediction_checks']==36
    assert checked['runs'][0]['operating_point_checks']==108
    assert set(result['summary'])==set(CONDITIONS)
