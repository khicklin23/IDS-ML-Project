"""Causal feature construction and frozen multi-budget evaluation checks."""
import numpy as np
import pandas as pd
import pytest

from ids_ml.experiments.mixed_scan import generate_windows
from ids_ml.experiments.scan_context import causal_features,build_context_preprocessor,pack_context,PACE_COLUMNS,HISTORY_COLUMNS
from ids_ml.experiments.mixed_scan_context import Config,run
from ids_ml.experiments.audit_scan_context import audit


def test_context_uses_only_prefix_and_is_independent_of_generator_truth():
    frame=generate_windows(941,12,'train')
    full=causal_features(frame)
    for _,w in frame.groupby('session_id',sort=False):
        prefix=w.iloc[:20]
        expected=full.loc[prefix.index].reset_index(drop=True)
        pd.testing.assert_frame_equal(causal_features(prefix.reset_index(drop=True)),expected)
    changed=frame.copy()
    for column in ('label','scenario','behavior','src_ip','dst_ip','run_id'):changed[column]='irrelevant'
    changed.window_target=1-changed.window_target
    changed.observation_duration_s=1
    pd.testing.assert_frame_equal(causal_features(changed),full)
    shifted=frame.copy();shifted.event_time += 1e6
    np.testing.assert_allclose(causal_features(shifted)[list(PACE_COLUMNS+HISTORY_COLUMNS)],full[list(PACE_COLUMNS+HISTORY_COLUMNS)],rtol=1e-7,atol=1e-7)


def test_counts_exclude_current_event_and_recent_history_expires():
    frame=generate_windows(941,4,'train').groupby('session_id',sort=False).get_group('train-window-000000').iloc[:4].copy().reset_index(drop=True)
    frame.event_time=[0.,10.,20.,400.];frame.prior_gap_s=[0.,10.,10.,380.]
    frame.dst_port=[22,22,443,22];frame.outcome=['rejected','established','timeout','rejected']
    a=causal_features(frame)
    assert list(a.past_flows)==[0,1,2,3]
    assert list(a.past_failed)==[0,1,1,2]
    assert list(a.past_unique_ports)==[0,1,1,2]
    assert list(a.current_port_is_new)==[1,0,1,0]
    assert a.past300_failed.iloc[-1]==0
    assert a.online_relative_gap.iloc[-1]==pytest.approx(np.log1p(38))
    assert np.isfinite(a.select_dtypes('number')).all().all()


def test_processors_fit_on_training_and_do_not_truncate_or_reorder_targets():
    train=generate_windows(941,12,'train');val=generate_windows(10941,8,'val')
    dimensions=[]
    for mode in ('raw','pace','context'):
        p=build_context_preprocessor(train,mode)
        X,ll,y,g=pack_context(val,p,160)
        dimensions.append(X.shape[-1])
        assert len(y)==8 and y.sum()==4
        for i,n in enumerate(ll):assert np.all(X[i,n:]==0)
        expected=p.encoder.named_transformers_['numeric'].mean_.copy()
        p.transform(val)
        np.testing.assert_array_equal(expected,p.encoder.named_transformers_['numeric'].mean_)
    assert dimensions[0]<dimensions[1]<dimensions[2]
    with pytest.raises(ValueError,match='cap'):pack_context(val,p,2)
    broken=val.copy();broken.iloc[0,broken.columns.get_loc('event_time')]=1e9
    with pytest.raises(ValueError,match='chronological'):causal_features(broken)


def test_tiny_full_study_reloads_all_models_and_all_budgets(tmp_path):
    r=run(Config(seeds=(941,),train_windows=20,val_windows=12,test_windows=12,epochs=2,min_epochs=1),tmp_path/'study')
    checked=audit(tmp_path/'study')
    assert checked['runs'][0]['prediction_checks']==40
    assert checked['runs'][0]['operating_point_checks']==120
    from ids_ml.experiments.summarize_scan_context import summarize
    summarize(tmp_path/'study')
    import csv
    rows=list(csv.DictReader((tmp_path/'study/per_seed.csv').open()))
    assert len(rows)==120
    assert {row['validation_budget'] for row in rows}=={'0.01','0.02','0.05'}
    assert 'Context neural pooling' in (tmp_path/'study/RESULTS.md').read_text()
    for models in r['runs'][0]['results']['val'].values():
        for b,m in models.items():assert m['at_validation_false_alarm_target']['false_positive_rate']<=float(b)
    with pytest.raises(FileExistsError):run(Config(),tmp_path/'study')
    with pytest.raises(ValueError,match='fresh seeds'):run(Config(seeds=(811,)),tmp_path/'old-seed')


def test_observer_cutoff_is_uniform_and_no_event_crosses_window_end():
    from ids_ml.experiments.scan_context import generate_context_windows
    f=generate_context_windows(941,80,'bounded',mode='lookalikes')
    assert (f.event_time>=f.observation_start_s).all()
    assert (f.event_time<f.observation_start_s+f.observation_duration_s).all()
    assert f.session_id.nunique()==80
    np.testing.assert_allclose(f.groupby('session_id').event_time.diff().fillna(0.),f.prior_gap_s)
    a=causal_features(f)
    changed=f.copy();changed.observation_start_s += 100000
    pd.testing.assert_frame_equal(a,causal_features(changed))
