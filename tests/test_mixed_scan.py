"""Mixed observations, provenance exclusion and operating-point regression checks."""
import numpy as np
import pandas as pd
import pytest
import torch

from ids_ml.data.schema import FEATURE_COLUMNS, validate
from ids_ml.features import build_preprocessor
from ids_ml.experiments.mixed_scan import generate_windows, pack_windows, summaries, PooledMLP, Config, run


def test_independent_mixed_windows_and_no_identity_overlap():
    frame=generate_windows(811,80,'train')
    pd.testing.assert_frame_equal(frame,generate_windows(811,80,'train'))
    other=generate_windows(10811,80,'val')
    for key in ('flow_id','session_id','src_host_id'):
        assert not set(frame[key]) & set(other[key])
    validate(frame)
    for _,w in frame.groupby('session_id'):
        assert w.event_time.is_monotonic_increasing
        np.testing.assert_allclose(np.diff(w.event_time),w.prior_gap_s.iloc[1:])
        assert w.window_target.nunique()==1
        if w.window_target.iloc[0]:
            assert set(w.scenario)=={'benign','port_scan'}
            assert (w.scenario=='benign').mean()>.5
        else:assert set(w.scenario)=={'benign'}
    # These are independently generated traces, not identical paired multisets.
    assert 'pair_id' not in frame
    assert set(frame.behavior)>={'gradual_scan','monitoring','discovery','retries'}


def test_provenance_never_changes_inputs_or_summary_baselines():
    frame=generate_windows(811,20,'train')
    processor=build_preprocessor(frame)
    X,lengths,y,groups=pack_windows(frame,processor,160)
    changed=frame.copy()
    changed['scenario']='benign';changed['label']='normal'
    changed['behavior']='changed';changed['window_target']=1-changed.window_target
    changed['src_ip']='203.0.113.1';changed['dst_ip']='203.0.113.2'
    changed['observation_duration_s']=99999
    altered=pack_windows(changed,processor,160)
    np.testing.assert_array_equal(X,altered[0])
    a,rule=summaries(frame,groups);b,other=summaries(changed,groups)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(rule,other)
    assert not set(FEATURE_COLUMNS)&{'window_target','behavior','scenario','label'}
    for i,n in enumerate(lengths):assert np.all(X[i,n:]==0)
    with pytest.raises(ValueError,match='cap'):pack_windows(frame,processor,10)


def test_pooling_is_order_invariant_and_ignores_padding():
    torch.manual_seed(1)
    model=PooledMLP(4).eval()
    X=torch.randn(2,8,4);lengths=torch.tensor([5,8])
    shuffled=X.clone();shuffled[0,:5]=X[0,[4,2,0,3,1]];shuffled[1]=X[1].flip(0)
    shuffled[0,5:]=9999
    with torch.no_grad():
        torch.testing.assert_close(model(X,lengths),model(shuffled,lengths),rtol=1e-5,atol=1e-6)


def test_shift_and_lookalikes_remain_mixed_and_chronological():
    for mode in ('shifted','lookalikes'):
        frame=generate_windows(811,80,mode,mode=mode)
        validate(frame)
        assert frame.observation_duration_s.eq(3600).all()
        assert frame.groupby('session_id').window_target.first().sum()==40
    with pytest.raises(ValueError):generate_windows(811,5,'odd')


def test_small_complete_run_freezes_thresholds_and_preserves_all_results(tmp_path):
    report=run(Config(seeds=(811,),train_windows=20,val_windows=12,test_windows=12,
                      epochs=2,min_epochs=1,threads=2),tmp_path/'study')
    expected={'beta2_vote','beta2_mean','pooled_mlp','sona','sona_shuffled_test',
              'shuffled_train','summary_boosting','temporal_rule'}
    from ids_ml.experiments.audit_mixed_scan import audit_seed
    assert audit_seed(tmp_path/'study/seed-811')['reloaded_model_split_checks']==32
    record=report['runs'][0]
    assert (tmp_path/'study/protocol.json').exists()
    assert record['audit']['status']=='passed'
    for split in ('val','test','shifted','lookalikes'):
        assert set(record['results'][split])==expected
        for model,result in record['results'][split].items():
            assert result['threshold_from_validation']==record['thresholds'][model]
            m=result['at_validation_false_alarm_target']
            assert m['tp']+m['fn']==6 and m['tn']+m['fp']==6
            if split=='val':assert m['false_positive_rate']<=.01
    with pytest.raises(FileExistsError):run(Config(),tmp_path/'study')


def test_summary_requires_audit_and_retains_all_seed_rows(tmp_path):
    import json
    from ids_ml.experiments.summarize_mixed_scan import summarize
    # Reuse a synthetic results contract, not a new model training run.
    models={'sona':{'recall':.5,'false_positive_rate':.02},
            'summary_boosting':{'recall':.8,'false_positive_rate':.01}}
    runs=[]
    for seed in (811,827):
        results={}
        for split in ('test','shifted','lookalikes'):
            results[split]={name:{'at_validation_false_alarm_target':{**m,'tp':5,'fp':1,'fn':5,'tn':49},
                'threshold_from_validation':.8,'roc_auc':.8,'average_precision':.7,
                'by_behavior':{'ordinary':{'windows':50,'flag_rate':.02}}} for name,m in models.items()}
        runs.append({'seed':seed,'flow_counts':{'test':100},'results':results})
    summary={split:{name:{k:{'mean':v,'min':v,'max':v} for k,v in m.items()} for name,m in models.items()}
             for split in ('test','shifted','lookalikes')}
    report={'config':{'seeds':[811,827],'train_windows':100,'val_windows':100,'test_windows':100},
            'runs':runs,'summary':summary}
    (tmp_path/'report.json').write_text(json.dumps(report))
    (tmp_path/'audit.json').write_text(json.dumps({'status':'failed'}))
    with pytest.raises(ValueError):summarize(tmp_path)
    # Summary normally receives all models; this small contract checks the CSV
    # without hiding a stronger comparator in the narrative.
    from ids_ml.experiments import summarize_mixed_scan as module
    original=module.LABELS
    module.LABELS={name:original[name] for name in models}
    try:
        (tmp_path/'audit.json').write_text(json.dumps({'status':'passed'}))
        summarize(tmp_path)
    finally:module.LABELS=original
    csv=(tmp_path/'per_seed.csv').read_text()
    assert len(csv.splitlines())==13
    assert '827,test,summary_boosting' in csv
    assert 'Summary boosting' in (tmp_path/'RESULTS.md').read_text()
