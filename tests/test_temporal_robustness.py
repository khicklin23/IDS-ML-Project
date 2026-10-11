"""Test training safeguards and timing transforms, without requiring a winning model."""
import dataclasses
import numpy as np
import pandas as pd
import pytest
from ids_ml.data.schema import FEATURE_COLUMNS
from ids_ml.experiments.temporal_patterns import generate_pairs
from ids_ml.experiments.temporal_robustness import relative_view, polling
from ids_ml.models.sequence import SequenceTrainConfig, train_sequence_model, predict_sequence_proba


def test_relative_timing_is_scale_invariant_and_preserves_matched_pairs():
    df, _ = generate_pairs(19, 8, 'relative')
    original = df.copy(deep=True)
    slow = df.copy()
    slow['prior_gap_s'] *= 17
    a, b = relative_view(df), relative_view(slow)
    np.testing.assert_allclose(a.prior_gap_s, b.prior_gap_s, rtol=1e-12)
    pd.testing.assert_frame_equal(df, original)
    for _, pair in a.groupby('pair_id'):
        benign, attack = [pair[pair.target == label] for label in (0, 1)]
        columns = list(FEATURE_COLUMNS)
        pd.testing.assert_frame_equal(benign[columns].sort_values(columns).reset_index(drop=True),
                                      attack[columns].sort_values(columns).reset_index(drop=True))
    changed = df.copy()
    changed['target'] = 1 - changed.target
    changed['scenario'] = 'ignore-provenance'
    np.testing.assert_array_equal(relative_view(changed).prior_gap_s, a.prior_gap_s)


def test_relative_timing_handles_zero_gaps():
    df, _ = generate_pairs(5, 2, 'zero')
    df['prior_gap_s'] = 0.
    assert np.isfinite(relative_view(df).prior_gap_s).all()
    assert (relative_view(df).prior_gap_s == 0).all()


def test_pace_factor_changes_only_timing_and_preserves_legacy_generation():
    a, _ = generate_pairs(31, 4, 'pace')
    b, _ = generate_pairs(31, 4, 'pace', pace=(10,50))
    np.testing.assert_allclose(b.prior_gap_s, a.prior_gap_s*2)
    pd.testing.assert_frame_equal(a.drop(columns=['prior_gap_s','event_time']),
                                  b.drop(columns=['prior_gap_s','event_time']))


def test_benign_polling_is_regular_labelled_normal_and_coherent():
    df = polling(77, 12, 'polling')
    assert set(df.target) == {0}
    assert df.session_id.nunique() == 12
    for _, session in df.groupby('session_id'):
        gaps = session.prior_gap_s.to_numpy()[1:]
        assert gaps.std()/gaps.mean() < .1
        np.testing.assert_allclose(np.diff(session.event_time), gaps)
    pd.testing.assert_frame_equal(df, polling(77, 12, 'polling'))


def test_minimum_training_window_delays_stopping_and_keeps_best_checkpoint():
    X = np.ones((8,2,3), dtype=np.float32)
    lengths = np.full(8,2)
    y = np.tile([0,1],4)
    cfg = SequenceTrainConfig(seed=33, lr=0, max_epochs=12, patience=3, batch_size=8)
    original, history = train_sequence_model(X,lengths,y,X,lengths,y,cfg,2)
    extended, warm_history = train_sequence_model(X,lengths,y,X,lengths,y,
                                                dataclasses.replace(cfg,min_epochs=7),2)
    assert len(history)-1 == 4
    assert len(warm_history)-1 == 9
    assert warm_history[-1]['best_epoch'] == 0
    np.testing.assert_array_equal(predict_sequence_proba(original,X,lengths),
                                  predict_sequence_proba(extended,X,lengths))
    with pytest.raises(ValueError,match='min_epochs'):
        train_sequence_model(X,lengths,y,X,lengths,y,dataclasses.replace(cfg,min_epochs=13),2)


def test_saved_relative_preprocessor_reuses_training_statistics(tmp_path):
    from ids_ml.features import build_preprocessor, transform_features, save_preprocessor, load_preprocessor
    train, _ = generate_pairs(23, 12, 'fit')
    test, _ = generate_pairs(1023, 6, 'heldout', pace=(40,80))
    processor = build_preprocessor(train, timing_mode='relative')
    expected = transform_features(processor, test)
    manual = build_preprocessor(relative_view(train))
    np.testing.assert_array_equal(transform_features(manual, relative_view(test)), expected)
    mean_before = processor.encoder.named_transformers_['numeric'].mean_.copy()
    slow = test.copy()
    slow['prior_gap_s'] *= 8
    np.testing.assert_allclose(transform_features(processor, slow),expected,atol=1e-6)
    np.testing.assert_array_equal(processor.encoder.named_transformers_['numeric'].mean_,mean_before)
    save_preprocessor(processor,tmp_path/'preprocessor.pkl')
    loaded = load_preprocessor(tmp_path/'preprocessor.pkl')
    assert loaded.timing_mode == 'relative'
    np.testing.assert_array_equal(transform_features(loaded,test),expected)
    processor.__dict__.pop('timing_mode')  # Earlier relative artifacts only stored encoder.
    save_preprocessor(processor,tmp_path/'legacy_relative.pkl')
    np.testing.assert_array_equal(transform_features(load_preprocessor(tmp_path/'legacy_relative.pkl'),test),expected)
    with pytest.raises(ValueError,match='session_id'):
        transform_features(loaded,test.drop(columns=['session_id']))
    with pytest.raises(ValueError,match='timing_mode'):
        build_preprocessor(train,timing_mode='invalid')
    missing_id = test.copy()
    missing_id.loc[0,'session_id'] = None
    with pytest.raises(ValueError,match='session_id'):
        transform_features(loaded,missing_id)


def test_relative_session_report_saves_matching_inference_transform(tmp_path):
    import json
    import torch
    from ids_ml.report import build_session_report
    from ids_ml.features import load_preprocessor, SCENARIO_CLASSES
    from ids_ml.models.sequence import SequenceClassifier
    from ids_ml.models.sessions import build_session_sequences
    cfg = SequenceTrainConfig(seed=81,max_epochs=2,batch_size=128)
    report = build_session_report(seed=81,out_dir=tmp_path,num_hosts=12,duration_minutes=15,
                                  episodes_per_family=4,sequence_config=cfg,timing_mode='relative')
    assert report['detector']['timing_mode'] == 'relative'
    artifact = tmp_path/'artifact_sequence'
    metadata = json.loads((artifact/'metadata.json').read_text())
    processor = load_preprocessor(artifact/'preprocessor.pkl')
    assert metadata['timing_mode'] == processor.timing_mode == 'relative'
    # Complete closed synthetic sessions with real schema class names.
    df,_ = generate_pairs(29,2,'reload')
    df['scenario'] = np.where(df.target==0,'benign','beaconing')
    X,lengths,_,_=build_session_sequences(df,processor,list(SCENARIO_CLASSES),max_len=64)
    model=SequenceClassifier(X.shape[-1],len(SCENARIO_CLASSES),metadata['hidden_dim'])
    model.load_state_dict(torch.load(artifact/'model.pt',weights_only=True))
    np.testing.assert_allclose(predict_sequence_proba(model,X,lengths).sum(axis=1),1,atol=1e-6)


def test_hybrid_timing_retains_absolute_gap_and_adds_relative_context(tmp_path):
    from ids_ml.features import build_preprocessor, transform_features, save_preprocessor, load_preprocessor, NUMERIC_FEATURES
    df,_=generate_pairs(91,4,'hybrid')
    absolute=build_preprocessor(df)
    hybrid=build_preprocessor(df,timing_mode='hybrid')
    A=transform_features(absolute,df)
    H=transform_features(hybrid,df)
    n=len(NUMERIC_FEATURES)
    assert H.shape[1] == A.shape[1]+1
    np.testing.assert_array_equal(H[:,:n],A[:,:n])
    np.testing.assert_array_equal(H[:,n+1:],A[:,n:])
    slow=df.copy()
    slow['prior_gap_s']*=10
    S=transform_features(hybrid,slow)
    np.testing.assert_allclose(S[:,n],H[:,n],atol=1e-6)
    assert not np.allclose(S[:,:n],H[:,:n])
    save_preprocessor(hybrid,tmp_path/'hybrid.pkl')
    np.testing.assert_array_equal(transform_features(load_preprocessor(tmp_path/'hybrid.pkl'),df),H)
    from ids_ml.models.classifier import MLPClassifier
    from ids_ml.models.train import TrainConfig, save_artifact, load_artifact
    from ids_ml.models.evaluate import predict_class_proba
    cfg=TrainConfig(seed=91)
    model=MLPClassifier(H.shape[1],hidden_dim=cfg.hidden_dim,dropout=cfg.dropout,output_dim=2)
    save_artifact(model,hybrid,cfg,{},tmp_path/'model',input_dim=H.shape[1],
                  output_dim=2,class_names=['benign','beaconing'])
    loaded_model,loaded_processor,metadata=load_artifact(tmp_path/'model')
    assert metadata['timing_mode']=='hybrid'
    assert 'session_relative_gap' in metadata['numeric_features']
    assert len(metadata['feature_names']) == H.shape[1]
    np.testing.assert_array_equal(predict_class_proba(model,H),
                                  predict_class_proba(loaded_model,transform_features(loaded_processor,df)))


def test_joint_six_class_examples_preserve_schema_and_feature_pairs():
    from ids_ml.experiments.joint_temporal import schema_frame
    from ids_ml.data.schema import validate, SCENARIO_LABEL
    df,_=generate_pairs(88,4,'joint-schema')
    framed=schema_frame(df,88,'joint-run')
    validate(framed)
    assert set(framed.scenario.astype(str)) == {'benign','beaconing'}
    assert (framed.label.astype(str)==framed.scenario.astype(str).map(SCENARIO_LABEL)).all()
    np.testing.assert_array_equal(framed[list(FEATURE_COLUMNS)].to_numpy(),df[list(FEATURE_COLUMNS)].to_numpy())
    assert framed.flow_id.is_unique


def test_tempered_class_weights_change_loss_without_changing_initial_model():
    import torch
    X=np.ones((8,2,3),dtype=np.float32)
    lengths=np.full(8,2)
    train_y=np.array([0]*7+[1])
    val_y=np.tile([0,1],4)
    losses=[]
    for power in (1.,.5):
        cfg=SequenceTrainConfig(seed=72,lr=0,max_epochs=1,batch_size=8,class_weight_power=power)
        model,history=train_sequence_model(X,lengths,train_y,X,lengths,val_y,cfg,2)
        model.eval()
        with torch.no_grad():logits=model(torch.tensor(X),torch.tensor(lengths))
        ratio=7**power
        weights=torch.tensor([2/(1+ratio),2*ratio/(1+ratio)],dtype=torch.float32)
        expected=torch.nn.CrossEntropyLoss(weight=weights)(logits,torch.tensor(val_y)).item()
        assert history[0]['val_loss']==pytest.approx(expected,rel=1e-6)
        losses.append(expected)
    assert losses[0] != pytest.approx(losses[1],abs=1e-5)
    cfg=SequenceTrainConfig(seed=72,max_epochs=1,class_weight_power=2)
    with pytest.raises(ValueError,match='class_weight_power'):
        train_sequence_model(X,lengths,train_y,X,lengths,val_y,cfg,2)


def test_diverse_polling_covers_short_normal_and_long_paces_without_duplicates():
    from ids_ml.experiments.joint_temporal import diverse_polling
    df=diverse_polling(103,12,'coverage')
    means=df.groupby('session_id').prior_gap_s.mean()
    assert len(means)==12
    assert (means<30).sum()==4
    assert ((means>=30)&(means<150)).sum()==4
    assert (means>=150).sum()==4
    assert df.flow_id.is_unique
    assert set(df.target)=={0}
    pd.testing.assert_frame_equal(df,diverse_polling(103,12,'coverage'))
