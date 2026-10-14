import numpy as np
import pandas as pd
import pytest
import torch
from ids_ml.experiments.scan_generalization_features import generate_condition, prepare as old_prepare
from ids_ml.experiments.scan_context import PACE_COLUMNS
from ids_ml.experiments.sona_candidate.features import CandidatePreprocessor, prepare, pack
from ids_ml.experiments.sona_candidate.models import HierarchicalMemory, PoolClassifier


def test_explicit_processor_matches_frozen_relative_and_is_causal():
    frame = generate_condition(1301, 8, 'features', 'test')
    new = prepare(frame)
    np.testing.assert_allclose(new[list(PACE_COLUMNS)], old_prepare(frame)[list(PACE_COLUMNS)], rtol=1e-12, atol=1e-12)
    processor = CandidatePreprocessor.fit(frame)
    first = frame[frame.session_id == frame.session_id.iloc[0]].iloc[:15].reset_index(drop=True)
    np.testing.assert_array_equal(processor.transform(first), processor.transform(frame)[:15])
    scaled = frame.copy(); scaled.event_time *= 4; scaled.prior_gap_s *= 4
    np.testing.assert_allclose(processor.transform(scaled), processor.transform(frame), atol=1e-6)
    changed = frame.copy(); changed.window_target = 1-changed.window_target
    for column in ('scenario', 'label', 'behavior'): changed[column] = 'hidden'
    np.testing.assert_array_equal(processor.transform(frame), processor.transform(changed))
    mean = processor.encoder.named_transformers_['numeric'].mean_.copy()
    processor.transform(scaled)
    np.testing.assert_array_equal(mean, processor.encoder.named_transformers_['numeric'].mean_)
    with pytest.raises(ValueError): pack(frame, processor, 3)
    bad = frame.copy(); bad.prior_gap_s += 1
    with pytest.raises(ValueError): processor.transform(bad)


def test_hierarchical_masking_gradients_and_pool_order_invariance():
    torch.set_num_threads(2); torch.manual_seed(4)
    x = torch.randn(4, 27, 12); lengths = torch.tensor([27, 19, 8, 1])
    model = HierarchicalMemory(12)
    padded = x.clone()
    for i, n in enumerate(lengths): padded[i, n:] = 900
    torch.testing.assert_close(model(x, lengths), model(padded, lengths))
    logits = model(x, lengths)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    pool = PoolClassifier(12)
    shuffled = x.clone()
    for i, n in enumerate(lengths): shuffled[i, :n] = x[i, :n].flip(0)
    torch.testing.assert_close(pool(x, lengths), pool(shuffled, lengths))


def test_complete_artifact_replay_and_resume(tmp_path):
    from ids_ml.experiments.sona_candidate.study import Config, run, TESTS
    from ids_ml.experiments.sona_candidate.audit import audit
    config = Config(seeds=(1301,), train_windows=40, val_windows=40, test_windows=8, epochs=2, min_epochs=1)
    result = run(config, tmp_path/'study')
    checked = audit(tmp_path/'study')
    assert checked['runs'][0]['prediction_checks'] == 64
    assert checked['runs'][0]['operating_point_checks'] == 192
    assert set(result['summary']) == set(TESTS)
    assert run(config, tmp_path/'study') == result
    from ids_ml.experiments.sona_candidate.summarize import summarize
    from ids_ml.experiments.sona_candidate.package import package
    assert summarize(tmp_path/'study')['public_promotion'] is False
    packaged = package(tmp_path/'study', tmp_path/'package')
    assert packaged[0]['equivalence_checks'] == ['val', 'familiar', 'duration', 'density', 'combined', 'inventory', 'intermediate', 'longer']
    from ids_ml.experiments.sona_candidate.parameter_control import run as run_control, audit as audit_control
    controls = run_control(tmp_path/'study', tmp_path/'control')
    assert len(controls) == 2
    verified = audit_control(tmp_path/'study', tmp_path/'control')
    assert verified['prediction_checks'] == 16 and verified['operating_point_checks'] == 48
    from ids_ml.experiments.sona_candidate.retrieval_study import run as run_extension, audit as audit_extension
    extension = run_extension(tmp_path/'study', tmp_path/'extension')
    assert extension['public_promotion'] is False
    assert audit_extension(tmp_path/'study', tmp_path/'extension')['prediction_checks'] == 16
    from ids_ml.experiments.sona_candidate.package_final import package as package_final
    exported = package_final(tmp_path/'study', tmp_path/'extension', tmp_path/'final_package')
    assert exported['public_promotion'] is False
    with pytest.raises(ValueError): run(Config(seeds=(1301,), train_windows=80, val_windows=40, test_windows=8, epochs=2, min_epochs=1), tmp_path/'study')


@pytest.mark.parametrize('architecture', ['hierarchical', 'retrieval'])
def test_independent_evaluation_adapter_roundtrip(tmp_path, architecture):
    from ids_ml.experiments.sona_candidate.evaluation_adapter import build_evaluation_adapter
    train = generate_condition(1301, 8, 'train-adapter', 'test')
    val = generate_condition(1327, 8, 'val-adapter', 'test')
    adapter = build_evaluation_adapter(architecture=architecture, mode='fit', epochs=2, min_epochs=1)
    adapter.fit(train, val, 1301)
    before = adapter.score_prepared(adapter.prepare(val))
    adapter.save(tmp_path/'adapter')
    loaded = build_evaluation_adapter().load(tmp_path/'adapter')
    np.testing.assert_array_equal(loaded.score_prepared(loaded.prepare(val)), before)
    assert loaded.window_order(loaded.prepare(val)) == val.groupby('session_id', sort=False).session_id.first().tolist()
    stripped = val.drop(columns=['window_target', 'label', 'scenario', 'behavior'])
    np.testing.assert_array_equal(loaded.score_prepared(loaded.prepare(stripped)), before)
    import json
    json.dumps(loaded.describe())
    assert loaded.describe()['parameter_count'] == sum(p.numel() for p in loaded.modules()[0].parameters())
    assert loaded.describe()['fit_context'] == {'seed': 1301, 'train_windows': 8, 'validation_windows': 8}


def test_optimized_processor_exact_equivalence_and_invalid_gap_rejection():
    from ids_ml.experiments.sona_candidate.fast_features import FastCandidatePreprocessor
    frame = generate_condition(1361, 12, 'fast', 'inventory')
    for mode in ('raw', 'relative'):
        original = CandidatePreprocessor.fit(frame, mode)
        fast = FastCandidatePreprocessor.from_processor(original)
        np.testing.assert_array_equal(original.transform(frame), fast.transform(frame))
        prefix = frame[frame.session_id == frame.session_id.iloc[0]].iloc[:9].reset_index(drop=True)
        np.testing.assert_array_equal(fast.transform(prefix), original.transform(prefix))
        bad = frame.copy(); bad.prior_gap_s += 1
        with pytest.raises(ValueError): fast.transform(bad)


def test_retrieval_event_states_are_causal_and_padding_invariant():
    from ids_ml.experiments.sona_candidate.retrieval import RetrievalMemory, ExtendedGRU
    torch.set_num_threads(2); torch.manual_seed(6)
    model = RetrievalMemory(21)
    x = torch.randn(3, 41, 21); lengths = torch.tensor([41, 19, 7])
    padded = x.clone()
    for i, n in enumerate(lengths): padded[i, n:] = 1000
    torch.testing.assert_close(model(x, lengths), model(padded, lengths))
    torch.testing.assert_close(model.encode(x)[:, :13], model.encode(x[:, :13]), atol=1e-6, rtol=1e-6)
    loss = torch.nn.functional.cross_entropy(model(x, lengths), torch.tensor([0, 1, 0])); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    count = sum(p.numel() for p in model.parameters())
    control = sum(p.numel() for p in ExtendedGRU(21).parameters())
    assert abs(count-control)/control < .03
