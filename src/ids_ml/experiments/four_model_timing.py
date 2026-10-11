"""Extend frozen timing datasets with controlled Beta 1/Beta 3 counterparts."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ids_ml.data.schema import validate
from ids_ml.features import build_preprocessor, extract_scenario_labels, transform_features, save_preprocessor, load_preprocessor
from ids_ml.models.train import TrainConfig, train_multiclass_model, save_artifact, load_artifact
from ids_ml.models.sequence import SequenceTrainConfig, train_sequence_model, SequenceClassifier
from ids_ml.models.sessions import build_session_sequences
from ids_ml.models.evaluate import predict_class_proba, multiclass_metrics_from_predictions
from .joint_temporal import sequence_probs
from .temporal_patterns import shuffled


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def narrow_frame(frame, seed):
    """Retain standard ordinary traffic and the same shared temporal practice."""
    return frame[frame.run_id.astype(str) != f'run-{seed+1000}'].copy().reset_index(drop=True)


def votes(model, processor, frame, ids, classes):
    flow = predict_class_proba(model, transform_features(processor, frame)).argmax(axis=1)
    groups = frame.groupby('session_id', sort=False).indices
    return np.array([np.bincount(flow[groups[sid]], minlength=len(classes)).argmax() for sid in ids])


def session_keys(frame):
    return set(zip(frame.run_id.astype(str), frame.session_id.astype(str).str.split('::').str[-1]))


def run(parent, output, *, resume_evaluation=False):
    if not resume_evaluation:
        output.mkdir(parents=True, exist_ok=False)
    else:
        assert (output/'beta1/model.pt').is_file() and (output/'beta3.pt').is_file()
    torch.set_num_threads(2)
    reference = json.loads((parent/'report.json').read_text())
    assert json.loads((parent/'audit.json').read_text())['status'] == 'passed'
    assert json.loads((parent/'extra_stress/report.json').read_text())['audit_status'] == 'passed'
    seed = reference['seed']
    metadata = reference['metadata']
    assert metadata['timing_mode'] == 'relative'
    classes = metadata['class_names']
    cap = metadata['max_seq_len']
    primary = list(reference['results'])
    supplemental = list(json.loads((parent/'extra_stress/report.json').read_text())['results'])
    paths = {name: (parent/'extra_stress' if name in supplemental else parent)/f'{name}.parquet'
             for name in ['train', 'val', *primary, *supplemental]}
    hashes = {name: digest(path) for name, path in paths.items()}
    artifact_paths = [parent/'sona.pt', parent/'sona_preprocessor.pkl', parent/'sona_metadata.json',
                      parent/'beta2/model.pt', parent/'beta2/preprocessor.pkl', parent/'beta2/metadata.json']
    artifact_hashes = {str(path.relative_to(parent)): digest(path) for path in artifact_paths}
    original_train = pd.read_parquet(paths['train'])
    original_val = pd.read_parquet(paths['val'])
    train = narrow_frame(original_train, seed)
    val = narrow_frame(original_val, seed)
    assert len(train) < len(original_train) and len(val) < len(original_val)
    for source, subset in [(original_train, train), (original_val, val)]:
        pd.testing.assert_frame_equal(source[source.pair_id.notna()].reset_index(drop=True),
                                      subset[subset.pair_id.notna()].reset_index(drop=True))
        assert f'run-{seed+1000}' not in set(subset.run_id.astype(str))
    processor = build_preprocessor(train, timing_mode='relative')
    train_pack = build_session_sequences(train, processor, classes, max_len=cap)
    val_pack = build_session_sequences(val, processor, classes, max_len=cap)
    flow_config = TrainConfig(seed=seed, max_epochs=80, patience=10, batch_size=512)
    sequence_config = SequenceTrainConfig(**metadata['train_config'])
    if not resume_evaluation:
        print(f'seed {seed}: train Beta 1 on {len(train):,} flows; shared timing practice retained', flush=True)
        def progress(name):
            def report(row):
                if row['epoch']%10 == 0:
                    print(f'seed {seed} {name} epoch {row["epoch"]+1}, validation F1 {row["val_f1"]:.3f}', flush=True)
            return report
        beta1, flow_history = train_multiclass_model(transform_features(processor, train), extract_scenario_labels(train),
            transform_features(processor, val), extract_scenario_labels(val), flow_config, len(classes), on_epoch=progress('Beta 1'))
        save_artifact(beta1, processor, flow_config, {}, output/'beta1', input_dim=train_pack[0].shape[-1],
                      output_dim=len(classes), class_names=classes)
        beta3, sequence_history = train_sequence_model(*train_pack[:3], *val_pack[:3], sequence_config, len(classes),
                                                       on_epoch=progress('Beta 3'))
        torch.save(beta3.state_dict(), output/'beta3.pt')
        save_preprocessor(processor, output/'beta3_preprocessor.pkl')
        beta3_meta = {**metadata, 'model_version': 'Beta 3 timing counterpart',
                     'input_dim': train_pack[0].shape[-1], 'feature_names': processor.get_feature_names_out().tolist()}
        (output/'beta3_metadata.json').write_text(json.dumps(beta3_meta, indent=2))
        (output/'training_history.json').write_text(json.dumps({'v1':flow_history,'v3':sequence_history},indent=2))
    else:
        beta1, processor, flow_meta = load_artifact(output/'beta1')
        assert flow_meta['train_config'] == dataclasses.asdict(flow_config)
        beta3_meta = json.loads((output/'beta3_metadata.json').read_text())
        assert beta3_meta['train_config'] == dataclasses.asdict(sequence_config)
        beta3 = SequenceClassifier(beta3_meta['input_dim'],len(classes),sequence_config.hidden_dim)
        beta3.load_state_dict(torch.load(output/'beta3.pt',weights_only=True))
        assert processor.get_feature_names_out().tolist() == beta3_meta['feature_names']
        history_path = output/'training_history.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else {'v1':[], 'v3':[]}
        flow_history, sequence_history = history['v1'], history['v3']
        print(f'seed {seed}: resume evaluation of frozen Beta 1/Beta 3 checkpoints; no retraining',flush=True)
    reloaded1, reloaded_processor1, _ = load_artifact(output/'beta1')
    reloaded3 = SequenceClassifier(beta3_meta['input_dim'], len(classes), sequence_config.hidden_dim)
    reloaded3.load_state_dict(torch.load(output/'beta3.pt', weights_only=True))
    reloaded_processor3 = load_preprocessor(output/'beta3_preprocessor.pkl')
    beta2, beta2_processor, _ = load_artifact(parent/'beta2')
    sona_processor = load_preprocessor(parent/'sona_preprocessor.pkl')
    sona = SequenceClassifier(metadata['input_dim'], len(classes), metadata['hidden_dim'])
    sona.load_state_dict(torch.load(parent/'sona.pt', weights_only=True))
    seen = session_keys(original_train) | session_keys(original_val)
    assert not session_keys(original_train) & session_keys(original_val)
    results = {}; checks = 0; pair_checks = 0
    for name in [*primary, *supplemental]:
        frame = pd.read_parquet(paths[name]); validate(frame)
        session_set = session_keys(frame)
        assert not session_set & seen, (name, 'split overlap')
        seen.update(session_set)
        assert frame.groupby('session_id').size().max() <= cap
        X, lengths, y, ids = build_session_sequences(frame, processor, classes, max_len=cap)
        loaded_X, loaded_lengths, loaded_y, loaded_ids = build_session_sequences(frame, reloaded_processor3, classes, max_len=cap)
        np.testing.assert_array_equal(X, loaded_X); np.testing.assert_array_equal(lengths, loaded_lengths)
        np.testing.assert_array_equal(y, loaded_y); assert ids == loaded_ids
        predictions1 = votes(beta1, processor, frame, ids, classes)
        np.testing.assert_array_equal(predictions1, votes(reloaded1, reloaded_processor1, frame, ids, classes))
        probs3 = sequence_probs(beta3, X, lengths)
        np.testing.assert_allclose(probs3, sequence_probs(reloaded3, loaded_X, loaded_lengths), atol=1e-6, rtol=1e-5)
        predictions3 = probs3.argmax(axis=1)
        saved_directory = parent/'extra_stress' if name in supplemental else parent
        with np.load(saved_directory/f'{name}_predictions.npz') as saved:
            np.testing.assert_array_equal(saved['y'], y); np.testing.assert_array_equal(saved['session_ids'], np.array(ids))
            predictions2 = saved['beta2'].copy(); predictions4 = saved['sona'].copy()
            np.testing.assert_array_equal(predictions2, votes(beta2, beta2_processor, frame, ids, classes))
            sona_X, sona_lengths, sona_y, sona_ids = build_session_sequences(frame, sona_processor, classes, max_len=cap)
            np.testing.assert_array_equal(sona_y, y); assert sona_ids == ids
            np.testing.assert_allclose(sequence_probs(sona, sona_X, sona_lengths), saved['sona_probs'], atol=1e-6, rtol=1e-5)
        results[name] = {key: multiclass_metrics_from_predictions(pred, y, classes)
                        for key, pred in [('v1', predictions1), ('v2', predictions2), ('v3', predictions3), ('v4', predictions4)]}
        if 'pair_id' in frame and name not in ('polling', 'unseen_polling', 'extreme_polling'):
            groups = frame[frame.pair_id.notna()].groupby('pair_id', sort=False)
            mapping = dict(zip(ids, predictions1))
            for _, pair in groups:
                sessions = list(pair.session_id.unique())
                assert len(sessions) == 2 and mapping[sessions[0]] == mapping[sessions[1]]
                pair_checks += 1
            if pair_checks:
                results[name]['v3_shuffled'] = multiclass_metrics_from_predictions(
                    sequence_probs(beta3, shuffled(X, lengths, seed+130000), lengths).argmax(axis=1), y, classes)
        np.savez_compressed(output/f'{name}_predictions.npz', y=y, session_ids=np.array(ids),
                            v1=predictions1, v2=predictions2, v3=predictions3, v4=predictions4, v3_probs=probs3)
        checks += 1
        print(f'seed {seed} {name}: '+', '.join(f'{key} {results[name][key]["accuracy"]:.3f}' for key in ('v1','v2','v3','v4')), flush=True)
    assert hashes == {name: digest(path) for name, path in paths.items()}
    assert artifact_hashes == {str(path.relative_to(parent)): digest(path) for path in artifact_paths}
    result = {'seed': seed, 'parent': str(parent), 'results': results, 'classes': classes,
              'training': {'ordinary': 'Standard only for v1/v3; standard plus stealth for v2/v4.',
                           'temporal': 'Identical shared timing pairs and polling for all four.',
                           'preprocessing': 'relative, fitted on each training recipe only',
                           'narrow_flows':len(train), 'narrow_sessions':train.session_id.nunique(),
                           'broad_flows':len(original_train), 'broad_sessions':original_train.session_id.nunique()},
              'configs': {'flow': dataclasses.asdict(flow_config), 'sequence':dataclasses.asdict(sequence_config)},
              'histories': {'v1': flow_history, 'v3':sequence_history},
              'evaluation_resumed_without_refitting':resume_evaluation,
              'history_note':'Histories not persisted before first audit interruption; frozen checkpoints retained.' if resume_evaluation and not flow_history else 'Persisted training histories.',
              'source_hashes': hashes, 'artifact_hashes':artifact_hashes,
              'audit': {'status':'passed','reload_checks_all_four_models':checks,'paired_vote_checks':pair_checks,
                        'split_ids':'disjoint','source_data_and_reused_artifacts':'unchanged'},
              'limitations':['Known benchmark extension; not independent confirmation.',
                             'Retrained architecture/recipe counterparts, not original public beta checkpoints.',
                             'Complete sessions only; speed not measured.']}
    (output/'report.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True); parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--resume-evaluation',action='store_true')
    args = parser.parse_args(); run(args.parent, args.out_dir,resume_evaluation=args.resume_evaluation)
