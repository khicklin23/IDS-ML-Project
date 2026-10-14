"""Reconstruct artifacts, train-only preprocessing, identities and operating points."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ids_ml.data.schema import validate
from ids_ml.experiments.mixed_scan import digest
from ids_ml.experiments.temporal_patterns import threshold_for_false_alarms
from .features import CandidatePreprocessor
from .inference import CandidatePredictor
from .study import metrics, targets, aggregate, SPECS, TESTS, write_json, source_hashes, select_global


def audit_seed(path, config):
    record = json.loads((path/'results.json').read_text())
    identities, frames = {}, {}
    for split, expected in record['source_sha256'].items():
        if digest(path/f'{split}.parquet') != expected: raise AssertionError('Source changed.')
        frame = pd.read_parquet(path/f'{split}.parquet')
        validate(frame)
        assert len(frame) == record['flow_counts'][split]
        assert (frame.event_time >= frame.observation_start_s).all()
        assert (frame.event_time < frame.observation_start_s+frame.observation_duration_s).all()
        identity = {field: set(frame[field]) for field in ('flow_id', 'session_id', 'src_host_id')}
        for previous in identities.values():
            for field in identity: assert not previous[field] & identity[field]
        identities[split] = identity
        truth = frame.scenario.eq('port_scan').groupby(frame.session_id, sort=False).any().to_numpy()
        np.testing.assert_array_equal(targets(frame).astype(bool), truth)
        frames[split] = frame
    checks, operating = 0, 0
    for name, (regime, mode, architecture) in SPECS.items():
        predictor = CandidatePredictor.load(path/name)
        manifest = predictor.manifest
        assert manifest['training_sha256'] == record['source_sha256'][f'train_{regime}']
        assert manifest['validation_sha256'] == record['source_sha256']['val']
        if predictor.processor is not None:
            fresh = CandidatePreprocessor.fit(frames[f'train_{regime}'], mode)
            np.testing.assert_array_equal(fresh.transform(frames['val']), predictor.processor.transform(frames['val']))
            assert not any(x in feature for feature in manifest['feature_names'] for x in ('window_target', 'scenario', 'behavior', 'session_id'))
            assert sum(p.numel() for p in predictor.model.parameters()) == manifest['efficiency']['parameter_count']
        for split in ('val', *TESTS):
            prediction = predictor.predict(frames[split]); scores = prediction['scores']
            if split == 'val':
                saved = np.load(path/name/'validation_scores.npy')
                for budget, threshold in manifest['thresholds'].items():
                    assert threshold == threshold_for_false_alarms(scores, targets(frames[split]), float(budget))
            else:
                saved_file = np.load(path/f'{split}_predictions.npz', allow_pickle=False)
                saved = saved_file[name]
                np.testing.assert_array_equal(saved_file['y'], targets(frames[split]))
                np.testing.assert_array_equal(saved_file['session_ids'], prediction['session_ids'])
            np.testing.assert_allclose(scores, saved, rtol=0, atol=1e-7)
            recomputed = metrics(scores, frames[split], manifest['thresholds'])
            assert recomputed == record['results'][split][name]
            if split == 'val':
                for budget, metric in recomputed.items():
                    assert metric['at_validation_false_alarm_target']['false_positive_rate'] <= float(budget)
            checks += 1; operating += len(recomputed)
        # Labels and bookkeeping must not be needed for inference.
        stripped = frames['familiar'].drop(columns=['label', 'scenario', 'window_target', 'behavior', 'condition', 'seed', 'run_id', 'flow_id', 'src_host_id', 'dst_host_id', 'src_ip', 'dst_ip', 'schema_version', 'observation_start_s', 'observation_duration_s'], errors='ignore')
        np.testing.assert_array_equal(predictor.predict(stripped)['scores'], predictor.predict(frames['familiar'])['scores'])
    selected = json.loads((path/'selection.json').read_text())
    assert selected['validation_results'] == record['results']['val']
    names = [n for n in SPECS if SPECS[n][2] != 'summary']
    choice = sorted(names, key=lambda n: (-record['results']['val'][n]['0.01']['at_validation_false_alarm_target']['recall'],
        CandidatePredictor.load(path/n).manifest['efficiency']['parameter_count'], n))[0]
    assert choice == selected['selected'] == record['selected']
    result = {'status': 'passed', 'seed': record['seed'], 'prediction_checks': checks, 'operating_point_checks': operating,
              'train_only_preprocessing': True, 'identities_capture_bounds_and_label_exclusion': True,
              'validation_only_selection_and_thresholds': True}
    write_json(path/'audit.json', result)
    return result


def audit(directory):
    report = json.loads((directory/'report.json').read_text())
    assert report['config'] == json.loads((directory/'protocol.json').read_text())
    assert source_hashes() == json.loads((directory/'implementation_sha256.json').read_text())
    runs = []
    for record in report['runs']:
        path = directory/f'seed-{record["seed"]}'
        assert json.loads((path/'results.json').read_text()) == record
        result = audit_seed(path, report['config']); runs.append(result)
        print(json.dumps(result), flush=True)
    assert aggregate(report['runs']) == report['summary']
    assert report['selected_global'] == select_global(report['runs'], directory)
    result = {'status': 'passed', 'runs': runs}
    write_json(directory/'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    audit(parser.parse_args().directory)
