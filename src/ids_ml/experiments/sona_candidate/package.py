"""Package validation-selected checkpoints with exact-equivalent faster inference."""
import argparse
import json
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ids_ml.experiments.mixed_scan import digest
from .fast_features import FastCandidatePreprocessor
from .inference import CandidatePredictor
from .study import write_json


def package(study, destination):
    report = json.loads((study/'report.json').read_text())
    audit = json.loads((study/'audit.json').read_text())
    if audit['status'] != 'passed': raise ValueError('Passed study audit required.')
    if destination.exists(): raise FileExistsError('Refuse to overwrite a candidate package.')
    destination.mkdir(parents=True)
    records = []
    for run in report['runs']:
        original = study/f'seed-{run["seed"]}'/report['selected_global']
        predictor = CandidatePredictor.load(original)
        fast = FastCandidatePreprocessor.from_processor(predictor.processor)
        checks = []
        for split in ('val', 'familiar', 'duration', 'density', 'combined', 'inventory', 'intermediate', 'longer'):
            frame = pd.read_parquet(original.parent/f'{split}.parquet')
            old = predictor.processor.transform(frame); new = fast.transform(frame)
            np.testing.assert_array_equal(old, new)
            checks.append(split)
        # Paired timing on a fixed validation subset; no detection decisions change.
        frame = pd.read_parquet(original.parent/'val.parquet')
        ids = frame.session_id.drop_duplicates().iloc[:128]
        subset = frame[frame.session_id.isin(ids)].reset_index(drop=True)
        old_times, new_times = [], []
        for _ in range(7):
            start = time.perf_counter(); predictor.processor.transform(subset); old_times.append(time.perf_counter()-start)
            start = time.perf_counter(); fast.transform(subset); new_times.append(time.perf_counter()-start)
        path = destination/f'seed-{run["seed"]}'
        shutil.copytree(original, path)
        (path/'preprocessor.pkl').write_bytes(pickle.dumps(fast))
        manifest = json.loads((path/'manifest.json').read_text())
        manifest['sha256']['preprocessor.pkl'] = digest(path/'preprocessor.pkl')
        manifest['preprocessor_version'] = fast.version
        manifest['source_candidate'] = str(original.resolve())
        manifest['exact_feature_equivalence_partitions'] = checks
        manifest['preprocessing_timing'] = {'windows': len(ids), 'old_median_ms': float(np.median(old_times))*1000,
            'fast_median_ms': float(np.median(new_times))*1000, 'repeats': 7,
            'limitation': 'Paired in-process measurements; independent isolated benchmark remains authoritative.'}
        write_json(path/'manifest.json', manifest)
        reloaded = CandidatePredictor.load(path)
        np.testing.assert_array_equal(predictor.predict(subset)['scores'], reloaded.predict(subset)['scores'])
        records.append({'seed': run['seed'], 'path': str(path.resolve()), 'manifest_sha256': digest(path/'manifest.json'),
                        'equivalence_checks': checks, 'preprocessing_timing': manifest['preprocessing_timing']})
    for file in ('protocol.json', 'report.json', 'audit.json', 'decision.json', 'RESULTS.md', 'per_seed.csv'):
        if (study/file).exists(): shutil.copy2(study/file, destination/file)
    write_json(destination/'package.json', {'candidate': report['selected_global'], 'public_promotion': False,
        'source_study': str(study.resolve()), 'runs': records,
        'description': 'Experimental binary-window checkpoint, exact-equivalent optimized preprocessing.'})
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path); parser.add_argument('destination', type=Path)
    args = parser.parse_args(); print(json.dumps(package(args.study, args.destination), indent=2))
