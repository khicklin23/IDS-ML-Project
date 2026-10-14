"""Final validation-selected experimental package; no public replacement."""
import argparse
import json
import pickle
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from ids_ml.experiments.mixed_scan import digest
from .extended_inference import load_predictor
from .fast_features import FastCandidatePreprocessor
from .study import TESTS, write_json


def package(study, extension, destination):
    base = json.loads((study/'report.json').read_text())
    final = json.loads((extension/'report.json').read_text())
    if json.loads((study/'audit.json').read_text())['status'] != 'passed' or json.loads((extension/'audit.json').read_text())['status'] != 'passed':
        raise ValueError('Both complete audits must pass.')
    if destination.exists(): raise FileExistsError('Preserve existing packages.')
    destination.mkdir(parents=True)
    chosen = final['selection']['selected']; records = []
    for record in base['runs']:
        seed = record['seed']
        source = (extension if chosen in ('retrieval', 'gru40') else study)/f'seed-{seed}'/chosen
        predictor = load_predictor(source)
        fast = FastCandidatePreprocessor.from_processor(predictor.processor)
        for split in ('val', *TESTS):
            frame = pd.read_parquet(study/f'seed-{seed}'/f'{split}.parquet')
            np.testing.assert_array_equal(predictor.processor.transform(frame), fast.transform(frame))
        target = destination/f'seed-{seed}'; shutil.copytree(source, target)
        (target/'preprocessor.pkl').write_bytes(pickle.dumps(fast))
        manifest = json.loads((target/'manifest.json').read_text())
        manifest['sha256']['preprocessor.pkl'] = digest(target/'preprocessor.pkl')
        manifest['preprocessor_version'] = fast.version
        manifest['exact_equivalence_partitions'] = ['val', *TESTS]
        manifest['source_candidate'] = str(source.resolve())
        write_json(target/'manifest.json', manifest)
        frame = pd.read_parquet(study/f'seed-{seed}'/'familiar.parquet')
        np.testing.assert_array_equal(predictor.predict(frame)['scores'], load_predictor(target).predict(frame)['scores'])
        records.append({'seed': seed, 'directory': str(target.resolve()), 'manifest_sha256': digest(target/'manifest.json')})
    root = Path(__file__).parent
    package_record = {'selected': chosen, 'architecture': final['selection']['architecture'], 'representation': final['selection']['representation'],
                      'public_promotion': False, 'unit': 'binary whole observation window', 'runs': records,
                      'runtime_source_sha256': {p.name: digest(p) for p in sorted(root.glob('*.py'))},
                      'source_study': str(study.resolve()), 'source_extension': str(extension.resolve())}
    write_json(destination/'package.json', package_record)
    (destination/'README.md').write_text('Experimental Sona candidate. Binary window scores, not public six-category Sona.\n'
        'Load with ids_ml.experiments.sona_candidate.extended_inference.load_predictor.\n'
        'Use only trusted local artifacts. See docs/SONA_CANDIDATE.md for the frozen protocol and limits.\n')
    return package_record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('study', type=Path); p.add_argument('extension', type=Path); p.add_argument('destination', type=Path)
    args = p.parse_args(); print(json.dumps(package(args.study, args.extension, args.destination), indent=2))
