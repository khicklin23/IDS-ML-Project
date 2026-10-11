"""Audit every saved split and paired prediction in a completed temporal benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ids_ml.data.schema import FEATURE_COLUMNS


def audit(directory: Path) -> dict:
    report = json.loads((directory / 'report.json').read_text())
    checked = {'seeds': [], 'flows': 0, 'sessions': 0, 'matched_pairs': 0}
    for run in report['runs']:
        seen = {column: set() for column in ('flow_id', 'session_id', 'pair_id')}
        root = directory / f'seed-{run["seed"]}'
        for split, manifest in run['manifests'].items():
            df = pd.read_parquet(root / f'{split}.parquet')
            assert len(df) == manifest['flows'], (run['seed'], split, 'flow count')
            assert df.flow_id.is_unique, (run['seed'], split, 'duplicate flow ID')
            for column in seen:
                ids = set(df[column])
                assert not ids & seen[column], (run['seed'], split, 'split overlap', column)
                seen[column].update(ids)
            hashes = pd.util.hash_pandas_object(df[list(FEATURE_COLUMNS)], index=False).to_numpy()
            pairs = df.groupby('pair_id', sort=False).indices
            assert len(pairs) == manifest['pairs']
            for indices in pairs.values():
                labels = df.target.to_numpy()[indices]
                benign = indices[labels == 0]
                attack = indices[labels == 1]
                assert len(benign) == len(attack) > 0
                # Hash-based whole-artifact audit supplements exact frame comparisons in unit tests.
                np.testing.assert_array_equal(np.sort(hashes[benign]), np.sort(hashes[attack]))
                np.testing.assert_array_equal(hashes[benign[[0, -1]]], hashes[attack[[0, -1]]])
            for indices in df.groupby('session_id', sort=False).indices.values():
                frame = df.iloc[indices]
                np.testing.assert_allclose(np.diff(frame.event_time), frame.prior_gap_s.to_numpy()[1:])
            checked['flows'] += len(df)
            checked['sessions'] += manifest['sessions']
            checked['matched_pairs'] += manifest['pairs']
            if split in ('test', 'shifted'):
                with np.load(root / f'{split}_predictions.npz') as scores:
                    np.testing.assert_array_equal(scores['y'], np.tile([0, 1], manifest['pairs']))
                    for model in ('beta2', 'last_flow_logistic'):
                        np.testing.assert_array_equal(scores[model][::2], scores[model][1::2])
        checked['seeds'].append(run['seed'])
    checked['status'] = 'passed'
    (directory / 'audit.json').write_text(json.dumps(checked, indent=2))
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))


if __name__ == '__main__':
    main()
