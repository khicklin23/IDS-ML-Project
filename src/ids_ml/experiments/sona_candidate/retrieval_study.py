"""Final pre-test-declared architecture extension; validation-only selection."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from ids_ml.experiments.mixed_scan import digest
from ids_ml.experiments.temporal_patterns import threshold_for_false_alarms
from .features import pack
from .models import probabilities
from .parameter_control import train_matched
from .retrieval import RetrievalMemory, ExtendedGRU, make_extended
from .study import Config, TESTS, SPECS, targets, metrics, write_json


def run(study, output):
    report = json.loads((study/'report.json').read_text())
    if json.loads((study/'audit.json').read_text())['status'] != 'passed': raise ValueError('Base audit required.')
    config = Config(**report['config']); torch.set_num_threads(config.threads)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).parent
    protocol = {'source_report_sha256': digest(study/'report.json'), 'config': report['config'],
                'source_sha256': {f: digest(root/f) for f in ('retrieval.py', 'retrieval_study.py', 'parameter_control.py', 'extended_inference.py')},
                'selection': 'Mean strict-budget validation recall, lower parameter count tie break.'}
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text()) != protocol: raise ValueError('Extension resume changed.')
    else: write_json(output/'protocol.json', protocol)
    records = []
    with threadpool_limits(limits=config.threads):
        for r in report['runs']:
            seed = r['seed']; source = study/f'seed-{seed}'
            for architecture, factory in [('retrieval', RetrievalMemory), ('gru40', ExtendedGRU)]:
                directory = output/f'seed-{seed}'/architecture
                if (directory/'validation.json').exists(): continue
                directory.mkdir(parents=True, exist_ok=True)
                processor = pickle.loads((source/'relative_mixed_gru'/'preprocessor.pkl').read_bytes())
                training = pd.read_parquet(source/'train_mixed.parquet'); validation = pd.read_parquet(source/'val.parquet')
                tr = pack(training, processor, config.max_length); va = pack(validation, processor, config.max_length)
                print(f'seed {seed}: training extended {architecture}', flush=True)
                model, history = train_matched((*tr[:2], targets(training)), (*va[:2], targets(validation)), config, seed, factory)
                scores = probabilities(model, *va[:2])[:, 1].astype(np.float64)
                thresholds = {f'{b:.2f}': threshold_for_false_alarms(scores, targets(validation), b) for b in config.budgets}
                torch.save(model.state_dict(), directory/'model.pt')
                (directory/'preprocessor.pkl').write_bytes(pickle.dumps(processor))
                write_json(directory/'history.json', history)
                write_json(directory/'manifest.json', {'artifact_version': 'sona-retrieval-1.0', 'architecture': architecture, 'representation': 'relative',
                    'classes': ['benign', 'suspicious'], 'feature_names': processor.get_feature_names_out().tolist(),
                    'input_dim': tr[0].shape[-1], 'max_length': config.max_length, 'thresholds': thresholds,
                    'parameter_count': sum(p.numel() for p in model.parameters()),
                    'training_sha256': digest(source/'train_mixed.parquet'),
                    'sha256': {f: digest(directory/f) for f in ('model.pt', 'preprocessor.pkl')}})
                np.save(directory/'validation_scores.npy', scores)
                write_json(directory/'validation.json', metrics(scores, validation, thresholds))
    # Freeze final architecture and representation before scoring extended test arms.
    ranks = {}
    for name in [n for n in SPECS if SPECS[n][2] != 'summary']:
        values = [r['results']['val'][name]['0.01']['at_validation_false_alarm_target']['recall'] for r in report['runs']]
        from .inference import CandidatePredictor
        params = np.mean([CandidatePredictor.load(study/f'seed-{r["seed"]}'/name).manifest['efficiency']['parameter_count'] for r in report['runs']])
        ranks[name] = {'recall': float(np.mean(values)), 'parameters': float(params), 'architecture': SPECS[name][2], 'representation': SPECS[name][1]}
    for architecture in ('retrieval', 'gru40'):
        values, parameters = [], []
        for r in report['runs']:
            path = output/f'seed-{r["seed"]}'/architecture
            values.append(json.loads((path/'validation.json').read_text())['0.01']['at_validation_false_alarm_target']['recall'])
            parameters.append(json.loads((path/'manifest.json').read_text())['parameter_count'])
        ranks[architecture] = {'recall': float(np.mean(values)), 'parameters': float(np.mean(parameters)), 'architecture': architecture, 'representation': 'relative'}
    selected = sorted(ranks, key=lambda name: (-ranks[name]['recall'], ranks[name]['parameters'], name))[0]
    selection = {'selected': selected, 'ranking': ranks, 'architecture': ranks[selected]['architecture'], 'representation': ranks[selected]['representation']}
    write_json(output/'selection.json', selection)
    for r in report['runs']:
        source = study/f'seed-{r["seed"]}'
        for architecture in ('retrieval', 'gru40'):
            path = output/f'seed-{r["seed"]}'/architecture
            if (path/'results.json').exists():
                records.append(json.loads((path/'results.json').read_text())); continue
            m = json.loads((path/'manifest.json').read_text())
            processor = pickle.loads((path/'preprocessor.pkl').read_bytes())
            model = make_extended(architecture, m['input_dim']); model.load_state_dict(torch.load(path/'model.pt', weights_only=True))
            results = {'val': json.loads((path/'validation.json').read_text())}
            for split in TESTS:
                frame = pd.read_parquet(source/f'{split}.parquet'); x, lengths, ids, _ = pack(frame, processor, config.max_length)
                scores = probabilities(model, x, lengths)[:, 1].astype(np.float64)
                results[split] = metrics(scores, frame, m['thresholds'])
                np.savez_compressed(path/f'{split}_predictions.npz', scores=scores, y=targets(frame), session_ids=ids)
            record = {'seed': r['seed'], 'architecture': architecture, 'results': results}
            write_json(path/'results.json', record); records.append(record)
            print(f'seed {r["seed"]}: evaluated extended {architecture}', flush=True)
    final = {'protocol': protocol, 'selection': selection, 'runs': records, 'public_promotion': False}
    write_json(output/'report.json', final)
    return final


def audit(study, output):
    report = json.loads((output/'report.json').read_text()); config = Config(**report['protocol']['config'])
    torch.set_num_threads(config.threads)
    for f, expected in report['protocol']['source_sha256'].items(): assert digest(Path(__file__).parent/f) == expected
    assert digest(study/'report.json') == report['protocol']['source_report_sha256']
    checks = 0
    for r in report['runs']:
        source = study/f'seed-{r["seed"]}'; path = output/f'seed-{r["seed"]}'/r['architecture']
        m = json.loads((path/'manifest.json').read_text())
        for f, expected in m['sha256'].items(): assert digest(path/f) == expected
        assert digest(source/'train_mixed.parquet') == m['training_sha256']
        processor = pickle.loads((path/'preprocessor.pkl').read_bytes())
        model = make_extended(r['architecture'], m['input_dim']); model.load_state_dict(torch.load(path/'model.pt', weights_only=True))
        for split in ('val', *TESTS):
            frame = pd.read_parquet(source/f'{split}.parquet'); x, lengths, ids, _ = pack(frame, processor, config.max_length)
            scores = probabilities(model, x, lengths)[:, 1].astype(np.float64)
            if split == 'val':
                np.testing.assert_array_equal(scores, np.load(path/'validation_scores.npy'))
                for budget, threshold in m['thresholds'].items(): assert threshold == threshold_for_false_alarms(scores, targets(frame), float(budget))
            else:
                saved = np.load(path/f'{split}_predictions.npz', allow_pickle=False)
                np.testing.assert_array_equal(scores, saved['scores']); np.testing.assert_array_equal(ids, saved['session_ids'])
            assert metrics(scores, frame, m['thresholds']) == r['results'][split]
            checks += 1
    result = {'status': 'passed', 'prediction_checks': checks, 'operating_point_checks': checks*3}
    ranking = report['selection']['ranking']
    base = json.loads((study/'report.json').read_text())
    for name, item in ranking.items():
        if name in ('retrieval', 'gru40'):
            values = [r['results']['val']['0.01']['at_validation_false_alarm_target']['recall'] for r in report['runs'] if r['architecture'] == name]
        else:
            values = [r['results']['val'][name]['0.01']['at_validation_false_alarm_target']['recall'] for r in base['runs']]
        assert item['recall'] == float(np.mean(values))
    choice = sorted(ranking, key=lambda n: (-ranking[n]['recall'], ranking[n]['parameters'], n))[0]
    assert choice == report['selection']['selected']
    assert report['selection'] == json.loads((output/'selection.json').read_text())
    result['validation_only_selection_verified'] = True
    write_json(output/'audit.json', result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('study', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--audit-only', action='store_true'); args = p.parse_args()
    if not args.audit_only: run(args.study, args.output)
    print(json.dumps(audit(args.study, args.output)))
