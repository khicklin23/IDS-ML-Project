"""Pre-test-declared capacity-matched pooling confirmation, separate from v1."""
import argparse
import copy
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from ids_ml.experiments.mixed_scan import digest
from ids_ml.experiments.temporal_patterns import threshold_for_false_alarms
from .features import pack
from .models import probabilities
from .study import Config, TESTS, targets, metrics, write_json


class MatchedPool(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.flow = nn.Sequential(nn.Linear(dimension, 96), nn.ReLU(), nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x, lengths):
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        return (self.flow(x)*valid[..., None]).sum(1)/lengths[:, None]


def train_matched(training, validation, config, seed, model_factory=MatchedPool):
    torch.manual_seed(seed)
    x, lengths, y = training
    v, vl, vy = validation
    model = model_factory(x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    tr = [torch.tensor(a) for a in training]; va = [torch.tensor(a) for a in validation]
    generator = torch.Generator().manual_seed(seed)
    best, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), float('inf'), -1, 0
    history = []; started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train(); total = 0.
        for rows in torch.randperm(len(x), generator=generator).split(config.batch_size):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(tr[0][rows], tr[1][rows]), tr[2][rows])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            total += float(loss.detach())*len(rows)
        model.eval(); losses = []
        with torch.inference_mode():
            for start in range(0, len(v), config.batch_size):
                stop = start+config.batch_size
                loss = nn.functional.cross_entropy(model(va[0][start:stop], va[1][start:stop]), va[2][start:stop])
                losses.append((float(loss), len(va[2][start:stop])))
        value = sum(a*b for a,b in losses)/len(v)
        history.append({'epoch': epoch, 'train_loss': total/len(x), 'val_loss': value})
        if epoch % 10 == 0: print(f'seed {seed} {type(model).__name__} epoch {epoch+1} val_loss={value:.5f}', flush=True)
        if value < best_loss:
            best, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), value, epoch, 0
        elif epoch+1 >= config.min_epochs:
            stale += 1
            if stale >= config.patience: break
    model.load_state_dict(best)
    history.append({'best_epoch': best_epoch, 'best_val_loss': best_loss, 'training_seconds': time.perf_counter()-started})
    return model, history


def run(study, output):
    original = json.loads((study/'report.json').read_text())
    if json.loads((study/'audit.json').read_text())['status'] != 'passed': raise ValueError('Original audit required.')
    config = Config(**original['config']); torch.set_num_threads(config.threads)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'source_study': str(study.resolve()), 'source_report_sha256': digest(study/'report.json'),
                'implementation_sha256': digest(Path(__file__)), 'config': original['config'],
                'candidate_selection_eligible': False, 'widths': [96, 32]}
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text()) != protocol: raise ValueError('Control resume changed.')
    else: write_json(output/'protocol.json', protocol)
    records = []
    with threadpool_limits(limits=config.threads):
        for source in original['runs']:
            seed = source['seed']; seed_path = study/f'seed-{seed}'
            dest = output/f'seed-{seed}'; dest.mkdir(exist_ok=True)
            for regime in ('ordinary', 'mixed'):
                directory = dest/f'relative_{regime}_pool_matched'
                if (directory/'results.json').exists():
                    records.append(json.loads((directory/'results.json').read_text())); continue
                directory.mkdir(exist_ok=True)
                processor = pickle.loads((seed_path/f'relative_{regime}_gru'/'preprocessor.pkl').read_bytes())
                train_frame = pd.read_parquet(seed_path/f'train_{regime}.parquet')
                val_frame = pd.read_parquet(seed_path/'val.parquet')
                tr = pack(train_frame, processor, config.max_length); va = pack(val_frame, processor, config.max_length)
                model, history = train_matched((*tr[:2], targets(train_frame)), (*va[:2], targets(val_frame)), config, seed)
                torch.save(model.state_dict(), directory/'model.pt')
                (directory/'preprocessor.pkl').write_bytes(pickle.dumps(processor))
                scores = probabilities(model, *va[:2])[:, 1].astype(np.float64)
                thresholds = {f'{b:.2f}': threshold_for_false_alarms(scores, targets(val_frame), b) for b in config.budgets}
                write_json(directory/'thresholds.json', thresholds)
                results, score_checks = {}, 0
                # Reload frozen weights before computing every saved test result.
                reloaded = MatchedPool(tr[0].shape[-1])
                reloaded.load_state_dict(torch.load(directory/'model.pt', weights_only=True))
                for split in ('val', *TESTS):
                    frame = val_frame if split == 'val' else pd.read_parquet(seed_path/f'{split}.parquet')
                    x, lengths, ids, _ = pack(frame, processor, config.max_length)
                    prediction = probabilities(model, x, lengths)[:, 1].astype(np.float64)
                    replay = probabilities(reloaded, x, lengths)[:, 1].astype(np.float64)
                    np.testing.assert_array_equal(prediction, replay); score_checks += 1
                    results[split] = metrics(prediction, frame, thresholds)
                    np.savez_compressed(directory/f'{split}_predictions.npz', scores=prediction, y=targets(frame), session_ids=ids)
                record = {'seed': seed, 'regime': regime, 'input_dim': tr[0].shape[-1], 'results': results,
                          'parameter_count': sum(p.numel() for p in model.parameters()), 'histories': history,
                          'model_sha256': digest(directory/'model.pt'), 'processor_sha256': digest(directory/'preprocessor.pkl'),
                          'thresholds': thresholds, 'training_sha256': digest(seed_path/f'train_{regime}.parquet'),
                          'audit': {'status': 'passed', 'prediction_replays': score_checks, 'operating_points': score_checks*len(thresholds)}}
                write_json(directory/'results.json', record); records.append(record)
                write_json(output/'checkpoint.json', records)
    write_json(output/'report.json', {'protocol': protocol, 'runs': records})
    return records


def audit(study, output):
    report = json.loads((output/'report.json').read_text()); config = Config(**report['protocol']['config'])
    assert digest(Path(__file__)) == report['protocol']['implementation_sha256']
    assert digest(study/'report.json') == report['protocol']['source_report_sha256']
    torch.set_num_threads(config.threads)
    checks = 0
    for record in report['runs']:
        seed_path = study/f'seed-{record["seed"]}'
        directory = output/f'seed-{record["seed"]}'/f'relative_{record["regime"]}_pool_matched'
        assert record == json.loads((directory/'results.json').read_text())
        assert digest(directory/'model.pt') == record['model_sha256']
        assert digest(directory/'preprocessor.pkl') == record['processor_sha256']
        assert digest(seed_path/f'train_{record["regime"]}.parquet') == record['training_sha256']
        processor = pickle.loads((directory/'preprocessor.pkl').read_bytes())
        model = MatchedPool(record['input_dim'])
        model.load_state_dict(torch.load(directory/'model.pt', weights_only=True))
        for split in ('val', *TESTS):
            frame = pd.read_parquet(seed_path/f'{split}.parquet')
            x, lengths, ids, _ = pack(frame, processor, config.max_length)
            scores = probabilities(model, x, lengths)[:, 1].astype(np.float64)
            saved = np.load(directory/f'{split}_predictions.npz', allow_pickle=False)
            np.testing.assert_array_equal(scores, saved['scores'])
            np.testing.assert_array_equal(ids, saved['session_ids']); np.testing.assert_array_equal(targets(frame), saved['y'])
            assert metrics(scores, frame, record['thresholds']) == record['results'][split]
            if split == 'val':
                for budget, threshold in record['thresholds'].items():
                    assert threshold == threshold_for_false_alarms(scores, targets(frame), float(budget))
            checks += 1
    result = {'status': 'passed', 'prediction_checks': checks, 'operating_point_checks': checks*3}
    write_json(output/'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if not args.audit_only: run(args.study, args.output)
    print(json.dumps(audit(args.study, args.output)))
