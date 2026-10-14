"""Prospective mixed-traffic gradual-scan study; no public checkpoint changes.

Independent behavioral generation, not reordered matched pairs. Complete host
windows have binary targets; event provenance never enters training features.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn
from threadpoolctl import threadpool_limits

from ids_ml.data.schema import FEATURE_COLUMNS, SCHEMA_VERSION, validate
from ids_ml.features import build_preprocessor, transform_features
from ids_ml.models.classifier import MLPClassifier
from ids_ml.models.sequence import SequenceClassifier, SequenceTrainConfig, train_sequence_model
from ids_ml.models.train import TrainConfig, train_multiclass_model
from .temporal_patterns import threshold_for_false_alarms, score_metrics, sequence_scores, shuffled


@dataclasses.dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (811, 827, 853)
    train_windows: int = 2400
    val_windows: int = 1600
    test_windows: int = 2400
    epochs: int = 80
    patience: int = 12
    min_epochs: int = 20
    batch_size: int = 128
    false_alarm_target: float = .01
    threads: int = 2
    max_length: int = 160


COMMON_PORTS = np.array([22, 25, 53, 80, 110, 135, 139, 143, 443, 445, 587, 993,
                         995, 1433, 1521, 3306, 3389, 5432, 5900, 6379, 8080, 8443])
SUMMARY_NAMES = ('flows', 'failed_fraction', 'probe_fraction', 'unique_ports',
                 'unique_failed_ports', 'probe_span_fraction', 'probe_gap_median',
                 'probe_gap_cv', 'max_300s_probes', 'mean_300s_probes',
                 'occupied_300s_bins', 'mean_duration', 'mean_packets', 'mean_bytes')


def generate_windows(seed, count, prefix, *, mode='usual'):
    """Disjoint, fixed-duration windows. Scan and benign traces sampled independently.

    Every window has background plus short probe-like connections. Labels are
    generator intent, and legitimate inventory can overlap attack behavior.
    """
    if count < 4 or count % 2 or mode not in ('usual', 'shifted', 'lookalikes'):
        raise ValueError('Use an even window count >=4 and a known mode.')
    rng = np.random.default_rng(seed)
    targets = rng.permutation(np.tile([0, 1], count // 2))
    records = []
    duration = 1800. if mode == 'usual' else 3600.
    for index, target in enumerate(targets):
        environment = index // 200
        host = f'{prefix}-env-{environment}-host-{index % 40}'
        window = f'{prefix}-window-{index:06d}'
        base_time = (index // 40) * duration
        kind = 'gradual_scan' if target else str(rng.choice(
            ['ordinary', 'retries', 'monitoring', 'discovery', 'inventory'],
            p=[.25, .25, .25, .20, .05] if mode != 'lookalikes' else [.15, .20, .20, .20, .25]))
        n_background = int(rng.integers(48, 85) if mode == 'usual' else rng.integers(64, 117))
        n_probes = int(rng.integers(8, 29))  # Same count distribution for every target.
        if kind == 'gradual_scan':
            start = rng.uniform(0, duration * .15)
            span = rng.uniform(duration * .55, duration * .85)
            gaps = rng.lognormal(0, .7, n_probes)
            probe_times = start + span * np.cumsum(gaps) / gaps.sum()
            # A scan can revisit ports; service discovery can cover many too.
            unique_count = int(rng.integers(6, min(n_probes, 20) + 1))
            pool = rng.choice(COMMON_PORTS, unique_count, replace=False)
            probe_ports = rng.choice(pool, n_probes, replace=True)
        elif kind in ('discovery', 'inventory'):
            start = rng.uniform(0, duration * .2)
            span = rng.uniform(100, 500) if kind == 'discovery' else rng.uniform(duration * .45, duration * .9)
            probe_times = np.sort(start + rng.uniform(0, span, n_probes))
            pool = rng.choice(COMMON_PORTS, int(rng.integers(6, 21)), replace=False)
            probe_ports = rng.choice(pool, n_probes, replace=True)
        elif kind == 'monitoring':
            probe_times = np.clip(np.linspace(rng.uniform(0, 80), duration - 80, n_probes)
                                  + rng.normal(0, 35, n_probes), 0, duration - 1)
            probe_ports = rng.choice(rng.choice(COMMON_PORTS, int(rng.integers(2, 6)), replace=False), n_probes)
        elif kind == 'retries':
            centers = rng.uniform(0, duration - 100, int(rng.integers(1, 4)))
            probe_times = np.clip(rng.choice(centers, n_probes) + rng.exponential(25, n_probes), 0, duration - 1)
            probe_ports = rng.choice(rng.choice(COMMON_PORTS, int(rng.integers(1, 4)), replace=False), n_probes)
        else:
            probe_times = rng.uniform(0, duration, n_probes)
            probe_ports = rng.choice(COMMON_PORTS, n_probes)
        background_times = rng.uniform(0, duration, n_background)
        # Environment-specific background mix, independently of target.
        env_rng = np.random.default_rng(seed + environment * 919)
        service_p = env_rng.dirichlet([8, 3, 2, 1])
        for j in range(n_background + n_probes):
            probe = j >= n_background
            event_t = probe_times[j - n_background] if probe else background_times[j]
            port = int(probe_ports[j - n_background]) if probe else int(rng.choice([443, 80, 53, 22], p=service_p))
            service = {443:'https', 80:'http', 53:'dns', 22:'ssh', 3389:'rdp'}.get(port, 'other')
            if probe:
                outcome = str(rng.choice(['rejected','timeout','established','reset'], p=[.48,.22,.20,.10]))
                packets = int(rng.integers(1, 6))
                flow_duration = float(rng.lognormal(-1.3, .8))
            else:
                outcome = str(rng.choice(['established','timeout','reset','rejected'], p=[.87,.05,.05,.03]))
                packets = max(1, int(rng.lognormal(2., 1.)))
                flow_duration = float(rng.lognormal(.6, 1.1))
            received = 0 if outcome in ('timeout','rejected') else max(1, int(packets * rng.uniform(.4, 1.2)))
            actual_scan = bool(target and probe)
            records.append(dict(schema_version=SCHEMA_VERSION,run_id=prefix,seed=seed,
                flow_id=f'{window}-flow-{j:03d}',session_id=window,event_time=base_time+float(event_t),
                src_host_id=host,dst_host_id=f'{prefix}-service-{port}',
                src_ip=f'10.{environment % 250}.{index % 40}.1',dst_ip=f'192.0.2.{port % 250 + 1}',
                duration_s=flow_duration,src_port=int(rng.integers(20000,65535)),dst_port=port,
                protocol='udp' if service=='dns' and not probe else 'tcp',service=service,
                packets_sent=packets,packets_recv=received,
                bytes_sent=packets*int(rng.integers(60,900)),bytes_recv=received*int(rng.integers(60,1100)),
                outcome=outcome,prior_gap_s=0.,scenario='port_scan' if actual_scan else 'benign',
                label='suspicious' if actual_scan else 'normal',window_target=int(target),
                behavior=kind,observation_duration_s=duration))
    frame = pd.DataFrame(records).sort_values(['session_id','event_time'], kind='stable').reset_index(drop=True)
    frame['prior_gap_s'] = frame.groupby('session_id').event_time.diff().fillna(0.)
    validate(frame)
    return frame


def pack_windows(frame, processor, cap):
    features = transform_features(processor, frame)
    groups = list(frame.groupby('session_id', sort=False).indices.values())
    if max(map(len, groups)) > cap:
        raise ValueError('Window exceeds predeclared cap; do not silently truncate.')
    X = np.zeros((len(groups), cap, features.shape[1]), dtype=np.float32)
    lengths = np.array(list(map(len, groups)), dtype=np.int64)
    y = np.array([frame.window_target.iloc[g[0]] for g in groups], dtype=np.int64)
    for i, g in enumerate(groups):
        if frame.window_target.iloc[g].nunique() != 1:
            raise ValueError('Inconsistent window targets.')
        X[i, :len(g)] = features[g]
    return X, lengths, y, groups


def summaries(frame, groups):
    """Only permitted flow fields and event-time differences, no label/identity inputs."""
    output, temporal = [], []
    for g in groups:
        w = frame.iloc[g]
        time_s = w.event_time.to_numpy() - w.event_time.iloc[0]
        failed = w.outcome.isin(['rejected','timeout']).to_numpy()
        probe = failed & (w.duration_s.to_numpy() < 1.5) & (w.packets_sent.to_numpy() <= 5)
        times = time_s[probe]
        span = max(float(time_s[-1]), 1.)
        counts = np.histogram(times, bins=np.arange(0, span + 300, 300))[0]
        gaps = np.diff(times)
        unique = w.dst_port[probe].nunique()
        occupied = int(np.count_nonzero(counts))
        output.append([len(w),failed.mean(),probe.mean(),w.dst_port.nunique(),unique,
                       np.ptp(times)/span if len(times)>1 else 0.,np.median(gaps) if len(gaps) else 0.,
                       np.std(gaps)/max(np.mean(gaps),1e-6) if len(gaps) else 0.,
                       counts.max(initial=0),counts.mean() if len(counts) else 0.,occupied,
                       w.duration_s.mean(),w.packets_sent.mean(),w.bytes_sent.mean()])
        # A legitimate temporal baseline: diverse failed attempts sustained
        # across time bins, rather than only a short burst.
        temporal.append(float(unique * occupied / max(len(counts), 1)))
    return np.asarray(output, dtype=np.float32), np.asarray(temporal, dtype=np.float64)


class PooledMLP(nn.Module):
    """Same small connection network, trained through order-independent mean pooling."""
    def __init__(self, dimension):
        super().__init__()
        self.flow = MLPClassifier(dimension, output_dim=2)

    def forward(self, x, lengths):
        logits = self.flow(x)
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        return (logits * mask[..., None]).sum(dim=1) / lengths[:, None]


def train_pool(X, lengths, y, V, vl, vy, config, seed):
    torch.manual_seed(seed)
    model = PooledMLP(X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    train = tuple(torch.as_tensor(a) for a in (X, lengths, y))
    val = tuple(torch.as_tensor(a) for a in (V, vl, vy))
    best, loss_best, stale, history = copy.deepcopy(model.state_dict()), float('inf'), 0, []
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(config.epochs):
        model.train()
        for ids in torch.randperm(len(X), generator=gen).split(config.batch_size):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(train[0][ids], train[1][ids]), train[2][ids])
            loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.cross_entropy(model(val[0], val[1]), val[2]))
        history.append({'epoch':epoch,'val_loss':val_loss})
        if val_loss < loss_best:
            loss_best, best, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        elif epoch + 1 >= config.min_epochs:
            stale += 1
            if stale >= config.patience:break
    model.load_state_dict(best)
    return model, history


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_seed(config, seed, output):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    frames, packs, hashes = {}, {}, {}
    split_specs = [('train',config.train_windows,'usual'),('val',config.val_windows,'usual'),
                   ('test',config.test_windows,'usual'),('shifted',config.test_windows,'shifted'),
                   ('lookalikes',config.test_windows,'lookalikes')]
    for offset,(name,n,mode) in enumerate(split_specs):
        frames[name] = generate_windows(seed+offset*10000,n,f'seed-{seed}-{name}',mode=mode)
        frames[name].to_parquet(output/f'{name}.parquet',index=False)
        hashes[name] = digest(output/f'{name}.parquet')
    processor = build_preprocessor(frames['train'])
    for name,frame in frames.items():packs[name] = pack_windows(frame,processor,config.max_length)
    X, lengths, y, groups = packs['train'];V, vl, vy, vg = packs['val']
    print(f'seed {seed}: {sum(map(len,frames.values())):,} flows, training',flush=True)
    histories, models = {}, {}
    # Flow labels deliberately inherit the same window target given to all
    # learners. Event-level generator truth is retained for audit only.
    beta, histories['beta2_vote'] = train_multiclass_model(
        transform_features(processor,frames['train']),frames['train'].window_target.to_numpy(),
        transform_features(processor,frames['val']),frames['val'].window_target.to_numpy(),
        TrainConfig(seed=seed,max_epochs=config.epochs,patience=config.patience,batch_size=1024),2)
    models['beta2_vote'] = beta
    seq_config = SequenceTrainConfig(seed=seed,max_epochs=config.epochs,patience=config.patience,
        min_epochs=config.min_epochs,batch_size=config.batch_size,class_weight=False)
    def progress(name):
        return lambda row: print(f'seed {seed} {name} epoch {row["epoch"]+1}: val loss {row["val_loss"]:.4f}',flush=True) if row['epoch']%10==0 else None
    models['sona'], histories['sona'] = train_sequence_model(X,lengths,y,V,vl,vy,seq_config,2,on_epoch=progress('Sona'))
    models['pooled_mlp'], histories['pooled_mlp'] = train_pool(X,lengths,y,V,vl,vy,config,seed)
    models['shuffled_train'], histories['shuffled_train'] = train_sequence_model(
        shuffled(X,lengths,seed+50000),lengths,y,shuffled(V,vl,seed+60000),vl,vy,
        seq_config,2,on_epoch=progress('shuffled control'))
    S,_ = summaries(frames['train'],groups)
    summary_model = HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=1.,random_state=seed)
    with threadpool_limits(limits=config.threads):
        summary_model.fit(S,y)
    def scores(name):
        xx,ll,yy,gg = packs[name];frame=frames[name]
        from ids_ml.models.evaluate import predict_class_proba
        flow = predict_class_proba(beta,transform_features(processor,frame))[:,1]
        mean = np.array([flow[g].mean(dtype=np.float64) for g in gg])
        vote = np.array([np.mean(flow[g]>=.5) for g in gg])
        s,rule = summaries(frame,gg)
        with threadpool_limits(limits=config.threads):
            summary_scores=summary_model.predict_proba(s)[:,1]
        return {'beta2_vote':vote,'beta2_mean':mean,'pooled_mlp':sequence_scores(models['pooled_mlp'],xx,ll),
                'sona':sequence_scores(models['sona'],xx,ll),
                'sona_shuffled_test':sequence_scores(models['sona'],shuffled(xx,ll,seed+70000),ll),
                'shuffled_train':sequence_scores(models['shuffled_train'],shuffled(xx,ll,seed+80000),ll),
                'summary_boosting':summary_scores,'temporal_rule':rule}
    val_scores = scores('val')
    thresholds = {name:threshold_for_false_alarms(value,vy,config.false_alarm_target) for name,value in val_scores.items()}
    results, predictions = {}, {}
    for name in ('val','test','shifted','lookalikes'):
        yy = packs[name][2];values = val_scores if name=='val' else scores(name)
        predictions[name] = values
        results[name] = {}
        kinds = frames[name].groupby('session_id',sort=False).behavior.first().to_numpy()
        for model,value in values.items():
            result = score_metrics(value,yy,thresholds[model])
            result['by_behavior'] = {kind:{'windows':int(np.sum(kinds==kind)),
                'flag_rate':float(np.mean(value[kinds==kind]>=thresholds[model]))} for kind in sorted(set(kinds))}
            # Rule scores are not probabilities; .5 default metrics have no
            # comparable meaning. Only validation-frozen metrics/AUC are used.
            if model=='temporal_rule':
                for key in ('accuracy','macro_f1','default_confusion'):result.pop(key)
            results[name][model] = result
        np.savez_compressed(output/f'{name}_predictions.npz',y=yy,
            window_ids=frames[name].groupby('session_id',sort=False).session_id.first().to_numpy(dtype=str),**values)
    for name,model in models.items():torch.save(model.state_dict(),output/f'{name}.pt')
    (output/'preprocessor.pkl').write_bytes(pickle.dumps(processor))
    (output/'summary_model.pkl').write_bytes(pickle.dumps(summary_model))
    # Reload checks use validation; all held-out arrays and source hashes saved.
    for name,model in models.items():
        original = copy.deepcopy(model.state_dict())
        model.load_state_dict(torch.load(output/f'{name}.pt',weights_only=True))
        for key in original:torch.testing.assert_close(original[key],model.state_dict()[key],rtol=0,atol=0)
    reloaded = pickle.loads((output/'preprocessor.pkl').read_bytes())
    np.testing.assert_array_equal(transform_features(reloaded,frames['val']),transform_features(processor,frames['val']))
    reloaded_summary=pickle.loads((output/'summary_model.pkl').read_bytes())
    with threadpool_limits(limits=config.threads):
        np.testing.assert_array_equal(summary_model.predict_proba(summaries(frames['val'],vg)[0]),
                                      reloaded_summary.predict_proba(summaries(frames['val'],vg)[0]))
    for i,(name,frame) in enumerate(frames.items()):
        assert digest(output/f'{name}.parquet')==hashes[name]
        for other in list(frames)[:i]:
            for key in ('flow_id','session_id','src_host_id'):
                assert not set(frame[key]) & set(frames[other][key])
    record={'seed':seed,'config':dataclasses.asdict(config),'results':results,'thresholds':thresholds,
        'histories':histories,'flow_counts':{k:len(v) for k,v in frames.items()},
        'feature_columns':list(FEATURE_COLUMNS),'encoded_features':X.shape[-1],
        'summary_columns':list(SUMMARY_NAMES),'source_sha256':hashes,
        'audit':{'status':'passed','identities_disjoint':True,'weights_reload_exact':True,
                 'preprocessor_reload_exact':True,'summary_reload_exact':True},
        'elapsed_seconds':time.perf_counter()-started}
    (output/'results.json').write_text(json.dumps(record,indent=2))
    print(f'seed {seed}: done; test recall/FPR '+str({n:r['at_validation_false_alarm_target'] for n,r in results['test'].items()}),flush=True)
    return record


def run(config, output):
    if len(set(config.seeds)) != len(config.seeds):raise ValueError('Seeds must be unique.')
    output.mkdir(parents=True,exist_ok=False)
    # Freeze the recipe before generation/training/test evaluation.
    (output/'protocol.json').write_text(json.dumps(dataclasses.asdict(config),indent=2))
    source_files=[Path(__file__),Path(__file__).parent/'audit_mixed_scan.py']
    (output/'implementation_sha256.json').write_text(json.dumps({p.name:digest(p) for p in source_files},indent=2))
    torch.set_num_threads(config.threads)
    records=[]
    for seed in config.seeds:
        records.append(run_seed(config,seed,output/f'seed-{seed}'))
        (output/'checkpoint.json').write_text(json.dumps(records,indent=2))
    summary={}
    for split in ('test','shifted','lookalikes'):
        summary[split]={}
        for model in records[0]['results'][split]:
            summary[split][model]={}
            for key in ('recall','false_positive_rate'):
                values=[r['results'][split][model]['at_validation_false_alarm_target'][key] for r in records]
                summary[split][model][key]={'mean':float(np.mean(values)),'min':float(min(values)),'max':float(max(values))}
    report={'benchmark_version':'1.0.0','config':dataclasses.asdict(config),'summary':summary,'runs':records,
        'limitations':['Synthetic generator-intent labels do not prove malicious intent.',
            'Binary retrained architecture candidates, not public six-class checkpoints.',
            'Complete non-overlapping host windows, not online detection or latency measurement.',
            'Same window targets for every learner; flow model uses weak repeated window labels.',
            'Summary features are engineered from allowed fields and relative time; no generator truth.',
            'Validation thresholds freeze a 1% budget; unseen test false alarms need not remain 1%.',
            'A legitimate inventory can be indistinguishable from scanning; all results retained.']}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(Config().seeds))
    parser.add_argument('--train-windows',type=int,default=2400)
    parser.add_argument('--val-windows',type=int,default=1600)
    parser.add_argument('--test-windows',type=int,default=2400)
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--min-epochs',type=int,default=20)
    args=parser.parse_args()
    run(Config(seeds=tuple(args.seeds),train_windows=args.train_windows,val_windows=args.val_windows,
               test_windows=args.test_windows,epochs=args.epochs,min_epochs=args.min_epochs),args.out_dir)


if __name__=='__main__':main()
