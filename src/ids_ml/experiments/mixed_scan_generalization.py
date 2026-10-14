"""Predeclared factorial generalization study, isolated from public Sona."""
import argparse
import dataclasses
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from ids_ml.models.sequence import SequenceTrainConfig, train_sequence_model
from ids_ml.models.train import TrainConfig, train_multiclass_model
from ids_ml.models.evaluate import predict_class_proba
from .mixed_scan import summaries, train_pool, digest
from .scan_context import pack_context
from .scan_generalization_features import generate_partition, prepare, build_processor, CONDITIONS
from .temporal_patterns import threshold_for_false_alarms, score_metrics, sequence_scores, shuffled


@dataclasses.dataclass(frozen=True)
class Config:
    seeds: tuple[int,...]=(1103,1129,1151)
    train_windows: int=3200
    val_windows: int=3200
    test_windows: int=1600
    epochs: int=80
    patience: int=12
    min_epochs: int=20
    batch_size: int=128
    budgets: tuple[float,...]=(.01,.02,.05)
    threads: int=2
    max_length: int=160


MODEL_NAMES=('raw_sona','relative_sona','small_sona','small_pool','summary_boosting','temporal_rule')

def budget_key(budget):return f'{budget:.2f}'


def evaluate_scores(values,y,kinds,thresholds):
    results={}
    for model,scores in values.items():
        results[model]={}
        for budget,threshold in thresholds[model].items():
            m=score_metrics(scores,y,threshold)
            m['by_behavior']={kind:{'windows':int(np.sum(kinds==kind)),
                'flag_rate':float(np.mean(scores[kinds==kind]>=threshold))} for kind in sorted(set(kinds))}
            if model=='temporal_rule':
                for key in ('accuracy','macro_f1','default_confusion'):m.pop(key)
            results[model][budget]=m
    return results


def score_models(frame,prepared,packs,processors,models,boosters,seed,threads):
    groups=packs['small'][3]
    s,rule=summaries(frame,groups)
    with threadpool_limits(limits=threads):
        basic_score=boosters['summary_boosting'].predict_proba(s)[:,1]
    values={name:sequence_scores(models[name],*packs[mode][:2]) for name,mode in
        [('raw_sona','raw'),('relative_sona','relative'),('small_sona','small'),('small_pool','small')]}
    values.update(summary_boosting=basic_score,temporal_rule=rule)
    return values


def run_seed(config,seed,output):
    started=time.perf_counter();output.mkdir(parents=True,exist_ok=False)
    frames,prepared,hashes={},{},{}
    specs=[('train',config.train_windows,'mixture'),('val',config.val_windows,'mixture')] + [(name,config.test_windows,name) for name in CONDITIONS]
    for offset,(split,n,mode) in enumerate(specs):
        frames[split]=generate_partition(seed+offset*10000,n,f'generalization-{seed}-{split}',mode)
        frames[split].to_parquet(output/f'{split}.parquet',index=False)
        hashes[split]=digest(output/f'{split}.parquet')
        prepared[split]=prepare(frames[split])
    processors={mode:build_processor(frames['train'],mode,prepared['train']) for mode in ('raw','relative','small')}
    packs={split:{mode:pack_context(frame,processors[mode],config.max_length,prepared[split])
                  for mode in processors} for split,frame in frames.items()}
    models,histories={},{}
    seq_config=SequenceTrainConfig(seed=seed,max_epochs=config.epochs,min_epochs=config.min_epochs,
        patience=config.patience,batch_size=config.batch_size,class_weight=False)
    def progress(name):
        return lambda row:print(f'seed {seed} {name} epoch {row["epoch"]+1}: val loss {row["val_loss"]:.4f}',flush=True) if row['epoch']%10==0 else None
    print(f'seed {seed}: {sum(map(len,frames.values())):,} fresh flows; training',flush=True)
    for mode,name in [('raw','raw_sona'),('relative','relative_sona'),('small','small_sona')]:
        tr,va=packs['train'][mode],packs['val'][mode]
        models[name],histories[name]=train_sequence_model(*tr[:3],*va[:3],seq_config,2,on_epoch=progress(name))
        torch.save(models[name].state_dict(),output/f'{name}.pt')
        (output/'training_checkpoint.json').write_text(json.dumps(histories,indent=2))
    tr,va=packs['train']['small'],packs['val']['small']
    models['small_pool'],histories['small_pool']=train_pool(*tr[:3],*va[:3],config,seed)
    s,_=summaries(frames['train'],tr[3])
    model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=1.,random_state=seed)
    with threadpool_limits(limits=config.threads):model.fit(s,tr[2])
    boosters={'summary_boosting':model}
    val_scores=score_models(frames['val'],prepared['val'],packs['val'],processors,models,boosters,seed,config.threads)
    vy=packs['val']['small'][2]
    thresholds={name:{budget_key(b):threshold_for_false_alarms(scores,vy,b) for b in config.budgets}
                for name,scores in val_scores.items()}
    results={}
    for split in ('val',*CONDITIONS):
        values=val_scores if split=='val' else score_models(frames[split],prepared[split],packs[split],processors,models,boosters,seed,config.threads)
        y=packs[split]['small'][2]
        kinds=frames[split].groupby('session_id',sort=False).behavior.first().to_numpy()
        results[split]=evaluate_scores(values,y,kinds,thresholds)
        np.savez_compressed(output/f'{split}_predictions.npz',y=y,
            window_ids=frames[split].groupby('session_id',sort=False).session_id.first().to_numpy(dtype=str),**values)
    for name,model in models.items():torch.save(model.state_dict(),output/f'{name}.pt')
    for name,processor in processors.items():(output/f'{name}_preprocessor.pkl').write_bytes(pickle.dumps(processor))
    for name,model in boosters.items():(output/f'{name}.pkl').write_bytes(pickle.dumps(model))
    record={'seed':seed,'config':dataclasses.asdict(config),'results':results,'thresholds':thresholds,
        'histories':histories,'source_sha256':hashes,'flow_counts':{k:len(v) for k,v in frames.items()},
        'dimensions':{k:packs['train'][k][0].shape[-1] for k in processors},
        'feature_names':{k:p.get_feature_names_out().tolist() for k,p in processors.items()},
        'parameter_counts':{k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
        'elapsed_seconds':time.perf_counter()-started}
    (output/'results.json').write_text(json.dumps(record,indent=2))
    print(f'seed {seed}: finished all candidates and budgets',flush=True)
    return record


def aggregate(records):
    result={}
    for split in CONDITIONS:
        result[split]={}
        for model in MODEL_NAMES:
            result[split][model]={}
            for budget in records[0]['results'][split][model]:
                result[split][model][budget]={}
                for metric in ('recall','false_positive_rate'):
                    values=[r['results'][split][model][budget]['at_validation_false_alarm_target'][metric] for r in records]
                    result[split][model][budget][metric]={'mean':float(np.mean(values)),'min':float(min(values)),'max':float(max(values))}
    return result


def run(config,output):
    if len(set(config.seeds))!=len(config.seeds):raise ValueError('Seeds must be unique.')
    if set(config.seeds)&{811,827,853,941,967,983}:raise ValueError('Use fresh seeds, not the inspected v1 seeds.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'protocol.json').write_text(json.dumps(dataclasses.asdict(config),indent=2))
    source=Path(__file__).parent
    files=['mixed_scan_generalization.py','scan_generalization_features.py','audit_scan_generalization.py','scan_context.py','mixed_scan.py']
    (output/'implementation_sha256.json').write_text(json.dumps({n:digest(source/n) for n in files},indent=2))
    torch.set_num_threads(config.threads)
    records=[]
    for seed in config.seeds:
        records.append(run_seed(config,seed,output/f'seed-{seed}'))
        (output/'checkpoint.json').write_text(json.dumps(records,indent=2))
    report={'benchmark_version':'generalization-1.0.0','config':dataclasses.asdict(config),'summary':aggregate(records),'runs':records,
        'limitations':['Binary fresh architecture candidates, not public six-class checkpoints.',
            'Same generator family, fresh environments/seeds; no captured traffic.',
            'Small pooling has engineered causal history; it is not history-free.',
            'Whole-window classification does not establish online alert timing.',
            'Equal validation budgets need not give equal realized test false alarms.',
            'Generator intent can remain indistinguishable from legitimate inventory.']}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(Config().seeds))
    p.add_argument('--train-windows',type=int,default=3200);p.add_argument('--val-windows',type=int,default=3200)
    p.add_argument('--test-windows',type=int,default=1600);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--min-epochs',type=int,default=20)
    a=p.parse_args()
    run(Config(seeds=tuple(a.seeds),train_windows=a.train_windows,val_windows=a.val_windows,
               test_windows=a.test_windows,epochs=a.epochs,min_epochs=a.min_epochs),a.out_dir)


if __name__=='__main__':main()
