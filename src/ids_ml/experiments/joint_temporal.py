"""Six-class candidate mixing ordinary/stealth telemetry with paired timing examples.

Preserves class meanings and public artifacts; this is an independent candidate.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ids_ml.data.generator import GenerationConfig, generate_dataset, split_dataset, combine_runs
from ids_ml.data.schema import ALL_COLUMNS, SCHEMA_VERSION, coerce_dtypes, validate
from ids_ml.features import (SCENARIO_CLASSES, build_preprocessor, transform_features,
                             extract_scenario_labels, save_preprocessor)
from ids_ml.models.train import TrainConfig, train_multiclass_model, save_artifact
from ids_ml.models.sequence import SequenceTrainConfig, train_sequence_model, SequenceClassifier, predict_sequence_proba
from ids_ml.models.sessions import build_session_sequences
from ids_ml.models.evaluate import predict_class_proba, multiclass_metrics_from_predictions
from .temporal_patterns import generate_pairs, shuffled
from .temporal_robustness import polling


def schema_frame(df, seed, prefix):
    out=df.copy().reset_index(drop=True)
    out['scenario']=np.where(out.target==1,'beaconing','benign')
    out['label']=np.where(out.target==1,'suspicious','normal')
    out['schema_version']=SCHEMA_VERSION
    out['run_id']=prefix
    out['seed']=seed
    out['src_host_id']=out.pair_id.map(lambda x:'host-'+x)
    out['dst_host_id']='destination-'+prefix
    pairs={name:i for i,name in enumerate(out.pair_id.unique())}
    out['src_ip']=out.pair_id.map(lambda x:f'10.240.{pairs[x]//254}.{pairs[x]%254+1}')
    out['dst_ip']='203.0.113.80'
    out=coerce_dtypes(out)
    validate(out)
    return out


def sequence_probs(model,X,lengths):
    return np.concatenate([predict_sequence_proba(model,X[i:i+128],lengths[i:i+128])
                           for i in range(0,len(X),128)])


def diverse_polling(seed, sessions, prefix):
    parts=[]
    for idx,pace in enumerate([(5,25),(30,120),(180,600)]):
        count=sessions//3 + (idx < sessions%3)
        if count:
            parts.append(polling(seed+idx*500,count,f'{prefix}-pace-{idx}',pace))
    return pd.concat(parts,ignore_index=True)


def run(output, seed=401, max_epochs=100, class_weight_power=1.0, pace_augmentation=False,
        timing_mode='hybrid', noise_augmentation=False):
    if max_epochs < 60 or not 0 <= class_weight_power <= 1:
        raise ValueError('Use epochs >=60 and class_weight_power in [0,1].')
    if timing_mode not in ('hybrid','hybrid_log','relative'):
        raise ValueError('Joint timing_mode must be hybrid, hybrid_log, or relative.')
    if output.exists():raise ValueError('Use a new candidate directory.')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    started=time.perf_counter()
    config=GenerationConfig(seed=seed,num_hosts=120,duration_minutes=180,
        port_scan_episodes=40,auth_bruteforce_episodes=40,beaconing_episodes=40,
        service_flood_episodes=40,outbound_transfer_episodes=40,realistic_background=True)
    ordinary={}
    manifests=[]
    for mode,delta in [('normal',0),('stealth',1000)]:
        df,manifest=generate_dataset(dataclasses.replace(config,seed=seed+delta,stealth_mode=bool(delta)))
        manifests.append(manifest)
        ordinary[mode]=split_dataset(df,val_frac=.15,test_frac=.15,seed=seed+delta)
    recipe={'blocks':[1,2,3,4,5,6],'contrast':(1.1,3.5),'jitter':.12}
    if pace_augmentation:
        recipe['pace']=(5,120)
    datasets={}
    for idx,(name,count) in enumerate([('train',1200),('val',400)]):
        if noise_augmentation:
            parts=[]
            for group,noise in enumerate((.08,.16,.25,.35)):
                group_count=count//4 + (group < count%4)
                part,_=generate_pairs(seed+10000*(idx+2)+group*1000,group_count,
                    f'{seed}-{name}-pairs-noise-{group}',**{**recipe,'jitter':noise})
                parts.append(part)
            pairs=pd.concat(parts,ignore_index=True)
        else:
            pairs,_=generate_pairs(seed+10000*(idx+2),count,f'{seed}-{name}-pairs',**recipe)
        make_polling=diverse_polling if pace_augmentation else polling
        benign=make_polling(seed+10000*(idx+4),300 if idx==0 else 100,f'{seed}-{name}-polling')
        temporal=schema_frame(pd.concat([pairs,benign],ignore_index=True),seed,f'joint-{name}')
        base=combine_runs(ordinary['normal'][idx],ordinary['stealth'][idx])
        datasets[name]=coerce_dtypes(pd.concat([base,temporal],ignore_index=True))
        validate(datasets[name])
    datasets['normal']=ordinary['normal'][2]
    datasets['stealth']=ordinary['stealth'][2]
    for idx,(name,kwargs) in enumerate([('timing',{}),('slow_timing',{'shifted':True}),
        ('known_long_cycles',{'blocks':[5]}),('unseen_long_cycles',{'blocks':[7]}),
        ('subtle_timing',{'contrast':(1.15,1.4)})],start=6):
        df,_=generate_pairs(seed+idx*10000,600,f'{seed}-{name}',**kwargs)
        datasets[name]=schema_frame(df,seed,name)
    if pace_augmentation:
        df,_=generate_pairs(seed+140000,600,f'{seed}-extreme-pace',pace=(150,300))
        datasets['extreme_pace']=schema_frame(df,seed,'extreme-pace')
    datasets['polling']=schema_frame(polling(seed+120000,300,f'{seed}-polling-slow',(30,70)),seed,'polling')
    for name,df in datasets.items():
        df.to_parquet(output/f'{name}.parquet',index=False)
    # Both architectures receive the same absolute plus closed-session-relative inputs.
    preprocessor=build_preprocessor(datasets['train'],timing_mode=timing_mode)
    classes=list(SCENARIO_CLASSES)
    cap=max(int(df.groupby('session_id').size().max()) for df in datasets.values())
    packs={name:build_session_sequences(df,preprocessor,classes,max_len=cap) for name,df in datasets.items()}
    tx,tl,ty,_=packs['train'];vx,vl,vy,_=packs['val']
    print(f'joint seed {seed}: {sum(len(df) for df in datasets.values()):,} flows, training six-class Beta2',flush=True)
    beta,bh=train_multiclass_model(transform_features(preprocessor,datasets['train']),extract_scenario_labels(datasets['train']),
        transform_features(preprocessor,datasets['val']),extract_scenario_labels(datasets['val']),
        TrainConfig(seed=seed,max_epochs=80,patience=10,batch_size=512),len(classes))
    cfg=SequenceTrainConfig(seed=seed,max_epochs=max_epochs,min_epochs=60,patience=20,batch_size=128,
                            class_weight_power=class_weight_power)
    def progress(row):
        if row['epoch']%10==0:print(f'joint Sona epoch {row["epoch"]+1}, validation F1 {row["val_f1"]:.3f}',flush=True)
    sona,sh=train_sequence_model(tx,tl,ty,vx,vl,vy,cfg,len(classes),on_epoch=progress)
    results={}
    for split in list(datasets)[2:]:
        df=datasets[split];X,lengths,y,ids=packs[split]
        flowprobs=predict_class_proba(beta,transform_features(preprocessor,df))
        indexed=df.groupby('session_id',sort=False).indices
        beta_preds=np.array([np.bincount(flowprobs[indexed[sid]].argmax(axis=1),minlength=len(classes)).argmax() for sid in ids])
        probs=sequence_probs(sona,X,lengths)
        guesses=probs.argmax(axis=1)
        metrics={'beta2':multiclass_metrics_from_predictions(beta_preds,y,classes),
                 'sona_hybrid':multiclass_metrics_from_predictions(guesses,y,classes)}
        if split not in ('normal','stealth','polling'):
            mixed=sequence_probs(sona,shuffled(X,lengths,seed+130000),lengths).argmax(axis=1)
            metrics['sona_shuffled']=multiclass_metrics_from_predictions(mixed,y,classes)
        # Conditional timing splits only contain two true classes: use accuracy there,
        # not six-class macro F1 (absent classes would otherwise contribute zeros).
        results[split]=metrics
        np.savez_compressed(output/f'{split}_predictions.npz',y=y,session_ids=np.array(ids),
                            beta2=beta_preds,sona=guesses,sona_probs=probs)
        print(f'joint {split}: accuracy Beta2={metrics["beta2"]["accuracy"]:.3f}, Sona={metrics["sona_hybrid"]["accuracy"]:.3f}',flush=True)
    save_artifact(beta,preprocessor,TrainConfig(seed=seed,max_epochs=80,patience=10,batch_size=512),results['normal']['beta2'],
                  output/'beta2',input_dim=tx.shape[-1],output_dim=len(classes),class_names=classes)
    torch.save(sona.state_dict(),output/'sona.pt')
    save_preprocessor(preprocessor,output/'sona_preprocessor.pkl')
    metadata={'architecture':'GRU (session-level)','model_version':'Sona temporal candidate',
        'schema_version':SCHEMA_VERSION,'class_names':classes,'input_dim':tx.shape[-1],
        'hidden_dim':cfg.hidden_dim,'max_seq_len':cap,'timing_mode':timing_mode,
        'feature_names':preprocessor.get_feature_names_out().tolist(),'train_config':dataclasses.asdict(cfg)}
    (output/'sona_metadata.json').write_text(json.dumps(metadata,indent=2))
    report={'seed':seed,'training_recipe':recipe,'pace_augmentation':pace_augmentation,
        'noise_augmentation':noise_augmentation,
        'training_jitters':[.08,.16,.25,.35] if noise_augmentation else [recipe['jitter']],
        'polling_training_paces':[(5,25),(30,120),(180,600)] if pace_augmentation else [(5,25)],
        'generation_manifests':manifests,'metadata':metadata,
        'counts':{name:{'flows':len(df),'sessions':df.session_id.nunique()} for name,df in datasets.items()},
        'histories':{'beta2':bh,'sona':sh},'results':results,'elapsed_seconds':time.perf_counter()-started,
        'limitations':['One fresh six-class candidate run, not a multi-seed deployment validation.',
            'Timing pairs are synthetic order-discrimination examples, not proof of attack intent.',
            'Conditional timing splits contain only benign/beaconing; report accuracy there.',
            'Hybrid uses completed session context. No streaming guarantee.']}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=401)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--class-weight-power',type=float,default=1.)
    parser.add_argument('--pace-augmentation',action='store_true')
    parser.add_argument('--timing-mode',choices=['hybrid','hybrid_log','relative'],default='hybrid')
    parser.add_argument('--noise-augmentation',action='store_true')
    args=parser.parse_args()
    run(args.out_dir,args.seed,args.epochs,args.class_weight_power,args.pace_augmentation,
        args.timing_mode,args.noise_augmentation)
