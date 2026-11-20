"""Bounded validation-only overnight stages; no calibration/test evaluation."""
import argparse
import json
import time
import numpy as np
import pandas as pd
from src import multiscenario_pilot as base

ROOT=base.ROOT
PLAN=[
    dict(id=4,title='Stage 4 - uncertainty',directory='results/stage4_uncertainty',models=['analysis'],epochs=1,unit='analysis passes'),
    dict(id=5,title='Stage 5 - P3 seed 7',directory='results/stage5_seed7',models=['P3'],epochs=6,unit='epochs'),
    dict(id=6,title='Stage 6 - P3 seed 13',directory='results/stage6_seed13',models=['P3'],epochs=6,unit='epochs'),
    dict(id=7,title='Stage 7 - comparison',directory='results/stage7_comparison',models=['analysis'],epochs=1,unit='analysis passes'),
]


def uncertainty():
    source=ROOT/'results/validation_diagnostics_v1'
    manifest=pd.read_json(source/'split_manifest.jsonl',lines=True)
    val=manifest[manifest.split=='val']
    blocks=val.block_key.to_numpy()
    rng=np.random.default_rng(20261109)
    hits={}
    for model in ('B1','P1','P3'):
        with np.load(source/f'{model}_predictions.npz') as a:
            if len(a['labels'])!=len(val) or not np.array_equal(a['scenarios'],val.scenario.to_numpy()):
                raise ValueError('Prediction ordering does not match manifest')
            if not np.array_equal(a['labels'],val.target_beam.to_numpy()): raise ValueError('Label alignment failed')
            hits[model]=(a['logits'].argmax(1)==a['labels']).astype(float)
    groups=[np.unique(blocks[val.scenario.to_numpy()==s]) for s in (36,37,38,39)]
    keys=np.concatenate(groups)
    counts={b:int((blocks==b).sum()) for b in keys}
    correct={m:{b:float(hits[m][blocks==b].sum()) for b in keys} for m in hits}
    draws={m:[] for m in hits}
    for i in range(2000):
        sampled=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        denominator=sum(counts[b] for b in sampled)
        for m in hits: draws[m].append(sum(correct[m][b] for b in sampled)/denominator)
        if i%100==0: base.status(status='analyzing',model='analysis',epoch=1,batch=i+1,batches=2000)
    report={'scope':'Validation-only, scenario-stratified trajectory-block bootstrap; conditional on selected checkpoints. Not independent test inference.',
            'replicates':2000,'blocks_by_scenario':[len(g) for g in groups],
            'models':{m:{'top1':float(hits[m].mean()),'percentile_95':np.quantile(draws[m],[.025,.975]).tolist()} for m in hits},
            'paired_top1_differences':{f'P3_minus_{m}':np.quantile(np.array(draws['P3'])-draws[m],[.025,.5,.975]).tolist() for m in ('B1','P1')}}
    base.save_json(base.OUT/'report.json',report)
    return report


def comparison():
    rows=[]
    for seed,folder in ((42,'multiscenario_extended_v1'),(7,'stage5_seed7'),(13,'stage6_seed13')):
        history=json.loads((ROOT/'results'/folder/'P3_history.json').read_text())
        history=[h for h in history if h['epoch']<=6]
        if len(history)!=6: raise ValueError(f'Seed {seed} has not completed six epochs')
        best=max(history,key=lambda h:h['val']['top1'])
        rows.append(dict(seed=seed,epoch=best['epoch'],val=best['val']))
    report=dict(scope='Matched six-epoch development runs; validation-selected per seed. No test claim.',
        runs=rows,mean_top1=float(np.mean([r['val']['top1'] for r in rows])),
        sample_sd_top1=float(np.std([r['val']['top1'] for r in rows],ddof=1)),
        next_gate='Review generalization and verified literature/protocol before broader ablations, calibration and locked test.')
    base.save_json(base.OUT/'report.json',report)
    lines=['# Unattended experiment results','',report['scope'],'']
    for r in rows: lines.append(f"- Seed {r['seed']}, selected epoch {r['epoch']}: validation top-1 {r['val']['top1']:.2%}, top-5 {r['val']['top5']:.2%}.")
    lines += ['',f"Mean validation top-1: {report['mean_top1']:.2%}; sample SD: {report['sample_sd_top1']:.2%}.", '',report['next_gate']]
    (base.OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return report


def run(stage):
    plan=next(p for p in PLAN if p['id']==stage)
    base.OUT=ROOT/plan['directory']; base.OUT.mkdir(parents=True,exist_ok=True)
    base.status(status='preparing',model=plan['models'][0])
    if stage in (5,6):
        base.CONFIG={**base.CONFIG, 'models':['P3'],'seed':7 if stage==5 else 13,
                     'epochs':6,'lr_after_epoch3':3e-5,'stage':'matched-six-epoch-replication'}
        # New initialization; never copy model weights across seeds.
        source=json.loads((ROOT/'results/multiscenario_extended_v1/protocol.json').read_text())
        base.save_json(base.OUT/'continuation_provenance.json',dict(source_manifest_sha256=source['manifest_sha256'],
            note='Split guard only: fresh model initialization, no inherited weights.'))
        base.main()
    else:
        start=time.monotonic()
        report=uncertainty() if stage==4 else comparison()
        base.save_json(base.OUT/'analysis_history.json',[dict(epoch=1,seconds=time.monotonic()-start,
            val={'top1':report.get('mean_top1',report.get('models',{}).get('P3',{}).get('top1',0)), 'top5':None})])
        base.save_json(base.OUT/'analysis_finished.json',dict(reason='analysis_complete'))
        base.status(status='complete',scope='Validation analysis; calibration/test reserved.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=int,choices=[4,5,6,7],required=True)
    args=parser.parse_args()
    try: run(args.stage)
    except Exception as exc:
        base.status(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
