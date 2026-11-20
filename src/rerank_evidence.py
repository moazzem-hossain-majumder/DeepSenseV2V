import json
import shutil
import numpy as np
import pandas as pd
from src.rerank_plan import ROOT,OUT,PLAN,PREVIOUS
from src.publication_data import load,partitions
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.camera_evidence import macro_interval
from src.acquire_scenarios import save_json

def coverage(folder,progress):
    m,_,power,_=load();train,val,__=partitions(m);sc=m.scenario.to_numpy();labels=m.target_beam.to_numpy()
    prior=np.load(PREVIOUS/'stage44/prior_pooled.npy',mmap_mode='r');records={}
    try:
        for split,ids in (('train',train),('val',val)):
            records[split]={}
            for k in (5,10,20):
                outcomes=per_sample(prior[ids],power[ids],labels[ids],k)
                by={str(s):float(outcomes['success'][sc[ids]==s].mean()) for s in (36,37,38,39)}
                records[split][str(k)]=dict(macro=float(np.mean(list(by.values()))),by_scenario=by,misses=int((~outcomes['success']).sum()))
            progress(status='measuring_shortlist_coverage',model='analysis',batch=1 if split=='train' else 2,batches=2)
        eligible=[k for k in (10,20) if records['train'][str(k)]['macro']>=records['train']['5']['macro']+.02]
        save_json(folder/'coverage.json',dict(records=records,eligible_k=eligible,
            scope='Top-K near-optimal coverage is an optimistic upper bound on five-beam reranking, not achievable performance. Candidate feasibility gate uses training coverage gain >=2 percentage points. Validation breakdown is exploratory.'))
        shutil.copy2(ROOT/'paper/RERANK_BASELINE_AUDIT.md',folder/'BASELINE_AUDIT.md')
    finally:prior._mmap.close()

def select(records,baseline):
    eligible=[]
    for k in (10,20):
        pair={r['vision']:r for r in records if r['k']==k}
        if len(pair)==2 and pair[True]['score']>=max(baseline,pair[False]['score'])+.0025:eligible.append(pair[True])
    return max(eligible,key=lambda r:r['score']) if eligible else None

def gate(folder,progress):
    feasibility=json.loads((OUT/'stage96/coverage.json').read_text());records=[]
    baseline=feasibility['records']['val']['5']['macro']
    for p in PLAN:
        if p['kind']=='pilot' and not (ROOT/p['directory']/'skipped.json').exists():
            r=json.loads((ROOT/p['directory']/'result.json').read_text())
            records.append(dict(stage=p['id'],k=p['k'],vision=p['vision'],score=r['val']['macro_success5']))
    save_json(folder/'selection.json',dict(chosen=select(records,baseline),records=records,baseline=baseline,
        scope='Expand only if visual reranker exceeds both analytic geometry and matched GPS-only pilot by 0.25 percentage points. Validation-selected exploratory screen.'))

def report(folder,progress):
    selection=json.loads((OUT/'stage101/selection.json').read_text());records=selection['records']
    evidence=dict(publication_ready=False,calibration_test_used=False,selection=selection)
    lines=['# Constrained reranking results','',selection['scope'],'']
    for r in records:lines.append(f"- Stage {r['stage']}, K={r['k']}, images={r['vision']}: {r['score']:.2%}.")
    if selection['chosen']:
        m=pd.read_json(CACHE/'manifest.jsonl',lines=True);preds={False:[],True:[]};scores={False:[],True:[]}
        with np.load(PREVIOUS/'stage44/baseline_pooled_val.npz') as a:
            ids=a['indices'];geometry=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
        for p in PLAN:
            if p['kind']=='expand' and p['holdout'] is None:
                path=ROOT/p['directory'];scores[p['vision']].append(json.loads((path/'result.json').read_text())['val']['macro_success5'])
                with np.load(path/'val_predictions.npz') as a:
                    if not np.array_equal(ids,a['indices']):raise ValueError('Prediction ID mismatch')
                    preds[p['vision']].append(per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float))
        visual=np.mean(preds[True],axis=0);gps=np.mean(preds[False],axis=0);blocks=m.block_key.to_numpy()[ids];sc=m.scenario.to_numpy()[ids]
        transfer={};geo=json.loads((PREVIOUS/'stage44/report.json').read_text())['configurations']
        for s in (36,37,38,39):
            case={'geometry':geo[f'without{s}']['unseen_val']['macro_success5']}
            for p in PLAN:
                if p['kind']=='expand' and p['holdout']==s:case['visual' if p['vision'] else 'gps']=json.loads((ROOT/p['directory']/'unseen_report.json').read_text())['macro_success5']
            transfer[str(s)]=case
        evidence.update(visual_mean=float(np.mean(scores[True])),gps_mean=float(np.mean(scores[False])),
            interval_vs_geometry=macro_interval(visual-geometry,blocks,sc),interval_vs_gps=macro_interval(visual-gps,blocks,sc),transfer=transfer)
        lines += ['',f"Three-seed visual mean {evidence['visual_mean']:.2%}; matched GPS {evidence['gps_mean']:.2%}; geometry {selection['baseline']:.2%}.",
            f"Paired macro intervals versus geometry: {evidence['interval_vs_geometry']}; versus GPS: {evidence['interval_vs_gps']}."]
        for s,r in transfer.items():lines.append(f"- Held S{s}: visual {r['visual']:.2%}, GPS {r['gps']:.2%}, geometry {r['geometry']:.2%}.")
    else:lines+=['','No visual pilot passed the expansion screen; paired longer runs were skipped.']
    lines+=['','This is exploratory development evidence. Shortlist size selection used pooled validation; transfer checks are therefore not independent architecture confirmation. Physical clock conversion and exact published-challenge reproduction remain unresolved. Calibration/test stay reserved.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');save_json(folder/'evidence.json',evidence)
