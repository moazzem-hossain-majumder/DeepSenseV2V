import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from src.rescheck_plan import ROOT,PREVIOUS,PLAN
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.camera_evidence import macro_interval
from src.acquire_scenarios import save_json
def report(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True);records=[];outcomes={}
    with np.load(PREVIOUS/'stage44/baseline_pooled_val.npz') as a:
        ids=a['indices'];geo=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
    scenarios=m.scenario.to_numpy()[ids];blocks=m.block_key.to_numpy()[ids]
    macro=lambda x:float(np.mean([x[scenarios==s].mean() for s in np.unique(scenarios)]))
    runs=[dict(id=i,directory=f'results/resolution_v1/stage{i}',seed=42,resolution=size,vision=True) for i,size in ((152,96),(153,192))]+[p for p in PLAN if p['kind']=='train']
    for p in runs:
        path=ROOT/p['directory'];r=json.loads((path/'result.json').read_text())
        with np.load(path/'val_predictions.npz') as a:
            if not np.array_equal(ids,a['indices']):raise ValueError('Prediction IDs differ')
            success=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
        arm=str(p['resolution']) if p['vision'] else 'gps';outcomes[p['seed'],arm]=success
        records.append(dict(stage=p['id'],seed=p['seed'],arm=arm,score=macro(success),best_epoch=r['best_epoch'],by_scenario={str(s):float(success[scenarios==s].mean()) for s in np.unique(scenarios)}))
    means={arm:np.mean([outcomes[s,arm] for s in (42,7,13)],axis=0) for arm in ('gps','96','192')}
    contrasts={}
    for a,b in (('192','96'),('96','gps'),('192','gps')):
        delta=means[a]-means[b]
        contrasts[f'{a}_minus_{b}']=dict(gain=macro(delta),interval=macro_interval(delta,blocks,scenarios),per_seed={str(s):macro(outcomes[s,a]-outcomes[s,b]) for s in (42,7,13)})
    for arm in ('96','192'):contrasts[f'{arm}_minus_geometry']=dict(gain=macro(means[arm]-geo),interval=macro_interval(means[arm]-geo,blocks,scenarios))
    evidence=dict(publication_ready=False,calibration_test_used=False,baseline=macro(geo),means={k:macro(v) for k,v in means.items()},records=records,contrasts=contrasts,
        scope='Exploratory blocked validation. Seed 42 was screened previously. Block intervals condition on selected checkpoints and do not account fully for repeated selection or seed uncertainty.',
        next_action='Assess consistency, uncertainty and per-scenario harms before predeclaring transfer evaluation; keep final test reserved.')
    save_json(folder/'evidence.json',evidence)
    lines=['# Matched resolution replication','',evidence['scope'],'']
    for arm,value in evidence['means'].items():lines.append(f'- {arm} mean: {value:.2%}.')
    for name,c in contrasts.items():lines.append(f"- {name}: {c['gain']*100:+.3f} percentage points; block interval [{c['interval'][0]*100:.3f}, {c['interval'][1]*100:.3f}] points.")
    lines+=['',evidence['next_action'],'No automatic publication-readiness claim.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    with (ROOT/'paper/EXPERIMENT_TRACKER.md').open('a',encoding='utf-8') as f:f.write(f'\nStage 163 completed {datetime.now(timezone.utc).isoformat()}; see results/rescheck_v1/stage163/REPORT.md.\n')
