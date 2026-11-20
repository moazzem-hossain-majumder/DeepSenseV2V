import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from src.robust_plan import ROOT,PLAN
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.camera_evidence import macro_interval
from src.acquire_scenarios import save_json
def report(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True);records=[];all_ids=[];outcomes={k:[] for k in ('geometry','gps','original','dropout','consistency')}
    for s in (36,37,38,39):
        with np.load(ROOT/f'results/conservative_v1/stage44/baseline_without{s}_unseen.npz') as a:
            ids=a['indices'];geo=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
        if not ((m.scenario.iloc[ids]==s)&(m.split.iloc[ids]=='val')).all():raise ValueError('Held scope mismatch')
        values={'geometry':geo};all_ids.extend(ids)
        paths={'gps':ROOT/f'results/transfer_v1/stage{165+2*(s-36)}','original':ROOT/f'results/transfer_v1/stage{166+2*(s-36)}'}
        for p in PLAN:
            if p['kind']=='train' and p['holdout']==s:paths['consistency' if p['consistency'] else 'dropout']=ROOT/p['directory']
        for name,path in paths.items():
            with np.load(path/'val_predictions.npz') as a:
                if (m.scenario.iloc[a['indices']]==s).any():raise ValueError('Held labels used for checkpoint selection')
            with np.load(path/'unseen_predictions.npz') as a:
                if not np.array_equal(ids,a['indices']):raise ValueError('IDs differ')
                values[name]=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
        records.append(dict(scenario=s,scores={k:float(v.mean()) for k,v in values.items()}))
        for name,v in values.items():outcomes[name].extend(v)
    ids=np.asarray(all_ids);sc=m.scenario.to_numpy()[ids];blocks=m.block_key.to_numpy()[ids];outcomes={k:np.asarray(v) for k,v in outcomes.items()}
    macro=lambda a:float(np.mean([a[sc==s].mean() for s in np.unique(sc)]))
    contrasts={}
    for arm in ('dropout','consistency'):
        for control in ('original','gps','geometry'):
            d=outcomes[arm]-outcomes[control];contrasts[f'{arm}_minus_{control}']=dict(gain=macro(d),interval=macro_interval(d,blocks,sc))
    action='Report both prespecified variants. Assess whether transfer harm decreased without selecting per-scenario variants or fallback from held labels. Further hyperparameter search requires inner splits restricted to training scenarios.'
    evidence=dict(publication_ready=False,calibration_test_used=False,records=records,means={k:macro(v) for k,v in outcomes.items()},contrasts=contrasts,next_action=action,scope='Fixed settings, seed-42 exploratory transfer development; excluded scenarios previously examined. No independent confirmation claim.')
    save_json(folder/'evidence.json',evidence)
    lines=['# Prespecified visual robustness comparison','',evidence['scope'],'']
    for r in records:lines.append(f"- S{r['scenario']}: "+', '.join(f'{k} {v:.2%}' for k,v in r['scores'].items()))
    lines+=['',action,'Calibration and final test remain reserved.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    with (ROOT/'paper/EXPERIMENT_TRACKER.md').open('a',encoding='utf-8') as f:f.write(f'\nStage 189 complete {datetime.now(timezone.utc).isoformat()}; see results/robust_v1/stage189/REPORT.md.\n')
