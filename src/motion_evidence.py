import json
import numpy as np
import pandas as pd
from src.motion_plan import ROOT, PLAN
from src.anchor_analysis import source
from src.publication_metrics import per_sample
from src.camera_evidence import macro_interval
from src.publication_plan import CACHE
from src.acquire_scenarios import save_json

def report(folder,progress):
    manifest=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    records={}
    for plan in PLAN:
        if plan['kind']!='train' or (ROOT/plan['directory']/'skipped.json').exists():
            continue
        scenario=plan['holdout']
        scope=f'without{scenario}_unseen' if scenario else 'pooled_val'
        expected=manifest.split.eq('val')
        if scenario: expected &= manifest.scenario.eq(scenario)
        with np.load(ROOT/f'results/conservative_v1/stage44/baseline_{scope}.npz') as data:
            ids,powers,labels=(data[k] for k in ('indices','profiles','labels'))
            if not np.array_equal(ids,manifest.index[expected]): raise ValueError('Validation IDs differ')
            outcomes={'geometry':per_sample(data['logits'],powers,labels)['success'].astype(float)}
        for arm,path in [('gps',source(42,scenario,False)),('mean',source(42,scenario,True)),('delta',plan['directory'])]:
            filename='unseen_predictions.npz' if scenario else 'val_predictions.npz'
            with np.load(ROOT/path/filename) as data:
                if not np.array_equal(data['indices'],ids) or not np.array_equal(data['profiles'],powers):raise ValueError('Pairing mismatch')
                outcomes[arm]=per_sample(data['logits'],powers,labels)['success'].astype(float)
        sc=manifest.scenario.to_numpy()[ids];blocks=manifest.block_key.to_numpy()[ids]
        records[scope]=dict(scores={k:float(np.mean([v[sc==s].mean() for s in np.unique(sc)])) for k,v in outcomes.items()},
            intervals={k:macro_interval(outcomes['delta']-outcomes[k],blocks,sc) for k in ('geometry','gps','mean')})
    save_json(folder/'evidence.json',dict(records=records,seed=42,scope='Exploratory paired development comparison'))
    (folder/'REPORT.md').write_text('# Endpoint difference versus temporal mean\n\n'+'\n'.join(f"- {k}: "+', '.join(f"{arm} {score:.3%}" for arm,score in v['scores'].items()) for k,v in records.items()),encoding='utf-8')
