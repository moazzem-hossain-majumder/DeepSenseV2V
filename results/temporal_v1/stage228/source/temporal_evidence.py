"""Paired comparisons of the prespecified temporal contrast on validation windows."""
import numpy as np
import pandas as pd
from src.anchor_analysis import source
from src.motion_plan import PLAN as PILOT
from src.temporal_plan import ROOT, PLAN
from src.camera_evidence import macro_interval
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.acquire_scenarios import save_json


def delta_path(seed, scenario):
    return ROOT/next(p['directory'] for p in PILOT+PLAN if p['kind']=='train'
                     and p['seed']==seed and p['holdout']==scenario)


def summarize(values, scenarios, blocks):
    means = {name:np.mean(rows,axis=0) for name,rows in values.items()}
    def macro(rows):
        return float(np.mean([rows[scenarios==s].mean() for s in np.unique(scenarios)]))
    return dict(scores={name:macro(v) for name,v in means.items()},
                seed_scores={name:[macro(row) for row in rows] for name,rows in values.items()},
                intervals={name:macro_interval(means['delta']-means[name],blocks,scenarios)
                           for name in ('mean','gps','geometry')})


def report(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    records={}
    transfers=[]
    for scenario in (None,36,37,38,39):
        scope=f'without{scenario}_unseen' if scenario else 'pooled_val'
        expected=m.split.eq('val')
        if scenario:
            expected &= m.scenario.eq(scenario)
        with np.load(ROOT/f'results/conservative_v1/stage44/baseline_{scope}.npz') as archive:
            ids,powers,labels=(archive[k] for k in ('indices','profiles','labels'))
            if not np.array_equal(ids,m.index[expected]):raise ValueError('Scope identity mismatch')
            geometry=per_sample(archive['logits'],powers,labels)['success'].astype(float)
        values={name:[] for name in ('delta','mean','gps','geometry')}
        for seed in (42,7,13):
            paths={'delta':delta_path(seed,scenario),'mean':ROOT/source(seed,scenario,True),
                   'gps':ROOT/source(seed,scenario,False)}
            if (paths['delta']/'skipped.json').exists():
                break
            for name,path in paths.items():
                file='unseen_predictions.npz' if scenario else 'val_predictions.npz'
                with np.load(path/file) as a:
                    if not np.array_equal(ids,a['indices']) or not np.array_equal(powers,a['profiles']):
                        raise ValueError('Paired prediction mismatch')
                    values[name].append(per_sample(a['logits'],powers,labels)['success'].astype(float))
            values['geometry'].append(geometry)
        if len(values['delta'])!=3:
            records[scope]={'incomplete':True}
            continue
        sc=m.scenario.to_numpy()[ids]
        blocks=m.block_key.to_numpy()[ids]
        records[scope]=summarize(values,sc,blocks)
        if scenario:
            transfers.append((values,sc,blocks))
    if len(transfers)==4:
        combined={k:np.concatenate([part[0][k] for part in transfers],axis=1) for k in transfers[0][0]}
        records['transfer_macro']=summarize(combined,np.concatenate([p[1] for p in transfers]),np.concatenate([p[2] for p in transfers]))
    save_json(folder/'evidence.json',dict(records=records,seeds=[42,7,13],scope='Development validation across four previously explored scenarios.'))
    lines=['# Temporal feature replication','','All differences compare matched windows with equal scenario weight.','']
    for name,r in records.items():
        if 'scores' in r:
            lines.append(f"- {name}: "+', '.join(f'{arm} {score:.3%}' for arm,score in r['scores'].items()))
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
