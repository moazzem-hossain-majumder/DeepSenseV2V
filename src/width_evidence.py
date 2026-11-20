import numpy as np
import pandas as pd
from src.anchor_analysis import source
from src.publication_metrics import per_sample
from src.publication_plan import CACHE
from src.camera_evidence import macro_interval
from src.width_plan import ROOT,PLAN
from src.acquire_scenarios import save_json


def report(folder,progress):
    manifest=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    groups=[]
    records={}
    for held in (36,37,38,39):
        ids=manifest.index[manifest.scenario.eq(held)&manifest.split.eq('val')].to_numpy()
        with np.load(ROOT/f'results/conservative_v1/stage44/baseline_without{held}_unseen.npz') as a:
            assert np.array_equal(a['indices'],ids)
            power,labels=a['profiles'],a['labels']
            losses={'geometry':per_sample(a['logits'],power,labels)['regret']}
        plans=[p for p in PLAN if p['kind']=='train' and p['holdout']==held]
        if any((ROOT/p['directory']/'skipped.json').exists() for p in plans):continue
        for p in plans:
            arm='visual' if p['vision'] else 'gps'
            for width,path in [(20,ROOT/source(42,held,p['vision'])),(40,ROOT/p['directory'])]:
                with np.load(path/'unseen_predictions.npz') as a:
                    assert np.array_equal(a['indices'],ids) and np.array_equal(a['profiles'],power)
                    losses[f'{arm}_{width}']=per_sample(a['logits'],power,labels)['regret']
        records[str(held)]={arm:dict(success=float((v<=3).mean()),mean_db=float(v.mean()),severe=float((v>10).mean())) for arm,v in losses.items()}
        groups.append((ids,losses))
    summary={}
    if len(groups)==4:
        ids=np.concatenate([x for x,_ in groups])
        sc=manifest.scenario.to_numpy()[ids]
        blocks=manifest.block_key.to_numpy()[ids]
        all_loss={k:np.concatenate([loss[k] for _,loss in groups]) for k in groups[0][1]}
        for arm,control in [('visual_40','visual_20'),('gps_40','gps_20'),('visual_40','gps_40'),('visual_40','geometry')]:
            first,second=all_loss[arm],all_loss[control]
            success=(first<=3).astype(float)-(second<=3)
            summary[f'{arm}_minus_{control}']=dict(delta_success=float(np.mean([success[sc==s].mean() for s in np.unique(sc)])),interval=macro_interval(success,blocks,sc))
    save_json(folder/'evidence.json',dict(records=records,contrasts=summary,seed=42))
    lines=['# Candidate width and transfer','','Each model selects five beams. Results use seed 42 and previously studied validation scenarios.','']
    for held,row in records.items():
        lines.append(f"- S{held}: "+', '.join(f"{arm} {v['success']:.2%}" for arm,v in row.items()))
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
