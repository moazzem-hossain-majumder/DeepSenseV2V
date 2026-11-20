"""Decompose the K40 change with the same trained scores restricted to the old shortlist."""
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.anchor_analysis import source
from src.width_plan import ROOT, PLAN
from src.publication_plan import CACHE
from src.publication_data import digest
from src.acquire_scenarios import save_json


def restriction(geometry, scores):
    order = np.argsort(-geometry, axis=1, kind='stable')
    ranks = np.argsort(order, axis=1)
    deployed = np.argsort(-scores, axis=1, kind='stable')[:, :5]
    used = np.take_along_axis(ranks, deployed, axis=1)
    if np.any(used >= 40):
        raise ValueError('Selected beam outside K40')
    restricted = np.argsort(-np.where(ranks < 20, scores, -np.inf), axis=1, kind='stable')[:, :5]
    return deployed, restricted, (used >= 20).any(1)


def loss(power, selected):
    return power.max(1) - np.take_along_axis(power, selected, axis=1).max(1)


def main():
    deadline=json.loads((ROOT/'results/autonomous_17h_window.json').read_text())['ends_utc']
    if datetime.now(timezone.utc)>=datetime.fromisoformat(deadline):
        raise ValueError('Research window closed')
    folder=ROOT/'results/width_attribution_v1'
    folder.mkdir(exist_ok=True)
    code=Path(__file__)
    frozen=folder/code.name
    if frozen.exists() and digest(frozen)!=digest(code):
        raise ValueError('Source differs from snapshot')
    if not frozen.exists(): frozen.write_bytes(code.read_bytes())
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    prior=json.loads((ROOT/'results/width_v1/stage249/evidence.json').read_text())['records']
    rows=[]
    for plan in PLAN:
        if plan['kind']!='train':continue
        s=plan['holdout']
        arm='visual' if plan['vision'] else 'gps'
        ids=m.index[m.scenario.eq(s)&m.split.eq('val')].to_numpy()
        with np.load(ROOT/f'results/conservative_v1/stage44/baseline_without{s}_unseen.npz') as a:
            if not np.array_equal(ids,a['indices']):raise ValueError('Baseline IDs differ')
            geo,power=a['logits'],a['profiles']
        losses={}
        for k,path in [(20,ROOT/source(42,s,plan['vision'])),(40,ROOT/plan['directory'])]:
            with np.load(path/'unseen_predictions.npz') as a:
                if not np.array_equal(ids,a['indices']) or not np.array_equal(power,a['profiles']):
                    raise ValueError('Paired inputs differ')
                scores=a['logits']
            if k==40:
                selected,restricted,used=restriction(geo,scores)
                losses['restricted']=loss(power,restricted).astype(float)
            else:
                selected=np.argsort(-scores,axis=1,kind='stable')[:,:5]
            losses[k]=loss(power,selected).astype(float)
            if abs(losses[k].mean()-prior[str(s)][f'{arm}_{k}']['mean_db'])>1e-4:
                raise ValueError('Reference loss mismatch')
        access=losses['restricted']-losses[40]
        reorder=losses[20]-losses['restricted']
        total=losses[20]-losses[40]
        np.testing.assert_allclose(access+reorder,total)
        if np.any(access[~used]!=0):raise ValueError('Difference without a new beam')
        rows.append(dict(scenario=s,arm=arm,n=len(ids),used=int(used.sum()),access_db=float(access.mean()),
                         reorder_db=float(reorder.mean()),total_db=float(total.mean()),
                         helped=int((access>0).sum()),harmed=int((access<0).sum())))
    save_json(folder/'evidence.json',dict(seed=42,rows=rows,source_sha256=digest(code)))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    main()
