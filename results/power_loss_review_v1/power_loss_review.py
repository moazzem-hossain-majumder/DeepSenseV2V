"""Development power-loss audit for the saved top-five predictions."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.anchor_analysis import source
from src.temporal_evidence import delta_path
from src.publication_plan import ROOT,CACHE
from src.publication_data import digest
from src.publication_metrics import per_sample
from src.camera_evidence import macro_interval
from src.acquire_scenarios import save_json

ARMS = ('geometry','gps','mean','delta')
SEEDS = (42,7,13)


def describe(rows, scenarios):
    # Each seed and each scenario contributes equally to these descriptive measures.
    cells = [row[scenarios == scenario] for row in rows for scenario in np.unique(scenarios)]
    misses = [cell[cell > 3] for cell in cells]
    return dict(mean_db=float(np.mean([cell.mean() for cell in cells])),
                success=float(np.mean([(cell <= 3).mean() for cell in cells])),
                q90_db=float(np.mean([np.quantile(cell,.90) for cell in cells])),
                above_10db=float(np.mean([(cell>10).mean() for cell in cells])),
                conditional_miss_db=float(np.mean([miss.mean() for miss in misses if len(miss)])),
                cells_with_misses=sum(bool(len(miss)) for miss in misses),cells=len(cells))


def read_scope(scenario, manifest):
    tag = f'without{scenario}_unseen' if scenario else 'pooled_val'
    selected = manifest.split.eq('val')
    if scenario:
        selected &= manifest.scenario.eq(scenario)
    ids = manifest.index[selected].to_numpy()
    with np.load(ROOT/f'results/conservative_v1/stage44/baseline_{tag}.npz') as archive:
        if not np.array_equal(ids,archive['indices']):
            raise ValueError('Baseline row mismatch')
        powers, labels = archive['profiles'], archive['labels']
        geometry = per_sample(archive['logits'],powers,labels)['regret'].astype(float)
    arms = {arm:[] for arm in ARMS}
    for seed in SEEDS:
        arms['geometry'].append(geometry)
        for arm,path in [('gps', ROOT/source(seed,scenario,False)),('mean',ROOT/source(seed,scenario,True)),
                         ('delta',delta_path(seed,scenario))]:
            name = 'unseen_predictions.npz' if scenario else 'val_predictions.npz'
            with np.load(path/name) as archive:
                if not np.array_equal(ids,archive['indices']) or not np.array_equal(powers,archive['profiles']):
                    raise ValueError('Unpaired prediction rows or powers')
                arms[arm].append(per_sample(archive['logits'],powers,labels)['regret'].astype(float))
    return ids,{arm:np.stack(values) for arm,values in arms.items()}


def main():
    deadline=json.loads((ROOT/'results/autonomous_17h_window.json').read_text())['ends_utc']
    if datetime.now(timezone.utc)>=datetime.fromisoformat(deadline):
        raise ValueError('Research window ended')
    out=ROOT/'results/power_loss_review_v1'
    out.mkdir(exist_ok=True)
    code=Path(__file__)
    frozen=out/code.name
    if frozen.exists() and digest(frozen)!=digest(code):
        raise ValueError('Source snapshot differs')
    if not frozen.exists():
        frozen.write_bytes(code.read_bytes())
    manifest=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    reference=json.loads((ROOT/'results/temporal_v1/stage239/evidence.json').read_text())['records']
    folds=[read_scope(s,manifest) for s in (36,37,38,39)]
    held_ids=np.concatenate([ids for ids,_ in folds])
    held={arm:np.concatenate([arms[arm] for _,arms in folds],axis=1) for arm in ARMS}
    records={}
    for scope,(ids,arms) in [('transfer_macro',(held_ids,held)),('pooled_val',read_scope(None,manifest))]:
        sc=manifest.scenario.to_numpy()[ids]
        blocks=manifest.block_key.to_numpy()[ids]
        values={arm:describe(rows,sc) for arm,rows in arms.items()}
        for arm,val in values.items():
            if not np.isclose(val['success'],reference[scope]['scores'][arm],atol=1e-10,rtol=0):
                raise ValueError('Success-rate reconstruction failed')
        contrasts={}
        for arm in ('mean','delta'):
            for control in ('gps','geometry'):
                diff=(arms[arm]-arms[control]).mean(axis=0)
                contrasts[f'{arm}_minus_{control}']=dict(mean_db=values[arm]['mean_db']-values[control]['mean_db'],
                    interval_db=macro_interval(diff,blocks,sc))
        records[scope]=dict(values=values,paired=contrasts)
    save_json(out/'evidence.json',dict(records=records,source_hash=digest(code),seeds=SEEDS))
    lines=['# Top-five received-power loss','','Loss is the difference in dB between the strongest measured beam and the strongest of the five suggested beams. Means, percentiles and rates below give each scenario and seed equal weight. Conditional severity uses each model\'s own set of misses over 3 dB.','']
    for scope,record in records.items():
        for arm,v in record['values'].items():
            lines.append(f"- {scope}, {arm}: mean {v['mean_db']:.3f} dB, mean within-cell 90th percentile {v['q90_db']:.3f} dB, losses above 10 dB {v['above_10db']:.2%}, mean on misses {v['conditional_miss_db']:.3f} dB.")
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[4:]))
    print(records['transfer_macro']['paired'])


if __name__=='__main__':
    main()
