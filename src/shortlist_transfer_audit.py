"""Attribute validation losses to candidate selection versus ordering, at fixed thresholds."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from src.power_loss_review import ROOT, CACHE, SEEDS, source, delta_path, digest, save_json


def decompose(oracle, chosen, threshold):
    if np.any(oracle > chosen + 1e-5):
        raise ValueError("Oracle loss exceeds deployed loss")
    excluded = oracle > threshold
    ranking = (chosen > threshold) & ~excluded
    return dict(miss=float((chosen>threshold).mean()), excluded=float(excluded.mean()),
                ranking=float(ranking.mean()))


def main():
    deadline=json.loads((ROOT/'results/autonomous_17h_window.json').read_text())['ends_utc']
    if datetime.now(timezone.utc)>=datetime.fromisoformat(deadline):raise ValueError('Window closed')
    folder=ROOT/'results/shortlist_transfer_v1'
    folder.mkdir(exist_ok=True)
    script=Path(__file__)
    frozen=folder/script.name
    if frozen.exists() and digest(frozen)!=digest(script):raise ValueError('Snapshot changed')
    if not frozen.exists():frozen.write_bytes(script.read_bytes())
    manifest=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    rows=[]
    for scenario in (36,37,38,39):
        ids=manifest.index[manifest.scenario.eq(scenario)&manifest.split.eq('val')].to_numpy()
        with np.load(ROOT/f'results/conservative_v1/stage44/baseline_without{scenario}_unseen.npz') as a:
            assert np.array_equal(a['indices'],ids)
            power=a['profiles']
            order=np.argsort(-a['logits'],axis=1,kind='stable')
        best=power.max(1)
        oracle=best-np.take_along_axis(power,order[:,:20],axis=1).max(1)
        candidates=np.zeros_like(power,dtype=bool)
        np.put_along_axis(candidates,order[:,:20],True,axis=1)
        for seed in SEEDS:
            for arm,path in [('gps',ROOT/source(seed,scenario,False)),('mean',ROOT/source(seed,scenario,True)),('delta',delta_path(seed,scenario))]:
                with np.load(path/'unseen_predictions.npz') as a:
                    assert np.array_equal(a['indices'],ids) and np.array_equal(a['profiles'],power)
                    selected=np.argsort(-a['logits'],axis=1,kind='stable')[:,:5]
                if not np.take_along_axis(candidates,selected,axis=1).all():raise ValueError('Candidate mismatch')
                loss=best-np.take_along_axis(power,selected,axis=1).max(1)
                for threshold in (3,10):
                    rows.append(dict(scenario=scenario,seed=seed,arm=arm,threshold=threshold,
                                     **decompose(oracle,loss,threshold)))
    macro=pd.DataFrame(rows).groupby(['arm','threshold'])[['miss','excluded','ranking']].mean().reset_index()
    macro['excluded_share']=macro.excluded/macro.miss
    assert np.allclose(macro.miss,macro.excluded+macro.ranking)
    prior=json.loads((ROOT/'results/power_loss_review_v1/evidence.json').read_text())['records']['transfer_macro']['values']
    for r in macro.itertuples():
        expected=prior[r.arm]['above_10db'] if r.threshold==10 else 1-prior[r.arm]['success']
        assert np.isclose(r.miss,expected,rtol=0,atol=1e-10)
    save_json(folder/'evidence.json',dict(rows=rows,macro=macro.to_dict('records'),source_sha256=digest(script)))
    print(macro.to_string(index=False))

if __name__=='__main__':main()
