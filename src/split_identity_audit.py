"""Audit train/validation identifiers and nearest recorded times without loading targets."""
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.publication_plan import ROOT, CACHE
from src.publication_data import digest
from src.acquire_scenarios import save_json

def nearest(reference, query):
    ordered = np.sort(reference)
    pos = np.searchsorted(ordered, query)
    left = ordered[np.maximum(0,pos-1)]
    right = ordered[np.minimum(len(ordered)-1,pos)]
    return np.minimum(np.abs(left-query),np.abs(right-query))

def main():
    deadline = json.loads((ROOT/'results/autonomous_17h_window.json').read_text())['ends_utc']
    if datetime.now(timezone.utc)>=datetime.fromisoformat(deadline):
        raise ValueError('Window closed')
    folder = ROOT/'results/split_identity_v1'
    folder.mkdir(exist_ok=True)
    source = Path(__file__)
    snapshot = folder/source.name
    if snapshot.exists() and digest(snapshot)!=digest(source):
        raise ValueError('Frozen source changed')
    snapshot.write_bytes(source.read_bytes())
    fields = ['scenario','split','block_key','seq_index','x_indices','y_index']
    records = []
    with (CACHE/'manifest.jsonl').open() as handle:
        for line in handle:
            row=json.loads(line)
            if row['split'] in ('train','val'):
                records.append({k:row[k] for k in fields})
    manifest = pd.DataFrame(records)
    receipt = json.loads((CACHE/'receipt.json').read_text())
    assert digest(CACHE/'manifest.jsonl')==receipt['manifest_sha256']
    results = {}
    for scenario, group in manifest.groupby('scenario'):
        path = ROOT/f'data/raw/scenario{scenario}/scenario{scenario}.csv'
        assert digest(path)==receipt['csv_sha256'][str(scenario)]
        raw=pd.read_csv(path,usecols=['timestamp','abs_index','seq_index','unit1_rgb5','unit1_rgb6'])
        clocks=pd.to_timedelta(raw.timestamp.str.replace('-',':',regex=False)).dt.total_seconds().to_numpy()
        used={}
        inputs={}
        parts={}
        for split, part in group.groupby('split'):
            x=np.stack(part.x_indices); y=part.y_index.to_numpy()
            assert (raw.seq_index.to_numpy()[x]==part.seq_index.to_numpy()[:,None]).all()
            assert np.array_equal(raw.seq_index.to_numpy()[y],part.seq_index)
            inputs[split]=np.unique(x)
            used[split]=np.union1d(inputs[split],y)
            parts[split]=part
        a,b=used['train'],used['val']
        paths={k:len(set(raw[k].iloc[inputs['train']]) & set(raw[k].iloc[inputs['val']])) for k in ('unit1_rgb5','unit1_rgb6')}
        times=nearest(clocks[a],clocks[b])
        results[str(scenario)]=dict(
            windows={k:len(v) for k,v in parts.items()},
            sequence_overlap=sorted(set(parts['train'].seq_index)&set(parts['val'].seq_index)),
            block_overlap=sorted(set(parts['train'].block_key)&set(parts['val'].block_key)),
            shared_raw_rows=len(np.intersect1d(a,b)), shared_image_paths=paths,
            chronology_strict=bool(np.all(np.diff(clocks[np.union1d(a,b)])>0)),
            first_validation_after_last_train_seconds=float(clocks[b].min()-clocks[a].max()),
            minimum_nearest_seconds=float(times.min()),median_nearest_seconds=float(np.median(times)),
            min_abs_index_distance=int(nearest(raw.abs_index.to_numpy()[a],raw.abs_index.to_numpy()[b]).min()))
    save_json(folder/'evidence.json',dict(scenarios=results,sha256=digest(source),manifest_sha256=receipt['manifest_sha256']))
    print(json.dumps(results))

if __name__=='__main__':main()
