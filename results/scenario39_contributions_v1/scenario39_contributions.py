"""Attribute the previously observed Scenario 39 contrast to recording sequences."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.anchor_analysis import source
from src.publication_data import digest
from src.publication_metrics import per_sample
from src.publication_plan import CACHE, ROOT
from src.acquire_scenarios import save_json


def load_success(path, expected_ids, keep, profiles):
    with np.load(path) as data:
        if not np.array_equal(data["indices"], expected_ids):
            raise ValueError(f"Unpaired validation rows in {path}")
        if not np.array_equal(data["profiles"][keep], profiles):
            raise ValueError(f"Power profiles differ in {path}")
        return per_sample(data["logits"][keep], profiles, data["labels"][keep])["success"].astype(float)


def decompose(frame, group):
    rows = []
    overall = frame["contrast"].sum()
    for key, part in frame.groupby(group):
        mass = part["contrast"].sum()
        rows.append(dict(group=str(key), windows=len(part), contribution_pp=100*mass/len(frame),
                         omit_pp=100*(overall-mass)/(len(frame)-len(part)),
                         transfer_vs_geometry_pp=100*part["vs_geometry"].mean(),
                         transfer_vs_gps_pp=100*part["vs_gps"].mean()))
    if not np.isclose(sum(row["contribution_pp"] for row in rows), 100*frame["contrast"].mean()):
        raise ValueError("Group contributions fail to recover overall contrast")
    return sorted(rows, key=lambda row: -abs(row["contribution_pp"]))


def main():
    deadline = json.loads((ROOT/"results/autonomous_17h_window.json").read_text())["ends_utc"]
    if datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
        raise ValueError("Review window closed")
    folder = ROOT/"results/scenario39_contributions_v1"
    folder.mkdir(exist_ok=True)
    code = Path(__file__)
    frozen = folder/code.name
    if frozen.exists() and digest(frozen)!=digest(code):
        raise ValueError("Frozen analysis differs")
    if not frozen.exists():
        frozen.write_bytes(code.read_bytes())
    manifest = pd.read_json(CACHE/"manifest.jsonl", lines=True)
    validation = manifest.loc[manifest.split.eq("val")]
    frame = validation.loc[validation.scenario.eq(39), ["seq_index","block_key"]].copy()
    differences = {}
    geometry_gains = []
    for scope in (None,39):
        ids = validation.index.to_numpy() if scope is None else frame.index.to_numpy()
        keep = np.isin(ids,frame.index)
        assert np.array_equal(ids[keep], frame.index)
        name = "pooled_val" if scope is None else "without39_unseen"
        filename = "val_predictions.npz" if scope is None else "unseen_predictions.npz"
        baseline = ROOT/f"results/conservative_v1/stage44/baseline_{name}.npz"
        with np.load(baseline) as data:
            if not np.array_equal(ids,data["indices"]):
                raise ValueError("Geometry row mismatch")
            profiles=data["profiles"][keep]
        geo=load_success(baseline,ids,keep,profiles)
        gains=[]
        for seed in (42,7,13):
            gps,visual=[load_success(ROOT/source(seed,scope,arm)/filename, ids,keep,profiles)
                        for arm in (False,True)]
            gains.append(visual-gps)
            if scope:
                geometry_gains.append(visual-geo)
        differences[scope]=np.mean(gains,axis=0)
    frame["contrast"]=differences[None]-differences[39]
    frame["vs_geometry"]=np.mean(geometry_gains,axis=0)
    frame["vs_gps"]=differences[39]
    expected=json.loads((ROOT/"results/variation_v1/stage219/audit.json").read_text())["by_scenario"]["39"]["effect"]
    if not np.isclose(100*frame.contrast.mean(),expected):
        raise ValueError("Scenario 39 contrast differs from Stage 219")
    result=dict(windows=len(frame),contrast_pp=expected,transfer_vs_geometry_pp=100*frame.vs_geometry.mean(),
                transfer_vs_gps_pp=100*frame.vs_gps.mean(),sequence=decompose(frame,"seq_index"),
                block=decompose(frame,"block_key"),source_sha256=digest(code))
    save_json(folder/"evidence.json",result)
    print(json.dumps({**result,"sequence":result["sequence"][:3]},indent=2))

if __name__=="__main__":
    main()
