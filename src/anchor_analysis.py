"""Evaluate one fixed beam-selection rule on saved development predictions."""
import json
import numpy as np
import pandas as pd

from src.anchor_plan import ROOT, OUT, PLAN
from src.steady_plan import PLAN as RUNS
from src.camera_evidence import macro_interval
from src.publication_plan import CACHE
from src.acquire_scenarios import save_json


def choose(geometry, learned):
    protected = np.argsort(-geometry, axis=1, kind="stable")[:, :3]
    candidates = np.argsort(-learned, axis=1, kind="stable")[:, :5]
    available = ~(candidates[:, :, None] == protected[:, None, :]).any(2)
    positions = np.argsort(~available, axis=1, kind="stable")[:, :2]
    return np.column_stack((protected, np.take_along_axis(candidates, positions, axis=1)))


def source(seed, scenario, visual):
    if scenario is None and not visual:
        return f"results/rescheck_v1/stage{ {42:156,7:157,13:160}[seed]}"
    if scenario and seed == 42:
        family, first = ("robust", 182) if visual else ("transfer", 165)
        return f"results/{family}_v1/stage{first+2*(scenario-36)}"
    return next(p["directory"] for p in RUNS if p["kind"]=="train" and
                (p["holdout"],p["seed"],p["vision"]) == (scenario,seed,visual))


def audit(plan, folder, progress):
    scenario = plan["scenario"]
    scope = f"without{scenario}_unseen" if scenario else "pooled_val"
    with np.load(ROOT/f"results/conservative_v1/stage44/baseline_{scope}.npz") as baseline:
        ids, geo, powers = (baseline[k] for k in ("indices","logits","profiles"))
    manifest = pd.read_json(CACHE/"manifest.jsonl",lines=True).iloc[ids]
    assert manifest.split.eq("val").all() and (not scenario or manifest.scenario.eq(scenario).all())
    sc = manifest.scenario.to_numpy()
    blocks = manifest.block_key.to_numpy()
    def success(beams):
        return (powers.max(1)-np.take_along_axis(powers,beams,axis=1).max(1)<=3).astype(float)
    def macro(values):
        return float(np.mean([values[sc==s].mean() for s in np.unique(sc)]))
    geometry = success(np.argsort(-geo,axis=1,kind="stable")[:,:5])
    arms = {}
    for i,seed in enumerate((42,7,13)):
        for visual in (False,True):
            arm = "visual" if visual else "gps"
            filename = "unseen_predictions.npz" if scenario else "val_predictions.npz"
            with np.load(ROOT/source(seed,scenario,visual)/filename) as archive:
                assert np.array_equal(archive["indices"],ids) and np.array_equal(archive["profiles"],powers)
                scores = archive["logits"]
            for key, beams in ((arm,np.argsort(-scores,axis=1,kind="stable")[:,:5]),
                               (f"protected_{arm}",choose(geo,scores))):
                arms.setdefault(key,[]).append(success(beams))
        progress(status="scoring",model="analysis",batch=i+1,batches=3)
    means = {key:np.mean(values,axis=0) for key,values in arms.items()}
    means["geometry"] = geometry
    contrasts = {}
    for key in ("geometry","visual","protected_gps"):
        delta = means["protected_visual"] - means[key]
        contrasts[key] = dict(gain=macro(delta),interval=macro_interval(delta,blocks,sc))
    save_json(folder/"audit.json",dict(scores={k:macro(v) for k,v in means.items()},contrasts=contrasts))


def report(folder,progress):
    rows = {}
    for stage in PLAN:
        if stage["kind"]=="audit":
            path=ROOT/stage["directory"]/"audit.json"
            if path.exists():
                rows[str(stage["scenario"] or "pooled")]=json.loads(path.read_text())
    save_json(folder/"evidence.json",dict(scopes=rows,seeds=[42,7,13],protected_count=3))
    lines=["# Protecting three geometry beam choices","","Three-seed development validation. Each model fills two of five unique beam positions.",""]
    for scope,row in rows.items():
        lines.append(f"- {scope}: "+", ".join(f"{name} {value:.3%}" for name,value in row["scores"].items()))
    (folder/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
