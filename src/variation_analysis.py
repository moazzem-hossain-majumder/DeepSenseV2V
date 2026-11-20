"""Paired validation contrasts for the regularized 192px model and matched GPS."""
import json
import numpy as np
import pandas as pd

from src.anchor_analysis import source
from src.variation_plan import ROOT, OUT
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.acquire_scenarios import save_json


def bootstrap(frame, cluster):
    rng = np.random.default_rng(218)
    draws = np.zeros(4000)
    for scenario, group in frame.groupby("scenario"):
        stats = group.groupby(cluster)["effect"].agg(["sum", "count"])
        indices = rng.integers(len(stats), size=(len(draws), len(stats)))
        draws += stats["sum"].to_numpy()[indices].sum(1) / stats["count"].to_numpy()[indices].sum(1)
    return (np.quantile(draws/frame.scenario.nunique(), [.025,.975])*100).tolist()


def audit(plan, folder, progress):
    m = pd.read_json(CACHE/"manifest.jsonl", lines=True)
    frame = m.loc[m.split.eq("val"), ["scenario","block_key","seq_index"]].copy()
    for seed in (42,7,13):
        for held in (None,36,37,38,39):
            subset = frame if held is None else frame.loc[frame.scenario.eq(held)]
            scope = "pooled_val" if held is None else f"without{held}_unseen"
            with np.load(ROOT/f"results/conservative_v1/stage44/baseline_{scope}.npz") as a:
                ids = a["indices"]
                if not np.array_equal(ids,subset.index): raise ValueError("Validation row identity mismatch")
                profiles, labels = a["profiles"], a["labels"]
            outcomes = []
            for visual in (False,True):
                filename = "val_predictions.npz" if held is None else "unseen_predictions.npz"
                with np.load(ROOT/source(seed,held,visual)/filename) as a:
                    if not np.array_equal(a["indices"],ids) or not np.array_equal(a["profiles"],profiles):
                        raise ValueError("Arm pairing mismatch")
                    outcomes.append(per_sample(a["logits"],profiles,labels)["success"].astype(float))
            key = "pooled" if held is None else "transfer"
            frame.loc[ids,f"{key}_{seed}"] = outcomes[1]-outcomes[0]
        progress(status="pairing_predictions",model="analysis",batch=(42,7,13).index(seed)+1,batches=3)
    frame["pooled"] = frame[[f"pooled_{s}" for s in (42,7,13)]].mean(axis=1)
    frame["transfer"] = frame[[f"transfer_{s}" for s in (42,7,13)]].mean(axis=1)
    if frame.isna().any().any(): raise ValueError("Incomplete pairing")
    frame["effect"] = frame["pooled"]-frame["transfer"]
    by = frame.groupby("scenario")[["pooled","transfer","effect"]].mean()*100
    result = dict(windows=len(frame),pooled_gain_pp=by.pooled.mean(),transfer_gain_pp=by.transfer.mean(),
                  difference_pp=by.effect.mean(),by_scenario=by.to_dict("index"),
                  leave_one_scenario_out_pp={str(s):by.effect.drop(s).mean() for s in by.index},
                  cluster_counts=frame.groupby("scenario")[["block_key","seq_index"]].nunique().to_dict("index"),
                  block_interval_pp=bootstrap(frame,"block_key"),sequence_interval_pp=bootstrap(frame,"seq_index"))
    save_json(folder/"audit.json",result)


def report(folder,progress):
    info=json.loads((OUT/"stage219/audit.json").read_text())
    lines=["# How scenario exclusion changes the visual contribution","",
           f"All {info['windows']:,} validation windows were paired by ID across both training protocols and arms. We averaged the three seed-specific success differences per window and weighted the four scenarios equally.","",
           f"The visual advantage over matched GPS was {info['pooled_gain_pp']:+.3f} points in pooled validation and {info['transfer_gain_pp']:+.3f} points in transfer. The paired difference was {info['difference_pp']:+.3f} percentage points.",
           f"Scenario-stratified 95% percentile intervals from 4,000 paired draws were {info['block_interval_pp']} points for recorded blocks and {info['sequence_interval_pp']} points for recorded sequence IDs.",""]
    for s,row in info["by_scenario"].items():
        lines.append(f"- S{s} contributed {row['effect']:+.3f} points within that scenario. Omitting S{s} gave {info['leave_one_scenario_out_pp'][s]:+.3f} points across the other three.")
    lines+=["","Intervals condition on the four observed scenarios and selected checkpoints. Dependence between recorded sequences, limited cluster counts and prior model selection restrict inference to this development comparison."]
    (folder/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
