"""Export block-level sufficient statistics for the full-codebook development study."""
import csv
import numpy as np
import pandas as pd
from src.publication_plan import ROOT, CACHE
from src.publication_metrics import per_sample
from src.swin_comparison_plan import PLAN

def main():
    manifest = pd.read_json(CACHE / "manifest.jsonl", lines=True)
    validation = manifest.loc[manifest.split.eq("val")]
    rows = []
    for run in PLAN:
        if run["kind"] != "train":
            continue
        held = run["holdout"]
        frame = validation.loc[validation.scenario.eq(held)] if held else validation
        scope = f"without{held}_unseen" if held else "pooled_val"
        pred = ROOT / run["directory"] / ("unseen_predictions.npz" if held else "val_predictions.npz")
        arms = [("rgb" if run["vision"] else "gps", pred)]
        if not run["vision"]:
            arms.append(("geometry", ROOT / f"results/conservative_v1/stage44/baseline_{scope}.npz"))
        for arm, path in arms:
            with np.load(path) as data:
                np.testing.assert_array_equal(data["indices"], frame.index)
                p, y, z = data["profiles"], data["labels"], data["logits"]
                assert np.all(p.argmax(1) == y) and np.all((p == p.max(1, keepdims=True)).sum(1) == 1)
                out = per_sample(z, p, y)
                block = frame[["scenario", "block_key"]].copy()
                block["n"] = 1
                block["hit"] = out["hit"]
                block["success"] = out["success"]
                block["alternative"] = out["success"] & ~out["hit"]
                block["regret"] = out["regret"].astype(float)
                block["top1_loss_db"] = per_sample(z, p, y, 1)["regret"].astype(float)
                for record in block.groupby(["scenario","block_key"]).sum().reset_index().to_dict("records"):
                    rows.append(dict(split="transfer" if held else "pooled", arm=arm, seed=run["seed"], **record))
    target = ROOT / "results/swin_comparison_v1/stage299/block_summary.csv"
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
