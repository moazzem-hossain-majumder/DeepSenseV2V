import json
import numpy as np
import pandas as pd
from src.acquire_scenarios import save_json
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.swin_comparison_plan import ROOT, PLAN
from src.variation_analysis import bootstrap


def report(folder):
    manifest = pd.read_json(CACHE / "manifest.jsonl", lines=True)
    frame = manifest.loc[manifest.split.eq("val"), ["scenario", "block_key", "seq_index"]].copy()
    records = []
    for run in PLAN:
        if run["kind"] != "train":
            continue
        held = run["holdout"]
        subset = frame[frame.scenario.eq(held)] if held else frame
        filename = "unseen_predictions.npz" if held else "val_predictions.npz"
        split = "transfer" if held else "pooled"
        name = f"without{held}_unseen" if held else "pooled_val"
        arm = "rgb" if run["vision"] else "gps"
        with np.load(ROOT / run["directory"] / filename) as pred, np.load(
                ROOT / f"results/conservative_v1/stage44/baseline_{name}.npz") as geo:
            for field in ("indices", "profiles", "labels"):
                np.testing.assert_array_equal(pred[field], geo[field])
            np.testing.assert_array_equal(pred["indices"], subset.index.to_numpy())
            for label, data in [(arm, pred), ("geometry", geo)]:
                metrics = per_sample(data["logits"], data["profiles"], data["labels"])
                metrics["top1_loss_db"] = per_sample(data["logits"], data["profiles"], data["labels"], 1)["regret"]
                for metric, values in metrics.items():
                    col = f"{split}_{label}_{metric}_{run['seed']}"
                    frame.loc[subset.index, col] = values.astype(float)
    if frame.isna().any().any():
        raise ValueError("Missing paired values")
    for metric in ("hit", "success", "regret", "top1_loss_db"):
        for split in ("pooled", "transfer"):
            for arm in ("rgb", "gps", "geometry"):
                cols = [f"{split}_{arm}_{metric}_{s}" for s in (42, 7, 13)]
                frame[f"{split}_{arm}"] = frame[cols].mean(axis=1)
            for comparator in ("gps", "geometry"):
                # bootstrap() returns 100 times the mean difference.
                factor = 100 if metric in ("hit", "success") else 1
                frame["effect"] = (frame[f"{split}_rgb"] - frame[f"{split}_{comparator}"]) * factor / 100
                by = 100 * frame.groupby("scenario").effect.mean()
                records.append(dict(metric=metric, split=split, comparator=comparator,
                    difference=by.mean(), by_scenario=by.to_dict(), interval=bootstrap(frame, "block_key")))
            frame[f"{split}_gain"] = frame[f"{split}_rgb"] - frame[f"{split}_gps"]
        frame["effect"] = (frame.pooled_gain - frame.transfer_gain) * factor / 100
        by = 100 * frame.groupby("scenario").effect.mean()
        records.append(dict(metric=metric, split="pooled_minus_transfer", comparator="matched_gps_gain",
            difference=by.mean(), by_scenario=by.to_dict(), interval=bootstrap(frame, "block_key")))
    save_json(folder / "contrasts.json", records)
    (folder / "REPORT.md").write_text("# Shifted-window adaptation on development validation\n\n" +
        "\n".join(f"- {r['metric']} / {r['split']} / {r['comparator']}: {r['difference']:+.3f} "
                  f"[{r['interval'][0]:+.3f}, {r['interval'][1]:+.3f}]" for r in records) +
        "\n\nAccuracy differences in percentage points. Power-loss differences in dB. Paired block intervals condition on four explored scenarios.\n")
