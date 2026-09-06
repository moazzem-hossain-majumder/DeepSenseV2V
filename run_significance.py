"""
Standalone script to recompute the paired trajectory-block bootstrap
significance test (P3 vs P1) using real per-sample inference.
Overwrites results/significance_tests.csv.
"""
import os, sys, json, types
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from src.dataset import prepare_multimodal_data
from src.paths import resolve_raw_data_root
from src.models import create_model
from src.evaluate import (
    compute_topk_accuracy,
    trajectory_block_bootstrap_ci,
    paired_trajectory_bootstrap_diff,
)

# ── config ──────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
mcfg = cfg.get("model", {})
model_kwargs = dict(
    d_model=mcfg.get("hidden_dim", 256),
    fusion_heads=mcfg.get("fusion_heads", 8),
    fusion_layers=mcfg.get("fusion_layers", 3),
    freeze_until=mcfg.get("freeze_backbone_until", "layer2"),
    dropout=mcfg.get("dropout", 0.12),
    n_beams=mcfg.get("n_beams", 256),
    gru_layers=mcfg.get("gru_layers", 3),
    head_hidden=mcfg.get("head_hidden", 512),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ── load dataset ─────────────────────────────────────────────────────────────
print("Loading dataset...", flush=True)
raw_root = resolve_raw_data_root(r"D:\DeepSense_V2V\data")
datasets, _, _ = prepare_multimodal_data(data_root=raw_root, img_size=(96, 96))
test_loader = DataLoader(
    datasets["test"], batch_size=256, shuffle=False,
    num_workers=0, pin_memory=(device.type == "cuda")
)

# ── helpers ──────────────────────────────────────────────────────────────────
def eval_on_test(model_name, seed):
    ckpt_path = os.path.join("results", "checkpoints", f"best_model_{model_name}_seed{seed}.pt")
    mdl = create_model(model_name, **model_kwargs)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.to(device).eval()
    all_logits, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            gps = batch["gps"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = mdl(rgb, gps)
            all_logits.append(out["logits"].float().cpu().numpy())
            all_labels.append(batch["beam_label"].numpy())
            all_seqs.append(batch["seq_index"].numpy())
    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        np.concatenate(all_seqs),
    )

# ── inference ────────────────────────────────────────────────────────────────
# Use seed 42 for both models — consistent with the main results table and
# avoids test-set peeking that would occur if we selected by best test Top-1.
p3_seed = 42
print(f"Loading P3 seed={p3_seed} checkpoint...", flush=True)
p3_logits, p3_labels, p3_seqs = eval_on_test("P3", p3_seed)
p3_top1 = compute_topk_accuracy(p3_logits, p3_labels)["top1"]
print(f"P3 test Top-1: {p3_top1*100:.2f}%", flush=True)

p1_seed = 42
print(f"Loading P1 seed={p1_seed} checkpoint...", flush=True)
p1_logits, p1_labels, p1_seqs = eval_on_test("P1", p1_seed)
p1_top1 = compute_topk_accuracy(p1_logits, p1_labels)["top1"]
print(f"P1 test Top-1: {p1_top1*100:.2f}%", flush=True)

assert np.array_equal(p3_seqs, p1_seqs), "seq_index mismatch between P3 and P1 test sets!"

# ── paired bootstrap ─────────────────────────────────────────────────────────
print("Running paired block bootstrap (n_boot=1000)...", flush=True)
diff = paired_trajectory_bootstrap_diff(
    metric_fn_a=lambda rows: compute_topk_accuracy(p3_logits[rows], p3_labels[rows])["top1"],
    metric_fn_b=lambda rows: compute_topk_accuracy(p1_logits[rows], p3_labels[rows])["top1"],
    seq_indices=p3_seqs,
    n_boot=1000,
    alpha_ci=0.05,
    seed=42,
)
p3_ci = trajectory_block_bootstrap_ci(
    lambda rows: compute_topk_accuracy(p3_logits[rows], p3_labels[rows])["top1"],
    p3_seqs, n_boot=1000, seed=42,
)
p1_ci = trajectory_block_bootstrap_ci(
    lambda rows: compute_topk_accuracy(p1_logits[rows], p3_labels[rows])["top1"],
    p3_seqs, n_boot=1000, seed=42,
)

# ── save ─────────────────────────────────────────────────────────────────────
with open("results/results_P3.json") as f:
    p3_runs = json.load(f)
with open("results/results_P1.json") as f:
    p1_runs = json.load(f)
seeds_used = sorted(set(
    [r["seed"] for r in p3_runs if "seed" in r] +
    [r["seed"] for r in p1_runs if "seed" in r]
))

sig_rows = [
    {
        "comparison": "P3_top1_block_bootstrap",
        "mean": p3_ci["mean"],
        "ci_95_low": p3_ci["ci_lower"],
        "ci_95_high": p3_ci["ci_upper"],
        "seeds_trained": str(seeds_used),
        "resample_unit": "trajectory block (seq_index), not frames",
    },
    {
        "comparison": "P1_top1_block_bootstrap",
        "mean": p1_ci["mean"],
        "ci_95_low": p1_ci["ci_lower"],
        "ci_95_high": p1_ci["ci_upper"],
        "seeds_trained": str(seeds_used),
        "resample_unit": "trajectory block (seq_index), not frames",
    },
    {
        "comparison": "P3_minus_P1_top1_paired_block_bootstrap",
        "mean": diff["mean_diff"],
        "ci_95_low": diff["ci_95"][0],
        "ci_95_high": diff["ci_95"][1],
        "seeds_trained": str(seeds_used),
        "resample_unit": "trajectory block (seq_index), paired",
        "p3_seed": p3_seed,
        "p1_seed": p1_seed,
        "n_boot": 1000,
        "significant": bool(diff["ci_95"][0] > 0.0 or diff["ci_95"][1] < 0.0),
    },
]

pd.DataFrame(sig_rows).to_csv("results/significance_tests.csv", index=False)

lo, hi = diff["ci_95"][0] * 100, diff["ci_95"][1] * 100
print(
    f"Paired diff = {diff['mean_diff']*100:+.2f}%  "
    f"95% CI = [{lo:+.2f}%, {hi:+.2f}%]  "
    f"Significant = {sig_rows[-1]['significant']}",
    flush=True,
)
print("Saved results/significance_tests.csv", flush=True)
