"""
Regenerate data/processed/eda_figures/fig_model_comparison.png with corrections:
  1. Profile MAE and Rank Corr bars suppressed (set to NaN) for models without
     a profile head (B0, B1, B3, P1).
  2. Dashed reference line label changed from "Mean-profile floor" to
     "Mean-profile baseline".
Reads from results/ablation_table.csv (no raw dataset required).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── load data ────────────────────────────────────────────────────────────────
abl = pd.read_csv("results/ablation_table.csv")
# Reorder to match the figure's display order
model_order = ["B1", "B3", "P1", "P3", "B0"]
abl["model"] = pd.Categorical(abl["model"], categories=model_order, ordered=True)
abl = abl.sort_values("model").reset_index(drop=True)

# Models WITHOUT a profile head — suppress MAE and rank-corr
NO_PROFILE_HEAD = {"B0", "B1", "B3", "P1"}
for col in ("profile_mae_db", "profile_rank_corr"):
    abl.loc[abl["model"].isin(NO_PROFILE_HEAD), col] = np.nan

maj_top1   = float(abl["majority_baseline_top1"].iloc[0])
mean_baseline = float(abl["mean_profile_mae_db"].iloc[0])   # 3.11 dB

# ── colour palette (matches original figure) ─────────────────────────────────
palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# ── subplots ──────────────────────────────────────────────────────────────────
metrics = [
    # (column,             y-label,                    pct?,  ref_val,      ref_label)
    ("test_top1",     "Top-1 Accuracy (%)",        True,  maj_top1*100, "Majority Top-1"),
    ("test_top5",     "Top-5 Accuracy (%)",        True,  maj_top1*100, "Majority Top-1"),
    ("test_top13",    "Top-13 Accuracy (%)",       True,  maj_top1*100, "Majority Top-1"),
    ("apl_db",        "Avg Power Loss (dB)",       False, None,         None),
    ("profile_mae_db","Profile MAE (dB)",          False, mean_baseline,"Mean-profile baseline"),
    ("profile_rank_corr","Profile Rank Corr",      True,  None,         None),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, (col, ylabel, as_pct, ref_val, ref_label) in zip(axes, metrics):
    vals = abl[col].values.copy()
    if as_pct:
        vals = vals * 100

    bars = ax.bar(
        abl["model"], vals,
        color=palette[:len(abl)],
        edgecolor="black", linewidth=0.6,
    )

    # Value labels on top of each bar (skip NaN)
    for bar, v in zip(bars, vals):
        if np.isnan(v):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    if ref_val is not None:
        ax.axhline(ref_val, color="black", linestyle="--", linewidth=1,
                   label=ref_label)
        ax.legend(fontsize=8)

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.suptitle(
    "Model Performance Comparison (Scenario 36, seed=42 best checkpoint)",
    fontsize=13,
)
plt.tight_layout()

out_path = os.path.join("data", "processed", "eda_figures", "fig_model_comparison.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved {out_path}")
