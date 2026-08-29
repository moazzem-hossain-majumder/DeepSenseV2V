"""
Generate Complete Annotated EDA Dataset and Publication Figures for Scenario 36.
Uses the trained best model (P3_MultiTaskProfile) to annotate all 24,179 sequence samples
with top-k predictions, power loss, profile MAE, and conformal candidate sets.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.dataset import prepare_multimodal_data, get_dataloaders
from src.models import create_model
from src.candidate_sets import StaticConformalRiskControl
from src.online_controller import run_online_aci_controller, select_best_eta_on_val
from src.paths import resolve_raw_data_root

# Styling for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def resolve_checkpoint(explicit_path=None, output_dir="results"):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend([
        os.path.join(output_dir, "checkpoints", "best_model_P3_seed42.pt"),
        os.path.join("results", "checkpoints", "best_model_P3_seed42.pt"),
        os.path.join("results_rtx5070", "checkpoints", "best_model_P3_seed42.pt"),
        os.path.join("checkpoints", "best_model_P3_seed42.pt"),
    ])
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No trained checkpoint found. Train first with `python train.py --config config.yaml`, "
        "or pass --checkpoint PATH."
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU inference + merged annotated EDA figures")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 80)
    print("GENERATING COMPLETE ANNOTATED EDA DATASET WITH MODEL INFERENCE")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    import yaml
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    mcfg = cfg.get("model", {})
    img_size = tuple(cfg.get("data", {}).get("image_size", [96, 96]))
    model_kwargs = {
        "d_model": mcfg.get("hidden_dim", 256),
        "fusion_heads": mcfg.get("fusion_heads", 8),
        "fusion_layers": mcfg.get("fusion_layers", 3),
        "freeze_until": mcfg.get("freeze_backbone_until", "layer2"),
        "dropout": mcfg.get("dropout", 0.12),
        "n_beams": mcfg.get("n_beams", 256),
        "gru_layers": mcfg.get("gru_layers", 3),
        "head_hidden": mcfg.get("head_hidden", 512),
    }

    print(f"\n[1/5] Loading Multimodal Dataset (img_size={img_size})...")
    raw_root = resolve_raw_data_root(args.data_root)
    datasets, metadata_df, raw_p_dict = prepare_multimodal_data(data_root=raw_root, img_size=img_size)

    # 2. Load Trained Best Model
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.output_dir)
    
    print(f"\n[2/5] Loading Best Trained Checkpoint: {checkpoint_path}...")
    model = create_model("P3", **model_kwargs)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from Epoch {ckpt['epoch']} (Best Val Top-1: {ckpt.get('best_val_top1', 0.0)*100:.2f}%)")

    # 3. Perform Inference Across All Splits
    print("\n[3/5] Performing High-Speed In-Memory Inference on All 24,179 Sequences...")
    from torch.utils.data import DataLoader
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=256, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()),
        "val": DataLoader(datasets["val"], batch_size=256, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()),
        "calib": DataLoader(datasets["calib"], batch_size=256, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()),
        "test": DataLoader(datasets["test"], batch_size=256, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    }

    split_predictions = {}
    with torch.no_grad():
        for split_name, loader in loaders.items():
            t0 = time.time()
            all_logits = []
            all_profiles = []
            all_labels = []
            all_true_profs = []
            all_seq_indices = []

            for batch in loader:
                rgb = batch["rgb"].to(device, non_blocking=True)
                gps = batch["gps"].to(device, non_blocking=True)
                labels = batch["beam_label"]
                true_profs = batch["profile_db"]
                seq_idx = batch["seq_index"]

                if torch.cuda.is_available():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model(rgb, gps)
                else:
                    out = model(rgb, gps)

                all_logits.append(out["logits"].cpu().float().numpy())
                all_profiles.append(out["pred_profile"].cpu().float().numpy())
                all_labels.append(labels.numpy())
                all_true_profs.append(true_profs.numpy())
                all_seq_indices.append(seq_idx.numpy())

            split_predictions[split_name] = {
                "logits": np.concatenate(all_logits, axis=0),
                "pred_profiles": np.concatenate(all_profiles, axis=0),
                "true_labels": np.concatenate(all_labels, axis=0),
                "true_profiles": np.concatenate(all_true_profs, axis=0),
                "seq_indices": np.concatenate(all_seq_indices, axis=0)
            }
            print(f"  Split '{split_name}': {len(all_labels)} sequences evaluated in {time.time()-t0:.2f}s")

    # 4. Conformal Calibration
    print("\n[4/5] Computing Conformal Prediction Risk Quantiles & ACI Sets...")
    calib_pred = split_predictions["calib"]["pred_profiles"]
    calib_true = split_predictions["calib"]["true_profiles"]
    crc = StaticConformalRiskControl(target_alpha=0.10, delta_db=3.0)
    q_hat = crc.fit(calib_pred, calib_true)
    print(f"  Static CRC Calibrated Quantile q_hat = {q_hat:.2f} dB (target alpha = 0.10, delta = 3.0 dB)")

    # 5. Build Enriched Annotations DataFrame
    print("\n[5/5] Assembling Master Annotated DataFrame & Computing Rich Metrics...")
    annotated_rows = []
    
    for split_name, preds in split_predictions.items():
        dataset_obj = datasets[split_name]
        seq_df = dataset_obj.seq_df.copy()
        
        logits = preds["logits"]
        pred_profs = preds["pred_profiles"]
        true_labels = preds["true_labels"]
        true_profs = preds["true_profiles"]

        # Top-k indices
        top1_preds = np.argmax(logits, axis=1)
        top3_preds = np.argsort(-logits, axis=1)[:, :3]
        top5_preds = np.argsort(-logits, axis=1)[:, :5]
        top13_preds = np.argsort(-logits, axis=1)[:, :13]

        # Top-k correctness
        top1_correct = (top1_preds == true_labels)
        top3_correct = np.array([true_labels[i] in top3_preds[i] for i in range(len(true_labels))])
        top5_correct = np.array([true_labels[i] in top5_preds[i] for i in range(len(true_labels))])
        top13_correct = np.array([true_labels[i] in top13_preds[i] for i in range(len(true_labels))])

        # RF Powers and APL
        opt_powers = np.array([true_profs[i, true_labels[i]] for i in range(len(true_labels))])
        pred_powers = np.array([true_profs[i, top1_preds[i]] for i in range(len(true_labels))])
        power_losses = opt_powers - pred_powers  # dB >= 0

        # Continuous profile metrics per sample
        profile_maes = np.mean(np.abs(pred_profs - true_profs), axis=1)
        profile_rmses = np.sqrt(np.mean((pred_profs - true_profs) ** 2, axis=1))
        
        profile_corrs = []
        for i in range(len(true_labels)):
            c, _ = spearmanr(pred_profs[i], true_profs[i])
            profile_corrs.append(0.0 if np.isnan(c) else float(c))
        profile_corrs = np.array(profile_corrs)

        # Conformal Candidate Sets
        conf_set_sizes, conf_covered = [], []
        for i in range(len(true_labels)):
            max_pred_p = np.max(pred_profs[i])
            # Set of beams within q_hat of predicted maximum
            c_set = np.where(pred_profs[i] >= (max_pred_p - q_hat))[0]
            conf_set_sizes.append(len(c_set))
            # True power of best beam in candidate set vs optimal
            best_cand_power = np.max(true_profs[i, c_set]) if len(c_set) > 0 else -100.0
            is_cov = (opt_powers[i] - best_cand_power) <= 3.0
            conf_covered.append(bool(is_cov))

        # Build records
        for i in range(len(seq_df)):
            row = seq_df.iloc[i]
            y_idx = row["y_index"]
            raw_meta = metadata_df.iloc[y_idx]

            # GPS Kinematic Features
            u1_x, u1_y = raw_meta.get("unit1_loc_x", 0.0), raw_meta.get("unit1_loc_y", 0.0)
            u2_x, u2_y = raw_meta.get("unit2_loc_x", 0.0), raw_meta.get("unit2_loc_y", 0.0)
            e_rel = u2_x - u1_x
            n_rel = u2_y - u1_y
            dist = np.sqrt(e_rel**2 + n_rel**2)
            speed = np.sqrt(raw_meta.get("unit1_speed_x", 0.0)**2 + raw_meta.get("unit1_speed_y", 0.0)**2)
            rot_z = raw_meta.get("unit1_rot_z", 0.0)

            annotated_rows.append({
                "split": split_name,
                "traj_id": int(row["seq_index"]),
                "seq_index": int(row["seq_index"]),
                "y_frame_idx": int(y_idx),
                "rgb_path": str(raw_meta.get("unit1_rgb5", "")),
                # GPS Kinematics
                "dist_m": float(dist),
                "rel_east_m": float(e_rel),
                "rel_north_m": float(n_rel),
                "speed_mps": float(speed),
                "heading_deg": float(rot_z),
                # True Ground Truth
                "true_best_beam": int(true_labels[i]),
                "optimal_power_db": float(opt_powers[i]),
                # Model Predictions
                "pred_best_beam_top1": int(top1_preds[i]),
                "pred_top3_beams": [int(b) for b in top3_preds[i]],
                "pred_top5_beams": [int(b) for b in top5_preds[i]],
                "pred_top13_beams": [int(b) for b in top13_preds[i]],
                # Accuracy & Error
                "is_top1_correct": bool(top1_correct[i]),
                "is_top3_correct": bool(top3_correct[i]),
                "is_top5_correct": bool(top5_correct[i]),
                "is_top13_correct": bool(top13_correct[i]),
                "beam_index_error": int(abs(top1_preds[i] - true_labels[i])),
                "predicted_beam_power_db": float(pred_powers[i]),
                "power_loss_db": float(power_losses[i]),
                # Continuous Profile Metrics
                "profile_mae_db": float(profile_maes[i]),
                "profile_rmse_db": float(profile_rmses[i]),
                "profile_rank_corr": float(profile_corrs[i]),
                # Conformal Prediction
                "conformal_set_size": int(conf_set_sizes[i]),
                "conformal_is_covered": bool(conf_covered[i])
            })

    master_df = pd.DataFrame(annotated_rows)
    print(f"Master Annotated DataFrame created with shape: {master_df.shape}")

    # Save to disk
    os.makedirs("data/processed", exist_ok=True)
    out_csv = "data/processed/annotated_scenario36_with_predictions.csv"
    master_df.to_csv(out_csv, index=False)
    print(f"Saved complete annotated dataset to {out_csv} ({os.path.getsize(out_csv)/(1024**2):.1f} MB)")

    # -------------------------------------------------------------
    # 6. Generate Publication-Quality Visualizations
    # -------------------------------------------------------------
    fig_dir = "data/processed/eda_figures"
    os.makedirs(fig_dir, exist_ok=True)
    print(f"\n[6/6] Generating Publication EDA Figures into '{fig_dir}'...")

    # Figure 1: True vs Predicted Beam Distribution
    plt.figure(figsize=(12, 5))
    plt.hist(master_df["true_best_beam"], bins=64, range=(0, 256), alpha=0.6, color="#1f77b4", label="Ground Truth Optimal Beam", density=True)
    plt.hist(master_df["pred_best_beam_top1"], bins=64, range=(0, 256), alpha=0.6, color="#ff7f0e", label="Model Predicted Top-1 Beam", density=True)
    plt.title("Figure 1: Ground Truth vs Model Predicted 256-Beam Distribution")
    plt.xlabel("Millimeter-Wave Beam Index (0 - 255)")
    plt.ylabel("Probability Density")
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(fig_dir, "fig1_beam_distribution_comparison.png"))
    plt.close()
    print("  -> Saved fig1_beam_distribution_comparison.png")

    # Figure 2: Average Power Loss (APL) CDF by Split
    plt.figure(figsize=(10, 6))
    colors = {"train": "#2ca02c", "val": "#1f77b4", "calib": "#9467bd", "test": "#d62728"}
    for split_name, df_sub in master_df.groupby("split"):
        sorted_losses = np.sort(df_sub["power_loss_db"])
        cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
        plt.plot(sorted_losses, cdf, label=f"{split_name.capitalize()} Split (Mean APL: {df_sub['power_loss_db'].mean():.2f} dB)", color=colors.get(split_name, "black"), linewidth=2)
    plt.axvline(3.0, color="gray", linestyle=":", label="3.0 dB Target Risk Threshold (Delta)")
    plt.title("Figure 2: Empirical Cumulative Distribution Function (ECDF) of Power Loss (APL)")
    plt.xlabel("Average Power Loss (dB)")
    plt.ylabel("Cumulative Probability P(Loss <= x)")
    plt.xlim(0, 30)
    plt.ylim(0, 1.02)
    plt.legend(frameon=True, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(fig_dir, "fig2_power_loss_distribution_by_split.png"))
    plt.close()
    print("  -> Saved fig2_power_loss_distribution_by_split.png")

    # Figure 3: Accuracy vs Distance and Speed
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Distance Bins
    master_df["dist_bin"] = pd.cut(master_df["dist_m"], bins=[0, 10, 20, 30, 50, 100], labels=["0-10m", "10-20m", "20-30m", "30-50m", ">50m"])
    dist_perf = master_df.groupby("dist_bin", observed=False)[["is_top1_correct", "is_top5_correct"]].mean() * 100
    dist_perf.plot(kind="bar", ax=axes[0], color=["#1f77b4", "#2ca02c"], edgecolor="black")
    axes[0].set_title("Accuracy vs Relative Vehicle Distance")
    axes[0].set_xlabel("Inter-Vehicle Distance (m)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(["Top-1 Accuracy", "Top-5 Accuracy"])
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Speed Bins
    master_df["speed_bin"] = pd.cut(master_df["speed_mps"], bins=[-1, 5, 10, 15, 25], labels=["<5 m/s", "5-10 m/s", "10-15 m/s", ">15 m/s"])
    speed_perf = master_df.groupby("speed_bin", observed=False)[["is_top1_correct", "is_top5_correct"]].mean() * 100
    speed_perf.plot(kind="bar", ax=axes[1], color=["#ff7f0e", "#9467bd"], edgecolor="black")
    axes[1].set_title("Accuracy vs Vehicle Speed")
    axes[1].set_xlabel("Vehicle Speed (m/s)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(["Top-1 Accuracy", "Top-5 Accuracy"])
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.suptitle("Figure 3: Multi-Modal Model Performance across Physical Kinematic Regimes", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_accuracy_vs_distance_and_speed.png"))
    plt.close()
    print("  -> Saved fig3_accuracy_vs_distance_and_speed.png")

    # Figure 4: Conformal Candidate Set Size vs Coverage
    plt.figure(figsize=(10, 5))
    test_df = master_df[master_df["split"] == "test"]
    sns.histplot(test_df["conformal_set_size"], bins=40, kde=True, color="#9467bd", edgecolor="black")
    plt.axvline(test_df["conformal_set_size"].mean(), color="red", linestyle="--", label=f"Mean Set Size: {test_df['conformal_set_size'].mean():.1f} beams")
    plt.axvline(256 * 0.10, color="green", linestyle=":", label="10% of Full Codebook (25.6 beams)")
    plt.title(f"Figure 4: Conformal Risk Control Prediction Set Size Distribution (Test Split, q={q_hat:.2f} dB)")
    plt.xlabel("Candidate Set Size |C(X)| (Number of Recommended Beams out of 256)")
    plt.ylabel("Sample Count")
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(fig_dir, "fig4_conformal_candidate_set_sizes.png"))
    plt.close()
    print("  -> Saved fig4_conformal_candidate_set_sizes.png")

    # Figure 5: Continuous Profile MAE vs Spearman Rank Correlation
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=master_df.sample(min(3000, len(master_df)), random_state=42), x="profile_mae_db", y="profile_rank_corr", hue="split", alpha=0.5, palette=colors)
    plt.title("Figure 5: Power Profile Reconstruction Fidelity: MAE vs Spearman Rank Correlation")
    plt.xlabel("Profile Mean Absolute Error (dB)")
    plt.ylabel("Profile Spearman Rank Correlation")
    plt.legend(frameon=True, title="Split")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(fig_dir, "fig5_profile_mae_vs_rank_correlation.png"))
    plt.close()
    print("  -> Saved fig5_profile_mae_vs_rank_correlation.png")

    # Figure 6: Chronological Trajectory Beam & Power Tracking Sample
    sample_traj = master_df[master_df["split"] == "test"]["traj_id"].iloc[0]
    traj_sub = master_df[master_df["traj_id"] == sample_traj].sort_values("seq_index")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    # Beams
    axes[0].plot(range(len(traj_sub)), traj_sub["true_best_beam"], label="Ground Truth Optimal Beam", color="#1f77b4", marker="o", markersize=4, linestyle="-")
    axes[0].plot(range(len(traj_sub)), traj_sub["pred_best_beam_top1"], label="Predicted Top-1 Beam", color="#ff7f0e", marker="x", markersize=5, linestyle="--")
    axes[0].set_ylabel("Beam Index (0-255)")
    axes[0].set_title(f"Figure 6: Time-Series Beam & Power Tracking over Test Trajectory ID: {sample_traj}")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Powers
    axes[1].plot(range(len(traj_sub)), traj_sub["optimal_power_db"], label="Optimal Beam Power (dB)", color="#2ca02c", linewidth=2)
    axes[1].plot(range(len(traj_sub)), traj_sub["predicted_beam_power_db"], label="Predicted Beam Power (dB)", color="#d62728", linestyle="--", linewidth=2)
    axes[1].fill_between(range(len(traj_sub)), traj_sub["predicted_beam_power_db"], traj_sub["optimal_power_db"], color="gray", alpha=0.2, label="Power Loss (APL)")
    axes[1].set_xlabel("Chronological Sequence Step in Trajectory")
    axes[1].set_ylabel("Received Power (dB)")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig6_trajectory_temporal_tracking_sample.png"))
    plt.close()
    print("  -> Saved fig6_trajectory_temporal_tracking_sample.png")

    # Figure 7: 2D Spatial Heatmap of Power Loss
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(master_df["rel_east_m"], master_df["rel_north_m"], c=np.clip(master_df["power_loss_db"], 0, 25), cmap="viridis", alpha=0.6, s=15)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Average Power Loss (dB, clipped at 25 dB)")
    plt.title("Figure 7: 2D Spatial Distribution of Power Loss vs Relative Vehicle Positions")
    plt.xlabel("Relative East Displacement (m)")
    plt.ylabel("Relative North Displacement (m)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(fig_dir, "fig7_spatial_error_heatmap.png"))
    plt.close()
    print("  -> Saved fig7_spatial_error_heatmap.png")

    # Figure 8: Multimodal Benchmark Summary Bar Chart
    split_summary = master_df.groupby("split").agg({
        "is_top1_correct": lambda x: np.mean(x)*100,
        "is_top3_correct": lambda x: np.mean(x)*100,
        "is_top5_correct": lambda x: np.mean(x)*100,
        "is_top13_correct": lambda x: np.mean(x)*100,
        "power_loss_db": "mean",
        "profile_mae_db": "mean"
    }).loc[["train", "val", "calib", "test"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(split_summary))
    width = 0.18
    ax.bar(x - 1.5*width, split_summary["is_top1_correct"], width, label="Top-1 Accuracy (%)", color="#1f77b4")
    ax.bar(x - 0.5*width, split_summary["is_top3_correct"], width, label="Top-3 Accuracy (%)", color="#2ca02c")
    ax.bar(x + 0.5*width, split_summary["is_top5_correct"], width, label="Top-5 Accuracy (%)", color="#ff7f0e")
    ax.bar(x + 1.5*width, split_summary["is_top13_correct"], width, label="Top-13 Accuracy (%)", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(["Train Split\n(68 Traj)", "Validation Split\n(19 Traj)", "Calibration Split\n(19 Traj)", "Test Split\n(18 Traj)"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Figure 8: Top-k Beam Alignment Performance Across All Partition Splits")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig8_multimodal_performance_breakdown.png"))
    plt.close()
    print("  -> Saved fig8_multimodal_performance_breakdown.png")

    print("\n" + "=" * 80)
    print(f"EDA PROCESSING COMPLETE! Elapsed Time: {time.time()-start_time:.1f}s")
    print(f"Annotated dataset saved: {out_csv}")
    print(f"Publication figures saved in: {fig_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
