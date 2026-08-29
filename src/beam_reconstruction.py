import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_ARR = 4
N_BEAMS_PER_ARR = 64
N_BEAMS_TOTAL = N_ARR * N_BEAMS_PER_ARR  # 256
EPS_LINEAR = 1e-12

def reconstruct_256_power_vector(unit1_pwr1, unit1_pwr2, unit1_pwr3, unit1_pwr4):
    """
    Reconstruct full 256-beam power vector from 4 subarrays of 64 beams each.
    Layout: [pwr1_0..pwr1_63, pwr2_0..pwr2_63, pwr3_0..pwr3_63, pwr4_0..pwr4_63]
    """
    pwr_arrs = [np.asarray(p, dtype=np.float32) for p in [unit1_pwr1, unit1_pwr2, unit1_pwr3, unit1_pwr4]]
    return np.concatenate(pwr_arrs, axis=-1)

def linear_to_db(linear_pwr):
    """Convert linear power values to dB scale (10 * log10(P + eps))."""
    return 10.0 * np.log10(np.maximum(linear_pwr, EPS_LINEAR))

def compute_power_gap_db(power_profile_db):
    """
    Compute power gap relative to the maximum beam in the profile:
    gap_b = max(P_db) - P_db(b) >= 0 dB.
    """
    max_pwr = np.max(power_profile_db, axis=-1, keepdims=True)
    return max_pwr - power_profile_db

def verify_reconstruction_and_feasibility(data_root=".", output_dir="results/eda"):
    """
    Phase 1 Feasibility Gate:
    1. Load scenario36 pickle and CSV
    2. Reconstruct 256-vector for all samples
    3. Verify argmax == unit1_overall-beam
    4. Compute 1 dB and 3 dB near-optimal beam counts
    5. Check best-beam class distribution and majority baseline
    """
    from src.paths import resolve_raw_data_root, scenario36_pkl_path
    os.makedirs(output_dir, exist_ok=True)
    raw_root = resolve_raw_data_root(data_root)
    pkl_path = scenario36_pkl_path(raw_root)

    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as fp:
        p_dict = pickle.load(fp)

    n_samples = len(p_dict["abs_index"])
    print(f"Total samples in pickle: {n_samples:,}")

    # Reconstruct all 256-vectors
    pwr1 = np.array(p_dict["unit1_pwr1"])
    pwr2 = np.array(p_dict["unit1_pwr2"])
    pwr3 = np.array(p_dict["unit1_pwr3"])
    pwr4 = np.array(p_dict["unit1_pwr4"])

    full_pwr_linear = np.concatenate([pwr1, pwr2, pwr3, pwr4], axis=1)  # (N, 256)
    full_pwr_db = linear_to_db(full_pwr_linear)

    # Argmax check
    recon_argmax = np.argmax(full_pwr_linear, axis=1)
    true_labels = np.array(p_dict["unit1_overall-beam"])

    match_count = np.sum(recon_argmax == true_labels)
    match_rate = match_count / n_samples
    print(f"\n[Feasibility Gate 1] 256-Beam Argmax Match Rate: {match_rate * 100:.2f}% ({match_count:,} / {n_samples:,})")
    assert match_rate == 1.0, f"Error: Argmax mismatch detected! Match rate = {match_rate}"

    # Near-optimal beam distribution
    gaps_db = compute_power_gap_db(full_pwr_db)
    beams_within_1db = np.sum(gaps_db <= 1.0, axis=1)
    beams_within_3db = np.sum(gaps_db <= 3.0, axis=1)

    print("\n[Feasibility Gate 2] Near-Optimal Beam Set Sizes:")
    print(f"  Within 1 dB: Mean = {np.mean(beams_within_1db):.2f}, Median = {np.median(beams_within_1db):.1f}, Max = {np.max(beams_within_1db)}")
    print(f"  Within 3 dB: Mean = {np.mean(beams_within_3db):.2f}, Median = {np.median(beams_within_3db):.1f}, Max = {np.max(beams_within_3db)}")

    # Class imbalance / Majority class baseline
    classes, counts = np.unique(true_labels, return_counts=True)
    maj_idx = classes[np.argmax(counts)]
    maj_acc = np.max(counts) / n_samples
    print("\n[Feasibility Gate 3] Best-Beam Class Distribution:")
    print(f"  Active beam classes: {len(classes)} / {N_BEAMS_TOTAL}")
    print(f"  Majority beam class: Beam #{maj_idx} ({maj_acc * 100:.2f}% of samples)")

    # Plot Near-Optimal Set Size Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(beams_within_1db, bins=np.arange(0.5, 30.5, 1), alpha=0.7, label=r"Within 1 dB ($\Delta=1$)")
    plt.hist(beams_within_3db, bins=np.arange(0.5, 30.5, 1), alpha=0.7, label=r"Within 3 dB ($\Delta=3$)")
    plt.xlabel("Number of Near-Optimal Beams", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    plt.title("Feasibility Gate: Near-Optimal Beam Count Distribution (Scenario 36)", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "near_optimal_set_sizes.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved feasibility figure to {plot_path}")

    plt.figure(figsize=(10, 4))
    plt.bar(classes, counts, width=1.0, color="#1f77b4")
    plt.xlabel("Best-beam class (0-255)")
    plt.ylabel("Count")
    plt.title("Phase 1: Best-beam class imbalance (Scenario 36)")
    plt.tight_layout()
    class_plot = os.path.join(output_dir, "beam_class_distribution.png")
    plt.savefig(class_plot, dpi=200)
    plt.close()

    import json
    stats = {
        "match_rate": float(match_rate),
        "majority_class_idx": int(maj_idx),
        "majority_class_acc": float(maj_acc),
        "mean_beams_1db": float(np.mean(beams_within_1db)),
        "median_beams_1db": float(np.median(beams_within_1db)),
        "mean_beams_3db": float(np.mean(beams_within_3db)),
        "median_beams_3db": float(np.median(beams_within_3db)),
        "active_classes": int(len(classes)),
        "total_samples": int(n_samples),
        "units": "pickle unit1_pwr* are linear power; APL uses 10*log10 after reconstruction",
    }
    with open(os.path.join(output_dir, "phase1_feasibility.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats

if __name__ == "__main__":
    verify_reconstruction_and_feasibility()
