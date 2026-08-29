import numpy as np
import scipy.stats as stats
import pandas as pd
from tqdm import tqdm

def compute_topk_accuracy(pred_logits, true_labels, topk=(1, 3, 5)):
    """
    Compute Top-k classification accuracies.
    pred_logits: (N, 256), true_labels: (N,)
    """
    N = len(true_labels)
    sorted_preds = np.argsort(pred_logits, axis=1)
    accs = {}
    for k in topk:
        topk_preds = sorted_preds[:, -k:]
        hits = np.sum([true_labels[i] in topk_preds[i] for i in range(N)])
        accs[f"top{k}"] = float(hits / N)
    return accs

def compute_profile_metrics(pred_profile_db, true_profile_db):
    """
    Compute regression MAE, RMSE, and rank correlation across full 256-dim profiles.
    """
    mae = float(np.mean(np.abs(pred_profile_db - true_profile_db)))
    rmse = float(np.sqrt(np.mean((pred_profile_db - true_profile_db) ** 2)))

    # Spearman rank correlation per sample
    N = len(pred_profile_db)
    corrs = []
    for i in range(N):
        r, _ = stats.spearmanr(pred_profile_db[i], true_profile_db[i])
        if not np.isnan(r):
            corrs.append(r)
    rank_corr = float(np.mean(corrs)) if len(corrs) > 0 else 0.0

    return {
        "profile_mae_db": mae,
        "profile_rmse_db": rmse,
        "profile_rank_corr": rank_corr
    }

def compute_average_power_loss(pred_chosen_beams, true_profile_db):
    """
    Compute Average Power Loss (APL in dB) relative to optimal beam:
    APL = mean_i (max_b P_i(b) - P_i(chosen_beam_i))
    """
    N = len(pred_chosen_beams)
    best_pwrs = np.max(true_profile_db, axis=1)
    chosen_pwrs = np.array([true_profile_db[i, pred_chosen_beams[i]] for i in range(N)])
    gaps = best_pwrs - chosen_pwrs
    return float(np.mean(gaps)), gaps

def compute_multi_delta_reliability(candidate_sets, true_profile_db, deltas=(0.0, 1.0, 3.0)):
    """
    Compute inclusion / reliability at multiple delta_dB thresholds.
    """
    N = len(candidate_sets)
    best_powers = np.max(true_profile_db, axis=1)
    results = {}

    for delta in deltas:
        misses = 0
        for i in range(N):
            c = candidate_sets[i]
            if len(c) == 0:
                misses += 1
            else:
                cand_max = np.max(true_profile_db[i, c])
                if best_powers[i] - cand_max > delta:
                    misses += 1
        results[f"miss_rate_delta_{delta}db"] = float(misses / N)
        results[f"coverage_delta_{delta}db"] = float(1.0 - (misses / N))

    return results

def trajectory_block_bootstrap_ci(metric_fn, seq_indices, n_boot=1000, alpha_ci=0.05, seed=42):
    """
    Cluster/Block Bootstrap by unique trajectory run (seq_index).
    metric_fn: function(sample_indices) -> float metric value.
    """
    np.random.seed(seed)
    unique_seqs = np.unique(seq_indices)
    n_seqs = len(unique_seqs)

    # Pre-map seq_index to row indices
    seq_to_rows = {s: np.where(seq_indices == s)[0] for s in unique_seqs}

    boot_values = []
    for _ in range(n_boot):
        sampled_seqs = np.random.choice(unique_seqs, size=n_seqs, replace=True)
        sampled_rows = np.concatenate([seq_to_rows[s] for s in sampled_seqs])
        val = metric_fn(sampled_rows)
        boot_values.append(val)

    boot_values = np.array(boot_values)
    mean_val = float(np.mean(boot_values))
    std_val = float(np.std(boot_values))
    lower = float(np.percentile(boot_values, 100 * (alpha_ci / 2.0)))
    upper = float(np.percentile(boot_values, 100 * (1.0 - alpha_ci / 2.0)))

    return {
        "mean": mean_val,
        "std": std_val,
        "ci_95": [lower, upper],
        "ci_lower": lower,
        "ci_upper": upper
    }

def paired_trajectory_bootstrap_diff(metric_fn_a, metric_fn_b, seq_indices, n_boot=1000, alpha_ci=0.05, seed=42):
    """
    Paired Block-Bootstrap test for the difference (Model A - Model B).
    """
    np.random.seed(seed)
    unique_seqs = np.unique(seq_indices)
    n_seqs = len(unique_seqs)
    seq_to_rows = {s: np.where(seq_indices == s)[0] for s in unique_seqs}

    diffs = []
    for _ in range(n_boot):
        sampled_seqs = np.random.choice(unique_seqs, size=n_seqs, replace=True)
        sampled_rows = np.concatenate([seq_to_rows[s] for s in sampled_seqs])
        val_a = metric_fn_a(sampled_rows)
        val_b = metric_fn_b(sampled_rows)
        diffs.append(val_a - val_b)

    diffs = np.array(diffs)
    lower = float(np.percentile(diffs, 100 * (alpha_ci / 2.0)))
    upper = float(np.percentile(diffs, 100 * (1.0 - alpha_ci / 2.0)))

    return {
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs)),
        "ci_95": [lower, upper]
    }
