import os
import sys
import numpy as np
import pandas as pd

X_SIZE = 5  # 5 historical steps
Y_SIZE = 1  # 1 future target step

def create_sliding_window_sequences(df, x_size=X_SIZE, y_size=Y_SIZE):
    """
    Construct 5-step historical inputs -> 1-step target sequences grouped by trajectory run (`seq_index`).
    Returns DataFrame where each row contains the 5 historical indices and the 1 target index.
    """
    sequences = []
    # Ensure df is sorted by abs_index
    df = df.sort_values("abs_index").reset_index(drop=True)

    for seq_id, group in df.groupby("seq_index", sort=False):
        group_len = len(group)
        if group_len < x_size + y_size:
            continue
        
        group_indices = group.index.values
        group_abs = group["abs_index"].values
        group_times = group["timestamp"].values
        group_beams = group["unit1_overall-beam"].values if "unit1_overall-beam" in group else group["true_beam"].values

        for i in range(group_len - x_size - y_size + 1):
            x_idx = group_indices[i : i + x_size]
            y_idx = group_indices[i + x_size : i + x_size + y_size]
            
            record = {
                "seq_index": seq_id,
                "x_indices": list(x_idx),
                "y_index": int(y_idx[0]),
                "target_abs_index": int(group_abs[i + x_size]),
                "target_timestamp": str(group_times[i + x_size]),
                "target_beam": int(group_beams[i + x_size]),
            }
            sequences.append(record)

    seq_df = pd.DataFrame(sequences)
    print(f"Generated {len(seq_df):,} 5-past -> 1-future sequence windows across {seq_df['seq_index'].nunique()} trajectory runs.")
    return seq_df

def assign_leakage_free_splits(seq_df, raw_df, train_pct=0.55, val_pct=0.15, calib_pct=0.15, test_pct=0.15):
    """
    Assign trajectory runs (seq_index) to Train/Val/Calib/Test chronologically based on initial appearance.
    Enforces strict block separation: every trajectory run belongs to EXACTLY one split.
    """
    first_seen = raw_df.groupby("seq_index")["abs_index"].min().sort_values()
    run_order = first_seen.index.tolist()
    total_runs = len(run_order)

    n_train = int(np.round(train_pct * total_runs))
    n_val = int(np.round(val_pct * total_runs))
    n_calib = int(np.round(calib_pct * total_runs))
    n_test = total_runs - n_train - n_val - n_calib

    train_runs = set(run_order[:n_train])
    val_runs = set(run_order[n_train : n_train + n_val])
    calib_runs = set(run_order[n_train + n_val : n_train + n_val + n_calib])
    test_runs = set(run_order[n_train + n_val + n_calib :])

    def get_split(s_id):
        if s_id in train_runs:
            return "train"
        elif s_id in val_runs:
            return "val"
        elif s_id in calib_runs:
            return "calib"
        elif s_id in test_runs:
            return "test"
        return "unknown"

    seq_df["split"] = seq_df["seq_index"].apply(get_split)

    # Verification: Zero leakage
    check = seq_df.groupby("seq_index")["split"].nunique()
    assert (check == 1).all(), "Leakage Error: Some seq_index spans multiple splits!"

    print("\n[Split Assignment Summary]")
    for split_name in ["train", "val", "calib", "test"]:
        cnt = np.sum(seq_df["split"] == split_name)
        n_r = len(set(seq_df[seq_df["split"] == split_name]["seq_index"]))
        print(f"  {split_name.upper():5s}: {cnt:6,d} sequences ({cnt / len(seq_df) * 100:5.1f}%) across {n_r:3d} trajectory runs")

    return seq_df

def build_partition_manifest(data_root=".", output_dir="data/processed", guard_abs=25):
    """
    Full partitioning execution creating split_manifest.csv.

    Phase 2 blueprint: trajectory runs closer than `guard_abs` frames are merged
    into a single block via guard-interval union-find before split assignment.
    This prevents a trajectory that briefly pauses from being split across Train/Val.
    """
    from src.paths import resolve_raw_data_root, scenario36_csv_path
    os.makedirs(output_dir, exist_ok=True)
    raw_root = resolve_raw_data_root(data_root)
    csv_path = scenario36_csv_path(raw_root)

    df = pd.read_csv(csv_path)

    # Phase 2: merge close trajectory runs into contiguous blocks
    block_map = merge_close_trajectory_blocks(df, guard_abs=guard_abs)
    df["seq_index"] = df["seq_index"].map(block_map).fillna(df["seq_index"]).astype(int)

    # Save merge stats for audit trail
    import json
    merge_stats = {"n_runs": len(set(block_map.keys())), "n_blocks": len(set(block_map.values()))}
    os.makedirs(os.path.join(os.path.abspath("."), "results", "eda"), exist_ok=True)
    with open(os.path.join(os.path.abspath("."), "results", "eda", "trajectory_guard_merge.json"), "w") as f:
        json.dump(merge_stats, f, indent=2)
    print(f"[Phase 2] Merge stats saved: {merge_stats}")

    seq_df = create_sliding_window_sequences(df)
    seq_df = assign_leakage_free_splits(seq_df, df)

    # Save parquet split tables (blueprint processed/ layout)
    for split_name in ["train", "val", "calib", "test"]:
        sub = seq_df[seq_df["split"] == split_name].copy()
        if "x_indices" in sub.columns:
            sub["x_indices_str"] = sub["x_indices"].apply(lambda x: ",".join(map(str, x)))
        pq_path = os.path.join(output_dir, f"{split_name}.parquet")
        try:
            sub.to_parquet(pq_path, index=False)
            print(f"Saved {pq_path}")
        except Exception:
            csv_path_split = os.path.join(output_dir, f"{split_name}.csv")
            sub.to_csv(csv_path_split, index=False)
            print(f"Saved {csv_path_split} (parquet engine unavailable)")

    # Save manifest
    manifest_path = os.path.join(output_dir, "split_manifest.csv")
    # For storage, convert list of indices to string format
    seq_df_out = seq_df.copy()
    seq_df_out["x_indices_str"] = seq_df_out["x_indices"].apply(lambda x: ",".join(map(str, x)))
    seq_df_out.to_csv(manifest_path, index=False)
    print(f"\nSaved split manifest to {manifest_path}")

    return seq_df


def merge_close_trajectory_blocks(raw_df, guard_abs=25):
    """
    Phase 2: connect runs that sit closer than `guard_abs` in abs_index.
    Returns mapping original seq_index -> block_id.
    """
    first = raw_df.groupby("seq_index")["abs_index"].min().sort_values()
    last = raw_df.groupby("seq_index")["abs_index"].max()
    runs = first.index.tolist()
    parent = {r: r for r in runs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(runs) - 1):
        a, b = runs[i], runs[i + 1]
        gap = int(first[b] - last[a])
        if gap <= guard_abs:
            union(a, b)
    mapping = {int(r): int(find(r)) for r in runs}
    n_blocks = len(set(mapping.values()))
    print(f"[Phase 2] Guard-interval merge (guard_abs={guard_abs}): {len(runs)} runs -> {n_blocks} blocks")
    return mapping


if __name__ == "__main__":
    build_partition_manifest()
