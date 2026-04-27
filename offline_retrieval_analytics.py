#!/usr/bin/env python

import os
import math
import argparse
from collections import Counter

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Helpers
# ---------------------------

def find_demo_group(h5file):
    """
    Try to find the group that holds demo_x groups.
    For LIBERO with your current config, this is 'data'.
    """
    candidates = ["data", "demo", "demos", "traj"]
    for c in candidates:
        if c in h5file:
            return c
    raise KeyError(f"Could not find demo group in file. Root keys: {list(h5file.keys())}")


def load_metadata(h5_path):
    """
    Load per-demo metadata from a retrieval HDF5 file.

    Returns a dict with:
        - costs: np.ndarray [N]
        - starts: np.ndarray [N]
        - ends: np.ndarray [N]
        - lengths: np.ndarray [N]
        - source_ids: np.ndarray [N]  (string key: file|traj_key)
        - demo_keys: list of demo_x
        - demo_group: string name of group (e.g., 'data')
    """
    with h5py.File(h5_path, "r") as f:
        demo_group = find_demo_group(f)
        grp = f[demo_group]

        # demo_0, demo_1, ... but make sure we sort numerically by index
        demo_keys = [k for k in grp.keys() if k.startswith("demo")]
        demo_keys_sorted = sorted(
            demo_keys,
            key=lambda k: int(k.split("_")[1])
        )

        costs = []
        starts = []
        ends = []
        lengths = []
        source_ids = []

        for dk in demo_keys_sorted:
            meta_path = f"{demo_group}/{dk}/metadata"
            if meta_path not in f:
                raise RuntimeError(f"Missing metadata group at: {meta_path}")

            meta = f[meta_path]
            cost = float(meta["cost"][()])
            start = int(meta["start"][()])
            end = int(meta["end"][()])
            length = end - start

            src_file = meta["source_file"][()].decode("utf-8") if isinstance(meta["source_file"][()], (bytes, bytearray)) else str(meta["source_file"][()])
            src_traj = meta["source_traj_key"][()].decode("utf-8") if isinstance(meta["source_traj_key"][()], (bytes, bytearray)) else str(meta["source_traj_key"][()])

            source_id = f"{os.path.basename(src_file)}|{src_traj}"

            costs.append(cost)
            starts.append(start)
            ends.append(end)
            lengths.append(length)
            source_ids.append(source_id)

    return {
        "demo_group": demo_group,
        "demo_keys": demo_keys_sorted,
        "costs": np.array(costs, dtype=float),
        "starts": np.array(starts, dtype=int),
        "ends": np.array(ends, dtype=int),
        "lengths": np.array(lengths, dtype=int),
        "source_ids": np.array(source_ids, dtype=object),
    }


def ensure_same_length(meta_sdtw, meta_ot):
    ns, no = len(meta_sdtw["demo_keys"]), len(meta_ot["demo_keys"])
    if ns != no:
        raise RuntimeError(f"SDTW demos={ns} vs OT demos={no} — cannot do paired analysis.")
    # We assume demo_i corresponds to same (query, rank) pair across methods.
    return ns


def safe_entropy(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -(p * np.log(p)).sum()


def make_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Analytics
# ---------------------------

def cost_analysis(meta_sdtw, meta_ot, out_dir):
    costs_s = meta_sdtw["costs"]
    costs_o = meta_ot["costs"]

    n = ensure_same_length(meta_sdtw, meta_ot)

    # Basic stats
    stats = {}
    for name, arr in [("sdtw", costs_s), ("ot", costs_o)]:
        stats[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    # Paired difference
    cost_diff = costs_o - costs_s
    pct_ot_better = float((cost_diff < 0).mean() * 100.0)

    # Save stats to text
    with open(os.path.join(out_dir, "cost_stats.txt"), "w") as f:
        f.write("=== Retrieval Cost Statistics ===\n\n")
        for name in ["sdtw", "ot"]:
            f.write(f"{name.upper()}:\n")
            for k, v in stats[name].items():
                f.write(f"  {k}: {v:.6f}\n")
            f.write("\n")
        f.write("Paired Comparison (OT - SDTW):\n")
        f.write(f"  mean diff: {cost_diff.mean():.6f}\n")
        f.write(f"  std  diff: {cost_diff.std():.6f}\n")
        f.write(f"  min  diff: {cost_diff.min():.6f}\n")
        f.write(f"  max  diff: {cost_diff.max():.6f}\n")
        f.write(f"  % queries where OT_cost < SDTW_cost: {pct_ot_better:.2f}%\n")

    # Histograms
    plt.figure()
    plt.hist(costs_s, bins=30, alpha=0.7)
    plt.title("SDTW Retrieval Costs")
    plt.xlabel("Cost")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_cost_sdtw.png"))
    plt.close()

    plt.figure()
    plt.hist(costs_o, bins=30, alpha=0.7)
    plt.title("OT Retrieval Costs")
    plt.xlabel("Cost")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_cost_ot.png"))
    plt.close()

    # Boxplot
    plt.figure()
    plt.boxplot([costs_s, costs_o], labels=["SDTW", "OT"])
    plt.ylabel("Cost")
    plt.title("Retrieval Cost Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_cost_sdtw_vs_ot.png"))
    plt.close()

    # Histogram of cost differences
    plt.figure()
    plt.hist(cost_diff, bins=30, alpha=0.7)
    plt.title("OT_cost - SDTW_cost")
    plt.xlabel("Cost Difference")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_cost_diff_ot_minus_sdtw.png"))
    plt.close()

    return {
        "stats": stats,
        "cost_diff": cost_diff,
        "pct_ot_better": pct_ot_better,
    }


def diversity_entropy_analysis(meta_sdtw, meta_ot, out_dir):
    src_s = meta_sdtw["source_ids"]
    src_o = meta_ot["source_ids"]

    cnt_s = Counter(src_s)
    cnt_o = Counter(src_o)

    # Convert counters to aligned arrays
    all_sources = sorted(set(cnt_s.keys()) | set(cnt_o.keys()))
    counts_s = np.array([cnt_s.get(s, 0) for s in all_sources], dtype=float)
    counts_o = np.array([cnt_o.get(s, 0) for s in all_sources], dtype=float)

    entropy_s = safe_entropy(counts_s)
    entropy_o = safe_entropy(counts_o)

    unique_s = (counts_s > 0).sum()
    unique_o = (counts_o > 0).sum()

    total_s = len(src_s)
    total_o = len(src_o)

    frac_unique_s = unique_s / total_s if total_s > 0 else 0.0
    frac_unique_o = unique_o / total_o if total_o > 0 else 0.0

    # Save stats
    with open(os.path.join(out_dir, "diversity_stats.txt"), "w") as f:
        f.write("=== Retrieval Diversity / Entropy ===\n\n")
        f.write("SDTW:\n")
        f.write(f"  unique demos: {unique_s}\n")
        f.write(f"  total retrieved: {total_s}\n")
        f.write(f"  frac unique: {frac_unique_s:.4f}\n")
        f.write(f"  entropy: {entropy_s:.6f}\n\n")

        f.write("OT:\n")
        f.write(f"  unique demos: {unique_o}\n")
        f.write(f"  total retrieved: {total_o}\n")
        f.write(f"  frac unique: {frac_unique_o:.4f}\n")
        f.write(f"  entropy: {entropy_o:.6f}\n\n")

    # Bar chart of retrieval frequency per source (top-K for readability)
    top_k = min(30, len(all_sources))

    # Sort by SDTW frequency
    idx_sorted_s = np.argsort(-counts_s)[:top_k]
    top_sources_s = [all_sources[i] for i in idx_sorted_s]
    top_counts_s = counts_s[idx_sorted_s]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(top_sources_s)), top_counts_s)
    plt.xticks(range(len(top_sources_s)), top_sources_s, rotation=90)
    plt.ylabel("Count")
    plt.title("Top Retrieval Sources - SDTW")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bar_source_freq_sdtw.png"))
    plt.close()

    # Sort by OT frequency
    idx_sorted_o = np.argsort(-counts_o)[:top_k]
    top_sources_o = [all_sources[i] for i in idx_sorted_o]
    top_counts_o = counts_o[idx_sorted_o]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(top_sources_o)), top_counts_o)
    plt.xticks(range(len(top_sources_o)), top_sources_o, rotation=90)
    plt.ylabel("Count")
    plt.title("Top Retrieval Sources - OT")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bar_source_freq_ot.png"))
    plt.close()

    # Entropy comparison bar chart
    plt.figure()
    plt.bar(["SDTW", "OT"], [entropy_s, entropy_o])
    plt.ylabel("Entropy")
    plt.title("Retrieval Entropy (Source Distribution)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bar_entropy_sdtw_vs_ot.png"))
    plt.close()

    return {
        "entropy_sdtw": entropy_s,
        "entropy_ot": entropy_o,
        "frac_unique_sdtw": frac_unique_s,
        "frac_unique_ot": frac_unique_o,
    }


def retrieval_consistency(meta_sdtw, meta_ot, out_dir, cost_diff):
    """
    Compare OT and SDTW retrievals for each index (same demo_i index).
    Check:
      - same (source_file, source_traj_key, start, end)?
      - if not, what are cost differences?
    """
    src_s = meta_sdtw["source_ids"]
    src_o = meta_ot["source_ids"]
    start_s, start_o = meta_sdtw["starts"], meta_ot["starts"]
    end_s, end_o = meta_sdtw["ends"], meta_ot["ends"]

    same_segment = (
        (src_s == src_o) &
        (start_s == start_o) &
        (end_s == end_o)
    )

    pct_same = float(same_segment.mean() * 100.0)

    diff_mask = ~same_segment
    num_diff = diff_mask.sum()

    with open(os.path.join(out_dir, "consistency_stats.txt"), "w") as f:
        f.write("=== Retrieval Consistency (OT vs SDTW) ===\n\n")
        f.write(f"Total retrievals: {len(src_s)}\n")
        f.write(f"Same retrieved segment: {pct_same:.2f}%\n")
        f.write(f"Different retrieved segment: {num_diff} ({100.0 - pct_same:.2f}%)\n\n")

        if num_diff > 0:
            diff_costs = cost_diff[diff_mask]
            f.write("Cost difference for non-matching segments (OT - SDTW):\n")
            f.write(f"  mean: {diff_costs.mean():.6f}\n")
            f.write(f"  std:  {diff_costs.std():.6f}\n")
            f.write(f"  min:  {diff_costs.min():.6f}\n")
            f.write(f"  max:  {diff_costs.max():.6f}\n")

    return {
        "pct_same_segment": pct_same,
        "num_diff": int(num_diff),
    }


def temporal_alignment_analysis(meta_sdtw, meta_ot, out_dir):
    start_s, start_o = meta_sdtw["starts"], meta_ot["starts"]
    end_s, end_o = meta_sdtw["ends"], meta_ot["ends"]
    len_s, len_o = meta_sdtw["lengths"], meta_ot["lengths"]

    diff_start = start_o - start_s
    diff_end = end_o - end_s
    diff_len = len_o - len_s

    with open(os.path.join(out_dir, "temporal_stats.txt"), "w") as f:
        f.write("=== Temporal Alignment Differences (OT - SDTW) ===\n\n")
        f.write("Start index diff (OT_start - SDTW_start):\n")
        f.write(f"  mean: {diff_start.mean():.6f}\n")
        f.write(f"  std:  {diff_start.std():.6f}\n")
        f.write(f"  min:  {diff_start.min():.6f}\n")
        f.write(f"  max:  {diff_start.max():.6f}\n\n")

        f.write("End index diff (OT_end - SDTW_end):\n")
        f.write(f"  mean: {diff_end.mean():.6f}\n")
        f.write(f"  std:  {diff_end.std():.6f}\n")
        f.write(f"  min:  {diff_end.min():.6f}\n")
        f.write(f"  max:  {diff_end.max():.6f}\n\n")

        f.write("Length diff (OT_len - SDTW_len):\n")
        f.write(f"  mean: {diff_len.mean():.6f}\n")
        f.write(f"  std:  {diff_len.std():.6f}\n")
        f.write(f"  min:  {diff_len.min():.6f}\n")
        f.write(f"  max:  {diff_len.max():.6f}\n")

    # Histograms for temporal differences
    plt.figure()
    plt.hist(diff_start, bins=30)
    plt.title("Start Index Difference (OT - SDTW)")
    plt.xlabel("Index Difference")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_diff_start.png"))
    plt.close()

    plt.figure()
    plt.hist(diff_end, bins=30)
    plt.title("End Index Difference (OT - SDTW)")
    plt.xlabel("Index Difference")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_diff_end.png"))
    plt.close()

    plt.figure()
    plt.hist(diff_len, bins=30)
    plt.title("Retrieved Length Difference (OT - SDTW)")
    plt.xlabel("Length Difference")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_diff_length.png"))
    plt.close()

    return {
        "diff_start": diff_start,
        "diff_end": diff_end,
        "diff_len": diff_len,
    }


def embedding_similarity_stub(out_dir):
    # Placeholder: retrieval HDF5 alone does not contain embeddings.
    # To implement this properly you would:
    #  - load embeddings from original embedding files
    #  - map (source_file, source_traj_key, start, end) → embedding slice
    #  - compute cosine / L2 distances
    with open(os.path.join(out_dir, "embedding_similarity_NOTE.txt"), "w") as f:
        f.write(
            "Embedding similarity analysis is SKIPPED.\n"
            "Reason: retrieval HDF5 files do not store embeddings.\n"
            "To enable this, you must load DINO embeddings from the original embedding files\n"
            "using the same indexing (source_file, source_traj_key, start, end).\n"
        )


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline SDTW vs OT retrieval analytics.")
    parser.add_argument(
        "--sdtw_file",
        type=str,
        default="/home/ubuntu/STRAP/data/retrieval_results/stove-pot_retrieved_dataset.hdf5",
        help="Path to SDTW retrieval HDF5 file.",
    )
    parser.add_argument(
        "--ot_file",
        type=str,
        default="/home/ubuntu/STRAP/data/retrieval_results/metadata-ot-stove-pot_retrieved_dataset.hdf5",
        help="Path to OT retrieval HDF5 file.",
    )
    parser.add_argument(
    "--out_dir",
    type=str,
    default="/home/ubuntu/STRAP/data/retrieval_results/analytics",
    help="Directory where analytics outputs (plots + stats) will be saved.",
)

    args = parser.parse_args()

    make_dir(args.out_dir)

    print("Loading SDTW metadata...")
    meta_sdtw = load_metadata(args.sdtw_file)
    print(f"  Loaded {len(meta_sdtw['demo_keys'])} demos from SDTW file.")

    print("Loading OT metadata...")
    meta_ot = load_metadata(args.ot_file)
    print(f"  Loaded {len(meta_ot['demo_keys'])} demos from OT file.")

    ensure_same_length(meta_sdtw, meta_ot)

    print("\n[1] Retrieval Cost Analysis...")
    cost_res = cost_analysis(meta_sdtw, meta_ot, args.out_dir)
    print(f"  % OT better (lower cost): {cost_res['pct_ot_better']:.2f}%")

    print("\n[2] Retrieval Diversity & Entropy...")
    div_res = diversity_entropy_analysis(meta_sdtw, meta_ot, args.out_dir)
    print(f"  Entropy SDTW: {div_res['entropy_sdtw']:.4f}")
    print(f"  Entropy OT:   {div_res['entropy_ot']:.4f}")

    print("\n[3] Embedding Similarity (stub)...")
    embedding_similarity_stub(args.out_dir)
    print("  (Skipped; see embedding_similarity_NOTE.txt for instructions.)")

    print("\n[4] Retrieval Consistency...")
    cons_res = retrieval_consistency(meta_sdtw, meta_ot, args.out_dir, cost_res["cost_diff"])
    print(f"  % identical segments: {cons_res['pct_same_segment']:.2f}%")

    print("\n[5] Temporal Alignment Analytics...")
    temporal_alignment_analysis(meta_sdtw, meta_ot, args.out_dir)
    print("  Temporal alignment stats + histograms saved.")

    print("\n✅ Done. All analytics written to:")
    print(f"   {args.out_dir}")


if __name__ == "__main__":
    main()
