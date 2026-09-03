#!/usr/bin/env python3
"""
Stage 24 — Run temporal DAGMA for any dataset/PH/seed combination.

Supports:
  - SZ-Taxi (shenzhen): N=156
  - Los-loop: N=207

Usage:
  python gsl_stage24/stage24_run_dagma.py --ph 1 --seed 42
  python gsl_stage24/stage24_run_dagma.py --ph 1 --seed 43
  python gsl_stage24/stage24_run_dagma.py --ph 1 --dataset losloop
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from dagma.linear import DagmaLinear

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage24_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset configurations
DATASET_CONFIGS = {
    "shenzhen": {
        "feat_path": "data/sz_speed.csv",
        "adj_path": "data/sz_adj.csv",
        "N": 156,
    },
    "losloop": {
        "feat_path": "data/los_speed.csv",
        "adj_path": "data/los_adj.csv",
        "N": 207,
    },
}


def build_Z_for_ph(train_norm, N, seq_len, ph):
    """Build DAGMA input Z for a given PH."""
    Xs = []
    for i in range(len(train_norm) - seq_len - 1):
        Xs.append(train_norm[i:i + seq_len])
    Xs = np.array(Xs)
    data = Xs[:, 0, :]  # (M, N) — first timestep of each window
    
    X_orig = data[::ph]
    M_orig = X_orig.shape[0]
    
    M = M_orig - 1
    Z = np.zeros((M, 2 * N), dtype=np.float32)
    Z[:, 0:N] = X_orig[:-1]
    Z[:, N:2 * N] = X_orig[1:]
    
    return Z, M_orig


def get_output_prefix(dataset, ph, seed):
    """Generate output prefix based on dataset/ph/seed."""
    if dataset == "shenzhen":
        return f"sz_ph{ph}_seed{seed}"
    elif dataset == "losloop":
        return f"los_ph{ph}_seed{seed}"
    else:
        return f"{dataset}_ph{ph}_seed{seed}"


def main():
    parser = argparse.ArgumentParser(description="Stage 24: Run temporal DAGMA")
    parser.add_argument("--ph", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Prediction horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default="shenzhen",
                        choices=["shenzhen", "losloop"],
                        help="Dataset name")
    parser.add_argument("--lambda1", type=float, default=0.01, help="DAGMA L1 coefficient")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if output exists")
    args = parser.parse_args()
    
    ph = args.ph
    seed = args.seed
    dataset = args.dataset
    config = DATASET_CONFIGS[dataset]
    N = config["N"]
    prefix = get_output_prefix(dataset, ph, seed)
    
    print("=" * 70)
    print(f"STAGE 24: Temporal DAGMA — {dataset}, PH={ph}, seed={seed}")
    print(f"N={N}, Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check if output already exists
    W_path = os.path.join(RESULTS_DIR, f"{prefix}_W_raw_temporal.npy")
    if os.path.exists(W_path) and not args.force:
        print(f"\nOutput already exists: {W_path}")
        print("Use --force to re-run.")
        return
    
    # Load and normalize data
    feat = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, config["feat_path"])),
                    dtype=np.float32)
    T_full, N_data = feat.shape
    split = int(T_full * 0.8)
    train = feat[:split]
    feat_max = float(np.max(train))
    train_norm = train / feat_max
    
    print(f"\nDataset: {dataset}, N={N_data}")
    print(f"Train timesteps: {split}")
    print(f"feat_max: {feat_max:.6f}")
    
    # Build Z for this PH
    seq_len = 12
    Z, M_orig = build_Z_for_ph(train_norm, N, seq_len, ph)
    print(f"PH={ph}: X_orig rows = {M_orig}, Z shape = {Z.shape}")
    
    # Run DAGMA
    print(f"\nRunning DAGMA (lambda1={args.lambda1}, w_threshold=0.0)...")
    np.random.seed(seed)
    t0 = time.time()
    model = DagmaLinear(loss_type='l2', verbose=True)
    W = model.fit(Z, lambda1=args.lambda1, w_threshold=0.0)
    runtime = time.time() - t0
    print(f"DAGMA completed in {runtime:.1f}s ({runtime/60:.1f} min)")
    
    # Save raw W
    np.save(W_path, W)
    
    # Extract correct temporal block
    W_cross = W[0:N, N:2*N]
    W_cross_path = os.path.join(RESULTS_DIR, f"{prefix}_W_cross_correct.npy")
    np.save(W_cross_path, W_cross)
    
    # Save metadata
    metadata = {
        "dataset": dataset,
        "N": N,
        "PH": ph,
        "seed": seed,
        "lambda1": args.lambda1,
        "loss_type": "l2",
        "w_threshold": 0.0,
        "seq_len": seq_len,
        "feat_max": feat_max,
        "split": split,
        "M_orig": M_orig,
        "Z_shape": list(Z.shape),
        "W_shape": list(W.shape),
        "runtime_s": round(runtime, 1),
        "runtime_min": round(runtime / 60, 1),
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "correct_block": "W[0:N, N:2N] (past -> current)",
        "script": "gsl_stage24/stage24_run_dagma.py",
    }
    meta_path = os.path.join(RESULTS_DIR, f"{prefix}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Report statistics
    abs_cross = np.abs(W_cross)
    print(f"\n--- Correct temporal block W[0:N, N:2N] ---")
    print(f"  Shape: {W_cross.shape}")
    print(f"  Nonzero: {np.sum(abs_cross > 0)}")
    print(f"  Max |W|: {abs_cross.max():.6f}")
    print(f"  Mean |W| (nonzero): {abs_cross[abs_cross > 0].mean():.6f}")
    
    print(f"\n  Top 10 temporal edges:")
    flat_idx = np.argsort(abs_cross.ravel())[::-1]
    for rank, idx in enumerate(flat_idx[:10], 1):
        i, j = divmod(idx, N)
        w = W_cross[i, j]
        edge_type = "self-loop" if i == j else "cross-sensor"
        print(f"    {rank:2d}. [{i:3d},{j:3d}] = {w:10.6f}  ({edge_type})")
    
    print(f"\nSaved:")
    print(f"  {W_path}")
    print(f"  {W_cross_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
