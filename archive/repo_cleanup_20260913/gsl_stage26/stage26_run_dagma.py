#!/usr/bin/env python3
"""
Stage 26 — Full-Sensor Multi-Lag DAGMA Extraction.

Constructs multi-lag input:
  Z = [x(t-L), x(t-L+1), ..., x(t-1), x(t)]  for ALL N sensors

DAGMA operates on (L+1)*N variables.

Block extraction (DAGMA convention W[i,j] = variable_i -> variable_j):
  - W[l*N:(l+1)*N, l*N:(l+1)*N] = within-block (lag l autocorrelation)
  - W[0:N, (l)*N:(l+1)*N]       = lag l sensor -> lag 0 sensor (cross-lag)
  - W[l*N:(l+1)*N, N:2N]        = lag l source -> lag 1 target

For forecasting we need:
  A_l[i,j] = "sensor i at time t-l influences sensor j at time t"

Using DAGMA convention W[i,j] = i -> j and the block structure:
  Z columns [0:N]     = x(t-L)
  Z columns [N:2N]    = x(t-L+1)
  ...
  Z columns [L*N:(L+1)*N] = x(t)

The lag-l-to-current block:
  W[l*N:(l+1)*N, L*N:(L+1)*N]
represents "variable in block l -> variable in block L (current)"

Since W[i,j] = variable_i -> variable_j:
  W[l*N+i, L*N+j] > 0  means  sensor_i(t-l) -> sensor_j(t)

This is the correct temporal dependency for forecasting.

Usage:
  python gsl_stage26/stage26_run_dagma.py --ph 1 --dataset shenzhen
  python gsl_stage26/stage26_run_dagma.py --ph 1 --dataset losloop
  python gsl_stage26/stage26_run_dagma.py --ph 1 --dataset shenzhen --lags 2
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

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage26_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

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


def build_multilag_Z(train_norm, N, n_lags):
    """
    Build multi-lag DAGMA input Z = [x(t-L), ..., x(t-1), x(t)].

    Args:
        train_norm: (T, N) normalized training data
        N: number of sensors (all sensors)
        n_lags: number of lagged snapshots L

    Returns:
        Z: (T - n_lags, (L+1)*N) multi-lag matrix

    Block structure:
        Z[:, 0:N]       = x(t-L)      (most distant past)
        Z[:, N:2N]      = x(t-L+1)
        ...
        Z[:, (L-1)*N:LN] = x(t-1)    (most recent past)
        Z[:, L*N:(L+1)*N] = x(t)     (current)

    DAGMA convention: W[i,j] = variable_i -> variable_j
    Therefore: W[lag_block_i, current_block_j] represents
               variable in lag block i -> variable in current block j
    """
    T = train_norm.shape[0]
    M = T - n_lags  # number of samples
    rows = []
    for t in range(n_lags, T):
        # Stack: [x(t-L), x(t-L+1), ..., x(t-1), x(t)]
        z = np.concatenate(
            [train_norm[t - l] for l in range(n_lags, 0, -1)] + [train_norm[t]]
        )
        rows.append(z)
    Z = np.array(rows, dtype=np.float32)
    return Z


def extract_lag_blocks(W_est, N, n_lags):
    """
    Extract lag-specific adjacency blocks from the full DAGMA weight matrix.

    DAGMA convention: W[i,j] = variable_i -> variable_j

    Block structure:
        Block 0 (rows 0:N)      = variables for x(t-L)
        Block 1 (rows N:2N)     = variables for x(t-L+1)
        ...
        Block L (rows L*N:...)  = variables for x(t) (current)

    For forecasting, we need:
        A_l[i,j] = sensor_i(t-l) -> sensor_j(t)

    This corresponds to:
        W[lag_block_start + i, current_block_start + j]

    Where lag_block_start = l_idx * N (with l_idx=0 for most distant past)
    and current_block_start = n_lags * N

    Returns dict: {lag_label: np.ndarray of shape (N, N)}
    """
    current_block_start = n_lags * N

    lag_blocks = {}
    for l_idx in range(n_lags):
        lag_block_start = l_idx * N
        # W[lag_block_start:N, current_block_start:current_block_start+N]
        # This is: variable in block l_idx -> variable in block n_lags (current)
        # Meaning: sensor_i at time t-(L-l_idx) -> sensor_j at time t
        W_block = W_est[lag_block_start:lag_block_start + N,
                        current_block_start:current_block_start + N]

        # The lag value: l_idx=0 corresponds to lag L (most distant),
        # l_idx=n_lags-1 corresponds to lag 1 (most recent)
        lag_value = n_lags - l_idx
        lag_label = f"lag_{lag_value}"
        lag_blocks[lag_label] = W_block.astype(np.float32)

    # Also extract the contemporaneous block (current -> current)
    current_self = W_est[current_block_start:current_block_start + N,
                         current_block_start:current_block_start + N]
    lag_blocks["current"] = current_self.astype(np.float32)

    return lag_blocks


def binary_graph(W, threshold):
    """Convert weighted W to binary adjacency at given threshold."""
    adj = (np.abs(W) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)
    return adj


def main():
    parser = argparse.ArgumentParser(description="Stage 26: Full-Sensor Multi-Lag DAGMA")
    parser.add_argument("--ph", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Prediction horizon")
    parser.add_argument("--dataset", type=str, default="shenzhen",
                        choices=["shenzhen", "losloop"], help="Dataset name")
    parser.add_argument("--lags", type=int, default=3,
                        help="Number of lagged snapshots L")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lambda1", type=float, default=0.01, help="DAGMA L1 coefficient")
    parser.add_argument("--warm-iter", type=int, default=30000,
                        help="DAGMA warm-up iterations")
    parser.add_argument("--max-iter", type=int, default=60000,
                        help="DAGMA max iterations")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if output exists")
    args = parser.parse_args()

    dataset = args.dataset
    ph = args.ph
    seed = args.seed
    n_lags = args.lags
    config = DATASET_CONFIGS[dataset]
    N = config["N"]
    prefix = "sz" if dataset == "shenzhen" else "los"
    total_vars = (n_lags + 1) * N

    print("=" * 80)
    print(f"STAGE 26: Full-Sensor Multi-Lag DAGMA — {dataset}, PH={ph}")
    print(f"N={N}, Lags={n_lags}, Total variables={total_vars}")
    print(f"Matrix size: {total_vars} x {total_vars}")
    print(f"Seed={seed}, lambda1={args.lambda1}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Output paths
    W_path = os.path.join(RESULTS_DIR, f"{prefix}_ph{ph}_seed{seed}_L{n_lags}_W_full.npy")
    meta_path = os.path.join(RESULTS_DIR, f"{prefix}_ph{ph}_seed{seed}_L{n_lags}_metadata.json")

    if os.path.exists(W_path) and not args.force:
        print(f"\nOutput already exists: {W_path}")
        print("Use --force to re-run.")
        return

    # Load and normalize data
    feat = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, config["feat_path"])),
                    dtype=np.float32)
    T_full, N_data = feat.shape
    assert N_data == N, f"Expected {N} sensors, got {N_data}"
    split = int(T_full * 0.8)
    train = feat[:split]
    feat_max = float(np.max(train))
    train_norm = train / feat_max

    print(f"\nDataset: {dataset}, T={T_full}, N={N}")
    print(f"Train timesteps: {split}")
    print(f"feat_max: {feat_max:.6f}")

    # Build multi-lag Z
    print(f"\nBuilding Z with {n_lags} lags and {N} sensors...")
    Z = build_multilag_Z(train_norm, N, n_lags)
    print(f"Z shape: {Z.shape} ({Z.shape[1]} variables = {n_lags+1} * {N})")

    # Critical assertion: verify block structure
    assert Z.shape[1] == total_vars, f"Z columns {Z.shape[1]} != (L+1)*N = {total_vars}"
    print(f"Assertion passed: Z has correct shape ({n_lags+1} blocks of {N})")

    # Block verification
    print(f"\n--- Block structure verification ---")
    for blk in range(n_lags + 1):
        start = blk * N
        end = (blk + 1) * N
        if blk < n_lags:
            lag_val = n_lags - blk
            print(f"  Block {blk} (cols {start}:{end}) = x(t-{lag_val})")
        else:
            print(f"  Block {blk} (cols {start}:{end}) = x(t) [current]")

    print(f"\nDAGMA block interpretation:")
    print(f"  W[lag_block_i, current_block_j] = variable_i(lag) -> variable_j(current)")
    print(f"  Current block starts at column {n_lags * N}")

    # Run DAGMA
    print(f"\nRunning DAGMA (lambda1={args.lambda1}, w_threshold=0.0)...")
    print(f"  warm_iter={args.warm_iter}, max_iter={args.max_iter}")
    np.random.seed(seed)
    t0 = time.time()
    model = DagmaLinear(loss_type="l2", verbose=True)
    W_est = model.fit(Z, lambda1=args.lambda1, w_threshold=0.0,
                      warm_iter=args.warm_iter, max_iter=args.max_iter)
    runtime = time.time() - t0
    print(f"\nDAGMA completed in {runtime:.1f}s ({runtime/60:.1f} min)")
    print(f"W_est shape: {W_est.shape}")

    # Save raw full W
    np.save(W_path, W_est)
    print(f"Saved: {W_path}")

    # Extract lag-specific blocks
    lag_blocks = extract_lag_blocks(W_est, N, n_lags)

    # Save each block
    block_paths = {}
    for lag_label, W_block in lag_blocks.items():
        block_path = os.path.join(RESULTS_DIR,
                                   f"{prefix}_ph{ph}_seed{seed}_L{n_lags}_{lag_label}.npy")
        np.save(block_path, W_block)
        block_paths[lag_label] = block_path

    # Analyze each block
    print(f"\n--- Lag-specific block analysis ---")
    block_stats = {}
    for lag_label, W_block in sorted(lag_blocks.items()):
        abs_w = np.abs(W_block)
        n_nonzero_01 = int(np.sum(abs_w > 0.1))
        n_nonzero_001 = int(np.sum(abs_w > 0.001))
        n_total_nonzero = int(np.sum(abs_w > 0))
        max_w = float(abs_w.max()) if abs_w.size > 0 else 0
        nonzero_vals = abs_w[abs_w > 0]
        mean_nonzero = float(nonzero_vals.mean()) if nonzero_vals.size > 0 else 0

        # Top 5 edges
        flat_idx = np.argsort(abs_w.ravel())[::-1][:5]
        top_edges = []
        for idx in flat_idx:
            i, j = divmod(idx, W_block.shape[1])
            if abs_w[i, j] > 0:
                top_edges.append({"src": int(i), "tgt": int(j),
                                  "w": round(float(W_block[i, j]), 6)})

        block_stats[lag_label] = {
            "shape": list(W_block.shape),
            "nonzero_total": n_total_nonzero,
            "nonzero_gt0.001": n_nonzero_001,
            "nonzero_gt0.1": n_nonzero_01,
            "density_gt0.1": round(n_nonzero_01 / (N * N), 6),
            "max_abs_weight": round(max_w, 6),
            "mean_abs_nonzero": round(mean_nonzero, 6),
            "top_edges": top_edges,
        }

        print(f"\n  {lag_label}:")
        print(f"    Shape: {W_block.shape}")
        print(f"    Nonzero (any): {n_total_nonzero}")
        print(f"    Nonzero (|w|>0.001): {n_nonzero_001}")
        print(f"    Nonzero (|w|>0.1): {n_nonzero_01}")
        print(f"    Density (>0.1): {block_stats[lag_label]['density_gt0.1']:.6f}")
        print(f"    Max |w|: {max_w:.6f}")
        print(f"    Mean |w| (nonzero): {mean_nonzero:.6f}")
        if top_edges:
            print(f"    Top edges:")
            for e in top_edges[:3]:
                print(f"      [{e['src']:3d} -> {e['tgt']:3d}] = {e['w']:.6f}")

    # Threshold sweep across all blocks
    print(f"\n--- Threshold sweep (total edges across all lag blocks) ---")
    for thr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]:
        total = 0
        for lag_label, W_block in lag_blocks.items():
            total += int(np.sum(np.abs(W_block) > thr))
        print(f"  thr={thr:.3f}: {total} total edges across {len(lag_blocks)} blocks")

    # Save metadata
    metadata = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "dataset": dataset,
        "N": N,
        "PH": ph,
        "n_lags": n_lags,
        "total_variables": total_vars,
        "matrix_shape": list(W_est.shape),
        "seed": seed,
        "lambda1": args.lambda1,
        "loss_type": "l2",
        "w_threshold": 0.0,
        "warm_iter": args.warm_iter,
        "max_iter": args.max_iter,
        "seq_len": 12,
        "feat_max": feat_max,
        "split": split,
        "T_full": T_full,
        "runtime_s": round(runtime, 1),
        "runtime_min": round(runtime / 60, 1),
        "block_interpretation": {
            "description": "W[i,j] = variable_i -> variable_j. Block l (rows l*N:(l+1)*N) corresponds to x(t-L+l). Current block at rows L*N:(L+1)*N. Lag-l-to-current block: W[l*N:(l+1)*N, L*N:(L+1)*N] means sensor_i(t-L+l) -> sensor_j(t).",
            "current_block_start_col": n_lags * N,
            "lag_blocks": {
                f"lag_{n_lags - l_idx}": f"W[{l_idx*N}:{(l_idx+1)*N}, {n_lags*N}:{(n_lags+1)*N}]"
                for l_idx in range(n_lags)
            },
        },
        "block_stats": block_stats,
        "block_paths": block_paths,
        "full_W_path": W_path,
        "script": "gsl_stage26/stage26_run_dagma.py",
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nMetadata saved: {meta_path}")

    print(f"\nCompleted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
