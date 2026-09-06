#!/usr/bin/env python3
"""
Stage 27 — Temporal Resolution Experiment.

Tests whether the marginal SZ-Taxi improvement is due to 15-min vs 5-min resolution.

Approach:
  1. Resample Los-loop from 5-min to 15-min intervals (avg every 3 steps)
  2. Run multi-lag DAGMA (L=3) on resampled data
  3. Compare edge structures with original 5-min DAGMA
  4. Run forecasting (T-GCN-NoSpatial, T-GCN-MultiGSL-Mix) on resampled data
  5. Compare RMSE: Los-loop-5min vs Los-loop-15min vs SZ-Taxi-15min

If Los-loop-15min shows reduced improvement (similar to SZ-Taxi),
temporal resolution is a key confounding factor.

Usage:
  python gsl_stage26/stage26_resolution_experiment.py --phase dagma
  python gsl_stage26/stage26_resolution_experiment.py --phase evaluate
  python gsl_stage26/stage26_resolution_experiment.py --phase analyze
  python gsl_stage26/stage26_resolution_experiment.py --phase all
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dagma.linear import DagmaLinear
from utils.graph_conv import calculate_laplacian_with_self_loop
from models.tgcn import TGCN

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage27_resolution")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_CONFIGS = {
    "losloop": {
        "feat_path": "data/los_speed.csv",
        "adj_path": "data/los_adj.csv",
        "N": 207, "prefix": "los",
    },
    "shenzhen": {
        "feat_path": "data/sz_speed.csv",
        "adj_path": "data/sz_adj.csv",
        "N": 156, "prefix": "sz",
    },
}


def resample_15min(data_5min):
    """Average every 3 consecutive time steps to get 15-min intervals."""
    T = data_5min.shape[0]
    T_resampled = T // 3
    # Trim to multiple of 3
    data_trimmed = data_5min[:T_resampled * 3]
    # Reshape and average
    resampled = data_trimmed.reshape(T_resampled, 3, -1).mean(axis=1)
    return resampled


def build_multilag_Z(train_norm, N, n_lags):
    """Build Z = [x(t-L), ..., x(t-1), x(t)]."""
    T = train_norm.shape[0]
    rows = []
    for t in range(n_lags, T):
        z = np.concatenate(
            [train_norm[t - l] for l in range(n_lags, 0, -1)] + [train_norm[t]]
        )
        rows.append(z)
    return np.array(rows, dtype=np.float32)


def extract_lag_blocks(W_est, N, n_lags):
    """Extract lag-specific blocks from DAGMA W matrix."""
    current_start = n_lags * N
    lag_blocks = {}
    for l_idx in range(n_lags):
        W_block = W_est[l_idx * N:(l_idx + 1) * N,
                        current_start:current_start + N]
        lag_value = n_lags - l_idx
        lag_blocks[f"lag_{lag_value}"] = W_block.astype(np.float32)
    current_self = W_est[current_start:current_start + N,
                         current_start:current_start + N]
    lag_blocks["current"] = current_self.astype(np.float32)
    return lag_blocks


def run_dagma_phase(args):
    """Phase 1: Run DAGMA on resampled Los-loop (15-min)."""
    print("=" * 80)
    print("PHASE 1: DAGMA on Resampled Los-loop (15-min intervals)")
    print("=" * 80)

    config = DATASET_CONFIGS["losloop"]
    N = config["N"]
    n_lags = args.lags
    seed = args.seed
    prefix = "los_15min"

    # Load data
    feat = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, config["feat_path"])),
                    dtype=np.float32)
    print(f"Original shape: {feat.shape} (5-min intervals)")

    # Resample to 15-min
    feat_15 = resample_15min(feat)
    print(f"Resampled shape: {feat_15.shape} (15-min intervals)")
    print(f"Reduction: {feat.shape[0]} -> {feat_15.shape[0]} timesteps ({feat_15.shape[0]/feat.shape[0]*100:.1f}%)")

    # Save resampled data
    resampled_path = os.path.join(RESULTS_DIR, f"{prefix}_resampled.csv")
    pd.DataFrame(feat_15).to_csv(resampled_path, header=False, index=False)
    print(f"Saved resampled data: {resampled_path}")

    # Train/test split
    split = int(feat_15.shape[0] * 0.8)
    train = feat_15[:split]
    feat_max = float(np.max(train))
    train_norm = train / feat_max

    total_vars = (n_lags + 1) * N
    print(f"\nN={N}, Lags={n_lags}, Total variables={total_vars}")
    print(f"Matrix size: {total_vars} x {total_vars}")
    print(f"Train timesteps: {split}")

    # Build Z
    Z = build_multilag_Z(train_norm, N, n_lags)
    print(f"Z shape: {Z.shape}")

    # Run DAGMA
    print(f"\nRunning DAGMA (seed={seed}, lambda1={args.lambda1})...")
    np.random.seed(seed)
    t0 = time.time()
    model = DagmaLinear(loss_type="l2", verbose=True)
    W_est = model.fit(Z, lambda1=args.lambda1, w_threshold=0.0,
                      warm_iter=args.warm_iter, max_iter=args.max_iter)
    runtime = time.time() - t0
    print(f"DAGMA completed in {runtime:.1f}s ({runtime/60:.1f} min)")

    # Save raw W
    W_path = os.path.join(RESULTS_DIR, f"{prefix}_ph{args.ph}_seed{seed}_L{n_lags}_W_full.npy")
    np.save(W_path, W_est)
    print(f"Saved: {W_path}")

    # Extract blocks
    lag_blocks = extract_lag_blocks(W_est, N, n_lags)
    block_paths = {}
    for lag_label, W_block in lag_blocks.items():
        block_path = os.path.join(RESULTS_DIR,
                                   f"{prefix}_ph{args.ph}_seed{seed}_L{n_lags}_{lag_label}.npy")
        np.save(block_path, W_block)
        block_paths[lag_label] = block_path

    # Save metadata
    metadata = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "experiment": "resolution_comparison",
        "dataset": "losloop_15min",
        "original_interval": "5min",
        "resampled_interval": "15min",
        "N": N,
        "PH": args.ph,
        "n_lags": n_lags,
        "total_variables": total_vars,
        "matrix_shape": list(W_est.shape),
        "seed": seed,
        "lambda1": args.lambda1,
        "warm_iter": args.warm_iter,
        "max_iter": args.max_iter,
        "feat_max": feat_max,
        "split": split,
        "T_original": feat.shape[0],
        "T_resampled": feat_15.shape[0],
        "runtime_s": round(runtime, 1),
        "block_paths": block_paths,
        "full_W_path": W_path,
    }

    meta_path = os.path.join(RESULTS_DIR, f"{prefix}_ph{args.ph}_seed{seed}_L{n_lags}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    # Print block stats
    print("\n--- Lag block statistics (15-min Los-loop) ---")
    for lag_label, W_block in sorted(lag_blocks.items()):
        abs_w = np.abs(W_block)
        n_01 = int(np.sum(abs_w > 0.1))
        n_001 = int(np.sum(abs_w > 0.001))
        max_w = float(abs_w.max())
        print(f"  {lag_label}: nonzero(>0.1)={n_01}, nonzero(>0.001)={n_001}, max|w|={max_w:.4f}")

    # Also load original 5-min stats for comparison
    orig_prefix = "los"
    orig_path = os.path.join(PROJECT_ROOT, "results", "stage26_validation",
                              f"{orig_prefix}_ph{args.ph}_seed{seed}_L{n_lags}_metadata.json")
    if os.path.exists(orig_path):
        with open(orig_path) as f:
            orig_meta = json.load(f)
        print("\n--- Comparison with original 5-min Los-loop ---")
        for lag_label in sorted(lag_blocks.keys()):
            orig_key = lag_label
            if orig_key in orig_meta.get("block_stats", {}):
                orig_stat = orig_meta["block_stats"][orig_key]
                new_abs = np.abs(lag_blocks[lag_label])
                n_01_new = int(np.sum(new_abs > 0.1))
                print(f"  {lag_label}: 5min={orig_stat['nonzero_gt0.1']} edges, "
                      f"15min={n_01_new} edges")

def run_evaluate_phase(args):
    """Phase 2: Run forecasting on resampled Los-loop (15-min)."""
    print("=" * 80)
    print("PHASE 2: Forecasting on Resampled Los-loop (15-min)")
    print("=" * 80)

    config = DATASET_CONFIGS["losloop"]
    N = config["N"]
    n_lags = args.lags
    seed = args.seed
    ph = args.ph
    prefix = "los_15min"

    # Load resampled data
    resampled_path = os.path.join(RESULTS_DIR, f"{prefix}_resampled.csv")
    if not os.path.exists(resampled_path):
        print(f"ERROR: Resampled data not found: {resampled_path}")
        print("Run --phase dagma first.")
        return

    feat_15 = np.array(pd.read_csv(resampled_path, header=None), dtype=np.float32)
    print(f"Loaded resampled data: {feat_15.shape}")

    # Load lag blocks
    lag_blocks = {}
    for lag_label in [f"lag_{l}" for l in range(1, n_lags + 1)] + ["current"]:
        block_path = os.path.join(RESULTS_DIR,
                                   f"{prefix}_ph{ph}_seed{seed}_L{n_lags}_{lag_label}.npy")
        if os.path.exists(block_path):
            lag_blocks[lag_label] = np.load(block_path)
            print(f"  Loaded {lag_label}: shape={lag_blocks[lag_label].shape}")

    if not lag_blocks:
        print("ERROR: No lag blocks found. Run --phase dagma first.")
        return

    # Load adjacency for NoGraph and Physical
    adj = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, config["adj_path"]),
                                header=None), dtype=np.float32)

    # Prepare data sequences
    seq_len = 12
    split = int(feat_15.shape[0] * 0.8)
    feat_max = float(np.max(feat_15[:split]))

    def make_sequences(data, seq_len, ph):
        X, Y = [], []
        for i in range(len(data) - seq_len - ph + 1):
            X.append(data[i:i + seq_len])
            Y.append(data[i + seq_len + ph - 1])
        return np.array(X), np.array(Y)

    X_all, Y_all = make_sequences(feat_15 / feat_max, seq_len, ph)
    X_train, Y_train = X_all[:split - seq_len], Y_all[:split - seq_len]
    X_test, Y_test = X_all[split - seq_len:], Y_all[split - seq_len:]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Build lag-specific graphs (threshold at 0.1)
    thr = 0.1
    adj_matrices = []
    for l in range(1, n_lags + 1):
        key = f"lag_{l}"
        if key in lag_blocks:
            A = (np.abs(lag_blocks[key]) > thr).astype(np.float32)
            np.fill_diagonal(A, 0)
            adj_matrices.append(A)
            print(f"  {key}: {(A > 0).sum()} edges")

    # Methods
    methods = {}
    methods["T-GCN-NoSpatial"] = {"adj": np.eye(N)}
    union_adj = np.zeros((N, N))
    for A in adj_matrices:
        union_adj = np.clip(union_adj + A, 0, 1)
    methods["T-GCN-MultiGSL-Mix"] = {"adj": union_adj}

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for method_name, method_cfg in methods.items():
        print(f"\n--- {method_name} ---")
        
        # ---- FIX: Keep adjacency on CPU when creating model ----
        adj_tensor = torch.FloatTensor(method_cfg["adj"])  # CPU tensor

        # 1. Create TGCN with CPU tensor (buffers will be on CPU)
        model = TGCN(adj_tensor, hidden_dim=64)
        # 2. Move model to device (moves all parameters and buffers)
        model = model.to(device)

        # 3. Projection layer – move to device after creation
        projection = nn.Linear(64, 1).to(device)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(projection.parameters()),
            lr=0.001, weight_decay=0.0001
        )
        criterion = nn.MSELoss()

        # Training
        model.train()
        projection.train()
        for epoch in range(50):
            total_loss = 0
            batch_size = 128
            indices = list(range(len(X_train)))
            random.seed(seed)
            random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]

                batch_x = torch.stack([torch.tensor(X_train[i]) for i in batch_idx]).to(device)  # (B, seq_len, N)
                batch_y = torch.stack([torch.tensor(Y_train[i]) for i in batch_idx]).unsqueeze(-1).to(device)  # (B, N, 1)

                optimizer.zero_grad()
                out = model(batch_x)          # (B, N, 64)
                out = projection(out)         # (B, N, 1)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: loss={total_loss/len(X_train):.6f}")

        # Evaluation
        model.eval()
        projection.eval()
        with torch.no_grad():
            test_x = torch.stack([torch.tensor(X_test[i]) for i in range(len(X_test))]).to(device)
            test_y = torch.stack([torch.tensor(Y_test[i]) for i in range(len(Y_test))]).unsqueeze(-1).to(device)

            pred = model(test_x)
            pred = projection(pred)

            rmse = torch.sqrt(torch.mean((pred - test_y) ** 2)).item()
            mae = torch.mean(torch.abs(pred - test_y)).item()

        results[method_name] = {"rmse": rmse, "mae": mae}
        print(f"  RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Save results (same as before)
    results_path = os.path.join(RESULTS_DIR, f"{prefix}_ph{ph}_seed{seed}_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "dataset": "losloop_15min",
            "ph": ph,
            "seed": seed,
            "interval": "15min",
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Optional comparison with 5-min results (unchanged)
    orig_results_path = os.path.join(PROJECT_ROOT, "results", "stage26_validation",
                                      f"stage26_results_los_ph{ph}_seed{seed}.json")
    if os.path.exists(orig_results_path):
        with open(orig_results_path) as f:
            orig_data = json.load(f)
        print("\n--- COMPARISON: 5-min vs 15-min Los-loop ---")
        print(f"{'Method':<25} {'5-min RMSE':>12} {'15-min RMSE':>14} {'Change':>10}")
        print("-" * 65)
        orig_map = {r["method"]: r["rmse"] for r in orig_data["results"]}
        for method_name, r in results.items():
            orig_key = "NoGraph" if "NoSpatial" in method_name else "GatedMulti_thr0.1"
            if orig_key in orig_map:
                orig_rmse = orig_map[orig_key]
                change = (orig_rmse - r["rmse"]) / orig_rmse * 100
                print(f"{method_name:<25} {orig_rmse:>12.4f} {r['rmse']:>14.4f} {change:>+9.1f}%")


def run_analyze_phase(args):
    """Phase 3: Compare edge structures across resolutions."""
    print("=" * 80)
    print("PHASE 3: Edge Structure Comparison (5-min vs 15-min)")
    print("=" * 80)

    config = DATASET_CONFIGS["losloop"]
    N = config["N"]
    n_lags = args.lags
    seed = args.seed
    ph = args.ph

    def jaccard(A, B):
        s1 = set(zip(*np.where(A > 0)))
        s2 = set(zip(*np.where(B > 0)))
        if len(s1 | s2) == 0:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    # Load 5-min blocks
    print("\n--- 5-min Los-loop ---")
    orig_blocks = {}
    for lag_label in [f"lag_{l}" for l in range(1, n_lags + 1)] + ["current"]:
        path = os.path.join(PROJECT_ROOT, "results", "stage26_validation",
                            f"los_ph{ph}_seed{seed}_L{n_lags}_{lag_label}.npy")
        if os.path.exists(path):
            W = np.load(path)
            orig_blocks[lag_label] = (np.abs(W) > 0.1).astype(float)
            print(f"  {lag_label}: {(orig_blocks[lag_label] > 0).sum()} edges")

    # Load 15-min blocks
    print("\n--- 15-min Los-loop ---")
    new_blocks = {}
    for lag_label in [f"lag_{l}" for l in range(1, n_lags + 1)] + ["current"]:
        path = os.path.join(RESULTS_DIR,
                            f"los_15min_ph{ph}_seed{seed}_L{n_lags}_{lag_label}.npy")
        if os.path.exists(path):
            W = np.load(path)
            new_blocks[lag_label] = (np.abs(W) > 0.1).astype(float)
            print(f"  {lag_label}: {(new_blocks[lag_label] > 0).sum()} edges")

    # Cross-resolution Jaccard (same lag, different resolution)
    print("\n--- Cross-resolution edge overlap (same lag) ---")
    for lag_label in sorted(set(orig_blocks.keys()) & set(new_blocks.keys())):
        jac = jaccard(orig_blocks[lag_label], new_blocks[lag_label])
        print(f"  {lag_label}: Jaccard(5min, 15min) = {jac:.4f}")

    # Cross-lag Jaccard within each resolution
    print("\n--- Cross-lag overlap (within 15-min) ---")
    lag_keys = [f"lag_{l}" for l in range(1, n_lags + 1)]
    for i, k1 in enumerate(lag_keys):
        for k2 in lag_keys[i+1:]:
            if k1 in new_blocks and k2 in new_blocks:
                jac = jaccard(new_blocks[k1], new_blocks[k2])
                print(f"  {k1} vs {k2}: Jaccard = {jac:.4f}")

    print("\n--- Cross-lag overlap (within 5-min) ---")
    for i, k1 in enumerate(lag_keys):
        for k2 in lag_keys[i+1:]:
            if k1 in orig_blocks and k2 in orig_blocks:
                jac = jaccard(orig_blocks[k1], orig_blocks[k2])
                print(f"  {k1} vs {k2}: Jaccard = {jac:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Stage 27: Resolution Experiment")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["dagma", "evaluate", "analyze", "all"],
                        help="Which phase to run")
    parser.add_argument("--ph", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--lags", type=int, default=3, help="Number of lags")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lambda1", type=float, default=0.01, help="DAGMA L1")
    parser.add_argument("--warm-iter", type=int, default=30000, help="DAGMA warm-up")
    parser.add_argument("--max-iter", type=int, default=60000, help="DAGMA max iter")
    args = parser.parse_args()

    if args.phase in ["dagma", "all"]:
        run_dagma_phase(args)
    if args.phase in ["evaluate", "all"]:
        run_evaluate_phase(args)
    if args.phase in ["analyze", "all"]:
        run_analyze_phase(args)


if __name__ == "__main__":
    main()
