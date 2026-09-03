#!/usr/bin/env python3
"""
Stage 25 — Experiment Family G: Multi-Lag DAGMA Pilot.

Tests whether DAGMA can discover lag-specific temporal dependencies:
  Z(t) = [x(t-L), x(t-L+1), ..., x(t-1), x(t)]

For L lags and N sensors, DAGMA operates on L*N variables.
The cross-lag block W[l*N:(l+1)*N, 0:N] represents lag-l → current.

Pilot: small subset of sensors (N_small) to verify feasibility.

Usage:
  python gsl_stage25/stage25_multilag_pilot.py --n-sensors 20 --lags 3
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage25_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_CONFIGS = {
    "shenzhen": {"feat_path": "data/sz_speed.csv", "adj_path": "data/sz_adj.csv", "N": 156, "prefix": "sz"},
    "losloop": {"feat_path": "data/los_speed.csv", "adj_path": "data/los_adj.csv", "N": 207, "prefix": "los"},
}


def load_raw_data(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    feat = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, config["feat_path"])), dtype=np.float32)
    adj = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, config["adj_path"]), header=None), dtype=np.float32)
    return feat, adj


def build_multilag_Z(data, n_lags, n_sensors=None, top_k_sensors=None):
    """
    Build multi-lag DAGMA input:
    Z = [x(t-L), x(t-L+1), ..., x(t-1), x(t)]

    Args:
        data: (T, N_full) full sensor data
        n_lags: number of lagged snapshots (L)
        n_sensors: if given, use only first n_sensors sensors
        top_k_sensors: if given, select top_k most variable sensors

    Returns:
        Z: (T - n_lags, L * N_small) multi-lag matrix
        sensor_indices: which sensors were selected
    """
    T, N_full = data.shape

    if n_sensors is not None and n_sensors < N_full:
        if top_k_sensors is not None:
            # Select most variable sensors
            variance = np.var(data, axis=0)
            sensor_indices = np.argsort(variance)[::-1][:n_sensors]
        else:
            sensor_indices = np.arange(n_sensors)
    else:
        sensor_indices = np.arange(N_full)
        n_sensors = N_full

    data_small = data[:, sensor_indices]  # (T, N_small)
    N_small = n_sensors

    # Normalize
    max_val = float(np.max(data_small))
    data_norm = data_small / max_val

    # Build Z
    rows = []
    for t in range(n_lags, T):
        z = np.concatenate([data_norm[t - l] for l in range(n_lags, 0, -1)] +
                          [data_norm[t]])  # [x(t-L), ..., x(t-1), x(t)]
        rows.append(z)

    Z = np.array(rows, dtype=np.float32)
    return Z, sensor_indices, max_val


def run_dagma(Z, lambda1=0.01, seed=42, warm_iter=30000, max_iter=60000):
    """Run DAGMA on the multi-lag input."""
    from dagma.linear import DagDagmaLinear

    model = DagDagmaLinear(lambda1=lambda1)
    W_est, loss = model.fit(Z, warm_iter=warm_iter, max_iter=max_iter)
    return W_est, loss


def analyze_multilag_W(W_est, n_lags, N_small, threshold=0.1):
    """
    Analyze the multi-lag DAGMA weight matrix.

    For Z = [x(t-L), ..., x(t-1), x(t)]:
      W[l*N_small:(l+1)*N_small, 0:N_small] represents lag-(L-l) → current

    Returns per-lag analysis.
    """
    results = {}
    for lag_idx in range(n_lags + 1):
        start_row = lag_idx * N_small
        end_row = (lag_idx + 1) * N_small
        W_block = W_est[start_row:end_row, 0:N_small]  # lag → current

        n_nonzero = int(np.sum(np.abs(W_block) > threshold))
        abs_weights = np.abs(W_block)
        nonzero_weights = abs_weights[abs_weights > 0]

        lag_label = f"lag_{n_lags - lag_idx}" if lag_idx < n_lags else "current_self"

        results[lag_label] = {
            "block_shape": W_block.shape,
            "n_nonzero_thr": n_nonzero,
            "density": round(n_nonzero / (N_small * N_small), 4),
            "max_weight": round(float(abs_weights.max()), 6) if abs_weights.size > 0 else 0,
            "mean_abs_nonzero": round(float(nonzero_weights.mean()), 6) if nonzero_weights.size > 0 else 0,
        }

        # Top 5 edges in this block
        flat_idx = np.argsort(abs_weights.flatten())[::-1][:5]
        top_edges = []
        for idx in flat_idx:
            i, j = divmod(idx, N_small)
            if abs_weights[i, j] > 0:
                top_edges.append({
                    "source": int(i), "target": int(j),
                    "weight": round(float(W_block[i, j]), 6),
                })
        results[lag_label]["top_edges"] = top_edges

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 25: Multi-Lag DAGMA Pilot")
    parser.add_argument("--dataset", type=str, default="shenzhen", choices=["shenzhen", "losloop"])
    parser.add_argument("--n-sensors", type=int, default=20, help="Number of sensors (pilot)")
    parser.add_argument("--lags", type=int, default=3, help="Number of lags L")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda1", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=60000)
    parser.add_argument("--warm-iter", type=int, default=30000)
    args = parser.parse_args()

    dataset = args.dataset
    config = DATASET_CONFIGS[dataset]
    N_full = config["N"]
    prefix = config["prefix"]

    print("=" * 80)
    print(f"STAGE 25 — MULTI-LAG DAGMA PILOT ({dataset})")
    print(f"Sensors: {args.n_sensors}/{N_full}, Lags: {args.lags}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load data
    feat, adj = load_raw_data(dataset)
    print(f"Raw data: {feat.shape}")

    # Build multi-lag input
    Z, sensor_idx, max_val = build_multilag_Z(
        feat, n_lags=args.lags, n_sensors=args.n_sensors, top_k_sensors=True)
    print(f"Z shape: {Z.shape} ({args.lags + 1} * {args.n_sensors} = {(args.lags + 1) * args.n_sensors} variables)")
    print(f"Selected sensors: {sensor_idx.tolist()}")

    # Run DAGMA
    print(f"\nRunning DAGMA (lambda1={args.lambda1}, max_iter={args.max_iter})...")
    t0 = time.time()
    W_est, loss = run_dagma(Z, lambda1=args.lambda1, seed=args.seed,
                            warm_iter=args.warm_iter, max_iter=args.max_iter)
    runtime = time.time() - t0
    print(f"DAGMA completed in {runtime:.1f}s")
    print(f"W_est shape: {W_est.shape}")
    print(f"Final loss: {loss}")

    # Analyze per-lag
    print(f"\n--- Per-lag analysis ---")
    analysis = analyze_multilag_W(W_est, args.lags, args.n_sensors, threshold=0.1)

    for lag_label, stats in analysis.items():
        print(f"\n  {lag_label}:")
        print(f"    Nonzero edges (thr=0.1): {stats['n_nonzero_thr']}")
        print(f"    Density: {stats['density']}")
        print(f"    Max weight: {stats['max_weight']}")
        print(f"    Mean |w| (nonzero): {stats['mean_abs_nonzero']}")
        if stats["top_edges"]:
            print(f"    Top edges:")
            for e in stats["top_edges"][:3]:
                print(f"      sensor_{e['source']} -> sensor_{e['target']}: {e['weight']:.6f}")

    # Save results
    result = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "dataset": dataset,
        "n_sensors": args.n_sensors,
        "n_lags": args.lags,
        "total_variables": (args.lags + 1) * args.n_sensors,
        "lambda1": args.lambda1,
        "runtime_s": round(runtime, 1),
        "final_loss": round(loss, 6),
        "sensor_indices": sensor_idx.tolist(),
        "per_lag_analysis": analysis,
    }

    json_path = os.path.join(RESULTS_DIR, f"stage25_multilag_pilot_{prefix}_L{args.lags}_N{args.n_sensors}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    npy_path = os.path.join(RESULTS_DIR, f"stage25_multilag_pilot_{prefix}_L{args.lags}_N{args.n_sensors}_W.npy")
    np.save(npy_path, W_est)

    print(f"\nResults saved to: {json_path}")
    print(f"Weight matrix saved to: {npy_path}")

    # Also run at threshold sweep
    print(f"\n--- Threshold sweep ---")
    for thr in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]:
        total_edges = int(np.sum(np.abs(W_est) > thr))
        print(f"  thr={thr:.3f}: {total_edges} total edges")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
