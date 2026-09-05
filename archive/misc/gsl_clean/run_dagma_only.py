#!/usr/bin/env python3
"""
Fresh DAGMA-only benchmark on traffic datasets.
No GCN, no T-GCN, no forecasting — pure DAGMA execution and profiling.

Usage:
    python gsl_clean/run_dagma_only.py --dataset sz --horizons 1 2 3 4 --w-threshold 0.3
    python gsl_clean/run_dagma_only.py --dataset both --horizons 1 2 3 4 --w-threshold 0.3
"""

import argparse
import json
import os
import sys
import time
import random

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dagma.linear import DagmaLinear
from utils.data.functions import load_features, generate_dataset


# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Dataset configs ──────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "sz": {
        "name": "shenzhen",
        "feat_path": "data/sz_speed.csv",
        "lambda1": 0.01,
        "N_expected": 156,
    },
    "los": {
        "name": "losloop",
        "feat_path": "data/los_speed.csv",
        "lambda1": 0.02,
        "N_expected": 207,
    },
}


def prepare_dagma_input(feat_path: str, seq_len: int, pre_len: int, split_ratio: float = 0.8):
    """
    Reproduce the exact DAGMA input construction from the paper's code.

    Pipeline:
        load CSV -> normalize -> generate train sequences -> extract x[0] -> subsample by PH
    """
    # Load raw features
    feat = load_features(feat_path, dtype=np.float32)
    raw_shape = feat.shape  # (T, N)

    # Generate train sequences (normalize=True, same as paper)
    train_X, train_Y, test_X, test_Y = generate_dataset(
        feat, seq_len=seq_len, pre_len=pre_len,
        split_ratio=split_ratio, normalize=True,
    )
    # train_X shape: (M, seq_len, N)

    # Extract first time step: data = np.array([x[0] for x in self.train_data])
    # This is exactly what SpatioTemporalCSVData.compute_adjacency_matrix() does
    data = train_X[:, 0, :]  # shape: (M, N)
    dagma_full_input = data.copy()

    # Subsample by PH: X = data[i::PH]
    X_sub = data[0::pre_len].copy()  # shape: (M_sub, N)

    return {
        "raw_shape": raw_shape,
        "train_X_shape": train_X.shape,
        "data_shape": data.shape,
        "dagma_full_input": dagma_full_input,
        "dagma_full_input_shape": dagma_full_input.shape,
        "X_sub_shape": X_sub.shape,
        "N": feat.shape[1],
        "M_full": data.shape[0],
        "M_sub": X_sub.shape[0],
        "feat_path": feat_path,
        "seq_len": seq_len,
        "pre_len": pre_len,
    }


def run_dagma(
    X: np.ndarray,
    lambda1: float,
    w_threshold: float,
    loss_type: str = "l2",
    verbose: bool = True,
):
    """
    Run fresh DAGMA. No cached W_est files are loaded.
    """
    print(f"\n{'='*70}")
    print(f"WARNING: Running FRESH DAGMA. No cached W_est is being used.")
    print(f"{'='*70}")
    print(f"  X shape: {X.shape}  (N={X.shape[1]}, M={X.shape[0]})")
    print(f"  lambda1: {lambda1}")
    print(f"  w_threshold: {w_threshold}")
    print(f"  loss_type: {loss_type}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  DAGMA device: CPU (pure numpy)")
    print()

    t0 = time.perf_counter()
    model = DagmaLinear(loss_type=loss_type, verbose=verbose)
    W = model.fit(
        X,
        lambda1=lambda1,
        w_threshold=w_threshold,
    )
    elapsed = time.perf_counter() - t0

    return W, elapsed


def compute_w_statistics(W: np.ndarray, N: int):
    """Compute comprehensive statistics of the resulting W matrix."""
    abs_W = np.abs(W)
    stats = {
        "shape": list(W.shape),
        "min": float(np.min(W)),
        "max": float(np.max(W)),
        "mean": float(np.mean(W)),
        "std": float(np.std(W)),
        "num_nonzero": int(np.count_nonzero(W)),
        "num_positive": int(np.sum(W > 0)),
        "num_negative": int(np.sum(W < 0)),
        "num_abs_gte_0001": int(np.sum(abs_W >= 0.001)),
        "num_abs_gte_001": int(np.sum(abs_W >= 0.01)),
        "num_abs_gte_01": int(np.sum(abs_W >= 0.1)),
        "num_abs_gte_03": int(np.sum(abs_W >= 0.3)),
        "density": float(np.count_nonzero(W) / (N * N)),
        "diagonal_nonzeros": int(np.count_nonzero(np.diag(W))),
        "offdiagonal_nonzeros": int(np.count_nonzero(W)) - int(np.count_nonzero(np.diag(W))),
        "total_entries": N * N,
    }
    return stats


def print_w_statistics(stats: dict, label: str):
    """Pretty-print W statistics."""
    print(f"\n--- W Statistics for {label} ---")
    print(f"  shape:          {stats['shape']}")
    print(f"  min:            {stats['min']:.6f}")
    print(f"  max:            {stats['max']:.6f}")
    print(f"  mean:           {stats['mean']:.6f}")
    print(f"  std:            {stats['std']:.6f}")
    print(f"  nonzero:        {stats['num_nonzero']}")
    print(f"  positive:       {stats['num_positive']}")
    print(f"  negative:       {stats['num_negative']}")
    print(f"  |W| >= 0.001:  {stats['num_abs_gte_0001']}")
    print(f"  |W| >= 0.01:   {stats['num_abs_gte_001']}")
    print(f"  |W| >= 0.1:    {stats['num_abs_gte_01']}")
    print(f"  |W| >= 0.3:    {stats['num_abs_gte_03']}")
    print(f"  density:        {stats['density']:.6f}")
    print(f"  diag nonzero:   {stats['diagonal_nonzeros']}")
    print(f"  offdiag nonzero:{stats['offdiagonal_nonzeros']}")


def main():
    parser = argparse.ArgumentParser(description="Fresh DAGMA-only benchmark")
    parser.add_argument("--dataset", type=str, default="sz",
                        choices=["sz", "los", "both"],
                        help="Dataset: sz (SZ-Taxi), los (Los-loop), or both")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Prediction horizons to run DAGMA on")
    parser.add_argument("--w-threshold", type=float, default=0.3,
                        help="DAGMA w_threshold (default: 0.3)")
    parser.add_argument("--lambda1", type=float, default=None,
                        help="Override lambda1 (default: per-dataset)")
    parser.add_argument("--seq-len", type=int, default=12,
                        help="Sequence length (default: 12)")
    parser.add_argument("--split-ratio", type=float, default=0.8,
                        help="Train/test split ratio (default: 0.8)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DAGMA verbose output")
    parser.add_argument("--save-w", action="store_true", default=True,
                        help="Save W matrices to results/dagma_fresh/")
    args = parser.parse_args()

    print("=" * 70)
    print("FRESH DAGMA BENCHMARK — No cached W_est files")
    print("=" * 70)
    print(f"  Seed:          {SEED}")
    print(f"  CUDA:          {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
    print(f"  DAGMA device:  CPU (pure numpy implementation)")
    print(f"  DAGMA default: warm_iter=30000, max_iter=60000, T=5")
    print(f"  Total iter:    4 × 30000 + 60000 = 180000 per DAGMA call")
    print(f"  w_threshold:   {args.w_threshold}")
    print()

    # Create output directory
    out_dir = "results/dagma_fresh"
    os.makedirs(out_dir, exist_ok=True)

    # Determine datasets to run
    datasets_to_run = []
    if args.dataset in ("sz", "both"):
        datasets_to_run.append(("sz", DATASET_CONFIGS["sz"]))
    if args.dataset in ("los", "both"):
        datasets_to_run.append(("los", DATASET_CONFIGS["los"]))

    all_results = []
    script_start = time.perf_counter()

    for ds_key, ds_cfg in datasets_to_run:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {ds_cfg['name']} ({ds_key})")
        print(f"{'#'*70}")

        # Prepare data once
        info = prepare_dagma_input(
            ds_cfg["feat_path"],
            seq_len=args.seq_len,
            pre_len=1,  # We prepare with PH=1, then subsample later
            split_ratio=args.split_ratio,
        )
        print(f"\n  Raw feature shape:  {info['raw_shape']}")
        print(f"  Train seq shape:    {info['train_X_shape']}")
        print(f"  DAGMA input (full): {info['dagma_full_input_shape']}")
        print(f"  N (nodes):          {info['N']}")
        print(f"  M (observations):   {info['M_full']}")
        print(f"  Lambda1:            {ds_cfg['lambda1']}")

        assert info["N"] == ds_cfg["N_expected"], (
            f"Expected N={ds_cfg['N_expected']}, got {info['N']}"
        )

        lambda1 = args.lambda1 if args.lambda1 is not None else ds_cfg["lambda1"]

        # Full data matrix (PH=1 equivalent — used as reference)
        full_data = info["dagma_full_input"].copy()

        for ph in args.horizons:
            print(f"\n{'─'*70}")
            print(f"  PH={ph}  (subsampled: X = data[0::{ph}])")

            # Subsample by PH
            X = full_data[0::ph].copy()
            N = info["N"]
            M = X.shape[0]

            print(f"  DAGMA input shape:  ({M}, {N})")
            print(f"  Lambda1:            {lambda1}")
            print(f"  w_threshold:        {args.w_threshold}")

            # Validate input
            assert X.shape == (M, N), f"Unexpected shape: {X.shape}"
            assert np.isfinite(X).all(), "Input contains non-finite values!"

            # Run DAGMA
            W, elapsed = run_dagma(
                X, lambda1=lambda1, w_threshold=args.w_threshold,
                verbose=args.verbose,
            )

            # Validate output
            assert W.shape == (N, N), f"Unexpected W shape: {W.shape}"
            assert np.isfinite(W).all(), "W contains non-finite values!"

            # Compute statistics
            stats = compute_w_statistics(W, N)
            print_w_statistics(stats, f"{ds_cfg['name']} PH={ph}")

            print(f"\n  DAGMA runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

            # Save W
            if args.save_w:
                w_path = os.path.join(out_dir, f"{ds_key}_PH{ph}_W.npy")
                np.save(w_path, W)
                print(f"  Saved W to: {w_path}")

            # Record result
            result = {
                "dataset": ds_cfg["name"],
                "dataset_key": ds_key,
                "ph": ph,
                "lambda1": lambda1,
                "w_threshold": args.w_threshold,
                "loss_type": "l2",
                "N": N,
                "M": M,
                "M_full": info["M_full"],
                "runtime_seconds": round(elapsed, 2),
                "runtime_minutes": round(elapsed / 60, 2),
                "w_statistics": stats,
            }
            all_results.append(result)

    script_elapsed = time.perf_counter() - script_start

    print(f"\n{'='*70}")
    print(f"COMPLETE — Total script runtime: {script_elapsed:.2f}s ({script_elapsed/60:.2f}min)")
    print(f"{'='*70}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    header = f"{'Dataset':<12} {'PH':>3} {'N':>4} {'M':>6} {'lambda1':>8} {'w_thr':>6} {'Edges':>7} {'Nonzero':>8} {'Runtime':>10}"
    print(header)
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['dataset']:<12} {r['ph']:>3} {r['N']:>4} {r['M']:>6} "
            f"{r['lambda1']:>8.3f} {r['w_threshold']:>6.2f} "
            f"{r['w_statistics']['num_nonzero']:>7} {r['w_statistics']['num_abs_gte_0001']:>8} "
            f"{r['runtime_seconds']:>8.1f}s"
        )

    # ── Save results ─────────────────────────────────────────────────────────
    report_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dagma_device": "CPU (pure numpy)",
        "dagma_defaults": {
            "warm_iter": 30000,
            "max_iter": 60000,
            "T": 5,
            "total_iterations": 180000,
        },
        "args": {
            "dataset": args.dataset,
            "horizons": args.horizons,
            "w_threshold": args.w_threshold,
            "seq_len": args.seq_len,
            "split_ratio": args.split_ratio,
        },
        "total_script_runtime_seconds": round(script_elapsed, 2),
        "experiments": all_results,
    }

    json_path = os.path.join(out_dir, "dagma_benchmark_report.json")
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)
    print(f"\n  JSON report saved to: {json_path}")


if __name__ == "__main__":
    main()
