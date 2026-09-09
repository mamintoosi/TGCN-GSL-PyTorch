#!/usr/bin/env python3
"""
Stage 40.3 — Fit missing SZ-Taxi contemporaneous DAGMA graphs (PH=1-4).

This script ONLY fits DAGMA and saves the graph artifacts. It does NOT
train any forecasting models.

Uses the exact same code path as Stage 33 (stage33_gsl_canonical.py):
  - learn_gsl_graph() from stage33_gsl_canonical.py
  - Same preprocessing: train_norm[0::PH]
  - Same DAGMA params: lambda1=0.01, w_threshold=0.3
  - Same support rule: A = 1(|W| > 0), diagonal removed

Output files (consumed by Stage 40 runner):
  results/stage33_gsl_canonical/sz_gsl_ph{1-4}_seed42_W_est.npy
  results/stage33_gsl_canonical/sz_gsl_ph{1-4}_seed42_A_binary.npy

Usage:
  python gsl_stage26/stage40_3_fit_sz_contemporaneous.py           # fit all PHs
  python gsl_stage26/stage40_3_fit_sz_contemporaneous.py --phs 1   # fit PH=1 only
"""
import os
import sys
import time
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import the exact DAGMA fitting function from Stage 33
from gsl_stage26.stage33_gsl_canonical import (
    learn_gsl_graph, load_data, DATASET_CONFIGS, RESULTS_DIR
)
import numpy as np

# Also save metadata to a separate JSON for provenance
METADATA_DIR = RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Stage 40.3: Fit SZ-Taxi contemporaneous DAGMA (PH=1-4)")
    parser.add_argument("--phs", type=int, nargs="+", default=[1, 2, 3, 4],
                        choices=[1, 2, 3, 4])
    parser.add_argument("--warm-iter", type=int, default=30000)
    parser.add_argument("--max-iter", type=int, default=60000)
    args = parser.parse_args()

    dataset = "shenzhen"
    config = DATASET_CONFIGS[dataset]
    N = config["N"]
    prefix = config["prefix"]
    seed = 42
    dagma_kwargs = {"warm_iter": args.warm_iter, "max_iter": args.max_iter}

    print("=" * 78)
    print("STAGE 40.3 — FIT SZ-TAXI CONTEMPORANEOUS DAGMA")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset} (N={N}), PHs: {args.phs}")
    print(f"lambda1={config['lambda1']}, w_threshold=0.3, seed={seed}")
    print(f"Output dir: {RESULTS_DIR}")
    print("=" * 78)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_meta = []
    total_t0 = time.time()

    for ph in args.phs:
        print(f"\n{'=' * 60}")
        print(f"PH={ph}")
        print(f"{'=' * 60}")

        W_path = os.path.join(RESULTS_DIR, f"{prefix}_gsl_ph{ph}_seed{seed}_W_est.npy")
        A_path = os.path.join(RESULTS_DIR, f"{prefix}_gsl_ph{ph}_seed{seed}_A_binary.npy")

        if os.path.exists(A_path):
            print(f"  Already exists: {A_path}")
            print(f"  Skipping (delete file to re-fit)")
            # Load existing for metadata
            W_est = np.load(W_path)
            A = np.load(A_path)
            meta = {
                "dataset": dataset, "ph": ph, "seed": seed,
                "lambda1": config["lambda1"],
                "n_edges": int(A.sum()),
                "n_coefficients": int((W_est != 0).sum()),
                "max_abs_weight": round(float(np.abs(W_est).max()), 6),
                "status": "reused",
            }
            all_meta.append(meta)
            continue

        t0 = time.time()
        W_est, A, meta = learn_gsl_graph(dataset, ph, seed, dagma_kwargs)
        runtime = time.time() - t0

        # Save
        np.save(W_path, W_est)
        np.save(A_path, A)

        # Save per-PH metadata
        meta_path = os.path.join(RESULTS_DIR, f"{prefix}_gsl_ph{ph}_seed{seed}_metadata.json")
        import json
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  W_est saved: {W_path}")
        print(f"  A_binary saved: {A_path}")
        print(f"  Metadata saved: {meta_path}")
        print(f"  Edges: {meta['n_edges']}, coefficients: {meta['n_coefficients_surviving_abs_threshold']}")
        print(f"  Max |w|: {meta['max_abs_weight']:.6f}")
        print(f"  Positive: {meta['n_positive_surviving']}, Negative: {meta['n_negative_surviving']}")
        print(f"  Runtime: {runtime:.1f}s ({runtime/60:.1f} min)")
        all_meta.append(meta)

    total_time = time.time() - total_t0

    # Save summary metadata
    summary_path = os.path.join(RESULTS_DIR, f"{prefix}_gsl_seed{seed}_summary.json")
    import json
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset,
            "N": N,
            "seed": seed,
            "lambda1": config["lambda1"],
            "w_threshold": 0.3,
            "support_rule": "A = 1(|W| > 0), diagonal removed",
            "total_runtime_s": round(total_time, 1),
            "ph_results": all_meta,
        }, f, indent=2)

    print(f"\n{'=' * 78}")
    print(f"COMPLETE in {total_time/60:.1f} min")
    print(f"Summary saved: {summary_path}")
    print(f"{'=' * 78}")

    # Quick validation
    print("\n--- Quick Validation ---")
    for ph in args.phs:
        A_path = os.path.join(RESULTS_DIR, f"{prefix}_gsl_ph{ph}_seed{seed}_A_binary.npy")
        A = np.load(A_path)
        print(f"  PH={ph}: shape={A.shape}, edges={int(A.sum())}, "
              f"binary={set(np.unique(A).tolist()) == {0.0, 1.0}}, "
              f"diag={int(A.diagonal().sum())}")


if __name__ == "__main__":
    main()
