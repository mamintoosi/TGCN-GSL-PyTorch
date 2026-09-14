#!/usr/bin/env python3
"""
R1-W5 fairer baseline: sparsified physical graph at the matched 30-edge budget.

Trains plain T-GCN on top-K physical edges (default K=30) under the same
protocol as Stage 32 sparse controls (Los-loop PH1, five seeds, 50 epochs,
batch 128, Adam, mse_with_regularizer). Expected runtime: ~10–20 minutes on
CPU for 5 seeds × 1 method.

Usage (from repo root):
  PYTHON=python bash run_experiments.sh   # not required
  python src/run_phys_topk_control.py
  python src/run_phys_topk_control.py --seeds 42 --epochs 2   # canary
  python src/run_phys_topk_control.py --n-edges 30 --seeds 42 43 44 45 46

Output: results/stage32_sparse_control/phys_topk30_control.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from models.multigsl import physical_topk_graph  # noqa: E402
from run_sparse_controls import (  # noqa: E402
    RESULTS_DIR,
    generate_sequences,
    load_data,
    train_and_eval,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sparsified physical-graph control (R1-W5)"
    )
    parser.add_argument("--dataset", type=str, default="losloop")
    parser.add_argument("--ph", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-edges", type=int, default=30)
    args = parser.parse_args()

    seq_len = 12
    train_norm, test_norm, adj_phys, feat_max = load_data(args.dataset)
    train_X, train_Y = generate_sequences(train_norm, seq_len, args.ph)
    test_X, test_Y = generate_sequences(test_norm, seq_len, args.ph)

    phys_sparse = physical_topk_graph(adj_phys, args.n_edges)
    n_edges = int(phys_sparse.sum())
    label = f"PhysTop{args.n_edges}"
    print(f"Dataset {args.dataset} PH={args.ph} feat_max={feat_max}")
    print(f"{label}: {n_edges} directed edges (full physical off-diag="
          f"{int((np.abs(adj_phys) > 0).sum() - np.trace(adj_phys != 0))})")

    rows = []
    for seed in args.seeds:
        print(f"\n--- Seed {seed}, {label} ---")
        m = train_and_eval(
            phys_sparse,
            train_X,
            train_Y,
            test_X,
            test_Y,
            feat_max,
            args.ph,
            seed=seed,
            max_epochs=args.epochs,
        )
        row = {
            "dataset": args.dataset,
            "ph": args.ph,
            "seed": seed,
            "method": label,
            "model": "TGCN",
            "n_edges": n_edges,
            "rmse": round(m["RMSE"], 4),
            "mae": round(m["MAE"], 4),
            "train_time_s": m["train_time_s"],
        }
        rows.append(row)
        print(
            f"  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  ({m['train_time_s']}s)"
        )

    rmses = [r["rmse"] for r in rows]
    summary = {
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "n_seeds": len(rmses),
    }
    print(
        f"\nSUMMARY {label}: {summary['mean_rmse']:.4f} "
        f"+- {summary['std_rmse']:.4f} (n={summary['n_seeds']})"
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phys_topk_control.json")
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "R1-W5 sparsified physical baseline at matched edge budget",
        "dataset": args.dataset,
        "ph": args.ph,
        "n_edges": args.n_edges,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "protocol": {
            "backbone": "TGCN (static graph)",
            "batch_size": 128,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "hidden_dim": 64,
            "loss": "mse_with_regularizer",
            "optimizer": "Adam",
            "seq_len": 12,
            "feat_max_source": "train split only",
        },
        "results": rows,
        "summary": summary,
        "compare_to": {
            "graph_free_5seed_mean_rmse_ph1_los": 5.25,
            "CorrTop30_5seed_mean_rmse_ph1_los": 5.39,
            "RandTop30_5seed_mean_rmse_ph1_los": 6.10,
            "TGCN_MultiGSL_5seed_mean_rmse_ph1_los": 4.84,
            "TGCN_MultiGSL_Mix_5seed_mean_rmse_ph1_los": 4.49,
            "note": "Reference means from manuscript Table 4 (same protocol family).",
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
