#!/usr/bin/env python3
"""
Stage 32 — Sparse-Control Experiment: matched-edge-count sparse baselines.

Answers Reviewer 1, Weakness 5 ("are the gains from the learned structure or
just from sparsity?") at matched edge parity. The Stage 26 oversmoothing table
compares graphs of very different densities (T-GCN-NoSpatial 207, single-lag
DAGMA 6-60, T-GCN-MultiGSL/Mix 30). Here we add, on the SAME 30 directed edges
as T-GCN-MultiGSL-Mix's lag graphs (sum over lag graphs = 12+3+15):

  - CorrTop30 : top-30 |Pearson correlation| training-data edges (directed,
                off-diagonal), trained with the standard TGCN (single static
                graph). Strongest non-DAGMA heuristic per the archived
                112-experiment report.
  - RandTop30 : 30 random off-diagonal directed edges (re-drawn per seed),
                trained with the standard TGCN. Floor control.

Both use the EXACT canonical Stage 26 pipeline:
  SupervisedForecastTask(loss="mse_with_regularizer"), set_seed(),
  generate_sequences(), Adam(lr=0.001, wd=0.0001), batch 128, 50 epochs,
  full-batch test evaluation, feat_max from training data only.

Usage:
  python gsl_stage26/stage32_sparse_control.py --seeds 42 43 44 45 46
  python gsl_stage26/stage32_sparse_control.py --seeds 42 --epochs 3   # canary
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tasks.supervised import SupervisedForecastTask
from models.multigsl import (
    correlation_topk_graph,
    random_edge_graph,
)
from models.tgcn import TGCN

DATASET_CONFIGS = {
    "losloop": {
        "feat_path": "data/los_speed.csv",
        "adj_path": "data/los_adj.csv",
        "N": 207, "prefix": "los",
    },
}
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage32_sparse_control")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CANONICAL DATA / SEED / SEQUENCES (identical to stage26_validation.py)
# ============================================================

def load_data(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    feat = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, config["feat_path"])),
                    dtype=np.float32)
    adj = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, config["adj_path"]),
                               header=None), dtype=np.float32)
    T, N = feat.shape
    train_size = int(T * 0.8)
    feat_max = float(np.max(feat[:train_size]))
    return feat[:train_size] / feat_max, feat[train_size:] / feat_max, adj, feat_max


def generate_sequences(data, seq_len, pre_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pre_len):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len:i + seq_len + pre_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# EDGE-SET CONSTRUCTION (all controls have exactly N_CTRL directed edges)
# ============================================================

def load_multilag_union(dataset, ph, seed=42, n_lags=3, threshold=0.1):
    """Lag graphs actually used by T-GCN-MultiGSL/-Mix.

    Returns (union, per_lag_counts):
      union          - binary (N,N) union of the thresholded lag graphs
      per_lag_counts - dict {lag_label: edge count}

    NOTE (important for edge parity): the canonical Stage 26 pipeline trains
    on the 3 SEPARATE lag graphs and reports total edges as their SUM
    (12+3+15=30 for losloop PH1 seed42). The UNION of those edges is 28.
    The paper's "30 edges" refers to the sum. Matched-budget controls here
    use k=30 (>= union), which is conservative in favor of the controls.
    """
    config = DATASET_CONFIGS[dataset]
    prefix = config["prefix"]
    results_dir = os.path.join(PROJECT_ROOT, "results", "stage26_validation")
    N = config["N"]
    union = np.zeros((N, N), dtype=np.float32)
    per_lag_counts = {}
    for l in range(1, n_lags + 1):
        path = os.path.join(results_dir,
                            f"{prefix}_ph{ph}_seed{seed}_L{n_lags}_lag_{l}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        W = np.load(path)
        A = (np.abs(W) > threshold).astype(np.float32)
        np.fill_diagonal(A, 0)
        per_lag_counts[f"lag_{l}"] = int(A.sum())
        union = np.clip(union + A, 0, 1)
    return union, per_lag_counts


# corr_topk_graph and rand_topk_graph moved to models.multigsl as
# correlation_topk_graph and random_edge_graph (canonical home, no duplication).


# ============================================================
# CANONICAL TRAIN/EVAL (identical to stage29_los15min.py train_and_eval)
# ============================================================

def train_and_eval(adj, train_X, train_Y, test_X, test_Y,
                   feat_max, pre_len, seed=42, max_epochs=50, hidden_dim=64):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TGCN(adj=adj, hidden_dim=hidden_dim)
    task = SupervisedForecastTask(
        model=model, loss="mse_with_regularizer", pre_len=pre_len,
        learning_rate=0.001, weight_decay=0.0001, feat_max_val=feat_max,
    )
    model = model.to(device)
    if task.regressor is not None:
        task.regressor = task.regressor.to(device)

    optimizer = task.configure_optimizer()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
        ),
        batch_size=128, shuffle=True,
    )

    t0 = time.time()
    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = task.training_step((xb, yb))
            loss.backward()
            optimizer.step()
    train_time = time.time() - t0

    model.eval()
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
        ),
        batch_size=len(test_X), shuffle=False,
    )
    metrics = task.validation_epoch(test_loader, device)
    metrics["train_time_s"] = round(train_time, 2)
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    metrics["n_params_total"] = int(metrics["n_params"] + sum(p.numel() for p in task.regressor.parameters()))
    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 32: matched-edge sparse controls")
    parser.add_argument("--dataset", type=str, default="losloop")
    parser.add_argument("--ph", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-edges", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["corr", "rand"],
                        choices=["corr", "rand", "multilag_union"],
                        help="multilag_union re-runs the T-GCN-MultiGSL-Mix graph "
                             "through the plain TGCN (same edges, no multi-lag processing)")
    args = parser.parse_args()
    config = DATASET_CONFIGS[args.dataset]
    N = config["N"]
    seq_len = 12

    train_norm, test_norm, adj_phys, feat_max = load_data(args.dataset)
    train_X, train_Y = generate_sequences(train_norm, seq_len, args.ph)
    test_X, test_Y = generate_sequences(test_norm, seq_len, args.ph)
    print(f"Dataset {args.dataset}: N={N}, PH={args.ph}, feat_max={feat_max}")
    print(f"Train {train_X.shape}, Test {test_X.shape}")

    # Reference: the lag graphs used by the proposed method (seed 42 DAGMA)
    union, per_lag_counts = load_multilag_union(args.dataset, args.ph, seed=42,
                                                threshold=args.threshold)
    n_union = int(union.sum())
    n_sum = sum(per_lag_counts.values())  # canonical 'total edges' = sum across lag graphs
    print(f"Multi-lag graphs: {per_lag_counts}  sum={n_sum}  union={n_union}")
    assert n_sum == args.n_edges, (f"lag-graph sum is {n_sum}, expected {args.n_edges}; "
                                   "pass --n-edges accordingly")

    # Corr control is seed-independent (computed once from training data)
    corr_adj = correlation_topk_graph(train_norm, args.n_edges)
    print(f"CorrTop{args.n_edges} edges: {int(corr_adj.sum())}")

    all_results = []
    for seed in args.seeds:
        for method in args.methods:
            if method == "corr":
                adj, label = corr_adj, f"CorrTop{args.n_edges}"
            elif method == "rand":
                adj, label = random_edge_graph(N, args.n_edges, seed=seed), f"RandTop{args.n_edges}"
            elif method == "multilag_union":
                adj, label = union, "MultiLagUnion(staticTGCN)"
            else:
                raise ValueError(method)

            print(f"\n--- Seed {seed}, {label} ---")
            m = train_and_eval(adj, train_X, train_Y, test_X, test_Y,
                               feat_max, args.ph, seed=seed, max_epochs=args.epochs)
            row = {
                "dataset": args.dataset, "ph": args.ph, "seed": seed,
                "method": label, "model": "TGCN",
                "n_edges": int(adj.sum()),
                "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                "n_params": m["n_params"], "train_time_s": m["train_time_s"],
            }
            all_results.append(row)
            print(f"  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
                  f"({m['train_time_s']}s, device={'cuda' if torch.cuda.is_available() else 'cpu'})")

    json_path = os.path.join(RESULTS_DIR, "stage32_sparse_control.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": args.dataset, "ph": args.ph,
            "n_edges": args.n_edges, "threshold": args.threshold,
            "seeds": args.seeds, "epochs": args.epochs,
            "protocol": {
                "backbone": "TGCN (static graph)",
                "batch_size": 128, "learning_rate": 0.001,
                "weight_decay": 0.0001, "hidden_dim": 64,
                "loss": "mse_with_regularizer", "optimizer": "Adam",
                "seq_len": 12, "feat_max_source": "train split only",
            },
            "multilag_reference": {
                "per_lag_counts": per_lag_counts,
                "sum_edges": n_sum, "union_edges": n_union,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved: {json_path}")

    print("\nSUMMARY (mean +- std over seeds)")
    labels = sorted({r["method"] for r in all_results})
    for label in labels:
        rmses = [r["rmse"] for r in all_results if r["method"] == label]
        print(f"  {label:28s} {np.mean(rmses):.4f} +- {np.std(rmses):.4f}  (n={len(rmses)})")


if __name__ == "__main__":
    main()
