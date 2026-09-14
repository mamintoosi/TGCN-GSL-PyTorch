#!/usr/bin/env python3
"""
Sparsity sweep: matched edge budgets on Los-loop PH1 (T-GCN backbone).

Compares, at several budgets K:
  - RandTopK   : K random directed off-diagonal edges (redrawn per seed)
  - CorrTopK   : top-K |Pearson| training edges
  - NoSpatial  : identity (reference, not edge-matched)

Uses the same training protocol as Stage 32 / Stage 40:
  mse_with_regularizer, Adam lr=1e-3 wd=1e-4, batch 128, 50 epochs,
  seq_len=12, hidden=64, full-batch test, feat_max from train only.

Usage (smoke):
  python gsl_stage58_sparsity/sparsity_edge_sweep.py --budgets 30 --seeds 42 --epochs 2

Usage (full — user runs):
  bash run_sparsity_edge_sweep.sh
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tasks.supervised import SupervisedForecastTask
from models.tgcn import TGCN
from models.multigsl import correlation_topk_graph, random_edge_graph

RESULTS_DIR = PROJECT_ROOT / "results" / "stage58_sparsity_sweep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_los():
    feat = np.array(pd.read_csv(PROJECT_ROOT / "data/los_speed.csv"), dtype=np.float32)
    adj = np.array(
        pd.read_csv(PROJECT_ROOT / "data/los_adj.csv", header=None), dtype=np.float32
    )
    T, N = feat.shape
    split = int(0.8 * T)
    feat_max = float(feat[:split].max())
    return feat[:split] / feat_max, feat[split:] / feat_max, adj, N, feat_max


def generate_sequences(data, seq_len, pre_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pre_len):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len : i + seq_len + pre_len])
    return np.array(X, np.float32), np.array(Y, np.float32)


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_eval(adj, train_xy, test_xy, feat_max, ph, seed, args, device):
    set_seed(seed)
    model = TGCN(adj=adj, hidden_dim=args.hidden_dim)
    task = SupervisedForecastTask(
        model=model,
        loss="mse_with_regularizer",
        pre_len=ph,
        learning_rate=args.lr,
        weight_decay=args.wd,
        feat_max_val=feat_max,
    )
    train_X, train_Y = train_xy
    test_X, test_Y = test_xy
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(train_X), torch.tensor(train_Y)
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(test_X), torch.tensor(test_Y)
        ),
        batch_size=len(test_X),
        shuffle=False,
    )
    task.model.to(device)
    if task.regressor is not None:
        task.regressor.to(device)
    opt = task.configure_optimizer()
    t0 = time.time()
    for _ in range(args.epochs):
        task.model.train()
        if task.regressor is not None:
            task.regressor.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = task.training_step((xb, yb))
            loss.backward()
            opt.step()
    metrics = task.validation_epoch(test_loader, device)
    nn = adj.shape[0]
    n_offdiag = int((np.abs(adj) > 0).sum())
    if np.allclose(adj, np.eye(nn), atol=1e-6):
        n_offdiag = 0  # identity / NoSpatial
    return {
        "RMSE": round(float(metrics["RMSE"]), 4),
        "MAE": round(float(metrics["MAE"]), 4),
        "train_seconds": round(time.time() - t0, 1),
        "n_edges": n_offdiag,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 30, 50, 80])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--ph", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--wd", type=float, default=0.0001)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dataset", default="losloop")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out-name", default="sparsity_sweep")
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    train_n, test_n, adj_phys, N, feat_max = load_los()
    train_xy = generate_sequences(train_n, args.seq_len, args.ph)
    test_xy = generate_sequences(test_n, args.seq_len, args.ph)

    rows = []
    jsonl = RESULTS_DIR / f"{args.out_name}.jsonl"
    # NoSpatial reference once per seed
    for seed in args.seeds:
        print(f"=== NoSpatial seed {seed} ===")
        rec = train_eval(
            np.eye(N, dtype=np.float32),
            train_xy,
            test_xy,
            feat_max,
            args.ph,
            seed,
            args,
            device,
        )
        rec.update(
            method="NoSpatial",
            budget=0,
            dataset=args.dataset,
            ph=args.ph,
            seed=seed,
            epochs=args.epochs,
        )
        rows.append(rec)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(" ", rec["RMSE"])

    for K in args.budgets:
        for method in ("RandTopK", "CorrTopK"):
            for seed in args.seeds:
                print(f"=== {method} K={K} seed {seed} ===")
                if method == "RandTopK":
                    A = random_edge_graph(N, K, seed=seed)
                else:
                    A = correlation_topk_graph(train_n, K)
                rec = train_eval(
                    A.astype(np.float32),
                    train_xy,
                    test_xy,
                    feat_max,
                    args.ph,
                    seed,
                    args,
                    device,
                )
                rec.update(
                    method=method,
                    budget=K,
                    dataset=args.dataset,
                    ph=args.ph,
                    seed=seed,
                    epochs=args.epochs,
                )
                rows.append(rec)
                with open(jsonl, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(" ", rec["RMSE"])

    csv_path = RESULTS_DIR / f"{args.out_name}.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("Wrote", csv_path)

    # summary
    df = pd.DataFrame(rows)
    print("\n=== mean ± std by method/budget ===")
    g = df.groupby(["method", "budget"])["RMSE"].agg(["mean", "std", "count"])
    print(g.to_string())


if __name__ == "__main__":
    main()
