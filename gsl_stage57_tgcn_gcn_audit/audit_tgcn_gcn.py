#!/usr/bin/env python3
"""
Stage 57 — T-GCN vs GCN fairness audit (loss / regularization isolation).

Reuses Stage 40 data pipeline and model classes. Does not change GSL.

Arms (see STAGE57_TGCN_GCN_AUDIT.md):
  A  GCN   + mse
  B  T-GCN + mse_with_regularizer   (canonical Stage 40 T-GCN)
  C  T-GCN + mse                    (no loss-level L2)
  D  GCN   + mse_with_regularizer   (same loss as T-GCN)
  E  GRU   + mse                    (optional, no graph)

Usage examples:
  # Smoke (2 epochs, Los PH1, seed 42, identity only):
  python gsl_stage57_tgcn_gcn_audit/audit_tgcn_gcn.py --smoke

  # Full command is issued by run_tgcn_gcn_audit.sh
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
from models.gcn import GCN
from models.gru import GRU

RESULTS = PROJECT_ROOT / "results" / "stage57_tgcn_gcn_audit"
RESULTS.mkdir(parents=True, exist_ok=True)

DATASET_CONFIGS = {
    "losloop": {"feat_path": "data/los_speed.csv", "adj_path": "data/los_adj.csv", "N": 207},
    "shenzhen": {"feat_path": "data/sz_speed.csv", "adj_path": "data/sz_adj.csv", "N": 156},
}

ARMS = {
    "A_gcn_mse": {"backbone": "gcn", "loss": "mse", "adj": "auto"},
    "B_tgcn_reg": {"backbone": "tgcn", "loss": "mse_with_regularizer", "adj": "auto"},
    "C_tgcn_mse": {"backbone": "tgcn", "loss": "mse", "adj": "auto"},
    "D_gcn_reg": {"backbone": "gcn", "loss": "mse_with_regularizer", "adj": "auto"},
    "E_gru_mse": {"backbone": "gru", "loss": "mse", "adj": "identity"},
}


def load_data(dataset_name: str):
    cfg = DATASET_CONFIGS[dataset_name]
    feat = np.array(pd.read_csv(PROJECT_ROOT / cfg["feat_path"]), dtype=np.float32)
    adj = np.array(pd.read_csv(PROJECT_ROOT / cfg["adj_path"], header=None), dtype=np.float32)
    T, N = feat.shape
    split = int(0.8 * T)
    train, test = feat[:split], feat[split:]
    feat_max = float(train.max())
    train_norm = train / feat_max
    test_norm = test / feat_max
    return train_norm, test_norm, adj, N, feat_max


def generate_sequences(data, seq_len, pre_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pre_len):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len : i + seq_len + pre_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(task: SupervisedForecastTask) -> int:
    return sum(p.numel() for p in task.parameters() if p.requires_grad)


def build_model(backbone: str, adj_mode: str, adj_phys, N: int, seq_len: int, hidden: int):
    if adj_mode == "identity":
        adj = np.eye(N, dtype=np.float32)
    else:
        adj = adj_phys.astype(np.float32)
    if backbone == "gcn":
        return GCN(adj=adj, seq_len=seq_len, hidden_dim=hidden)
    if backbone == "tgcn":
        return TGCN(adj=adj, hidden_dim=hidden)
    if backbone == "gru":
        return GRU(num_nodes=N, hidden_dim=hidden)
    raise ValueError(backbone)


def train_eval(
    model,
    loss_name: str,
    train_xy,
    test_xy,
    feat_max: float,
    ph: int,
    seed: int,
    max_epochs: int,
    batch_size: int,
    lr: float,
    wd: float,
    device: str,
    hidden: int,
):
    set_seed(seed)
    task = SupervisedForecastTask(
        model=model,
        loss=loss_name,
        pre_len=ph,
        learning_rate=lr,
        weight_decay=wd,
        feat_max_val=feat_max,
    )
    n_params = count_params(task)
    train_X, train_Y = train_xy
    test_X, test_Y = test_xy
    train_X_t = torch.tensor(train_X)
    train_Y_t = torch.tensor(train_Y)
    test_X_t = torch.tensor(test_X)
    test_Y_t = torch.tensor(test_Y)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_X_t, train_Y_t),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_X_t, test_Y_t),
        batch_size=len(test_X_t),
        shuffle=False,
    )
    task.model.to(device)
    if task.regressor is not None:
        task.regressor.to(device)
    opt = task.configure_optimizer()
    t0 = time.time()
    last_train_loss = None
    for _epoch in range(max_epochs):
        task.model.train()
        if task.regressor is not None:
            task.regressor.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = task.training_step((xb, yb))
            loss.backward()
            opt.step()
            last_train_loss = float(loss.item())
    metrics = task.validation_epoch(test_loader, device)
    elapsed = time.time() - t0
    return {
        "n_params": n_params,
        "last_train_loss": last_train_loss,
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "val_loss": metrics["val_loss"],
        "train_seconds": elapsed,
    }


def run_one(
    arm: str,
    dataset: str,
    ph: int,
    seed: int,
    adj_kind: str,
    args,
    train_norm,
    test_norm,
    adj_phys,
    N,
    feat_max,
):
    """adj_kind: identity | physical"""
    cfg = ARMS[arm]
    if cfg["adj"] == "identity":
        adj_kind = "identity"
    elif cfg["adj"] == "auto":
        pass  # use requested adj_kind
    model = build_model(cfg["backbone"], adj_kind, adj_phys, N, args.seq_len, args.hidden_dim)
    train_xy = generate_sequences(train_norm, args.seq_len, ph)
    test_xy = generate_sequences(test_norm, args.seq_len, ph)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    res = train_eval(
        model,
        cfg["loss"],
        train_xy,
        test_xy,
        feat_max,
        ph,
        seed,
        args.max_epochs,
        args.batch_size,
        args.lr,
        args.wd,
        device,
        args.hidden_dim,
    )
    out = {
        "arm": arm,
        "backbone": cfg["backbone"],
        "loss": cfg["loss"],
        "adj_kind": adj_kind,
        "dataset": dataset,
        "ph": ph,
        "seed": seed,
        "seq_len": args.seq_len,
        "hidden_dim": args.hidden_dim,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wd": args.wd,
        "feat_max": feat_max,
        **res,
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=list(ARMS.keys()))
    p.add_argument("--datasets", nargs="+", default=["losloop"])
    p.add_argument("--phs", type=int, nargs="+", default=[1])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--adj-kinds", nargs="+", default=["identity", "physical"],
                   help="For arms with adj=auto. E_gru always identity.")
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--wd", type=float, default=0.0001)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs, losloop PH1 seed42, identity+physical, all arms")
    p.add_argument("--out-name", default=None)
    args = p.parse_args()

    if args.smoke:
        args.max_epochs = 2
        args.datasets = ["losloop"]
        args.phs = [1]
        args.seeds = [42]
        args.arms = list(ARMS.keys())

    rows = []
    jsonl_path = RESULTS / (args.out_name or f"audit_{int(time.time())}.jsonl")
    csv_path = jsonl_path.with_suffix(".csv")

    for dataset in args.datasets:
        train_norm, test_norm, adj_phys, N, feat_max = load_data(dataset)
        for arm in args.arms:
            for adj_kind in args.adj_kinds:
                if ARMS[arm]["adj"] == "identity" and adj_kind != "identity":
                    continue
                if ARMS[arm]["adj"] == "identity" and adj_kind == "identity":
                    pass
                for ph in args.phs:
                    for seed in args.seeds:
                        print(f"=== {arm} | {dataset} | adj={adj_kind} | PH{ph} | seed{seed} ===")
                        rec = run_one(
                            arm, dataset, ph, seed, adj_kind, args,
                            train_norm, test_norm, adj_phys, N, feat_max,
                        )
                        rows.append(rec)
                        with open(jsonl_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                        print(
                            f"  RMSE={rec['RMSE']:.4f} MAE={rec['MAE']:.4f} "
                            f"params={rec['n_params']} loss={rec['loss']}"
                        )

    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {csv_path}")
        print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
