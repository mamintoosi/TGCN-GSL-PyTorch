#!/usr/bin/env python3
"""
Stage 33 — SZ-Taxi multi-seed validation under the canonical pipeline.

Purpose
-------
The main-text SZ-Taxi table (Table tab:sz_multiph) currently reports a single
seed (seed 42). This script extends the verified Stage 26 SZ-Taxi protocol to
the same five-seed reporting standard as the Los-loop multi-seed table,
reusing the EXISTING SZ DAGMA lag blocks (results/stage26_validation/
sz_ph*_seed42_L3_*.npy) — no DAGMA recomputation.

Scientific question: is the marginal SZ-Taxi improvement (≤0.3% at PH1-3,
-0.02% at PH4, single seed) stable across seeds, or is the PH=4 dip seed
noise? This directly supports the manuscript's dataset-dependence claim.

Methods follow the canonical manuscript terminology:
  T-GCN-NoSpatial, T-GCN-MultiGSL, T-GCN-MultiGSL-Mix
(legacy JSON keys retained: NoGraph, MultiGraphTGCN_fixed,
GatedMultiGraphTGCN — see doc/METHOD_NAMING_MAP.md).

Usage:
  python gsl_stage26/stage33_sz_multiseed.py --seeds 42 43 44 45 46 --phs 1 2 3 4
  python gsl_stage26/stage33_sz_multiseed.py --seeds 42 --phs 1 --epochs 2   # smoke test
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tasks.supervised import SupervisedForecastTask
from models.multigsl import GatedMultiGraphTGCN, MultiGraphTGCNFixed, binary_graph
from models.tgcn import TGCN

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage33_sz_multiseed")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_CONFIGS = {
    "shenzhen": {
        "feat_path": "data/sz_speed.csv",
        "adj_path": "data/sz_adj.csv",
        "N": 156, "prefix": "sz",
    },
}


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


def load_multilag_blocks(ph, seed=42, n_lags=3):
    """Load the EXISTING SZ DAGMA lag blocks (Stage 26 artifacts, seed 42)."""
    prefix = DATASET_CONFIGS["shenzhen"]["prefix"]
    results_dir = os.path.join(PROJECT_ROOT, "results", "stage26_validation")
    lag_blocks = {}
    for lag_label in [f"lag_{l}" for l in range(1, n_lags + 1)] + ["current"]:
        path = os.path.join(results_dir,
                             f"{prefix}_ph{ph}_seed{seed}_L{n_lags}_{lag_label}.npy")
        if os.path.exists(path):
            lag_blocks[lag_label] = np.load(path)
    return lag_blocks if lag_blocks else None


# ============================================================
# CANONICAL TRAIN/EVAL (identical to stage26_validation.py)
# ============================================================
def train_and_eval(adj_or_model_factory, model_type, train_X, train_Y, test_X, test_Y,
                   feat_max, pre_len, seed=42, max_epochs=50, hidden_dim=64):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "standard":
        model = TGCN(adj=adj_or_model_factory, hidden_dim=hidden_dim)
    elif model_type == "gated_multi":
        model = GatedMultiGraphTGCN(adj_list=adj_or_model_factory, hidden_dim=hidden_dim)
    elif model_type == "multi_graph_fixed":
        model = MultiGraphTGCNFixed(adj_list=adj_or_model_factory, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

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
    return metrics


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stage 33: SZ-Taxi multi-seed validation (canonical pipeline)")
    parser.add_argument("--phs", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    N = DATASET_CONFIGS["shenzhen"]["N"]
    print("=" * 80)
    print("STAGE 33 — SZ-TAXI MULTI-SEED VALIDATION (canonical pipeline)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PHs: {args.phs}, Seeds: {args.seeds}, epochs: {args.epochs}")
    print("=" * 80)

    train_data, test_data, adj_phys, feat_max = load_data("shenzhen")
    all_results = []

    for ph in args.phs:
        print(f"\n{'=' * 70}\nPH={ph}\n{'=' * 70}")

        lag_blocks = load_multilag_blocks(ph, seed=42, n_lags=3)
        if lag_blocks is None:
            print(f"ERROR: SZ DAGMA blocks missing for PH={ph} "
                  f"(results/stage26_validation/sz_ph{ph}_seed42_L3_*.npy).")
            continue
        lag_keys = sorted([k for k in lag_blocks if k.startswith("lag_")],
                           key=lambda x: int(x.split("_")[1]))
        adj_list = [binary_graph(lag_blocks[k], args.threshold) for k in lag_keys]
        total_edges = sum(int(a.sum()) for a in adj_list)
        print(f"  Lag graphs: {[int(a.sum()) for a in adj_list]} edges "
              f"(sum={total_edges})")

        train_X, train_Y = generate_sequences(train_data, 12, ph)
        test_X, test_Y = generate_sequences(test_data, 12, ph)
        print(f"  Train: {train_X.shape}, Test: {test_X.shape}")

        for seed in args.seeds:
            # T-GCN-NoSpatial
            adj_id = np.eye(N, dtype=np.float32)
            m = train_and_eval(adj_id, "standard", train_X, train_Y,
                               test_X, test_Y, feat_max, ph, seed, args.epochs)
            all_results.append({
                "dataset": "shenzhen", "ph": ph, "seed": seed,
                "method": "NoGraph", "canonical_name": "T-GCN-NoSpatial",
                "model": "TGCN",
                "n_edges": N, "rmse": round(m["RMSE"], 4),
                "mae": round(m["MAE"], 4), "n_params": m["n_params"],
            })
            print(f"  seed {seed} NoSpatial:    RMSE={m['RMSE']:.4f}")

            # T-GCN-MultiGSL
            m = train_and_eval(adj_list, "multi_graph_fixed", train_X, train_Y,
                               test_X, test_Y, feat_max, ph, seed, args.epochs)
            all_results.append({
                "dataset": "shenzhen", "ph": ph, "seed": seed,
                "method": "MultiGraphTGCN_fixed", "canonical_name": "T-GCN-MultiGSL",
                "model": "MultiGraphTGCNFixed",
                "n_edges": total_edges, "rmse": round(m["RMSE"], 4),
                "mae": round(m["MAE"], 4), "n_params": m["n_params"],
            })
            print(f"  seed {seed} MultiGSL:     RMSE={m['RMSE']:.4f}")

            # T-GCN-MultiGSL-Mix
            m = train_and_eval(adj_list, "gated_multi", train_X, train_Y,
                               test_X, test_Y, feat_max, ph, seed, args.epochs)
            all_results.append({
                "dataset": "shenzhen", "ph": ph, "seed": seed,
                "method": "GatedMultiGraphTGCN", "canonical_name": "T-GCN-MultiGSL-Mix",
                "model": "GatedMultiGraphTGCN",
                "n_edges": total_edges, "rmse": round(m["RMSE"], 4),
                "mae": round(m["MAE"], 4), "n_params": m["n_params"],
            })
            print(f"  seed {seed} MultiGSL-Mix: RMSE={m['RMSE']:.4f}")

    json_path = os.path.join(RESULTS_DIR, "stage33_sz_multiseed_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "shenzhen", "seeds": args.seeds, "phs": args.phs,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Summary
    print("\nSUMMARY (mean +- std over seeds)")
    for ph in args.phs:
        for method, canon in [("NoGraph", "T-GCN-NoSpatial"),
                              ("MultiGraphTGCN_fixed", "T-GCN-MultiGSL"),
                              ("GatedMultiGraphTGCN", "T-GCN-MultiGSL-Mix")]:
            rmses = [r["rmse"] for r in all_results
                     if r["method"] == method and r["ph"] == ph]
            if rmses:
                print(f"  PH={ph}  {canon:20s} {np.mean(rmses):.4f} +- {np.std(rmses):.4f}"
                      f"  (n={len(rmses)})")


if __name__ == "__main__":
    main()
