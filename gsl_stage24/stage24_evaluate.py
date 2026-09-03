#!/usr/bin/env python3
"""
Stage 24 — Evaluate corrected temporal DAGMA across PH=1,2,3,4.

Reuses existing DAGMA matrices. No DAGMA computation needed.

Usage:
  python gsl_stage24/stage24_evaluate.py
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime
from typing import Dict, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.gcn import GCN
from models.tgcn import TGCN
from tasks.supervised import SupervisedForecastTask

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage24_validation")
STAGE20_DIR = os.path.join(PROJECT_ROOT, "results", "stage20_5_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_normalize_train_only(dataset_name="shenzhen", split_ratio=0.8):
    paths = {
        "shenzhen": ("data/sz_speed.csv", "data/sz_adj.csv"),
    }
    feat = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, paths[dataset_name][0])), dtype=np.float32)
    adj = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, paths[dataset_name][1]), header=None), dtype=np.float32)
    T, N = feat.shape
    train_size = int(T * split_ratio)
    feat_max_val = float(np.max(feat[:train_size]))
    return feat[:train_size] / feat_max_val, feat[train_size:] / feat_max_val, adj, feat_max_val


def generate_sequences(data, seq_len, pre_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pre_len):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len:i + seq_len + pre_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_evaluate(adj, model_name, train_X, train_Y, test_X, test_Y,
                        feat_max_val, pre_len=1, seq_len=12, seed=42, max_epochs=50):
    set_seed(seed)
    if model_name == "GCN":
        model = GCN(adj=adj, seq_len=seq_len, hidden_dim=64)
        loss_name = "mse"
    else:
        model = TGCN(adj=adj, hidden_dim=64)
        loss_name = "mse_with_regularizer"
    
    task = SupervisedForecastTask(model=model, loss=loss_name, pre_len=pre_len,
                                   learning_rate=0.001, weight_decay=0.0001,
                                   feat_max_val=feat_max_val)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if task.regressor is not None:
        task.regressor = task.regressor.to(device)
    
    optimizer = task.configure_optimizer()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(train_X), torch.FloatTensor(train_Y)),
        batch_size=128, shuffle=True)
    
    t0 = time.time()
    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = task.training_step((xb, yb))
            loss.backward()
            optimizer.step()
    train_time = time.time() - t0
    
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_Y)),
        batch_size=len(test_X), shuffle=False)
    metrics = task.validation_epoch(test_loader, device)
    metrics["train_time_s"] = round(train_time, 2)
    return metrics


def main():
    print("=" * 80)
    print("STAGE 24 — MULTI-PH EVALUATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    t_total = time.time()
    train_data, test_data, adj_phys, feat_max = load_and_normalize_train_only()
    N = 156
    
    all_results = []
    
    for ph in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"PH={ph}")
        print(f"{'='*60}")
        
        # Load DAGMA matrix
        if ph == 1:
            W_path = os.path.join(STAGE20_DIR, "sz_ph1_W_raw_temporal.npy")
        else:
            W_path = os.path.join(RESULTS_DIR, f"sz_ph{ph}_W_raw_temporal.npy")
        
        if not os.path.exists(W_path):
            print(f"  WARNING: {W_path} not found. Skipping PH={ph}.")
            print(f"  Run: python gsl_stage24/stage24_run_dagma.py --ph {ph}")
            continue
        
        W_raw = np.load(W_path)
        W_cross = W_raw[0:N, N:2*N]  # CORRECT block
        print(f"  Loaded W: {W_raw.shape}, correct block: {W_cross.shape}")
        
        # Generate sequences for this PH
        train_X, train_Y = generate_sequences(train_data, 12, ph)
        test_X, test_Y = generate_sequences(test_data, 12, ph)
        print(f"  Train: {train_X.shape}, Test: {test_Y.shape}")
        
        # Baselines
        baselines = {
            "NoGraph": np.eye(N, dtype=np.float32),
            "Physical": adj_phys,
            "Corr-K8": None,  # built below
            "Corr-K16": None,
        }
        
        # Build correlation graphs
        corr = np.corrcoef(train_data.T)
        corr = np.nan_to_num(corr, nan=0.0)
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 0)
        upper = np.triu_indices(N, k=1)
        
        for k_name, k_val in [("Corr-K8", 8), ("Corr-K16", 16)]:
            adj = np.zeros((N, N), dtype=np.float32)
            sorted_idx = np.argsort(abs_corr[upper])[::-1]
            for i in range(min(k_val, len(sorted_idx))):
                r, c = upper[0][sorted_idx[i]], upper[1][sorted_idx[i]]
                adj[r, c] = 1.0
                adj[c, r] = 1.0
            baselines[k_name] = adj
        
        # Temporal DAGMA thresholds
        thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
        
        # Evaluate baselines
        for bname, badj in baselines.items():
            adj_eval = badj.copy()
            np.fill_diagonal(adj_eval, 0)
            n_edges = int(np.sum(adj_eval > 0))
            for model_name in ["GCN", "TGCN"]:
                set_seed(42)
                m = train_and_evaluate(adj_eval, model_name, train_X, train_Y,
                                       test_X, test_Y, feat_max, ph)
                all_results.append({
                    "ph": ph, "method": bname, "model": model_name,
                    "n_edges": n_edges, "rmse": round(m["RMSE"], 4),
                    "mae": round(m["MAE"], 4), "r2": round(m["R2"], 6),
                    "block": "baseline", "threshold": None,
                })
            print(f"  {bname:12s}: {n_edges:5d} edges")
        
        # Evaluate temporal DAGMA at different thresholds
        for thr in thresholds:
            adj = (np.abs(W_cross) > thr).astype(np.float32)
            np.fill_diagonal(adj, 0)  # Remove self-loops
            n_edges = int(np.sum(adj > 0))
            for model_name in ["GCN", "TGCN"]:
                set_seed(42)
                m = train_and_evaluate(adj, model_name, train_X, train_Y,
                                       test_X, test_Y, feat_max, ph)
                all_results.append({
                    "ph": ph, "method": f"TempDAGMA_thr{thr}", "model": model_name,
                    "n_edges": n_edges, "rmse": round(m["RMSE"], 4),
                    "mae": round(m["MAE"], 4), "r2": round(m["R2"], 6),
                    "block": "correct_W0_N_2N", "threshold": thr,
                })
            print(f"  TempDAGMA_{thr:.3f}: {n_edges:5d} edges")
        
        # Also evaluate with self-loops retained
        adj_selfloop = (np.abs(W_cross) > 0.2).astype(np.float32)
        n_edges_sl = int(np.sum(adj_selfloop > 0))
        for model_name in ["GCN", "TGCN"]:
            set_seed(42)
            m = train_and_evaluate(adj_selfloop, model_name, train_X, train_Y,
                                   test_X, test_Y, feat_max, ph)
            all_results.append({
                "ph": ph, "method": "TempDAGMA_thr0.2_selfloop", "model": model_name,
                "n_edges": n_edges_sl, "rmse": round(m["RMSE"], 4),
                "mae": round(m["MAE"], 4), "r2": round(m["R2"], 6),
                "block": "correct_W0_N_2N", "threshold": 0.2,
                "self_loop": True,
            })
        print(f"  TempDAGMA_0.2_sl:  {n_edges_sl:5d} edges (self-loops retained)")
    
    # Save results
    csv_path = os.path.join(RESULTS_DIR, "stage24_results.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    
    # Summary table
    print("\n" + "=" * 100)
    print("STAGE 24 SUMMARY")
    print("=" * 100)
    for ph in [1, 2, 3, 4]:
        ph_results = [r for r in all_results if r["ph"] == ph]
        if not ph_results:
            continue
        print(f"\n--- PH={ph} ---")
        print(f"{'Method':30s} | {'Model':5s} | {'Edges':>6s} | {'RMSE':>8s} | {'MAE':>8s}")
        print("-" * 80)
        for r in ph_results:
            print(f"{r['method']:30s} | {r['model']:5s} | {r['n_edges']:6d} | {r['rmse']:8.4f} | {r['mae']:8.4f}")
    
    # Save summary JSON
    summary = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "n_results": len(all_results),
        "phs_evaluated": list(set(r["ph"] for r in all_results)),
    }
    with open(os.path.join(RESULTS_DIR, "stage24_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTotal time: {time.time()-t_total:.1f}s ({(time.time()-t_total)/60:.1f} min)")
    print(f"Results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
