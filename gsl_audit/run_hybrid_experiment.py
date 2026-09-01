#!/usr/bin/env python3
"""
Stage 18-B/C: Hybrid Graph Forecasting Experiment

Tests whether combining DAGMA-learned edges with physical or correlation
structure improves traffic forecasting over individual graph methods.

Graph types tested (SZ-Taxi only):
  1. GSL         — DAGMA thr=0.3, 8 edges (existing baseline)
  2. GSL+Phys    — GSL edges ∪ top-K physical (16 edges total)
  3. GSL+Corr    — GSL edges ∪ top-K correlation (16 edges total)
  4. GSL+PhysC   — 8 GSL + 4 phys + 4 corr = 16 edges
  5. PhysSparseDir — Top-8 physical directed (existing baseline)
  6. Corr-K8     — Top-8 correlation (downsampled)
  7. Corr-K16    — Top-16 correlation (existing baseline)
  8. PhysSparse  — Top-16 physical symmetric (existing baseline)
"""
import os
import sys
import json
import time
import csv
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gsl_clean.data_pipeline import load_data, generate_sequences
from gsl_clean.graph_utils import build_gsl_adjacency, build_cgsl_adjacency, graph_statistics
from models.gcn import GCN
from models.tgcn import TGCN
from tasks.supervised import SupervisedForecastTask

SEED = 42
MAX_EPOCHS = 50
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage18_hybrid")
DATASET = "shenzhen"

os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_gsl_adj_from_fresh_W(dataset_name, pre_len, threshold=0.3):
    """Build GSL adjacency from fresh DAGMA W (not cached)."""
    w_path = os.path.join(PROJECT_ROOT, "results", "dagma_fresh",
                          "sz_PH" + str(pre_len) + "_W.npy")
    if not os.path.exists(w_path):
        raise FileNotFoundError(f"Fresh DAGMA W not found: {w_path}")
    W = np.load(w_path)
    # Apply threshold then binary
    W_thresh = W.copy()
    W_thresh[np.abs(W_thresh) < threshold] = 0
    adj = (W_thresh > 0).astype(np.float32)
    return adj


def build_correlation_top_k(train_data, k):
    """Top-K edges by absolute Pearson correlation (symmetric)."""
    N = train_data.shape[1]
    corr = np.corrcoef(train_data.T)
    corr = np.nan_to_num(corr, nan=0.0)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0)
    upper = np.triu_indices(N, k=1)
    vals = abs_corr[upper]
    sorted_idx = np.argsort(vals)[::-1]
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(min(k, len(sorted_idx))):
        r, c = upper[0][sorted_idx[i]], upper[1][sorted_idx[i]]
        adj[r, c] = 1.0
        adj[c, r] = 1.0
    return adj


def build_phys_top_k_directed(adj_phys, k):
    """Top-K entries from physical adjacency (directed, like PhysSparseDir)."""
    adj_b = (adj_phys > 0).astype(float)
    np.fill_diagonal(adj_b, 0)
    upper = np.triu_indices(adj_b.shape[0], k=1)
    weights = adj_phys[upper]
    sorted_idx = np.argsort(weights)[::-1]
    adj = np.zeros_like(adj_phys, dtype=np.float32)
    for i in range(min(k, len(sorted_idx))):
        r, c = upper[0][sorted_idx[i]], upper[1][sorted_idx[i]]
        adj[r, c] = 1.0  # directed (upper triangle only)
    return adj


def build_phys_top_k_symmetric(adj_phys, k):
    """Top-K edges from physical adjacency (symmetric)."""
    adj_b = (adj_phys > 0).astype(float)
    np.fill_diagonal(adj_b, 0)
    upper = np.triu_indices(adj_b.shape[0], k=1)
    weights = adj_phys[upper]
    sorted_idx = np.argsort(weights)[::-1]
    adj = np.zeros_like(adj_phys, dtype=np.float32)
    for i in range(min(k, len(sorted_idx))):
        r, c = upper[0][sorted_idx[i]], upper[1][sorted_idx[i]]
        adj[r, c] = 1.0
        adj[c, r] = 1.0
    return adj


def build_hybrid_graphs(dataset_name, pre_len, threshold=0.3):
    """Build all hybrid graph variants for one prediction horizon."""
    feat_raw, adj_physical = load_data(dataset_name)
    N = adj_physical.shape[0]

    # Training data for correlation
    T = feat_raw.shape[0]
    train_size = int(T * 0.8)
    max_val = np.max(feat_raw)
    train_data = feat_raw[:train_size] / max_val

    # GSL edges
    gsl_adj = build_gsl_adj_from_fresh_W(dataset_name, pre_len, threshold)
    gsl_edges = set()
    for i in range(N):
        for j in range(N):
            if gsl_adj[i, j] > 0:
                gsl_edges.add((i, j))
    n_gsl = len(gsl_edges)

    # Physical top edges (symmetric, sorted by weight)
    adj_b = (adj_physical > 0).astype(float)
    np.fill_diagonal(adj_b, 0)
    upper = np.triu_indices(N, k=1)
    phys_weights = adj_physical[upper]
    phys_sorted = np.argsort(phys_weights)[::-1]
    phys_top_edges = [(upper[0][idx], upper[1][idx]) for idx in phys_sorted]

    # Correlation top edges
    corr_adj_full = build_correlation_top_k(train_data, N * N)
    corr_upper = np.triu_indices(N, k=1)
    corr_vals = corr_adj_full[corr_upper]
    corr_sorted = np.argsort(corr_vals)[::-1]
    corr_top_edges = [(corr_upper[0][idx], corr_upper[1][idx]) for idx in corr_sorted]

    graphs = {}

    # 1. GSL only (existing)
    graphs["GSL"] = gsl_adj

    # 2. GSL+Phys: GSL edges + top physical edges to reach 16 total
    hybrid = gsl_adj.copy()
    added = 0
    target = 16 - n_gsl
    for r, c in phys_top_edges:
        if added >= target:
            break
        if (r, c) not in gsl_edges and (c, r) not in gsl_edges:
            hybrid[r, c] = 1.0
            hybrid[c, r] = 1.0
            added += 1
    graphs["GSL+Phys"] = hybrid

    # 3. GSL+Corr: GSL edges + top correlation edges to reach 16 total
    hybrid = gsl_adj.copy()
    added = 0
    for r, c in corr_top_edges:
        if added >= target:
            break
        if (r, c) not in gsl_edges and (c, r) not in gsl_edges:
            hybrid[r, c] = 1.0
            hybrid[c, r] = 1.0
            added += 1
    graphs["GSL+Corr"] = hybrid

    # 4. GSL+Phys+Corr: 8 GSL + 4 physical + 4 correlation = 16
    hybrid = gsl_adj.copy()
    added_p, added_c = 0, 0
    for r, c in phys_top_edges:
        if added_p >= 4:
            break
        if (r, c) not in gsl_edges and (c, r) not in gsl_edges:
            hybrid[r, c] = 1.0
            hybrid[c, r] = 1.0
            added_p += 1
    for r, c in corr_top_edges:
        if added_c >= 4:
            break
        if hybrid[r, c] == 0 and hybrid[c, r] == 0:
            hybrid[r, c] = 1.0
            hybrid[c, r] = 1.0
            added_c += 1
    graphs["GSL+PhysC"] = hybrid

    # 5. PhysSparseDir (existing): top-8 physical directed
    graphs["PhysSparseDir"] = build_phys_top_k_directed(adj_physical, 8)

    # 6. Corr-K8: top-8 correlation (downsampled)
    graphs["Corr-K8"] = build_correlation_top_k(train_data, 8)

    # 7. Corr-K16: top-16 correlation (existing)
    graphs["Corr-K16"] = build_correlation_top_k(train_data, 16)

    # 8. PhysSparse: top-16 physical symmetric (existing)
    graphs["PhysSparse"] = build_phys_top_k_symmetric(adj_physical, 16)

    return graphs


def train_and_evaluate(adj, model_name, dataset_name, train_X, train_Y,
                        test_X, test_Y, pre_len, seed=42, max_epochs=50,
                        device="cuda"):
    """Train model and evaluate. Returns metrics dict."""
    set_seed(seed)
    hidden_dim = 64
    seq_len = 12

    if model_name == "GCN":
        model = GCN(adj=adj, seq_len=seq_len, hidden_dim=hidden_dim)
        loss_name = "mse"
    elif model_name == "TGCN":
        model = TGCN(adj=adj, hidden_dim=hidden_dim)
        loss_name = "mse_with_regularizer"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    feat_raw, _ = load_data(dataset_name)
    feat_max_val = float(np.max(feat_raw))

    model_task = SupervisedForecastTask(
        model=model, loss=loss_name, pre_len=pre_len,
        learning_rate=0.001, weight_decay=0.0001,
        feat_max_val=feat_max_val,
    )

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    actual_device = "cuda" if use_cuda else "cpu"
    model = model.to(actual_device)
    if model_task.regressor is not None:
        model_task.regressor = model_task.regressor.to(actual_device)

    optimizer = model_task.configure_optimizer()
    train_X_t = torch.FloatTensor(train_X)
    train_Y_t = torch.FloatTensor(train_Y)
    test_X_t = torch.FloatTensor(test_X)
    test_Y_t = torch.FloatTensor(test_Y)

    train_dataset = torch.utils.data.TensorDataset(train_X_t, train_Y_t)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )

    start = time.time()
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(actual_device), yb.to(actual_device)
            optimizer.zero_grad()
            loss = model_task.training_step((xb, yb))
            loss.backward()
            optimizer.step()
    train_time = time.time() - start

    model.eval()
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_X_t, test_Y_t),
        batch_size=len(test_X_t), shuffle=False,
    )
    metrics = model_task.validation_epoch(test_loader, actual_device)
    metrics["train_time_s"] = round(train_time, 2)
    return metrics


def main():
    print("=" * 80)
    print("Stage 18-B/C: Hybrid Graph Forecasting Experiment")
    print("=" * 80)
    print(f"Dataset: {DATASET}")
    print(f"Seed: {SEED}, Max epochs: {MAX_EPOCHS}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data once
    feat_raw, adj_physical = load_data(DATASET)
    N = adj_physical.shape[0]
    n_phys_edges = int(np.sum(adj_physical > 0) / 2)
    print(f"Nodes: {N}, Physical edges: {n_phys_edges}")

    results = []

    for pre_len in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Prediction Horizon = {pre_len}")
        print(f"{'='*60}")

        # Generate sequences
        train_X, train_Y, test_X, test_Y = generate_sequences(
            feat_raw, seq_len=12, pre_len=pre_len,
            split_ratio=0.8, normalize=True,
        )

        # Build all hybrid graphs
        graphs = build_hybrid_graphs(DATASET, pre_len, threshold=0.3)

        for graph_name, adj in graphs.items():
            stats = graph_statistics(adj, graph_name)
            n_edges = stats["n_edges"]

            for model_name in ["GCN", "TGCN"]:
                print(f"\n  {graph_name:15s} | PH={pre_len} | {model_name:4s} | edges={n_edges:4d}", end="  ", flush=True)

                metrics = train_and_evaluate(
                    adj=adj, model_name=model_name, dataset_name=DATASET,
                    train_X=train_X, train_Y=train_Y,
                    test_X=test_X, test_Y=test_Y,
                    pre_len=pre_len, seed=SEED, max_epochs=MAX_EPOCHS,
                )

                row = {
                    "dataset": DATASET,
                    "model": model_name,
                    "pre_len": pre_len,
                    "graph_type": graph_name,
                    "seed": SEED,
                    "n_edges": n_edges,
                    "n_active": N - stats["n_isolated_nodes"],
                    "n_isolated": stats["n_isolated_nodes"],
                    "density": round(stats["density"], 8),
                    "rmse": round(metrics["RMSE"], 4),
                    "mae": round(metrics["MAE"], 4),
                    "r2": round(metrics["R2"], 6),
                    "train_time_s": metrics["train_time_s"],
                }
                results.append(row)
                print(f"RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} ({metrics['train_time_s']:.1f}s)")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "hybrid_forecasting_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {csv_path}")

    # Summary table
    print("\n" + "=" * 100)
    print("HYBRID GRAPH FORECASTING RESULTS — SZ-Taxi")
    print("=" * 100)

    for model in ["GCN", "TGCN"]:
        print(f"\n--- {model} ---")
        print(f"{'Graph':15s} | {'PH=1':>8s} | {'PH=2':>8s} | {'PH=3':>8s} | {'PH=4':>8s} | {'Edges':>6s} | {'Active':>6s}")
        print("-" * 80)

        graph_names = list(dict.fromkeys(r["graph_type"] for r in results if r["model"] == model))
        for gn in graph_names:
            row_data = [r for r in results if r["model"] == model and r["graph_type"] == gn]
            rmses = {r["pre_len"]: r["rmse"] for r in row_data}
            n_edges = row_data[0]["n_edges"]
            n_active = row_data[0]["n_active"]
            print(f"{gn:15s} | {rmses.get(1, 0):8.4f} | {rmses.get(2, 0):8.4f} | {rmses.get(3, 0):8.4f} | {rmses.get(4, 0):8.4f} | {n_edges:6d} | {n_active:6d}")

    # Save JSON summary
    summary = {
        "experiment": "Stage 18 Hybrid Graph Forecasting",
        "timestamp": datetime.now().isoformat(),
        "dataset": DATASET,
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "results": results,
    }
    json_path = os.path.join(RESULTS_DIR, "hybrid_forecasting_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {json_path}")


if __name__ == "__main__":
    main()
