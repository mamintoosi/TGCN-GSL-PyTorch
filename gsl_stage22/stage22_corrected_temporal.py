#!/usr/bin/env python3
"""
Stage 22 — Correct Temporal DAGMA Extraction and Re-evaluation

Loads the existing 312×312 raw DAGMA matrix from Stage 20.5 and extracts
the CORRECT temporal block W[0:N, N:2*N] (past → current) instead of
the WRONG block W[N:2*N, 0:N] (current → past) used in Stage 20.5.

This script:
1. Loads the saved raw W matrix
2. Extracts the correct temporal block
3. Analyzes the corrected temporal graph
4. Runs threshold sweep and Top-K experiments
5. Compares against baselines
6. Reports corrected results

No DAGMA re-computation is needed.
"""
import os
import sys
import json
import time
import csv
import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime
from typing import Dict, Tuple, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.gcn import GCN
from models.tgcn import TGCN
from tasks.supervised import SupervisedForecastTask

# Directories
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage22_corrected_temporal")
STAGE20_DIR = os.path.join(PROJECT_ROOT, "results", "stage20_5_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================================================================
# Data Loading (matching Stage 20 pipeline)
# ======================================================================

def load_and_normalize_train_only(
    dataset_name: str = "shenzhen",
    split_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load data with train-only normalization."""
    paths = {
        "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
        "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    }
    feat_df = pd.read_csv(os.path.join(PROJECT_ROOT, paths[dataset_name]["feat"]))
    feat = np.array(feat_df, dtype=np.float32)
    adj_df = pd.read_csv(os.path.join(PROJECT_ROOT, paths[dataset_name]["adj"]), header=None)
    adj_physical = np.array(adj_df, dtype=np.float32)
    T, N = feat.shape
    train_size = int(T * split_ratio)
    feat_max_val = float(np.max(feat[:train_size]))
    train_data = feat[:train_size] / feat_max_val
    test_data = feat[train_size:] / feat_max_val
    return train_data, test_data, adj_physical, feat_max_val


def generate_sequences(data: np.ndarray, seq_len: int, pre_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for i in range(len(data) - seq_len - pre_len):
        X.append(data[i: i + seq_len])
        Y.append(data[i + seq_len: i + seq_len + pre_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# Forecasting Evaluation
# ======================================================================

def train_and_evaluate(
    adj: np.ndarray,
    model_name: str,
    train_X: np.ndarray,
    train_Y: np.ndarray,
    test_X: np.ndarray,
    test_Y: np.ndarray,
    feat_max_val: float,
    pre_len: int = 1,
    seq_len: int = 12,
    hidden_dim: int = 64,
    seed: int = 42,
    max_epochs: int = 50,
    device: str = "cuda",
    batch_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
) -> Dict:
    set_seed(seed)
    if model_name == "GCN":
        model = GCN(adj=adj, seq_len=seq_len, hidden_dim=hidden_dim)
        loss_name = "mse"
    elif model_name == "TGCN":
        model = TGCN(adj=adj, hidden_dim=hidden_dim)
        loss_name = "mse_with_regularizer"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_task = SupervisedForecastTask(
        model=model, loss=loss_name, pre_len=pre_len,
        learning_rate=learning_rate, weight_decay=weight_decay,
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
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


# ======================================================================
# Graph Construction
# ======================================================================

def build_graph_from_weights(W_block: np.ndarray, threshold: float, N: int,
                             remove_diagonal: bool = True,
                             symmetrize: bool = False) -> np.ndarray:
    """Build binary adjacency from a weight block."""
    W_abs = np.abs(W_block.copy())
    adj = (W_abs > threshold).astype(np.float32)
    if remove_diagonal:
        np.fill_diagonal(adj, 0)
    if symmetrize:
        adj = np.maximum(adj, adj.T)
    return adj


def build_graph_topk(W_block: np.ndarray, K: int, N: int,
                     remove_diagonal: bool = True) -> np.ndarray:
    """Build graph using top-K edges by absolute weight."""
    W_abs = np.abs(W_block.copy())
    if remove_diagonal:
        np.fill_diagonal(W_abs, 0)
    
    # Get all off-diagonal entries
    indices = np.unravel_index(np.argsort(W_abs.ravel())[::-1], W_abs.shape)
    adj = np.zeros((N, N), dtype=np.float32)
    count = 0
    for idx in range(len(indices[0])):
        i, j = indices[0][idx], indices[1][idx]
        if W_abs[i, j] > 0 and count < K:
            adj[i, j] = 1.0
            count += 1
    return adj


def build_correlation_graph(train_data: np.ndarray, k: int) -> np.ndarray:
    """Top-K edges by absolute Pearson correlation."""
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


def graph_stats(adj: np.ndarray, name: str) -> dict:
    N = adj.shape[0]
    adj_b = (adj > 0).astype(int)
    np.fill_diagonal(adj_b, 0)
    n_entries = int(np.sum(adj_b))
    degrees = adj_b.sum(axis=1)
    n_active = int(np.sum(degrees > 0))
    return {
        "name": name, "n_entries": n_entries,
        "n_active_nodes": n_active, "n_isolated": N - n_active,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 80)
    print("STAGE 22 — CORRECTED TEMPORAL DAGMA EVALUATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    t_total = time.time()
    
    # ======================================================================
    # Step 1: Load saved raw W matrix
    # ======================================================================
    print("\n--- Step 1: Load saved raw W matrix ---")
    W_path = os.path.join(STAGE20_DIR, "sz_ph1_W_raw_temporal.npy")
    W_raw = np.load(W_path)
    N = 156
    D = 312
    print(f"  Loaded: {W_path}")
    print(f"  Shape: {W_raw.shape}")
    assert W_raw.shape == (D, D), f"Expected ({D},{D}), got {W_raw.shape}"
    
    # Load metadata
    with open(os.path.join(STAGE20_DIR, "phase_a_metadata.json")) as f:
        meta = json.load(f)
    print(f"  Metadata: N={meta['N']}, D={meta['D']}, seed={meta['seed']}")
    print(f"  DAGMA: lambda1={meta['lambda1']}, loss_type={meta['loss_type']}")
    
    # ======================================================================
    # Step 2: Extract CORRECT temporal block
    # ======================================================================
    print("\n--- Step 2: Extract correct temporal block ---")
    print(f"  CORRECT: W_cross = W[0:N, N:2N]  (past → current)")
    print(f"  WRONG:   W_cross = W[N:2N, 0:N]  (current → past) — Stage 20.5 used this!")
    
    W_cross_correct = W_raw[0:N, N:2*N]
    W_cross_wrong = W_raw[N:2*N, 0:N]
    
    print(f"\n  CORRECT block (W[0:N, N:2N]):")
    print(f"    Shape: {W_cross_correct.shape}")
    print(f"    Nonzero: {np.sum(np.abs(W_cross_correct) > 0)}")
    print(f"    |W| range: [{np.abs(W_cross_correct).min():.6f}, {np.abs(W_cross_correct).max():.6f}]")
    
    print(f"\n  WRONG block (W[N:2N, 0:N]) — Stage 20.5 result:")
    print(f"    Shape: {W_cross_wrong.shape}")
    print(f"    Nonzero: {np.sum(np.abs(W_cross_wrong) > 0)}")
    print(f"    |W| range: [{np.abs(W_cross_wrong).min():.6f}, {np.abs(W_cross_wrong).max():.6f}]")
    
    # Save correct block
    np.save(os.path.join(RESULTS_DIR, "W_cross_correct_raw.npy"), W_cross_correct)
    
    # ======================================================================
    # Step 3: Analyze corrected temporal graph
    # ======================================================================
    print("\n--- Step 3: Analyze corrected temporal graph ---")
    
    abs_correct = np.abs(W_cross_correct)
    nonzero_vals = abs_correct[abs_correct > 0]
    
    print(f"\n  Weight statistics:")
    print(f"    Entries: {N*N}")
    print(f"    Nonzero: {len(nonzero_vals)}")
    print(f"    Density: {len(nonzero_vals)/(N*N):.6f}")
    print(f"    Min |W|: {nonzero_vals.min():.8f}")
    print(f"    Max |W|: {nonzero_vals.max():.6f}")
    print(f"    Mean |W|: {nonzero_vals.mean():.6f}")
    print(f"    Median |W|: {np.median(nonzero_vals):.6f}")
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"    Quantiles of |W|:")
    for p in percentiles:
        print(f"      {p}th: {np.percentile(nonzero_vals, p):.6f}")
    
    # Top 20 edges
    print(f"\n  Top 20 temporal edges (W_cross_correct[i,j] = past_i → current_j):")
    print(f"  {'Rank':>4s}  {'past_i':>7s}  {'curr_j':>7s}  {'weight':>12s}  {'|weight|':>12s}  {'type':>10s}")
    flat_idx = np.argsort(abs_correct.ravel())[::-1]
    for rank, idx in enumerate(flat_idx[:20], 1):
        i, j = divmod(idx, N)
        w = W_cross_correct[i, j]
        edge_type = "self-loop" if i == j else "cross-sensor"
        print(f"  {rank:4d}  {i:7d}  {j:7d}  {w:12.6f}  {abs(w):12.6f}  {edge_type:>10s}")
    
    # Count self-loops vs cross-sensor
    n_selfloops = sum(1 for idx in flat_idx[:100] if divmod(idx, N)[0] == divmod(idx, N)[1])
    print(f"\n  In top 100 edges: {n_selfloops} self-loops, {100-n_selfloops} cross-sensor")
    
    # Self-loop analysis
    print(f"\n  Self-loop weights (diagonal of W_cross_correct):")
    diag = np.diag(W_cross_correct)
    abs_diag = np.abs(diag)
    print(f"    Max self-loop: {abs_diag.max():.6f} (sensor {np.argmax(abs_diag)})")
    print(f"    Mean self-loop: {abs_diag.mean():.6f}")
    n_selfloops_above = {thr: int(np.sum(abs_diag > thr)) for thr in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]}
    print(f"    Self-loops above threshold: {n_selfloops_above}")
    
    # ======================================================================
    # Step 4: Load data and generate sequences
    # ======================================================================
    print("\n--- Step 4: Load data and generate sequences ---")
    train_data, test_data, adj_phys, feat_max = load_and_normalize_train_only("shenzhen")
    seq_len = 12
    pre_len = 1
    train_X, train_Y = generate_sequences(train_data, seq_len, pre_len)
    test_X, test_Y = generate_sequences(test_data, seq_len, pre_len)
    print(f"  Train: {train_X.shape}, Test: {test_X.shape}")
    print(f"  feat_max: {feat_max:.6f}")
    
    # ======================================================================
    # Step 5: Threshold sweep on corrected block
    # ======================================================================
    print("\n--- Step 5: Threshold sweep on corrected temporal block ---")
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = []
    
    for thr in thresholds:
        adj = build_graph_from_weights(W_cross_correct, thr, N, remove_diagonal=True)
        stats = graph_stats(adj, f"TempDAGMA_correct_thr{thr}")
        
        for model_name in ["GCN", "TGCN"]:
            set_seed(42)
            metrics = train_and_evaluate(
                adj=adj, model_name=model_name,
                train_X=train_X, train_Y=train_Y,
                test_X=test_X, test_Y=test_Y,
                feat_max_val=feat_max, pre_len=pre_len,
                max_epochs=50, device="cuda",
            )
            row = {
                "method": f"TempDAGMA_correct_thr{thr}",
                "threshold": thr, "model": model_name,
                "n_edges": stats["n_entries"], "n_active": stats["n_active_nodes"],
                "rmse": round(metrics["RMSE"], 4), "mae": round(metrics["MAE"], 4),
                "r2": round(metrics["R2"], 6), "train_time_s": metrics["train_time_s"],
                "block": "correct_W0_N_2N", "seed": 42,
            }
            results.append(row)
            print(f"  thr={thr:.3f} | {model_name:4s} | edges={stats['n_entries']:5d} | RMSE={metrics['RMSE']:.4f}")
    
    # ======================================================================
    # Step 6: Top-K sweep on corrected block
    # ======================================================================
    print("\n--- Step 6: Top-K sweep on corrected temporal block ---")
    topk_values = [1, 2, 4, 8, 16, 32]
    
    for K in topk_values:
        adj = build_graph_topk(W_cross_correct, K, N, remove_diagonal=True)
        stats = graph_stats(adj, f"TempDAGMA_top{K}")
        
        for model_name in ["GCN", "TGCN"]:
            set_seed(42)
            metrics = train_and_evaluate(
                adj=adj, model_name=model_name,
                train_X=train_X, train_Y=train_Y,
                test_X=test_X, test_Y=test_Y,
                feat_max_val=feat_max, pre_len=pre_len,
                max_epochs=50, device="cuda",
            )
            row = {
                "method": f"TempDAGMA_top{K}",
                "threshold": None, "model": model_name,
                "n_edges": stats["n_entries"], "n_active": stats["n_active_nodes"],
                "rmse": round(metrics["RMSE"], 4), "mae": round(metrics["MAE"], 4),
                "r2": round(metrics["R2"], 6), "train_time_s": metrics["train_time_s"],
                "block": "correct_W0_N_2N", "seed": 42, "K": K,
            }
            results.append(row)
            print(f"  Top-{K:2d} | {model_name:4s} | edges={stats['n_entries']:5d} | RMSE={metrics['RMSE']:.4f}")
    
    # ======================================================================
    # Step 7: Baseline comparisons
    # ======================================================================
    print("\n--- Step 7: Baseline comparisons ---")
    
    # Physical
    adj_phys_clean = adj_phys.copy()
    np.fill_diagonal(adj_phys_clean, 0)
    stats_phys = graph_stats(adj_phys_clean, "Physical")
    for model_name in ["GCN", "TGCN"]:
        set_seed(42)
        metrics = train_and_evaluate(
            adj=adj_phys_clean, model_name=model_name,
            train_X=train_X, train_Y=train_Y,
            test_X=test_X, test_Y=test_Y,
            feat_max_val=feat_max, pre_len=pre_len,
            max_epochs=50, device="cuda",
        )
        row = {
            "method": "Physical", "threshold": None, "model": model_name,
            "n_edges": stats_phys["n_entries"], "n_active": stats_phys["n_active_nodes"],
            "rmse": round(metrics["RMSE"], 4), "mae": round(metrics["MAE"], 4),
            "r2": round(metrics["R2"], 6), "train_time_s": metrics["train_time_s"],
            "block": "physical", "seed": 42,
        }
        results.append(row)
        print(f"  Physical | {model_name:4s} | edges={stats_phys['n_entries']:5d} | RMSE={metrics['RMSE']:.4f}")
    
    # Correlation graphs
    for k in [8, 16]:
        adj_corr = build_correlation_graph(train_data, k)
        stats_corr = graph_stats(adj_corr, f"Corr-K{k}")
        for model_name in ["GCN", "TGCN"]:
            set_seed(42)
            metrics = train_and_evaluate(
                adj=adj_corr, model_name=model_name,
                train_X=train_X, train_Y=train_Y,
                test_X=test_X, test_Y=test_Y,
                feat_max_val=feat_max, pre_len=pre_len,
                max_epochs=50, device="cuda",
            )
            row = {
                "method": f"Corr-K{k}", "threshold": None, "model": model_name,
                "n_edges": stats_corr["n_entries"], "n_active": stats_corr["n_active_nodes"],
                "rmse": round(metrics["RMSE"], 4), "mae": round(metrics["MAE"], 4),
                "r2": round(metrics["R2"], 6), "train_time_s": metrics["train_time_s"],
                "block": "correlation", "seed": 42,
            }
            results.append(row)
            print(f"  Corr-K{k:2d} | {model_name:4s} | edges={stats_corr['n_entries']:5d} | RMSE={metrics['RMSE']:.4f}")
    
    # Original DAGMA (raw, no threshold)
    W_orig_raw = np.load(os.path.join(STAGE20_DIR, "sz_ph1_W_orig_contemp.npy"))
    adj_orig_raw = build_graph_from_weights(W_orig_raw, 0.0, N, remove_diagonal=True)
    stats_orig_raw = graph_stats(adj_orig_raw, "OriginalDAGMA_raw")
    for model_name in ["GCN", "TGCN"]:
        set_seed(42)
        metrics = train_and_evaluate(
            adj=adj_orig_raw, model_name=model_name,
            train_X=train_X, train_Y=train_Y,
            test_X=test_X, test_Y=test_Y,
            feat_max_val=feat_max, pre_len=pre_len,
            max_epochs=50, device="cuda",
        )
        row = {
            "method": "OriginalDAGMA_raw", "threshold": 0.0, "model": model_name,
            "n_edges": stats_orig_raw["n_entries"], "n_active": stats_orig_raw["n_active_nodes"],
            "rmse": round(metrics["RMSE"], 4), "mae": round(metrics["MAE"], 4),
            "r2": round(metrics["R2"], 6), "train_time_s": metrics["train_time_s"],
            "block": "original_contemp", "seed": 42,
        }
        results.append(row)
        print(f"  OrigDAGMA_raw | {model_name:4s} | edges={stats_orig_raw['n_entries']:5d} | RMSE={metrics['RMSE']:.4f}")
    
    # Original DAGMA (threshold 0.3)
    adj_orig_03 = build_graph_from_weights(W_orig_raw, 0.3, N, remove_diagonal=True)
    stats_orig_03 = graph_stats(adj_orig_03, "OriginalDAGMA_0.3")
    for model_name in ["GCN", "TGCN"]:
        set_seed(42)
        metrics = train_and_evaluate(
            adj=adj_orig_03, model_name=model_name,
            train_X=train_X, train_Y=train_Y,
            test_X=test_X, test_Y=test_Y,
            feat_max_val=feat_max, pre_len=pre_len,
            max_epochs=50, device="cuda",
        )
        row = {
            "method": "OriginalDAGMA_0.3", "threshold": 0.3, "model": model_name,
            "n_edges": stats_orig_03["n_entries"], "n_active": stats_orig_03["n_active_nodes"],
            "rmse": round(metrics["RMSE"], 4), "mae": round(metrics["MAE"], 4),
            "r2": round(metrics["R2"], 6), "train_time_s": metrics["train_time_s"],
            "block": "original_contemp", "seed": 42,
        }
        results.append(row)
        print(f"  OrigDAGMA_0.3 | {model_name:4s} | edges={stats_orig_03['n_entries']:5d} | RMSE={metrics['RMSE']:.4f}")
    
    # ======================================================================
    # Step 8: Summary table
    # ======================================================================
    print("\n" + "=" * 100)
    print("STAGE 22 RESULTS SUMMARY")
    print("=" * 100)
    
    for model in ["GCN", "TGCN"]:
        print(f"\n--- {model} ---")
        print(f"{'Method':35s} | {'Edges':>6s} | {'RMSE':>8s} | {'MAE':>8s} | {'Block':>15s}")
        print("-" * 90)
        for r in results:
            if r["model"] == model:
                block_label = r.get("block", "?")
                print(f"{r['method']:35s} | {r['n_edges']:6d} | {r['rmse']:8.4f} | {r['mae']:8.4f} | {block_label:>15s}")
    
    # ======================================================================
    # Step 9: Save results
    # ======================================================================
    print("\n--- Step 9: Save results ---")
    
    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "stage22_results.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")
    
    # Save summary JSON
    summary = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "dataset": "shenzhen",
        "N": N, "D": D,
        "seed": 42, "max_epochs": 50,
        "source_W_file": W_path,
        "correct_block": "W[0:N, N:2N]",
        "wrong_block_Was": "W[N:2N, 0:N]",
        "n_results": len(results),
    }
    with open(os.path.join(RESULTS_DIR, "stage22_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save weight stats
    weight_stats = {
        "nonzero": int(np.sum(np.abs(W_cross_correct) > 0)),
        "max_abs": float(np.abs(W_cross_correct).max()),
        "mean_abs_nonzero": float(nonzero_vals.mean()),
        "self_loop_max": float(abs_diag.max()),
        "self_loop_max_sensor": int(np.argmax(abs_diag)),
    }
    with open(os.path.join(RESULTS_DIR, "stage22_weight_stats.json"), "w") as f:
        json.dump(weight_stats, f, indent=2)
    
    total_time = time.time() - t_total
    
    # ======================================================================
    # Final Summary
    # ======================================================================
    print("\n" + "=" * 80)
    print("STAGE 22 COMPLETE")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 80)
    
    print("\n--- KEY FINDINGS ---")
    print(f"\nCorrect block (W[0:N, N:2N]):")
    print(f"  Nonzero entries: {np.sum(np.abs(W_cross_correct) > 0)}")
    print(f"  Max |weight|: {np.abs(W_cross_correct).max():.6f}")
    
    # Find best method for each model
    for model in ["GCN", "TGCN"]:
        model_results = [r for r in results if r["model"] == model]
        best = min(model_results, key=lambda x: x["rmse"])
        print(f"\n  Best {model}: {best['method']} (RMSE={best['rmse']:.4f}, edges={best['n_edges']})")
    
    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
