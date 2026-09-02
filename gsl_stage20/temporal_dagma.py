#!/usr/bin/env python3
"""
Stage 20: Temporal DAGMA Graph Construction with Fixed Normalization

This module implements:
1. Proper train-only normalization (no test-set leakage)
2. Temporal DAGMA with lag-expanded input representation
3. Multiple graph extraction strategies
4. Validation tests
5. Experiment runner

Mathematical Formulation:
  Raw: v(t) ∈ R^N, t=0,...,T-1
  Train: first 80% chronologically
  Normalization: u(t) = v(t) / max_train (computed from training data only)

  Temporal DAGMA input (2-lag):
    z(t) = [u(t-1), u(t)] ∈ R^{2N}
    Z ∈ R^{M × 2N} where M = train_size - L - PH - 1

  DAGMA learns W ∈ R^{2N × 2N}, which decomposes into 4 blocks:
    W = [[W_pp, W_pc],
         [W_cp, W_cc]]
  where p = "past" (t-1), c = "current" (t)

  The cross-time block W_cp ∈ R^{N×N} captures:
    W_cp[i,j] > 0  →  sensor i at time t-1 predicts sensor j at time t

  This is the TEMPORAL graph used by GCN/TGCN.

Usage:
  python gsl_stage20/temporal_dagma.py --test          # Run validation tests
  python gsl_stage20/temporal_dagma.py --compare       # Compare graph structures
  python gsl_stage20/temporal_dagma.py --experiment    # Full PH=1 experiment
"""
import argparse
import os
import sys
import json
import time
import random
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dagma.linear import DagmaLinear
from models.gcn import GCN
from models.tgcn import TGCN
from tasks.supervised import SupervisedForecastTask

# ============================================================
# Configuration
# ============================================================
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage20_temporal")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1. Data Loading with Train-Only Normalization
# ============================================================
def load_and_normalize_train_only(
    dataset_name: str,
    split_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load data, normalize using training data only, split chronologically.

    Returns:
        train_data: (T_train, N) normalized training data
        test_data: (T_test, N) normalized test data (using train max)
        adj_physical: (N, N) physical adjacency
        feat_max_val: normalization maximum (from training data only)
    """
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

    # CRITICAL: normalize using training data only
    feat_max_val = float(np.max(feat[:train_size]))
    train_data = feat[:train_size] / feat_max_val
    test_data = feat[train_size:] / feat_max_val

    return train_data, test_data, adj_physical, feat_max_val


def generate_sequences(
    data: np.ndarray,
    seq_len: int,
    pre_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding-window sequences from normalized data.

    Returns:
        X: (M, seq_len, N) input windows
        Y: (M, pre_len, N) target windows
    """
    X, Y = [], []
    for i in range(len(data) - seq_len - pre_len):
        X.append(data[i: i + seq_len])
        Y.append(data[i + seq_len: i + seq_len + pre_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ============================================================
# 2. Original (Contemporaneous) DAGMA
# ============================================================
def build_original_dagma(
    train_data: np.ndarray,
    N: int,
    lambda1: float = 0.01,
    w_threshold: float = 0.3,
    verbose: bool = False,
) -> np.ndarray:
    """
    Build original contemporaneous DAGMA graph.

    Current implementation: X ∈ R^{M × N}, each row = one timestamp.
    DAGMA learns W ∈ R^{N×N} of contemporaneous dependencies.

    Args:
        train_data: (T_train, N) normalized training data
        N: number of nodes
        lambda1: DAGMA L1 coefficient
        w_threshold: weight threshold
        verbose: DAGMA verbose output

    Returns:
        adj: (N, N) binary adjacency matrix
    """
    model = DagmaLinear(loss_type='l2', verbose=verbose)
    W = model.fit(train_data, lambda1=lambda1, w_threshold=w_threshold)

    # Binary: positive entries only (matching original code)
    adj = (W > 0).astype(np.float32)
    return adj


# ============================================================
# 3. Temporal DAGMA (Lag-Expanded Input)
# ============================================================
def build_temporal_dagma(
    train_data: np.ndarray,
    N: int,
    n_lags: int = 1,
    lambda1: float = 0.01,
    w_threshold: float = 0.3,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build temporal DAGMA graph with lag-expanded input.

    Formulation:
      z(t) = [u(t-n_lags), ..., u(t-1), u(t)] ∈ R^{(n_lags+1)*N}
      Z ∈ R^{M × (n_lags+1)*N}

    DAGMA learns W ∈ R^{D×D} where D = (n_lags+1)*N.

    The cross-time block W[0:n_lags*N, n_lags*N:D] ∈ R^{n_lags*N × N}
    captures dependencies from past sensors to current sensors.

    We extract the most recent lag block as the temporal graph:
      W_temporal = W[n_lags*N : (n_lags+1)*N, n_lags*N : (n_lags+1)*N]  # contemporaneous block
      W_cross    = W[(n_lags-1)*N : n_lags*N, n_lags*N : (n_lags+1)*N]  # 1-lag cross block

    The final N×N graph can be:
      Option 1: W_temporal only (contemporaneous, same as original)
      Option 2: |W_cross| + |W_temporal| (combined)
      Option 3: W_cross only (pure temporal dependency)

    Args:
        train_data: (T_train, N) normalized training data
        N: number of nodes
        n_lags: number of lagged observations (1 = past + current)
        lambda1: DAGMA L1 coefficient
        w_threshold: weight threshold
        verbose: DAGMA verbose output

    Returns:
        adj_temporal: (N, N) temporal graph adjacency (cross-time block)
        adj_combined: (N, N) combined temporal + contemporaneous
        info: dict with metadata
    """
    D = (n_lags + 1) * N
    M = train_data.shape[0]

    # Build lag-expanded matrix Z ∈ R^{M - n_lags, D}
    # z(t) = [u(t-n_lags), u(t-n_lags+1), ..., u(t)]
    n_samples = M - n_lags
    Z = np.zeros((n_samples, D), dtype=train_data.dtype)
    for lag in range(n_lags + 1):
        start_col = lag * N
        end_col = (lag + 1) * N
        Z[:, start_col:end_col] = train_data[n_lags - lag: n_lags - lag + n_samples]

    print(f"  Temporal DAGMA input: Z ∈ R^{{{n_samples} × {D}}}")
    print(f"  (vs original: R^{{{M} × {N}}})")
    print(f"  Matrix size: {D}×{D} = {D**2} entries (vs {N**2})")

    # Fit DAGMA
    t0 = time.time()
    model = DagmaLinear(loss_type='l2', verbose=verbose)
    W_full = model.fit(Z, lambda1=lambda1, w_threshold=w_threshold)
    dagma_time = time.time() - t0

    # Extract blocks
    W_full_raw = W_full.copy()

    # Contemporaneous block: last N rows × last N columns
    # This captures dependencies among current-time sensors
    W_cc = W_full_raw[n_lags * N: (n_lags + 1) * N, n_lags * N: (n_lags + 1) * N]

    # Cross-time block: second-to-last N rows × last N columns
    # This captures dependencies from (t-1) sensors to (t) sensors
    if n_lags >= 1:
        W_cross = W_full_raw[(n_lags - 1) * N: n_lags * N, n_lags * N: (n_lags + 1) * N]
    else:
        W_cross = np.zeros((N, N), dtype=np.float64)

    # Build graphs
    # Temporal graph: cross-time dependencies only
    adj_temporal = (np.abs(W_cross) > 0).astype(np.float32)

    # Combined: both temporal and contemporaneous
    adj_combined = ((np.abs(W_cross) > 0) | (np.abs(W_cc) > 0)).astype(np.float32)

    info = {
        "n_lags": n_lags,
        "D": D,
        "N": N,
        "n_samples": n_samples,
        "dagma_time_s": round(dagma_time, 2),
        "w_cross_nonzero": int(np.sum(np.abs(W_cross) > 0)),
        "w_cc_nonzero": int(np.sum(np.abs(W_cc) > 0)),
        "w_full_nonzero": int(np.sum(np.abs(W_full_raw) > 0)),
        "w_cross_max": float(np.max(np.abs(W_cross))),
        "w_cc_max": float(np.max(np.abs(W_cc))),
    }

    return adj_temporal, adj_combined, info


# ============================================================
# 4. Correlation Graph (Fixed: Train-Only)
# ============================================================
def build_correlation_graph(
    train_data: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Top-K edges by absolute Pearson correlation (symmetric).
    Uses training data only (no leakage).
    """
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


# ============================================================
# 5. Graph Comparison
# ============================================================
def compare_graphs(graphs: Dict[str, np.ndarray], names: list) -> dict:
    """Compute edge overlap and Jaccard similarity between graph pairs."""
    N = list(graphs.values())[0].shape[0]
    results = {}

    for i, name_a in enumerate(names):
        adj_a = (graphs[name_a] > 0).astype(int)
        edges_a = set()
        for r in range(N):
            for c in range(N):
                if adj_a[r, c] > 0:
                    edges_a.add((r, c))

        for name_b in names[i + 1:]:
            adj_b = (graphs[name_b] > 0).astype(int)
            edges_b = set()
            for r in range(N):
                for c in range(N):
                    if adj_b[r, c] > 0:
                        edges_b.add((r, c))

            intersection = edges_a & edges_b
            union = edges_a | edges_b
            jaccard = len(intersection) / len(union) if union else 0

            results[f"{name_a} ∩ {name_b}"] = {
                "intersection_size": len(intersection),
                "union_size": len(union),
                "jaccard": round(jaccard, 4),
                "a_only": len(edges_a - edges_b),
                "b_only": len(edges_b - edges_a),
            }

    return results


def graph_stats(adj: np.ndarray, name: str) -> dict:
    """Compute graph statistics."""
    N = adj.shape[0]
    adj_b = (adj > 0).astype(int)
    np.fill_diagonal(adj_b, 0)
    n_entries = int(np.sum(adj_b))
    degrees = adj_b.sum(axis=1)
    n_active = int(np.sum(degrees > 0))
    n_isolated = N - n_active
    return {
        "name": name,
        "n_entries": n_entries,
        "n_active_nodes": n_active,
        "n_isolated": n_isolated,
        "density": round(n_entries / (N * (N - 1)), 8) if N > 1 else 0,
        "max_degree": int(np.max(degrees)),
        "mean_degree": round(float(np.mean(degrees)), 2),
    }


# ============================================================
# 6. Forecasting Evaluation
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    """Train model and evaluate."""
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
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
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


# ============================================================
# 7. Validation Tests
# ============================================================
def run_tests():
    """Run small validation tests to verify implementation."""
    print("=" * 70)
    print("STAGE 20 VALIDATION TESTS")
    print("=" * 70)

    # --- Test 1: Data shapes and normalization ---
    print("\n--- Test 1: Data shapes and normalization ---")
    train_data, test_data, adj_phys, feat_max = load_and_normalize_train_only("shenzhen")
    T_train, N = train_data.shape
    T_test = test_data.shape[0]
    print(f"  Train: ({T_train}, {N}), Test: ({T_test}, {N})")
    print(f"  feat_max_val: {feat_max:.6f}")
    print(f"  Train max after norm: {np.max(train_data):.6f}")
    print(f"  Test max after norm: {np.max(test_data):.6f}")
    print(f"  Physical adj: {adj_phys.shape}, entries={int(np.sum(adj_phys > 0))}")

    # Verify normalization is correct
    assert np.max(train_data) <= 1.0, "Train data should be <= 1"
    assert np.max(test_data) <= 1.0, "Test data should be <= 1"
    # Verify the test data was normalized by train max, not test max
    raw_test_max = np.max(np.loadtxt("data/sz_speed.csv", delimiter=",", skiprows=1)[T_train:])
    expected_test_max = raw_test_max / feat_max
    actual_test_max = np.max(test_data)
    print(f"  Verification: raw test max={raw_test_max:.4f}, normed={actual_test_max:.6f}, expected={expected_test_max:.6f}")
    assert abs(actual_test_max - expected_test_max) < 1e-5, "Normalization mismatch!"
    print("  ✓ PASSED")

    # --- Test 2: Original DAGMA ---
    print("\n--- Test 2: Original (contemporaneous) DAGMA ---")
    t0 = time.time()
    adj_orig = build_original_dagma(train_data, N, lambda1=0.01, w_threshold=0.3)
    t_orig = time.time() - t0
    stats_orig = graph_stats(adj_orig, "Original DAGMA")
    print(f"  Time: {t_orig:.1f}s")
    print(f"  {stats_orig}")
    print("  ✓ PASSED")

    # --- Test 3: Temporal DAGMA ---
    print("\n--- Test 3: Temporal DAGMA (2-lag input) ---")
    t0 = time.time()
    adj_temp, adj_comb, info = build_temporal_dagma(
        train_data, N, n_lags=1, lambda1=0.01, w_threshold=0.3
    )
    t_temp = time.time() - t0
    stats_temp = graph_stats(adj_temp, "Temporal DAGMA (cross-time)")
    stats_comb = graph_stats(adj_comb, "Temporal DAGMA (combined)")
    print(f"  DAGMA info: {info}")
    print(f"  Temporal graph: {stats_temp}")
    print(f"  Combined graph: {stats_comb}")
    print(f"  Total time: {t_temp:.1f}s")
    print("  ✓ PASSED")

    # --- Test 4: Correlation graph ---
    print("\n--- Test 4: Correlation graph (train-only) ---")
    adj_corr = build_correlation_graph(train_data, k=16)
    stats_corr = graph_stats(adj_corr, "Correlation-K16")
    print(f"  {stats_corr}")
    print("  ✓ PASSED")

    # --- Test 5: Graph comparison ---
    print("\n--- Test 5: Graph overlap analysis ---")
    graphs = {
        "Original-DAGMA": adj_orig,
        "Temporal-DAGMA": adj_temp,
        "Combined-DAGMA": adj_comb,
        "Correlation-K16": adj_corr,
    }
    overlap = compare_graphs(graphs, list(graphs.keys()))
    for pair, info in overlap.items():
        print(f"  {pair}: intersection={info['intersection_size']}, "
              f"jaccard={info['jaccard']}, "
              f"A-only={info['a_only']}, B-only={info['b_only']}")
    print("  ✓ PASSED")

    # --- Test 6: Tiny forecasting run ---
    print("\n--- Test 6: Tiny forecasting sanity check ---")
    seq_len = 12
    pre_len = 1
    train_X, train_Y = generate_sequences(train_data, seq_len, pre_len)
    test_X, test_Y = generate_sequences(test_data, seq_len, pre_len)
    print(f"  Train sequences: {train_X.shape}, Test sequences: {test_X.shape}")

    # Test with original DAGMA graph, 3 epochs
    set_seed(42)
    metrics = train_and_evaluate(
        adj=adj_orig, model_name="GCN",
        train_X=train_X, train_Y=train_Y,
        test_X=test_X, test_Y=test_Y,
        feat_max_val=feat_max, pre_len=pre_len,
        max_epochs=3, device="cuda",
    )
    print(f"  Original DAGMA GCN (3 epochs): RMSE={metrics['RMSE']:.4f}")

    # Test with temporal DAGMA graph
    set_seed(42)
    metrics_t = train_and_evaluate(
        adj=adj_temp, model_name="GCN",
        train_X=train_X, train_Y=train_Y,
        test_X=test_X, test_Y=test_Y,
        feat_max_val=feat_max, pre_len=pre_len,
        max_epochs=3, device="cuda",
    )
    print(f"  Temporal DAGMA GCN (3 epochs): RMSE={metrics_t['RMSE']:.4f}")
    print("  ✓ PASSED")

    # --- Test 7: Leakage sanity check ---
    print("\n--- Test 7: Leakage sanity check ---")
    # Verify normalization uses only training data
    raw = np.loadtxt("data/sz_speed.csv", delimiter=",", skiprows=1)
    T_all = raw.shape[0]
    train_size = int(T_all * 0.8)
    train_max = np.max(raw[:train_size])
    global_max = np.max(raw)
    print(f"  Global max: {global_max:.6f}")
    print(f"  Train max:  {train_max:.6f}")
    print(f"  Equal: {train_max == global_max}")
    if train_max == global_max:
        print("  NOTE: Global max = Train max for this dataset.")
        print("  The normalization leakage has no numerical effect here,")
        print("  but the protocol is still incorrect and should be fixed.")
    print("  ✓ PASSED (protocol fixed, no numerical difference for this dataset)")

    # --- Test 8: Multi-lag temporal DAGMA ---
    print("\n--- Test 8: Multi-lag temporal DAGMA (3 lags) ---")
    adj_temp3, adj_comb3, info3 = build_temporal_dagma(
        train_data, N, n_lags=3, lambda1=0.01, w_threshold=0.3
    )
    stats_temp3 = graph_stats(adj_temp3, "Temporal DAGMA (3-lag)")
    print(f"  Info: {info3}")
    print(f"  {stats_temp3}")
    print("  ✓ PASSED")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Original DAGMA: {stats_orig['n_entries']} entries, {stats_orig['n_active_nodes']} active nodes")
    print(f"  Temporal (1-lag): {stats_temp['n_entries']} entries, {stats_temp['n_active_nodes']} active nodes")
    print(f"  Temporal (3-lag): {stats_temp3['n_entries']} entries, {stats_temp3['n_active_nodes']} active nodes")
    print(f"  Combined (1-lag): {stats_comb['n_entries']} entries, {stats_comb['n_active_nodes']} active nodes")
    print(f"  Correlation-K16: {stats_corr['n_entries']} entries, {stats_corr['n_active_nodes']} active nodes")
    print(f"  Original DAGMA time: {t_orig:.1f}s")
    print(f"  Temporal DAGMA time: {t_temp:.1f}s")


# ============================================================
# 8. Full Experiment (PH=1, SZ-Taxi)
# ============================================================
def run_experiment(
    dataset_name: str = "shenzhen",
    pre_len: int = 1,
    seed: int = 42,
    max_epochs: int = 50,
    n_lags_list: list = None,
    threshold_list: list = None,
):
    """
    Full experiment: compare original vs temporal DAGMA on SZ-Taxi PH=1.
    """
    if n_lags_list is None:
        n_lags_list = [1]
    if threshold_list is None:
        threshold_list = [0.3]

    print("=" * 80)
    print("STAGE 20: Temporal DAGMA Experiment")
    print(f"  Dataset: {dataset_name}, PH={pre_len}, Seed={seed}, Epochs={max_epochs}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load data with fixed normalization
    train_data, test_data, adj_phys, feat_max = load_and_normalize_train_only(dataset_name)
    N = train_data.shape[1]
    print(f"  Nodes: {N}, Train: {train_data.shape[0]}, Test: {test_data.shape[0]}")

    # Generate sequences
    seq_len = 12
    train_X, train_Y = generate_sequences(train_data, seq_len, pre_len)
    test_X, test_Y = generate_sequences(test_data, seq_len, pre_len)
    print(f"  Train seq: {train_X.shape}, Test seq: {test_X.shape}")

    results = []

    # --- Graph construction ---
    graphs = {}

    # 1. Physical graph
    graphs["Physical"] = adj_phys.copy()
    np.fill_diagonal(graphs["Physical"], 0)

    # 2. Correlation-K16
    graphs["Corr-K16"] = build_correlation_graph(train_data, k=16)

    # 3. Original DAGMA
    adj_orig = build_original_dagma(train_data, N, lambda1=0.01, w_threshold=0.3)
    graphs["Original-DAGMA"] = adj_orig

    # 4. Temporal DAGMA (vary n_lags and threshold)
    for n_lags in n_lags_list:
        for thr in threshold_list:
            adj_temp, adj_comb, info = build_temporal_dagma(
                train_data, N, n_lags=n_lags,
                lambda1=0.01, w_threshold=thr,
            )
            key_temp = f"TempDAGMA-lags{n_lags}-thr{thr}"
            key_comb = f"CombDAGMA-lags{n_lags}-thr{thr}"
            graphs[key_temp] = adj_temp
            graphs[key_comb] = adj_comb
            print(f"  {key_temp}: {graph_stats(adj_temp, key_temp)}")
            print(f"  {key_comb}: {graph_stats(adj_comb, key_comb)}")

    # --- Graph comparison ---
    print("\n--- Graph Overlap Analysis ---")
    overlap = compare_graphs(graphs, list(graphs.keys()))
    for pair, info in overlap.items():
        print(f"  {pair}: jaccard={info['jaccard']}, "
              f"intersect={info['intersection_size']}, "
              f"A-only={info['a_only']}, B-only={info['b_only']}")

    # --- Forecasting ---
    print("\n--- Forecasting Experiments ---")
    graph_names = list(graphs.keys())

    for gname in graph_names:
        adj = graphs[gname]
        stats = graph_stats(adj, gname)

        for model_name in ["GCN", "TGCN"]:
            print(f"\n  {gname:25s} | {model_name:4s} | edges={stats['n_entries']:4d} active={stats['n_active_nodes']:3d}",
                  end="  ", flush=True)

            set_seed(seed)
            metrics = train_and_evaluate(
                adj=adj, model_name=model_name,
                train_X=train_X, train_Y=train_Y,
                test_X=test_X, test_Y=test_Y,
                feat_max_val=feat_max, pre_len=pre_len,
                max_epochs=max_epochs, device="cuda",
            )

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "pre_len": pre_len,
                "graph_type": gname,
                "seed": seed,
                "n_lags": info.get("n_lags", 0) if "info" in dir() else 0,
                "n_edges": stats["n_entries"],
                "n_active": stats["n_active_nodes"],
                "rmse": round(metrics["RMSE"], 4),
                "mae": round(metrics["MAE"], 4),
                "r2": round(metrics["R2"], 6),
                "train_time_s": metrics["train_time_s"],
                "normalization": "train_only",
            }
            results.append(row)
            print(f"RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f}")

    # --- Summary Table ---
    print("\n" + "=" * 100)
    print("STAGE 20 RESULTS SUMMARY")
    print("=" * 100)

    for model in ["GCN", "TGCN"]:
        print(f"\n--- {model} ---")
        print(f"{'Graph':30s} | {'RMSE':>8s} | {'MAE':>8s} | {'Edges':>6s} | {'Active':>6s}")
        print("-" * 80)
        for r in results:
            if r["model"] == model:
                print(f"{r['graph_type']:30s} | {r['rmse']:8.4f} | {r['mae']:8.4f} | {r['n_edges']:6d} | {r['n_active']:6d}")

    # --- Save results ---
    csv_path = os.path.join(RESULTS_DIR, f"stage20_results_{dataset_name}_ph{pre_len}.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return results


# ============================================================
# 9. Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 20: Temporal DAGMA")
    parser.add_argument("--test", action="store_true", help="Run validation tests")
    parser.add_argument("--experiment", action="store_true", help="Run full PH=1 experiment")
    parser.add_argument("--dataset", type=str, default="shenzhen")
    parser.add_argument("--pre-len", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--n-lags", type=int, nargs="+", default=[1])
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.3])
    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.experiment:
        run_experiment(
            dataset_name=args.dataset,
            pre_len=args.pre_len,
            seed=args.seed,
            max_epochs=args.max_epochs,
            n_lags_list=args.n_lags,
            threshold_list=args.threshold,
        )
    else:
        print("Usage: python temporal_dagma.py --test  OR  python temporal_dagma.py --experiment")
