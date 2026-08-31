#!/usr/bin/env python
"""
Independent Graph Structure Learning module.

All parameters are explicitly specified. No undocumented library defaults.

Usage:
    from gsl_audit.independent_gsl import learn_graph, build_adjacency, audit_dagma
    
    W = learn_graph(X, lambda1=0.01, w_threshold=0.0, max_iter=180000)
    A = build_adjacency(W, mode='gsl')
"""

import numpy as np
from dagma.linear import DagmaLinear


def extract_dagma_input(train_X):
    """
    Extract the DAGMA input exactly as the existing code does.
    
    The existing code does:
        data = np.array([x[0] for x in self.train_data])
    
    Where self.train_data = np.array([x[0].numpy() for x in train_dataset])
    
    For TensorDataset(train_X, train_Y):
        train_dataset[i] = (train_X[i], train_Y[i])
        x = train_X[i]  # shape (seq_len, N)
        x[0] = first time step  # shape (N,)
    
    So data = np.array([train_X[i][0] for i in range(len(train_X))])
    Shape: (num_samples, N) — contemporaneous snapshots.
    
    Args:
        train_X: (num_samples, seq_len, N) array of training sequences
    
    Returns:
        data: (num_samples, N) — each row is a contemporaneous snapshot
    """
    # Verify shape
    assert train_X.ndim == 3, f"Expected 3D array, got {train_X.ndim}D"
    num_samples, seq_len, N = train_X.shape
    
    # Extract first time step from each sequence (EXACTLY as the code does)
    data = np.array([train_X[i][0] for i in range(num_samples)])
    
    assert data.shape == (num_samples, N), f"Expected ({num_samples}, {N}), got {data.shape}"
    
    print(f"  DAGMA input: {data.shape} (M samples, N nodes)")
    print(f"  Each row = one contemporaneous snapshot of all {N} sensors")
    
    return data


def learn_graph(X, lambda1=0.01, w_threshold=0.0, max_iter=180000, 
                loss_type='l2', verbose=True):
    """
    Learn a DAG structure using DAGMA.
    
    ALL parameters are explicitly specified — no library defaults.
    
    Args:
        X: (M, N) data matrix — M observations of N variables
        lambda1: L1 regularization strength (controls sparsity)
        w_threshold: threshold for zeroing out small weights
            - 0.0: preserve ALL learned weights (raw output)
            - 0.3: library default (used in original paper)
        max_iter: maximum DAGMA iterations
        loss_type: 'l2' for linear Gaussian model
        verbose: whether to print DAGMA progress
    
    Returns:
        W: (N, N) weighted adjacency matrix
    """
    assert X.ndim == 2, f"Expected 2D array, got {X.ndim}D"
    M, N = X.shape
    
    if verbose:
        print(f"\n  DAGMA configuration:")
        print(f"    Input shape: {X.shape}")
        print(f"    lambda1: {lambda1}")
        print(f"    w_threshold: {w_threshold}")
        print(f"    loss_type: {loss_type}")
        print(f"    max_iter: {max_iter}")
    
    model = DagmaLinear(loss_type=loss_type)
    
    # Pass w_threshold explicitly to override the default 0.3
    W = model.fit(X, lambda1=lambda1, w_threshold=w_threshold, max_iter=max_iter)
    
    if verbose:
        print(f"  DAGMA output:")
        print(f"    Shape: {W.shape}")
        print(f"    Nonzero: {np.count_nonzero(W)}")
        print(f"    Positive: {np.sum(W > 0)}")
        print(f"    Negative: {np.sum(W < 0)}")
        print(f"    Max |W|: {np.max(np.abs(W)):.6f}")
        print(f"    Mean |W nonzero: {np.mean(np.abs(W[W != 0])):.6f}" if np.any(W != 0) else "    No nonzero weights")
    
    return W


def build_adjacency(W, mode='gsl', threshold=0.0):
    """
    Convert DAGMA output W to binary adjacency matrix.
    
    Args:
        W: (N, N) weighted adjacency matrix
        mode: 
            'gsl' — directed, binary: A = (W > threshold)
            'cgsl' — symmetrized: A = (W > threshold) + (W > threshold).T
        threshold: minimum weight to keep as edge
    
    Returns:
        A: (N, N) binary adjacency matrix (int)
    """
    assert W.ndim == 2
    N = W.shape[0]
    
    if mode == 'gsl':
        A = (W > threshold).astype(int)
    elif mode == 'cgsl':
        A_raw = (W > threshold).astype(int)
        A = A_raw + A_raw.T
        A = (A > 0).astype(int)  # ensure binary
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    edges = int(np.sum(A > 0))
    density = edges / (N * (N - 1))
    
    print(f"  Adjacency ({mode}): {N} nodes, {edges} edges, density={density:.6f}")
    
    return A


def audit_dagma(W, label=""):
    """Comprehensive audit of a DAGMA output matrix."""
    N = W.shape[0]
    total = N * N
    
    print(f"\n  DAGMA audit [{label}]:")
    print(f"    Shape: {W.shape}, dtype: {W.dtype}")
    print(f"    Total entries: {total}")
    print(f"    Exact zeros: {np.sum(W == 0)} ({np.sum(W == 0)/total*100:.2f}%)")
    print(f"    Positive: {np.sum(W > 0)} ({np.sum(W > 0)/total*100:.4f}%)")
    print(f"    Negative: {np.sum(W < 0)} ({np.sum(W < 0)/total*100:.4f}%)")
    
    if np.any(W != 0):
        abs_w = np.abs(W[W != 0])
        print(f"    Nonzero |W| statistics:")
        print(f"      Min: {np.min(abs_w):.6f}")
        print(f"      Max: {np.max(abs_w):.6f}")
        print(f"      Mean: {np.mean(abs_w):.6f}")
        print(f"      Median: {np.median(abs_w):.6f}")
        
        # Threshold sensitivity
        thresholds = [0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.5]
        print(f"    Threshold sensitivity (|W| >= t):")
        for t in thresholds:
            count = np.sum(np.abs(W) >= t)
            edges = count - N  # exclude diagonal (self-loops)
            print(f"      |W| >= {t:<8}: {count:>6} entries ({count/total*100:>6.2f}%)")
    
    return W


def full_audit_pipeline(dataset_name, seq_len=12, pre_len=3, 
                        lambda1=None, w_threshold=0.0, max_iter=180000):
    """
    Complete audit pipeline: load data → extract DAGMA input → run DAGMA → audit.
    
    This is the INDEPENDENT reimplementation, not relying on existing W_est files.
    """
    from utils.data.functions import load_features, generate_dataset
    
    DATA_PATHS = {
        "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
        "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    }
    
    if lambda1 is None:
        lambda1 = 0.01 if dataset_name == "shenzhen" else 0.02
    
    print(f"\n{'='*70}")
    print(f"FULL AUDIT PIPELINE: {dataset_name}")
    print(f"  lambda1={lambda1}, w_threshold={w_threshold}, max_iter={max_iter}")
    print(f"{'='*70}")
    
    # Load and preprocess
    feat = load_features(DATA_PATHS[dataset_name]["feat"])
    N = feat.shape[1]
    max_val = np.max(feat)
    feat_norm = feat / max_val
    
    train_X, train_Y, test_X, test_Y = generate_dataset(
        feat_norm, seq_len=seq_len, pre_len=pre_len, split_ratio=0.8, normalize=False
    )
    
    print(f"\n  Data: {feat.shape} → train_X={train_X.shape}")
    
    # Extract DAGMA input
    data = extract_dagma_input(train_X)
    
    # Subsample for PH=1 (all data)
    X_ph1 = data[1::1]  # same as data[1:] essentially
    
    print(f"\n  Running DAGMA with w_threshold={w_threshold}...")
    W = learn_graph(X_ph1, lambda1=lambda1, w_threshold=w_threshold, max_iter=max_iter)
    
    # Audit
    audit_dagma(W, label=f"{dataset_name} PH=1 w_thresh={w_threshold}")
    
    return W


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Independent GSL audit")
    parser.add_argument("--dataset", choices=["shenzhen", "losloop"], required=True)
    parser.add_argument("--lambda1", type=float, default=None)
    parser.add_argument("--w-threshold", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=180000)
    args = parser.parse_args()
    
    W = full_audit_pipeline(
        args.dataset, 
        lambda1=args.lambda1, 
        w_threshold=args.w_threshold,
        max_iter=args.max_iter
    )
