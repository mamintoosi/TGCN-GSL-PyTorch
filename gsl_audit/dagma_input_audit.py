#!/usr/bin/env python
"""
FORENSIC AUDIT: What exactly is the DAGMA input?

This script traces the complete data pipeline from raw CSV to DAGMA input,
logging every shape, type, and statistical property at each step.

The goal is to determine whether DAGMA receives:
  (A) Contemporaneous snapshots: X[i] = [x_1(t), x_2(t), ..., x_N(t)]
  (B) Lagged temporal windows:   X[i] = [x(t-k), ..., x(t-1), x(t)]
  (C) Something else

This is the single most critical question for interpreting the paper's claims.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data.functions import load_features, load_adjacency_matrix, generate_dataset


def audit_dataset(dataset_name, seq_len=12, pre_len=3):
    """Trace the complete data pipeline and audit DAGMA input."""
    
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*70}")
    
    DATA_PATHS = {
        "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
        "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    }
    
    # Step 1: Load raw data
    feat = load_features(DATA_PATHS[dataset_name]["feat"])
    adj = load_adjacency_matrix(DATA_PATHS[dataset_name]["adj"])
    
    print(f"\n--- Step 1: Raw data ---")
    print(f"  Features shape: {feat.shape}  (T timesteps, N sensors)")
    print(f"  Adjacency shape: {adj.shape}  (N nodes, N nodes)")
    print(f"  Feature range: [{feat.min():.4f}, {feat.max():.4f}]")
    print(f"  Feature mean: {feat.mean():.4f}, std: {feat.std():.4f}")
    print(f"  Adjacency nonzero: {int(np.sum(adj > 0))}")
    print(f"  Adjacency dtype: {adj.dtype}")
    
    N = feat.shape[1]
    T = feat.shape[0]
    
    # Step 2: Normalize
    max_val = np.max(feat)
    feat_norm = feat / max_val
    
    print(f"\n--- Step 2: After normalization ---")
    print(f"  Max val: {max_val:.4f}")
    print(f"  Normalized range: [{feat_norm.min():.6f}, {feat_norm.max():.6f}]")
    print(f"  Normalized mean: {feat_norm.mean():.6f}, std: {feat_norm.std():.6f}")
    
    # Step 3: Generate sequences
    train_X, train_Y, test_X, test_Y = generate_dataset(
        feat_norm, seq_len=seq_len, pre_len=pre_len, split_ratio=0.8, normalize=False
    )
    
    print(f"\n--- Step 3: Generated sequences ---")
    print(f"  train_X shape: {train_X.shape}  (samples, seq_len, N)")
    print(f"  train_Y shape: {train_Y.shape}  (samples, pre_len, N)")
    print(f"  test_X shape:  {test_X.shape}  (samples, seq_len, N)")
    print(f"  test_Y shape:  {test_Y.shape}  (samples, pre_len, N)")
    print(f"  train_X[0] first 5 values: {train_X[0, 0, :5]}")
    print(f"  train_X[0] last 5 values:  {train_X[0, -1, :5]}")
    
    # Step 4: What the code ACTUALLY extracts for DAGMA
    # From spatiotemporal_csv_data.py:
    #   self.train_data = np.array([x[0].numpy() for x in train_dataset])
    #   data = np.array([x[0] for x in self.train_data])
    #   X = data[i::pre_len]
    
    # Simulate what x[0] means for a TensorDataset
    # train_dataset[i] returns (train_X[i], train_Y[i])
    # x[0] extracts train_X[i][0] = first time step of sequence i
    
    print(f"\n--- Step 4: DAGMA input construction ---")
    print(f"  In the code: self.train_data = np.array([x[0] for x in train_dataset])")
    print(f"  This extracts the FIRST TIME STEP from each training sequence.")
    print(f"")
    print(f"  For TensorDataset(train_X, train_Y):")
    print(f"    train_dataset[i] = (train_X[i], train_Y[i])")
    print(f"    x = train_X[i] has shape {train_X.shape[1:]}  (seq_len={seq_len}, N={N})")
    print(f"    x[0] = first time step, shape ({N},)")
    print(f"")
    
    # What does x[0] actually produce?
    # For a TensorDataset with train_X shape (M, seq_len, N):
    # iterating gives tuples of (tensor(seq_len, N), tensor(pre_len, N))
    # x[0] takes the first row: shape (N,)
    
    # BUT: x[0] on a tuple gives the first tensor (the input, not the target)
    # So for (x, y) in train_dataset:
    #   x has shape (seq_len, N) = (12, N)
    #   x[0] has shape (N,) — first time step
    
    # Therefore data = np.array([x[0] for x in train_dataset]) has shape (M, N)
    
    data_for_dagma = np.array([train_X[i][0] for i in range(len(train_X))])
    
    print(f"  data_for_dagma shape: {data_for_dagma.shape}  (M samples, N nodes)")
    print(f"  Each ROW is one contemporaneous snapshot of all {N} sensors")
    print(f"  Each COLUMN is the time series of one sensor (subsampled)")
    print(f"")
    print(f"  DAGMA receives X of shape ({data_for_dagma.shape[0]}, {data_for_dagma.shape[1]})")
    print(f"  This is M={data_for_dagma.shape[0]} i.i.d. observations of {N}-dim random vector")
    print(f"  Each observation = speeds of all {N} sensors at ONE time step")
    
    # Step 5: What DAGMA actually does with this
    print(f"\n--- Step 5: DAGMA interpretation ---")
    print(f"  DAGMA learns W such that X ≈ X @ W (linear SEM)")
    print(f"  W[i,j] = effect of node j on node i (given all other nodes)")
    print(f"  This captures CONTEMPORANEOUS dependencies, not temporal ones")
    print(f"  The 'DAG' property means: among {N} sensors, the dependency structure")
    print(f"  has no cycles — this is a statistical structure, not a temporal one")
    
    # Step 6: Check the subsampling
    print(f"\n--- Step 6: Subsampling for each PH ---")
    for ph in [1, 2, 3, 4]:
        X_ph = data_for_dagma[ph::ph]
        print(f"  PH={ph}: X shape = {X_ph.shape} (every {ph}-th sample)")
        print(f"    Row 0 = sample {ph}, Row 1 = sample {ph+ph}, ...")
        print(f"    This is STILL contemporaneous — just fewer samples")
    
    # Step 7: Correlation analysis
    print(f"\n--- Step 7: Cross-sectional correlation structure ---")
    corr = np.corrcoef(data_for_dagma.T)  # (N, N) correlation matrix
    print(f"  Correlation matrix shape: {corr.shape}")
    print(f"  Diagonal (self-correlation): all {corr[0,0]:.4f}")
    print(f"  Off-diagonal mean: {np.mean(corr[np.triu_indices(N, k=1)]):.4f}")
    print(f"  Off-diagonal std:  {np.std(corr[np.triu_indices(N, k=1)]):.4f}")
    print(f"  Max off-diagonal:  {np.max(corr[np.triu_indices(N, k=1)]):.4f}")
    print(f"  Min off-diagonal:  {np.min(corr[np.triu_indices(N, k=1)]):.4f}")
    
    # Strongest correlations
    np.fill_diagonal(corr, 0)
    top_k = 10
    flat_idx = np.argsort(np.abs(corr).ravel())[-top_k:]
    rows, cols = np.unravel_index(flat_idx, corr.shape)
    print(f"\n  Top {top_k} strongest correlations:")
    for r, c, idx in zip(rows, cols, flat_idx):
        print(f"    node {r} <-> node {c}: r = {corr[r, c]:.4f}")
    
    return data_for_dagma, adj, feat_norm


def main():
    print("=" * 70)
    print("FORENSIC AUDIT: DAGMA Input Analysis")
    print("=" * 70)
    print()
    print("CRITICAL QUESTION: Does DAGMA receive contemporaneous or temporal data?")
    print()
    
    for dataset in ["shenzhen", "losloop"]:
        data, adj, feat = audit_dataset(dataset, seq_len=12, pre_len=3)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The DAGMA input X has shape (M, N) where:")
    print("  M = number of training samples (subsampled)")
    print("  N = number of sensors")
    print()
    print("Each ROW of X is a CONTEMPORANEOUS snapshot:")
    print("  X[i] = [speed_sensor_1(t_i), speed_sensor_2(t_i), ..., speed_sensor_N(t_i)]")
    print()
    print("This is CROSS-SECTIONAL data, not temporal/lagged data.")
    print("DAGMA learns contemporaneous dependencies between sensors.")
    print()
    print("The paper's Section 5 claim that 'edge j→i means j at time t")
    print("predicts i at time t+1' is NOT supported by this input construction.")
    print("The learned graph captures which sensors have concurrent statistical")
    print("dependencies, not which sensors influence each other over time.")


if __name__ == "__main__":
    main()
