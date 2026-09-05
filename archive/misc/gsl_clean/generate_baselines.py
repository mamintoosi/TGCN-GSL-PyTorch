#!/usr/bin/env python3
"""
Generate baseline graph structures for controlled comparison.

This script creates two types of graphs that are essential for answering
the question: "Does DAGMA learn useful edges?"

1. CORRELATION GRAPH: Top-K edges by absolute Pearson correlation
   - Tests whether DAGMA's specific edges are better than simple correlation
   - Uses only TRAINING data (no data leakage)

2. PHYSICAL-SPARSE GRAPH: Top-K edges from the physical adjacency
   - Tests whether keeping the "best" physical edges matches GSL
   - Uses the existing physical graph weights

Both graphs have approximately the same edge count as the GSL graph,
enabling a fair density-matched comparison.

Usage:
    python gsl_clean/generate_baselines.py
    python gsl_clean/generate_baselines.py --dataset shenzhen
"""
import argparse
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


def load_training_data(dataset_name: str, seq_len: int = 12, split_ratio: float = 0.8) -> np.ndarray:
    """
    Load and preprocess training data (identical to main pipeline).
    
    Returns:
        train_data: (T_train, N) normalized training data
    """
    paths = {
        "shenzhen": {"feat": "data/sz_speed.csv"},
        "losloop": {"feat": "data/los_speed.csv"},
    }
    
    feat_df = pd.read_csv(paths[dataset_name]["feat"])
    feat = np.array(feat_df, dtype=np.float32)
    
    T = feat.shape[0]
    train_size = int(T * split_ratio)
    train_data = feat[:train_size]
    
    # Normalize (same as pipeline)
    max_val = np.max(feat)
    train_data = train_data / max_val
    
    return train_data


def compute_correlation_graph(train_data: np.ndarray, n_edges: int) -> np.ndarray:
    """
    Compute correlation-based graph: top-K edges by |Pearson correlation|.
    
    Uses only training data to avoid leakage.
    
    Args:
        train_data: (T_train, N) normalized training data
        n_edges: Number of edges to keep
        
    Returns:
        adj: (N, N) symmetric binary adjacency matrix
    """
    N = train_data.shape[1]
    
    # Compute Pearson correlation matrix
    corr_matrix = np.corrcoef(train_data.T)  # (N, N)
    
    # Replace NaN with 0 (constant nodes have undefined correlation)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Use absolute correlation
    abs_corr = np.abs(corr_matrix)
    
    # Zero out diagonal
    np.fill_diagonal(abs_corr, 0)
    
    # Get upper triangle indices sorted by correlation
    upper_tri_idx = np.triu_indices(N, k=1)
    corr_values = abs_corr[upper_tri_idx]
    
    # Sort by correlation (descending)
    sorted_idx = np.argsort(corr_values)[::-1]
    
    # Select top-K edges
    top_k = min(n_edges, len(sorted_idx))
    
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(top_k):
        row = upper_tri_idx[0][sorted_idx[i]]
        col = upper_tri_idx[1][sorted_idx[i]]
        adj[row, col] = 1.0
        adj[col, row] = 1.0  # Symmetric
    
    return adj


def compute_physical_sparse_graph(adj_physical: np.ndarray, n_edges: int, seed: int = 42) -> np.ndarray:
    """
    Compute physical-sparse graph: top-K edges from physical adjacency.
    
    The physical graph may have edge weights (from geographic distance).
    We keep the top-K edges by weight.
    
    Args:
        adj_physical: (N, N) physical adjacency matrix
        n_edges: Number of edges to keep
        seed: Random seed for tie-breaking
        
    Returns:
        adj: (N, N) symmetric binary adjacency matrix
    """
    N = adj_physical.shape[0]
    
    # Get edge indices (excluding diagonal)
    adj_binary = (adj_physical > 0).astype(float)
    np.fill_diagonal(adj_binary, 0)
    
    edge_indices = np.where(np.triu(adj_binary, k=1) > 0)
    edge_weights = adj_physical[edge_indices]
    
    if len(edge_weights) == 0:
        return np.zeros((N, N), dtype=np.float32)
    
    # Sort by weight (descending) - keep strongest connections
    sorted_idx = np.argsort(edge_weights)[::-1]
    
    # Select top-K edges
    top_k = min(n_edges, len(sorted_idx))
    
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(top_k):
        row = edge_indices[0][sorted_idx[i]]
        col = edge_indices[1][sorted_idx[i]]
        adj[row, col] = 1.0
        adj[col, row] = 1.0  # Symmetric
    
    return adj


def main():
    parser = argparse.ArgumentParser(description="Generate baseline graphs")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["shenzhen", "losloop"],
                        help="Dataset (default: both)")
    parser.add_argument("--seq_len", type=int, default=12,
                        help="Sequence length (default: 12)")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Train/test split ratio (default: 0.8)")
    args = parser.parse_args()
    
    datasets = [args.dataset] if args.dataset else ["shenzhen", "losloop"]
    
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Generating baselines for {ds}")
        print(f"{'='*60}")
        
        # Load training data and physical adjacency
        train_data = load_training_data(ds, args.seq_len, args.split_ratio)
        
        adj_physical_path = f"data/{'sz' if ds == 'shenzhen' else 'los'}_adj.csv"
        adj_physical = np.loadtxt(adj_physical_path, delimiter=',').astype(np.float32)
        
        N = adj_physical.shape[0]
        n_phys_edges = int(np.sum(adj_physical > 0) / 2)  # undirected
        print(f"  Nodes: {N}, Physical edges: {n_phys_edges}")
        
        # Get GSL edge counts from existing W_est files
        # IMPORTANT: GSL is directed (asymmetric), so the adjacency has
        # N_entries nonzero entries. Correlation/phys-sparse are symmetric,
        # so N_entries/2 unique edges. For fair comparison, we need to
        # match UNIQUE edges, not matrix entries.
        
        for ph in range(1, 5):
            w_est_path = f"data/W_est_{ds}_pre_len{ph}.npy"
            if not os.path.exists(w_est_path):
                print(f"  WARNING: {w_est_path} not found, skipping PH={ph}")
                continue
            
            W_est = np.load(w_est_path)
            if W_est.ndim == 3:
                gsl_entries = int(np.sum(np.any(W_est > 0, axis=2)))
            else:
                gsl_entries = int(np.sum(W_est > 0))
            
            # Unique edges in GSL (directed): each nonzero entry is one edge
            gsl_unique_edges = gsl_entries
            # Unique edges in cGSL (symmetrized): approximately double
            cgsl_unique_edges = gsl_entries * 2
            
            # For correlation/phys-sparse (symmetric), each unique edge = 2 entries
            # So we need: n_entries = 2 * target_unique_edges
            
            print(f"\n  PH={ph}: GSL={gsl_entries} entries ({gsl_unique_edges} unique directed)")
            
            # Generate correlation graph with same number of MATRIX ENTRIES as GSL
            # This means ~gsl_entries/2 unique undirected edges (symmetric)
            # vs GSL's gsl_entries unique directed edges (asymmetric)
            # Total connectivity seen by GCN is comparable
            n_target_entries = gsl_entries  # Match matrix entry count
            corr_graph = compute_correlation_graph(train_data, n_target_entries)
            corr_entries = int(np.sum(corr_graph > 0))
            corr_unique = corr_entries // 2
            corr_path = f"data/correlation_{ds}_pre_len{ph}.npy"
            np.save(corr_path, corr_graph)
            print(f"    Correlation: {corr_entries} entries ({corr_unique} unique undirected)")
            
            # Generate physical-sparse graph with same number of MATRIX ENTRIES
            phys_sparse = compute_physical_sparse_graph(adj_physical, n_target_entries)
            ps_entries = int(np.sum(phys_sparse > 0))
            ps_unique = ps_entries // 2
            ps_path = f"data/physical_sparse_{ds}_pre_len{ph}.npy"
            np.save(ps_path, phys_sparse)
            print(f"    PhysSparse:  {ps_entries} entries ({ps_unique} unique undirected)")
    
    print(f"\n{'='*60}")
    print("Done! All baseline graphs generated.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
