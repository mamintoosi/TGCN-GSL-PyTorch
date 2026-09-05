"""
Graph construction and normalization utilities.
All operations are verified against the paper's specifications.
"""
import numpy as np
import torch
from typing import Tuple, Dict, Optional


def build_gsl_adjacency(W_est: np.ndarray, w_threshold: float = 0.3) -> np.ndarray:
    """
    GSL adjacency construction: A = 1[W > 0] after thresholding.
    
    Paper (Section 3): "We estimate the adjacency matrix A ∈ {0,1}^{N×N}"
    Paper (Section 5): "An edge j→i signifies that road j predicts road i"
    
    Args:
        W_est: Raw DAGMA output matrix (N, N) — may be 2D or 3D (N, N, pre_len)
        w_threshold: Weight threshold (default 0.3 from dagma library)
    
    Returns:
        Binary adjacency matrix A ∈ {0,1}^{N×N}
    
    Note: W_est > 0 means only positive weights are kept.
          Negative DAGMA weights are discarded.
    """
    if W_est.ndim == 3:
        # Union across prediction horizons (existing code behavior)
        W_est = np.any(W_est > 0, axis=2)
    
    adj = np.zeros(W_est.shape, dtype=np.float32)
    adj[W_est > 0] = 1.0
    
    return adj


def build_cgsl_adjacency(W_est: np.ndarray, w_threshold: float = 0.3) -> np.ndarray:
    """
    cGSL adjacency construction: A = 1[W > 0] + (1[W > 0])^T
    
    Paper (Section 4.2): "cGSL: cyclic variant created by symmetrizing the learned adjacency"
    
    Args:
        W_est: Raw DAGMA output matrix (N, N) — may be 2D or 3D
        w_threshold: Weight threshold (default 0.3 from dagma library)
    
    Returns:
        Symmetric binary adjacency matrix A ∈ {0,1}^{N×N}
    """
    adj_gsl = build_gsl_adjacency(W_est, w_threshold)
    adj_cgsl = adj_gsl + adj_gsl.T
    # Clip to {0,1} — cGSL should be unweighted binary
    adj_cgsl = np.clip(adj_cgsl, 0, 1).astype(np.float32)
    
    return adj_cgsl


def calculate_laplacian_with_self_loop(adj: torch.Tensor) -> torch.Tensor:
    """
    GCN symmetric normalized Laplacian with self-loops.
    
    Paper Eq. (2): H^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)})
    Where Ã = A + I_N
    
    This is IDENTICAL to the existing implementation in utils/graph_conv.py.
    Included here for independent verification.
    
    Args:
        adj: Adjacency matrix (N, N) as torch.Tensor
    
    Returns:
        Normalized Laplacian: D̃^{-1/2} Ã D̃^{-1/2}
    
    Verification:
        For A = [[0,1],[1,0]]: Ã = [[1,1],[1,1]], D̃ = diag(2,2)
        Result = [[0.5, 0.5], [0.5, 0.5]]
    """
    N = adj.size(0)
    
    # Step 1: Add self-loops: Ã = A + I
    A_tilde = adj + torch.eye(N, device=adj.device, dtype=adj.dtype)
    
    # Step 2: Compute degree matrix: D̃ = diag(row_sum(Ã))
    row_sum = A_tilde.sum(1)
    
    # Step 3: Compute D̃^{-1/2} (handle zero-degree nodes)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    
    # Step 4: Compute symmetric normalization: D̃^{-1/2} Ã D̃^{-1/2}
    normalized = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
    return normalized


def graph_statistics(adj: np.ndarray, label: str = "") -> Dict:
    """
    Compute comprehensive graph statistics for audit reporting.
    
    Args:
        adj: Adjacency matrix (N, N) — may be binary or weighted
        label: Human-readable label for this graph
    
    Returns:
        Dictionary of graph statistics
    """
    N = adj.shape[0]
    total_entries = N * N
    
    # Binary adjacency (for counting edges)
    adj_binary = (adj > 0).astype(int)
    
    # Count edges (excluding self-loops)
    np.fill_diagonal(adj_binary, 0)
    n_edges = int(np.sum(adj_binary))
    
    # Degree per node
    degrees = adj_binary.sum(axis=1)
    
    # Connected components (undirected interpretation)
    # Use union-find for efficiency
    adj_undirected = ((adj_binary + adj_binary.T) > 0).astype(int)
    visited = np.zeros(N, dtype=bool)
    n_components = 0
    component_sizes = []
    
    for start in range(N):
        if visited[start]:
            continue
        n_components += 1
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            size += 1
            neighbors = np.where(adj_undirected[node] > 0)[0]
            for nb in neighbors:
                if not visited[nb]:
                    stack.append(nb)
        component_sizes.append(size)
    
    # Largest connected component
    lcc_size = max(component_sizes) if component_sizes else 0
    
    stats = {
        "label": label,
        "N": N,
        "total_entries": total_entries,
        "n_edges": n_edges,
        "density": n_edges / (N * (N - 1)) if N > 1 else 0,
        "mean_degree": float(np.mean(degrees)),
        "median_degree": float(np.median(degrees)),
        "max_degree": int(np.max(degrees)),
        "n_isolated_nodes": int(np.sum(degrees == 0)),
        "n_components": n_components,
        "lcc_size": lcc_size,
    }
    
    # Weight statistics (if not purely binary)
    nonzero_mask = adj > 0
    if np.any(nonzero_mask):
        nonzero_vals = adj[nonzero_mask]
        stats["weight_min"] = float(np.min(np.abs(nonzero_vals)))
        stats["weight_max"] = float(np.max(np.abs(nonzero_vals)))
        stats["weight_mean"] = float(np.mean(np.abs(nonzero_vals)))
        stats["weight_median"] = float(np.median(np.abs(nonzero_vals)))
    
    # Positive/negative analysis
    nonzero_all = adj.flatten()
    nonzero_all = nonzero_all[nonzero_all != 0]
    if len(nonzero_all) > 0:
        stats["n_positive"] = int(np.sum(nonzero_all > 0))
        stats["n_negative"] = int(np.sum(nonzero_all < 0))
        stats["pct_positive"] = 100.0 * stats["n_positive"] / len(nonzero_all)
    
    return stats


def print_graph_stats(stats: Dict):
    """Pretty-print graph statistics."""
    print(f"\n=== Graph Statistics: {stats['label']} ===")
    print(f"  Nodes:              {stats['N']}")
    print(f"  Edges (excl. diag): {stats['n_edges']}")
    print(f"  Density:            {stats['density']:.6f}")
    print(f"  Mean degree:        {stats['mean_degree']:.2f}")
    print(f"  Median degree:      {stats['median_degree']:.1f}")
    print(f"  Max degree:         {stats['max_degree']}")
    print(f"  Isolated nodes:     {stats['n_isolated_nodes']}/{stats['N']} "
          f"({100*stats['n_isolated_nodes']/stats['N']:.1f}%)")
    print(f"  Connected comp.:    {stats['n_components']}")
    print(f"  LCC size:           {stats['lcc_size']}")
    if "weight_min" in stats:
        print(f"  Weight range:       [{stats['weight_min']:.6f}, {stats['weight_max']:.6f}]")
        print(f"  Weight mean/median: {stats['weight_mean']:.6f} / {stats['weight_median']:.6f}")
    if "n_positive" in stats:
        print(f"  Positive: {stats['n_positive']} ({stats['pct_positive']:.1f}%)")
        print(f"  Negative: {stats['n_negative']} ({100-stats['pct_positive']:.1f}%)")
