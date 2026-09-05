#!/usr/bin/env python
"""
Generate sparse random physical graphs for ablation study.

For each dataset and PH, creates a random directed graph with the same
number of edges as the GSL graph, sampled from the physical graph's
edge pool.

This isolates whether the forecasting improvement comes from:
  (a) graph sparsification (reduced oversmoothing), or
  (b) the specific learned topology.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

np.random.seed(42)  # Reproducible

# Load physical adjacency matrices
adj_sz = np.loadtxt("data/sz_adj.csv", delimiter=",").astype(np.float32)
adj_ll = np.loadtxt("data/los_adj.csv", delimiter=",").astype(np.float32)

print(f"SZ-Taxi physical: shape={adj_sz.shape}, edges={int(np.sum(adj_sz > 0))}")
print(f"Los-loop physical: shape={adj_ll.shape}, edges={int(np.sum(adj_ll > 0))}")

# Load GSL edge counts from stored W_est files
gsl_edges = {}
for dataset, adj, name in [("shenzhen", adj_sz, "SZ"), ("losloop", adj_ll, "Los")]:
    gsl_edges[dataset] = {}
    for ph in [1, 2, 3, 4]:
        W = np.load(f"data/W_est_{dataset}_pre_len{ph}.npy")
        if W.ndim == 3:
            # Count union of edges across all slices
            combined = np.any(W > 0, axis=2)
        else:
            combined = W > 0
        n_edges = int(np.sum(combined))
        gsl_edges[dataset][ph] = n_edges
        print(f"  {name} PH={ph}: GSL edges = {n_edges}")

# Generate sparse random graphs
# Strategy: from all possible directed edges in the physical graph,
# randomly select `gsl_edges` edges and set them to 1.
# This preserves the edge density ratio but randomizes topology.

N_SEEDS = 5  # Multiple random seeds for robustness

for dataset, adj, name in [("shenzhen", adj_sz, "SZ"), ("losloop", adj_ll, "Los")]:
    n_nodes = adj.shape[0]
    
    # Get all possible directed edges from physical graph
    phys_rows, phys_cols = np.where(adj > 0)
    phys_edge_pool = list(zip(phys_rows.tolist(), phys_cols.tolist()))
    print(f"\n{name}: {len(phys_edge_pool)} physical directed edges in pool")
    
    for ph in [1, 2, 3, 4]:
        target_edges = gsl_edges[dataset][ph]
        
        for seed_idx in range(N_SEEDS):
            rng = np.random.RandomState(42 + seed_idx)
            
            # Randomly sample target_edges from physical edge pool
            if target_edges > len(phys_edge_pool):
                # If GSL has more edges than physical (shouldn't happen), use all
                selected = phys_edge_pool.copy()
            else:
                selected_indices = rng.choice(
                    len(phys_edge_pool), size=target_edges, replace=False
                )
                selected = [phys_edge_pool[i] for i in selected_indices]
            
            # Build sparse adjacency matrix
            sparse_adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            for r, c in selected:
                sparse_adj[r, c] = 1.0
            
            # Save
            if seed_idx == 0:
                fname = f"data/sparse_random_{dataset}_pre_len{ph}.npy"
                np.save(fname, sparse_adj)
                print(f"  Saved {fname}: {int(np.sum(sparse_adj > 0))} edges (seed={42+seed_idx})")
            else:
                fname = f"data/sparse_random_{dataset}_pre_len{ph}_seed{42+seed_idx}.npy"
                np.save(fname, sparse_adj)
                print(f"  Saved {fname}: {int(np.sum(sparse_adj > 0))} edges (seed={42+seed_idx})")

print("\nDone. All sparse random graphs generated.")
