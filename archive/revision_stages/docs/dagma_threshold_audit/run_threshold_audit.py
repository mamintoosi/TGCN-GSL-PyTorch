#!/usr/bin/env python
"""
DAGMA Threshold Sensitivity Audit
==================================
Runs DAGMA with w_threshold=0 (no thresholding) on both datasets,
then analyzes the raw weighted matrices for threshold sensitivity.

This script does NOT modify any existing files.
It saves raw outputs to results/dagma_threshold_audit/
"""

import sys
import os
import numpy as np
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data.functions import load_features, generate_dataset
from dagma.linear import DagmaLinear

# ── Configuration (matching original) ──────────────────────────────────────
DATASETS = {
    "shenzhen": {
        "feat": "data/sz_speed.csv",
        "nodes": 156,
        "lambda1": 0.01,
    },
    "losloop": {
        "feat": "data/los_speed.csv",
        "nodes": 207,
        "lambda1": 0.02,
    },
}

SEQ_LEN = 1
PRE_LENS = [1, 2, 3, 4]
SPLIT_RATIO = 0.8

# ── Output directory ──────────────────────────────────────────────────────
OUTPUT_DIR = "results/dagma_threshold_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_dagma_input(feat_path, pre_len, split_ratio=0.8):
    """
    Reproduce the EXACT DAGMA input construction from the codebase.
    
    From spatiotemporal_csv_data.py:
      - load_features -> feat (T, N)
      - generate_torch_datasets -> TensorDataset
      - self.train_data = np.array([x[0].numpy() for x in train_dataset])
        => shape (num_samples, seq_len, num_nodes) = (num_samples, 1, N)
      - data = np.array([x[0] for x in self.train_data])
        => iterating over train_data (3D array), x[0] gets first seq_len dim
        => shape (num_samples, num_nodes) -- THIS IS THE DAGMA INPUT
      - X = data[i::pre_len]
        => every pre_len-th row starting from i
    """
    feat = load_features(feat_path)
    max_val = np.max(feat)
    feat_norm = feat / max_val
    
    train_size = int(feat_norm.shape[0] * split_ratio)
    train_data = feat_norm[:train_size]
    
    # Reproduce generate_dataset to get training samples
    train_X = []
    for i in range(len(train_data) - SEQ_LEN - pre_len):
        train_X.append(np.array(train_data[i : i + SEQ_LEN]))
    train_X = np.array(train_X)  # (num_samples, seq_len=1, N)
    
    # Reproduce the exact code: np.array([x[0] for x in train_data])
    # train_data here is train_X (the numpy array from TensorDataset)
    data = np.array([x[0] for x in train_X])  # (num_samples, N)
    
    return data, max_val


def run_dagma_threshold_audit(dataset_name, pre_len, w_threshold=0.0):
    """Run DAGMA with specified w_threshold and return the raw W matrix."""
    cfg = DATASETS[dataset_name]
    
    data, max_val = extract_dagma_input(cfg["feat"], pre_len)
    
    # Take the same subslice as the original code: X = data[i::pre_len]
    # For i=0 (first slice), X = data[0::pre_len]
    X = data[0::pre_len]
    
    print(f"  Dataset: {dataset_name}, PH={pre_len}")
    print(f"  DAGMA input X shape: {X.shape}")
    print(f"  lambda1: {cfg['lambda1']}, w_threshold: {w_threshold}")
    
    model = DagmaLinear(loss_type='l2')
    t0 = time.time()
    W_raw = model.fit(X, lambda1=cfg['lambda1'], w_threshold=w_threshold)
    elapsed = time.time() - t0
    
    print(f"  DAGMA completed in {elapsed:.1f}s")
    print(f"  Raw W: nonzero={np.count_nonzero(W_raw)}, shape={W_raw.shape}")
    
    return W_raw, X, elapsed


def analyze_matrix(W, label=""):
    """Compute comprehensive statistics for a DAGMA weight matrix."""
    stats = {}
    stats["shape"] = list(W.shape)
    stats["total_entries"] = int(W.size)
    stats["exact_zeros"] = int(np.sum(W == 0))
    stats["nonzero"] = int(np.count_nonzero(W))
    stats["density"] = float(stats["nonzero"] / stats["total_entries"])
    stats["positive"] = int(np.sum(W > 0))
    stats["negative"] = int(np.sum(W < 0))
    
    if stats["nonzero"] > 0:
        abs_nonzero = np.abs(W[W != 0])
        stats["min_abs_nonzero"] = float(np.min(abs_nonzero))
        stats["max_abs_nonzero"] = float(np.max(abs_nonzero))
        stats["mean_abs_nonzero"] = float(np.mean(abs_nonzero))
        stats["median_abs_nonzero"] = float(np.median(abs_nonzero))
        stats["std_abs_nonzero"] = float(np.std(abs_nonzero))
        
        # Quantiles
        for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            stats[f"q{q}_abs"] = float(np.percentile(abs_nonzero, q))
    
    # Threshold counts (using abs(W))
    for t in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        stats[f"abs_gte_{t}"] = int(np.sum(np.abs(W) >= t))
    
    # Positive-only threshold counts (matching repo's W > 0 pattern)
    for t in [0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.30]:
        stats[f"pos_gt_{t}"] = int(np.sum(W > t))
    
    return stats


def check_dag_validity(W, threshold=0.0):
    """Check if the thresholded directed graph is a DAG using topological sort."""
    try:
        import networkx as nx
        # Build directed graph from nonzero entries above threshold
        G = nx.DiGraph()
        n = W.shape[0]
        G.add_nodes_from(range(n))
        
        rows, cols = np.where(np.abs(W) > threshold)
        for r, c in zip(rows, cols):
            if r != c:  # skip self-loops
                G.add_edge(int(r), int(c))
        
        is_dag = nx.is_directed_acyclic_graph(G)
        num_edges = G.number_of_edges()
        
        try:
            topo_order = list(nx.topological_sort(G))
            has_topo = True
        except nx.NetworkXUnfeasible:
            has_topo = False
            topo_order = []
        
        return {
            "is_dag": is_dag,
            "num_edges": num_edges,
            "has_topological_order": has_topo,
            "num_self_loops": int(np.sum(np.diag(W) != 0)),
        }
    except ImportError:
        return {"error": "networkx not available"}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    all_results = {}
    
    for dataset_name in ["shenzhen", "losloop"]:
        all_results[dataset_name] = {}
        
        for pre_len in PRE_LENS:
            print(f"\n{'='*60}")
            print(f"Running DAGMA with w_threshold=0: {dataset_name} PH={pre_len}")
            print(f"{'='*60}")
            
            W_raw, X, elapsed = run_dagma_threshold_audit(
                dataset_name, pre_len, w_threshold=0.0
            )
            
            # Save raw matrix
            out_path = os.path.join(
                OUTPUT_DIR,
                f"W_raw_{dataset_name}_pre_len{pre_len}_thresh0.npy"
            )
            np.save(out_path, W_raw)
            print(f"  Saved raw W to: {out_path}")
            
            # Analyze
            stats_raw = analyze_matrix(W_raw, f"{dataset_name} PH={pre_len} raw")
            dag_raw = check_dag_validity(W_raw, threshold=0.0)
            
            # Also analyze at various thresholds
            threshold_analysis = {}
            for t in [0.0, 0.01, 0.05, 0.10, 0.20, 0.30]:
                abs_edges = int(np.sum(np.abs(W_raw) >= t))
                pos_edges = int(np.sum(W_raw > t))
                dag_info = check_dag_validity(W_raw, threshold=t)
                threshold_analysis[str(t)] = {
                    "abs_edges": abs_edges,
                    "pos_edges": pos_edges,
                    "is_dag": dag_info.get("is_dag", None),
                    "dag_edges": dag_info.get("num_edges", None),
                }
            
            all_results[dataset_name][str(pre_len)] = {
                "stats_raw": stats_raw,
                "dag_validity": dag_raw,
                "threshold_analysis": threshold_analysis,
                "elapsed_seconds": elapsed,
                "input_shape": list(X.shape),
            }
            
            # Print summary
            print(f"\n  ── Raw Matrix Summary ──")
            print(f"  Nonzero (raw): {stats_raw['nonzero']}")
            print(f"  Positive: {stats_raw['positive']}, Negative: {stats_raw['negative']}")
            print(f"  Density: {stats_raw['density']:.6f}")
            if stats_raw['nonzero'] > 0:
                print(f"  |W| range: [{stats_raw['min_abs_nonzero']:.6f}, {stats_raw['max_abs_nonzero']:.6f}]")
                print(f"  Mean |W|: {stats_raw['mean_abs_nonzero']:.6f}")
            print(f"\n  ── Threshold Sensitivity ──")
            print(f"  {'Threshold':>10} {'Abs edges':>10} {'Pos edges':>10} {'Is DAG':>8}")
            for t_str, ta in threshold_analysis.items():
                print(f"  {t_str:>10} {ta['abs_edges']:>10} {ta['pos_edges']:>10} {str(ta['is_dag']):>8}")
            print(f"\n  DAG validity (raw): {dag_raw}")
    
    # Save all results as JSON
    json_path = os.path.join(OUTPUT_DIR, "threshold_audit_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {json_path}")
    
    print("\n\nDONE. Threshold audit complete.")
