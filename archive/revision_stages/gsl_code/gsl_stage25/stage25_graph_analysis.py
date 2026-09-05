#!/usr/bin/env python3
"""
Stage 25 — Experiment Families A, B, H, I: Structural Analysis of Existing DAGMA Graphs.

Analyzes PH-specific temporal DAGMA graphs for:
  - Edge count vs threshold
  - Top-K overlap
  - Jaccard similarity
  - Weight correlation
  - Persistent vs PH-specific edges
  - Seed stability

Uses ONLY existing DAGMA matrices. No new DAGMA computation.

Usage:
  python gsl_stage25/stage25_graph_analysis.py
  python gsl_stage25/stage25_graph_analysis.py --dataset losloop
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage25_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_CONFIGS = {
    "shenzhen": {"N": 156, "prefix": "sz"},
    "losloop": {"N": 207, "prefix": "los"},
}


def load_W(dataset, ph, seed=42):
    """Load the raw temporal DAGMA matrix and extract the correct cross block."""
    config = DATASET_CONFIGS[dataset]
    prefix = config["prefix"]
    N = config["N"]
    
    path = os.path.join(PROJECT_ROOT, "results", "stage24_validation",
                        f"{prefix}_ph{ph}_seed{seed}_W_raw_temporal.npy")
    if not os.path.exists(path):
        return None
    W_raw = np.load(path)
    W_cross = W_raw[:N, N:2*N]  # CORRECT: past -> current
    return W_cross


def binary_graph(W, threshold):
    """Convert weighted W to binary adjacency at given threshold."""
    adj = (np.abs(W) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)
    return adj


def top_k_graph(W, k):
    """Keep only top-K edges by absolute weight."""
    N = W.shape[0]
    W_abs = np.abs(W.copy())
    np.fill_diagonal(W_abs, 0)
    adj = np.zeros_like(W_abs)
    flat = W_abs.flatten()
    top_idx = np.argsort(flat)[::-1][:k]
    adj.flat[top_idx] = 1.0
    return adj


def edge_set(adj):
    """Return set of directed edges as frozenset of (i,j) tuples."""
    return frozenset(zip(*np.where(adj > 0)))


def jaccard(s1, s2):
    """Jaccard similarity between two sets."""
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def weight_correlation(W1, W2):
    """Spearman-like correlation of absolute weights (using Pearson on abs values)."""
    abs1 = np.abs(W1).flatten()
    abs2 = np.abs(W2).flatten()
    # Only consider entries where at least one is nonzero
    mask = (abs1 > 0) | (abs2 > 0)
    if mask.sum() < 3:
        return 0.0
    return float(np.corrcoef(abs1[mask], abs2[mask])[0, 1])


def analyze_ph_graphs(dataset, seed=42):
    """Family A + B + H: Analyze how graphs change across PH."""
    config = DATASET_CONFIGS[dataset]
    N = config["N"]
    
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    top_k_values = [1, 4, 8, 16, 32, 64, 128]
    
    # Load all PH matrices
    W_ph = {}
    for ph in [1, 2, 3, 4]:
        W = load_W(dataset, ph, seed)
        if W is not None:
            W_ph[ph] = W
    
    if len(W_ph) < 2:
        print(f"  WARNING: Only {len(W_ph)} PH matrices available. Need at least 2.")
        return {}
    
    results = {
        "dataset": dataset,
        "seed": seed,
        "N": N,
        "n_ph_available": len(W_ph),
        "available_phs": sorted(W_ph.keys()),
    }
    
    # === Family A: Edge count vs threshold ===
    edge_counts = {}
    for thr in thresholds:
        edge_counts[thr] = {}
        for ph, W in W_ph.items():
            adj = binary_graph(W, thr)
            edge_counts[thr][ph] = int(np.sum(adj))
    results["edge_counts_vs_threshold"] = edge_counts
    
    # === Family A: Top-K edge counts ===
    top_k_edge_counts = {}
    for k in top_k_values:
        top_k_edge_counts[k] = {}
        for ph, W in W_ph.items():
            adj = top_k_graph(W, k)
            top_k_edge_counts[k][ph] = int(np.sum(adj))
    results["top_k_edge_counts"] = top_k_edge_counts
    
    # === Family B + H: Cross-PH similarity at each threshold ===
    cross_ph_similarity = {}
    for thr in thresholds:
        binary_ph = {ph: binary_graph(W, thr) for ph, W in W_ph.items()}
        edges_ph = {ph: edge_set(adj) for ph, adj in binary_ph.items()}
        
        pairs = {}
        for ph1 in sorted(W_ph.keys()):
            for ph2 in sorted(W_ph.keys()):
                if ph1 >= ph2:
                    continue
                key = f"PH{ph1}_vs_PH{ph2}"
                pairs[key] = {
                    "jaccard": round(jaccard(edges_ph[ph1], edges_ph[ph2]), 4),
                    "intersection": len(edges_ph[ph1] & edges_ph[ph2]),
                    "union": len(edges_ph[ph1] | edges_ph[ph2]),
                    "only_in_ph1": len(edges_ph[ph1] - edges_ph[ph2]),
                    "only_in_ph2": len(edges_ph[ph2] - edges_ph[ph1]),
                }
        cross_ph_similarity[thr] = pairs
    results["cross_ph_similarity"] = cross_ph_similarity
    
    # === Family B: Persistent vs PH-specific edges (at threshold 0.1) ===
    thr_ref = 0.1
    binary_ref = {ph: binary_graph(W, thr_ref) for ph, W in W_ph.items()}
    edges_ref = {ph: edge_set(adj) for ph, adj in binary_ref.items()}
    
    all_edges = set()
    for e in edges_ref.values():
        all_edges |= e
    
    persistent_all = frozenset.intersection(*edges_ref.values()) if edges_ref else set()
    persistent_any = set().union(*edges_ref.values())
    
    edge_persistence = {}
    for e in sorted(all_edges):
        phs_present = [ph for ph, es in edges_ref.items() if e in es]
        edge_persistence[str(e)] = {
            "n_phs": len(phs_present),
            "phs": phs_present,
        }
    
    results["persistence_analysis"] = {
        "threshold": thr_ref,
        "total_unique_edges": len(all_edges),
        "persistent_all_phs": len(persistent_all),
        "persistent_any_phs": len(persistent_any),
        "fraction_persistent_all": round(len(persistent_all) / max(len(all_edges), 1), 4),
        "edge_details": edge_persistence,
    }
    
    # === Family B: Weight-level analysis ===
    abs_weights = {ph: np.abs(W) for ph, W in W_ph.items()}
    np.fill_diagonal(abs_weights[sorted(W_ph.keys())[0]], 0)  # zero diag for first PH
    
    weight_corr = {}
    for ph1 in sorted(W_ph.keys()):
        for ph2 in sorted(W_ph.keys()):
            if ph1 >= ph2:
                continue
            key = f"PH{ph1}_vs_PH{ph2}"
            weight_corr[key] = round(weight_correlation(W_ph[ph1], W_ph[ph2]), 4)
    results["weight_correlation"] = weight_corr
    
    # === Family A: Degree distribution at each PH (threshold 0.1) ===
    degree_dists = {}
    for ph, W in W_ph.items():
        adj = binary_graph(W, 0.1)
        out_deg = adj.sum(axis=1)
        in_deg = adj.sum(axis=0)
        degree_dists[ph] = {
            "out_degree_mean": round(float(out_deg.mean()), 2),
            "out_degree_max": int(out_deg.max()),
            "in_degree_mean": round(float(in_deg.mean()), 2),
            "in_degree_max": int(in_deg.max()),
            "n_isolated_out": int(np.sum(out_deg == 0)),
            "n_isolated_in": int(np.sum(in_deg == 0)),
        }
    results["degree_distributions"] = degree_dists
    
    # === Family A: Top edges at each PH ===
    top_edges = {}
    for ph, W in W_ph.items():
        W_abs = np.abs(W.copy())
        np.fill_diagonal(W_abs, 0)
        # Get top 10
        flat_idx = np.argsort(W_abs.flatten())[::-1][:10]
        edges = []
        for idx in flat_idx:
            i, j = divmod(idx, W.shape[0])
            edges.append({
                "source": int(i), "target": int(j),
                "weight": round(float(W[i, j]), 6),
                "abs_weight": round(float(W_abs[i, j]), 6),
            })
        top_edges[ph] = edges
    results["top_edges"] = top_edges
    
    return results


def analyze_seed_stability(dataset, seeds=None):
    """Family I: Seed stability for PH=1."""
    config = DATASET_CONFIGS[dataset]
    N = config["N"]
    
    if seeds is None:
        seeds = [42, 43, 44, 45, 46] if dataset == "shenzhen" else [42]
    
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    top_k_values = [1, 4, 8, 16, 32]
    
    # Load all seed matrices for PH=1
    W_seeds = {}
    for seed in seeds:
        W = load_W(dataset, 1, seed)
        if W is not None:
            W_seeds[seed] = W
    
    if len(W_seeds) < 2:
        print(f"  WARNING: Only {len(W_seeds)} seed matrices available.")
        return {}
    
    results = {
        "dataset": dataset,
        "ph": 1,
        "n_seeds": len(W_seeds),
        "seeds": sorted(W_seeds.keys()),
    }
    
    # Edge count variance across seeds
    edge_counts = {}
    for thr in thresholds:
        edge_counts[thr] = {}
        for seed, W in W_seeds.items():
            adj = binary_graph(W, thr)
            edge_counts[thr][seed] = int(np.sum(adj))
        
        counts = list(edge_counts[thr].values())
        edge_counts[thr]["mean"] = round(np.mean(counts), 1)
        edge_counts[thr]["std"] = round(np.std(counts), 1)
    results["edge_counts_vs_threshold"] = edge_counts
    
    # Cross-seed Jaccard at each threshold
    cross_seed_jaccard = {}
    for thr in thresholds:
        edges_seeds = {seed: edge_set(binary_graph(W, thr)) for seed, W in W_seeds.items()}
        pair_jaccards = []
        for s1 in sorted(W_seeds.keys()):
            for s2 in sorted(W_seeds.keys()):
                if s1 >= s2:
                    continue
                j = jaccard(edges_seeds[s1], edges_seeds[s2])
                pair_jaccards.append(j)
        cross_seed_jaccard[thr] = {
            "mean_jaccard": round(np.mean(pair_jaccards), 4) if pair_jaccards else None,
            "min_jaccard": round(np.min(pair_jaccards), 4) if pair_jaccards else None,
            "max_jaccard": round(np.max(pair_jaccards), 4) if pair_jaccards else None,
        }
    results["cross_seed_jaccard"] = cross_seed_jaccard
    
    # Cross-seed top-K overlap
    cross_seed_topk = {}
    for k in top_k_values:
        edges_seeds = {seed: edge_set(top_k_graph(W, k)) for seed, W in W_seeds.items()}
        pair_overlaps = []
        for s1 in sorted(W_seeds.keys()):
            for s2 in sorted(W_seeds.keys()):
                if s1 >= s2:
                    continue
                overlap = len(edges_seeds[s1] & edges_seeds[s2]) / k
                pair_overlaps.append(overlap)
        cross_seed_topk[k] = {
            "mean_overlap": round(np.mean(pair_overlaps), 4) if pair_overlaps else None,
        }
    results["cross_seed_topk_overlap"] = cross_seed_topk
    
    # Cross-seed weight correlation
    pair_corrs = []
    for s1 in sorted(W_seeds.keys()):
        for s2 in sorted(W_seeds.keys()):
            if s1 >= s2:
                continue
            c = weight_correlation(W_seeds[s1], W_seeds[s2])
            pair_corrs.append(c)
    results["cross_seed_weight_correlation"] = {
        "mean": round(np.mean(pair_corrs), 4) if pair_corrs else None,
        "min": round(np.min(pair_corrs), 4) if pair_corrs else None,
    }
    
    # Persistent edges across all seeds (at threshold 0.1)
    edges_per_seed = {seed: edge_set(binary_graph(W, 0.1)) for seed, W in W_seeds.items()}
    persistent_all = set.intersection(*[set(x) for x in edges_per_seed.values()])
    all_edges = set().union(*edges_per_seed.values())
    results["persistence_across_seeds"] = {
        "threshold": 0.1,
        "total_unique_edges": len(all_edges),
        "persistent_all_seeds": len(persistent_all),
        "fraction_persistent": round(len(persistent_all) / max(len(all_edges), 1), 4),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 25: Graph structural analysis")
    parser.add_argument("--dataset", type=str, default="shenzhen",
                        choices=["shenzhen", "losloop"], help="Dataset")
    args = parser.parse_args()
    
    dataset = args.dataset
    
    print("=" * 80)
    print("STAGE 25 — GRAPH STRUCTURAL ANALYSIS")
    print(f"Dataset: {dataset}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # === Family A + B + H: PH graph analysis ===
    print("\n--- Family A+B+H: PH-specific graph analysis ---")
    ph_results = analyze_ph_graphs(dataset)
    
    # Print key findings
    if "edge_counts_vs_threshold" in ph_results:
        print("\nEdge counts by threshold and PH:")
        print(f"  {'Threshold':>10s}", end="")
        for ph in ph_results["available_phs"]:
            print(f"  {'PH'+str(ph):>8s}", end="")
        print()
        for thr, counts in ph_results["edge_counts_vs_threshold"].items():
            print(f"  {thr:>10.3f}", end="")
            for ph in ph_results["available_phs"]:
                print(f"  {counts.get(ph, 'N/A'):>8}", end="")
            print()
    
    if "cross_ph_similarity" in ph_results:
        print("\nCross-PH Jaccard similarity:")
        for thr, pairs in ph_results["cross_ph_similarity"].items():
            for pair_key, stats in pairs.items():
                print(f"  thr={thr:.3f} {pair_key}: Jaccard={stats['jaccard']:.4f} "
                      f"(shared={stats['intersection']}, only_1={stats['only_in_ph1']}, only_2={stats['only_in_ph2']})")
    
    if "persistence_analysis" in ph_results:
        pa = ph_results["persistence_analysis"]
        print(f"\nPersistence (threshold={pa['threshold']}):")
        print(f"  Total unique edges: {pa['total_unique_edges']}")
        print(f"  Persistent across ALL PHs: {pa['persistent_all_phs']}")
        print(f"  Fraction persistent: {pa['fraction_persistent_all']:.1%}")
    
    if "weight_correlation" in ph_results:
        print("\nWeight correlation across PHs:")
        for pair, corr in ph_results["weight_correlation"].items():
            print(f"  {pair}: {corr:.4f}")
    
    # === Family I: Seed stability ===
    print("\n--- Family I: Seed stability (PH=1) ---")
    seed_results = analyze_seed_stability(dataset)
    
    if "edge_counts_vs_threshold" in seed_results:
        print("\nEdge count mean±std across seeds:")
        for thr, counts in seed_results["edge_counts_vs_threshold"].items():
            if "mean" in counts:
                print(f"  thr={thr:.3f}: {counts['mean']:.1f} ± {counts['std']:.1f}")
    
    if "cross_seed_jaccard" in seed_results:
        print("\nCross-seed Jaccard (mean across pairs):")
        for thr, stats in seed_results["cross_seed_jaccard"].items():
            if stats["mean_jaccard"] is not None:
                print(f"  thr={thr:.3f}: {stats['mean_jaccard']:.4f}")
    
    if "cross_seed_topk_overlap" in seed_results:
        print("\nCross-seed top-K overlap (mean):")
        for k, stats in seed_results["cross_seed_topk_overlap"].items():
            if stats["mean_overlap"] is not None:
                print(f"  K={k}: {stats['mean_overlap']:.4f}")
    
    if "persistence_across_seeds" in seed_results:
        ps = seed_results["persistence_across_seeds"]
        print(f"\nPersistence across seeds (threshold={ps['threshold']}):")
        print(f"  Total unique edges: {ps['total_unique_edges']}")
        print(f"  Persistent across ALL seeds: {ps['persistent_all_seeds']}")
        print(f"  Fraction: {ps['fraction_persistent']:.1%}")
    
    # Save results
    prefix = DATASET_CONFIGS[dataset]["prefix"]
    json_path = os.path.join(RESULTS_DIR, f"stage25_graph_analysis_{prefix}.json")
    with open(json_path, "w") as f:
        json.dump(ph_results, f, indent=2, default=str)
    print(f"\nPH analysis saved to: {json_path}")
    
    json_path2 = os.path.join(RESULTS_DIR, f"stage25_seed_stability_{prefix}.json")
    with open(json_path2, "w") as f:
        json.dump(seed_results, f, indent=2, default=str)
    print(f"Seed stability saved to: {json_path2}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
