#!/usr/bin/env python
"""
Forensic diagnostic script for DAGMA weight analysis.
Investigates the root cause of extreme sparsity in learned graphs.
"""

import numpy as np
import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

datasets = ['shenzhen', 'losloop']
pre_lens = [1, 2, 3, 4]


def analyze_single_slice(W, dataset, pre_len, slice_idx):
    """Analyze a single 2D W matrix."""
    N = W.shape[0]
    total_elements = N * N
    diag = np.diag(W)
    W_no_diag = W.copy()
    np.fill_diagonal(W_no_diag, 0)
    off_diag = W_no_diag[W_no_diag != 0]
    
    result = {
        'dataset': dataset, 'pre_len': pre_len, 'slice': slice_idx,
        'N': N, 'total_elements': total_elements,
    }
    
    # Basic stats
    result['min'] = float(np.min(W))
    result['max'] = float(np.max(W))
    result['mean'] = float(np.mean(W))
    result['std'] = float(np.std(W))
    result['max_abs'] = float(np.max(np.abs(W)))
    
    # Sign analysis
    result['exact_zero'] = int(np.sum(W == 0))
    result['positive_total'] = int(np.sum(W > 0))
    result['negative_total'] = int(np.sum(W < 0))
    result['nonzero_total'] = int(np.sum(W != 0))
    
    # Off-diagonal sign analysis
    result['offdiag_nonzero'] = int(len(off_diag))
    result['offdiag_positive'] = int(np.sum(off_diag > 0))
    result['offdiag_negative'] = int(np.sum(off_diag < 0))
    
    # Diagonal
    result['diag_mean'] = float(np.mean(diag))
    result['diag_max'] = float(np.max(diag))
    result['diag_min'] = float(np.min(diag))
    result['diag_nonzero'] = int(np.sum(diag != 0))
    
    # Magnitude thresholds (off-diagonal)
    thresholds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1]
    mag_analysis = {}
    for t in thresholds:
        count = int(np.sum(np.abs(W_no_diag) > t))
        density = count / (N * (N - 1))
        mag_analysis[f'|W|>{t}'] = {'count': count, 'density': density}
    result['magnitude_analysis'] = mag_analysis
    
    # W > 0 analysis
    W_pos = (W_no_diag > 0).astype(int)
    result['w_positive_edges'] = int(np.sum(W_pos))
    result['w_positive_density'] = result['w_positive_edges'] / (N * (N - 1))
    
    # cGSL: W>0 symmetrized
    W_cGSL = W_pos + W_pos.T
    W_cGSL[W_cGSL > 0] = 1
    result['cgsl_edges'] = int(np.sum(W_cGSL) // 2)
    
    # Top 30 entries by absolute magnitude
    abs_flat = np.abs(W_no_diag)
    indices = np.unravel_index(np.argsort(abs_flat.ravel())[::-1], W_no_diag.shape)
    top_entries = []
    for rank in range(min(30, len(indices[0]))):
        i, j = indices[0][rank], indices[1][rank]
        val = W_no_diag[i, j]
        top_entries.append({
            'rank': rank + 1, 'source': int(i), 'target': int(j),
            'W_ij': float(val), 'abs_W_ij': float(abs(val)),
            'sign': '+' if val > 0 else '-'
        })
    result['top_entries'] = top_entries
    
    # All nonzero off-diagonal values
    result['all_nonzero_values'] = sorted(off_diag.tolist(), key=lambda x: abs(x), reverse=True)
    
    return result


def analyze_connectivity(W_binary, N):
    """Analyze graph connectivity from binary adjacency matrix."""
    W_undir = ((W_binary + W_binary.T) > 0).astype(int)
    np.fill_diagonal(W_undir, 0)
    
    degrees = W_undir.sum(axis=1)
    
    # BFS connected components
    visited = np.zeros(N, dtype=bool)
    components = []
    for start in range(N):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        comp = [start]
        while queue:
            node = queue.pop(0)
            neighbors = np.where(W_undir[node] > 0)[0]
            for n in neighbors:
                if not visited[n]:
                    visited[n] = True
                    queue.append(n)
                    comp.append(n)
        components.append(comp)
    components.sort(key=len, reverse=True)
    
    return {
        'edges_undirected': int(np.sum(W_undir) // 2),
        'min_degree': int(np.min(degrees)),
        'max_degree': int(np.max(degrees)),
        'mean_degree': float(np.mean(degrees)),
        'median_degree': float(np.median(degrees)),
        'std_degree': float(np.std(degrees)),
        'isolated_nodes': int(np.sum(degrees == 0)),
        'isolated_pct': float(np.sum(degrees == 0) / N * 100),
        'nodes_deg1': int(np.sum(degrees == 1)),
        'nodes_deg_ge2': int(np.sum(degrees >= 2)),
        'nodes_deg_ge5': int(np.sum(degrees >= 5)),
        'nodes_deg_ge10': int(np.sum(degrees >= 10)),
        'num_components': len(components),
        'largest_component_size': len(components[0]),
        'component_sizes': [len(c) for c in components[:10]],
        'degree_dist': degrees.tolist(),
    }


def main():
    print("=" * 90)
    print("DAGMA SPARSITY FORENSIC DIAGNOSTIC")
    print("=" * 90)
    
    all_slice_results = []
    all_combined_results = []
    
    for dataset in datasets:
        for pre_len in pre_lens:
            filepath = f"data/W_est_{dataset}_pre_len{pre_len}.npy"
            if not os.path.exists(filepath):
                print(f"WARNING: {filepath} not found, skipping.")
                continue
            
            W_all = np.load(filepath)
            N = W_all.shape[0]
            
            print(f"\n{'=' * 90}")
            print(f"FILE: {filepath}  |  Shape: {W_all.shape}  |  Nodes: {N}")
            print(f"{'=' * 90}")
            
            # Analyze each slice
            nonzero_abs_all = []
            for s in range(pre_len):
                W_slice = W_all[:, :, s]
                r = analyze_single_slice(W_slice, dataset, pre_len, s)
                all_slice_results.append(r)
                
                print(f"\n--- Slice {s} (PH component {s}) ---")
                print(f"  Off-diag nonzero: {r['offdiag_nonzero']}")
                print(f"  Off-diag positive: {r['offdiag_positive']}")
                print(f"  Off-diag negative: {r['offdiag_negative']}")
                print(f"  GSL edges (W>0): {r['w_positive_edges']}")
                print(f"  cGSL edges: {r['cgsl_edges']}")
                
                print(f"\n  Magnitude thresholds:")
                for key, info in r['magnitude_analysis'].items():
                    print(f"    {key:>14}: {info['count']:>6} edges  density={info['density']:.6f}")
                
                print(f"\n  Top 15 entries:")
                print(f"    {'Rank':>4} | {'Src':>4}->{'Tgt':>4} | {'W_ij':>12} | {'|W|':>12} | Sign")
                print(f"    " + "-" * 60)
                for e in r['top_entries'][:15]:
                    print(f"    {e['rank']:>4} | {e['source']:>4}->{e['target']:>4} | {e['W_ij']:>12.6f} | {e['abs_W_ij']:>12.6f} | {e['sign']}")
                
                # Collect nonzero abs values for threshold analysis
                W_nd = W_slice.copy()
                np.fill_diagonal(W_nd, 0)
                nz = W_nd[W_nd != 0]
                if len(nz) > 0:
                    nonzero_abs_all.extend(np.abs(nz).tolist())
            
            # DAGMA w_threshold evidence
            print(f"\n--- DAGMA w_threshold=0.3 Evidence ---")
            if nonzero_abs_all:
                nonzero_abs = np.array(nonzero_abs_all)
                print(f"  Total nonzero off-diag values across all slices: {len(nonzero_abs)}")
                print(f"  Min |W|: {np.min(nonzero_abs):.8f}")
                print(f"  Max |W|: {np.max(nonzero_abs):.8f}")
                print(f"  All |W| >= 0.3 - 1e-6: {bool(np.all(nonzero_abs >= 0.3 - 1e-6))}")
                print(f"  All |W| >= 0.25: {bool(np.all(nonzero_abs >= 0.25))}")
                print(f"  All |W| >= 0.2: {bool(np.all(nonzero_abs >= 0.2))}")
                print(f"  Count |W| < 0.3: {int(np.sum(nonzero_abs < 0.3 - 1e-6))}")
                print(f"  Count |W| < 0.25: {int(np.sum(nonzero_abs < 0.25))}")
                print(f"  Count |W| < 0.1: {int(np.sum(nonzero_abs < 0.1))}")
                print(f"  Count |W| < 0.05: {int(np.sum(nonzero_abs < 0.05))}")
                
                # Histogram
                hist, bin_edges = np.histogram(nonzero_abs, bins=50)
                print(f"\n  Histogram of |W| (nonzero values):")
                for i in range(len(hist)):
                    bar = '#' * min(hist[i], 60)
                    print(f"    [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}): {hist[i]:>5} {bar}")
            else:
                print(f"  No nonzero off-diagonal values found!")
            
            # Connectivity
            W_for_conn = (W_all[:, :, 0] > 0).astype(int)
            conn = analyze_connectivity(W_for_conn, N)
            print(f"\n--- GSL Graph Connectivity (slice 0, W>0) ---")
            print(f"  Edges: {conn['edges_undirected']}")
            print(f"  Isolated nodes: {conn['isolated_nodes']} / {N} ({conn['isolated_pct']:.1f}%)")
            print(f"  Mean degree: {conn['mean_degree']:.2f}")
            print(f"  Max degree: {conn['max_degree']}")
            print(f"  Connected components: {conn['num_components']}")
            print(f"  Largest component: {conn['largest_component_size']}")
            print(f"  Component sizes: {conn['component_sizes']}")
            
            all_combined_results.append({
                'dataset': dataset, 'pre_len': pre_len,
                'nonzero_all_slices': len(nonzero_abs_all),
                'conn_slice0': conn,
            })
    
    # PH stability
    print("\n" + "=" * 90)
    print("PH STABILITY: Edge Overlap Analysis")
    print("=" * 90)
    for dataset in datasets:
        print(f"\n--- {dataset.upper()} ---")
        graphs = {}
        for ph in pre_lens:
            filepath = f"data/W_est_{dataset}_pre_len{ph}.npy"
            W = np.load(filepath)
            W_01 = (W[:, :, 0] > 0).astype(int)
            np.fill_diagonal(W_01, 0)
            graphs[ph] = set(zip(*np.where(W_01 > 0)))
        
        for ph1 in [1]:
            for ph2 in [2, 3, 4]:
                s1, s2 = graphs[ph1], graphs[ph2]
                if s1 and s2:
                    jaccard = len(s1 & s2) / len(s1 | s2)
                    print(f"  PH={ph1} vs PH={ph2}: Jaccard={jaccard:.4f}  intersect={len(s1 & s2)}  union={len(s1 | s2)}")
                else:
                    print(f"  PH={ph1} vs PH={ph2}: one or both empty")
    
    # Physical graph reference
    print("\n" + "=" * 90)
    print("PHYSICAL GRAPH REFERENCE")
    print("=" * 90)
    for ds_name, prefix in [('shenzhen', 'sz'), ('losloop', 'los')]:
        adj_path = f"data/{prefix}_adj.csv"
        adj = np.loadtxt(adj_path, delimiter=',')
        N = adj.shape[0]
        edges = int(np.sum(adj > 0))
        density = edges / (N * (N - 1))
        degrees = adj.sum(axis=1)
        print(f"\n{ds_name}:")
        print(f"  Nodes: {N}")
        print(f"  Edges: {edges}")
        print(f"  Density: {density:.6f}")
        print(f"  Mean degree: {np.mean(degrees):.1f}")
        print(f"  Max degree: {np.max(degrees):.0f}")
        print(f"  Min degree: {np.min(degrees):.0f}")
    
    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, 'w_est_analysis.json')
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(json_path, 'w') as f:
        json.dump({'slices': all_slice_results, 'combined': all_combined_results}, f, indent=2, cls=NumpyEncoder)
    print(f"\nFull analysis saved to: {json_path}")


if __name__ == '__main__':
    main()
