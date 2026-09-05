#!/usr/bin/env python3
"""
Comprehensive Post-Hoc Analysis of Fresh DAGMA Weight Matrices
=============================================================
TGCN-GSL-PyTorch — SZ-Taxi Dataset

Uses ONLY the fresh W files under results/dagma_fresh/sz_PH{1,2,3,4}_W.npy
Generated with w_threshold=0.0 (no thresholding applied).

DO NOT run DAGMA. Analysis only.
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRESH_DIR = os.path.join(ROOT, "results", "dagma_fresh")
OUT_DIR = os.path.join(ROOT, "results", "dagma_fresh", "threshold_analysis")
CACHED_DIR = os.path.join(ROOT, "data")
os.makedirs(OUT_DIR, exist_ok=True)

N = 156  # number of sensors for SZ-Taxi
HORIZONS = [1, 2, 3, 4]
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Load and verify fresh W files
# ═══════════════════════════════════════════════════════════════════════════

def load_fresh_W():
    """Load and verify the 4 fresh W matrices."""
    ws = {}
    print("=" * 80)
    print("SECTION 1: FRESH W FILE VERIFICATION")
    print("=" * 80)
    for ph in HORIZONS:
        path = os.path.join(FRESH_DIR, f"sz_PH{ph}_W.npy")
        assert os.path.exists(path), f"Missing: {path}"
        W = np.load(path)
        assert W.shape == (N, N), f"Unexpected shape for PH={ph}: {W.shape}"
        assert W.dtype == np.float64, f"Unexpected dtype: {W.dtype}"
        assert np.isfinite(W).all(), f"Non-finite values in PH={ph}"
        ws[ph] = W
        diag = np.diag(W)
        offdiag = W - np.diag(diag)
        print(f"\n  PH={ph}:")
        print(f"    shape: {W.shape}, dtype: {W.dtype}, finite: True")
        print(f"    range: [{W.min():.10f}, {W.max():.10f}]")
        print(f"    mean:  {W.mean():.10f}, std: {W.std():.10f}")
        print(f"    exact zeros: {np.sum(W == 0)}")
        print(f"    positive: {np.sum(W > 0)}, negative: {np.sum(W < 0)}")
        print(f"    diagonal nonzero: {np.count_nonzero(diag)}")
        print(f"    off-diagonal nonzero: {np.count_nonzero(offdiag)}")
        print(f"    max abs off-diagonal: {np.max(np.abs(offdiag)):.10f}")
        print(f"    max abs diagonal: {np.max(np.abs(diag)):.10f}")
    return ws


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Off-diagonal extraction helper
# ═══════════════════════════════════════════════════════════════════════════

def get_offdiag(W):
    """Return W with diagonal zeroed out."""
    W_off = W.copy()
    np.fill_diagonal(W_off, 0)
    return W_off


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Threshold sweep
# ═══════════════════════════════════════════════════════════════════════════

def threshold_sweep_entry(W_off, thr, mode="abs"):
    """Compute graph statistics for a given threshold."""
    N = W_off.shape[0]
    if mode == "abs":
        mask = np.abs(W_off) >= thr
    else:  # positive only
        mask = W_off >= thr

    n_edges = int(np.sum(mask))
    n_pos = int(np.sum(W_off[mask] > 0))
    n_neg = int(np.sum(W_off[mask] < 0))
    density = n_edges / (N * (N - 1))

    # Degrees (treating as directed)
    out_deg = mask.sum(axis=1)
    in_deg = mask.sum(axis=0)
    mean_out = float(np.mean(out_deg))
    mean_in = float(np.mean(in_deg))

    # Isolated nodes
    isolated = int(np.sum((out_deg == 0) & (in_deg == 0)))

    # Weakly connected components (via scipy)
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    # Make undirected for weak connectivity
    undirected = np.maximum(mask, mask.T).astype(int)
    n_components, labels = connected_components(csr_matrix(undirected), directed=False)
    component_sizes = np.bincount(labels)
    largest_cc = int(np.max(component_sizes))

    # Weight statistics
    abs_weights = np.abs(W_off[mask])
    if len(abs_weights) > 0:
        min_abs = float(np.min(abs_weights))
        median_abs = float(np.median(abs_weights))
        max_abs = float(np.max(abs_weights))
    else:
        min_abs = median_abs = max_abs = 0.0

    return {
        "threshold": thr, "mode": mode, "n_edges": n_edges,
        "n_positive": n_pos, "n_negative": n_neg, "density": density,
        "mean_out_degree": mean_out, "mean_in_degree": mean_in,
        "isolated_nodes": isolated, "n_components": n_components,
        "largest_cc": largest_cc,
        "min_abs_weight": min_abs, "median_abs_weight": median_abs,
        "max_abs_weight": max_abs,
    }


def run_threshold_sweep(ws):
    """Full threshold sweep for all PH values."""
    thresholds = [0, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3,
                  1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 3e-1]
    all_rows = []

    print("\n" + "=" * 80)
    print("SECTION 3: THRESHOLD SWEEP")
    print("=" * 80)

    for ph in HORIZONS:
        W_off = get_offdiag(ws[ph])
        print(f"\n  --- PH={ph} ---")
        for thr in thresholds:
            for mode in ["abs", "pos"]:
                row = threshold_sweep_entry(W_off, thr, mode)
                row["ph"] = ph
                all_rows.append(row)

        # Print summary table for this PH
        print(f"  {'thr':>10}  {'mode':>4}  {'edges':>7}  {'pos':>6}  {'neg':>6}  {'dens':>10}  {'isol':>5}  {'comp':>5}  {'max_cc':>6}  {'med_abs':>10}")
        for row in all_rows:
            if row["ph"] == ph:
                print(f"  {row['threshold']:10.5f}  {row['mode']:>4}  {row['n_edges']:>7d}  {row['n_positive']:>6d}  {row['n_negative']:>6d}  {row['density']:10.8f}  {row['isolated_nodes']:>5d}  {row['n_components']:>5d}  {row['largest_cc']:>6d}  {row['median_abs_weight']:10.8f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "threshold_sweep.csv"), index=False)
    print(f"\n  Saved: threshold_sweep.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Reproduce old graph construction at key thresholds
# ═══════════════════════════════════════════════════════════════════════════

def reproduce_old_graph(ws):
    """Reproduce old code's graph construction at different thresholds."""
    key_thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
    results = []

    print("\n" + "=" * 80)
    print("SECTION 4: REPRODUCE OLD GRAPH CONSTRUCTION")
    print("=" * 80)
    print("  adj = (W > 0).astype(int)  after  |W| < threshold → 0")

    for ph in HORIZONS:
        W = ws[ph]
        print(f"\n  --- PH={ph} ---")
        print(f"  {'threshold':>10}  {'edges':>7}  {'density':>10}  {'frac_of_all':>12}")
        for thr in key_thresholds:
            # Old code: W_est[np.abs(W_est) < w_threshold] = 0  (happens inside DAGMA)
            # Then: adj = (W_est > 0).astype(int)
            W_thr = W.copy()
            W_thr[np.abs(W_thr) < thr] = 0
            adj = (W_thr > 0).astype(int)
            np.fill_diagonal(adj, 0)
            n_edges = int(np.sum(adj))
            density = n_edges / (N * (N - 1))
            total_possible = N * (N - 1)
            frac = n_edges / total_possible if total_possible > 0 else 0
            print(f"  {thr:10.4f}  {n_edges:>7d}  {density:10.8f}  {frac:12.8f}")
            results.append({"ph": ph, "threshold": thr, "edges": n_edges,
                            "density": density, "fraction": frac})
    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: PH graph stability
# ═══════════════════════════════════════════════════════════════════════════

def ph_graph_stability(ws):
    """Compare graph structure across PH=1..4."""
    thresholds = [0.001, 0.01, 0.1, 0.3]
    results = []

    print("\n" + "=" * 80)
    print("SECTION 5: PH GRAPH STABILITY (POSITIVE-ONLY EDGES)")
    print("=" * 80)

    for thr in thresholds:
        print(f"\n  --- threshold={thr} (positive-only) ---")
        # Build edge sets for each PH
        edge_sets = {}
        for ph in HORIZONS:
            W_off = get_offdiag(ws[ph])
            mask = W_off >= thr
            edges = set(zip(*np.where(mask)))
            edge_sets[ph] = edges

        # Pairwise comparison
        for i in range(len(HORIZONS)):
            for j in range(i + 1, len(HORIZONS)):
                ph1, ph2 = HORIZONS[i], HORIZONS[j]
                e1, e2 = edge_sets[ph1], edge_sets[ph2]
                common = e1 & e2
                union = e1 | e2
                jaccard = len(common) / len(union) if len(union) > 0 else 0
                print(f"    PH={ph1} vs PH={ph2}: |E1|={len(e1)}, |E2|={len(e2)}, "
                      f"|common|={len(common)}, Jaccard={jaccard:.4f}")
                results.append({
                    "threshold": thr, "mode": "positive",
                    "ph1": ph1, "ph2": ph2,
                    "edges_ph1": len(e1), "edges_ph2": len(e2),
                    "common": len(common), "jaccard": jaccard,
                })

    # Also do absolute-value comparison at one threshold
    print(f"\n  --- Absolute-value edges at threshold=0.01 ---")
    edge_sets_abs = {}
    for ph in HORIZONS:
        W_off = get_offdiag(ws[ph])
        mask = np.abs(W_off) >= 0.01
        edges = set(zip(*np.where(mask)))
        edge_sets_abs[ph] = edges
    for i in range(len(HORIZONS)):
        for j in range(i + 1, len(HORIZONS)):
            ph1, ph2 = HORIZONS[i], HORIZONS[j]
            e1, e2 = edge_sets_abs[ph1], edge_sets_abs[ph2]
            common = e1 & e2
            union = e1 | e2
            jaccard = len(common) / len(union) if len(union) > 0 else 0
            print(f"    PH={ph1} vs PH={ph2}: |E1|={len(e1)}, |E2|={len(e2)}, "
                  f"|common|={len(common)}, Jaccard={jaccard:.4f}")
            results.append({
                "threshold": 0.01, "mode": "absolute",
                "ph1": ph1, "ph2": ph2,
                "edges_ph1": len(e1), "edges_ph2": len(e2),
                "common": len(common), "jaccard": jaccard,
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "ph_overlap.csv"), index=False)
    print(f"\n  Saved: ph_overlap.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Weight distribution
# ═══════════════════════════════════════════════════════════════════════════

def weight_distribution(ws):
    """Analyze weight distributions and find top edges."""
    quantile_probs = [0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
    all_quantile_rows = []
    all_top_edges = []

    print("\n" + "=" * 80)
    print("SECTION 6: WEIGHT DISTRIBUTION")
    print("=" * 80)

    for ph in HORIZONS:
        W_off = get_offdiag(ws[ph])
        abs_w = np.abs(W_off)
        nonzero_abs = abs_w[abs_w > 0]

        print(f"\n  --- PH={ph} ---")
        print(f"  Off-diagonal absolute weight quantiles:")
        quantiles = np.quantile(nonzero_abs, quantile_probs)
        print(f"  {'quantile':>10}  {'value':>14}")
        for q, v in zip(quantile_probs, quantiles):
            print(f"  {q:10.1%}  {v:14.10f}")
            all_quantile_rows.append({"ph": ph, "quantile": q, "value": v})

        # Top 20 edges
        print(f"\n  Top 20 off-diagonal edges (by absolute weight):")
        print(f"  {'rank':>4}  {'src':>4}  {'tgt':>4}  {'W':>14}  {'|W|':>14}  {'sign':>6}")
        abs_flat = abs_w.flatten()
        top_idx = np.argsort(abs_flat)[::-1][:20]
        for rank, idx in enumerate(top_idx, 1):
            i, j = divmod(idx, N)
            w_val = W_off[i, j]
            print(f"  {rank:>4d}  {i:>4d}  {j:>4d}  {w_val:14.10f}  {abs(w_val):14.10f}  {'+' if w_val > 0 else '-':>6}")
            all_top_edges.append({
                "ph": ph, "rank": rank, "src": i, "tgt": j,
                "weight": w_val, "abs_weight": abs(w_val),
                "sign": "positive" if w_val > 0 else "negative",
            })

    pd.DataFrame(all_quantile_rows).to_csv(os.path.join(OUT_DIR, "weight_quantiles.csv"), index=False)
    pd.DataFrame(all_top_edges).to_csv(os.path.join(OUT_DIR, "top_edges.csv"), index=False)
    print(f"\n  Saved: weight_quantiles.csv, top_edges.csv")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: w_threshold=0.3 aggressiveness
# ═══════════════════════════════════════════════════════════════════════════

def threshold_aggressiveness(ws):
    """Quantify fraction retained at various thresholds."""
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]

    print("\n" + "=" * 80)
    print("SECTION 7: IS w_threshold=0.3 AGGRESSIVELY REMOVING EDGES?")
    print("=" * 80)

    for ph in HORIZONS:
        W_off = get_offdiag(ws[ph])
        abs_w = np.abs(W_off)
        total_nonzero = np.count_nonzero(abs_w)
        total_entries = N * (N - 1)

        print(f"\n  --- PH={ph} ---")
        print(f"  Total off-diagonal entries: {total_entries}")
        print(f"  Total nonzero (at w_threshold=0): {total_nonzero}")
        print(f"  {'threshold':>10}  {'n_abs>=thr':>12}  {'n_pos>=thr':>12}  {'frac_total':>12}  {'frac_nonzero':>14}")
        for thr in thresholds:
            n_abs = int(np.sum(abs_w >= thr))
            n_pos = int(np.sum(W_off >= thr))
            frac_total = n_abs / total_entries if total_entries > 0 else 0
            frac_nz = n_abs / total_nonzero if total_nonzero > 0 else 0
            print(f"  {thr:10.4f}  {n_abs:>12d}  {n_pos:>12d}  {frac_total:12.8f}  {frac_nz:14.8f}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Near-zero values analysis
# ═══════════════════════════════════════════════════════════════════════════

def near_zero_analysis(ws):
    """Investigate near-zero values."""
    print("\n" + "=" * 80)
    print("SECTION 8: NEAR-ZERO VALUES ANALYSIS")
    print("=" * 80)

    for ph in HORIZONS:
        W_off = get_offdiag(ws[ph])
        abs_w = np.abs(W_off)

        print(f"\n  --- PH={ph} ---")
        # Count entries in various magnitude bins
        bins = [0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 1.0]
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            count = int(np.sum((abs_w >= lo) & (abs_w < hi)))
            frac = count / (N * (N - 1))
            bar = "#" * min(50, int(frac * 500))
            print(f"    [{lo:.0e}, {hi:.0e}):  {count:>8d}  ({frac:8.6f})  {bar}")
        count_ge_03 = int(np.sum(abs_w >= 0.3))
        print(f"    [3e-01, inf):  {count_ge_03:>8d}  ({count_ge_03/(N*(N-1)):8.6f})")

        # Check if the optimizer is producing these or if it's numerical noise
        # The DAGMA minimize function uses Adam with lr=0.0003
        # Values << lr are likely near convergence floor
        lr = 0.0003
        below_lr = int(np.sum(abs_w < lr))
        above_lr = int(np.sum(abs_w >= lr))
        print(f"\n    Entries < lr (0.0003): {below_lr}  ({below_lr/(N*(N-1)):.6f})")
        print(f"    Entries >= lr (0.0003): {above_lr}  ({above_lr/(N*(N-1)):.6f})")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: Sign analysis
# ═══════════════════════════════════════════════════════════════════════════

def sign_analysis(ws):
    """Check positive vs negative coefficients at various thresholds."""
    thresholds = [0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1]

    print("\n" + "=" * 80)
    print("SECTION 9: SIGN ANALYSIS (POSITIVE vs NEGATIVE)")
    print("=" * 80)

    for ph in HORIZONS:
        W_off = get_offdiag(ws[ph])
        print(f"\n  --- PH={ph} ---")
        print(f"  {'threshold':>10}  {'pos':>8}  {'neg':>8}  {'total':>8}  {'frac_pos':>10}")
        for thr in thresholds:
            if thr == 0:
                mask = np.ones_like(W_off, dtype=bool)
            else:
                mask = np.abs(W_off) >= thr
            n_pos = int(np.sum((W_off > 0) & mask))
            n_neg = int(np.sum((W_off < 0) & mask))
            total = n_pos + n_neg
            frac_pos = n_pos / total if total > 0 else 0
            print(f"  {thr:10.5f}  {n_pos:>8d}  {n_neg:>8d}  {total:>8d}  {frac_pos:10.4f}")

        # Does W > 0 (old code) discard meaningful negative edges at practical thresholds?
        print(f"\n  At threshold=0.01: W>0 keeps {int(np.sum(W_off >= 0.01))} edges")
        print(f"  At threshold=0.01: |W|>=0.01 keeps {int(np.sum(np.abs(W_off) >= 0.01))} edges")
        print(f"  Negative edges discarded: {int(np.sum((W_off < 0) & (np.abs(W_off) >= 0.01)))}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: Fresh vs cached comparison
# ═══════════════════════════════════════════════════════════════════════════

def fresh_vs_cached(ws):
    """Compare fresh W against cached W_est files."""
    results = []

    print("\n" + "=" * 80)
    print("SECTION 11: FRESH W vs CACHED W_est COMPARISON")
    print("=" * 80)

    for ph in HORIZONS:
        cached_path = os.path.join(CACHED_DIR, f"W_est_shenzhen_pre_len{ph}.npy")
        if not os.path.exists(cached_path):
            print(f"\n  PH={ph}: CACHED FILE NOT FOUND: {cached_path}")
            continue

        W_cached_all = np.load(cached_path)
        print(f"\n  --- PH={ph} ---")
        print(f"  Cached shape: {W_cached_all.shape}")
        print(f"  Cached dtype: {W_cached_all.dtype}")

        if W_cached_all.ndim == 3:
            W_cached = W_cached_all[:, :, ph - 1]
        else:
            W_cached = W_cached_all

        print(f"  Cached W shape: {W_cached.shape}")

        # Check if cached is already thresholded
        n_zero = np.sum(W_cached == 0)
        n_nonzero = np.count_nonzero(W_cached)
        total = W_cached.size
        print(f"  Cached zeros: {n_zero} / {total} ({n_zero/total:.6f})")
        print(f"  Cached nonzero: {n_nonzero} / {total} ({n_nonzero/total:.6f})")

        # Fresh W at threshold 0.3
        W_fresh = ws[ph]
        W_thr = W_fresh.copy()
        W_thr[np.abs(W_thr) < 0.3] = 0

        # Compare positive-only graphs
        adj_fresh = (W_thr > 0).astype(int)
        np.fill_diagonal(adj_fresh, 0)
        adj_cached = (W_cached > 0).astype(int)
        np.fill_diagonal(adj_cached, 0)

        fresh_edges = set(zip(*np.where(adj_fresh > 0)))
        cached_edges = set(zip(*np.where(adj_cached > 0)))
        common = fresh_edges & cached_edges
        union = fresh_edges | cached_edges
        jaccard = len(common) / len(union) if len(union) > 0 else 0

        print(f"  Fresh (thr=0.3, pos>0) edges: {len(fresh_edges)}")
        print(f"  Cached (pos>0) edges: {len(cached_edges)}")
        print(f"  Common: {len(common)}")
        print(f"  Jaccard: {jaccard:.4f}")
        print(f"  Only in fresh: {len(fresh_edges - cached_edges)}")
        print(f"  Only in cached: {len(cached_edges - fresh_edges)}")

        # Check exact match
        exact_match = np.array_equal(adj_fresh, adj_cached)
        print(f"  Binary adjacency exact match: {exact_match}")

        # Check at threshold 0.01
        W_thr01 = W_fresh.copy()
        W_thr01[np.abs(W_thr01) < 0.01] = 0
        adj_fresh01 = (W_thr01 > 0).astype(int)
        np.fill_diagonal(adj_fresh01, 0)
        fresh01_edges = set(zip(*np.where(adj_fresh01 > 0)))
        common01 = fresh01_edges & cached_edges
        union01 = fresh01_edges | cached_edges
        jaccard01 = len(common01) / len(union01) if len(union01) > 0 else 0
        print(f"\n  At threshold=0.01 vs cached:")
        print(f"  Fresh edges: {len(fresh01_edges)}, Common: {len(common01)}, Jaccard: {jaccard01:.4f}")

        results.append({
            "ph": ph, "cached_zeros": n_zero, "cached_nonzero": n_nonzero,
            "fresh_edges_03": len(fresh_edges), "cached_edges": len(cached_edges),
            "common_03": len(common), "jaccard_03": jaccard,
            "exact_match": exact_match,
            "fresh_edges_01": len(fresh01_edges), "common_01": len(common01),
            "jaccard_01": jaccard01,
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "fresh_vs_cached.csv"), index=False)
    print(f"\n  Saved: fresh_vs_cached.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("DAGMA THRESHOLD ANALYSIS — SZ-Taxi (Fresh W, w_threshold=0.0)")
    print("=" * 80)
    print(f"  Seed: {SEED}")
    print(f"  DAGMA version: 1.1.1")
    print(f"  All W matrices have NO thresholding applied (w_threshold=0.0)")
    print()

    ws = load_fresh_W()
    run_threshold_sweep(ws)
    reproduce_old_graph(ws)
    ph_graph_stability(ws)
    weight_distribution(ws)
    threshold_aggressiveness(ws)
    near_zero_analysis(ws)
    sign_analysis(ws)
    fresh_vs_cached(ws)

    print("\n" + "=" * 80)
    print("ALL ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"  Output directory: {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"    {f}")


if __name__ == "__main__":
    main()
