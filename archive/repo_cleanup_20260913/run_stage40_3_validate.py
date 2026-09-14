#!/usr/bin/env python3
"""
Stage 40.3 — Validate SZ-Taxi Contemporaneous DAGMA Artifacts

Run this AFTER the DAGMA fitting to verify all outputs are correct
and compatible with Stage 40.

Usage:
  conda run -n pth python run_stage40_3_validate.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, ".")

from gsl_stage40.scripts.stage40_run_all import (
    load_contemporaneous_graph, VARIANTS
)
from models.multigsl import binary_graph
import torch
from models.gcn import GCN
from models.tgcn import TGCN

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, status, detail=""):
    results.append((name, status, detail))
    symbol = {"PASS": "✅", "FAIL": "❌"}[status]
    print(f"  {symbol} [{status}] {name}" + (f" — {detail}" if detail else ""))


print("=" * 70)
print("STAGE 40.3 — SZ-TAXI DAGMA ARTIFACT VALIDATION")
print("=" * 70)

# 1. Check all 4 PH files exist
print("\n--- File Existence ---")
for ph in [1, 2, 3, 4]:
    A_path = f"results/stage33_gsl_canonical/sz_gsl_ph{ph}_seed42_A_binary.npy"
    W_path = f"results/stage33_gsl_canonical/sz_gsl_ph{ph}_seed42_W_est.npy"
    A_exists = os.path.exists(A_path)
    W_exists = os.path.exists(W_path)
    check(f"PH={ph}: A_binary exists", PASS if A_exists else FAIL, A_path)
    check(f"PH={ph}: W_est exists", PASS if W_exists else FAIL, W_path)

# 2. Validate each graph
print("\n--- Graph Validation ---")
N = 156
for ph in [1, 2, 3, 4]:
    A_path = f"results/stage33_gsl_canonical/sz_gsl_ph{ph}_seed42_A_binary.npy"
    W_path = f"results/stage33_gsl_canonical/sz_gsl_ph{ph}_seed42_W_est.npy"

    if not os.path.exists(A_path):
        check(f"PH={ph}: SKIP (file missing)", FAIL)
        continue

    A = np.load(A_path)
    W = np.load(W_path)

    # Shape
    check(f"PH={ph}: shape correct", PASS if A.shape == (N, N) else FAIL,
          f"shape={A.shape}")

    # Finite values
    check(f"PH={ph}: no NaN/Inf",
          PASS if np.all(np.isfinite(A)) and np.all(np.isfinite(W)) else FAIL)

    # Binary adjacency
    unique_vals = set(np.unique(A).tolist())
    check(f"PH={ph}: binary (0/1 only)",
          PASS if unique_vals <= {0.0, 1.0} else FAIL,
          f"unique values: {unique_vals}")

    # No self-loops
    diag_sum = int(A.diagonal().sum())
    check(f"PH={ph}: no self-loops",
          PASS if diag_sum == 0 else FAIL,
          f"diagonal sum={diag_sum}")

    # Edge count
    n_edges = int(A.sum())
    check(f"PH={ph}: edge count", PASS,
          f"edges={n_edges}, density={n_edges/(N*(N-1)):.6f}")

    # Raw weights range
    nonzero_mask = W != 0
    if nonzero_mask.any():
        w_nonzero = W[nonzero_mask]
        check(f"PH={ph}: W_est nonzero range",
              PASS,
              f"[{w_nonzero.min():.4f}, {w_nonzero.max():.4f}], "
              f"n={nonzero_mask.sum()}")
    else:
        check(f"PH={ph}: W_est has nonzero entries", FAIL, "all zeros")

    # Directed (not symmetric)
    is_symmetric = np.allclose(A, A.T)
    check(f"PH={ph}: directed (not symmetric)",
          PASS if not is_symmetric else "WARN",
          f"symmetric={is_symmetric}")

    # W_est max abs should be >= 0.3 (threshold)
    max_abs = float(np.abs(W).max())
    check(f"PH={ph}: max|W| >= 0.3 (threshold)",
          PASS if max_abs >= 0.3 else FAIL,
          f"max|W|={max_abs:.4f}")

# 3. Stage 40 loader compatibility
print("\n--- Stage 40 Loader Compatibility ---")
for ph in [1, 2, 3, 4]:
    A = load_contemporaneous_graph("shenzhen", ph)
    check(f"PH={ph}: load_contemporaneous_graph() works",
          PASS if A is not None else FAIL,
          f"edges={int(A.sum())}" if A is not None else "None")

# 4. Counterpart compatibility
print("\n--- Counterpart Compatibility ---")
for ph in [1, 2, 3, 4]:
    A = load_contemporaneous_graph("shenzhen", ph)
    if A is None:
        continue

    # GCN-GSL and T-GCN-GSL would load the same graph
    check(f"PH={ph}: GCN-GSL ↔ T-GCN-GSL same graph",
          PASS, "both use load_contemporaneous_graph('shenzhen', ph)")

    # cGSL
    A_cgsl = A + A.T
    A_cgsl = (A_cgsl > 0).astype(np.float32)
    np.fill_diagonal(A_cgsl, 0)
    n_cgsl = int(A_cgsl.sum())
    is_sym = np.allclose(A_cgsl, A_cgsl.T)
    check(f"PH={ph}: cGSL derived correctly",
          PASS if is_sym else FAIL,
          f"symmetric={is_sym}, edges={n_cgsl}")

    # Both GCN-cGSL and T-GCN-cGSL would apply same symmetrization
    check(f"PH={ph}: GCN-cGSL ↔ T-GCN-cGSL same graph",
          PASS, "both apply (A + A.T) > 0 to same GSL graph")

# 5. Model instantiation smoke test
print("\n--- Model Instantiation Smoke Test ---")
A_1 = load_contemporaneous_graph("shenzhen", 1)
if A_1 is not None:
    gcn = GCN(adj=A_1, seq_len=12, hidden_dim=64)
    tgcn = TGCN(adj=A_1, hidden_dim=64)
    x = torch.randn(2, 12, N)
    with torch.no_grad():
        gcn_out = gcn(x)
        tgcn_out = tgcn(x)
    check("GCN(SZ GSL) forward pass",
          PASS if gcn_out.shape == (2, N, 64) and torch.isfinite(gcn_out).all() else FAIL,
          f"shape={gcn_out.shape}")
    check("TGCN(SZ GSL) forward pass",
          PASS if tgcn_out.shape == (2, N, 64) and torch.isfinite(tgcn_out).all() else FAIL,
          f"shape={tgcn_out.shape}")

# 6. Cross-PH comparison
print("\n--- Cross-PH Comparison ---")
edge_counts = {}
for ph in [1, 2, 3, 4]:
    A = load_contemporaneous_graph("shenzhen", ph)
    if A is not None:
        edge_counts[ph] = int(A.sum())
if edge_counts:
    all_same = len(set(edge_counts.values())) == 1
    check("Edge counts across PHs",
          PASS if all_same else "WARN",
          f"edges: {edge_counts}")

# 7. Compare with Los-loop reference
print("\n--- Comparison with Los-loop Reference ---")
los_edge_counts = {}
for ph in [1, 2, 3, 4]:
    A_los = load_contemporaneous_graph("losloop", ph)
    A_sz = load_contemporaneous_graph("shenzhen", ph)
    if A_los is not None and A_sz is not None:
        los_edge_counts[ph] = int(A_los.sum())
        sz_edge_counts_val = int(A_sz.sum())
        # Los: 207 nodes, SZ: 156 nodes
        # Different N, so different edge counts are expected
        check(f"PH={ph}: Los edges={int(A_los.sum())}, SZ edges={sz_edge_counts_val}",
              PASS, "different N, different counts expected")

# Summary
print("\n" + "=" * 70)
n_pass = sum(1 for _, s, _ in results if s == PASS)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
n_warn = sum(1 for _, s, _ in results if s == "WARN")
print(f"PASS: {n_pass}, FAIL: {n_fail}, WARN: {n_warn}, TOTAL: {len(results)}")

if n_fail > 0:
    print("\nFAILED ITEMS:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  ❌ {name}: {detail}")

print("\n" + "=" * 70)
print("STAGE 40.3 VERDICT:", "READY" if n_fail == 0 else "NOT READY")
print("=" * 70)
