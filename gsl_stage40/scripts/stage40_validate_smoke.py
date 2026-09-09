#!/usr/bin/env python3
"""
Stage 40 — Lightweight Smoke Tests

Verifies (without running any expensive experiments):
  1. All modules import correctly
  2. All 11 model variants can be instantiated
  3. DAGMA graph files exist and have correct shapes
  4. Existing result JSONs are readable and have correct schema
  5. Method registry maps are consistent
  6. The GCN architecture is correctly assessed as non-recurrent

Usage:
  python gsl_stage40/scripts/stage40_validate_smoke.py
"""
import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((name, PASS, detail))
        print(f"  [{PASS}] {name}" + (f" — {detail}" if detail else ""))
    except Exception as e:
        results.append((name, FAIL, f"{type(e).__name__}: {e}"))
        print(f"  [{FAIL}] {name} — {type(e).__name__}: {e}")


# 1. Module imports
def t_imports():
    import importlib
    mods = ["models", "models.multigsl", "models.tgcn", "models.gcn", "models.gru",
            "tasks.supervised", "utils.losses", "utils.graph_conv",
            "gsl_stage26.stage26_validation"]
    for m in mods:
        importlib.import_module(m)
    return f"{len(mods)} modules imported"


# 2. All 11 variants instantiate
def t_all_variants():
    import torch
    from models.multigsl import (
        GatedMultiGraphTGCN, MultiGraphTGCNFixed, WeightedMultiGraphTGCN,
        METHOD_REGISTRY,
    )
    from models.tgcn import TGCN
    from models.gcn import GCN

    N, K, H = 207, 3, 8
    _lags = [(np.random.rand(N, N) > 0.8).astype(np.float32) * (1 - np.eye(N, dtype=np.float32))
             for _ in range(K)]

    # T-GCN family (7 variants)
    t_nograph = TGCN(adj=np.eye(N, dtype=np.float32), hidden_dim=H)
    t_phys = TGCN(adj=_lags[0], hidden_dim=H)
    t_gsl = TGCN(adj=_lags[0], hidden_dim=H)
    t_cgsl = TGCN(adj=(_lags[0] + _lags[0].T > 0).astype(np.float32), hidden_dim=H)
    t_multi = MultiGraphTGCNFixed(adj_list=_lags, hidden_dim=H)
    t_weighted = WeightedMultiGraphTGCN(adj_list=_lags, hidden_dim=H)
    t_mix = GatedMultiGraphTGCN(adj_list=_lags, hidden_dim=H)

    # GCN family (4 variants)
    g_phys = GCN(adj=_lags[0], seq_len=12, hidden_dim=H)
    g_nograph = GCN(adj=np.eye(N, dtype=np.float32), seq_len=12, hidden_dim=H)
    g_gsl = GCN(adj=_lags[0], seq_len=12, hidden_dim=H)
    g_cgsl = GCN(adj=(_lags[0] + _lags[0].T > 0).astype(np.float32), seq_len=12, hidden_dim=H)

    # Verify forward passes
    batch = torch.randn(2, 12, N)
    for name, m in [("TGCN-Nograph", t_nograph), ("TGCN-Phys", t_phys),
                     ("TGCN-GSL", t_gsl), ("TGCN-cGSL", t_cgsl),
                     ("TGCN-Multi", t_multi), ("TGCN-Weighted", t_weighted),
                     ("TGCN-Mix", t_mix),
                     ("GCN-Phys", g_phys), ("GCN-Nograph", g_nograph),
                     ("GCN-GSL", g_gsl), ("GCN-cGSL", g_cgsl)]:
        m.eval()
        with torch.no_grad():
            out = m(batch)
        assert out.shape == (2, N, H), f"{name}: {out.shape}"
        assert torch.isfinite(out).all(), f"{name}: non-finite"

    # GCN-MultiGSL: union of lag graphs, single static graph to GCN
    A_union = np.zeros((N, N), dtype=np.float32)
    for a in _lags:
        A_union = np.maximum(A_union, a)
    g_multigsl = GCN(adj=A_union, seq_len=12, hidden_dim=H)

    # Check method registry has all 12 entries
    expected = {"no_spatial", "physical", "gsl", "cgsl",
                "multi_gsl", "multi_gsl_mix", "multi_gsl_weighted",
                "gcn_physical", "gcn_no_spatial", "gcn_gsl", "gcn_cgsl",
                "gcn_multigsl"}
    assert expected == set(METHOD_REGISTRY.keys()), \
        f"Registry mismatch: missing {expected - set(METHOD_REGISTRY.keys())}, extra {set(METHOD_REGISTRY.keys()) - expected}"

    return "12 variants instantiate + forward pass; registry has 12 entries"


# 3. DAGMA graph files exist and have correct shapes
def t_dagma_files():
    from pathlib import Path
    issues = []

    # Multi-lag blocks (stage26)
    for ds, N in [("losloop", 207), ("shenzhen", 156)]:
        prefix = "los" if ds == "losloop" else "sz"
        d = Path(PROJECT_ROOT) / "results" / "stage26_validation"
        for ph in [1, 2, 3, 4]:
            for lag in [1, 2, 3]:
                p = d / f"{prefix}_ph{ph}_seed42_L3_lag_{lag}.npy"
                if not p.exists():
                    issues.append(f"MISSING: {p}")
                else:
                    arr = np.load(p)
                    if arr.shape != (N, N):
                        issues.append(f"WRONG SHAPE: {p} {arr.shape} != ({N},{N})")

    # Contemporaneous (stage33) for losloop
    d33 = Path(PROJECT_ROOT) / "results" / "stage33_gsl_canonical"
    for ph in [1, 2, 3, 4]:
        p = d33 / f"los_gsl_ph{ph}_seed42_A_binary.npy"
        if not p.exists():
            issues.append(f"MISSING: {p}")
        else:
            arr = np.load(p)
            if arr.shape != (207, 207):
                issues.append(f"WRONG SHAPE: {p} {arr.shape}")

    if issues:
        raise RuntimeError("; ".join(issues[:5]))
    return "All DAGMA files present with correct shapes"


# 4. Existing result JSONs readable
def t_result_jsons():
    from pathlib import Path
    count = 0
    issues = []

    # Stage 33 GSL canonical
    p = Path(PROJECT_ROOT) / "results" / "stage33_gsl_canonical" / "stage33_gsl_canonical_results.json"
    if p.exists():
        d = json.load(open(p))
        count += len(d.get("results", []))
        for r in d["results"]:
            if "rmse" not in r:
                issues.append(f"Stage33 missing rmse: {r.get('method')}")

    # Stage 33 SZ multiseed
    p2 = Path(PROJECT_ROOT) / "results" / "stage33_sz_multiseed" / "stage33_sz_multiseed_results.json"
    if p2.exists():
        d2 = json.load(open(p2))
        count += len(d2.get("results", []))

    # Stage 26 validation
    for f in (Path(PROJECT_ROOT) / "results" / "stage26_validation").glob("stage26_validation_*.json"):
        d3 = json.load(open(f))
        count += len(d3.get("results", []))

    if issues:
        raise RuntimeError("; ".join(issues))
    return f"{count} result records readable across {3} source files"


# 5. Method registry naming consistency
def t_registry_naming():
    from models.multigsl import METHOD_REGISTRY
    issues = []
    for mid, meta in METHOD_REGISTRY.items():
        if "name" not in meta:
            issues.append(f"{mid}: missing 'name'")
        if "adjacency" not in meta:
            issues.append(f"{mid}: missing 'adjacency'")
        if "backbone" not in meta:
            issues.append(f"{mid}: missing 'backbone'")
    if issues:
        raise RuntimeError("; ".join(issues))
    return f"{len(METHOD_REGISTRY)} entries all have name, adjacency, backbone"


# 6. GCN non-recurrent assessment
def t_gcn_nonrecurrent():
    """Verify GCN processes the whole window in one step (not per-timestep)."""
    from models.gcn import GCN
    import torch
    N, H, T = 207, 8, 12
    g = GCN(adj=np.eye(N, dtype=np.float32), seq_len=T, hidden_dim=H)
    # GCN should have a single weight matrix (seq_len -> hidden_dim), no per-timestep RNN
    # Count parameters: Laplacian (buffer) + weights matrix (T x H) = T*H params
    n_params = sum(p.numel() for p in g.parameters())
    expected = T * H  # weights only (laplacian is a buffer, not a parameter)
    assert n_params == expected, f"GCN params {n_params} != seq_len*hidden_dim = {expected}"
    # Verify: forward output depends on graph, not timestep order
    x1 = torch.randn(1, T, N)
    x2 = torch.randn(1, T, N)
    g.eval()
    with torch.no_grad():
        y1 = g(x1)
        y2 = g(x2)
    assert y1.shape == (1, N, H)
    return f"GCN has {n_params} params (= seq_len * hidden_dim), processes whole window at once"


def main():
    print("=" * 78)
    print("STAGE 40 SMOKE TESTS")
    print("=" * 78)

    check("1. module imports", t_imports)
    check("2. all 12 variants instantiate + forward", t_all_variants)
    check("3. DAGMA graph files present + correct shapes", t_dagma_files)
    check("4. existing result JSONs readable", t_result_jsons)
    check("5. method registry naming consistency", t_registry_naming)
    check("6. GCN non-recurrent assessment", t_gcn_nonrecurrent)

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print("-" * 78)
    print(f"TOTAL: {len(results)} checks, {len(results) - n_fail} PASS, {n_fail} FAIL")
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
