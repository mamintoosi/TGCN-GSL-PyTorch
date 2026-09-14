#!/usr/bin/env python3
"""
Stage 40 — Canonical Experiment Pipeline (T-GCN + GCN families, both datasets)

Resumable launcher covering:
  T-GCN  : Physical, NoSpatial, GSL, cGSL, MultiGSL, MultiGSL-Weighted, MultiGSL-Mix
  GCN    : Physical, NoSpatial, GSL, cGSL

Datasets  : losloop, shenzhen
PHs       : 1, 2, 3, 4
Seeds     : 42, 43, 44, 45, 46

Reuses all existing DAGMA outputs (multilag blocks from stage26, contemporaneous
from stage33).  Never recomputes DAGMA unless the output file is missing.

Resumability:
  - Every experiment writes a result JSON atomically (write to temp, rename).
  - On re-launch, each (variant, dataset, ph, seed) is checked; if a valid
    result exists, the experiment is skipped.
  - Logs are appended to a per-run log file.

Usage:
  # Full run (T-GCN + GCN, both datasets, all PHs, all seeds):
  python gsl_stage40/scripts/stage40_run_all.py

  # Subset:
  python gsl_stage40/scripts/stage40_run_all.py --variants physical no_spatial --datasets losloop --phs 1 --seeds 42

  # T-GCN-GSL / cGSL only (need contemporaneous DAGMA):
  python gsl_stage40/scripts/stage40_run_all.py --variants gsl cgsl --datasets losloop

  # GCN family only:
  python gsl_stage40/scripts/stage40_run_all.py --backbone gcn

  # Dry-run (print what would be run):
  python gsl_stage40/scripts/stage40_run_all.py --dry-run

  # Smoke test (2 epochs, single seed):
  python gsl_stage40/scripts/stage40_run_all.py --max-epochs 2 --seeds 42
"""
import os
import sys
import json
import time
import shutil
import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tasks.supervised import SupervisedForecastTask
from models.tgcn import TGCN
from models.gcn import GCN
from models.multigsl import (
    GatedMultiGraphTGCN, MultiGraphTGCNFixed, WeightedMultiGraphTGCN,
    binary_graph, normalize_method, METHOD_REGISTRY,
)

# ======================================================================
# Configuration
# ======================================================================
RESULTS_DIR = PROJECT_ROOT / "results" / "stage40_canonical"
DAGMA_DIR = RESULTS_DIR / "dagma"
TRAINING_DIR = RESULTS_DIR / "training"
LOG_DIR = RESULTS_DIR / "logs"
FIG_DIR = RESULTS_DIR / "figures"

for d in [RESULTS_DIR, DAGMA_DIR, TRAINING_DIR, LOG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASET_CONFIGS = {
    "losloop": {
        "feat_path": "data/los_speed.csv",
        "adj_path": "data/los_adj.csv",
        "N": 207, "prefix": "los",
        "lambda1_multilag": 0.02, "lambda1_contemporaneous": 0.02,
    },
    "shenzhen": {
        "feat_path": "data/sz_speed.csv",
        "adj_path": "data/sz_adj.csv",
        "N": 156, "prefix": "sz",
        "lambda1_multilag": 0.01, "lambda1_contemporaneous": 0.01,
    },
}

# ======================================================================
# Canonical variant definitions
# ======================================================================
VARIANTS = {
    # T-GCN family
    "physical":      {"display": "T-GCN",                "backbone": "tgcn", "adj_type": "single",    "dagma": None},
    "no_spatial":    {"display": "T-GCN-NoSpatial",      "backbone": "tgcn", "adj_type": "identity",  "dagma": None},
    "gsl":           {"display": "T-GCN-GSL",            "backbone": "tgcn", "adj_type": "single",    "dagma": "contemporaneous", "dagma_thr": 0.3},
    "cgsl":          {"display": "T-GCN-cGSL",           "backbone": "tgcn", "adj_type": "single",    "dagma": "contemporaneous", "dagma_thr": 0.3, "symmetrize": True},
    "multi_gsl":     {"display": "T-GCN-MultiGSL",       "backbone": "tgcn", "adj_type": "lag_list",  "dagma": "multilag", "dagma_thr": 0.1},
    "multi_gsl_weighted": {"display": "T-GCN-MultiGSL-Weighted", "backbone": "tgcn", "adj_type": "lag_list", "dagma": "multilag", "dagma_thr": 0.1},
    "multi_gsl_mix": {"display": "T-GCN-MultiGSL-Mix",   "backbone": "tgcn", "adj_type": "lag_list",  "dagma": "multilag", "dagma_thr": 0.1},
    # GCN family
    "gcn_physical":  {"display": "GCN",                  "backbone": "gcn",  "adj_type": "single",    "dagma": None},
    "gcn_no_spatial":{"display": "GCN-NoSpatial",        "backbone": "gcn",  "adj_type": "identity",  "dagma": None},
    "gcn_gsl":       {"display": "GCN-GSL",              "backbone": "gcn",  "adj_type": "single",    "dagma": "contemporaneous", "dagma_thr": 0.3},
    "gcn_cgsl":      {"display": "GCN-cGSL",             "backbone": "gcn",  "adj_type": "single",    "dagma": "contemporaneous", "dagma_thr": 0.3, "symmetrize": True},
    "gcn_multigsl":  {"display": "GCN-MultiGSL",          "backbone": "gcn",  "adj_type": "single",    "dagma": "multilag_union", "dagma_thr": 0.1},
}

# ======================================================================
# Data loading
# ======================================================================
def load_data(dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]
    feat = np.array(pd.read_csv(PROJECT_ROOT / cfg["feat_path"]), dtype=np.float32)
    adj = np.array(pd.read_csv(PROJECT_ROOT / cfg["adj_path"], header=None), dtype=np.float32)
    T, N = feat.shape
    train_size = int(T * 0.8)
    feat_max = float(np.max(feat[:train_size]))
    return feat[:train_size] / feat_max, feat[train_size:] / feat_max, adj, feat_max


def generate_sequences(data, seq_len, pre_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pre_len):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len:i + seq_len + pre_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ======================================================================
# DAGMA graph loading (reuse existing outputs)
# ======================================================================
def load_multilag_graphs(dataset, ph, threshold=0.1, n_lags=3):
    """Load existing multi-lag DAGMA blocks and threshold them."""
    cfg = DATASET_CONFIGS[dataset]
    prefix = cfg["prefix"]
    lag_blocks_dir = PROJECT_ROOT / "results" / "stage26_validation"
    lag_list = []
    for l in range(1, n_lags + 1):
        path = lag_blocks_dir / f"{prefix}_ph{ph}_seed42_L{n_lags}_lag_{l}.npy"
        if not path.exists():
            return None  # missing
        W = np.load(path)
        A = binary_graph(W, threshold)
        lag_list.append(A)
    return lag_list


def load_contemporaneous_graph(dataset, ph, threshold=0.3):
    """Load existing contemporaneous single-graph DAGMA output."""
    cfg = DATASET_CONFIGS[dataset]
    prefix = cfg["prefix"]
    gsl_dir = PROJECT_ROOT / "results" / "stage33_gsl_canonical"
    A_path = gsl_dir / f"{prefix}_gsl_ph{ph}_seed42_A_binary.npy"
    if not A_path.exists():
        return None  # missing
    A = np.load(A_path).astype(np.float32)
    return A

# ======================================================================
# Model construction
# ======================================================================
def build_model_for_variant(variant_id, adj_phys, multilag_graphs, cgsl_graph,
                            N, seq_len=12, hidden_dim=64):
    """Construct the model for a given variant."""
    v = VARIANTS[variant_id]
    backbone = v["backbone"]
    adj_type = v["adj_type"]

    if adj_type == "identity":
        adj_input = np.eye(N, dtype=np.float32)
    elif adj_type == "single":
        if v.get("symmetrize") and cgsl_graph is not None:
            adj_input = cgsl_graph
        elif v["dagma"] == "contemporaneous":
            adj_input = cgsl_graph if v.get("symmetrize") else load_contemporaneous_graph_by_variant(variant_id, N)
            if adj_input is None:
                adj_input = adj_phys  # fallback
        else:
            adj_input = adj_phys
    elif adj_type == "lag_list":
        adj_input = multilag_graphs
    else:
        adj_input = adj_phys

    if backbone == "tgcn":
        if adj_type == "identity":
            return TGCN(adj=np.eye(N, dtype=np.float32), hidden_dim=hidden_dim)
        elif adj_type == "single":
            return TGCN(adj=adj_input, hidden_dim=hidden_dim)
        else:  # lag_list
            cls_map = {
                "multi_gsl": MultiGraphTGCNFixed,
                "multi_gsl_mix": GatedMultiGraphTGCN,
                "multi_gsl_weighted": WeightedMultiGraphTGCN,
            }
            return cls_map[variant_id](adj_list=adj_input, hidden_dim=hidden_dim)
    elif backbone == "gcn":
        if adj_type == "identity":
            return GCN(adj=np.eye(N, dtype=np.float32), seq_len=seq_len, hidden_dim=hidden_dim)
        elif adj_type == "single":
            return GCN(adj=adj_input, seq_len=seq_len, hidden_dim=hidden_dim)
    raise ValueError(f"Cannot build model for {variant_id}")


def load_contemporaneous_graph_by_variant(variant_id, N):
    """Load the contemporaneous GSL graph for a given dataset (infers from variant)."""
    # We need the dataset; this is called from a context where we know the dataset.
    # For now, return None and let the caller handle it.
    return None

# ======================================================================
# Training and evaluation
# ======================================================================
def train_and_eval(model, train_X, train_Y, test_X, test_Y,
                   feat_max, pre_len, seed, max_epochs, loss_name,
                   batch_size=128, lr=0.001, wd=0.0001, hidden_dim=64):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task = SupervisedForecastTask(
        model=model, loss=loss_name, pre_len=pre_len,
        learning_rate=lr, weight_decay=wd, feat_max_val=feat_max,
    )
    model = model.to(device)
    if task.regressor is not None:
        task.regressor = task.regressor.to(device)

    optimizer = task.configure_optimizer()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
        ),
        batch_size=batch_size, shuffle=True,
    )

    t0 = time.time()
    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = task.training_step((xb, yb))
            loss.backward()
            optimizer.step()
    train_time = time.time() - t0

    model.eval()
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
        ),
        batch_size=len(test_X), shuffle=False,
    )
    metrics = task.validation_epoch(test_loader, device)
    metrics["train_time_s"] = round(train_time, 2)
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    return metrics

# ======================================================================
# Result I/O (atomic writes)
# ======================================================================
def result_path(variant, dataset, ph, seed):
    return TRAINING_DIR / f"{dataset}_ph{ph}_seed{seed}_{variant}.json"


def result_exists(variant, dataset, ph, seed):
    p = result_path(variant, dataset, ph, seed)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        return data.get("status") == "complete"
    except Exception:
        return False


def save_result(result, variant, dataset, ph, seed):
    p = result_path(variant, dataset, ph, seed)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    tmp.rename(p)

# ======================================================================
# Single experiment
# ======================================================================
def run_experiment(variant, dataset, ph, seed, args, log_file=None):
    """Run one (variant, dataset, ph, seed) experiment."""
    if result_exists(variant, dataset, ph, seed):
        print(f"  [SKIP] {variant} {dataset} PH={ph} seed={seed} (already exists)")
        return "skipped"

    v = VARIANTS[variant]
    cfg = DATASET_CONFIGS[dataset]
    N = cfg["N"]
    backbone = v["backbone"]
    loss_name = "mse_with_regularizer" if backbone == "tgcn" else "mse"

    # Load data
    train_norm, test_norm, adj_phys, feat_max = load_data(dataset)
    train_X, train_Y = generate_sequences(train_norm, args.seq_len, ph)
    test_X, test_Y = generate_sequences(test_norm, args.seq_len, ph)

    # Load DAGMA graphs
    multilag_graphs = None
    contemporaneous_A = None
    if v["dagma"] in ("multilag", "multilag_union"):
        multilag_graphs = load_multilag_graphs(dataset, ph, v["dagma_thr"])
        if multilag_graphs is None:
            print(f"  [FAIL] {variant} {dataset} PH={ph} seed={seed}: multilag DAGMA blocks missing")
            return "missing_dagma"
    elif v["dagma"] == "contemporaneous":
        contemporaneous_A = load_contemporaneous_graph(dataset, ph, v["dagma_thr"])
        if contemporaneous_A is None:
            print(f"  [FAIL] {variant} {dataset} PH={ph} seed={seed}: contemporaneous DAGMA missing")
            return "missing_dagma"

    # Build adjacency for the variant
    if v["adj_type"] == "identity":
        adj_model = np.eye(N, dtype=np.float32)
    elif v["adj_type"] == "single":
        if v.get("symmetrize"):
            if contemporaneous_A is None:
                print(f"  [FAIL] {variant} {dataset} PH={ph}: cGSL needs contemporaneous DAGMA")
                return "missing_dagma"
            adj_model = contemporaneous_A + contemporaneous_A.T
            adj_model = (adj_model > 0).astype(np.float32)
            np.fill_diagonal(adj_model, 0)
        elif v["dagma"] == "contemporaneous":
            adj_model = contemporaneous_A if contemporaneous_A is not None else adj_phys
        elif v["dagma"] == "multilag_union":
            # GCN-MultiGSL: union of lag-specific graphs into single static graph
            adj_model = np.zeros((N, N), dtype=np.float32)
            for a in multilag_graphs:
                adj_model = np.maximum(adj_model, a)
        else:
            adj_model = adj_phys
    elif v["adj_type"] == "lag_list":
        adj_model = multilag_graphs
    else:
        adj_model = adj_phys

    # Build model
    if backbone == "tgcn":
        if v["adj_type"] == "identity":
            model = TGCN(adj=np.eye(N, dtype=np.float32), hidden_dim=args.hidden_dim)
        elif v["adj_type"] == "single":
            model = TGCN(adj=adj_model, hidden_dim=args.hidden_dim)
        else:
            cls_map = {
                "multi_gsl": MultiGraphTGCNFixed,
                "multi_gsl_mix": GatedMultiGraphTGCN,
                "multi_gsl_weighted": WeightedMultiGraphTGCN,
            }
            model = cls_map[variant](adj_list=adj_model, hidden_dim=args.hidden_dim)
    elif backbone == "gcn":
        if v["adj_type"] == "identity":
            model = GCN(adj=np.eye(N, dtype=np.float32), seq_len=args.seq_len, hidden_dim=args.hidden_dim)
        elif v["adj_type"] == "single":
            model = GCN(adj=adj_model, seq_len=args.seq_len, hidden_dim=args.hidden_dim)

    n_edges = int((np.asarray(adj_model) > 0).sum()) if adj_model is not None else N

    print(f"  [RUN] {v['display']} {dataset} PH={ph} seed={seed} "
          f"(edges={n_edges}, params={sum(p.numel() for p in model.parameters())})")

    metrics = train_and_eval(
        model, train_X, train_Y, test_X, test_Y,
        feat_max, ph, seed, args.max_epochs, loss_name,
        batch_size=args.batch_size, lr=args.lr, wd=args.wd,
        hidden_dim=args.hidden_dim,
    )

    result = {
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "variant": variant,
        "display_name": v["display"],
        "backbone": backbone,
        "dataset": dataset,
        "ph": ph,
        "seed": seed,
        "n_edges": n_edges,
        "n_params": metrics["n_params"],
        "rmse": round(metrics["RMSE"], 4),
        "mae": round(metrics["MAE"], 4),
        "train_time_s": metrics["train_time_s"],
        "protocol": {
            "batch_size": args.batch_size, "lr": args.lr, "wd": args.wd,
            "hidden_dim": args.hidden_dim, "seq_len": args.seq_len,
            "epochs": args.max_epochs, "loss": loss_name,
        },
    }
    save_result(result, variant, dataset, ph, seed)
    print(f"  [DONE] RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
          f"({metrics['train_time_s']}s)")
    return "done"

# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 40 canonical experiment launcher")
    parser.add_argument("--variants", type=str, nargs="*",
                        default=list(VARIANTS.keys()),
                        help="Variant IDs to run (default: all)")
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["tgcn", "gcn"],
                        help="Filter to one backbone family")
    parser.add_argument("--datasets", type=str, nargs="*",
                        default=["losloop", "shenzhen"])
    parser.add_argument("--phs", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0001)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiments without running them")
    args = parser.parse_args()

    # Filter by backbone if requested
    if args.backbone:
        args.variants = [v for v in args.variants
                         if VARIANTS[v]["backbone"] == args.backbone]

    print("=" * 78)
    print("STAGE 40 — CANONICAL EXPERIMENT PIPELINE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Variants: {args.variants}")
    print(f"Datasets: {args.datasets}, PHs: {args.phs}, Seeds: {args.seeds}")
    print(f"Epochs: {args.max_epochs}, Hidden: {args.hidden_dim}")
    print("=" * 78)

    # Count experiments
    n_total = len(args.variants) * len(args.datasets) * len(args.phs) * len(args.seeds)
    n_skipped = sum(1 for v in args.variants for ds in args.datasets
                    for ph in args.phs for seed in args.seeds
                    if result_exists(v, ds, ph, seed))
    n_to_run = n_total - n_skipped

    print(f"\nTotal experiments: {n_total}")
    print(f"Already complete:  {n_skipped}")
    print(f"To run:            {n_to_run}")

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for v in args.variants:
            for ds in args.datasets:
                for ph in args.phs:
                    for seed in args.seeds:
                        status = "EXISTS" if result_exists(v, ds, ph, seed) else "NEW"
                        print(f"  {VARIANTS[v]['display']:30s} {ds:12s} PH={ph} seed={seed}  [{status}]")
        return

    # Run experiments
    stats = {"done": 0, "skipped": 0, "failed": 0, "missing_dagma": 0}
    t0 = time.time()
    for v in args.variants:
        for ds in args.datasets:
            for ph in args.phs:
                for seed in args.seeds:
                    status = run_experiment(v, ds, ph, seed, args)
                    stats[status] = stats.get(status, 0) + 1

    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print(f"COMPLETE in {elapsed/60:.1f} min")
    print(f"  Run: {stats.get('done', 0)}, Skip: {stats.get('skipped', 0)}, "
          f"Fail: {stats.get('failed', 0)}, Missing DAGMA: {stats.get('missing_dagma', 0)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
