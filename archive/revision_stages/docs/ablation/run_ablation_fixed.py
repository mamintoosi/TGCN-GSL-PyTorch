#!/usr/bin/env python
"""
Sparsified-Physical-Graph Ablation — FIXED version.

Purpose: Determine whether the forecasting improvement from GSL/cGSL comes from
learned topology or simply from graph sparsification (reduced oversmoothing).

Experiment:
  - Physical graph:    full road-network adjacency (532 edges for SZ, 2833 for LA)
  - Sparse random:     random graph with SAME edge count as GSL (8 for SZ, 28 for LA)
  - GSL:               DAGMA-learned directed graph
  - cGSL:              DAGMA-learned, symmetrized (directed cyclic graph)

If sparse_random ≈ physical, the sparsification itself doesn't matter.
If sparse_random ≈ GSL, the topology doesn't matter — only the edge count.
If sparse_random is between physical and GSL, both factors contribute.

Usage:
    cd /data/git/mamintoosi/TGCN-GSL-PyTorch
    /data/python-envs/pytorch/bin/python doc/ablation/run_ablation_fixed.py

    # Or run specific subset:
    /data/python-envs/pytorch/bin/python doc/ablation/run_ablation_fixed.py --dataset shenzhen --model GCN
    /data/python-envs/pytorch/bin/python doc/ablation/run_ablation_fixed.py --dataset losloop --model TGCN

Estimated runtime: ~20-30 min total on RTX 3090 (all 64 experiments)
"""

import sys
import os
import time
import json
import csv
import random
import argparse
import numpy as np
import torch

# Ensure we can import from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models import GCN, TGCN
from tasks.supervised import SupervisedForecastTask
from utils.data.spatiotemporal_csv_data import SpatioTemporalCSVData
from utils import metrics
import torchmetrics

# ─── Configuration ────────────────────────────────────────────────────────────
MODELS = [("GCN", 100), ("TGCN", 100)]
DATASETS = [("shenzhen", "SZ-Taxi"), ("losloop", "Los-loop")]
PRE_LENS = [1, 2, 3, 4]
GRAPH_TYPES = [
    (0, "physical",      "Physical graph baseline"),
    (3, "sparse_random", "Sparse random (same edge count as GSL)"),
    (1, "gsl",           "DAGMA GSL"),
    (2, "dcg",           "DAGMA cGSL (symmetrized)"),
]
NUM_EPOCHS = 50
BATCH_SIZE = 64
LR = 0.001
SEED = 42

# Output file
RESULTS_FILE = os.path.join(PROJECT_ROOT, "doc", "ablation", "ablation_results_fixed.json")
CSV_FILE = os.path.join(PROJECT_ROOT, "doc", "ablation", "ablation_results_fixed.csv")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_experiment_list(datasets=None, models=None, pre_lens=None):
    """Build experiment list, optionally filtered."""
    exps = []
    for dk, dl in datasets or DATASETS:
        for mn, hd in models or MODELS:
            for pl in pre_lens or PRE_LENS:
                for gv, gl, gd in GRAPH_TYPES:
                    exps.append({
                        "dataset": dk, "dataset_label": dl,
                        "model": mn, "hidden_dim": hd,
                        "pre_len": pl,
                        "gsl": gv, "gsl_label": gl, "gsl_desc": gd,
                    })
    return exps


def train_and_eval(exp, device="cuda"):
    """Train a model and evaluate on validation set."""
    set_seed(SEED)

    # ─── Load data ───
    data_module = SpatioTemporalCSVData(
        dataset_name=exp["dataset"],
        seq_len=12,
        pre_len=exp["pre_len"],
        split_ratio=0.8,
        normalize=True,
        use_gsl=exp["gsl"],
    )
    train_dataset, val_dataset = data_module.get_datasets()

    # CRITICAL FIX: always call compute_adjacency_matrix for ALL graph types
    # use_gsl=3 (sparse random) is also handled inside this method
    data_module.compute_adjacency_matrix()

    adj = data_module.adj
    num_nodes = data_module.num_nodes
    feat_max_val = data_module.feat_max_val

    # ─── Create model ───
    if exp["model"] == "GCN":
        model = GCN(adj=adj, hidden_dim=exp["hidden_dim"], seq_len=12)
    else:
        model = TGCN(adj=adj, hidden_dim=exp["hidden_dim"], seq_len=12)
    model = model.to(device)

    # ─── Create task ───
    task = SupervisedForecastTask(
        model=model, loss="mse",
        pre_len=exp["pre_len"], learning_rate=LR, weight_decay=0,
        feat_max_val=feat_max_val,
    )
    task.model.to(device)
    if task.regressor is not None:
        task.regressor.to(device)
    optimizer = task.configure_optimizer()

    # ─── Train ───
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False,
    )

    for epoch in range(NUM_EPOCHS):
        task.model.train()
        if task.regressor is not None:
            task.regressor.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = task.training_step((x, y))
            loss.backward()
            optimizer.step()

    # ─── Evaluate ───
    task.model.eval()
    if task.regressor is not None:
        task.regressor.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = task.forward(x)
            preds = preds * feat_max_val
            y = y * feat_max_val
            p2d = preds.transpose(1, 2).reshape((-1, x.size(2)))
            y2d = y.reshape((-1, y.size(2)))

            rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(p2d, y2d)).item()
            mae = torchmetrics.functional.mean_absolute_error(p2d, y2d).item()
            acc = metrics.accuracy(p2d, y2d).item()
            r2_val = metrics.r2(p2d, y2d).item()

    return {"RMSE": rmse, "MAE": mae, "Accuracy": acc, "R2": r2_val}


def main():
    parser = argparse.ArgumentParser(description="Sparsified-physical-graph ablation")
    parser.add_argument("--dataset", choices=["shenzhen", "losloop"], default=None,
                        help="Run only this dataset")
    parser.add_argument("--model", choices=["GCN", "TGCN"], default=None,
                        help="Run only this model")
    parser.add_argument("--pre-len", type=int, nargs="+", default=None,
                        help="Run only these prediction horizons")
    args = parser.parse_args()

    # Filter experiments
    datasets = [d for d in DATASETS if args.dataset is None or d[0] == args.dataset]
    models = [m for m in MODELS if args.model is None or m[0] == args.model]
    pre_lens = args.pre_len or PRE_LENS

    experiments = build_experiment_list(datasets, models, pre_lens)
    print(f"Running {len(experiments)} experiments")
    print(f"  Datasets: {[d[1] for d in datasets]}")
    print(f"  Models:   {[m[0] for m in models]}")
    print(f"  Horizons: {pre_lens}")
    print(f"  Graph types: {[g[1] for g in GRAPH_TYPES]}")
    print(f"  Epochs: {NUM_EPOCHS}, Seed: {SEED}")
    print()

    # Load existing results if any (for resuming)
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {RESULTS_FILE}")

    # Determine which experiments are already done
    done_keys = set()
    for r in results:
        if r.get("success"):
            key = (r["dataset"], r["model"], r["pre_len"], r["graph_type"])
            done_keys.add(key)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    total = len(experiments)
    skipped = 0
    for i, exp in enumerate(experiments):
        key = (exp["dataset_label"], exp["model"], exp["pre_len"], exp["gsl_label"])
        label = f"{exp['dataset_label']} {exp['model']} PH={exp['pre_len']} [{exp['gsl_label']}]"

        if key in done_keys:
            skipped += 1
            print(f"[{i+1}/{total}] {label} — SKIP (already done)")
            continue

        print(f"[{i+1}/{total}] {label} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            m = train_and_eval(exp, device=device)
            m["elapsed"] = time.time() - t0
            m["success"] = True
            print(f"RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} Acc={m['Accuracy']:.4f} R2={m['R2']:.4f}  ({m['elapsed']:.1f}s)")
        except Exception as e:
            m = {"success": False, "error": str(e), "elapsed": time.time() - t0}
            print(f"FAILED: {e}")

        results.append({
            "dataset": exp["dataset_label"],
            "model": exp["model"],
            "pre_len": exp["pre_len"],
            "graph_type": exp["gsl_label"],
            "graph_desc": exp["gsl_desc"],
            **m,
        })

        # Save after each experiment (for resume support)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSkipped {skipped} already-completed experiments.")
    print(f"Completed {len(experiments) - skipped} new experiments.")

    # ─── Print summary table ───
    print("\n" + "=" * 110)
    print("ABLATION RESULTS: Physical vs Sparse-Random vs GSL vs cGSL")
    print("=" * 110)
    print(f"{'Dataset':<12} {'Model':<6} {'PH':<4} {'Graph':<16} {'RMSE':>8} {'MAE':>8} {'Acc':>8} {'R2':>8} {'Time':>6}")
    print("-" * 110)

    for r in results:
        if r.get("success"):
            print(f"{r['dataset']:<12} {r['model']:<6} {r['pre_len']:<4} "
                  f"{r['graph_type']:<16} {r['RMSE']:>8.4f} {r['MAE']:>8.4f} "
                  f"{r['Accuracy']:>8.4f} {r['R2']:>8.4f} {r.get('elapsed',0):>5.0f}s")
        else:
            print(f"{r['dataset']:<12} {r['model']:<6} {r['pre_len']:<4} "
                  f"{r['graph_type']:<16} {'FAILED':>8}")

    # ─── Comparison summary ───
    print("\n" + "=" * 110)
    print("KEY COMPARISON: Does sparse random match GSL? (If yes → sparsification, not topology)")
    print("=" * 110)
    for r in results:
        if not r.get("success"):
            continue
        if r["graph_type"] == "sparse_random":
            # Find GSL and physical
            gsl_r = [x for x in results if x["dataset"] == r["dataset"] and x["model"] == r["model"]
                     and x["pre_len"] == r["pre_len"] and x["graph_type"] == "gsl"]
            phys_r = [x for x in results if x["dataset"] == r["dataset"] and x["model"] == r["model"]
                      and x["pre_len"] == r["pre_len"] and x["graph_type"] == "physical"]
            if gsl_r and phys_r:
                g = gsl_r[0]
                p = phys_r[0]
                rmse_diff_vs_gsl = abs(r["RMSE"] - g["RMSE"])
                rmse_diff_vs_phys = abs(r["RMSE"] - p["RMSE"])
                closer = "GSL" if rmse_diff_vs_gsl < rmse_diff_vs_phys else "Physical"
                print(f"  {r['dataset']:<12} {r['model']:<6} PH={r['pre_len']}: "
                      f"sparse_random={r['RMSE']:.4f} | GSL={g['RMSE']:.4f} | phys={p['RMSE']:.4f} "
                      f"| closer_to={closer}")

    # ─── Save CSV ───
    if results:
        keys = list(results[0].keys())
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    print(f"\nResults saved to:\n  {RESULTS_FILE}\n  {CSV_FILE}")


if __name__ == "__main__":
    main()
