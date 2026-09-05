#!/usr/bin/env python
"""
Direct ablation runner — imports modules directly instead of subprocess.
Much faster than subprocess approach.
"""

import sys
import os
import yaml
import time
import json
import csv
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import GCN, TGCN
from tasks.supervised import SupervisedForecastTask
from utils.data.spatiotemporal_csv_data import SpatioTemporalCSVData
from utils.data.functions import load_features
from utils.metrics import accuracy, r2, explained_variance
import torchmetrics

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Configuration
EXPERIMENTS = []
MODELS = [("GCN", 100), ("TGCN", 100)]
DATASETS = [("shenzhen", "SZ-Taxi"), ("losloop", "Los-loop")]
PRE_LENS = [1, 2, 3, 4]
GRAPH_TYPES = [
    (0, "physical", "Physical graph baseline"),
    (3, "sparse_random", "Sparse random (same edge count as GSL)"),
    (1, "gsl", "DAGMA GSL"),
    (2, "dcg", "DAGMA cGSL (symmetrized)"),
]

for dataset_key, dataset_label in DATASETS:
    for model_name, hidden_dim in MODELS:
        for pre_len in PRE_LENS:
            for gsl_val, gsl_label, gsl_desc in GRAPH_TYPES:
                EXPERIMENTS.append({
                    "dataset": dataset_key,
                    "dataset_label": dataset_label,
                    "model": model_name,
                    "hidden_dim": hidden_dim,
                    "pre_len": pre_len,
                    "gsl": gsl_val,
                    "gsl_label": gsl_label,
                    "gsl_desc": gsl_desc,
                })

print(f"Total experiments: {len(EXPERIMENTS)}")


def train_and_evaluate(exp, num_epochs=50, batch_size=64, lr=0.001, device="cuda"):
    """Run a single training + evaluation experiment."""
    # Reset seeds for fair comparison
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load data
    data_module = SpatioTemporalCSVData(
        dataset_name=exp["dataset"],
        seq_len=12,
        pre_len=exp["pre_len"],
        split_ratio=0.8,
        normalize=True,
        use_gsl=exp["gsl"],
    )

    train_dataset, val_dataset = data_module.get_datasets()
    if data_module.use_gsl > 0 and data_module.use_gsl != 3:
        data_module.compute_adjacency_matrix()
    elif data_module.use_gsl == 3:
        # Already loaded in __init__ via the new code path
        pass

    adj = data_module.adj
    num_nodes = data_module.num_nodes
    feat_max_val = data_module.feat_max_val

    # Create model
    if exp["model"] == "GCN":
        model = GCN(adj=adj, hidden_dim=exp["hidden_dim"], seq_len=12)
    elif exp["model"] == "TGCN":
        model = TGCN(adj=adj, hidden_dim=exp["hidden_dim"], seq_len=12)

    model = model.to(device)

    # Create task
    model_task = SupervisedForecastTask(
        model=model,
        loss="mse",
        pre_len=exp["pre_len"],
        learning_rate=lr,
        weight_decay=0,
        feat_max_val=feat_max_val,
    )
    model_task.model.to(device)
    if model_task.regressor is not None:
        model_task.regressor.to(device)

    optimizer = model_task.configure_optimizer()

    # Training loop
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False
    )

    for epoch in range(num_epochs):
        model_task.model.train()
        if model_task.regressor is not None:
            model_task.regressor.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model_task.training_step((x, y))
            loss.backward()
            optimizer.step()

    # Evaluate
    model_task.model.eval()
    if model_task.regressor is not None:
        model_task.regressor.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            predictions = model_task.forward(x)
            predictions = predictions * feat_max_val
            y = y * feat_max_val
            preds_2d = predictions.transpose(1, 2).reshape((-1, x.size(2)))
            y_2d = y.reshape((-1, y.size(2)))

            rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(preds_2d, y_2d)).item()
            mae = torchmetrics.functional.mean_absolute_error(preds_2d, y_2d).item()
            acc = metrics.accuracy(preds_2d, y_2d).item()
            r2_val = metrics.r2(preds_2d, y_2d).item()

    return {"RMSE": rmse, "MAE": mae, "Accuracy": acc, "R2": r2_val}


from utils import metrics

# Run experiments
results = []
total = len(EXPERIMENTS)

for i, exp in enumerate(EXPERIMENTS):
    label = f"{exp['dataset_label']} {exp['model']} PH={exp['pre_len']} [{exp['gsl_label']}]"
    print(f"\n[{i+1}/{total}] {label}", flush=True)

    t0 = time.time()
    try:
        m = train_and_evaluate(exp, num_epochs=50, device="cuda")
        m["elapsed"] = time.time() - t0
        m["success"] = True
        print(f"  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  Acc={m['Accuracy']:.4f}  R2={m['R2']:.4f}  ({m['elapsed']:.1f}s)")
    except Exception as e:
        m = {"success": False, "error": str(e), "elapsed": time.time() - t0}
        print(f"  FAILED: {e}")

    results.append({
        "dataset": exp["dataset_label"],
        "model": exp["model"],
        "pre_len": exp["pre_len"],
        "graph_type": exp["gsl_label"],
        "graph_desc": exp["gsl_desc"],
        **m,
    })

    # Save intermediate results
    with open("doc/ablation/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

# Print summary
print("\n" + "=" * 100)
print("ABLATION RESULTS SUMMARY")
print("=" * 100)
print(f"{'Dataset':<12} {'Model':<6} {'PH':<4} {'Graph':<16} {'RMSE':>8} {'MAE':>8} {'Acc':>8} {'R2':>8}")
print("-" * 100)
for row in results:
    if row.get("success"):
        print(f"{row['dataset']:<12} {row['model']:<6} {row['pre_len']:<4} "
              f"{row['graph_type']:<16} {row['RMSE']:>8.4f} {row['MAE']:>8.4f} "
              f"{row['Accuracy']:>8.4f} {row['R2']:>8.4f}")
    else:
        print(f"{row['dataset']:<12} {row['model']:<6} {row['pre_len']:<4} "
              f"{row['graph_type']:<16} {'FAILED':>8}")

# Save CSV
csv_path = "doc/ablation/ablation_results.csv"
if results:
    keys = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

print(f"\nResults saved to doc/ablation/ablation_results.json and .csv")
