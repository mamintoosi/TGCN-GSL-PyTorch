#!/usr/bin/env python3
"""
DAGMA Threshold Sensitivity Forecasting Experiment
===================================================
TGCN-GSL-PyTorch — SZ-Taxi Dataset

Tests whether the DAGMA threshold (w_threshold) affects forecasting performance.

Graph source: results/dagma_fresh/sz_PH{1,2,3,4}_W.npy (fresh DAGMA output)
DAGMA execution: SKIPPED — using previously computed fresh W

The ONLY experimental variable is the DAGMA threshold.
All other parameters are preserved from the original paper configuration.
"""

import os, sys, time, json, csv, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.utils.data

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

warnings.filterwarnings("ignore")

# ── Reproducibility ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Seeds: Python={SEED}, NumPy={SEED}, PyTorch={SEED}, CUDA={SEED}")

# ── Configuration ───────────────────────────────────────────────
DATASET = "shenzhen"
N = 156  # number of sensors for SZ-Taxi
SEQ_LEN = 12
SPLIT_RATIO = 0.8
BATCH_SIZE = 64
MAX_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
LOSS = "mse"
GCN_HIDDEN_DIM = 100
TGCN_HIDDEN_DIM = 64

THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.30]
HORIZONS = [1, 2, 3, 4]
MODELS = ["GCN", "TGCN"]

FRESH_DIR = os.path.join(ROOT, "results", "dagma_fresh")
OUT_DIR = os.path.join(ROOT, "results", "dagma_fresh", "threshold_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load fresh DAGMA W matrices ─────────────────────────────────
print("\n" + "=" * 80)
print("LOADING FRESH DAGMA W MATRICES")
print("=" * 80)

W_fresh = {}
for ph in HORIZONS:
    path = os.path.join(FRESH_DIR, f"sz_PH{ph}_W.npy")
    W_fresh[ph] = np.load(path)
    print(f"  PH={ph}: shape={W_fresh[ph].shape}, range=[{W_fresh[ph].min():.6f}, {W_fresh[ph].max():.6f}]")

# ── Graph construction (matching original pipeline) ─────────────
def construct_graph(W, threshold, merge_ph=True):
    """
    Construct binary adjacency matrix from DAGMA W, matching original code.
    
    Original code (spatiotemporal_csv_data.py):
        W_est_all shape: (N, N, pre_len)
        W_est = np.any(W_est_all > 0, axis=2)  # merge PH slices
        adj = np.zeros(W_est.shape, dtype=int)
        adj[W_est > 0] = 1
    
    Our fresh W is already 2D per PH. For merge_ph=True, we stack all PH
    and apply np.any(axis=2) to reproduce the original behavior.
    """
    if merge_ph:
        # Stack all PH-specific W matrices into 3D: (N, N, num_PH)
        W_3d = np.stack([W_fresh[ph] for ph in HORIZONS], axis=2)
        # Apply threshold
        W_3d[np.abs(W_3d) < threshold] = 0
        # Reproduce original: np.any(W_est_all > 0, axis=2)
        W_est = np.any(W_3d > 0, axis=2)
    else:
        # PH-specific: use the W matrix for this PH directly
        W_thr = W.copy()
        W_thr[np.abs(W_thr) < threshold] = 0
        W_est = W_thr > 0
    
    # Convert to binary adjacency (original code)
    adj = np.zeros(W_est.shape, dtype=int)
    adj[W_est > 0] = 1
    np.fill_diagonal(adj, 0)  # ensure no self-loops
    
    return adj


def graph_stats(adj, threshold):
    """Compute graph statistics."""
    n_edges = int(np.sum(adj))
    n_nodes = adj.shape[0]
    out_deg = adj.sum(axis=1)
    in_deg = adj.sum(axis=0)
    n_active = int(np.sum((out_deg > 0) | (in_deg > 0)))
    n_isolated = n_nodes - n_active
    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
    
    return {
        "threshold": threshold,
        "num_edges": n_edges,
        "num_active_nodes": n_active,
        "num_isolated_nodes": n_isolated,
        "density": density,
        "max_out_degree": int(out_deg.max()),
        "max_in_degree": int(in_deg.max()),
    }


# ── Data loading (shared across experiments) ────────────────────
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

from utils.data.functions import load_features, generate_torch_datasets

FEAT_PATH = os.path.join(ROOT, "data", "sz_speed.csv")
feat = load_features(FEAT_PATH)
feat_max_val = np.max(feat)
print(f"  Features: shape={feat.shape}, max={feat_max_val:.4f}")

# Pre-compute datasets for each PH (data loading is the same, only pre_len differs)
datasets = {}
for ph in HORIZONS:
    train_ds, val_ds = generate_torch_datasets(
        feat, SEQ_LEN, ph, split_ratio=SPLIT_RATIO, normalize=True
    )
    datasets[ph] = (train_ds, val_ds)
    print(f"  PH={ph}: train={len(train_ds)}, val={len(val_ds)}")

# ── Import models ───────────────────────────────────────────────
from models import GCN, TGCN
from tasks.supervised import SupervisedForecastTask


def run_single_experiment(model_name, ph, threshold, adj, train_ds, val_ds):
    """Train and evaluate one model configuration."""
    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # Create model
    if model_name == "GCN":
        model = GCN(adj=adj, seq_len=SEQ_LEN, hidden_dim=GCN_HIDDEN_DIM)
    elif model_name == "TGCN":
        model = TGCN(adj=adj, hidden_dim=TGCN_HIDDEN_DIM)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(DEVICE)
    
    task = SupervisedForecastTask(
        model=model,
        loss=LOSS,
        pre_len=ph,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        feat_max_val=feat_max_val,
    )
    task.model.to(DEVICE)
    if task.regressor is not None:
        task.regressor.to(DEVICE)
    
    optimizer = task.configure_optimizer()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=len(val_ds))
    
    # Train
    t_start = time.time()
    best_val_rmse = float("inf")
    best_epoch = 0
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = task.training_step((x, y))
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        val_metrics = task.validation_epoch(val_loader, DEVICE)
        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_epoch = epoch + 1
    
    training_time = time.time() - t_start
    
    return {
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
        "final_val_rmse": val_metrics["RMSE"],
        "final_val_mae": val_metrics["MAE"],
        "final_val_r2": val_metrics["R2"],
        "training_time_sec": training_time,
    }


# ── Run experiments ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("RUNNING THRESHOLD SENSITIVITY EXPERIMENTS")
print(f"  Thresholds: {THRESHOLDS}")
print(f"  Horizons: {HORIZONS}")
print(f"  Models: {MODELS}")
print(f"  Epochs: {MAX_EPOCHS}")
print(f"  Estimated time: ~50 minutes")
print("=" * 80)

all_results = []
total_experiments = len(THRESHOLDS) * len(HORIZONS) * len(MODELS)
experiment_count = 0

csv_path = os.path.join(OUT_DIR, "threshold_forecasting_results.csv")
csv_fields = [
    "dataset", "model", "ph", "threshold", "graph_mode",
    "num_edges", "num_active_nodes", "num_isolated_nodes", "density",
    "best_val_rmse", "best_epoch", "final_val_rmse", "final_val_mae", "final_val_r2",
    "training_time_sec", "seed", "max_out_degree", "max_in_degree",
]

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()
    
    for threshold in THRESHOLDS:
        for ph in HORIZONS:
            # Construct graph using merged PH (original pipeline behavior)
            adj = construct_graph(W_fresh, threshold, merge_ph=True)
            stats = graph_stats(adj, threshold)
            
            print(f"\n--- Threshold={threshold}, PH={ph}, merged_graph ---")
            print(f"  Graph source: results/dagma_fresh/sz_PH{{1,2,3,4}}_W.npy")
            print(f"  DAGMA execution: SKIPPED — using previously computed fresh W")
            print(f"  Graph: {stats['num_edges']} edges, {stats['num_active_nodes']} active nodes, "
                  f"{stats['num_isolated_nodes']} isolated, density={stats['density']:.8f}")
            
            for model_name in MODELS:
                experiment_count += 1
                print(f"  [{experiment_count}/{total_experiments}] {model_name} PH={ph} ...", end=" ", flush=True)
                
                result = run_single_experiment(
                    model_name, ph, threshold, adj,
                    datasets[ph][0], datasets[ph][1]
                )
                
                row = {
                    "dataset": DATASET,
                    "model": model_name,
                    "ph": ph,
                    "threshold": threshold,
                    "graph_mode": "merged",
                    "seed": SEED,
                    **stats,
                    **result,
                }
                writer.writerow(row)
                all_results.append(row)
                csvfile.flush()
                
                print(f"RMSE={result['best_val_rmse']:.6f} "
                      f"(epoch={result['best_epoch']}, "
                      f"time={result['training_time_sec']:.1f}s)")

# ── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

df = pd.DataFrame(all_results)

# Pivot table: model × threshold × PH → RMSE
for model_name in MODELS:
    print(f"\n--- {model_name} ---")
    model_df = df[df["model"] == model_name]
    
    # Table: PH × threshold → RMSE
    print(f"\n  Best Validation RMSE:")
    header = f"  {'PH':>4}"
    for t in THRESHOLDS:
        header += f"  {t:>8}"
    print(header)
    print("  " + "-" * (6 + 10 * len(THRESHOLDS)))
    
    for ph in HORIZONS:
        row_str = f"  {ph:>4}"
        ph_df = model_df[model_df["ph"] == ph]
        for t in THRESHOLDS:
            match = ph_df[ph_df["threshold"] == t]
            if len(match) > 0:
                rmse = match.iloc[0]["best_val_rmse"]
                row_str += f"  {rmse:>8.4f}"
            else:
                row_str += f"  {'N/A':>8}"
        print(row_str)
    
    # Average across PH
    print(f"\n  Average RMSE across PH:")
    avg_row = f"  {'AVG':>4}"
    for t in THRESHOLDS:
        t_df = model_df[model_df["threshold"] == t]
        if len(t_df) > 0:
            avg_rmse = t_df["best_val_rmse"].mean()
            avg_row += f"  {avg_rmse:>8.4f}"
        else:
            avg_row += f"  {'N/A':>8}"
    print(avg_row)
    
    # Best threshold
    avg_by_threshold = model_df.groupby("threshold")["best_val_rmse"].mean()
    best_t = avg_by_threshold.idxmin()
    best_rmse = avg_by_threshold.min()
    print(f"\n  Best threshold (by avg RMSE): {best_t} (RMSE={best_rmse:.6f})")
    
    # Compare with threshold=0.3
    if 0.3 in avg_by_threshold.index:
        rmse_03 = avg_by_threshold[0.3]
        improvement = (rmse_03 - best_rmse) / rmse_03 * 100
        print(f"  vs threshold=0.3 (RMSE={rmse_03:.6f}): {improvement:+.2f}% {'improvement' if improvement > 0 else 'degradation'}")

# Save CSV
print(f"\nResults saved to: {csv_path}")

# ── Analysis ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

for model_name in MODELS:
    model_df = df[df["model"] == model_name]
    avg_by_t = model_df.groupby("threshold")["best_val_rmse"].mean()
    
    print(f"\n--- {model_name} ---")
    print(f"  Q1: Does lowering threshold from 0.3 improve performance?")
    best_t = avg_by_t.idxmin()
    if best_t < 0.3:
        print(f"    YES — best threshold={best_t} outperforms 0.3")
    elif best_t == 0.3:
        print(f"    NO — threshold=0.3 is already optimal")
    else:
        print(f"    UNEXPECTED — best threshold={best_t}")
    
    print(f"\n  Q2-Q3: Best threshold by RMSE: {best_t} ({avg_by_t[best_t]:.6f})")
    
    print(f"\n  Q6: Is 8-edge graph (thr=0.3) hurting?")
    if 0.3 in avg_by_t.index and best_t < 0.3:
        pct = (avg_by_t[0.3] - avg_by_t[best_t]) / avg_by_t[0.3] * 100
        print(f"    YES — lowering to {best_t} improves by {pct:.2f}%")
    else:
        print(f"    NO — 8-edge graph performs as well or better")

# Save summary JSON
summary = {
    "experiment": "DAGMA threshold sensitivity",
    "dataset": DATASET,
    "seed": SEED,
    "thresholds_tested": THRESHOLDS,
    "horizons_tested": HORIZONS,
    "models_tested": MODELS,
    "epochs": MAX_EPOCHS,
    "graph_mode": "merged (np.any across PH, matching original pipeline)",
    "graph_source": "results/dagma_fresh/sz_PH{1,2,3,4}_W.npy",
    "dAGMA_rerun": False,
}

for model_name in MODELS:
    model_df = df[df["model"] == model_name]
    avg_by_t = model_df.groupby("threshold")["best_val_rmse"].mean()
    summary[model_name] = {
        "avg_rmse_by_threshold": {str(t): float(v) for t, v in avg_by_t.items()},
        "best_threshold": float(avg_by_t.idxmin()),
        "best_avg_rmse": float(avg_by_t.min()),
    }

json_path = os.path.join(OUT_DIR, "threshold_forecasting_summary.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to: {json_path}")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)
