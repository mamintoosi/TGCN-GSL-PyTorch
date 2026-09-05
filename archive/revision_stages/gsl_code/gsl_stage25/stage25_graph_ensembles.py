#!/usr/bin/env python3
"""
Stage 25 — Experiment Families C + D: Graph Ensembles and Physical-DAGMA Fusion.

Constructs and evaluates:
  C1. Union of PH-specific graphs
  C2. Intersection of PH-specific graphs
  C3. Frequency-based graph (edge in >= K/4 PHs)
  C4. Weighted ensemble of PH graphs
  D1. Physical only
  D2. DAGMA only
  D3. Physical AND DAGMA (intersection)
  D4. Weighted fusion: alpha * Physical + (1-alpha) * DAGMA
  D5. Physical OR DAGMA (union)

Uses existing DAGMA matrices. No new DAGMA computation.

Usage:
  python gsl_stage25/stage25_graph_ensembles.py
  python gsl_stage25/stage25_graph_ensembles.py --dataset losloop
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.gcn import GCN
from models.tgcn import TGCN
from tasks.supervised import SupervisedForecastTask

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage25_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_CONFIGS = {
    "shenzhen": {
        "feat_path": "data/sz_speed.csv",
        "adj_path": "data/sz_adj.csv",
        "N": 156, "prefix": "sz",
    },
    "losloop": {
        "feat_path": "data/los_speed.csv",
        "adj_path": "data/los_adj.csv",
        "N": 207, "prefix": "los",
    },
}


def load_data(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    feat = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, config["feat_path"])), dtype=np.float32)
    adj = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, config["adj_path"]), header=None), dtype=np.float32)
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_eval(adj, model_name, train_X, train_Y, test_X, test_Y,
                   feat_max, pre_len, seed=42, max_epochs=50):
    set_seed(seed)
    if model_name == "GCN":
        model = GCN(adj=adj, seq_len=12, hidden_dim=64)
        loss_name = "mse"
    else:
        model = TGCN(adj=adj, hidden_dim=64)
        loss_name = "mse_with_regularizer"

    task = SupervisedForecastTask(model=model, loss=loss_name, pre_len=pre_len,
                                   learning_rate=0.001, weight_decay=0.0001,
                                   feat_max_val=feat_max)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if task.regressor is not None:
        task.regressor = task.regressor.to(device)

    optimizer = task.configure_optimizer()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(train_X), torch.FloatTensor(train_Y)),
        batch_size=128, shuffle=True)

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
        torch.utils.data.TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_Y)),
        batch_size=len(test_X), shuffle=False)
    metrics = task.validation_epoch(test_loader, device)
    metrics["train_time_s"] = round(train_time, 2)
    return metrics


def load_W(dataset, ph, seed=42):
    prefix = DATASET_CONFIGS[dataset]["prefix"]
    N = DATASET_CONFIGS[dataset]["N"]
    path = os.path.join(PROJECT_ROOT, "results", "stage24_validation",
                        f"{prefix}_ph{ph}_seed{seed}_W_raw_temporal.npy")
    if not os.path.exists(path):
        return None
    W_raw = np.load(path)
    return W_raw[:N, N:2*N]  # Correct block: past -> current


def binary_graph(W, threshold):
    adj = (np.abs(W) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)
    return adj


def top_k_graph(W, k):
    N = W.shape[0]
    W_abs = np.abs(W.copy())
    np.fill_diagonal(W_abs, 0)
    adj = np.zeros_like(W_abs)
    flat = W_abs.flatten()
    top_idx = np.argsort(flat)[::-1][:k]
    adj.flat[top_idx] = 1.0
    return adj


# ============================================================
# Family C: Multi-PH Graph Ensembles
# ============================================================
def build_ensemble_graphs(W_ph_dict, threshold=0.1):
    """Build ensemble graphs from multiple PH-specific DAGMA matrices."""
    binary = {ph: binary_graph(W, threshold) for ph, W in W_ph_dict.items()}
    phs = sorted(binary.keys())
    N = binary[phs[0]].shape[0]

    results = {}

    # C1: Union
    union = np.zeros((N, N), dtype=np.float32)
    for ph in phs:
        union = np.maximum(union, binary[ph])
    results["ensemble_union"] = union

    # C2: Intersection
    intersection = np.ones((N, N), dtype=np.float32)
    for ph in phs:
        intersection *= binary[ph]
    results["ensemble_intersection"] = intersection

    # C3: Frequency graphs
    freq = np.zeros((N, N), dtype=np.float32)
    for ph in phs:
        freq += binary[ph]
    for min_freq in [2, 3, 4]:
        results[f"ensemble_freq_geq{min_freq}"] = (freq >= min_freq).astype(np.float32)

    # C4: Weighted ensemble (equal weights)
    weighted = np.zeros((N, N), dtype=np.float32)
    for ph in phs:
        weighted += np.abs(W_ph_dict[ph])
    weighted /= len(phs)
    for thr in [0.01, 0.05, 0.1]:
        results[f"ensemble_weighted_thr{thr}"] = binary_graph(W=weighted, threshold=thr) if False else \
            ((weighted > thr).astype(np.float32))

    return results


# ============================================================
# Family D: Physical-DAGMA Fusion
# ============================================================
def build_fusion_graphs(adj_phys, W_temporal, threshold=0.1):
    """Build fusion graphs combining physical and temporal DAGMA."""
    N = adj_phys.shape[0]
    adj_dagma = binary_graph(W_temporal, threshold)

    # Make physical binary if not already
    adj_phys_bin = (adj_phys > 0).astype(np.float32)
    np.fill_diagonal(adj_phys_bin, 0)

    results = {}

    # D3: Intersection
    results["fusion_intersection"] = adj_phys_bin * adj_dagma

    # D4: Weighted fusion
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        fused = alpha * adj_phys_bin + (1 - alpha) * adj_dagma
        results[f"fusion_alpha{alpha:.1f}"] = fused

    # D5: Union
    results["fusion_union"] = np.maximum(adj_phys_bin, adj_dagma)

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 25: Graph Ensembles & Fusion")
    parser.add_argument("--dataset", type=str, default="shenzhen", choices=["shenzhen", "losloop"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--ph", type=int, default=1, help="PH for fusion experiments")
    args = parser.parse_args()

    dataset = args.dataset
    seed = args.seed
    config = DATASET_CONFIGS[dataset]
    N = config["N"]
    prefix = config["prefix"]

    print("=" * 80)
    print(f"STAGE 25 — GRAPH ENSEMBLES & FUSION ({dataset}, seed={seed})")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load data
    train_data, test_data, adj_phys, feat_max = load_data(dataset)

    # Load all PH DAGMA matrices
    W_ph = {}
    for ph in [1, 2, 3, 4]:
        W = load_W(dataset, ph, seed)
        if W is not None:
            W_ph[ph] = W
    print(f"\nLoaded DAGMA matrices for PHs: {sorted(W_ph.keys())}")

    if len(W_ph) < 2:
        print("ERROR: Need at least 2 PH DAGMA matrices for ensemble analysis.")
        return

    all_results = []

    # ============================================================
    # Family C: Multi-PH Ensembles (at PH=args.ph)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"FAMILY C: Multi-PH Graph Ensembles (evaluated at PH={args.ph})")
    print(f"{'='*60}")

    train_X, train_Y = generate_sequences(train_data, 12, args.ph)
    test_X, test_Y = generate_sequences(test_data, 12, args.ph)

    ensembles = build_ensemble_graphs(W_ph, threshold=0.1)

    for ens_name, adj_ens in ensembles.items():
        n_edges = int(np.sum(adj_ens > 0))
        for model_name in ["GCN", "TGCN"]:
            m = train_and_eval(adj_ens, model_name, train_X, train_Y,
                               test_X, test_Y, feat_max, args.ph, seed=seed,
                               max_epochs=args.max_epochs)
            all_results.append({
                "dataset": dataset, "ph": args.ph, "method": ens_name,
                "model": model_name, "n_edges": n_edges,
                "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                "r2": round(m["R2"], 6), "family": "C_ensemble",
            })
        print(f"  {ens_name:35s}: {n_edges:6d} edges")

    # ============================================================
    # Baselines for comparison
    # ============================================================
    baselines = {"NoGraph": np.eye(N, dtype=np.float32), "Physical": adj_phys}

    # Correlation graphs
    corr = np.corrcoef(train_data.T)
    corr = np.nan_to_num(corr, nan=0.0)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0)
    upper = np.triu_indices(N, k=1)

    for k_name, k_val in [("Corr-K8", 8), ("Corr-K16", 16)]:
        adj = np.zeros((N, N), dtype=np.float32)
        sorted_idx = np.argsort(abs_corr[upper])[::-1]
        for i in range(min(k_val, len(sorted_idx))):
            r, c = upper[0][sorted_idx[i]], upper[1][sorted_idx[i]]
            adj[r, c] = 1.0
            adj[c, r] = 1.0
        baselines[k_name] = adj

    # Temp DAGMA at threshold 0.1
    W_ref = W_ph[args.ph]
    adj_dagma_01 = binary_graph(W_ref, 0.1)
    baselines["TempDAGMA_0.1"] = adj_dagma_01

    print(f"\n{'='*60}")
    print("BASELINES")
    print(f"{'='*60}")

    for bname, badj in baselines.items():
        n_edges = int(np.sum(badj > 0))
        for model_name in ["GCN", "TGCN"]:
            m = train_and_eval(badj, model_name, train_X, train_Y,
                               test_X, test_Y, feat_max, args.ph, seed=seed,
                               max_epochs=args.max_epochs)
            all_results.append({
                "dataset": dataset, "ph": args.ph, "method": bname,
                "model": model_name, "n_edges": n_edges,
                "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                "r2": round(m["R2"], 6), "family": "baseline",
            })
        print(f"  {bname:35s}: {n_edges:6d} edges")

    # ============================================================
    # Family D: Physical-DAGMA Fusion
    # ============================================================
    print(f"\n{'='*60}")
    print(f"FAMILY D: Physical-DAGMA Fusion (at PH={args.ph})")
    print(f"{'='*60}")

    fusions = build_fusion_graphs(adj_phys, W_ref, threshold=0.1)

    for fus_name, adj_fus in fusions.items():
        n_edges = int(np.sum(adj_fus > 0))
        for model_name in ["GCN", "TGCN"]:
            m = train_and_eval(adj_fus, model_name, train_X, train_Y,
                               test_X, test_Y, feat_max, args.ph, seed=seed,
                               max_epochs=args.max_epochs)
            all_results.append({
                "dataset": dataset, "ph": args.ph, "method": fus_name,
                "model": model_name, "n_edges": n_edges,
                "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                "r2": round(m["R2"], 6), "family": "D_fusion",
            })
        print(f"  {fus_name:35s}: {n_edges:6d} edges")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*100}")
    print(f"STAGE 25 ENSEMBLE/FUSION SUMMARY ({dataset}, PH={args.ph}, seed={seed})")
    print(f"{'='*100}")
    print(f"{'Method':35s} | {'Model':5s} | {'Edges':>6s} | {'RMSE':>8s} | {'MAE':>8s}")
    print("-" * 85)
    for r in sorted(all_results, key=lambda x: x["rmse"]):
        print(f"{r['method']:35s} | {r['model']:5s} | {r['n_edges']:6d} | {r['rmse']:8.4f} | {r['mae']:8.4f}")

    # Save
    csv_path = os.path.join(RESULTS_DIR, f"stage25_ensembles_{prefix}_ph{args.ph}_seed{seed}.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    print(f"Total time: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
