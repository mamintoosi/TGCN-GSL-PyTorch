#!/usr/bin/env python3
"""
Stage 26 Validation — Targeted experiments for GatedMultiGraphTGCN.

Three experiment families:
  A. Multi-seed validation (seeds 42-46, Los-loop, PH=1)
  B. Parameter-matched NoGraph control (hidden_dim=74 vs 64)
  C. Lag ablation (which lags contribute?)

All use EXISTING DAGMA matrices — no DAGMA recomputation.

Usage:
  python gsl_stage26/stage26_validation.py --experiment A
  python gsl_stage26/stage26_validation.py --experiment B
  python gsl_stage26/stage26_validation.py --experiment C
  python gsl_stage26/stage26_validation.py --experiment all
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from datetime import datetime
from itertools import combinations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.graph_conv import calculate_laplacian_with_self_loop
from tasks.supervised import SupervisedForecastTask
from models.tgcn import TGCN

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage26_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_CONFIGS = {
    "losloop": {
        "feat_path": "data/los_speed.csv",
        "adj_path": "data/los_adj.csv",
        "N": 207, "prefix": "los",
    },
}


# ============================================================
# DATA LOADING (identical to stage26_evaluate.py)
# ============================================================
def load_data(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    feat = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, config["feat_path"])),
                    dtype=np.float32)
    adj = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, config["adj_path"]),
                               header=None), dtype=np.float32)
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


# ============================================================
# GRAPH UTILITIES
# ============================================================
def binary_graph(W, threshold):
    adj = (np.abs(W) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)
    return adj


def load_multilag_blocks(dataset, ph, seed=42, n_lags=3):
    prefix = DATASET_CONFIGS[dataset]["prefix"]
    results_dir = os.path.join(PROJECT_ROOT, "results", "stage26_validation")
    lag_blocks = {}
    for lag_label in [f"lag_{l}" for l in range(1, n_lags + 1)] + ["current"]:
        path = os.path.join(results_dir,
                             f"{prefix}_ph{ph}_seed{seed}_L{n_lags}_{lag_label}.npy")
        if os.path.exists(path):
            lag_blocks[lag_label] = np.load(path)
    return lag_blocks if lag_blocks else None


# ============================================================
# MODELS — same as stage26_evaluate.py
# ============================================================
class GatedMultiGraphTGCN(nn.Module):
    """Per-node, per-timestep adaptive graph selection (unchanged from Stage 26)."""
    def __init__(self, adj_list, hidden_dim=64, **kwargs):
        super().__init__()
        self._input_dim = adj_list[0].shape[0]
        self._hidden_dim = hidden_dim
        self._n_graphs = len(adj_list)
        laps = [calculate_laplacian_with_self_loop(torch.FloatTensor(adj)) for adj in adj_list]
        self.register_buffer("lap_stack", torch.stack(laps))
        self.gate_net = nn.Sequential(
            nn.Linear(1 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self._n_graphs),
        )
        self.W_z = nn.Linear(1 + hidden_dim, hidden_dim * 2)
        self.W_n = nn.Linear(1 + hidden_dim, hidden_dim)

    def forward(self, inputs):
        B, T, N = inputs.shape
        h = torch.zeros(B, N * self._hidden_dim, device=inputs.device, dtype=inputs.dtype)
        for t in range(T):
            x = inputs[:, t, :].reshape(B, N, 1)
            hh = h.reshape(B, N, self._hidden_dim)
            gate_input = torch.cat([x, hh], dim=2)
            gate_logits = self.gate_net(gate_input)
            gate_w = F.softmax(gate_logits, dim=-1)
            adj_weighted = torch.einsum('bnk,kij->bnj', gate_w, self.lap_stack)
            gh = torch.cat([x, hh], dim=2)
            ag = torch.bmm(adj_weighted, gh)
            z = torch.sigmoid(self.W_z(ag))
            r, u = torch.chunk(z, chunks=2, dim=2)
            c = torch.tanh(self.W_n(torch.cat([x, r * hh], dim=2)))
            h = u * hh + (1 - u) * c
        return h.reshape(B, N, self._hidden_dim)

    @property
    def hyperparameters(self):
        return {"hidden_dim": self._hidden_dim}


class MultiGraphTGCNFixed(nn.Module):
    """
    MultiGraphTGCN with CORRECTED graph-timestep alignment.

    FIXED MAPPING:
      input step t (0=most recent) -> lag graph for temporal gap (seq_len - 1 - t)
      Since we have lag_1, lag_2, lag_3:
        gap=1 -> lag_1 (index 0)
        gap=2 -> lag_2 (index 1)
        gap=3 -> lag_3 (index 2)
        gap>3 -> cycles: index = (gap - 1) % n_graphs

    This is the CORRECTED version replacing the buggy t % n_graphs.
    """
    def __init__(self, adj_list, hidden_dim=64, seq_len=12, **kwargs):
        super().__init__()
        self._input_dim = adj_list[0].shape[0]
        self._hidden_dim = hidden_dim
        self._n_graphs = len(adj_list)
        self._seq_len = seq_len
        laps = [calculate_laplacian_with_self_loop(torch.FloatTensor(adj)) for adj in adj_list]
        for i, lap in enumerate(laps):
            self.register_buffer(f"lap_{i}", lap)
        self.W_z = nn.Linear(1 + hidden_dim, hidden_dim * 2)
        self.W_n = nn.Linear(1 + hidden_dim, hidden_dim)

    def _graph_conv(self, lap, x):
        B, N, D = x.shape
        x_flat = x.permute(1, 2, 0).reshape(N, D * B)
        out = lap @ x_flat
        return out.reshape(N, D, B).permute(2, 0, 1)

    def forward(self, inputs):
        B, T, N = inputs.shape
        h = torch.zeros(B, N * self._hidden_dim, device=inputs.device, dtype=inputs.dtype)
        for t in range(T):
            # CORRECTED: map input step to lag graph
            temporal_gap = (T - 1) - t  # gap from current: 0,1,2,...,T-1
            graph_idx = (temporal_gap) % self._n_graphs  # 0->lag_1, 1->lag_2, 2->lag_3, 3->lag_1,...
            lap = getattr(self, f"lap_{graph_idx}")
            x = inputs[:, t, :].reshape(B, N, 1)
            hh = h.reshape(B, N, self._hidden_dim)
            gh = self._graph_conv(lap, torch.cat([x, hh], dim=2))
            z = torch.sigmoid(self.W_z(gh))
            r, u = torch.chunk(z, chunks=2, dim=2)
            c = torch.tanh(self.W_n(torch.cat([x, r * hh], dim=2)))
            h = u * hh + (1 - u) * c
        return h.reshape(B, N, self._hidden_dim)

    @property
    def hyperparameters(self):
        return {"hidden_dim": self._hidden_dim}


# ============================================================
# TRAINING AND EVALUATION
# ============================================================
def train_and_eval(adj_or_model_factory, model_type, train_X, train_Y, test_X, test_Y,
                   feat_max, pre_len, seed=42, max_epochs=50, hidden_dim=64):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "standard":
        model = TGCN(adj=adj_or_model_factory, hidden_dim=hidden_dim)
    elif model_type == "gated_multi":
        model = GatedMultiGraphTGCN(adj_list=adj_or_model_factory, hidden_dim=hidden_dim)
    elif model_type == "multi_graph_fixed":
        model = MultiGraphTGCNFixed(adj_list=adj_or_model_factory, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    task = SupervisedForecastTask(
        model=model, loss="mse_with_regularizer", pre_len=pre_len,
        learning_rate=0.001, weight_decay=0.0001, feat_max_val=feat_max,
    )
    model = model.to(device)
    if task.regressor is not None:
        task.regressor = task.regressor.to(device)

    optimizer = task.configure_optimizer()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
        ),
        batch_size=128, shuffle=True,
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


# ============================================================
# EXPERIMENT A: Multi-seed validation
# ============================================================
def run_experiment_a(dataset, ph, seeds, n_lags, threshold, max_epochs):
    print("\n" + "=" * 80)
    print("EXPERIMENT A — Multi-Seed Validation")
    print(f"Dataset: {dataset}, PH={ph}, Seeds: {seeds}")
    print("=" * 80)

    train_data, test_data, adj_phys, feat_max = load_data(dataset)
    train_X, train_Y = generate_sequences(train_data, 12, ph)
    test_X, test_Y = generate_sequences(test_data, 12, ph)
    N = DATASET_CONFIGS[dataset]["N"]
    print(f"Train: {train_X.shape}, Test: {test_X.shape}")

    # Load DAGMA lag blocks (fixed across seeds)
    lag_blocks = load_multilag_blocks(dataset, ph, seed=42, n_lags=n_lags)
    if lag_blocks is None:
        print("ERROR: No DAGMA blocks found.")
        return
    lag_keys = sorted([k for k in lag_blocks if k.startswith("lag_")],
                       key=lambda x: int(x.split("_")[1]))
    adj_list = [binary_graph(lag_blocks[k], threshold) for k in lag_keys]
    total_edges = sum(int(a.sum()) for a in adj_list)
    print(f"Lag graphs: {len(adj_list)} graphs, {total_edges} total edges")
    for k, a in zip(lag_keys, adj_list):
        print(f"  {k}: {int(a.sum())} edges")

    all_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # NoGraph
        adj_nograph = np.eye(N, dtype=np.float32)
        m = train_and_eval(adj_nograph, "standard", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, max_epochs)
        all_results.append({
            "experiment": "A_multiseed", "dataset": dataset, "ph": ph,
            "seed": seed, "method": "NoGraph", "model": "TGCN",
            "n_edges": N, "rmse": round(m["RMSE"], 4),
            "mae": round(m["MAE"], 4), "n_params": m["n_params"],
        })
        print(f"  NoGraph:       RMSE={m['RMSE']:.4f}  params={m['n_params']}")

        # MultiGraphTGCN (corrected alignment)
        m = train_and_eval(adj_list, "multi_graph_fixed", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, max_epochs)
        all_results.append({
            "experiment": "A_multiseed", "dataset": dataset, "ph": ph,
            "seed": seed, "method": "MultiGraphTGCN_fixed", "model": "MultiGraphTGCNFixed",
            "n_edges": total_edges, "rmse": round(m["RMSE"], 4),
            "mae": round(m["MAE"], 4), "n_params": m["n_params"],
        })
        print(f"  MultiGraph:    RMSE={m['RMSE']:.4f}  params={m['n_params']}")

        # GatedMultiGraphTGCN
        m = train_and_eval(adj_list, "gated_multi", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, max_epochs)
        all_results.append({
            "experiment": "A_multiseed", "dataset": dataset, "ph": ph,
            "seed": seed, "method": "GatedMultiGraphTGCN", "model": "GatedMultiGraphTGCN",
            "n_edges": total_edges, "rmse": round(m["RMSE"], 4),
            "mae": round(m["MAE"], 4), "n_params": m["n_params"],
        })
        print(f"  GatedMulti:    RMSE={m['RMSE']:.4f}  params={m['n_params']}")

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT A SUMMARY")
    print("=" * 80)
    for method in ["NoGraph", "MultiGraphTGCN_fixed", "GatedMultiGraphTGCN"]:
        rmses = [r["rmse"] for r in all_results if r["method"] == method]
        if rmses:
            print(f"  {method:25s}: mean={np.mean(rmses):.4f}  std={np.std(rmses):.4f}  "
                  f"min={np.min(rmses):.4f}  max={np.max(rmses):.4f}  (n={len(rmses)})")

    # Save
    csv_path = os.path.join(RESULTS_DIR, f"stage26_validation_A_{dataset}_ph{ph}.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    return all_results


# ============================================================
# EXPERIMENT B: Parameter-matched NoGraph control
# ============================================================
def run_experiment_b(dataset, ph, seed, n_lags, threshold, max_epochs):
    print("\n" + "=" * 80)
    print("EXPERIMENT B — Parameter-Matched NoGraph Control")
    print(f"Dataset: {dataset}, PH={ph}, Seed: {seed}")
    print("=" * 80)

    train_data, test_data, adj_phys, feat_max = load_data(dataset)
    train_X, train_Y = generate_sequences(train_data, 12, ph)
    test_X, test_Y = generate_sequences(test_data, 12, ph)
    N = DATASET_CONFIGS[dataset]["N"]

    # Load DAGMA lag blocks
    lag_blocks = load_multilag_blocks(dataset, ph, seed=42, n_lags=n_lags)
    lag_keys = sorted([k for k in lag_blocks if k.startswith("lag_")],
                       key=lambda x: int(x.split("_")[1]))
    adj_list = [binary_graph(lag_blocks[k], threshold) for k in lag_keys]

    all_results = []

    # Standard NoGraph (hidden_dim=64)
    adj_nograph = np.eye(N, dtype=np.float32)
    m = train_and_eval(adj_nograph, "standard", train_X, train_Y,
                       test_X, test_Y, feat_max, ph, seed, max_epochs, hidden_dim=64)
    all_results.append({
        "experiment": "B_param_match", "dataset": dataset, "ph": ph,
        "seed": seed, "method": "NoGraph_h64", "model": "TGCN",
        "hidden_dim": 64, "n_edges": N,
        "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
        "n_params": m["n_params"],
    })
    print(f"  NoGraph (h=64):       RMSE={m['RMSE']:.4f}  params={m['n_params']}")

    # Parameter-matched NoGraph (hidden_dim=74, ~16872 params)
    m = train_and_eval(adj_nograph, "standard", train_X, train_Y,
                       test_X, test_Y, feat_max, ph, seed, max_epochs, hidden_dim=74)
    all_results.append({
        "experiment": "B_param_match", "dataset": dataset, "ph": ph,
        "seed": seed, "method": "NoGraph_h74", "model": "TGCN",
        "hidden_dim": 74, "n_edges": N,
        "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
        "n_params": m["n_params"],
    })
    print(f"  NoGraph (h=74):       RMSE={m['RMSE']:.4f}  params={m['n_params']}")

    # GatedMultiGraphTGCN (hidden_dim=64, ~17091 params)
    m = train_and_eval(adj_list, "gated_multi", train_X, train_Y,
                       test_X, test_Y, feat_max, ph, seed, max_epochs, hidden_dim=64)
    all_results.append({
        "experiment": "B_param_match", "dataset": dataset, "ph": ph,
        "seed": seed, "method": "GatedMultiGraphTGCN", "model": "GatedMultiGraphTGCN",
        "hidden_dim": 64, "n_edges": sum(int(a.sum()) for a in adj_list),
        "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
        "n_params": m["n_params"],
    })
    print(f"  GatedMulti (h=64):    RMSE={m['RMSE']:.4f}  params={m['n_params']}")

    # Summary
    print("\n  Parameter comparison:")
    for r in all_results:
        print(f"    {r['method']:25s}: {r['n_params']:6d} params, RMSE={r['rmse']:.4f}")

    csv_path = os.path.join(RESULTS_DIR, f"stage26_validation_B_{dataset}_ph{ph}.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    return all_results


# ============================================================
# EXPERIMENT C: Lag ablation
# ============================================================
def run_experiment_c(dataset, ph, seed, n_lags, threshold, max_epochs):
    print("\n" + "=" * 80)
    print("EXPERIMENT C — Lag Ablation")
    print(f"Dataset: {dataset}, PH={ph}, Seed: {seed}")
    print("=" * 80)

    train_data, test_data, adj_phys, feat_max = load_data(dataset)
    train_X, train_Y = generate_sequences(train_data, 12, ph)
    test_X, test_Y = generate_sequences(test_data, 12, ph)
    N = DATASET_CONFIGS[dataset]["N"]

    # Load DAGMA lag blocks
    lag_blocks = load_multilag_blocks(dataset, ph, seed=42, n_lags=n_lags)
    lag_keys = sorted([k for k in lag_blocks if k.startswith("lag_")],
                       key=lambda x: int(x.split("_")[1]))
    lag_graphs = {k: binary_graph(lag_blocks[k], threshold) for k in lag_keys}

    # Also need NoGraph baseline
    adj_nograph = np.eye(N, dtype=np.float32)

    all_results = []

    # Baseline: NoGraph
    m = train_and_eval(adj_nograph, "standard", train_X, train_Y,
                       test_X, test_Y, feat_max, ph, seed, max_epochs)
    all_results.append({
        "experiment": "C_lag_ablation", "dataset": dataset, "ph": ph,
        "seed": seed, "method": "NoGraph", "model": "TGCN",
        "n_edges": N, "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
        "n_params": m["n_params"], "lags_used": "none",
    })
    print(f"  NoGraph:  RMSE={m['RMSE']:.4f}")

    # All single lags
    for lag_name in lag_keys:
        adj_subset = [lag_graphs[lag_name]]
        n_e = int(adj_subset[0].sum())
        m = train_and_eval(adj_subset, "gated_multi", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, max_epochs)
        all_results.append({
            "experiment": "C_lag_ablation", "dataset": dataset, "ph": ph,
            "seed": seed, "method": f"GatedMulti_{lag_name}", "model": "GatedMultiGraphTGCN",
            "n_edges": n_e, "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
            "n_params": m["n_params"], "lags_used": lag_name,
        })
        print(f"  {lag_name} only:  RMSE={m['RMSE']:.4f}  edges={n_e}")

    # All 2-lag combinations
    for combo in combinations(lag_keys, 2):
        adj_subset = [lag_graphs[k] for k in combo]
        n_e = sum(int(a.sum()) for a in adj_subset)
        combo_name = "+".join(combo)
        m = train_and_eval(adj_subset, "gated_multi", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, max_epochs)
        all_results.append({
            "experiment": "C_lag_ablation", "dataset": dataset, "ph": ph,
            "seed": seed, "method": f"GatedMulti_{combo_name}", "model": "GatedMultiGraphTGCN",
            "n_edges": n_e, "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
            "n_params": m["n_params"], "lags_used": combo_name,
        })
        print(f"  {combo_name}:  RMSE={m['RMSE']:.4f}  edges={n_e}")

    # All 3 lags
    adj_all = [lag_graphs[k] for k in lag_keys]
    n_e = sum(int(a.sum()) for a in adj_all)
    m = train_and_eval(adj_all, "gated_multi", train_X, train_Y,
                       test_X, test_Y, feat_max, ph, seed, max_epochs)
    all_results.append({
        "experiment": "C_lag_ablation", "dataset": dataset, "ph": ph,
        "seed": seed, "method": "GatedMulti_all", "model": "GatedMultiGraphTGCN",
        "n_edges": n_e, "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
        "n_params": m["n_params"], "lags_used": "lag_1+lag_2+lag_3",
    })
    print(f"  all lags:  RMSE={m['RMSE']:.4f}  edges={n_e}")

    # Summary table sorted by RMSE
    print("\n  Lag ablation summary (sorted by RMSE):")
    for r in sorted(all_results, key=lambda x: x["rmse"]):
        print(f"    {r['method']:30s}: RMSE={r['rmse']:.4f}  edges={r['n_edges']:4d}  lags={r['lags_used']}")

    csv_path = os.path.join(RESULTS_DIR, f"stage26_validation_C_{dataset}_ph{ph}.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    return all_results


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 26 Validation Experiments")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["A", "B", "C", "all"],
                        help="Experiment to run")
    parser.add_argument("--dataset", type=str, default="losloop",
                        choices=["losloop"])
    parser.add_argument("--ph", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46",
                        help="Comma-separated seeds for Experiment A")
    parser.add_argument("--n-lags", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=50)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 80)
    print("STAGE 26 VALIDATION EXPERIMENTS")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiment: {args.experiment}")
    print("=" * 80)

    all_results = []

    if args.experiment in ("A", "all"):
        results = run_experiment_a(
            args.dataset, args.ph, seeds, args.n_lags,
            args.threshold, args.max_epochs)
        if results:
            all_results.extend(results)

    if args.experiment in ("B", "all"):
        results = run_experiment_b(
            args.dataset, args.ph, args.seed, args.n_lags,
            args.threshold, args.max_epochs)
        if results:
            all_results.extend(results)

    if args.experiment in ("C", "all"):
        results = run_experiment_c(
            args.dataset, args.ph, args.seed, args.n_lags,
            args.threshold, args.max_epochs)
        if results:
            all_results.extend(results)

    # Save combined results
    if all_results:
        json_path = os.path.join(RESULTS_DIR,
                                  f"stage26_validation_{args.experiment}_{args.dataset}_ph{args.ph}.json")
        with open(json_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "experiment": args.experiment,
                "dataset": args.dataset,
                "ph": args.ph,
                "results": all_results,
            }, f, indent=2, default=str)
        print(f"\nCombined results saved: {json_path}")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
