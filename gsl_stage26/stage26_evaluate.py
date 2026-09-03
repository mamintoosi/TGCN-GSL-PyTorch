#!/usr/bin/env python3
"""
Stage 26 — Multi-Lag Forecasting Evaluation.

Evaluates the following methods on traffic forecasting:

BASELINES:
  A. NoGraph         — TGCN/GCN with identity adjacency (no spatial info)
  B. Physical         — TGCN/GCN with physical road network adjacency
  C. SingleDAGMA      — TGCN/GCN with existing Stage 24 temporal DAGMA (2-lag, PH-specific)

MULTI-LAG METHODS:
  D. UnionGraph       — Union of lag-specific DAGMA graphs into one adjacency
  E. MultiGraphTGCN   — Different adjacency per timestep (t-l uses A_l)
  F. WeightedMultiGraph — A = sum(alpha_l * A_l), weights learned end-to-end
  G. GatedMultiGraph   — Per-step adaptive gate over lag-specific graphs
  H. Corr-K8 / Corr-K16 — Correlation-based graphs (sparsity controls)

Also includes:
  I. AggregatedDAG    — Mean of lag-specific absolute weights, thresholded

For each method, reports:
  - Number of nodes, edges, density
  - RMSE, MAE
  - Graph threshold (where applicable)

Usage:
  python gsl_stage26/stage26_evaluate.py --dataset shenzhen --ph 1
  python gsl_stage26/stage26_evaluate.py --dataset losloop --ph 1
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.graph_conv import calculate_laplacian_with_self_loop
from tasks.supervised import SupervisedForecastTask
from models.tgcn import TGCN

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage26_validation")
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


# ============================================================
# DATA LOADING
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


def top_k_graph(W, k):
    N = W.shape[0]
    W_abs = np.abs(W.copy())
    np.fill_diagonal(W_abs, 0)
    adj = np.zeros_like(W_abs)
    flat_idx = np.argsort(W_abs.ravel())[::-1][:k]
    adj.flat[flat_idx] = 1.0
    return adj


# ============================================================
# MULTI-LAG DAGMA LOADING
# ============================================================
def load_multilag_blocks(dataset, ph, seed=42, n_lags=3):
    """Load pre-extracted lag-specific DAGMA blocks."""
    prefix = DATASET_CONFIGS[dataset]["prefix"]
    results_dir = os.path.join(PROJECT_ROOT, "results", "stage26_validation")

    lag_blocks = {}
    for lag_label in [f"lag_{l}" for l in range(1, n_lags + 1)] + ["current"]:
        path = os.path.join(results_dir,
                             f"{prefix}_ph{ph}_seed{seed}_L{n_lags}_{lag_label}.npy")
        if os.path.exists(path):
            lag_blocks[lag_label] = np.load(path)

    return lag_blocks if lag_blocks else None


def load_single_lag_dagma(dataset, ph, seed=42):
    """Load the existing Stage 24 single-lag temporal DAGMA (2-lag formulation)."""
    prefix = DATASET_CONFIGS[dataset]["prefix"]
    N = DATASET_CONFIGS[dataset]["N"]
    path = os.path.join(PROJECT_ROOT, "results", "stage24_validation",
                         f"{prefix}_ph{ph}_seed{seed}_W_raw_temporal.npy")
    if not os.path.exists(path):
        # Fallback
        path = os.path.join(PROJECT_ROOT, "results", "stage24_validation",
                             f"{prefix}_ph{ph}_W_raw_temporal.npy")
    if os.path.exists(path):
        W_raw = np.load(path)
        return W_raw[:N, N:2*N]  # Correct block: past -> current
    return None


# ============================================================
# MODEL: MultiGraphTGCN — different adjacency per timestep
# ============================================================
class MultiGraphTGCNCell(nn.Module):
    """
    TGCN cell that uses a different adjacency matrix at each input timestep.

    At timestep i in the input sequence, applies:
        A_i = laplacians[i % len(laplacians)]

    This allows the model to use lag-specific dependency graphs:
        - timestep 0: uses A_0 (most recent lag)
        - timestep 1: uses A_1 (lag-2)
        - etc.
    """
    def __init__(self, adj_list, input_dim, hidden_dim):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._n_graphs = len(adj_list)

        # Precompute Laplacians for each adjacency
        laps = []
        for adj in adj_list:
            laps.append(calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        # Register as buffer list
        self.register_buffer("lap_0", laps[0])
        for i, lap in enumerate(laps[1:], 1):
            self.register_buffer(f"lap_{i}", lap)

        # Shared weights
        self.graph_conv1_W = nn.Linear(1 + hidden_dim, hidden_dim * 2)
        self.graph_conv2_W = nn.Linear(1 + hidden_dim, hidden_dim)

    def _graph_conv(self, lap, x):
        """x: (B, N, D) -> (B, N, D)"""
        B, N, D = x.shape
        x_flat = x.permute(1, 2, 0).reshape(N, D * B)
        out = lap @ x_flat
        return out.reshape(N, D, B).permute(2, 0, 1)

    def forward(self, inputs, hidden_state, graph_idx):
        B, N = inputs.shape
        h = hidden_state.reshape(B, N, self._hidden_dim)
        x = inputs.reshape(B, N, 1)

        # Select graph
        lap = getattr(self, f"lap_{graph_idx % self._n_graphs}")

        # Graph convolution
        gh = self._graph_conv(lap, torch.cat([x, h], dim=2))

        # GRU-like update
        z = torch.sigmoid(self.graph_conv1_W(gh))
        r, u = torch.chunk(z, chunks=2, dim=2)
        c = torch.tanh(self.graph_conv2_W(torch.cat([x, r * h], dim=2)))
        new_h = u * h + (1 - u) * c

        return new_h.reshape(B, N * self._hidden_dim), new_h.reshape(B, N * self._hidden_dim)


class MultiGraphTGCN(nn.Module):
    """
    TGCN that applies a different adjacency at each input timestep.

    adj_list[0] is used for the most recent input,
    adj_list[1] for the next, etc. Cyclic if fewer graphs than timesteps.
    """
    def __init__(self, adj_list, hidden_dim=64, **kwargs):
        super().__init__()
        self._input_dim = adj_list[0].shape[0]
        self._hidden_dim = hidden_dim
        self._n_graphs = len(adj_list)
        self.cell = MultiGraphTGCNCell(adj_list, self._input_dim, hidden_dim)

    def forward(self, inputs):
        # inputs: (B, seq_len, N)
        B, T, N = inputs.shape
        h = torch.zeros(B, N * self._hidden_dim, device=inputs.device, dtype=inputs.dtype)
        for t in range(T):
            # Most recent input (t=0) uses adj_list[0], etc.
            graph_idx = t % self._n_graphs
            out, h = self.cell(inputs[:, t, :], h, graph_idx)
        return out.reshape(B, N, self._hidden_dim)

    @property
    def hyperparameters(self):
        return {"hidden_dim": self._hidden_dim}


# ============================================================
# MODEL: WeightedMultiGraphTGCN — learnable weighted combination
# ============================================================
class WeightedMultiGraphTGCN(nn.Module):
    """
    TGCN that learns a weighted combination of lag-specific graphs.

    A = softmax(w) . [A_1, A_2, ..., A_L]

    The weights w are learnable scalars (one per lag graph).
    """
    def __init__(self, adj_list, hidden_dim=64, **kwargs):
        super().__init__()
        self._input_dim = adj_list[0].shape[0]
        self._hidden_dim = hidden_dim
        self._n_graphs = len(adj_list)

        # Precompute Laplacians
        laps = []
        for adj in adj_list:
            laps.append(calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        # Register as stack
        self.register_buffer("lap_stack", torch.stack(laps))  # (K, N, N)

        # Learnable weights for combining graphs
        self.log_weights = nn.Parameter(torch.zeros(self._n_graphs))

        # Shared TGCN-like components
        self.W_z = nn.Linear(1 + hidden_dim, hidden_dim * 2)
        self.W_n = nn.Linear(1 + hidden_dim, hidden_dim)

    def forward(self, inputs):
        B, T, N = inputs.shape
        h = torch.zeros(B, N * self._hidden_dim, device=inputs.device, dtype=inputs.dtype)

        # Compute weighted adjacency: (N, N)
        w = torch.softmax(self.log_weights, dim=0)  # (K,)
        lap = torch.einsum('k,kij->ij', w, self.lap_stack)  # (N, N)

        for t in range(T):
            x = inputs[:, t, :].reshape(B, N, 1)  # (B, N, 1)
            hh = h.reshape(B, N, self._hidden_dim)  # (B, N, H)

            # Graph conv
            gh = torch.cat([x, hh], dim=2)  # (B, N, 1+H)
            D_B = (1 + self._hidden_dim) * B
            gh_flat = gh.permute(1, 2, 0).reshape(N, D_B)
            ag = lap @ gh_flat  # (N, D_B)
            ag = ag.reshape(N, 1 + self._hidden_dim, B).permute(2, 0, 1)  # (B, N, 1+H)

            # GRU
            z = torch.sigmoid(self.W_z(ag))
            r, u = torch.chunk(z, chunks=2, dim=2)
            c = torch.tanh(self.W_n(torch.cat([x, r * hh], dim=2)))
            h = u * hh + (1 - u) * c

        return h.reshape(B, N, self._hidden_dim)

    @property
    def hyperparameters(self):
        return {"hidden_dim": self._hidden_dim}

    def get_graph_weights(self):
        return torch.softmax(self.log_weights, dim=0).detach().cpu().numpy()


# ============================================================
# MODEL: GatedMultiGraphTGCN — adaptive gate per timestep
# ============================================================
class GatedMultiGraphTGCN(nn.Module):
    """
    TGCN that adaptively selects which lag-specific graph to use at each step.

    At each timestep, a gate network decides the blend of lag graphs
    based on input + hidden state.
    """
    def __init__(self, adj_list, hidden_dim=64, **kwargs):
        super().__init__()
        self._input_dim = adj_list[0].shape[0]
        self._hidden_dim = hidden_dim
        self._n_graphs = len(adj_list)

        # Precompute Laplacians
        laps = []
        for adj in adj_list:
            laps.append(calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self.register_buffer("lap_stack", torch.stack(laps))  # (K, N, N)

        # Gate: decides weight for each graph based on [x, h]
        self.gate_net = nn.Sequential(
            nn.Linear(1 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self._n_graphs),
        )

        # GRU components
        self.W_z = nn.Linear(1 + hidden_dim, hidden_dim * 2)
        self.W_n = nn.Linear(1 + hidden_dim, hidden_dim)

    def forward(self, inputs):
        B, T, N = inputs.shape
        h = torch.zeros(B, N * self._hidden_dim, device=inputs.device, dtype=inputs.dtype)

        for t in range(T):
            x = inputs[:, t, :].reshape(B, N, 1)
            hh = h.reshape(B, N, self._hidden_dim)

            # Per-node gate
            gate_input = torch.cat([x, hh], dim=2)  # (B, N, 1+H)
            gate_logits = self.gate_net(gate_input)  # (B, N, K)
            gate_w = F.softmax(gate_logits, dim=-1)  # (B, N, K)

            # Weighted adjacency per sample and node
            # lap_stack: (K, N, N)
            # gate_w: (B, N, K)
            # We want: for each batch b, node j, the weighted sum over k:
            #   adj_weighted[j, i] = sum_k gate_w[b, j, k] * lap_stack[k, j, i]
            # This means: adj_weighted = gate_w @ lap_stack  -> (B, N, N)
            adj_weighted = torch.einsum('bnk,kij->bnj', gate_w, self.lap_stack)  # (B, N, N)

            # Graph conv
            gh = torch.cat([x, hh], dim=2)  # (B, N, 1+H)
            D_B = (1 + self._hidden_dim) * B
            gh_flat = gh.permute(1, 2, 0).reshape(N, D_B)

            # This is more complex because adj_weighted varies per batch element
            # For efficiency, iterate over batch (or use einsum)
            ag_list = []
            for b in range(B):
                ag_b = adj_weighted[b] @ gh_flat  # Would need per-batch gh
            # Simplify: use batched matmul
            gh_batched = gh  # (B, N, 1+H)
            # adj_weighted: (B, N, N), gh_batched: (B, N, 1+H)
            # ag = adj_weighted @ gh_batched -> (B, N, 1+H)
            ag = torch.bmm(adj_weighted, gh_batched)  # (B, N, 1+H)

            # GRU
            z = torch.sigmoid(self.W_z(ag))
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
                   feat_max, pre_len, seed=42, max_epochs=50):
    """
    Train and evaluate a model.

    model_type: 'standard' | 'multi_graph' | 'weighted_multi' | 'gated_multi'
    adj_or_model_factory: either an adjacency matrix or a callable that returns a model
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "standard":
        adj = adj_or_model_factory
        model = TGCN(adj=adj, hidden_dim=64)
        loss_name = "mse_with_regularizer"
    elif model_type == "multi_graph":
        adj_list = adj_or_model_factory
        model = MultiGraphTGCN(adj_list=adj_list, hidden_dim=64)
        loss_name = "mse_with_regularizer"
    elif model_type == "weighted_multi":
        adj_list = adj_or_model_factory
        model = WeightedMultiGraphTGCN(adj_list=adj_list, hidden_dim=64)
        loss_name = "mse_with_regularizer"
    elif model_type == "gated_multi":
        adj_list = adj_or_model_factory
        model = GatedMultiGraphTGCN(adj_list=adj_list, hidden_dim=64)
        loss_name = "mse_with_regularizer"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    task = SupervisedForecastTask(
        model=model, loss=loss_name, pre_len=pre_len,
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

    # For weighted model, report learned weights
    if model_type == "weighted_multi" and hasattr(model, "get_graph_weights"):
        metrics["graph_weights"] = model.get_graph_weights().tolist()

    return metrics


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 26: Multi-Lag Evaluation")
    parser.add_argument("--dataset", type=str, default="shenzhen",
                        choices=["shenzhen", "losloop"])
    parser.add_argument("--ph", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--n-lags", type=int, default=3,
                        help="Number of lags for multi-lag DAGMA")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Threshold for binary graph construction")
    args = parser.parse_args()

    dataset = args.dataset
    seed = args.seed
    ph = args.ph
    n_lags = args.n_lags
    thr = args.threshold
    config = DATASET_CONFIGS[dataset]
    N = config["N"]
    prefix = config["prefix"]

    print("=" * 80)
    print(f"STAGE 26 — MULTI-LAG FORECASTING EVALUATION")
    print(f"Dataset: {dataset} (N={N}), PH={ph}, seed={seed}")
    print(f"Lags: {n_lags}, Threshold: {thr}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load data
    train_data, test_data, adj_phys, feat_max = load_data(dataset)
    train_X, train_Y = generate_sequences(train_data, 12, ph)
    test_X, test_Y = generate_sequences(test_data, 12, ph)
    print(f"Train: {train_X.shape}, Test: {test_X.shape}")

    # Load multi-lag DAGMA blocks
    lag_blocks = load_multilag_blocks(dataset, ph, seed, n_lags)
    if lag_blocks is None:
        print("ERROR: No multi-lag DAGMA blocks found.")
        print(f"Expected files in results/stage26_validation/")
        print("Run: python gsl_stage26/stage26_run_dagma.py --ph {ph} --dataset {dataset}")
        return

    print(f"\nLoaded lag blocks: {sorted(lag_blocks.keys())}")
    for lbl, Wb in sorted(lag_blocks.items()):
        n_e = int(np.sum(np.abs(Wb) > thr))
        print(f"  {lbl:12s}: {Wb.shape}, {n_e} edges at thr={thr}")

    # Load single-lag DAGMA for comparison
    W_single = load_single_lag_dagma(dataset, ph, seed)
    if W_single is not None:
        print(f"  SingleDAGMA: {W_single.shape}")

    # Build correlation graphs
    corr = np.corrcoef(train_data.T)
    corr = np.nan_to_num(corr, nan=0.0)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0)
    upper = np.triu_indices(N, k=1)

    # ============================================================
    # DEFINE ALL EXPERIMENTS
    # ============================================================
    all_results = []

    # ----- BASELINES -----
    print(f"\n{'='*60}")
    print("BASELINES")
    print(f"{'='*60}")

    # A. NoGraph
    adj_nograph = np.eye(N, dtype=np.float32)
    n_e = N  # self-loops
    for mn in ["TGCN"]:
        m = train_and_eval(adj_nograph, "standard", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, args.max_epochs)
        all_results.append({
            "dataset": dataset, "ph": ph, "method": "NoGraph",
            "model": mn, "n_edges": n_e, "rmse": round(m["RMSE"], 4),
            "mae": round(m["MAE"], 4), "family": "baseline",
        })
    print(f"  NoGraph: {n_e} edges (self-loops only)")

    # B. Physical
    n_e_phys = int(np.sum(adj_phys > 0))
    for mn in ["TGCN"]:
        m = train_and_eval(adj_phys, "standard", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, args.max_epochs)
        all_results.append({
            "dataset": dataset, "ph": ph, "method": "Physical",
            "model": mn, "n_edges": n_e_phys, "rmse": round(m["RMSE"], 4),
            "mae": round(m["MAE"], 4), "family": "baseline",
        })
    print(f"  Physical: {n_e_phys} edges")

    # C. SingleDAGMA (existing Stage 24, threshold sweep)
    if W_single is not None:
        for t in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]:
            adj_s = binary_graph(W_single, t)
            n_e = int(np.sum(adj_s > 0))
            for mn in ["TGCN"]:
                m = train_and_eval(adj_s, "standard", train_X, train_Y,
                                   test_X, test_Y, feat_max, ph, seed, args.max_epochs)
                all_results.append({
                    "dataset": dataset, "ph": ph,
                    "method": f"SingleDAG_thr{t}",
                    "model": mn, "n_edges": n_e,
                    "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                    "family": "C_single_dagma", "threshold": t,
                })
            print(f"  SingleDAG_{t:.3f}: {n_e:6d} edges")

    # Correlation graphs
    for k_name, k_val in [("Corr-K8", 8), ("Corr-K16", 16), ("Corr-K32", 32)]:
        adj_corr = top_k_graph(abs_corr, k_val * N)  # k_val per node
        n_e = int(np.sum(adj_corr > 0))
        for mn in ["TGCN"]:
            m = train_and_eval(adj_corr, "standard", train_X, train_Y,
                               test_X, test_Y, feat_max, ph, seed, args.max_epochs)
            all_results.append({
                "dataset": dataset, "ph": ph, "method": k_name,
                "model": mn, "n_edges": n_e,
                "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                "family": "baseline",
            })
        print(f"  {k_name}: {n_e:6d} edges")

    # ----- MULTI-LAG METHODS -----
    print(f"\n{'='*60}")
    print(f"MULTI-LAG METHODS (thr={thr})")
    print(f"{'='*60}")

    # Prepare lag-specific binary graphs
    lag_graphs = {}
    for lbl, Wb in lag_blocks.items():
        lag_graphs[lbl] = binary_graph(Wb, thr)

    # Get lag-specific graph list (excluding 'current')
    lag_keys = sorted([k for k in lag_graphs.keys() if k.startswith("lag_")],
                       key=lambda x: int(x.split("_")[1]))
    adj_list = [lag_graphs[k] for k in lag_keys]

    total_edges_multi = sum(int(np.sum(adj_list[i] > 0)) for i in range(len(adj_list)))

    # D. UnionGraph — union of all lag graphs into one adjacency
    adj_union = np.zeros((N, N), dtype=np.float32)
    for adj_l in adj_list:
        adj_union = np.maximum(adj_union, adj_l)
    n_e_union = int(np.sum(adj_union > 0))
    for mn in ["TGCN"]:
        m = train_and_eval(adj_union, "standard", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, args.max_epochs)
        all_results.append({
            "dataset": dataset, "ph": ph, "method": f"UnionGraph_thr{thr}",
            "model": mn, "n_edges": n_e_union,
            "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
            "family": "D_union", "threshold": thr,
        })
    print(f"  UnionGraph_{thr}: {n_e_union:6d} edges")

    # Intersection of lag graphs
    adj_intersect = np.ones((N, N), dtype=np.float32)
    for adj_l in adj_list:
        adj_intersect *= adj_l
    n_e_intersect = int(np.sum(adj_intersect > 0))
    for mn in ["TGCN"]:
        m = train_and_eval(adj_intersect, "standard", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, args.max_epochs)
        all_results.append({
            "dataset": dataset, "ph": ph, "method": f"IntersectGraph_thr{thr}",
            "model": mn, "n_edges": n_e_intersect,
            "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
            "family": "D_intersect", "threshold": thr,
        })
    print(f"  IntersectGraph_{thr}: {n_e_intersect:6d} edges")

    # I. AggregatedDAG — mean of abs weights, thresholded
    W_agg = np.zeros((N, N), dtype=np.float32)
    for lbl in lag_keys:
        W_agg += np.abs(lag_blocks[lbl])
    W_agg /= len(lag_keys)
    adj_agg = binary_graph(W_agg, thr)
    n_e_agg = int(np.sum(adj_agg > 0))
    for mn in ["TGCN"]:
        m = train_and_eval(adj_agg, "standard", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, args.max_epochs)
        all_results.append({
            "dataset": dataset, "ph": ph, "method": f"AggregatedDAG_thr{thr}",
            "model": mn, "n_edges": n_e_agg,
            "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
            "family": "I_aggregated", "threshold": thr,
        })
    print(f"  AggregatedDAG_{thr}: {n_e_agg:6d} edges")

    # E. MultiGraphTGCN — different adjacency per timestep
    print(f"\n  Training MultiGraphTGCN ({len(adj_list)} graphs, {total_edges_multi} total edges)...")
    for mn in ["TGCN"]:
        m = train_and_eval(adj_list, "multi_graph", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, args.max_epochs)
        all_results.append({
            "dataset": dataset, "ph": ph, "method": f"MultiGraphTGCN_thr{thr}",
            "model": "MultiGraphTGCN", "n_edges": total_edges_multi,
            "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
            "family": "E_multi_graph", "threshold": thr,
            "n_lag_graphs": len(adj_list),
        })
    print(f"  MultiGraphTGCN: RMSE={all_results[-1]['rmse']}")

    # F. WeightedMultiGraphTGCN — learnable weights
    print(f"\n  Training WeightedMultiGraphTGCN...")
    for mn in ["TGCN"]:
        m = train_and_eval(adj_list, "weighted_multi", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, args.max_epochs)
        result = {
            "dataset": dataset, "ph": ph,
            "method": f"WeightedMulti_thr{thr}",
            "model": "WeightedMultiGraphTGCN", "n_edges": total_edges_multi,
            "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
            "family": "F_weighted_multi", "threshold": thr,
            "n_lag_graphs": len(adj_list),
        }
        if "graph_weights" in m:
            result["learned_weights"] = m["graph_weights"]
        all_results.append(result)
    print(f"  WeightedMulti: RMSE={all_results[-1]['rmse']}")
    if "learned_weights" in all_results[-1]:
        w = all_results[-1]["learned_weights"]
        for i, k in enumerate(lag_keys):
            print(f"    {k}: weight = {w[i]:.4f}")

    # G. GatedMultiGraphTGCN — adaptive gate
    print(f"\n  Training GatedMultiGraphTGCN...")
    for mn in ["TGCN"]:
        m = train_and_eval(adj_list, "gated_multi", train_X, train_Y,
                           test_X, test_Y, feat_max, ph, seed, args.max_epochs)
        all_results.append({
            "dataset": dataset, "ph": ph,
            "method": f"GatedMulti_thr{thr}",
            "model": "GatedMultiGraphTGCN", "n_edges": total_edges_multi,
            "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
            "family": "G_gated_multi", "threshold": thr,
            "n_lag_graphs": len(adj_list),
        })
    print(f"  GatedMulti: RMSE={all_results[-1]['rmse']}")

    # Per-lag standalone evaluation
    print(f"\n--- Per-lag standalone ---")
    for lbl in lag_keys:
        adj_l = lag_graphs[lbl]
        n_e = int(np.sum(adj_l > 0))
        for mn in ["TGCN"]:
            m = train_and_eval(adj_l, "standard", train_X, train_Y,
                               test_X, test_Y, feat_max, ph, seed, args.max_epochs)
            all_results.append({
                "dataset": dataset, "ph": ph,
                "method": f"{lbl}_standalone_thr{thr}",
                "model": mn, "n_edges": n_e,
                "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                "family": "per_lag", "threshold": thr,
            })
        print(f"  {lbl}: {n_e:6d} edges, RMSE={all_results[-1]['rmse']}")

    # ----- SUMMARY -----
    print(f"\n{'='*100}")
    print(f"STAGE 26 SUMMARY ({dataset}, PH={ph}, seed={seed}, thr={thr})")
    print(f"{'='*100}")
    print(f"{'Method':40s} | {'Model':25s} | {'Edges':>6s} | {'RMSE':>8s} | {'MAE':>8s}")
    print("-" * 105)
    for r in sorted(all_results, key=lambda x: x["rmse"]):
        print(f"{r['method']:40s} | {r['model']:25s} | {r['n_edges']:6d} | "
              f"{r['rmse']:8.4f} | {r['mae']:8.4f}")

    # Save results
    csv_path = os.path.join(RESULTS_DIR,
                             f"stage26_results_{prefix}_ph{ph}_seed{seed}.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)

    json_path = os.path.join(RESULTS_DIR,
                              f"stage26_results_{prefix}_ph{ph}_seed{seed}.json")
    summary = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "dataset": dataset, "N": N, "ph": ph, "seed": seed,
        "n_lags": n_lags, "threshold": thr,
        "n_results": len(all_results),
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
