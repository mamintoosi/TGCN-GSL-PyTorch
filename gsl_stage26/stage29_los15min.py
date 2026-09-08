#!/usr/bin/env python3
"""
Stage 29 — Los-Loop as a 15-Minute Dataset.

Treats Los-Loop at 15-minute temporal resolution, independent of the original
5-minute version. Uses the exact same pipeline as Stage 26
(SupervisedForecastTask, set_seed, mse_with_regularizer_loss).

Scientific question:
  When Los-Loop is treated as a 15-minute-resolution dataset,
  does T-GCN-MultiGSL-Mix provide a meaningful improvement over T-GCN-NoSpatial?

Usage:
  python gsl_stage26/stage29_los15min.py --phase dagma
  python gsl_stage26/stage29_los15min.py --phase forecast
  python gsl_stage26/stage29_los15min.py --phase analyze
  python gsl_stage26/stage29_los15min.py --phase all
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

from dagma.linear import DagmaLinear
from utils.graph_conv import calculate_laplacian_with_self_loop
from tasks.supervised import SupervisedForecastTask
from models.tgcn import TGCN

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage29_los15min")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# DATA: Los-Loop at 15-minute resolution
# ============================================================

def resample_15min(data_5min):
    """Average every 3 consecutive 5-min timesteps -> 15-min interval."""
    T = data_5min.shape[0]
    T_15 = T // 3
    data_trimmed = data_5min[:T_15 * 3]
    return data_trimmed.reshape(T_15, 3, -1).mean(axis=1)


def load_los15_data():
    """Load Los-Loop, resample to 15-min, return (feat_15, adj, N, feat_max, split)."""
    feat_5 = np.array(
        pd.read_csv(os.path.join(PROJECT_ROOT, "data/los_speed.csv")),
        dtype=np.float32,
    )
    adj = np.array(
        pd.read_csv(os.path.join(PROJECT_ROOT, "data/los_adj.csv"), header=None),
        dtype=np.float32,
    )
    N = feat_5.shape[1]  # 207
    feat_15 = resample_15min(feat_5)
    split = int(feat_15.shape[0] * 0.8)
    feat_max = float(np.max(feat_15[:split]))
    return feat_15, adj, N, feat_max, split


# ============================================================
# SEQUENCE GENERATION (identical to Stage 26)
# ============================================================

def load_data_15min():
    """Return (train_data, test_data, adj, feat_max) normalized."""
    feat_15, adj, N, feat_max, split = load_los15_data()
    train = feat_15[:split] / feat_max
    test = feat_15[split:] / feat_max
    return train, test, adj, feat_max


def generate_sequences(data, seq_len, pre_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pre_len):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len : i + seq_len + pre_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ============================================================
# SEED HANDLING (identical to Stage 26)
# ============================================================

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


def build_multilag_Z(train_norm, N, n_lags):
    """Build DAGMA input Z = [x(t-L), ..., x(t-1), x(t)]."""
    T = train_norm.shape[0]
    rows = []
    for t in range(n_lags, T):
        z = np.concatenate(
            [train_norm[t - l] for l in range(n_lags, 0, -1)] + [train_norm[t]]
        )
        rows.append(z)
    return np.array(rows, dtype=np.float32)


def extract_lag_blocks(W_est, N, n_lags):
    """Extract lag-specific blocks from DAGMA W matrix."""
    current_start = n_lags * N
    blocks = {}
    for l_idx in range(n_lags):
        W_block = W_est[l_idx * N : (l_idx + 1) * N, current_start : current_start + N]
        lag_value = n_lags - l_idx
        blocks[f"lag_{lag_value}"] = W_block.astype(np.float32)
    blocks["current"] = W_est[current_start : current_start + N,
                              current_start : current_start + N].astype(np.float32)
    return blocks


# ============================================================
# MODELS (identical to Stage 26)
# ============================================================

class GatedMultiGraphTGCN(nn.Module):
    """Per-node, per-timestep adaptive graph selection (Stage 26)."""
    def __init__(self, adj_list, hidden_dim=64, **kwargs):
        super().__init__()
        self._input_dim = adj_list[0].shape[0]
        self._hidden_dim = hidden_dim
        self._n_graphs = len(adj_list)
        laps = [calculate_laplacian_with_self_loop(torch.FloatTensor(adj)) for adj in adj_list]
        self.register_buffer("lap_stack", torch.stack(laps))
        self.gate_net = nn.Sequential(
            nn.Linear(1 + hidden_dim, hidden_dim), nn.ReLU(),
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
            gate_w = F.softmax(self.gate_net(gate_input), dim=-1)
            adj_w = torch.einsum("bnk,kij->bnj", gate_w, self.lap_stack)
            gh = torch.cat([x, hh], dim=2)
            ag = torch.bmm(adj_w, gh)
            z = torch.sigmoid(self.W_z(ag))
            r, u = torch.chunk(z, chunks=2, dim=2)
            c = torch.tanh(self.W_n(torch.cat([x, r * hh], dim=2)))
            h = u * hh + (1 - u) * c
        return h.reshape(B, N, self._hidden_dim)

    @property
    def hyperparameters(self):
        return {"hidden_dim": self._hidden_dim}


class MultiGraphTGCNFixed(nn.Module):
    """Fixed lag-specific multi-graph (Stage 26 corrected alignment)."""
    def __init__(self, adj_list, hidden_dim=64, seq_len=12, **kwargs):
        super().__init__()
        self._input_dim = adj_list[0].shape[0]
        self._hidden_dim = hidden_dim
        self._n_graphs = len(adj_list)
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
            gap = (T - 1) - t
            idx = gap % self._n_graphs
            lap = getattr(self, f"lap_{idx}")
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
# TRAINING AND EVALUATION (identical to Stage 26)
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
# PHASE 1: DAGMA
# ============================================================

def run_dagma_phase(args):
    """DAGMA phase: reuse Stage 27 results if available, otherwise compute once.

    DAGMA output does NOT depend on PH (Z is built from training data only).
    So we only need one computation and copy it for all PH values.
    """
    print("=" * 80)
    print("PHASE 1: DAGMA on Los-Loop-15min")
    print("=" * 80)

    n_lags = args.lags
    stage27_dir = os.path.join(PROJECT_ROOT, "results", "stage27_resolution")

    # Stage 27 files: los_15min_ph1_seed42_L3_{label}.npy
    # Stage 29 files: los15_ph{ph}_seed42_L3_{label}.npy
    labels = [f"lag_{l}" for l in range(1, n_lags + 1)] + ["current", "W_full"]

    # Check if Stage 27 has the DAGMA results
    stage27_files = {}
    for label in labels:
        s27_path = os.path.join(stage27_dir, f"los_15min_ph1_seed{args.seed}_L{n_lags}_{label}.npy")
        if os.path.exists(s27_path):
            stage27_files[label] = s27_path

    if len(stage27_files) == len(labels):
        print(f"Found Stage 27 DAGMA results ({len(stage27_files)} files). Reusing...")
        for label in labels:
            for ph in args.phs:
                dest = os.path.join(RESULTS_DIR, f"los15_ph{ph}_seed{args.seed}_L{n_lags}_{label}.npy")
                if not os.path.exists(dest):
                    import shutil
                    shutil.copy2(stage27_files[label], dest)
                    print(f"  Copied {label} -> PH={ph}")
                else:
                    print(f"  Already exists: {label} PH={ph}")

        # Print edge counts from the existing W matrix
        W_est = np.load(stage27_files["W_full"])
        print(f"\nW matrix shape: {W_est.shape}")
        feat_15, _, _, feat_max, split = load_los15_data()
        print(f"DAGMA was computed on {split} training samples, feat_max={feat_max}")
        print(f"\nEdge counts (off-diagonal, threshold={args.threshold}):")
        blocks = extract_lag_blocks(W_est, feat_15.shape[1], n_lags)
        for label_name, W_block in sorted(blocks.items()):
            A = binary_graph(W_block, args.threshold)
            n_edges = int(A.sum())
            n_self = int(np.trace(A))
            n_cross = n_edges - n_self
            print(f"  {label_name:10s}: {n_edges:4d} total, {n_self:4d} self-loops, {n_cross:4d} cross-sensor, max|w|={np.abs(W_block).max():.4f}")
        return

    # Otherwise, compute DAGMA once (PH-independent)
    print("No existing DAGMA results found. Computing...")
    train_data, test_data, adj, feat_max = load_data_15min()
    N = train_data.shape[1]

    print(f"Train shape: {train_data.shape}, N={N}, feat_max={feat_max}")
    print(f"Lags: {n_lags}, Total vars: {(n_lags + 1) * N}")

    # DAGMA output is PH-independent: compute once
    print(f"\n--- Computing DAGMA (PH-independent, computed once) ---")
    Z = build_multilag_Z(train_data, N, n_lags)
    print(f"Z shape: {Z.shape}")

    np.random.seed(args.seed)
    t0 = time.time()
    model = DagmaLinear(loss_type="l2", verbose=True)
    W_est = model.fit(Z, lambda1=args.lambda1, w_threshold=0.0,
                      warm_iter=args.warm_iter, max_iter=args.max_iter)
    runtime = time.time() - t0
    print(f"DAGMA completed in {runtime:.1f}s ({runtime/60:.1f} min)")

    blocks = extract_lag_blocks(W_est, N, n_lags)

    # Save for all PH values (they're identical)
    for ph in args.phs:
        prefix = f"los15_ph{ph}_seed{args.seed}_L{n_lags}"
        np.save(os.path.join(RESULTS_DIR, f"{prefix}_W_full.npy"), W_est)
        for label, W_block in blocks.items():
            np.save(os.path.join(RESULTS_DIR, f"{prefix}_{label}.npy"), W_block)

        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "losloop_15min",
            "N": N, "PH": ph, "n_lags": n_lags,
            "matrix_shape": list(W_est.shape),
            "seed": args.seed, "lambda1": args.lambda1,
            "warm_iter": args.warm_iter, "max_iter": args.max_iter,
            "feat_max": feat_max, "split": int(train_data.shape[0] / 0.8 * 0.8),
            "runtime_s": round(runtime, 1),
            "note": "DAGMA is PH-independent; result is identical for all PH values",
        }
        with open(os.path.join(RESULTS_DIR, f"{prefix}_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\nEdge counts (off-diagonal, threshold={args.threshold}):")
    for label_name, W_block in sorted(blocks.items()):
        A = binary_graph(W_block, args.threshold)
        n_edges = int(A.sum())
        n_self = int(np.trace(A))
        n_cross = n_edges - n_self
        print(f"  {label_name:10s}: {n_edges:4d} total, {n_self:4d} self-loops, {n_cross:4d} cross-sensor, max|w|={np.abs(W_block).max():.4f}")


# ============================================================
# PHASE 2: FORECASTING
# ============================================================

def run_forecast_phase(args):
    print("=" * 80)
    print("PHASE 2: Forecasting on Los-Loop-15min")
    print("=" * 80)

    train_data, test_data, adj, feat_max = load_data_15min()
    N = train_data.shape[1]
    n_lags = args.lags

    all_results = []

    for ph in args.phs:
        print(f"\n{'='*80}")
        print(f"PH={ph}  (physical horizon: {ph * 15} min)")
        print(f"{'='*80}")

        # Load DAGMA blocks
        prefix_dagma = f"los15_ph{ph}_seed{args.seed}_L{n_lags}"
        lag_blocks = {}
        for l in range(1, n_lags + 1):
            path = os.path.join(RESULTS_DIR, f"{prefix_dagma}_lag_{l}.npy")
            if os.path.exists(path):
                lag_blocks[f"lag_{l}"] = np.load(path)
        path = os.path.join(RESULTS_DIR, f"{prefix_dagma}_current.npy")
        if os.path.exists(path):
            lag_blocks["current"] = np.load(path)

        if not lag_blocks:
            print(f"ERROR: No DAGMA blocks for PH={ph}. Run --phase dagma first.")
            continue

        # Build adjacency matrices
        lag_keys = sorted([k for k in lag_blocks if k.startswith("lag_")],
                          key=lambda x: int(x.split("_")[1]))
        adj_list = [binary_graph(lag_blocks[k], args.threshold) for k in lag_keys]
        total_edges = sum(int(a.sum()) for a in adj_list)
        print(f"Lag graphs: {len(adj_list)} graphs, {total_edges} total edges (off-diagonal+diag)")
        for k, a in zip(lag_keys, adj_list):
            n_off = int(a.sum()) - int(np.trace(a))
            print(f"  {k}: {int(a.sum())} total, {n_off} cross-sensor")

        # Generate sequences
        train_X, train_Y = generate_sequences(train_data, 12, ph)
        test_X, test_Y = generate_sequences(test_data, 12, ph)
        print(f"Train: {train_X.shape}, Test: {test_X.shape}")

        for seed in args.seeds:
            print(f"\n--- Seed {seed}, PH={ph} ---")

            # T-GCN-NoSpatial
            adj_id = np.eye(N, dtype=np.float32)
            m = train_and_eval(adj_id, "standard", train_X, train_Y,
                               test_X, test_Y, feat_max, ph, seed, args.epochs)
            all_results.append({
                "dataset": "losloop_15min", "ph": ph, "seed": seed,
                "method": "NoGraph", "model": "TGCN",
                "n_edges": N, "rmse": round(m["RMSE"], 4),
                "mae": round(m["MAE"], 4), "n_params": m["n_params"],
            })
            print(f"  NoGraph:      RMSE={m['RMSE']:.4f}")

            # T-GCN-MultiGSL (fixed alignment)
            m = train_and_eval(adj_list, "multi_graph_fixed", train_X, train_Y,
                               test_X, test_Y, feat_max, ph, seed, args.epochs)
            all_results.append({
                "dataset": "losloop_15min", "ph": ph, "seed": seed,
                "method": "MultiGraphTGCN_fixed", "model": "MultiGraphTGCNFixed",
                "n_edges": total_edges, "rmse": round(m["RMSE"], 4),
                "mae": round(m["MAE"], 4), "n_params": m["n_params"],
            })
            print(f"  MultiGSL:     RMSE={m['RMSE']:.4f}")

            # T-GCN-MultiGSL-Mix (proposed method)
            m = train_and_eval(adj_list, "gated_multi", train_X, train_Y,
                               test_X, test_Y, feat_max, ph, seed, args.epochs)
            all_results.append({
                "dataset": "losloop_15min", "ph": ph, "seed": seed,
                "method": "GatedMultiGraphTGCN", "model": "GatedMultiGraphTGCN",
                "n_edges": total_edges, "rmse": round(m["RMSE"], 4),
                "mae": round(m["MAE"], 4), "n_params": m["n_params"],
            })
            print(f"  MultiGSL-Mix: RMSE={m['RMSE']:.4f}")

    # Save results
    json_path = os.path.join(RESULTS_DIR, "stage29_los15min_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "losloop_15min",
            "seeds": args.seeds, "phs": args.phs,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    for method in ["NoGraph", "MultiGraphTGCN_fixed", "GatedMultiGraphTGCN"]:
        print(f"\n{method}:")
        for ph in args.phs:
            rmses = [r["rmse"] for r in all_results
                     if r["method"] == method and r["ph"] == ph]
            if rmses:
                print(f"  PH={ph} ({ph*15:2d} min): "
                      f"mean={np.mean(rmses):.4f}  std={np.std(rmses):.4f}  "
                      f"seeds={rmses}")


# ============================================================
# PHASE 3: ANALYSIS
# ============================================================

def run_analyze_phase(args):
    print("=" * 80)
    print("PHASE 3: Analysis of Los-Loop-15min Results")
    print("=" * 80)

    results_path = os.path.join(RESULTS_DIR, "stage29_los15min_results.json")
    if not os.path.exists(results_path):
        print("ERROR: Results not found. Run --phase forecast first.")
        return

    with open(results_path) as f:
        data = json.load(f)
    results = data["results"]

    # --- Improvement over NoGraph by PH ---
    print("\n--- Improvement over NoGraph by PH ---")
    print(f"{'PH':>4s} {'Physical':>10s} {'Method':>25s} {'RMSE':>8s} {'Improv':>8s}")
    print("-" * 60)
    for ph in args.phs:
        no_rmses = [r["rmse"] for r in results if r["method"] == "NoGraph" and r["ph"] == ph]
        if not no_rmses:
            continue
        no_mean = np.mean(no_rmses)
        for method in ["MultiGraphTGCN_fixed", "GatedMultiGraphTGCN"]:
            rmses = [r["rmse"] for r in results if r["method"] == method and r["ph"] == ph]
            if rmses:
                m_mean = np.mean(rmses)
                imp = (no_mean - m_mean) / no_mean * 100
                print(f"{ph:4d} {ph*15:8d} min {method:>25s} {m_mean:8.4f} {imp:>+7.2f}%")

    # --- Per-seed results ---
    print("\n--- Per-Seed Results (PH=1) ---")
    print(f"{'Seed':>6s} {'NoGraph':>10s} {'MultiGSL':>10s} {'MultiGSL-Mix':>14s}")
    print("-" * 45)
    for seed in args.seeds:
        no = [r["rmse"] for r in results if r["method"] == "NoGraph" and r["ph"] == 1 and r["seed"] == seed]
        mg = [r["rmse"] for r in results if r["method"] == "MultiGraphTGCN_fixed" and r["ph"] == 1 and r["seed"] == seed]
        gm = [r["rmse"] for r in results if r["method"] == "GatedMultiGraphTGCN" and r["ph"] == 1 and r["seed"] == seed]
        if no and mg and gm:
            imp = (no[0] - gm[0]) / no[0] * 100
            print(f"{seed:6d} {no[0]:10.4f} {mg[0]:10.4f} {gm[0]:14.4f}  ({imp:+.2f}%)")

    # --- Mean ± std across seeds ---
    print("\n--- Mean ± Std Across Seeds ---")
    for method in ["NoGraph", "MultiGraphTGCN_fixed", "GatedMultiGraphTGCN"]:
        print(f"\n{method}:")
        for ph in args.phs:
            rmses = [r["rmse"] for r in results if r["method"] == method and r["ph"] == ph]
            if rmses:
                print(f"  PH={ph:2d} ({ph*15:2d} min): {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

    # --- Comparison with SZ-Taxi (cross-dataset, not causal) ---
    sz_path = os.path.join(PROJECT_ROOT, "results", "stage26_validation",
                            "stage26_results_sz_ph1_seed42.json")
    if os.path.exists(sz_path):
        with open(sz_path) as f:
            sz_data = json.load(f)
        print("\n--- Cross-Dataset Comparison (PH=1, seed=42) ---")
        print(f"{'Dataset':<25s} {'NoGraph':>10s} {'GatedMulti':>12s} {'Improvement':>12s}")
        print("-" * 62)
        # SZ-Taxi
        sz_map = {r["method"]: r for r in sz_data["results"]}
        sz_no = sz_map["NoGraph"]["rmse"]
        sz_gm = sz_map["GatedMulti_thr0.1"]["rmse"]
        sz_imp = (sz_no - sz_gm) / sz_no * 100
        print(f"{'SZ-Taxi 15min':<25s} {sz_no:10.4f} {sz_gm:12.4f} {sz_imp:>+11.2f}%")
        # Los-Loop-15min (seed=42)
        no_42 = [r["rmse"] for r in results if r["method"] == "NoGraph" and r["ph"] == 1 and r["seed"] == 42]
        gm_42 = [r["rmse"] for r in results if r["method"] == "GatedMultiGraphTGCN" and r["ph"] == 1 and r["seed"] == 42]
        if no_42 and gm_42:
            los_imp = (no_42[0] - gm_42[0]) / no_42[0] * 100
            print(f"{'Los-Loop-15min':<25s} {no_42[0]:10.4f} {gm_42[0]:12.4f} {los_imp:>+11.2f}%")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 29: Los-Loop as 15-min dataset")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["dagma", "forecast", "analyze", "all"])
    parser.add_argument("--phs", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--seed", type=int, default=42, help="DAGMA seed")
    parser.add_argument("--lags", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--lambda1", type=float, default=0.01)
    parser.add_argument("--warm-iter", type=int, default=30000)
    parser.add_argument("--max-iter", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print(f"Stage 29: Los-Loop as 15-min dataset")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PHs: {args.phs}, Seeds: {args.seeds}")

    if args.phase in ("dagma", "all"):
        run_dagma_phase(args)
    if args.phase in ("forecast", "all"):
        run_forecast_phase(args)
    if args.phase in ("analyze", "all"):
        run_analyze_phase(args)


if __name__ == "__main__":
    main()
