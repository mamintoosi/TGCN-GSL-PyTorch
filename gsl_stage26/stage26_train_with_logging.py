#!/usr/bin/env python3
"""
Stage 26 — Training with checkpoint saving and loss logging.

Extends stage26_validation.py to save:
  1. Per-epoch train/val loss (for convergence curves)
  2. Best model checkpoint (for predicted vs actual plots)
  3. Predictions and ground truth (for post-hoc analysis)

All DAGMA matrices are loaded from existing saved files — no DAGMA recomputation.

Usage:
  python gsl_stage26/stage26_train_with_logging.py --method gated_multi --dataset losloop --ph 1 --seed 42
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
from models.tgcn import TGCN

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage26_checkpoint")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_CONFIGS = {
    "losloop": {
        "feat_path": "data/los_speed.csv",
        "adj_path": "data/los_adj.csv",
        "N": 207, "prefix": "los",
    },
    "shenzhen": {
        "feat_path": "data/sz_speed.csv",
        "adj_path": "data/sz_adj.csv",
        "N": 156, "prefix": "sz",
    },
}


# ============================================================
# DATA LOADING (identical to stage26_validation.py)
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
# MODELS (same as stage26_validation.py)
# ============================================================
class GatedMultiGraphTGCN(nn.Module):
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


class MultiGraphTGCNFixed(nn.Module):
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
            temporal_gap = (T - 1) - t
            graph_idx = temporal_gap % self._n_graphs
            lap = getattr(self, f"lap_{graph_idx}")
            x = inputs[:, t, :].reshape(B, N, 1)
            hh = h.reshape(B, N, self._hidden_dim)
            gh = self._graph_conv(lap, torch.cat([x, hh], dim=2))
            z = torch.sigmoid(self.W_z(gh))
            r, u = torch.chunk(z, chunks=2, dim=2)
            c = torch.tanh(self.W_n(torch.cat([x, r * hh], dim=2)))
            h = u * hh + (1 - u) * c
        return h.reshape(B, N, self._hidden_dim)


# ============================================================
# TRAINING WITH LOGGING
# ============================================================
def train_with_logging(adj_or_model_factory, model_type, train_X, train_Y,
                       test_X, test_Y, feat_max, pre_len, seed=42,
                       max_epochs=50, hidden_dim=64, save_dir=None):
    """
    Train model and save per-epoch loss, checkpoint, and predictions.
    Uses SupervisedForecastTask for consistent loss computation.
    """
    from tasks.supervised import SupervisedForecastTask

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

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
        ),
        batch_size=len(test_X), shuffle=False,
    )

    train_losses = []
    best_loss = float("inf")
    best_epoch = 0

    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = task.training_step((xb, yb))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_epoch = epoch

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{max_epochs}: train_loss={avg_train_loss:.6f}")

    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s. Best epoch: {best_epoch+1} (loss={best_loss:.6f})")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "regressor_state_dict": task.regressor.state_dict() if task.regressor is not None else None,
            "model_type": model_type,
            "hidden_dim": hidden_dim,
            "pre_len": pre_len,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "train_losses": train_losses,
        }
        ckpt_path = os.path.join(save_dir, "best_model.pt")
        torch.save(checkpoint, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

        loss_path = os.path.join(save_dir, "train_loss_history.json")
        with open(loss_path, "w") as f:
            json.dump({
                "train_losses": train_losses,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "train_time_s": round(train_time, 2),
                "max_epochs": max_epochs,
                "seed": seed,
            }, f, indent=2)
        print(f"  Loss history saved: {loss_path}")

    model.eval()
    metrics = task.validation_epoch(test_loader, device)

    with torch.no_grad():
        all_preds = []
        for xb, _ in test_loader:
            xb = xb.to(device)
            pred = task.forward(xb)
            all_preds.append(pred.cpu().numpy())
        preds_np = np.concatenate(all_preds, axis=0)
        preds_np = preds_np.transpose(0, 2, 1)

    if save_dir:
        np.save(os.path.join(save_dir, "y_pred.npy"), preds_np)
        np.save(os.path.join(save_dir, "y_true.npy"), test_Y)
        print(f"  Predictions saved: {save_dir}/y_pred.npy, y_true.npy")

    n_params = sum(p.numel() for p in model.parameters())
    if task.regressor is not None:
        n_params += sum(p.numel() for p in task.regressor.parameters())

    return {
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "train_time_s": round(train_time, 2),
        "n_params": n_params,
        "best_epoch": best_epoch,
    }



# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stage 26 Training with Checkpoint Saving")
    parser.add_argument("--method", type=str, required=True,
                        choices=["nograph", "gated_multi", "multi_graph_fixed"],
                        help="Model type to train")
    parser.add_argument("--dataset", type=str, default="losloop",
                        choices=["losloop", "shenzhen"])
    parser.add_argument("--ph", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-lags", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 26 — TRAINING WITH CHECKPOINT SAVING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Method: {args.method}, Dataset: {args.dataset}, PH: {args.ph}, Seed: {args.seed}")
    print("=" * 70)

    # Load data
    train_data, test_data, adj_phys, feat_max = load_data(args.dataset)
    train_X, train_Y = generate_sequences(train_data, 12, args.ph)
    test_X, test_Y = generate_sequences(test_data, 12, args.ph)
    N = DATASET_CONFIGS[args.dataset]["N"]
    print(f"Train: {train_X.shape}, Test: {test_X.shape}, N={N}")

    # Build adjacency
    if args.method == "nograph":
        adj_factory = np.eye(N, dtype=np.float32)
        model_type = "standard"
    elif args.method in ("gated_multi", "multi_graph_fixed"):
        lag_blocks = load_multilag_blocks(args.dataset, args.ph, seed=42, n_lags=args.n_lags)
        if lag_blocks is None:
            print("ERROR: No DAGMA blocks found. Run DAGMA first.")
            sys.exit(1)
        lag_keys = sorted([k for k in lag_blocks if k.startswith("lag_")],
                           key=lambda x: int(x.split("_")[1]))
        adj_list = [binary_graph(lag_blocks[k], args.threshold) for k in lag_keys]
        total_edges = sum(int(a.sum()) for a in adj_list)
        print(f"Lag graphs: {len(adj_list)} graphs, {total_edges} total edges")
        for k, a in zip(lag_keys, adj_list):
            print(f"  {k}: {int(a.sum())} edges")
        adj_factory = adj_list
        model_type = args.method
    else:
        print(f"ERROR: Unknown method {args.method}")
        sys.exit(1)

    # Save directory
    save_dir = os.path.join(RESULTS_DIR,
                            f"{DATASET_CONFIGS[args.dataset]['prefix']}_ph{args.ph}_"
                            f"seed{args.seed}_{args.method}")

    # Train
    metrics = train_with_logging(
        adj_factory, model_type, train_X, train_Y,
        test_X, test_Y, feat_max, args.ph,
        seed=args.seed, max_epochs=args.max_epochs,
        hidden_dim=args.hidden_dim, save_dir=save_dir,
    )

    print(f"\n{'='*70}")
    print(f"RESULTS: {args.method} / {args.dataset} / PH={args.ph} / seed={args.seed}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  Params: {metrics['n_params']}")
    print(f"  Best epoch: {metrics['best_epoch']+1}")
    print(f"  Train time: {metrics['train_time_s']:.1f}s")
    print(f"{'='*70}")

    # Save summary
    summary_path = os.path.join(save_dir, "metrics.json")
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {summary_path}")


if __name__ == "__main__":
    main()
