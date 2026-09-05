#!/usr/bin/env python3
"""
Stage 25 — Experiment Families E + F: Dual-Graph Architecture & Warm-Up Refinement.

Family E: Dual-graph TGCN that processes physical and functional graphs separately.
Family F: Warm-up → extract representations → refine graph → retrain.

Usage:
  python gsl_stage25/stage25_dual_graph.py
  python gsl_stage25/stage25_dual_graph.py --dataset losloop
"""
import os, sys, json, time, argparse, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.graph_conv import calculate_laplacian_with_self_loop
from tasks.supervised import SupervisedForecastTask

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage25_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_CONFIGS = {
    "shenzhen": {"feat_path": "data/sz_speed.csv", "adj_path": "data/sz_adj.csv", "N": 156, "prefix": "sz"},
    "losloop": {"feat_path": "data/los_speed.csv", "adj_path": "data/los_adj.csv", "N": 207, "prefix": "los"},
}


# ============================================================
# Family E: Dual-Graph TGCN
# ============================================================
class DualGraphTGCNCell(nn.Module):
    """TGCN cell that applies graph convolution with two separate adjacencies."""
    def __init__(self, adj1, adj2, input_dim, hidden_dim):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("lap1", calculate_laplacian_with_self_loop(torch.FloatTensor(adj1)))
        self.register_buffer("lap2", calculate_laplacian_with_self_loop(torch.FloatTensor(adj2)))

        # Gate for blending two graph convolutions
        self.gate = nn.Sequential(nn.Linear(2 * (1 + hidden_dim) + 1, 1), nn.Sigmoid())

        # Shared components
        self.W_z = nn.Linear(1 + hidden_dim, hidden_dim)
        self.W_n = nn.Linear(1 + hidden_dim, hidden_dim)

    def _graph_conv(self, lap, x):
        """Apply graph convolution: lap @ x."""
        # x: (batch, num_nodes, dim)
        B, N, D = x.shape
        x_flat = x.permute(1, 2, 0).reshape(N, D * B)  # (N, D*B)
        out = lap @ x_flat  # (N, D*B)
        return out.reshape(N, D, B).permute(2, 0, 1)  # (B, N, D)

    def forward(self, inputs, hidden_state):
        B, N = inputs.shape
        h = hidden_state.reshape(B, N, self._hidden_dim)
        x = inputs.reshape(B, N, 1)

        # Two graph convolutions
        gh1 = self._graph_conv(self.lap1, torch.cat([x, h], dim=2))  # (B, N, 1+H)
        gh2 = self._graph_conv(self.lap2, torch.cat([x, h], dim=2))  # (B, N, 1+H)

        # Learnable blend
        g_input = torch.cat([gh1, gh2, x], dim=2)  # (B, N, 2*(1+H)+1)
        alpha = self.gate(g_input)  # (B, N, 1)
        gh = alpha * gh1 + (1 - alpha) * gh2  # (B, N, 1+H)

        # GRU-like update
        z = torch.sigmoid(self.W_z(gh))
        n = torch.tanh(self.W_n(torch.cat([x, z * h], dim=2)))
        new_h = (1 - z) * h + z * n

        return new_h.reshape(B, N * self._hidden_dim), new_h.reshape(B, N * self._hidden_dim)


class DualGraphTGCN(nn.Module):
    def __init__(self, adj1, adj2, hidden_dim=64, **kwargs):
        super().__init__()
        self._input_dim = adj1.shape[0]
        self._hidden_dim = hidden_dim
        self.cell = DualGraphTGCNCell(adj1, adj2, self._input_dim, hidden_dim)

    def forward(self, inputs):
        B, T, N = inputs.shape
        h = torch.zeros(B, N * self._hidden_dim, device=inputs.device, dtype=inputs.dtype)
        for t in range(T):
            out, h = self.cell(inputs[:, t, :], h)
        return out.reshape(B, N, self._hidden_dim)

    @property
    def hyperparameters(self):
        return {"hidden_dim": self._hidden_dim}


class DualGraphGCN(nn.Module):
    """GCN that blends two graph convolutions with a learnable gate."""
    def __init__(self, adj1, adj2, seq_len=12, hidden_dim=64, **kwargs):
        super().__init__()
        self._num_nodes = adj1.shape[0]
        self._input_dim = seq_len
        self._output_dim = hidden_dim
        self.register_buffer("lap1", calculate_laplacian_with_self_loop(torch.FloatTensor(adj1)))
        self.register_buffer("lap2", calculate_laplacian_with_self_loop(torch.FloatTensor(adj2)))
        self.weights = nn.Parameter(torch.FloatTensor(self._input_dim, self._output_dim))
        self.alpha = nn.Parameter(torch.tensor(0.5))  # blend scalar
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, inputs):
        B = inputs.shape[0]
        x = inputs.transpose(0, 2).transpose(1, 2).reshape(self._num_nodes, B * self._input_dim)
        a1x = self.lap1 @ x
        a2x = self.lap2 @ x
        alpha = torch.sigmoid(self.alpha)
        ax = alpha * a1x + (1 - alpha) * a2x
        ax = ax.reshape(self._num_nodes, B, self._input_dim)
        out = torch.tanh(ax.reshape(self._num_nodes * B, self._input_dim) @ self.weights)
        return out.reshape(self._num_nodes, B, self._output_dim).transpose(0, 1)

    @property
    def hyperparameters(self):
        return {"num_nodes": self._num_nodes, "input_dim": self._input_dim, "hidden_dim": self._output_dim}


# ============================================================
# Warm-up / Graph Refinement (Family F)
# ============================================================
def warmup_refine(adj_phys, train_X, train_Y, test_X, test_Y, feat_max,
                  model_name="TGCN", seed=42, warmup_epochs=30, refine_epochs=20,
                  top_k_edges=32):
    """
    F1: Warm-up → extract hidden representation → build similarity graph → retrain.
    """
    # Step 1: Warm-up with physical graph
    set_seed(seed)
    if model_name == "TGCN":
        warmup_model = __import__("models.tgcn", fromlist=["TGCN"]).TGCN(adj=adj_phys, hidden_dim=64)
    else:
        warmup_model = __import__("models.gcn", fromlist=["GCN"]).GCN(adj=adj_phys, seq_len=12, hidden_dim=64)

    task = SupervisedForecastTask(model=warmup_model, loss="mse" if model_name == "GCN" else "mse_with_regularizer",
                                   pre_len=train_Y.shape[1], learning_rate=0.001, weight_decay=0.0001,
                                   feat_max_val=feat_max)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warmup_model = warmup_model.to(device)
    if task.regressor is not None:
        task.regressor = task.regressor.to(device)

    optimizer = task.configure_optimizer()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(train_X), torch.FloatTensor(train_Y)),
        batch_size=128, shuffle=True)

    for _ in range(warmup_epochs):
        warmup_model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = task.training_step((xb, yb))
            loss.backward()
            optimizer.step()

    # Step 2: Extract hidden representations
    warmup_model.eval()
    with torch.no_grad():
        # Use training data for representation
        inp = torch.FloatTensor(train_X).to(device)
        hidden = warmup_model(inp)  # (B, N, H)
        repr = hidden.mean(dim=0).cpu().numpy()  # (N, H)

    # Step 3: Build functional graph from representation similarity
    from numpy.linalg import norm
    N = repr.shape[0]
    sim = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j:
                sim[i, j] = np.dot(repr[i], repr[j]) / (norm(repr[i]) * norm(repr[j]) + 1e-8)

    # Keep top-K edges by similarity
    np.fill_diagonal(sim, 0)
    adj_refined = np.zeros((N, N), dtype=np.float32)
    flat_idx = np.argsort(sim.flatten())[::-1][:top_k_edges]
    adj_refined.flat[flat_idx] = 1.0

    # Step 4: Retrain with refined graph
    set_seed(seed)
    if model_name == "TGCN":
        refined_model = __import__("models.tgcn", fromlist=["TGCN"]).TGCN(adj=adj_refined, hidden_dim=64)
    else:
        refined_model = __import__("models.gcn", fromlist=["GCN"]).GCN(adj=adj_refined, seq_len=12, hidden_dim=64)

    task2 = SupervisedForecastTask(model=refined_model, loss="mse" if model_name == "GCN" else "mse_with_regularizer",
                                    pre_len=train_Y.shape[1], learning_rate=0.001, weight_decay=0.0001,
                                    feat_max_val=feat_max)
    refined_model = refined_model.to(device)
    if task2.regressor is not None:
        task2.regressor = task2.regressor.to(device)

    optimizer2 = task2.configure_optimizer()
    for _ in range(refine_epochs):
        refined_model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer2.zero_grad()
            loss = task2.training_step((xb, yb))
            loss.backward()
            optimizer2.step()

    refined_model.eval()
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_Y)),
        batch_size=len(test_X), shuffle=False)
    metrics = task2.validation_epoch(test_loader, device)

    return metrics, adj_refined


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_W(dataset, ph, seed=42):
    prefix = DATASET_CONFIGS[dataset]["prefix"]
    N = DATASET_CONFIGS[dataset]["N"]
    path = os.path.join(PROJECT_ROOT, "results", "stage24_validation",
                        f"{prefix}_ph{ph}_seed{seed}_W_raw_temporal.npy")
    if not os.path.exists(path):
        return None
    W_raw = np.load(path)
    return W_raw[:N, N:2*N]


def binary_graph(W, threshold):
    adj = (np.abs(W) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)
    return adj


def train_and_eval_standard(adj, model_name, train_X, train_Y, test_X, test_Y,
                             feat_max, pre_len, seed=42, max_epochs=50):
    set_seed(seed)
    if model_name == "GCN":
        from models.gcn import GCN
        model = GCN(adj=adj, seq_len=12, hidden_dim=64)
        loss_name = "mse"
    else:
        from models.tgcn import TGCN
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

    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = task.training_step((xb, yb))
            loss.backward()
            optimizer.step()

    model.eval()
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_Y)),
        batch_size=len(test_X), shuffle=False)
    return task.validation_epoch(test_loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="shenzhen", choices=["shenzhen", "losloop"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--ph", type=int, default=1)
    parser.add_argument("--skip-dual", action="store_true", help="Skip dual-graph experiments")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warm-up refinement experiments")
    args = parser.parse_args()

    dataset = args.dataset
    seed = args.seed
    config = DATASET_CONFIGS[dataset]
    N = config["N"]
    prefix = config["prefix"]

    print("=" * 80)
    print(f"STAGE 25 — DUAL-GRAPH & WARM-UP REFINEMENT ({dataset}, seed={seed})")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    train_data, test_data, adj_phys, feat_max = load_data(dataset)
    train_X, train_Y = generate_sequences(train_data, 12, args.ph)
    test_X, test_Y = generate_sequences(test_data, 12, args.ph)

    W = load_W(dataset, args.ph, seed)
    if W is None:
        print("ERROR: No DAGMA matrix found.")
        return
    adj_dagma = binary_graph(W, 0.1)

    all_results = []

    # Baselines first
    print("\n--- Standard baselines ---")
    for bm_name, bm_adj in [("Physical", adj_phys), ("NoGraph", np.eye(N, dtype=np.float32)),
                              ("TempDAGMA_0.1", adj_dagma)]:
        n_e = int(np.sum(bm_adj > 0))
        for mn in ["GCN", "TGCN"]:
            m = train_and_eval_standard(bm_adj, mn, train_X, train_Y, test_X, test_Y,
                                         feat_max, args.ph, seed=seed, max_epochs=args.max_epochs)
            all_results.append({
                "dataset": dataset, "ph": args.ph, "method": bm_name, "model": mn,
                "n_edges": n_e, "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                "family": "baseline",
            })
        print(f"  {bm_name:25s}: {n_e:6d} edges")

    # Family E: Dual-Graph
    if not args.skip_dual:
        print("\n--- Family E: Dual-Graph Architecture ---")
        for model_cls_name, model_cls in [("DualTGCN", DualGraphTGCN), ("DualGCN", DualGraphGCN)]:
            set_seed(seed)
            if model_cls_name == "DualTGCN":
                model = model_cls(adj1=adj_phys, adj2=adj_dagma, hidden_dim=64)
                loss_name = "mse_with_regularizer"
            else:
                model = model_cls(adj1=adj_phys, adj2=adj_dagma, seq_len=12, hidden_dim=64)
                loss_name = "mse"

            task = SupervisedForecastTask(model=model, loss=loss_name, pre_len=args.ph,
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
            for _ in range(args.max_epochs):
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
            m = task.validation_epoch(test_loader, device)

            n_e = int(np.sum(adj_phys > 0)) + int(np.sum(adj_dagma > 0))
            all_results.append({
                "dataset": dataset, "ph": args.ph,
                "method": f"dual_{model_cls_name.lower()}_phys+dagma",
                "model": model_cls_name,
                "n_edges": n_e, "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                "family": "E_dual_graph",
            })
            print(f"  {model_cls_name:25s}: RMSE={m['RMSE']:.4f} ({train_time:.1f}s)")

    # Family F: Warm-up Refinement
    if not args.skip_warmup:
        print("\n--- Family F: Warm-up → Graph Refinement ---")
        for top_k in [16, 32, 64]:
            for mn in ["TGCN"]:
                m, adj_r = warmup_refine(
                    adj_phys, train_X, train_Y, test_X, test_Y, feat_max,
                    model_name=mn, seed=seed, warmup_epochs=30, refine_epochs=20,
                    top_k_edges=top_k)
                n_e = int(np.sum(adj_r > 0))
                all_results.append({
                    "dataset": dataset, "ph": args.ph,
                    "method": f"warmup_refine_{mn}_K{top_k}",
                    "model": mn,
                    "n_edges": n_e, "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                    "family": "F_warmup_refine",
                })
                print(f"  Warmup-{mn}-K{top_k:3d}: {n_e:6d} edges, RMSE={m['RMSE']:.4f}")

    # Summary
    print(f"\n{'='*90}")
    print(f"STAGE 25 DUAL/REFINE SUMMARY ({dataset}, PH={args.ph}, seed={seed})")
    print(f"{'='*90}")
    print(f"{'Method':40s} | {'Edges':>6s} | {'RMSE':>8s} | {'MAE':>8s}")
    print("-" * 75)
    for r in sorted(all_results, key=lambda x: x["rmse"]):
        print(f"{r['method']:40s} | {r['n_edges']:6d} | {r['rmse']:8.4f} | {r['mae']:8.4f}")

    csv_path = os.path.join(RESULTS_DIR, f"stage25_dual_warmup_{prefix}_ph{args.ph}_seed{seed}.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
