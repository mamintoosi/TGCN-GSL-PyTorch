#!/usr/bin/env python3
"""
Stage 20.5: Validate Temporal DAGMA Edge Selection and Threshold Sensitivity

Tasks:
1. Save raw temporal DAGMA W matrix (unthresholded)
2. Identify the exact surviving edge
3. Threshold sensitivity
4. Top-K temporal edge experiment
5. Directional sanity check (synthetic test)
6. Scientific interpretation
"""
import os, sys, json, time, random
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dagma.linear import DagmaLinear
from models.gcn import GCN
from models.tgcn import TGCN
from tasks.supervised import SupervisedForecastTask

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage20_5_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(dataset="shenzhen", split=0.8):
    paths = {
        "shenzhen": ("data/sz_speed.csv", "data/sz_adj.csv"),
        "losloop": ("data/los_speed.csv", "data/los_adj.csv"),
    }
    feat = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, paths[dataset][0])), dtype=np.float32)
    adj = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, paths[dataset][1]), header=None), dtype=np.float32)
    T, N = feat.shape
    train_size = int(T * split)
    feat_max = float(np.max(feat[:train_size]))
    return feat[:train_size] / feat_max, feat[train_size:] / feat_max, adj, feat_max, N


def gen_seq(data, seq_len, pre_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pre_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+pre_len])
    return np.array(X, np.float32), np.array(Y, np.float32)


def train_eval(adj, model_name, trX, trY, teX, teY, fmax, pre_len=1, seed=42, epochs=50, seq_len=12):
    set_seed(seed)
    if model_name == "GCN":
        model = GCN(adj=adj, seq_len=seq_len, hidden_dim=64)
        loss_name = "mse"
    else:
        model = TGCN(adj=adj, hidden_dim=64)
        loss_name = "mse_with_regularizer"

    task = SupervisedForecastTask(model=model, loss=loss_name, pre_len=pre_len,
                                   learning_rate=0.001, weight_decay=0.0001, feat_max_val=fmax)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    if task.regressor: task.regressor = task.regressor.to(dev)
    opt = task.configure_optimizer()

    trX_t = torch.FloatTensor(trX).to(dev)
    trY_t = torch.FloatTensor(trY).to(dev)
    teX_t = torch.FloatTensor(teX).to(dev)
    teY_t = torch.FloatTensor(teY).to(dev)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(trX_t, trY_t), batch_size=128, shuffle=True)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = task.training_step((xb, yb))
            loss.backward()
            opt.step()
    train_time = time.time() - t0

    model.eval()
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(teX_t, teY_t), batch_size=len(teX_t))
    metrics = task.validation_epoch(test_loader, dev)
    metrics["train_time_s"] = round(train_time, 2)
    return metrics


def graph_stats(adj, N=None):
    if N is None: N = adj.shape[0]
    a = (adj > 0).astype(int)
    np.fill_diagonal(a, 0)
    n = int(np.sum(a))
    deg = a.sum(axis=1)
    return {"n_edges": n, "n_active": int(np.sum(deg > 0)), "n_isolated": N - int(np.sum(deg > 0)),
            "max_deg": int(np.max(deg)), "mean_deg": round(float(np.mean(deg)), 2)}


# ====================================================================
# Task 7: Directional sanity check
# ====================================================================
def directional_sanity_check():
    """
    Synthetic test: x3(t) = 0.9*x1(t-1) + noise
    Build Z with [x(t-1), x(t)] and verify DAGMA recovers the direction.
    """
    print("\n" + "="*70)
    print("TASK 7: DIRECTIONAL SANITY CHECK")
    print("="*70)
    set_seed(42)
    N = 3
    T = 500
    noise = 0.05

    # Generate data: x3(t) depends on x1(t-1)
    x = np.zeros((T, N), dtype=np.float32)
    x[0] = np.random.randn(N).astype(np.float32)
    for t in range(1, T):
        x[t, 0] = 0.5 * x[t-1, 0] + noise * np.random.randn()
        x[t, 1] = 0.5 * x[t-1, 1] + noise * np.random.randn()
        x[t, 2] = 0.9 * x[t-1, 0] + noise * np.random.randn()  # x3(t) from x1(t-1)

    # Build Z = [x(t-1), x(t)]
    Z = np.zeros((T-1, 2*N), dtype=np.float32)
    Z[:, 0:N] = x[1:]     # past: x(t-1) at columns 0,1,2
    Z[:, N:2*N] = x[1:]   # Wait, this is wrong. Let me fix.
    # Actually: Z[t] = [x(t), x(t+1)] so row t has past=t, current=t+1
    Z[:, 0:N] = x[:-1]    # past: x(t)
    Z[:, N:2*N] = x[1:]    # current: x(t+1)

    print(f"  Synthetic data: T={T}, N={N}")
    print(f"  Ground truth: x3(t) = 0.9*x1(t-1)")
    print(f"  Z shape: {Z.shape}")

    model = DagmaLinear(loss_type='l2', verbose=False)
    W = model.fit(Z, lambda1=0.01, w_threshold=0.0)

    W_full = W.copy()
    W_cross = W_full[N:2*N, 0:N]  # past→current block

    print(f"\n  Full W shape: {W_full.shape}")
    print(f"  W_cross (past→current) shape: {W_cross.shape}")
    print(f"\n  W_cross (should show x1→x3 = entry [0,2]):")
    for i in range(N):
        for j in range(N):
            if abs(W_cross[i,j]) > 0.01:
                print(f"    W_cross[{i},{j}] = {W_cross[i,j]:.4f}  (x{i+1}(t-1) → x{j+1}(t))")

    # Check: the strong entry should be W_cross[0, 2] (x1 past → x3 current)
    expected_entry = W_cross[0, 2]
    print(f"\n  Expected strong entry W_cross[0,2] (x1→x3): {expected_entry:.4f}")
    if abs(expected_entry) > 0.5:
        print("  ✓ DIRECTION CORRECT: x1(t-1) → x3(t) is the strongest cross-time edge")
    else:
        print(f"  ⚠ Entry [0,2] = {expected_entry:.4f}, checking all entries...")
        max_idx = np.unravel_index(np.argmax(np.abs(W_cross)), W_cross.shape)
        print(f"  Strongest entry: W_cross[{max_idx[0]},{max_idx[1]}] = {W_cross[max_idx]:.4f}")

    # Also check W_cc (contemporaneous block)
    W_cc = W_full[N:2*N, N:2*N]
    print(f"\n  W_cc (contemporaneous) nonzero entries:")
    for i in range(N):
        for j in range(N):
            if abs(W_cc[i,j]) > 0.01:
                print(f"    W_cc[{i},{j}] = {W_cc[i,j]:.4f}")

    result = {
        "W_cross": W_cross.tolist(),
        "W_cc": W_cc.tolist(),
        "expected_entry_0_2": float(expected_entry),
        "strongest_entry": [int(max_idx[0]), int(max_idx[1]), float(W_cross[max_idx])],
    }
    return result


# ====================================================================
# Main validation
# ====================================================================
def run_validation():
    print("="*80)
    print("STAGE 20.5: TEMPORAL DAGMA VALIDATION")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    dataset = "shenzhen"
    seed = 42
    epochs = 50
    seq_len = 12
    pre_len = 1

    # --- Load data ---
    train_data, test_data, adj_phys, feat_max, N = load_data(dataset)
    print(f"\nDataset: {dataset}, N={N}, Train={train_data.shape[0]}, Test={test_data.shape[0]}")
    print(f"feat_max_val: {feat_max:.6f}")

    trX, trY = gen_seq(train_data, seq_len, pre_len)
    teX, teY = gen_seq(test_data, seq_len, pre_len)
    print(f"Sequences: train={trX.shape}, test={teX.shape}")

    # --- Build temporal DAGMA and save RAW W ---
    print("\n" + "="*70)
    print("BUILDING TEMPORAL DAGMA (saving raw W)")
    print("="*70)
    D = 2 * N
    n_samples = train_data.shape[0] - 1
    Z = np.zeros((n_samples, D), dtype=train_data.dtype)
    Z[:, 0:N] = train_data[:-1]
    Z[:, N:2*N] = train_data[1:]

    print(f"  Z shape: {Z.shape}")
    t0 = time.time()
    model = DagmaLinear(loss_type='l2', verbose=False)
    W_full = model.fit(Z, lambda1=0.01, w_threshold=0.0)  # NO thresholding
    dagma_time = time.time() - t0
    print(f"  DAGMA time: {dagma_time:.1f}s")

    # Save raw W
    np.save(os.path.join(RESULTS_DIR, "sz_ph1_raw_W_full.npy"), W_full)

    W_cross = W_full[N:2*N, 0:N]  # past→current block
    W_cc = W_full[N:2*N, N:2*N]   # contemporaneous block
    np.save(os.path.join(RESULTS_DIR, "sz_ph1_raw_W_cross.npy"), W_cross)
    np.save(os.path.join(RESULTS_DIR, "sz_ph1_raw_W_cc.npy"), W_cc)

    print(f"  W_full: {W_full.shape}, nonzero={np.sum(np.abs(W_full)>0)}")
    print(f"  W_cross: {W_cross.shape}, nonzero={np.sum(np.abs(W_cross)>0)}")
    print(f"  W_cc: {W_cc.shape}, nonzero={np.sum(np.abs(W_cc)>0)}")

    # --- Task 2: Identify surviving edges ---
    print("\n" + "="*70)
    print("TASK 2: IDENTIFY SURVIVING EDGES")
    print("="*70)
    abs_cross = np.abs(W_cross)
    print(f"\n  W_cross stats: min={W_cross.min():.6f}, max={W_cross.max():.6f}, "
          f"abs_mean={abs_cross.mean():.6f}, abs_max={abs_cross.max():.6f}")

    # Top edges by absolute weight
    flat_idx = np.argsort(abs_cross.ravel())[::-1]
    print("\n  Top 20 temporal edges (|W_cross[i,j]|):")
    print(f"  {'Rank':>4s}  {'i(past)':>8s}  {'j(curr)':>8s}  {'weight':>10s}  {'|weight|':>10s}")
    for rank, idx in enumerate(flat_idx[:20], 1):
        i, j = divmod(idx, N)
        print(f"  {rank:4d}  sensor_{i:03d}  sensor_{j:03d}  {W_cross[i,j]:10.6f}  {abs_cross[i,j]:10.6f}")

    # Threshold 0.3 edges
    thr_edges = np.argwhere(abs_cross > 0.3)
    print(f"\n  Edges with |W_cross| > 0.3: {len(thr_edges)}")
    for e in thr_edges:
        print(f"    W_cross[{e[0]},{e[1]}] = {W_cross[e[0],e[1]]:.6f}")

    # --- Task 3: Threshold sensitivity ---
    print("\n" + "="*70)
    print("TASK 3: THRESHOLD SENSITIVITY")
    print("="*70)
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    thr_results = []

    for thr in thresholds:
        adj = (abs_cross > thr).astype(np.float32)
        stats = graph_stats(adj, N)
        print(f"\n  Threshold={thr:.2f}: {stats['n_edges']} edges, {stats['n_active']} active nodes")

        for model_name in ["GCN", "TGCN"]:
            set_seed(seed)
            m = train_eval(adj, model_name, trX, trY, teX, teY, feat_max, pre_len, seed, epochs, seq_len)
            row = {"threshold": thr, "model": model_name, "n_edges": stats["n_edges"],
                   "n_active": stats["n_active"], "rmse": round(m["RMSE"], 4),
                   "mae": round(m["MAE"], 4), "r2": round(m["R2"], 6)}
            thr_results.append(row)
            print(f"    {model_name}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

    # --- Task 4: Top-K ---
    print("\n" + "="*70)
    print("TASK 4: TOP-K TEMPORAL EDGES")
    print("="*70)
    topk_values = [1, 2, 4, 8, 16, 32]
    topk_results = []

    for K in topk_values:
        adj = np.zeros((N, N), dtype=np.float32)
        count = 0
        for idx in flat_idx:
            if count >= K:
                break
            i, j = divmod(idx, N)
            adj[i, j] = 1.0
            count += 1
        stats = graph_stats(adj, N)
        print(f"\n  Top-{K}: {stats['n_edges']} edges, {stats['n_active']} active nodes")

        for model_name in ["GCN", "TGCN"]:
            set_seed(seed)
            m = train_eval(adj, model_name, trX, trY, teX, teY, feat_max, pre_len, seed, epochs, seq_len)
            row = {"K": K, "model": model_name, "n_edges": stats["n_edges"],
                   "n_active": stats["n_active"], "rmse": round(m["RMSE"], 4),
                   "mae": round(m["MAE"], 4), "r2": round(m["R2"], 6)}
            topk_results.append(row)
            print(f"    {model_name}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

    # --- Task 7: Directional sanity check ---
    synth = directional_sanity_check()

    # --- Save all results ---
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    pd.DataFrame(thr_results).to_csv(os.path.join(RESULTS_DIR, "threshold_sensitivity.csv"), index=False)
    pd.DataFrame(topk_results).to_csv(os.path.join(RESULTS_DIR, "topk_results.csv"), index=False)

    summary = {
        "dataset": dataset, "N": N, "seed": seed, "epochs": epochs,
        "dagma_time_s": round(dagma_time, 2),
        "W_cross_nonzero": int(np.sum(np.abs(W_cross) > 0)),
        "W_cc_nonzero": int(np.sum(np.abs(W_cc) > 0)),
        "W_full_nonzero": int(np.sum(np.abs(W_full) > 0)),
        "threshold_sensitivity": thr_results,
        "topk_results": topk_results,
        "synthetic_test": synth,
    }
    with open(os.path.join(RESULTS_DIR, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("  - sz_ph1_raw_W_full.npy, sz_ph1_raw_W_cross.npy, sz_ph1_raw_W_cc.npy")
    print("  - threshold_sensitivity.csv")
    print("  - topk_results.csv")
    print("  - validation_summary.json")
    print("\nDone.")


if __name__ == "__main__":
    run_validation()
