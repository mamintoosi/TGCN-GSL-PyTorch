#!/usr/bin/env python3
"""
Stage 20.5: Split into Part A (DAGMA + analysis) and Part B (forecasting).
This script does Part A: runs DAGMA, saves raw W, identifies edges, runs synthetic test.
Part B (threshold sensitivity + Top-K) uses the saved W and is much faster.
"""
import os, sys, json, time, random
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from dagma.linear import DagmaLinear

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage20_5_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    import torch; torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_data(dataset="shenzhen", split=0.8):
    paths = {"shenzhen": ("data/sz_speed.csv", "data/sz_adj.csv"),
             "losloop": ("data/los_speed.csv", "data/los_adj.csv")}
    feat = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, paths[dataset][0])), dtype=np.float32)
    adj = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, paths[dataset][1]), header=None), dtype=np.float32)
    T, N = feat.shape
    train_size = int(T * split)
    feat_max = float(np.max(feat[:train_size]))
    return feat[:train_size] / feat_max, feat[train_size:] / feat_max, adj, feat_max, N


def gen_seq(data, seq_len, pre_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pre_len):
        X.append(data[i:i+seq_len]); Y.append(data[i+seq_len:i+seq_len+pre_len])
    return np.array(X, np.float32), np.array(Y, np.float32)


def train_eval(adj, model_name, trX, trY, teX, teY, fmax, pre_len=1, seed=42, epochs=50, seq_len=12):
    import torch
    from models.gcn import GCN
    from models.tgcn import TGCN
    from tasks.supervised import SupervisedForecastTask
    set_seed(seed)
    if model_name == "GCN":
        model = GCN(adj=adj, seq_len=seq_len, hidden_dim=64); loss_name = "mse"
    else:
        model = TGCN(adj=adj, hidden_dim=64); loss_name = "mse_with_regularizer"
    task = SupervisedForecastTask(model=model, loss=loss_name, pre_len=pre_len,
                                   learning_rate=0.001, weight_decay=0.0001, feat_max_val=fmax)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    if task.regressor: task.regressor = task.regressor.to(dev)
    opt = task.configure_optimizer()
    trX_t, trY_t = torch.FloatTensor(trX).to(dev), torch.FloatTensor(trY).to(dev)
    teX_t, teY_t = torch.FloatTensor(teX).to(dev), torch.FloatTensor(teY).to(dev)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(trX_t, trY_t), batch_size=128, shuffle=True)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = task.training_step((xb, yb)); loss.backward(); opt.step()
    train_time = time.time() - t0
    model.eval()
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(teX_t, teY_t), batch_size=len(teX_t))
    metrics = task.validation_epoch(test_loader, dev)
    metrics["train_time_s"] = round(train_time, 2)
    return metrics


def graph_stats(adj, N=None):
    if N is None: N = adj.shape[0]
    a = (adj > 0).astype(int); np.fill_diagonal(a, 0)
    n = int(np.sum(a)); deg = a.sum(axis=1)
    return {"n_edges": n, "n_active": int(np.sum(deg > 0)), "n_isolated": N - int(np.sum(deg > 0)),
            "max_deg": int(np.max(deg)), "mean_deg": round(float(np.mean(deg)), 2)}


def run_part_a():
    """Part A: DAGMA + edge identification + synthetic test. ~2.5 hours."""
    print("="*80)
    print("STAGE 20.5 PART A: DAGMA + ANALYSIS")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    train_data, test_data, adj_phys, feat_max, N = load_data("shenzhen")
    print(f"N={N}, Train={train_data.shape[0]}, Test={test_data.shape[0]}")

    # --- Build temporal DAGMA (NO thresholding) ---
    D = 2 * N
    n_samples = train_data.shape[0] - 1
    Z = np.zeros((n_samples, D), dtype=train_data.dtype)
    Z[:, 0:N] = train_data[:-1]
    Z[:, N:2*N] = train_data[1:]
    print(f"Z shape: {Z.shape} ({n_samples} samples × {D} variables)")

    t0 = time.time()
    model = DagmaLinear(loss_type='l2', verbose=False)
    W_full = model.fit(Z, lambda1=0.01, w_threshold=0.0)
    dagma_time = time.time() - t0
    print(f"DAGMA time: {dagma_time:.1f}s ({dagma_time/60:.1f} min)")

    # Save raw W
    np.save(os.path.join(RESULTS_DIR, "sz_ph1_raw_W_full.npy"), W_full)
    W_cross = W_full[N:2*N, 0:N]
    W_cc = W_full[N:2*N, N:2*N]
    np.save(os.path.join(RESULTS_DIR, "sz_ph1_raw_W_cross.npy"), W_cross)
    np.save(os.path.join(RESULTS_DIR, "sz_ph1_raw_W_cc.npy"), W_cc)

    abs_cross = np.abs(W_cross)
    print(f"\nW_full: nonzero={np.sum(np.abs(W_full)>0)}/{D*D}")
    print(f"W_cross: nonzero={np.sum(np.abs(W_cross)>0)}/{N*N}")
    print(f"W_cc: nonzero={np.sum(np.abs(W_cc)>0)}/{N*N}")

    # --- Task 2: Identify edges ---
    print("\n" + "="*70)
    print("TASK 2: TOP TEMPORAL EDGES")
    print("="*70)
    flat_idx = np.argsort(abs_cross.ravel())[::-1]
    print(f"\nTop 20 temporal edges (W_cross = past→current):")
    print(f"{'Rank':>4s}  {'i(past)':>8s}  {'j(curr)':>8s}  {'weight':>10s}  {'|w|':>10s}")
    for rank, idx in enumerate(flat_idx[:20], 1):
        i, j = divmod(idx, N)
        print(f"{rank:4d}  sensor_{i:03d}  sensor_{j:03d}  {W_cross[i,j]:10.6f}  {abs_cross[i,j]:10.6f}")

    for thr in [0.1, 0.2, 0.3, 0.5]:
        n = int(np.sum(abs_cross > thr))
        print(f"\nEdges with |W_cross| > {thr}: {n}")

    # --- Task 7: Synthetic directional test ---
    print("\n" + "="*70)
    print("TASK 7: DIRECTIONAL SANITY CHECK")
    print("="*70)
    set_seed(42)
    N_s, T_s, noise_s = 3, 2000, 0.05
    x = np.zeros((T_s, N_s), dtype=np.float32)
    x[0] = np.random.randn(N_s).astype(np.float32)
    for t in range(1, T_s):
        x[t, 0] = 0.5 * x[t-1, 0] + noise_s * np.random.randn()
        x[t, 1] = 0.5 * x[t-1, 1] + noise_s * np.random.randn()
        x[t, 2] = 0.9 * x[t-1, 0] + noise_s * np.random.randn()
    Z_s = np.zeros((T_s-1, 2*N_s), dtype=np.float32)
    Z_s[:, 0:N_s] = x[:-1]; Z_s[:, N_s:2*N_s] = x[1:]
    model_s = DagmaLinear(loss_type='l2', verbose=False)
    W_s = model_s.fit(Z_s, lambda1=0.01, w_threshold=0.0)
    W_cross_s = W_s[N_s:2*N_s, 0:N_s]
    print(f"Ground truth: x3(t) = 0.9*x1(t-1)")
    print(f"W_cross (should show [0,2] as strong):")
    for i in range(N_s):
        for j in range(N_s):
            if abs(W_cross_s[i,j]) > 0.01:
                print(f"  W_cross[{i},{j}] = {W_cross_s[i,j]:.4f}  (x{i+1}(t-1)→x{j+1}(t))")
    exp = W_cross_s[0, 2]
    print(f"\nExpected entry [0,2] = {exp:.4f}")
    if abs(exp) > 0.3:
        print("✓ DIRECTION RECOVERED CORRECTLY")
    else:
        mx = np.unravel_index(np.argmax(np.abs(W_cross_s)), W_cross_s.shape)
        print(f"Strongest: [{mx[0]},{mx[1]}] = {W_cross_s[mx]:.4f}")

    summary = {
        "dataset": "shenzhen", "N": N, "dagma_time_s": round(dagma_time, 2),
        "W_cross_nonzero": int(np.sum(abs_cross > 0)),
        "W_cc_nonzero": int(np.sum(np.abs(W_cc) > 0)),
        "W_full_nonzero": int(np.sum(np.abs(W_full) > 0)),
        "top_edges": [{"rank": r+1, "i": int(divmod(flat_idx[r], N)[0]),
                       "j": int(divmod(flat_idx[r], N)[1]),
                       "weight": float(W_cross[divmod(flat_idx[r], N)]),
                       "abs_weight": float(abs_cross[divmod(flat_idx[r], N)])}
                      for r in range(20)],
        "synthetic_test": {"expected_0_2": float(exp),
                           "strongest": [int(mx[0]), int(mx[1]), float(W_cross_s[mx])]}
    }
    with open(os.path.join(RESULTS_DIR, "part_a_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPart A done. Results in {RESULTS_DIR}/")


def run_part_b():
    """Part B: Threshold sensitivity + Top-K using saved W_cross. ~10 min."""
    print("="*80)
    print("STAGE 20.5 PART B: THRESHOLD SENSITIVITY + TOP-K")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    W_cross = np.load(os.path.join(RESULTS_DIR, "sz_ph1_raw_W_cross.npy"))
    N = W_cross.shape[0]
    abs_cross = np.abs(W_cross)
    flat_idx = np.argsort(abs_cross.ravel())[::-1]

    train_data, test_data, adj_phys, feat_max, _ = load_data("shenzhen")
    trX, trY = gen_seq(train_data, 12, 1)
    teX, teY = gen_seq(test_data, 12, 1)

    results = []

    # --- Threshold sensitivity ---
    print("\n--- THRESHOLD SENSITIVITY ---")
    for thr in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        adj = (abs_cross > thr).astype(np.float32)
        s = graph_stats(adj, N)
        print(f"\n  thr={thr:.2f}: {s['n_edges']} edges, {s['n_active']} active")
        for mn in ["GCN", "TGCN"]:
            m = train_eval(adj, mn, trX, trY, teX, teY, feat_max, 1, 42, 50, 12)
            row = {"method": "threshold", "value": thr, "model": mn,
                   "n_edges": s["n_edges"], "n_active": s["n_active"],
                   "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                   "r2": round(m["R2"], 6)}
            results.append(row)
            print(f"    {mn}: RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f}")

    # --- Top-K ---
    print("\n--- TOP-K EDGES ---")
    for K in [1, 2, 4, 8, 16, 32]:
        adj = np.zeros((N, N), dtype=np.float32)
        for ci in range(min(K, len(flat_idx))):
            i, j = divmod(flat_idx[ci], N)
            adj[i, j] = 1.0
        s = graph_stats(adj, N)
        print(f"\n  Top-{K}: {s['n_edges']} edges, {s['n_active']} active")
        for mn in ["GCN", "TGCN"]:
            m = train_eval(adj, mn, trX, trY, teX, teY, feat_max, 1, 42, 50, 12)
            row = {"method": "topk", "value": K, "model": mn,
                   "n_edges": s["n_edges"], "n_active": s["n_active"],
                   "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                   "r2": round(m["R2"], 6)}
            results.append(row)
            print(f"    {mn}: RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f}")

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "part_b_results.csv"), index=False)
    print(f"\nPart B done. Results saved to {RESULTS_DIR}/part_b_results.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["a", "b", "both"], default="both")
    args = parser.parse_args()
    if args.part in ("a", "both"):
        run_part_a()
    if args.part in ("b", "both"):
        run_part_b()
