#!/usr/bin/env python3
"""
Stage 20.5 Phase B: Analyze temporal DAGMA results.
Loads saved W matrices from Phase A, then runs:
  - Edge identification and threshold sensitivity
  - Top-K edge experiment
  - Directional sanity check (synthetic)
  - Training comparison (threshold sweep + Top-K)
Estimated time: ~5-8 min (24 small training runs, no DAGMA).
"""
import os, sys, time, json, random
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage20_5_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

from dagma.linear import DagmaLinear
from models.gcn import GCN
from models.tgcn import TGCN
from tasks.supervised import SupervisedForecastTask


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
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len:i + seq_len + pre_len])
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
    if task.regressor:
        task.regressor = task.regressor.to(dev)
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


def graph_stats(adj, N):
    a = (adj > 0).astype(int)
    np.fill_diagonal(a, 0)
    n = int(np.sum(a))
    deg = a.sum(axis=1)
    return {"n_edges": n, "n_active": int(np.sum(deg > 0)), "n_isolated": N - int(np.sum(deg > 0))}


def directional_sanity_check():
    """Synthetic test: x3(t) = 0.9*x1(t-1) + noise"""
    print("\n" + "=" * 70)
    print("DIRECTIONAL SANITY CHECK")
    print("=" * 70)
    set_seed(42)
    N = 3
    T = 500
    noise = 0.05

    x = np.zeros((T, N), dtype=np.float32)
    x[0] = np.random.randn(N).astype(np.float32)
    for t in range(1, T):
        x[t, 0] = 0.5 * x[t - 1, 0] + noise * np.random.randn()
        x[t, 1] = 0.5 * x[t - 1, 1] + noise * np.random.randn()
        x[t, 2] = 0.9 * x[t - 1, 0] + noise * np.random.randn()  # x3(t) from x1(t-1)

    Z = np.zeros((T - 1, 2 * N), dtype=np.float32)
    Z[:, 0:N] = x[:-1]
    Z[:, N:2 * N] = x[1:]

    print(f"  Ground truth: x3(t) = 0.9*x1(t-1) + noise")
    print(f"  Z shape: {Z.shape}")

    model = DagmaLinear(loss_type='l2', verbose=False)
    W = model.fit(Z, lambda1=0.01, w_threshold=0.0)

    W_cross = W[N:2 * N, 0:N]
    W_cc = W[N:2 * N, N:2 * N]

    print(f"\n  W_cross (past→current):")
    for i in range(N):
        for j in range(N):
            if abs(W_cross[i, j]) > 0.01:
                print(f"    W_cross[{i},{j}] = {W_cross[i, j]:.4f}  (x{i+1}(t-1) → x{j+1}(t))")

    expected = W_cross[0, 2]
    print(f"\n  Expected strong entry W_cross[0,2] (x1→x3): {expected:.4f}")
    if abs(expected) > 0.5:
        print("  ✓ DIRECTION CORRECT: x1(t-1) → x3(t) is the strongest cross-time edge")
    else:
        max_idx = np.unravel_index(np.argmax(np.abs(W_cross)), W_cross.shape)
        print(f"  ⚠ Entry [0,2] = {expected:.4f}")
        print(f"  Strongest entry: W_cross[{max_idx[0]},{max_idx[1]}] = {W_cross[max_idx]:.4f}")

    return {"W_cross": W_cross.tolist(), "expected_entry_0_2": float(expected)}


def main():
    print("=" * 80)
    print("STAGE 20.5 PHASE B: ANALYSIS & TRAINING COMPARISON")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # --- Load saved matrices from Phase A ---
    W_thresh = np.load(os.path.join(RESULTS_DIR, "sz_ph1_W_thresh_temporal.npy"))
    W_raw = np.load(os.path.join(RESULTS_DIR, "sz_ph1_W_raw_temporal.npy"))
    W_cross = np.load(os.path.join(RESULTS_DIR, "sz_ph1_W_cross_raw.npy"))
    W_cc = np.load(os.path.join(RESULTS_DIR, "sz_ph1_W_cc_raw.npy"))
    W_orig = np.load(os.path.join(RESULTS_DIR, "sz_ph1_W_orig_contemp.npy"))

    with open(os.path.join(RESULTS_DIR, "phase_a_metadata.json")) as f:
        meta = json.load(f)

    N = meta["N"]
    print(f"Loaded Phase A results: N={N}, W_raw nonzero={meta['W_raw_nonzero']}")
    print(f"  W_cross nonzero: {meta['W_cross_nonzero']}, W_cc nonzero: {meta['W_cc_nonzero']}")
    print(f"  W_orig nonzero: {meta['W_orig_nonzero']}")
    print(f"  Phase A time: {meta['time_thresh_s'] + meta['time_raw_s'] + meta['time_orig_s']:.1f}s")

    # --- Load data for training ---
    train_data, test_data, adj_phys, feat_max, N = load_data("shenzhen")
    seq_len = 12
    pre_len = 1
    trX, trY = gen_seq(train_data, seq_len, pre_len)
    teX, teY = gen_seq(test_data, seq_len, pre_len)
    print(f"Training data: trX={trX.shape}, teX={teX.shape}")

    seed = 42
    epochs = 50
    all_results = []

    # ================================================================
    # TASK: Threshold sensitivity on temporal DAGMA
    # ================================================================
    print("\n" + "=" * 70)
    print("TASK: THRESHOLD SENSITIVITY (temporal DAGMA)")
    print("=" * 70)

    abs_cross = np.abs(W_cross)
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    for thr in thresholds:
        adj = (abs_cross > thr).astype(np.float32)
        stats = graph_stats(adj, N)
        print(f"\n  Threshold={thr:.3f}: {stats['n_edges']} edges, {stats['n_active']} active nodes")

        for model_name in ["GCN", "TGCN"]:
            set_seed(seed)
            m = train_eval(adj, model_name, trX, trY, teX, teY, feat_max, pre_len, seed, epochs, seq_len)
            row = {"method": f"TempDAGMA_thr{thr}", "threshold": thr, "model": model_name,
                   "n_edges": stats["n_edges"], "n_active": stats["n_active"],
                   "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                   "r2": round(m["R2"], 6), "train_time_s": m["train_time_s"]}
            all_results.append(row)
            print(f"    {model_name}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

    # ================================================================
    # TASK: Top-K temporal edges
    # ================================================================
    print("\n" + "=" * 70)
    print("TASK: TOP-K TEMPORAL EDGES")
    print("=" * 70)

    flat_idx = np.argsort(abs_cross.ravel())[::-1]
    topk_values = [1, 2, 4, 8, 16, 32]

    for K in topk_values:
        adj = np.zeros((N, N), dtype=np.float32)
        for count, idx in enumerate(flat_idx[:K]):
            i, j = divmod(idx, N)
            adj[i, j] = 1.0
        stats = graph_stats(adj, N)
        print(f"\n  Top-{K}: {stats['n_edges']} edges, {stats['n_active']} active nodes")

        for model_name in ["GCN", "TGCN"]:
            set_seed(seed)
            m = train_eval(adj, model_name, trX, trY, teX, teY, feat_max, pre_len, seed, epochs, seq_len)
            row = {"method": f"TempDAGMA_top{K}", "K": K, "model": model_name,
                   "n_edges": stats["n_edges"], "n_active": stats["n_active"],
                   "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                   "r2": round(m["R2"], 6), "train_time_s": m["train_time_s"]}
            all_results.append(row)
            print(f"    {model_name}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

    # ================================================================
    # Baselines (using Stage 20 existing results or retrain)
    # ================================================================
    print("\n" + "=" * 70)
    print("BASELINES")
    print("=" * 70)

    baselines = {
        "Physical": adj_phys.astype(np.float32),
        "OriginalDAGMA": (np.abs(W_orig) > 0.3).astype(np.float32),
        "TempDAGMA_thr0.3": (abs_cross > 0.3).astype(np.float32),
    }

    for name, adj in baselines.items():
        stats = graph_stats(adj, N)
        print(f"\n  {name}: {stats['n_edges']} edges, {stats['n_active']} active nodes")
        for model_name in ["GCN", "TGCN"]:
            set_seed(seed)
            m = train_eval(adj, model_name, trX, trY, teX, teY, feat_max, pre_len, seed, epochs, seq_len)
            row = {"method": name, "model": model_name,
                   "n_edges": stats["n_edges"], "n_active": stats["n_active"],
                   "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                   "r2": round(m["R2"], 6), "train_time_s": m["train_time_s"]}
            all_results.append(row)
            print(f"    {model_name}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

    # --- Directional sanity check ---
    synth = directional_sanity_check()

    # --- Save results ---
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "phase_b_results.csv")
    df.to_csv(csv_path, index=False)

    # Summary table
    print("\n=== SUMMARY TABLE ===")
    pivot = df.pivot_table(index=["method", "n_edges"], columns="model", values=["rmse", "mae"])
    print(pivot.to_string())

    # Save full summary
    summary = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "dataset": "shenzhen", "PH": 1, "seed": seed, "epochs": epochs,
        "N": N,
        "phase_a_times": {
            "thresholded_s": meta["time_thresh_s"],
            "raw_s": meta["time_raw_s"],
            "original_s": meta["time_orig_s"],
        },
        "results": all_results,
        "synthetic_test": synth,
    }
    with open(os.path.join(RESULTS_DIR, "phase_b_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {RESULTS_DIR}/")
    print(f"  phase_b_results.csv")
    print(f"  phase_b_summary.json")
    print(f"\nPHASE B COMPLETE")


if __name__ == "__main__":
    main()
