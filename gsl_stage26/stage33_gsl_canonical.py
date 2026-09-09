#!/usr/bin/env python3
"""
Stage 33 — Canonical rerun of the ORIGINAL single-graph GSL baselines
(T-GCN-GSL, GCN-GSL, and optional cyclic variant), under the exact canonical
Stage 26/29/32 training-and-evaluation pipeline.

WHY THIS RUN IS NEEDED
----------------------
The original submission's GSL results (Appendix "Original GSL/cGSL Results")
were produced by main.py + configs/*.yaml with a materially different
protocol:

    factor            original protocol                  canonical protocol
    ---------------------------------------------------------------------
    normalization     max over FULL series               max over TRAIN split
                      (Stage 19 audit: leakage path;     (feat_max=train only)
                      numerically identical here:
                      global max = train max for both
                      committed datasets)
    batch size        64                                 128
    weight decay      0                                  0.0001
    hidden dim (GCN)  100                                64 (canonical table uses 64)
    seeds             42 only                            42-46 (5 seeds)
    loss              TGCN: mse_with_regularizer         mse_with_regularizer
                      GCN:  mse                          (GCN: mse preserved)
    W provenance      committed data/W_est_*.npy,        re-learned per PH from
                      generation not reproducible        training data only
                      (Stage 19 #11)
    evaluation        torchmetrics weighted avg          identical (shared task)

The old numbers are therefore retained in the appendix as historical,
clearly-labelled results. For any MAIN-TEXT comparison involving the
original single-graph GSL, the numbers must come from this canonical rerun.

WHAT IS RE-LEARNED
------------------
DAGMA is fit on contemporaneous training snapshots X ∈ R^{T_train × N} with
the ORIGINAL formulation (single-lag special case of the multi-lag model;
no multi-lag blocking). Positive entries are kept (the original adjacency
rule A = 1(W>0), which matches the committed W_est files where min|w|>0.30,
i.e. already past any threshold); self-loops are removed. No data leakage:
only the 80% training split is used, normalized by the training maximum.

Usage:
  python gsl_stage26/stage33_gsl_canonical.py --models tgcn --ph 1            # canary-ish single run
  python gsl_stage26/stage33_gsl_canonical.py --models tgcn gcn --ph 1 2 3 4  # full PH sweep
  python gsl_stage26/stage33_gsl_canonical.py --models tgcn --ph 1 --epochs 2 # quick smoke test
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dagma.linear import DagmaLinear
from tasks.supervised import SupervisedForecastTask
from models.multigsl import binary_graph, normalize_method
from models.tgcn import TGCN
from models.gcn import GCN

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage33_gsl_canonical")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_CONFIGS = {
    "losloop": {
        "feat_path": "data/los_speed.csv",
        "adj_path": "data/los_adj.csv",
        "N": 207, "prefix": "los",
        "lambda1": 0.02,   # original protocol value for Los-loop
    },
    "shenzhen": {
        "feat_path": "data/sz_speed.csv",
        "adj_path": "data/sz_adj.csv",
        "N": 156, "prefix": "sz",
        "lambda1": 0.01,   # original protocol value for SZ-Taxi
    },
}


# ============================================================
# CANONICAL DATA / SEED / SEQUENCES (identical to stage26_validation.py)
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
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ORIGINAL-FORMULATION SINGLE-GRAPH GSL (training data only)
# ============================================================
def learn_gsl_graph(dataset, ph, seed, dagma_kwargs):
    """Fit DAGMA on contemporaneous training snapshots; return binary adjacency.

    Positive entries kept (A = 1(W>0)), self-loops removed — the original
    adjacency rule, applied to a freshly learned, provenance-clean W.
    """
    config = DATASET_CONFIGS[dataset]
    N = config["N"]
    train_norm, _, _, feat_max = load_data(dataset)  # normalized by train max only

    # Contemporaneous snapshots from the training split (original formulation)
    X = train_norm  # (T_train, N)

    np.random.seed(seed)
    t0 = time.time()
    model = DagmaLinear(loss_type="l2", verbose=False)
    W_est = model.fit(X, lambda1=config["lambda1"], w_threshold=0.0,
                      warm_iter=dagma_kwargs["warm_iter"],
                      max_iter=dagma_kwargs["max_iter"])
    runtime = time.time() - t0

    A = (W_est > 0).astype(np.float32)
    np.fill_diagonal(A, 0)
    meta = {
        "dataset": dataset, "ph": ph, "seed": seed,
        "lambda1": config["lambda1"], "loss_type": "l2",
        "warm_iter": dagma_kwargs["warm_iter"],
        "max_iter": dagma_kwargs["max_iter"],
        "feat_max": feat_max, "train_rows": int(X.shape[0]),
        "n_edges": int(A.sum()), "runtime_s": round(runtime, 1),
        "formulation": "contemporaneous single-graph (original GSL)",
        "adjacency_rule": "A = 1(W>0), self-loops removed",
    }
    return W_est.astype(np.float32), A, meta


# ============================================================
# CANONICAL TRAIN/EVAL (identical to stage32_sparse_control.py)
# ============================================================
def train_and_eval(adj, backbone, train_X, train_Y, test_X, test_Y,
                   feat_max, pre_len, seed=42, max_epochs=50, hidden_dim=64):
    """Canonical pipeline with static-graph backbones (TGCN or GCN)."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if backbone == "tgcn":
        model = TGCN(adj=adj, hidden_dim=hidden_dim)
    elif backbone == "gcn":
        # GCN consumes the whole window in one graph convolution
        model = GCN(adj=adj, seq_len=train_X.shape[1], hidden_dim=hidden_dim)
    else:
        raise ValueError(backbone)

    loss_name = "mse_with_regularizer" if backbone == "tgcn" else "mse"
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
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    return metrics


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stage 33: canonical rerun of original single-graph GSL baselines")
    parser.add_argument("--dataset", type=str, default="losloop",
                        choices=["losloop", "shenzhen"])
    parser.add_argument("--models", type=str, nargs="+", default=["tgcn"],
                        choices=["tgcn", "gcn"],
                        help="backbones to run; the GSL adjacency is shared per PH")
    parser.add_argument("--cyclic", action="store_true",
                        help="also run the symmetrized cyclic variant (cGSL, appendix only)")
    parser.add_argument("--phs", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--warm-iter", type=int, default=30000)
    parser.add_argument("--max-iter", type=int, default=60000)
    args = parser.parse_args()

    config = DATASET_CONFIGS[args.dataset]
    N = config["N"]
    dagma_kwargs = {"warm_iter": args.warm_iter, "max_iter": args.max_iter}

    print("=" * 80)
    print("STAGE 33 — CANONICAL GSL BASELINE RERUN")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset} (N={N}), models: {args.models}, PHs: {args.phs}")
    print(f"Seeds: {args.seeds}, epochs: {args.epochs}, hidden_dim: {args.hidden_dim}")
    print("=" * 80)

    train_data, test_data, adj_phys, feat_max = load_data(args.dataset)
    all_results = []
    graph_meta = []

    for ph in args.phs:
        print(f"\n{'=' * 70}\nPH={ph}\n{'=' * 70}")

        # --- learn the GSL graph ONCE per PH (training data only) ---
        W_path = os.path.join(RESULTS_DIR,
                              f"{config['prefix']}_gsl_ph{ph}_seed42_W_est.npy")
        A_path = os.path.join(RESULTS_DIR,
                              f"{config['prefix']}_gsl_ph{ph}_seed42_A_binary.npy")
        if os.path.exists(A_path):
            W_est = np.load(W_path)
            A_gsl = np.load(A_path)
            meta = {"n_edges": int(A_gsl.sum()), "reused": True}
            print(f"  GSL graph reused: {int(A_gsl.sum())} edges ({A_path})")
        else:
            W_est, A_gsl, meta = learn_gsl_graph(args.dataset, ph, 42, dagma_kwargs)
            np.save(W_path, W_est)
            np.save(A_path, A_gsl)
            print(f"  GSL graph learned: {int(A_gsl.sum())} edges "
                  f"(lambda1={meta['lambda1']}, {meta['runtime_s']}s)")
        graph_meta.append(meta)

        # cyclic (cGSL) variant — appendix-only comparison
        A_cgsl = None
        if args.cyclic:
            A_cgsl = A_gsl + A_gsl.T
            np.fill_diagonal(A_cgsl, 0)
            A_cgsl = (A_cgsl > 0).astype(np.float32)
            print(f"  cGSL (symmetrized): {int(A_cgsl.sum())} directed entries")

        train_X, train_Y = generate_sequences(train_data, 12, ph)
        test_X, test_Y = generate_sequences(test_data, 12, ph)
        print(f"  Train: {train_X.shape}, Test: {test_X.shape}")

        # graph variants to evaluate: physical, GSL, (optional cGSL)
        variants = [("physical", adj_phys), ("gsl", A_gsl)]
        if A_cgsl is not None:
            variants.append(("cgsl", A_cgsl))

        for backbone in args.models:
            for variant, adj in variants:
                for seed in args.seeds:
                    m = train_and_eval(adj, backbone, train_X, train_Y,
                                       test_X, test_Y, feat_max, ph,
                                       seed=seed, max_epochs=args.epochs,
                                       hidden_dim=args.hidden_dim)
                    label = {"tgcn": "T-GCN", "gcn": "GCN"}[backbone]
                    name = {"physical": f"{label}", "gsl": f"{label}-GSL",
                            "cgsl": f"{label}-cGSL"}[variant]
                    all_results.append({
                        "dataset": args.dataset, "ph": ph, "seed": seed,
                        "backbone": backbone, "variant": variant,
                        "method": name, "canonical_name": name,
                        "graph": variant, "n_edges": int(adj.sum()),
                        "rmse": round(m["RMSE"], 4), "mae": round(m["MAE"], 4),
                        "n_params": m["n_params"], "epochs": args.epochs,
                        "hidden_dim": args.hidden_dim,
                        "batch_size": 128, "weight_decay": 0.0001,
                        "loss": "mse_with_regularizer" if backbone == "tgcn" else "mse",
                    })
                    print(f"  {name:12s} seed={seed}: RMSE={m['RMSE']:.4f}  "
                          f"MAE={m['MAE']:.4f}  ({m['train_time_s']}s)")

    # Save
    json_path = os.path.join(RESULTS_DIR, "stage33_gsl_canonical_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": args.dataset,
            "protocol": "canonical stage26 (batch 128, wd 1e-4, feat_max=train only)",
            "gsl_graphs": graph_meta,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Summary
    print("\nSUMMARY (mean +- std over seeds)")
    for ph in args.phs:
        for name in sorted({r["method"] for r in all_results if r["ph"] == ph}):
            rmses = [r["rmse"] for r in all_results if r["method"] == name and r["ph"] == ph]
            if rmses:
                print(f"  PH={ph}  {name:12s} {np.mean(rmses):.4f} +- {np.std(rmses):.4f}  (n={len(rmses)})")


if __name__ == "__main__":
    main()
