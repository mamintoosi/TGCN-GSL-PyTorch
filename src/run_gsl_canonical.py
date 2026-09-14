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
DAGMA is fit on the same per-PH input construction as the ORIGINAL pipeline
(utils/data/spatiotemporal_csv_data.py): contemporaneous training snapshots
subsampled at every PH-th row (X = train[0::PH]), a (T_train, N) matrix —
the single-lag special case of the multi-lag model (no multi-lag blocking).

Thresholding reproduces the ORIGINAL adjacency semantics exactly. The original
code called model.fit(X, lambda1) WITHOUT w_threshold, so DAGMA's library
default w_threshold=0.3 zeroed |W|<0.3 INSIDE fit() (dagma/linear.py:
"self.W_est[np.abs(self.W_est) < w_threshold] = 0"), after which the project
kept positive entries (A = 1(W>0)). On a raw unthresholded fit, A = 1(|W|>0)
alone would keep thousands of tiny noise entries (e.g. 13,704 edges on SZ
PH1) and destroy the DAG property; the committed W_est files (all nonzero
entries in [0.31, 0.78]) confirm the effective rule was 1(|W|>=0.3).
We therefore pass w_threshold=0.3 explicitly to fit().

SUPPORT RULE (Stage 36 canonical policy): the binary adjacency retains the
NONZERO support regardless of sign, A = 1(|W| > 0), diagonal removed — i.e.
an edge is kept whenever |W_ij| >= 0.3, positive or negative. The original
pipeline discarded negative survivors via A = 1(W>0); the Stage 36 audit
showed that on every audited artifact (24 Stage 26 lag blocks, 4 archived
raw single-graph SZ fits, 8 committed W_est files) NO negative coefficient
reaches the threshold (max |negative| = 0.013), so the revised rule is
numerically identical to the original support on this data — but it is now
explicit, sign-symmetric, and recorded in provenance. The models remain
binary (the sign is not passed as an edge weight); signed graph convolution
is explicitly out of scope. No data leakage: only the 80% training split is
used, normalized by the training maximum (numerically identical to the
original global max for both committed datasets — verified: los 70.0,
sz 86.4292).

Usage:
  python gsl_stage26/stage33_gsl_canonical.py --models tgcn --phs 1            # canary-ish single run
  python gsl_stage26/stage33_gsl_canonical.py --models tgcn gcn --phs 1 2 3 4  # full PH sweep
  python gsl_stage26/stage33_gsl_canonical.py --models tgcn --phs 1 --epochs 2 --seeds 42   # smoke test
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
    """Fit DAGMA exactly as the original pipeline did; return binary adjacency.

    Input:  X = train_norm[0::ph] — the same per-PH subsampling of the training
            snapshots used by the original SpatioTemporalCSVData
            (utils/data/spatiotemporal_csv_data.py), normalized by the train
            maximum (numerically identical to the original global max).
            Documented deviation: the original loop fitted one DAGMA per
            offset i in {0..PH-1} and merged them with np.any(W>0, axis=2);
            the committed per-offset edge sets were identical (SZ PH1-4: 8
            edges at every offset), so the canonical rerun uses the offset-0
            fit (one DAGMA fit per PH) as the clean single-graph definition.
    Rule:   w_threshold=0.3 inside fit() (the original code relied on the
            library default), then the Stage 36 canonical support rule
            A = 1(|W| > 0) with self-loops removed: an edge is kept whenever
            |W_ij| >= 0.3, regardless of sign. On a raw unthresholded fit,
            A = 1(|W|>0) alone would keep thousands of near-zero noise
            entries (verified: 13,704 edges on SZ PH1) and destroy the DAG
            property. The original pipeline used A = 1(W>0) (positive only);
            the audited artifacts contain no negative survivor at |W| >= 0.3,
            so both rules coincide numerically on this data (recorded in
            provenance). Models stay binary — no signed convolution.
    """
    config = DATASET_CONFIGS[dataset]
    N = config["N"]
    train_norm, _, _, feat_max = load_data(dataset)  # normalized by train max only

    # Per-PH subsampling of contemporaneous snapshots (original construction)
    X = train_norm[0::ph]  # (ceil(T_train/ph), N)

    np.random.seed(seed)
    t0 = time.time()
    model = DagmaLinear(loss_type="l2", verbose=False)
    # w_threshold=0.3 = the ORIGINAL effective protocol (library default that
    # the original fit() call relied on); see module docstring for evidence.
    W_est = model.fit(X, lambda1=config["lambda1"], w_threshold=0.3,
                      warm_iter=dagma_kwargs["warm_iter"],
                      max_iter=dagma_kwargs["max_iter"])
    runtime = time.time() - t0

    # Stage 36 canonical support rule: absolute-magnitude support (both signs).
    # After fit()'s internal threshold, nonzero <=> |W| >= 0.3.
    A = (np.abs(W_est) > 0).astype(np.float32)
    np.fill_diagonal(A, 0)
    n_pos = int((W_est > 0).sum())
    n_neg = int((W_est < 0).sum())
    meta = {
        "dataset": dataset, "ph": ph, "seed": seed,
        "lambda1": config["lambda1"], "loss_type": "l2",
        "w_threshold": 0.3,  # original protocol: DAGMA library default
        "threshold_rule": "abs(W) >= w_threshold, applied inside DAGMA fit()",
        "support_rule": "A = 1(|W| > 0), diagonal removed (Stage 36 canonical "
                        "policy; negative survivors retained as edges)",
        "dagma_input": f"train_norm[0::{ph}] (original per-PH subsampling)",
        "warm_iter": dagma_kwargs["warm_iter"],
        "max_iter": dagma_kwargs["max_iter"],
        "feat_max": feat_max, "train_rows": int(X.shape[0]),
        "n_edges": int(A.sum()), "runtime_s": round(runtime, 1),
        "formulation": "contemporaneous single-graph, per-PH subsampled (original GSL)",
        "freshly_fitted": True,
        # --- full provenance (Stage 35 Goal 5 / Stage 36 Goal 5) ---
        "n_nodes": N,
        "n_input_rows": int(X.shape[0]),
        "n_coefficients_surviving_abs_threshold": int((W_est != 0).sum()),
        "n_positive_surviving": n_pos,
        "n_negative_surviving": n_neg,
        "n_final_binary_edges": int(A.sum()),
        "n_diagonal_removed": int((np.diagonal(W_est) != 0).sum()),
        "max_abs_weight": round(float(np.abs(W_est).max()), 6),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "dagma_file": __import__("dagma.linear", fromlist=["x"]).__file__,
        },
        "determinism_note": "DAGMA-linear is deterministic (zero-init, no RNG); "
                            "see doc/STAGE35_DAGMA_DETERMINISM_AND_PROVENANCE_REPORT.md",
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
    parser.add_argument("--warm-iter", type=int, default=30000,
                        help="DAGMA warm-start iterations (original pipeline used "
                             "the library defaults: 30000/60000)")
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
                    # Canonical manuscript name (doc/METHOD_NAMING_MAP.md):
                    # the road-adjacency T-GCN row is "Physical"; the GSL
                    # family keeps its descriptive labels.
                    canon = {"physical": "Physical", "gsl": f"{label}-GSL",
                             "cgsl": f"{label}-cGSL"}[variant]
                    all_results.append({
                        "dataset": args.dataset, "ph": ph, "seed": seed,
                        "backbone": backbone, "variant": variant,
                        "method": name, "canonical_name": canon,
                        "graph": variant,
                        # NOTE: count nonzeros, not the weight sum — los_adj.csv
                        # stores fractional edge weights (~0.1), so adj.sum()
                        # is NOT the edge count.
                        "n_edges": int((np.asarray(adj) > 0).sum()),
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
