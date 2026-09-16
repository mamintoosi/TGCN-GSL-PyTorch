#!/usr/bin/env python3
"""
DIAGNOSTIC — Reproduce and time the ORIGINAL DAGMA structure-learning protocol
==============================================================================

WHY THIS SCRIPT EXISTS
----------------------
The submitted manuscript states (archive/repo_cleanup_20260913/paper/
submitted_version/sn-article.tex, line 519, §Model Configurations):

    "The graph structure learning process leverages the DAGMA method
     [Bello2024DAGMA], which builds on prior work [zheng2018dags] to
     efficiently estimate adjacency matrices. This process is
     computationally efficient, taking approximately 2 minutes on
     standard hardware."

(The same sentence also appears, commented out, at line 966 as a
limitations-checklist bullet.)

This script tests whether that "approximately 2 minutes" figure is
reproducible, by re-running the ORIGINAL protocol exactly as committed in
the submitted code base (utils/data/spatiotemporal_csv_data.py,
`compute_adjacency_matrix`, plus utils/data/functions.py preprocessing).

WHAT THE ORIGINAL PROTOCOL IS (verified against the committed code)
-------------------------------------------------------------------
1. Dataset: data/los_speed.csv (Los-loop, T x 207) or data/sz_speed.csv
   (SZ-Taxi, T x 156), loaded with utils.data.functions.load_features
   (pd.read_csv -> float32).
2. Preprocessing (utils.data.functions.generate_torch_datasets):
   - normalize=True -> divide by max over the FULL series (np.max(data),
     train+test; on the committed data this equals the train max:
     los 70.0, sz 86.4292);
   - 80/20 chronological split (split_ratio=0.8);
   - sliding windows seq_len=12, pre_len=PH -> train_X of shape
     (num_windows, 12, N), num_windows = int(0.8*T) - 12 - PH.
3. Input matrix to DAGMA (the original two lines, verbatim):
       train_data = np.array([x[0].numpy() for x in train_dataset])
       data       = np.array([x[0] for x in train_data])
   i.e. the FIRST time step of every sliding window -> a (num_windows, N)
   matrix of contemporaneous snapshots; window k starts at time k.
4. DAGMA (verbatim from the original loop):
       for i in range(self.pre_len):
           X = data[i::self.pre_len]
           lambda1 = 0.01 if self.dataset_name == "shenzhen" else 0.02
           model = DagmaLinear(loss_type='l2')
           w_est = model.fit(X, lambda1=lambda1)
           W_est_all[:, :, i] = w_est
   i.e. ONE DagmaLinear fit per offset i in {0..PH-1} (PH fits per
   (dataset, PH)), on the stride-i subset; NO w_threshold and NO
   warm_iter/max_iter overrides are passed, so the DAGMA library defaults
   apply (the library zeroes |W| < w_threshold=0.3 INSIDE fit()).
5. Thresholding (verbatim semantics): keep positive survivors of the
   thresholded fit, A = 1(any_i(W_est_all[:,:,i] > 0)) — the original
   use_gsl=1 binary adjacency; use_gsl=2 symmetrizes it (+ its transpose).
6. The result was saved per (dataset, PH) as W_est_{dataset}_pre_len{PH}.npy
   of shape (N, N, PH). Those committed artifacts live in
   archive/historical_submission/ and are NOT touched by this script.

DEVIATIONS FROM THE ORIGINAL CODE PATH (all documented, none silent)
--------------------------------------------------------------------
(a) The original `compute_adjacency_matrix` short-circuits: if
    archive/historical_submission/W_est_{dataset}_pre_len{PH}.npy exists it
    LOADS it and never fits. To time the fit we must skip that shortcut.
    This is the ONLY behavioural deviation.
(b) Optional CLI flags --warm-iter/--max-iter, if you pass them, add
    keyword arguments the original did not pass (default: not passed).
(c) The optional --threads flag pins OMP/MKL thread counts for a cleaner
    timing environment; the original ran with default threading.

Everything else (loader, normalization, split, windowing, first-timestep
input construction, per-offset loop, lambda1 choice, bare fit() call,
thresholding) is executed via the ORIGINAL committed functions and lines.

WARNING — COMPARABILITY CAVEAT
------------------------------
The manuscript's "approximately 2 minutes" does NOT specify dataset,
horizon, or which operation is included (a single fit? the whole loop?
loading included?). The comparable quantity is the wall-clock of the
DAGMA fit() call(s) for a given (dataset, PH). If the measured fit time
is much larger than 2 minutes on this machine, that still does not prove
the original claim wrong: machine/hardware differences and the ambiguity
of the claim itself must be reported alongside the measurement. See the
comparability note printed at the end of the run.

COST WARNING
------------
Each DAGMA-linear fit on (num_windows, N) is CPU-bound and, per prior
canonical measurements, can take tens of minutes at N ~ 156-207. One
(dataset, PH) pair for PH=1 costs ONE such fit. Running every offset of
every horizon of both datasets costs 8 fits (PH+1 fits each for PH in
1..4, summed over 2 datasets) — budget accordingly.

NO SIDE EFFECTS: the script never writes .npy artifacts, never modifies
data/ or archive/, and only writes one small JSON report into a NEW
directory results/diag_original_dagma_runtime/ (optional, --no-json to
disable). Nothing is overwritten there (timestamped filenames).

Usage examples:
  python3 tools/check_original_dagma_runtime.py --help
  python3 tools/check_original_dagma_runtime.py --dataset losloop --pre-len 1
  python3 tools/check_original_dagma_runtime.py --dataset losloop shenzhen --pre-len 1 2 3 4
  python3 tools/check_original_dagma_runtime.py --dataset shenzhen --pre-len 1 --threads 8
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
import time
from datetime import datetime

# stdlib-only at module level so that `--help` works without numpy/torch
# and missing dependencies produce a clean, actionable error message.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Make the repository's packages importable. When running
# `python3 tools/check_original_dagma_runtime.py`, sys.path[0] is tools/ (the
# script's directory), NOT the repo root, so `import utils.data...` would fail
# with "No module named 'utils'" without this insert (same convention as
# src/run_gsl_canonical.py).
sys.path.insert(0, REPO_ROOT)

DATA_FILES = {
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv",
                "lambda1": 0.02, "N": 207},
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv",
                 "lambda1": 0.01, "N": 156},
}
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "diag_original_dagma_runtime")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Time the ORIGINAL DAGMA structure-learning protocol "
                    "(submitted-manuscript code path). NOTE: running this "
                    "script DOES execute DAGMA fits; use --help only to "
                    "inspect options.",
        epilog="Example: python3 tools/check_original_dagma_runtime.py "
               "--dataset losloop --pre-len 1")
    p.add_argument("--dataset", nargs="+", default=["losloop"],
                   choices=sorted(DATA_FILES.keys()),
                   help="dataset(s) to run (default: losloop)")
    p.add_argument("--pre-len", type=int, nargs="+", default=[1],
                   choices=[1, 2, 3, 4],
                   help="prediction horizon(s) PH; the original protocol runs "
                        "PH DAGMA fits (one per offset i in 0..PH-1) per pair "
                        "(default: 1 = a single fit on all training rows)")
    p.add_argument("--threads", type=int, default=None,
                   help="optional: pin OMP/MKL/torch thread counts before "
                        "importing torch (timing hygiene; the original used "
                        "default threading)")
    p.add_argument("--repeats", type=int, default=1,
                   help="repeat the offset-0 DAGMA fit this many extra times "
                        "and report each (default: 1 = no repeats; "
                        "DAGMA-linear is deterministic, so repeats only "
                        "characterize machine jitter)")
    p.add_argument("--warm-iter", type=int, default=None,
                   help="DEVIATION if passed: forward warm_iter to fit() "
                        "(the original passed nothing). Default: not passed.")
    p.add_argument("--max-iter", type=int, default=None,
                   help="DEVIATION if passed: forward max_iter to fit() "
                        "(the original passed nothing). Default: not passed.")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                   help="diagnostic output directory for the small JSON "
                        "timing report (default: results/diag_original_dagma_runtime)")
    p.add_argument("--no-json", action="store_true",
                   help="do not write the JSON timing report")
    return p.parse_args()


def banner(msg: str) -> None:
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78, flush=True)


def import_dependencies():
    """Lazy imports with clear, actionable failure messages."""
    try:
        import numpy as np
        import pandas as pd
        import torch
    except ImportError as exc:
        sys.exit(
            f"[FATAL] Missing dependency: {exc}\n"
            "This diagnostic needs the same environment as the experiments:\n"
            "    pip install numpy pandas torch\n"
            "and the dagma package (pip install dagma / dagma-linear).")
    try:
        from dagma.linear import DagmaLinear
    except ImportError as exc:
        sys.exit(
            f"[FATAL] Cannot import DAGMA: {exc}\n"
            "Install the DAGMA library used by the original pipeline "
            "(from dagma.linear import DagmaLinear).")
    return np, pd, torch, DagmaLinear


def dagma_signature_defaults(DagmaLinear):
    """Introspect the library defaults the original fit() call relied on."""
    info = {}
    try:
        init_sig = inspect.signature(DagmaLinear.__init__)
        info["DagmaLinear.__init__ defaults"] = {
            k: (v.default if v.default is not inspect.Parameter.empty else "<required>")
            for k, v in init_sig.parameters.items() if k != "self"}
    except (TypeError, ValueError):
        info["DagmaLinear.__init__ defaults"] = "<unavailable>"
    try:
        fit_sig = inspect.signature(DagmaLinear.fit)
        info["DagmaLinear.fit defaults"] = {
            k: (v.default if v.default is not inspect.Parameter.empty else "<required>")
            for k, v in fit_sig.parameters.items() if k != "self"}
    except (TypeError, ValueError):
        info["DagmaLinear.fit defaults"] = "<unavailable>"
    return info


def environment_report(torch, np) -> dict:
    cuda = torch.cuda.is_available()
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "os_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": bool(cuda),
        "cuda_device_name": (torch.cuda.get_device_name(0) if cuda else None),
        "dagma_device_note": "DagmaLinear (loss_type='l2') runs on CPU; "
                             "the original protocol never used a GPU for it.",
    }
    return env


def run_one(dataset: str, ph: int, args, np, torch, DagmaLinear,
            generate_torch_datasets, load_features) -> dict:
    cfg = DATA_FILES[dataset]
    feat_path = os.path.join(REPO_ROOT, cfg["feat"])
    if not os.path.exists(feat_path):
        sys.exit(f"[FATAL] Missing data file: {feat_path}\n"
                 "Run this diagnostic from a checkout that contains the "
                 "committed data/ directory.")

    N_expected = cfg["N"]

    # ---- 1. data loading (ORIGINAL loader: utils.data.functions.load_features)
    t0 = time.perf_counter()
    feat = load_features(feat_path)                      # pd.read_csv -> float32
    t_load = time.perf_counter() - t0
    T, N = feat.shape
    if N != N_expected:
        sys.exit(f"[FATAL] {feat_path} has N={N} columns, expected {N_expected}; "
                 "the committed dataset does not match the original protocol.")

    # ---- 2. preprocessing / input construction (ORIGINAL functions, verbatim)
    #   normalize=True -> divide by max over the FULL series (original
    #   behaviour); sliding windows seq_len=12; 80/20 chronological split.
    t0 = time.perf_counter()
    train_dataset, _val_dataset = generate_torch_datasets(
        feat, 12, ph, split_ratio=0.8, normalize=True)
    # The original two lines that build the DAGMA input (verbatim):
    train_data = np.array([x[0].numpy() for x in train_dataset])
    data = np.array([x[0] for x in train_data])          # first timestep per window
    t_construct = time.perf_counter() - t0

    global_max = float(np.max(feat))
    train_size = int(feat.shape[0] * 0.8)
    num_windows = data.shape[0]
    # Context only: train-split max (the canonical protocol's normalization).
    train_max = float(np.max(feat[:train_size]))

    print(f"\n[{dataset} PH={ph}] input matrix: shape={data.shape} "
          f"(num_windows={num_windows}, N={N}); T={T}, train_size={train_size}; "
          f"global max={global_max:.4f}, train max={train_max:.4f}")

    # ---- 3. DAGMA fits (ORIGINAL loop, verbatim semantics)
    W_est_all = np.zeros((N, N, ph))
    fit_times, repeat_times, per_offset = [], [], []
    fit_kwargs = {}
    if args.warm_iter is not None:
        fit_kwargs["warm_iter"] = args.warm_iter
    if args.max_iter is not None:
        fit_kwargs["max_iter"] = args.max_iter

    for i in range(ph):
        X = data[i::ph]
        lambda1 = 0.01 if dataset == "shenzhen" else 0.02   # original selection
        model = DagmaLinear(loss_type="l2")
        t0 = time.perf_counter()
        w_est = model.fit(X, lambda1=lambda1, **fit_kwargs)
        dt = time.perf_counter() - t0
        fit_times.append(dt)
        W_est_all[:, :, i] = w_est
        nnz = int(np.count_nonzero(w_est))
        per_offset.append({
            "offset": i, "input_rows": int(X.shape[0]),
            "lambda1": lambda1, "fit_time_s": round(dt, 2),
            "nonzero_after_fit": nnz,
            "max_abs_weight": round(float(np.abs(w_est).max()), 6),
        })
        print(f"[{dataset} PH={ph}] offset {i}: fit on {X.shape[0]} rows, "
              f"lambda1={lambda1} -> {dt:.1f}s, nonzero={nnz}")
        del model
        if args.repeats > 1 and i == 0:
            # deterministic fit; repeats only characterize machine jitter and
            # are kept OUT of the dagma_fit_s total
            for r in range(1, args.repeats):
                model = DagmaLinear(loss_type="l2")
                t0 = time.perf_counter()
                model.fit(X, lambda1=lambda1, **fit_kwargs)
                dt = time.perf_counter() - t0
                repeat_times.append(dt)
                print(f"[{dataset} PH={ph}] offset 0 repeat {r}: {dt:.1f}s")
                del model

    # ---- 4. thresholding (ORIGINAL semantics: positive survivors, PH-way merge)
    t0 = time.perf_counter()
    W_est_binary = np.any(W_est_all > 0, axis=2)   # original use_gsl=1 support
    A_gsl = np.zeros(W_est_binary.shape, dtype=int)
    A_gsl[W_est_binary > 0] = 1
    A_cgsl = A_gsl + A_gsl.T                        # original use_gsl=2 (cyclic)
    t_threshold = time.perf_counter() - t0

    t_total = t_load + t_construct + sum(fit_times) + t_threshold

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "reproduce+time the ORIGINAL DAGMA protocol "
                   "(utils/data/spatiotemporal_csv_data.py compute_adjacency_matrix)",
        "claim_under_test": "~2 minutes (submitted manuscript line 519)",
        "dataset": dataset, "ph": ph,
        "data": {
            "feat_path": cfg["feat"],
            "T": int(T), "N": int(N),
            "split_ratio": 0.8, "seq_len": 12,
            "normalization": "max over FULL series (original generate_dataset)",
            "global_max": global_max, "train_max": train_max,
            "train_size": int(train_size), "num_windows": int(num_windows),
            "dagma_input": f"first timestep of each sliding window -> "
                           f"({num_windows}, {N}); rows = data[i::{ph}]",
        },
        "dagma": {
            "model": "DagmaLinear(loss_type='l2')  [verbatim original call]",
            "fit_kwargs_passed": fit_kwargs or "(none — library defaults, as original)",
            "per_offset": per_offset,
            "n_fits_per_pair": ph,
            "total_fit_time_s": round(sum(fit_times), 2),
            "signature_defaults": dagma_signature_defaults(DagmaLinear),
        },
        "timing": {
            "data_load_s": round(t_load, 4),
            "preprocessing_input_construction_s": round(t_construct, 4),
            "dagma_fit_s": round(sum(fit_times), 2),
            "thresholding_s": round(t_threshold, 6),
            "total_s": round(t_total, 2),
            "clock": "time.perf_counter()",
            "repeats_offset0_s": ([round(t, 2) for t in repeat_times]
                                  if args.repeats > 1 else None),
        },
        "output_graph": {
            "W_est_all_shape": list(W_est_all.shape),
            "A_gsl_edges": int(A_gsl.sum()),
            "A_cgsl_directed_entries": int((A_cgsl > 0).sum()),
            "threshold_note": "fit() applied |W|<w_threshold(=0.3 library "
                              "default) internally; project kept W>0 "
                              "(original use_gsl=1).",
        },
        "environment": environment_report(torch, np),
        "deviations": [
            "skips the existing-artifact shortcut (must re-fit to time it)",
            *(["--warm-iter/--max-iter passed (original passed nothing)"]
              if fit_kwargs else []),
            *(["--threads pinning (original used default threading)"]
              if args.threads else []),
        ],
    }

    print(f"[{dataset} PH={ph}] TIMING  load={t_load:.3f}s  "
          f"preprocess={t_construct:.3f}s  DAGMA fit={sum(fit_times):.1f}s "
          f"({ph} fit(s))  threshold={t_threshold * 1000:.1f}ms  "
          f"TOTAL={t_total:.1f}s")
    print(f"[{dataset} PH={ph}] graph: A_gsl edges={int(A_gsl.sum())}, "
          f"A_cgsl directed entries={int((A_cgsl > 0).sum())}, "
          f"W_est_all shape={W_est_all.shape}")
    return report


def main() -> None:
    args = parse_args()

    if args.threads is not None:
        # must happen BEFORE torch is imported (hence lazy imports below)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = str(args.threads)

    banner("DIAGNOSTIC: original DAGMA structure-learning runtime\n"
           "Reproduces the committed submitted-version protocol "
           "(utils/data/spatiotemporal_csv_data.py).\n"
           "The manuscript's 'approximately 2 minutes' does not specify "
           "dataset/horizon/operation —\ncompare only the DAGMA fit() wall-clock "
           "against that claim (see comparability note at the end).")

    np, pd, torch, DagmaLinear = import_dependencies()
    # NOTE: importing utils.data triggers utils/data/__init__.py, which imports
    # spatiotemporal_csv_data and therefore dagma at module level; wrap it so a
    # broken dagma install still produces the clean message below.
    try:
        from utils.data.functions import generate_torch_datasets, load_features
    except ImportError as exc:
        sys.exit(
            f"[FATAL] Cannot import the original preprocessing module: {exc}\n"
            "(utils/data/__init__.py pulls in dagma via "
            "spatiotemporal_csv_data; ensure the dagma package is installed "
            "and that you run this from the repository root.)")

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    torch_version = torch.__version__
    print(f"[env] python={sys.version.split()[0]} torch={torch_version} "
          f"numpy={np.__version__} threads={torch.get_num_threads()} "
          f"cpus={os.cpu_count()} cuda={torch.cuda.is_available()}")

    all_reports = []
    for dataset in args.dataset:
        for ph in args.pre_len:
            all_reports.append(run_one(dataset, ph, args, np, torch,
                                       DagmaLinear, generate_torch_datasets,
                                       load_features))

    banner("COMPARABILITY NOTE")
    print(
        "The quantity comparable to the manuscript's 'approximately 2 minutes'\n"
        "is the wall-clock of the DagmaLinear.fit() call(s) itself (the\n"
        "'dagma_fit_s' field), not data loading or window construction.\n"
        "Interpretation rules:\n"
        "  * If dagma_fit_s is on the order of minutes for the pair you ran,\n"
        "    the claim is consistent with that (dataset, PH) on this machine.\n"
        "  * If it is far larger, the claim is NOT reproduced here — but do\n"
        "    NOT declare it wrong on that basis alone: hardware differences\n"
        "    (original table: Intel Core i3 9th-gen CPU, 16GB RAM) and the\n"
        "    claim's unspecified dataset/horizon must be reported too.\n"
        "  * Do not average across datasets or horizons; report per fit.")

    if not args.no_json:
        os.makedirs(args.out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            args.out_dir, f"original_dagma_timing_{ts}.json")
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "args": {k: v for k, v in vars(args).items()},
            "runs": all_reports,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nJSON timing report written: {out_path} "
              f"(~{os.path.getsize(out_path)} bytes; no graph matrices saved)")


if __name__ == "__main__":
    main()
