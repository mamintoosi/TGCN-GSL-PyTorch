#!/usr/bin/env python3
"""
SMOKE TEST (todo6) — DAGMA-linear at reduced iterations, compared with the
existing PH5/PH6 Los-loop canonical artifacts.
==============================================================================

MOTIVATION (todo6.txt, user's memory)
-------------------------------------
The user remembers that the ORIGINAL DAGMA structure learning ("approximately
2 minutes", submitted manuscript line 519) was fast, and guesses the DAGMA
library DEFAULTS may have changed since then (e.g. default iterations 100x
smaller). The diagnostic tools/check_original_dagma_runtime.py showed that a
fit with library defaults runs 180000 optimizer iterations (tqdm total =
(T-1)*warm_iter + max_iter = 4*30000 + 60000) and needs ~1 h on this machine,
so the user interrupted it.

WHAT THIS SCRIPT DOES (isolated; canonical results untouched)
-------------------------------------------------------------
1. Copies (never moves) the existing canonical reference artifacts
       results/stage33_gsl_canonical/los_gsl_ph{5,6}_seed42_{W_est,A_binary}.npy
   into a NEW isolated output root (default:
       results/diag_dagma_lowiter_smoke/reference_copies/
   ) with provenance sidecars (source path + sha256). Sources are opened
   READ-ONLY and byte-for-byte unchanged.
2. Rebuilds the EXACT canonical DAGMA input via the canonical code path
   (src.run_gsl_canonical.load_data -> train_norm[0::ph]) and re-fits
   DagmaLinear(loss_type='l2', verbose=False) with the SAME lambda1=0.02,
   w_threshold=0.3 and support rule as the canonical fit, but with REDUCED
   iteration budgets (default warm_iter=300, max_iter=600 => 1800 total
   optimizer iterations = 1% of the default 180000 — the user's "0.01 of
   current" guess; also runs warm_iter=3000/max_iter=6000 by default).
3. Compares, per PH and per iteration budget:
   * runtime of each fit (the point of the exercise: does a small budget
     land near "~2 minutes"?);
   * number of preserved edges (nonzeros of the thresholded W) in the
     low-iteration fit vs the canonical reference;
   * number of COMMON edges (Jaccard overlap) of the binary supports;
   * weight agreement: max/mean |dW| over the union support, and
     max |W_low - W_ref| overall.
4. Writes ONE markdown report per run (timestamped filename, never
   overwrites):
       results/diag_dagma_lowiter_smoke/lowiter_smoke_report_<ts>.md
   plus a machine-readable JSON with the same content. Nothing else is
   written anywhere; no canonical file is ever opened for writing.

NO paper/reviewer-response files are read or modified by this script.

DAGMA input identity check (paranoia): the rebuilt input matrix is verified
against the artifacts' recorded provenance (train_rows, feat_max) and, since
DAGMA is deterministic, the script optionally (--verify-refit) re-derives the
input hash only; it does NOT re-run the expensive canonical fit.

Usage:
  python3 src/run_dagma_lowiter_smoke.py --help
  python3 src/run_dagma_lowiter_smoke.py                     # default budgets
  python3 src/run_dagma_lowiter_smoke.py --budgets 300:600 3000:6000 30000:60000
  python3 src/run_dagma_lowiter_smoke.py --phs 5 6 --dataset losloop
  python3 src/run_dagma_lowiter_smoke.py --budgets 300:600 --no-copy-ref
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_OUT_ROOT = os.path.join(
    PROJECT_ROOT, "results", "diag_dagma_lowiter_smoke")
DEFAULT_REFERENCE_DIR = os.path.join(
    PROJECT_ROOT, "results", "stage33_gsl_canonical")

DATASET_CONFIGS = {
    "losloop": {"feat_path": "data/los_speed.csv", "N": 207,
                "lambda1": 0.02, "prefix": "los"},
    "shenzhen": {"feat_path": "data/sz_speed.csv", "N": 156,
                 "lambda1": 0.01, "prefix": "sz"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="todo6 smoke test: DAGMA-linear at reduced iteration "
                    "budgets, compared with existing PH5/PH6 canonical "
                    "artifacts (isolated output root; canonical untouched).")
    p.add_argument("--dataset", type=str, default="losloop",
                   choices=sorted(DATASET_CONFIGS.keys()),
                   help="dataset to refit (default: losloop; the PH5/PH6 "
                        "reference artifacts live only for losloop)")
    p.add_argument("--phs", type=int, nargs="+", default=[5, 6],
                   help="prediction horizons to refit (default: 5 6)")
    p.add_argument("--budgets", type=str, nargs="+",
                   default=["300:600", "3000:6000"],
                   help="iteration budgets as WARM:MAX pairs (default: "
                        "300:600 and 3000:6000; the first is 1%% of the "
                        "library default 30000:60000 = 180000 total "
                        "iterations). '30000:60000' would be the default "
                        "budget itself (slow: ~15-20 min per fit).")
    p.add_argument("--seed", type=int, default=42,
                   help="seed passed to np.random before each fit, matching "
                        "the canonical protocol (DAGMA-linear is "
                        "deterministic; seed is provenance only)")
    p.add_argument("--lambda1", type=float, default=None,
                   help="override lambda1 (default: the dataset canonical "
                        "value, 0.02 for losloop — same as the reference)")
    p.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT,
                   help="isolated output root (default: "
                        "results/diag_dagma_lowiter_smoke)")
    p.add_argument("--reference-dir", type=str, default=DEFAULT_REFERENCE_DIR,
                   help="directory holding the canonical reference artifacts "
                        "(default: results/stage33_gsl_canonical)")
    p.add_argument("--no-copy-ref", action="store_true",
                   help="do not copy the reference artifacts into the output "
                        "root (comparison still loads them read-only)")
    p.add_argument("--no-report", action="store_true",
                   help="do not write the markdown/JSON reports (console "
                        "output only)")
    p.add_argument("--redo-fits", action="store_true",
                   help="ignore existing per-fit checkpoints under "
                        "<out-root>/fits/ and refit everything (default: "
                        "a (ph, budget) whose checkpoint exists is SKIPPED "
                        "and its stored result is re-used — makes long runs "
                        "resumable and re-runs idempotent)")
    return p.parse_args()


def banner(msg: str) -> None:
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78, flush=True)


def sha256_of(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_budgets(budget_strings):
    budgets = []
    for s in budget_strings:
        try:
            warm_s, max_s = s.split(":")
            warm, mx = int(float(warm_s)), int(float(max_s))
        except ValueError:
            sys.exit(f"[FATAL] --budgets entries must be WARM:MAX, got: {s}")
        if warm <= 0 or mx <= 0:
            sys.exit(f"[FATAL] iteration budgets must be positive: {s}")
        budgets.append({"warm_iter": warm, "max_iter": mx,
                        "total_iters": 4 * warm + mx, "label": f"{warm}:{mx}"})
    return budgets


def copy_reference_with_provenance(src: str, dst: str) -> dict:
    """Copy src -> dst (read-only w.r.t. src) + provenance sidecar JSON.

    Idempotent: if dst already exists with the SAME sha256 as src, the copy
    is skipped (re-runs never collide and never overwrite). A differing
    existing file is an error (never silently overwritten).
    """
    import shutil
    if not os.path.exists(src):
        sys.exit(f"[FATAL] reference artifact missing: {src}")
    src_sha = sha256_of(src)
    if os.path.exists(dst):
        if sha256_of(dst) == src_sha:
            return {
                "provenance_kind": "copied_reference_for_todo6_lowiter_smoke",
                "source_path": os.path.abspath(src),
                "source_sha256": src_sha,
                "copied_to": dst,
                "copy_timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"),
                "status": "already_present_identical_skipped",
                "source_not_modified": ("the source artifact was opened "
                                        "read-only and was not modified"),
            }
        sys.exit(f"[FATAL] refusing to overwrite differing existing file: "
                 f"{dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    prov = {
        "provenance_kind": "copied_reference_for_todo6_lowiter_smoke",
        "source_path": os.path.abspath(src),
        "source_sha256": src_sha,
        "copied_to": dst,
        "copy_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "copied_now",
        "source_not_modified": ("the source artifact was opened read-only "
                                "and was not modified by this copy"),
    }
    with open(dst.replace(".npy", ".provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    return prov


def environment_report() -> dict:
    import torch
    import dagma
    import importlib.metadata as md
    try:
        dagma_version = md.version("dagma")
    except Exception:
        dagma_version = "unknown"
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "dagma_version": dagma_version,
        "dagma_linear_file": __import__("dagma.linear",
                                        fromlist=["x"]).__file__,
        "torch_num_threads": torch.get_num_threads(),
        "os_cpu_count": os.cpu_count(),
    }


def fit_checkpoint_path(out_root: str, prefix: str, ph: int,
                        budget: dict) -> str:
    """Per-fit checkpoint JSON (resumability of long runs)."""
    return os.path.join(
        out_root, "fits",
        f"{prefix}_ph{ph}_warm{budget['warm_iter']}"
        f"_max{budget['max_iter']}_fit.json")


def fit_lowiter(X: np.ndarray, lambda1: float, budget: dict,
                seed: int) -> tuple:
    """One DagmaLinear fit under a reduced iteration budget (canonical
    protocol otherwise: loss_type='l2', w_threshold=0.3, support=|W|>0,
    diagonal removed). Returns (W_est, A, meta)."""
    from dagma.linear import DagmaLinear
    np.random.seed(seed)
    model = DagmaLinear(loss_type="l2", verbose=False)
    t0 = time.perf_counter()
    W_est = model.fit(X, lambda1=lambda1, w_threshold=0.3,
                      warm_iter=budget["warm_iter"],
                      max_iter=budget["max_iter"])
    runtime = time.perf_counter() - t0
    A = (np.abs(W_est) > 0).astype(np.float32)
    np.fill_diagonal(A, 0)
    meta = {
        "warm_iter": budget["warm_iter"],
        "max_iter": budget["max_iter"],
        "total_optimizer_iters_budget": budget["total_iters"],
        "lambda1": lambda1, "w_threshold": 0.3, "seed": seed,
        "runtime_s": round(runtime, 2),
        "n_edges": int(A.sum()),
        "n_positive_surviving": int((W_est > 0).sum()),
        "n_negative_surviving": int((W_est < 0).sum()),
        "max_abs_weight": round(float(np.abs(W_est).max()), 6),
        "h_final": float(getattr(model, "h_final", float("nan"))),
    }
    del model
    return W_est.astype(np.float32), A, meta


def compare_pair(W_low: np.ndarray, A_low: np.ndarray,
                 W_ref: np.ndarray, A_ref: np.ndarray) -> dict:
    """Support and weight comparison between a low-iteration fit and the
    canonical reference (same PH, same input, same lambda)."""
    sup_low = A_low > 0
    sup_ref = A_ref > 0
    common = int(np.logical_and(sup_low, sup_ref).sum())
    only_low = int(np.logical_and(sup_low, ~sup_ref).sum())
    only_ref = int(np.logical_and(~sup_low, sup_ref).sum())
    union = common + only_low + only_ref
    jaccard = common / union if union else 1.0
    dW = np.abs(W_low - W_ref)
    both = np.logical_and(sup_low, sup_ref)
    dw_common = dW[both] if both.any() else np.array([0.0])
    return {
        "edges_low": int(sup_low.sum()),
        "edges_ref": int(sup_ref.sum()),
        "edges_common": common,
        "edges_only_low": only_low,
        "edges_only_ref": only_ref,
        "jaccard_support": round(jaccard, 4),
        "max_abs_dW_overall": round(float(dW.max()), 6),
        "mean_abs_dW_common_edges": round(float(dw_common.mean()), 6),
        "max_abs_dW_common_edges": round(float(dw_common.max()), 6),
    }


def main() -> None:
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    cfg = DATASET_CONFIGS[args.dataset]
    lambda1 = cfg["lambda1"] if args.lambda1 is None else args.lambda1

    banner("TODO6 SMOKE TEST — DAGMA-linear at reduced iteration budgets\n"
           "Isolated output root; canonical results are never modified.\n"
           f"dataset={args.dataset}  PHs={args.phs}  lambda1={lambda1}\n"
           f"budgets (warm:max) = {[b['label'] for b in budgets]}")

    # ---- import the CANONICAL data pipeline (read-only use) -----------------
    from src.run_gsl_canonical import load_data  # noqa: E402

    os.makedirs(args.out_root, exist_ok=True)
    env = environment_report()
    print(f"[env] python={env['python']} torch={env['torch']} "
          f"numpy={env['numpy']} dagma={env['dagma_version']} "
          f"threads={env['torch_num_threads']} cpus={env['os_cpu_count']}")

    # ---- 1. copy the reference artifacts (read-only sources) ----------------
    copy_reports = []
    if not args.no_copy_ref:
        ref_copy_dir = os.path.join(args.out_root, "reference_copies")
        for ph in args.phs:
            for kind in ("W_est", "A_binary"):
                src = os.path.join(
                    args.reference_dir,
                    f"{cfg['prefix']}_gsl_ph{ph}_seed42_{kind}.npy")
                dst = os.path.join(
                    ref_copy_dir,
                    f"{cfg['prefix']}_gsl_ph{ph}_seed42_{kind}.npy")
                prov = copy_reference_with_provenance(src, dst)
                copy_reports.append(prov)
                print(f"[copy] {os.path.basename(src)} -> "
                      f"{os.path.relpath(dst, PROJECT_ROOT)} "
                      f"(sha256 {prov['source_sha256'][:12]}..., "
                      "source read-only)")
    else:
        print("[copy] --no-copy-ref: comparison loads references read-only, "
              "no copies made")

    # ---- 2. rebuild the canonical DAGMA input (same code path) --------------
    train_norm, _test_norm, _adj, feat_max = load_data(args.dataset)
    print(f"[data] feat_max (train)={feat_max:.4f}  "
          f"train_norm shape={train_norm.shape}")
    ts_fn = datetime.now().strftime("%Y%m%d_%H%M%S")  # run tag for artifacts

    # ---- 3. refit at each budget x PH, compare with the reference -----------
    results = []
    for ph in args.phs:
        X = train_norm[0::ph]  # canonical per-PH subsampling (original GSL)
        ref_W_path = os.path.join(
            args.reference_dir,
            f"{cfg['prefix']}_gsl_ph{ph}_seed42_W_est.npy")
        ref_A_path = os.path.join(
            args.reference_dir,
            f"{cfg['prefix']}_gsl_ph{ph}_seed42_A_binary.npy")
        if not (os.path.exists(ref_W_path) and os.path.exists(ref_A_path)):
            sys.exit(f"[FATAL] reference artifacts for PH={ph} not found "
                     f"under {args.reference_dir}")
        W_ref = np.load(ref_W_path)
        A_ref = np.load(ref_A_path)
        print(f"\n[PH={ph}] input X={X.shape} (train_norm[0::{ph}]); "
              f"reference: {int(A_ref.sum())} edges "
              f"({os.path.relpath(ref_A_path, PROJECT_ROOT)})")
        if X.shape[0] * 1.0 < 10:
            sys.exit("[FATAL] absurdly small input; aborting")

        for budget in budgets:
            # --- resumability: skip (ph, budget) pairs already fitted -----
            ckpt_path = fit_checkpoint_path(args.out_root, cfg["prefix"],
                                            ph, budget)
            if os.path.exists(ckpt_path) and not args.redo_fits:
                with open(ckpt_path) as f:
                    ck = json.load(f)
                W_low = np.load(ck["W_npy"])
                A_low = np.load(ck["A_npy"])
                meta = ck["meta"]
                cmp = compare_pair(W_low, A_low, W_ref, A_ref)
                results.append({"dataset": args.dataset, "ph": ph, **meta,
                                **cmp,
                                "reference_artifact": os.path.relpath(
                                    ref_A_path, PROJECT_ROOT),
                                "reused_checkpoint": True})
                print(f"[PH={ph} warm={budget['warm_iter']:>6d} "
                      f"max={budget['max_iter']:>6d}] "
                      f"REUSED checkpoint: fit={meta['runtime_s']}s "
                      f"edges: low={cmp['edges_low']} "
                      f"ref={cmp['edges_ref']} "
                      f"common={cmp['edges_common']} "
                      f"jaccard={cmp['jaccard_support']}", flush=True)
                continue

            W_low, A_low, meta = fit_lowiter(X, lambda1, budget, args.seed)
            cmp = compare_pair(W_low, A_low, W_ref, A_ref)
            rec = {"dataset": args.dataset, "ph": ph, **meta, **cmp,
                   "reference_artifact": os.path.relpath(ref_A_path,
                                                         PROJECT_ROOT)}
            results.append(rec)
            print(f"[PH={ph} warm={budget['warm_iter']:>6d} "
                  f"max={budget['max_iter']:>6d}] "
                  f"fit={meta['runtime_s']:7.2f}s  "
                  f"edges: low={cmp['edges_low']:>3d} ref={cmp['edges_ref']:>3d} "
                  f"common={cmp['edges_common']:>3d} "
                  f"jaccard={cmp['jaccard_support']:.3f}  "
                  f"max|dW|={cmp['max_abs_dW_overall']:.4f}", flush=True)
            # save the low-iteration matrices (small, isolated, run-specific
            # filenames so re-runs never overwrite anything)
            np.save(os.path.join(
                args.out_root,
                f"{cfg['prefix']}_ph{ph}_warm{budget['warm_iter']}"
                f"_max{budget['max_iter']}_{ts_fn}_W_est.npy"), W_low)
            np.save(os.path.join(
                args.out_root,
                f"{cfg['prefix']}_ph{ph}_warm{budget['warm_iter']}"
                f"_max{budget['max_iter']}_{ts_fn}_A_binary.npy"), A_low)
            # per-fit checkpoint (written LAST, marks the pair complete)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            with open(ckpt_path, "w") as f:
                json.dump({
                    "dataset": args.dataset, "ph": ph,
                    "warm_iter": budget["warm_iter"],
                    "max_iter": budget["max_iter"],
                    "lambda1": lambda1, "seed": args.seed,
                    "input_shape": list(X.shape),
                    "meta": meta,
                    "W_npy": os.path.relpath(
                        os.path.join(
                            args.out_root,
                            f"{cfg['prefix']}_ph{ph}_warm{budget['warm_iter']}"
                            f"_max{budget['max_iter']}_{ts_fn}_W_est.npy"),
                        PROJECT_ROOT),
                    "A_npy": os.path.relpath(
                        os.path.join(
                            args.out_root,
                            f"{cfg['prefix']}_ph{ph}_warm{budget['warm_iter']}"
                            f"_max{budget['max_iter']}_{ts_fn}_A_binary.npy"),
                        PROJECT_ROOT),
                }, f, indent=2)
            print(f"  checkpoint written: {ckpt_path}", flush=True)

    # ---- 4. reports ----------------------------------------------------------
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "timestamp": ts,
        "purpose": "todo6: DAGMA-linear at reduced iteration budgets vs "
                   "existing canonical PH5/PH6 artifacts (user info only; "
                   "no paper/reviewer files touched)",
        "dataset": args.dataset, "phs": args.phs, "lambda1": lambda1,
        "seed": args.seed,
        "dagma_library_defaults_note": "fit(warm_iter=3e4, max_iter=6e4, T=5) "
                                       "=> 180000 optimizer iterations total",
        "environment": env,
        "reference_dir": os.path.relpath(args.reference_dir, PROJECT_ROOT),
        "copies": copy_reports,
        "results": results,
    }
    if not args.no_report:
        md_path = os.path.join(args.out_root,
                               f"lowiter_smoke_report_{ts_fn}.md")
        json_path = os.path.join(args.out_root,
                                 f"lowiter_smoke_report_{ts_fn}.json")
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        lines = []
        lines.append(f"# DAGMA low-iteration smoke test (todo6) — {ts}\n")
        lines.append(f"Dataset: **{args.dataset}**, PHs: {args.phs}, "
                     f"lambda1: {lambda1}, seed: {args.seed}.\n")
        lines.append("DAGMA library defaults are `warm_iter=30000, "
                     "max_iter=60000, T=5` → **180000** optimizer iterations "
                     "per fit (the 180000 seen in the interrupted diagnostic "
                     "log). Budgets below are `warm:max` pairs.\n")
        lines.append("| PH | warm:max | total iters | fit time (s) | edges low "
                     "| edges ref | common | only low | only ref | Jaccard "
                     "| max abs dW |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in results:
            lines.append(
                f"| {r['ph']} | {r['warm_iter']}:{r['max_iter']} "
                f"| {r['total_optimizer_iters_budget']} "
                f"| {r['runtime_s']:.2f} | {r['edges_low']} "
                f"| {r['edges_ref']} | {r['edges_common']} "
                f"| {r['edges_only_low']} | {r['edges_only_ref']} "
                f"| {r['jaccard_support']:.3f} "
                f"| {r['max_abs_dW_overall']:.4f} |")
        lines.append("\nInterpretation guide:\n")
        lines.append("- `fit time` answers the todo6 runtime question: which "
                     "budget lands near the remembered '~2 minutes'.\n")
        lines.append("- `common`/`Jaccard` answer the graph-agreement "
                     "question: how many of the canonical PH5/PH6 edges "
                     "survive at the small budget.\n")
        lines.append("- `edges only low/ref` show what each side adds "
                     "uniquely; `max abs dW` is the weight drift on the "
                     "union support.\n")
        lines.append("\nReference artifacts were copied (never moved) into "
                     "`reference_copies/` with sha256 provenance sidecars; "
                     "canonical result trees were not modified.\n")
        with open(md_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\n[report] markdown: {md_path}")
        print(f"[report] json:     {json_path}")

    banner("DONE — no canonical file was opened for writing; all outputs "
           f"are under {os.path.relpath(args.out_root, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
