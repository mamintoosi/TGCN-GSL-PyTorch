#!/usr/bin/env python3
"""
Long-horizon extension (PH=5, PH=6) — support utilities.

Two purposes, selected with --phase:

  --phase graphs   Fit the MISSING contemporaneous DAGMA graphs (GSL input)
                   for the requested datasets/PHs, reusing the exact Stage 33
                   protocol (learn_gsl_graph: X = train_norm[0::PH],
                   lambda1 per dataset, w_threshold=0.3 inside fit(),
                   warm 30k / max 60k, seed 42).  Existing artifacts are
                   reused; nothing is recomputed or overwritten.

  --phase report   Aggregate the PH=5/PH=6 forecasting JSONs written by
                   src/run_canonical_matrix.py into a mean±std summary table
                   (five-seed RMSE/MAE per variant), including the GSL/cGSL
                   gap the reviewer asked about.

The multi-lag DAGMA blocks need NO new fit for PH=5/6: the Stage 26 fit is
PH-independent (input built from the training split only; blocks verified
byte-identical across PH1-4) and is reused via load_multilag_graphs().

Physical / NoSpatial variants need no graph artifacts at all.

Output conventions:
  graphs phase : results/stage33_gsl_canonical/{prefix}_gsl_ph{ph}_seed42_{W_est,A_binary}.npy
                 (same naming as Stage 33; never overwrites existing files)
  report phase : results/stage59_ph56_horizon/ph56_horizon_summary.json

Usage:
  python src/run_ph56_horizon.py --phase graphs --datasets losloop --phs 5 6
  python src/run_ph56_horizon.py --phase report --datasets losloop --phs 5 6
"""
import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage59_ph56_horizon")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]


def main():
    parser = argparse.ArgumentParser(
        description="PH=5/6 long-horizon extension: graph prep and reporting")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["graphs", "report"])
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["losloop"], choices=["losloop", "shenzhen"])
    parser.add_argument("--phs", type=int, nargs="+", default=[5, 6])
    args = parser.parse_args()

    if args.phase == "graphs":
        run_graphs(args)
    else:
        run_report(args)


def run_graphs(args):
    # Imported lazily so --phase report works without dagma/torch installed.
    from src.run_gsl_canonical import RESULTS_DIR as GSL_DIR, DATASET_CONFIGS
    from src.run_gsl_canonical import learn_gsl_graph
    dagma_kwargs = {"warm_iter": 30000, "max_iter": 60000}  # Stage 33 defaults

    print("=" * 78)
    print("PH=5/6 EXTENSION — CONTEMPORANEOUS DAGMA GRAPHS (Stage 33 protocol)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    for dataset in args.datasets:
        cfg = DATASET_CONFIGS[dataset]
        for ph in args.phs:
            w_path = os.path.join(
                GSL_DIR, f"{cfg['prefix']}_gsl_ph{ph}_seed42_W_est.npy")
            a_path = os.path.join(
                GSL_DIR, f"{cfg['prefix']}_gsl_ph{ph}_seed42_A_binary.npy")
            if os.path.exists(a_path):
                A = np.load(a_path)
                print(f"  [REUSE] {dataset} PH={ph}: {int(A.sum())} edges ({a_path})")
                continue
            print(f"  [FIT ] {dataset} PH={ph}: fitting contemporaneous DAGMA "
                  f"(lambda1={cfg['lambda1']}, X=train_norm[0::{ph}]) ...")
            W_est, A, meta = learn_gsl_graph(dataset, ph, 42, dagma_kwargs)
            np.save(w_path, W_est)
            np.save(a_path, A)
            print(f"  [DONE] {dataset} PH={ph}: {int(A.sum())} edges "
                  f"(runtime {meta['runtime_s']}s)")
    print("\nAll required contemporaneous graphs are in place.")


def run_report(args):
    train_dir = os.path.join(PROJECT_ROOT, "results", "stage40_canonical", "training")

    def load_cell(dataset, ph, variant):
        rows = []
        for seed in SEEDS:
            p = os.path.join(train_dir, f"{dataset}_ph{ph}_seed{seed}_{variant}.json")
            if not os.path.exists(p):
                return None  # incomplete cell
            with open(p) as f:
                d = json.load(f)
            if d.get("status") != "complete":
                return None
            rows.append(d)
        return rows

    def mean_std(rows, key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return round(float(vals.mean()), 4), round(float(vals.std(ddof=1)), 4)

    payload = {
        "stage": 59,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Long-horizon extension (PH=5, PH=6) — Reviewer 1 W7",
        "protocol": "identical to Stage 40 canonical (batch 128, Adam lr 1e-3, "
                    "wd 1e-4, hidden 64, 50 epochs, seq_len 12, seeds 42-46, "
                    "feat_max = train-split max)",
        "datasets": {},
    }

    print("=" * 78)
    print("PH=5/6 EXTENSION — FIVE-SEED SUMMARY")
    print("=" * 78)
    incomplete = []

    for dataset in args.datasets:
        per_ph = {}
        for ph in args.phs:
            variants = {}
            for variant in ("physical", "no_spatial", "gsl", "cgsl",
                            "multi_gsl", "multi_gsl_mix"):
                rows = load_cell(dataset, ph, variant)
                if rows is None:
                    incomplete.append(f"{dataset}/ph{ph}/{variant}")
                    continue
                rm, rs = mean_std(rows, "rmse")
                ma, mas = mean_std(rows, "mae")
                variants[variant] = {
                    "display": rows[0]["display_name"],
                    "rmse_mean": rm, "rmse_std": rs,
                    "mae_mean": ma, "mae_std": mas,
                    "n_seeds": len(rows),
                    "n_edges": rows[0]["n_edges"],
                    "dagma_multilag_source_ph": rows[0].get("dagma_multilag_source_ph"),
                }
            per_ph[ph] = variants
            if variants:
                print(f"\n{dataset} PH={ph}:")
                for v, s in variants.items():
                    print(f"  {s['display']:26s} RMSE {s['rmse_mean']:.4f} "
                          f"± {s['rmse_std']:.4f}  (n={s['n_seeds']}, "
                          f"edges={s['n_edges']})")
                if "gsl" in variants and "cgsl" in variants and "no_spatial" in variants:
                    gap = variants["cgsl"]["rmse_mean"] - variants["gsl"]["rmse_mean"]
                    print(f"  GSL/cGSL gap (cGSL - GSL): {gap:+.4f} RMSE")
        payload["datasets"][dataset] = per_ph

    if incomplete:
        payload["incomplete_cells"] = incomplete
        print("\n[NOTE] Incomplete cells (JSON missing or not complete):")
        for c in incomplete:
            print(f"  - {c}")
        print("Run src/run_canonical_matrix.py for the missing cells first.")

    out_path = os.path.join(RESULTS_DIR, "ph56_horizon_summary.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
