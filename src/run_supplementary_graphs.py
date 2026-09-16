#!/usr/bin/env python3
"""
Supplementary PH5/PH6 graph preparation — rev2 (todo5 policy)
=============================================================

Two experiment kinds, each with its own isolated output root:

  canonical_completion (--experiment-kind canonical_completion)
      root: results/supplementary_ph56/
      lambda_1 = canonical per dataset (losloop 0.02, shenzhen 0.01).
      Reuse policy (todo5 #2):
        * Los-loop PH5/PH6 gsl graphs at lambda 0.02 EXIST canonically
          (results/stage33_gsl_canonical, fitted in stage59) -> COPIED into
          graphs_lambda002/ by atomic_copy_with_provenance (sha256 + source
          path + timestamp; source opened read-only, never modified).
        * SZ-Taxi PH5/PH6 gsl at lambda 0.01 do NOT exist -> fresh DAGMA fit
          (the only fresh fits in this kind).
        * multi-lag Stage 26 blocks (PH-independent) -> COPIED with
          provenance (renamed without PH ambiguity: ..._from_ph1_...).
        * cGSL -> derived from the gsl W_est (no DAGMA fit), saved here.

  lambda_sensitivity (--experiment-kind lambda_sensitivity)
      root: results/supplementary_ph56_lambda001/
      Los-loop only, lambda_1 = 0.01 (todo5 #3). NOTHING canonical exists at
      this lambda -> every graph is freshly fitted; DAGMA verifiably receives
      0.01 (learn_gsl_graph passes the effective lambda straight into
      model.fit; assertion in code).

Everything is written ONLY under the kind's root; canonical result trees and
the other kind's root are write-protected (src/supp_paths.check_root_safety).
Default behaviour is refuse-to-overwrite; --overwrite relaxes it for THESE
supplementary files only.

Usage:
  # plans (safe anywhere):
  python3 src/run_supplementary_graphs.py --experiment-kind canonical_completion --print-plan-only
  python3 src/run_supplementary_graphs.py --experiment-kind lambda_sensitivity --print-plan-only

  # real runs (on the experiment machine):
  python3 src/run_supplementary_graphs.py --experiment-kind canonical_completion
  python3 src/run_supplementary_graphs.py --experiment-kind lambda_sensitivity
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.supp_paths import (  # noqa: E402
    SEEDS, DATASETS, DATASET_PREFIX, LAMBDA_CANONICAL, SUPP_VARIANTS,
    graph_paths, graph_dir, multilag_blocks_dir,
    multilag_block_copy_path, multilag_provenance_path,
    atomic_save_json, atomic_save_npy, atomic_copy_with_provenance,
    add_root_args, resolve_root, kind_note, lambda_tag, sha256_of,
)

STAGE26_DIR = PROJECT_ROOT / "results" / "stage26_validation"
STAGE33_DIR = PROJECT_ROOT / "results" / "stage33_gsl_canonical"
N_LAGS = 3


def effective_lambda(args, ds: str) -> float:
    """Effective lambda_1 for a dataset under the kind's policy."""
    if args.experiment_kind == "lambda_sensitivity":
        # todo5 #3: sensitivity experiment is lambda_1 = 0.01, Los-loop only.
        if args.lambda1 is not None and abs(args.lambda1 - 0.01) > 1e-12:
            raise SystemExit(
                "[FATAL] --lambda1 for the lambda_sensitivity kind must be "
                f"0.01 (got {args.lambda1:g}); refusing a mixed sensitivity "
                "value.")
        return 0.01
    # canonical_completion: canonical per dataset (explicit --lambda1 allowed
    # only if it equals the canonical value, to prevent accidental mixing).
    canon = LAMBDA_CANONICAL[ds]
    if args.lambda1 is not None and abs(args.lambda1 - canon) > 1e-12:
        raise SystemExit(
            f"[FATAL] --lambda1 {args.lambda1:g} differs from the canonical "
            f"value {canon:g} for {ds} in the canonical_completion kind; "
            "refusing (use --experiment-kind lambda_sensitivity for the "
            "0.01 Los-loop study).")
    return canon


def canonical_gsl_paths(dataset: str, ph: int) -> dict:
    prefix = DATASET_PREFIX[dataset]
    base = STAGE33_DIR / f"{prefix}_gsl_ph{ph}_seed42"
    return {"A": base.with_name(base.name + "_A_binary.npy"),
            "W": base.with_name(base.name + "_W_est.npy")}


def canonical_multilag_blocks(dataset: str, source_ph: int = 1) -> list:
    prefix = DATASET_PREFIX[dataset]
    return [STAGE26_DIR / f"{prefix}_ph{source_ph}_seed42_L{N_LAGS}_lag_{l}.npy"
            for l in range(1, N_LAGS + 1)]


# ----------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------
def build_plan(args, root: Path) -> list:
    plan = []
    for ds in args.datasets:
        lam = effective_lambda(args, ds)
        canon = canonical_gsl_paths(ds, 5)  # existence pattern same for 5/6
        canon_exists = canonical_gsl_paths(ds, args.phs[0])["A"].exists()
        for ph in args.phs:
            for gt in ("gsl", "cgsl"):
                paths = graph_paths(root, ds, ph, lam, gt)
                exists = paths["A"].exists()
                if gt == "gsl":
                    if exists:
                        action = "exists -> reuse"
                    elif args.experiment_kind == "canonical_completion" \
                            and canon_exists:
                        action = "COPY canonical (no fit, provenance sidecar)"
                    else:
                        action = "DAGMA FIT (~1 h CPU at N=207)"
                else:  # cgsl
                    action = ("exists -> reuse" if exists
                              else "derive from gsl W_est (no fit)")
                plan.append((ds, ph, lam, gt, action,
                             paths["A"].name))
            plan.append((ds, ph, lam, "multilag", "COPY blocks (no fit)",
                         multilag_provenance_path(root, ds, lam).name))
    return plan


def print_plan(args, root: Path) -> None:
    print("=" * 78)
    print("SUPPLEMENTARY GRAPH PLAN (no DAGMA / no copy in this mode)")
    print("=" * 78)
    print(f"experiment-kind: {args.experiment_kind}")
    print(f"output root    : {root}")
    print(f"datasets       : {args.datasets}")
    print(f"PHs            : {args.phs}")
    lam_desc = ("canonical per dataset (losloop 0.02, shenzhen 0.01)"
                if args.experiment_kind == "canonical_completion"
                else "0.01 (sensitivity; Los-loop only)")
    print(f"lambda_1       : {lam_desc}")
    print(f"variants later : {len(SUPP_VARIANTS)} (10-variant matrix; no "
          "MultiGSL-Weighted)")
    print(f"overwrite      : {args.overwrite}")
    print("-" * 78)
    n_fits = n_copies = n_derived = 0
    for ds, ph, lam, gt, action, name in build_plan(args, root):
        if gt == "multilag":
            print(f"  {ds:9s} PH={ph} multilag : {action} "
                  f"[lambda1={lam:g}]")
            continue
        print(f"  {ds:9s} PH={ph} {gt:5s}   : {action} [lambda1={lam:g}]")
        print(f"        -> {name}")
        if "DAGMA FIT" in action:
            n_fits += 1
        elif "COPY" in action:
            n_copies += 1
        elif "derive" in action:
            n_derived += 1
    print("-" * 78)
    print(f"summary: {n_fits} fresh DAGMA fit(s), {n_copies} canonical "
          f"cop(ies) with provenance, {n_derived} cGSL derivation(s) per "
          "kind-run")
    print(kind_note(args.experiment_kind))
    print("Canonical result directories are never written "
          "(src/supp_paths.check_root_safety).")


# ----------------------------------------------------------------------
# execution
# ----------------------------------------------------------------------
def copy_or_report_multilag(root: Path, ds: str, lam: float,
                            overwrite: bool) -> None:
    """Copy the PH-independent Stage 26 blocks with provenance (or keep)."""
    prov_path = multilag_provenance_path(root, ds, lam)
    blocks = canonical_multilag_blocks(ds)
    missing = [str(p) for p in blocks if not p.exists()]
    if missing:
        print(f"  [WARN] {ds}: multi-lag blocks missing: {missing}")
        return
    copies = []
    for lag, src in enumerate(blocks, start=1):
        dst = multilag_block_copy_path(root, ds, lam, lag)
        if dst.exists() and not overwrite:
            copies.append({"path": str(dst), "sha256": sha256_of(dst),
                           "status": "already_present"})
            continue
        atomic_copy_with_provenance(
            src, dst, overwrite=overwrite,
            meta_extra={"dataset": ds, "ph": "PH-independent (from PH=1 fit)",
                        "lambda_tag": lambda_tag(lam), "graph_type": "multilag",
                        "lag_index": lag,
                        "note": "Stage 26 multi-lag fit is PH-independent "
                                "(training-split input); blocks are "
                                "byte-identical across PH1-4."})
        copies.append({"path": str(dst), "sha256": sha256_of(dst),
                       "status": "copied"})
    payload = {
        "kind": "multilag_blocks_copy_provenance",
        "dataset": ds,
        "lambda_tag": lambda_tag(lam),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dagma_refit_required": False,
        "reason": "Stage 26 multi-lag DAGMA fit input is the training split "
                  "only; PH-independent; blocks byte-identical across PH1-4.",
        "consumer_threshold": 0.1,
        "blocks": copies,
        "missing_blocks": missing,
    }
    if not (prov_path.exists() and not overwrite):
        atomic_save_json(prov_path, payload, overwrite=overwrite)
        print(f"  [COPY] {ds} multilag blocks ({len(copies)}) with "
              f"provenance -> {prov_path.parent.name}/")
    else:
        print(f"  [KEEP] {ds} multilag provenance exists: {prov_path.name}")


def run(args) -> None:
    root = resolve_root(args)
    kind = args.experiment_kind
    print("=" * 78)
    print(f"SUPPLEMENTARY GRAPH PREPARATION — {kind}")
    print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Root      : {root}")
    print(f"Overwrite : {args.overwrite}")
    print(kind_note(kind))
    print("=" * 78)

    for ds in args.datasets:
        lam = effective_lambda(args, ds)
        print(f"\n--- {ds} (lambda_1={lam:g} [{lambda_tag(lam)}]) ---")

        # ---- 1. multi-lag blocks: copy with provenance (both kinds) --------
        copy_or_report_multilag(root, ds, lam, args.overwrite)

        # ---- 2. contemporaneous gsl ---------------------------------------
        for ph in args.phs:
            gp = graph_paths(root, ds, ph, lam, "gsl")
            canon = canonical_gsl_paths(ds, ph)
            if gp["A"].exists() and not args.overwrite:
                print(f"  [REUSE] {ds} PH={ph} gsl: {gp['A'].name}")
                W_est = None
            elif kind == "canonical_completion" and canon["A"].exists():
                atomic_copy_with_provenance(
                    canon["A"], gp["A"], overwrite=args.overwrite,
                    meta_extra={"dataset": ds, "ph": ph,
                                "lambda_1": lam, "graph_type": "gsl"})
                atomic_copy_with_provenance(
                    canon["W"], gp["W"], overwrite=args.overwrite,
                    meta_extra={"dataset": ds, "ph": ph,
                                "lambda_1": lam, "graph_type": "gsl_W_est"})
                atomic_save_json(gp["meta"], {
                    "kind": "supplementary_gsl_copied_from_canonical",
                    "dataset": ds, "ph": ph, "lambda_1": lam,
                    "lambda_tag": lambda_tag(lam), "graph_type": "gsl",
                    "canonical_source_A": str(canon["A"]),
                    "canonical_source_W": str(canon["W"]),
                    "dagma_refit_required": False,
                    "generated": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"),
                }, overwrite=args.overwrite)
                import numpy as np
                A = np.load(gp["A"])
                print(f"  [COPY ] {ds} PH={ph} gsl: canonical -> "
                      f"{gp['A'].name} ({int(A.sum())} edges, provenance "
                      "sidecar)")
                W_est = None
            else:
                from src.run_gsl_canonical import learn_gsl_graph, load_data
                train_norm, _t, _a, _fm = load_data(ds)
                T_train, N = train_norm.shape
                print(f"  [FIT  ] {ds} PH={ph} gsl: DAGMA lambda1={lam:g}, "
                      f"X=train_norm[0::{ph}] "
                      f"({(T_train + ph - 1) // ph} rows x {N} vars)")
                t0 = datetime.now()
                W_est, A, meta = learn_gsl_graph(
                    ds, ph, SEEDS[0], {"warm_iter": 30000, "max_iter": 60000},
                    lambda1=lam)
                # todo5 #3: verify lambda actually reached the fit call.
                assert abs(float(meta["lambda1"]) - lam) < 1e-12, \
                    f"fit metadata lambda {meta['lambda1']} != requested {lam}"
                fit_s = (datetime.now() - t0).total_seconds()
                atomic_save_npy(gp["A"], A, overwrite=args.overwrite)
                atomic_save_npy(gp["W"], W_est, overwrite=args.overwrite)
                atomic_save_json(gp["meta"], {
                    "kind": "supplementary_gsl_fresh_fit",
                    "dataset": ds, "ph": ph,
                    "lambda_1": lam, "lambda_tag": lambda_tag(lam),
                    "lambda1_verified_in_fit_call": True,
                    "graph_type": "gsl",
                    "threshold_w": 0.3,
                    "n_nodes": int(N),
                    "n_retained_directed_edges": int(A.sum()),
                    "source_data_shape": [int(T_train), int(N)],
                    "dagma_input": f"train_norm[0::{ph}]",
                    "n_input_rows": int(meta["train_rows"]),
                    "fitting_time_s": round(fit_s, 1),
                    "dagma_runtime_s_from_meta": meta["runtime_s"],
                    "seed": SEEDS[0],
                    "software": meta.get("software", {}),
                    "generated": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"),
                }, overwrite=args.overwrite)
                print(f"  [DONE ] {ds} PH={ph} gsl: {int(A.sum())} edges, "
                      f"{fit_s:.0f}s (lambda {lam:g} verified in fit meta)")

            # ---- 3. cgsl (derived; no DAGMA) ------------------------------
            cp = graph_paths(root, ds, ph, lam, "cgsl")
            if cp["A"].exists() and not args.overwrite:
                print(f"  [REUSE] {ds} PH={ph} cgsl: {cp['A'].name}")
                continue
            import numpy as np
            if W_est is None:
                if gp["W"].exists():
                    W_est = np.load(gp["W"])
                elif kind == "canonical_completion" and canon["W"].exists():
                    W_est = np.load(canon["W"])          # read-only
                else:
                    print(f"  [SKIP ] {ds} PH={ph} cgsl: no W_est available")
                    continue
            A_cgsl = (np.abs(W_est) > 0).astype("float32")
            A_cgsl = A_cgsl + A_cgsl.T
            import numpy as _np
            _np.fill_diagonal(A_cgsl, 0)
            A_cgsl = (A_cgsl > 0).astype("float32")
            atomic_save_npy(cp["A"], A_cgsl, overwrite=args.overwrite)
            atomic_save_json(cp["meta"], {
                "kind": "supplementary_cgsl_derived",
                "dataset": ds, "ph": ph,
                "lambda_1": lam, "lambda_tag": lambda_tag(lam),
                "graph_type": "cgsl",
                "n_nodes": int(A_cgsl.shape[0]),
                "n_retained_directed_entries": int(A_cgsl.sum()),
                "derived_from": gp["A"].name,
                "derivation": "symmetrize(A_gsl); no DAGMA fit",
                "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, overwrite=args.overwrite)
            print(f"  [DONE ] {ds} PH={ph} cgsl: "
                  f"{int(A_cgsl.sum())} directed entries -> {cp['A'].name}")

    print("\nAll supplementary graphs are in place under:", root)
    print(kind_note(kind))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated supplementary PH5/PH6 graph preparation (rev2)")
    parser.add_argument("--datasets", nargs="+",
                        default=None,
                        help="datasets to prepare (default: losloop+shenzhen "
                             "for canonical_completion, losloop only for "
                             "lambda_sensitivity)")
    parser.add_argument("--phs", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--lambda1", type=float, default=None,
                        help="optional; must match the kind's policy value "
                             "(canonical per dataset, or 0.01 for the "
                             "sensitivity kind) — mainly a guard")
    add_root_args(parser)
    parser.add_argument("--print-plan-only", action="store_true",
                        help="print the plan and exit (no DAGMA, no copies, "
                             "no writes)")
    args = parser.parse_args()

    # Default dataset set depends on the kind (sensitivity = Los-loop only).
    if args.datasets is None:
        args.datasets = (["losloop"]
                         if args.experiment_kind == "lambda_sensitivity"
                         else list(DATASETS))

    root = resolve_root(args)
    if args.print_plan_only:
        print_plan(args, root)
        return
    run(args)


if __name__ == "__main__":
    main()
