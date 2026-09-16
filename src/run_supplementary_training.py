#!/usr/bin/env python3
"""
Supplementary PH5/PH6 training — rev2 (todo5 policy)
====================================================

Two experiment kinds, each with its own isolated output root:

  canonical_completion   root: results/supplementary_ph56/
      lambda_1 = canonical per dataset (losloop 0.02, shenzhen 0.01).
      10-variant matrix (todo5 #1: T-GCN-MultiGSL-Weighted REMOVED).
      Cell reuse (todo5 #4): Los-loop T-GCN cells at lambda 0.02 that already
      exist canonically (results/stage40_canonical, stage59 protocol) are
      COPIED into training_lambda002/ with inline provenance fields
      (source path, sha256, timestamp) — NOT re-trained. Everything else
      (all GCN-family cells, all SZ cells) is newly trained.

  lambda_sensitivity     root: results/supplementary_ph56_lambda001/
      Los-loop only, lambda_1 = 0.01. 10-variant matrix, all cells newly
      trained; no canonical counterpart exists at this lambda.

Training protocol is the canonical one (imported from
src/run_canonical_matrix.py): 80/20 chronological split, train-max
normalization, seq_len 12, batch 128, Adam lr 1e-3, wd 1e-4, hidden 64,
50 epochs, seeds 42-46. Nothing is silently changed.

Graph sources:
  gsl/cgsl     : the kind's graphs_<tag>/ artifacts prepared by
                 src/run_supplementary_graphs.py (copied canonical or fresh
                 fits); for canonical_completion a read-only canonical
                 fallback remains as a safety net.
  multi_gsl /  : PH-independent Stage 26 blocks — the COPIED copies in
  gcn_multigsl   multilag_blocks/ (preferred, with provenance) or the
                 canonical files read-only as fallback; consumer threshold
                 |W|>0.1; no refit.
  no_spatial / physical: no DAGMA artifact involved.

Cell counts (10-variant matrix, todo5 #5):
  canonical_completion : 2 datasets x 2 PHs x 10 variants x 5 seeds = 200
  lambda_sensitivity   : 1 dataset  x 2 PHs x 10 variants x 5 seeds = 100

Usage:
  python3 src/run_supplementary_training.py --experiment-kind canonical_completion --print-plan-only
  python3 src/run_supplementary_training.py --experiment-kind lambda_sensitivity --print-plan-only
  python3 src/run_supplementary_training.py --experiment-kind canonical_completion
  python3 src/run_supplementary_training.py --experiment-kind lambda_sensitivity
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# numpy/torch are imported lazily (inside run_one) so --print-plan-only and
# --dry-run work in any environment, matching the other runners.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.supp_paths import (  # noqa: E402
    SEEDS, DATASETS, DATASET_PREFIX, LAMBDA_CANONICAL, SUPP_VARIANTS,
    graph_paths, multilag_block_copy_path, training_path,
    atomic_save_json, add_root_args, resolve_root, kind_note, lambda_tag,
    sha256_of,
)

STAGE26_DIR = PROJECT_ROOT / "results" / "stage26_validation"
STAGE33_DIR = PROJECT_ROOT / "results" / "stage33_gsl_canonical"
STAGE40_TRAINING = PROJECT_ROOT / "results" / "stage40_canonical" / "training"
N_LAGS = 3

# Los-loop T-GCN cells eligible for copy-from-canonical (lambda 0.02 only):
# these five variants were run canonically at PH5/PH6 (stage40/stage59).
COPY_ELIGIBLE_VARIANTS = ("no_spatial", "physical", "gsl", "cgsl", "multi_gsl")

# Canonical module may be unavailable in plan-only environments (no numpy).
try:
    from src.run_canonical_matrix import (  # noqa: E402
        VARIANTS, DATASET_CONFIGS, load_data, generate_sequences,
        train_and_eval,
    )
    CANONICAL_AVAILABLE = True
except ImportError as _exc:  # no numpy/torch here
    CANONICAL_AVAILABLE = False
    CANONICAL_IMPORT_ERROR = _exc
    # Static display tables only (plain dicts, no numerical code) so plans
    # can still be printed; real training uses the canonical module.
    VARIANTS = {
        "no_spatial":      {"display": "T-GCN-NoSpatial", "backbone": "tgcn", "adj_type": "identity", "dagma": None},
        "physical":        {"display": "T-GCN",           "backbone": "tgcn", "adj_type": "single",   "dagma": None},
        "gsl":             {"display": "T-GCN-GSL",       "backbone": "tgcn", "adj_type": "single",   "dagma": "contemporaneous", "dagma_thr": 0.3},
        "cgsl":            {"display": "T-GCN-cGSL",      "backbone": "tgcn", "adj_type": "single",   "dagma": "contemporaneous", "dagma_thr": 0.3, "symmetrize": True},
        "multi_gsl":       {"display": "T-GCN-MultiGSL",  "backbone": "tgcn", "adj_type": "lag_list", "dagma": "multilag", "dagma_thr": 0.1},
        "gcn_no_spatial":  {"display": "GCN-NoSpatial",   "backbone": "gcn",  "adj_type": "identity", "dagma": None},
        "gcn_physical":    {"display": "GCN",             "backbone": "gcn",  "adj_type": "single",   "dagma": None},
        "gcn_gsl":         {"display": "GCN-GSL",         "backbone": "gcn",  "adj_type": "single",   "dagma": "contemporaneous", "dagma_thr": 0.3},
        "gcn_cgsl":        {"display": "GCN-cGSL",        "backbone": "gcn",  "adj_type": "single",   "dagma": "contemporaneous", "dagma_thr": 0.3, "symmetrize": True},
        "gcn_multigsl":    {"display": "GCN-MultiGSL",    "backbone": "gcn",  "adj_type": "single",   "dagma": "multilag_union", "dagma_thr": 0.1},
    }
    DATASET_CONFIGS = {
        "losloop": {"feat_path": "data/los_speed.csv", "adj_path": "data/los_adj.csv", "N": 207, "prefix": "los"},
        "shenzhen": {"feat_path": "data/sz_speed.csv", "adj_path": "data/sz_adj.csv", "N": 156, "prefix": "sz"},
    }
    load_data = generate_sequences = train_and_eval = None


def effective_lambda(args, ds: str) -> float:
    """Effective lambda_1 for a dataset under the kind's policy."""
    if args.experiment_kind == "lambda_sensitivity":
        if args.lambda1 is not None and abs(args.lambda1 - 0.01) > 1e-12:
            raise SystemExit(
                "[FATAL] --lambda1 for the lambda_sensitivity kind must be "
                f"0.01 (got {args.lambda1:g}).")
        return 0.01
    canon = LAMBDA_CANONICAL[ds]
    if args.lambda1 is not None and abs(args.lambda1 - canon) > 1e-12:
        raise SystemExit(
            f"[FATAL] --lambda1 {args.lambda1:g} differs from the canonical "
            f"value {canon:g} for {ds} in the canonical_completion kind; "
            "refusing.")
    return canon


def default_datasets(kind: str) -> list:
    return ["losloop"] if kind == "lambda_sensitivity" else list(DATASETS)


def canonical_result_path(ds: str, ph: int, seed: int, variant: str) -> Path:
    return STAGE40_TRAINING / f"{ds}_ph{ph}_seed{seed}_{variant}.json"


def resolve_contemporaneous_A(ds: str, ph: int, lam: float, gtype: str,
                              root: Path, kind: str):
    """Order: kind's graphs_<tag>/ artifact -> canonical read-only fallback
    (canonical_completion only) -> missing."""
    supp = graph_paths(root, ds, ph, lam, gtype)["A"]
    if supp.exists():
        return supp, f"supplementary {gtype} artifact {supp.name}"
    if kind == "canonical_completion":
        canon = (STAGE33_DIR
                 / f"{DATASET_PREFIX[ds]}_gsl_ph{ph}_seed42_A_binary.npy")
        if canon.exists():
            return canon, (f"canonical artifact {canon.name} "
                           "(read-only fallback; lambda == canonical)")
    return None, f"missing supplementary {gtype} graph for {ds} PH={ph}"


def multilag_sources(root: Path, ds: str, lam: float):
    """The 3 lag blocks: COPIED copies preferred, canonical read-only
    fallback; None when unavailable."""
    out = []
    for lag in range(1, N_LAGS + 1):
        copied = multilag_block_copy_path(root, ds, lam, lag)
        if copied.exists():
            out.append(copied)
            continue
        canon = (STAGE26_DIR
                 / f"{DATASET_PREFIX[ds]}_ph1_seed42_L{N_LAGS}_lag_{lag}.npy")
        if not canon.exists():
            return None
        out.append(canon)
    return out


def cell_status(variant: str, ds: str, ph: int, seed: int, lam: float,
                root: Path, kind: str):
    """'exists' | 'ready' | ('copy_from_canonical', src) |
    ('missing_graph', remediation) for one cell."""
    out = training_path(root, ds, ph, seed, variant, lam)
    if out.exists():
        try:
            with open(out) as f:
                if json.load(f).get("status") == "complete":
                    return "exists"
        except Exception:
            pass
    # copy-from-canonical eligibility (todo5 #4)
    if (kind == "canonical_completion" and ds == "losloop"
            and variant in COPY_ELIGIBLE_VARIANTS
            and abs(lam - 0.02) < 1e-12):
        src = canonical_result_path(ds, ph, seed, variant)
        if src.exists():
            try:
                with open(src) as f:
                    if json.load(f).get("status") == "complete":
                        return ("copy_from_canonical", src)
            except Exception:
                pass
    # graph availability
    v = VARIANTS[variant]
    if v["dagma"] == "contemporaneous":
        gtype = "cgsl" if v.get("symmetrize") else "gsl"
        p, _desc = resolve_contemporaneous_A(ds, ph, lam, gtype, root, kind)
        if p is None:
            return ("missing_graph",
                    "python3 src/run_supplementary_graphs.py "
                    f"--experiment-kind {kind} "
                    f"--datasets {ds} --phs {ph}")
    elif v["dagma"] in ("multilag", "multilag_union"):
        if multilag_sources(root, ds, lam) is None:
            return ("missing_graph",
                    f"multi-lag blocks missing for {ds}: run "
                    "python3 src/run_multilag_dagma.py --dataset "
                    f"{ds} --ph 1")
    return "ready"


def print_plan(args, root: Path) -> None:
    kind = args.experiment_kind
    print("=" * 78)
    print("SUPPLEMENTARY TRAINING PLAN" +
          (" (no training in this mode)" if args.print_plan_only else ""))
    print("=" * 78)
    print(f"experiment-kind: {kind}")
    print(f"output root    : {root}")
    print(f"datasets       : {args.datasets}")
    print(f"PHs            : {args.phs} (native 5-min Los-loop; 15-min SZ)")
    print(f"variants ({len(args.variants)})   : {args.variants}")
    print(f"seeds          : {args.seeds}")
    if kind == "canonical_completion":
        print("lambda_1       : canonical per dataset (losloop 0.02, "
              "shenzhen 0.01)")
    else:
        print("lambda_1       : 0.01 (sensitivity; Los-loop only)")
    print("protocol       : canonical Stage 40 (seq_len 12, batch 128, Adam "
          "lr 1e-3, wd 1e-4, hidden 64, 50 epochs)")
    print(f"overwrite      : {args.overwrite}")
    print("-" * 78)
    counts = {"exists": 0, "ready": 0, "copy": 0, "missing": 0}
    for ds in args.datasets:
        lam = effective_lambda(args, ds)
        for ph in args.phs:
            for variant in args.variants:
                v = VARIANTS[variant]
                for seed in args.seeds:
                    st = cell_status(variant, ds, ph, seed, lam, root, kind)
                    out = training_path(root, ds, ph, seed, variant, lam)
                    if st == "exists":
                        counts["exists"] += 1
                        status_s = "EXISTS -> skip"
                    elif st == "ready":
                        counts["ready"] += 1
                        status_s = "RUN (newly trained)"
                    elif st[0] == "copy_from_canonical":
                        counts["copy"] += 1
                        status_s = (f"COPY from canonical "
                                    f"({Path(st[1]).name})")
                    else:
                        counts["missing"] += 1
                        status_s = f"MISSING GRAPH (remediation: {st[1]})"
                    print(f"  {ds:9s} PH={ph} {v['display']:18s} seed={seed} "
                          f"-> {out.name}  [{status_s}]")
    total = sum(counts.values())
    print("-" * 78)
    print(f"cells: {total} logical | {counts['ready']} newly executed | "
          f"{counts['copy']} copied from canonical | "
          f"{counts['exists']} already in supplementary tree | "
          f"{counts['missing']} blocked on missing graphs")
    if kind == "canonical_completion":
        print(f"expected matrix total: 2 datasets x 2 PHs x 10 variants x "
              f"5 seeds = 200 cells (MultiGSL-Weighted excluded per todo5)")
    else:
        print(f"expected matrix total: 1 dataset x 2 PHs x 10 variants x "
              f"5 seeds = 100 cells")
    print(kind_note(kind))
    print("Canonical result directories are never written "
          "(src/supp_paths.check_root_safety).")


def copy_canonical_cell(src: Path, ds: str, ph: int, seed: int, variant: str,
                        lam: float, root: Path, overwrite: bool) -> str:
    """Copy a canonical result JSON into the supplementary tree with
    inline provenance (todo5 #4: copied-from-canonical is explicit)."""
    import json as _json
    with open(src) as f:
        data = _json.load(f)
    data["cell_provenance"] = "copied_from_canonical"
    data["source_result"] = str(src)
    data["source_sha256"] = sha256_of(src)
    data["copy_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["note"] = ("copied read-only from the canonical run; the source "
                    "file was not modified; NOT newly trained")
    out = training_path(root, ds, ph, seed, variant, lam)
    atomic_save_json(out, data, overwrite=overwrite)
    print(f"  [COPY] {ds} PH={ph} seed={seed} {variant}: "
          f"{src.name} -> {out.name}")
    return "copied_from_canonical"


def run_one(variant: str, ds: str, ph: int, seed: int, lam: float,
            root: Path, kind: str, args) -> str:
    import numpy as np  # lazy: plan-only mode must work without numpy
    v = VARIANTS[variant]
    cfg = DATASET_CONFIGS[ds]
    N = cfg["N"]
    backbone = v["backbone"]
    loss_name = "mse_with_regularizer" if backbone == "tgcn" else "mse"

    train_norm, test_norm, adj_phys, feat_max = load_data(ds)
    train_X, train_Y = generate_sequences(train_norm, args.seq_len, ph)
    test_X, test_Y = generate_sequences(test_norm, args.seq_len, ph)

    graph_src_name = None
    if v["adj_type"] == "identity":
        adj_model = np.eye(N, dtype=np.float32)
    elif v["dagma"] is None and v["adj_type"] == "single":
        adj_model = adj_phys
    elif v["dagma"] == "contemporaneous":
        gtype = "cgsl" if v.get("symmetrize") else "gsl"
        A_path, desc = resolve_contemporaneous_A(ds, ph, lam, gtype, root,
                                                 kind)
        if A_path is None:
            print(f"  [FAIL] {desc}")
            return "missing_graph"
        A = np.load(A_path).astype(np.float32)
        graph_src_name = f"{gtype}: {A_path.name}"
        if v.get("symmetrize"):
            adj_model = A + A.T
            adj_model = (adj_model > 0).astype(np.float32)
            np.fill_diagonal(adj_model, 0)
        else:
            adj_model = A
    elif v["dagma"] in ("multilag", "multilag_union"):
        srcs = multilag_sources(root, ds, lam)
        if srcs is None:
            print(f"  [FAIL] multi-lag blocks missing for {ds}")
            return "missing_graph"
        from models.multigsl import binary_graph
        lag_list = [binary_graph(np.load(p), v["dagma_thr"]) for p in srcs]
        graph_src_name = ("copied/canonical Stage 26 blocks (PH-independent; "
                          f"{len(lag_list)} lags, |W|>0.1)")
        if v["dagma"] == "multilag_union":      # GCN-MultiGSL single graph
            adj_model = np.zeros((N, N), dtype=np.float32)
            for a in lag_list:
                adj_model = np.maximum(adj_model, a)
        else:                                    # T-GCN lag_list
            adj_model = lag_list
    else:
        adj_model = adj_phys

    from models.tgcn import TGCN
    from models.gcn import GCN
    from models.multigsl import MultiGraphTGCNFixed
    if backbone == "tgcn":
        if v["adj_type"] == "identity":
            model = TGCN(adj=np.eye(N, dtype=np.float32),
                         hidden_dim=args.hidden_dim)
        elif v["adj_type"] == "single":
            model = TGCN(adj=adj_model, hidden_dim=args.hidden_dim)
        else:
            model = MultiGraphTGCNFixed(adj_list=adj_model,
                                        hidden_dim=args.hidden_dim)
    elif backbone == "gcn":
        model = GCN(adj=adj_model, seq_len=args.seq_len,
                    hidden_dim=args.hidden_dim)
    else:
        raise ValueError(backbone)

    n_edges = int((np.asarray(adj_model) > 0).sum()) \
        if isinstance(adj_model, np.ndarray) else \
        int(sum(int((a > 0).sum()) for a in adj_model))

    print(f"  [RUN] {v['display']} {ds} PH={ph} seed={seed} "
          f"(edges={n_edges}, params={sum(p.numel() for p in model.parameters())})")
    metrics = train_and_eval(
        model, train_X, train_Y, test_X, test_Y,
        feat_max, ph, seed, args.max_epochs, loss_name,
        batch_size=args.batch_size, lr=args.lr, wd=args.wd,
        hidden_dim=args.hidden_dim,
    )

    result = {
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_kind": kind,
        "cell_provenance": "newly_trained",
        "variant": variant,
        "display_name": v["display"],
        "backbone": backbone,
        "dataset": ds,
        "ph": ph,
        "seed": seed,
        "lambda1": lam,
        "lambda_tag": lambda_tag(lam),
        "lambda1_canonical": LAMBDA_CANONICAL[ds],
        "graph_source": graph_src_name,
        "n_edges": n_edges,
        "n_params": metrics["n_params"],
        "rmse": round(metrics["RMSE"], 4),
        "mae": round(metrics["MAE"], 4),
        "train_time_s": metrics["train_time_s"],
        "protocol": {
            "batch_size": args.batch_size, "lr": args.lr, "wd": args.wd,
            "hidden_dim": args.hidden_dim, "seq_len": args.seq_len,
            "epochs": args.max_epochs, "loss": loss_name,
            "identical_to": "src/run_canonical_matrix.py defaults",
        },
    }
    out = training_path(root, ds, ph, seed, variant, lam)
    atomic_save_json(out, result, overwrite=args.overwrite)
    print(f"  [DONE] RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
          f"({metrics['train_time_s']}s) -> {out.name}")
    return "newly_trained"


def run(args) -> None:
    if not CANONICAL_AVAILABLE:
        sys.exit(f"[FATAL] Training requires the canonical pipeline modules "
                 f"(numpy/torch). Import failed with: {CANONICAL_IMPORT_ERROR}\n"
                 "Run this inside the experiment environment "
                 "(conda activate pytorch).")
    kind = args.experiment_kind
    root = resolve_root(args)

    print_plan(args, root)   # complete plan BEFORE any training

    if args.dry_run:
        print("\n[DRY RUN] no training executed.")
        return

    stats = {}
    t0 = time.time()
    for ds in args.datasets:
        lam = effective_lambda(args, ds)
        for ph in args.phs:
            for variant in args.variants:
                for seed in args.seeds:
                    st = cell_status(variant, ds, ph, seed, lam, root, kind)
                    if st == "exists":
                        print(f"  [SKIP] {variant} {ds} PH={ph} seed={seed} "
                              "(complete result exists in supplementary tree)")
                        key = "already_present"
                    elif isinstance(st, tuple) and st[0] == "copy_from_canonical":
                        key = copy_canonical_cell(st[1], ds, ph, seed,
                                                  variant, lam, root,
                                                  args.overwrite)
                    elif isinstance(st, tuple):
                        print(f"  [FAIL] {variant} {ds} PH={ph} seed={seed}: "
                              f"{st[0]}; remediation: {st[1]}")
                        key = st[0]
                    else:
                        key = run_one(variant, ds, ph, seed, lam, root,
                                      kind, args)
                    stats[key] = stats.get(key, 0) + 1
    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print(f"SUPPLEMENTARY TRAINING COMPLETE in {elapsed/60:.1f} min ({kind})")
    for k, n in sorted(stats.items()):
        print(f"  {k}: {n}")
    print(f"Results under: {root}")
    print("Next: python3 src/aggregate_supplementary_results.py")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated supplementary PH5/PH6 training (rev2, "
                    "10-variant matrix)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=DATASETS,
                        help="default: losloop+shenzhen for "
                             "canonical_completion, losloop only for "
                             "lambda_sensitivity")
    parser.add_argument("--phs", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--variants", nargs="+", default=list(SUPP_VARIANTS),
                        choices=SUPP_VARIANTS,
                        help="10-variant matrix; multi_gsl_weighted is "
                             "intentionally NOT available (todo5 #1)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0001)
    parser.add_argument("--lambda1", type=float, default=None,
                        help="optional guard; must match the kind's policy "
                             "(canonical per dataset, or 0.01 for the "
                             "sensitivity kind)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit before training")
    add_root_args(parser)
    parser.add_argument("--print-plan-only", action="store_true",
                        help="print the plan and exit (no torch import)")
    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = default_datasets(args.experiment_kind)
    if (args.experiment_kind == "lambda_sensitivity"
            and set(args.datasets) != {"losloop"}):
        sys.exit("[FATAL] The lambda_sensitivity experiment is Los-loop only "
                 "(todo5 #3).")

    root = resolve_root(args)
    if args.print_plan_only:
        print_plan(args, root)
        return
    run(args)


if __name__ == "__main__":
    main()
