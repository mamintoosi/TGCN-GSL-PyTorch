#!/usr/bin/env python3
"""
Aggregate supplementary PH5/PH6 results — rev2 (todo5), stdlib-only
===================================================================

Reads ONLY from one supplementary root (selected by --experiment-kind or
--output-root), never from canonical directories, and writes a CSV + JSON
summary under that same root. Main-paper tables are never touched.

todo5 requirements implemented here:
  * the 10-variant matrix is expected; multi_gsl_weighted / multi_gsl_mix
    files, if any were ever produced, are reported as unexpected rather than
    silently mixed into the 10-variant table;
  * provenance is preserved and reported per cell: newly_trained vs
    copied_from_canonical (from the result JSONs' cell_provenance field);
  * the lambda_1 distinction is preserved via training_lambda*/ tags and the
    recorded effective lambda in every JSON;
  * missing experiments are reported explicitly (missing_seeds listed,
    status column) and are NEVER treated as zero;
  * per-seed RMSE values are kept in the JSON payload;
  * graph edge counts / fit times / graph provenance come from the
    graphs_<tag>/*_meta.json and *.provenance.json sidecars when present.

Usage:
  python3 src/aggregate_supplementary_results.py                       # completion root
  python3 src/aggregate_supplementary_results.py --experiment-kind lambda_sensitivity
  python3 src/aggregate_supplementary_results.py --output-root <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.supp_paths import (  # noqa: E402
    SEEDS, DATASETS, SUPP_VARIANTS, default_root_for_kind,
    add_root_args, resolve_root, training_dir, graph_dir, summary_paths,
    lambda_tag,
)

# The expected 10-variant matrix (todo5 #1). Anything else found on disk is
# reported as unexpected, not silently aggregated.
VARIANT_ORDER = list(SUPP_VARIANTS)

LAM_BY_TAG = {"lambda001": 0.01, "lambda002": 0.02}


def find_lambda_tags(root: Path):
    return sorted(d.name.replace("training_", "")
                  for d in root.glob("training_lambda*") if d.is_dir())


def load_cell(train_d: Path, dataset: str, ph: int, variant: str, tag: str):
    """(rows, missing_seeds); missing/invalid files are explicit."""
    rows, missing = [], []
    for seed in SEEDS:
        p = train_d / f"{dataset}_ph{ph}_seed{seed}_{variant}_{tag}.json"
        if not p.exists():
            missing.append(seed)
            continue
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            missing.append(seed)
            continue
        if d.get("status") != "complete":
            missing.append(seed)
            continue
        rows.append(d)
    return rows, missing


def unexpected_variants(root: Path, tag: str):
    """Report variant files outside the 10-variant matrix (e.g. Weighted)."""
    train_d = root / f"training_{tag}"
    found = set()
    for p in train_d.glob("*.json"):
        parts = p.stem.split("_")
        # {dataset}_ph{ph}_seed{seed}_{variant...}_{tag}
        if len(parts) >= 5 and parts[-1] == tag:
            variant = "_".join(parts[3:-1])
            if variant not in VARIANT_ORDER:
                found.add(variant)
    return sorted(found)


def graph_meta_for(root: Path, tag: str, dataset: str, ph: int):
    prefix = {"losloop": "los", "shenzhen": "sz"}[dataset]
    lam = LAM_BY_TAG.get(tag)
    gd = graph_dir(root, lam) if lam else root / f"graphs_{tag}"
    info = {}
    for gtype in ("gsl", "cgsl"):
        meta_p = gd / f"{prefix}_ph{ph}_{tag}_{gtype}_meta.json"
        prov_p = gd / f"{prefix}_ph{ph}_{tag}_{gtype}_A_binary.provenance.json"
        entry = {}
        if meta_p.exists():
            try:
                d = json.loads(meta_p.read_text())
                entry["n_edges"] = d.get(
                    "n_retained_directed_edges",
                    d.get("n_retained_directed_entries"))
                entry["fit_time_s"] = d.get("fitting_time_s")
                entry["graph_provenance"] = (
                    "fresh_fit" if d.get("kind", "").endswith("fresh_fit")
                    else "copied_from_canonical"
                    if "copied_from_canonical" in d.get("kind", "")
                    else d.get("kind"))
            except (json.JSONDecodeError, OSError):
                entry = {"error": "unreadable meta"}
        elif prov_p.exists():
            try:
                d = json.loads(prov_p.read_text())
                entry["graph_provenance"] = "copied_from_canonical"
                entry["source_path"] = d.get("source_path")
                entry["source_sha256"] = d.get("source_sha256")
            except (json.JSONDecodeError, OSError):
                entry = {"error": "unreadable provenance"}
        if entry:
            info[gtype] = entry
    ml_prov = gd / "multilag_blocks" / \
        f"{prefix}_multilag_blocks_provenance_{tag}.json"
    if ml_prov.exists():
        try:
            d = json.loads(ml_prov.read_text())
            info["multilag"] = {
                "graph_provenance": "copied_from_canonical",
                "refit_required": d.get("dagma_refit_required"),
                "n_blocks": len(d.get("blocks", [])),
            }
        except (json.JSONDecodeError, OSError):
            info["multilag"] = {"error": "unreadable provenance"}
    return info


def aggregate(args) -> None:
    root = resolve_root(args)
    tags = find_lambda_tags(root)
    if not tags:
        sys.exit(f"[FATAL] No training_lambda*/ directories under {root}. "
                 "Run src/run_supplementary_training.py for this "
                 "experiment kind first.")

    print("=" * 78)
    print("SUPPLEMENTARY RESULTS AGGREGATION (rev2; read-only over one root)")
    print(f"Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output root  : {root}")
    print(f"Lambda tags  : {tags}")
    print(f"Expected     : 10-variant matrix (MultiGSL-Weighted excluded)")
    print("=" * 78)

    rows_out = []
    totals = {"logical": 0, "newly_trained": 0, "copied": 0,
              "partial": 0, "missing_cells": 0}
    unexpected_all = {}

    for tag in tags:
        lam = LAM_BY_TAG.get(tag)
        train_d = training_dir(root, lam) if lam else root / f"training_{tag}"
        unexpected = unexpected_variants(root, tag)
        if unexpected:
            unexpected_all[tag] = unexpected
        for dataset in DATASETS:
            for ph in (5, 6):
                for variant in VARIANT_ORDER:
                    rows, missing = load_cell(train_d, dataset, ph,
                                              variant, tag)
                    if not rows and len(missing) == len(SEEDS):
                        totals["missing_cells"] += 1
                        continue          # cell not part of this run
                    provenances = sorted({r.get("cell_provenance", "unknown")
                                          for r in rows})
                    status = ("complete" if not missing
                              else f"partial (missing seeds: {missing})")
                    if missing:
                        totals["partial"] += 1
                    rmses = [r["rmse"] for r in rows]
                    maes = [r["mae"] for r in rows]
                    lam_vals = sorted({r.get("lambda1") for r in rows})
                    row = {
                        "dataset": dataset, "ph": ph,
                        "variant": variant,
                        "display": rows[0]["display_name"] if rows else "",
                        "lambda_tag": tag,
                        "lambda1": (lam_vals[0] if len(lam_vals) == 1
                                    else lam_vals),
                        "n_seeds": len(rows),
                        "missing_seeds": missing,
                        "cell_provenance": "+".join(provenances),
                        "rmse_mean": round(st.mean(rmses), 4) if rmses else None,
                        "rmse_std": round(st.stdev(rmses), 4) if len(rmses) > 1 else None,
                        "mae_mean": round(st.mean(maes), 4) if maes else None,
                        "mae_std": round(st.stdev(maes), 4) if len(maes) > 1 else None,
                        "rmse_per_seed": {str(r["seed"]): r["rmse"] for r in rows},
                        "n_edges": rows[0].get("n_edges") if rows else None,
                        "graph_source": rows[0].get("graph_source") if rows else None,
                        "graph_meta": graph_meta_for(root, tag, dataset, ph),
                        "status": status,
                    }
                    rows_out.append(row)
                    totals["logical"] += 1
                    if provenances == ["newly_trained"]:
                        totals["newly_trained"] += 1
                    elif "copied_from_canonical" in provenances:
                        totals["copied"] += 1

    # ---- print table -------------------------------------------------------
    hdr = (f"{'dataset':9s} {'PH':>2s} {'method':18s} {'lambda':>6s} "
           f"{'n':>2s} {'RMSE mean':>9s} {'std':>6s} {'provenance':24s} status")
    print(hdr)
    print("-" * len(hdr))
    for r in rows_out:
        lam_s = (f"{r['lambda1']:g}" if isinstance(r["lambda1"], float)
                 else str(r["lambda1"]))
        rm = (f"{r['rmse_mean']:.4f}" if r["rmse_mean"] is not None else "--")
        rs = (f"{r['rmse_std']:.4f}" if r["rmse_std"] is not None else "--")
        print(f"{r['dataset']:9s} {r['ph']:>2d} {r['display']:18s} "
              f"{lam_s:>6s} {r['n_seeds']:>2d} {rm:>9s} {rs:>6s} "
              f"{r['cell_provenance']:24s} {r['status']}")

    if unexpected_all:
        print("\n[NOTE] Unexpected variant files (outside the 10-variant "
              "matrix; reported, NOT aggregated):")
        for tag, vs in unexpected_all.items():
            print(f"  {tag}: {vs}")
    if any(r["status"] != "complete" for r in rows_out):
        print("\n[NOTE] Missing experiments are listed with their missing "
              "seeds — NOT treated as zero, excluded from means.")
    if totals["missing_cells"]:
        print(f"[NOTE] {totals['missing_cells']} expected (dataset, PH, "
              "variant) cells have no results at all and are absent from "
              "the table (they were never part of a completed run).")

    print("\nCell totals:", json.dumps(totals, indent=2))

    # ---- write summary (JSON + CSV) ----------------------------------------
    sp = summary_paths(root)
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_root": str(root),
        "matrix": "10 variants (MultiGSL-Weighted excluded per todo5)",
        "totals": totals,
        "unexpected_variants": unexpected_all,
        "groups": rows_out,
    }
    tmp = sp["json"].with_name(sp["json"].name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(sp["json"])

    with open(sp["csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "ph", "variant", "display", "lambda_tag", "lambda1",
            "n_seeds", "missing_seeds", "cell_provenance", "rmse_mean",
            "rmse_std", "mae_mean", "mae_std", "n_edges", "status"])
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    print(f"\nWrote: {sp['json']}")
    print(f"Wrote: {sp['csv']}")
    print("(Both under the supplementary root; no main-paper table or "
          "canonical result was touched.)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate supplementary PH5/PH6 results (rev2, "
                    "stdlib-only)")
    add_root_args(parser)
    args = parser.parse_args()
    aggregate(args)


if __name__ == "__main__":
    main()
