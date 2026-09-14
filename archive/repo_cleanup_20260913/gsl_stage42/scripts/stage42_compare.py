"""Stage 42 — read-only consistency audit: submitted manuscript vs Stage 40 canonical results.

Constraints honored: no training, no DAGMA fitting, no modification of experimental code.
Reads only:
  - results/stage40_canonical/training/*.json  (Stage 40 canonical per-run results)
  - the submitted numbers hard-coded below, verified digit-for-digit against
    paper/sn-article_original.tex and paper/appendix/original_gsl_results.tex.

Outputs:
  - gsl_stage42/stage42_comparison.csv
  - gsl_stage42/stage42_comparison.json
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "results" / "stage40_canonical" / "training"
OUT_DIR = PROJECT_ROOT / "gsl_stage42"

# ---------------------------------------------------------------------------
# Submitted-manuscript RMSE values (single-run, original pipeline).
# Source of truth: paper/sn-article_original.tex
#   tab:gcn_gsl_comparison_combined  (GCN family)
#   tab:tgcn_gsl_comparison_combined (TGCN family)
# and paper/appendix/original_gsl_results.tex (same numbers).
# Key: (dataset, ph) -> {method -> rmse}
# ---------------------------------------------------------------------------
SUBMITTED = {
    ("shenzhen", 1): {"GCN": 5.958, "GCN-GSL": 4.886, "GCN-cGSL": 4.648,
                      "T-GCN": 4.866, "T-GCN-GSL": 4.214, "T-GCN-cGSL": 4.821},
    ("shenzhen", 2): {"GCN": 5.983, "GCN-GSL": 4.904, "GCN-cGSL": 4.672,
                      "T-GCN": 4.506, "T-GCN-GSL": 4.239, "T-GCN-cGSL": 4.534},
    ("shenzhen", 3): {"GCN": 5.991, "GCN-GSL": 4.958, "GCN-cGSL": 4.712,
                      "T-GCN": 4.685, "T-GCN-GSL": 4.344, "T-GCN-cGSL": 4.630},
    ("shenzhen", 4): {"GCN": 6.002, "GCN-GSL": 4.933, "GCN-cGSL": 4.726,
                      "T-GCN": 4.934, "T-GCN-GSL": 4.366, "T-GCN-cGSL": 4.774},
    ("losloop", 1): {"GCN": 7.724, "GCN-GSL": 7.527, "GCN-cGSL": 5.440,
                     "T-GCN": 6.588, "T-GCN-GSL": 4.818, "T-GCN-cGSL": 6.550},
    ("losloop", 2): {"GCN": 7.940, "GCN-GSL": 7.867, "GCN-cGSL": 5.806,
                     "T-GCN": 6.960, "T-GCN-GSL": 5.400, "T-GCN-cGSL": 6.915},
    ("losloop", 3): {"GCN": 8.102, "GCN-GSL": 8.073, "GCN-cGSL": 6.171,
                     "T-GCN": 7.361, "T-GCN-GSL": 5.846, "T-GCN-cGSL": 7.331},
    ("losloop", 4): {"GCN": 8.285, "GCN-GSL": 9.067, "GCN-cGSL": 6.745,
                     "T-GCN": 7.568, "T-GCN-GSL": 6.257, "T-GCN-cGSL": 7.539},
}

# Stage 40 variant_id for each submitted method name.
VARIANT_OF = {
    "GCN": "gcn_physical",
    "GCN-GSL": "gcn_gsl",
    "GCN-cGSL": "gcn_cgsl",
    "T-GCN": "physical",
    "T-GCN-GSL": "gsl",
    "T-GCN-cGSL": "cgsl",
}
FAMILY_OF = {"GCN": "GCN", "GCN-GSL": "GCN", "GCN-cGSL": "GCN",
             "T-GCN": "T-GCN", "T-GCN-GSL": "T-GCN", "T-GCN-cGSL": "T-GCN"}

SEEDS = [42, 43, 44, 45, 46]
DATASETS = ["shenzhen", "losloop"]
PHS = [1, 2, 3, 4]


def load_stage40_mean(dataset: str, ph: int, variant_id: str):
    vals = []
    for seed in SEEDS:
        p = TRAINING_DIR / f"{dataset}_ph{ph}_seed{seed}_{variant_id}.json"
        with open(p) as f:
            rec = json.load(f)
        if rec.get("status") != "complete":
            raise RuntimeError(f"incomplete record: {p}")
        vals.append(rec["rmse"])
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, var ** 0.5, min(vals), max(vals), vals


def classify(rel_diff: float, is_baseline: bool, rank_preserved: bool,
             relation_preserved: bool, favorable_reversal: bool) -> str:
    """Classify a submitted-vs-Stage40 discrepancy for one (dataset, ph, method) cell.

    Categories (per Stage 42 spec):
      essentially consistent
      consistent in qualitative trend but numerically different
      materially inconsistent
      directly contradictory

    rel_diff: percent change of Stage 40 mean vs submitted value.
    is_baseline: method is the family's physical-graph baseline (its scientific role
        is to serve as reference; within-family rank flips caused by another method's
        reversal are attributed to that method's row, not the baseline's).
    rank_preserved: method's rank within its family unchanged.
    relation_preserved: relation to the family's physical baseline unchanged
        (trivially True for the baseline itself).
    favorable_reversal: relation to baseline changed, but in the method's favor
        (e.g. cGSL now beats the physical graph where the submission had it worse).
    """
    if is_baseline:
        # The baseline keeps its role; rank movement is attributed to the other rows.
        if abs(rel_diff) <= 5.0:
            return "essentially consistent"
        return "consistent in qualitative trend but numerically different"
    if relation_preserved:
        if rank_preserved:
            if abs(rel_diff) <= 5.0:
                return "essentially consistent"
            return "consistent in qualitative trend but numerically different"
        return "materially inconsistent"  # ordering among learned variants flipped
    if favorable_reversal:
        return "materially inconsistent"
    return "directly contradictory"


def main():
    rows = []
    for dataset in DATASETS:
        for ph in PHS:
            sub = SUBMITTED[(dataset, ph)]
            stage40 = {}
            for method, variant in VARIANT_OF.items():
                mean, std, mn, mx, per_seed = load_stage40_mean(dataset, ph, variant)
                stage40[method] = mean
                s = sub[method]
                abs_diff = stage40[method] - s
                rel_diff = abs_diff / s * 100.0
                row = {
                    "dataset": dataset,
                    "ph": ph,
                    "family": FAMILY_OF[method],
                    "method": method,
                    "submitted_rmse": s,
                    "stage40_mean_rmse": round(mean, 4),
                    "stage40_std_rmse": round(std, 4),
                    "stage40_min_rmse": round(mn, 4),
                    "stage40_max_rmse": round(mx, 4),
                    "abs_diff_stage40_minus_submitted": round(abs_diff, 4),
                    "rel_diff_pct": round(rel_diff, 2),
                }
                rows.append(row)
            # Ranking preservation and baseline-relation analysis per family
            for family in ("GCN", "T-GCN"):
                methods = [m for m in VARIANT_OF if FAMILY_OF[m] == family]
                baseline = "GCN" if family == "GCN" else "T-GCN"
                s_rank = {m: i + 1 for i, (m, _) in enumerate(
                    sorted(((m, sub[m]) for m in methods), key=lambda t: t[1]))}
                c_rank = {m: i + 1 for i, (m, _) in enumerate(
                    sorted(((m, stage40[m]) for m in methods), key=lambda t: t[1]))}
                for m in methods:
                    row = next(r for r in rows
                               if r["dataset"] == dataset and r["ph"] == ph and r["method"] == m)
                    row["submitted_rank"] = s_rank[m]
                    row["stage40_rank"] = c_rank[m]
                    rank_preserved = s_rank[m] == c_rank[m]
                    if m == baseline:
                        s_rel, c_rel = 0, 0
                    else:
                        s_rel = 1 if sub[m] < sub[baseline] else -1
                        c_rel = 1 if stage40[m] < stage40[baseline] else -1
                    relation_preserved = (s_rel == c_rel)
                    favorable_reversal = (not relation_preserved) and (c_rel == 1)
                    row["rank_preserved"] = "yes" if rank_preserved else "no"
                    row["baseline_relation_submitted"] = {0: "is-baseline", 1: "better", -1: "worse"}[s_rel]
                    row["baseline_relation_stage40"] = {0: "is-baseline", 1: "better", -1: "worse"}[c_rel]
                    row["classification"] = classify(
                        row["rel_diff_pct"], m == baseline, rank_preserved,
                        relation_preserved, favorable_reversal)

    OUT_DIR.mkdir(exist_ok=True)
    fields = ["dataset", "ph", "family", "method", "submitted_rmse", "stage40_mean_rmse",
              "stage40_std_rmse", "stage40_min_rmse", "stage40_max_rmse",
              "abs_diff_stage40_minus_submitted", "rel_diff_pct", "classification",
              "submitted_rank", "stage40_rank", "rank_preserved",
              "baseline_relation_submitted", "baseline_relation_stage40"]
    with open(OUT_DIR / "stage42_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(OUT_DIR / "stage42_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"wrote {len(rows)} rows -> {OUT_DIR / 'stage42_comparison.csv'}")


if __name__ == "__main__":
    main()
