#!/usr/bin/env python3
"""Summarize Stage 58 sparsity sweep and write analysis markdown."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "results/stage58_sparsity_sweep/full.csv")

COLORS = {
    "NoSpatial": "#2196F3",
    "RandTopK": "#FF9800",
    "CorrTopK": "#FFB74D",
    "T-GCN-MultiGSL": "#009688",
    "T-GCN-MultiGSL-Mix": "#E91E63",
}

# Stage 40 MultiGSL / Mix / NoSpatial (already in stage41)
MULTIGSL = 4.84078
MULTIGSL_STD = 0.114613
MIX = 4.49144
MIX_STD = 0.140346

summary = (
    df.groupby(["method", "budget"])["RMSE"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
print(summary.to_string(index=False))

# Figure: RMSE vs budget, log-ish x, with DAGMA reference lines
fig, ax = plt.subplots(figsize=(8.5, 5))
for method, col in [("RandTopK", COLORS["RandTopK"]), ("CorrTopK", COLORS["CorrTopK"])]:
    sub = summary[summary.method == method].sort_values("budget")
    ax.errorbar(
        sub.budget,
        sub["mean"],
        yerr=sub["std"],
        marker="o",
        color=col,
        capsize=3,
        label=method,
        lw=1.8,
    )
    for b, m in zip(sub.budget, sub["mean"]):
        ax.text(b, m + 0.08, f"{m:.2f}", ha="center", fontsize=8)

ns = summary[summary.method == "NoSpatial"]["mean"].iloc[0]
ns_s = summary[summary.method == "NoSpatial"]["std"].iloc[0]
ax.axhline(ns, color=COLORS["NoSpatial"], ls="--", lw=1.2, label=f"Graph-free baseline ({ns:.2f})")
ax.axhline(MULTIGSL, color=COLORS["T-GCN-MultiGSL"], ls=":", lw=1.6, label=f"T-GCN-MultiGSL @30 ({MULTIGSL:.2f})")
ax.axhline(MIX, color=COLORS["T-GCN-MultiGSL-Mix"], ls="-.", lw=1.6, label=f"Mix @30 ({MIX:.2f})")

ax.set_xlabel("Matched edge budget K (directed off-diagonal)")
ax.set_ylabel("Test RMSE (km/h)")
ax.set_title("Los-loop PH1: sparsity sweep (five-seed mean ± SD)\nT-GCN; K-edge random vs correlation graphs vs graph-free / DAGMA")
ax.set_xticks([10, 20, 30, 50, 80])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout()

OUT = ROOT / "paper" / "revised_version" / "figures"
fig.savefig(OUT / "sparsity_sweep_los_ph1.pdf", bbox_inches="tight")
fig.savefig(OUT / "sparsity_sweep_los_ph1.png", bbox_inches="tight")
print("Wrote sparsity_sweep_los_ph1")

# Markdown analysis
lines = [
    "# Stage 58 — Sparsity edge-budget sweep (Los-loop PH1)",
    "",
    "## Summary (five-seed mean ± sample std, RMSE)",
    "",
    "| Method | K | mean | std | vs graph-free |",
    "|--------|---|------|-----|----------------|",
]
for _, r in summary.iterrows():
    d = r["mean"] - ns
    lines.append(
        f"| {r.method} | {int(r.budget)} | {r['mean']:.3f} | {r['std']:.3f} | {d:+.3f} |"
    )
lines += [
    "",
    f"Graph-free baseline: **{ns:.3f} ± {ns_s:.3f}**",
    f"T-GCN-MultiGSL (Stage 40, K=30 lag slots): **{MULTIGSL:.3f} ± {MULTIGSL_STD:.3f}**",
    f"T-GCN-MultiGSL-Mix (Stage 40, same graphs, gated): **{MIX:.3f} ± {MIX_STD:.3f}**",
    "",
    "## Findings",
    "",
    "1. **Sparsity alone is not enough.** For every K in {10,20,30,50,80}, both RandTopK and CorrTopK are **at or above** the graph-free baseline (worse or ~equal).",
    "2. **More random edges make things worse** (RandTopK: 5.37 → 7.03 from K=10 to 80).",
    "3. **Correlation graphs are near the graph-free baseline at small K and degrade as K grows** (CorrTopK K=10 ≈ 5.27 vs baseline 5.23; K=80 → 5.64).",
    "4. **At the same 30-edge budget**, DAGMA MultiGSL (4.84) and Mix (4.49) beat both heuristic graphs and the graph-free baseline.",
    "5. Conclusion for the paper: **edge count / sparsity alone does not produce the Los-loop multi-lag gain**; learned placement (+ lag-aligned use) is required.",
    "",
    "## Protocol",
    "",
    "- Dataset: Los-loop, PH=1, T-GCN, 50 epochs, seeds 42–46",
    "- RandTopK: K random directed off-diagonal edges (redrawn per seed)",
    "- CorrTopK: top-K |Pearson| on training data",
    "- Same training pipeline as Stage 32 / Stage 40",
    "- DAGMA references from Stage 40 (not re-run in this sweep)",
    "",
]
(ROOT / "results" / "stage58_sparsity_sweep" / "ANALYSIS.md").write_text(
    "\n".join(lines), encoding="utf-8"
)
print("Wrote ANALYSIS.md")
