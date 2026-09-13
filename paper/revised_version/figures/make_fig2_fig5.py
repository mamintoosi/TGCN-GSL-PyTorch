"""Regenerate revised-version figures 2 (5-seed RMSE bars) and 5 (lag ablation).

fig2: Stage 41 five-seed mean RMSE (Los-loop PH1), error bars = sample std.
fig5: Stage 26 validation C lag ablation (seed 42), labeled as supplementary.
fig8: copied from previous_revision (compact convergence); not regenerated here.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "revised_version" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
S41 = ROOT / "gsl_stage41" / "stage41_summary.csv"
C_ABL = ROOT / "results" / "stage26_validation" / "stage26_validation_C_losloop_ph1.csv"


def load_stage41_los_ph1():
    rows = {}
    with open(S41, newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "losloop" and int(r["ph"]) == 1:
                rows[r["variant_id"]] = {
                    "method": r["method"],
                    "rmse_mean": float(r["rmse_mean"]),
                    "rmse_std": float(r["rmse_std"]),
                    "n_edges": int(float(r["n_edges"])),
                }
    return rows


def fig2_rmse_bars():
    """Five-seed mean RMSE bars for key Los-loop PH1 methods (Stage 40)."""
    rows = load_stage41_los_ph1()
    order = [
        ("physical", "T-GCN\n(Physical)"),
        ("gcn_physical", "GCN\n(Physical)"),
        ("no_spatial", "T-GCN-NoSpatial"),
        ("gcn_no_spatial", "GCN-NoSpatial"),
        ("gsl", "T-GCN-GSL"),
        ("cgsl", "T-GCN-cGSL"),
        ("multi_gsl", "T-GCN-MultiGSL"),
        ("multi_gsl_weighted", "T-GCN-MultiGSL\n-Weighted"),
        ("multi_gsl_mix", "T-GCN-MultiGSL-Mix"),
        ("gcn_multigsl", "GCN-MultiGSL\n(union)"),
    ]
    labels, means, stds, edges = [], [], [], []
    for vid, lab in order:
        if vid not in rows:
            continue
        labels.append(lab)
        means.append(rows[vid]["rmse_mean"])
        stds.append(rows[vid]["rmse_std"])
        edges.append(rows[vid]["n_edges"])

    colors = []
    for lab in labels:
        if "NoSpatial" in lab:
            colors.append("#1d3557")
        elif "Mix" in lab:
            colors.append("#1f7a6f")
        elif "MultiGSL" in lab or "union" in lab:
            colors.append("#2a9d8f")
        elif "GSL" in lab:
            colors.append("#8ab17d")
        else:
            colors.append("#9aa0a6")

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3.5,
        width=0.72,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        error_kw=dict(elinewidth=1.0, ecolor="#333333", capthick=1.0),
    )
    for xi, m, ne in zip(x, means, edges):
        ax.text(
            xi,
            m + 0.12,
            f"{m:.2f}\n{ne}e",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ns = rows["no_spatial"]["rmse_mean"]
    ax.axhline(ns, color="#1d3557", ls="--", lw=1.1, alpha=0.8)
    ax.text(
        len(labels) - 0.35,
        ns + 0.05,
        "T-GCN-NoSpatial",
        color="#1d3557",
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test RMSE (mean over 5 seeds)")
    ax.set_title("Los-loop PH1: five-seed mean RMSE (whiskers: sample SD)")
    ax.set_ylim(0, max(means) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_rmse_bars_stage40.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_rmse_bars_stage40.png", bbox_inches="tight")
    print("Wrote fig2_rmse_bars_stage40")


def fig5_lag_ablation():
    """Supplementary lag ablation (Stage 26 C, seed 42)."""
    rows = []
    with open(C_ABL, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    def short(m):
        if m == "NoGraph":
            return "T-GCN-NoSpatial"
        if m == "GatedMulti_all":
            return "All 3 lags"
        return m.replace("GatedMulti_", "").replace("lag_", "lag ")

    dag = [r for r in rows if r["method"] != "NoGraph"]
    dag.sort(key=lambda r: float(r["rmse"]))
    nograph = next(r for r in rows if r["method"] == "NoGraph")

    labels = [short(r["method"]) for r in dag] + [short(nograph["method"])]
    rmses = [float(r["rmse"]) for r in dag] + [float(nograph["rmse"])]
    edges = [int(r["n_edges"]) for r in dag] + [int(nograph["n_edges"])]

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=160)
    y = np.arange(len(labels))[::-1]
    colors = []
    for r in dag:
        lu = r.get("lags_used", "")
        if lu == "lag_1+lag_2+lag_3":
            colors.append("#1f7a6f")
        elif "+" in lu:
            colors.append("#2a9d8f")
        else:
            colors.append("#8ab17d")
    colors.append("#1d3557")

    ax.barh(y, rmses, color=colors, edgecolor="black", linewidth=0.5, height=0.65)
    for yi, rmse, ne in zip(y, rmses, edges):
        ax.text(rmse + 0.02, yi, f"{rmse:.3f}  ({ne} edges)", va="center", fontsize=9)
    ns = float(nograph["rmse"])
    ax.axvline(ns, color="#1d3557", ls="--", lw=1.1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("RMSE")
    ax.set_xlim(0, max(rmses) * 1.35)
    ax.set_title(
        "Lag ablation — Los-loop PH1 (seed 42)\n"
        "Supplementary; not part of the five-seed main matrix"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_lag_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig5_lag_ablation.png", bbox_inches="tight")
    print("Wrote fig5_lag_ablation")


if __name__ == "__main__":
    fig2_rmse_bars()
    fig5_lag_ablation()
