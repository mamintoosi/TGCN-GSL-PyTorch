#!/usr/bin/env python3
"""Regenerate manuscript figures 2, 5, 8 from repo artifacts (Stage 41 / Stage 26).

Outputs (paper/revised_version/figures/):
  fig2_rmse_keymethods.pdf  — 5-seed mean±std, Los PH1 key methods (from stage41)
  fig5_lag_ablation.pdf     — lag ablation, seed 42 (from stage26_validation_C)
  fig8_train_loss.pdf       — compact train-loss curves, seed 42 (checkpoints)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Physical": "#9aa0a6",
    "NoSpatial": "#1d3557",
    "GSL": "#8ab17d",
    "cGSL": "#6b8f71",
    "MultiGSL": "#2a9d8f",
    "Weighted": "#7eb8a8",
    "Mix": "#1f7a6f",
    "GCN_Physical": "#b0b0b0",
    "GCN_NoSpatial": "#4a6fa5",
}


def fig2_stage41_rmse():
    """Five-seed mean±std RMSE, Los-loop PH1, key T-GCN methods + GCN-NoSpatial."""
    path = ROOT / "gsl_stage41" / "stage41_summary.csv"
    rows = list(csv.DictReader(open(path, newline="")))
    key = [
        ("physical", "T-GCN\n(Physical)", COLORS["Physical"]),
        ("no_spatial", "T-GCN-\nNoSpatial", COLORS["NoSpatial"]),
        ("gsl", "T-GCN-GSL", COLORS["GSL"]),
        ("cgsl", "T-GCN-cGSL", COLORS["cGSL"]),
        ("multi_gsl", "T-GCN-\nMultiGSL", COLORS["MultiGSL"]),
        ("multi_gsl_weighted", "MultiGSL-\nWeighted", COLORS["Weighted"]),
        ("multi_gsl_mix", "MultiGSL-\nMix", COLORS["Mix"]),
        ("gcn_no_spatial", "GCN-\nNoSpatial", COLORS["GCN_NoSpatial"]),
    ]
    means, stds, labels, colors = [], [], [], []
    for vid, lab, col in key:
        r = next(
            x
            for x in rows
            if x["dataset"] == "losloop" and int(x["ph"]) == 1 and x["variant_id"] == vid
        )
        means.append(float(r["rmse_mean"]))
        stds.append(float(r["rmse_std"]))
        labels.append(lab)
        colors.append(col)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        error_kw=dict(elinewidth=1.0, ecolor="#333333"),
    )
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.08, f"{m:.2f}", ha="center", fontsize=9)
    ns = means[1]
    ax.axhline(ns, color=COLORS["NoSpatial"], ls="--", lw=1.2, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Test RMSE (km/h)", fontsize=10)
    ax.set_title(
        "Los-loop PH1 — key configurations (five-seed mean ± sample SD)",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(means) * 1.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_rmse_keymethods.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_rmse_keymethods.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Wrote fig2_rmse_keymethods")


def fig5_lag_ablation():
    path = ROOT / "results" / "stage26_validation" / "stage26_validation_C_losloop_ph1.csv"
    rows = list(csv.DictReader(open(path, newline="")))
    dag = [r for r in rows if r["method"] != "NoGraph"]
    dag.sort(key=lambda r: float(r["rmse"]))
    nograph = next(r for r in rows if r["method"] == "NoGraph")
    labels, rmses, edges = [], [], []
    for r in dag:
        labels.append(r.get("lags_used", r["method"]).replace("lag_", ""))
        rmses.append(float(r["rmse"]))
        edges.append(int(r["n_edges"]))
    labels.append("T-GCN-NoSpatial")
    rmses.append(float(nograph["rmse"]))
    edges.append(int(nograph["n_edges"]))
    n_lags = [0]
    for lab in labels[:-1]:
        n_lags.append(lab.count("+") + 1)

    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(0, 3)
    colors = [COLORS["NoSpatial"] if nl == 0 else cmap(norm(nl)) for nl in n_lags]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, rmses, color=colors, edgecolor="black", linewidth=0.5, height=0.65)
    for yi, rmse, ne in zip(y, rmses, edges):
        ax.text(rmse + 0.02, yi, f"{rmse:.3f}  ({ne} edges)", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("RMSE (km/h)", fontsize=10)
    ax.set_title(
        "Lag ablation — Los-loop PH1 (seed 42)\nlonger bars = worse; "
        "supplementary single-seed study",
        fontsize=11,
    )
    ax.axvline(float(nograph["rmse"]), color=COLORS["NoSpatial"], ls="--", lw=1.2, alpha=0.6)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=cmap(norm(1)), label="1 lag"),
            Patch(facecolor=cmap(norm(2)), label="2 lags"),
            Patch(facecolor=cmap(norm(3)), label="3 lags (all)"),
            Patch(facecolor=COLORS["NoSpatial"], label="T-GCN-NoSpatial"),
        ],
        loc="lower right",
        fontsize=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(rmses) * 1.28)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_lag_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig5_lag_ablation.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Wrote fig5_lag_ablation")


def fig8_train_loss():
    base = ROOT / "results" / "stage26_checkpoint"
    methods = [
        ("los_ph1_seed42_nograph", "T-GCN-NoSpatial", COLORS["NoSpatial"]),
        ("los_ph1_seed42_multi_graph_fixed", "T-GCN-MultiGSL", COLORS["MultiGSL"]),
        ("los_ph1_seed42_gated_multi", "T-GCN-MultiGSL-Mix", COLORS["Mix"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for folder, name, color in methods:
        hist = json.load(open(base / folder / "train_loss_history.json"))
        losses = hist["train_losses"]
        ep = np.arange(1, len(losses) + 1)
        axes[0].plot(ep, losses, label=name, color=color, lw=1.6)
        axes[1].plot(ep, np.log10(np.maximum(losses, 1e-12)), label=name, color=color, lw=1.6)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("(a) Training loss (Los-loop PH1, seed 42)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("log10(training loss)")
    axes[1].set_title("(b) Log-scale training loss")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Compact convergence diagnostics (checkpoint train-loss histories)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_train_loss.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig8_train_loss.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Wrote fig8_train_loss")


if __name__ == "__main__":
    fig2_stage41_rmse()
    fig5_lag_ablation()
    fig8_train_loss()
