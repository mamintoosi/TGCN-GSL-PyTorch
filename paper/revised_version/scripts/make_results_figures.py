#!/usr/bin/env python3
"""Regenerate Results figures with unified method colors and naming.

Canonical manuscript method names (match Tables 2–3):
  T-GCN, GCN, T-GCN-NoSpatial, GCN-NoSpatial, T-GCN-GSL, T-GCN-cGSL,
  T-GCN-MultiGSL, T-GCN-MultiGSL-Weighted, T-GCN-MultiGSL-Mix, GCN-MultiGSL

Palette (same color for the same method in every figure):
  T-GCN (Physical) = red; GCN (Physical) = deep orange;
  *NoSpatial = blue family; GSL/cGSL = green family;
  MultiGSL = teal; Weighted = cyan; Mix = magenta.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
S41 = ROOT / "gsl_stage41" / "stage41_summary.csv"
CKPT = ROOT / "results" / "stage26_checkpoint"

# Unified palette (extends previous_revision Material palette)
COLORS = {
    "T-GCN": "#F44336",
    "GCN": "#FF5722",
    "T-GCN-NoSpatial": "#2196F3",
    "GCN-NoSpatial": "#3F51B5",
    "T-GCN-GSL": "#4CAF50",
    "T-GCN-cGSL": "#8BC34A",
    "GCN-GSL": "#9CCC65",
    "GCN-cGSL": "#CDDC39",
    "T-GCN-MultiGSL": "#009688",
    "T-GCN-MultiGSL-Weighted": "#00BCD4",
    "T-GCN-MultiGSL-Mix": "#7B1FA2",  # purple
    "GCN-MultiGSL": "#795548",
    "RandTop30": "#FF9800",
    "CorrTop30": "#FFB74D",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

# Stage 41 variant_id -> manuscript display name
VID = {
    "physical": "T-GCN",
    "gcn_physical": "GCN",
    "no_spatial": "T-GCN-NoSpatial",
    "gcn_no_spatial": "GCN-NoSpatial",
    "gsl": "T-GCN-GSL",
    "cgsl": "T-GCN-cGSL",
    "gcn_gsl": "GCN-GSL",
    "gcn_cgsl": "GCN-cGSL",
    "multi_gsl": "T-GCN-MultiGSL",
    "multi_gsl_weighted": "T-GCN-MultiGSL-Weighted",
    "multi_gsl_mix": "T-GCN-MultiGSL-Mix",
    "gcn_multigsl": "GCN-MultiGSL",
}

SHORT = {
    "T-GCN": "T-GCN\n(Physical)",
    "GCN": "GCN\n(Physical)",
    "T-GCN-NoSpatial": "Graph-free\nbaseline",
    "GCN-NoSpatial": "Graph-free\nbaseline",
    "T-GCN-GSL": "T-GCN-GSL",
    "T-GCN-cGSL": "T-GCN-cGSL",
    "GCN-GSL": "GCN-GSL",
    "GCN-cGSL": "GCN-cGSL",
    "T-GCN-MultiGSL": "T-GCN-\nMultiGSL",
    "T-GCN-MultiGSL-Weighted": "MultiGSL-\nWeighted",
    "T-GCN-MultiGSL-Mix": "MultiGSL-Mix",
    "GCN-MultiGSL": "GCN-MultiGSL\n(union)",
}


def load_los_ph1():
    rows = {}
    with open(S41, newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "losloop" and int(r["ph"]) == 1:
                rows[r["variant_id"]] = r
    return rows


def _bar_family(order, title, out_stem, ns_vid):
    rows = load_los_ph1()
    labels, means, stds, colors = [], [], [], []
    for vid in order:
        r = rows[vid]
        name = VID[vid]
        labels.append(SHORT[name])
        means.append(float(r["rmse_mean"]))
        stds.append(float(r["rmse_std"]))
        colors.append(COLORS[name])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(labels))
    ax.bar(
        x, means, yerr=stds, capsize=3.5, width=0.7,
        color=colors, edgecolor="black", linewidth=0.5,
        error_kw=dict(elinewidth=1.0, ecolor="#333333"),
    )
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.18, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    ns = float(rows[ns_vid]["rmse_mean"])
    ns_name = VID[ns_vid]
    ax.axhline(ns, color=COLORS[ns_name], ls="--", lw=1.1, alpha=0.85)
    ax.text(len(labels) - 0.35, ns + 0.08, "Graph-free baseline", color=COLORS[ns_name],
            fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Test RMSE (km/h)")
    ax.set_title(title)
    ax.set_ylim(0, max(means) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{out_stem}.png", bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_stem)


def fig_rmse_comparison():
    """Separate figures for T-GCN family and GCN family."""
    _bar_family(
        ["physical", "gsl", "cgsl", "multi_gsl", "multi_gsl_weighted", "multi_gsl_mix"],
        "Los-loop PH1: T-GCN family (five-seed mean RMSE ± sample SD)",
        "rmse_comparison_tgcn_los_ph1",
        "no_spatial",
    )
    _bar_family(
        ["gcn_physical", "gcn_gsl", "gcn_cgsl", "gcn_multigsl"],
        "Los-loop PH1: GCN family (five-seed mean RMSE ± sample SD)",
        "rmse_comparison_gcn_los_ph1",
        "gcn_no_spatial",
    )
    # remove old combined file if present
    for ext in (".pdf", ".png"):
        old = OUT / f"rmse_comparison_los_ph1{ext}"
        if old.exists():
            old.unlink()
            print("Removed", old.name)


def fig_sparsity_controls():
    """Matched-sparsity bar chart for Table 4 (Los-loop PH1, five seeds)."""
    rows = load_los_ph1()
    # Stage 41 sparse summary embedded in stage41_summary.json if available
    import json as _json

    s41 = _json.load(open(ROOT / "gsl_stage41" / "stage41_summary.json"))
    sparse = s41.get("sparse_controls_stage32", {}).get("summary", {})
    items = [
        ("RandTop30", sparse.get("RandTop30", {}), COLORS["RandTop30"]),
        ("CorrTop30", sparse.get("CorrTop30", {}), COLORS["CorrTop30"]),
        ("T-GCN-NoSpatial", {
            "mean": float(rows["no_spatial"]["rmse_mean"]),
            "std": float(rows["no_spatial"]["rmse_std"]),
        }, COLORS["T-GCN-NoSpatial"]),
        ("T-GCN-MultiGSL", {
            "mean": float(rows["multi_gsl"]["rmse_mean"]),
            "std": float(rows["multi_gsl"]["rmse_std"]),
        }, COLORS["T-GCN-MultiGSL"]),
        ("T-GCN-MultiGSL-Mix", {
            "mean": float(rows["multi_gsl_mix"]["rmse_mean"]),
            "std": float(rows["multi_gsl_mix"]["rmse_std"]),
        }, COLORS["T-GCN-MultiGSL-Mix"]),
    ]
    labels, means, stds, colors = [], [], [], []
    for lab, d, col in items:
        labels.append(lab.replace("-", "-\n") if len(lab) > 14 else lab)
        # simpler multi-line
        labels[-1] = lab
        means.append(float(d["mean"]))
        stds.append(float(d["std"]))
        colors.append(col)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(labels))
    ax.bar(
        x, means, yerr=stds, capsize=4, width=0.68,
        color=colors, edgecolor="black", linewidth=0.5,
        error_kw=dict(elinewidth=1.0, ecolor="#333333"),
    )
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.15, f"{m:.2f}", ha="center", fontsize=8.5)
    ns = means[2]
    ax.axhline(ns, color=COLORS["T-GCN-NoSpatial"], ls="--", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("Test RMSE (km/h)")
    ax.set_title("Los-loop PH1: matched 30-edge sparsity controls (five-seed mean ± SD)")
    ax.set_ylim(0, max(np.array(means) + np.array(stds)) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "sparsity_controls_los_ph1.pdf", bbox_inches="tight")
    fig.savefig(OUT / "sparsity_controls_los_ph1.png", bbox_inches="tight")
    plt.close(fig)
    print("Wrote sparsity_controls_los_ph1")


def fig_dissociation():
    rows = load_los_ph1()
    items = [
        ("physical", "T-GCN\n(Physical)"),
        ("gcn_physical", "GCN\n(Physical)"),
        ("gcn_multigsl", "GCN-MultiGSL\n(static union)"),
        ("multi_gsl", "T-GCN-MultiGSL\n(per-timestep)"),
        ("multi_gsl_mix", "T-GCN-MultiGSL-Mix\n(gated)"),
    ]
    labels, means, stds, colors = [], [], [], []
    for vid, lab in items:
        name = VID[vid]
        labels.append(lab)
        means.append(float(rows[vid]["rmse_mean"]))
        stds.append(float(rows[vid]["rmse_std"]))
        colors.append(COLORS[name])
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        width=0.68,
        color=colors,
        edgecolor="#222222",
        linewidth=0.6,
        error_kw=dict(elinewidth=1.0, ecolor="#333333"),
    )
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.12, f"{m:.2f}", ha="center", fontsize=9)
    ns = float(rows["no_spatial"]["rmse_mean"])
    ax.axhline(ns, color=COLORS["T-GCN-NoSpatial"], ls="--", lw=1.2)
    ax.text(len(labels) - 0.3, ns + 0.12, f"T-GCN-NoSpatial  {ns:.2f}",
            color=COLORS["T-GCN-NoSpatial"], fontsize=8, ha="right")
    ax.annotate(
        "",
        xy=(3, 5.15),
        xytext=(2, 9.5),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0,
                        connectionstyle="arc3,rad=-0.25"),
    )
    ax.text(
        2.5,
        7.4,
        "identical multi-lag\nartifacts (union 28\nof 30 lag slots)",
        ha="center",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Test RMSE (km/h)")
    ax.set_title("Los-loop PH1: consumption of the same multi-lag graphs")
    ax.set_ylim(0, 11.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "consumption_dissociation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "consumption_dissociation.png", bbox_inches="tight")
    plt.close(fig)
    print("Wrote consumption_dissociation")


def fig_perseed():
    methods = [
        ("no_spatial", "T-GCN-NoSpatial"),
        ("multi_gsl", "T-GCN-MultiGSL"),
        ("multi_gsl_weighted", "T-GCN-MultiGSL-Weighted"),
        ("multi_gsl_mix", "T-GCN-MultiGSL-Mix"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, ph in zip(axes, range(1, 5)):
        for i, (vid, name) in enumerate(methods):
            vals = []
            for seed in (42, 43, 44, 45, 46):
                p = (
                    ROOT
                    / "results"
                    / "stage40_canonical"
                    / "training"
                    / f"losloop_ph{ph}_seed{seed}_{vid}.json"
                )
                vals.append(float(json.load(open(p))["rmse"]))
            vals = np.array(vals)
            col = COLORS[name]
            ax.vlines(i, vals.min(), vals.max(), color=col, lw=1.2, alpha=0.7)
            ax.hlines(
                [vals.min(), vals.max()],
                i - 0.12,
                i + 0.12,
                color=col,
                lw=1.0,
            )
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.add_patch(
                plt.Rectangle(
                    (i - 0.18, q1),
                    0.36,
                    q3 - q1,
                    facecolor=col,
                    edgecolor="#222",
                    alpha=0.35,
                    linewidth=0.6,
                )
            )
            ax.hlines(med, i - 0.18, i + 0.18, color="#222", lw=1.0)
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=14,
                color=col,
                edgecolor="#222",
                linewidth=0.3,
                zorder=4,
            )
            ax.plot(i, vals.mean(), marker="D", ms=4, color="#F44336", zorder=5)
        ax.set_title(f"PH{ph}", fontsize=10)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(
            ["NoSpatial", "MultiGSL", "Weighted", "Mix"], rotation=30, ha="right", fontsize=8
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(4.0, 7.0)
    axes[0].set_ylabel("Test RMSE (km/h)")
    fig.suptitle("Los-loop per-seed RMSE (seeds 42–46); diamond = mean", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "perseed_rmse_los.pdf", bbox_inches="tight")
    fig.savefig(OUT / "perseed_rmse_los.png", bbox_inches="tight")
    plt.close(fig)
    print("Wrote perseed_rmse_los")


def fig_pred_vs_actual():
    """3 rows (nodes) × 1 column: Actual + NoSpatial + Mix."""
    base = CKPT
    # High-variance nodes from previous fig9: 149, 163, 12 (0-indexed)
    nodes = [149, 163, 12]
    methods = [
        ("los_ph1_seed42_nograph", "T-GCN-NoSpatial", COLORS["T-GCN-NoSpatial"]),
        ("los_ph1_seed42_gated_multi", "T-GCN-MultiGSL-Mix", COLORS["T-GCN-MultiGSL-Mix"]),
    ]
    # Load first method's y_true (same test set); shape (n, PH, N) with PH=1
    yt = np.load(base / methods[0][0] / "y_true.npy")
    preds = {
        name: np.load(base / folder / "y_pred.npy") for folder, name, _ in methods
    }
    if yt.ndim == 3 and yt.shape[1] == 1:
        yt = yt[:, 0, :]
    for k in list(preds):
        if preds[k].ndim == 3 and preds[k].shape[1] == 1:
            preds[k] = preds[k][:, 0, :]
    T = min(100, yt.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    t = np.arange(T)
    for ax, node in zip(axes, nodes):
        actual = yt[:T, node]
        ax.plot(t, actual, color="black", lw=1.4, label="Actual", zorder=3)
        for name, col in [(n, c) for _, n, c in methods]:
            ax.plot(t, preds[name][:T, node], color=col, lw=1.2, alpha=0.9, label=name)
        ax.set_ylabel(f"Node {node}\n(norm.)", fontsize=9)
        ax.legend(loc="upper right", fontsize=7.5, ncol=3, framealpha=0.9)
        ax.grid(True, ls=":", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Test time step")
    fig.suptitle(
        "Los-loop PH1, seed 42: predicted vs actual (normalized)\n"
        "three high-variance nodes; 100 consecutive test steps",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "pred_vs_actual_los_ph1.pdf", bbox_inches="tight")
    fig.savefig(OUT / "pred_vs_actual_los_ph1.png", bbox_inches="tight")
    plt.close(fig)
    print("Wrote pred_vs_actual_los_ph1")


def fig_graph_structure():
    # Reuse existing logic: physical vs multi-lag union
    import csv as _csv

    adj_path = ROOT / "data" / "los_adj.csv"
    phys = np.array(
        [list(map(float, r)) for r in _csv.reader(open(adj_path)) if r], dtype=float
    )
    np.fill_diagonal(phys, 0)
    phys_bin = (np.abs(phys) > 0).astype(float)
    lag_dir = ROOT / "results" / "stage26_validation"
    lags = []
    for k in (1, 2, 3):
        m = np.load(lag_dir / f"los_ph1_seed42_L3_lag_{k}.npy").astype(float)
        np.fill_diagonal(m, 0)
        lags.append(np.abs(m))
    union = (np.stack(lags).max(axis=0) > 0.1).astype(float)
    n = phys_bin.shape[0]
    deg_p = phys_bin.sum(1)
    deg_u = union.sum(1)
    fig = plt.figure(figsize=(9.5, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(phys_bin, cmap="Greys", interpolation="nearest")
    ax1.set_title(f"(a) Physical\n{int(phys_bin.sum())} off-diag edges", fontsize=9)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(union, cmap="Blues", interpolation="nearest")
    ax2.set_title(f"(b) Multi-lag union\n{int(union.sum())} edges", fontsize=9)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(np.arange(n), deg_p, s=8, color=COLORS["T-GCN"], label="Physical")
    ax3.scatter(np.arange(n), deg_u, s=8, color=COLORS["T-GCN-MultiGSL"], label="Union")
    ax3.set_xlabel("Sensor index", fontsize=8)
    ax3.set_ylabel("Degree", fontsize=8)
    ax3.set_title(
        f"(c) Degrees\nmean {deg_p.mean():.2f} vs {deg_u.mean():.2f}", fontsize=9
    )
    ax3.legend(fontsize=7, frameon=False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax4 = fig.add_subplot(gs[1, :])
    bins = np.arange(0, max(int(deg_p.max()), 1) + 2) - 0.5
    ax4.hist(deg_p, bins=bins, color=COLORS["T-GCN"], alpha=0.75,
             label=f"Physical (mean {deg_p.mean():.2f})", edgecolor="#333", lw=0.4)
    ax4.hist(deg_u, bins=bins, color=COLORS["T-GCN-MultiGSL"], alpha=0.75,
             label=f"Union (mean {deg_u.mean():.2f})", edgecolor="#333", lw=0.4)
    ax4.set_xlabel("Node degree")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=8, frameon=False)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "graph_structure_los.pdf", bbox_inches="tight")
    fig.savefig(OUT / "graph_structure_los.png", bbox_inches="tight")
    plt.close(fig)
    print("Wrote graph_structure_los")


def fig_lag_ablation():
    rows = list(csv.DictReader(open(ROOT / "results/stage26_validation/stage26_validation_C_losloop_ph1.csv")))
    dag = [r for r in rows if r["method"] != "NoGraph"]
    dag.sort(key=lambda r: float(r["rmse"]))
    nograph = next(r for r in rows if r["method"] == "NoGraph")
    labels, rmses = [], []
    for r in dag:
        labels.append(r.get("lags_used", r["method"]).replace("lag_", ""))
        rmses.append(float(r["rmse"]))
    labels.append("T-GCN-NoSpatial")
    rmses.append(float(nograph["rmse"]))
    n_lags = [lab.count("+") + 1 for lab in labels[:-1]] + [0]
    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(0, 3)
    colors = [COLORS["T-GCN-NoSpatial"] if nl == 0 else cmap(norm(nl)) for nl in n_lags]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, rmses, color=colors, edgecolor="black", linewidth=0.5, height=0.65)
    for yi, m in zip(y, rmses):
        ax.text(m + 0.02, yi, f"{m:.2f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("RMSE (km/h)")
    ax.set_title("Lag ablation — Los-loop PH1 (seed 42, supplementary)")
    ax.axvline(float(nograph["rmse"]), color=COLORS["T-GCN-NoSpatial"], ls="--", lw=1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "lag_ablation_los_ph1.pdf", bbox_inches="tight")
    fig.savefig(OUT / "lag_ablation_los_ph1.png", bbox_inches="tight")
    plt.close(fig)
    print("Wrote lag_ablation_los_ph1")


def fig_train_loss():
    """Training-loss: Physical vs winning Mix (fallback: MultiGSL if Physical missing)."""
    phys = CKPT / "los_ph1_seed42_physical" / "train_loss_history.json"
    mix = ("los_ph1_seed42_gated_multi", "T-GCN-MultiGSL-Mix", COLORS["T-GCN-MultiGSL-Mix"])
    if phys.exists():
        methods = [
            ("los_ph1_seed42_physical", "T-GCN (Physical)", COLORS["T-GCN"]),
            mix,
        ]
    else:
        print("WARN: Physical train_loss_history missing; using MultiGSL+Mix")
        methods = [
            ("los_ph1_seed42_multi_graph_fixed", "T-GCN-MultiGSL", COLORS["T-GCN-MultiGSL"]),
            mix,
        ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for folder, name, col in methods:
        hist = json.load(open(CKPT / folder / "train_loss_history.json"))
        losses = hist["train_losses"]
        ep = np.arange(1, len(losses) + 1)
        axes[0].plot(ep, losses, label=name, color=col, lw=1.6)
        axes[1].plot(ep, np.log10(np.maximum(losses, 1e-12)), label=name, color=col, lw=1.6)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("(a) Training loss (Los PH1, seed 42)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("log10(loss)")
    axes[1].set_title("(b) Log-scale")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "train_loss_curves_los_ph1.pdf", bbox_inches="tight")
    fig.savefig(OUT / "train_loss_curves_los_ph1.png", bbox_inches="tight")
    plt.close(fig)
    print("Wrote train_loss_curves_los_ph1")


if __name__ == "__main__":
    fig_sparsity_controls()
    fig_rmse_comparison()
    fig_dissociation()
    fig_perseed()
    fig_pred_vs_actual()
    fig_graph_structure()
    fig_lag_ablation()
    fig_train_loss()
