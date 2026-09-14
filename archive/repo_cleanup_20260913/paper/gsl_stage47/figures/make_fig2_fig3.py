"""Generate Results Figures 2 and 3 from Stage 40 artifacts.

Fig 2: per-seed RMSE strip/box, Los-loop, PH1-4, NoSpatial/MultiGSL/Weighted/Mix
Fig 3: physical vs multi-lag union structure (adjacency + degree)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "results" / "stage40_canonical" / "training"
FIGDIR = Path(__file__).resolve().parent
SEEDS = [42, 43, 44, 45, 46]
METHODS = {
    "no_spatial": "T-GCN-NoSpatial",
    "multi_gsl": "T-GCN-MultiGSL",
    "multi_gsl_weighted": "T-GCN-MultiGSL-Weighted",
    "multi_gsl_mix": "T-GCN-MultiGSL-Mix",
}
COLORS = {
    "no_spatial": "#1d3557",
    "multi_gsl": "#2a9d8f",
    "multi_gsl_weighted": "#8ab17d",
    "multi_gsl_mix": "#1f7a6f",
}


def load_rmse(dataset: str, method: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for ph in range(1, 5):
        vals = []
        for seed in SEEDS:
            p = TRAIN / f"{dataset}_ph{ph}_seed{seed}_{method}.json"
            if not p.exists():
                raise FileNotFoundError(p)
            with open(p) as f:
                vals.append(float(json.load(f)["rmse"]))
        out[ph] = vals
    return out


def fig2_perseed() -> None:
    data = {m: load_rmse("losloop", m) for m in METHODS}
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.8), dpi=200, sharey=True)
    rng = np.random.default_rng(0)
    for ax, ph in zip(axes, range(1, 5)):
        positions = np.arange(len(METHODS))
        for i, (key, label) in enumerate(METHODS.items()):
            vals = np.array(data[key][ph], dtype=float)
            # boxplot-style summary without full box machinery
            mean = vals.mean()
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            lo, hi = vals.min(), vals.max()
            ax.vlines(i, lo, hi, color=COLORS[key], lw=1.2, alpha=0.7)
            ax.hlines([lo, hi], i - 0.12, i + 0.12, color=COLORS[key], lw=1.0)
            ax.add_patch(
                plt.Rectangle(
                    (i - 0.18, q1),
                    0.36,
                    q3 - q1,
                    facecolor=COLORS[key],
                    edgecolor="#222222",
                    alpha=0.35,
                    linewidth=0.6,
                    zorder=2,
                )
            )
            ax.hlines(med, i - 0.18, i + 0.18, color="#222222", lw=1.0, zorder=3)
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=14,
                color=COLORS[key],
                edgecolor="#222222",
                linewidth=0.3,
                zorder=4,
            )
            ax.plot(i, mean, marker="D", ms=4, color="#c45c48", zorder=5)
        ax.set_title(f"PH{ph}", fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            ["NoSpatial", "MultiGSL", "Weighted", "Mix"], rotation=30, ha="right", fontsize=8
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(4.0, 7.0)
    axes[0].set_ylabel("Test RMSE (km/h)", fontsize=9)
    fig.suptitle(
        "Los-loop per-seed RMSE (seeds 42–46); diamond = mean  ·  n=5",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_perseed.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig2_perseed.png", bbox_inches="tight")
    print("Wrote fig2_perseed.{pdf,png}")
    # print means for audit
    for m in METHODS:
        for ph in range(1, 5):
            print(f"  {m} PH{ph}: {np.mean(data[m][ph]):.4f}")


def load_multilag_union() -> np.ndarray:
    p = ROOT / "results" / "stage26_validation" / "los_ph1_seed42_L3_lag_1.npy"
    if not p.exists():
        # fallback path pattern
        candidates = list((ROOT / "results").rglob("los_ph1_seed42_L3_lag_1.npy"))
        if not candidates:
            raise FileNotFoundError("multi-lag lag_1 npy not found")
        p = candidates[0]
    base = p.parent
    mats = []
    for k in (1, 2, 3):
        m = np.load(base / f"los_ph1_seed42_L3_lag_{k}.npy").astype(float)
        np.fill_diagonal(m, 0.0)
        mats.append(np.abs(m))
    # Stage 40 consumer threshold: |W| > 0.1 after diagonal removal
    union = (np.stack(mats).max(axis=0) > 0.1).astype(float)
    np.fill_diagonal(union, 0)
    return union


def load_physical() -> np.ndarray:
    import csv

    adj_path = ROOT / "data" / "los_adj.csv"
    rows = []
    with open(adj_path, newline="") as f:
        for row in csv.reader(f):
            if row:
                rows.append([float(x) for x in row])
    A = np.array(rows, dtype=float)
    return A


def fig3_graphstruct() -> None:
    phys = load_physical()
    union = load_multilag_union()
    n = phys.shape[0]
    # degrees ignoring self-loops for display stats
    P = phys.copy()
    np.fill_diagonal(P, 0)
    phys_bin = (np.abs(P) > 0).astype(float)
    deg_phys = phys_bin.sum(axis=1)
    deg_union = union.sum(axis=1)

    fig = plt.figure(figsize=(9.5, 5.2), dpi=200)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(phys_bin, cmap="Greys", interpolation="nearest")
    ax1.set_title(
        f"(a) Physical adjacency\n{int(phys_bin.sum())} off-diag edges (dense)",
        fontsize=9,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(union, cmap="Blues", interpolation="nearest")
    ax2.set_title(
        f"(b) Multi-lag union\n{int(union.sum())} off-diag edges",
        fontsize=9,
    )
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(np.arange(n), deg_phys, s=8, color="#9aa0a6", label="Physical")
    ax3.scatter(np.arange(n), deg_union, s=8, color="#1f7a6f", label="Multi-lag union")
    ax3.set_xlabel("Sensor index", fontsize=8)
    ax3.set_ylabel("Degree", fontsize=8)
    ax3.set_title(
        f"(c) Degrees\nmean {deg_phys.mean():.2f} vs {deg_union.mean():.2f}",
        fontsize=9,
    )
    ax3.legend(fontsize=7, frameon=False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    ax4 = fig.add_subplot(gs[1, :])
    bins = np.arange(0, max(deg_phys.max(), 1) + 2) - 0.5
    ax4.hist(
        deg_phys,
        bins=bins,
        color="#9aa0a6",
        alpha=0.75,
        label=f"Physical (mean {deg_phys.mean():.2f})",
        edgecolor="#333333",
        linewidth=0.4,
    )
    ax4.hist(
        deg_union,
        bins=bins,
        color="#2a9d8f",
        alpha=0.75,
        label=f"Multi-lag union (mean {deg_union.mean():.2f})",
        edgecolor="#333333",
        linewidth=0.4,
    )
    ax4.set_xlabel("Node degree", fontsize=9)
    ax4.set_ylabel("Count", fontsize=9)
    ax4.legend(fontsize=8, frameon=False)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.set_title(
        "Los-loop: physical proximity graph vs learned multi-lag statistical-dependency union "
        "(static estimates; not causal influence)",
        fontsize=10,
        pad=8,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_graphstruct.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig3_graphstruct.png", bbox_inches="tight")
    print("Wrote fig3_graphstruct.{pdf,png}")
    print(f"  physical off-diag edges: {int(phys_bin.sum())}")
    print(f"  union edges: {int(union.sum())}")
    print(f"  mean degree: {deg_phys.mean():.3f} vs {deg_union.mean():.3f}")


if __name__ == "__main__":
    fig2_perseed()
    fig3_graphstruct()
