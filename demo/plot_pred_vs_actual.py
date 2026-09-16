#!/usr/bin/env python3
"""Plot Figures 9–10 style overlays from compact demo assets (no training).

Usage (from repo root):
  python demo/plot_pred_vs_actual.py
  python demo/plot_pred_vs_actual.py --out demo/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "demo" / "assets"

COL_PHYS = "#FF8A80"
COL_MIX = "#7B1FA2"


def plot_one(npz_path: Path, out_dir: Path, title_prefix: str) -> None:
    z = np.load(npz_path)
    yt = z["y_true"]
    yp = z["y_pred_physical"]
    ym = z["y_pred_mix"]
    nodes = z["nodes"]
    T = yt.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    for i, (ax, node) in enumerate(zip(axes, nodes)):
        ax.plot(t, yt[:, i], color="black", lw=1.4, label="Actual", zorder=3)
        ax.plot(t, yp[:, i], color=COL_PHYS, lw=1.2, ls="--", alpha=0.95,
                label="T-GCN (Physical)")
        ax.plot(t, ym[:, i], color=COL_MIX, lw=1.2, alpha=0.95,
                label="T-GCN-MultiGSL-Mix")
        ax.set_ylabel(f"Node {int(node)}\n(norm.)", fontsize=9)
        ax.legend(loc="lower right", fontsize=7.5, ncol=3, framealpha=0.92)
        ax.grid(True, ls=":", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Test time step")
    fig.suptitle(
        f"{title_prefix} PH1, seed 42: predicted vs actual (normalized)\n"
        "three high-variance nodes; 100 consecutive test steps",
        fontsize=11,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=160)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_dir / f"{stem}.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "demo" / "figures")
    args = p.parse_args()
    mapping = {
        "pred_vs_actual_los_ph1_seed42.npz": "Los-loop",
        "pred_vs_actual_sz_ph1_seed42.npz": "SZ-Taxi",
    }
    for name, label in mapping.items():
        path = ASSETS / name
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run demo/build_pred_vs_actual_assets.py first")
        plot_one(path, args.out, label)


if __name__ == "__main__":
    main()
