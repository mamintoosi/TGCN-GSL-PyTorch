#!/usr/bin/env python3
"""Build compact demo assets for paper Figures 9–10 (pred vs actual).

Extracts only the three high-variance nodes × 100 test steps used in the
manuscript (PH1, seed 42) from results/stage26_checkpoint (if present).
Does not retrain models. Writes a small .npz under demo/assets/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results" / "stage26_checkpoint"
OUT = ROOT / "demo" / "assets"

# Manuscript settings (make_pred_vs_actual.py / Figures 9–10)
LOS_NODES = [149, 163, 12]
T_STEPS = 100


def load(path: Path) -> np.ndarray:
    a = np.load(path)
    if a.ndim == 3 and a.shape[1] == 1:
        a = a[:, 0, :]
    return a


def high_var_nodes(y: np.ndarray, k: int = 3) -> list[int]:
    v = y.var(axis=0)
    return list(np.argsort(v)[-k:][::-1])


def extract(dataset: str, seed42_dir: str) -> dict:
    phys = CKPT / f"{dataset}_ph1_seed42_physical"
    mix = CKPT / f"{dataset}_ph1_seed42_gated_multi"
    if not (phys / "y_true.npy").exists() or not (mix / "y_pred.npy").exists():
        raise FileNotFoundError(
            f"Missing checkpoint arrays under {phys} or {mix}. "
            "Run the physical/mix training first, or restore results/stage26_checkpoint."
        )
    yt = load(phys / "y_true.npy")
    yp_phys = load(phys / "y_pred.npy")
    yp_mix = load(mix / "y_pred.npy")
    nodes = LOS_NODES if dataset == "los" else high_var_nodes(yt, 3)
    T = min(T_STEPS, yt.shape[0])
    return {
        "y_true": yt[:T, nodes].astype(np.float32),
        "y_pred_physical": yp_phys[:T, nodes].astype(np.float32),
        "y_pred_mix": yp_mix[:T, nodes].astype(np.float32),
        "nodes": np.asarray(nodes, dtype=np.int32),
        "dataset": dataset,
        "ph": 1,
        "seed": 42,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in ("los", "sz"):
        data = extract(ds, "")
        path = OUT / f"pred_vs_actual_{ds}_ph1_seed42.npz"
        np.savez_compressed(path, **data)
        print(f"Wrote {path} ({path.stat().st_size} bytes) nodes={data['nodes'].tolist()}")


if __name__ == "__main__":
    main()
