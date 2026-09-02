#!/usr/bin/env python3
"""
Stage 20.5 Phase A: Run 2 DAGMA experiments and save raw matrices.
  1. Temporal DAGMA with w_threshold=0.0 (save full raw W)
  2. Original (contemporaneous) DAGMA with w_threshold=0.0

w_threshold is PURELY POST-PROCESSING inside DagmaLinear:
  self.W_est[np.abs(self.W_est) < w_threshold] = 0
Therefore running with w_threshold=0.3 and again with w_threshold=0.0 is redundant.
Phase B will apply any desired threshold to the saved raw matrices.

Estimated time: ~10-12 min total (2 DAGMA runs on 312x312 and 156x156).
"""
import os, sys, time, json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from dagma.linear import DagmaLinear

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage20_5_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 80)
    print("STAGE 20.5 PHASE A: DAGMA RUNS (optimized — 2 runs only)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # --- Load and normalize data ---
    feat = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, "data/sz_speed.csv")), dtype=np.float32)
    adj = np.array(pd.read_csv(os.path.join(PROJECT_ROOT, "data/sz_adj.csv"), header=None), dtype=np.float32)
    T_full, N = feat.shape
    split = int(T_full * 0.8)
    train = feat[:split]
    feat_max = float(np.max(train))
    train_norm = train / feat_max

    seq_len = 12
    Xs = []
    for i in range(len(train_norm) - seq_len - 1):
        Xs.append(train_norm[i:i + seq_len])
    Xs = np.array(Xs)
    data = Xs[:, 0, :]  # (M, N) — contemporaneous snapshots

    # PH=1: no subsampling
    X_orig = data[::1]  # (M_orig, N)
    M_orig = X_orig.shape[0]
    print(f"N={N}, M_orig={M_orig}, feat_max={feat_max:.4f}")

    # --- Temporal input: Z = [x(t-1), x(t)] ---
    M = M_orig - 1
    Z = np.zeros((M, 2 * N), dtype=np.float32)
    Z[:, 0:N] = X_orig[:-1]
    Z[:, N:2 * N] = X_orig[1:]
    print(f"Temporal Z shape: {Z.shape}  (2N = {2 * N})")

    # --- Run 1: Temporal DAGMA, w_threshold=0.0 (raw) ---
    print("\n" + "=" * 60)
    print("RUN 1: Temporal DAGMA, w_threshold=0.0 (raw)")
    print("=" * 60)
    np.random.seed(42)
    t0 = time.time()
    m1 = DagmaLinear(loss_type='l2', verbose=True)
    W_raw_temp = m1.fit(Z, lambda1=0.01, w_threshold=0.0)
    dt1 = time.time() - t0

    # Save full raw temporal W
    np.save(os.path.join(RESULTS_DIR, "sz_ph1_W_raw_temporal.npy"), W_raw_temp)

    # Extract blocks from the raw W
    W_cross = W_raw_temp[N:2 * N, 0:N]       # past → current
    W_cc = W_raw_temp[N:2 * N, N:2 * N]      # contemporaneous
    W_past_past = W_raw_temp[0:N, 0:N]        # past → past
    W_past_curr = W_raw_temp[0:N, N:2 * N]    # current → past (should be ~0)

    np.save(os.path.join(RESULTS_DIR, "sz_ph1_W_cross_raw.npy"), W_cross)
    np.save(os.path.join(RESULTS_DIR, "sz_ph1_W_cc_raw.npy"), W_cc)
    np.save(os.path.join(RESULTS_DIR, "sz_ph1_W_past_past_raw.npy"), W_past_past)
    np.save(os.path.join(RESULTS_DIR, "sz_ph1_W_past_curr_raw.npy"), W_past_curr)

    print(f"  Time: {dt1:.1f}s")
    print(f"  Full W nonzero: {np.sum(np.abs(W_raw_temp) > 0)}")
    print(f"  W_cross nonzero: {np.sum(np.abs(W_cross) > 0)}")
    print(f"  W_cc nonzero: {np.sum(np.abs(W_cc) > 0)}")
    print(f"  W_past_past nonzero: {np.sum(np.abs(W_past_past) > 0)}")
    print(f"  W_past_curr nonzero: {np.sum(np.abs(W_past_curr) > 0)}")

    # Also save thresholded versions from the same raw W
    for thr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        W_thresh = W_raw_temp.copy()
        W_thresh[np.abs(W_thresh) < thr] = 0
        n = np.sum(np.abs(W_thresh) > 0)
        n_cross = np.sum(np.abs(W_cross) > thr)
        print(f"  Threshold {thr:.3f}: full nonzero={n}, cross edges={n_cross}")

    # --- Run 2: Original (contemporaneous) DAGMA, w_threshold=0.0 (raw) ---
    print("\n" + "=" * 60)
    print("RUN 2: Original DAGMA (contemporaneous), w_threshold=0.0")
    print("=" * 60)
    np.random.seed(42)
    t0 = time.time()
    m2 = DagmaLinear(loss_type='l2', verbose=True)
    W_orig = m2.fit(X_orig, lambda1=0.01, w_threshold=0.0)
    dt2 = time.time() - t0
    np.save(os.path.join(RESULTS_DIR, "sz_ph1_W_orig_contemp.npy"), W_orig)

    print(f"  Time: {dt2:.1f}s, nonzero: {np.sum(np.abs(W_orig) > 0)}")

    # --- Edge analysis on raw W_cross ---
    print("\n" + "=" * 60)
    print("TEMPORAL DAGMA EDGE ANALYSIS (from raw W_cross)")
    print("=" * 60)

    abs_cross = np.abs(W_cross)
    print(f"\nW_cross shape: {W_cross.shape}, nonzero: {np.sum(abs_cross > 0)}")
    nonzero_vals = W_cross[abs_cross > 0]
    if len(nonzero_vals) > 0:
        print(f"Weight range: [{W_cross.min():.6f}, {W_cross.max():.6f}]")
        print(f"|weight| stats: min={abs_cross[abs_cross > 0].min():.6f} "
              f"max={abs_cross[abs_cross > 0].max():.6f} "
              f"mean={abs_cross[abs_cross > 0].mean():.6f}")

    # Top 20 edges
    flat_idx = np.argsort(abs_cross.ravel())[::-1]
    print(f"\nTop 20 temporal edges (|W_cross[i,j]|):")
    print(f"  {'Rank':>4s}  {'past_i':>7s}  {'curr_j':>7s}  {'weight':>12s}  {'|weight|':>12s}")
    for rank, idx in enumerate(flat_idx[:20], 1):
        i, j = divmod(idx, N)
        if abs_cross[i, j] < 1e-10:
            break
        print(f"  {rank:4d}  sensor_{i:03d}  sensor_{j:03d}  {W_cross[i, j]:12.6f}  {abs_cross[i, j]:12.6f}")

    # Contemporaneous block analysis
    abs_cc = np.abs(W_cc)
    print(f"\nW_cc (contemporaneous) nonzero: {np.sum(abs_cc > 0)}")
    if np.sum(abs_cc > 0) > 0:
        flat_cc = np.argsort(abs_cc.ravel())[::-1]
        print("Top 5 contemporaneous edges:")
        count = 0
        for idx in flat_cc:
            i, j = divmod(idx, N)
            if abs_cc[i, j] > 0:
                count += 1
                print(f"  {count}: sensor_{i:03d}<->sensor_{j:03d}  w={W_cc[i, j]:.6f}")
                if count >= 5:
                    break

    # Save metadata
    meta = {
        "N": N, "D": 2 * N, "M_temporal": M, "M_orig": M_orig,
        "feat_max": float(feat_max), "seed": 42,
        "lambda1": 0.01, "loss_type": "l2",
        "time_temporal_s": round(dt1, 1),
        "time_original_s": round(dt2, 1),
        "W_raw_temporal_nonzero": int(np.sum(np.abs(W_raw_temp) > 0)),
        "W_cross_nonzero": int(np.sum(np.abs(W_cross) > 0)),
        "W_cc_nonzero": int(np.sum(np.abs(W_cc) > 0)),
        "W_past_past_nonzero": int(np.sum(np.abs(W_past_past) > 0)),
        "W_past_curr_nonzero": int(np.sum(np.abs(W_past_curr) > 0)),
        "W_orig_nonzero": int(np.sum(np.abs(W_orig) > 0)),
    }
    with open(os.path.join(RESULTS_DIR, "phase_a_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 80}")
    print("PHASE A COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"  sz_ph1_W_raw_temporal.npy     (full raw 2Nx2N temporal)")
    print(f"  sz_ph1_W_cross_raw.npy        (NxN past→current block)")
    print(f"  sz_ph1_W_cc_raw.npy           (NxN contemporaneous block)")
    print(f"  sz_ph1_W_past_past_raw.npy    (NxN past→past block)")
    print(f"  sz_ph1_W_past_curr_raw.npy    (NxN current→past block, should be ~0)")
    print(f"  sz_ph1_W_orig_contemp.npy     (raw original DAGMA)")
    print(f"  phase_a_metadata.json")
    print(f"Total DAGMA time: {dt1 + dt2:.1f}s ({(dt1 + dt2) / 60:.1f} min)")
    print(f"  Temporal (2N={2*N} vars): {dt1:.1f}s")
    print(f"  Original (N={N} vars): {dt2:.1f}s")


if __name__ == "__main__":
    main()
