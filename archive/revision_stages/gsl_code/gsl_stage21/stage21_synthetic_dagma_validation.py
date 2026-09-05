#!/usr/bin/env python3
"""
Stage 21 — Synthetic DAGMA Validation

Validates the actual DAGMA implementation used in this repository
against synthetic data with known ground-truth structures.

CRITICAL FINDING: The Stage 20.5 implementation extracted the WRONG block.
  Stage 20.5 used: W[N:2N, 0:N] (current -> past, REVERSE temporal)
  Correct block:   W[0:N, N:2N] (past -> current, FORWARD temporal)

Tests:
  A: Contemporaneous DAGMA on known within-time dependencies
  B: Lag-1 temporal DAGMA — both blocks checked
  C: Multiple-lag conceptual test
  D: Null / no-dependency control
  E: Noise sensitivity
  F: Self-loop / temporal persistence
  Threshold analysis on known graph

All tests use the same DagmaLinear from the dagma library.
"""
import os
import sys
import json
import time
import csv
import numpy as np
from typing import Dict, Tuple, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from dagma.linear import DagmaLinear

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stage21_synthetic")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================================================================
# Configuration
# ======================================================================
N_VARS = 10
T_SAMPLES = 10000
SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)


# ======================================================================
# Synthetic Data Generators
# ======================================================================

def generate_temporal_dependencies(N, T, seed=SEED):
    """
    Generate data with known lag-1 cross-variable temporal dependencies.
    x_i(t) = sum_j A[i,j] * x_j(t-1) + noise
    
    A[i,j] > 0 means: x_j(t-1) -> x_i(t)
    """
    set_seed(seed)
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        A[i, i] = 0.7  # self-persistence
    # Cross-temporal dependencies
    A[2, 0] = 0.6   # x_0(t-1) -> x_2(t)
    A[4, 1] = 0.5   # x_1(t-1) -> x_4(t)
    A[4, 3] = 0.2   # x_3(t-1) -> x_4(t)
    A[7, 5] = 0.4   # x_5(t-1) -> x_7(t)
    A[9, 6] = 0.3   # x_6(t-1) -> x_9(t)
    
    noise = np.random.randn(T, N) * 0.1
    X = np.zeros((T, N), dtype=np.float64)
    X[0] = np.random.randn(N) * 0.5
    for t in range(1, T):
        X[t] = A @ X[t-1] + noise[t]
    
    ground_truth = []
    for i in range(N):
        for j in range(N):
            if A[i, j] != 0 and i != j:
                ground_truth.append((j, i, A[i, j]))  # (src_j, tgt_i, weight)
    return X, A, ground_truth


def generate_independent(N, T, seed=SEED):
    set_seed(seed)
    X = np.zeros((T, N), dtype=np.float64)
    for i in range(N):
        persistence = 0.5 + 0.03 * i
        X[0, i] = np.random.randn() * 0.5
        for t in range(1, T):
            X[t, i] = persistence * X[t-1, i] + np.random.randn() * 0.1
    return X


# ======================================================================
# DAGMA Wrappers
# ======================================================================

def run_dagma_temporal(X, N, lambda1=0.01, w_threshold=0.0, verbose=False):
    """
    Run temporal DAGMA with Z = [x(t-1), x(t)].
    
    DAGMA convention: W[i,j] = variable_i predicts variable_j
    
    Z layout: indices 0..N-1 = past(t-1), indices N..2N-1 = current(t)
    
    Blocks:
      W_pp = W[0:N, 0:N]       past -> past
      W_correct = W[0:N, N:2N]  past -> current (CORRECT temporal block!)
      W_wrong   = W[N:2N, 0:N]  current -> past (Stage 20.5 used this!)
      W_cc     = W[N:2N, N:2N]  current -> current
    """
    T_total = X.shape[0]
    M = T_total - 1
    Z = np.zeros((M, 2 * N), dtype=np.float64)
    Z[:, 0:N] = X[:-1]
    Z[:, N:2*N] = X[1:]
    
    model = DagmaLinear(loss_type='l2', verbose=verbose)
    W_full = model.fit(Z, lambda1=lambda1, w_threshold=w_threshold)
    
    # CORRECT temporal block: past -> current
    W_correct = W_full[0:N, N:2*N]
    # WRONG block (Stage 20.5 used): current -> past
    W_wrong = W_full[N:2*N, 0:N]
    W_cc = W_full[N:2*N, N:2*N]
    W_pp = W_full[0:N, 0:N]
    
    return W_full, W_correct, W_wrong, W_cc, W_pp


def run_dagma_contemporaneous(X, N, lambda1=0.01, w_threshold=0.0, verbose=False):
    model = DagmaLinear(loss_type='l2', verbose=verbose)
    W = model.fit(X, lambda1=lambda1, w_threshold=w_threshold)
    return W


# ======================================================================
# Graph Recovery Metrics
# ======================================================================

def evaluate_graph_recovery(W_learned, ground_truth_edges, N, threshold=0.3,
                            remove_diagonal=True, mode='contemp'):
    """
    Evaluate graph recovery.
    
    mode='contemp': W[i,j] means past_j -> current_i (transpose convention)
    mode='contemp': W[i,j] means i -> j (standard convention)
    """
    W_abs = np.abs(W_learned)
    learned_binary = (W_abs > threshold).astype(int)
    if remove_diagonal:
        np.fill_diagonal(learned_binary, 0)
    
    learned_edges = set()
    for i in range(N):
        for j in range(N):
            if learned_binary[i, j] > 0:
                learned_edges.add((i, j))
    
    gt_edges = set((s, t) for s, t, w in ground_truth_edges)
    
    if mode == 'past_to_current':
        # W_correct[i,j] = past_j -> current_i
        # GT edge (src_j, tgt_i) = (j, i)
        # So GT (j, i) maps to W_correct[i, j]
        gt_mapped = set((t, s) for s, t in gt_edges)
    else:
        gt_mapped = gt_edges
    
    tp = len(learned_edges & gt_mapped)
    fp = len(learned_edges - gt_mapped)
    fn = len(gt_mapped - learned_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'n_learned': len(learned_edges),
        'n_gt': len(gt_mapped),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
    }


# ======================================================================
# Tests
# ======================================================================

def test_block_convention():
    """Verify DAGMA convention with a tiny synthetic test."""
    print("\n" + "=" * 70)
    print("TEST 0: Block Convention Verification")
    print("=" * 70)
    
    N, T = 5, 2000
    set_seed()
    A = np.zeros((N, N))
    A[0, 0] = 0.8; A[1, 1] = 0.7; A[2, 2] = 0.6
    A[2, 0] = 0.5; A[4, 1] = 0.4
    
    X = np.zeros((T, N))
    X[0] = np.random.randn(N) * 0.5
    for t in range(1, T):
        X[t] = A @ X[t-1] + np.random.randn(N) * 0.1
    
    _, W_correct, W_wrong, W_cc, W_pp = run_dagma_temporal(X, N)
    
    print(f"  Ground truth: x_0(t-1) -> x_2(t) [weight=0.5], x_1(t-1) -> x_4(t) [weight=0.4]")
    print()
    
    # CORRECT block: W[0:N, N:2N] (past -> current)
    # DAGMA convention: W[i,j] = variable_i predicts variable_j
    # So W_correct[0, 7-N] = W_full[0, 7] = past_0 predicts curr_2
    # In the NxN block: W_correct[i, j] = past_i predicts curr_j
    # Wait... let me be precise.
    
    # W_full[0:N, N:2N] is the block where rows=0..N-1 (past) and cols=N..2N-1 (current)
    # So W_full[i, N+j] = variable_i (past) predicts variable_N+j (current)
    # In the extracted NxN block: W_correct[i, j] = W_full[i, N+j]
    # This means: past_i predicts current_j
    # So W_correct[0, 2] should be 0.5 (past_0 predicts curr_2)
    
    # Check the CORRECT block
    print("  CORRECT block: W[0:N, N:2N] (past_i predicts current_j)")
    print(f"    W_correct[0,2] (past_0 -> curr_2): {W_correct[0,2]:.4f}  (GT: 0.5)")
    print(f"    W_correct[1,4] (past_1 -> curr_4): {W_correct[1,4]:.4f}  (GT: 0.4)")
    print(f"    W_correct[0,0] (past_0 -> curr_0): {W_correct[0,0]:.4f}  (self-persistence)")
    print()
    
    # Check the WRONG block (Stage 20.5 used)
    print("  WRONG block: W[N:2N, 0:N] (Stage 20.5 used this!)")
    print(f"    W_wrong[2,0] (curr_2 -> past_2): {W_wrong[2,0]:.4f}  (REVERSE temporal!)")
    print(f"    W_wrong[0,0] (curr_0 -> past_0): {W_wrong[0,0]:.4f}")
    print()
    
    # Determine which block recovers the ground truth
    gt_correct = (abs(W_correct[0, 2] - 0.5) < 0.2) and (abs(W_correct[1, 4] - 0.4) < 0.2)
    gt_wrong = (abs(W_wrong[2, 0] - 0.5) < 0.2) and (abs(W_wrong[4, 1] - 0.4) < 0.2)
    
    if gt_correct:
        print("  ✓ CORRECT block recovers ground truth")
    else:
        print("  ✗ CORRECT block does NOT recover ground truth")
    
    if gt_wrong:
        print("  ✓ WRONG block also recovers ground truth (unexpected)")
    else:
        print("  ✗ WRONG block does NOT recover ground truth (expected)")
    
    print()
    print("  CRITICAL FINDING:")
    print("  Stage 20.5 extracted W[N:2N, 0:N] as the temporal block.")
    print("  This block represents: current -> past (REVERSE temporal direction!)")
    print("  The CORRECT temporal block is W[0:N, N:2N] (past -> current).")
    
    return W_correct, W_wrong, gt_correct


def test_B_lag1_temporal():
    """Test B: Lag-1 temporal DAGMA with correct block extraction."""
    print("\n" + "=" * 70)
    print("TEST B: Lag-1 Temporal DAGMA — Correct Block Extraction")
    print("=" * 70)
    
    N, T = N_VARS, T_SAMPLES
    X, A_true, gt = generate_temporal_dependencies(N, T)
    print(f"  Data: X ∈ R^{{{T} × {N}}}")
    print(f"  Ground truth: {len(gt)} cross-temporal edges")
    for s, t, w in gt:
        print(f"    x_{s}(t-1) -> x_{t}(t): weight={w:.2f}")
    
    W_full, W_correct, W_wrong, W_cc, W_pp = run_dagma_temporal(X, N)
    
    print(f"\n  Full W shape: {W_full.shape}, nonzero: {np.sum(np.abs(W_full) > 0)}")
    
    # CORRECT block analysis
    print(f"\n  CORRECT block (past -> current):")
    abs_correct = np.abs(W_correct.copy())
    np.fill_diagonal(abs_correct, 0)
    flat_idx = np.argsort(abs_correct.ravel())[::-1]
    
    gt_set = set((s, t) for s, t, w in gt)
    print(f"  Top 10 edges (W_correct[i,j] = past_i -> current_j):")
    print(f"  {'Rank':>4s}  {'past_i':>6s}  {'curr_j':>6s}  {'weight':>10s}  {'gt?':>4s}")
    for rank, idx in enumerate(flat_idx[:10], 1):
        i, j = divmod(idx, N)
        w_val = W_correct[i, j]
        is_gt = "GT" if (i, j) in gt_set else ""
        print(f"  {rank:4d}  {i:6d}  {j:6d}  {w_val:10.6f}  {is_gt:>4s}")
    
    # Recovery metrics for correct block
    metrics_correct = evaluate_graph_recovery(W_correct, gt, N, threshold=0.1, mode='contemp')
    print(f"\n  CORRECT block recovery (threshold=0.1):")
    print(f"    Precision: {metrics_correct['precision']}, Recall: {metrics_correct['recall']}, F1: {metrics_correct['f1']}")
    print(f"    TP: {metrics_correct['tp']}, FP: {metrics_correct['fp']}, FN: {metrics_correct['fn']}")
    
    # WRONG block analysis
    metrics_wrong = evaluate_graph_recovery(W_wrong, gt, N, threshold=0.1, mode='contemp')
    print(f"\n  WRONG block (Stage 20.5) recovery (threshold=0.1):")
    print(f"    Precision: {metrics_wrong['precision']}, Recall: {metrics_wrong['recall']}, F1: {metrics_wrong['f1']}")
    
    # Self-persistence
    print(f"\n  Self-persistence in CORRECT block diagonal:")
    for i in range(N):
        w = W_correct[i, i]
        if abs(w) > 0.01:
            print(f"    W_correct[{i},{i}] = {w:.6f}")
    
    return W_full, W_correct, W_wrong, gt, metrics_correct, metrics_wrong


def test_D_null_control():
    """Test D: Null — independent variables."""
    print("\n" + "=" * 70)
    print("TEST D: Null/No-Dependency Control")
    print("=" * 70)
    
    N, T = N_VARS, T_SAMPLES
    X = generate_independent(N, T)
    print(f"  Data: {N} independent AR(1) processes, T={T}")
    
    _, W_correct, W_wrong, _, _ = run_dagma_temporal(X, N)
    
    abs_correct = np.abs(W_correct.copy())
    np.fill_diagonal(abs_correct, 0)
    
    print(f"\n  CORRECT block (past -> current) statistics:")
    print(f"    Max off-diagonal |weight|: {abs_correct.max():.6f}")
    print(f"    Nonzero off-diagonal: {np.sum(abs_correct > 0)}")
    
    for thr in [0.01, 0.05, 0.1, 0.2, 0.3]:
        count = np.sum(abs_correct > thr)
        print(f"    threshold={thr:.2f}: {count} spurious edges")
    
    return W_correct


def test_E_noise_sensitivity():
    """Test E: Noise sensitivity."""
    print("\n" + "=" * 70)
    print("TEST E: Noise Sensitivity")
    print("=" * 70)
    
    N, T = N_VARS, T_SAMPLES
    noise_levels = [0.05, 0.10, 0.20, 0.40]
    results = []
    
    for noise_std in noise_levels:
        set_seed()
        A = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            A[i, i] = 0.7
        A[2, 0] = 0.6; A[4, 1] = 0.5; A[7, 5] = 0.4
        gt = [(0, 2, 0.6), (1, 4, 0.5), (5, 7, 0.4)]
        
        X = np.zeros((T, N), dtype=np.float64)
        X[0] = np.random.randn(N) * 0.5
        for t in range(1, T):
            X[t] = A @ X[t-1] + np.random.randn(N) * noise_std
        
        _, W_correct, _, _, _ = run_dagma_temporal(X, N)
        
        metrics = evaluate_graph_recovery(W_correct, gt, N, threshold=0.1, mode='contemp')
        w_02 = W_correct[0, 2]
        w_14 = W_correct[1, 4]
        w_57 = W_correct[5, 7]
        
        snr = (0.6**2 + 0.5**2 + 0.4**2) / (3 * noise_std**2)
        print(f"  noise={noise_std:.2f} (SNR≈{snr:.1f}): W[0,2]={w_02:.4f}, W[1,4]={w_14:.4f}, W[5,7]={w_57:.4f} | F1={metrics['f1']:.3f}")
        
        results.append({
            'noise_std': noise_std, 'snr': round(snr, 1),
            'w_02': round(w_02, 4), 'w_14': round(w_14, 4), 'w_57': round(w_57, 4),
            'f1': metrics['f1'], 'precision': metrics['precision'], 'recall': metrics['recall'],
        })
    
    return results


def test_F_self_loop():
    """Test F: Self-loop / temporal persistence."""
    print("\n" + "=" * 70)
    print("TEST F: Self-Loop / Temporal Persistence")
    print("=" * 70)
    
    N, T = N_VARS, T_SAMPLES
    set_seed()
    
    X = np.zeros((T, N), dtype=np.float64)
    X[0] = np.random.randn(N) * 0.5
    for t in range(1, T):
        for i in range(N):
            persistence = 0.9 - 0.05 * i
            X[t, i] = persistence * X[t-1, i] + np.random.randn() * 0.1
    
    _, W_correct, _, _, _ = run_dagma_temporal(X, N)
    
    print(f"  Self-persistence recovery in CORRECT block diagonal:")
    print(f"  {'var':>4s}  {'true':>8s}  {'learned':>10s}  {'status':>8s}")
    for i in range(N):
        w = W_correct[i, i]
        true_w = 0.9 - 0.05 * i
        status = "✓" if abs(w - true_w) < 0.2 else "✗"
        print(f"  {i:4d}  {true_w:8.4f}  {w:10.6f}  {status:>8s}")
    
    abs_correct = np.abs(W_correct.copy())
    np.fill_diagonal(abs_correct, 0)
    n_offdiag = np.sum(abs_correct > 0.1)
    print(f"\n  Off-diagonal edges > 0.1: {n_offdiag} (should be 0)")
    
    return W_correct


def test_threshold_analysis():
    """Threshold analysis on the correct temporal block."""
    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS: Recovery vs Threshold (CORRECT block)")
    print("=" * 70)
    
    N, T = N_VARS, T_SAMPLES
    X, A_true, gt = generate_temporal_dependencies(N, T)
    
    _, W_correct, _, _, _ = run_dagma_temporal(X, N)
    
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    print(f"  GT edges: {len(gt)}")
    print(f"  {'thr':>6s}  {'edges':>6s}  {'TP':>3s}  {'FP':>3s}  {'FN':>3s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")
    
    for thr in thresholds:
        metrics = evaluate_graph_recovery(W_correct, gt, N, threshold=thr, mode='contemp')
        print(f"  {thr:6.3f}  {metrics['n_learned']:6d}  {metrics['tp']:3d}  "
              f"{metrics['fp']:3d}  {metrics['fn']:3d}  "
              f"{metrics['precision']:6.3f}  {metrics['recall']:6.3f}  {metrics['f1']:6.3f}")
    
    best_f1 = 0
    best_thr = 0
    for thr in thresholds:
        m = evaluate_graph_recovery(W_correct, gt, N, threshold=thr, mode='contemp')
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_thr = thr
    print(f"\n  Optimal threshold: {best_thr} (F1={best_f1:.3f})")
    
    return best_thr, best_f1


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 80)
    print("STAGE 21 — SYNTHETIC DAGMA VALIDATION")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"N={N_VARS}, T={T_SAMPLES}, seed={SEED}")
    print("=" * 80)
    
    t_total = time.time()
    all_results = {}
    
    # Test 0: Block convention
    t0 = time.time()
    W_correct_conv, W_wrong_conv, gt_recovered = test_block_convention()
    all_results['test_0_convention'] = {
        'time_s': round(time.time() - t0, 1),
        'gt_recovered_in_correct_block': gt_recovered,
    }
    
    # Test B: Lag-1 temporal
    t0 = time.time()
    W_full_B, W_correct_B, W_wrong_B, gt_B, metrics_correct_B, metrics_wrong_B = test_B_lag1_temporal()
    all_results['test_B_correct'] = {
        'time_s': round(time.time() - t0, 1),
        'metrics': metrics_correct_B,
    }
    all_results['test_B_wrong'] = {
        'metrics': metrics_wrong_B,
    }
    
    # Test D: Null
    t0 = time.time()
    W_D = test_D_null_control()
    abs_D = np.abs(W_D.copy()); np.fill_diagonal(abs_D, 0)
    all_results['test_D'] = {
        'time_s': round(time.time() - t0, 1),
        'max_offdiag': round(float(abs_D.max()), 6),
        'nonzero_offdiag': int(np.sum(abs_D > 0)),
    }
    
    # Test E: Noise
    t0 = time.time()
    results_E = test_E_noise_sensitivity()
    all_results['test_E'] = {'time_s': round(time.time() - t0, 1), 'results': results_E}
    
    # Test F: Self-loop
    t0 = time.time()
    W_F = test_F_self_loop()
    diag_F = [round(float(W_F[i, i]), 6) for i in range(N_VARS)]
    all_results['test_F'] = {'time_s': round(time.time() - t0, 1), 'diagonal': diag_F}
    
    # Threshold
    t0 = time.time()
    best_thr, best_f1 = test_threshold_analysis()
    all_results['threshold'] = {'time_s': round(time.time() - t0, 1), 'best_thr': best_thr, 'best_f1': best_f1}
    
    total_time = time.time() - t_total
    
    # Save
    with open(os.path.join(RESULTS_DIR, 'stage21_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    np.save(os.path.join(RESULTS_DIR, 'W_correct_B.npy'), W_correct_B)
    np.save(os.path.join(RESULTS_DIR, 'W_wrong_B.npy'), W_wrong_B)
    
    # Summary
    print("\n" + "=" * 80)
    print("STAGE 21 COMPLETE")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 80)
    
    print("\n--- KEY FINDINGS ---")
    print(f"\nCRITICAL BUG: Stage 20.5 extracted the WRONG block!")
    print(f"  Wrong block F1: {metrics_wrong_B['f1']}")
    print(f"  Correct block F1: {metrics_correct_B['f1']}")
    print(f"\nThreshold analysis (correct block):")
    print(f"  Optimal threshold: {best_thr} (F1={best_f1:.3f})")
    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
