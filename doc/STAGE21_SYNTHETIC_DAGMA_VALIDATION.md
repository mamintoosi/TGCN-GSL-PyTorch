# Stage 21 — Synthetic DAGMA Validation Report

**Date:** 2026-09-03  
**Repository:** TGCN-GSL-PyTorch  
**Configuration:** N=10, T=10,000, seed=42, lambda1=0.01, 180K iterations

---

## 1. Executive Summary

Stage 21 validated the DAGMA implementation against synthetic data with known ground-truth structures.

### CRITICAL BUG FOUND

**Stage 20.5 extracted the WRONG block from the DAGMA weight matrix.**

| | Stage 20.5 Used | Correct Block |
|---|---|---|
| Block | W[N:2N, 0:N] | W[0:N, N:2N] |
| Meaning | current → past (REVERSE) | past → current (FORWARD) |
| Synthetic F1 | **0.0** | **0.75** |

DAGMA convention: W[i,j] = variable_i predicts variable_j.  
For Z = [x(t-1), x(t)], the temporal dependencies live in W[0:N, N:2N] (past predicts current), **not** W[N:2N, 0:N] (current predicts past).

### Other Key Findings

1. **DAGMA correctly recovers strong lag-1 dependencies** (Precision=1.0 at threshold 0.1).
2. **Weak dependencies (< 0.3) may be missed** — recall drops from 0.8 to 0.6 as threshold increases.
3. **Optimal threshold for graph recovery is ~0.001** (F1=0.889), much lower than the paper's 0.3.
4. **DAGMA does NOT recover self-persistence well** — only 1/10 variables recovered in the self-loop test.
5. **Independent variables produce zero spurious edges** — DAGMA is clean on null data.
6. **Low noise (SNR>100) paradoxically hurts recovery** — DAGMA needs some noise to avoid degenerate solutions.

---

## 2. Implementation Audit

### 2.1 DAGMA Convention

From the source code of `dagma.linear.DagmaLinear`:

```python
# Loss function:
dif = self.Id - W
rhs = self.cov @ dif
loss = 0.5 * np.trace(dif.T @ rhs)
```

This minimizes ||X(I - W)||², which is equivalent to X ≈ X @ W.  
**Convention: W[i,j] = variable_i predicts variable_j.**

### 2.2 Z Construction

```python
Z[:, 0:N] = X[:-1]    # past (t-1)
Z[:, N:2N] = X[1:]    # current (t)
```

### 2.3 Block Interpretation

For W_full ∈ R^{2N × 2N}:

| Block | Rows | Cols | Meaning |
|-------|------|------|---------|
| W[0:N, 0:N] | past | past | past → past |
| **W[0:N, N:2N]** | **past** | **current** | **past → current (CORRECT temporal)** |
| W[N:2N, 0:N] | current | past | current → past (REVERSE!) |
| W[N:2N, N:2N] | current | current | current → current |

### 2.4 The Bug

Stage 20.5 code in `phase_a_run_dagma.py`:
```python
W_cross = W_raw_temp[N:2*N, 0:N]       # past → current  ← WRONG COMMENT!
```

The comment says "past → current" but the actual block is "current → past".  
The correct extraction should be:
```python
W_cross = W_raw_temp[0:N, N:2*N]       # past → current  ← CORRECT
```

---

## 3. Synthetic Data

### Ground truth (Test B, N=10):

```
x_i(t) = 0.7 * x_i(t-1) + Σ_{j≠i} A[i,j] * x_j(t-1) + noise(0.1)
```

| Edge | Weight | Strength |
|------|--------|----------|
| x_0(t-1) → x_2(t) | 0.60 | Strong |
| x_1(t-1) → x_4(t) | 0.50 | Strong |
| x_5(t-1) → x_7(t) | 0.40 | Medium |
| x_6(t-1) → x_9(t) | 0.30 | Medium |
| x_3(t-1) → x_4(t) | 0.20 | Weak |

---

## 4. Results

### 4.1 Block Convention Verification (Test 0, N=5)

```
CORRECT block: W[0:N, N:2N]
  W_correct[0,2] = 0.5832  (GT: 0.5)  ← RECOVERED
  W_correct[1,4] = 0.0001  (GT: 0.4)  ← NOT recovered (too weak for N=5)

WRONG block: W[N:2N, 0:N] (Stage 20.5)
  W_wrong[2,0] = 0.0000    ← Empty (correctly — reverse direction)
```

### 4.2 Lag-1 Temporal DAGMA (Test B, N=10)

**CORRECT block top edges:**

| Rank | past_i | curr_j | Weight | GT? |
|------|--------|--------|--------|-----|
| 1 | 0 | 2 | 0.675 | ✓ |
| 2 | 1 | 4 | 0.471 | ✓ |
| 3 | 5 | 7 | 0.335 | ✓ |
| 4 | 6 | 9 | 0.074 | ✓ (weak) |
| 5 | 8 | 6 | 0.000 | |
| 6 | 3 | 4 | 0.000 | ✓ (missed) |

**Recovery at threshold=0.1:**

| Block | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|-----|
| CORRECT (past→current) | **1.0** | **0.6** | **0.75** | 3 | 0 | 2 |
| WRONG (Stage 20.5) | 0.0 | 0.0 | 0.0 | 0 | 0 | 5 |

### 4.3 Threshold Sensitivity (Correct Block)

| Threshold | Edges | TP | FP | FN | Precision | Recall | F1 |
|-----------|-------|----|----|-----|-----------|--------|-----|
| 0.001 | 4 | 4 | 0 | 1 | 1.000 | 0.800 | **0.889** |
| 0.01 | 4 | 4 | 0 | 1 | 1.000 | 0.800 | 0.889 |
| 0.05 | 4 | 4 | 0 | 1 | 1.000 | 0.800 | 0.889 |
| 0.1 | 3 | 3 | 0 | 2 | 1.000 | 0.600 | 0.750 |
| 0.3 | 3 | 3 | 0 | 2 | 1.000 | 0.600 | 0.750 |
| 0.5 | 1 | 1 | 0 | 4 | 1.000 | 0.200 | 0.333 |

**Key:** Precision is always 1.0 (no false positives). Recall decreases as threshold increases. Optimal F1 at threshold 0.001.

### 4.4 Null Control (Test D)

| Threshold | Spurious Edges |
|-----------|----------------|
| 0.01 | 0 |
| 0.05 | 0 |
| 0.1 | 0 |
| 0.2 | 0 |
| 0.3 | 0 |

**DAGMA produces zero spurious edges on independent data.** Max off-diagonal weight: 0.000065.

### 4.5 Noise Sensitivity (Test E)

| Noise | SNR | W[0,2] | W[1,4] | W[5,7] | F1 |
|-------|-----|--------|--------|--------|-----|
| 0.05 | 102.7 | 0.0001 | 0.0000 | 0.0001 | 0.000 |
| 0.10 | 25.7 | 0.6736 | 0.4787 | 0.3346 | **1.000** |
| 0.20 | 6.4 | 1.0480 | 0.8656 | 0.7044 | **1.000** |
| 0.40 | 1.6 | 1.1415 | 0.9624 | 0.7910 | **1.000** |

**Paradox:** Very low noise (SNR>100) makes recovery worse. The near-deterministic structure may cause DAGMA to find degenerate solutions. Moderate noise (SNR~25) is optimal.

### 4.6 Self-Loop Recovery (Test F)

| Variable | True | Learned | Status |
|----------|------|---------|--------|
| 0 | 0.90 | 0.707 | ✓ |
| 1 | 0.85 | 0.000 | ✗ |
| 2 | 0.80 | 0.000 | ✗ |
| 3 | 0.75 | 0.000 | ✗ |
| 4 | 0.70 | 0.205 | ✗ |
| 5 | 0.65 | 0.100 | ✗ |
| 6–9 | 0.60–0.45 | ~0 | ✗ |

**DAGMA struggles to recover self-persistence in the temporal block.** Only variable 0 (strongest: 0.9) is recovered. Off-diagonal spurious edges: 0.

---

## 5. Implications for Stage 20.5

### Was the temporal DAGMA implementation correct?

**NO — the block extraction was wrong.** Stage 20.5 extracted W[N:2N, 0:N] (current→past) instead of W[0:N, N:2N] (past→current). The wrong block has F1=0.0 on synthetic data.

### Was the block indexing correct?

**NO.** The code comment said "past → current" but the actual slice was "current → past".

### Was the edge direction interpretation correct?

**The interpretation was correct for the WRONG block.** The Stage 20.5 code correctly understood that W[i,j] means i→j, but applied it to the wrong block.

### Is the 0.2–0.3 threshold defensible?

**Partially.** On synthetic data:
- Threshold 0.3 still achieves Precision=1.0 (no false positives)
- But Recall drops to 0.6 (misses 2 of 5 edges)
- Optimal F1 is at threshold 0.001 (F1=0.889)
- The paper's 0.3 threshold is conservative but not unreasonable for avoiding false positives

### Is removing temporal self-loops justified?

**Ambiguous.** DAGMA does recover self-persistence (Test F shows variable 0 at 0.707), but:
- Only the strongest self-loop is recovered
- Self-loops are scientifically meaningful (temporal persistence)
- The current code removes them, which may discard useful information

### Does the current 2N formulation actually model temporal dependencies?

**YES — in the correct block.** The synthetic test confirms:
- W[0:N, N:2N] correctly captures past→current dependencies
- Strong edges (≥0.3) are recovered with Precision=1.0
- The formulation is scientifically valid when the correct block is used

---

## 6. Implications for Multi-Lag DAGMA

For N=156 sensors and L=12 lags:

```
Z(t) = [x(t-12), ..., x(t-1), x(t)] ∈ R^{(L+1)*N} = R^{2028}
W ∈ R^{2028 × 2028} ≈ 4.1M entries
```

Computational cost scales roughly as O(D² × T × iterations):
- Current (2N=312): ~180K iterations, ~5500s
- Multi-lag (2028): ~180K iterations, estimated ~230,000s (~64 hours)

**Not computationally feasible without significant optimization.**

---

## 7. Reproducibility

```bash
cd /data/git/mamintoosi/TGCN-GSL-PyTorch
/data/python-envs/pytorch/bin/python gsl_stage21/stage21_synthetic_dagma_validation.py \
    2>&1 | tee results/stage21_synthetic/stage21_log.txt
```

**Total runtime: ~45 seconds**

---

## 8. Files Generated

| File | Description |
|------|-------------|
| `results/stage21_synthetic/stage21_summary.json` | Full results |
| `results/stage21_synthetic/stage21_results.csv` | CSV results |
| `results/stage21_synthetic/W_correct_B.npy` | Correct temporal block |
| `results/stage21_synthetic/W_wrong_B.npy` | Wrong block (Stage 20.5) |
| `results/stage21_synthetic/stage21_log.txt` | Full log |

---

## 9. Recommendation for Stage 22

**Option B: Fix the temporal DAGMA implementation first.**

The block extraction bug must be fixed before any further real-data experiments. The corrected implementation should:
1. Extract W[0:N, N:2N] as the temporal block
2. Re-run Stage 20 experiments with the correct block
3. Report the corrected results

This is a critical prerequisite for scientific validity.
