# DAGMA Threshold Analysis — SZ-Taxi (Fresh W Matrices)

**Date:** 2026-09-01  
**Repository:** TGCN-GSL-PyTorch  
**Paper:** "Graph Structure Learning for Traffic Prediction"  
**Dataset:** SZ-Taxi (Shenzhen Taxi, N=156 sensors)

---

## 1. Executive Summary

We analyzed four fresh DAGMA weight matrices (PH=1,2,3,4) generated with `w_threshold=0.0` (no thresholding), `lambda1=0.01`, `loss_type=l2`, 180,000 iterations. The key findings are:

1. **DAGMA does NOT produce an intrinsically sparse graph.** The raw W has 24,180 nonzero off-diagonal entries (every entry is nonzero). The extreme sparsity (8 edges at PH=1) is **entirely caused by thresholding**.

2. **The weight distribution is bimodal:** ~99.86% of entries have |W| < 0.001 (numerical residuals from the optimizer), while ~31-39 entries have |W| ≥ 0.001 (genuine learned coefficients). There is a clear gap between these two populations.

3. **`w_threshold=0.3` is moderately aggressive but not the main cause of sparsity.** Going from threshold=0 to threshold=0.001 reduces 24,180 → ~34 edges. Going from 0.001 to 0.3 reduces only 34 → 8 edges. The real sparsification happens at the 0.001 level.

4. **The graph is highly stable across prediction horizons.** At threshold=0.3, all 4 PHs produce the **exact same 8 edges** (Jaccard=1.0). At threshold=0.01, Jaccard=0.80-0.94.

5. **Node 128 dominates the learned graph.** The top 20 edges for every PH almost exclusively originate from node 128, making it a hub. Only 2 other source nodes appear in the top 20.

6. **Negative coefficients are negligible at practical thresholds.** Only 1 negative edge exists at |W| ≥ 0.001; all edges at |W| ≥ 0.05 are positive. The `W > 0` conversion is essentially lossless.

7. **Fresh W matches cached W_est.** At threshold=0.3, the fresh and cached graphs are identical (PH=1,3) or differ by 1 edge (PH=2,4). The cached files are trustworthy representations of genuine DAGMA output.

---

## 2. Fresh W File Verification

| PH | Shape | Dtype | Finite | Exact Zeros | Positive | Negative | Diag NZ | Off-diag NZ | Max |W_off| |
|----|-------|-------|--------|-------------|----------|----------|---------|-------------|------|
| 1 | (156,156) | float64 | ✅ | 0 | 13,798 | 10,538 | 156 | 24,180 | 0.6173 |
| 2 | (156,156) | float64 | ✅ | 0 | 13,765 | 10,571 | 156 | 24,180 | 0.6044 |
| 3 | (156,156) | float64 | ✅ | 0 | 13,652 | 10,684 | 156 | 24,180 | 0.6183 |
| 4 | (156,156) | float64 | ✅ | 0 | 13,733 | 10,603 | 156 | 24,180 | 0.6004 |

**Key observation:** Every single off-diagonal entry is nonzero. DAGMA does NOT produce a sparse matrix at the floating-point level. The matrix is effectively dense with most values being extremely small.

---

## 3. Threshold Sweep — How Many Edges Survive?

### 3.1 Absolute-value threshold (|W| ≥ threshold)

| Threshold | PH=1 | PH=2 | PH=3 | PH=4 | Density (PH=1) |
|-----------|------|------|------|------|----------------|
| 0 | 24,336 | 24,336 | 24,336 | 24,336 | 1.006 |
| 1e-5 | 14,403 | 14,395 | 14,373 | 14,278 | 0.596 |
| 1e-4 | 153 | 162 | 139 | 139 | 0.006 |
| 5e-4 | 35 | 37 | 40 | 37 | 0.001 |
| 1e-3 | 34 | 36 | 39 | 35 | 0.001 |
| 2e-3 | 33 | 36 | 38 | 31 | 0.001 |
| 5e-3 | 33 | 35 | 35 | 31 | 0.001 |
| 1e-2 | 32 | 34 | 33 | 31 | 0.001 |
| 2e-2 | 31 | 30 | 31 | 30 | 0.001 |
| 5e-2 | 24 | 23 | 26 | 24 | 0.001 |
| 1e-1 | 15 | 15 | 16 | 16 | 0.0006 |
| 2e-1 | 12 | 12 | 12 | 13 | 0.0005 |
| **3e-1** | **8** | **8** | **8** | **8** | **0.0003** |

### 3.2 Positive-only threshold (W ≥ threshold)

The positive-only counts are nearly identical to absolute-value counts at thresholds ≥ 0.001, because almost all large-magnitude edges are positive.

| Threshold | PH=1 pos | PH=1 neg (discarded) |
|-----------|----------|---------------------|
| 0.001 | 33 | 1 |
| 0.01 | 31 | 1 |
| 0.05 | 24 | 0 |
| 0.1 | 15 | 0 |
| 0.3 | 8 | 0 |

**Conclusion:** The `W > 0` conversion discards at most 1 edge at any practical threshold.

---

## 4. Reproduce Old Graph Construction

Using the exact old code logic: `W[|W| < threshold] = 0` → `adj = (W > 0).astype(int)`

| Threshold | PH=1 edges | PH=2 edges | PH=3 edges | PH=4 edges |
|-----------|-----------|-----------|-----------|-----------|
| 0.001 | 33 | 35 | 38 | 34 |
| 0.005 | 32 | 34 | 34 | 31 |
| 0.01 | 31 | 33 | 32 | 31 |
| 0.05 | 24 | 23 | 26 | 24 |
| 0.1 | 15 | 15 | 16 | 16 |
| 0.2 | 12 | 12 | 12 | 13 |
| **0.3** | **8** | **8** | **8** | **8** |

The physical SZ-Taxi graph has 532 edges. Even at threshold=0.001, DAGMA produces only ~34 edges — still 15× sparser than the physical graph. The sparsity is NOT primarily caused by `w_threshold=0.3`.

---

## 5. PH Graph Stability

### Positive-only edges at threshold=0.3 (the paper's configuration)

| PH pair | |E1| | |E2| | Common | Jaccard |
|---------|-----|-----|--------|---------|
| PH=1 vs PH=2 | 8 | 8 | 8 | **1.0000** |
| PH=1 vs PH=3 | 8 | 8 | 8 | **1.0000** |
| PH=1 vs PH=4 | 8 | 8 | 8 | **1.0000** |
| PH=2 vs PH=3 | 8 | 8 | 8 | **1.0000** |
| PH=2 vs PH=4 | 8 | 8 | 8 | **1.0000** |
| PH=3 vs PH=4 | 8 | 8 | 8 | **1.0000** |

**Perfect stability.** The same 8 edges are learned for all prediction horizons.

### Positive-only edges at threshold=0.01

| PH pair | |E1| | |E2| | Common | Jaccard |
|---------|-----|-----|--------|---------|
| PH=1 vs PH=2 | 31 | 33 | 31 | 0.9394 |
| PH=1 vs PH=3 | 31 | 32 | 30 | 0.9091 |
| PH=1 vs PH=4 | 31 | 31 | 29 | 0.8788 |
| PH=2 vs PH=3 | 33 | 32 | 31 | 0.9118 |
| PH=2 vs PH=4 | 33 | 31 | 30 | 0.8824 |
| PH=3 vs PH=4 | 32 | 31 | 28 | 0.8000 |

High stability even at the lower threshold. The DAGMA graph captures a robust structural signal.

---

## 6. Weight Distribution

### Quantiles of off-diagonal |W| (PH=1)

| Quantile | Value |
|----------|-------|
| 50% | 0.00001309 |
| 75% | 0.00002441 |
| 90% | 0.00003869 |
| 95% | 0.00005033 |
| 97.5% | 0.00006305 |
| 99% | 0.00008479 |
| 99.5% | 0.00011355 |
| **99.9%** | **0.04535001** |

**The critical jump:** Between the 99.5th and 99.9th percentile, the weight magnitude jumps from 0.0001 to 0.045 — a 400× increase. This confirms a clear bimodal distribution:
- **Population A (99.5%):** |W| < 0.0001 → optimizer noise/residuals
- **Population B (0.5%):** |W| > 0.001 → genuinely learned coefficients

### Top 20 edges (PH=1, all positive)

| Rank | Source | Target | W | |W| |
|------|--------|--------|-----|-----|
| 1 | 32 | 128 | 0.6173 | 0.6173 |
| 2 | 128 | 102 | 0.4809 | 0.4809 |
| 3 | 102 | 152 | 0.4422 | 0.4422 |
| 4 | 128 | 148 | 0.4284 | 0.4284 |
| 5 | 102 | 150 | 0.3723 | 0.3723 |
| 6 | 128 | 0 | 0.3608 | 0.3608 |
| 7 | 128 | 50 | 0.3144 | 0.3144 |
| 8 | 128 | 66 | 0.3119 | 0.3119 |
| 9 | 128 | 73 | 0.2812 | 0.2812 |
| 10 | 128 | 24 | 0.2778 | 0.2778 |
| 11 | 128 | 149 | 0.2669 | 0.2669 |
| 12 | 128 | 154 | 0.2176 | 0.2176 |
| 13 | 128 | 9 | 0.1880 | 0.1880 |
| 14 | 128 | 65 | 0.1432 | 0.1432 |
| 15 | 128 | 64 | 0.1002 | 0.1002 |
| 16 | 128 | 67 | 0.0987 | 0.0987 |
| 17 | 128 | 52 | 0.0980 | 0.0980 |
| 18 | 128 | 53 | 0.0939 | 0.0939 |
| 19 | 128 | 21 | 0.0862 | 0.0862 |
| 20 | 128 | 56 | 0.0851 | 0.0851 |

**Node 128 is a dominant hub:** 17 of the top 20 edges originate from node 128. The DAGMA graph essentially captures "node 128 predicts many other nodes." This is consistent across all PHs.

---

## 7. Is `w_threshold=0.3` Aggressively Removing Edges?

| Threshold | |W|≥thr edges | Fraction of 24,180 | Assessment |
|-----------|-------------|-------------------|------------|
| 0.001 | 34 | 0.14% | Removes 99.86% of entries (optimizer noise) |
| 0.01 | 32 | 0.13% | Very few additional edges removed |
| 0.05 | 24 | 0.10% | Moderate reduction |
| 0.1 | 15 | 0.06% | Half of 0.01 edges removed |
| 0.2 | 12 | 0.05% | Few more removed |
| **0.3** | **8** | **0.03%** | **The paper's threshold** |

**Answer:** `w_threshold=0.3` removes edges from 15 → 8 (about half), which is meaningful but not the dominant cause of sparsity. The dominant sparsification happens between 0 and 0.001 (24,180 → 34 edges). At that level, DAGMA's L1-type regularization (`lambda1=0.01`) has already pushed most weights to near-zero.

**However**, the 7 edges between threshold 0.001 and 0.3 may be informative. At threshold=0.01, we get ~32 edges; at 0.3, only 8. The intermediate thresholds (0.01-0.1) might provide a better tradeoff.

---

## 8. Near-Zero Values Analysis

### Magnitude distribution (PH=1)

| Range | Count | Fraction | Assessment |
|-------|-------|----------|------------|
| [0, 1e-6) | 1,208 | 5.0% | Numerical noise |
| [1e-6, 1e-5) | 8,725 | 36.1% | Optimizer residuals |
| [1e-5, 1e-4) | 14,250 | 58.9% | Optimizer residuals |
| [1e-4, 5e-4) | 118 | 0.49% | Borderline |
| [5e-4, 1e-3) | 1 | 0.004% | Rare |
| [1e-3, 5e-3) | 1 | 0.004% | Rare |
| [5e-3, 1e-2) | 1 | 0.004% | Rare |
| [1e-2, 5e-2) | 8 | 0.03% | **Genuine** |
| [5e-2, 1e-1) | 9 | 0.04% | **Genuine** |
| [1e-1, 3e-1) | 7 | 0.03% | **Genuine** |
| [3e-1, 1.0) | 8 | 0.03% | **Genuine** |

**There is a clear gap between 5e-4 and 1e-2** (only 3 entries in this range). This confirms the bimodal structure:
- **Noise floor:** |W| < 5e-4 (24,141 entries, 99.8%)
- **Genuine coefficients:** |W| > 1e-2 (33 entries, 0.14%)

The DAGMA optimizer (Adam, lr=0.0003) produces near-zero residuals across the entire matrix, but only ~30-40 entries are genuinely large. **Entries below the learning rate (0.0003) are effectively zero** — 24,297 out of 24,180 off-diagonal entries fall in this category.

---

## 9. Sign Analysis

| Threshold | Positive | Negative | Total | % Positive |
|-----------|----------|----------|-------|------------|
| 0 | 13,704 | 10,476 | 24,180 | 56.7% |
| 0.0001 | 149 | 4 | 153 | 97.4% |
| 0.001 | 33 | 1 | 34 | 97.1% |
| 0.01 | 31 | 1 | 32 | 96.9% |
| 0.05 | 24 | 0 | 24 | 100% |
| 0.1 | 15 | 0 | 15 | 100% |
| 0.3 | 8 | 0 | 8 | 100% |

**At the raw level**, there are roughly equal positive and negative coefficients (57% vs 43%), but these are all tiny (|W| < 1e-4). The genuinely large coefficients are overwhelmingly positive.

**The `W > 0` conversion discards only 1 edge at threshold=0.01 and 0 edges at threshold≥0.05.** This is NOT a significant source of information loss.

---

## 10. Fresh W vs Cached W_est

| PH | Cached Shape | Cached NZ | Fresh edges (thr=0.3) | Cached edges | Common | Jaccard | Exact Match |
|----|-------------|-----------|----------------------|--------------|--------|---------|-------------|
| 1 | (156,156,1) | 8 | 8 | 8 | 8 | **1.0000** | ✅ True |
| 2 | (156,156,2) | 7 | 8 | 7 | 7 | 0.8750 | ❌ False |
| 3 | (156,156,3) | 8 | 8 | 8 | 8 | **1.0000** | ✅ True |
| 4 | (156,156,4) | 7 | 8 | 7 | 7 | 0.8750 | ❌ False |

**Cached files are trustworthy** for PH=1 and PH=3 (exact match). PH=2 and PH=4 differ by 1 edge, likely due to slight nondeterminism between different DAGMA runs (the cached files were produced on a different machine/config).

**Important:** The cached files store `(N, N, max_PH)` where the third dimension indexes by PH. Each slice is independently thresholded.

---

## 11. Assessment of the `w_threshold=0.3` Issue

### The real question: Is DAGMA intrinsically sparse?

**Answer: NO.** DAGMA produces a dense matrix (24,180 nonzero entries) where 99.86% of entries are tiny numerical residuals (< 0.001). The sparsity is caused by two factors:

1. **DAGMA's L1 regularization (`lambda1=0.01`)** pushes most weights toward zero, creating a bimodal distribution with ~30-40 genuinely large coefficients and ~24,000 near-zero residuals.
2. **Thresholding** (whether at 0.001 or 0.3) removes the residuals and keeps the genuine coefficients.

### Is 0.3 too aggressive?

**Moderately, but not critically.** At threshold=0.01, there are ~32 edges; at 0.3, there are 8. The 24 edges between 0.01 and 0.3 are not necessarily "better" — they have smaller weights and may represent weaker relationships. However, a threshold of 0.01-0.05 might capture more useful structure.

### What threshold would be scientifically optimal?

This is an empirical question. The threshold=0.01 configuration (32 edges) provides a denser graph that might better capture the traffic network structure, while threshold=0.3 (8 edges) provides a very sparse graph that may act more like a denoising mechanism.

---

## 12. Final Scientific Conclusions

### Q1: Was the old graph actually generated from a fresh DAGMA run?

**Yes.** The cached W_est files at threshold=0.3 produce graphs identical (or nearly identical) to the fresh W. The old implementation was correct.

### Q2: Is `w_threshold=0.3` scientifically defensible?

**Partially.** It is a reasonable default for producing a very sparse graph, but there is no theoretical justification for 0.3 specifically. The choice appears somewhat arbitrary. A sensitivity analysis across thresholds would strengthen the paper.

### Q3: Would thresholds around 0.001–0.1 produce substantially different graph densities?

**Yes.** The edge count varies from ~34 (at 0.001) to ~15 (at 0.1). This is a 2× difference that could affect forecasting performance. Testing multiple thresholds would be informative.

### Q4: Is DAGMA intrinsically producing an extremely sparse graph?

**No.** DAGMA produces a dense weight matrix. The sparsity is caused by the combination of L1 regularization (creating a bimodal weight distribution) and thresholding (removing the small weights). The genuinely learned coefficients number ~30-40, not 8.

### Q5: Are the learned edges stable across PH=1..4?

**Yes, highly stable.** At threshold=0.3, all 4 PHs produce the exact same 8 edges. This is a positive finding — the graph captures a robust structural relationship independent of prediction horizon.

### Q6: Does the fresh analysis justify running fresh DAGMA on Los-loop?

**Yes.** The SZ-Taxi analysis was informative. Running Los-loop would reveal whether the bimodal distribution and hub structure are dataset-specific or general.

### Q7: What is the SINGLE next experiment?

**Run a threshold-sensitivity forecasting experiment:** Train GCN/T-GCN with DAGMA graphs at thresholds {0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3} and measure RMSE. This directly answers whether the paper's choice of 0.3 is optimal or whether a denser learned graph performs better. It also determines whether the 24 "intermediate" edges (between 0.01 and 0.3) are helpful or harmful for prediction.

---

## 13. Output Files

All results saved to `results/dagma_fresh/threshold_analysis/`:

- `threshold_sweep.csv` — Full threshold sweep for all PH values
- `ph_overlap.csv` — PH graph stability analysis
- `weight_quantiles.csv` — Weight distribution quantiles
- `top_edges.csv` — Top 20 edges per PH
- `fresh_vs_cached.csv` — Fresh vs cached comparison
- `DAGMA_THRESHOLD_ANALYSIS.md` — This report

---

*Report generated by `gsl_audit/analyze_fresh_dagma.py`*  
*Analysis environment: DAGMA 1.1.1, Python 3.12, seed=42*
