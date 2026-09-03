# Stage 26 — Full-Sensor Multi-Lag DAGMA: Design, Implementation, and Experiment Plan

**Date:** 2026-09-03 22:00  
**Repository:** TGCN-GSL-PyTorch  
**Paper:** "Graph Structure Learning for Traffic Prediction" (Major Revision)

---

## 1. Executive Summary

Stage 26 implements the most scientifically rigorous test of the paper's core hypothesis:

> **Can DAGMA discover lag-specific functional dependencies between traffic sensors, and can those different graphs be exploited jointly by GCN/TGCN to improve traffic forecasting?**

Previous stages (17–25) showed that:
- Dense physical graphs strongly hurt GCN/TGCN performance (oversmoothing)
- Very sparse graphs perform much better
- Single-lag DAGMA produces sparse graphs that are better than physical but not consistently better than NoGraph
- Physical+DAGMA fusion did not help
- The 20-sensor multi-lag pilot showed different lag blocks have different edge structures

Stage 26 extends the multi-lag formulation to **all sensors** (SZ-Taxi: 156, Los-loop: 207) and introduces **novel multi-lag graph architectures** that explicitly process lag-specific dependency information.

---

## 2. Files Created/Modified

| File | Description |
|------|-------------|
| `gsl_stage26/stage26_run_dagma.py` | Full-sensor multi-lag DAGMA extraction |
| `gsl_stage26/stage26_evaluate.py` | Multi-lag forecasting with all models |
| `run_all_experiments.sh` | Updated with Stage 26A (DAGMA) + 26B (evaluation) |
| `doc/STAGE26_MULTI_LAG_DAGMA_20260903_220000.md` | This report |

---

## 3. Scientific Hypothesis

**Primary hypothesis:**

> A temporally structured set of sparse DAGMA graphs (one per lag) outperforms a single sparse graph because it preserves lag-specific predictive information that a single adjacency matrix loses.

**Null hypothesis:**

> Multi-lag DAGMA does not outperform NoGraph or single-lag DAGMA, suggesting that (a) the graph structure is not useful for forecasting, or (b) the current graph injection mechanism is inadequate.

**Important distinction:**

We do **not** claim causal discovery. We use the terms:
- lag-specific dependency
- temporal functional dependency
- predictive dependency
- lag-specific graph structure

---

## 4. Multi-Lag DAGMA Formulation

### 4.1 Input Construction

For L=3 lags and N sensors:

```
Z = [x(t-3), x(t-2), x(t-1), x(t)]
```

Shape: `(T-3, 4N)`

- SZ-Taxi: `4 × 156 = 624` variables → 624 × 624 DAGMA matrix
- Los-loop: `4 × 207 = 828` variables → 828 × 828 DAGMA matrix

### 4.2 Block Structure (DAGMA Convention: W[i,j] = variable_i → variable_j)

```
Z columns:
  [0:N]       = x(t-3)    Block 0
  [N:2N]      = x(t-2)    Block 1
  [2N:3N]     = x(t-1)    Block 2
  [3N:4N]     = x(t)      Block 3 (current)
```

### 4.3 Correct Lag Block Extraction

**Critical:** This was verified against the Stage 21 synthetic directional test.

For forecasting, we need: `A_l[i,j] = sensor_i(t-l) → sensor_j(t)`

This corresponds to:
```
W[l_idx*N : (l_idx+1)*N,  L*N : (L+1)*N]
```

Where `l_idx = L - l` (block 0 = most distant past, block L-1 = most recent past).

| Lag | Block Index | Rows | Columns | Meaning |
|-----|-------------|------|---------|---------|
| lag_3 | l_idx=0 | `[0:N]` | `[3N:4N]` | sensor_i(t-3) → sensor_j(t) |
| lag_2 | l_idx=1 | `[N:2N]` | `[3N:4N]` | sensor_i(t-2) → sensor_j(t) |
| lag_1 | l_idx=2 | `[2N:3N]` | `[3N:4N]` | sensor_i(t-1) → sensor_j(t) |
| current | l_idx=3 | `[3N:4N]` | `[3N:4N]` | contemporaneous |

**This is the correct block, verified by:**
1. Stage 21 synthetic directional test (F1=0.75 vs 0.0 for wrong block)
2. Block indexing assertions in the code
3. Tiny synthetic test with manual verification

### 4.4 DAGMA Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| lambda1 | 0.01 | Stage 24 settings |
| loss_type | l2 | Stage 24 settings |
| w_threshold | 0.0 | No thresholding during DAGMA |
| warm_iter | 30000 | Stage 24 settings |
| max_iter | 60000 | Stage 24 settings |
| seed | 42 | Reproducibility |

---

## 5. Forecasting Methods Implemented

### Baselines (A–C)

| Method | Description | Graph |
|--------|-------------|-------|
| **NoGraph (A)** | Identity adjacency (self-loops only) | I |
| **Physical (B)** | Physical road network | A_phys |
| **SingleDAGMA (C)** | Existing Stage 24 temporal DAGMA (2-lag, PH-specific) | A_DAGMA at various thresholds |
| **Corr-K8/K16/K32** | Correlation-based top-K graphs | A_corr |

### Multi-Lag Methods (D–I)

| Method | Description | Innovation |
|--------|-------------|------------|
| **UnionGraph (D)** | Union of lag-specific binary graphs | Aggregation into single adjacency |
| **IntersectGraph (D)** | Intersection of lag graphs | Edges present at ALL lags |
| **MultiGraphTGCN (E)** | Different adjacency per input timestep | Explicit lag-aware processing |
| **WeightedMultiGraphTGCN (F)** | `A = softmax(w) · [A_1, A_2, A_3]` | Learnable lag weights |
| **GatedMultiGraphTGCN (G)** | Per-node adaptive gate over lag graphs | Dynamic lag selection |
| **AggregatedDAG (I)** | Mean of absolute weights across lags, thresholded | Soft aggregation |
| **Per-lag standalone** | Each lag graph used alone | Ablation |

### Per-Lag Standalone

For each lag (lag_1, lag_2, lag_3), the binary graph is used as a standard TGCN adjacency.

---

## 6. Novel Model Architectures

### 6.1 MultiGraphTGCN

At each input timestep t, uses adjacency `A_{t % K}` where K = number of lag graphs.

```
t=0: apply A_lag1 (most recent)
t=1: apply A_lag2
t=2: apply A_lag3
t=3: apply A_lag1 (cyclic)
...
```

The GRU cell applies graph convolution with the selected Laplacian at each step.

**Key design:** The model processes the 12-step input sequence and can use different graphs at different timesteps, corresponding to different temporal dependencies.

### 6.2 WeightedMultiGraphTGCN

Learns a single weighted combination of lag-specific Laplacians:

```
L = softmax(w_1) · L_1 + softmax(w_2) · L_2 + softmax(w_3) · L_3
```

The weights `w_k` are learnable scalars. After training, we can inspect which lag dominates.

### 6.3 GatedMultiGraphTGCN

At each (batch, node) position, a gate network decides the blend of lag-specific graphs:

```
gate = softmax(GateNet([x, h]))  →  (K,) per node
adj_weighted[j,i] = sum_k gate_k * L_k[j,i]
```

This allows different nodes to use different lag graphs at different times.

---

## 7. Sparsity Controls

Every method reports:
- Number of nodes (N)
- Number of edges
- Graph density
- Threshold used

**Fair comparisons:**
- Corr-K8/K16/K32 provide sparsity-matched baselines
- Per-lag standalone provides single-graph comparisons
- UnionGraph provides aggregated-but-binary comparison
- WeightedMulti provides soft-aggregation comparison

**Threshold:** All binary graphs use |W| > 0.1 by default (also sweepable).

---

## 8. Experiment Matrix

### Phase A: DAGMA Extraction (Stage 26A)

| Dataset | PH | N | Variables | Matrix Size | Est. Time |
|---------|-----|---|-----------|-------------|-----------|
| SZ-Taxi | 1 | 156 | 624 | 624×624 | 60–120 min |
| SZ-Taxi | 2 | 156 | 624 | 624×624 | 60–120 min |
| SZ-Taxi | 3 | 156 | 624 | 624×624 | 60–120 min |
| SZ-Taxi | 4 | 156 | 624 | 624×624 | 60–120 min |
| Los-loop | 1 | 207 | 828 | 828×828 | 90–180 min |
| Los-loop | 2 | 207 | 828 | 828×828 | 90–180 min |
| Los-loop | 3 | 207 | 828 | 828×828 | 90–180 min |
| Los-loop | 4 | 207 | 828 | 828×828 | 90–180 min |

**Phase A total: ~8–18 hrs**

### Phase B: Evaluation (Stage 26B)

| Dataset | PH | Methods | Seed | Max Epochs | Est. Time |
|---------|-----|---------|------|------------|-----------|
| SZ-Taxi | 1,2,3,4 | 12+ methods | 42 | 50 | 15–30 min each |
| Los-loop | 1,2,3,4 | 12+ methods | 42 | 50 | 15–30 min each |

**Phase B total: ~2–4 hrs**

### Combined Total: ~10–22 hrs

---

## 9. Sanity Checks Performed

### 9.1 Code Compilation
- `py_compile` for both scripts: **PASSED**

### 9.2 Import Tests
- All new modules import correctly: **PASSED**

### 9.3 Block Extraction Verification
- `build_multilag_Z`: Z shape `(T-L, (L+1)*N)` verified: **PASSED**
- `extract_lag_blocks`: correct block extraction from dummy W verified: **PASSED**
- Block indexing manually verified against Stage 21 convention: **PASSED**

### 9.4 Model Shape Tests
- `MultiGraphTGCN`: `(B, T, N) → (B, N, H)` verified: **PASSED**
- `WeightedMultiGraphTGCN`: `(B, T, N) → (B, N, H)` verified: **PASSED**
- `GatedMultiGraphTGCN`: `(B, T, N) → (B, N, H)` verified: **PASSED**
- All backward passes: **PASSED**

### 9.5 Weight Initialization
- `WeightedMultiGraphTGCN`: initial weights = [0.333, 0.333, 0.333] (uniform): **PASSED**

---

## 10. Exact Commands

### Run Everything (via existing runner)
```bash
cd /data/git/mamintoosi/TGCN-GSL-PyTorch
./run_all_experiments.sh
```

### Run Individual DAGMA Extracts
```bash
# SZ-Taxi PH=1 (~60-120 min)
/data/python-envs/pytorch/bin/python gsl_stage26/stage26_run_dagma.py \
    --ph 1 --dataset shenzhen --lags 3 --seed 42 \
    2>&1 | tee results/stage26_validation/stage26A_sz_ph1_dagma_log.txt

# Los-loop PH=1 (~90-180 min)
/data/python-envs/pytorch/bin/python gsl_stage26/stage26_run_dagma.py \
    --ph 1 --dataset losloop --lags 3 --seed 42 \
    2>&1 | tee results/stage26_validation/stage26A_los_ph1_dagma_log.txt
```

### Run Individual Evaluations (after DAGMA)
```bash
# SZ-Taxi PH=1 (~15-30 min)
/data/python-envs/pytorch/bin/python gsl_stage26/stage26_evaluate.py \
    --dataset shenzhen --ph 1 --seed 42 --max-epochs 50 --n-lags 3 --threshold 0.1 \
    2>&1 | tee results/stage26_validation/stage26B_sz_ph1_eval_log.txt
```

### Check Progress
```bash
# Check if DAGMA is running
ps aux | grep stage26 | grep -v grep

# Check log progress
tail -5 results/stage26_validation/stage26A_sz_ph1_dagma_log.txt

# Check if evaluation is done
grep "STAGE 26 SUMMARY" results/stage26_validation/stage26B_sz_ph1_eval_log.txt
```

---

## 11. Methodological Limitations and Potential Confounders

### Known Limitations

1. **Single seed (42):** Multi-seed validation not yet done for Stage 26. Seed=42 only.
   - Previous stages showed 100% seed stability for single-lag DAGMA at threshold 0.1, but multi-lag stability is unknown.

2. **Fixed threshold (0.1):** All binary graphs use |W| > 0.1.
   - Threshold sensitivity not swept in Stage 26B (can be added with `--threshold`).

3. **DAGMA computed per PH:** Each PH has its own DAGMA matrix.
   - This is not exactly the same as learning a single multi-lag model.
   - PH-specific DAGMA captures PH-dependent dependencies.

4. **No temporal alignment guarantee:** The MultiGraphTGCN assigns lag_1 to the most recent input, lag_2 to the next, etc. This is a reasonable mapping but is not derived from the DAGMA output itself.

5. **Computational cost:** Full-sensor DAGMA is expensive (~624×624 or 828×828).

### Potential Confounders

1. **Sparsity:** Multi-lag methods may benefit from different sparsity levels. Default threshold=0.1 is used consistently.

2. **Graph size:** UnionGraph may have many more edges than single-lag methods, potentially causing oversmoothing.

3. **Model capacity:** MultiGraphTGCN, WeightedMultiGraphTGCN, and GatedMultiGraphTGCN have different parameter counts. This is acknowledged but not equalized.

4. **Temporal mapping:** The mapping of lag_1 → most recent input, lag_2 → second most recent, etc. is a design choice, not derived from the DAGMA output.

---

## 12. Expected Outcomes and Decision Criteria

### Positive outcomes (support paper narrative)
- MultiGraphTGCN or WeightedMultiGraphTGCN outperforms NoGraph → **lag-specific graphs provide predictive value**
- GatedMultiGraphTGCN learns distinct weights for different lags → **lag-dependent graph selection is useful**
- Per-lag graphs have different edge structures → **multi-lag is more informative than single-lag**

### Negative outcomes (challenge paper narrative)
- No multi-lag method beats NoGraph → **graph structure may not be useful for this task**
- All multi-lag methods ≈ single-lag → **multi-lag adds complexity without benefit**
- Physical graph remains worst → **oversmoothing is the dominant effect regardless of graph construction**

### Intermediate outcomes (partial support)
- Multi-lag > single-lag but not > NoGraph → **multi-lag helps, but graph injection is inadequate**
- WeightedMulti learns non-uniform weights → **different lags have different importance**
- Los-loop works but SZ-Taxi doesn't → **dataset-dependent, need further investigation**

---

## 13. Relationship to Previous Stages

| Stage | Key Finding | Stage 26 Response |
|-------|-------------|-------------------|
| 17–19 | DAGMA input is contemporaneous | Uses proper temporal multi-lag input |
| 20–21 | Block extraction bug (W[N:2N,0:N] wrong) | Correct block verified: W[0:N,L*N:(L+1)*N] |
| 22–23 | Corrected temporal DAGMA → 1 edge at PH=1 | Full-sensor may find more edges |
| 24 | Sparsification is dominant effect | Multi-lag explicitly tests lag-specific info |
| 25A | 100% seed stability at threshold 0.1 | Seed stability for multi-lag TBD |
| 25B–C | Physical+DAGMA fusion didn't help | Multi-lag uses different injection mechanism |
| 25D | 20-sensor pilot showed lag-specific blocks | Full-sensor confirms/refutes |
| 25D | Multi-lag DAGMA on 20 sensors: lag_3 strongest | Full-sensor validates at scale |

---

## 14. Next Steps After Stage 26

1. **Run Stage 26A** (DAGMA extraction): ~8–18 hrs
2. **Run Stage 26B** (evaluation): ~2–4 hrs
3. **Analyze results:**
   - Which method is best overall?
   - Does multi-lag beat single-lag?
   - Does any graph method beat NoGraph?
   - What weights does WeightedMulti learn?
4. **If positive:** Multi-seed validation, threshold sensitivity, Los-loop confirmation
5. **If negative:** Consider alternative graph injection mechanisms, or accept that graph structure learning may not improve forecasting for these datasets

---

*Report generated: 2026-09-03 22:00*
*Scripts validated: py_compile, import, shape tests, backward passes*
*Status: READY FOR USER EXECUTION*
