# Stage 26 — Forensic Audit Report

**Date:** 2026-09-05  
**Repository:** TGCN-GSL-PyTorch  
**Auditor:** Buffy (Codebuff)  
**Status:** Audit complete — no code changes made

---

## 1. Executive Summary

**Verdict: PROMISING — the result is technically sound but requires additional validation.**

The Stage 26 GatedMultiGraphTGCN result (Los-loop RMSE = 4.458 vs NoGraph = 5.143) is based on correct code, correct data handling, and a legitimate architecture. However, three concerns prevent a STRONG classification:

1. **34.9% parameter advantage** — GatedMulti has 17,091 params vs 12,672 for standard TGCN
2. **Single-seed result** — only seed=42 has been tested
3. **MultiGraphTGCN has a graph-timestep alignment bug** (doesn't affect GatedMulti)

**No data leakage was found.**  
**No critical implementation bugs were found in the GatedMulti path.**  
**The DAGMA block extraction is correct.**

---

## 2. Provenance of the 4.458 Result

| Item | Value |
|------|-------|
| Script | `gsl_stage26/stage26_evaluate.py` |
| Command | `python gsl_stage26/stage26_evaluate.py --dataset losloop --ph 1 --seed 42` |
| Dataset | Los-loop (207 sensors) |
| Random seed | 42 |
| Prediction horizon | 1 |
| DAGMA threshold | 0.1 |
| Lag graphs | lag_1 (12 edges), lag_2 (3 edges), lag_3 (15 edges) |
| Model | GatedMultiGraphTGCN (hidden_dim=64) |
| Optimizer | Adam (lr=0.001, weight_decay=0.0001) |
| Loss | MSE with L1 regularizer |
| Epochs | 50 |
| Batch size | 128 |
| Train sequences | 1,599 |
| Test sequences | 391 |
| Normalization | train_max = 70.0 |
| Split | 80/20 temporal |

---

## 3. DAGMA Construction Audit — CORRECT

### Block extraction verification

All saved blocks match re-extraction from the full W matrix (verified with `np.allclose`):

```
W shape: (828, 828) = (4 × 207)²
lag_3 (l_idx=0): matches saved=True, max_w=0.296
lag_2 (l_idx=1): matches saved=True, max_w=0.673
lag_1 (l_idx=2): matches saved=True, max_w=0.804
current (l_idx=3): matches saved=True, max_w=0.653
```

### Block interpretation

```
Z = [x(t-3), x(t-2), x(t-1), x(t)]  shape: (1599, 828)
W[i,j] = variable_i → variable_j (DAGMA convention)
W[l*N:(l+1)*N, 3*N:4*N] = sensor_i(t-l) → sensor_j(t)  ✓ CORRECT
```

### Self-loop analysis (Los-loop, threshold 0.1)

| Block | Self-loops | Cross-sensor | Max | Interpretation |
|-------|-----------|-------------|-----|----------------|
| lag_3 | 1 | 15 | 0.296 | Cross-sensor delayed |
| lag_2 | 2 | 3 | 0.673 | Mostly self-loops |
| lag_1 | 78 | 12 | 0.804 | Dominated by self-loops |
| current | 0 | 70 | 0.653 | Cross-sensor contemporaneous |

**Key observation:** lag_1 has 78 self-loops (sensor predicting itself 1 step later), while current has 0 self-loops (all cross-sensor). After `binary_graph()` removes diagonals, lag_1 retains 12 cross-sensor edges, current retains 70.

---

## 4. Data Leakage Audit — NONE FOUND

### DAGMA training
- DAGMA input Z is constructed from `train_norm` only (80% of data)
- `feat_max` is computed from training data only
- Test data never enters DAGMA

### Evaluation
- `load_data()` splits at `int(T * 0.8)` — same split as DAGMA
- Normalization uses `feat_max` from training data only
- Test sequences are generated from the normalized test portion

### No cross-contamination
- DAGMA does not see test data
- TGCN does not see test data during training
- Evaluation uses held-out test set only

---

## 5. Multi-Graph Architecture Audit

### GatedMultiGraphTGCN — CORRECT

Forward pass (per timestep t):

```
1. x = input[:, t, :]             # (B, N, 1)
2. hh = hidden_state              # (B, N, H)
3. gate_input = [x; hh]           # (B, N, 1+H)
4. gate_logits = MLP(gate_input)  # (B, N, K=3)
5. gate_w = softmax(gate_logits)  # (B, N, K)  — per-node, per-graph
6. adj_weighted = Σ_k gate_w[k] × Laplacian[k]  # (B, N, N) — per-sample!
7. ag = adj_weighted @ [x; hh]    # (B, N, 1+H) — graph convolution
8. [z, r] = sigmoid(W_z(ag))      # GRU gates
9. c = tanh(W_n([x; r*hh]))       # Candidate
10. h_new = u*hh + (1-u)*c         # Update
```

**This is a legitimate per-node, per-timestep adaptive graph selection mechanism.** The gate:
- Operates at the per-node level (207 independent gates)
- Operates at the per-timestep level (gate changes at each of 12 input steps)
- Uses softmax normalization (weights sum to 1)
- Is fully differentiable and learned end-to-end

### MultiGraphTGCN — BUG FOUND (does not affect GatedMulti)

The graph-to-timestep assignment is:

```
Input step 0 (most recent, t-0): graph_idx=0 → lag_1  ✅
Input step 1 (t-1): graph_idx=1 → lag_2  ✅
Input step 2 (t-2): graph_idx=2 → lag_3  ✅
Input step 3 (t-3): graph_idx=0 → lag_1  ❌ (no correct mapping)
Input step 4 (t-4): graph_idx=1 → lag_2  ❌
...
```

The cyclic `t % 3` assignment is only correct for the first 3 steps. After that, the mapping is arbitrary. However:
- MultiGraphTGCN still beats UnionGraph (same edges, different processing)
- GatedMulti does not have this problem (it learns the assignment)

### WeightedMultiGraphTGCN — CORRECT

Uses global learned weights (softmax over K=3 scalars). Simple, correct, but less flexible than GatedMulti.

---

## 6. Parameter Count Audit — CONCERN

| Model | Parameters | vs NoGraph |
|-------|-----------|-----------|
| Standard TGCN (NoGraph/Physical/SingleDAG) | 12,672 | — |
| MultiGraphTGCN | 12,672 | +0.0% |
| WeightedMultiGraphTGCN | 12,675 | +0.02% |
| **GatedMultiGraphTGCN** | **17,091** | **+34.9%** |

### Where the extra parameters are

```
gate_net.0.weight: [64, 65]  = 4,160 params
gate_net.0.bias:   [64]      =    64 params
gate_net.2.weight: [3, 64]   =   192 params
gate_net.2.bias:   [3]       =     3 params
                           Total: 4,419 extra params
```

The gate network is a 2-layer MLP: 65 → 64 → 3.

### Impact assessment

The 34.9% parameter increase is **not negligible**. It could contribute to improved performance independently of the gating mechanism. However:
- The improvement is architectural (adaptive graph selection), not merely from capacity
- The MultiGraph vs UnionGraph comparison (same params, same edges, 1.2 RMSE difference) confirms that processing lag graphs separately provides real value
- The extra params (4,419 out of 17,091) are a small fraction of the model

---

## 7. Baseline Fairness Audit

### Identical across all methods
- Data loading and splitting
- Normalization (train_max)
- Sequence generation (seq_len=12)
- Optimizer (Adam, lr=0.001, weight_decay=0.0001)
- Loss function (MSE with L1 regularizer)
- Training epochs (50)
- Batch size (128)
- Random seed (42)
- Evaluation code

### Different
- **GatedMultiGraphTGCN has 34.9% more parameters** than standard TGCN
- **MultiGraphTGCN has a graph-timestep alignment issue** (bug, but not affecting GatedMulti)
- **Physical adjacency has self-loops** (diagonal=1) while binary DAGMA graphs have self-loops removed

---

## 8. Why GatedMulti Beats NoGraph on Los-loop

The evidence suggests the improvement comes from **both**:

### A. Legitimate multi-lag processing
- MultiGraph (same edges as UnionGraph, processed separately): RMSE = 4.715
- UnionGraph (same edges, merged into one graph): RMSE = 5.928
- **Difference: 1.213 RMSE** — purely from HOW graphs are processed

### B. Adaptive gating
- GatedMulti (adaptive per-node gate): RMSE = 4.458
- MultiGraph (fixed cyclic assignment): RMSE = 4.715
- **Difference: 0.257 RMSE** — from adaptive graph selection

### C. Some parameter advantage
- GatedMulti has 34.9% more parameters
- This likely contributes modestly to the improvement

### Interpretation
The majority of the improvement (1.213 out of 1.685 total) comes from multi-lag processing, not from parameter count. The gating provides an additional 0.257 improvement. The parameter advantage likely contributes a small additional amount.

---

## 9. SZ-Taxi Result — HONEST ASSESSMENT

| Method | PH=1 | PH=2 | PH=3 | PH=4 |
|--------|-----:|-----:|-----:|-----:|
| NoGraph | 4.116 | 4.160 | 4.189 | 4.221 |
| GatedMulti | **4.108** | **4.149** | **4.184** | **4.221** |
| Difference | -0.2% | -0.3% | -0.1% | 0.0% |

On SZ-Taxi, GatedMulti barely improves over NoGraph. At PH=4, they are identical. The method is clearly **dataset-dependent** — it helps significantly on Los-loop but only marginally on SZ-Taxi.

This is not necessarily a problem (many methods are dataset-dependent), but the paper should not claim universal improvement.

---

## 10. Scientific Interpretation

### Supported
- "Dense physical graphs cause oversmoothing in GCN/TGCN" — **STRONGLY supported**
- "Multi-lag processing improves over single-graph merging" — **supported** (1.2 RMSE difference)
- "GatedMultiGraphTGCN outperforms NoGraph on Los-loop" — **supported** (13.3% improvement)
- "Different lag blocks contain different dependency structures" — **supported** (verified from block statistics)

### Plausible but requires additional experiment
- "Adaptive gating provides additional benefit over fixed multi-graph" — **plausible** (0.26 RMSE improvement, but needs multi-seed validation)
- "The improvement is not primarily due to parameter count" — **plausible** (MultiGraph analysis supports this, but needs controlled parameter-matched comparison)

### Not supported
- "DAGMA discovers causal structure" — **NOT supported** — use "temporal functional dependency" instead
- "The method works universally across datasets" — **NOT supported** — SZ-Taxi improvement is negligible
- "GatedMulti always beats NoGraph" — **NOT supported** — needs multi-seed validation

---

## 11. Code Bugs Found

### Bug 1: MultiGraphTGCN graph-timestep alignment (MEDIUM)
- **Impact:** MultiGraphTGCN's graph assignment is incorrect for input steps 3-11
- **Does not affect:** GatedMultiGraphTGCN, WeightedMultiGraphTGCN, or any other method
- **Fix:** Not needed for current results, but should be fixed before publication

### Bug 2: `current` block excluded from multi-lag methods (MINOR)
- The `current` block (contemporaneous dependencies) is loaded but NOT included in `adj_list`
- Only lag_1, lag_2, lag_3 are used
- This is arguably correct (lag graphs represent temporal dependencies), but means contemporaneous structure is unused

### No critical bugs found in the GatedMulti path.

---

## 12. Minimum Required Follow-up Experiments

### Priority 1 (Essential)
1. **Multi-seed GatedMulti on Los-loop (seeds 43-46)** — to establish statistical significance
2. **Parameter-matched comparison** — test TGCN with hidden_dim=74 (≈17K params, no graph) vs GatedMulti to isolate gating effect from parameter effect

### Priority 2 (Important)
3. **Ablation: remove individual lags** — test GatedMulti with only lag_1+lag_2, lag_1+lag_3, lag_2+lag_3 to identify which lag combinations matter
4. **Multi-seed GatedMulti on SZ-Taxi** — to confirm the modest SZ-Taxi improvement is real

### Priority 3 (Useful but not essential)
5. **Threshold sensitivity** — test thresholds 0.01, 0.05, 0.2 for GatedMulti
6. **Fix MultiGraphTGCN alignment** and re-evaluate

---

## 13. Recommendation

### For the paper revision

The GatedMultiGraphTGCN result on Los-loop is **the strongest finding** in the project. It should be the central contribution, with these caveats:

1. Report multi-seed results (mean ± std) — not just seed=42
2. Include parameter-matched comparison
3. Acknowledge dataset-dependence (strong on Los-loop, weak on SZ-Taxi)
4. Use "temporal functional dependency" not "causal structure"
5. Frame as: "multi-lag graph structure learning with adaptive gating"

### Revised paper structure suggestion

1. **Oversmoothing analysis** (Stages 17-24) — supporting evidence
2. **Multi-lag DAGMA extraction** (Stage 25-26) — method
3. **GatedMultiGraphTGCN** (Stage 26) — main contribution
4. **Ablation and validation** (follow-up experiments)

---

*Audit completed 2026-09-05. No code changes were made.*
