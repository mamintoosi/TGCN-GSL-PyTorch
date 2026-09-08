# Stage 30 — Forensic Audit: Stage 27 vs Stage 29

**Date:** 2026-09-08
**Auditor:** Buffy (Codebuff Agent)
**Purpose:** Determine why Stage 27 and Stage 29 produce radically different results for Los-Loop at 15-minute resolution, and whether Stage 29 is valid.

---

## 1. Executive Summary

Stage 27 contains a **fundamental implementation error**: it tested the wrong model class. The "T-GCN-MultiGSL-Mix" in Stage 27 was actually **standard TGCN with a static union graph** — not the `GatedMultiGraphTGCN` model with per-node, per-timestep adaptive graph selection that constitutes the paper's proposed method. Stage 29 correctly implements the proposed method using the canonical pipeline from Stage 26.

Additionally, Stage 27's evaluation code saved **normalized** RMSE values (dividing by feat_max internally) while comparing them against **denormalized** values from Stage 26, creating an apples-to-oranges comparison in the comparison tables.

### VERDICT

- **Stage 27:** **INVALID** — tested wrong model class, reported mixed normalized/denormalized values
- **Stage 29:** **VALID** — uses canonical pipeline, correct model classes, multi-seed evaluation
- **Recommended Los-Loop-15min result for manuscript:** Stage 29 results (GatedMultiGraphTGCN mean RMSE = 6.24 ± 0.19 at PH=1, +27.44% over NoGraph)
- **Is Stage 29 suitable for manuscript use?** **YES**
- **Does 15-minute resolution explain SZ-Taxi's weak performance?** **NOT ESTABLISHED** — Los-Loop-15min shows strong improvement, contradicting the simple resolution hypothesis

---

## 2. Stage 27 Configuration

### 2.1 DAGMA Phase

| Parameter | Value |
|-----------|-------|
| Dataset | Los-Loop resampled to 15-min |
| Original shape | (2016, 207) |
| Resampled shape | (672, 207) |
| Train/test split | 80/20 → 537 train, 135 test |
| feat_max | 70.0 |
| N | 207 |
| L (lags) | 3 |
| Total variables | 828 |
| lambda1 | 0.01 |
| warm_iter | 30000 |
| max_iter | 60000 |
| seed | 42 |

### 2.2 Forecasting Phase (Critical Errors)

| Parameter | Stage 27 Value | Correct Value (Stage 26/29) |
|-----------|---------------|---------------------------|
| **Model class for MultiGSL-Mix** | **`TGCN` with union adjacency** | **`GatedMultiGraphTGCN` with 3 lag graphs** |
| Loss function | `nn.MSELoss()` (plain MSE) | `mse_with_regularizer_loss` (MSE + L2 reg, λ=1.5e-3) |
| Target shape | Y = single timestep (B, N, 1) | Y = pre_len timesteps (B, N, pre_len) |
| Sequence generation | `make_sequences`: range(len - seq_len - ph + 1) | `generate_sequences`: range(len - seq_len - pre_len) |
| Test samples | 135 (inflated) | 122 (correct) |
| Training samples | 525 (inflated) | 524 (correct) |
| Shuffling | `random.seed(seed)` inside epoch loop | DataLoader with shuffle=True |
| RMSE reporting | Normalized (×feat_max in comparison table only) | Denormalized (×feat_max in validation_epoch) |
| Seeds | 1 (seed=42 only) | 5 (seeds 42-46) |

---

## 3. Root Cause: Wrong Model Class (MOST CRITICAL)

### Stage 27 MultiGSL-Mix Implementation

From `stage26_resolution_experiment.py` lines ~195-210:

```python
# Stage 27 builds a SINGLE union graph from all lags
union_adj = np.zeros((N, N))
for A in adj_matrices:
    union_adj = np.clip(union_adj + A, 0, 1)
methods["T-GCN-MultiGSL-Mix"] = {"adj": union_adj}

# Stage 27 creates a STANDARD TGCN — not GatedMultiGraphTGCN
adj_tensor = torch.FloatTensor(method_cfg["adj"])
model = TGCN(adj=adj_tensor, hidden_dim=64)
projection = nn.Linear(64, 1)
```

**This is TGCN with a static adjacency matrix — there is no gating, no per-timestep graph selection, no adaptive weighting.** The name "MultiGSL-Mix" is misleading.

### Stage 29 MultiGSL-Mix Implementation

From `stage29_los15min.py` in `train_and_eval()`:

```python
# Stage 29 passes separate lag graphs to the correct model class
model = GatedMultiGraphTGCN(adj_list=adj_or_model_factory, hidden_dim=hidden_dim)
```

**This IS the paper's proposed method** — per-node, per-timestep adaptive graph selection via learned gating.

### Impact

The difference between a static union graph TGCN and an adaptive-gating GatedMultiGraphTGCN is the **entire point of the paper's contribution**. Stage 27 never tested the proposed method.

---

## 4. Secondary Issues in Stage 27

### 4.1 Loss Function Difference

| | Stage 27 | Stage 26/29 |
|---|---------|------------|
| Loss | `nn.MSELoss()` | `mse_with_regularizer_loss` |
| Formula | `mean((y-ŷ)²)` | `sum((y-ŷ)²)/2 + λ·sum(θ²)/2` |
| Regularization | weight_decay=0.0001 only | weight_decay=0.0001 + explicit L2 (λ=1.5e-3) |

The L2 regularization in `mse_with_regularizer_loss` adds parameter penalty directly to the loss, which provides stronger regularization than weight_decay alone. This could affect model convergence and generalization, especially for the multi-graph model with more parameters (17091 vs 12672).

### 4.2 Sequence Construction Difference

Stage 27 uses: `range(len(data) - seq_len - ph + 1)`
Stage 29 uses: `range(len(data) - seq_len - pre_len)`

For PH=1:
- Stage 27: `range(n - 13)` → n-12 samples
- Stage 29: `range(n - 13)` → n-13 samples

Stage 27 produces one extra sample (525 train, 135 test) vs Stage 29 (524 train, 122 test). This is a minor off-by-one that doesn't materially affect RMSE, but indicates the Stage 27 code is not aligned with the canonical pipeline.

### 4.3 RMSE Reporting Error in Comparison Table

Stage 27's comparison table output:
```
Method                      5-min RMSE    15-min RMSE     Change
-----------------------------------------------------------------
T-GCN-NoSpatial                 5.1432         0.1254     +97.6%
```

This compares **denormalized** 5-min RMSE (5.1432 from Stage 26) with **normalized** 15-min RMSE (0.1254 from Stage 27). The "97.6% improvement" is meaningless. The denormalized 15-min NoGraph RMSE is actually 0.1254 × 70 = 8.78.

---

## 5. DAGMA Verification

### 5.1 Matrix Integrity

| Check | Result |
|-------|--------|
| W_full shape | (828, 828) — correct |
| Stage 27 vs Stage 29 W_full | **Identical** (np.allclose = True) |
| All lag blocks identical | **Yes** — all 4 blocks verified |
| DAGMA parameters | seed=42, λ=0.01, L=3 — consistent |

**The DAGMA computation is the same in both stages.** The divergence is entirely in the forecasting phase.

### 5.2 Edge Counts (threshold=0.1, off-diagonal only)

| Block | Edges (cross-sensor) | max|w| |
|-------|---------------------:|--------:|
| lag_1 | 22 | 0.7820 |
| lag_2 | 9 | 0.7087 |
| lag_3 | 1 | 0.1152 |
| current | 105 | 0.7031 |
| **Total (lag blocks only)** | **32** | — |

Note: Stage 29 uses only lag_1, lag_2, lag_3 (32 edges) as input to GatedMultiGraphTGCN. The current block is extracted but not used in forecasting.

### 5.3 Stage 27 Report Edge Count Discrepancy

The Stage 27 report stated: `current=105, lag_1=75, lag_2=10, lag_3=1`

My independent verification from .npy files: `lag_1=22, lag_2=9, lag_3=1, current=105`

The lag_1 discrepancy (22 vs 75) suggests the Stage 27 report may have used a different threshold or included the current block in the lag counts. The .npy files confirm 22/9/1 cross-sensor edges at threshold=0.1.

---

## 6. Stage 29 Verification

### 6.1 Pipeline Consistency with Stage 26

| Component | Stage 26 (canonical) | Stage 29 |
|-----------|---------------------|----------|
| Model classes | TGCN, GatedMultiGraphTGCN, MultiGraphTGCNFixed | **Identical** |
| Loss | `mse_with_regularizer_loss` | **Identical** |
| Training loop | `train_and_eval()` via `SupervisedForecastTask` | **Identical** |
| Seed handling | `set_seed()` | **Identical** |
| Sequence generation | `generate_sequences()` | **Identical** |
| Graph construction | `binary_graph()` threshold=0.1 | **Identical** |
| Optimizer | Adam, lr=0.001, wd=0.0001 | **Identical** |
| Batch size | 128 | **Identical** |
| Epochs | 50 | **Identical** |
| Hidden dim | 64 | **Identical** |
| Evaluation | `SupervisedForecastTask.validation_epoch` | **Identical** |

**Stage 29 uses the exact same pipeline as Stage 26.** The only difference is the dataset (resampled Los-Loop).

### 6.2 Results Summary (Stage 29, Los-Loop-15min, denormalized RMSE)

#### PH=1 (15 min ahead)

| Method | RMSE (mean ± std) | Improvement |
|--------|-------------------:|------------:|
| NoGraph | 8.6000 ± 0.2492 | — |
| MultiGraphTGCN_fixed | 7.2462 ± 0.2975 | +15.74% |
| **GatedMultiGraphTGCN** | **6.2402 ± 0.1873** | **+27.44%** |

#### Per-seed detail (PH=1)

| Seed | NoGraph | MultiGSL | MultiGSL-Mix | Improvement |
|-----:|--------:|---------:|-------------:|------------:|
| 42 | 8.3457 | 7.1031 | 6.2022 | +25.68% |
| 43 | 8.6211 | 7.0729 | 6.4048 | +25.71% |
| 44 | 9.0528 | 7.0764 | 6.2239 | +31.25% |
| 45 | 8.5805 | 7.8394 | 5.9199 | +31.01% |
| 46 | 8.3998 | 7.1392 | 6.4500 | +23.21% |

#### All horizons (GatedMultiGraphTGCN improvement over NoGraph)

| PH | Physical Horizon | NoGraph Mean | GatedMulti Mean | Improvement |
|---:|-----------------:|-------------:|----------------:|------------:|
| 1 | 15 min | 8.6000 | 6.2402 | +27.44% |
| 2 | 30 min | 9.3109 | 7.3220 | +21.36% |
| 3 | 45 min | 10.0475 | 8.0626 | +19.76% |
| 4 | 60 min | 10.4604 | 8.7770 | +16.09% |

### 6.3 Statistical Observations

- Improvement is consistent across all 5 seeds (range: 23.21% to 31.25% at PH=1)
- Improvement persists across all 4 prediction horizons
- Improvement decreases monotonically with longer horizons (27% at PH=1 → 16% at PH=4)
- NoGraph baseline variance: std=0.25 (3.0% of mean) — reasonably stable
- GatedMulti variance: std=0.19 (3.0% of mean) — comparable stability

---

## 7. Cross-Dataset Comparison (SECONDARY, not causal)

### PH=1, seed=42

| Dataset | NoGraph | GatedMulti | Improvement | Notes |
|---------|--------:|-----------:|------------:|-------|
| Los-Loop 5-min | 5.1432 | 4.4578 | +13.33% | Stage 26, canonical pipeline |
| Los-Loop-15min | 8.3457 | 6.2022 | +25.68% | Stage 29, canonical pipeline |
| SZ-Taxi 15-min | 4.1156 | 4.1076 | +0.19% | Stage 26, canonical pipeline |

**Observation:** Los-Loop-15min shows **stronger** improvement than Los-Loop 5-min, not weaker. This contradicts the Stage 27 hypothesis that 15-minute resolution causes reduced improvement.

**Important caveat:** These are cross-dataset comparisons, not controlled experiments. Differences may be caused by dataset-specific factors (sensor count, traffic dynamics, network topology, signal statistics) rather than temporal resolution alone.

---

## 8. Assessment of Claims

### Claim 1: "Temporal resolution is the primary confounding factor"

**NOT SUPPORTED by Stage 29.** Los-Loop-15min shows +27.44% improvement — stronger than the +13.33% at 5-min resolution. If temporal resolution were the primary factor, we would expect weaker performance at 15-min resolution.

### Claim 2: "SZ-Taxi matches Los-Loop at 15-min resolution"

**CONTRADICTED by Stage 29.** Los-Loop-15min shows strong improvement (+27.44%), while SZ-Taxi shows marginal improvement (+0.19%). They do NOT match at 15-minute resolution.

### Claim 3: "Learned edge structure is substantially changed at 15-min resolution"

**PARTIALLY SUPPORTED.** The edge structure does change (different DAGMA graphs), but the method still works well. The structural change does not prevent the method from being effective.

### Claim 4: "Effectiveness depends on temporal granularity"

**NOT ESTABLISHED as a general claim.** The Stage 27 evidence for this was invalid (wrong model class). Stage 29 shows the method works well at 15-min granularity on Los-Loop. The SZ-Taxi weak result requires a different explanation.

---

## 9. Alternative Explanations for SZ-Taxi's Weak Performance

Given that Los-Loop-15min shows strong improvement, the original hypothesis (temporal resolution causes weak SZ-Taxi results) is no longer viable. Alternative explanations include:

1. **Dataset-specific traffic dynamics** — SZ-Taxi may have fundamentally different temporal dependency patterns
2. **Sensor/network topology** — SZ-Taxi's 156 sensors may have different spatial correlation structure
3. **Signal-to-noise ratio** — SZ-Taxi data may be noisier or have different variability
4. **Scale differences** — different traffic speed ranges or distributions
5. **Network connectivity** — SZ-Taxi's physical graph may be less informative for prediction
6. **DAGMA optimization** — SZ-Taxi's DAGMA may produce less useful graphs (only 4 edges at threshold=0.1 vs 32 for Los-Loop-15min)
7. **Graph sparsity** — SZ-Taxi DAGMA discovers very few edges (4 cross-sensor vs 32 for Los-Loop-15min), suggesting weak temporal dependencies

---

## 10. Recommended Next Steps for the Manuscript

### Immediate

1. **Discard Stage 27's forecasting results entirely** (DAGMA results are valid)
2. **Use Stage 29 results** for Los-Loop-15min analysis
3. **Reframe the narrative**: The method works well on both 5-min and 15-min Los-Loop data. SZ-Taxi's weak result is likely due to dataset-specific factors, not temporal resolution.

### Suggested manuscript changes

- Add Los-Loop-15min as an additional experimental condition showing the method works across temporal resolutions
- Replace the "temporal resolution as confounding factor" narrative with a more nuanced discussion
- Discuss what makes SZ-Taxi different (fewer DAGMA edges, different traffic dynamics)

### No additional experiments needed

Stage 29 already provides a clean, well-controlled result using the canonical pipeline with 5 seeds and 4 prediction horizons. No further experiments are required.

---

## 11. Files Created/Modified

### Created
- `results/stage30_forensic_audit/STAGE30_FORENSIC_AUDIT.md` (this report)
- `gsl_stage26/stage29_los15min.py` (Stage 29 experiment script)
- `run_stage29_los15min.sh` (Stage 29 runner)
- `results/stage29_los15min/` (all Stage 29 artifacts)

### Modified (earlier stages)
- `run_resolution_experiment.sh` (PIPESTATUS fix only)

### Not modified
- `paper/*.tex` (manuscript unchanged)
- `gsl_stage26/stage26_resolution_experiment.py` (Stage 27 preserved as-is)
- `results/stage27_resolution/` (preserved as-is)

---

## 12. Reproducibility

### Stage 29 can be reproduced by:

```bash
cd /data/git/mamintoosi/TGCN-GSL-PyTorch && bash run_stage29_los15min.sh
```

- Phase 1 (DAGMA): instant (reuses Stage 27 results)
- Phase 2 (Forecasting): ~2-3 hours (60 training runs)
- Phase 3 (Analysis): ~1 minute
- Total: ~2-3 hours

### Stage 27 can be reproduced by:

```bash
cd /data/git/mamintoosi/TGCN-GSL-PyTorch && bash run_resolution_experiment.sh
```

But note: Stage 27's results should be considered invalid for the manuscript due to the wrong model class.

---

## 13. Bottom Line

**Stage 27 is invalid.** It never tested the paper's proposed method (`GatedMultiGraphTGCN`); it tested a standard TGCN with a static union graph. The reported -3.14% "degradation" is an artifact of testing the wrong model.

**Stage 29 is valid.** It uses the exact same canonical pipeline as Stage 26, correctly implements `GatedMultiGraphTGCN`, and reports +27.44% improvement at PH=1 across 5 seeds. This is a strong, reproducible result.

**The scientific narrative must change.** The hypothesis that "15-minute temporal resolution causes weak SZ-Taxi performance" is contradicted by Stage 29. Los-Loop-15min shows even stronger improvement than Los-Loop 5-min. The explanation for SZ-Taxi's weak performance lies elsewhere — likely in dataset-specific factors such as graph sparsity (SZ-Taxi DAGMA produces only 4 edges vs 32 for Los-Loop-15min).
