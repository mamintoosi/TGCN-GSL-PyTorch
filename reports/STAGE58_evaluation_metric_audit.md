# Evaluation Metric Audit

**Stage:** 58  
**Date:** 2026-09-12  
**Scope:** static code inspection + light recomputation from stored artifacts. No training reruns. No file modifications in model/training code.

---

## 1. Executive conclusion

**PARTIALLY VERIFIED**

- **All 12 main Table 1–2 methods** (Stage 40 canonical matrix) use the **same** evaluation function, the same inverse scaling, the same full-batch RMSE/MAE formulas, and the same five-seed **mean ± sample std (ddof=1)** aggregation. Verified by recomputation from `results/stage40_canonical/training/*.json` against `gsl_stage41/stage41_summary.csv`.
- **Training loss is not the table metric.** GCN uses MSE and T-GCN uses `mse_with_regularizer` for **optimization only**; both are evaluated with the same RMSE/MAE path after training.
- **Inconsistency found (std convention only):** the **15-minute variant table** (`tab:los15`) uses **population std (ddof=0)**, matching Stage 29’s `np.std` default, while the main tables use **sample std (ddof=1)**. Means are consistent; parenthetical stds are on different conventions.
- Predictions are **not** stored, so an independent recompute of RMSE from raw $\hat{y}$ vs $y$ is **not possible** without a small re-forward pass (optional follow-up).

---

## 2. Evaluation pipeline

### 2.1 Stage 40 (main 12 methods — source of Tables 1–2)

```
generate_sequences(train_norm/test_norm, seq_len=12, pre_len=PH)
  X[i] = data[i : i+12]           # (12, N)
  Y[i] = data[i+12 : i+12+PH]     # (PH, N)

train (Adam) with backbone-specific LOSS (not used as table metric)

test DataLoader(batch_size=len(test_X), shuffle=False)   # FULL BATCH
→ SupervisedForecastTask.validation_epoch
    pred = model(x) * feat_max     # inverse scale
    y    = y * feat_max
    flatten to (B*PH, N)
    RMSE = sqrt(MSE(pred, y))      # torchmetrics mean over ALL elements
    MAE  = mean(|pred - y|)
→ JSON: rmse/mae rounded to 4 decimals
→ Stage 41: mean and std(ddof=1) over seeds 42–46
→ manuscript tables (rounded to 2 decimals)
```

**Code:**  
- Loader / full-batch: `gsl_stage40/scripts/stage40_run_all.py` ~261–267, 399–400  
- Metric: `tasks/supervised.py` `validation_epoch` ~100–107, 114–124  
- Aggregation: `gsl_stage41/scripts/stage41_audit.py` `mean_std` ~117–119 (`a.std(ddof=1)`)

### 2.2 Sparsity controls (RandTop30, CorrTop30)

Same `SupervisedForecastTask.validation_epoch` and full-batch test loader  
(`gsl_stage26/stage32_sparse_control.py` ~177, 244).  
Stage 41 recomputes mean/std with **ddof=1** (`stage41_audit.py` ~205–206).

### 2.3 Capacity control (h=74)

Single-seed Stage 26 validation B artifact; no multi-seed std.

### 2.4 15-minute variant

Same `validation_epoch` (`stage29_los15min.py` ~211, 385–410).  
Stage 29 console summary uses `np.std(rmses)` → **ddof=0**.  
Manuscript `tab:los15` stds match **ddof=0** (see §6).

---

## 3. Exact formulas

### 3.1 Per-seed evaluation (all Stage 40 methods)

From `tasks/supervised.py` (after `* feat_max`):

\[
\mathrm{RMSE}
= \sqrt{
  \mathrm{mean}_{b,i,h,n}\bigl((y-\hat y)^2\bigr)
}
\]

\[
\mathrm{MAE}
= \mathrm{mean}_{b,i,h,n}\bigl(|y-\hat y|\bigr)
\]

where the mean is over **all test windows** ($i$), **all PH steps in the target** ($h$), and **all sensors** ($n$), in **one full-batch pass**.

**Not used:**
- mean of per-batch RMSEs (only one batch in Stage 40);
- per-node then average of RMSEs;
- average of per-horizon RMSEs computed separately.

**Horizon pooling:** for a model trained with `pre_len=PH`, all `PH` target steps enter **one** MSE. This is **not** “RMSE of horizon PH alone.” It is the RMSE of the multi-step target for that PH-specific model. All methods share this rule.

**Inverse scaling:** yes — multiply by training-split `feat_max` before metrics (`supervised.py` 100–101). Values are in the original speed unit (km/h scale as stored).

**Mask / node drop / sample drop:** none found.

### 3.2 Loss vs metric (do not confuse)

| Role | GCN | T-GCN |
|------|-----|-------|
| **Training loss** | `F.mse_loss` (mean) | `sum((y-ŷ)²)/2 + 1.5e-3·∑θ²/2` |
| **Table RMSE/MAE** | `validation_epoch` (shared) | **same** `validation_epoch` |

Training loss differences do **not** change the evaluation formula.

### 3.3 Five-seed aggregation (main tables)

\[
\overline{\mathrm{RMSE}} = \frac{1}{5}\sum_{s=1}^{5} \mathrm{RMSE}_s,
\qquad
\mathrm{std} = \sqrt{\frac{1}{4}\sum_{s=1}^{5}(\mathrm{RMSE}_s-\overline{\mathrm{RMSE}})^2}
\]

i.e. **mean of per-seed RMSEs**, **sample std ddof=1** — **not** $\sqrt{\overline{\mathrm{MSE}}}$ and **not** pooling all seeds’ predictions.

`gsl_stage41/scripts/stage41_audit.py`:
```python
return float(a.mean()), float(a.std(ddof=1))
```

JSON stores `round(rmse, 4)`; Stage 41 means use those 4-decimal values; manuscript displays 2 decimals.

---

## 4. Method-by-method audit

All twelve Stage 40 methods share one evaluation path.

| Method | Evaluation function | RMSE formula | MAE formula | Inverse scaling | Aggregation | Same as others? | Evidence |
|--------|---------------------|--------------|-------------|-----------------|-------------|-----------------|----------|
| GCN (Physical) | `validation_epoch` | global sqrt-MSE | global MAE | × feat_max | mean±std ddof=1 over 5 seeds | **Yes** | stage40_run_all + stage41 |
| GCN-NoSpatial | same | same | same | same | same | **Yes** | same |
| GCN-GSL | same | same | same | same | same | **Yes** | same |
| GCN-cGSL | same | same | same | same | same | **Yes** | same |
| GCN-MultiGSL | same | same | same | same | same | **Yes** | same |
| T-GCN (Physical) | same | same | same | same | same | **Yes** | same |
| T-GCN-NoSpatial | same | same | same | same | same | **Yes** | same |
| T-GCN-GSL | same | same | same | same | same | **Yes** | same |
| T-GCN-cGSL | same | same | same | same | same | **Yes** | same |
| T-GCN-MultiGSL | same | same | same | same | same | **Yes** | same |
| T-GCN-MultiGSL-Weighted | same | same | same | same | same | **Yes** | same |
| T-GCN-MultiGSL-Mix | same | same | same | same | same | **Yes** | same |
| RandTop30 / CorrTop30 | `validation_epoch` (stage32) | same | same | same | mean±std ddof=1 (stage41) | **Yes** (formula) | stage32 + stage41 |
| Capacity h=74 | stage26 val B | same family | — | same | **single seed** | N/A (n=1) | stage26 JSON |
| 15-min (3 methods) | `validation_epoch` (stage29) | same | same | same | mean; **std ddof=0 in table** | **std NO** | stage29 JSON |

---

## 5. Table-generation audit

### Tables 1–2 (T-GCN / GCN families) and controls table

| Step | Artifact |
|------|----------|
| Raw per-seed | `results/stage40_canonical/training/{dataset}_ph{ph}_seed{seed}_{variant}.json` keys `rmse`, `mae` |
| Aggregation | `gsl_stage41/scripts/stage41_audit.py` → `stage41_summary.csv` / `.json` |
| Sparse | `stage32` JSON + stage41 sparse block (ddof=1) |
| Manuscript | `paper/revised_version/sections/results.tex` (manual 2-decimal transcription from Stage 41 / Stage 44 §8) |

**Improvement percentages** in text = relative reduction of **five-seed mean RMSEs** (`stage41_improvements.csv`), not a different metric.

### Table 4 (15-min)

| Step | Artifact |
|------|----------|
| Raw | `results/stage29_los15min/stage29_los15min_results.json` |
| Means | 5-seed mean of stored `rmse` |
| Std in manuscript | **ddof=0** (matches Stage 29 `np.std` / Stage 44 §8 text `8.600±0.249`) |

### Appendix MAE table

Built from `stage41_summary.csv` `mae_mean` / `mae_std` (ddof=1) — consistent with main RMSE aggregation.

---

## 6. Numerical verification

### 6.1 Stage 41 vs Stage 40 JSONs (recomputed)

Script: `reports/_stage58_check.py` (read-only).

| Cell | JSON mean | CSV mean | JSON std ddof=1 | CSV std | MAE match |
|------|-----------|----------|-----------------|---------|-----------|
| Los PH1 T-GCN-NoSpatial | 5.251440 | 5.251440 | 0.186250 | 0.186250 | OK |
| Los PH1 T-GCN | 7.877140 | 7.877140 | 0.283865 | 0.283865 | OK |
| Los PH1 Mix | 4.491440 | 4.491440 | 0.140346 | 0.140346 | OK |
| Los PH1 GCN-NoSpatial | 4.879620 | 4.879620 | 0.306566 | 0.306566 | OK |
| SZ PH1 Mix | 4.119000 | 4.119000 | 0.019637 | 0.019637 | OK |
| Los PH4 GCN | 8.764820 | 8.764820 | 0.285809 | 0.285809 | OK |

**Result: ALL_MATCH** for main-table source pipeline.

### 6.2 ddof illustration (Los PH1 NoSpatial)

| Stat | Value |
|------|-------|
| Per-seed RMSEs | 5.1695, 4.9701, 5.4025, 5.2961, 5.4190 |
| mean | 5.25144 |
| std ddof=0 | 0.16659 |
| std ddof=1 | **0.18625** ← Stage 41 / main tables |

### 6.3 15-min std convention

| Method | mean | std ddof=0 | std ddof=1 | Manuscript `tab:los15` |
|--------|------|------------|------------|-------------------------|
| NoSpatial | 8.6000 | **0.2492** | 0.2787 | **0.249** (ddof=0) |
| MultiGSL | 7.2462 | **0.2975** | 0.3327 | **0.298** (ddof=0) |
| Mix | 6.2402 | **0.1873** | 0.2094 | **0.187** (ddof=0) |

**Inconsistency:** main tables ddof=1; 15-min table ddof=0.

### 6.4 Independent RMSE from predictions

**Not performed.** Stage 40 JSONs do not store $\hat y$. Optional small check: one forward pass on one seed and compare to stored `rmse` (would confirm `validation_epoch` vs an independent `sqrt(mean((y-ŷ)²))`).

---

## 7. Problems and risks

| Issue | Severity | Detail |
|-------|----------|--------|
| **Std ddof mismatch: 15-min vs main** | **Medium (reporting)** | Same mean; different parentheses. Fix by recomputing 15-min stds with ddof=1 or stating ddof=0 in the caption. |
| RMSE pools all PH steps of a PH-step model | Low (definition) | Consistent across methods; not “horizon-3-only RMSE.” |
| JSON rounds to 4 decimals | Negligible | Affects aggregation at ~1e-4. |
| Stage 29 / Stage 32 **console** prints use `np.std` default ddof=0 | Low | Main tables do not use those prints; Stage 41 uses ddof=1 for sparse. |
| Capacity control n=1 | Low | Already labeled single-seed. |
| No saved predictions | Low for consistency | Blocks independent metric recompute without re-forward. |
| Training loss ≠ eval metric | N/A | Correctly separated; not a table error. |

---

## 8. Final verdict

1. **Are table RMSEs computed with one formula?**  
   **Yes for all 12 main methods and sparse controls:**  
   \(\mathrm{RMSE}=\sqrt{\mathrm{mean}((y-\hat y)^2)}\) on de-normalized full-batch test tensors, pooling windows × PH steps × sensors.  
   The **15-min table uses the same RMSE formula** but a **different std ddof**.

2. **Do all methods share one evaluation pipeline?**  
   **Yes** for Stage 40 / stage32 / stage29: all call `SupervisedForecastTask.validation_epoch` with full-batch test loaders and `feat_max` inverse scaling.

3. **Does training loss affect table RMSE formula?**  
   **No.** Loss only affects optimization. Evaluation is shared.

4. **Are the table numbers reliable for scientific comparison?**  
   **Yes for relative comparison of Stage 40 methods** (same formula, same protocol, recomputed means match artifacts).  
   **Caveat:** do not compare the 15-min **std** directly to main-table stds without converting ddof; **means are fine**.

5. **If anything remains unclear:**  
   - Recompute 15-min stds with `ddof=1` and update `tab:los15` (or caption).  
   - Optional: one-seed re-forward to independently verify `validation_epoch` RMSE against stored JSON (predictions not archived).

---

## Appendix — File map

| Role | Path |
|------|------|
| Eval function | `tasks/supervised.py` (`validation_epoch`) |
| Stage 40 runner / save | `gsl_stage40/scripts/stage40_run_all.py` |
| Stage 41 aggregation | `gsl_stage41/scripts/stage41_audit.py` |
| Stage 41 outputs | `gsl_stage41/stage41_summary.csv`, `.json` |
| Raw Stage 40 metrics | `results/stage40_canonical/training/*.json` |
| Sparse controls | `gsl_stage26/stage32_sparse_control.py` + results |
| 15-min | `gsl_stage26/stage29_los15min.py` + `results/stage29_los15min/` |
| Manuscript tables | `paper/revised_version/sections/results.tex` |
| This audit’s recompute | `reports/_stage58_check.py` |
