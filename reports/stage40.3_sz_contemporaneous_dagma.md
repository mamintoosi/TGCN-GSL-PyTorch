# Stage 40.3 — SZ-Taxi Contemporaneous DAGMA Artifacts

**Date:** 2026-09-10
**Status:** READY TO RUN (code prepared, awaiting user execution on GPU system)
**Auditor:** Buffy (Codebuff)

---

## 1. Exact Code Path for DAGMA Fitting

**Script:** `gsl_stage26/stage40_3_fit_sz_contemporaneous.py`

This script imports `learn_gsl_graph()` from `gsl_stage26/stage33_gsl_canonical.py` — the exact same function used to generate the Los-loop contemporaneous DAGMA artifacts. No new DAGMA code was written.

**Code path:**
```
stage40_3_fit_sz_contemporaneous.py
  └── learn_gsl_graph("shenzhen", ph, seed=42, dagma_kwargs)
        ├── load_data("shenzhen")  →  train_norm, test_norm, adj, feat_max
        ├── X = train_norm[0::ph]  (per-PH subsampling)
        ├── DagmaLinear(loss_type="l2").fit(X, lambda1=0.01, w_threshold=0.3, ...)
        ├── A = (np.abs(W_est) > 0).astype(np.float32)
        ├── np.fill_diagonal(A, 0)
        └── return W_est, A, meta
```

**No model training is performed.** The script only fits DAGMA and saves graph artifacts.

---

## 2. Exact Preprocessing / Input Construction

| Step | Operation | Details |
|---|---|---|
| 1. Load data | `pd.read_csv("data/sz_speed.csv")` | (2976, 156) raw speed measurements |
| 2. Chronological split | `train = feat[:2380]`, `test = feat[2380:]` | 80% train, 20% test |
| 3. Normalize | `feat_max = max(train) = 86.4292` | Normalized by training maximum only |
| 4. Per-PH subsampling | `X = train_norm[0::PH]` | Contemporaneous snapshots at every PH-th row |

**PH-specific input shapes:**

| PH | X shape | Description |
|---|---|---|
| 1 | (2380, 156) | All training rows (no subsampling) |
| 2 | (1190, 156) | Every 2nd row |
| 3 | (794, 156) | Every 3rd row |
| 4 | (595, 156) | Every 4th row |

---

## 3. Exact Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| lambda1 | 0.01 | Original SZ-Taxi protocol (`DATASET_CONFIGS["shenzhen"]["lambda1"]`) |
| w_threshold | 0.3 | DAGMA library default, applied inside `fit()` |
| warm_iter | 30000 | Default |
| max_iter | 60000 | Default |
| loss_type | "l2" | DAGMALinear default |
| seed | 42 | Standard seed (DAGMA is deterministic: zero-init, no RNG) |
| verbose | False | Suppress DAGMA convergence output |

---

## 4. Output Files Generated

**After running:** `bash run_stage40_3_sz_dagma.sh`

| File | Description |
|---|---|
| `results/stage33_gsl_canonical/sz_gsl_ph1_seed42_W_est.npy` | Raw DAGMA weight matrix PH=1 |
| `results/stage33_gsl_canonical/sz_gsl_ph1_seed42_A_binary.npy` | Binary adjacency PH=1 |
| `results/stage33_gsl_canonical/sz_gsl_ph2_seed42_W_est.npy` | Raw DAGMA weight matrix PH=2 |
| `results/stage33_gsl_canonical/sz_gsl_ph2_seed42_A_binary.npy` | Binary adjacency PH=2 |
| `results/stage33_gsl_canonical/sz_gsl_ph3_seed42_W_est.npy` | Raw DAGMA weight matrix PH=3 |
| `results/stage33_gsl_canonical/sz_gsl_ph3_seed42_A_binary.npy` | Binary adjacency PH=3 |
| `results/stage33_gsl_canonical/sz_gsl_ph4_seed42_W_est.npy` | Raw DAGMA weight matrix PH=4 |
| `results/stage33_gsl_canonical/sz_gsl_ph4_seed42_A_binary.npy` | Binary adjacency PH=4 |
| `results/stage33_gsl_canonical/sz_gsl_ph{1-4}_seed42_metadata.json` | Per-PH provenance metadata |
| `results/stage33_gsl_canonical/sz_gsl_seed42_summary.json` | Summary of all PHs |

**Filename format is identical to Los-loop:** `{prefix}_gsl_ph{ph}_seed42_{W_est,A_binary}.npy`

---

## 5. Edge Counts and Densities (Expected)

Based on Los-loop reference (28 edges at 207 nodes), SZ-Taxi (156 nodes) is expected to produce a sparse graph. The exact edge counts will be determined after fitting.

**Los-loop reference (for calibration):**

| PH | Edges | Density | Positive | Negative |
|---|---|---|---|---|
| 1 | 28 | 0.000654 | 28 | 0 |
| 2 | 28 | 0.000654 | 28 | 0 |
| 3 | 28 | 0.000654 | 28 | 0 |
| 4 | 28 | 0.000654 | 28 | 0 |

**Note:** Los-loop graphs are identical across PHs (28 edges each). Whether SZ-Taxi shows the same pattern or PH-dependent variation will be determined after fitting.

---

## 6. Threshold Semantics

| Stage | Operation | Effect |
|---|---|---|
| DAGMA fit | `w_threshold=0.3` (internal) | Zeroes \|W\| < 0.3 inside `fit()` |
| Post-fit | `A = 1(|W| > 0)` | Retains all nonzero coefficients (both signs) |
| Self-loops | `np.fill_diagonal(A, 0)` | Removes diagonal entries |
| Support rule | Absolute magnitude | An edge exists iff \|W_ij\| >= 0.3 |

**Sign handling:** The Stage 36 canonical policy retains both positive and negative coefficients. On Los-loop, no negative coefficient reaches the threshold (max \|negative\| = 0.013), so the rule is numerically equivalent to positive-only. Whether SZ-Taxi shows negative survivors will be verified after fitting.

---

## 7. Loader Compatibility

**Verified pre-fitting:** The output filenames exactly match what `load_contemporaneous_graph()` in `gsl_stage40/scripts/stage40_run_all.py` expects:

```python
def load_contemporaneous_graph(dataset, ph, threshold=0.3):
    cfg = DATASET_CONFIGS[dataset]
    prefix = cfg["prefix"]  # "sz" for shenzhen
    gsl_dir = PROJECT_ROOT / "results" / "stage33_gsl_canonical"
    A_path = gsl_dir / f"{prefix}_gsl_ph{ph}_seed42_A_binary.npy"
    # ...
```

**Filename match verified:**
- PH=1: `sz_gsl_ph1_seed42_A_binary.npy` ✅
- PH=2: `sz_gsl_ph2_seed42_A_binary.npy` ✅
- PH=3: `sz_gsl_ph3_seed42_A_binary.npy` ✅
- PH=4: `sz_gsl_ph4_seed42_A_binary.npy` ✅

---

## 8. Counterpart Compatibility

After generation, the following counterpart relationships must hold:

| Pair | Same Graph? | Mechanism |
|---|---|---|
| GCN-GSL ↔ T-GCN-GSL | ✅ Yes | Both call `load_contemporaneous_graph("shenzhen", ph)` |
| GCN-cGSL ↔ T-GCN-cGSL | ✅ Yes | Both apply `(A + A.T) > 0` to same GSL graph |

**No code changes needed.** The counterpart compatibility is guaranteed by the shared `load_contemporaneous_graph()` function and the symmetric cGSL construction in the runner.

---

## 9. Runtime

**Estimated runtime per PH:** ~1.5-2.5 hours (based on Los-loop experience)
**Total estimated runtime:** ~6-10 hours for PH=1-4

The script runs PHs sequentially. Each PH is independent.

**Actual runtime will be reported after execution.**

---

## 10. Anomalies or Warnings

1. **SZ-Taxi multilag graphs are extremely sparse:** Only 2 edges total across 3 lags (at threshold 0.1). This is a property of the DAGMA fit with lambda1=0.01 on SZ data. The contemporaneous DAGMA uses a different construction (w_threshold=0.3) and may produce a denser graph.

2. **Lambda1 difference:** SZ-Taxi uses lambda1=0.01 while Los-loop uses lambda1=0.02. This is the original protocol value and must not be changed.

3. **DAGMA is deterministic:** The fitting is deterministic (zero-init, no RNG), so the same output will be produced on every run. The script checks for existing files and skips if already fitted.

---

## 11. Final Readiness for Stage 40

### Pre-fitting checklist:
- [x] DAGMA-only script created and importable
- [x] Bash launcher created
- [x] Validation script created
- [x] Output filenames match Stage 40 loader
- [x] Protocol matches Los-loop reference
- [x] Manifest updated with expected outputs

### Post-fitting checklist (run after execution):
- [ ] All 4 PH A_binary files exist
- [ ] All 4 PH W_est files exist
- [ ] All graphs have shape (156, 156)
- [ ] All graphs are binary (0/1)
- [ ] All graphs have no self-loops
- [ ] `load_contemporaneous_graph("shenzhen", ph)` works for all PHs
- [ ] GCN and TGCN can be instantiated with the graphs
- [ ] Validation script passes all checks

### How to run:

```bash
# Step 1: Fit DAGMA (estimated ~6-10 hours)
bash run_stage40_3_sz_dagma.sh

# Step 2: Validate (takes seconds)
conda run -n pth python run_stage40_3_validate.py

# Step 3: Run full Stage 40 (after validation passes)
conda run -n pth python gsl_stage40/scripts/stage40_run_all.py
```

---

## Files Created/Modified

| File | Action | Purpose |
|---|---|---|
| `gsl_stage26/stage40_3_fit_sz_contemporaneous.py` | Created | DAGMA-only fitting script |
| `run_stage40_3_sz_dagma.sh` | Created | Bash launcher |
| `run_stage40_3_validate.py` | Created | Post-fitting validation |
| `gsl_stage40/dagma/dagma_artifacts_manifest.json` | Updated | Added expected SZ file entries |

---

```
STAGE 40.3 VERDICT: READY (code prepared, awaiting user execution)
```
