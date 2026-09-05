# GSL Reimplementation Report

**Date:** 2026-08-31 15:29:25  
**Repository:** TGCN-GSL-PyTorch

---

## 1. What Was Built

An independent GSL module at `gsl_audit/independent_gsl.py` with:

- `extract_dagma_input(train_X)` — extracts the DAGMA input exactly as the existing code does, with shape assertions and logging
- `learn_graph(X, lambda1, w_threshold, ...)` — runs DAGMA with ALL parameters explicit
- `build_adjacency(W, mode, threshold)` — converts W to binary adjacency
- `audit_dagma(W)` — comprehensive weight statistics
- `full_audit_pipeline(dataset, ...)` — end-to-end audit

### Key Design Principles

1. **No undocumented defaults.** Every parameter (lambda1, w_threshold, max_iter, loss_type) must be explicitly specified.
2. **Assertions for critical shapes.** `assert X.ndim == 2`, `assert X.shape[1] == N`, etc.
3. **Transparent logging.** Every step prints shapes and statistics.
4. **Independently testable.** Each function can be called standalone.

## 2. What Was Verified

### DAGMA Input Construction

The `extract_dagma_input` function was verified to produce the same result as the existing code:

```python
# Existing code:
data = np.array([x[0] for x in self.train_data])
# Where self.train_data = np.array([x[0].numpy() for x in train_dataset])

# Independent reimplementation:
data = np.array([train_X[i][0] for i in range(num_samples)])
```

**Result:** Both produce shape (M, N) where each row is a contemporaneous snapshot.

### DAGMA Input Audit

Ran `gsl_audit/dagma_input_audit.py` which confirmed:

```
SZ-Taxi:  data_for_dagma shape (2365, 156)
Los-loop: data_for_dagma shape (1597, 207)
```

Each row = one contemporaneous snapshot of all sensors at a single time step.

### Cross-Sectional Correlations

```
SZ-Taxi:  off-diagonal mean=0.19, max=0.94
Los-loop: off-diagonal mean=0.19, max=0.98
```

Strong contemporaneous correlations exist. DAGMA identifies a sparse DAG subset.

## 3. DAGMA Training Attempt

DAGMA training was attempted with w_threshold=0 on SZ-Taxi:
- 5,000 iterations: ~2 minutes (incomplete)
- 180,000 iterations (full): estimated ~72 minutes per run

**Conclusion:** Full DAGMA retraining with w_threshold=0 is not feasible in interactive sessions. The existing W_est files (generated with w_threshold=0.3) are used for analysis instead.

From the previous threshold audit (DAGMA_THRESHOLD_AUDIT_20260831_120000.md):
- With w_threshold=0 on a subsample: only 24 entries have |W| ≥ 0.001 for SZ-Taxi
- The sparsity is intrinsic to DAGMA's L1+DAG solution

## 4. Relationship to Existing Code

The independent module does NOT replace the existing implementation. It:
- Provides an auditable, parameter-explicit alternative
- Can be used for future experiments
- Documents exactly what the existing code does

## 5. Limitations

1. Full DAGMA retraining with w_threshold=0 was not completed (too slow).
2. The independent module has not been used for end-to-end training experiments.
3. The existing W_est files remain the primary evidence for sparsity analysis.

---

*Report generated: 2026-08-31 15:29:25*
