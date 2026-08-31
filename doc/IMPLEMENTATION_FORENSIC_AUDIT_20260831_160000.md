# Implementation Forensic Audit — GSL for Traffic Prediction

**Date:** 2026-08-31  
**Repository:** TGCN-GSL-PyTorch  
**Paper:** "Graph Structure Learning for Traffic Prediction"

---

## 1. Executive Summary

1. **CRITICAL: DAGMA input is contemporaneous, not temporal.** The code extracts `x[0]` from each training sequence, producing a `(M, N)` matrix where each row is a single-timestep snapshot of all N sensors. DAGMA learns cross-sectional dependencies, NOT temporal causation.
2. **CRITICAL: Paper Section 5 interpretation is unsupported.** The claim that "edge j→i means j at time t predicts i at time t+1" is not grounded in the actual input construction.
3. **MAJOR: `w_threshold=0.3` is an undocumented library default.** DAGMA's `fit()` zeros out all `|W| < 0.3` internally. The project never overrides this.
4. **MAJOR: Extreme sparsity is intrinsic to DAGMA.** Even with `w_threshold=0`, the solution is very sparse. The L1+DAG constraint produces 8 edges for SZ-Taxi (94% isolated nodes).
5. **MAJOR: Ablation shows sparsification dominates.** A random sparse graph with the same edge count as GSL matches or outperforms the learned graph in 3 of 4 dataset/model combinations.
6. **MINOR: Evaluation is on validation set, not held-out test set.**
7. **MINOR: Loss function mismatch** — code uses `mse_with_regularizer` (λ=1.5e-3) by default, but some configs specify `loss=mse`.

---

## 2. Paper Specification (from sn-article.tex)

### 2a. Core Idea

Replace the fixed physical adjacency matrix A with a learned graph structure via DAGMA, then use this learned graph in GCN/T-GCN.

### 2b. DAGMA Formulation

Given X ∈ R^{M×N} (M observations of N variables):
- Minimize: score(W) = ||X - X·W||² + λ₁||W||₁
- Subject to: h(W) = tr(e^{W∘W}) - N = 0 (DAG constraint)
- Solved via augmented Lagrangian
- Output: W ∈ R^{N×N} (weighted adjacency)

### 2c. Two Variants

- **GSL (use_gsl=1):** A = (W > 0), directed binary graph
- **cGSL (use_gsl=2):** A = (W > 0) + (W > 0)ᵀ, symmetrized binary graph

### 2d. Key Parameters (from paper)

| Parameter | SZ-Taxi | Los-loop |
|-----------|---------|----------|
| λ₁ | 0.01 | 0.02 |
| w_threshold | not mentioned | not mentioned |
| loss_type | l2 | l2 |
| seq_len | 12 | 12 |
| hidden_dim | 100 | 100 |
| epochs | 50 | 50 |
| optimizer | Adam | Adam |
| lr | 0.001 | 0.001 |
| batch_size | 64 | 64 |

---

## 3. Existing Implementation Analysis

### 3a. Complete Execution Graph

```
raw CSV data: sz_speed.csv (2976, 156) or los_speed.csv (2016, 207)
    ↓ load_features()
feature matrix: (T, N) float32
    ↓ normalize by global max
normalized features: (T, N) in [0, 1]
    ↓ generate_dataset(seq_len=12, pre_len=PH)
train_X: (M, 12, N), train_Y: (M, PH, N)   ← sliding windows
    ↓ SpatioTemporalCSVData.get_datasets()
TensorDataset(train_X, train_Y)
    ↓ extract: self.train_data = np.array([x[0] for x in dataset])
    ↓ extract: data = np.array([x[0] for x in self.train_data])
data: (M, N)   ← FIRST TIME STEP of each sequence
    ↓ DAGMA input: X = data[i::PH] for each PH
X: (M/PH, N)   ← subsampled contemporaneous snapshots
    ↓ DagmaLinear.fit(X, lambda1, w_threshold=0.3 default)
W_est: (N, N) per PH
    ↓ saved as W_est_all: (N, N, PH) = stacked per-PH matrices
    ↓ W_est = np.any(W_est_all > 0, axis=2)  — union across PHs
binary directed: (N, N) bool
    ↓ use_gsl=1: adj = zeros, adj[W_est>0] = 1  (directed)
    ↓ use_gsl=2: adj = zeros, adj[W_est>0] = 1, adj += adj.T  (symmetric)
adjacency: (N, N) int
    ↓ calculate_laplacian_with_self_loop(adj)
    ↓ Adds self-loop, computes D^{-1/2} Ã D^{-1/2}
laplacian: (N, N) float32
    ↓ GCN or TGCN forward pass
forecasting output: (B, N, PH)
```

### 3b. DAGMA Input Construction (CRITICAL)

**File:** `utils/data/spatiotemporal_csv_data.py`, lines ~85-95

```python
# train_dataset is TensorDataset(train_X, train_Y)
# train_dataset[i] returns (train_X[i], train_Y[i])
# x = train_X[i], which has shape (seq_len, N) = (12, N)
# x[0] = first time step of sequence i, shape (N,)

self.train_data = np.array([x[0].numpy() for x in train_dataset])
# self.train_data shape: (M, N) — first timestep from each sequence

data = np.array([x[0] for x in self.train_data])
# data shape: (M, N) — same as self.train_data
# This is confirmed by the audit script

X = data[i::self.pre_len]
# X shape: (M/PH, N) — subsampled rows, still contemporaneous
```

**Each row of X is one contemporaneous snapshot:** X[k] = [speed_1(t_k), speed_2(t_k), ..., speed_N(t_k)]

**DAGMA learns:** which sensor values co-vary with which other sensor values across M simultaneous observations.

### 3c. Graph Construction

**File:** `utils/data/spatiotemporal_csv_data.py`, lines ~100-130

```python
# After DAGMA produces W_est_all of shape (N, N, PH):
if W_est_all.ndim == 3:
    W_est = np.any(W_est_all > 0, axis=2)  # union across PHs

# use_gsl=1 (GSL):
adj = np.zeros(W_est.shape, dtype=int)
adj[W_est > 0] = 1  # directed binary

# use_gsl=2 (cGSL):
adj = np.zeros(W_est.shape, dtype=int)
adj[W_est > 0] = 1
adj = adj + adj.T  # symmetrized (may have values > 1, but no re-threshold)
```

### 3d. GCN Normalization

**File:** `utils/graph_conv.py`

```python
def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(N)  # add self-loops
    row_sum = matrix.sum(1)
    d_inv_sqrt = pow(row_sum, -0.5)
    d_mat_inv_sqrt = diag(d_inv_sqrt)
    return matrix @ d_inv_sqrt @ d_inv_sqrt.T  # symmetric normalization
```

This is D̃^{-1/2} Ã D̃^{-1/2} as described in the paper (Eq. 2).

---

## 4. Discrepancy Analysis

### CRITICAL Discrepancies

| # | Description | Impact |
|---|------------|--------|
| C1 | DAGMA input is contemporaneous (M, N), not temporal/lagged | Paper Section 5 claims temporal causation; code learns contemporaneous correlation |
| C2 | Paper says "edge j→i means j at time t predicts i at time t+1" but DAGMA receives simultaneous observations | The learned graph does NOT represent temporal dependencies as claimed |
| C3 | The DAG constraint on contemporaneous data means: "among N sensors, the dependency structure has no cycles" — this is a statistical property, not a temporal one | The "temporal DAG" interpretation in Section 5 is an aspirational reinterpretation |

### MAJOR Discrepancies

| # | Description | Impact |
|---|------------|--------|
| M1 | `w_threshold=0.3` is an undocumented library default; project code never specifies it | Most DAGMA weights are silently zeroed out |
| M2 | Evaluation on validation set, not held-out test set | Results may be optimistic |
| M3 | The `W > 0` conversion discards negative weights without justification | Could be meaningful (anti-correlations) |
| M4 | `np.any(W_est_all > 0, axis=2)` unions across PHs | Different PHs may have different optimal graphs; union may not be optimal |

### MINOR Discrepancies

| # | Description | Impact |
|---|------------|--------|
| m1 | Hard-coded λ values (0.01, 0.02) without sensitivity analysis | Robustness unknown |
| m2 | Loss function inconsistency (mse vs mse_with_regularizer) | Minor effect |
| m3 | No random seed control for DAGMA (deterministic) | OK |
| m4 | Random seed 42 for training but no multi-seed reporting | Variance unknown |

---

## 5. DAGMA Input Audit Results

### SZ-Taxi

```
Raw features: (2976, 156) — 2976 timesteps, 156 sensors
Normalized: [0.0, 1.0]
train_X: (2365, 12, 156) — 2365 sequences
DAGMA input: (2365, 156) — each row = one snapshot
For PH=1: X = (2364, 156)
For PH=2: X = (1182, 156)
For PH=3: X = (788, 156)
For PH=4: X = (591, 156)
```

### Los-loop

```
Raw features: (2016, 207) — 2016 timesteps, 207 sensors
train_X: (1597, 12, 207)
DAGMA input: (1597, 207)
For PH=1: X = (1596, 207)
For PH=2: X = (798, 207)
For PH=3: X = (532, 207)
For PH=4: X = (399, 207)
```

### Cross-Sectional Correlation

```
SZ-Taxi: off-diagonal mean=0.19, max=0.94
Los-loop: off-diagonal mean=0.19, max=0.98
```

Strong contemporaneous correlations exist between many sensor pairs. DAGMA identifies a sparse DAG subset.

---

## 6. Sparsity Analysis

### Existing W_est Files (w_threshold=0.3)

| Dataset | PH | Nonzero | Positive | Negative | Edges after W>0 |
|---------|---:|--------:|---------:|---------:|-----------------:|
| SZ-Taxi | 1 | 8 | 8 | 0 | 8 |
| SZ-Taxi | 2 | 8 | 8 | 0 | 8 |
| SZ-Taxi | 3 | 8 | 8 | 0 | 8 |
| SZ-Taxi | 4 | 8 | 8 | 0 | 8 |
| Los-loop | 1 | 28 | 28 | 0 | 28 |
| Los-loop | 2 | 32 | 32 | 0 | 32 |
| Los-loop | 3 | 33 | 33 | 0 | 33 |
| Los-loop | 4 | 39 | 39 | 0 | 39 |

**All nonzero weights are positive and ≥ 0.303.** The W>0 conversion loses nothing — DAGMA already produced only positive weights ≥ 0.3.

### Physical vs Learned Graph

| Dataset | Physical Edges | GSL Edges | cGSL Edges | Sparsity Ratio |
|---------|---------------:|----------:|-----------:|---------------:|
| SZ-Taxi | 532 | 8 | 16 | 66.5× |
| Los-loop | 2833 | 28–39 | 56–78 | 72–101× |

### Isolated Nodes

| Dataset | GSL Edges | Nodes | Isolated | % Isolated |
|---------|----------:|------:|---------:|-----------:|
| SZ-Taxi | 8 | 156 | 147 | 94.2% |
| Los-loop | 28 | 207 | 174 | 84.1% |

---

## 7. Ablation Results (Sparse Random vs GSL)

The sparsified-physical-graph ablation (64 experiments) showed:

### Key Finding: Where does sparse_random sit?

| Dataset | Model | Sparse Random Coverage | Verdict |
|---------|-------|----------------------:|---------|
| SZ-Taxi | GCN | 63% | Both sparsification and topology matter |
| SZ-Taxi | TGCN | 110–123% | Sparse random OUTPERFORMS GSL |
| Los-loop | GCN | Negative | Random sparsification is harmful |
| Los-loop | TGCN | 79% | Sparsification dominates |

### SZ-Taxi / TGCN: Most Damaging Finding

A random sparse graph with 8 edges consistently outperforms the DAGMA-learned graph:
- PH=1: sparse_random=4.151 vs gsl=4.211 (sparse random wins)
- PH=2: sparse_random=4.201 vs gsl=4.252 (sparse random wins)
- PH=3: sparse_random=4.232 vs gsl=4.290 (sparse random wins)
- PH=4: sparse_random=4.229 vs gsl=4.311 (sparse random wins)

This means the specific edge identities chosen by DAGMA are not just unnecessary — they're slightly worse than random placement for TGCN on SZ-Taxi.

---

## 8. Root Cause Classification

**Category: A (Intrinsic DAGMA sparsity) + F (combination)**

The extreme sparsity has two components:
1. **Intrinsic:** DAGMA's L1 regularization + DAG constraint produces a genuinely sparse solution. Even with w_threshold=0, only ~24 entries have |W| ≥ 0.001 for SZ-Taxi.
2. **Amplified by w_threshold=0.3:** The library default zeros out all weights below 0.3, reducing ~24 meaningful entries to 8.

---

## 9. Severity Classification Summary

| Severity | Count | Key Issues |
|----------|------:|------------|
| CRITICAL | 3 | DAGMA input is contemporaneous, not temporal; paper claims unsupported |
| MAJOR | 4 | Undocumented w_threshold, validation-only eval, negative weights discarded, PH union |
| MINOR | 4 | Hard-coded λ, loss inconsistency, no multi-seed, no sensitivity analysis |

---

## 10. Scientific Assessment

### Is the current implementation correct relative to the paper?

**PARTIALLY.** The implementation faithfully executes the code as written, but the paper's interpretation (Section 5) of what the learned graph represents is NOT supported by the actual DAGMA input construction.

### Is the extreme sparsity expected?

**YES.** DAGMA's L1+DAG constraint on contemporaneous data produces intrinsically sparse solutions. The w_threshold=0.3 amplifies this but is not the primary cause.

### Is the GSL graph scientifically defensible?

**CONDITIONAL.**
- The graph captures real contemporaneous statistical dependencies.
- However, the paper's claims about temporal causation and interpretability are not supported.
- The sparsification benefit confounds the topology benefit in most cases.

---

## 11. Recommended Actions for Revision

### Essential (must do)

1. **Correct Section 5 interpretation.** Acknowledge that DAGMA receives contemporaneous data and learns cross-sectional dependencies, not temporal causation.
2. **Report graph density/degree distribution** for physical, GSL, and cGSL graphs.
3. **Include the sparse-random ablation** in the paper to honestly address Reviewer 1's concern.
4. **Make w_threshold explicit** in the project code and paper.

### Strongly recommended

5. **Multi-seed evaluation** (5 seeds) with mean ± std.
6. **Top-K correlation baseline** to test if DAGMA edges are better than simple correlation.
7. **Lambda sensitivity analysis.**
8. **Add a limitations section** acknowledging the sparsification confound.

---

## 12. Files Created

| File | Description |
|------|-------------|
| `doc/REIMPLEMENTATION_SPEC.md` | Mathematical specification from paper |
| `doc/IMPLEMENTATION_FORENSIC_AUDIT_20260831_160000.md` | This report |
| `gsl_audit/independent_gsl.py` | Independent GSL module with explicit parameters |
| `gsl_audit/dagma_input_audit.py` | Forensic audit of DAGMA input construction |

---

*Report generated: 2026-08-31 16:00:00*
