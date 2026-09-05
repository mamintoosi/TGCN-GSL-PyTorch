# Reimplementation Specification — GSL for Traffic Prediction

**Derived from:** paper/sn-article.tex (not from existing code)
**Date:** 2026-08-31

---

## 1. Problem Statement

Predict future traffic speed: $[X_{t+1}, \ldots, X_{t+T}] = f(G; [X_{t-n}, \ldots, X_t])$

Where:
- $X_t \in \mathbb{R}^N$ is traffic speed across $N$ roads at time $t$
- $G$ is a graph with adjacency matrix $\mathbf{A}$
- $n$ = historical window (12)
- $T$ = prediction horizon (1–4)

## 2. Baseline Models

### 2a. GCN (Kipf & Welling 2017)

$$\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$

Where:
- $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}_N$ (self-loops added by GCN normalization, NOT in the adjacency)
- $\tilde{\mathbf{D}}$ = degree matrix of $\tilde{\mathbf{A}}$
- Input: $(B, \text{seq\_len}, N)$
- Output: $(B, N, \text{hidden\_dim})$
- Then a linear regressor maps $(B \cdot N, \text{hidden\_dim}) \to (B \cdot N, \text{pre\_len})$

### 2b. T-GCN (Zhao et al. 2019)

GCN captures spatial features at each time step, fed into GRU for temporal modeling:
- GCN part: $Z_t = f(\mathbf{A}, X_t)$
- GRU part: $S_t = \text{GRU}(Z_t, S_{t-1})$
- Output: $(B, N, \text{hidden\_dim})$

### Key architectural detail

Both models use `calculate_laplacian_with_self_loop` which computes:
$$\hat{A} = (A + I)^{-1/2} (A + I) (A + I)^{-1/2}$$

This is the **symmetric normalized Laplacian with self-loops**. Self-loops are added during normalization, not in the adjacency matrix itself.

## 3. Graph Structure Learning (the proposed method)

### 3a. DAGMA Formulation

Given data matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ (n observations of d variables):

$$\min_{W \in \mathbb{R}^{d \times d}} \text{score}(W) \quad \text{subject to} \quad h(W) = 0$$

Where:
- $h(W) = \text{tr}(e^{W \circ W}) - d = 0$ iff $W$ is a DAG
- Score function: L2 loss + L1 regularization
- Solved via augmented Lagrangian

### 3b. DAGMA Hyperparameters (from paper)

| Parameter | SZ-Taxi | Los-loop |
|-----------|---------|----------|
| lambda1 (L1) | 0.01 | 0.02 |
| loss_type | l2 | l2 |
| w_threshold | 0.3 (library default) | 0.3 (library default) |
| iterations | 180,000 (library default) | 180,000 (library default) |

### 3c. DAGMA Input (CRITICAL — paper interpretation vs implementation)

**Paper Section 5 claims:** "An edge $j \rightarrow i$ signifies that the traffic state on road $j$ at time $t$ has a predictive, causal influence on the traffic state of road $i$ at time $t+1$."

**Actual implementation (from code):**
```python
data = np.array([x[0] for x in self.train_data])
# data shape: (num_samples, N) — each row is one time step's snapshot

X = data[i::pre_len]  # subsample for this horizon
# X shape: (num_samples/pre_len, N)

model = DagmaLinear(loss_type='l2')
w_est = model.fit(X, lambda1=lambda1)
```

**Analysis:** DAGMA receives $\mathbf{X} \in \mathbb{R}^{M \times N}$ where each row is a **contemporaneous** snapshot of all $N$ sensors. This is a **cross-sectional** input, not a lagged/temporal one.

**Interpretation:** The learned $W_{ij}$ represents: "among the $M$ time-stamped observations, how does the value of sensor $j$ co-vary with sensor $i$ after accounting for all other sensors' effects and the DAG constraint?"

This is closer to **contemporaneous correlation** than **temporal causation**. The paper's Section 5 interpretation of "temporal dependency graph" appears to be an aspirational reinterpretation, not what the code actually computes.

### 3d. DAGMA Output Processing

```python
# W_est_all shape: (N, N, pre_len) — one matrix per horizon

if W_est_all.ndim == 3:
    W_est = np.any(W_est_all > 0, axis=2)  # union across horizons

# use_gsl=1 (GSL): directed, binary
adj = np.zeros(W_est.shape, dtype=int)
adj[W_est > 0] = 1

# use_gsl=2 (cGSL): symmetrized, binary
adj = np.zeros(W_est.shape, dtype=int)
adj[W_est > 0] = 1
adj = adj + adj.T  # symmetrize
```

### 3e. Key observations about output processing

1. **Thresholding:** `W_est > 0` — only positive weights become edges. Negative weights are discarded.
2. **DAGMA's internal threshold:** `w_threshold=0.3` zeros out all $|W| < 0.3$ before returning.
3. **Union across horizons:** `np.any(W_est_all > 0, axis=2)` takes the OR of all PH graphs.
4. **Result:** Binary adjacency matrix, no weights, no self-loops (self-loops added later by GCN normalization).

## 4. Data Pipeline

### Raw data
- `sz_speed.csv`: (8624, 156) — 15-minute intervals
- `los_speed.csv`: (207, 2016) — 5-minute intervals (transposed?)

### Preprocessing
1. Load features: `(T, N)` matrix of speeds
2. Normalize: divide by global max
3. Generate sequences: sliding window of `seq_len=12` → `(B, 12, N)`
4. Targets: next `pre_len` steps → `(B, pre_len, N)`
5. Split: 80% train, 20% validation (temporal order preserved)

### Graph construction timing
- DAGMA runs ONCE before training, using the normalized training data
- Graph is fixed during training
- Graph is loaded from `.npy` files if they exist (cached)

## 5. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Batch size | 64 |
| Epochs | 50 |
| Loss | MSE (with optional L2 regularization, λ=1.5e-3) |
| Hidden dim | 100 |
| Seq len | 12 |
| Device | CUDA (RTX 3090) |

## 6. Evaluation

- Metrics: RMSE, MAE, Accuracy, R²
- Evaluated on validation set (not test set — the code uses `val_dataset` for evaluation)
- Predictions are de-normalized by multiplying by `feat_max_val`

## 7. Discrepancies Identified

| # | Severity | Description |
|---|----------|-------------|
| 1 | CRITICAL | DAGMA input is contemporaneous (rows of X are time snapshots), but paper Section 5 claims temporal causation |
| 2 | MAJOR | `w_threshold=0.3` is an undocumented library default that zeros out most weights |
| 3 | MAJOR | Evaluation is on validation set, not a held-out test set |
| 4 | MINOR | Loss function uses `mse_with_regularizer_loss` with λ=1.5e-3 by default, but config says loss="mse" |

## 8. What the Independent Reimplementation Must Verify

1. DAGMA receives `(M, N)` matrix where M = number of training samples (subsampled)
2. The learned W is mostly zeros even before `w_threshold=0.3`
3. The graph construction correctly handles directed vs symmetrized
4. GCN normalization correctly adds self-loops
5. The training loop matches the paper's description
6. Results with the independent implementation match (or don't match) previous results

---

*This spec is derived from the paper source, not from the code.*
