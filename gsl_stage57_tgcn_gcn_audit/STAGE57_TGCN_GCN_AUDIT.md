# Stage 57 — T-GCN vs GCN Implementation Audit

**Date:** 2026-09-11  
**Scope:** diagnosis only. No hyperparameter search. No multi-seed training run in this stage.  
**Focus models:** GCN-NoSpatial, GCN-Physical, T-GCN-NoSpatial, T-GCN-Physical.

---

## 1. Problem statement

Under the Stage 40 canonical protocol, **GCN-NoSpatial** is highly competitive and often better than **T-GCN-NoSpatial** (and other T-GCN variants). Example Los-loop PH1 five-seed means (`gsl_stage41/stage41_summary.csv`):

| Model | RMSE mean (std) |
|-------|-----------------|
| GCN-NoSpatial | **4.88 (0.31)** |
| T-GCN-NoSpatial | 5.25 (0.19) |
| GCN (Physical) | 8.14 (0.11) |
| T-GCN (Physical) | 7.88 (0.28) |

On SZ-Taxi PH1, GCN-NoSpatial (4.11) and T-GCN-NoSpatial (4.12) are nearly tied.  
The question is **not** whether GSL is wrong, but whether the **GCN vs T-GCN baseline comparison** is implementation- or protocol-fair.

---

## 2. Current GCN implementation

File: `models/gcn.py`

- Input: `(B, T, N)` with `T = seq_len = 12`.
- One graph convolution on the **flattened window as node features**:
  - Reshape to `(N, B·T)`, multiply by Laplacian, reshape, then `tanh(A X W)` with `W ∈ R^{T×hidden}`.
- Output: `(B, N, hidden)`.
- **Not** a multi-layer Kipf GCN; it is the classic “GCN as spatial filter over the window” used in T-GCN baselines.
- **No** temporal recurrence inside the model.
- Head: shared `Linear(hidden, pre_len)` in `SupervisedForecastTask`.

**With identity adjacency (NoSpatial):** Laplacian of `I` is `I` (after the same normalization). The model becomes, **per node independently**, a linear map from the 12-step window to `hidden`, then to `PH` — i.e. a **per-node linear autoregression** on the window. That is a strong, low-variance baseline for short horizons.

---

## 3. Current T-GCN implementation

File: `models/tgcn.py`

- Recurrent cell `TGCNCell` over `t = 0…T-1` (chronological if `X` is stored as `[t-11,…,t]`).
- Graph convolution is **inside the GRU-style gates**, not “GCN then GRU” as a two-stage pipeline:
  - `graph_conv1([x_t, h_{t-1}])` → sigmoid → reset/update gates;
  - `graph_conv2([x_t, r ⊙ h_{t-1}])` → candidate state.
- Output: last hidden state `(B, N, hidden)` → same `Linear(hidden, pre_len)` head.
- This matches the **official PyTorch/TF “T-GCN” cell style** of Zhao et al.’s public repo (`lehaifeng/T-GCN`: graph conv applied to `[x, h]` inside the cell), **not** a literal “GCN(Z_t) then vanilla GRU(Z_t)” pipeline from the paper’s high-level figure.

**With identity adjacency:** graph conv reduces to a per-node feature mixing of `[x, h]` — a **GRU-like recurrent** model without spatial mixing. More parameters and a harder optimization problem than the linear GCN-NoSpatial map.

**Identity ⇒ GRU equivalence (verified).**  
`calculate_laplacian_with_self_loop(I)` yields $\hat{A}=I$ (self-loop and degree scaling cancel). Then `TGCNGraphConvolution` is exactly `[x,h]@W`, i.e. the same computation as `models/gru.py`. **T-GCN-NoSpatial is architecturally a vanilla GRU**, not “GCN with a dummy graph.”  
A 1-epoch sanity run (seed 42, Los PH1, identity) produced **identical** RMSE for arm C (T-GCN+mse) and arm E (GRU+mse): `14.6728`.  
Therefore Stage 40 **GCN-NoSpatial vs T-GCN-NoSpatial** is primarily **linear window map vs GRU**, not a pure graph-vs-no-graph contrast.

**Comparison to paper text:** Zhao et al. describe GCN for spatial dependence and GRU for temporal dependence. The official code integrates graph convolution into the recurrent cell. Our code follows the **official code lineage**. This is **not an obvious bug**, but it is **not identical** to a strict two-stage GCN→GRU reading of the paper.

---

## 4. Input / output semantics (shared)

From `gsl_stage40/scripts/stage40_run_all.py`:

```text
X[i] = data[i : i+seq_len]           # (T, N), chronological
Y[i] = data[i+seq_len : i+seq_len+PH]
```

- Same for GCN and T-GCN.
- Train max normalization only; same split.
- Head: `Linear(hidden_dim, PH)` jointly predicts all horizons (not autoregressive).
- Temporal order in T-GCN: `for i in range(seq_len): cell(inputs[:, i, :], h)` — **not reversed**.

**No I/O mismatch found** between GCN and T-GCN.

---

## 5. Graph handling

Shared: `utils/graph_conv.py::calculate_laplacian_with_self_loop`

```text
Ã = A + I
L̂ = D̃^{-1/2} Ãᵀ D̃^{-1/2}
```

- Same operator for physical, identity, and learned graphs in both backbones.
- Identity → effectively `I` after self-loop + normalization (degree 2 on diagonal after +I, scaling cancels to identity on off-diagonal zeros).
- Physical Los adjacency is dense (order 10³ off-diag entries) — oversmoothing risk is architecture-independent.

**No unfair normalization asymmetry** between GCN and T-GCN for the same `A`.

---

## 6. Training protocol (shared vs asymmetric)

| Item | GCN (Stage 40) | T-GCN (Stage 40) |
|------|----------------|------------------|
| Optimizer | Adam | Adam |
| LR | 1e-3 | 1e-3 |
| Adam `weight_decay` | 1e-4 | 1e-4 |
| Batch | 128 | 128 |
| Epochs | 50 | 50 |
| Hidden | 64 | 64 |
| Seeds | 42–46 | 42–46 |
| **Loss** | **`F.mse_loss` (mean)** | **`mse_with_regularizer` (sum + λ‖θ‖²)** |
| **λ_reg in loss** | **none** | **1.5e-3** |

Code evidence: `stage40_run_all.py`  
`loss_name = "mse_with_regularizer" if backbone == "tgcn" else "mse"`.

`utils/losses.py`:

```python
mse_loss = torch.sum((inputs - targets) ** 2) / 2   # SUM, not mean
reg_loss = lamda * sum(θ²) / 2
```

### Why this matters

1. **Reduction mismatch:** sum-of-squares vs mean MSE → gradient scale differs by ~number of elements in the batch tensor. Training dynamics are not comparable.
2. **Double regularization on T-GCN:** loss-level L2 (`1.5e-3`) **plus** Adam `weight_decay=1e-4`. GCN gets only Adam weight decay.
3. **Likely bias direction:** heavier regularization + harder recurrent optimization can **hurt T-GCN relative to GCN**, especially with identity graphs where GCN is a simple linear map.

This is the **primary experimental confound** for “GCN-NoSpatial beats T-GCN-NoSpatial.”

---

## 7. Parameter counts (approx., backbone only)

| Model | Back-of-envelope |
|-------|------------------|
| GCN | `T × hidden` = 12×64 = **768** (+ head `64×PH`) |
| T-GCN | cell graphs: ~`(H+1)×2H + 2H` + `(H+1)×H + H` ≈ **12,672** at H=64 (+ head) |

T-GCN is **much larger**, not unfairly small. Capacity is **not** the reason T-GCN loses to GCN-NoSpatial. If anything, extra capacity without matched loss makes the confound **worse to diagnose**.

---

## 8. Potential implementation issues (ranked)

| # | Issue | Severity | Evidence |
|---|--------|----------|----------|
| 1 | **Asymmetric loss (mean MSE vs sum+L2)** | **High** | `stage40_run_all.py:311`, `utils/losses.py` |
| 2 | **Double L2 on T-GCN** (loss + Adam wd) | **High** | same |
| 3 | **T-GCN-NoSpatial ≡ vanilla GRU** (identity Laplacian is I) | **High (interpretation)** | `graph_conv.py`; 1-epoch C≡E = 14.6728 |
| 4 | T-GCN cell ≠ strict paper “GCN then GRU” pipeline | Medium (design) | `models/tgcn.py` vs Zhao paper prose; matches `lehaifeng/T-GCN` code style |
| 5 | GCN-NoSpatial is a very strong linear AR baseline | Medium (interpretation) | Identity + one-shot window map |
| 6 | Val loss for T-GCN uses reg loss on **de-normalized** scale | Low (metrics RMSE OK) | `supervised.py` validation |
| 7 | GCN uses `tanh`; GRU cell uses sigmoid/tanh gates | Low | different but standard |

**No bug found** in temporal order, shared I/O, or Laplacian application.

---

## 9. Potential experimental confounders

1. Loss / regularization asymmetry (**main**).
2. Architecture family difference (linear window map vs recurrent cell) **with identity graph** — expected even under a fair loss.
3. Physical graph density (oversmoothing) affecting both families when `A` is physical.
4. Short horizons PH=1–4 favor simple models; GRU may need longer sequences or different hidden size to show gains (do **not** retune yet).

---

## 10. Recommended next experiment (prepared, not run here)

Isolate **loss** while holding data, seed, optimizer settings, and graphs fixed:

| Arm | Backbone | Adj | Loss |
|-----|----------|-----|------|
| A | GCN | I / physical | **mse** (canonical GCN) |
| B | T-GCN | I / physical | **mse_with_regularizer** (canonical T-GCN) |
| C | T-GCN | I / physical | **mse** (no loss L2) |
| D | GCN | I / physical | **mse_with_regularizer** (same as T-GCN) |
| E | GRU (no graph) | — | **mse** (optional control) |

Smoke: 1 dataset, PH1, seed 42, few epochs.  
Full: both datasets, PH1–4, seeds 42–46 — **for the user to run**.

Scripts:

- `gsl_stage57_tgcn_gcn_audit/audit_tgcn_gcn.py`
- `run_tgcn_gcn_audit.sh`

---

## 11. What should NOT be changed yet

- No GSL / DAGMA changes.
- No architecture rewrite of T-GCN toward a different paper variant.
- No hyperparameter search.
- No replacement of Stage 40 numbers in the manuscript until this audit experiment is run and interpreted.
- Do not assume “T-GCN is broken”; assume **protocol asymmetry until arm C/D refute or confirm it**.

---

## 12. Conclusion (verdict)

**Verdict: C — An important experimental confound was found.**

- T-GCN implementation is **consistent with the official T-GCN code lineage** (graph conv in the recurrent cell); it is **not** an obvious coding bug versus `lehaifeng/T-GCN`.
- The **GCN vs T-GCN comparison is not loss-fair**: different reduction and extra L2 on T-GCN can systematically disadvantage T-GCN, including on NoSpatial where GCN is only a linear per-node map.
- Therefore **GCN-NoSpatial ≻ T-GCN-NoSpatial in Stage 40 does not yet prove that the T-GCN architecture is inferior** under a matched protocol.
- **Action:** run `run_tgcn_gcn_audit.sh` (user) and compare arms A–D before any new GSL work or manuscript claim changes about GCN vs T-GCN baselines.
