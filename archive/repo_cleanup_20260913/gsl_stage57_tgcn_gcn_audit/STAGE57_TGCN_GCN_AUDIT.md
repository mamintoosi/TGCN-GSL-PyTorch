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
Full: both datasets, PH1–4, seeds 42–46 — **completed** (`results/stage57_tgcn_gcn_audit/full.csv`, 320 rows, 50 epochs).

Scripts:

- `gsl_stage57_tgcn_gcn_audit/audit_tgcn_gcn.py`
- `gsl_stage57_tgcn_gcn_audit/analyze_full.py`
- `run_tgcn_gcn_audit.sh`

---

## 11. Full-run results (Mode=full)

**Protocol:** 50 epochs, batch 128, Adam lr 1e-3, wd 1e-4, hidden 64, seq_len 12; arms A–D × {identity, physical} × {losloop, shenzhen} × PH1–4 × seeds 42–46.

**Reproduction of Stage 40:**  
T-GCN+reg (arm B) matches Stage 40 NoSpatial/Physical almost exactly (e.g. Los identity PH1 **5.251** vs Stage 40 **5.25**; Los physical PH1 **7.883** vs **7.88**). GCN+mse (arm A) matches GCN Stage 40 (Los identity PH1 **4.965** vs **4.88** — small gap; physical **8.145** exact).

### Identity (NoSpatial) — mean RMSE

| Dataset | PH | A GCN+mse | D GCN+reg | B T-GCN+reg | C T-GCN+mse |
|---------|----|-----------|-----------|-------------|-------------|
| Los | 1 | **4.97** | 4.90 | 5.25 | 5.42 |
| Los | 2 | **5.60** | 5.63 | 5.76 | 5.90 |
| Los | 3 | **6.02** | 6.04 | 6.11 | 6.23 |
| Los | 4 | **6.26** | 6.27 | 6.58 | 6.67 |
| SZ | 1–4 | **best or tied** | ≈A | slightly worse | slightly worse than B |

**GCN+mse beats T-GCN+mse in 8/8 identity cells.**

### Physical — mean RMSE

| Dataset | Pattern |
|---------|---------|
| Los | **T-GCN (B/C) beats GCN (A/D)** in 4/4 PHs (e.g. PH1: 7.88 vs 8.15) |
| SZ | **T-GCN+reg (B) much better** than T-GCN+mse (C) and GCN (PH1: **5.45** vs 5.79 vs 5.96) |

### Loss isolation

| Comparison | Identity | Physical |
|------------|----------|----------|
| C − B (drop T-GCN loss L2) | **+0.07 to +0.17** (C **worse**) | ≈0 on Los; **+0.2 to +0.34 on SZ** (C **worse**) |
| D − A (add GCN loss L2) | ≈0 after 50 epochs (slightly better at Los PH1) | ≈0 |
| A − C (GCN vs T-GCN, both mse) | **GCN better** | **T-GCN better** |

**Conclusion on loss:** the Stage 40 loss asymmetry does **not** explain GCN-NoSpatial ≻ T-GCN-NoSpatial. After 50 epochs, removing T-GCN’s loss-level L2 does **not** help T-GCN (often slightly hurts). On SZ+physical, the regularizer **helps** T-GCN.

### Architecture interpretation

1. **Identity + GCN** ≈ per-node linear map on the 12-step window → strong short-horizon baseline.  
2. **Identity + T-GCN** ≡ **GRU** → recurrent model, slightly worse at PH≤4 under this protocol.  
3. **Physical + T-GCN** ≻ **Physical + GCN** → when a (dense) graph is used, the recurrent T-GCN cell uses it better than the one-shot GCN.  
4. Stage 40’s “GCN-NoSpatial is competitive” is therefore an **architecture/inductive-bias** result, not a broken T-GCN implementation.

---

## 12. What should NOT be changed yet

- No GSL / DAGMA changes based on this audit alone.
- No forced “T-GCN must beat GCN everywhere” claim.
- Optional later: document in the paper that NoSpatial GCN is a linear-window baseline and NoSpatial T-GCN is a GRU (identity Laplacian).
- Do not discard Stage 40 numbers — they reproduce arm B.

---

## 13. Conclusion (verdict — updated after Mode=full)

**Verdict: C (refined) — Confound identified and largely ruled out as the cause of GCN ≻ T-GCN on NoSpatial.**

- T-GCN implementation matches the official code lineage; Stage 40 numbers reproduce (arm B).  
- **Loss asymmetry is real but not the driver** of GCN-NoSpatial ≻ T-GCN-NoSpatial (C ≈ B or worse; A still beats C on identity).  
- **The main explanation is architectural:** identity-graph GCN is a linear window model; identity-graph T-GCN is a GRU. On the **physical** graph, T-GCN **beats** GCN on both datasets — the recurrent+graph design is working.  
- **GSL is not implicated** by this audit; multi-lag / Mix results are unaffected.  
- **Manuscript implication:** keep NoSpatial as a graph-free control; avoid implying that T-GCN is a weaker backbone in general — it is weaker than linear GCN only when **no graph** is used, and stronger when the physical graph is used.

