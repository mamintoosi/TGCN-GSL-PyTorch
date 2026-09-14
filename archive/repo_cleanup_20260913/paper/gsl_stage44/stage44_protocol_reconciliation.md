# Stage 44 — Protocol and Provenance Reconciliation

**Date:** 2026-09-10
**Scope:** read-only audit. No manuscript edits, no code edits, no experiments, no changes to `previous_revision` or `submitted_version`, no new numerical results.
**Method:** every claim below was verified against actual implementation (`gsl_stage40/scripts/stage40_run_all.py`, `models/multigsl.py`, `models/tgcn.py`, `models/gcn.py`, `gsl_stage26/stage26_run_dagma.py`, `gsl_stage26/stage33_gsl_canonical.py`, `gsl_stage26/stage29_los15min.py`, `gsl_stage26/stage32_sparse_control.py`, `tasks/supervised.py`, `utils/graph_conv.py`, `utils/data/spatiotemporal_csv_data.py`, `configs/tgcn-los-gsl-pre_len1.yaml`), stored artifacts (`results/stage26_validation/`, `results/stage33_gsl_canonical/`, `results/stage29_los15min/`, `results/stage32_sparse_control/`, `results/stage40_canonical/training/`), the Stage 41 audit script/outputs, and the Stage 40.1/40.2/40.3 provenance reports. Manuscript statements were never used to override code evidence.

---

## 1. Executive summary

The Stage 40 canonical pipeline is fully traceable end-to-end, internally consistent, and its protocol can now be stated unambiguously. The decisive findings:

1. **Every result used in the new manuscript is traceable to a verified code path.** All 480 Stage 40 result JSONs exist, are marked `status: "complete"`, match the execution log (480 `[DONE]`, 0 `[FAIL]`, 0 `[SKIP]`), and carry their own `protocol` block (batch 128, lr 0.001, wd 1e-4, hidden 64, seq_len 12, epochs 50, loss per backbone).
2. **The two DAGMA threshold regimes are intentional, documented, and correctly implemented:** multi-lag fits use `w_threshold=0.0` (raw weights saved) with consumer thresholding `|W| > 0.1`; contemporaneous fits use `w_threshold=0.3` inside `fit()` (reproducing the original pipeline's reliance on the library default) with post-fit support `A = 1(|W| > 0)`. The earlier draft's "λ₁=0.02, ω=0.3" vs "λ₁=0.01, τ=0.1" tension resolves cleanly: **λ₁ is dataset-dependent (Los 0.02 / SZ 0.01 for the contemporaneous fits; 0.01 for both datasets' multi-lag fits), and 0.3 vs 0.1 are not competing values of one knob but the two construction-specific thresholds.**
3. **cGSL is a binary symmetrization of the exact same GSL artifact** — `A_cGSL = (A_GSL + A_GSLᵀ > 0)`, diagonal removed — applied identically for GCN and T-GCN. GSL and cGSL therefore share the underlying learned graph by construction.
4. **The multi-lag graphs are PH-independent** (byte-identical npy files across PH1–4, verified by md5), fitted once on the 5-minute training split, and consumed three different ways by T-GCN-MultiGSL (fixed per-timestep assignment `idx=(T−1−t) mod 3`), T-GCN-MultiGSL-Weighted (3 learned global scalars), and T-GCN-MultiGSL-Mix (per-node, per-timestep gate, +4,419 params) — while GCN-MultiGSL consumes the **same files** as a static union (`max` over lag graphs).
5. **NoSpatial is exactly the identity adjacency** passed through the same Laplacian-with-self-loop normalization as every other graph (the normalized Laplacian of I is I), so it is a true graph-free control, not a missing-graph hack.
6. **No blocking provenance issues remain.** The Stage 43 open issues §11.1–11.3 are resolved (protocol reconciliation, control provenance, 15-minute provenance). The 15-minute experiment and the sparse/capacity controls are **canonical-pipeline** experiments (same task, same hyperparameters, 5 seeds) but are *not part of the 480-experiment canonical matrix*; they are usable as main-text evidence with a one-line scope/provenance label, and the "27.4%" figure is legitimately sourced from them (Section 8).
7. **Manuscript-facing corrections identified:** the previous revision's τ=0.1/30-edge description of the GSL baseline is an earlier-pipeline description (canonical: 28 edges, ω=0.3); its λ₁ statements are partially wrong for SZ multi-lag; its SZ win counts predate Stage 40; its "32.8%"/"14.9%"/"21.7%" headline figures are earlier-pipeline values. All are editorial-to-protocol-level issues, none blocking.

**Verdict: READY FOR STAGE 45** (Section 14).

---

## 2. DAGMA threshold/lambda audit

### 2.1 The complete pipeline, as implemented

```
raw CSV (data/los_speed.csv T=2976×207; data/sz_speed.csv T=2976×156)
→ chronological 80/20 split (train = feat[:int(0.8T)])
→ normalization: divide by feat_max = max(TRAIN split) only   [Stage 40/26/29/32/33: identical]
→ DAGMA input construction (two distinct constructions, §2.2)
→ DAGMA-linear fit (DagmaLinear(loss_type="l2"), zero-init, deterministic, seed 42)
→ raw W_est  (multilag: saved raw; contemporaneous: saved AFTER internal threshold)
→ internal DAGMA thresholding (construction-dependent: 0.0 or 0.3)
→ consumer threshold / support rule (construction-dependent)
→ binary adjacency, diagonal zeroed
→ [cGSL only: binary symmetrization (A + Aᵀ > 0), diagonal zeroed]
→ graph normalization: calculate_laplacian_with_self_loop (A + I, D^-1/2 (A+I) D^-1/2, transposed form)
→ GCN / T-GCN backbone
```

### 2.2 The two DAGMA constructions (verified from code)

| Property | **Multi-lag construction** (`stage26_run_dagma.py`) | **Contemporaneous construction** (`stage33_gsl_canonical.py`, incl. Stage 40.3 SZ fits) |
|---|---|---|
| DAGMA input | `Z = [x(t−L), …, x(t−1), x(t)]`, (L+1)·N variables (L=3 → 828 Los / 624 SZ), full training split, all timesteps | `X = train_norm[0::PH]` — every PH-th **simultaneous snapshot** of the training split (contemporaneous rows, no lag blocks) |
| Variables | N per block × 4 blocks | N (single block) |
| λ₁ | **0.01 both datasets** (script default 0.01; metadata confirms) | **0.02 Los-loop, 0.01 SZ-Taxi** (hard-coded as "original protocol value") |
| `w_threshold` inside `fit()` | **0.0** (raw weights preserved) | **0.3** — the original pipeline's *effective* threshold (it called `fit(X, lambda1)` relying on the library default 0.3; committed W_est values all ≥ 0.31 confirm) |
| Post-fit support rule | consumer: `binary_graph(W, 0.1)` = `(|W| > 0.1)`, diagonal zeroed | `A = 1(|W| > 0)` = `1(|W| ≥ 0.3)`, diagonal zeroed — **absolute-magnitude** rule (Stage 36 policy; audited artifacts contain no negative survivor at threshold, max |negative| = 0.013, so sign handling is empirically irrelevant here but explicitly recorded) |
| Binary adjacency produced | at load time by the Stage 40 runner (`load_multilag_graphs`) | at graph-generation time (stored `A_binary.npy`); Stage 40 loads the binary artifact |
| Symmetrization | never | only for cGSL, after the binary adjacency exists |
| Edge budget (result) | Los: 12+3+15 = 30 per-lag sum, 28 union; SZ: 0+0+2 = 2 | Los: 28 edges at every PH; SZ: 8 edges at every PH |
| Seed | 42 (deterministic; no RNG in DAGMA-linear) | 42 (same) |
| Warm/max iterations | 30,000 / 60,000 (library defaults) | 30,000 / 60,000 (same) |
| Loss type | l2 | l2 |
| PH dependence | **None** — one fit, saved per PH under different names (md5-identical across PH1–4, verified directly) | **None in graph, yes in input** — the fit is per-PH because `X = train_norm[0::PH]` differs; PH changes the subsampling stride, hence the input (on this data the resulting support is 28/28/28/28 Los, 8/8/8/8 SZ) |
| Normalization of data | train-split max (70.0 Los / 86.4292 SZ) | same |
| DAGMA sees test data | never | never |

**Key clarification for the manuscript:** "threshold" is two different operations in the two constructions and must never be conflated. There is no single "τ" — the multi-lag graph is thresholded at |W|>0.1 by the *consumer* on raw weights; the contemporaneous graph is thresholded at |W|≥0.3 *inside the fit* on DAGMA's standardized coefficients. Both use the absolute-magnitude rule (sign-symmetric), consistent with the DAGMA library's own semantics.

### 2.3 GSL vs cGSL: same underlying learned graph

Verified in `stage40_run_all.py::run_experiment`:

```python
adj_model = contemporaneous_A + contemporaneous_A.T
adj_model = (adj_model > 0).astype(np.float32)
np.fill_diagonal(adj_model, 0)
```

- cGSL = `(A_GSL + A_GSLᵀ) > 0`, diagonal removed, computed from the **same stored `A_binary.npy`** that GSL consumes. No refit, no different threshold.
- Consequence: |cGSL| directed entries ∈ {28, 56} for Los (56 = full symmetrization, no reciprocal overlap loss) and 16 for SZ (from 8; the Stage 41 graph-statistics section derives 16 SZ / 56 Los).
- Symmetrization happens **after** the binary adjacency exists (absolute-value thresholding is conceptually and temporally separate from symmetrization — Stage 44 check D satisfied). There is no weighted symmetrization and no re-thresholding after symmetrization.

---

## 3. Multi-lag graph audit (Stage 40 protocol)

| Question | Verified answer |
|---|---|
| Number of temporal blocks | 4 (lag_3, lag_2, lag_1, current), L=3 |
| Lag interpretation | `lag_k` = dependency of the **current** step x(t) on x(t−k): block `W[(L−k)·N : (L−k+1)·N, L·N : (L+1)·N]`. "lag 1" = most recent previous step (t−1 → t); "lag 3" = most distant (t−3 → t). The `current` block (x(t)→x(t)) exists in the fit but is **not consumed** by any Stage 40 method |
| DAGMA input dimensionality | (L+1)·N = 828 (Los) / 624 (SZ); matrix 828×828 / 624×624 |
| Exact λ₁ | 0.01 (both datasets) |
| Threshold(s) | fit: 0.0 (raw saved); consumer: \|W\| > 0.1, absolute magnitude, diagonal removed |
| DAGMA fitting parameters | DagmaLinear, loss l2, warm_iter 30,000, max_iter 60,000, seed 42, deterministic |
| Whether refit per PH | **No** — one fit; files copied per PH |
| Whether PH changes DAGMA input | **No** — Z is built from the full training split regardless of PH |
| Graphs identical across PH | **Yes — verified by md5:** `los_ph{1,2,3,4}_seed42_L3_lag_1.npy` all `f1057c1e…` (lag_2, lag_3 analogous) |
| Edges per lag (Los, \|W\|>0.1) | lag_1: 12, lag_2: 3, lag_3: 15 (sum 30) |
| Union graph (Los) | 28 distinct off-diagonal edges (2-edge overlap between lag sets) |
| Edges per lag (SZ) | lag_1: 0, lag_2: 0, lag_3: 2 (sum 2; union 2) |
| Self-loops in DAGMA output | None retained — `binary_graph` zeroes the diagonal; the raw blocks' diagonal mass (e.g. lag_1's self-correlation) is discarded |
| Self-loops added by normalization | Yes — `calculate_laplacian_with_self_loop` adds I before degree normalization, for **every** method including NoSpatial |
| Causal description permitted? | No. The construction is *consistent with* a lag-specific statistical-dependency reading (explicit lag blocks, per-lag consumption demonstrably matters), but no direct validation of the lag interpretation was run; DAGMA's acyclicity constraint is a fitting regularizer on a linear SEM score, not evidence of causal traffic mechanisms |

Note on a manuscript-facing nuance: the earlier draft's Table `tab:lag_stats` (Los: 70/90/5/16 edges incl. self-loops at τ=0.1) describes the same stored artifacts with a *different counting convention* (diagonal included; per-block raw counts). The canonical manuscript should use the Stage 40 consumption convention (diagonal removed): 12/3/15, union 28.

---

## 4. Graph-consumption audit (the four MultiGSL variants)

All four verified in `models/multigsl.py` + `stage40_run_all.py`. All three T-GCN multi-graph variants receive the **identical `adj_list`** (same three thresholded lag graphs); they differ only in consumption.

| Property | T-GCN-MultiGSL (`MultiGraphTGCNFixed`) | T-GCN-MultiGSL-Weighted (`WeightedMultiGraphTGCN`) | T-GCN-MultiGSL-Mix (`GatedMultiGraphTGCN`) | GCN-MultiGSL (GCN + union) |
|---|---|---|---|---|
| Graph matrices | 3 (lag_1, lag_2, lag_3) | 3 | 3 | 1 (union) |
| Mapping timestep → graph | `graph_idx = (T−1−t) mod 3`: most recent input step ↔ lag_1, oldest ↔ lag_3, cycling in between (T=12) | none (single mixed Laplacian used at all timesteps) | none (per-step mixed Laplacian) | none (static) |
| Cyclic mapping? | Yes (period 3 over the 12 input steps) | — | — | — |
| Assigned to specific historical lags? | Yes — corrected alignment: temporal gap g gets lag graph (g mod 3) + 1 | — | — | — |
| GCN unions the lag graphs? | — | — | — | Yes: `A_union = max_l A_l` (element-wise OR), single static graph |
| Global learned graph weights? | No | Yes: `A = Σ_k softmax(w)_k · L_k`, one scalar per lag (3 params) | No | No |
| Node/time-dependent gates? | No | No | Yes: gate MLP `Linear(1+H,H) → ReLU → Linear(H,3)`, softmax over graphs, **per node per timestep**, mixing Laplacians before the GRU update | No |
| Extra trainable params (H=64) | **0** (12,672 total) | **3** (12,675) | **4,419** (17,091) | **0** (768 total GCN) |
| Included in capacity comparisons? | Yes — same 12,672 as T-GCN | Yes (+3, negligible) | Yes — and explicitly controlled by the parameter-matched control (NoSpatial h=74: 16,872 ≈ 17,091) | Yes — 768, identical to all GCN variants |
| "Adaptive" terminology justified? | Method name is **T-GCN-MultiGSL** — no "Adaptive" anywhere in code, registry, or results | same | same | same |

Mechanistic reading (for the manuscript): the Fixed variant's cyclic mapping is *architecturally aligned* (recent steps ↔ recent-lag graphs); Weighted collapses to a learned static mixture; Mix lets each node select its lag-graph mixture at each step. The union variant (GCN-MultiGSL) destroys lag identity entirely. The Stage 40 outcome ordering on Los (Mix 4.49 < Fixed 4.84 ≪ union-via-GCN 9.78) is therefore a consumption-mechanism contrast on identical edge sets (28 of 30 edges in the union).

---

## 5. Contemporaneous single-graph GSL audit (GCN-GSL/cGSL, T-GCN-GSL/cGSL)

| Question | Verified answer |
|---|---|
| DAGMA input | `train_norm[0::PH]` — every PH-th training snapshot, all N sensors, one row = one simultaneous observation vector |
| Training subset | 80% chronological training split only (Los 2380 rows; SZ 2380 rows); PH-specific row counts: PH1 2380, PH2 1190, PH3 794, PH4 595 (per PH) |
| Normalization | train-split max only (70.0 Los / 86.4292 SZ) |
| λ₁ | Los 0.02, SZ 0.01 (original-protocol values, preserved) |
| Threshold | w_threshold=0.3 inside `fit()`; support `A = 1(\|W\| > 0)` ≡ `\|W\| ≥ 0.3` post-fit; diagonal removed |
| Graph density (N(N−1) convention, off-diagonal) | Los: 28 / (207·206) = 6.54e-4; SZ: 8 / (156·155) = 3.31e-4 |
| Directed edges | Los 28 per PH; SZ 8 per PH (verified from stored artifacts, `stage41_summary.json`) |
| Self-loops in learned adjacency | None (diagonal zeroed; DAGMA graphs have no self-loops; graph convolution adds them internally) |
| cGSL construction | `(A_GSL + A_GSLᵀ) > 0`, diagonal removed → Los 56, SZ 16 directed entries |
| Shared between GCN and T-GCN counterparts? | **Yes — exactly the same file** per (dataset, PH): `{los,sz}_gsl_ph{ph}_seed42_A_binary.npy`, loaded by `load_contemporaneous_graph()` for both backbones (Stage 40.2 audit: 125/125 checks) |
| Genuinely contemporaneous? | **Yes.** DAGMA receives rows of simultaneous sensor observations (no lag blocks, no time-shifted variables). It must be described as a **contemporaneous statistical dependency graph**, never as a temporal or causal graph — its later use by a temporal forecasting model does not make the graph temporal. (The multi-lag construction is the explicitly lagged one.) |
| Original-pipeline deviation recorded | The original loop fitted one DAGMA per offset i∈{0..PH−1} and unioned the supports; the committed per-offset supports were identical (SZ: 8 edges at every offset), so the canonical rerun uses the offset-0 fit — documented in `stage33_gsl_canonical.py` |

---

## 6. Data split and normalization audit

| Item | Stage 40 canonical value (verified) |
|---|---|
| Split | chronological 80/20, `train = feat[:int(0.8·T)]`; Los T=2976 → 2380/596; SZ T=2976 → 2380/1596·0.2 → 2380 train rows (T is identical for both datasets) |
| Normalization statistic | global maximum of the **training split** (`feat_max = np.max(feat[:train_size])`) |
| Train-only normalization? | **Yes** — computed before any test access |
| Sequence length | 12 input steps (`seq_len=12`) |
| Prediction horizons | PH ∈ {1,2,3,4}, 5-minute steps on the canonical 5-min data (PH4 = 20 min ahead) |
| Sampling intervals | Los-loop 5 min (canonical); SZ-Taxi 15 min (aggregated); plus the 15-min Los-loop protocol variant (Section 8) |
| Training samples seen by DAGMA (contemporaneous) | PH1 2380, PH2 1190, PH3 794, PH4 595 rows (per PH subsampling) |
| Training samples seen by DAGMA (multi-lag) | T_train − L = 2377 sliding windows (828 variables) |
| DAGMA sees test data | **Never** (both constructions; correlation controls likewise training-only) |
| Sequence generation | `X[i] = data[i:i+12]`, `Y[i] = data[i+12 : i+12+PH]`; windows drawn separately from train and test splits (no window straddles the split) |

**Historical vs canonical protocol differences (verified):** the submitted pipeline (`utils/data/spatiotemporal_csv_data.py`, `configs/*.yaml`) used `feat_max = np.max(self._feat)` over the **full series** (Stage 19 audit flagged the leakage path; numerically identical on both committed datasets since the global max = train max: Los 70.0, SZ 86.4292), batch 64, weight decay 0, GCN hidden 100, seed 42 only, and DAGMA graphs from committed `W_est_*.npy` with non-reproducible generation provenance. Stage 40 uses train-split max, batch 128, wd 1e-4, hidden 64, seeds 42–46, and re-learned graphs with full provenance. **Absolute RMSE values are not comparable across the two protocols** — the Stage 42/43 "contemporaneous re-baselining" framing is the correct integration, and only *relative* effects may be compared.

**Old-manuscript statements inconsistent with Stage 40 (flagged):**
- previous_revision §4.2 "DAGMA hyperparameters: λ₁ = 0.01" — wrong for the Los-loop contemporaneous GSL baseline (0.02) and for Los multi-lag it is right (0.01) but stated without distinguishing the constructions.
- previous_revision §5.3 "λ₁ = 0.02, … ω = 0.3 … 28-edge DAG" — correct for the canonical contemporaneous construction; but the same subsection's earlier draft text elsewhere references τ=0.1/30-edge semantics belonging to the multi-lag construction — the two must not be blended in one protocol description.
- previous_revision `tab:lag_stats` edge counts use a different counting convention (see §3 note).
- submitted-version statements that DAGMA input was temporally lagged — contradicted by the re-derivation (contemporaneous); already corrected in previous_revision.

---

## 7. Sparsity and capacity controls audit

**Provenance: canonical pipeline.** `stage32_sparse_control.py` uses the *identical* task, loss (`mse_with_regularizer`), seed handling, sequence generation, split, normalization, optimizer (Adam lr 1e-3, wd 1e-4), batch 128, epochs 50, hidden 64, full-batch evaluation, and 5 seeds (42–46) as Stage 40; the Stage 41 audit pairs its results with Stage 40's NoSpatial on the same cell. These are canonical-protocol experiments, though **outside the 480-run matrix** and **Los-loop PH1 only**.

| Control | Edge budget | Selection | Directionality | Self-loops | Normalization | Seeds | Model | Params | Type |
|---|---|---|---|---|---|---|---|---|---|
| RandTop30 | exactly 30 directed off-diagonal | uniform random (per-seed redraw, `np.random.RandomState(seed)`) | directed, asymmetric | none | same Laplacian-with-self-loop | 42–46 | TGCN | 12,672 | sparsity floor control |
| CorrTop30 | exactly 30 directed off-diagonal | top-30 \|Pearson\| on **training data only** (`np.corrcoef(train_norm.T)`, diagonal killed) | directed (asymmetric by construction) | none | same | 42–46 | TGCN | 12,672 | sparsity + "heuristic structure" control |
| DAGMA lag blocks (T-GCN-MultiGSL) | 30 (12+3+15) | DAGMA multi-lag fit, \|W\|>0.1 per lag | directed per lag | none | same | 42–46 | MultiGraphTGCNFixed | 12,672 | the learned condition |
| DAGMA + Mix | 30 (same graphs) | same | same | none | same | 42–46 | GatedMultiGraphTGCN | 17,091 | learned + gating |
| Capacity control | identity graph | — | — | — | same | 42 (single-seed, labeled) | TGCN h=64 (12,672) vs h=74 (16,872) vs Mix (17,091) | — | **capacity** control |

Stored results (`results/stage32_sparse_control/stage32_sparse_control.json`, timestamped 2026-09-09): RandTop30 6.10±0.12, CorrTop30 5.39±0.10 (mean±std over 5 seeds; per-seed values in the artifact), against Stage 40 canonical NoSpatial 5.2514±0.1863 on the identical cell.

**What these controls support (and nothing more):** at a matched 30-edge budget on Los-loop PH1, neither random nor correlation-placed sparse graphs beat the no-graph baseline, while the DAGMA-placed edges with per-lag consumption do (5/5 seeds). Conclusion: **sparsity per se does not explain the multi-lag gain at this cell; the specific learned edge placement and its per-lag consumption do.** Not supported: any claim about SZ-Taxi, PH>1, other datasets, other edge budgets, or that the DAGMA placement is optimal. The capacity control supports: the Mix gain is not explained by its +4,419 parameters (h=74 NoSpatial ≈ NoSpatial). The 15-min/other-PH generalization is prohibited.

---

## 8. 15-minute sampling experiment provenance (Stage 29)

| Question | Verified answer |
|---|---|
| Exact dataset | Los-loop 5-minute series resampled to 15 minutes: `reshape(T//3, 3, N).mean(axis=1)` → T=992 (793 train / 199 test) |
| Sampling interval | 15 minutes |
| Prediction horizons | PH 1–4 at 15-minute steps |
| Real-time horizon mapping | PH1 = 15 min, PH2 = 30 min, PH3 = 45 min, PH4 = 60 min ahead — **must not be equated with PH1–4 at 5-minute sampling** (5–20 min ahead) |
| Methods compared | T-GCN-NoSpatial (key `NoGraph`), T-GCN-MultiGSL (`MultiGraphTGCN_fixed`), T-GCN-MultiGSL-Mix (`GatedMultiGraphTGCN`) — the canonical model classes, imported verbatim from `models.multigsl` |
| Number of seeds | 5 (42–46) |
| Exact numerical results (mean±std, computed from the stored JSON) | NoSpatial 8.600±0.249 / 9.311±0.197 / 10.048±0.074 / 10.460±0.196; MultiGSL 7.246±0.298 / 8.571±0.327 / 9.112±0.077 / 9.756±0.210; Mix 6.240±0.187 / 7.322±0.338 / 8.063±0.203 / 8.777±0.326 (PH1–4) |
| Mix vs NoSpatial improvement | +27.4% / +21.4% / +19.8% / +16.1% (PH1–4) — **Mix wins all 5 seeds at all 4 PHs** |
| Same canonical pipeline as Stage 40? | **Yes in protocol** — identical task/loss/optimizer/batch/epochs/hidden/seq_len/normalization/split/5-seed design and the identical model classes; **not part of the 480-run matrix** (separate dataset variant, 3 methods, its own DAGMA fit) |
| Its own DAGMA fit | PH-independent, computed **once** on the 15-min training split (828 variables), λ₁=0.01, w_threshold=0.0, consumer \|W\|>0.1, warm/max 30k/60k, seed 42; 32 cross-sensor edges across the three lags; md5-verified identical across PH |
| Does the 27.4% come from this experiment? | **Yes** — computed from `results/stage29_los15min/stage29_los15min_results.json` (timestamped 2026-09-08), not from another pipeline |
| Legitimate as a headline abstract result? | **Conditionally.** It is a genuine 5-seed canonical-protocol result, but it is a *dataset-variant* experiment (resampled Los-loop), not one of the two canonical datasets, and its DAGMA fit is shared across PH. Recommendation: keep it in the main text as a **temporal-resolution robustness experiment** with the explicit label "15-minute sampling variant; PH denotes 15-minute steps", and keep the abstract's primary headline on the canonical 5-minute Los result (Mix vs NoSpatial +14.2% PH1). If quoted in the abstract, it must carry the "at 15-minute sampling" qualifier in the same sentence. |

**PH convention warning (Stage 44 check):** "PH4 at 15-minute sampling" = 60 minutes ahead; "PH4 at 5-minute sampling" = 20 minutes ahead. No manuscript sentence may mix these without the qualifier.

---

## 9. Stage 40 numerical provenance

| Check | Result |
|---|---|
| All 12 methods × 2 datasets × 4 PH × 5 seeds = 480 records | **480/480 present**, all `status:"complete"`, 0 missing, 0 corrupt, 0 duplicates (Stage 41 audit; reconfirmed by file inventory this stage) |
| Log cross-check | `archive/misc/stage40_run_all.txt`: 480 `[DONE]`, 0 `[FAIL]`, 0 `[SKIP]` — a single clean full run |
| GCN family (GCN, GCN-NoSpatial, GCN-GSL, GCN-cGSL, GCN-MultiGSL) | Traceable: runner variant ids `gcn_*`; GCN-MultiGSL loads the same lag blocks as T-GCN-MultiGSL and unions them |
| T-GCN family (7 methods) | Traceable: runner variant ids `physical`, `no_spatial`, `gsl`, `cgsl`, `multi_gsl`, `multi_gsl_weighted`, `multi_gsl_mix` |
| NoSpatial (both families) | Traceable: `adj_type:"identity"` → `np.eye(N)`; n_edges logged as N (207/156) |
| GSL / cGSL | Traceable: stored `A_binary.npy` per (dataset, PH); cGSL symmetrized in-runner |
| MultiGSL / Weighted / Mix | Traceable: same `load_multilag_graphs()` output; consumption per Section 4 |
| Both datasets, PH1–4, five seeds | Every record carries `dataset`, `ph`, `seed`, `variant`, `protocol` block; naming matches `doc/METHOD_NAMING_MAP.md` |
| Mean/std definitions consistent? | **Yes:** Stage 41 computes `mean` and **sample std (ddof=1)** over the 5 unrounded per-seed RMSE/MAE values; paired tests use the same per-seed vectors (`scipy.stats.ttest_rel`, exact Wilcoxon); improvements use mean-RMSE ratios. Result JSONs store 4-decimal rounded values; the summary CSV carries 6-decimal means. The new manuscript should state: "mean ± sample standard deviation (ddof=1) over 5 training seeds; DAGMA graphs deterministic and shared across seeds" |
| Recomputation needed? | **None.** No ambiguity requiring recomputation was found |

One nuance to disclose in the manuscript methods: the multi-lag graphs are PH-independent by construction (verified by md5), while the contemporaneous graphs are per-PH fits (input subsampling differs); both facts are protocol features, not inconsistencies, but tables listing "graph edges per PH" should say "identical across PH (multi-lag)" vs "refit per PH (contemporaneous; identical supports on this data)".

---

## 10. Old-vs-canonical protocol table

| # | Issue | Evidence | Canonical Stage 40 value/protocol | Previous manuscript value/protocol | Difference | Scientific impact | Required action in new manuscript |
|---|---|---|---|---|---|---|---|
| 1 | λ₁ (contemporaneous) | `stage33_gsl_canonical.py` DATASET_CONFIGS | Los 0.02, SZ 0.01 | prev-rev §5.3 "λ₁=0.02" (Los-only statement); §4.2 "λ₁=0.01" | prev-rev states each value without the dataset split | Editorial-to-protocol | State both: "λ₁=0.02 (Los-loop), 0.01 (SZ-Taxi), original-protocol values" |
| 2 | λ₁ (multi-lag) | `stage26_run_dagma.py` default + metadata | 0.01 both datasets | prev-rev §4.2 "λ₁=0.01" | consistent (once constructions are separated) | None | State per construction |
| 3 | Threshold (contemporaneous) | `fit(..., w_threshold=0.3)` + support `1(\|W\|>0)` | \|W\|≥0.3 inside fit; sign-symmetric support | prev-rev "ω=0.3" (correct); submitted "λ₁ only, threshold unstated" | submitted under-specified | Editorial | Report both fit-internal and support rules explicitly |
| 4 | Threshold (multi-lag) | `binary_graph(W, 0.1)` consumer-side | fit 0.0 (raw), consumer \|W\|>0.1 | prev-rev "τ=0.1" (correct for multi-lag); blended with GSL baseline in one subsection | blended presentation | Editorial | Separate the two constructions in Method; never list "0.3 vs 0.1" as competing choices |
| 5 | DAGMA input (GSL) | `train_norm[0::PH]` | contemporaneous snapshots, per-PH subsampled | prev-rev: correct (contemporaneous); submitted: implied temporal | submitted interpretation contradicted | Interpretation (already corrected in prev-rev) | Keep the contemporaneous-vs-lagged distinction explicit |
| 6 | DAGMA input (multi-lag) | `build_multilag_Z` | Z=[x(t−3),…,x(t)], 828/624 vars | prev-rev correct | none | None | — |
| 7 | Normalization | `load_data()` all canonical scripts | train-split max only | submitted: full-series max (`spatiotemporal_csv_data.py`); prev-rev: train-only | protocol difference; numerically identical on this data (global max = train max) | None on results; protocol hygiene | State train-split max; footnote the historical equivalence |
| 8 | Train/test split | all canonical scripts | chronological 80/20 | same | none | None | — |
| 9 | Temporal interpretation | code + Stage 35/36/40 audits | contemporaneous graph = statistical dependency; multi-lag = explicitly lagged blocks | submitted: "temporal DAG"; prev-rev: corrected | submitted wrong, prev-rev right | Interpretation | Keep prev-rev wording |
| 10 | Graph construction (GSL) | stored A_binary | Los 28 edges/PH, SZ 8/PH | prev-rev `tab:oversmoothing` "Single DAGMA τ=0.3, 6/60 edges" — that row is the earlier-pipeline artifact; `tab:lag_stats` counts differ | protocol/counting differences | Editorial–protocol | Use canonical 28/8 and the diagonal-removed lag counts (12/3/15) |
| 11 | cGSL | runner symmetrization | (A+Aᵀ)>0 binary, same artifact | prev-rev appendix: same formula | none | None | — |
| 12 | Multi-lag graph PH-dependence | md5 check (this stage) | PH-independent (one fit) | prev-rev `tab:los15min` caption implies per-PH graphs for the 15-min variant ("single PH-independent run" — actually correct there); main tables silent | under-specified | Editorial | State PH-independence explicitly |
| 13 | Graph consumption (MultiGSL) | `models/multigsl.py` | fixed cyclic mapping idx=(T−1−t) mod 3; Mix = per-node per-timestep gate | prev-rev "fixed cyclic pattern (corrected alignment)" — correct | none (pending wording check) | None | Verify the Method §3.4 text against Section 4 of this report |
| 14 | GCN-MultiGSL | runner `multilag_union` | union = max over lag graphs, static | absent from prev-rev main text | missing evidence | Structural (Stage 43 F4/T2) | Add per Stage 43 plan |
| 15 | Seeds | runner default [42–46] | 5 training seeds; graphs seed-42 deterministic | submitted: single seed 42; prev-rev: 5 seeds | protocol difference | Variance claims only valid canonically | Canonical numbers only in main text |
| 16 | Model capacity | n_params in every JSON | TGCN 12,672; Weighted +3; Mix +4,419 (17,091); GCN 768 | prev-rev: correct (incl. capacity control) | none | None | — |
| 17 | Batch/wd/epochs | JSON protocol blocks | 128 / 1e-4 / 50 | submitted: 64 / 0 / 50 | protocol difference | part of why absolutes are incomparable | Keep "contemporaneous re-baselining" framing |
| 18 | Sparsity controls | `stage32_sparse_control.py` + JSON | canonical pipeline, Los PH1 only, 30 edges, 5 seeds | prev-rev: same values (6.10/5.39), earlier framing as if parallel to canonical table | scope + pairing nuance | Editorial | Present as canonical-pipeline, Los-PH1-scoped controls vs canonical NoSpatial 5.2514 |
| 19 | 15-minute experiment | `stage29_los15min.py` + JSON | canonical protocol, dataset variant, 3 methods, 5 seeds; 27.4% PH1 | prev-rev §5.6: same numbers | none (provenance now verified) | Editorial | Add provenance/scope label per Section 8 |
| 20 | SZ-Taxi win counts | `stage41_paired_tests.csv` | Mix vs NoSpatial: 4/5, 4/5, 5/5, 4/5 (p=0.064–0.908) | prev-rev: 5/5, 5/5, 5/5, 3/5 | earlier-pipeline values | Result-level (numbers change) | Migrate to canonical values (Stage 43 §4.1 checklist) |

**Classification summary:** 12× NO DIFFERENCE (once the construction split is made), 5× EDITORIAL, 3× PROTOCOL DIFFERENCE (normalization family, batch/wd/hidden family, seed family — all subsumed under the "absolute values not comparable" rule), 2× INTERPRETATION DIFFERENCE (both inherited from the submitted version and already corrected in prev-rev), 0× RESULT-PROVENANCE ISSUE, 0× BLOCKING ISSUE.

---

## 11. Discrepancy classification (per Stage 44 taxonomy)

1. **NO DIFFERENCE** — split, sequence generation, evaluation, cGSL formula, parameter counts, consumption mechanisms, multi-lag construction, PH conventions, graph artifacts, statistical conventions (within Stage 40/41), model classes.
2. **EDITORIAL** — λ₁ presentation (dataset/construction split), threshold presentation (two constructions), lag-edge counting convention, PH-independence statement, control-experiment scope labels, 15-minute variant labels, "Physical" row-label phrasing.
3. **PROTOCOL DIFFERENCE** — submitted-vs-canonical: normalization source (full-series vs train max; numerically identical here), batch size (64 vs 128), weight decay (0 vs 1e-4), GCN hidden (100 vs 64), seeds (1 vs 5), graph provenance (committed unreproducible W_est vs re-learned with full provenance). Consequence already adopted: absolute RMSE not comparable; appendix isolation of historical numbers.
4. **INTERPRETATION DIFFERENCE** — submitted version's temporal/causal reading of the contemporaneous DAG (retired in Stage 42; prev-rev already carries the correction); submitted "learned graphs universally beneficial" (retired).
5. **RESULT-PROVENANCE ISSUE** — none. Every main-text-bound number traces to a verified artifact.
6. **BLOCKING ISSUE** — none. Stage 43's three blocking open issues (§11.1 protocol reconciliation, §11.2 control provenance, §11.3 15-minute provenance) are all resolved by this audit.

---

## 12. Canonical protocol for the new manuscript

> **Canonical Protocol (verified against implementation; use verbatim in Methods)**
>
> **Datasets.** Los-loop: 207 highway sensors, `data/los_speed.csv` (T=2976, 5-minute intervals); SZ-Taxi: 156 urban sensors, `data/sz_speed.csv` (T=2976, 15-minute intervals). Chronological 80/20 train/test split (Los 2380/596 timesteps); no window straddles the split.
>
> **Normalization.** Features divided by the training-split maximum only (Los 70.0, SZ 86.4292); test data never influences normalization or graph learning.
>
> **Forecasting task.** Input window 12 steps; prediction horizons PH ∈ {1,2,3,4} steps of the dataset's sampling interval (Los: 5–20 min; SZ: 15–60 min).
>
> **Graph learning — contemporaneous construction (GSL/cGSL).** DAGMA-linear fitted per PH to `train_norm[0::PH]` (simultaneous per-sensor snapshots subsampled every PH-th training row; PH1/2/3/4 → 2380/1190/794/595 rows), N variables; loss l2; λ₁ = 0.02 (Los) / 0.01 (SZ); w_threshold = 0.3 inside `fit()`; binary support A = 1(|W| > 0) (absolute magnitude; diagonal removed) → 28 edges/PH (Los), 8 edges/PH (SZ); deterministic (zero-init), seed 42. cGSL = (A_GSL + A_GSLᵀ) > 0 with diagonal removed (56 / 16 directed entries). The identical graph artifacts serve both GCN and T-GCN counterparts.
>
> **Graph learning — multi-lag construction (MultiGSL family).** DAGMA-linear fitted once (PH-independent) to the lag-stacked matrix Z = [x(t−3), x(t−2), x(t−1), x(t)] of the training split ((L+1)·N = 828 Los / 624 SZ variables; 2377 windows); loss l2; λ₁ = 0.01 (both datasets); w_threshold = 0.0 (raw weights retained); consumer thresholding |W| > 0.1 (absolute magnitude, diagonal removed) → lag blocks of 12/3/15 edges (Los; union 28 distinct off-diagonal) and 0/0/2 (SZ; union 2). Identical across PH (byte-identical artifacts).
>
> **Graph consumption.** T-GCN-MultiGSL: the three lag graphs consumed separately, one per input timestep, graph index = (T−1−t) mod 3 (most recent step ↔ lag-1 graph; cyclic over the 12-step window); no extra parameters. T-GCN-MultiGSL-Weighted: learned global scalar mixture (softmax over 3 lags) of the graph Laplacians; +3 parameters. T-GCN-MultiGSL-Mix: per-node, per-timestep gate (MLP over [x_t; h_{t−1}], softmax over 3 graphs) mixing Laplacians before the GRU update; +4,419 parameters. GCN-MultiGSL: union graph max_l A_l (element-wise OR) consumed as a single static graph by the standard GCN; no extra parameters. All graphs normalized identically by the symmetric Laplacian with self-loops (A + I, D^{-1/2}(A+I)D^{-1/2}).
>
> **NoSpatial.** Identity adjacency (N×N I) through the same normalization; a true graph-free control.
>
> **Models.** T-GCN family: GRU with per-timestep graph convolution, hidden 64, loss MSE with L2-on-weights regularizer (`mse_with_regularizer`). GCN family: single graph convolution over the whole window, hidden 64, loss MSE. Capacity: T-GCN-family single-graph variants 12,672 parameters (GCN 768); capacity-matched control at hidden 74 confirms the Mix gain is not capacity-driven.
>
> **Training.** Adam, lr 0.001, weight decay 1e-4, batch 128, 50 epochs; seeds 42–46 (5 training seeds); DAGMA graphs deterministic and shared across training seeds.
>
> **Evaluation.** RMSE (primary) and MAE on the de-normalized scale; full-batch test evaluation; mean ± sample standard deviation (ddof=1) over the 5 seeds; paired per-seed win counts; paired t-tests with the explicit caveat that n=5 caps the exact Wilcoxon p at 0.0625 — no definitive significance claims.
>
> **Sparsity/capacity controls (Los-loop PH1 only).** Matched 30-edge budget: RandTop30 (random directed edges, redrawn per seed), CorrTop30 (top-30 |Pearson| from training data only), vs DAGMA lag blocks with fixed and gated consumption; 5 seeds; canonical pipeline. Conclusion scope: this cell only.
>
> **15-minute variant (Los-loop).** 5-minute series averaged over triplets → T=992, 15-minute steps; PH1–4 = 15–60 minutes ahead (never equated with 5-minute PHs); separate PH-independent DAGMA fit on the 15-minute training split (λ₁=0.01, |W|>0.1; 32 cross-sensor lag edges); methods T-GCN-NoSpatial / T-GCN-MultiGSL / T-GCN-MultiGSL-Mix; 5 seeds; canonical training protocol. Labeled throughout as a temporal-resolution variant, not a third dataset.
>
> **Statistical analysis.** Descriptive (means, sample std, per-seed wins); paired t p-values reported for reference with the n=5 caveat; no causal language; learned graphs described as statistical dependency structure; static graphs with only the consumption mechanism operating per timestep.

---

## 13. Remaining blocking issues

**None.** Explicitly resolved from Stage 43's open-issue list:

1. ~~λ₁/τ-ω protocol reconciliation~~ → resolved (Section 2: two constructions, two thresholds, dataset-dependent λ₁).
2. ~~Control-experiment provenance~~ → resolved (Section 7: canonical pipeline, Los-PH1 scope, 5 seeds, stored artifact).
3. ~~15-minute variant provenance and the 27.4% figure~~ → resolved (Section 8: canonical protocol, dataset variant, 5 seeds; usable with qualifier).

Non-blocking items carried forward for the writing stage (from Stage 43 §11.4–11.10, unchanged): Method §3.4 wording check against Section 4 here; T1 table layout; MAE appendix placement; response-letter number sweep; figure regeneration; equivalence targets; `revised_version/` population strategy.

---

## 14. Final readiness verdict

**READY FOR STAGE 45.**

Every number that the revised manuscript will use is traceable to a verified implementation path and a stored artifact; the protocol descriptions that were ambiguous or blended in the previous revision now have a single authoritative formulation (Section 12); the historical protocol differences are fully characterized and confined to the "contemporaneous re-baselining" appendix framing; and no discrepancy requires rerunning anything.

---

## Appendix — Verification actions performed (all read-only)

- Read the full Stage 40 runner, model implementations, task class, graph utility, and data loaders; read the Stage 26/29/32/33 experiment scripts and the historical config YAML.
- Cross-read the Stage 40.1/40.2/40.3 provenance reports and the Stage 41 audit script and outputs.
- Recomputed (presentation-only, from stored artifacts): RandTop30/CorrTop30 means (6.0953/5.3891), Stage 29 per-cell means/stds, Mix-vs-NoSpatial 15-min improvements (27.4/21.4/19.8/16.1%), GCN parameter count (seq_len × hidden = 768).
- md5-verified: multi-lag lag-1 blocks byte-identical across PH1–4 (`los_ph*` → `f1057c1e…`) and across PH in the 15-minute variant (`los15_ph{1,2}` → `97cc43b6…`).
- Confirmed the submitted pipeline's `feat_max = np.max(self._feat)` (full series) vs canonical train-split max, and their numerical equality on both datasets.
- Confirmed SZ contemporaneous edge counts (8 per PH) from `stage41_summary.json` graph statistics.

*End of Stage 44 report. No files other than this report were created or modified.*
