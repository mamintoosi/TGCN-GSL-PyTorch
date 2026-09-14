# Stage 46 — Writing Report: Sections 3 and 4

**Date:** 2026-09-10
**Scope:** writing stage — `method.tex` (Section 3) and `setup.tex` (Section 4) for the new manuscript, plus this verification report. No other files created or modified; no experiments run; no numbers changed.

---

## 1. Files written

| File | Content | Approx. words |
|---|---|---|
| `paper/gsl_stage46/sections/method.tex` | Section 3 — Problem Formulation and Graph Learning Framework (§3.1 task/baselines/normalization, §3.2 contemporaneous GSL/cGSL, §3.3 multi-lag, §3.4 consumption mechanisms) | ~1,945 (source incl. LaTeX) |
| `paper/gsl_stage46/sections/setup.tex` | Section 4 — Experimental Setup (§4.1–§4.10: datasets, split, protocol, methods, training, metrics, statistics, controls, 15-min variant, reproducibility) | ~1,082 (source incl. LaTeX) |
| `paper/gsl_stage46/stage46_writing_report.md` | This report | — |

Both files are self-contained section bodies compatible with
`\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}`; they contain no `\input`
of other section files and no table/figure floats.

---

## 2. Protocol verification

Every protocol number inserted into the manuscript, with its source(s).
"Stage 44 §12" = `paper/gsl_stage44/stage44_protocol_reconciliation.md` §12;
"Code" = directly verified from canonical Stage 40 implementation/artifacts
this session.

| # | Item | Value used | Where it appears | Stage 44 §12 source | Additionally verified from code/artifacts |
|---|---|---|---|---|---|
| 1 | Los-loop sensors / interval | N=207, 5-min | §4.1 | Yes | Yes — `stage40_run_all.py` DATASET_CONFIGS; CSV load |
| 2 | SZ-Taxi sensors / interval | N=156, 15-min | §4.1 | Yes | Yes — same |
| 3 | **Los-loop series length** | **2016 timesteps** | §4.1 | **Contradicted** (§12 says 2976) | **Yes — artifact-verified** (see Deviation D1) |
| 4 | SZ-Taxi series length | 2976 timesteps | §4.1 | Yes | Yes — CSV row count 2977 (2976 rows) |
| 5 | Train/test split | 80/20 chronological; Los 1612/404; SZ 2380/596 | §4.2 | Partial (§12 says Los 2380/596 — SZ values) | **Yes** — Stage 26 metadata `T_full=2016, split=1612`; SZ metadata `n_input_rows=2380` |
| 6 | Normalization maxima | Los 70.0, SZ 86.4292 | §4.2 | Yes | Yes — Stage 26 metadata `feat_max=70.0`; Stage 33 SZ metadata `feat_max=86.429…` |
| 7 | Input window | T=12 | §3.1, §4.3 | Yes | Yes — `args.seq_len=12`, `generate_sequences` |
| 8 | Horizons | PH ∈ {1,2,3,4}; Los 5–20 min; SZ 15–60 min | §4.3 | Yes | Yes |
| 9 | **PH target semantics** | **Multi-step vector `[x(t+1),…,x(t+PH)]`, produced jointly by a shared linear output layer; not autoregressive** | §3.1 | §12 says "PH ∈ {1,2,3,4} steps" without target semantics | **Yes** — `generate_sequences`: `Y[i] = data[i+T : i+T+PH]`; `SupervisedForecastTask` regressor `nn.Linear(hidden, pre_len)` |
| 10 | DAGMA variable counts | d=N contemporaneous; d=(L+1)N multi-lag; 828 Los / 624 SZ | §3.1, §3.3 | Yes | Yes — Stage 26 metadata `matrix_shape=[828,828]` |
| 11 | Contemporaneous DAGMA input rows | Los 1612/806/538/403; SZ 2380/1190/794/595 | §3.2 | Partial (§12 gives only the SZ values as "PH1/2/3/4 → 2380/1190/794/595") | **Yes** — SZ: `n_input_rows` in Stage 33 metadata per PH; Los: ceil(1612/PH) consistent with the loader construction |
| 12 | λ₁ contemporaneous | 0.02 Los / 0.01 SZ | §3.2 | Yes | Yes — `stage33_gsl_canonical.py` DATASET_CONFIGS; SZ metadata `lambda1=0.01` |
| 13 | Internal threshold contemporaneous | 0.3 inside `fit()` | §3.2 | Yes | Yes — `model.fit(X, lambda1=…, w_threshold=0.3)` |
| 14 | Support rule | A = 1(\|W\|>0), diagonal removed ≡ \|W\|≥0.3 retained | §3.2 | Yes | Yes — `A = (np.abs(W_est) > 0)` after internal threshold |
| 15 | GSL edge counts | 28 Los / 8 SZ, every PH | §3.2 | Yes | **Yes** — loaded all 8 `A_binary.npy` artifacts: Los [28,28,28,28], SZ [8,8,8,8], diag=0 |
| 16 | cGSL construction & counts | (A+Aᵀ)>0, diag removed; 56 Los / 16 SZ | §3.2 | Yes | Yes — computed from the A_binary artifacts |
| 17 | Multi-lag windows | 1609 Los / 2377 SZ | §3.3 | **Contradicted** (§12 says 2377 for both) | **Yes** — T_train−L: 1612−3 and 2380−3 (see Deviation D1) |
| 18 | λ₁ multi-lag | 0.01 both datasets | §3.3 | Yes | **Yes** — Stage 26 Los metadata `lambda1=0.01` (overrides the stale 0.02 field in `stage40_run_all.py`; consistent with stage40.1 manifest correction) |
| 19 | Internal threshold multi-lag | 0.0 (raw weights) | §3.3 | Yes | Yes — `model.fit(Z, lambda1, w_threshold=0.0)` |
| 20 | Consumer threshold | \|W\|>0.1, diagonal removed | §3.3 | Yes | Yes — `binary_graph(W, 0.1)` |
| 21 | Multi-lag edge counts | Los 12/3/15 = 30 slots, union 28; SZ 0/0/2 = 2 | §3.3 | Yes | **Yes** — recomputed from `los_ph1_seed42_L3_lag_{1,2,3}.npy` and SZ counterparts |
| 22 | Multi-lag PH-independence | One fit shared across PH | §3.3 | Yes | Yes (Stage 44 md5 check; per-PH files are copies) |
| 23 | Contemporaneous PH-dependence | Input subsampled `train_norm[0::PH]` | §3.2, §3.3 | Yes | Yes — `X = train_norm[0::ph]` |
| 24 | Lag assignment rule | graph_idx = (T−1−t) mod 3; most recent step ↔ lag-1 graph | §3.4 | Yes | **Yes** — `MultiGraphTGCNFixed.forward`: `temporal_gap=(T−1)−t; graph_idx = temporal_gap % n_graphs`; lag blocks labeled lag_1…lag_3 by temporal gap |
| 25 | Weighted mechanism | softmax over 3 learned scalars; mixes normalized lag operators | §3.4 | Yes | Yes — `WeightedMultiGraphTGCN` |
| 26 | Mix gate | gate on [x_t; h_{t−1}], softmax over K=3, mixes Laplacians before GRU update | §3.4 | Yes | Yes — `GatedMultiGraphTGCN.forward` |
| 27 | GCN-MultiGSL union | element-wise max/OR of the three lag graphs, single static graph | §3.4 | Yes | Yes — `np.maximum` over `multilag_graphs` |
| 28 | Parameter counts | GCN 768; T-GCN 12,672; Weighted +3; Mix +4,419 | §3.4 | Yes | Yes — hand-derived from class definitions AND stored artifacts `n_params` (12,672 T-GCN; 17,091 Mix) |
| 29 | Capacity-control counts | h=74 NoSpatial 16,872 vs Mix 17,091 | §4.8 | Yes (§7 table) | Yes — same derivation path; single-seed status per Stage 44 §7 |
| 30 | Optimizer/training | Adam, lr 1e-3, wd 1e-4, batch 128, epochs 50, hidden 64 | §4.5 | Yes | Yes — `stage40_run_all.py` defaults + `SupervisedForecastTask` |
| 31 | Losses | T-GCN: sum-of-squares + 1.5e-3·½Σ‖θ‖²; GCN: MSE | §4.5 | Yes ("L2-on-weights regularizer") | **Yes** — `utils/losses.py`: `lamda=1.5e-3`, `mse_loss = Σ(y−ŷ)²/2`, `reg = lamda·Σ‖θ‖²/2` |
| 32 | Seeds | {42,43,44,45,46}; DAGMA deterministic, shared | §4.5 | Yes | Yes — `--seeds` default; DAGMA zero-init determinism (Stage 35 report) |
| 33 | DAGMA iterations | loss l2; warm 30,000; max 60,000; seed 42 | §4.5 | Yes | Yes — both DAGMA scripts |
| 34 | Metrics | RMSE primary, MAE secondary, de-normalized scale, full-batch test eval | §4.6 | Yes | Yes — `validation_epoch` (predictions×feat_max; single test batch) |
| 35 | Statistical policy | 5-seed means, sample std (ddof=1); win counts; paired t; Wilcoxon floor p=0.0625; no definitive significance claims | §4.7 | Yes | Yes — Stage 41 audit (`np.std(ddof=1)`, exact Wilcoxon) |
| 36 | Sparsity controls | Los PH1 only; 30-edge budget; RandTop30 / CorrTop30 / DAGMA fixed / DAGMA Mix; 5 seeds; outside the 12-method matrix; **no sparsified-physical control; no λ/threshold sweep** | §4.8 | Yes | Yes — `stage32_sparse_control.py`, `random_edge_graph`, `correlation_topk_graph` |
| 37 | Capacity control | h=74 check, single-seed, labeled | §4.8 | Yes | Yes |
| 38 | **15-min variant length** | **T=672; 80/20 split** | §4.9 | **Contradicted** (§12 says T=992, 793/199) | **Yes** — Stage 27 metadata `split=537` ⇒ T=672 (see Deviation D1) |
| 39 | 15-min variant settings | triplet averaging; PH=15/30/45/60 min; separate PH-independent DAGMA (λ₁=0.01, \|W\|>0.1); 3 methods; 5 seeds; not a third dataset; PH indices not equivalent | §4.9 | Yes | Yes — `stage29_los15min.py` (`resample_15min`, args defaults); lag edges 22+9+1=32 recomputed from Stage 27 artifacts |
| 40 | Repository | `https://github.com/mamintoosi/TGCN-GSL-PyTorch` | §4.10 | Yes | Yes — submitted manuscript footer |

---

## 3. Implementation checks

Explicitly verified from canonical Stage 40 code this session:

- **PH target semantics.** `generate_sequences` (in `stage40_run_all.py`, identical in `stage33_gsl_canonical.py` and `stage29_los15min.py`) produces `Y[i] = data[i+T : i+T+PH]` — a PH-length multi-step vector per sensor. `SupervisedForecastTask` maps the hidden state through `nn.Linear(hidden_dim, pre_len)` to the full horizon vector at once. Manuscript states: multi-step target, jointly produced, not autoregressive. ✓
- **Graph normalization.** `utils/graph_conv.py::calculate_laplacian_with_self_loop`: `A+I`, row sums → D̃, then `matrix @ D̃^{-1/2} )ᵀ @ D̃^{-1/2}`, i.e. **D̃^{-1/2} Ãᵀ D̃^{-1/2}** (transposed middle factor). Self-loops added; symmetric scaling; the same function is applied to physical, identity, GSL, cGSL, and every lag graph (all classes call it). Manuscript Eq. (1) states exactly this, and notes the identity reduction. ✓
- **Self-loop handling.** Added inside the normalization operator (A+I) for all configurations; learned-graph diagonals are removed at graph-construction time. ✓
- **Graph orientation/indexing.** DAGMA convention W[i,j] = i→j; lag blocks extracted as W[l·N:(l+1)N, L·N:(L+1)N] with `lag_value = L − l_idx` (verified in `stage26_run_dagma.py::extract_lag_blocks` and the stored metadata `block_interpretation`). The contemporaneous block is extracted by the fitting script but never loaded by the Stage 40 runner (`load_multilag_graphs` loads only `lag_1..lag_3`). ✓
- **MultiGSL lag indexing.** `MultiGraphTGCNFixed.forward`: `temporal_gap = (T−1) − t`, `graph_idx = temporal_gap % n_graphs`, with t the loop index over the window (t=T−1 is the most recent step). Therefore most recent step → gap 1 → lag-1 graph; the oldest step (gap 11) → lag-3 graph. Manuscript Eq. (2) states this; the prompt's tentative per-step correspondence ("oldest step → lag-3") holds under the code's convention. ✓
- **Parameter counts.** Derived from the class definitions: GCN `12×64` weights = 768; T-GCN cell `65×128+128 + 65×64+64` = 12,672; Weighted `log_weights` = 3; Mix gate `65×64+64 + 64×3+3` = 4,419 (17,091 total). Cross-checked against stored artifact `n_params` fields (12,672 / 17,091). The torch runtime check could not be executed in this environment (torch not installed in the local shell), so the derivation + artifact cross-check is the verification path; both agree. ✓
- **Training loss.** `utils/losses.py::mse_with_regularizer_loss`: `Σ(y−ŷ)²/2 + 1.5e-3 · Σ‖θ‖²/2` summed over **all** task parameters (backbone + output layer); GCN family uses plain `F.mse_loss`. Manuscript states both with the exact coefficient. ✓
- **Seed handling.** `set_seed(seed)` seeds `random`, `numpy`, `torch` (and CUDA if available) before each training run; DAGMA-linear is deterministic (zero-init) and run once per construction with seed 42, shared across the five training seeds. ✓

---

## 4. Deliberate deviations

**Deviation D1 — dataset lengths corrected against artifacts (Los-loop 2016/1612/404, multi-lag windows 1609, contemporaneous rows 1612/806/538/403, 15-min variant T=672).**
Stage 44 §6/§12 state Los-loop T=2976, split 2380/596, multi-lag windows 2377, and a 15-min variant of T=992 (793/199). Direct artifact inspection proves the Los-loop values are different: `los_speed.csv` has 2,016 data rows (7 days × 288 steps — consistent with "March 1–7, 2012" at 5-min sampling); the Stage 26 Los metadata records `T_full=2016, split=1612`; Stage 27 (15-min) records `split=537` ⇒ T=672. The 2976/2380/596/2377/992 figures are the **SZ-Taxi** values (SZ CSV = 2,976 rows; SZ contemporaneous metadata `n_input_rows=2380`), which Stage 44 §6/§12 appear to have carried over to Los-loop. Per the Stage 46 source rule ("inspect the canonical code/artifacts; record discrepancies"), the manuscript uses the artifact-verified values and this report records the discrepancy. **Note for Stage 47+:** no result number is affected — the 480 Stage 40 runs and both DAGMA constructions consumed exactly these splits; only the descriptive prose in Stage 44 §6/§12 is affected.

**Deviation D2 — "2976 timesteps" retained for SZ-Taxi.** Verified directly (CSV row count; metadata). Matches Stage 44 §12.

**Deviation D3 — normalization equation stated in its implemented (transposed) form.** The prompt's template formula was `D̃^{-1/2} Ã D̃^{-1/2}`, but the implementation applies the scaling to `Ãᵀ`. Per the source rule the manuscript states the implemented operator exactly (Eq. 1) and notes where the two coincide. This also corrects a latent imprecision in Stage 44 §12's inline formula.

**Deviation D4 — the spec's §3.1 pointer "Sect.~\ref{sec:task} below" for normalization.** The normalization paragraph is placed at the end of §3.1 (`sec:task`) rather than in a separate subsection; the internal reference in the graph-free bullet points there. No other structural deviation from Stage 45.1's architecture.

**Deviation D5 — physical-adjacency description kept value-agnostic.** `los_adj.csv` stores fractional weights (~0.1) while `sz_adj.csv` is binary; the manuscript therefore says "dataset-provided road-network adjacency" without asserting binarity of the physical baseline, and the "backbone adjacencies are binary" claim is scoped to the *learned* graphs. (Stage 42's appendices should retain their explicit per-dataset density conventions.)

No other deviations from Stage 45.1 or Stage 44 §12. In particular: the two threshold regimes, the cGSL same-artifact definition, the PH-dependence contrast, the lag-assignment rule, the consumption-mechanism definitions, the statistical-policy paragraph (verbatim), the control scope labels, and the 15-min variant labels all follow Stage 44 §12 exactly.

---

## 5. Citations

| Key | Used for | Resolved? |
|---|---|---|
| `Bello2024DAGMA` | DAGMA method (§3.2, §4.5) | ✓ `MyReferences.bib` line 592 |
| `kipf2017semi` | GCN normalization convention (§3.1) | ✓ line 628 |
| `zhao2019t` | Dataset provenance/source for T-GCN benchmarks (§4.1) | ✓ line 514 |

No other citation keys are used in the two sections. No `\cite` for the repository is used; the URL appears as `\url{...}` (matching the submitted manuscript's footnote convention, kept as plain URL here since section files carry no footnote macros). **Missing keys: none.**

Deliberately **not** cited: NOTEARS/`zheng2018dags` (per instructions — no NOTEARS framing; the log-det/M-matrix characterization is described as DAGMA's own), `Fan2023DAG` (not needed).

---

## 6. Forward references

Only two forward references exist, both to Results/Background, and neither uses
`\ref`:

- §3.4: "the corresponding empirical comparison is presented in Sect.~5" —
textual reference to Results; no `\ref` placeholder was inserted.
- §3.1: "derivations of both architectures are given in Sect.~2" — textual
reference to the Background section; no `\ref` placeholder was inserted.
- All other cross-references (`\ref{sec:task}`, `\ref{sec:gsl_contemp}`,
`\ref{sec:gsl_multilag}`, `\ref{sec:consumption}`, `\ref{sec:setup}`) point
to labels declared within the two Stage 46 files themselves.

**Action item for Stage 47:** when assembling the full manuscript, either
keep the numeric forward references (if Background lands as Section 2 and
Results as Section 5, per the Stage 45.1 TOC) or convert them to
`\ref{sec:background}`/`\ref{sec:results}` once those labels are declared.
No `\ref` placeholders to tables/figures were inserted.

---

## 7. R1/R2 compliance

- **R1 (no over-claim that consumption alone causes the benefit):** ✓ §3.4 states the GCN-union vs T-GCN-per-timestep comparison "is designed to assess whether the utility of the same learned multi-lag structure depends on how it is consumed" and explicitly adds that "because GCN and T-GCN also differ in backbone architecture, the comparison does not isolate consumption as the sole differing factor." The stronger "located in consumption, not the edge set" phrasing is absent from both files.
- **R2 (no ambiguous "+X% improvement"):** ✓ Neither section contains any improvement percentage, comparison result, or performance claim (percentages appear only as hyperparameter values, split ratios, and iteration counts). Any future improvement statement in Results must use "relative RMSE reduction" phrasing.

Additional wording-policy checks: no "adaptive" in any method name or description (the Mix mechanism is described as changing consumption, never the graph); no claim that graphs evolve/change over time (§3.3 states the opposite explicitly); no causal language (both constructions are "statistical dependency structures"; acyclicity is "an optimization and fitting constraint… not causal evidence"); no development-history, version-comparison, or protocol-difference language anywhere.

---

## 8. Open issues for Stage 47

1. **Labels `sec:results` and `sec:background`** must be declared by the corresponding section files (or the two references in §3.1/§3.4 adjusted). Only forward-reference issue carried by Sections 3–4.
2. **Final table/figure labels and ordering** — untouched here per out-of-scope; Sections 3–4 deliberately contain no floats.
3. **Placement of per-seed results and appendix references** (e.g., where the 15-min variant's numbers and the capacity-control table live) — open per Stage 45.1 §9.
4. **Results wording must respect the Stage 46 conventions** established here: "relative RMSE reduction" phrasing; scope labels for the two controls; the 15-min variant never equated with 5-minute PHs; graph-free = NoSpatial named consistently.
5. **Unresolved implementation/provenance questions: none blocking.** The single provenance correction (Deviation D1) is fully resolved by artifact inspection; Stage 44 §6/§12 prose is affected but no stored result is.
6. **Optional Stage 47+ check:** the SZ-Taxi dataset date provenance ("January 2015") is carried from the submitted manuscript; the repo's data files carry no date metadata, so this descriptive fact rests on the original benchmark documentation (as in the submitted version).

---

## 9. Claim and style checklist (final status)

- [x] No history language in either section
- [x] No causal verbs claiming traffic influence
- [x] Learned graphs described as statistical dependency structures
- [x] Static graph fitting clearly distinguished from graph consumption (§3.4 opening principle)
- [x] No claim that the learned graph changes over time
- [x] cGSL defined completely in Section 3 before Results
- [x] A/W notation convention explicit (§3 opening + §3.1)
- [x] Contemporaneous threshold regime (0.3 internal) separated from MultiGSL consumer thresholding (0.1) — enumerated explicitly
- [x] Multi-lag graphs explicitly PH-independent
- [x] Contemporaneous graph learning explicitly PH-dependent through `train_norm[0::PH]`
- [x] Method names match the Stage 40 registry display names exactly (12 canonical configurations; no "Physical" as an extra method name; no "Adaptive")
- [x] Graph normalization verified from the actual implementation (transposed form stated)
- [x] PH target definition verified from the actual data loader (multi-step vector, jointly produced)
- [x] T-GCN MultiGSL lag assignment verified from the actual implementation
- [x] Exact parameter counts verified from code + artifacts before being stated
- [x] Statistical policy paragraph present in Section 4 (verbatim required wording, including the final sentence)
- [x] Five-seed means/std policy stated
- [x] 15-minute material labeled as a temporal-resolution variant only, with the PH-index caveat
- [x] No sparsified-physical control claimed (explicitly negated)
- [x] No lambda/threshold sweep claimed (explicitly negated)
- [x] R1 wording respected
- [x] R2 wording respected
- [x] Citations resolve against `MyReferences.bib`
- [x] Missing citation keys: none
- [x] Forward references are textual (Sect.~2 / Sect.~5), no `\ref` to undeclared labels
- [x] No files outside `paper/gsl_stage46/` were modified

---

## 10. Verdict

**STAGE 46 COMPLETE — READY FOR RESULTS WRITING (STAGE 47).**

Both section files are publication-ready under `sn-jnl`; every implementation-dependent statement was checked against the canonical Stage 40 code or the stored artifacts; the two sections contain no results, no percentages, no historical language, and no causal claims. One provenance discrepancy (Los-loop series length and derived counts) was resolved from artifacts and documented in Deviation D1 for the record.
