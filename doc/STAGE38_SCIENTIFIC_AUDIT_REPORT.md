# Stage 38 — Scientific Results & Reviewer-Response Mapping Audit

**Date:** 2026-09-09 · **Mode:** analysis-only (no experiments run, manuscript untouched)
**Inputs:** Stage 37 result JSONs, Stage 26 artifacts, B graph files, historical archives,
`paper/Reviewers-comments.txt`, `paper/sn-article.tex` + `paper/sections/*.tex`, experiment
scripts, Stage 34–37 reports, git history (`16ba21b`, `650747c`, `1e84b28`, `61f6434`).

> Note on file location: the deliverable is saved as `doc/STAGE38_SCIENTIFIC_AUDIT_REPORT.md`
> (repo convention for stage reports since Stage 34; the root `Stage38.md` holds the stage
> instructions and was not overwritten). Copy/rename on request.

---

## 1. Executive Summary

- **All three final experiments completed and audited.** Every number below was recomputed
  from the actual Stage 37 JSONs and cross-checked against raw artifacts; nothing is taken
  from prior reports on trust.
- **A closes the sparsity confound** (Reviewer 1, W5): at a matched 30-edge budget,
  random sparse graph = 6.096 (worse than no graph, 5.234 multi-seed), correlation = 5.389,
  while the same number of DAGMA lag-block edges inside MultiGSL = 4.794 and inside the
  gated Mix = 4.452. Sparsity is necessary but not sufficient; the learned structure
  carries the gain.
- **B re-establishes the single-graph GSL baseline with full provenance** and — unexpectedly —
  **reproduces the historical submission's slice-0 DAGMA fits exactly** (PH1/2/4:
  28/28-edge support identity; PH3: 26/28; shared-edge weights within 0.016). The
  historical protocol was decoded from the restored loader: it was a **union of K
  offset-stratified fits** (`data[i::K]`, 28/32/33/39 edges), which B deliberately does
  *not* replicate; B's single-DAG-per-horizon protocol is cleaner and matches the
  manuscript's appendix description. The original headline PH1 relative claim (26.9%)
  is validated at 25.5% under the revised protocol.
- **C upgrades the SZ table to 5 seeds:** Mix beats NoSpatial in 5/5 seeds at PH1–3 but
  the effect is marginal (+0.09% to +0.26% RMSE) and **not stable at PH4** (3/5 seeds,
  +0.09%). The ungated MultiGSL does not help on SZ (−0.27% to +0.05%). Correct framing:
  *marginal, dataset-dependent, gated-variant-only*.
- **The two Stage 37 metadata corrections are verified correct and were reporting-only** —
  no RMSE/MAE was affected; no rerun is needed (see §8).
- **Signed-weight policy** (§7): the `|W|` rule is the library's own threshold semantics;
  empirically neutral on all traffic artifacts (zero negative survivors); should be
  documented as a methodological clarification, not as an effect.
- **No additional experiment is necessary** before manuscript revision. Two *optional*
  cheap additions are identified (§11), one of which (sparsified-physical control) would
  fully close Reviewer 1 W5's "fairer baseline" suggestion.
- **Final verdict: GO for manuscript revision (Stage 39).**

---

## 2. Reviewer-to-Experiment Mapping

Reviewer text quoted/paraphrased from `paper/Reviewers-comments.txt`. Nothing invented;
issues not addressed by A/B/C are marked as such.

| Reviewer issue | Exact scientific concern | Experiment | Evidence/result | What the revised paper should say |
|---|---|---|---|---|
| **R1-W5** | "learned graphs are likely much sparser than the physical… check whether some of the reported gains are attributable to reduced oversmoothing from sparsity rather than the specific learned structure"; suggests "a fairer baseline (e.g., the physical graph's own sparsification)" | **A** (+ Stage 26 C-family) | Matched 30 edges: RandTop30 6.096±0.107, CorrTop30 5.389±0.088, vs NoSpatial 5.234±0.090, MultiGSL 4.794±0.102, Mix 4.452±0.143. Also: same DAGMA edges as a single union adjacency = 5.928 | Sparsity alone does not explain the gains — a random sparse graph is *worse than no graph*; correlation placement is still worse than no graph; only the DAGMA lag-block structure (used per-lag) beats it. Density statistics already in `fig:graph_comparison`. Remaining gap: physical-graph *sparsification* control (see §11) |
| **R1-W6** | "No significance testing, variance, or multiple seeds… report mean ± std over at least 3–5 seeds… given DAGMA itself is deterministic" | **B, C** (+ Stage 26 multiseed) | B: 5 seeds × 4 PHs (GSL 5.792±0.196 … 6.999±0.084; Physical 7.772±0.140 … 8.554±0.169). C: 5 seeds × 4 PHs × 3 methods. Stage 26: Los-loop PH1 5 seeds (Mix 4.452±0.143 vs NoSpatial 5.234±0.090, 14.9%) | All updated/new tables report mean±std over seeds 42–46; variance isolates training randomness; DAGMA determinism verified (Stage 35: zero-init, no RNG; cross-process bit-identity; ~1 ULP across thread counts with identical thresholded support) |
| **R1-W7** | "Results are only reported up to horizon 4… report longer horizons" | **None** | Not covered by A/B/C (all PH≤4). Nearest existing evidence: Stage 29 15-min sampling extends *wall-clock* horizon to 60 min (PH4) with gains growing (27.4% mean) | Either run PH5–8 (optional, §11) or add an explicit scope limitation with the 15-min-sampling result as the long-horizon proxy |
| **R1-W4 / R1-Q4** | "temporal interpretation of the learned DAG is asserted rather than demonstrated… is the DAG learned over simultaneous per-road observations?"; Q4 asks for "direct evidence… feeding genuinely simultaneous vs. lagged data into DAGMA" | **B** (+ Stage 26 multi-lag & lag ablation) | B makes the contemporaneous construction explicit and reproduces the historical contemporaneous fits exactly; the multi-lag formulation (lag-stacked 828-var fit) is the *direct* lagged-data construction; lag ablation shows each lag contributes (~10% individually, 13.3% together) | State clearly: the original single-graph GSL was contemporaneous (now proven by exact reproduction); the temporal interpretation is replaced by the explicit multi-lag formulation, which learns per-lag graphs from explicit temporal blockings `Z=[x(t−L)…x(t)]` |
| **R1-Q3** | time-varying graph extension: "concrete plan… and roughly what computational overhead" | **B (runtimes)** | Measured: DAGMA ~16 min/PH (207 nodes, 4 CPU cores, warm 30k/max 60k iters); multi-lag 828-var fit ~4 h; forecasting reruns are minutes. A sliding-window DAGMA would multiply the fit cost by the number of windows | Provide these measured numbers as the overhead estimate in the response; keep the conclusion-section plan qualitative |
| **R1-W9** | limitations: scalability of DAGMA, sensitivity to λ, only two backbones | **Partial (B, A)** | B gives scalability datapoints (207 nodes: ~16 min/PH; 156-node SZ archived audits: 52–76 min/fit). λ: two values in active use (0.02 single-graph protocol, 0.01 multi-lag) but **no sweep exists** | Add a limitations paragraph; do not imply a sweep was performed. Optional λ-sensitivity run listed in §11 |
| **R1-Q1, R1-Q2, R2-4, R2-5, R2-6** | graph visualization; pred-vs-actual plots; cGSL definition placement; convergence-plot readability; "avergae" typo | **None (editorial/already handled)** | `fig:graph_comparison` and `fig:pred_vs_actual` already in the revised manuscript; the rest are text edits | Handle editorially in Stage 39; not experiment-dependent |
| **R2-1** | "abstract… 21.6% and 24.7%… looks like a mix-up… T-GCN best average as 21.8%" | **B** | The revised abstract no longer quotes those numbers (now 14.9% / 27.4% multi-seed). B's canonical rerun independently gives mean 21.7% (PH1 25.5%) for GSL-vs-Physical — consistent in magnitude with the original 21.8% claim | Confirm the correction in the response letter; the original single-seed 21.8% remains in the appendix as historical |
| **R2-2** | "explicit insights into hidden causal structure… no heatmap… hard to tell if interpretability has been tested" | **Partial (B, A)** | `fig:graph_comparison` (physical vs DAGMA union, degree distributions) exists; B adds provenance-grade graph statistics (28 edges, all positive, weights 0.30–0.78, zero negative survivors); C bounds the claim: SZ marginality is itself interpretive evidence (the learned structure only helps where functional dependencies dominate proximity) | Temper causal language; present the visualizations and graph statistics as *descriptive* structure analysis, not causal validation |
| **R2-3** | static-graph statement vs "adapt to changing traffic patterns" inconsistency | **None (editorial)** | — | Rephrase item 3 of §3.2 as suggested by the reviewer (graph fixed at training time; GRU models temporal dynamics; learned structure captures real dependencies unlike fixed proximity) |

---

## 3. Experiment A — Scientific Interpretation

**Protocol match — verified in code, not assumed:**
- `train_and_eval` in `stage32_sparse_control.py` is byte-for-byte the canonical path:
  `SupervisedForecastTask(loss="mse_with_regularizer")`, `set_seed(seed)`, Adam(lr 1e-3,
  wd 1e-4), batch 128, 50 epochs, full-batch test evaluation, `feat_max` from train split.
  Same data pipeline (`generate_sequences`, PH=1) as `stage26_validation.py`.
- Budget match: script asserts `n_sum == args.n_edges` — the three lag graphs must sum to
  exactly 30 (verified against blocks: lag_1=12, lag_2=3, lag_3=15; union 28).
- `CorrTop30` = top-30 |Pearson| **directed** edges from **training data only**
  (`models/multigsl.py:303`, diagonal removed) — no test leakage.
- `RandTop30` = deterministic per seed (`np.random.RandomState(seed)`), off-diagonal.
- Seeds {42,43,44,45,46}; n_edges=30; both methods 12,672 params (= NoSpatial TGCN).
- Std convention: population (ddof=0) — **same as the manuscript** (Stage 26 multiseed
  5.234±0.090 reproduces `tab:multiseed` exactly; A's log prints the same convention).

**Results (RMSE, mean±std, n=5):**

| Method | Edges | RMSE | MAE | vs NoSpatial (multiseed mean 5.234) |
|---|---|---|---|---|
| RandTop30 | 30 | 6.096 ± 0.107 | 3.756 ± 0.150 | **−16.5%** (worse) |
| CorrTop30 | 30 | 5.389 ± 0.088 | 3.247 ± 0.056 | −3.0% (worse) |
| T-GCN-NoSpatial (Stage 26, ref) | 207 (identity) | 5.234 ± 0.090 | — | — |
| T-GCN-MultiGSL (Stage 26, ref) | 30 | 4.794 ± 0.102 | — | +8.4% |
| T-GCN-MultiGSL-Mix (Stage 26, ref) | 30 | 4.452 ± 0.143 | — | **+14.9%** |

**What A establishes, decomposed by the five requested factors:**

1. **Sparsity effect:** sparsity per se is not beneficial — a random 30-edge graph
   *degrades* RMSE by 16.5% relative to no graph. Sparsity is at best a necessary
   condition (dense graphs hurt — `tab:oversmoothing`), never a sufficient one.
2. **Graph topology / edge placement:** correlation-guided placement recovers only part
   of the gap and remains worse than no graph (−3.0%). Placement quality matters, but
   marginal-signal topology (highest single-lag correlation) is not enough.
3. **DAGMA-derived structure:** the *only* 30-edge graph that beats NoSpatial is the
   DAGMA lag-block edge set. Two independent corroborations from Stage 26: (a) the same
   DAGMA edges assembled into a *single* union adjacency with plain T-GCN
   (`UnionGraph_thr0.1`, 28 edges) give 5.928 — worse than NoSpatial; (b) single-DAG
   DAGMA graphs thresholded from the multi-lag fit (`SingleDAG_thr0.1`, 60 edges) give
   6.057. So neither the DAGMA edges alone in one graph, nor a thresholded single DAG,
   explains the gain either.
4. **MultiGSL mechanism:** the jump from 5.928 (same edges, single adjacency) to 4.794
   (same edges, per-lag multi-graph) isolates the multi-graph mechanism as the primary
   source of the gain. A itself cannot show this without the Stage 26 union control —
   the paper should cite both together.
5. **Mix/gating effect:** 4.794 → 4.452 (gating adds a further 6.7% relative to
   MultiGSL; total 14.9% vs NoSpatial), consistent with the parameter-matched control
   (`tab:param_control`: +35% params without gating → only 0.1% better).

**What A does NOT establish (do not overclaim):**
- It does **not** sparsify the *physical* graph — R1-W5's suggested "fairer baseline"
  (sparsified road-graph adjacency) was **not** run (see §11).
- It is PH=1, Los-loop only; no SZ controls.
- It does not test λ sensitivity of the DAGMA graphs.
- No significance tests were performed (report win counts and overlapping/not-overlapping
  ±1σ only). Here: RandTop30 vs NoSpatial bands are clearly separated; CorrTop30's
  advantage over NoSpatial is within ~1σ per-seed and should be described as "no better
  than no graph", not as significantly worse.

**Safe conclusion for the paper (suggested wording):**
> "At a matched 30-edge budget, neither a random sparse graph (RMSE 6.10±0.11) nor a
> top-correlation graph (5.39±0.09) matches T-GCN-NoSpatial (5.23±0.09), whereas the
> same number of DAGMA lag-block edges yields 4.79±0.10 (MultiGSL) and 4.45±0.14
> (MultiGSL-Mix). The benefit of sparse learned structure therefore arises from the
> *combination* of DAGMA-learned lag-specific edges and their per-lag use in the
> multi-graph architecture, not from sparsity alone."

---

## 4. Experiment B — Scientific Interpretation

### 4.1 The two protocols, stated precisely

**Historical submitted protocol** (decoded from the restored loader,
`utils/data/spatiotemporal_csv_data.py` — this is code, not inference):
```python
for i in range(pre_len):                      # i = 0..K-1
    X = data[i::K]                            # offset-i, stride-K subsample of train windows
    W_est_all[:, :, i] = DagmaLinear(loss_type='l2').fit(X, lambda1=0.02)   # library-default w_threshold=0.3
W_est = np.any(W_est_all > 0, axis=2)         # graph = UNION of the K fits' supports
adj[W_est > 0] = 1                            # (GSL mode; +adj.T for cGSL; +physical for GSL+Adj)
```
→ Los-loop as-used graphs: **28 / 32 / 33 / 39 edges for K = 1 / 2 / 3 / 4**.

**Canonical revised B protocol** (`stage33_gsl_canonical.py`):
```python
X = train_norm[0::PH]                                 # one subsample, no offsets
W = DagmaLinear(loss_type='l2').fit(X, lambda1=0.02, w_threshold=0.3,
                                    warm_iter=30000, max_iter=60000)
A = (np.abs(W) > 0); np.fill_diagonal(A, 0)           # Stage 36 absolute-magnitude rule
```
→ **28 edges for every PH**; the graph is horizon-independent by construction.

### 4.2 Reproduction levels — explicitly graded

| Level | Status |
|---|---|
| **Exact reproduction of graph support** | ✅ for slice-0 at PH1, PH2, PH4: 28/28 edges, Jaccard 1.000 (fresh PH1 ≡ `pre_len1[s0]` ≡ `pre_len2[s0]` ≡ `pre_len4[s0]`; PH2 ≡ `pre_len2[s0]`/`pre_len4[s0]`; PH4 ≡ `pre_len4[s0]`). PH3: 26/28 (Jaccard 0.897) |
| **Approximate reproduction of weights** | ✅ on shared support: max \|ΔW\| = 0.0096/0.0101/0.0137/0.0164 (PH1→4), mean ≈ 0.003; signs agree 28/28. Not bit-identical (float64→float32 path and two years of library/environment drift) |
| **Reproduction of the original headline PH1 result** | ✅ *in relative terms*: original single-seed 6.588→4.818 (−26.9%); revised 5-seed 7.772±0.140→5.792±0.196 (−25.5%), non-overlapping ±1σ. Absolute RMSEs differ because the training pipeline itself was revised (the appendix already flags this: "not directly comparable in RMSE magnitude") |
| **Reproduction of the full historical multi-PH protocol** | ❌ **not replicated, by design.** B fits only offset 0; the historical graph unioned K offset fits (32/33/39 edges at PH2–4). Every fresh edge lies inside the historical union (strict subset), but B's graphs are 4–11 edges sparser. B's protocol is a *refinement*: a genuine single DAG per horizon (the union of K DAGs need not be acyclic), matching the manuscript's own description ("a single N×N weight matrix W") |

PH3 detail (the only non-exact case): the two fresh-only edges have |W| = 0.398 and 0.505
at cells where the historical fit instead placed a 0.362 edge in the **transposed
direction** — the same sensor neighborhood, resolved differently by the optimizer path.
This is an optimizer-path-level divergence, not a threshold artifact (values exceed 0.3
comfortably on both sides).

### 4.3 Forecasting results (5 seeds, revised pipeline)

| PH | Physical (2833 edges) | T-GCN-GSL (28 edges) | Improvement |
|----|------|------|------|
| 1 | 7.772 ± 0.140 | 5.792 ± 0.196 | 25.5% |
| 2 | 8.118 ± 0.189 | 6.328 ± 0.166 | 22.0% |
| 3 | 8.456 ± 0.047 | 6.669 ± 0.080 | 21.1% |
| 4 | 8.554 ± 0.169 | 6.999 ± 0.084 | 18.2% |

Mean 21.7% (original claim: 21.8%). Protocol-integrity check: B's Physical row at seed 42
(7.6582) is **bit-identical** to the manuscript's `tab:multiph`/`tab:oversmoothing`
Physical values (7.658) — B's training path is the canonical one, so its GSL rows slot
into the same table legitimately.

**Suggested paper wording (conservative, graded):**
> "The fresh protocol fit reproduces the historical submission's first-offset DAGMA
> support exactly at PH = 1, 2, and 4 (28/28 edges) and 26/28 at PH = 3, with shared-edge
> weights agreeing to within 0.016. The historical pipeline, recovered from the original
> code, additionally unioned K offset-stratified fits (28/32/33/39 edges); the revised
> protocol instead uses a single DAG per horizon, which the historical graph strictly
> contains. The original PH=1 finding — a sparse learned DAG substantially outperforming
> the dense physical graph — is reproduced at five seeds under the revised pipeline
> (25.5% vs the originally reported 26.9%)."

---

## 5. Experiment C — Scientific Interpretation

All values recomputed from `stage33_sz_multiseed_results.json` (5 seeds × 4 PHs × 3 methods).

| PH | NoSpatial | MultiGSL | Mix | Mix vs NoSpatial (from means) | MultiGSL vs NoSpatial | Mix wins (paired) |
|----|------|------|------|------|------|------|
| 1 | 4.1192 ± 0.0069 | 4.1302 ± 0.0152 | 4.1091 ± 0.0052 | **+0.25%** | −0.27% | 5/5 |
| 2 | 4.1623 ± 0.0055 | 4.1600 ± 0.0036 | 4.1515 ± 0.0026 | **+0.26%** | +0.05% | 5/5 |
| 3 | 4.1884 ± 0.0011 | 4.1988 ± 0.0121 | 4.1804 ± 0.0052 | **+0.19%** | −0.25% | 5/5 |
| 4 | 4.2196 ± 0.0013 | 4.2273 ± 0.0092 | 4.2159 ± 0.0073 | +0.09% | −0.18% | **3/5** |

- **Does MultiGSL itself help on SZ?** No. The ungated 2-edge lag-graph variant is at or
  below NoSpatial at 3 of 4 PHs (−0.27% to +0.05%). On SZ, the lag graphs (2 edges) carry
  too little structure to help on their own.
- **Does MultiGSL-Mix help?** Yes, marginally and consistently at PH1–3 (5/5 paired seeds,
  +0.19–0.26%), but the effect is two orders of magnitude smaller than on Los-loop
  (14.9%).
- **PH4: unstable.** 3/5 paired seeds, +0.09% mean, per-seed deltas straddling zero
  (−0.0007…+0.0124). The correct description is "not stable / within noise", not
  "dips" and not "stable".
- **No significance tests were performed** (no paired t-test/Wilcoxon in any JSON). The
  paper must report win counts, means, and stds only — no p-values, no "significant"
  language. (If desired later, a paired test on 5 seeds is possible but underpowered;
  not recommended as a headline claim.)
- **Correct headline:** "marginal and dataset-dependent; consistent direction for the
  gated variant at PH1–3; no effect for the ungated variant; PH4 within noise." This
  *strengthens* the manuscript's dataset-dependence section (an honest boundary result
  with 5-seed evidence is more credible than a single-seed 0.00%).

---

## 6. DAGMA Product Reconciliation (authoritative table)

Purpose: prevent any future accidental mixing. Paths, constructions, and roles — all
verified from artifact bytes and generating code.

| | **1. Stage 26 multi-lag blocks** | **2. B fresh single-graph fits** | **3. Archived audit fits** | **4. Historical W_est stacks** |
|---|---|---|---|---|
| Location | `results/stage26_validation/{los,sz}_ph{1..4}_seed42_L3_*.npy` (+ stage27/29 variants) | `results/stage33_gsl_canonical/{los}_gsl_ph{1..4}_seed42_{W_est,A_binary}.npy` | `archive/revision_stages/results/dagma_fresh/sz_PH{1..4}_W.npy` (+ threshold_audit, stage20.5/24/25 exploratory) | `archive/historical_submission/W_est_{losloop,shenzhen}_pre_len{1..4}.npy` |
| Optimization problem | Multi-lag joint fit, **828 vars** (207×4 blocks: current, lag_1..3) | Contemporaneous single DAG, **207 vars** | Contemporaneous single DAG, 156 vars (SZ) | Contemporaneous single DAG per offset, 207 (Los) / 156 (SZ) vars, **K fits stacked** |
| Input construction | every consecutive row, lag-stacked windows `Z=[x(t−L)…x(t)]` | `train_norm[0::PH]` | SZ subsampled rows | `data[i::K]` per slice i (loader code) |
| lambda1 | 0.01 | 0.02 | — | 0.02 Los / 0.01 SZ (loader code) |
| Threshold at fit | 0.0 → **raw W saved** | **0.3 inside fit()** | 0.0 → raw W saved | 0.3 inside fit() (confirmed by exact reproduction) |
| Sign rule at use | `abs(W) > 0.1` (consumer) | `abs(W) > 0` post-threshold (Stage 36) | `abs(W) > τ` post-hoc | `W > 0` on library-thresholded W (positive-only, historical) |
| Graph construction | per-lag binary graphs; diagonal removed | single binary DAG; diagonal removed | post-hoc thresholded | `np.any(W_est_all>0, axis=2)` = **union of K fits** |
| Number of graphs | 4 blocks + W_full per PH (×2 datasets) | 1 per PH (×4) | 1 per PH (SZ) | K stacked per file (8 files) |
| Edge count | Los: 12/3/15 (+70 current) per PH1; SZ: 0/0/2 (+11 current) | **28 every PH** | raw (unthresholded) | **28/32/33/39** (K=1..4, Los, as-used union); slice-0 = 28/28/27/28 |
| Intended method | **T-GCN-MultiGSL / -Mix** (Experiments A, C) | **T-GCN-GSL / GCN-GSL baseline** (Experiment B) | audits only | original submission only |
| Status | **active** (do not regenerate; deterministic) | **active, canonical** | **archival** — never wire into experiments | **archival reference only** — reproduced by B; must not silently enter new experiments |

Mixing hazards flagged: (2) must never be substituted into MultiGSL (different problem);
(4) must never re-enter `data/` (Stage 36 moved it); (3) is unthresholded raw W — using
it without applying the threshold protocol would reintroduce the Stage 34 bug class.

---

## 7. Signed-Weight Policy Audit

1. **Is `|W|` thresholding the correct interpretation of DAGMA's `w_threshold`?**
   **Yes — it is the library's own semantics.** Verified from the installed source
   (Stage 35/36): `linear.py` applies `W_est[np.abs(W_est) < w_threshold] = 0` post-hoc
   inside `fit()`. The library therefore always thresholded by absolute magnitude; the
   historical `A = 1(W>0)` was an *additional projection* on top of it.
2. **Is retaining both signs a defensible support policy?** Yes. DAGMA's W is a signed
   linear-coefficient matrix; the acyclicity constraint acts on the support, not the
   sign. Dropping negative survivors would silently impose an excitatory-only assumption
   on traffic dependencies without justification. The Stage 36 rule (`1(|W|>0)` after the
   library threshold, diagonal removed) is sign-symmetric and faithful to the library.
3. **Did it change any Stage 37 number?** **No — measured, zero effect.** Negative
   coefficients at/above threshold: **0** in all 24 Stage 26 lag blocks, all 4 fresh B
   fits (`n_negative_surviving: 0` in provenance), and all 8 historical stacks (max
   \|negative\| = 0.013 << 0.3). Old and new rules produce bit-identical supports on
   every artifact examined.
4. **Should the manuscript discuss it?** Yes, briefly — one methodological sentence and
   one reproducibility sentence. It preempts a code-inspection question and documents the
   generalization honestly.
5. **Conservative framing (suitable for reviewers):**
   > "The graph-support rule was generalized to absolute magnitude, matching the
   > threshold semantics of the DAGMA library itself: an edge is kept if the estimated
   > coefficient survives the sparsity threshold, regardless of sign. This is a
   > methodological clarification; on the datasets studied, no negative coefficient
   > survived the threshold, so the learned graphs and all reported results are
   > numerically identical to those of the original positive-only rule."

   Do **not** state or imply negative coefficients were observed or mattered.

---

## 8. Stage 37 Metadata Corrections — verified

**Correction 1 (C's edge counts).** Code confirms MultiGSL's input graphs are built from
**lag blocks only**: `lag_keys = sorted(k for k in lag_blocks if k.startswith("lag_"))`,
`adj_list = [binary_graph(lag_blocks[k], threshold) for k in lag_keys]` — the
contemporaneous `current` block is never loaded into the model (same as Stage 26).
Actual per-PH lag edges: **[0, 0, 2] → 2 edges/PH** (verified from block bytes and
`run.log` "Lag graphs: [0, 0, 2]"; consistent with every model row `n_edges: 2`).
The previous provenance value (5) had counted the `current` block — wrong.

**Correction 2 (B's physical edge count).** `los_adj.csv` has **2833 positive entries**
(weight sum 1307.158, weights ≈ 0.1 each). `int(adj.sum())` reported the weight sum
(1307) as an edge count; `int((adj > 0).sum()) = 2833` is the correct definition and
matches the manuscript (`fig:graph_comparison`: "2833 edges"). All 20 physical rows in
the JSON were corrected 1307 → 2833; GSL rows (28) were always correct.

**Did either error affect RMSE/MAE? No.** Both defects lived in post-hoc *reporting*
code paths: B's `n_edges` was computed from the adjacency *after* training, and training
received the full weighted matrix unchanged; C's provenance numbers were written in the
JSON-dump section, after all trainings, from a hardcoded literal — the model used the
real 2-edge graphs built earlier from the blocks. With fixed forecasting seeds and
deterministic DAGMA (Stage 35), a rerun would reproduce identical RMSE/MAE.
**Classification: provenance/reporting defects only; no rerun required.** Scripts fixed
at commit `61f6434`; local result JSONs corrected in place (gitignored).

---

## 9. Old → New Results Map

"Replace" = use in the revised main text under the canonical protocol. Historical values
are retained in the appendix, never silently overwritten.

| Manuscript location / claim | Original submitted value | New canonical value | Source | Should replace? | Reason |
|---|---|---|---|---|---|
| GSL PH1 (TGCN-GSL) | 4.818 (single seed, union-of-offsets graph, old pipeline) | **5.792 ± 0.196** (5 seeds, single-DAG protocol) | B | Yes — as the *revised-protocol* baseline row, labeled as such | Protocol-matched to the revised pipeline; multi-seed; graph reproduced from historical |
| GSL PH2 | 5.400 | **6.328 ± 0.166** | B | Yes (same caveat) | as above |
| GSL PH3 | 5.846 | **6.669 ± 0.080** | B | Yes (same caveat) | as above |
| GSL PH4 | 6.257 | **6.999 ± 0.084** | B | Yes (same caveat) | as above |
| GSL-vs-Physical mean improvement | 21.8% (single seed) | **21.7%** (5 seeds; PH1 25.5% vs 26.9%) | B | Yes — with reproduction note | Validates the original claim's magnitude under the revised protocol |
| Physical baseline (Los, PH1–4) | 6.588 / 6.960 / 7.361 / 7.568 (old pipeline) | 7.658**2** / 8.002 / 8.512 / 8.540 (seed 42, canonical; B reproduces seed-42 exactly) — 5-seed: 7.772±0.140 … 8.554±0.169 | Stage 26 / B | Main text keeps canonical seed-42 rows; use 5-seed values where multiseed framing is used | Two protocols are not comparable in absolute RMSE; appendix already states this |
| T-GCN-NoSpatial | not in original submission | 5.143 (PH1, seed 42); 5.234 ± 0.090 (5 seeds) | Stage 26 | Already in revised text | New control introduced by the revision |
| T-GCN-MultiGSL | not in original | 4.715 (seed 42); 4.794 ± 0.102 (5 seeds) | Stage 26 | Already in revised text | Proposed method |
| T-GCN-MultiGSL-Mix | not in original | 4.458 (seed 42); 4.452 ± 0.143 (5 seeds), 14.9% | Stage 26 | Already in revised text (`tab:multiseed` matches to the digit) | Proposed method |
| CorrTop30 | not in original | **5.389 ± 0.088** | A | **Add** to `tab:oversmoothing` | Closes R1-W5 sparsity confound |
| RandTop30 | not in original | **6.096 ± 0.107** | A | **Add** to `tab:oversmoothing` | Closes R1-W5 sparsity confound |
| SZ-Taxi (Mix, PH1–4) | 4.108 / 4.149 / 4.184 / 4.221 (seed 42) | **4.1091 ± 0.0052 / 4.1515 ± 0.0026 / 4.1804 ± 0.0052 / 4.2159 ± 0.0073** (5 seeds) | C | **Replace** in `tab:sz_multiph` (add MultiGSL row: 4.1302/4.1600/4.1988/4.2273) | R1-W6; PH4 re-characterized as not stable |
| SZ NoSpatial | 4.116 / 4.160 / 4.189 / 4.221 (seed 42) | **4.1192 ± 0.0069 / 4.1623 ± 0.0055 / 4.1884 ± 0.0011 / 4.2196 ± 0.0013** | C | **Replace** | multi-seed |
| Graph edge counts | never stated numerically in original text (rule only) | Physical **2833** (Los; verified), fresh GSL **28/PH**, historical as-used **28/32/33/39**, MultiGSL input **30** (Los) / **2** (SZ) | B, A, C, loader | Annotate appendix with historical counts; keep 2833 (already correct in fig) | Prevents the 1307-class error; documents the union protocol |

---

## 10. Main Text / Appendix / Historical Classification

**A. Main-text canonical results** (revised pipeline, current protocol):
- Stage 26 Los-loop results: multiseed (5.234/4.794/4.452 ± stds), `tab:multiph`
  (seed 42), lag ablation, parameter control, 15-min resolution table.
- **A:** CorrTop30, RandTop30 (new `tab:oversmoothing` rows).
- **B:** Physical + T-GCN-GSL rows, 5 seeds × 4 PHs, with the §4.3 reproduction note.
- **C:** SZ 5-seed table (NoSpatial / MultiGSL / Mix).

**B. Appendix results** (retain, clearly labeled):
- Original GSL/cGSL appendix (single-seed, old pipeline) — retain as historical context
  and for the GCN/cGSL asymmetry finding; annotate the union-of-offsets protocol and the
  28/32/33/39 edge counts now known.
- 15-min-resolution results; GCN variants; Stage 29 outputs.

**C. Historical submission results** (archival reference only):
- `archive/historical_submission/` W_est stacks (8 files) — reproduced by B; never load
  into new experiments.
- Original appendix numbers (GCN 21.6%-class, TGCN 24.7%-class claims as originally
  worded; the R2-1 mix-up).

**D. Obsolete / no longer cite:**
- The abstract's original "21.6% / 24.7%" framing (already removed from the revised
  abstract; confirm absence in response letter).
- The "1307 edges" physical count and the "5 edges/PH" SZ provenance (fixed at `61f6434`;
  never appeared in the manuscript).
- Any future citation of `results/stage33_*` JSONs should use the corrected local copies.
- Historical TGCN-GSL absolute RMSEs (4.818 etc.) as *current-protocol* numbers — they
  are appendix-only, different pipeline.

Nothing is deleted.

---

## 11. Need for Additional Experiments

**Verdict: no additional experiment is necessary** before manuscript revision. A/B/C +
existing Stage 26–29 evidence address R1-W4/W5/W6, R2-1/2 substantively, and the
remaining concerns are editorial or limitations-paragraph material.

**Optional (cheap, high reviewer value) — recommended, not required:**
1. **Sparsified-physical control** (directly requested by R1-W5's "physical graph's own
   sparsification"): threshold/kNN-sparsify `los_adj.csv` to exactly 30 edges, run the
   canonical T-GCN path, 5 seeds ≈ **5 min GPU**. A/B/C do not cover it: A's controls are
   correlation/random, not sparsified-physical. Outcome either way strengthens the
   oversmoothing section (if sparsified-physical still loses to NoSpatial, density is
   *entirely* ruled out as the physical graph's problem; if it helps, the paper must
   credit sparsity for part of the physical-vs-learned gap and the MultiGSL comparison
   still stands via the union-control argument).
2. **Multi-seed `tab:multiph` Los-loop rows** for MultiGSL/Mix (PH2–4): upgrades the
   seed-42 main table to 5-seed mean±std, ~30–60 min GPU. Addresses R1-W6 exhaustively.
3. **Longer horizons (PH5–8)** — the only *expensive* option (new DAGMA fits per PH +
   trainings). Recommend **against** running it now: present the 15-min-sampling result
   (up to 60 min ahead, gains grow to 27.4%) as the long-horizon proxy and add a scope
   limitation. This honestly answers R1-W7 without a multi-hour computation.
4. **λ-sensitivity mini-sweep** (e.g., λ ∈ {0.01, 0.02, 0.05} for the multi-lag fit on
   one PH): would substantiate the R1-W9 limitations paragraph. ~4–12 h DAGMA; defer.

---

## 12. Manuscript Integration Plan (no edits made yet)

| # | Location | Change | Scientific purpose |
|---|---|---|---|
| 1 | `tab:oversmoothing` + surrounding text (§5.1) | Add CorrTop30 (5.389±0.088) and RandTop30 (6.096±0.107) rows; add the union-control sentence (5.928) | Close the sparsity confound (R1-W5); support §3-Item-3/4 decomposition |
| 2 | §5.1 text | Insert the §3 "safe conclusion" sentence | Precise, non-overclaimed statement of what sparsity does/doesn't explain |
| 3 | New short subsection (or extension of §5.2) "The single-graph GSL baseline revisited" | B's 5-seed Physical/GSL table + reproduction paragraph (§4.3 wording) | Re-establish the original baseline with provenance (R2-1, R2-2; ties appendix to main text) |
| 4 | `tab:sz_multiph` + §5.7 text | Replace seed-42 rows with C's 5-seed mean±std; add MultiGSL row; rewrite PH4 sentence as "within noise / not stable" | R1-W6; honest dataset-dependence boundary |
| 5 | `tab:multiph` (optional) | Add T-GCN-GSL row (5.548 seed 42 / 5.792±0.196 5-seed) alongside Physical | Places the reproduced baseline next to the proposed method |
| 6 | Method §3.2 (one sentence) + Reproducibility/appendix (one sentence) | Insert the §7.5 signed-weight clarification wording | Methodological transparency; preempts code-inspection questions |
| 7 | §3.2 item 3 (R2-3) | Rephrase per reviewer's suggested intent (static graph; GRU models dynamics; learned structure captures real dependencies) | Resolve the static-vs-adaptive contradiction |
| 8 | §5.1 temporal-interpretation paragraphs (R1-W4/Q4) | Rewrite: original GSL was contemporaneous (now proven by exact reproduction); the multi-lag formulation supplies the explicit lagged construction the interpretation requires | Convert an asserted interpretation into a demonstrated construction |
| 9 | §6 Conclusion/limitations (R1-W9) | Add: DAGMA scalability datapoints (207 nodes ~16 min/PH, 4 CPU cores; 156-node SZ 52–76 min archived), λ not swept, two backbones, PH≤4 scope (with 15-min proxy) | Required limitations paragraph |
| 10 | Appendix `app:original_gsl` | Annotate: protocol was union-of-K-offset-fits (28/32/33/39 edges); slice-0 graphs reproduced exactly by the revised protocol; tables remain single-seed historical | Accuracy of historical description; strengthens the reproduction story |
| 11 | Abstract | No numerical change needed (14.9%/27.4% already multi-seed); optionally add one clause that the original baseline claim was reproduced (21.7% vs 21.8%) | Consistency with R2-1 correction |
| 12 | Figures | No new figures strictly required (`fig:graph_comparison`, `fig:pred_vs_actual` already answer R1-Q1/Q2). Optional: error-bar bar-charts if R1-W11 is pursued editorially | — |

Claims to **strengthen**: GSL-vs-physical result (now 5-seed, protocol-reproduced);
dataset-dependence (now multi-seed); reproducibility of the historical pipeline.
Claims to **weaken/temper**: any causal-structure language (R2-2); PH4 SZ stability;
"sparsity" as an explanation of gains.

---

## 13. Reviewer Response Plan (planning only — no polished rebuttal yet)

| Issue | Concern | Action | Experiment | Numerical evidence | Conclusion | Response strategy |
|---|---|---|---|---|---|---|
| R1-W5 | sparsity vs structure; fairer baseline | Ran matched-budget controls | **A** (+Stage 26 union control) | Rand 6.096±0.107, Corr 5.389±0.088, NoSpatial 5.234±0.090, MultiGSL 4.794±0.102, Mix 4.452±0.143; union-of-same-edges 5.928 | Gains require DAGMA lag edges *used per-lag*, not sparsity | Thank reviewer; present table; note density stats (mean degree 12.7 vs 0.14) already visualized; optionally add sparsified-physical control (§11.1) if run |
| R1-W6 | no seeds/variance | 5-seed protocol everywhere new | **B, C** (+Stage 26) | All §4/§5 tables; DAGMA determinism audit | Variance isolates training randomness; effects survive | State seed policy, std convention, and the determinism audit; no p-values claimed |
| R1-W7 | longer horizons | Scope limitation + proxy | — | 15-min sampling: 27.4% mean at ≤60 min ahead, gains grow with horizon | Structure matters more at longer *wall-clock* horizons | Acknowledge PH≤4 in 5-min sampling as scope; offer PH5–8 as future work (or run if requested) |
| R1-W4/Q4 | contemporaneous vs lagged DAG | Decoded + reproduced historical construction; multi-lag as the fix | **B** + Stage 26 | Slice-0 exact reproduction (§4.3); lag ablation 10%/13.3% | Original DAG was contemporaneous; interpretation now carried by explicit multi-lag graphs | Show the loader code result + reproduction table; present multi-lag as the constructive answer |
| R1-Q3 | time-varying overhead | Measured runtimes | **B** | 16 min/PH (207 nodes), multi-lag ~4 h; sliding-window = windows × fit cost | Feasible but linear-in-windows cost | Give the measured numbers; keep plan qualitative |
| R1-W9 | limitations | Added limitations paragraph (planned §12.9) | — | Scalability/λ/backbone facts above | Honest scoping | Concede directly; point to measured runtimes |
| R2-1 | abstract number mix-up | Numbers re-derived multi-seed; canonical rerun of original claim | **B** | 21.7% mean (5 seeds) vs original 21.8% (single seed); abstract now 14.9%/27.4% | Corrected and independently validated | Thank reviewer; show the correction and the reproduction |
| R2-2 | interpretability unsupported | Visualizations + graph statistics + tempered claims | **B, A** | `fig:graph_comparison`; B's 28-edge stats; SZ boundary | Structure is descriptive, not causal | Temper language; point to figure and statistics |
| R2-3/4/5/6 | editorial | Text edits in Stage 39 | — | — | — | Fix as suggested |

---

## 14. Reproducibility & Provenance Status

| Artifact | Status |
|---|---|
| `results/stage32_sparse_control/stage32_sparse_control.json` (+ `run.log`) | ✅ present, provenance block correct (checked against blocks: 12/3/15=30, union 28, 0 negative survivors) |
| `results/stage33_gsl_canonical/stage33_gsl_canonical_results.json` + 8 `.npy` graph files + `run_losloop_tgcn.log` | ✅ present; per-graph provenance complete (input construction, threshold, sign counts, software versions, determinism note); physical `n_edges` now 2833 |
| `results/stage33_sz_multiseed/stage33_sz_multiseed_results.json` (+ `run.log`) | ✅ present; provenance now computes lag edges from actual blocks (2/PH) |
| Stage 26 blocks | ✅ 20 Los + 20 SZ `.npy` + 8 metadata JSONs + DAGMA logs |
| Historical archives | ✅ 8 W_est files in `archive/historical_submission/` (git-tracked); loader path updated |
| Scripts / commits | ✅ `16ba21b` (Stage 34), `650747c` (35), `1e84b28` (36), `61f6434` (37) — all pushed to `origin/main` |
| Determinism evidence | ✅ Stage 35 report + synthetic tests (cross-process bit-identity; thread-count ~1 ULP; identical thresholded support) |

**Gaps (minor, none blocking):**
1. **Result JSONs and B's graph `.npy` files are gitignored** — they exist locally but are
   not in version control. If the machine is lost, the *graphs* are regenerable
   deterministically (~69 min) but the JSONs' trained-model results would need a full
   rerun (~2 h total). **Recommend archiving them** (e.g., a `results-archive/` bundle or
   external backup) before Stage 39.
2. **Environment pinning:** `requirements.txt` pins `torch==2.0.1`, `dagma==0.1.0`, but
   the actual run environment recorded in B's JSON is Python 3.12.13, numpy 2.5.2,
   scipy 1.18.1; the torch version actually used is not recorded in any JSON. Record the
   full `pip freeze` of `/data/python-envs/pytorch` once and store it next to the results.
3. A's and C's JSONs do not record software versions (B's does) — acceptable, since they
   ran in the same environment minutes apart; note it in the archive README.

No artifact is missing that would prevent complete future reproduction.

---

## 15. Final GO/NO-GO & Stage 39 Recommendation

```text
GO for manuscript revision — Stage 39.
```

**Stage 39 should:** execute the §12 integration plan (12 items, in that order),
draft the point-by-point response from §13, apply the §10 classification when deciding
what stays in the appendix, and add the limitations paragraph. Do **not** rerun A/B/C;
do not modify historical artifacts.

**Recommended Stage 39 preamble (optional, user's call):** the ~5-minute
sparsified-physical control (§11.1) and/or the ~30–60-minute multi-seed `tab:multiph`
upgrade (§11.2) — both strengthen reviewer responses at trivial cost. Everything else
in §11 is deferred or declined.

Stage 39 may begin as soon as this report is inspected.
