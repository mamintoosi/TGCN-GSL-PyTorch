# Stage 31 — Manuscript Reconstruction & Experimental Evidence Audit

**Date:** 2026-09-08 (rev. 2 — updated after the result folders were copied from the training machine)
**Repository:** TGCN-GSL-PyTorch
**Auditor:** Buffy (Freebuff)
**Status:** Read-only audit. No `.tex`, source-code, figure, or configuration file was modified. Nothing was committed. No experiment was run. *Rev. 2 note:* the first audit pass found `results/stage29_los15min/` missing; the author has since copied the result folders (`results/stage29_los15min/`, `results/stage30_forensic_audit/`) from the training machine into the repo. This revision incorporates and verifies them; all other findings of the first pass stand unchanged.

**Evidence sources used:** `paper/sn-article_original.tex`, `paper/sn-article.tex` + `paper/sections/*` + `paper/appendix/*`, `paper/Reviewers-comments.txt`, `paper/revision_notes.md`, `doc/RESPONSE_TO_REVIEWERS.md`, `doc/REVISION_PHASE1_AUDIT.md`, `stage30-prompt.md`, `results/stage26_validation/*`, `results/stage26_checkpoint/*`, `results/stage27_resolution/*`, `results/stage29_los15min/*` (recovered from the training machine), `results/stage30_forensic_audit/STAGE30_FORENSIC_AUDIT.md` (executed on the training machine), `archive/revision_stages/docs/*`, `gsl_stage26/*.py`, and full git history (62 commits, from `95ff91e` to `8b975fe`).

---

## 1. Executive Summary

1. **The original submission and the current revision are the same research program at two stages of maturity, not two unrelated studies.** The revision's central finding — dense physical graphs *hurt* T-GCN on Los-loop (RMSE 7.66 physical vs 5.14 no-graph) — was invisible in the original paper, which compared only against the dense graph and never tested a no-graph baseline. Once the no-graph control was added (Stages 24–26), the entire story inverted: the original "GSL beats physical" result was real, but for a reason the original paper did not identify (sparsity/anti-oversmoothing), and the multi-lag extension is the natural continuation.

2. **All headline numbers in the current manuscript verify against the experiment artifacts** (Stage 26 JSON files, checkpoints, validation reports) — with one rounding inconsistency (T-GCN-MultiGSL-Mix mean 4.452 vs 4.458, both derivable from real numbers, different contexts) and minor parameter-count bookkeeping differences. No number was found to be fabricated or contradicted by the artifacts. The Stage 29 15-minute results, initially missing from the repository, were recovered from the training machine and are likewise verified (§6.3).

3. **Stage 27 is invalid for forecasting** — direct evidence in `results/stage27_resolution/phase2_evaluate.log`: it trained with plain `nn.MSELoss()` via an inline loop, used a different sequence-construction (`pre_len` off-by-one), different train/test slicing, and — critically — ran "T-GCN-MultiGSL-Mix" through the **single-graph `TGCN` class with the union adjacency**, i.e., the wrong model class. Its saved JSON holds *normalized* RMSE (0.1165/0.1282); the 8.7772/9.0531 numbers in `STAGE27_REPORT.md` have no artifact backing and are arithmetically inconsistent with the JSON (implied `feat_max` would have to be 75.34 for NoGraph and 70.64 for the proposed method — impossible, since one run has one `feat_max`).

4. **Stage 29 is VALID and its artifacts are now in the repository and verified.** `gsl_stage26/stage29_los15min.py` verifiably reuses the canonical Stage 26 pipeline (`SupervisedForecastTask`, `mse_with_regularizer`, `set_seed`, identical `GatedMultiGraphTGCN`/`MultiGraphTGCNFixed` implementations, same resampling `T//3`, split, `feat_max` from train only). The outputs — produced on the Linux training machine (`/data/git/mamintoosi/TGCN-GSL-PyTorch`, run timestamp 2026-09-08 10:02) and copied into `results/stage29_los15min/` — were independently re-analyzed from `stage29_los15min_results.json` in this audit: **every quoted figure verifies exactly** (NoGraph 8.6000 ± 0.2492; T-GCN-MultiGSL 7.2462 ± 0.2975, +15.74%; T-GCN-MultiGSL-Mix 6.2402 ± 0.1873, **+27.44%** at PH=1; +21.36 / +19.76 / +16.09% at PH 2–4; Mix wins 5/5 seeds, per-seed +23.2%…+31.3%). The recovered Stage 30 audit independently confirms the same verdicts and additionally proves the Stage 27 and Stage 29 DAGMA W matrices are bit-identical (`np.allclose = True`).

5. **The "temporal resolution explains SZ-Taxi" hypothesis is REFUTED by the verified evidence.** Los-loop at 15 min improves *more* than at 5 min (+27.44% vs +13.33% seed-42 / +14.9% 5-seed mean), while SZ-Taxi at the same 15-min resolution shows ~0 (+0.19%). Two 15-minute datasets with opposite outcomes rule out temporal resolution as the cause; dataset-specific factors dominate (traffic regularity, network size, and — concretely — graph learnability: SZ yields only 2 learned edges at τ=0.1, vs 30 for Los 5-min and 32 for Los 15-min). The Stage 27 report's claim "temporal resolution is the primary confounding factor" rests on invalid forecast numbers and must be discarded; the honest framing (already in `limitations.tex`) is dataset dependence.

6. **Answer to the central question (§14): YES** — the original GSL/cGSL results and the new T-GCN-MultiGSL-Mix results can coexist in one coherent paper. The logical bridge: the original experiments established that *sparse learned structure* helps but left the mechanism unexplained (Section 5 of the original was "asserted rather than demonstrated" — Reviewer 1 #4); the audit-driven discovery that a no-graph baseline beats the dense graph reframed the mechanism as sparsity-vs-oversmoothing; asking *what* the learned DAG actually encodes then forces the temporal question, which the multi-lag formulation answers explicitly. The original result is not an appendix relic — it is the first act of the story. The current manuscript already has this bridge but underuses it: the original results sit in an appendix (`appendix/original_gsl_results.tex`) and Results §5.7 ("Relation to Original GSL/cGSL Results") contains an empty citation `()` and is misplaced after the main tables.

7. **Recommended framing: evolution, not replacement** (details in §8) — this is also the framing that best answers Reviewer 1 #4 (direct evidence for the temporal interpretation) and Reviewer 2 #2 (interpretability claims), and it preserves ~100% of the original evidence with zero re-runs.

8. **No experiment needs to be run before manuscript restructuring.** With the Stage 29 artifacts now present and verified, the entire revised Results section — including the 15-minute subsection — can be assembled from existing, verified artifacts. The only remaining housekeeping action is provenance: the result directories are excluded from version control (`.gitignore` contains `results*/`), so the small JSON/CSV artifacts backing the manuscript should be archived or committed before submission.

---

## 2. Original Submission Reconstruction

### 2.1 Identity of the two sources

| Source | Content | Assessment |
|---|---|---|
| Commit `dac8978` ("Update README.md", 2026-02-27) | Only implementation code: `main.py`, `models/`, `configs/` (48 YAMLs: {gcn,tgcn}×{los,sz}×{plain,gsl,gsl-dcg}×pre_len1–4), `data/W_est_*.npy` (8 DAGMA matrices), notebooks. **No paper source, no PDF** | Confirms the todo note: the repo at submission time contained only code |
| `paper/sn-article_original.tex` (1,166 lines, single file) | Full submitted manuscript ("Version 3.1 December 2024" Springer template) | The authoritative reconstruction source |

**Verdict:** they represent the same manuscript state *indirectly* — the commit is the code snapshot accompanying the submission, and `sn-article_original.tex` is the submitted paper source (added to the repo later, at `5204e60` "Add audit report and paper source files for revision"). No PDF of the submission exists in the repo; the text is verified only against the `.tex`. The config grid in `dac8978` matches the paper's method grid exactly (GCN/T-GCN × physical/GSL/cGSL(`gsl-dcg`) × PH 1–4), which cross-confirms the correspondence.

### 2.2 Extracted content

- **Title:** "Graph Structure Learning for Traffic Prediction"
- **Abstract:** GSL via continuous DAG optimization (DAGMA) replaces physical-proximity adjacency in GCN/T-GCN; two variants — acyclic (GSL) and symmetrized cyclic (cGSL); "up to 21.6% and 24.7% reduction in RMSE for GCN and T-GCN baselines, respectively"; spatial-only models benefit from cyclic, temporal models from acyclic; "computationally efficient and interpretable … explicit insights into hidden traffic patterns"; code URL.
- **Research problem:** physical proximity is a poor proxy for functional traffic dependency.
- **Stated contributions:** (i) first application of NOTEARS/DAGMA-style GSL to GCN-based traffic forecasting; (ii) explicit persistent dependency discovery (vs implicit attention weighting); (iii) interpretability for urban planners; (iv) GSL as plug-and-play module.
- **Method names:** GCN, T-GCN, GCN-GSL, TGCN-GSL, GCN-cGSL, TGCN-cGSL.
- **Method:** apply DAGMA (augmented-Lagrangian, trace-exponential acyclicity h(W)) to traffic speed data X ∈ R^{T×N}; threshold; GSL = directed DAG adjacency; cGSL = A + Aᵀ. **Input to DAGMA: contemporaneous (single-time-step) rows** — no explicit temporal blocking (later confirmed by the repo's own `CLEAN_REIMPLEMENTATION_AUDIT_20260831_170534.md`: "DAGMA input is contemporaneous, not temporal").
- **Datasets:** SZ-Taxi (156 sensors, 15-min) and Los-loop (207 sensors, 5-min); 80/20 temporal split; window 12; PH 1–4; λ = 0.01 (SZ) / 0.02 (Los) (§4.2).
- **Configurations:** single seed (fixed, stated as a feature), 50 epochs, RTX 3090.
- **Major tables:** Table `gcn_gsl_comparison_combined` (GCN/GSL/cGSL × 4 PH × 4 metrics × 2 datasets); Table `tgcn_gsl_comparison_combined` (same for T-GCN); hyperparameter table.
- **Major figures:** GCN layer schematic; dummy-network concept; 2 RMSE-vs-PH tikz figures (GCN, T-GCN); **4 convergence-curve figures** (Figs 5–8); DAG interpretation figure (Fig 9).
- **Key quantitative claims:** GCN-cGSL best everywhere: −21.6% RMSE SZ, −24.7% Los (vs GCN). TGCN-GSL best: −9.5% SZ, −21.8% Los (vs T-GCN). (Abstract's "24.7% for T-GCN" is the GCN Los number — the discrepancy Reviewer 2 flagged; Section 4.4 itself says ~21.8% for TGCN.)
- **Conclusions:** 5 enumerated findings (adjacency ≠ influence; temporal propagation matters; causal structure learning helps; interpretable causal graphs; simpler than attention).
- **Limitations:** none (no dedicated section).
- **Claims on GSL/sparsity/graph/temporal/dataset:** single static learned graph; temporal interpretation of the DAG **asserted post hoc** in Section 5 ("edge j→i signifies j at t influences i at t+1") with no direct evidence; no sparsity/oversmoothing analysis; no per-dataset mechanism discussion (SZ vs Los differences reported but not explained).

---

## 3. Current Manuscript Reconstruction

`paper/sn-article.tex` (modular; `sections/abstract,introduction,background,method,experiments,results,discussion,limitations,conclusion` + `appendix/{bibliometric,additional_diagnostics,convergence,original_gsl_results}`).

- **Title:** unchanged. **Keywords:** now include Multi-Lag Dependency Learning, Learned Graph Mixing.
- **Abstract (rewritten):** multi-lag DAGMA on explicit blockings Z=[x(t−L)..x(t)]; T-GCN-MultiGSL-Mix with per-node per-timestep learned mixing; 14.9% mean RMSE improvement over no-graph on Los-loop (5 seeds); parameter-matched control; lag ablation; marginal on SZ-Taxi ("dataset dependence").
- **Research problem:** extended to two problems — (1) physical proximity ≠ functional dependency (as before), (2) **dense physical graphs cause oversmoothing** (new).
- **Stated contributions:** (1) Multi-lag DAGMA; (2) T-GCN-MultiGSL-Mix.
- **Method (sections/method.tex):** DAGMA recap with W[i,j] convention note (answers R1#3); multi-lag formulation with block algebra and per-lag binary graphs (§3.3); T-GCN-MultiGSL-Mix architecture — softmax gate MLP([x_t; h_{t−1}]) → per-node/per-timestep convex mix of K=3 lag Laplacians → GRU update; contrasted with UnionGraph / WeightedMulti / MultiGraph fixed assignment; parameter count 17,091 vs 12,672.
- **Method names (current authoritative):** T-GCN-NoSpatial, Physical, T-GCN-MultiGSL, T-GCN-MultiGSL-Mix (manuscript) ⇄ code/JSON: NoGraph, Physical, MultiGraphTGCN_fixed, GatedMultiGraphTGCN.
- **Experiments (experiments.tex):** datasets as before + physical-graph edge counts (2833 Los / 532 SZ); L=3 → 828 (Los) / 624 (SZ) variables; DAGMA λ₁=0.01, L2, warm 30k / max 60k iters, τ=0.1, self-loops removed, train-only; models H=64, Adam lr 1e-3 wd 1e-4, batch 128, window 12, 50 epochs, MSE + L1 regularizer; mean ± std over seeds 42–46.
- **Major tables (results.tex):** `tab:oversmoothing`, `tab:lag_stats`, `tab:multiseed`, `tab:param_control`, `tab:lag_ablation`, `tab:multiph`; appendix: `tab:gcn_gsl_combined`, `tab:tgcn_gsl_combined`, `tab:density_stats`, `tab:sz_multiph`.
- **Major figures (paper/figures/):** fig1 graph comparison; fig2 RMSE bars; fig3 multiseed boxplot; fig4 param control; fig5 lag ablation; fig6 threshold sensitivity; fig7 lag edge stats; fig8 convergence (also in Discussion §5.5); fig9 predicted-vs-actual.
- **Major quantitative claims:** 32.8% (NoSpatial vs Physical, seed 42); 14.9% mean (5-seed); 13.3% (seed 42, PH1); 8.4% and +7.1% (MultiGSL vs NoSpatial; Mix vs MultiGSL); 0.1% parameter-control gap ("99% from architecture"); all-lags 13.3% vs ~10% per single lag; SZ-Taxi marginal (4.108 vs 4.116 at PH1; 4.221 vs 4.221 at PH4).
- **Conclusions:** 6 enumerated findings; future work = learned lag selection, sliding-window DAGMA, scalability, other backbones.
- **Limitations:** 8 explicit items (dataset dependence, two datasets, DAGMA scalability, fixed L=3, static graphs, one backbone, no causal claim, PH≤4).
- **Claims on GSL/sparsity/graph/temporal/dataset:** sparse learned graphs help *relative to no-graph*; multi-lag graphs are structurally distinct (low Jaccard); learned mixing matters; benefit is dataset-dependent (honestly reported); **causal language removed** (only the disclaiming mention in Limitations).

---

## 4. Reviewer-Criticism → Evidence Mapping

| Reviewer criticism (condensed, meaning preserved) | What the original paper did | What the revised paper currently does | Evidence added since submission | Still unresolved? |
|---|---|---|---|---|
| **R1#4** — Temporal interpretation of the learned DAG is *asserted, not demonstrated*; DAGMA input construction unclear (contemporaneous vs lagged?) — load-bearing for the whole Section 5 story | Section 5.1 asserts edge j→i = "j at t predicts i at t+1"; DAGMA actually fed contemporaneous snapshots | Multi-lag DAGMA on explicit temporal blockings Z=[x(t−L)..x(t)] makes lag-specific structure *directly observable* (Table `lag_stats`, Fig 7); cGSL moved to appendix; old Section 5 removed | Stage 25D pilot (Jaccard < 0.08 across lags), Stage 26 full-sensor W matrices + metadata (verified `los_ph1_seed42_L3_metadata.json` block algebra) | **Resolved in design.** Residual: the current text does not explicitly say "the original interpretation could not be verified because the original input was contemporaneous" — one honest sentence would close it |
| **R1#5** — No density/degree reporting; gains may come from reduced oversmoothing due to sparsity, not the learned structure; sparse the physical graph as a control | No density analysis; no sparse-physical control | Table `tab:oversmoothing` (2833 vs 207 vs 6/60/30 edges); Fig 1 degree distributions; Fig 6 threshold sensitivity | Stage 24–26 experiments incl. `SingleDAG_thr*`, `Corr-K*`, `PhysSp` (in `EXPERIMENTAL_RESULTS_REPORT_20260831_220418.md`), `UnionGraph`, `IntersectGraph` | **Mostly resolved.** Residual: no *sparsified-physical* (top-K physical) row in the manuscript table, though the experiment exists (Corr-K8/16/32 are in the JSON; PhysSp only in archived report). Adding one row (e.g., Corr-K8: 1656 edges, RMSE 6.915) would fully close it |
| **R1#6** — No seeds/variance/significance | Single fixed seed | 5 seeds (42–46), mean ± std, 5/5 wins (Table `tab:multiseed`) | `stage26_validation_A_losloop_ph1.json` (verified per-seed values) | **Resolved.** n=5 precludes strong significance tests; report as such |
| **R1#7** — Only PH ≤ 4; show longer horizons | PH 1–4 only | PH 1–4 multi-horizon table (Table `tab:multiph`); longer horizons listed as limitation | Stage 26 PH 1–4 JSONs (verified) | **Partially resolved.** The reviewer asked for *longer* horizons; the revision re-presents the same range plus a limitation. Acceptable if argued; a cheap PH 5–8 run on Los-loop would close it fully |
| **R1#1,2,8,10,11,12** — bibliometric underused; background too long; Section 5 repetitive; metric defs; dense convergence figures; citation style | Long background; 4 convergence figures; repetitive Section 5 | Compressed background; convergence → Appendix C; restructured Results/Discussion; 2-line metrics | Editorial only | **Resolved** (citation style still "will standardize" — a final-pass item) |
| **R1#3** — A→W notation switch unexplained | Footnote only | Explicit convention note in §3.2 ("W[i,j]: i → j") | Editorial | **Resolved** |
| **R1#9** — No limitations section | None | Section 7 with 8 items | Editorial | **Resolved** |
| **R1-Q1** — Visualize physical vs learned graph | Fig 9 conceptual only | Fig 1 (heatmaps + degree distributions) | Generated from Stage 26 W matrices | **Resolved** |
| **R1-Q2** — Predicted vs actual curves for nodes | Not present | Fig 9 predicted-vs-actual (3 high-variance nodes, r > 0.99) | Stage 26 checkpoint `y_pred.npy`/`y_true.npy` (checkpoints exist under `results/stage26_checkpoint/`) | **Resolved** |
| **R1-Q4** — Direct evidence for temporal-DAG interpretation (ablation feeding simultaneous vs lagged data) | Post-hoc interpretation | Multi-lag formulation is exactly the requested "lagged data" arm; current-block (contemporaneous) arm exists as the SingleDAG/`current` block evidence | Stage 25D/26 | **Resolved** — and worth saying explicitly in Discussion |
| **R2#1** — Abstract 24.7%/21.6% mismatch vs Section 4.4 | Abstract wrong (24.7% is the GCN-Los number) | Abstract rewritten with 14.9% (5-seed mean) | Stage 26 Validation A | **Resolved** |
| **R2#2** — "Hidden causal structure" claims unsupported; no learned-matrix visualization vs physical | Causal claims throughout; no heatmap | All causal claims removed (single disclaiming mention in Limitations); Fig 1 heatmaps added | Stage 26 W matrices | **Resolved.** NB: the *original* strong cGSL/GSL tables remain in the appendix, whose surrounding text no longer claims causality |
| **R2#3** — Static graph vs "adapts over time" contradiction (§3.1 item 3) | Contradictory wording | Distinguished: static learned graphs / GRU dynamics / per-timestep gating (§3.4, Limitations item 5) | Code inspection | **Resolved** |
| **R2#4** — cGSL defined too late (§5.3) after use in §4 | Definition late | cGSL confined to appendix with its definition up front | Editorial | **Resolved** |
| **R2#5** — 16-panel convergence figures unreadable | Figs 5–8 | Moved to Appendix C; main text fig8 is a 2-panel loss curve | Editorial | **Resolved** |
| **R2 misc** — "avergae" typo | Present | Gone (text rewritten) | — | **Resolved** |

**Summary:** every substantive criticism has a designed response backed by an artifact in the repo. The three with residual exposure are R1#5 (missing sparsified-physical row in the manuscript), R1#7 (no horizons beyond 4), and the single-seed status of the SZ-Taxi conclusions (optional 5-seed run). The 15-min analysis — previously listed here as pending — is now verified (§6.3).

---

## 5. Method Evolution

**Timeline (from git + archived reports):**

| Stage (report) | What was done | Key finding |
|---|---|---|
| Submission (`dac8978`) | DAGMA on contemporaneous snapshots; GSL/cGSL; GCN/T-GCN; seed 42 | Sparse learned graphs beat dense physical on both datasets; acyclic best for T-GCN, cyclic best for GCN |
| Clean-room audit (`CLEAN_REIMPLEMENTATION_AUDIT_20260831`) | Reimplementation + 64 experiments | Two critical discoveries: Laplacian asymmetry; **DAGMA input was contemporaneous, not temporal** → original temporal story unverified |
| 112-experiment report (`EXPERIMENTAL_RESULTS_REPORT_20260831`) | 7 graph types incl. random, Corr-K, PhysSp | Correlation graphs beat DAGMA in 15/16 configs → **the win is sparsity, not the specific learned edges** |
| Stage 24 | Single-lag temporal DAGMA | Dense physical graphs severely oversmooth |
| Stage 25 / 25D | Multi-PH validation; multi-lag pilot (N=20) | Lag blocks are structurally distinct (Jaccard < 0.08) |
| **Stage 26** (`654b7be`…`9e90a59`) | Full-sensor multi-lag DAGMA (828×828) + GatedMultiGraphTGCN | Los PH1: Mix 4.458 vs NoSpatial 5.143 (seed 42); forensic audit → PROMISING, needs seeds/param control |
| **Stage 26 Validation** (`9e90a59`) | 5 seeds; parameter-matched h=74; lag ablation | 14.9% mean; 99% from architecture; all 3 lags contribute |
| Revision Phases 1–3 (`565a199`, `dea6667`, `cc46391`, …) | Manuscript restructure; model renames (`8a9605c`); figures | Current modular manuscript |
| Stage 27 (`4fece88`, `424df94`) | Los-loop resampled to 15-min; investigate SZ weakness | Report claims −3.14% (Mix worse than baseline) — **invalid (wrong pipeline/model class)** |
| Stage 29 (`8b975fe` adds script; no outputs in repo) | 15-min experiment on the canonical pipeline, 5 seeds | Numbers quoted in prompts; **artifacts missing on this machine** |

**The seven questions:**

1. **Original GSL/cGSL approach:** DAGMA on contemporaneous per-timestep snapshots X ∈ R^{T×N}; threshold → directed DAG (GSL) or symmetrized (cGSL); insert as GCN/T-GCN adjacency. Learned *once* per dataset/P`H`-subset (the original even merged PH-specific graphs with a logical OR — commented-out text in `sn-article_original.tex`).
2. **Original model variants:** GCN, T-GCN × {physical, GSL, cGSL}.
3. **Original results demonstrated:** large RMSE gains from *any* sparse replacement of the dense graph (up to ~22–25% GCN; ~22% TGCN on Los); backbone-dependent preference for cyclic vs acyclic.
4. **What motivated multi-lag graphs:** the convergence of (a) R1#4's demand for direct evidence of the temporal interpretation, (b) the clean-room discovery that the original DAGMA input was contemporaneous (so the story was unverified), and (c) Stage 25D's observation that lag-specific blocks have genuinely different, nearly disjoint edge structures. The original paper itself planted the seed: a commented-out passage already proposes "learning a separate adjacency matrix for each time lag" as future work.
5. **Current architecture:** T-GCN-MultiGSL-Mix — K=3 lag-specific binary graphs (τ=0.1) from multi-lag DAGMA; per-node, per-timestep softmax gate over their Laplacians; mixed operator applied to [x_t; h_{t−1}]; GRU update; 17,091 params (H=64).
6. **Which criticism it answers:** directly R1#4 (temporal evidence), R1#5 (sparsity/oversmoothing framing), R1#6 (seeds), R2#2 (interpretability now via measurable lag blocks instead of causal rhetoric), R2#3 (static-graph vs adaptation confusion resolved by the gate's explicit semantics).
7. **What remains scientifically useful from the original:** everything that survives the reframing — the GSL/cGSL tables as the *historical first evidence* that structure (not density) matters; the cyclic-vs-acyclic asymmetry as the *puzzle* that the multi-lag formulation resolves; the datasets/protocol; the DAGMA machinery itself (now applied to blockings).

**Name lineage (do not rename code; map in prose):** NoGraph → T-GCN-NoSpatial; Physical → Physical; MultiGraphTGCN_fixed → T-GCN-MultiGSL; GatedMultiGraphTGCN → T-GCN-MultiGSL-Mix; WeightedMultiGraphTGCN → "WeightedMulti" (baseline only); MultiGraphTGCN (Stage 26 evaluate variant, cyclic `t%3` alignment bug documented in the forensic audit) → superseded by MultiGraphTGCNFixed. The word "Adaptive" appears only in the Stage-26 validation report prose ("Adaptive graph gating") and in code comments — it does not occur in the current manuscript's method names, and no rename is needed (per todo constraint, it is not introduced).

---

## 6. Stage 26 / 27 / 29 Evidence Audit

### 6.1 Stage 26 (canonical) — VERIFIED

Verified against `results/stage26_validation/*.json` and `results/stage26_checkpoint/*/metrics.json` (this machine, byte-level):

| Experiment | Dataset | Method (JSON name) | PH | Seeds | RMSE | Purpose |
|---|---|---|---|---|---|---|
| `stage26_results_los_ph{1..4}` | Los-loop | NoGraph / Physical / SingleDAG_thr{0.001..0.3} / Corr-K{8,16,32} / UnionGraph / IntersectGraph / AggregatedDAG / MultiGraphTGCN_thr0.1 / WeightedMulti_thr0.1 / GatedMulti_thr0.1 / per-lag standalones | 1–4 | 42 | PH1: NoGraph 5.1432, Physical 7.6582, MultiGraph 4.7149, **Mix 4.4578**; PH2 5.6422/8.0023/5.5492/5.3075; PH3 6.1635/8.5124/5.9341/5.6868; PH4 6.5016/8.5395/6.3363/6.0043 | Full method comparison per PH |
| `stage26_results_sz_ph{1..4}` | SZ-Taxi | same | 1–4 | 42 | PH1: NoGraph 4.1156, Physical 5.2674, **Mix 4.1076** (edges 2); PH2 4.1596/…/4.1486; PH3 4.1887/…/4.1841; PH4 4.2207/…/4.2214 | Dataset-dependence evidence |
| Validation A | Los-loop PH1 | NoGraph / MultiGraphTGCN_fixed / GatedMultiGraphTGCN | 1 | 42–46 | NoGraph 5.234±0.090; MultiGSL 4.794±0.102; **Mix 4.452±0.143** | Seed robustness |
| Validation B | Los-loop PH1 | NoGraph h64 / h74 / Mix | 1 | 42 | 5.1432 / 5.1373 / 4.4578 | Parameter-matched control |
| Validation C | Los-loop PH1 | GatedMulti lag subsets | 1 | 42 | all-3-lags 4.4578 best; singles ≈4.60–4.62 | Lag ablation |

- Checkpoint `los_ph1_seed42_gated_multi/metrics.json`: RMSE 4.457819938659668, n_params 17,156 (the 17,091 figure in the paper counts gates differently — see §12); `nograph`: 5.143194675445557, n_params 12,737 (paper: 12,672). Both discrepancies are parameter-count bookkeeping, not results.
- DAGMA metadata `los_ph1_seed42_L3_metadata.json`: λ₁=0.01, L2, warm 30k/max 60k, feat_max 70.0, split 1612, runtime 241 min — matches manuscript §4.2 claims (240 min Los, ~28 h total for 8 runs: SZ runs ≈100 min each ⇒ 4×241 + 4×100 ≈ 22 h, slightly under the stated 28 h; minor).

### 6.2 Stage 27 — INVALID for forecasting (evidence on this machine)

Direct artifact evidence (`results/stage27_resolution/`):

1. **Wrong model class:** `stage26_resolution_experiment.py` lines ~288–292 define `"T-GCN-MultiGSL-Mix"` as **plain `TGCN` with the union adjacency** — not `GatedMultiGraphTGCN`. The reported "-3.14%" compares NoGraph vs a *fixed single-graph* model.
2. **Different training code path:** inline loop with `nn.MSELoss()` (line 319), no `SupervisedForecastTask`, no `mse_with_regularizer` (L1), no per-sample gate; optimizer/step identical otherwise.
3. **Different sequence/target construction:** `Y = data[i+seq_len+ph−1]` (single target step, off-by-one vs canonical `generate_sequences`), and train/test slicing `X_all[:split−seq_len]` vs canonical per-split generation.
4. **One seed, no variance.**
5. **The published Stage-27 headline numbers are not backed by the JSON:** `los_15min_ph1_seed42_results.json` stores *normalized* RMSE (NoGraph 0.11650, Mix 0.12816). `STAGE27_REPORT.md` reports 8.7772 / 9.0531 "denormalized". Dividing back gives implied `feat_max` = 75.34 (NoGraph) and 70.64 (Mix) — impossible for a single run with `feat_max = 70.0` (recorded in metadata). The report's denormalized numbers are therefore **of unverifiable provenance** (consistent with a manual copy error, e.g. 8.155→8.7772 / 8.971→9.0531, but UNKNOWN). What *is* solid: the raw normalized values, the DAGMA W matrices (legitimately computed; reusable), and the edge-structure/Jaccard analyses in `phase3_analyze.log`.
6. **Edge-structure claims of Stage 27 remain usable:** 15-min block edges (current 105, lag1 75, lag2 10, lag3 1) and cross-resolution Jaccard (current 0.535, lag1 0.422, lag2 0.154, lag3 0.000) are computed directly from the W matrices and are independent of the forecasting bug. Note however the report's Jaccard analysis included self-loops (Stage 28 audit's criticism applies to the *report*, not necessarily the log values).
7. **Cross-check by the recovered Stage 30 audit** (`results/stage30_forensic_audit/STAGE30_FORENSIC_AUDIT.md`, executed on the training machine with access to both stages' raw outputs): it independently reaches the same verdict (wrong model class, plain MSE, mixed normalized/denormalized reporting, off-by-one sequence construction, one seed) and adds two hard facts: (a) the Stage 27 and Stage 29 DAGMA `W_full` matrices and all lag blocks are **bit-identical** (`np.allclose = True`), so the 15-min graph is shared and valid; (b) the Stage 27 report's per-block edge counts (e.g., lag_1 = 75) counted *total* entries including self-loops, whereas cross-sensor counts at τ=0.1 are lag_1 = 22, lag_2 = 9, lag_3 = 1 (32 edges — the graph actually fed to T-GCN-MultiGSL-Mix in Stage 29).

**Verdict: INVALID** as a forecasting experiment; the 15-min DAGMA graph and the cross-resolution structure analysis are valid, and are exactly what Stage 29 reuses.

### 6.3 Stage 29 — VERIFIED (artifacts recovered from the training machine)

*Rev. 2 update:* the result folders missing during the first audit pass were copied from the training machine into the repository. `results/stage29_los15min/` now contains the DAGMA matrices (`los15_ph{1..4}_seed42_L3_*.npy`, PH-labeled copies of the single PH-independent DAGMA run, dated 2026-09-06), the phase logs, and the master result file `stage29_los15min_results.json` (run timestamp 2026-09-08 10:02:11; 60 rows = 3 methods × 5 seeds × 4 PHs).

- **Pipeline:** `gsl_stage26/stage29_los15min.py` (committed at `8b975fe`) verifiably reuses the canonical Stage 26 machinery: `SupervisedForecastTask` with `loss="mse_with_regularizer"`, `set_seed` (random/np/torch/cuda), identical `GatedMultiGraphTGCN` and `MultiGraphTGCNFixed` forward passes, `generate_sequences` with the same off-by-none targets, resampling `T//3` + 80% split + train-only `feat_max`, identical DAGMA config (λ₁=0.01, warm 30k/max 60k), and — importantly — **reuses the Stage 27 DAGMA `.npy` files** rather than recomputing. The recovered Stage 30 audit confirms the reused matrices are bit-identical to Stage 27's.
- **Verification (this audit, recomputed from the JSON):**

| PH (physical horizon) | T-GCN-NoSpatial | T-GCN-MultiGSL | T-GCN-MultiGSL-Mix | Mix improvement |
|---|---|---|---|---|
| 1 (15 min) | 8.6000 ± 0.2492 | 7.2462 ± 0.2975 (+15.74%) | **6.2402 ± 0.1873** | **+27.44%** |
| 2 (30 min) | 9.3109 ± 0.1968 | 8.5711 ± 0.3266 (+7.95%) | 7.3220 ± 0.3380 | +21.36% |
| 3 (45 min) | 10.0475 ± 0.0740 | 9.1117 ± 0.0774 (+9.31%) | 8.0626 ± 0.2033 | +19.76% |
| 4 (60 min) | 10.4604 ± 0.1965 | 9.7559 ± 0.2097 (+6.73%) | 8.7770 ± 0.3258 | +16.09% |

- **Per-seed PH=1 (Mix wins 5/5):** 8.3457→6.2022 (+25.68%), 8.6211→6.4048 (+25.71%), 9.0528→6.2239 (+31.25%), 8.5805→5.9199 (+31.01%), 8.3998→6.4500 (+23.21%). All figures match the values quoted in `todo.txt`, `stage30-prompt.md`, and the Stage 30 audit exactly.
- **Graph:** 32 cross-sensor edges across the three lag blocks (lag_1 = 22, lag_2 = 9, lag_3 = 1) at τ=0.1; the current block (105 cross-sensor edges) is extracted but unused in forecasting.
- **Manuscript-relevant observation:** at 15-min resolution the *relative* benefit of the proposed method is larger than at 5-min (+27.4% vs +13.3% seed-42 / +14.9% 5-seed), while absolute RMSE of all methods is higher (NoGraph 8.60 vs 5.14) — a harder prediction task on which the learned lag structure matters more. Improvement decreases monotonically with horizon (+27.4% → +16.1%), mirroring the 5-min pattern.

**Verdict: Stage 29 is the authoritative Los-loop-15min result: VALID, multi-seed, canonical pipeline. Stage 27 must not be cited for forecasting in any form.**

---

## 7. Original Results Worth Preserving (recovery from git/artifacts)

| Original result | Dataset | Method | PH | Seeds | RMSE (as published) | Purpose | Still useful? | Retain without re-run? |
|---|---|---|---|---|---|---|---|---|
| GCN vs GCN-GSL vs GCN-cGSL | SZ + Los | GCN/cGSL | 1–4 | 1 (42) | SZ: 5.958→4.648 (−21.6%); Los: 7.724→5.440 (−24.7%) | Historical first evidence that sparsity helps GCN | Yes — as evolution stage / GCN-side evidence | **Yes** — numbers in `appendix/original_gsl_results.tex`; raw `W_est_*.npy` exist in `data/` |
| T-GCN vs TGCN-GSL vs TGCN-cGSL | SZ + Los | T-GCN/GSL | 1–4 | 1 (42) | Los: 6.588→4.818 (−21.8%); SZ: 4.866→4.214 (−9.5%) | Same, T-GCN side; acyclic>cyclic asymmetry = motivating puzzle | Yes — the asymmetry motivates multi-lag | **Yes** — same sources |
| Cyclic-vs-acyclic asymmetry (GCN↔cGSL, T-GCN↔GSL) | both | — | 1–4 | 1 | — | The scientific anomaly that the multi-lag formulation resolves | **Essential** — it is the narrative hinge | Yes |
| Convergence curves (orig. Figs 5–8) | both | GCN/T-GCN | — | 1 | — | Training-dynamics evidence | Marginal — appendix only (already in Appendix C) | Yes (`GCN_*_images_table-main.pdf` etc. still in `paper/`) |

No original result requires a re-run to be retained. The single caveat: original results are single-seed and must be labeled as such when cited alongside 5-seed results (the revised appendix text should state "single seed (42), original submission protocol").

**Stage 29 anchor numbers (status):** Los 5-min canonical: NoGraph 5.1432, Mix 4.4578, +13.33% (seed 42) — verified. Los 15-min: NoGraph 8.6000 ± 0.2492, Mix 6.2402 ± 0.1873, +27.44% (5-seed mean) — **verified** against `results/stage29_los15min/stage29_los15min_results.json`. SZ-Taxi: NoGraph 4.1156, Mix 4.1076, +0.19% (seed 42) — verified.

---

## 8. Determining the Best Scientific Story

### Narrative A — "New method replaces old" (current draft's de-facto posture)
1. RQ: Can lag-specific learned graphs + learned mixing improve T-GCN?
2. Original method's role: footnote/appendix ancestry.
3. New method: T-GCN-MultiGSL-Mix.
4. Main experiments: Stage 26 + Validation A/B/C.
5. Main result: +14.9% (5 seeds) Los; marginal SZ.
6. Reviewers: R1#4/5/6 answered by the new machinery, but the *motivation* for multi-lag rests on asserted prior work.
7. Objection: "Why multi-lag? What was wrong with the single learned graph you published before? Where are those results?" — the paper looks like it disowns its own baseline; also wastes the strongest existing evidence.

### Narrative B — "Evolution: from single learned graph to temporally resolved graph structure" (recommended)
1. RQ: *What does a learned traffic graph actually encode, and can making its temporal content explicit improve forecasting?*
2. Original method's role: **Act I** — first evidence that sparse learned structure beats dense physical (GSL/cGSL tables), and the acyclic/cyclic asymmetry as an unexplained anomaly.
3. New method's role: **Act II** — the anomaly is resolved by discovering (a) oversmoothing (no-graph control) and (b) temporal heterogeneity (multi-lag blocks); T-GCN-MultiGSL-Mix is the natural exploit.
4. Main experiments: original tables (appendix, summarized in main text) → oversmoothing table → multi-lag structure → multi-seed/param-control → horizons & datasets → 15-min analysis (Stage 29, once verified).
5. Main result: dense physical graph is worse than no graph (−32.8%); sparse multi-lag learned graphs with learned mixing give +14.9% over no-graph (5 seeds) and +13.3% at seed 42; benefit is dataset-dependent.
6. Reviewers: R1#4 is answered *by construction* (the multi-lag formulation is the direct lagged-vs-contemporaneous evidence they requested); R2#2's interpretability demand is met by measurable lag-block statistics instead of causal rhetoric; all other items as mapped in §4.
7. Objection: "Two methods in one paper — where is the contribution boundary?" Counter: the paper's contribution is the *resolution of the anomaly*, which requires presenting both; the original method is presented as an ablation-like baseline stage, not as a competing proposal. A second objection — "your original gains may have been mostly anti-oversmoothing, not structure" — is not a weakness here; it is *part of the findings* and is already supported (SingleDAG_thr0.3 with 6 edges ≈ 5.213 vs NoSpatial 5.143: at extreme sparsity, structure adds little; at 30 curated multi-lag edges it adds 13.3%).

### Narrative C — "Oversmoothing paper: sparsity is all you need"
1. RQ: Does graph density, not graph identity, determine GCN/T-GCN accuracy?
2. Original method: one of several sparsification heuristics.
3. New method: proof that curated multi-lag edges beat generic sparsity.
4. Main experiments: oversmoothing table + Corr-K/random baselines.
5. Result: −32.8% from dropping the graph; +13.3% from 30 learned edges.
6. Reviewers: R1#5 fully central; R1#4 becomes secondary.
7. Objection: throws away the multi-lag novelty and the journal-submission continuity; the 112-experiment finding that Corr-K beats single-DAGMA complicates (though multi-lag gating answers it). Weakest fit for the venue and for the reviewer responses already drafted.

**Ranking: B > A > C.**

### Testing the user's proposed narrative (§8 of todo)
> "…original GSL experiments demonstrated that replacing dense physical connectivity with sparse learned graph structures can improve T-GCN forecasting. This motivated a deeper investigation of temporal dependency structure… substantially improves Los-loop across both 5-minute and 15-minute resolutions, while marginal on SZ-Taxi, revealing dataset dependence."

Verdict: **largely defensible, with three required corrections derived from the evidence:**
1. "Original GSL demonstrated… improving T-GCN" — true, but the evidence is *stronger and more honest* if restated: the original showed sparse-learned beats dense-physical; the no-graph control (new) revealed the dense graph was the main problem. Keep the original claim but scope it to its actual control.
2. "Improves Los-loop across both 5-minute and 15-minute temporal resolutions" — **now fully supported by verified artifacts**: +13.33% (seed 42) / +14.9% (5-seed mean) at 5-min and +27.44% (5-seed mean) at 15-min PH=1, with 5/5 seed wins at 15-min (per-seed +23.2%…+31.3%). One presentation requirement stands: the 15-min result is at a *different physical horizon* (15 min ahead, not 5), so the paper must state physical horizons explicitly to avoid the apples-to-oranges trap flagged in the Stage 30 prompt.
3. "Dataset dependence rather than universal gains" — fully supported (Los +13.3/+14.9% at 5-min, +27.4% at 15-min; SZ +0.19% seed-42, +0.2–0.9% across PHs). Do **not** attribute the SZ weakness to temporal resolution: Los-15min improves *more* at 15 min than at 5 min, so resolution alone is ruled out; the two 15-min datasets diverge, implicating dataset-specific dynamics (traffic regularity, sensor count 156 vs 207, graph learnability — SZ DAGMA finds only 2 edges at τ=0.1 vs Los 5-min's 30 and Los 15-min's 32). This is the defensible statement.

---

## 9. Which Original Results Should Stay

| Original result | Keep? | Where in revised paper? | Why? | Need rerun? |
|---|---|---|---|---|
| GCN/GSL/cGSL table (both datasets, PH1–4) | **KEEP** (summary row in main text + full table in appendix) | Results §"Baseline" 1-row summary; Appendix "Original GSL/cGSL Results" | First evidence of the sparsity effect; GCN side never re-examined in revision; establishes continuity | No |
| T-GCN/GSL/cGSL table | **KEEP** (same treatment) | Same | Contains T-GCN−21.8% (Los) — still the largest single-graph gain; motivates "what does the DAG encode?" | No |
| Cyclic-vs-acyclic asymmetry finding | **KEEP — promoted to main text** | Results §"From Single-Graph GSL…" + Discussion | The anomaly that the multi-lag formulation resolves; turning it into a puzzle makes the paper one story | No |
| "First application of DAGMA-GSL to traffic forecasting" claim | KEEP (as background fact) | Introduction | True and worth stating; avoids re-litigating novelty | No |
| Convergence figures (orig. Figs 5–8) | KEEP, appendix only | Appendix C | Reviewers asked for their removal from main text; they remain supportive | No |
| Original abstract claims (21.6/24.7%, causal/interpretability rhetoric) | REMOVE | — | Superseded and partly erroneous (R2#1); rhetoric withdrawn by revision | — |
| Original Section 5 temporal-DAG interpretation | **KEEP BUT REFRAME** | Discussion ("what the single DAG did and did not show") | Do not delete the reasoning; present it as the hypothesis the multi-lag formulation then *tests* | No |
| Original single-seed protocol | KEEP with label | Appendix note | Transparent about why 5-seed numbers supersede | No |

The original strong results therefore do not merely "stay" — they *introduce* the new architecture: dense-graph baseline → sparse wins (original) → but no-graph beats dense (new control) → so what does the learned graph add? → its lag-specific content (multi-lag) → and mixing it pays (Mix). Each step is an existing table.

---

## 10. Simplifying the Experimental Story

**Redundant / reducible in the current draft:**
- **Training curves:** fig8 (Discussion §5.5) duplicates Appendix C's role. Cut fig8 from Discussion or reduce to a one-sentence reference to the appendix. *(REMOVE from main text)*
- **Excessive PH plots:** the manuscript has no per-PH line plots (good). Keep Table `tab:multiph` only; do not add PH curves. *(KEEP as table only)*
- **Redundant visualizations:** fig2 (RMSE bars) duplicates Table `tab:oversmoothing` exactly. Keep one — the table (numbers) and demote fig2, or keep fig2 and slim the table. *(MERGE)*
- **Duplicate graph visualizations:** fig1 (graph comparison) and fig7a (edge counts per lag) overlap; fig7's Jaccard panel is the unique content. Keep fig1 + fig7b/c; fig7a optional. *(TRIM)*
- **Repeated seed-level plots:** fig3 shows both boxplot and mean±std of the same 15 numbers in Table `tab:multiseed`. Keep the table; keep only the boxplot panel of fig3. *(TRIM)*
- **Threshold sensitivity:** fig6 (2 panels) mainly documents that τ=0.1 is a good operating point. One panel (RMSE vs edge count with the Mix star) suffices; full sweep → appendix/additional diagnostics. *(CONDENSE)*
- **Stage 27 outputs:** never cite the −3.14% forecast numbers; optionally keep the cross-resolution Jaccard table as supplementary structure evidence (recomputed without self-loops if cited). *(REMOVE forecasting; CONDENSE structure analysis)*

**Proposed compact Results structure (5 subsections):**

1. **Dense Physical Graphs Hurt: the Oversmoothing Baseline** — purpose: establish the control-corrected baseline; key table `tab:oversmoothing` (+ optionally one row `Corr-K8` for the sparsity-vs-structure question); key number: Physical 7.658 vs NoSpatial 5.143 (−32.8%); addresses R1#5. *(KEEP BUT REFRAME current §5.1; merge fig2 into it or drop fig2)*
2. **What the Single Learned Graph Encodes: from GSL/cGSL to Multi-Lag Structure** — purpose: the narrative hinge; 1-row summary of original GSL/cGSL + the asymmetry; key table: original appendix table summarized + `tab:lag_stats` + fig7; key numbers: lag blocks 70/90/5/16 edges, low cross-lag Jaccard; addresses R1#4, R2#4. *(MERGE current §5.2 + §5.7 + appendix summary)*
3. **T-GCN-MultiGSL-Mix: Multi-Seed Validation and Parameter Control** — purpose: the primary claim; key tables `tab:multiseed`, `tab:param_control`; key figure fig3 (boxplot panel); key numbers: 4.452±0.143 vs 5.234±0.090 (14.9%), 5/5 seeds, h=74 control 5.137; addresses R1#6. *(KEEP current §5.3+§5.4, merged)*
4. **Lag Ablation and Robustness Across Prediction Horizons** — purpose: internal validity; key tables `tab:lag_ablation`, `tab:multiph`; key numbers: all-3 13.3% vs ~10% singles; advantage persists PH2–4; addresses R1#7 partially. *(MERGE current §5.5+§5.6; keep fig5, drop fig6 to appendix or 1 panel)*
5. **Dataset Dependence: Los-loop vs SZ-Taxi (and the 15-Minute Question)** — purpose: honest boundary + the verified Stage 29 result; key table: SZ multi-horizon (from appendix `tab:sz_multiph`) + Stage 29 15-min table; key numbers: SZ +0.19% (seed 42) / +0.2–0.9% (PH1–4); Los-15min +27.44% PH1 (5 seeds, verified); addresses R1#7, R2 honesty; explains dataset dependence without the resolution-causal claim. *(KEEP BUT REFRAME current §5.6 "Dataset dependence" paragraph into its own subsection; supersede Stage 27 entirely)*

The "Predicted vs. actual" material (fig9) moves to the end of subsection 3 or to Additional Diagnostics (it answers R1-Q2; it is qualitative support, not a headline).

---

## 11. Proposed Revised Results Architecture

Concrete proposal (adapting the suggested 6-part skeleton to the evidence — item 2 inserted, item 3 promoted, everything mapped to existing artifacts):

1. **Baseline: Dense Physical Graph vs No-Graph vs Sparse Learned Graph**
   - Table: `tab:oversmoothing` (add optional `Corr-K8` row from `stage26_results_los_ph1_seed42.json`: 1656 edges, RMSE 6.9153) — file: `results/stage26_validation/stage26_results_los_ph1_seed42.json`
   - Figure: fig1 (graph comparison). Fig2 optional/merged.
   - Key numbers: 7.658 / 5.143 / 4.458; SingleDAG τ=0.3 (6 edges) 5.213.
2. **From Single-Graph GSL to Multi-Lag GSL (the evolution)**
   - Tables: one-paragraph summary + pointer to appendix `tab:gcn_gsl_combined`/`tab:tgcn_gsl_combined` (files: `paper/appendix/original_gsl_results.tex`, numbers from original tables); `tab:lag_stats` (file: `results/stage26_validation/los_ph1_seed42_L3_metadata.json` block stats).
   - Figure: fig7 (lag stats; panels b/c primary).
   - Key numbers: T-GCN−GSL −21.8% Los (single seed, original); cross-lag Jaccard < 0.11 (within-15min values from `phase3_analyze.log` for Los 5-min: 0.0106/0.0095/0.1053 — prefer the Stage 25D/26 values currently cited).
3. **Effectiveness of T-GCN-MultiGSL-Mix (multi-seed + parameter control)**
   - Tables: `tab:multiseed`, `tab:param_control` (files: `stage26_validation_A_losloop_ph1.json`, `stage26_validation_B_losloop_ph1.json`).
   - Figure: fig3 (boxplot panel), fig4 optional.
   - Key numbers: 14.9% mean, 5/5 seeds; +7.1% over fixed MultiGSL; parameter control 5.137 vs 4.458.
4. **Ablation: Lags, Threshold, and Horizon Robustness**
   - Tables: `tab:lag_ablation` (`stage26_validation_C_losloop_ph1.json`), `tab:multiph` (`stage26_results_los_ph{1..4}_seed42.json`).
   - Figure: fig5; fig6 condensed (single panel or appendix).
   - Key numbers: all-3 13.3%; advantage at every PH (largest at PH1).
5. **Robustness Across Temporal Resolution: Los-loop at 5 min and 15 min**
   - Table (new): NoGraph / T-GCN-MultiGSL / T-GCN-MultiGSL-Mix × PH1–4, mean±std; source: `results/stage29_los15min/stage29_los15min_results.json` (**now present and verified** — see §6.3).
   - Key numbers: 8.6000±0.2492 vs 6.2402±0.1873 (+27.44%) at PH1 (=15 min ahead); persists PH2–4 (+21.36/+19.76/+16.09%); Mix wins 5/5 seeds at every PH.
   - Explicitly: PH=1 here means 15 minutes ahead; not comparable numerically to 5-min PH=1.
6. **Dataset Dependence: the SZ-Taxi Boundary**
   - Table: `tab:sz_multiph` (move from appendix; `stage26_results_sz_ph{1..4}_seed42.json`).
   - Key numbers: +0.19% (PH1, seed 42), +0.2–0.9% across PHs; SZ DAGMA yields only 2 edges at τ=0.1 (vs 30 Los) — graph learnability as the honest mechanism hypothesis.

This structure uses **only existing, verified artifacts**. It absorbs all eight current subsections and the two "orphan" blocks (§5.7 and the predicted-vs-actual paragraph) without any new experiments.

---

## 12. Quantitative Consistency Audit

| # | Claim | Current manuscript value | Verified value (artifact) | Status | Required action |
|---|---|---|---|---|---|
| 1 | Los PH1 seed42 NoGraph RMSE | 5.143 | 5.1432 (`stage26_results_los_ph1_seed42.json`; checkpoint 5.143194…) | ✅ OK (rounding) | none |
| 2 | Los PH1 seed42 Physical RMSE | 7.658 | 7.6582 | ✅ OK | none |
| 3 | Los PH1 seed42 T-GCN-MultiGSL | 4.715 (oversmoothing table) / 4.717 (multiseed S42) | JSON eval: 4.7149; Validation A: 4.7167 | ⚠️ both real but from *different runs* (Stage 26 evaluate vs Validation A rerun) | add one sentence or footnote that multiseed table values come from the validation rerun; or harmonize to one source |
| 4 | Los PH1 seed42 T-GCN-MultiGSL-Mix | 4.458 | 4.4578 (eval; checkpoint 4.457819…) | ✅ OK | none |
| 5 | 5-seed means: NoSpatial 5.234±0.090; MultiGSL 4.794±0.102; Mix 4.452±0.143 | same | recompute from `stage26_validation_A_losloop_ph1.json`: means 5.23388→5.234 ✓ / 4.79420→4.794 ✓ / 4.45172→**4.452** ✓; stds (population, ddof=0): 0.0896→0.090 ✓ / 0.1018→0.102 ✓ / 0.1432→0.143 ✓ | ✅ OK | none — but note Mix mean is 4.452 here vs 4.458 (seed42-only) in Tables 1/6; both correct, context differs |
| 6 | "14.9% improvement" (abstract/intro/conclusion) | 14.9% | (5.23388−4.45172)/5.23388 = 14.95% | ✅ OK | none |
| 7 | "13.3% improvement" (seed42 PH1, lag-ablation & param-control context) | 13.3% | (5.1432−4.4578)/5.1432 = 13.31% | ✅ OK | none; ensure prose says "13.3% (seed 42)" where the 5-seed 14.9% is also cited |
| 8 | MultiGSL vs NoSpatial 8.4%; Mix vs MultiGSL +7.1% | 8.4% / 7.1% | (5.23388−4.79420)/5.23388=8.40% ✓; (4.79420−4.45172)/4.79420=7.15%→7.1% ✓ | ✅ OK | none |
| 9 | Parameter-matched control: h74 = 16,872 params, RMSE 5.137; "only 0.1%" | 5.137 / 0.1% | 5.1373, 16,872 (`stage26_validation_B_losloop_ph1.json`); (5.1432−5.1373)/5.1432 = 0.11% | ✅ OK | none |
| 10 | Parameter counts 12,672 / 17,091 | same | checkpoint `metrics.json` reports 12,737 / 17,156 | ⚠️ MISMATCH (bookkeeping: checkpoints count projection/bias params; formula in method.tex §3.4.3 gives 17,091) | standardize: state the formula-based count in the text and reconcile or annotate the checkpoint-derived counts in any appendix |
| 11 | Lag ablation table (4.619/4.605/4.605/4.646/4.638/4.559/4.458; edges 3/15/12/27/18/15/30) | same | `stage26_validation_C_losloop_ph1.json`: lag2 4.6190(3), lag3 4.6049(15), lag1 4.6049(12), l1+l3 4.6461(27), l2+l3 4.6378(18), l1+l2 4.5592(15), all 4.4578(30) | ✅ OK | none |
| 12 | Multi-PH Los table (5.143/5.642/6.164/6.502 NoSpatial; 4.715/5.549/5.934/6.336 MultiGSL; 4.458/5.308/5.687/6.004 Mix; Physical 7.658/8.002/8.512/8.540) | same | JSONs: 5.1432/5.6422/6.1635/6.5016; 4.7149/5.5492/5.9341/6.3363; 4.4578/5.3075/5.6868/6.0043; 7.6582/8.0023/8.5124/8.5395 | ⚠️ PH3/PH4 rounded inconsistently: 6.164 vs 6.1635 (should be 6.164 ✓ — ok), 6.502 vs 6.5016 (✓ 6.502), 8.540 vs 8.5395 (✓), 5.308 vs 5.3075 (✓ 5.308), 6.004 vs 6.0043 (✓) — all consistent at 3 d.p. | none — print at 3 d.p. uniformly |
| 13 | SZ PH1: NoGraph 4.116 / Mix 4.108 (+0.19%); PH4 4.221/4.221 | same | 4.1156 / 4.1076 → +0.194% ✓; PH4 4.2207 / 4.2214 → −0.017% (paper says "4.221 vs 4.221") | ✅ OK (note PH4 direction is *negative* at 4 d.p.; the appendix table `tab:sz_multiph` shows Mix 4.221 vs NoSpatial 4.221 — fine, but do not round it into an improvement) | phrase PH4 as "no improvement" |
| 14 | Oversmoothing table Single DAGMA τ=0.3: 6 edges, 5.213; τ=0.1: 60 edges, 6.057 | same | 5.2131 (6 edges) / 6.0569 (60 edges) | ✅ OK | none |
| 15 | Lag stats table (current 70, lag1 90, lag2 5, lag3 16; max w 0.653/0.804/0.673/0.296) | same | `los_ph1_seed42_L3_metadata.json` + Stage 26 forensic audit self-loop analysis: identical | ✅ OK | none |
| 16 | Los-loop physical 2833 edges; SZ 532; identity 207/156 | same | JSON `n_edges` fields | ✅ OK | none |
| 17 | DAGMA cost: Los ≈240 min, SZ ≈100 min, "≈28 h total" | same | metadata: 241.2 min (Los PH1); SZ runtime not in the checked metadata; 4×241+4×100≈22.7 h | ⚠️ minor | soften to "approximately 24 h" or cite per-run times only |
| 18 | Los-loop 15-min (Stage 29): NoGraph 8.6000±0.2492; MultiGSL 7.2462±0.2975; Mix 6.2402±0.1873; +27.44% (PH1); +21.36/+19.76/+16.09% (PH2–4) | not yet in manuscript | `results/stage29_los15min/stage29_los15min_results.json` (recovered from training machine): recomputed means/stds match **exactly** for all 3 methods × 4 PHs; per-seed values match; Mix wins 5/5 seeds at PH1 | ✅ VERIFIED | ready for manuscript use; keep the artifacts with the repo (they are gitignored via `results*/`) |
| 19 | Los-loop 15-min (Stage 27): NoGraph 8.7772; Mix 9.0531; −3.14% | not in current manuscript (good) | JSON holds normalized 0.11650/0.12816; denormalization inconsistent (implied feat_max 75.34/70.64) | ❌ INVALID | never cite; discard |
| 20 | Stage 27 report "SZ matches Los-15min; resolution is the primary factor" | not in manuscript (good) | contradicted by Stage 29 pattern (two 15-min datasets, opposite outcomes) | ❌ UNSUPPORTED | do not adopt the resolution-causal claim |
| 21 | Abstract "14.9% over a no-graph baseline" vs results "13.3%" | both present | both real (5-seed mean vs seed 42) | ⚠️ presentational | ensure each occurrence states its context (mean-of-5 vs seed 42) |
| 22 | Original abstract "24.7% for T-GCN" | historical only | original Table shows 24.7% is GCN-Los; TGCN-Los is 21.8% | ✅ already withdrawn in revision | none (keep the correction documented in the response letter) |

No contradiction was found between manuscript, JSONs, logs, source code, and previous audit reports for any Stage 26 or Stage 29 number. The only dead quantitative claims remaining are item 19 (Stage 27 forecasting numbers), which must simply never be cited.

---

## 13. Terminology Audit

**Authoritative manuscript names:** `T-GCN-NoSpatial`, `Physical`, `T-GCN-MultiGSL`, `T-GCN-MultiGSL-Mix`.

| Alternative name | Where it occurs (files) | Recommendation |
|---|---|---|
| `NoGraph` | all Stage 26/29 JSONs; `gsl_stage26/*.py`; `generate_figures.py` | Standardize in manuscript only (already done). No mapping note needed if figure labels already use manuscript names (they do — `generate_figures.py` maps `NoGraph→T-GCN-NoSpatial` etc.) |
| `GatedMultiGraphTGCN` | code, JSONs, checkpoints, `doc/RESPONSE_TO_REVIEWERS.md`, Stage 26/30 audit docs | Manuscript: never appears (good). In repo docs: add a one-line mapping note in the next audit/report, not in the paper |
| `MultiGraphTGCN_fixed` / `MultiGraphTGCN` / `MultiGraphTGCNFixed` | JSONs, code (`stage26_evaluate.py` vs `stage29_los15min.py` differ!) | The manuscript name `T-GCN-MultiGSL` covers the *fixed-alignment* variant. A mapping note is needed in supplementary/repo README because `stage26_evaluate.py`'s `MultiGraphTGCN` is the **buggy cyclic variant** while `stage26_validation.py`/`stage29_los15min.py`'s `MultiGraphTGCNFixed` is the corrected one — a reviewer or reader diffing the code must not confuse them |
| `WeightedMulti` / `WeightedMultiGraphTGCN` | JSONs, `method.tex` (as contrast baseline, line 95), `generate_figures.py` | Fine as a named baseline; keep manuscript label "WeightedMulti" with one clause defining it (fixed global learned weights) |
| `MultiGraph` (figure legend shorthand) | `generate_figures_extra.py`, fig3/fig8 labels | Prefer "T-GCN-MultiGSL" in any regenerated figure legend |
| `GatedMulti_thr0.1` / `GatedMulti_lag_*` | JSON method strings | JSON-internal; no change; note the threshold suffix convention somewhere (repo README) |
| "adaptive" (lowercase, prose) | Stage 26 validation report ("Adaptive graph gating"); `stage29_los15min.py` docstring "adaptive graph selection" | Not used in method names in the manuscript — compliant with the todo constraint; leave as prose, avoid promoting it into a name |
| `TGCN-GSL`, `GCN-cGSL`, `TGCN-cGSL` | `paper/appendix/original_gsl_results.tex` (original results) | Keep — they are the *historical* method names; the appendix explicitly frames them as the original submission's variants. A half-sentence "naming follows the original submission" avoids confusion with the new names |

**Where a mapping note is genuinely needed:** (1) supplementary material or repo README: code-name ↔ manuscript-name table (5 rows, above); (2) the appendix original-results section: one sentence that GSL/cGSL names are inherited from the submission; (3) nowhere else — the main text is already consistent.

---

## 14. Critical Scientific Question: Can both result sets coexist?

**YES — with one logical bridge and one presentation rule.**

**The bridge (three steps, each backed by an existing artifact):**
1. *Original (appendix + 1 summary row):* replacing the dense physical graph with a sparse DAGMA-learned graph improves GCN (−21.6%/−24.7%) and T-GCN (−9.5%/−21.8%) — single seed. → Structure matters.
2. *Reframing control (Table `tab:oversmoothing`):* but a no-graph T-GCN (5.143) beats the dense graph (7.658) by 32.8% — much of the original gain is *anti-oversmoothing*. → Then what does the learned graph add beyond sparsity?
3. *Answer (multi-lag + Mix):* the learned graph encodes *lag-specific* functional dependencies (structurally distinct blocks, Table `tab:lag_stats`); exploiting them per-node/per-timestep yields 4.452±0.143 vs 5.234±0.090 (14.9%) — a genuine structure effect beyond sparsity, robust to seeds and parameters.

This is exactly "an evolution of the original GSL approach motivated by the limitations revealed by reviewer concerns and subsequent analysis" — and the user's suspicion is confirmed by the evidence, with the corrections listed in §8. The presentation rule: the original results must appear **in the main-text storyline** (as step 1, at least in summary), not only in an appendix; otherwise the paper reads as two studies.

**Narrative transition (drop-in, for §5.7's replacement):** *"The original submission's GSL/cGSL experiments (Appendix A) established that a sparse DAGMA-learned graph substantially outperforms the dense physical adjacency for both GCN and T-GCN, with a striking asymmetry: the symmetrized cyclic variant (cGSL) was optimal for the purely spatial GCN, while the acyclic DAG was optimal for the temporally equipped T-GCN. That asymmetry raised the question this revision answers: what does a single learned DAG actually encode? Two controls were missing. First, a no-graph baseline, which reveals that the dense physical graph itself is harmful (Section 5.1). Second, an explicit temporal decomposition of the learned structure, which the multi-lag formulation provides (Section 3.3) — converting a post-hoc interpretation of one ambiguous matrix into directly measurable, structurally distinct lag-specific graphs."*

**Essential original results:** the two GSL/cGSL tables (in full, appendix; in summary, main text) and the asymmetry finding. **Removable:** the original convergence curves from any main-text role (appendix only, already done), the original abstract's 21.6/24.7 framing, the causal rhetoric, the original Section-5 *interpretation* as a claim (retain as the hypothesis being tested). **No original experiment needs a re-run.**

**If the answer had to be NO, it would be because** the original numbers were produced under an incomparable protocol (different Laplacian normalization, seq_len=5 in the clean-room audit vs 12 in the revision) — but this does not apply, because the original tables are presented as the original submission's results with their own protocol stated, and the bridge relies on their *qualitative* finding (sparse ≫ dense), which was reproduced and strengthened by every subsequent stage.

---

## 15. Experiments: Rerun vs No-Rerun

### Experiments that actually need (re)running
**None.** The previously required Stage 29 regeneration is resolved: the artifacts were copied from the training machine and verified (§6.3, §12 item 18). Two optional — not required — cheap runs remain on the table:

| Experiment | Why | Cost | Notes |
|---|---|---|---|
| *(Optional, cheap)* 5-seed SZ-Taxi PH1 | SZ conclusions currently rest on seed 42 (+0.19%); 5 seeds would make the dataset-dependence claim as robust as the Los one | ~1 h | Would upgrade item 13 of §12 from single-seed to mean±std |
| *(Optional)* Los-loop PH 5–8 | Fully closes R1#7 ("longer horizons") | ~2–4 h | Only if the response letter promises it; otherwise the limitation wording suffices |

### Experiments that do NOT need rerunning
- All Stage 26 result JSONs (Los & SZ, PH1–4) — verified.
- Stage 26 Validation A (5 seeds), B (param control), C (lag ablation) — verified.
- Stage 27 **DAGMA W matrices** — valid and reused by Stage 29 (the forecasting part is dead, the structure analysis is salvageable).
- Original GSL/cGSL tables — retained as historical evidence with protocol stated.
- All figures currently in `paper/figures/` (regenerable from the above artifacts via `generate_figures*.py`).

---

## 16. Recommended Final Experimental Package (for the revised paper)

**Tables (7):**
1. `tab:oversmoothing` (+ optional Corr-K8 row) — Los, PH1, seed 42
2. `tab:lag_stats` — Los, L=3, τ=0.1
3. Original GSL/cGSL **summary row** (one line: "TGCN-GSL −21.8% Los, single seed, Appendix A") — main text
4. `tab:multiseed` — Los PH1, 5 seeds
5. `tab:param_control` — Los PH1
6. `tab:lag_ablation` + `tab:multiph` (may merge into one ablation table block)
7. `tab:los15min` (NEW, from verified `results/stage29_los15min/stage29_los15min_results.json`) + `tab:sz_multiph` (promoted from appendix)

**Figures (5–6 in main text):** fig1 (graphs), fig7 (lag structure b/c), fig3 (boxplot panel), fig5 (lag ablation), fig9 (predicted-vs-actual, optionally to diagnostics); fig2/fig4/fig6/fig8 → appendix or cut.

**Appendices:** A = original GSL/cGSL full tables + method + naming note; B = bibliometric (unchanged); C = convergence curves; D (additional diagnostics) = density stats, threshold sweep, DAGMA cost.

**Explicitly excluded:** Stage 27 forecasting numbers (any form) and the resolution-causal hypothesis.

---

## 17. Risks / Potential Reviewer Objections

| Risk | Severity | Mitigation |
|---|---|---|
| Result directories are gitignored (`results*/`) — the quantitative chain lives outside version control | **Medium** (provenance risk: it already happened once with Stage 29) | Commit the small JSON/CSV artifacts (not the `.npy`/log bulk) or archive them alongside the paper before submission |
| "Two methods, one paper" continuity objection | Medium | Use the §14 bridge; present original method as an evolutionary stage with its protocol stated; keep the asymmetry-as-puzzle framing |
| "Original gains were mostly anti-oversmoothing" | Medium | Make it a finding, not a weakness: the paper *discovers* this via the no-graph control; single-DAGMA-at-extreme-sparsity (5.213 vs 5.143) + multi-lag-30-edges (4.458) shows structure adds real value beyond sparsity |
| "Corr-K8 (1656 edges) vs your 30-edge graph" — why is DAGMA needed if dense learned graphs are bad? | Medium | The comparison is not DAGMA-vs-correlation at equal edges; the added value shown is the *multi-lag decomposition + mixing* (4.794→4.452 fixed-vs-mix), not the edge-selection algorithm. State this explicitly; consider adding PhysSp/Corr top-K sparse controls at 30 edges if a reviewer insists (small run) |
| 15-min PH=1 (15 min ahead) vs 5-min PH=1 (5 min ahead) apples-to-oranges | Medium | State physical horizons in hours everywhere; the Stage 30 prompt already prescribes the correct interpretation |
| n=5 seeds, no significance tests | Low | Report mean±std and 5/5 wins; state the limitation (already in revision_notes) |
| Single seed in original results vs 5 seeds in new | Low | Label protocols explicitly in the appendix |
| SZ-Taxi only 2 learned edges at τ=0.1 → is the method even "on" for SZ? | Low–Medium | Already honest in the draft; strengthen with the edge-count observation as the mechanism hypothesis |
| R1#7 (longer horizons) unresolved | Low | Limitations item 8 + optional PH 5–8 run |

---

## 18. Concrete Next Steps for Stage 32

1. ~~Recover or regenerate Stage 29~~ **DONE** — artifacts recovered from the training machine and verified in this revision of the audit (§6.3). Recommended housekeeping: preserve the small result artifacts in version control or an archive (see §17 risk 1).
2. Restructure `paper/sections/results.tex` per §11 (and Discussion §5.4/§5.5 adjustments), preserving the current text in git history.
3. Fix §5.7: replace the "()" placeholder citation, apply the §14 narrative transition, and delete the orphan "Predicted vs. actual" paragraph header (give it a proper subsection or move to diagnostics).
4. Reconcile the parameter-count bookkeeping (item 10, §12) and the 4.715/4.717 dual-source MultiGSL numbers (item 3).
5. Add the mapping-note sentence to Appendix A (GSL/cGSL = original naming) and the code↔manuscript name table to the repo README or supplementary.
6. Decide on the optional cheap runs (SZ 5-seed; PH 5–8) based on the response-letter promises.
7. Final pass: citation style (R1#12), "PH4 no improvement" phrasing for SZ, uniform 3-d.p. table formatting.
8. Update the point-by-point response letter to reference the new multi-seed and 15-min evidence explicitly.

---

## 19. Final Decision Matrix

| Question | Answer |
|---|---|
| Can original strong results be retained? | **YES** — verified against artifacts; no rerun needed |
| Can they be integrated into the main story? | **YES** — as Act I of the evolution narrative (summary row in main text, full tables in Appendix A) |
| Is the new architecture necessary to answer reviewers? | **YES** — R1#4 (direct temporal evidence), R1#6 (seeds), R1#5 (sparsity framing), R2#2/#3 are answered by the multi-lag formulation + MultiGSL-Mix + validation experiments |
| Is Stage 29 valid evidence? | **YES** — canonical pipeline verified in code; artifacts recovered from the training machine and every quoted statistic verified exactly against `stage29_los15min_results.json` (§6.3) |
| Does 15-min Los-loop support the temporal-resolution hypothesis? | **NO** — refuted: Los-15min improves *more* than Los-5min (+27.44% vs +13.33%/+14.9%) while SZ-15min shows ~0, so resolution alone cannot explain SZ's weakness; dataset-specific factors dominate (incl. graph learnability: SZ 2 edges vs Los-15min 32). Do not adopt the resolution-causal claim |
| Does SZ-Taxi remain a marginal-result dataset? | **YES** — verified: +0.19% (PH1, seed 42), +0.2–0.9% across PHs; only 2 learned edges at τ=0.1 |
| Do we need more expensive experiments before manuscript restructuring? | **NO** — every number in the proposed revised Results is verified against artifacts; only optional cheap runs (SZ 5-seed, PH 5–8) remain on the table |
| Which existing experiments must be rerun? | **None.** (Optional: SZ-Taxi 5-seed PH1; Los-loop PH 5–8) |
| Which existing experiments should be retained unchanged? | All Stage 26 result/validation JSONs and checkpoints; Stage 29 results JSON + DAGMA matrices; Stage 27 DAGMA W matrices (structure analysis only — never its forecasting numbers); original GSL/cGSL tables (single-seed, protocol-labeled); all current paper figures |
| Recommended next stage | **Stage 32: Manuscript restructuring** (Results per §11, narrative bridge per §14, terminology/numbering fixes per §12–13) — no longer conditional on any pending experiment. Optional sub-stages: SZ 5-seed validation; PH 5–8 extension |

---

*Rev. 2 audit complete. No repository files were modified by the auditor; the Stage 29/30 result folders were supplied by the author, and this report is the only artifact created by the audit.*
