# Stage 45.1 — Final Architecture of the New Manuscript

**Date:** 2026-09-10
**Scope:** planning only. No manuscript file edited or created except this report; `submitted_version`, `previous_revision`, code, results, figures, and tables untouched.
**Input:** `paper/gsl_stage45/stage45_new_manuscript_architecture.md` (Stage 45 blueprint), refined against the Stage 45.1 editorial principle.

**Governing principle adopted throughout:** the final manuscript presents **one self-contained scientific study** — the final experimental design and the final evidence. It contains **no development history**: no "historical results", no "first version / previous version / submitted version", no "re-baselining", no explanations of why numbers differ from any earlier document. All such material is marked **RESPONSE-LETTER ONLY** and lives exclusively in the confidential revision process. Where Stage 45 contained history-dependent elements, they are corrected below and the reason is stated.

---

## 1. Executive Summary

Stage 45 produced a sound scientific skeleton (question-first thesis, consumption-dissociation centerpiece, flat 8-section structure), but four of its elements violated the reader-coherence principle, and a closer pass on tables/figures shows the main text can be leaner without losing any scientific message. The corrections:

1. **Removed the historical-results appendix** (Stage 45 Appendix A). The manuscript now contains zero historical tables. The submitted results remain preserved in `paper/submitted_version/` for the revision process only. *(Stage 45.1 §1, §4 — the manuscript is not a revision document; history in the paper would both confuse readers and invite the exact "your numbers changed" question the response letter is designed to answer.)*
2. **Removed Contribution 5** ("corrected historical baseline / preserved first version's results"). The final manuscript has **4 contributions**, each a statement about traffic forecasting, not about the manuscript. *(Stage 45.1 §2 — manuscript history is process, not science.)*
3. **Purged all history-dependent language** from the scientific story; every such element is re-homed as RESPONSE-LETTER ONLY material (§10 claim policy, §11 reviewer map, §12 abstract strategy). *(Stage 45.1 §3.)*
4. **Table and figure diet:** the per-seed table (Stage 45 Table 3) is merged into the main results table's caption/statistics; the predicted-vs-actual figure (Stage 45 Figure 5) is demoted to the appendix as illustrative; Figures 1–2 are kept with tightened scope framing. Main text: **3 tables + 3 figures**. *(Stage 45.1 §10–11 — mean±std plus a distribution figure already carries the seed evidence; a dedicated per-seed table is redundant, and a qualitative time-series plot supports, but does not carry, the argument.)*
5. **Results ordering confirmed** with one sharpening: the single-graph subsection now explicitly ends with the pivot question that motivates multi-lag (§6), and the positive Los-loop result is given its own subsection (5.5) so it is not buried inside a validation discussion.
6. **Title re-confirmed** with scientific (not stylistic) rationale (§3).
7. **Appendix structure simplified to three**, each justified by reader value (§13): Additional Experimental Results; Graph Statistics and Implementation Details; Bibliometric Analysis.

Everything else in Stage 45 (central question, TOC skeleton, claim policy structure, canonical numbers, protocol basis) survives unchanged in substance.

**Verdict: READY FOR MANUSCRIPT WRITING** (§16).

---

## 2. Final Scientific Position

Unchanged in substance from Stage 45 §2; restated without any historical reference:

> **Central question:** When and how does learned graph structure help graph-based traffic forecasting — and how much of the benefit comes from the learned graph itself versus the mechanism that consumes it?

The study is presented as a controlled evaluation on two public benchmarks (Los-loop, 5-minute sampling, 207 sensors; SZ-Taxi, 15-minute sampling, 156 sensors), 12 forecasting configurations spanning physical, graph-free (identity-adjacency), learned contemporaneous, and learned multi-lag graphs with four consumption mechanisms, 4 prediction horizons, 5 seeds, plus matched-sparsity and capacity controls. Findings, in evidence order:

1. **Graph-free beats the physical graph decisively** (both datasets, both backbones, all horizons, 5/5 seeds) — the default reference graph is harmful at these operating points.
2. **A single learned contemporaneous graph does not close the gap**: GSL/cGSL improve on the physical graph in most configurations (T-GCN-GSL, T-GCN-cGSL, and GCN-cGSL win 5/5 seeds everywhere; GCN-GSL is the notable exception — it does not consistently beat even the physical graph) but never the graph-free control.
3. **Sparsity alone explains nothing**: matched 30-edge random and correlation controls are worse than no graph (Los-loop PH1).
4. **Multi-lag structure helps only through lag-specific consumption**: per-timestep consumption of the three lag graphs beats graph-free on Los-loop (5/5 seeds, all horizons); the *identical* graphs unioned into one static adjacency (GCN-MultiGSL) are worse than the physical graph — a clean dissociation locating the benefit in consumption.
5. **The benefit is conditional**: on SZ-Taxi every learned variant is within seed variability of graph-free; the learned multi-lag graphs there contain 2 edges vs Los-loop's 30.
6. **The learned graphs are descriptive statistical dependency structure** — static, fitted once from training data; only their *use* varies over time (gate).

Scope guards built into the position: no causal language; no universal-benefit claim; n=5 statistical qualifications; the 15-minute Los-loop experiment is a temporal-resolution *variant*, not a third dataset and not PH5–8 evidence.

---

## 3. Final Title Recommendation

**Re-confirmed: "Graph Structure or Graph Consumption? Dissecting the Benefit of Learned Multi-Lag Graphs for Traffic Forecasting"** (Stage 45 Candidate 2).

Scientific rationale (not stylistic): the paper's two genuinely novel elements are (i) the graph-free-baseline reversal and (ii) the same-graph consumption dissociation. The title names exactly these, and "dissecting" accurately describes a controlled decomposition rather than a claim of superiority. The alternatives remain viable but weaker:

- *"When Does Learned Graph Structure Help Traffic Forecasting? A Graph-Free Baseline Perspective"* — covers finding (i) but not (ii); the dissociation is the paper's most distinctive evidence, and a question title spends the strongest element on framing.
- *"From Physical Graphs to Multi-Lag Dependency Graphs: A Controlled Reassessment of Graph Structure Learning for Traffic Prediction"* — "reassessment" subtly frames the paper as a correction exercise, which is exactly the development-history flavor Stage 45.1 removes; rejected now on scientific-position grounds, not style.

No history appears in any candidate; the recommended title stands because it matches the final contribution set (§9), which is unchanged by the history purge.

---

## 4. Final Table of Contents

```
Abstract
Keywords

1  Introduction
2  Background and Related Work
3  Problem Formulation and Graph Learning Framework
   3.1  Forecasting task and model baselines
   3.2  Contemporaneous graph learning (GSL / cGSL)
   3.3  Multi-lag graph learning
   3.4  Graph consumption mechanisms
4  Experimental Setup
5  Results
   5.1  The physical graph is not a sound default
   5.2  A single learned graph does not close the gap
   5.3  Sparsity and capacity controls
   5.4  Multi-lag graphs: consumption matters more than the graph
   5.5  Gated mixing on Los-loop: the positive result
   5.6  Boundaries: temporal resolution and dataset dependence
6  Discussion
   6.1  Why dense graphs fail and what learned structure adds
   6.2  Structure versus consumption
   6.3  What the learned graph represents
7  Limitations
8  Conclusion

Appendix A  Additional Experimental Results
Appendix B  Graph Statistics and Implementation Details
Appendix C  Bibliometric Analysis
```

Changes from Stage 45 TOC, with reasons: subsection titles in §5 are now **message titles** (findings, not topic labels) — a reader of the section headers alone gets the paper's argument; "Gated mixing and multi-seed validation" is renamed "Gated mixing on Los-loop: the positive result" so the paper's affirmative finding is visible at TOC level; the historical appendix is gone (§1); appendices reduced from four to three.

---

## 5. Final Section-by-Section Plan

### 1 Introduction (no subsections)
- **Purpose:** pose the central question; establish that graph choice is an assumption rarely tested.
- **Evidence:** bibliometric hook (near-universal reliance on heuristic graphs; one paragraph, details in Appendix C); proximity-vs-dependency motivation; oversmoothing risk.
- **Reviewers:** R1-W1 (lean on bibliometrics); R2-2/R2-3 register-setting (no causal or adaptive language).
- **Reuse from submitted:** problem-framing paragraphs with canonical numbers.
- **NOT carried over:** improvement-percent claims, "hidden causal structure", any allusion to earlier versions of this study. The paper is introduced as a standalone evaluation.
- **CHANGE from Stage 45:** the Introduction no longer frames the paper as examining "the gains reported for learned graphs" in a way that implies prior internal claims; it frames the field's assumption as untested. Same evidence, self-contained framing.

### 2 Background and Related Work
- **Purpose:** minimal background + the related work the new story needs.
- **Reviewers:** R1-W2 (condense GCN/T-GCN), R1-W12 (citation style).
- **Reuse:** submitted overview prose trimmed; equation-free references to GCN/T-GCN formulations.
- **NOT carried over:** full derivation subsections.

### 3 Problem Formulation and Graph Learning Framework
- **Purpose:** define task, notation (A/W convention in text — R1-W3), and the three graph sources + four consumption mechanisms; separate *graph structure* from *graph consumption* conceptually.
- **Evidence:** Stage 44 canonical protocol verbatim (contemporaneous: `train_norm[0::PH]`, λ₁ 0.02/0.01, w_threshold 0.3, support |W|≥0.3, 28/8 edges; multi-lag: Z=[x(t−3)…x(t)], 828/624 vars, λ₁ 0.01, consumer |W|>0.1, blocks 12/3/15 → union 28, 0/0/2 → 2; consumption: fixed cyclic idx=(T−1−t) mod 3 with 0 params, Weighted 3 params, Mix per-node-per-timestep gate 4,419 params, GCN union static 0 params; cGSL=(A+Aᵀ)>0 binary symmetrization of the same artifact).
- **Reviewers:** R1-W3, R1-W4 + R1-Q4, R2-4 (cGSL defined here, before any result), R2-3 (static-graphs/adaptive-gate wording).
- **NOT carried over:** the submitted temporal reading of the contemporaneous DAG; the DAG/cyclic asymmetry explanation.
- **CHANGE from Stage 45:** none in substance; wording audited so no sentence motivates the constructions by reference to what "earlier work by the authors" did.

### 4 Experimental Setup
- **Purpose:** complete reproducible protocol (Stage 44 §12 nearly verbatim).
- **Key content:** datasets and sampling intervals; chronological 80/20 split; train-max normalization; seq_len 12; PH1–4; seeds 42–46; DAGMA determinism; Adam lr 1e-3 / wd 1e-4 / batch 128 / 50 epochs / hidden 64; per-backbone losses; the 12 canonical methods by exact name; RMSE primary + one-line metric definitions; **statistical reporting policy paragraph**: "mean ± sample standard deviation (ddof=1) over five training seeds; paired per-seed win counts; paired t-tests reported with the explicit caveat that with n=5 the exact Wilcoxon signed-rank test cannot fall below p=0.0625, so no definitive significance claims are made."
- **Reviewers:** R1-W6, R1-W10, R1-W5 (protocol transparency), R2-1 (unambiguous numbers).
- **CHANGE from Stage 45:** the "first version protocol-difference sentence" is **removed** — it was history. The Setup simply states the final protocol as *the* protocol. (The protocol-difference table is RESPONSE-LETTER ONLY, §11.)

### 6 Discussion
- **6.1 Why dense graphs fail and what learned structure adds** — oversmoothing reading of 5.1–5.2, tied back to §1's framing (R1-W8 tie-back).
- **6.2 Structure versus consumption** — the dissociation as interpretation centerpiece: identical 28-of-30-edge budget, opposite outcomes; Weighted ≈ Fixed (global weights insufficient); Mix adds per-node/per-timestep selection; the cGSL-vs-GSL contrast as the aggregation-compatibility miniature (robust for GCN, null for T-GCN). Language: "consistent with", "cannot be explained by the graph alone" — never causal.
- **6.3 What the learned graph represents** — lag-specific statistical dependency structure; the lag reading is consistent with the construction but not independently validated; graphs are static, only their use varies (R2-3).
- **NOT carried over:** the submitted Section 5 apparatus (its temporal-DAG premise is outside the final evidence set). Where the GSL/cGSL asymmetry is discussed, it is presented purely as an aggregation-compatibility observation of the current study.

### 7 Limitations
- **Content (evidence-scoped):** n=5 power (exact Wilcoxon floor 0.0625); sparsity/capacity controls are Los-loop PH1-only — **no sparsified-physical control and no λ/threshold sweep were performed**; two datasets; linear DAGMA assumptions; measured DAGMA runtimes (~16 min/PH for 207-variable contemporaneous fits, ~4 h for the 828-variable multi-lag fit); **the gate changes graph usage, not the learned graph itself — genuinely time-varying learned graphs remain future work**; PH ≤ 4 at 5-minute sampling (the 15-minute variant covers 15–60 min wall-clock but is a resolution variant, not PH5–8); backbone scope (GCN/T-GCN family).
- **CHANGE from Stage 45:** the R1-Q3 clarification ("gate changes usage, not the graph") is now explicit here, per Stage 45.1 §13.

### 8 Conclusion
- Five sentences answering the six sub-questions; contribution stated conditionally; future work list (longer horizons at 5-minute resolution, additional datasets, λ/threshold sensitivity, sparsified-physical controls, other backbones, time-varying graphs).
- **CHANGE from Stage 45:** the "reproduced the earlier baseline" closing clause is removed; the conclusion ends on the scientific conditional, not on continuity with anything.

---

## 6. Final Results Architecture

Stage 45's six-subsection progression is **confirmed as the clearest order**, with the Stage 45.1 checklist satisfied by construction:

| § | Title (message form) | Question answered | Evidence | Why this position |
|---|---|---|---|---|
| 5.1 | The physical graph is not a sound default | Q1 | Table 1 (T-GCN family): T-GCN worst in family everywhere (Los PH1: 7.88 vs NoSpatial 5.25); Table 2 (GCN family) supports | **Negative result first** — resets the frame before anything is claimed |
| 5.2 | A single learned graph does not close the gap | Q2 | T1 rows T-GCN-GSL/cGSL (5.86/5.82 vs 5.25); T2 rows GCN-GSL/cGSL (7.83/5.76 vs 4.88); cGSL≫GSL for GCN noted, mechanism deferred to 6.2 | Establishes **why multi-lag** is needed next: one static graph, learned or not, is insufficient |
| 5.3 | Sparsity and capacity controls | Q3 | Table 3 (controls): RandTop30 6.10±0.12, CorrTop30 5.39±0.10 vs MultiGSL 4.84±0.11, Mix 4.49±0.14, NoSpatial 5.25±0.19; capacity block | Kills the sparsity confound **before** the positive result, so 5.5 cannot be dismissed |
| 5.4 | Multi-lag graphs: consumption matters more than the graph | Q4a | Figure 1 (F1): GCN-MultiGSL union 9.78 vs T-GCN-MultiGSL 4.84 on identical graphs (union holds 28/30 edges); lag statistics 12/3/15 | **Dissociation prominent** — immediately after the controls, before the headline number, so the mechanism is in place |
| 5.5 | Gated mixing on Los-loop: the positive result | Q4b | Table 1 + Figure 2: Mix 4.49 vs NoSpatial 5.25 (PH1), wins 5/5 at all horizons, +14.5/+11.9/+9.2/+10.9% vs NoSpatial, +43.0/+37.6/+33.7/+32.3% vs T-GCN; Weighted ≈ Fixed | **Positive result not buried** — its own subsection, following the mechanism that explains it |
| 5.6 | Boundaries: temporal resolution and dataset dependence | Q5 | SZ null (Mix deltas ≤0.015 RMSE, wins 4/5–5/5, p 0.006–0.908; 2 vs 30 lag edges as candidate mechanism); 15-minute variant table (Appendix A, main-text summary sentence) | **Boundary explicit**; the variant is a labeled resolution observation, not a third dataset |

Two structural notes:
- **The 15-minute variant is summarized in one sentence in 5.6** ("resampling Los-loop to 15-minute intervals preserves and strengthens the relative improvement over the graph-free baseline at matched wall-clock horizons of 15–60 minutes; see Appendix A") with the full table in the appendix. Rationale: it is supporting robustness evidence, and keeping its table out of the main text removes any risk of it reading as a third dataset or as PH5–8 evidence. The main text never compares its PH values against 5-minute PH values.
- **No repetition:** 5.1/5.2 share Table 1 (one table, two messages); statistical qualifications are stated once in §4 and applied by reference; the cGSL null is a sentence in 5.2 plus mechanism in 6.2, never a separate subsection.

---

## 7. Final Main Tables

| ID | Stage 45 | Decision | Scientific message | Methods | Datasets | Horizons | Redundancy risk |
|---|---|---|---|---|---|---|---|
| **Table 1** | T1 | **KEEP** — main T-GCN results | Physical worst-in-family; GSL/cGSL lose to NoSpatial; MultiGSL family wins on Los | T-GCN, T-GCN-NoSpatial, T-GCN-GSL, T-GCN-cGSL, T-GCN-MultiGSL, T-GCN-MultiGSL-Weighted, T-GCN-MultiGSL-Mix | Los + SZ (two half-tables) | PH1–4 | Low — the paper's primary table |
| **Table 2** | T2 | **KEEP** — GCN family | Counterpart evidence; cGSL≫GSL for GCN; union-consumption failure (GCN-MultiGSL row) | GCN, GCN-NoSpatial, GCN-GSL, GCN-cGSL, GCN-MultiGSL | Los + SZ | PH1–4 | Low |
| **Table 3** | T4 | **KEEP** (renumbered from T4) | Sparsity/capacity confounds ruled out (Los-PH1-scoped) | RandTop30, CorrTop30, T-GCN-MultiGSL, T-GCN-MultiGSL-Mix, NoSpatial ref.; capacity block (NoSpatial h64/h74, Mix) | Los | PH1 | Low — unique evidence |
| **Table A1** | T5 (15-min) | **MOVE to Appendix A** | Temporal-resolution robustness (qualified observation) | NoSpatial, MultiGSL, Mix | Los-15min | PH1–4 (=15–60 min) | Would risk "third dataset" reading in main text |
| ~~Table 3 (Stage 45 per-seed table)~~ | T3 | **REMOVE as separate table — MERGE** | Per-seed consistency | — | — | — | **High**: mean±std in Table 1 + Figure 2's distributions already carry the seed evidence; a third presentation of the same 20 numbers overloads the main text. The 5/5 win counts appear in Table 1's caption; full per-seed values live in Appendix A |

**Main text: 3 tables.** Every retained table states its dataset/horizon scope and, where relevant, the control scope (Los PH1) in its caption. No table mixes method families without labeling; no table contains anything but final-protocol results.

---

## 8. Final Main Figures

| ID | Stage 45 | Decision | Question it answers | Notes | Redundancy risk |
|---|---|---|---|---|---|
| **Figure 1** | F3 (dissociation) | **KEEP — promoted to Figure 1; conceptual centerpiece** | Is the benefit in the graph or its consumption? | GCN-MultiGSL (union) 9.78 vs T-GCN-MultiGSL 4.84, identical graphs, union = 28/30 edges; "+Mix −0.35, 5/5 seeds" annotation; Los PH1 (PH2–4 inset optional). **No causal implication possible: the figure shows error bars and an annotation that the graphs are identical statistical-dependency estimates — no arrow-of-influence iconography, no "influence flow" rendering** | None — unique evidence |
| **Figure 2** | F4 (per-seed) | **KEEP** | Is the Los-loop gain consistent across seeds? | Box/strip plots, seeds 42–46, 4 panels PH1–4, methods NoSpatial/MultiGSL/Weighted/Mix; means must match Table 1 exactly; "n=5" annotated. **Adds beyond mean±std**: shows distribution overlap (or lack of it) and the SZ contrast panel goes to Appendix A | Low — complements, not repeats, Table 1 |
| **Figure 3** | F1 (graph structure) | **KEEP, reframed caption** | Does the learned graph differ from the physical one? | (a) physical adjacency (2833 entries incl. 207 self-loops); (b) multi-lag union (28 distinct off-diagonal); (c) degree distributions (12.7 vs 0.14). **Caption must say "learned statistical dependency structure" and must not use "influence"/"causal"/"flow" language; panels show structure, not mechanisms** | Low |
| **Figure 4** | F2 (sparsity bars) | **APPENDIX A** (was main) | Is it just sparsity? | Bar chart: RandTop30 6.10, CorrTop30 5.39, NoSpatial 5.25, MultiGSL 4.84, Mix 4.49 (Los PH1, mean±std). **Scope "Los-loop PH1 only" stated in the caption.** Demoted because Table 3 contains the identical numbers; kept in the appendix for reviewers who want the visual | **High vs Table 3** — that is why it moves |
| ~~Figure 5~~ | F5 (predicted-vs-actual) | **APPENDIX A — illustrative** | Do predictions track reality at node level? | 3 high-variance Los nodes, 100-step window, PH1 seed 42, Mix vs NoSpatial, r=0.95/0.93. **Judged illustrative, not load-bearing**: it supports readability of the aggregate results but carries no argument the tables don't; retained (not removed) because it answers R1-Q2 and aids qualitative assessment | Low in appendix |

**Main text: 3 figures** (dissociation, per-seed, graph structure). Every figure earns its place by answering a question the tables cannot; no figure is retained solely because a reviewer requested one (Figure 4/former-F5 satisfy requests *and* retain value; the demotions are for redundancy, not compliance).

---

## 9. Final Scientific Contributions

**Four contributions** (Stage 45's five, minus the history contribution, with the dataset boundary promoted into the conditional-benefit contribution):

1. **A graph-free-baseline reassessment of graph structure learning for traffic forecasting.** Across two public benchmarks, 12 configurations, 4 horizons, and 5 seeds, identity-adjacency (graph-free) models outperform both the physical road-network graph and learned contemporaneous graphs, showing that the standard reference graph — physical or learned-single — is not the right yardstick at these operating points. *Evidence: Tables 1–2.*
2. **A controlled dissociation between graph structure and graph consumption.** Identical multi-lag DAGMA graphs, consumed as a static union by a non-recurrent model or per-timestep by a recurrent model, produce opposite outcomes (9.78 vs 4.84 RMSE on Los-loop), with the union holding 28 of the 30 edges — the benefit is located in the consumption mechanism, not the edge set. *Evidence: Figure 1; Table 2.*
3. **A conditional positive result: gated per-timestep consumption of lag-specific learned graphs.** T-GCN-MultiGSL-Mix improves over the graph-free baseline on Los-loop at every horizon (up to +14.5% at PH1; 5/5 paired seeds; up to +43.0% over the physical baseline), while on SZ-Taxi — where the learned multi-lag graphs retain only 2 edges — all learned variants remain within seed variability of graph-free. *Evidence: Tables 1–2; Figure 2.*
4. **Matched-sparsity evidence that learned edge placement, not sparsity, drives the remaining gains.** At a matched 30-edge budget on Los-loop PH1, random and top-correlation graphs are worse than no graph while DAGMA-placed edges with per-lag consumption are better. *Evidence: Table 3.* (Judged strong enough to be a contribution rather than mere support because it is a controlled, budget-matched comparison — but it is stated with its single-cell scope.)

Traceability: every contribution cites final-protocol artifacts only; no promotional language; scopes (dataset/model/horizon/statistical) are part of each claim (see §10).

---

## 10. Final Claim Policy

### KEEP (directly supported; scope qualifiers attached)

| Claim | Dataset scope | Model scope | Horizon scope | Statistical qualification |
|---|---|---|---|---|
| Graph-free beats the physical graph | both datasets | GCN and T-GCN backbones | PH1–4 | 5/5 seeds, all 8 dataset×PH cells; large gaps |
| Single contemporaneous learned graphs beat the physical graph but not graph-free | both | GSL/cGSL, both backbones | PH1–4 | T-GCN-GSL, T-GCN-cGSL, GCN-cGSL: 5/5 seeds vs physical everywhere; GCN-GSL: not consistently better than GCN-physical (aggregation-compatibility failure, §6.2); 0/8 cells better than NoSpatial for all four |
| Per-timestep consumption of multi-lag graphs beats graph-free on Los-loop | **Los-loop only** | T-GCN-MultiGSL and Mix | PH1–4 | 5/5 paired seeds; paired-t p≤0.0016 with n=5 caveat |
| Mix improves over fixed per-timestep consumption | **Los-loop** (SZ null) | Mix vs MultiGSL | PH1–4 | 5/5 seeds; 0.35–0.40 RMSE (Los), ≤0.023 (SZ) |
| Sparsity alone insufficient | **Los-loop PH1 only** | T-GCN backbone | PH1 | 5 seeds; matched 30-edge budget |
| Learned graphs are static statistical dependency estimates | both | all learned variants | — | descriptive statement |

### REFRAME (require careful qualification)

| Original framing | Final framing |
|---|---|
| "GSL improves traffic prediction" | "Replacing the physical adjacency — learned or identity — consistently reduces error; the learned contemporaneous graph recovers only part of the gap that removing the graph recovers" |
| "cGSL is better than GSL" | "Symmetrization matters specifically when downstream aggregation is symmetric (GCN: robust advantage); for the temporal model the difference is within seed noise" |
| "Learned graphs are beneficial" | "The benefit is dataset-dependent and most evident when lag-specific graphs are consumed in a temporally aligned manner" |
| "DAGMA discovers temporal dependencies" | "Multi-lag DAGMA yields lag-specific dependency estimates; the construction is consistent with a lag-structured reading, which is not independently validated here" |
| "Longer horizons favor graph structure" | "At 15-minute sampling (15–60 min ahead) the relative improvement over graph-free is larger than at 5-minute sampling (5–20 min); horizon counts are not comparable across sampling intervals" |
| "The model adapts to traffic" | "The learned graphs are static; the gate varies their use per node and timestep" |

### RETIRE (disappear entirely from the manuscript)

1. "GSL is better than NoSpatial" — refuted in all 8 cells.
2. "T-GCN-GSL / GCN-cGSL is the best method" — both lose to graph-free.
3. "Learned graphs are universally beneficial" — refuted on SZ and by GCN-MultiGSL on Los.
4. Any causal claim ("hidden causal structure", "discovers causal relationships") — never supported; acyclicity is a fitting regularizer.
5. "The contemporaneous graph is a temporal graph" — it is fitted on simultaneous snapshots.
6. "The graph adapts to changing traffic" — static graphs.
7. Definitive significance statements from n=5 — descriptive statistics and win counts only.
8. **RESPONSE-LETTER ONLY (not in the manuscript):** all comparisons to previously submitted numbers, protocol-difference tables, "re-baselining" narrative, reproduction-agreement percentages, and any history of how the study evolved. These are prepared for the confidential response letter and appear nowhere in the paper.

---

## 11. Final Reviewer-to-Manuscript Map (final manuscript only)

**Reviewer 1**

| Comment | Where addressed (final MS) | Evidence used | Status |
|---|---|---|---|
| W1 bibliometrics underused | §1 hook + Appendix C | bibliometric analysis | Fully addressed (existing material) |
| W2 background too long | §2 condensed | — | Fully addressed (editorial) |
| W3 A→W notation | §3.1 in-text convention | — | Fully addressed (editorial) |
| W4 temporal interpretation asserted | §3.2 vs §3.3 explicit constructions; §6.3 | Stage 44 protocol | Fully addressed (existing evidence) |
| W5 sparsity confound | §5.3 + Table 3 | matched-budget controls | **Partially addressed**: the 30-edge version of the question is answered; **the sparsified-physical control and λ/threshold sweep were not run** → Limitations + future work |
| W6 seeds/variance/significance | §4 policy paragraph + Tables 1–2 + Figure 2 | Stage 40/41 | Fully addressed (existing evidence) |
| W7 longer horizons | §5.6 (one sentence) + Appendix A variant table | 15-min variant | **Partially addressed**: PH5–8 at 5-minute resolution were **not run**; the 15-minute experiment is a temporal-resolution analysis, not longer-horizon evidence at 5-minute resolution → Limitations + future work |
| W8 repetitive results | §5 message-titled subsections; §6.1 tie-back | — | Fully addressed (editorial + structure) |
| W9 limitations | §7 | — | Fully addressed |
| W10 metric definitions | §4 one-liners | — | Fully addressed (editorial) |
| W11 dense convergence plots | Appendix A (two-panel summary) | Stage 26 diagnostics | Fully addressed (editorial) |
| W12 citation style | global pass | — | Fully addressed (editorial) |
| Q1 physical-vs-learned visualization | Figure 3 | existing artifact | Fully addressed |
| Q2 predicted-vs-actual | Appendix A figure | existing artifact | Fully addressed (as supplementary, per §8 decision) |
| Q3 time-varying graphs plan | §7 Limitations + §8 future work; **the gate changes graph usage, not the learned graph itself** | — | **Limitation / future work** — genuinely time-varying learned graphs were not implemented or run |
| Q4 direct contemporaneous-vs-lagged evidence | §3.2 vs §3.3 structural contrast; §5.4 | the two constructions | Fully addressed (existing evidence) |

**Reviewer 2**

| Comment | Where addressed | Status |
|---|---|---|
| R2-1 abstract numbers | §12 abstract strategy (canonical numbers only) | Fully addressed |
| R2-2 causal claims | retired everywhere; Figure 3 caption + §6.3 give the descriptive reading | Fully addressed |
| R2-3 static vs adaptive wording | §3.4/§6.3 verified wording | Fully addressed (editorial) |
| R2-4 cGSL defined late | §3.2 defines cGSL before any result appears | Fully addressed |
| R2-5 convergence plots | Appendix A two-panel summary | Fully addressed (editorial) |
| R2-6 typo | final pass | Fully addressed (editorial) |

**RESPONSE-LETTER ONLY material** (never in the manuscript): the protocol-difference table (submitted-protocol batch/normalization/seeds vs final protocol); the submitted-results reproduction percentages; any "the earlier draft said X" statement; the historical appendix that Stage 45 proposed. The response letter explains the evolution; the paper presents the study.

---

## 12. Final Abstract Strategy

Describes **only the final study**, in this order (not the abstract text itself):

1. **Problem (1–2 sentences):** graph-based traffic forecasters rely on an adjacency — physical or learned — that is rarely tested against a graph-free control.
2. **Design (2 sentences):** controlled evaluation on two public benchmarks (Los-loop 5-min; SZ-Taxi 15-min): 12 configurations spanning physical, identity-adjacency, learned contemporaneous (GSL/cGSL), and learned multi-lag graphs under four consumption mechanisms; 4 horizons; 5 seeds; matched-sparsity and capacity controls.
3. **Finding 1 — the reversal (1–2 sentences):** graph-free models outperform the physical graph and all single learned contemporaneous graphs on both datasets; learned edges beat the physical graph but never the graph-free control.
4. **Finding 2 — the mechanism (1–2 sentences):** the only configuration that improves over graph-free consumes *lag-specific* multi-lag graphs *per timestep*; feeding the identical graphs as a static union is markedly worse, so consumption matters as much as learning. Headline number: **T-GCN-MultiGSL-Mix on Los-loop: +14.5% RMSE over graph-free at PH1 (5/5 paired seeds), up to +43.0% over the physical baseline across horizons.**
5. **Finding 3 — the boundary (1 sentence):** on SZ-Taxi all learned variants remain within seed variability of graph-free; at a matched edge budget, random and correlation graphs do not reproduce the Los-loop gain, so sparsity alone is not the explanation.
6. **Closing (1 sentence):** learned multi-lag graphs are a conditional tool — worthwhile where the learned dependency structure is informative, dependent on temporally aligned consumption — with learned structure characterized as descriptive statistical dependency, no causal claims.

**Numerical guards:** only canonical five-seed numbers; **no Stage 29 (15-minute) number appears as a main result** — the variant is either absent from the abstract or appears solely as a qualified clause ("a 15-minute-sampling variant of Los-loop shows the same pattern at longer wall-clock horizons") with no percentage; the improvement figures above are vs NoSpatial (14.5%) and vs physical T-GCN (43.0%) and are never mixed with any other pipeline's values; no absolute-RMSE cross-dataset comparison.

---

## 13. Appendix Policy

| Appendix | Content | Kept? | Reader-value rationale |
|---|---|---|---|
| **A — Additional Experimental Results** | Full MAE tables; per-seed RMSE values; SZ-side per-seed distributions; sparsity-control bar chart; 15-minute variant table + per-seed data; two-panel convergence summary; predicted-vs-actual figure | **Yes** | Every item is final-protocol evidence that a replicator or reviewer needs but the narrative does not; nothing here is historical |
| **B — Graph Statistics and Implementation Details** | Physical/learned graph statistics (Los 2833 entries incl. 207 self-loops, off-diag density 0.0616 over N(N−1), symmetric; SZ 532 (incl. 156 self-loops), off-diag density 0.0156, not exactly symmetric; learned 28/8 contemporaneous; 12/3/15+union 28 and 0/0/2+union 2 multi-lag); density-convention statement; threshold-semantics table; parameter counts; DAGMA runtime table | **Yes** | Directly serves R1-W5/R1-Q1 reproducibility and the sparsity-confound audit; all final-study material |
| **C — Bibliometric Analysis** | The bibliometric methodology and findings | **Yes** | Independent motivating analysis (R1-W1 praised it); self-contained, no history |
| ~~Stage 45 Appendix A (historical results)~~ | — | **REMOVED** | Violates the reader-coherence principle; the paper must not contain historical comparisons (§1) |
| ~~Stage 45 Appendix D (supplementary)~~ | — | **MERGED into A** | One supplementary-results appendix is cleaner than two |

No appendix exists to preserve earlier-version material; all three earn their place by reader value.

---

## 14. Writing Sequence

Recommended order for the writing stage (each step verifiable against this architecture):

1. **§3 Problem Formulation and Graph Learning Framework** — hardest, most load-bearing; use the Stage 44 §12 canonical protocol text as the skeleton; everything downstream references it.
2. **§4 Experimental Setup** — largely assembled from Stage 44 §12 + the statistical-policy paragraph; contains no interpretive risk.
3. **Tables 1–3 + Figure 1–2 data** — generate from `gsl_stage41/stage41_summary.csv` and per-run JSONs with the Stage 43 §11.9 equivalence targets (means match to 4 decimals; figure annotations match tables exactly).
4. **§5 Results** — in subsection order 5.1→5.6; each subsection written against its table/figure, applying the §10 claim policy verbatim.
5. **§6 Discussion** — after Results, so the mechanism prose binds to the actual numbers.
6. **§2 Background, §1 Introduction, §8 Conclusion** — written after the body exists (framing sections last, so no framing promise exceeds the evidence).
7. **§7 Limitations** — from §5/§11 of this report.
8. **Abstract + title finalization** — last; check against §12 guards.
9. **Appendices A–C** — assemble from existing artifacts; verify no historical content slipped in.
10. **Global passes:** terminology audit (§10 avoid-list), citation style (R1-W12), figure-caption causality check, response-letter drafting (separately, using §10 RETIRE item 8 + §11).

---

## 15. Remaining Decisions

Non-blocking, for the authors during writing:

1. **Title** — recommended in §3; author sign-off needed.
2. **Figure 1 inset** — whether the dissociation figure shows PH1 only or PH1–4 as small multiples (PH1 recommended; PH2–4 identical in direction).
3. **Abstract mention of the 15-minute variant** — include the qualified clause or omit entirely (recommend: omit; the abstract stays purely canonical).
4. **Response-letter drafting order** — must follow, not precede, the manuscript (§14 step 10).
5. **Contribution 4 prominence** — whether the matched-sparsity result stays a numbered contribution or is folded into Contribution 1's evidence (recommend: keep as Contribution 4; it is the strongest controlled evidence in the paper).

None of these blocks writing; all have recommended defaults.

---

## 16. Final Verdict

**READY FOR MANUSCRIPT WRITING.**

The architecture is internally coherent: one self-contained study, one central question, a Results section whose subsection titles state the argument, 3 tables + 3 figures in the main text with zero redundancy, four traceable contributions, a claim policy with explicit scopes, a reviewer map that never cites historical material, and an appendix set justified purely by reader value. All history-dependent elements from Stage 45 are removed or re-homed as RESPONSE-LETTER ONLY; all numbers trace to Stage 40–44 canonical artifacts; no claim requires evidence that does not exist.

---

*End of Stage 45.1 report. No manuscript, code, figure, table, or result file was modified or created except this report.*
