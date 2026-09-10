# Stage 45 — New Manuscript Architecture Blueprint

**Date:** 2026-09-10
**Scope:** planning only. No manuscript file was edited, created (except this report), or replaced; `previous_revision` and `submitted_version` untouched; no experiments; no code or result changes.
**Source of truth, per the Stage 45 priority order:** (1) `paper/submitted_version/sn-article.tex` — structure inspected directly (Section 2 "Background" with 2.1 traffic-prediction overview / 2.2 GCN / 2.3 T-GCN; Section 3 "The Proposed Method: Estimating the Adjacency Matrix with Graph Structure Learning" with 3.1 static-graph/temporal integration / 3.2 GSL approaches; Section 4 "Experimental Results" (datasets/config/metrics/ablations); Section 5 "Analysis of Learned Graph Structures: Reconciling DAG Assumptions with Cyclic Traffic Networks" (5.1 spatial-vs-temporal / 5.2 why-DAGs-for-T-GCN / 5.3 why-cyclic-for-GCN); Section 6 Conclusion; Appendix A bibliometric methodology). (2) `paper/Reviewers-comments.txt` (R1: 12 weaknesses + 4 questions; R2: 6 comments). (3) Stage 40 canonical results, Stage 41 statistics, Stage 42 audit + claim reconciliation, Stage 43 results-structure plan, Stage 44 protocol reconciliation (canonical protocol text, Section 12 there). (4) `previous_revision` inspected as *material* (prose templates, figures, response letter) but explicitly NOT as the structural basis.

---

## 1. Executive summary

The new manuscript is a **conceptual reconstruction**, not a revision of `previous_revision`. Its thesis changes from the submitted version's "learned graphs improve traffic prediction" to an evidence-first question: **when does learned graph structure help traffic forecasting, and is the benefit in the graph or in how it is consumed?** The Stage 40 canonical evidence forces this reframing: the graph-free baselines (T-GCN-NoSpatial / GCN-NoSpatial) beat both the physical graph and every contemporaneous learned graph on both datasets; the only learned configuration that beats graph-free is T-GCN-MultiGSL(-Mix) — and only on Los-loop; and the GCN-union vs T-GCN-per-timestep dissociation on *identical* graphs localizes the benefit in the consumption mechanism.

The blueprint: 8 main sections (deliberately flat — no nested subsection chains beyond one level), 4 main tables, 5 main figures, 4 appendices. The submitted manuscript contributes its motivating framing, its bibliometric analysis, and its datasets/protocols as historical material; its Section 5 "DAG vs cyclic" analysis is retired in its old explanatory role (the Stage 33/44 contemporaneous re-derivation removed its premise) but survives as a symmetrization-mechanism observation. The previous revision contributes prose templates and reviewer-response scaffolding but not structure.

**Verdict: READY FOR MANUSCRIPT WRITING** (Section 15; three writing-time decisions flagged, none blocking).

---

## 2. Recommended central research question

> **When and how does learned graph structure help graph-based traffic forecasting — and how much of the benefit comes from the learned graph itself versus the mechanism that consumes it?**

Sub-questions the Results section answers in order:

1. **Q1 — Is the physical graph a sound default?** No: graph-free models beat it decisively on both datasets (5/5 seeds, all horizons). The submitted paper's reference point is the *problem*, not the baseline.
2. **Q2 — Does learning a single contemporaneous graph recover the loss?** No: T-GCN-GSL/cGSL and GCN-GSL/cGSL beat the physical graph but never beat NoSpatial. Learning recovers only a fraction of the gap that removing the graph recovers.
3. **Q3 — Is the benefit merely sparsity?** No: matched-budget controls (RandTop30, CorrTop30) on Los-loop PH1 are worse than no graph; only DAGMA-placed edges with per-lag consumption win (5/5 seeds). Los-PH1-scoped.
4. **Q4 — Does multi-lag structure help?** Only with lag-specific consumption: T-GCN-MultiGSL beats NoSpatial on Los (5/5 seeds, all horizons); Mix adds a further 0.30–0.42 RMSE; GCN-MultiGSL — the *same graphs* unioned into one static adjacency — is catastrophic on Los (worse than the physical graph).
5. **Q5 — Is the effect universal?** No: on SZ-Taxi every learned variant lands within seed noise of NoSpatial (Mix deltas ≤ 0.015 RMSE; wins 4/5–5/5, p 0.064–0.908). The benefit is dataset-dependent, tracking the informativeness of the fitted graphs (30 vs 2 lag edges).
6. **Q6 — What does the learned graph represent?** Lag-specific statistical dependency structure; descriptive only; static, fitted once; nothing adapts except the consumption gate.

Claims explicitly **not** retained (Stage 42 reconciliation): general GSL improvement; GSL > NoSpatial; cGSL generally better than GSL; single learned graph sufficiency; universal MultiGSL benefit; causal relationships; adaptation to changing traffic.

---

## 3. Recommended title

**Candidates:**

1. *When Does Learned Graph Structure Help Traffic Forecasting? A Graph-Free Baseline Perspective* — leads with the question; signals the NoSpatial finding.
2. *Graph Structure or Graph Consumption? Dissecting the Benefit of Learned Multi-Lag Graphs for Traffic Forecasting* — leads with the mechanism dissociation; most distinctive.
3. *From Physical Graphs to Multi-Lag Dependency Graphs: A Controlled Reassessment of Graph Structure Learning for Traffic Prediction* — continuity with the submitted title; emphasizes reassessment.

**Recommended: Candidate 2 — "Graph Structure or Graph Consumption? Dissecting the Benefit of Learned Multi-Lag Graphs for Traffic Forecasting."** It names the paper's actual contribution (the dissociation), is honest about scope (multi-lag, not GSL in general), and pre-empts the reviewer expectation that the paper claims universal GSL superiority. Candidate 1 is the fallback if the authors prefer a question-form title.

---

## 4. Complete new table of contents

```
Abstract
Keywords
1  Introduction
2  Background and Related Work
3  Problem Formulation and Graph Learning Framework
   3.1  Forecasting task and baselines
   3.2  Contemporaneous DAGMA graphs (GSL / cGSL)
   3.3  Multi-lag DAGMA graphs
   3.4  Graph consumption mechanisms (T-GCN-MultiGSL family; GCN union)
4  Experimental Setup
5  Results
   5.1  The baseline landscape: physical vs graph-free
   5.2  Single-graph learned structure does not beat no graph
   5.3  Ruling out sparsity and capacity
   5.4  Multi-lag structure and the consumption dissociation
   5.5  Gated mixing and multi-seed validation
   5.6  Temporal resolution and dataset dependence
6  Discussion
   6.1  Why the physical graph fails; what the learned graph adds
   6.2  Structure vs consumption: the mechanism
   6.3  What the learned graph represents
7  Limitations
8  Conclusion
Appendix A  Historical results from the first version of the manuscript
Appendix B  Graph statistics and protocol details
Appendix C  Bibliometric analysis methodology
Appendix D  Supplementary results and diagnostics
```

---

## 5. Detailed section plan

Format per Stage 45 §3: purpose · question answered · key evidence · reviewer concerns · reusable old material · material NOT carried over.

### 1 Introduction (no subsections)

- **Purpose:** motivate the study as a *controlled reassessment*: proximity graphs are the default, learned graphs the popular alternative, but the field lacks the graph-free baseline that decides whether learning helps.
- **Question:** the central question (Section 2 above).
- **Evidence:** bibliometric hook (784 articles; near-universal reliance on heuristic graphs); the physical-proximity-vs-functional-dependency motivation.
- **Reviewers:** R1-W1 (lean on bibliometrics in the intro — keep one paragraph + Appendix C); R2-2/R2-3 (no causal/hidden-structure language; no "adapts to traffic" language — the intro sets the cautious register).
- **Reusable from submitted:** paragraphs 1–3 of its Introduction (problem framing, proximity-imperfect-proxy argument, oversmoothing motivation) with numbers updated to canonical (2833 edges/207 nodes, density conventions per Stage 44).
- **NOT carried over:** the submitted Introduction's claims that GSL "improves RMSE by up to 21.6%/24.7%" (R2-1 misattribution; retired by Stage 42), "explicit insights into hidden causal structure", and the Conclusion-forward promise of adaptive graphs.

### 2 Background and Related Work

- **Purpose:** compress the submitted Section 2 (2.1 overview + 2.2 GCN + 2.3 T-GCN, ~180 lines) into a short background; add the related-work the new story requires (graph-free/NoSpatial ablations in GNN practice, oversmoothing literature, multi-graph/multi-view traffic models, GSL methods).
- **Question:** what must a reader know to follow the design?
- **Evidence:** standard GCN/T-GCN equations reduced to referenced statements (R1-W2: "condense both to a short paragraph with equation references" — the submitted subsections 2.2–2.3 are exactly what the reviewer flagged).
- **Reviewers:** R1-W2; R1-W12 (citation style applied throughout this section).
- **Reusable from submitted:** the overview-of-approaches prose (2.1) trimmed; citation base from `MyReferences.bib`.
- **NOT carried over:** full GCN/T-GCN derivation subsections; the "Problem Definition" commented-out block.

### 3 Problem Formulation and Graph Learning Framework

- **Purpose:** one place that defines task, notation (resolve the A/W switch in main text, not a footnote — R1-W3), and the three graph sources plus the consumption mechanisms. This is the section the submitted paper lacked: it never distinguished graph *source* from graph *consumption*.
- **Question:** what exactly is learned, from what input, and how is it consumed?
- **Evidence:** Stage 44 canonical protocol verbatim (Section 12 of that report): contemporaneous construction (input `train_norm[0::PH]`, λ₁ 0.02/0.01, w_threshold 0.3, support |W|≥0.3, 28/8 edges) vs multi-lag construction (Z=[x(t−3)…x(t)], 828/624 vars, λ₁ 0.01, consumer |W|>0.1, blocks 12/3/15 → union 28; 0/0/2 → 2); consumption: fixed cyclic mapping idx=(T−1−t) mod 3 (0 params), Weighted (3 params), Mix (per-node per-timestep gate, 4,419 params), GCN union (static, 0 params); cGSL = (A+Aᵀ)>0 binary symmetrization of the same artifact.
- **Reviewers:** R1-W3 (A→W notation); R1-W4 + R1-Q4 (explicit input construction; contemporaneous vs lagged distinction made structurally); R2-4 (cGSL defined before any result); R2-3 (static-graphs/adaptive-gate wording exactly as Stage 44 verified); terminology restrictions (no "Adaptive", no causal language).
- **Reusable:** submitted Section 3's DAGMA equations (h(W), augmented Lagrangian, thresholding) — these remain correct; previous_revision §3.3/3.4 prose as drafting templates.
- **NOT carried over:** the submitted 3.1 claim that the learned DAG encodes "j at time t predicting i at time t+1" (contradicted by the contemporaneous re-derivation); the Section 5 premise that the DAG/cyclic split explains the GCN/T-GCN asymmetry temporally.

### 4 Experimental Setup

- **Purpose:** complete, reproducible protocol — Stage 44 §12 text nearly verbatim.
- **Evidence:** datasets (Los 207 nodes/5-min; SZ 156/15-min), chronological 80/20 split, train-max normalization, seq_len 12, PH1–4, seeds 42–46, DAGMA determinism, batch 128 / lr 1e-3 / wd 1e-4 / 50 epochs / hidden 64, losses (mse_with_regularizer for T-GCN, mse for GCN), the 12 canonical methods with exact names, evaluation metrics (RMSE primary; one-line definitions per R1-W10), the explicit statement that main-text numbers are five-seed mean±sample-std (ddof=1) with no definitive significance claims (n=5; exact Wilcoxon floor 0.0625), and the "first version of the manuscript" protocol-difference sentence (single run, full-series max, batch 64 — absolute values not comparable).
- **Reviewers:** R1-W6 (seeds/variance); R1-W10 (metrics); R1-W5 partially (protocol transparency); R2-1 (numbers now unambiguous); R1-W9 (protocol-level limitations stated here, expanded in Section 7).
- **Reusable:** submitted 4.1/4.2 dataset descriptions (updated: SZ sampling interval stated as 15 min); previous_revision §4 reproducibility prose (determinism audit sentence).
- **NOT carried over:** the submitted single-seed configuration tables as *protocol*; the "16 subplots" convergence figure references.

### 5 Results

See Section 6 of this report for the full architecture and Section 7–8 for tables/figures.

### 6 Discussion

- **6.1 Why the physical graph fails; what the learned graph adds** — oversmoothing reading of Q1/Q2, tied back to the Introduction's proximity-vs-dependency framing (R1-W8's tie-back requirement). Reuse submitted Figure 2 (spatial–temporal motivation) if it survives the terminology check.
- **6.2 Structure vs consumption: the mechanism** — the dissociation as the paper's interpretation centerpiece: identical 28-edge budget, opposite outcomes; Weighted ≈ Fixed (global weights insufficient); Mix adds per-node/per-timestep selection; cGSL-vs-GSL as the miniature version (symmetrization matters only for the symmetric GCN aggregator — Stage 41: robust for GCN, null for T-GCN). No causal language; "consistent with", "cannot be explained by the graph alone".
- **6.3 What the learned graph represents** — lag-specific statistical dependency structure; the lag interpretation is *consistent with* the construction and the ablation, but not independently validated (honest statement); the static-graph/adaptive-use distinction restated (R2-3).
- **NOT carried over:** the entire submitted Section 5 ("Reconciling DAG Assumptions with Cyclic Traffic Networks") as an explanatory apparatus — its temporal-DAG premise is retired; its GSL/cGSL asymmetry observation survives inside 6.2 as a mechanism footnote, with the temporal reading removed.

### 7 Limitations

- **Purpose:** R1-W9's dedicated limitations, evidence-scoped per Stage 41 §12: n=5 power (Wilcoxon floor); sparsity controls are Los-PH1-only (no SZ control, no sparsified-physical control, no λ/threshold sweep); two datasets; linear-DAGMA assumptions; DAGMA scalability (measured runtimes: ~16 min/PH 207-var single-graph, ~4 h 828-var multi-lag); static graphs; PH≤4 at 5-min sampling (15-min variant covers up to 60 min wall-clock but is a variant, not PH5–8); backbone scope (T-GCN/GCN family only).
- **NOT carried over:** any limitation text implying sensitivity was measured, or that longer horizons were evaluated at 5-minute resolution.

### 8 Conclusion

- **Purpose:** answer the six sub-questions in five sentences; state the contribution as a *conditional* (learned multi-lag graphs with lag-specific consumption help where the graphs are informative — Los-loop — and the consumption mechanism is at least as important as the graph); future work (longer horizons, more datasets, λ/threshold sweeps, sparsified-physical controls, other backbones, time-varying graphs as *future* work).
- **NOT carried over:** submitted Conclusion's causal/hidden-structure sentence, universal-improvement phrasing, and "the graph adapts" future-work framing.

### Appendix plan

- **A — Historical results from the first version of the manuscript:** the submitted paper's four result tables verbatim, under the "contemporaneous re-baselining" label (Stage 43 §8.1 wording); includes the cGSL symmetrization formula derivation. R2-4's "cGSL defined where evaluated" is satisfied because the only live GSL/cGSL definitions are in Section 3.
- **B — Graph statistics and protocol details:** physical-graph stats (Los 2833 entries incl. 207 self-loops, off-diag density 0.0616 over N(N−1), symmetric; SZ 532, no self-loops, 0.0220, not exactly symmetric), learned-graph stats (28/8 contemporaneous; 12/3/15+union 28 / 0/0/2+union 2 multi-lag), density convention statement, per-PH refit vs PH-independence note, parameter counts, threshold semantics table (Stage 44 §2.2).
- **C — Bibliometric analysis methodology:** carried over essentially intact from the submitted Appendix (it was praised by R1; only the main-text hook stays in Section 1).
- **D — Supplementary results and diagnostics:** full canonical MAE tables; SZ-side per-seed box plots (contrast panel); per-seed paired-difference tables; compact convergence summary (two panels — R1-W11/R2-5); the Stage 29 15-min per-seed table.

---

## 6. Results architecture (Stage 45 §5 progression evaluated)

The Stage 45 nine-block progression is **merged to six subsections** — blocks 5.1/5.2 fold together (the physical-vs-graph-free and GSL-vs-NoSpatial comparisons are one table and one message), and 5.7 (statistics) dissolves into 5.5 and a Setup paragraph, because a separate statistics subsection invites repetition. Reordering rationale: consumption dissociation (new 5.4) must come *with* multi-lag introduction, not after it, since it is the same experiment; the 15-minute variant moves into 5.6 with dataset dependence so both "boundary" results share one subsection.

| New subsection | Stage 45 block | Content and evidence | Status of proposed block |
|---|---|---|---|
| 5.1 Baseline landscape: physical vs graph-free | 5.1 + 5.2 | Table 1 (T1): 7 T-GCN methods, both datasets, PH1–4, five-seed mean±std. Messages: physical worst-in-family everywhere (T-GCN 7.88 vs NoSpatial 5.25, Los PH1); GSL/cGSL beat physical, lose to NoSpatial (5.86/5.82 vs 5.25). | merged |
| 5.2 Single-graph learned structure | (inside 5.1) + GCN half | GCN family table (T2): GCN-NoSpatial 4.88 vs GCN-cGSL 5.76 vs GCN 8.14 (Los PH1); cGSL≫GSL for GCN (mechanism observation, full discussion deferred to 6.2). | merged/absorbed |
| 5.3 Sparsity and capacity controls | 5.3 | Table 4 (T4) + Figure 2: RandTop30 6.10±0.12, CorrTop30 5.39±0.10 vs MultiGSL 4.84±0.11, Mix 4.49±0.14, NoSpatial 5.25±0.19; parameter-matched control; Los-PH1 scope stated. | kept |
| 5.4 Multi-lag structure + consumption dissociation | 5.4 + 5.5 merged | Figure 3 (F3, the dissociation): GCN-MultiGSL (union) 9.78 vs T-GCN-MultiGSL 4.84 on identical graphs; lag-graph statistics (12/3/15); union = 28 of 30 edges — edge content cannot explain the gap. | merged; moved earlier than proposed |
| 5.5 Gated mixing and multi-seed validation | 5.6 + 5.7 | Table 3 (T3) per-seed (5/5 wins at all horizons); Mix vs MultiGSL vs Weighted; Figure 4 box plots; statistics (paired-t p≤0.0015; Wilcoxon floor stated once here). | merged |
| 5.6 Temporal resolution and dataset dependence | 5.8 + 5.9 | Table 5 (T5, 15-min variant, labeled); SZ boundary: Mix deltas ≤0.015, wins 4/5–5/5, null verdict; graph-informativeness explanation (30 vs 2 edges); "PH4@15-min = 60 min ≠ PH4@5-min = 20 min" warning. | merged; partly Discussion-linked |

The main contribution is made obvious by construction: **Table 1 delivers the negative results that reset the frame (5.1–5.2); Figure 3 delivers the mechanism (5.4); Table 3 + Figure 4 deliver the positive result with its statistical honesty (5.5); Table 5 delivers the boundary (5.6).** Nothing is listed without a message; every subsection answers one of Q1–Q6.

---

## 7. Main tables

All values Stage 40 canonical five-seed mean±std (sample std, ddof=1) except T4 (controls: five seeds for graph rows, labeled single-seed capacity block) and T5 (15-min variant). RMSE primary; MAE to Appendix D.

| ID | Title | Purpose | Methods | Datasets | PH | Metrics | Placement | Reviewer concern |
|---|---|---|---|---|---|---|---|---|
| **Table 1** | T-GCN family results (RMSE) | Primary evidence: physical-graph failure; contemporaneous GSL/cGSL verdict vs NoSpatial; MultiGSL result | T-GCN, T-GCN-NoSpatial, T-GCN-GSL, T-GCN-cGSL, T-GCN-MultiGSL, T-GCN-MultiGSL-Weighted, T-GCN-MultiGSL-Mix | Los-loop + SZ-Taxi (two half-tables or 9-col table) | 1–4 | RMSE mean±std; improvement block vs T-GCN (Los) and vs NoSpatial | **Main** | R1-W6, R2-1 |
| **Table 2** | GCN family results (RMSE) | Consumable counterpart evidence; cGSL≫GSL for GCN; union-consumption failure | GCN, GCN-NoSpatial, GCN-GSL, GCN-cGSL, GCN-MultiGSL | Los-loop + SZ-Taxi | 1–4 | RMSE mean±std | **Main** | R1-W4/Q4, R2-4 |
| **Table 3** | Multi-seed paired validation (Los-loop) | Per-seed honesty for the headline claim | T-GCN-NoSpatial, T-GCN-MultiGSL, T-GCN-MultiGSL-Weighted, T-GCN-MultiGSL-Mix | Los-loop | PH1 (per-seed) + win counts PH1–4 in caption | per-seed RMSE + mean±std; **no p-values in table** | **Main** | R1-W6 |
| **Table 4** | Sparsity and capacity controls (Los-loop, PH1) | Rule out sparsity/capacity confounds | RandTop30, CorrTop30, T-GCN-MultiGSL, T-GCN-MultiGSL-Mix (+ NoSpatial reference; capacity block: NoSpatial h64/h74, Mix) | Los-loop | 1 | RMSE mean±std (graph rows); single-seed labeled (capacity) | **Main** | R1-W5 |
| **Table 5** | Temporal-resolution variant (Los-loop at 15-minute sampling) | Longer *wall-clock* horizons; sampling robustness | T-GCN-NoSpatial, T-GCN-MultiGSL, T-GCN-MultiGSL-Mix | Los-15min | 1–4 (=15–60 min) | RMSE mean±std + improvement row | **Main**, with provenance/scope label (Stage 44 §8) | R1-W7 (partially) |
| Appendix D.1 | Full MAE tables | Metric-robustness check | all 12 | both | 1–4 | MAE mean±std | Appendix | R1-W6 |

Anti-proliferation rules: Weighted appears only in T1/T3 (its null is one sentence elsewhere); cGSL rows appear in T1/T2 (the symmetrization null is one sentence in 6.2); no per-PH GCN table beyond T2; the 480-run raw grid stays in the repository.

---

## 8. Main figures

| ID | Title | Purpose | Content | Datasets | Methods shown | Placement | Reviewer concern |
|---|---|---|---|---|---|---|---|
| **Figure 1** | Physical vs learned graph structure | Make "proximity ≠ dependency" concrete | (a) physical adjacency (2833 edges incl. 207 self-loops); (b) multi-lag union (28 distinct off-diagonal); (c) degree distributions (mean degree 12.7 vs 0.14) | Los-loop | graphs only | **Main** | R1-Q1 |
| **Figure 2** | Sparsity controls at matched edge budget | Visual confound control | Bar chart, mean±std, 5 bars: RandTop30 6.10, CorrTop30 5.39, NoSpatial 5.25, MultiGSL 4.84, Mix 4.49 | Los PH1 | 5 | **Main** | R1-W5 |
| **Figure 3** | The consumption dissociation | The mechanism centerpiece | GCN-MultiGSL (union) 9.78 vs T-GCN-MultiGSL 4.84 (+ physical refs 8.14/7.88; "+Mix −0.35, 5/5 seeds" annotation); note "identical 28-edge budget (union = 28/30 edges)" | Los-loop (PH1; PH2–4 optional inset) | 4–6 selected | **Main** — **new figure** | R1-W4, R1-Q4 |
| **Figure 4** | Per-seed distributions | Statistical honesty + consistency | Box/strip plots over seeds 42–46, 4 panels PH1–4: NoSpatial, MultiGSL, Weighted, Mix (means must match T3 exactly; "n=5" annotated) | Los-loop; SZ contrast panel → Appendix D | 4 | **Main** (regenerated from Stage 40 per-seed JSONs) | R1-W6, R2-5 |
| **Figure 5** | Predicted vs actual time series | Qualitative complement | 3 high-variance Los nodes, 100-step window, PH1 seed 42; Mix vs NoSpatial; r=0.95 vs 0.93 | Los-loop | 2 | **Main** | R1-Q2 |

Demoted to Appendix D: convergence curve summary (two panels, R1-W11/R2-5), lag-ablation bar chart (Stage 26 evidence; single dataset — mention in 6.2 text, keep table in appendix), SZ box plots. Not included: the submitted Figures 5–8 per-epoch grids (16 subplots — explicitly rejected by both reviewers).

---

## 9. Claims: keep / reframe / retire

Based explicitly on `gsl_stage42/stage42_claim_reconciliation.md` (Stage 42) and Stage 44 verification.

### A. KEEP (supported as-is, with canonical numbers)

1. **"The dense physical road-network graph is a harmful default for these backbones."** 5/5 seeds, 8/8 dataset×PH cells; NoSpatial beats physical everywhere (Stage 41 finding 1).
2. **"T-GCN-MultiGSL-Mix is the strongest configuration on Los-loop."** 42.98/37.60/33.72/32.34% vs T-GCN; 5/5 paired wins vs NoSpatial at all horizons (paired-t p≤0.0015, reported with the n=5 caveat).
3. **"Sparsity alone does not explain the Los-loop PH1 gain."** Matched-budget controls; scope stated in the same sentence.
4. **"Traffic dependencies are temporally heterogeneous in the learned structure."** Lag blocks are structurally distinct (12/3/15 edges; low overlap) — as a *descriptive* statement about the fitted graphs.
5. **"The learned graphs are static; the gate adapts their use per node and timestep."** Verified wording (Stage 44); R2-3 resolution.

### B. REFRAME

1. **"GSL improves over the physical graph"** → *"Replacing the physical road-network adjacency — learned or identity — consistently reduces error; the learned contemporaneous graph recovers only a fraction of the gap that removing the graph recovers."* (Stage 42 A; direction survives, interpretation does not.)
2. **"cGSL is superior to GSL"** → *"Symmetrizing the learned adjacency matters specifically when downstream aggregation is symmetric (GCN), and is immaterial for the temporal model."* (Stage 42 C: robust for GCN — 7/8 cells 5/5 wins; null for T-GCN.)
3. **"Learned graphs are beneficial"** → *"The benefit of learned graph structure is dataset-dependent and is most evident when lag-specific graphs are consumed in a temporally aligned manner."* (Stage 42 F replacement.)
4. **"DAGMA discovers temporal dependencies"** → *"Multi-lag DAGMA yields lag-specific dependency estimates; the construction is consistent with a lag-structured reading, which we do not independently validate."* (Stage 44 §3; Stage 41 claims-to-avoid.)
5. **"Longer horizons favor structure"** → *"At 15-minute sampling (PH = 15–60 minutes ahead) the relative improvement is larger than at 5-minute sampling; PH counts are not comparable across sampling intervals."* (Stage 44 §8.)
6. **"GCN benefits more from cyclic graphs; T-GCN from DAGs"** (submitted Section 5) → *"The GCN-family benefit of symmetrization is an aggregation-compatibility effect; the original temporal explanation is retired because the learned graph is contemporaneous."* (Stage 33/44 re-derivation.)

### C. RETIRE

1. **"GSL is better than NoSpatial"** — refuted in all 8 cells (Stage 42 B).
2. **"T-GCN-GSL is the best method" / "GCN-cGSL is the best method"** — both lose to NoSpatial everywhere (Stage 42 D, E).
3. **"Learned graphs are universally beneficial"** — refuted on SZ and by GCN-MultiGSL on Los (Stage 42 F).
4. **Any causal claim** ("hidden causal structure", "discovers causal relationships") — never supported; DAGMA's acyclicity constraint is a fitting regularizer (Stage 41 §10, Stage 44 check E).
5. **"The graph adapts to changing traffic"** — static graphs, fitted once (R2-3).
6. **"The learned DAG is a temporal graph"** — the contemporaneous construction sees simultaneous snapshots (Stage 44 §5).
7. **The submitted abstract's "21.6% / 24.7%"** — misattributed (R2-1) and superseded.
8. **Definitive significance statements** — n=5; exact Wilcoxon floor 0.0625; descriptive statistics + win counts only.

---

## 10. New contributions (3–5, each traceable)

1. **A graph-free-baseline reassessment of graph structure learning for traffic forecasting**, showing on two public benchmarks (12 methods × 4 horizons × 5 seeds) that graph-free models outperform both physical-proximity and learned contemporaneous graphs — relocating the earlier reported gains from "learned structure" to "a harmful reference graph". *Evidence: Table 1/2; Stage 42 §2.*
2. **A controlled dissociation between graph structure and graph consumption**: identical multi-lag DAGMA graphs, consumed by a static union (GCN-MultiGSL) or per-timestep (T-GCN-MultiGSL), produce opposite outcomes on Los-loop (9.78 vs 4.84 RMSE; union holds 28/30 edges). *Evidence: Figure 3; Stage 41 §6.*
3. **T-GCN-MultiGSL-Mix**, a per-node, per-timestep gating mechanism over lag-specific learned graphs, which — together with fixed per-timestep consumption — is the only learned configuration that beats the graph-free baseline, and only where the learned graphs are informative (Los-loop; +14.2% PH1, 5/5 seeds; SZ-Taxi null). *Evidence: Tables 1/3; Stage 41 findings 2/3.*
4. **Matched-budget controls isolating the contribution of learned edge placement**: random and correlation-placed 30-edge graphs are worse than no graph, while DAGMA-placed edges with per-lag consumption are better — sparsity per se is not the operative variable (Los-loop PH1). *Evidence: Table 4; Stage 41 §8.*
5. **A fully reproducible five-seed protocol and a corrected historical baseline**, preserving the first version's results as a clearly-labeled contemporaneous re-baselining whose relative claims reproduce (21.8% → 21.9% mean) under the canonical protocol. *Evidence: Stage 44 canonical protocol; Stage 42 §2; Appendix A.*

Style check: no "novel breakthrough", no "significantly outperforms" without qualification, no causal verbs; every claim carries its dataset/scope.

---

## 11. Abstract strategy (one paragraph, not the abstract itself)

The new abstract should: open with the field-default problem (proximity graphs; learned graphs as the popular alternative; the missing graph-free baseline); state the controlled design (two datasets, 12 methods including graph-free and identity-adjacency controls, 4 horizons, 5 seeds, matched-sparsity and capacity controls); deliver the three findings in evidence order — (i) graph-free models beat physical and contemporaneous learned graphs on both datasets, so single-graph GSL does not recover what removing the graph does; (ii) the only benefit appears with lag-specific *consumption* of multi-lag DAGMA graphs — T-GCN-MultiGSL-Mix on Los-loop (+14.2% over graph-free at PH1, 5/5 seeds; up to +43.0% over the physical baseline), with the identical-graph union dissociation showing consumption matters as much as learning; (iii) the benefit is dataset-dependent — on SZ-Taxi all learned variants are within seed noise of graph-free — and sparsity alone does not explain the gain (matched-budget controls). Close with the defensible takeaway (when graphs are informative, how they are consumed matters at least as much as how they are learned) and the explicit absence of causal claims. All numbers five-seed canonical; the 15-minute result appears only with its sampling qualifier; the historical single-run study is mentioned, if at all, as an appendix-preserved precursor.

---

## 12. Reviewer-to-manuscript map

**Reviewer 1**

| Comment | Where addressed | Evidence | New experiment? | Existing sufficient? | Editorial only? |
|---|---|---|---|---|---|
| W1 bibliometrics underused | §1 hook + Appendix C | submitted bibliometric analysis | No | Yes | Partly |
| W2 GCN/T-GCN background too long | §2 condensed | — | No | Yes | Yes |
| W3 A→W notation | §3.1 (in-text convention at W's introduction) | — | No | Yes | Yes |
| W4 temporal interpretation asserted | §3.2/3.3 (explicit constructions); §6.3 | Stage 44 §2/§3; Stage 33 re-derivation | No | Yes | No |
| W5 sparsity confound | §5.3 + Table 4 + Fig 2; Limitations (no sparsified-physical control, no λ sweep) | Stage 32 controls (canonical) | **Partially** — reviewer's sparsified-physical control and λ sweep remain unrun; scoped in Limitations | Yes, for the 30-edge version of the question | No |
| W6 seeds/variance/significance | §4 + Tables 1/3 + Fig 4 | Stage 40/41 | No | Yes | No |
| W7 longer horizons | §5.6 + Table 5 (15-min variant); Limitations (PH5–8 unrun) | Stage 29 (5 seeds) | **Partially** — PH5–8 at 5-min not run; 15-min variant is proxy evidence, labeled | Yes, as variant evidence | No |
| W8 Section 5 repetitive | §5 architecture (one message per subsection); §6.1 tie-back | Stage 43 §3 mapping | No | Yes | No |
| W9 limitations | §7 | Stage 41 §12 | No | Yes | No |
| W10 metrics spelled out | §4 one-line definitions | — | No | Yes | Yes |
| W11 dense convergence plots | Appendix D (two-panel summary only) | Stage 26 diagnostics | No | Yes | Yes |
| W12 citation style | global pass | — | No | — | Yes |
| Q1 physical-vs-learned visualization | Fig 1 | existing fig1 | No | Yes | No |
| Q2 predicted-vs-actual | Fig 5 | existing fig9 artifact | No | Yes | No |
| Q3 time-varying plan | §8 future work (qualitative, with measured DAGMA runtimes) | Stage 41/44 runtime data | No (explicitly future) | Yes | No |
| Q4 direct contemporaneous-vs-lagged evidence | §3.2 vs §3.3 structural contrast + §5.4 | the two constructions ARE the contrast (Stage 40.1/44) | No | Yes | No |

**Reviewer 2**

| Comment | Where addressed | Evidence | New experiment? | Existing sufficient? | Editorial only? |
|---|---|---|---|---|---|
| R2-1 abstract numbers | new abstract (canonical numbers only); Appendix A labels the historical derivation | Stage 42 §2 | No | Yes | No |
| R2-2 causal/hidden-structure claims | retired everywhere (§9.C.4); Fig 1 + §6.3 give the structural (not causal) reading | Stage 41/44 | No | Yes | No |
| R2-3 static vs adaptive wording | §3.4/§6.3 exact Stage 44-verified wording | Stage 44 §4 | No | Yes | Yes |
| R2-4 cGSL defined late | §3.2 defines cGSL before any result; historical cGSL tables in Appendix A | — | No | Yes | No |
| R2-5 unreadable convergence plots | Appendix D two-panel summary | — | No | Yes | Yes |
| R2-6 typo | final pass | — | No | — | Yes |

**Partially addressed requests (explicit):** R1-W5 (no sparsified-physical control; no λ/threshold sweep), R1-W7 (no PH5–8 at 5-minute resolution), and — inherent to the evidence — no third dataset (dataset-dependence rests on one contrast). All three are scoped in §7 Limitations and the response letter must state them as such.

---

## 13. Submitted-to-new migration plan

- **KEEP (from submitted, valid as-is):** Introduction's problem framing (paragraphs 1–3, numbers updated); DAGMA optimization equations (§3.2); dataset descriptions (§4.1, corrected: SZ = 15-min); RMSE/MAE definitions (condensed); bibliometric appendix (Appendix C); reference base; the motivation figure (Fig 2 of submitted, pending terminology check).
- **REWRITE (meaning changed):** Abstract (new claims/numbers); contributions list (Section 10 here); all Results narrative (Section 6 here); Discussion (submitted Section 5's DAG/cyclic explanation → mechanism story; "temporal DAG" → contemporaneous + multi-lag distinction); Conclusion (conditional claims); the GSL/cGSL asymmetry explanation (aggregation-compatibility, not temporality).
- **REMOVE:** submitted Section 5 as an explanatory apparatus (its premise retired); "hidden causal structure" sentence (Intro/Conclusion); "21.6%/24.7%" abstract claims; "adapts to changing traffic" item; universal-improvement phrasing; single-seed results as main-text evidence.
- **MOVE TO APPENDIX:** submitted main result tables (→ Appendix A, verbatim, labeled "Historical results from the first version of the manuscript"); bibliometric details (→ C, keep a §1 hook); per-epoch convergence grids (→ dropped from main; two-panel summary in D); the original W_est provenance discussion (→ A protocol notes).
- **ADD (required by reviewers/new story):** NoSpatial baselines in all main tables (T1/T2); GCN family table (T2); consumption-dissociation figure (F3 — new); per-seed table + box plots (T3/F4, from Stage 40 per-seed JSONs); sparsity/capacity controls table (T4); 15-minute variant table (T5, labeled); graph-statistics appendix (B); limitations section (§7); reviewer-response mapping (this report §12 → response letter); canonical protocol paragraph (Stage 44 §12 → §4).

---

## 14. One-paragraph final paper story

> Graph-based traffic forecasters almost universally assume that some graph — usually the physical road network, more recently a learned adjacency — should mediate spatial aggregation; this paper asks, under a fully controlled five-seed protocol, whether that assumption earns its keep. The answer begins negatively: replacing the physical road network with *no graph at all* improves accuracy on both benchmarks for both backbones, and learning a single contemporaneous dependency graph with DAGMA — while clearly better than the physical graph — never closes the gap to the graph-free control, so the improvements reported in the first version of this manuscript largely reflect a harmful reference graph rather than extracted structure. Matched-budget controls then show that sparsity alone cannot explain what gains remain. The positive result is specific: when DAGMA is applied to explicitly lag-stacked inputs, yielding separate graphs per temporal lag, and those graphs are consumed *per timestep* by a recurrent forecaster — with a per-node, per-timestep gate selecting the mixture (T-GCN-MultiGSL-Mix) — forecasting improves over the graph-free baseline consistently across seeds on the Los-loop highway network, while feeding the *same* learned graphs to a non-recurrent model as one union graph is markedly worse, indicating that how a learned graph is consumed matters at least as much as how it is learned. That benefit, however, is a property of the data as much as the method: on the SZ-Taxi urban network, where the fitted graphs retain almost no edges, every learned variant remains within seed variability of the graph-free control, and we report this boundary rather than a universal claim. We conclude that learned multi-lag graphs are a conditional tool — worthwhile where lag-specific dependency structure is informative, dependent on temporally aligned consumption, and neutral elsewhere — and we make no causal interpretation of the learned structure, which we characterize throughout as descriptive statistical dependency estimated once from training data.

---

## 15. Open issues / decisions required before writing

1. **Title choice** (Section 3) — author decision; Candidate 2 recommended.
2. **Figure regeneration capacity** — F3 must be created and F4 regenerated from Stage 40 per-seed JSONs; `previous_revision/generate_figures*.py` provide tooling (code changes belong to the writing stage).
3. **Figure 2-of-submitted reuse** — verify the spatial–temporal motivation figure contains no retired-claim annotation before reuse.
4. **Response-letter update** — this report's §12 map must be converted into the point-by-point letter with canonical numbers (Stage 43 §8.3 list) when the manuscript exists; the letter should not precede the manuscript.
5. **Journal formatting** — the submitted version uses `sn-jnl` (sn-mathphys-num); the new manuscript should reuse the class and the modular-section workflow (`previous_revision`'s build pattern) without importing its structure.

**Final check (Stage 45 §14):** based primarily on `submitted_version` structure ✓ (its TOC, framing, equations, appendix analyzed directly); incorporates Stage 40–42 evidence ✓ (every table/figure/claim mapped to canonical aggregates); respects Stage 44 protocol ✓ (Section 3/4 use the canonical protocol text); does not resurrect retired claims ✓ (Section 9C); no causality without evidence ✓; NoSpatial treated as the decisive baseline ✓ (Sections 2, 6); contemporaneous GSL distinguished from multi-lag GSL ✓ (Sections 3, 5); structure separated from consumption ✓ (Sections 2, 6.2); Los-loop benefit and SZ-Taxi boundary explicit ✓ (Sections 5, 6, 14); all reviewer comments mapped ✓ (Section 12, partial items flagged); no unvalidated results required ✓ (every number exists in Stage 40/41/29/32 artifacts per Stage 44 §9).

**VERDICT: READY FOR MANUSCRIPT WRITING.**

---

*End of Stage 45 report. No manuscript file was edited or created except this report.*
