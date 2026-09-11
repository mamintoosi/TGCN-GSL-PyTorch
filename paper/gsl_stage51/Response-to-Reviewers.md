# Response to Reviewers

**Manuscript:** Graph Structure Learning for Traffic Prediction  
**Journal:** International Journal of Data Science and Analytics  
**Revision type:** Major revision (complete rewrite of the experimental design and evidence base)

---

## General response

We thank the editor and both reviewers. Their comments exposed three structural issues in the submitted version: (1) the absence of a graph-free baseline, making it impossible to tell whether learned graphs help or merely beat a harmful physical graph; (2) an asserted temporal reading of a contemporaneous DAGMA fit; and (3) missing multi-seed statistics and incomplete scope statements.

In this revision we rebuild the study around those points. We do **not** present the revision as a patch of the old tables. The new manuscript is a self-contained controlled evaluation with:

1. A **graph-free identity control** (T-GCN-NoSpatial / GCN-NoSpatial) as the primary reference.
2. An **explicit multi-lag DAGMA construction** and a family of **consumption mechanisms**, so structure and consumption are separate design dimensions.
3. A **complete five-seed protocol** (seeds 42–46), sample standard deviations, paired win counts, and an honest $n=5$ significance caveat.
4. **Matched-sparsity and capacity controls** (scoped and labeled).
5. A **Limitations** section and claim language limited to statistical dependency (no causal claims).

**Headline result of the revision.** Graph-free models outperform the physical graph and all single learned contemporaneous graphs on both datasets. The only configuration that improves over the graph-free control on Los-loop consumes lag-specific multi-lag graphs per timestep (T-GCN-MultiGSL-Mix): up to **14.5% relative RMSE reduction vs NoSpatial at PH1** (5/5 paired seeds) and up to **43.0% vs the physical T-GCN baseline** across horizons. On SZ-Taxi all learned variants remain within seed variability of graph-free. Sparsity alone does not explain the Los-loop gain.

Below we address each comment. Locations refer to the revised manuscript (Section numbers as in the new draft).

---

## Reviewer 1

### W1 — Bibliometric analysis underused

**Response.** We keep a single motivating observation in the Introduction and move the full methodology and findings to Appendix C. The main-text hook is one sentence, not a second results section.

**Location.** §1; Appendix C.

### W2 — GCN/T-GCN background too long

**Response.** Background is condensed to short paragraphs with citations; full layer-by-layer derivations are removed. Architecture properties needed for the experimental design are stated in §3.1.

**Location.** §2; §3.1.

### W3 — A→W notation switch unexplained

**Response.** The main text now states the convention where it is introduced: $\mathbf{A}$ is the binary adjacency consumed by the backbone; $W$ is the continuous DAGMA coefficient matrix. The switch is no longer a footnote.

**Location.** §3 opening and §3.1.

### W4 — Temporal interpretation of the DAG asserted

**Response.** This was the most serious technical criticism, and we agree. The contemporaneous construction (§3.2) is fitted on simultaneous snapshots `train_norm[0::PH]` and is **not** described as a temporal graph. Temporal organization enters only through the **multi-lag construction** (§3.3): $Z=[x(t-3),\ldots,x(t)]$, lag blocks, consumer threshold. We also retire the old Section 5 apparatus that inferred a temporal DAG from the GSL/cGSL split. Remaining asymmetries are discussed as aggregation-compatibility observations, not as proof of temporal DAG semantics.

**Location.** §3.2 vs §3.3; §6.2; §6.3.

### W5 — Sparsity / oversmoothing / missing controls

**Response.** We add (i) a graph-free control, which is the cleanest anti-oversmoothing baseline; (ii) matched-sparsity controls at a 30-edge budget on Los-loop PH1 (RandTop30, CorrTop30 vs DAGMA lag graphs); (iii) structure visualization (physical vs multi-lag union and degree histograms). We do **not** claim a full hyperparameter sweep or a sparsified-physical control — those are stated as limitations and future work.

**Location.** §5.1, §5.3; Fig. 3; Table 3; §7.

### W6 — No seeds / variance / significance

**Response.** All main results are five-seed means with sample standard deviations (ddof=1). We report paired per-seed win counts and paired $t$-tests for reference, and explicitly state that with $n=5$ the exact Wilcoxon test cannot fall below $p=0.0625$, so we make no definitive significance claims.

**Location.** §4.7; Tables 1–2; Fig. 2.

### W7 — Horizons only up to 4

**Response.** We keep PH≤4 at native sampling as the canonical grid. We add a labeled **temporal-resolution variant** of Los-loop at 15-minute sampling covering 15–60 minutes wall-clock, with the explicit warning that PH indices are not comparable across sampling intervals. PH5–8 at 5-minute resolution were **not** run and are listed as future work / limitation, not as implied evidence.

**Location.** §5.6; Table 4; §7.

### W8 — Section 5 repetitive; weak tie-back

**Response.** Results are restructured into message-titled subsections (5.1–5.6). Discussion (§6.1) ties the physical-graph failure back to the Introduction’s proximity-vs-dependency framing. The old long “DAG/traffic paradox” section is removed.

**Location.** §5; §6.1.

### W9 — Missing limitations

**Response.** Dedicated §7 covers $n=5$ power, control scope, two datasets, linear DAGMA, runtime, static graphs vs gated use, horizon/sampling limits, and backbone scope.

**Location.** §7.

### W10 — Metric definitions too long

**Response.** Metrics are one-liners in §4.6 (RMSE primary, MAE secondary, de-normalized scale, full-batch test evaluation).

**Location.** §4.6.

### W11 — Dense convergence grids

**Response.** Per-epoch 16-subplot grids are removed from the main text. A compact two-panel convergence summary is reserved for Appendix A (not load-bearing for the argument).

**Location.** Appendix A (to be assembled); main text uses final RMSE with error bars (Figs. 1–2).

### W12 — Citation style

**Response.** Global pass to `\citep`/`\citet` consistent with the journal style.

**Location.** Throughout.

### Q1 — Side-by-side physical vs learned graph

**Response.** Figure 3 shows physical adjacency, multi-lag union, and degree distributions for Los-loop, with captions that describe statistical dependency structure (no causal language).

**Location.** Fig. 3; §5.1.

### Q2 — Predicted vs actual time series

**Response.** A qualitative predicted-vs-actual figure for selected high-variance Los-loop nodes is retained in Appendix A (illustrative, not load-bearing).

**Location.** Appendix A.

### Q3 — Time-varying graph plan

**Response.** We clarify that the Mix gate changes **how static graphs are used**, not the edge set. Genuinely time-varying graphs (e.g., sliding-window refits) remain future work; we do not claim incremental updates in this paper.

**Location.** §3.4; §6.3; §7; §8.

### Q4 — Direct contemporaneous vs lagged evidence

**Response.** The two constructions are now explicit in §3.2–§3.3 (input matrices, thresholds, edge counts). The consumption comparison (§5.4, Fig. 1) is the empirical counterpart: identical multi-lag artifacts, static union vs per-timestep use. We state the backbone confound explicitly.

**Location.** §3.2–3.3; §5.4; Fig. 1.

---

## Reviewer 2

### Abstract number mix-up (21.6% / 24.7%)

**Response.** Thank you. Those figures are removed from the abstract. The revised abstract quotes only five-seed canonical numbers with clear references: **14.5% vs NoSpatial** and **43.0% vs physical T-GCN** on Los-loop, with the SZ boundary stated in the same abstract.

**Location.** Abstract; §5.5.

### Causal / “hidden causal structure” claims

**Response.** All causal language is retired. Graphs are described as static statistical dependency estimates; acyclicity is a fitting regularizer. Figure captions and §6.3 restate this explicitly. We no longer claim “explicit insights into hidden causal structure.”

**Location.** §3.2–3.3; §6.3; Fig. 3 caption; Abstract.

### Static graph vs “adapts to changing traffic” inconsistency

**Response.** Agreed; the old list item was inconsistent with the implementation. The revised text states that graphs are fitted once; only consumption (Mix) varies per node and timestep.

**Location.** §3.4; §6.3.

### cGSL formula defined too late

**Response.** cGSL is defined in §3.2 as a binary symmetrization of the same stored GSL artifact, before any result appears.

**Location.** §3.2.

### Convergence plots unreadable

**Response.** 16-subplot grids removed from the main text (see R1-W11).

**Location.** Appendix A.

### Typo “avergae”

**Response.** Fixed in the rewritten Results section (the typo no longer exists in the new draft).

---

## Note on relationship to the submitted version

The submitted manuscript remains archived for the confidential revision record. Absolute RMSE values are **not** compared across the submitted single-run protocol and the new five-seed protocol in the scientific text. Differences in protocol (batch size, normalization scope, seeds, graph provenance) are explained in this response letter only, not in the paper body, so that the manuscript reads as one coherent study.

---

## Checklist of requested actions

| Request | Addressed? | Where |
|---------|------------|--------|
| Bibliometrics | Yes | §1, App. C |
| Condense background | Yes | §2 |
| A/W notation | Yes | §3 |
| Temporal DAG evidence | Yes | §3.2–3.3, §6 |
| Sparsity/degree/controls | Yes | §5.1, §5.3, Fig. 3, §7 |
| Multi-seed statistics | Yes | §4.7, Tables, Fig. 2 |
| Longer horizons | Partial (variant + limitation) | §5.6, §7 |
| Less repetition / tie-back | Yes | §5, §6.1 |
| Limitations | Yes | §7 |
| Metric one-liners | Yes | §4.6 |
| Convergence figures | Moved to appendix | App. A |
| Citation style | Yes | throughout |
| Graph visualization | Yes | Fig. 3 |
| Predicted vs actual | Appendix | App. A |
| Time-varying plan | Clarified as future work | §7–8 |
| Abstract numbers | Corrected | Abstract |
| No causal claims | Yes | global |
| Static vs adaptive wording | Yes | §3.4, §6.3 |
| cGSL early definition | Yes | §3.2 |
| Typo | Yes | §5 |

We believe the revision now answers the reviewers’ core scientific concerns with a single, auditable experimental design.
