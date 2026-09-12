# Response to Reviewers

**Manuscript:** Graph Structure Learning for Traffic Prediction  
**Journal:** International Journal of Data Science and Analytics  
**Document type:** Staged point-by-point response (updated as each reviewer item is addressed)  
**Manuscript source of truth:** `paper/revised_version/sn-article-flat.tex`

---

## General response (summary)

Thank you for the constructive reviews. The revision keeps the paper’s identity as a study of **graph structure learning for traffic forecasting**, while incorporating a graph-free control, an explicit multi-lag construction, repeated evaluation over **five forecasting seeds for the main experiments**, and matched-sparsity controls. Causal language has been removed.

**Two comments, in particular, motivated substantial clarification and additional controls** in the revision:

- **W4 (temporal interpretation of the learned DAG):** We clarified that the contemporaneous DAGMA fit is based on simultaneous sensor snapshots and should not be interpreted as a temporal graph. We therefore introduced an explicit multi-lag construction, audited the relevant experimental protocol, and revised the Method and Results sections accordingly.
- **W5 (sparsity, oversmoothing, and fair controls):** We added a graph-free identity control and matched-sparsity and capacity controls. These experiments allow the roles of graph structure, graph density, and model capacity to be examined explicitly rather than attributing improvements to learned structure without appropriate controls (with the sparsity control scoped to Los-loop at PH1).

We are grateful for these comments: they led to a **more comprehensive and better-controlled evaluation** than that presented in the submitted version.

This letter is filled in **stage by stage** as each comment is mapped onto the revised text.

---

# Reviewer 1

## Strengths (thank you)

We thank the reviewer for highlighting these strengths. Below we state, for each strength, what was **kept** and where it appears in the revised manuscript, and what was **adjusted** in light of the new experiments.

### S1 — Core motivation: physical proximity is a poor proxy for functional dependency (and figure)

**Kept.**  
The motivation is unchanged in substance and is developed in the Introduction and Method motivation, with the conceptual network figure (`dummy-network`) still illustrating that association need not follow geographic distance alone. In the letter and manuscript we refer to **statistical associations relevant to forecasting** (rather than causal or functional “influence”).

**Location:** Introduction; Section 3 (Method), motivation paragraph and conceptual figure.

---

### S2 — Clear articulation of explicit structure discovery vs implicit attention weighting

**Kept (with non-causal wording).**  
The reviewer praised the contrast between attention (implicit weights on a given graph) and GSL (explicit structure discovery). That paragraph has been **restored** in the Introduction in a form consistent with the new experiments:

- In the attention-based methods considered here, attention typically reweights messages over a given or predefined connectivity pattern; it does not by itself estimate the adjacency.
- “Causal dependency” wording was replaced by a **thresholded, estimator-dependent statistical association** graph (not a causal map).
- We add that whether the estimated graph helps is **not automatic**: it is tested against physical and graph-free controls.

**Location:** Introduction, paragraph beginning “Attention-based methods…”; also Background (GSL vs attention).

---

### S3 — Consistent evaluation across two datasets and four horizons

**Kept and strengthened.**  
The revised study still uses **SZ-Taxi** and **Los-loop** and evaluates **PH1–PH4** under a **common main-experiment protocol** with five forecasting seeds, sample standard deviations, and paired comparisons. Absolute claims of universal improvement were **removed**; the paper now reports conditional gains (especially Los-loop multi-lag with lag-aligned use) and an explicit SZ-Taxi boundary. Auxiliary controls are scope-labeled (e.g.\ capacity control is single-seed; matched sparsity is Los-loop PH1).

**Location.** Section 4 (Setup); Section 5 (Results), Tables 1–2.

**Note.** The submitted abstract’s “21.6% / 24.7%” figures are **not** reused. The revised abstract reports a **14.5% RMSE reduction relative to the graph-free baseline for Los-loop PH1** (T-GCN-MultiGSL-Mix) and a **maximum reduction of 43.0% relative to the physical T-GCN baseline** across the evaluated Los-loop horizons.

---

### S4 — GSL vs cGSL asymmetry between GCN and T-GCN

**Kept as an empirical finding; interpretation qualified.**  
The asymmetry (cGSL helps GCN more than T-GCN) remains in the Results. The earlier interpretation of the contemporaneous DAG as a **temporal** graph has been **removed** because the DAGMA fit uses simultaneous snapshots. The revised Discussion presents **aggregation compatibility** as a **possible interpretation consistent with** the observed contrast, rather than as a definitive causal explanation.

**Location.** Section 5 (GSL/cGSL rows); Section 6 (Discussion).

---

### S5 — Bibliometric analysis as motivating work

**Kept, framed as motivation only.**  
The bibliometric study remains as **motivation** (one sentence in the Introduction) with methodology and **three figures** in **Appendix A** (initial keyword network, word clouds, refined clustered network). It is not presented as a forecasting contribution.

**Location.** Introduction (one sentence); **Appendix A** (Bibliometric Analysis Methodology).

---

## Weaknesses and questions

### W1 — Bibliometric analysis underused in the main text

**Response.** We thank the reviewer. We adopted a mixed strategy that follows both of the suggested options without turning the bibliometrics into a second contribution:

1. **Lean on it in the Introduction (one load-bearing sentence).**  
   Immediately before the paper’s central question, the Introduction now states that a bibliometric analysis of $784$ traffic-forecasting articles ($2000$–$2025$) shows graph-based methods as a dominant recent focus **while many systems still construct the adjacency heuristically from road topology**, with a pointer to **Appendix A**. This motivates GSL at the decision point rather than as a footnote.

2. **Keep the appendix complete but secondary.**  
   **Appendix A** retains the full bibliometric pipeline (Scopus collection and filtering, keyword cleaning, semantic clustering) and **three figures**: the initial keyword network (Fig. A.1), representative word clouds (Fig. A.2), and the refined clustered co-occurrence network (Fig. A.3). It is framed as motivation only, not as a forecasting contribution.

**Location.** Introduction (motivation paragraph); **Appendix A** (Bibliometric Analysis Methodology).

---

### W2 — Sections 2.2–2.3 (GCN and T-GCN) too long

**Response.** We condensed both subsections to short, equation-referenced summaries, as suggested.

1. **Removed** lengthy lists of GCN application domains and GRU/loss exposition that duplicated the Method section.
2. **Kept** the standard propagation equations, numbered **(2.2)** and **(2.3)** in the revised manuscript, and a brief T-GCN description consistent with the implementation used here (graph convolution inside the gated cell; see Method).
3. **Clarified** that the **forecasting backbones actually used in the experiments** are specified in **Section 3** (single window convolution + linear head for GCN; graph-convolutional gated cell for T-GCN), so Background is not read as a full architecture specification.
4. Freed space is used for **variance reporting** (five-seed mean ± sample standard deviation and paired win counts) and **structure figures** in the Results, as the reviewer recommended.

**Location.** Background **Sections 2.2–2.3**; backbones in **Method Section 3**.

---

### W3 — The A→W notation switch is unexplained beyond a footnote

**Response.** We now introduce the convention **in the main text**, at the start of the Method section (not only in a footnote).

- $\mathbf{A}\in\{0,1\}^{N\times N}$ is the **binary adjacency supplied to the forecasting backbone** (same symbol as in the GCN/T-GCN background).
- $W\in\mathbb{R}^{d\times d}$ is the **continuous coefficient matrix** estimated by the graph-learning procedure.
- The text states explicitly that the two symbols are **not interchangeable**: $W$ is the estimator’s internal representation; $\mathbf{A}$ is the thresholded graph used by GCN or T-GCN.

$W$ is then used in the continuous-optimization subsection (NOTEARS/DAGMA) and in the contemporaneous and multi-lag constructions; $\mathbf{A}_{\mathrm{GSL}}$, $\mathbf{A}_{\mathrm{cGSL}}$, and $\mathbf{A}_k$ denote binary graphs after thresholding (and symmetrization, where relevant).

**Location.** Method, opening of Section 3.1 (paragraph before “Integration of Learned Structure with Temporal Modeling”).

---

### W4 — Temporal interpretation of the learned DAG was asserted rather than demonstrated

**Response.** We agree this was load-bearing and was not demonstrated in the submitted version. The revision separates two constructions and **retires** the claim that the contemporaneous DAG is a temporal graph.

1. **Contemporaneous GSL (Method Section 3.2).**  
   DAGMA is fitted to simultaneous per-sensor snapshots  
   $\mathbf{X}=\mathrm{train\_norm}[0::\mathrm{PH}]$  
   (one row = all $N$ sensors at the same time; **no lag blocks**).  
   An edge is a **statistical association among present observations**; it does **not** encode “sensor $j$ at time $t$ predicts sensor $i$ at time $t+1$.”

2. **Multi-lag construction (Method Section 3.3) carries temporal organization explicitly.**  
   Input $Z=[x(t-3),\,x(t-2),\,x(t-1),\,x(t)]$ with $(L+1)N$ variables; lag-specific binary blocks $\mathbf{A}_1,\mathbf{A}_2,\mathbf{A}_3$ are extracted by absolute-magnitude thresholding. This supports a lag-structured **statistical association** reading, not causal propagation.

3. **Old Section 5 apparatus removed.**  
   The submitted “spatial graph vs temporal dependency graph” paradox section is **not** in the revised paper. The GSL/cGSL asymmetry is discussed only as an **aggregation-compatibility** observation (Discussion), consistent with the contemporaneous fit being non-temporal.

4. **Contemporaneous vs lagged evidence (also relevant to Q4).**  
   The two constructions are specified side by side in Section 3; Results compare single contemporaneous graphs (GSL/cGSL) with multi-lag graphs and with static-union vs lag-aligned use of the **same** multi-lag artifacts.

**Location.** Method Section 3.2 (interpretation); Section 3.3 (multi-lag); Discussion (asymmetry); no standalone acyclicity section.

---

### W5 — No hyperparameter optimization; learned graphs much sparser than physical; possible gains from sparsity/oversmoothing rather than structure

**Response.** We thank the reviewer for this comment. Together with W4, it led to a **major redesign of the evaluation** (graph-free baseline, multi-seed protocol, structure figures, matched-budget controls) rather than only an editorial fix. We address the three parts as follows.

#### (a) Density / degree of learned vs physical graphs

We report the structural contrast explicitly:

- **Physical Los-loop:** $2626$ off-diagonal edges; **mean degree $\approx 12.69$**.
- **Learned multi-lag union (Los-loop):** $28$ edges after consumer thresholding; **mean degree $\approx 0.14$**.
- **Contemporaneous GSL:** $28$ (Los-loop) and $8$ (SZ-Taxi) directed edges at every PH.

These statistics appear in **Figure 5** (*Los-loop graph structure*: physical adjacency, multi-lag union, and degree histogram) and in Method (edge budgets).

**Location.**  
- **Figure 5** (Los-loop graph structure): physical $2626$ off-diagonal edges, mean degree $12.69$; multi-lag union $28$ edges, mean degree $0.14$.  
- **Section 3.4** (*Contemporaneous Graph Learning (GSL and cGSL)*): contemporaneous edge counts $28$ (Los-loop) and $8$ (SZ-Taxi) at every PH.  
- **Section 3.5** (*Multi-Lag Graph Learning*): lag-slot counts $12/3/15$ (sum $30$; union $28$) on Los-loop and $0/0/2$ (union $2$) on SZ-Taxi.

#### (b) Are gains only reduced oversmoothing / sparsity?

We no longer credit a learned graph solely for beating the physical graph. The evaluation includes:

1. **Graph-free identity control (T-GCN-NoSpatial / GCN-NoSpatial).**  
   Removing the graph entirely often **reduces** error relative to the dense physical graph on both datasets. Therefore a learned graph that merely improves on Physical is **not** sufficient evidence; it must be compared with NoSpatial. Single contemporaneous GSL/cGSL **do not** beat NoSpatial on these benchmarks.

2. **Matched-sparsity controls (Los-loop PH1 only, labeled as such).**  
   At a common budget of $30$ directed edges:
   - RandTop30 (random): **worse** than NoSpatial;
   - CorrTop30 (top training correlations): **worse** than NoSpatial;
   - DAGMA multi-lag edges with lag-aligned use: **better** than NoSpatial.  
   Thus, on that cell, **sparsity alone does not explain** the multi-lag gain; **where edges are placed** (and how they are used) matters. We do **not** claim this for all datasets or horizons.

3. **Capacity control (single seed, labeled).**  
   A hidden-74 NoSpatial model matching Mix’s parameter count stays near the hidden-64 NoSpatial error, so the gate’s extra parameters do not by themselves explain the Los-loop multi-lag result.

4. **Oversmoothing language.**  
   We describe dense-graph degradation as **consistent with** excessive neighborhood mixing or an unsuitable inductive bias; we do **not** claim we measured oversmoothing directly.

**Location.**  
- **Section 4** (*Experimental Setup*), *Matched-Sparsity and Capacity Controls* subsection.  
- **Section 5.3** (*Is the Gain Only Sparsity?*) and the controls table.  
- **Discussion** (why the physical graph can be a poor default).  
- **Limitations** (control scope: Los-loop PH1 only; no sparsified physical graph; no $\lambda$/threshold sweep).

#### (c) Hyperparameter optimization / fairer baseline

We **did not** run a full hyperparameter search, a sparsified **physical** graph, or a $\lambda$/threshold sweep. Those remain **explicit limitations and future work**, so the paper does not overclaim that the physical graph was optimally tuned. What we **did** add is the identity baseline and budget-matched controls, which are the minimal fair checks the reviewer’s concern requires under a five-seed protocol.

**Location.** Limitations; Conclusion (future work).

#### Summary for W5

| Reviewer concern | How addressed |
|------------------|---------------|
| Learned graphs much sparser | Quantified in **Figure 5** and **Sections 3.4–3.5** |
| Gains may be oversmoothing only | Graph-free control; no credit vs Physical alone |
| Gains may be sparsity only | Matched 30-edge controls in **Section 5.3** (Los PH1) |
| Fairer baseline / HPO | Identity baseline + capacity check; full sweep **not** claimed (Limitations) |

---

### W6 — No significance testing, variance, or multiple seeds

**Response.** We thank the reviewer. The submitted version reported essentially single-run numbers; the revision adds repeated evaluation and a clear reporting policy.

**Multiple seeds.** All main forecasting configurations are trained under five seeds $\{42,43,44,45,46\}$. DAGMA graphs are fitted once and remain deterministic across those seeds, so the five runs quantify **training variability of the forecasting models**, not re-estimation of the graph.

**Variance in the tables.** The main T-GCN and GCN results tables report **mean RMSE with sample standard deviations** ($\mathrm{ddof}=1$) over the five seeds. Captions note that bold marks the lowest mean, not a claim of statistical significance.

**How multi-seed evidence is interpreted.** Primary weight is placed on **effect size** (mean gaps and relative RMSE reductions), **five-seed means and spreads**, and **paired per-seed consistency**. For example, on Los-loop, Mix beats NoSpatial in $5/5$ seeds at every PH1–PH4, and NoSpatial beats Physical in $5/5$ seeds in all eight dataset$\times$horizon cells. On SZ-Taxi, Mix and NoSpatial are within seed spread.

**Paired $t$-tests.** Where the Mix vs NoSpatial comparison is discussed on Los-loop, we also report paired $t$-tests on per-seed RMSE differences **for reference** (Section 5.4). These are exploratory: we do not present them as formal proof of superiority, and they are not repeated in the tables.

**Significance language.** With $n=5$, the exact two-sided Wilcoxon signed-rank test cannot attain $p<0.0625$. We therefore avoid definitive “statistically significant” claims and state this limitation once in the statistical-reporting policy and again under Limitations. Conclusions are scoped to the experiments and rely on consistency and effect size rather than on $p$-values alone.

A per-seed distribution figure on Los-loop (NoSpatial, MultiGSL, Weighted, Mix) complements the tables; its means match the main results table.

**Location.** Section 4.7 (*Statistical Reporting*); Tables 2–3 (T-GCN and GCN families); Section 5.1 (*Baselines: Physical Graph versus Graph-Free Control*); Section 5.4 (*Multi-Lag Learned Graphs*); Figure 4 (per-seed Los-loop); Limitations (statistical power).

---

| ID | Status |
|----|--------|
| W1 bibliometrics underused | **Addressed** — stronger Intro sentence + full Appendix A (3 figures) |
| W2 GCN/T-GCN background length | **Addressed** — condensed; Eqs. (2.2)–(2.3) kept; backbones pointed to Method |
| W3 A→W notation | **Addressed** — main-text convention in Method Section 3.1 |
| W4 temporal interpretation of DAG | **Addressed** — contemporaneous vs multi-lag separated; temporal-DAG claim retired; **major experimental redesign** |
| W5 sparsity / oversmoothing / controls | **Addressed** — NoSpatial, matched sparsity, capacity, structure stats (Fig. 5; §3.4–3.5; §5.3); HPO/sweep deferred to Limitations |
| W6 seeds / variance / significance | **Addressed** — see W6 below |
| W7 horizons > 4 | In progress |
| W8 repetitive Results / tie-back | In progress |
| W9 limitations | In progress |
| W10 metric definitions | In progress |
| W11 dense convergence plots | In progress |
| W12 citation style | In progress |
| Q1 graph visualization | In progress |
| Q2 predicted vs actual | In progress |
| Q3 time-varying graphs | In progress |
| Q4 contemporaneous vs lagged evidence | In progress |

---

# Reviewer 2

*(To be filled stage by stage.)*

---

## Checklist of manuscript locations (living)

| Item | Where in `sn-article-flat.tex` |
|------|--------------------------------|
| Attention vs explicit structure | Introduction (restored paragraph) |
| Sparsity control | Setup controls; Results Table controls + structure figure |
| Graph-free control | Intro; Setup; Results Section 5.1 |
| Multi-lag + lag-aligned use | Method §3.3–3.4; Results §5.4–5.5 |
| Five-seed stats policy | Setup §statistical reporting |
| Sparsity control | Setup controls; Results Table controls |
| 15-min resolution control | Results §boundaries; Table los15 |
| No causal claims | Global wording policy |
| Bibliometric | **Intro** (decision-point sentence) + **Appendix A** (full, 3 figures) |
| GCN/T-GCN background | **Condensed** §2.2–2.3; backbones → Method §3 |
| A/W notation | Method §3.1 main text (not footnote-only) |
| Temporal DAG claim | **Removed**; contemporaneous vs multi-lag in Method §3.2–3.3 |

---

*This file will be extended as we proceed through each reviewer comment.*
