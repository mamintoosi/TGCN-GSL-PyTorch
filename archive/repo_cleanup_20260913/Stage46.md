# Stage 46 — Write Manuscript Sections 3 and 4

## Objective

Write the two load-bearing scientific sections of the new manuscript:

1. Section 3 — Problem Formulation and Graph Learning Framework
2. Section 4 — Experimental Setup

This is a WRITING stage. Produce publication-ready LaTeX section bodies that can later be assembled into the Springer Nature `sn-jnl` manuscript.

Do NOT:

- write Introduction, Background, Results, Discussion, Limitations, Conclusion, Abstract, or appendices;
- edit `paper/submitted_version/` or `paper/previous_revision/`;
- run new experiments or change any numbers;
- invent statistics not present in Stage 40–44 artifacts;
- infer implementation details when they can be verified from the canonical Stage 40 code;
- mention development history of the paper or of this repository.

---

# Governing editorial principle

The manuscript presents ONE self-contained scientific study: the final experimental design and the final evidence.

Sections 3 and 4 must not contain:

- "historical results", "first version", "previous version", "submitted version", "earlier version";
- "original results", "re-baselining", or protocol-difference narratives;
- explanations of why numbers differ from any earlier document;
- causal language about traffic influence or hidden causal structure;
- the word "Adaptive" in any method name;
- claims that the learned graphs themselves change over time;
- claims that DAGMA's acyclicity constraint establishes causality.

Historical/protocol-difference material belongs in the RESPONSE LETTER ONLY.

---

# Sources of truth

Use the following priority order:

1. `Stage45.md` and especially
   `paper/gsl_stage45_1/stage45_1_final_architecture.md`

2. `paper/gsl_stage44/stage44_protocol_reconciliation.md`,
   especially §12 Canonical Protocol

3. Canonical Stage 40 implementation and configuration files, including:
   - `gsl_stage40/scripts/stage40_run_all.py`
   - `gsl_stage40/`
   - `models/multigsl.py`
   - `utils/graph_conv.py`
   - relevant data-loader / training files used by Stage 40

4. `paper/submitted_version/sn-article.tex`
   for LaTeX class/style, citation commands, equation style

5. `paper/submitted_version/commands.tex`
   for shared macros

6. `paper/submitted_version/MyReferences.bib`
   for citation keys

7. `doc/METHOD_NAMING_MAP.md`

8. `paper/previous_revision/sections/method.tex`
   only as a drafting/template reference.
   It is NOT structural authority. Correct any NOTEARS-vs-DAGMA or temporal-DAG errors rather than copying them.

9. Stage 41–42 claim policy for wording restrictions.

## Important source rule

Whenever a statement depends on implementation, inspect the canonical Stage 40 code rather than relying only on the prose in this prompt.

In particular, verify from code:

- exact definition of PH and the target tensor;
- whether prediction is single-step or multi-step;
- exact graph normalization;
- self-loop handling;
- graph orientation/indexing;
- T-GCN MultiGSL lag-to-input-step mapping;
- parameter counts where reported;
- exact training loss;
- exact seed handling.

If Stage 44 prose and the code appear inconsistent, do NOT silently choose one. Record the discrepancy in the writing report and use the implementation that actually generated the Stage 40 results, unless the Stage 44 reconciliation explicitly resolves the issue.

---

# Output paths

Create ONLY:

`paper/gsl_stage46/sections/method.tex`

`paper/gsl_stage46/sections/setup.tex`

`paper/gsl_stage46/stage46_writing_report.md`

Do not create or overwrite a full `sn-article.tex`.

Do not modify files outside `paper/gsl_stage46/`.

---

# LaTeX conventions

The files must be compatible with:

`\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}`

Use:

- `\citep` / `\citet` as in the submitted manuscript;
- shared macros from `commands.tex` such as `\tr`, `\dagspace`, `\wadj`, etc., when appropriate;
- `\mathbf{A}` for the backbone adjacency;
- `$W$` for the continuous DAGMA coefficient matrix.

State this A/W convention explicitly in the main text:

> `\mathbf{A}` denotes the adjacency consumed by the forecasting backbone, whereas `W` denotes the continuous coefficient matrix estimated by DAGMA.

Do NOT describe `\mathbf{A}` as "weighted binary". Stage 40 backbone adjacencies are binary after graph construction.

Use:

`\label{sec:method}`

`\label{sec:setup}`

and appropriate subsection labels.

Do not use `\input` of other section files inside these files.

---

# Section 3 — Problem Formulation and Graph Learning Framework

## Title

`Problem Formulation and Graph Learning Framework`

## Label

`\label{sec:method}`

## Purpose

Define the forecasting task, clearly separate GRAPH SOURCE from GRAPH CONSUMPTION, and give the exact graph constructions used in the study.

This section defines the methods BEFORE any results are discussed.

---

## 3.1 Forecasting task and model baselines

Label:

`\label{sec:task}`

Include:

- Traffic speed observations at $N$ sensors:
  \[
  x(t)\in\mathbb{R}^N.
  \]

- Input window length:
  \[
  T=12.
  \]

- Prediction horizons:
  \[
  \mathrm{PH}\in\{1,2,3,4\}.
  \]

- Define the forecasting task as mapping the 12-step input window to the target used by the actual Stage 40 data loader.

### Critical implementation requirement

Inspect the actual Stage 40 data-loading and training code and determine exactly whether PH corresponds to:

- a single future time step, or
- a multi-step target/vector.

Do NOT write:

> "future speeds at $t+\mathrm{PH}$ (or vector of horizons...)"

Instead, resolve the ambiguity from the implementation and state the actual target definition precisely.

Do not infer this from generic T-GCN terminology.

---

### Graph notation

Use:

\[
\mathbf{A}\in\{0,1\}^{N\times N}
\]

for the binary adjacency consumed by the forecasting backbone.

Use:

\[
W\in\mathbb{R}^{d\times d}
\]

for the continuous DAGMA coefficient matrix, where $d=N$ for contemporaneous DAGMA and $d=(L+1)N$ for multi-lag DAGMA.

State explicitly:

> `\mathbf{A}` denotes the backbone adjacency, whereas $W$ denotes the continuous DAGMA coefficient matrix.

Do not use A and W interchangeably.

---

### Graph-free baseline

Define:

\[
\mathbf{A}=\mathbf{I}_N
\]

for the graph-free control.

Names:

- T-GCN-NoSpatial
- GCN-NoSpatial

State that the identity adjacency is passed through exactly the same graph-normalization pipeline as the other graph configurations.

Do not call it "no graph computation" if the implementation still applies graph convolution with identity adjacency.

---

### Physical baseline

Define the dataset-provided road-network adjacency.

Names:

- T-GCN for the T-GCN family
- GCN for the GCN family

Do not introduce an additional method called "Physical" unless required by a table label.

---

### Backbone description

Keep this equation-light.

State that:

- GCN performs graph convolution over the input representation;
- T-GCN incorporates graph convolution into a GRU-based recurrent architecture.

Detailed generic GCN/T-GCN derivations belong to Background and Related Work, not here.

---

### Graph normalization

Inspect the actual implementation, especially:

`utils/graph_conv.py`

and determine exactly how adjacency matrices are transformed.

If the implementation confirms:

\[
\tilde{\mathbf{A}}=\mathbf{A}+\mathbf{I},
\qquad
\hat{\mathbf{A}}
=
\tilde{\mathbf{D}}^{-1/2}
\tilde{\mathbf{A}}
\tilde{\mathbf{D}}^{-1/2},
\]

state this.

However, DO NOT assume this formula merely because it is standard GCN notation.

Match the actual Stage 40 implementation, including:

- whether self-loops are added;
- whether normalization is symmetric;
- whether transposition is applied;
- whether the same function is used for physical, identity, GSL, cGSL, and MultiGSL graphs.

The manuscript should describe the implementation that generated the reported experiments.

---

# 3.2 Contemporaneous graph learning (GSL / cGSL)

Label:

`\label{sec:gsl_contemp}`

Purpose: define the single learned graph constructions before Results.

Use DAGMA \citep{Bello2024DAGMA}.

Do NOT describe this method as NOTEARS.

Present the DAGMA acyclicity characterization actually used by the library, based on the log-determinant / M-matrix formulation.

Do not introduce the NOTEARS trace-exponential constraint unless explicitly discussing it as prior work, which is not necessary here.

---

## Continuous optimization

Describe:

- linear structural-equation-model score;
- $\ell_2$ loss;
- $\ell_1$ sparsity penalty with coefficient $\lambda_1$;
- DAGMA optimization.

Keep the mathematical presentation compact.

---

## Exact contemporaneous construction

Define:

\[
\mathbf{X}
=
\mathrm{train\_norm}[0::\mathrm{PH}]
\]

as the simultaneous sensor snapshots subsampled every PH-th training row.

Explain that:

- rows are simultaneous observations;
- columns correspond to sensors;
- there are $N$ variables;
- there are NO lag blocks.

For Los-loop, the resulting row counts are:

- PH1: 2380
- PH2: 1190
- PH3: 794
- PH4: 595

For SZ-Taxi, state the corresponding values only if verified from the canonical implementation/protocol.

Parameters:

- Los-loop: $\lambda_1=0.02$
- SZ-Taxi: $\lambda_1=0.01$
- DAGMA internal `w_threshold=0.3`

After DAGMA's internal thresholding, construct the binary support using the absolute-magnitude convention:

\[
\mathbf{A}_{\mathrm{GSL}}
=
\mathbf{1}(|W|>0),
\]

with the diagonal removed.

Explain that, under the canonical Stage 40 implementation, this is equivalent to retaining coefficients whose magnitude reaches the internal 0.3 threshold.

DO NOT conflate this with the 0.1 consumer threshold used by MultiGSL.

The resulting directed off-diagonal edge counts are:

- Los-loop: 28 for every PH;
- SZ-Taxi: 8 for every PH.

The graph is learned using training data only.

If seed/initialization details are mentioned, place them primarily in Section 4 rather than here.

---

## cGSL

Define cGSL as symmetrization of the SAME GSL artifact:

\[
\mathbf{A}_{\mathrm{cGSL}}
=
\mathbf{1}
\left(
\mathbf{A}_{\mathrm{GSL}}
+
\mathbf{A}_{\mathrm{GSL}}^\top
>0
\right),
\]

with diagonal removed.

Important:

- no second DAGMA fit;
- no second threshold;
- cGSL is not a separately learned graph;
- it is the symmetrized version of the corresponding GSL graph.

Directed-entry counts:

- Los-loop: 56
- SZ-Taxi: 16

The same graph artifacts are used by both GCN and T-GCN counterparts.

---

## Interpretation restriction

Explicitly state:

- the contemporaneous DAGMA graph is a statistical dependency structure;
- it is NOT a temporal graph;
- it does NOT mean sensor $j$ at time $t$ predicts sensor $i$ at time $t+1$;
- DAGMA acyclicity is an optimization/fitting constraint and is not causal evidence.

---

# 3.3 Multi-lag graph learning

Label:

`\label{sec:gsl_multilag}`

Define lag depth:

\[
L=3.
\]

Use the explicit lag-stacked representation:

\[
Z=
\left[
x(t-3),x(t-2),x(t-1),x(t)
\right].
\]

Dimensions:

- Los-loop:
  \[
  (L+1)N=4(207)=828
  \]
- SZ-Taxi:
  \[
  (L+1)N=4(156)=624.
  \]

State that the canonical construction contains 2377 training windows, if verified from the implementation.

DAGMA is fitted ONCE on the training split and is shared across PH=1,...,4.

State explicitly:

> The multi-lag graph is PH-independent: the same learned graph artifact is consumed for all four prediction horizons.

Also state the contrast:

> Unlike the multi-lag construction, the contemporaneous graph is PH-dependent because its input is subsampled as `train_norm[0::PH]`.

---

## Multi-lag optimization

Parameters:

- $\lambda_1=0.01$ for both datasets;
- internal DAGMA `w_threshold=0.0`;
- raw continuous weights retained;
- consumer threshold:
  \[
  |W|>0.1;
  \]
- diagonal removed.

Make the distinction between the two threshold regimes explicit:

1. Contemporaneous GSL:
   - DAGMA internal threshold = 0.3
   - support extracted after that internal threshold

2. Multi-lag:
   - DAGMA internal threshold = 0.0
   - raw weights retained
   - consumer threshold = 0.1

Never collapse these into a single "DAGMA threshold".

---

## Block interpretation

Define the lag blocks carefully according to the actual code indexing.

The intended manuscript interpretation is:

- lag-1 block: activity at $t-1$ to sensors at $t$;
- lag-2 block: activity at $t-2$ to sensors at $t$;
- lag-3 block: activity at $t-3$ to sensors at $t$.

The contemporaneous $x(t)\rightarrow x(t)$ block is present in the DAGMA fit but is NOT consumed by the Stage 40 MultiGSL forecasting methods.

Before writing this, verify the block orientation/indexing against the actual Stage 40 code.

---

## Edge counts

After consumer threshold:

Los-loop:

- lag-1: 12
- lag-2: 3
- lag-3: 15
- total lagged edge slots: 30
- union: 28 distinct off-diagonal edges

SZ-Taxi:

- lag-1: 0
- lag-2: 0
- lag-3: 2
- total: 2
- union: 2

If the canonical artifact uses a different terminology for "edge slots" versus "distinct edges", preserve that distinction.

---

## Interpretation restrictions

Describe the learned lag graphs as:

> descriptive statistical dependency estimates.

State that:

- the lag-structured reading is consistent with the construction and the consumption experiments;
- it is NOT independently validated as a causal or predictive graph;
- DAGMA acyclicity is not causal evidence;
- graphs are static and fitted once;
- only their consumption can vary by timestep.

Do not say that the graph itself "adapts", "evolves", or "changes with traffic".

---

# 3.4 Graph consumption mechanisms

Label:

`\label{sec:consumption}`

This is the conceptual centerpiece.

Start from the explicit principle:

> Graph source and graph consumption are treated as separate design dimensions.

The subsection should explain that the same or similarly budgeted learned graphs can be consumed in different ways.

---

## Static graph mechanisms

For:

- Physical
- NoSpatial
- GSL
- cGSL

state:

> one adjacency is used for all input steps.

Graph-side extra parameters: 0.

---

## Fixed cyclic lag assignment

Method name:

**T-GCN-MultiGSL**

At input step $t$ where:

- $t=0$ denotes the most recent input step,

use the exact code rule:

\[
\mathrm{graph\_idx}
=
(T-1-t)\bmod 3.
\]

Verify this against the actual implementation before writing.

Explain the resulting correspondence:

- most recent input step $\rightarrow$ lag-1 graph;
- preceding step $\rightarrow$ lag-2 graph;
- oldest step $\rightarrow$ lag-3 graph;

if and only if this matches the code.

Extra parameters: 0.

---

## Global lag mixture

Method name:

**T-GCN-MultiGSL-Weighted**

Use three learned scalar weights followed by a softmax.

The resulting normalized weights mix the three lag-specific graph/Laplacian representations before graph convolution.

Extra parameters: +3.

This method may be described briefly as a supplementary mechanism; do not promote it as the central contribution.

---

## Per-node, per-timestep gate

Method name:

**T-GCN-MultiGSL-Mix**

The gate is based on the implementation using:

\[
[x_t;h_{t-1}]
\]

and produces a softmax over:

\[
K=3
\]

lag graphs.

The graph/Laplacian representations are mixed before the GRU update.

Extra parameters:

+4,419.

Crucial interpretation:

> Mix varies how the static lag-specific graphs are consumed; it does not relearn or modify the graph edge set at each timestep.

Do not call this an adaptive graph or adaptive graph learning method.

---

## Static union in GCN

Method name:

**GCN-MultiGSL**

Define:

\[
\mathbf{A}_{\cup}
=
\bigvee_{k=1}^{3}\mathbf{A}_k
\]

or equivalently element-wise maximum for binary adjacency matrices.

This produces one static graph for the complete input window.

The same multi-lag graph artifacts are therefore used by:

- GCN-MultiGSL through a static union;
- T-GCN-MultiGSL through lag-specific consumption.

This distinction is central to the later experimental comparison.

Do NOT claim that the comparison isolates consumption alone, because GCN and T-GCN also differ in backbone architecture.

Use evidence-aligned wording such as:

> The comparison is designed to assess whether the utility of the same learned multi-lag structure depends on how it is consumed.

Do NOT write:

> the benefit is located in the consumption mechanism, not the edge set.

---

## Parameter counts

Verify these against the canonical implementation.

If confirmed, state:

- GCN: 768 trainable parameters;
- T-GCN: 12,672 trainable parameters;
- Weighted: +3;
- Mix: +4,419.

Use exact counts rather than "on the order of".

Keep this concise; the detailed capacity-control discussion belongs in Results/Appendix.

---

# Section 3 length and style

Keep Section 3 compact but complete.

Use equations only where they establish essential notation:

- forecasting task;
- graph normalization;
- cGSL symmetrization;
- multi-lag $Z$;
- MultiGSL lag assignment;
- Weighted/Mix mechanism where needed.

Do not include:

- result tables;
- result percentages;
- statistical significance;
- improvement claims;
- discussion of why one dataset benefits more than another.

---

# Section 4 — Experimental Setup

## Title

`Experimental Setup`

## Label

`\label{sec:setup}`

Purpose: provide a complete, reproducible experimental protocol.

Use Stage 44 §12 as the protocol skeleton, but verify implementation-dependent details against Stage 40 code.

Do not include any historical/protocol-difference narrative.

---

# 4.1 Datasets

Describe:

### Los-loop

- 207 highway loop detectors;
- Los Angeles County;
- March 2012;
- 5-minute sampling;
- 2976 timesteps.

### SZ-Taxi

- 156 urban sensors;
- Shenzhen / Luohu;
- January 2015;
- 15-minute sampling;
- 2976 timesteps.

State that the physical adjacency supplied with each dataset is used for the physical baseline.

---

# 4.2 Split and normalization

Use chronological 80/20 splitting.

Los-loop:

- training: 2380 timesteps;
- test: 596 timesteps.

No forecasting window crosses the train/test boundary.

Normalize features using the TRAINING-SPLIT maximum only.

Values:

- Los-loop: 70.0
- SZ-Taxi: 86.4292

State explicitly:

> The test split is not used for normalization or graph learning.

---

# 4.3 Forecasting protocol

Input window:

\[
T=12.
\]

Prediction horizons:

\[
\mathrm{PH}\in\{1,2,3,4\}.
\]

Translate these into wall-clock horizons:

### Los-loop

- PH1 = 5 min
- PH2 = 10 min
- PH3 = 15 min
- PH4 = 20 min

### SZ-Taxi

- PH1 = 15 min
- PH2 = 30 min
- PH3 = 45 min
- PH4 = 60 min

Use the exact target definition established by inspecting the Stage 40 loader.

Do not call PH=1,...,4 "four-step forecasting" unless the implementation actually performs multi-step forecasting.

---

# 4.4 Methods

List exactly the 12 canonical configurations.

## T-GCN family

- T-GCN
- T-GCN-NoSpatial
- T-GCN-GSL
- T-GCN-cGSL
- T-GCN-MultiGSL
- T-GCN-MultiGSL-Weighted
- T-GCN-MultiGSL-Mix

## GCN family

- GCN
- GCN-NoSpatial
- GCN-GSL
- GCN-cGSL
- GCN-MultiGSL

Do not invent additional method names.

Cross-reference Section 3 for graph construction and graph consumption.

Do not redefine the methods here.

---

# 4.5 Training protocol

Use the canonical Stage 40 settings, verified against code:

- optimizer: Adam;
- learning rate:
  \[
  10^{-3};
  \]
- weight decay:
  \[
  10^{-4};
  \]
- batch size: 128;
- epochs: 50;
- hidden dimension: 64.

Loss:

- T-GCN family: MSE with the implementation's L2-on-weights regularization (`mse_with_regularizer`);
- GCN family: MSE.

Training seeds:

\[
\{42,43,44,45,46\}.
\]

Five seeds.

DAGMA graphs are deterministic and shared across the five forecasting-model training seeds.

DAGMA-linear:

- loss type: $\ell_2$;
- warm iterations: 30,000;
- maximum iterations: 60,000;
- DAGMA seed: 42.

If the actual Stage 40 code specifies additional reproducibility settings that materially affect the protocol, include them.

---

# 4.6 Evaluation metrics

Keep this concise.

Primary metric:

**RMSE**

Secondary metric:

**MAE**

Both are computed on the de-normalized original speed scale.

Test evaluation uses the complete test set / full-batch evaluation as implemented.

Give one-line mathematical definitions only if needed for completeness.

Do not turn this subsection into a metrics tutorial.

---

# 4.7 Statistical reporting

Include an explicit paragraph substantially following this wording:

> Main-text numbers are five-seed means with sample standard deviations ($\mathrm{ddof}=1$). We report paired per-seed win counts and paired $t$-tests for reference. With $n=5$, the exact Wilcoxon signed-rank test cannot fall below $p=0.0625$; we therefore make no definitive significance claims.

Do not weaken or omit the final sentence.

---

# 4.8 Matched-sparsity and capacity controls

Describe these as SCOPE-LABELED supplementary controls.

## Matched sparsity

Los-loop PH1 only.

30-edge budget.

Methods:

- RandTop30: random directed edges, redrawn per seed;
- CorrTop30: top-30 absolute Pearson correlations computed from training data only;
- DAGMA multi-lag fixed consumption;
- DAGMA multi-lag Mix consumption.

Five seeds.

Same training protocol.

Explicitly state that this control cell is outside the complete 12-method dataset × horizon matrix and is reported separately.

Do NOT claim that a sparsified physical graph control was performed.

Do NOT claim that a lambda or threshold sweep was performed.

---

## Capacity control

Describe the hidden-74 NoSpatial / Mix capacity-matched check briefly.

Use the verified parameter counts.

Do not over-explain here; details may appear in Results or Appendix.

---

# 4.9 Temporal-resolution variant

Describe the 15-minute Los-loop experiment in ONE short paragraph.

Procedure:

- average consecutive groups of three 5-minute observations;
- obtain a 15-minute series;
- resulting length:
  \[
  T=992;
  \]
- PH1–4 correspond to 15, 30, 45, and 60 minutes.

This is a TEMPORAL-RESOLUTION VARIANT, NOT a third dataset.

Explicitly state:

> Its PH indices must not be interpreted as equivalent to PH indices in the original 5-minute Los-loop series.

Use a separate PH-independent DAGMA fit on this training split:

- $\lambda_1=0.01$;
- consumer threshold:
  \[
  |W|>0.1.
  \]

Methods:

- T-GCN-NoSpatial;
- T-GCN-MultiGSL;
- T-GCN-MultiGSL-Mix.

Five seeds.

Same forecasting training protocol.

Do not include any improvement percentages in Section 4.

---

# 4.10 Implementation and reproducibility

State the repository:

`https://github.com/mamintoosi/TGCN-GSL-PyTorch`

Use the appropriate LaTeX hyperlink/citation convention already present in the manuscript if needed.

State that DAGMA graph learning uses training data only and has no access to the test split.

Do not discuss legacy repository folders or historical experiment pipelines.

---

# Section 4 style

Complete but lean.

Do not include:

- results;
- percentages;
- interpretation;
- reviewer-response language;
- historical comparisons;
- protocol-difference tables.

---

# Claim and style checklist

Before declaring Stage 46 complete, verify every item:

- [ ] No history language in either section.
- [ ] No causal verbs claiming traffic influence.
- [ ] Learned graphs described as statistical dependency structures.
- [ ] Static graph fitting is clearly distinguished from graph consumption.
- [ ] No claim that the learned graph changes over time.
- [ ] cGSL is defined completely in Section 3 before Results.
- [ ] A/W notation convention is explicit.
- [ ] Contemporaneous DAGMA threshold regime is clearly separated from MultiGSL consumer thresholding.
- [ ] Multi-lag graphs are explicitly PH-independent.
- [ ] Contemporaneous graph learning is explicitly PH-dependent through `train_norm[0::PH]`.
- [ ] Method names match the Stage 40 registry/display names exactly.
- [ ] Graph normalization is verified from the actual implementation.
- [ ] PH target definition is verified from the actual data loader.
- [ ] T-GCN MultiGSL lag assignment is verified from the actual implementation.
- [ ] Exact parameter counts are verified from the code before being stated.
- [ ] Statistical policy paragraph is present in Section 4.
- [ ] Five-seed means/std policy is stated.
- [ ] 15-minute material is labeled as a temporal-resolution variant only.
- [ ] No sparsified-physical control is claimed.
- [ ] No lambda/threshold sweep is claimed.
- [ ] R1 wording is respected: do not claim that the benefit is located in consumption rather than the edge set.
- [ ] R2 wording is respected: any future improvement statement must use "relative RMSE reduction" and not ambiguous "+X% improvement".
- [ ] Citations resolve against `MyReferences.bib`.
- [ ] Missing citation keys, if any, are listed in the report.
- [ ] No files outside `paper/gsl_stage46/` were modified.

---

# Writing report

Create:

`paper/gsl_stage46/stage46_writing_report.md`

Include:

## 1. Files written

List the three files and approximate word counts.

## 2. Protocol verification

For every protocol number inserted into the manuscript, record:

- the value;
- where it appears;
- Stage 44 §12 source;
- whether it was additionally verified from Stage 40 code.

At minimum track:

- dataset sizes;
- time-series lengths;
- train/test split;
- normalization maxima;
- input length;
- PH values;
- DAGMA dimensions;
- lambda values;
- internal DAGMA thresholds;
- consumer threshold;
- edge counts;
- DAGMA iterations;
- training hyperparameters;
- seed set;
- parameter counts;
- 15-minute variant settings.

## 3. Implementation checks

Explicitly report what was verified from code for:

- PH target semantics;
- graph normalization;
- self-loop handling;
- graph orientation;
- MultiGSL lag indexing;
- parameter counts;
- training loss.

## 4. Deliberate deviations

Record every deliberate deviation from Stage 45.1 or Stage 44 and explain why.

If there were no deviations, state so explicitly.

## 5. Citations

List:

- citation keys used;
- citation keys checked and resolved;
- missing keys, if any.

## 6. Forward references

Do NOT invent table/figure labels that do not yet exist.

Where Section 3 or 4 needs to refer forward to Results, use textual references such as:

> "The corresponding empirical comparison is presented in Section 5."

Do not insert `\ref` placeholders unless the referenced label has already been declared in the Stage 46 files.

## 7. R1/R2 compliance

Explicitly confirm:

- R1: no over-claim that consumption alone causes the benefit;
- R2: no ambiguous "+X% improvement" phrasing.

## 8. Open issues for Stage 47

List only issues that genuinely remain for Results writing, such as:

- final table/figure labels;
- exact Results ordering;
- placement of per-seed results;
- appendix references;
- any unresolved implementation/provenance question.

---

# Important writing rule

Do NOT simply copy `paper/previous_revision/sections/method.tex`.

Use it only as a source of useful notation or wording.

The final Section 3 must reflect the conceptual architecture established in Stage 45.1:

1. graph source;
2. contemporaneous versus multi-lag graph learning;
3. graph consumption;
4. separation between learned structure and its downstream use.

The final Section 4 must describe the canonical Stage 40 experimental protocol as one coherent study.

---

# Out of scope

Do NOT perform any of the following:

- Tables 1–3 generation;
- Figures 1–3 generation;
- Section 5 Results;
- Discussion;
- Limitations;
- Conclusion;
- Abstract;
- Introduction;
- Background;
- Response-to-Reviewers;
- repo cleanup;
- new experiments;
- rerunning Stage 40;
- changing any Stage 40 result.

---

# Verdict

Stage 46 is complete only when:

1. `paper/gsl_stage46/sections/method.tex` exists;
2. `paper/gsl_stage46/sections/setup.tex` exists;
3. `paper/gsl_stage46/stage46_writing_report.md` exists;
4. the complete checklist passes;
5. all implementation-dependent claims have been checked against canonical Stage 40 code;
6. no file outside `paper/gsl_stage46/` was modified.

End of Stage 46.