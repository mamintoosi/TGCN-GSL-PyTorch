Stage 44 — Protocol and Provenance Reconciliation
=================================================

Goal
----
Audit and reconcile the exact experimental protocol and provenance of the
results that will be used in the new manuscript.

This stage is an audit only.

Do NOT:
- edit the manuscript;
- edit previous_revision;
- edit submitted_version;
- modify code;
- rerun experiments;
- change experimental results;
- generate new numerical results.

The goal is to establish a single, unambiguous description of the experiments
that can be used later when writing the new manuscript.

-------------------------------------------------------
1. SOURCES TO INSPECT
-------------------------------------------------------

Inspect the actual files/code/configuration corresponding to:

1. Stage40 canonical experiments
2. Stage41 statistical analysis
3. Stage42 audit and claim reconciliation
4. Stage43 manuscript architecture report
5. The current manuscript versions where the experimental protocol is
   described, especially:
   - paper/submitted_version/sn-article.tex
   - paper/previous_revision/ where relevant

Also inspect the actual Stage40 code and configuration files rather than
relying only on summaries.

Stage40 is the canonical source for the main 5-seed experimental results.

-------------------------------------------------------
2. PRIMARY QUESTIONS
-------------------------------------------------------

Resolve the following protocol/provenance issues.

A. DAGMA lambda1 and threshold

Determine exactly:

- lambda1 used during DAGMA fitting;
- DAGMA internal w_threshold, if any;
- post-processing threshold used by the consumer;
- whether thresholding uses W > tau, |W| > tau, or another rule;
- whether binary adjacency is produced before or after symmetrization;
- how cGSL is constructed;
- whether GSL and cGSL use exactly the same underlying learned graph.

Trace the complete pipeline:

raw data
→ normalization
→ train/test split
→ DAGMA input construction
→ DAGMA fitting
→ raw W
→ internal DAGMA thresholding
→ consumer threshold
→ binary adjacency
→ symmetrization (if cGSL)
→ graph normalization / Laplacian
→ GCN/T-GCN.

Do not infer any value from an old manuscript statement if the code/configuration
shows something different.

-------------------------------------------------------
3. STAGE40 MULTI-LAG GRAPH PROTOCOL
-------------------------------------------------------

Determine exactly how the multi-lag DAGMA graphs were generated.

Report:

- number of temporal blocks;
- lag interpretation;
- dimensionality of the DAGMA input;
- exact lambda1;
- exact threshold(s);
- DAGMA fitting parameters;
- warm iterations;
- maximum iterations;
- loss type;
- random seed;
- whether the graph is refit separately for each PH;
- whether PH changes the DAGMA input;
- whether the resulting graphs are identical across PH;
- number of edges per lag for each dataset;
- number of edges in the union graph;
- whether self-loops occur in DAGMA output;
- whether self-loops are subsequently added by the GCN/T-GCN graph
  normalization.

Be precise about what "lag 1", "lag 2", etc. mean.

Do not describe the graph as causal unless the implementation and evidence
justify such a claim.

-------------------------------------------------------
4. T-GCN-MultiGSL GRAPH CONSUMPTION
-------------------------------------------------------

Audit the actual implementation of:

- T-GCN-MultiGSL
- T-GCN-MultiGSL-Weighted
- T-GCN-MultiGSL-Mix
- GCN-MultiGSL

Determine exactly how the learned graphs are consumed.

For each method report:

- number of graph matrices;
- mapping from input timestep to graph index;
- whether the mapping is cyclic;
- whether graphs are assigned to specific historical lags;
- whether the GCN version unions the lag graphs;
- whether Weighted uses global learned graph weights;
- whether Mix uses node/time-dependent gates;
- number of additional trainable parameters;
- whether those additional parameters are included in reported model
  capacity comparisons.

Do not use the word "adaptive" for a method name unless that terminology is
actually present and justified.

-------------------------------------------------------
5. CONTEMPORANEOUS SINGLE-GRAPH GSL
-------------------------------------------------------

Audit the exact Stage40 implementation of:

- GCN-GSL
- GCN-cGSL
- T-GCN-GSL
- T-GCN-cGSL

Determine:

- DAGMA input;
- training subset;
- normalization;
- lambda1;
- threshold;
- graph density;
- number of directed edges;
- whether self-loops are present in learned adjacency;
- cGSL construction;
- whether the same graph artifact is shared between the GCN and T-GCN
  counterpart experiments.

Also determine whether the graph is genuinely contemporaneous.

If DAGMA receives rows representing simultaneous sensor observations,
describe it as a contemporaneous statistical dependency graph.

Do not interpret it as a temporal graph merely because it is later used by a
temporal forecasting model.

-------------------------------------------------------
6. DATA SPLIT AND NORMALIZATION
-------------------------------------------------------

Verify the exact data protocol used by Stage40.

Report:

- chronological train/test split;
- normalization statistic;
- whether normalization is computed using training data only;
- sequence length;
- prediction horizons PH1–PH4;
- dataset-specific sampling intervals;
- exact number of training samples used by DAGMA;
- whether DAGMA ever sees test data.

Explicitly identify any difference between the historical/submitted protocol
and the Stage40 canonical protocol.

If the old manuscript contains a protocol statement that is not consistent
with Stage40, flag it.

-------------------------------------------------------
7. SPARSITY AND CAPACITY CONTROLS
-------------------------------------------------------

Audit the provenance of:

- RandTop30
- CorrTop30
- DAGMA sparse graph
- DAGMA + Mix

For each determine:

- exact edge budget;
- how edges were selected;
- whether directionality is preserved;
- whether self-loops are added;
- whether the graph is normalized in the same way;
- seed(s);
- model architecture;
- parameter count;
- whether the comparison is intended as a sparsity control or a capacity
  control.

Determine precisely what scientific conclusion these controls support.

Do not generalize a Los-loop PH1 result to all datasets or all horizons.

-------------------------------------------------------
8. 15-MINUTE SAMPLING EXPERIMENT
-------------------------------------------------------

Audit the provenance of the 15-minute sampling experiment.

Determine:

- exact dataset;
- sampling interval;
- prediction horizons;
- corresponding real-time forecast horizons;
- methods compared;
- number of seeds;
- exact numerical results;
- whether this experiment belongs to the same canonical pipeline as Stage40;
- whether the 27.4% improvement reported elsewhere comes from this experiment
  or from another pipeline;
- whether the 15-minute experiment can legitimately be used as a headline
  result in the revised abstract.

Clearly distinguish:

"PH4 at 15-minute sampling"

from

"PH4 at 5-minute sampling".

Do not equate them.

-------------------------------------------------------
9. STAGE40 NUMERICAL PROVENANCE
-------------------------------------------------------

Verify that the following canonical results are traceable to the Stage40
artifacts:

- GCN family;
- T-GCN family;
- NoSpatial;
- GSL;
- cGSL;
- MultiGSL;
- MultiGSL-Weighted;
- MultiGSL-Mix;
- both datasets;
- PH1–PH4;
- all five seeds.

Do not recompute results unless absolutely necessary to resolve a provenance
ambiguity. This is an audit, not a reproduction stage.

Check that mean and standard deviation definitions are consistent.

-------------------------------------------------------
10. OLD MANUSCRIPT VS CANONICAL PROTOCOL
-------------------------------------------------------

Create a comparison table with columns:

Issue
Evidence
Canonical Stage40 value/protocol
Previous manuscript value/protocol
Difference
Scientific impact
Required action in new manuscript

At minimum check:

- lambda1;
- threshold;
- DAGMA input;
- normalization;
- train/test split;
- temporal interpretation;
- graph construction;
- cGSL;
- multi-lag construction;
- graph consumption;
- seeds;
- model capacity;
- sparsity controls;
- 15-minute experiment.

-------------------------------------------------------
11. CLASSIFY EVERY DISCREPANCY
-------------------------------------------------------

Classify discrepancies as one of:

1. NO DIFFERENCE
   The old description is consistent with Stage40.

2. EDITORIAL
   Wording should change but scientific interpretation is unchanged.

3. PROTOCOL DIFFERENCE
   The old manuscript describes a different experimental protocol.

4. INTERPRETATION DIFFERENCE
   The numerical experiment is the same, but the old scientific
   interpretation is no longer defensible.

5. RESULT-PROVENANCE ISSUE
   A result cannot currently be confidently traced to the canonical
   Stage40 pipeline.

6. BLOCKING ISSUE
   The discrepancy must be resolved before the new manuscript can be written.

-------------------------------------------------------
12. IMPORTANT SCIENTIFIC CHECKS
-------------------------------------------------------

Explicitly verify the following points.

A. NoSpatial

Confirm exactly what "NoSpatial" means in the implementation.

Determine whether it means:

- identity adjacency;
- no graph aggregation;
- another graph-free mechanism.

Do not assume.

B. Physical graph

Determine whether the reported physical graph includes self-loops only
because the graph normalization/Laplacian adds them internally.

Distinguish:

- raw physical adjacency;
- normalized adjacency;
- effective graph used by the model.

C. Learned graph density

Report density consistently.

For directed graphs, state clearly whether density is calculated over:

N(N-1)

or

N^2.

Do not mix conventions.

D. cGSL

Verify that cGSL is produced by symmetrizing the learned graph and that
absolute-value thresholding, if used, is conceptually separate from
symmetrization.

E. Causality

Explicitly identify any manuscript wording that should be removed because
the DAGMA constraint alone does not establish causal relationships.

-------------------------------------------------------
13. FINAL CANONICAL PROTOCOL
-------------------------------------------------------

At the end of the report, write a concise section:

"Canonical Protocol for the New Manuscript"

It should contain the exact protocol that later manuscript-writing stages
should use.

Include:

- datasets;
- sampling;
- train/test split;
- normalization;
- sequence length;
- PH;
- DAGMA input;
- DAGMA hyperparameters;
- thresholds;
- graph construction;
- cGSL;
- multi-lag graph construction;
- graph consumption;
- model variants;
- seeds;
- evaluation metrics;
- statistical analysis;
- sparsity controls;
- 15-minute experiment.

This section must be based on verified implementation/configuration, not
assumptions.

-------------------------------------------------------
14. FINAL DECISION
-------------------------------------------------------

Conclude with one of:

READY FOR STAGE 45

or

NOT READY FOR STAGE 45

If NOT READY, list only the genuinely blocking issues.

Do not recommend rerunning experiments unless the provenance problem cannot be
resolved from existing artifacts.

-------------------------------------------------------
15. OUTPUT
-------------------------------------------------------

Create:

paper/gsl_stage44/stage44_protocol_reconciliation.md

The report should contain:

1. Executive summary
2. DAGMA threshold/lambda audit
3. Multi-lag graph audit
4. Graph-consumption audit
5. Single-graph GSL audit
6. Data split and normalization audit
7. Sparsity/capacity control audit
8. 15-minute experiment provenance
9. Stage40 numerical provenance
10. Old-vs-canonical protocol table
11. Discrepancy classification
12. Canonical protocol for the new manuscript
13. Remaining blocking issues
14. Final readiness verdict

Again:

NO manuscript edits.
NO code edits.
NO new experiments.
NO changes to previous_revision.
NO changes to submitted_version.