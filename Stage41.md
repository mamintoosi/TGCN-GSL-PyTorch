
You are working in the repository:

`/data/git/mamintoosi/TGCN-GSL-PyTorch`

Run **Stage 41: Statistical and Scientific Result Audit**.

These scripts were ran on another system:
- run_stage40_3_sz_dagma.sh
- run_stage40_3_validate.py
- stage40_run_all.py

and the results were copied to here.

### Main objective

Audit the complete results of **Stage 40** and produce a rigorous, manuscript-oriented statistical report.

**Do NOT run any new model training or DAGMA fitting.**

Use only the already completed Stage 40 results.

Stage 40 contains:

* 12 canonical variants
* 2 datasets: `losloop`, `shenzhen`
* PH = 1, 2, 3, 4
* seeds = 42, 43, 44, 45, 46
* 480 experiments total
* RMSE and MAE

The Stage 40 log is:

`archive/misc/stage40_run_all.txt`

and the canonical results are under the Stage 40 results directory.

### 1. First inspect the implementation

Inspect the Stage 40 code and result files to determine:

* exact result-file format and locations
* exact method names
* dataset names
* PH convention
* seed convention
* graph construction used by every variant
* which variants are GCN and which are T-GCN
* how `GCN-MultiGSL` consumes the multi-lag graphs
* how `T-GCN-MultiGSL` consumes them
* how `T-GCN-MultiGSL-Weighted` works
* how `T-GCN-MultiGSL-Mix` works

Do not assume anything from the log if it can be verified from code/results.

### 2. Reconstruct the complete results table

For every:

`dataset × PH × method`

compute:

* mean RMSE
* standard deviation of RMSE
* mean MAE
* standard deviation of MAE
* minimum RMSE
* maximum RMSE
* number of seed wins when appropriate

Use the five independent seeds 42–46.

Do not round intermediate calculations.

Produce a machine-readable CSV/JSON summary in the Stage 41 results directory.

### 3. Canonical methods

Use these display names exactly:

#### GCN family

* GCN
* GCN-NoSpatial
* GCN-GSL
* GCN-cGSL
* GCN-MultiGSL

#### T-GCN family

* T-GCN
* T-GCN-NoSpatial
* T-GCN-GSL
* T-GCN-cGSL
* T-GCN-MultiGSL
* T-GCN-MultiGSL-Weighted
* T-GCN-MultiGSL-Mix

Do not introduce names such as "Original GSL", "Original cGSL", or "Adaptive".

### 4. Improvement analysis

For each dataset and PH calculate percentage RMSE improvement relative to:

1. T-GCN
2. T-GCN-NoSpatial

Use:

`100 * (baseline - method) / baseline`

Do this using the **mean RMSE**.

Clearly distinguish:

* improvement over the physical-graph baseline
* improvement over the no-spatial baseline

This distinction is scientifically important.

### 5. Seed consistency

For the principal comparisons, report:

* wins out of 5 seeds
* mean difference
* standard deviation
* per-seed differences where useful

At minimum analyze:

* T-GCN vs T-GCN-NoSpatial
* T-GCN vs T-GCN-GSL
* T-GCN vs T-GCN-cGSL
* T-GCN vs T-GCN-MultiGSL
* T-GCN vs T-GCN-MultiGSL-Mix
* T-GCN-NoSpatial vs T-GCN-MultiGSL-Mix

Do not call a method robust merely because its mean is lower.

### 6. Statistical tests

Where statistically meaningful and technically appropriate, perform paired tests across the five seeds.

Because there are only five seeds, do not overstate statistical significance.

Report:

* test used
* p-value
* effect direction
* whether the result should be regarded as statistically convincing

If a conventional significance test is unreliable because n=5 is too small, explicitly say so.

Do not manufacture statistical significance.

### 7. Dataset comparison

Explicitly compare `losloop` and `shenzhen`.

Determine:

* which methods consistently help on Los-loop
* which methods help on SZ-Taxi
* whether the effect of GSL is dataset-dependent
* whether multi-lag modeling is consistently useful
* whether mixing/gating provides additional benefit beyond fixed multi-lag graphs
* whether the physical graph itself is beneficial

Pay special attention to the possibility that the two datasets support **different conclusions**.

Do not try to force a universal conclusion.

### 8. GCN vs T-GCN interpretation

Analyze the paired architecture results.

In particular, investigate:

* GCN-MultiGSL
* T-GCN-MultiGSL

and explain quantitatively what happens when lag-specific graphs are:

* unioned into one static graph for GCN
* consumed separately according to lag for T-GCN

This is an important scientific result.

Do not claim that multi-lag graphs are useful merely because they contain more edges.

### 9. Sparsity confound

Review the available sparse-control results from previous stages if they are already stored in the repository.

Do NOT run new experiments.

Determine what the existing results actually establish about:

* graph sparsity
* random sparse graphs
* correlation-based sparse graphs
* learned DAGMA graphs

Clearly distinguish evidence from interpretation.

### 10. Graph statistics

Collect or verify, where available:

* number of nodes
* number of edges
* graph density
* self-loops
* number of lag-specific edges
* union edge count

For physical, GSL, cGSL, and multi-lag graphs.

If a requested statistic is not available without recomputing/fitting graphs, report that it is unavailable rather than generating a new experiment.

### 11. Identify the strongest defensible findings

Produce a section:

`Strongest Defensible Findings`

List approximately 5–8 findings.

For every finding provide:

* quantitative evidence
* datasets/PHs involved
* strength of evidence
* whether it is suitable for the main manuscript

### 12. Identify claims that should NOT be made

Produce a section:

`Claims to Avoid`

Be particularly strict about:

* causal interpretation
* universal superiority of GSL
* universal superiority of multi-lag modeling
* claims that DAGMA discovers true traffic causality
* claims that sparsity alone explains the gains
* claims of statistical significance unsupported by n=5
* claims that the method adapts to changing traffic if the graph is static

Use scientifically conservative language.

### 13. Manuscript recommendation

Produce a final section:

`Recommended Main-Text Results`

Recommend:

* which methods belong in the main results table
* which methods should be supplementary/ablation results
* which comparisons are essential
* which results can be removed from the main text because they are repetitive

Do not rewrite the manuscript yet.

### 14. Reviewer-oriented audit

Map the current evidence to the relevant reviewer concerns:

* multiple seeds / variance
* sparse-graph confound
* longer horizons
* GSL vs cGSL
* temporal interpretation
* dataset dependence
* scalability
* limitations

For each issue classify:

* `Addressed`
* `Partially addressed`
* `Not addressed`

and explain why.

Do not claim that Stage 40 solved a reviewer concern unless the evidence really does.

### 15. Important scientific constraints

Be conservative.

In particular:

* DAGMA gives a statistical dependency structure; do not call it causal without evidence.
* A contemporaneous DAGMA graph should not automatically be described as a temporal graph.
* Multi-lag DAGMA graphs provide lag-specific statistical dependencies only if the construction justifies that interpretation.
* A lower RMSE on one dataset does not establish a universal property.
* Five seeds provide useful variance estimates but weak statistical power.
* Do not tune conclusions to obtain a preferred story.

### 16. Output

Create:

`gsl_stage41/`

if it does not already exist.

Save:

1. `stage41_result_audit.md`
2. `stage41_summary.csv`
3. `stage41_summary.json`
4. `stage41_claim_audit.md`

The main report must contain complete numerical tables and concise scientific interpretation.

At the end print:

* total result records audited
* number of missing results
* number of duplicate results
* datasets audited
* methods audited
* PHs audited
* seeds audited
* output paths
* final verdict: `READY`, `READY WITH CAVEATS`, or `NOT READY`

**No new training. No new DAGMA fitting. No modification of experimental code.**
