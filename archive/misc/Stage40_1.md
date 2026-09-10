You are working on the current `TGCN-GSL-PyTorch` repository.

Execute **Stage 40.1: Pre-Execution Scientific and Code Audit** before any long Stage 40 experiments.

Do NOT run the full experiment matrix or other long GPU experiments. Short smoke tests, code inspection, shape checks, and existing-result inspection are allowed.

## Main objective

Audit the Stage 40 implementation and determine whether the planned 11-model matrix is scientifically and technically correct.

An important issue must be explicitly investigated:

A standard GCN requires an `N x N` graph matrix. Therefore, if DAGMA is run for multiple temporal lags and produces

`A_0, A_1, ..., A_L`,

these matrices can be combined into a single `N x N` matrix, for example by union, and that combined graph can be directly supplied to a standard GCN.

Therefore, do NOT assume that multi-lag DAGMA is inherently incompatible with GCN.

Investigate whether a meaningful `GCN-MultiGSL` baseline should be added.

## 1. Inspect the current Stage 40 code

Inspect:

* `gsl_stage40/`
* `models/multigsl.py`
* all Stage 40 experiment scripts
* configuration files
* graph-loading/reuse code
* existing Stage 26 and Stage 33 result directories
* any scripts used to construct DAGMA graphs

Verify the actual implementation rather than relying on reports.

## 2. Audit the model definitions

For every current variant, verify:

### GCN family

* `GCN`
* `GCN-NoSpatial`
* `GCN-GSL`
* `GCN-cGSL`

### T-GCN family

* `T-GCN`
* `T-GCN-GSL`
* `T-GCN-cGSL` if implemented/planned
* `T-GCN-MultiGSL`
* `T-GCN-MultiGSL-Weighted`
* `T-GCN-MultiGSL-Mix`
* `T-GCN-GatedMultiGSL`

Check:

* exact graph input
* tensor shapes
* whether the graph is static or timestep/lag-specific
* where graph convolution occurs
* whether the implementation actually matches the intended scientific description
* whether any variant is only a naming variant rather than a genuinely different model

Report any mismatch.

## 3. Explicitly investigate GCN-MultiGSL

Determine whether the following is technically valid in the current codebase:

DAGMA:
`A_0, A_1, ..., A_L`

then:

`A_union = 1(A_0 + A_1 + ... + A_L > 0)`

then:

`GCN(X, A_union)`

Also inspect whether a weighted combination is technically possible:

`A_weighted = sum_l alpha_l A_l`

Do not implement a new architecture yet unless necessary for a smoke test.

Determine the cleanest and fairest definition of a `GCN-MultiGSL` baseline.

Prefer a simple, parameter-free union baseline if scientifically appropriate.

Important:
The purpose is to test whether preserving separate lag-specific graphs is necessary, compared with simply combining those graphs into one graph before GCN.

## 4. Compare the conceptual roles

Clearly distinguish:

### Single-graph GSL

One DAGMA graph is supplied to GCN/T-GCN.

### Combined multi-lag GSL for GCN

Several DAGMA graphs are combined into one `N x N` graph and supplied to GCN.

### Multi-graph T-GCN

Several lag-specific DAGMA graphs remain separate and are used by different graph/temporal branches.

Explain whether these represent genuinely different hypotheses.

## 5. Audit DAGMA graph provenance

For every reusable graph file, determine:

* dataset
* train/test split
* input representation
* lag definition
* number of nodes
* DAGMA threshold
* whether raw weights or thresholded adjacency were saved
* whether absolute or signed thresholding was used
* whether self-loops are present
* whether the graph is directed or symmetrized
* exact source experiment/stage

Reuse existing valid DAGMA outputs whenever possible.

Do NOT recompute DAGMA merely because Stage 40 uses a different experiment name.

Flag any graph whose provenance is insufficient.

## 6. Audit threshold semantics

Verify the exact relationship between:

* DAGMA `w_threshold`
* consumer-side threshold
* absolute-value thresholding
* positive-only thresholding
* cGSL symmetrization

In particular, verify the previously observed distinction between multi-lag DAGMA outputs and contemporaneous DAGMA outputs.

Do not silently change threshold semantics.

## 7. Audit T-GCN vs GCN architecture

Verify that:

* GCN processes the graph/sequence exactly as implemented.
* T-GCN applies graph processing consistently with the temporal GRU mechanism.
* Any claim that a multi-lag construction is impossible for GCN is rejected unless code inspection demonstrates a genuine architectural limitation.

If GCN-MultiGSL is possible, document exactly how it should be implemented without artificially changing the basic GCN architecture.

## 8. Audit the experiment matrix

Propose the final Stage 40 matrix after the audit.

If `GCN-MultiGSL` is technically and scientifically valid, explicitly recommend whether to add it.

For each proposed model, state:

* graph source
* graph construction
* graph combination
* architecture
* trainable additional parameters, if any
* whether existing results can be reused
* whether a new training run is required

Do not run those long experiments yet.

## 9. Audit naming

The canonical name must be:

`T-GCN`

for the physical/static-graph baseline.

Do not use `Physical` in the canonical Stage 40 model names.

Do not introduce the word `Adaptive` into model names.

Use consistent naming across:

* Python registry
* scripts
* result directories
* CSV/JSON outputs
* plots
* tables

## 10. Audit result reuse

Build a clear reuse table:

| Required result | Existing source | Reusable? | Reason |
| --------------- | --------------- | --------- | ------ |

Especially inspect:

* Stage 26 multi-lag DAGMA outputs
* Stage 33 contemporaneous DAGMA outputs
* existing five-seed results
* sparse controls
* Los 15-minute results

Do not duplicate expensive DAGMA computation if an existing output is scientifically identical.

## 11. Audit the Stage 40 runner

Verify:

* command-line arguments are actually honored
* seed handling is correct
* dataset selection works
* PH selection works
* model selection works
* graph selection works
* output paths are deterministic
* existing results are not accidentally overwritten
* no hidden fixed `SEED=42`
* no accidental single-seed execution when multiple seeds are requested

Use a short smoke test if necessary.

## 12. Check scientific fairness

For every comparison, verify that differences are attributable to the intended graph/model change rather than:

* different data split
* different normalization
* different threshold
* different number of training epochs
* different random seed
* different graph density
* accidental extra trainable parameters
* different input representation

Do not modify the experimental protocol yet. Report problems first.

## 13. Deliverable

Create a concise but technically detailed report:

`reports/stage40.1_pre_execution_audit.md`

The report must contain:

1. Executive verdict
2. Current Stage 40 architecture audit
3. GCN-MultiGSL feasibility analysis
4. T-GCN multi-lag analysis
5. DAGMA provenance/reuse audit
6. Threshold audit
7. Seed/runner audit
8. Proposed final Stage 40 model matrix
9. Required code changes before long runs
10. Existing results that can be reused
11. Estimated number of genuinely new training experiments
12. Explicit statement of whether Stage 40 is ready to run

If code changes are necessary for Stage 40.1, make only **small, structural fixes required for correctness**. Do not start the long experiments.

At the end, print a short terminal summary with:

* PASS items
* FAIL items
* WARNING items
* final recommendation: `READY` or `NOT READY`

The goal is to make Stage 40 scientifically defensible and internally consistent before spending substantial GPU time.
