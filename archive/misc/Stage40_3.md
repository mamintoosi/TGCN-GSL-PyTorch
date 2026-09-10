You are continuing the `TGCN-GSL-PyTorch` project after successful Stage 40.1 and Stage 40.2 audits.

Execute **Stage 40.3: Prepare and Generate the Missing SZ-Taxi Contemporaneous DAGMA Artifacts**.

Do NOT run the full Stage 40 experiment matrix yet.

## Objective

Stage 40 is now scientifically and architecturally ready, except that the contemporaneous DAGMA graphs for SZ-Taxi are missing.

These graphs are required by:

* T-GCN-GSL
* T-GCN-cGSL
* GCN-GSL
* GCN-cGSL

for PH=1,2,3,4.

The goal of this stage is ONLY to generate and validate those missing DAGMA artifacts.

## 1. First inspect the existing Stage 33 DAGMA code

Inspect:

* `gsl_stage26/stage33_gsl_canonical.py`
* related DAGMA utilities
* Stage 40 graph loaders
* Stage 40 manifest files
* existing Los-loop contemporaneous DAGMA artifacts
* existing Stage 26/33 result structures

Determine exactly which part of the existing Stage 33 command performs DAGMA fitting and which part performs model training.

Do not assume that the command suggested in Stage 40.2 is DAGMA-only.

## 2. Do not unnecessarily train GCN/T-GCN

The immediate requirement is to create the graph artifacts.

If the existing Stage 33 script combines DAGMA fitting and model training, do NOT blindly run the full command.

Instead:

* identify whether an existing DAGMA-only execution mode exists;
* if it exists, use it;
* otherwise create a minimal DAGMA-only script or invoke the existing DAGMA function directly.

Do not redesign the Stage 33 pipeline.

The desired operation is:

For SZ-Taxi and PH=1,2,3,4:

1. load the canonical training data;
2. use the same preprocessing/protocol as the existing Los contemporaneous DAGMA;
3. construct the PH-specific DAGMA input;
4. fit DAGMA;
5. save the required raw/binary graph artifact;
6. save metadata/provenance.

Do not train forecasting models in this stage unless the existing implementation makes that unavoidable and there is no safe DAGMA-only route.

## 3. Preserve the established contemporaneous DAGMA protocol

Use exactly the protocol already established for Stage 33.

Verify from code, not assumptions:

* dataset: SZ-Taxi
* chronological 80/20 split
* normalization using training maximum
* PH-specific subsampling/input construction
* lambda1 for SZ-Taxi = 0.01
* DAGMA `w_threshold = 0.3`
* absolute-magnitude support semantics
* diagonal removal
* directed graph
* seed=42 if the fitting interface requires a seed

Do not change the threshold or lambda merely to make it match Los-loop.

The protocol must remain dataset-specific where that was already established.

## 4. Required output compatibility

The resulting artifacts must be directly consumable by Stage 40.

Inspect the existing Los-loop filenames and metadata and produce the SZ-Taxi equivalents with the same schema.

Before fitting, determine the exact filenames expected by:

`load_contemporaneous_graph()`

and the Stage 40 runner.

Do not invent an incompatible naming scheme.

## 5. Validate every generated graph

For each:

* PH1
* PH2
* PH3
* PH4

check:

* shape = `(156,156)`
* finite values
* no NaN/Inf
* diagonal removed in binary adjacency
* binary adjacency contains only 0/1
* graph is directed for GSL
* number of edges reported
* threshold/provenance recorded
* file can be loaded by the Stage 40 graph loader

If raw weights are saved, also inspect their range and nonzero count.

## 6. Verify counterpart compatibility

After generation, explicitly verify that:

`GCN-GSL` and `T-GCN-GSL`

would load the exact same SZ contemporaneous graph.

Likewise:

`GCN-cGSL` and `T-GCN-cGSL`

must derive cGSL from that same graph using the same symmetrization operation.

Do not train these models yet.

## 7. Update manifests only if necessary

If the Stage 40 artifact manifest currently says:

`SZ contemporaneous DAGMA = NOT_YET_FITTED`

update it to the appropriate completed status only after the artifacts have been successfully generated and validated.

Record:

* dataset
* PH
* lambda1
* w_threshold
* support threshold
* graph shape
* edge count
* source script
* date
* seed/provenance

Do not alter unrelated Stage 40 configuration.

## 8. Scientific sanity checks

Compare the resulting SZ graphs across PH=1..4.

Report:

* edge count
* density
* number of positive/negative retained coefficients if raw weights are available
* whether the graphs are identical across PH or PH-dependent
* major structural observations

Do not interpret the graph as causal.

Do not claim that more/fewer edges imply better predictive performance.

## 9. Runtime management

DAGMA fitting may take several hours.

It is acceptable to run the required DAGMA computation.

However:

* do not run the 480-model Stage 40 matrix;
* do not run forecasting training;
* do not recompute existing Los DAGMA artifacts;
* do not recompute Stage 26 multi-lag DAGMA artifacts.

Run only the missing SZ contemporaneous DAGMA computation.

If PH=1..4 can safely be executed in one command, do so. Otherwise run them sequentially.

## 10. Required report

Create:

`reports/stage40.3_sz_contemporaneous_dagma.md`

Include:

1. Exact code path used for DAGMA fitting
2. Exact preprocessing/input construction
3. Exact hyperparameters
4. Output files generated
5. Edge counts and densities for PH1–PH4
6. Threshold semantics
7. Loader compatibility
8. Counterpart compatibility
9. Runtime for each PH and total runtime
10. Any anomalies or warnings
11. Final readiness for Stage 40

At the end print:

```text
STAGE 40.3 VERDICT: READY
```

only if all four SZ contemporaneous DAGMA artifacts are successfully generated, validated, and loadable by Stage 40.

Otherwise print:

```text
STAGE 40.3 VERDICT: NOT READY
```

Do not start the full Stage 40 experiment matrix in this stage.
