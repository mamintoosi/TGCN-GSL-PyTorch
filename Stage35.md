# Stage 35 — DAGMA Determinism, Threshold and Provenance Audit

We are preparing to run the final scientific experiments A/B/C for the revised manuscript.

The repository is currently at the Stage 33/34 state. Stage 33 introduced the canonical naming system and Stage 34 prepared experiments A/B/C.

Before running any expensive experiment, perform a focused audit of the DAGMA pipeline.

## Background

The current scientific plan is:

* Experiment A: sparse matched-edge controls
* Experiment B: canonical T-GCN-GSL rerun
* Experiment C: SZ-Taxi multi-seed validation

No full experiment has yet been executed.

The current Stage 34 report established an important correction for Experiment B:

* DAGMA `w_threshold=0.3` is explicitly passed to `fit()`.
* This corresponds to the default threshold used by the original DAGMA call in the original pipeline.
* Graph construction subsequently uses the project rule `A = 1(W > 0)`, followed by removal of the diagonal.
* DAGMA input for PH is `train_norm[0::PH]`.
* The forecasting stage uses seeds 42–46.
* Stage 34 reports that DAGMA itself appears deterministic in the current linear implementation, but this must now be audited carefully.

Do NOT run the expensive A/B/C experiments.

Do NOT modify the manuscript.

---

# Goal 1 — Audit whether DAGMA uses randomness

Inspect the actual DAGMA implementation used by this repository/environment.

Determine whether the complete execution path used by our scripts contains:

* NumPy random initialization
* Python random
* PyTorch random initialization
* random graph initialization
* random permutations
* stochastic optimization
* random subsampling
* randomized BLAS operations
* any seed-dependent initialization

Do not infer this from the DAGMA paper alone. Inspect the actual installed/library code and the exact code path used by:

`gsl_stage26/stage33_gsl_canonical.py`

Report the exact conclusion.

Distinguish:

1. algorithmic randomness;
2. numerical nondeterminism caused by parallelism/threading;
3. randomness in the downstream T-GCN forecasting model.

---

# Goal 2 — Determine whether CPU thread count can change DAGMA output

Inspect the actual numerical stack used by the current environment:

* NumPy
* SciPy
* BLAS backend
* OpenMP/threading if relevant

Determine whether running DAGMA with different thread counts could produce numerically different `W_est`.

Do not make an unsupported claim that "DAGMA is deterministic" merely because it has no explicit random seed.

If practical, create a very small synthetic determinism test using the actual DAGMA implementation.

The test should:

1. create one fixed synthetic matrix X;
2. run DAGMA twice with identical configuration;
3. compare `W_est`;
4. if practical, repeat with different thread counts such as 1 and 4;
5. compare both:

   * numerical difference in W;
   * binary adjacency after the project thresholding rule.

The test must be small and must take only a short time.

Do NOT run a full 207-node traffic DAGMA fit.

---

# Goal 3 — Verify sign and threshold semantics

Inspect both:

* DAGMA's `w_threshold` implementation;
* our project-level adjacency construction.

Establish precisely:

1. whether DAGMA can return negative weights;
2. whether `w_threshold` is applied using absolute magnitude;
3. whether the project adjacency keeps only positive weights;
4. whether a negative weight with `|W| >= 0.3` would become an edge in our current pipeline;
5. whether diagonal entries are removed.

Use the actual implementation, not assumptions.

The final report should include a compact pseudocode description such as:

```text
W = DAGMA.fit(...)
W_thresholded = ...
A = ...
A[i,i] = 0
```

Do not change the scientific rule unless there is a demonstrable implementation bug.

---

# Goal 4 — Audit the historical W_est files

Find all historical `W_est` files related to the original submission.

Determine:

* where they are located;
* which scripts currently load them;
* whether any active Stage 33/34 experiment can accidentally use them;
* whether the new canonical B experiment creates its own graph artifacts;
* whether historical files should be moved to an explicit archive location to prevent accidental use.

IMPORTANT:

Do not delete historical artifacts yet.

We want to preserve reproducibility/history while ensuring that active experiments cannot silently use old graph estimates.

If necessary, recommend a minimal repository change such as:

```text
archive/historical_submission/
```

or another clearly isolated location.

Do not move/delete files unless necessary and safe.

---

# Goal 5 — Audit Experiment B provenance

Inspect:

`gsl_stage26/stage33_gsl_canonical.py`

and verify that its graph-learning configuration is explicitly recorded.

For every PH, the provenance should make it possible to know:

* dataset
* PH
* exact DAGMA input construction
* lambda1
* w_threshold
* max_iter
* other relevant DAGMA parameters
* number of input rows
* number of graph nodes
* number of nonzero weights before/after threshold if available
* number of positive edges retained
* number of negative weights above threshold if any
* software/library version if practical

Do not run the full experiment.

If provenance is missing, make only minimal code changes to record it.

---

# Goal 6 — Distinguish DAGMA seeds from forecasting seeds

Verify that the five seeds:

```text
42, 43, 44, 45, 46
```

actually affect the forecasting stage and do not unnecessarily trigger five independent DAGMA fits.

The intended final design is:

```text
For each dataset and PH:

    DAGMA
       ↓
    one deterministic W_est
       ↓
    one adjacency

    Forecasting:
       seed 42
       seed 43
       seed 44
       seed 45
       seed 46
```

If the current implementation violates this design, identify the exact problem and propose the smallest fix.

Do not execute the expensive runs.

---

# Goal 7 — Review whether retained historical methods need canonical reruns

Briefly inspect the repository/manuscript context and give a recommendation for methods that should be rerun under the final canonical protocol.

At minimum evaluate:

* T-GCN-GSL
* GCN-GSL
* T-GCN-NoSpatial
* Physical
* T-GCN-MultiGSL
* T-GCN-MultiGSL-Mix
* T-GCN-cGSL
* GCN-cGSL

The principle is:

A method that remains a main-text representative of the old approach should preferably have a clean canonical result under the final protocol.

Do not automatically rerun anything.

Do not change manuscript tables yet.

Give a recommendation only.

---

# Deliverables

Create:

`doc/STAGE34_5_DAGMA_DETERMINISM_AND_PROVENANCE_REPORT.md`

The report must contain:

1. **DAGMA randomness audit**
2. **Thread/numerical determinism audit**
3. **Threshold and sign semantics**
4. **Historical W_est isolation audit**
5. **Experiment B provenance audit**
6. **DAGMA-seed vs forecasting-seed analysis**
7. **Recommendation on canonical reruns**
8. **Exact code changes made, if any**
9. **Tests executed and their results**
10. **Clear GO/NO-GO recommendation for experiments A/B/C**

## Execution restriction

You may execute:

* source inspection
* imports
* grep/search
* version checks
* syntax checks
* tiny synthetic determinism tests
* other tests that take only a few minutes

You must NOT execute:

* full Experiment A
* full Experiment B
* full Experiment C
* full traffic-data DAGMA fits
* multi-hour training

At the end, give me:

```text
GO/NO-GO for A
GO/NO-GO for B
GO/NO-GO for C
```

and explain any required change before I start the expensive experiments.

Do not ask me questions. Make the best technically justified decision from the repository and actual implementation.

