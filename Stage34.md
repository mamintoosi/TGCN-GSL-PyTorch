# Stage 34 — Final Experimental Runs A/B/C

We are now at the experimental stage of the revised manuscript for the paper
**“Graph Structure Learning for Traffic Prediction.”** 

First read `paper/sn-article.tex` 

You have access to the complete repository and its history.

## Important repository context

The latest repository commit introduced the Stage 33 code-naming harmonization and experiment preparation.

In particular:

* `models/multigsl.py` is now the canonical implementation/registry for the multi-graph T-GCN variants.

* The manuscript-facing method names are now:

  * `T-GCN-NoSpatial`
  * `Physical`
  * `T-GCN-MultiGSL`
  * `T-GCN-MultiGSL-Mix`

* `T-GCN-MultiGSL-Mix` is the proposed method.

* The word **“Adaptive” must not be introduced into the new terminology.

* `normalize_method()` provides compatibility with historical method names.

* Historical result files and checkpoint directories were intentionally NOT renamed.

* Stage 33 also verified the refactoring with a 10/10 canary test, including numerical faithfulness of the refactored `MultiGSL` and `MultiGSL-Mix` implementations.

Please inspect the current repository state before doing anything else. Do not undo the Stage 33 refactoring.

The detailed Stage 33 documents are:

* `doc/STAGE33_CODE_NAMING_AND_EXPERIMENT_PREPARATION_REPORT.md`
* `doc/METHOD_NAMING_MAP.md`

The current manuscript is:

* `paper/sn-article.tex`

The purpose of this stage is **not to rewrite the manuscript yet**. We first need the remaining scientific experiments.

---

# Overall scientific objective

The revised paper is organized around the following scientific story:

1. Dense physical graphs can hurt T-GCN because of oversmoothing.
2. Sparse learned graph structure can outperform the dense physical graph.
3. A single graph may conflate dependencies associated with different temporal lags.
4. Lag-specific graphs expose structurally different temporal dependencies.
5. `T-GCN-MultiGSL` provides fixed lag-specific graph assignment.
6. `T-GCN-MultiGSL-Mix` learns how to mix these lag-specific graphs per node and timestep.
7. We must demonstrate that the improvement is not merely caused by sparsity.
8. We must establish the canonical single-graph GSL baseline under the same experimental protocol.
9. We must determine whether the marginal result on SZ-Taxi is stable across random seeds.

Three experiments remain especially important:

* **A — Sparse matched-edge controls**
* **B — Canonical T-GCN-GSL rerun**
* **C — SZ-Taxi multi-seed validation**

Do NOT invent additional expensive experiments unless they are necessary to make A/B/C scientifically valid.

---

# Experiment A — Sparse matched-edge controls

## Scientific question

Is the improvement of `T-GCN-MultiGSL-Mix` explained simply by having a sparse graph rather than by the proposed lag-specific graph/mixing mechanism?

This directly addresses the sparsity/structure concern.

## Required experiment

Dataset:

* Los-loop

Prediction horizon:

* PH=1

Seeds:

* 42, 43, 44, 45, 46

Compare at least:

* `CorrTop30`
* `RandTop30`
* `T-GCN-NoSpatial`
* `T-GCN-MultiGSL`
* `T-GCN-MultiGSL-Mix`

Use the existing Stage 32 sparse-control implementation where appropriate.

The graph-budget comparison must be explicit and auditable.

Important:

* Preserve the existing MultiGSL edge-count convention.
* The Stage 33 report notes that the three lag graphs contain 30 edges in total, while their union contains 28 unique edges.
* Do not silently change this convention.

## What to do

First inspect:

`gsl_stage26/stage32_sparse_control.py`

and the existing Stage 32 artifacts and protocol.

Reuse existing verified DAGMA matrices when appropriate. **Do not recompute DAGMA unnecessarily.**

If code modification is required, make the smallest scientifically justified modification.

Then:

1. Run only a short smoke test yourself if necessary.
2. Prepare/fix the final experiment script.
3. Do NOT run the full A experiment automatically.
4. Give me the exact command I should execute.

Expected final artifact:

`results/stage32_sparse_control/stage32_sparse_control.json`

The result should contain enough information to compare:

* method
* seed
* number of edges
* RMSE
* MAE
* parameter count
* relevant graph-budget information

---

# Experiment B — Canonical T-GCN-GSL rerun

## Scientific question

The original single-graph GSL result is the empirical starting point of the revised story.

We need to establish the performance of a **single DAGMA-learned graph under the canonical experimental protocol**, rather than relying on the old protocol.

This result should potentially become a main-text baseline.

## Required protocol

Dataset:

* Los-loop

Prediction horizons:

* PH=1, 2, 3, 4

Seeds:

* 42, 43, 44, 45, 46

Primary method:

* `T-GCN-GSL`

The comparison should include the appropriate physical baseline under the same protocol.

The canonical protocol from Stage 33 specifies at least:

* batch size = 128
* weight decay = `1e-4`

Inspect the current Stage 33 implementation and repository before changing anything.

The DAGMA graph must be generated from the correct training data and must have clean provenance.

Do not reuse an incompatible historical graph if the experimental protocol requires a fresh canonical DAGMA fit.

The Stage 33 report explicitly identifies this as the experiment that supplies the ACT-I main-text single-graph GSL result.

## Critical scientific constraint

Do not assume in advance that:

`T-GCN-NoSpatial < T-GCN-GSL < T-GCN-MultiGSL-Mix`

or any other ordering must hold.

The experiment must report what actually happens.

If the result differs from the expected narrative, do not manipulate the experiment to force the desired ordering. Report the actual result and explain its implications.

## What to do

Inspect:

`gsl_stage26/stage33_gsl_canonical.py`

and:

`run_stage33B_gsl_canonical.sh`

Verify:

* data split
* DAGMA input construction
* thresholding
* positive-weight handling
* adjacency construction
* training protocol
* seed handling
* PH handling
* model naming
* result recording
* checkpoint/output provenance

Run only a small smoke test yourself if needed.

Do NOT run the full B experiment automatically.

Give me the exact command for the full run.

Expected artifact:

`results/stage33_gsl_canonical/stage33_gsl_canonical_results.json`

The JSON should contain sufficient provenance to reconstruct:

* dataset
* PH
* seed
* model/backbone
* graph variant
* number of edges
* RMSE
* MAE
* parameter count
* protocol parameters

The fresh DAGMA artifacts should also be retained as specified by Stage 33.

---

# Experiment C — SZ-Taxi multi-seed validation

## Scientific question

The current manuscript reports only a marginal advantage of `T-GCN-MultiGSL-Mix` on SZ-Taxi.

For example, the current single-seed result is approximately:

* PH=1: 4.108 vs 4.116
* PH=4: 4.221 vs 4.221

This could be a genuine small effect or simply random-seed variation.

We need a 5-seed experiment.

## Required protocol

Dataset:

* SZ-Taxi

Prediction horizons:

* PH=1, 2, 3, 4

Seeds:

* 42, 43, 44, 45, 46

Methods:

* `T-GCN-NoSpatial`
* `T-GCN-MultiGSL`
* `T-GCN-MultiGSL-Mix`

Reuse the already verified SZ-Taxi DAGMA blocks.

**Do not recompute DAGMA unless inspection proves that the existing matrices are incompatible with the current canonical protocol.**

The main purpose is to obtain mean ± standard deviation across seeds and determine whether the apparent marginal improvement is stable.

## What to do

Inspect:

`gsl_stage26/stage33_sz_multiseed.py`

and:

`run_stage33C_sz_multiseed.sh`

Verify that:

* the correct SZ-Taxi data are used;
* existing DAGMA blocks are reused correctly;
* all three methods use exactly the same data/protocol;
* seeds are actually propagated to model/training initialization;
* PH=1..4 are all evaluated;
* the result JSON is complete.

Run only a short smoke test yourself if necessary.

Do NOT run the full C experiment automatically.

Give me the exact command for the full run.

Expected artifact:

`results/stage33_sz_multiseed/stage33_sz_multiseed_results.json`

---

# Execution policy

This is important.

You are operating on the same repository, but I will execute the expensive experiments myself.

Therefore:

### You MAY execute

* imports
* syntax checks
* CLI parsing tests
* model construction tests
* very short smoke tests
* a few epochs on a tiny configuration
* tests taking roughly a couple of minutes or less

### DO NOT execute automatically

* full DAGMA training
* full multi-seed training
* the complete A/B/C experiments
* multi-hour GPU jobs

Instead, inspect and prepare the code and provide the exact commands for me.

---

# Before giving me the commands

For each A/B/C:

1. Inspect the current implementation.
2. Check whether the existing script is actually correct.
3. Check compatibility with the Stage 33 naming refactor.
4. Check that no historical result is accidentally overwritten.
5. Check that the result files contain sufficient provenance.
6. Run a short smoke test if useful.
7. If you modify code, clearly state what was changed and why.
8. Do not modify the manuscript's scientific claims yet.

If an existing script is already correct, do not rewrite it unnecessarily.

---

# Important issue: manuscript integration

While preparing the experiments, also inspect the current manuscript and think about how the results should eventually be integrated.

The goal is NOT to preserve old results merely because they existed in an earlier submission.

The final manuscript must be a coherent scientific story for a reader who has never seen the previous version.

Therefore:

* Do not describe results as “old”, “previous”, “historical”, or “kept for continuity” in the main scientific narrative.
* The main text should contain the results that are scientifically necessary for the revised story.
* Protocol-incompatible historical results may remain in the appendix only when useful for reproducibility/transparency.
* The main text should not become a catalogue of every experiment ever performed.
* `GCN-cGSL` and `T-GCN-cGSL` do not need to be promoted to the main story unless there is a strong scientific reason.
* The core of the previous GSL approach should remain represented by a clean `T-GCN-GSL` baseline.
* The new methods should be presented as a methodological progression rather than as unrelated alternatives.

Think specifically about how A, B and C will fit into the Results section without making it unnecessarily long.

Do not rewrite `results.tex` at this stage unless a tiny correction is necessary for experimental correctness.

---

# Deliverable

At the end, provide a concise Stage 34 report containing:

## 1. Code verification

For A, B, C:

* script status
* required modifications
* smoke-test status
* whether the script is ready for full execution

## 2. Exact execution commands

Give me copy-paste-ready commands for:

```bash
# A
...

# B
...

# C
...
```

Use the repository's actual Python environment and current paths.

## 3. Expected outputs

List the expected result files/directories.

## 4. Runtime estimates

Give realistic GPU runtime estimates based on the current implementation.

## 5. Scientific role

For each experiment, explain in 1–3 sentences what question it answers and where its result is likely to appear in the final manuscript.

## 6. Do not fabricate results

No scientific result should be invented or inferred before the experiments are actually run.

The purpose of this stage is to leave the repository in a **clean, reproducible, execution-ready state** for A/B/C while leaving the expensive computation to me.

