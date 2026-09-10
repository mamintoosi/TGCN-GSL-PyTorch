# Stage 40 — Repository and Experiment-Code Restructuring

## Project Context

The project studies **Graph Structure Learning (GSL) for traffic prediction** and compares predefined physical road connectivity with graphs learned from traffic observations.

The current repository contains artifacts and code accumulated over many previous stages. The manuscript is being revised after reviewer comments, but **this Stage 40 is strictly a repository/code/experiment-organization task**.

Do **not** modify the manuscript, LaTeX files, reviewer-response letter, or manuscript wording in this stage.

The immediate goal is to create a clean, reproducible experimental structure from which the final manuscript can later be rebuilt.

---

## Main Objective

Restructure the repository and experiment code so that the following model families are explicitly and consistently represented.

### T-GCN family

Use these names:

1. `T-GCN`

   * Standard T-GCN using the physical/predefined road-network adjacency.
   * This replaces the old experimental name `Physical`.

2. `T-GCN-NoSpatial`

   * T-GCN with identity adjacency.
   * This is the no-graph baseline.

3. `T-GCN-GSL`

   * T-GCN using the contemporaneous single-graph DAGMA construction.

4. `T-GCN-cGSL`

   * T-GCN using the cyclic/symmetrized version of the single-graph learned structure.

5. `T-GCN-MultiGSL`

   * T-GCN using lag-specific DAGMA graphs assigned to input timesteps according to the corrected alignment.

6. `T-GCN-MultiGSL-Weighted`

   * MultiGSL using lag-specific learned graphs with the corresponding learned edge weights.

7. `T-GCN-MultiGSL-Mix`

   * The proposed model using per-node, per-timestep learned mixing over lag-specific graphs.

### GCN family

Also investigate and, where architecturally meaningful and supported by the existing implementation, provide the corresponding GCN family:

1. `GCN`
2. `GCN-NoSpatial`
3. `GCN-GSL`
4. `GCN-cGSL`
5. `GCN-MultiGSL`
6. `GCN-MultiGSL-Weighted`
7. `GCN-MultiGSL-Mix`

Do not invent an artificial implementation merely to obtain naming symmetry. First inspect the existing GCN architecture and determine which MultiGSL/Mix variants are technically meaningful. If a variant requires a specific adaptation, implement it consistently with the existing architecture and document the adaptation in the Stage 40 report.

---

## Important Naming Rule

The old name:

`Physical`

must be replaced in the **new experiment/code structure** by:

`T-GCN`

Do not preserve `Physical` as the primary model name in new result files, scripts, experiment configurations, plots, or generated tables.

Historical artifacts may remain untouched when they are genuinely archival and changing them would destroy provenance.

The distinction should be:

* historical artifacts = preserve
* new/restructured experimental pipeline = use the new canonical names

---

## Critical Requirement: Reuse Existing DAGMA Outputs

Before running or generating any DAGMA computation:

1. Search the entire repository for existing DAGMA outputs.
2. Identify their provenance, dimensions, horizons/lags, dataset, threshold, lambda, and construction.
3. Determine whether they can be reused for the new experiment organization.
4. Reuse valid existing DAGMA outputs whenever possible.
5. **Do not recompute an expensive DAGMA fit if an equivalent valid output already exists.**
6. If an output is ambiguous, do not silently reuse it. Record the ambiguity and inspect the generating code/configuration.
7. Clearly distinguish:

   * contemporaneous single-graph DAGMA outputs
   * lag-specific/multi-lag DAGMA outputs
   * historical outputs
   * outputs generated for the new canonical experiment pipeline

DAGMA computation is the major expensive component, so avoiding unnecessary recomputation is a primary requirement.

---

## Repository Restructuring

Inspect the complete repository before modifying anything.

The goal is to eliminate the current situation in which related DAGMA outputs, scripts, configurations, and results are scattered across different historical folders.

Create a clean structure in which the experimental pipeline is understandable from the directory tree.

In particular:

* Keep DAGMA-related outputs for the relevant experiment family together.
* Prefer a single coherent `gsl_stage40/` (or similarly clearly named Stage-40 directory) for the new canonical GSL experiment artifacts rather than scattering equivalent outputs between `gsl_stage26`, miscellaneous result directories, and unrelated folders.
* Do not delete historical artifacts.
* Do not overwrite historical results.
* If files must be moved, preserve provenance through clear names or a manifest.
* Separate:

  * source code
  * experiment configurations
  * DAGMA graph outputs
  * model-training results
  * figures
  * logs
  * manifests/metadata

The resulting structure should make it possible to answer:

> Which graph was used by this experiment, how was it generated, which dataset/horizon/lag does it correspond to, and which model consumed it?

without searching through unrelated historical folders.

---

## First Step: Repository Audit

Before making changes, inspect:

* current directory tree
* Git status
* existing experiment scripts
* DAGMA scripts
* model implementations
* configuration files
* result JSON/CSV files
* graph files
* figure-generation scripts
* previous Stage reports
* historical GSL artifacts

Identify all existing implementations corresponding to:

* T-GCN
* no-spatial baseline
* physical graph
* single-graph GSL
* cGSL
* MultiGSL
* Weighted MultiGSL
* Mix MultiGSL
* GCN counterparts

Also identify which experiments already exist for:

* Los-loop
* SZ-Taxi
* different prediction horizons
* different seeds
* different graph constructions

Do not modify anything during this audit phase.

Produce an inventory before restructuring.

---

## Canonical Experiment Matrix

Build a machine-readable experiment configuration/manifest covering at least:

### Dataset

* Los-loop
* SZ-Taxi

### Model family

* GCN
* T-GCN

### Graph/model variant

* NoSpatial
* physical/T-GCN
* GSL
* cGSL
* MultiGSL
* MultiGSL-Weighted
* MultiGSL-Mix

### Prediction horizons

Use the horizons already supported by the existing experimental protocol.

Do not silently change the established horizon definition.

### Seeds

Use the existing five-seed policy where applicable:

`42, 43, 44, 45, 46`

Do not rerun an experiment merely because its result already exists.

The manifest should explicitly identify whether each result is:

* already available and reusable
* requires model training
* requires DAGMA computation
* requires neither

---

## DAGMA Organization

Create a canonical DAGMA artifact structure.

Every DAGMA output should have enough metadata to identify:

* dataset
* construction type
* contemporaneous vs lagged
* horizon
* lag index
* number of variables
* lambda
* threshold
* edge-selection rule
* random seed, if applicable
* source script/configuration
* generation stage/date where known

Do not rely on filenames alone when metadata can be stored separately.

A manifest such as JSON/YAML/CSV is strongly preferred.

### Threshold semantics

Preserve the experimentally established threshold semantics.

If thresholding uses absolute magnitude, explicitly encode that in the graph-generation code/configuration.

Do not remove cGSL merely because the threshold uses absolute values.

The fact that negative coefficients may or may not survive thresholding is an empirical property of the resulting graph, not a reason to silently eliminate the cyclic/symmetrized model family.

---

## Historical vs Canonical Results

Do not destroy historical reproducibility.

The repository must distinguish between:

### Historical artifacts

Results generated under previous experimental protocols.

These should remain identifiable as historical.

### Canonical Stage-40 artifacts

Results intended for the final experimental pipeline.

These must use the new naming convention and standardized directory structure.

Do not silently relabel historical numerical results as new results.

---

## Model Naming and Code Interface

Where practical, introduce a canonical model identifier such as:

```text
T-GCN
T-GCN-NoSpatial
T-GCN-GSL
T-GCN-cGSL
T-GCN-MultiGSL
T-GCN-MultiGSL-Weighted
T-GCN-MultiGSL-Mix

GCN
GCN-NoSpatial
GCN-GSL
GCN-cGSL
GCN-MultiGSL
GCN-MultiGSL-Weighted
GCN-MultiGSL-Mix
```

Use these identifiers consistently in:

* experiment configurations
* result files
* logs
* scripts
* figure generation
* table-generation scripts
* manifests

Avoid having multiple aliases for the same method in the new pipeline.

---

## Correctness Checks

Before running expensive experiments, verify:

1. Existing DAGMA outputs have been correctly mapped to the new names.
2. `T-GCN` is exactly the existing physical-graph T-GCN baseline, except for naming/organization.
3. `T-GCN-NoSpatial` uses identity adjacency.
4. Single-graph GSL uses the contemporaneous construction established by the previous scientific audit.
5. cGSL is implemented according to the established symmetrization definition.
6. MultiGSL uses the corrected lag alignment.
7. MultiGSL-Weighted and MultiGSL-Mix are distinct implementations and are not accidentally aliases.
8. Dataset-specific preprocessing is unchanged unless a concrete bug is found.
9. No historical result is accidentally overwritten.
10. Existing DAGMA outputs are reused whenever they are valid equivalents.

Perform lightweight smoke tests where possible.

---

## Experiments to Prepare

The objective of Stage 40 is to prepare the complete experiment pipeline.

For short/lightweight tests, you may execute them yourself.

For expensive/main experiments:

**Do not execute them.**

Instead:

* create the required Python scripts
* create configuration files
* create Bash scripts
* create batch execution scripts where useful
* make them resumable
* make them detect existing outputs
* make them skip already completed experiments
* make logs explicit
* make output paths deterministic

I will execute the expensive/main scripts myself.

(for an example, see 'run_resolution_experiment.sh')

This includes long DAGMA fits and large batches of model training.

---

## Resumability

The main experiment launcher must support interruption and continuation.

For every experiment:

* detect an existing valid result
* skip it
* otherwise execute it
* write results atomically where practical
* record completion status
* preserve logs

The launcher should make it safe to run the same command again after interruption.

Do not force a complete rerun merely because one experiment failed.

---

## Figures and Tables

Do not edit the manuscript.

However, update the **code that generates figures and tables** so that it understands the new canonical names.

Regenerate affected figures/tables only when this can be done from existing results without expensive retraining.

In particular inspect all visualizations and result-generation code affected by:

* `Physical` → `T-GCN`
* addition/organization of GCN variants
* GSL/cGSL naming
* MultiGSL naming
* dataset comparison
* model-family comparison

Generated figures must use canonical method names.

Do not manually edit generated figures.

---

## Los-loop and SZ-Taxi

The new structure must support both datasets.

Do not design the pipeline around Los-loop only.

Every model/variant should have a clear status for both:

* Los-loop
* SZ-Taxi

If an existing method was previously evaluated only on one dataset, do not fabricate the missing result.

Instead, mark it as:

`missing / requires execution`

and generate the script needed to obtain it.

This is important because the final manuscript will compare behavior across both datasets.

---

## Git Safety

Before modifications:

```bash
git status
```

Record the initial state.

Do not delete or modify historical experiment artifacts unless explicitly required for restructuring and their provenance is preserved.

At the end:

```bash
git status
git diff --stat
git diff
```

Inspect the diff carefully.

Do not commit anything.

---

## Required Final Deliverables

At the end of Stage 40, provide:

### 1. Repository restructuring report

Create:

```text
doc/STAGE40_REPOSITORY_RESTRUCTURE_REPORT.md
```

Include:

* initial repository structure
* final structure
* files moved
* files renamed
* files created
* historical artifacts preserved
* DAGMA outputs discovered
* DAGMA outputs reused
* DAGMA outputs still missing
* model variants implemented
* model variants already available
* model variants requiring execution

### 2. Experiment manifest

Create a machine-readable manifest containing the complete experiment matrix.

### 3. Main execution scripts

Create scripts for the expensive experiments, but **do not run the expensive/main jobs**.

The scripts must:

* reuse existing DAGMA outputs
* skip existing valid results
* be resumable
* log progress
* save results in the canonical Stage-40 structure

### 4. Validation report

Run only lightweight validation/smoke tests yourself.

Verify imports, configurations, paths, graph dimensions, adjacency formats, and model construction.

Do not claim full experimental validation unless the corresponding experiment has actually been executed.

### 5. User execution instructions

At the end of the Stage-40 report, provide the exact commands I should run to execute:

* DAGMA jobs that are genuinely missing
* GCN experiments
* T-GCN experiments
* Los-loop experiments
* SZ-Taxi experiments
* any remaining figure/table regeneration that depends on newly generated results

Separate short tests from long-running jobs.

---

## Important Restrictions

* Do not modify the manuscript.
* Do not modify `Response-to-Reviewers.md`.
* Do not write reviewer responses.
* Do not make scientific claims based on experiments that have not been run.
* Do not rerun existing DAGMA fits unnecessarily.
* Do not delete historical artifacts.
* Do not overwrite historical results.
* Do not ask me for decisions during the task.
* Make reasonable technical decisions based on the existing repository and previous Stage reports.
* If something genuinely cannot be determined from the repository, record it explicitly in the Stage-40 report rather than guessing.

## Final Goal

At the end of Stage 40, the repository should have a clean, canonical, reproducible experimental pipeline covering the GCN/T-GCN model families, both datasets, all technically meaningful graph variants, and the existing DAGMA artifacts.

The repository should then be ready for me to run the long experiments and obtain the final numerical results.

Only after those experiments are complete should we return to the manuscript and reviewer-response documents.
