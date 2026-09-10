You are working directly inside the repository:

/data/git/mamintoosi/TGCN-GSL-PyTorch

We are preparing the final experimental phase of the revision of the paper:

"Graph Structure Learning for Traffic Prediction"

# IMPORTANT CONTEXT

The repository contains the current revised manuscript and the complete experimental history.

Please inspect the repository itself rather than relying only on this prompt.

Important manuscript files:

* paper/sn-article.tex
* paper/sections/*.tex
* paper/appendix/*.tex
* paper/sn-article_original.tex

Important reports:

* paper/STAGE31_MANUSCRIPT_RECONSTRUCTION.md
* paper/STAGE32_MANUSCRIPT_RESTRUCTURING_REPORT.md
* paper/revision_notes.md

Important historical reference:

* Git commit:
  dac89785f47968d4aaf1d728b6ec6556d9874e1e

This commit corresponds approximately to the submitted version of the paper.

The original submitted manuscript is also preserved as:

* paper/sn-article_original.tex

The current manuscript has already been substantially restructured. DO NOT undo that restructuring.

The current naming intended for the manuscript is:

1. T-GCN-NoSpatial
   T-GCN with identity adjacency.

2. Physical
   T-GCN with the predefined physical road-network adjacency.

3. T-GCN-MultiGSL
   T-GCN using separate lag-specific learned graphs assigned to input timesteps in a fixed pattern.

4. T-GCN-MultiGSL-Mix
   The proposed method: learned per-node, per-timestep mixing over the lag-specific graphs.

DO NOT introduce "Adaptive" into the names of the new methods.
In particular, do NOT rename the proposed method to anything containing "Adaptive".

The current conceptual distinction is:

* MultiGSL = fixed assignment of lag-specific graphs.
* MultiGSL-Mix = learned mixing/gating of lag-specific graphs.

The implementation class may internally use a term such as GatedMultiGraphTGCN if necessary, but all externally visible experiment/model names should follow the manuscript terminology above. Prefer eliminating unnecessary terminology duplication between code and manuscript.

# BACKGROUND FROM THE FORENSIC AUDITS

Stage 27 was found invalid for forecasting because:

* it used the wrong model class for "MultiGSL-Mix";
* it actually used standard TGCN with a static union graph;
* it used a different training loop;
* it used a different loss;
* it had sequence-construction differences;
* it mixed normalized and denormalized RMSE reporting.

Therefore:

DO NOT reuse Stage 27 forecasting numbers.

Stage 29 was independently verified as valid:

* canonical Stage 26 pipeline;
* correct GatedMultiGraphTGCN implementation;
* 5 seeds;
* 4 prediction horizons;
* Los-loop 15-minute data;
* PH=1 improvement +27.44%;
* PH=2,3,4 improvements +21.36%, +19.76%, +16.09%.

Stage 31/32 also established that the original and revised research are best understood as one methodological evolution:

Single-graph GSL
->
recognition of dense-graph oversmoothing
->
multi-lag graph construction
->
learned mixing over lag-specific graphs.

The current manuscript already follows this general narrative.

# CURRENT EXPERIMENTAL STATUS

The important verified results currently available are:

Los-loop, 5-minute resolution:

* T-GCN-NoSpatial: approximately 5.143
* T-GCN-MultiGSL: approximately 4.715
* T-GCN-MultiGSL-Mix: approximately 4.458
* five-seed mean for MultiGSL-Mix: 4.452 ± 0.143
* improvement over NoSpatial: 14.9%

Los-loop, 15-minute resolution:

* T-GCN-NoSpatial: 8.600 ± 0.249
* T-GCN-MultiGSL: 7.246 ± 0.298
* T-GCN-MultiGSL-Mix: 6.240 ± 0.187
* improvement at PH=1: 27.44%

SZ-Taxi:

* T-GCN-MultiGSL-Mix provides only marginal improvement over NoSpatial;
* PH=1 improvement is approximately +0.19%;
* PH=2 approximately +0.26%;
* PH=3 approximately +0.11%;
* PH=4 approximately -0.02%;
* the learned lag structure is extremely sparse compared with Los-loop.

The simple hypothesis that 15-minute temporal resolution explains the weak SZ-Taxi result is therefore NOT supported.

# TASK

This stage has FOUR goals.

DO NOT run expensive experiments.

DO NOT run DAGMA.

DO NOT train the full experiment suite.

DO NOT modify the scientific results.

## Goal 1 — Harmonize CODE NAMING

Inspect all relevant Python source files, experiment scripts, model definitions, functions, dictionaries, configuration objects, and command-line options.

Create a coherent naming scheme based on the current manuscript terminology.

The canonical externally visible names should be:

* T-GCN-NoSpatial
* Physical
* T-GCN-MultiGSL
* T-GCN-MultiGSL-Mix

Also establish sensible Python identifiers, for example:

* no_spatial
* physical
* multi_gsl
* multi_gsl_mix

or another clean naming scheme if the repository architecture suggests something better.

Do NOT mechanically rename every historical file.

Historical stage names such as:

* stage26
* stage27
* stage29
* stage32

must remain unchanged because they identify historical experiments.

Likewise, historical result directories must not be renamed merely for cosmetic reasons.

However, active/current experiment code should use the new canonical terminology consistently.

Pay particular attention to:

* model registries;
* method dictionaries;
* command-line choices;
* experiment names;
* result JSON keys;
* logging strings;
* comments;
* function names;
* variable names;
* plot labels;
* table-generation code;
* source-code comments that are copied into the manuscript.

Where a legacy name is necessary to read historical results, add a clear compatibility mapping rather than silently changing historical artifacts.

Create a concise mapping in a repository documentation file, e.g.:

LEGACY NAME -> CANONICAL NAME -> IMPLEMENTATION CLASS

For example:

T-GCN-MultiGSL-Mix
-> canonical manuscript name
-> GatedMultiGraphTGCN

T-GCN-MultiGSL
-> canonical manuscript name
-> MultiGraphTGCNFixed

But verify the actual implementation classes before writing the mapping.

IMPORTANT:
Do not rename the proposed method to "Adaptive...".
The manuscript terminology must remain MultiGSL-Mix.

## Goal 2 — VERIFY THE REFACTOR WITH A SMALL CANARY TEST

After the naming/refactoring changes:

DO NOT run the complete experiments.

Run only a small, fast canary test that verifies:

1. all relevant modules import successfully;
2. all four model variants can be instantiated;
3. the MultiGSL model receives the expected lag-specific graphs;
4. the MultiGSL-Mix model receives the expected lag-specific graphs;
5. a tiny synthetic batch can pass through each model;
6. loss calculation works;
7. one optimizer step works;
8. output tensor shapes are correct;
9. command-line argument parsing works for the planned experiment scripts.

The canary must be deliberately tiny.

Report:

* exact command;
* duration;
* models tested;
* tensor shapes;
* PASS/FAIL;
* any warnings.

Do not interpret canary results as scientific results.

## Goal 3 — UPDATE MANUSCRIPT SOURCE-CODE REFERENCES

The current manuscript contains statements near figures/tables identifying the source code or experiment artifact associated with them.

Inspect ALL files under:

paper/

especially:

* sections/results.tex
* sections/experiments.tex
* appendix/*.tex
* figure captions;
* table notes;
* comments;
* source-code references.

Update those references so that they use the new canonical experiment/model names.

Do not change scientific content merely for naming purposes.

Do not alter numerical values.

Do not remove historical provenance information where it is useful.

Where a figure/table is generated by code whose filename itself is historical (for example a stage-specific script), distinguish:

* historical artifact filename
  from
* current canonical method name.

Do not rename historical experiment scripts just to make source references look modern.

After these edits, compile the manuscript.

Verify:

* zero LaTeX errors;
* zero undefined references;
* zero undefined citations;
* no newly introduced overfull boxes of concern.

## Goal 4 — PREPARE THE MAIN EXPERIMENTS, BUT DO NOT RUN THEM

Prepare clean scripts and Linux bash runners for the remaining experiments that should be used for the final manuscript.

Do NOT execute the expensive experiments.

The scripts must be ready for me to run later on the Linux/GPU machine.

First inspect the current repository and determine exactly which experiments are scientifically necessary.

At minimum evaluate whether the following should be prepared:

A. Stage 32 sparse-control experiment

Purpose:
Determine whether the gain of MultiGSL-Mix is explained merely by sparsity.

The intended controls are:

* CorrTop30
* RandTop30

using the canonical Stage 26/29 training/evaluation pipeline.

Use training data only when constructing the correlation graph.

Use a matched edge budget consistent with the multi-lag reference.

Preserve the edge-count distinction between:

* sum of per-lag edge slots;
* union of distinct edges.

Do not accidentally introduce data leakage.

B. Original/single-graph GSL baseline under the canonical protocol

Determine whether any important original GSL/T-GCN baseline needs to be rerun because its original implementation/protocol differs from the final canonical protocol.

In particular investigate:

* normalization;
* loss;
* sequence construction;
* train/test split;
* graph threshold;
* evaluation;
* seed handling.

Do NOT automatically rerun everything.

Identify the smallest set of reruns needed to make the final main-text comparison scientifically clean.

C. Any GCN-side baseline that should remain in the final manuscript

The final paper may retain the original GCN-GSL result as part of the evolution of the work while dropping cGSL from the main story.

Determine whether a canonical rerun is needed for GCN-GSL.

Again, do not run it now.

D. Any other experiment that is genuinely necessary for the final manuscript

Do not add experiments merely because they are possible.

Prioritize experiments that directly answer reviewer concerns or resolve a methodological ambiguity.

For every proposed experiment, provide:

* scientific question;
* exact methods;
* dataset;
* temporal resolution;
* prediction horizons;
* seeds;
* graph construction;
* edge budget/threshold;
* loss;
* training settings;
* expected runtime;
* output directory;
* exact JSON/CSV artifacts to produce;
* exact command to run.

Create separate scripts when appropriate and corresponding bash runners.

All new experiment scripts must use the same canonical training/evaluation pipeline wherever scientifically appropriate.

Do not create another ad-hoc training loop.

Do not duplicate model implementations unnecessarily.

# MANUSCRIPT STORY: THINK BEFORE EDITING

This is important.

Do NOT rewrite the manuscript in this stage.

Instead, after inspecting the repository, produce a strategic recommendation about how the old results should appear in the final paper.

We do NOT want to say in the manuscript:

"We kept these results because they were in the previous version."

The final reader sees only one paper.

The scientific story should therefore be self-contained.

Consider the following candidate narrative:

ACT I:
Original/single-graph GSL establishes that replacing a dense physical graph with a learned sparse graph can substantially improve T-GCN forecasting.

ACT II:
A stronger no-graph and sparsity analysis reveals that part of the apparent GSL advantage is related to dense-graph oversmoothing.

ACT III:
This motivates examining whether the learned dependencies are temporally heterogeneous rather than representable by one static graph.

ACT IV:
Multi-lag GSL constructs lag-specific graphs, and T-GCN-MultiGSL-Mix learns how to combine them.

ACT V:
Multi-seed, parameter-control, lag-ablation, horizon, temporal-resolution, and dataset-dependence experiments validate the behavior.

The old GSL result should therefore not be presented as "historical baggage".
It should be presented as the empirical starting point that motivates the deeper analysis.

At the same time:

* do not overload the paper with every historical GSL/cGSL experiment;
* do not preserve cGSL merely for historical continuity if it is not needed scientifically;
* do not preserve duplicated convergence plots;
* do not retain old tables whose protocol is incompatible with the final protocol unless clearly labeled;
* if an old baseline is scientifically important but its protocol differs materially, prepare a canonical rerun instead.

Please explicitly decide whether the final paper should retain:

1. GCN-GSL;
2. GCN-cGSL;
3. T-GCN-GSL;
4. T-GCN-cGSL;
5. T-GCN-NoSpatial;
6. Physical;
7. T-GCN-MultiGSL;
8. T-GCN-MultiGSL-Mix.

For each, classify it as:

* MAIN TEXT
* APPENDIX
* REMOVE
* RERUN UNDER CANONICAL PROTOCOL

Do not make this decision based on "what was in the old paper".
Make it based on the final scientific narrative and reviewer concerns.

# IMPORTANT SCIENTIFIC CONSTRAINT

The final manuscript must not imply that the 15-minute experiment supports a temporal-resolution explanation for SZ-Taxi's weak result.

The verified evidence shows:

Los-loop:
5-min -> strong improvement
15-min -> strong improvement, even larger

SZ-Taxi:
15-min -> marginal improvement

Therefore the final interpretation should be dataset-dependent, with temporal resolution explicitly not established as the causal explanation.

Similarly, do not make causal claims from DAGMA.

Use terms such as:

* dependency;
* association;
* learned structure;
* lag-specific dependency;
* predictive structure;

rather than unsupported causal language.

# IMPORTANT PROTOCOL CONSISTENCY

The canonical pipeline should be treated as the reference for new experiments:

* same sequence generation;
* same train/test handling;
* feat_max computed from training data only;
* same loss;
* same optimizer;
* same batch size;
* same number of epochs;
* same evaluation;
* same seed handling;

unless an experiment explicitly tests one of these factors.

Do not silently change these settings.

# OUTPUT REQUIRED

At the end of this stage produce a report:

doc/STAGE33_CODE_NAMING_AND_EXPERIMENT_PREPARATION_REPORT.md

The report must contain:

1. Summary of code naming changes.
2. Legacy-to-canonical naming map.
3. Files changed.
4. Canary test command and PASS/FAIL result.
5. Manuscript source-code reference changes.
6. Manuscript compilation result.
7. Recommended final status of each old/new method:
   MAIN TEXT / APPENDIX / REMOVE / RERUN.
8. Recommended experiment suite to run next.
9. Exact commands for each prepared experiment.
10. Expected output directories and artifacts.
11. Any remaining scientific ambiguity.
12. A proposed high-level Results narrative, WITHOUT rewriting the manuscript itself.

DO NOT:

* run DAGMA;
* run full training suites;
* spend hours training models;
* modify historical Stage 27 results;
* claim Stage 27 forecasting results are valid;
* change numerical results;
* rewrite the manuscript extensively;
* add "Adaptive" to the method names.

The goal of this stage is to leave the repository in a clean, internally consistent, experimentally ready state so that the next stage can consist primarily of running the selected final experiments.
