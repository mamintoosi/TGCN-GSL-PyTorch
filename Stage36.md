# Stage 36 — DAGMA Signed-Weight Audit and Final Experiment Readiness

We are preparing the final experimental runs A/B/C for the revised version of the paper.

The repository is:

`/data/git/mamintoosi/TGCN-GSL-PyTorch`

Use the current repository state, including the latest commits and the changes documented in Stage 35.

Do NOT ask me questions. Make the best technically and scientifically justified decisions yourself, and report them.

## Important scientific clarification

Stage 35 established that DAGMA itself thresholds coefficients by absolute magnitude:

```text
W_est[abs(W_est) < w_threshold] = 0
```

The previous project code subsequently constructed a binary adjacency using:

```text
A = (W_est > 0)
```

This means that negative DAGMA coefficients were discarded by the project projection rule.

We now consider this an issue that must be explicitly audited rather than assuming that positive-only edges are scientifically correct.

For the revised canonical experiments, the preferred graph-support rule is:

```text
W_est = DAGMA(...)
W_est[abs(W_est) < threshold] = 0
A = (abs(W_est) > 0)
A[i,i] = 0
```

Equivalently, an edge is retained whenever:

```text
abs(W_ij) >= threshold
```

regardless of whether `W_ij` is positive or negative.

The direction of the DAGMA edge is preserved, but the current GCN/T-GCN models still use a binary adjacency, so the sign itself is not passed as an edge weight.

Do NOT introduce signed graph convolution or another new model in this stage. That would be a separate future experiment.

The threshold `0.3` should be regarded as the explicit form of the original DAGMA library default used implicitly by the submitted implementation, not as an arbitrary newly selected threshold.

## Main goals

### Goal 1 — Audit all DAGMA-to-graph conversions

Trace every relevant path from DAGMA output to the final graph used by experiments.

At minimum inspect:

* `gsl_stage26/stage33_gsl_canonical.py`
* `gsl_stage26/stage32_sparse_control.py`
* `gsl_stage26/stage33_sz_multiseed.py`
* Stage 26 graph-generation code
* `utils/data/spatiotemporal_csv_data.py`
* any other code that creates or loads DAGMA-derived adjacency matrices
* all relevant `.npy` graph/W-estimate artifacts used by A/B/C

For each path explicitly determine:

1. What does the `.npy` file contain: W, binary adjacency, weighted adjacency, or something else?
2. What threshold is used?
3. Is thresholding based on `W > threshold`, `abs(W) > threshold`, or another rule?
4. Are negative DAGMA coefficients retained or discarded?
5. Is the graph directed?
6. Is the diagonal removed?
7. Is `A+I` subsequently used only for message passing?
8. Does the resulting graph differ if negative coefficients are retained?

Do not infer these answers from filenames or comments. Inspect the actual code and data-generation path.

---

### Goal 2 — Determine exactly which experiments are affected

Audit experiments A, B, and C separately.

For each one classify it as:

* **Unaffected**
* **Affected but existing graph can be regenerated cheaply**
* **Affected and must be rerun**
* **No longer scientifically valid under the revised rule**

In particular:

#### Experiment B

B directly fits DAGMA and therefore must use the final canonical rule:

```text
w_threshold = 0.3
A = 1(abs(W_est) > 0)
```

unless the code/data audit finds a compelling repository-specific reason otherwise.

Make the smallest necessary code modification.

Record both:

* number of coefficients surviving `abs(W)>=0.3`
* number of positive surviving coefficients
* number of negative surviving coefficients
* number of final binary edges

This is important provenance.

#### Experiment A

Determine whether the Stage 26 lag-graph artifacts used by A were generated with positive-only or absolute-magnitude support.

If they were generated using positive-only support and the revised rule changes the graph, regenerate the relevant graph artifacts using the revised rule and determine whether A must be rerun.

Do not preserve old numerical results merely for continuity.

The method names and conceptual roles should remain stable so that the revised paper clearly represents an evolution of the submitted work rather than an unrelated method.

#### Experiment C

Trace the origin of:

`results/stage26_validation/sz_ph*_seed42_L3_lag_*.npy`

and determine exactly how they were generated.

If their graph construction is inconsistent with the revised signed-weight policy and materially changes the graph, regenerate them and mark C for rerun.

If they are already based on absolute magnitude and are unaffected, document that fact and do not unnecessarily rerun C.

---

## Goal 3 — Historical W_est isolation

The committed files:

```text
data/W_est_{losloop,shenzhen}_pre_len{1,2,3,4}.npy
```

belong to the original submitted implementation.

We do NOT require their numerical results to remain valid.

The priority is:

1. prevent accidental reuse by active experiments;
2. preserve provenance only if useful;
3. keep the conceptual/method names stable where possible.

Inspect all active code paths.

If these historical files are not required by the final active pipeline, move them out of the active `data/` path into an explicitly historical/archive location, preferably something like:

```text
archive/historical_submission/
```

Do not modify the historical files themselves.

Update only those code paths that genuinely need to distinguish historical artifacts from current experiment artifacts.

Do NOT make unrelated repository changes.

---

## Goal 4 — Final canonical policy

Unless the audit reveals a concrete contradiction in the repository, establish the following as the canonical DAGMA graph policy for the revised experiments:

```text
DAGMA fit
    ↓
absolute-magnitude thresholding at 0.3
    ↓
retain both positive and negative coefficients
    ↓
binary directed adjacency based on nonzero support
    ↓
remove diagonal
    ↓
A + I only where required by the existing message-passing implementation
```

Do not change:

* model architecture
* loss
* optimizer
* learning rate
* weight decay
* batch size
* hidden dimension
* number of epochs
* data split
* forecasting seed protocol

unless a concrete bug is discovered.

The purpose of this stage is to resolve graph-construction semantics, not to redesign the forecasting model.

---

## Goal 5 — Reproducibility and provenance

Update the relevant scripts so that saved graph metadata explicitly records:

* threshold
* threshold rule
* adjacency rule
* positive coefficients retained
* negative coefficients retained
* number of positive surviving coefficients
* number of negative surviving coefficients
* final number of edges
* DAGMA version
* numpy/scipy versions
* input construction
* dataset
* prediction horizon
* graph generation seed, if applicable
* whether the graph was freshly fitted or reused

The metadata must make it impossible to confuse a historical graph with a revised canonical graph.

---

## Goal 6 — Validation without expensive experiments

Perform only short tests needed to verify the implementation.

Acceptable tests include:

* syntax checks
* import checks
* CLI `--help`
* tiny synthetic DAGMA tests
* tiny graph-construction tests
* loading and inspecting existing graph artifacts
* comparing old versus revised graph support
* short smoke training if needed

Do NOT run the full A/B/C experiments.

Do NOT perform multi-hour DAGMA fits.

Do NOT produce scientific final results.

---

## Goal 7 — Preserve method identity

Do not rename the established methods merely because the implementation has been corrected.

Prefer keeping names such as:

* `T-GCN`
* `T-GCN-GSL`
* `T-GCN-MultiGSL`
* `T-GCN-MultiGSL-Mix`

unless a name is demonstrably technically incorrect.

The revised implementation should be presented as the corrected/evolved version of the submitted methodology, not as an entirely new method.

Numerical results from the original submission do NOT need to be preserved.

---

# Required final report

Produce a concise but technically complete report titled:

`Stage 36 — DAGMA Signed-Weight Audit and Final Experiment Readiness`

Include:

1. Exact DAGMA → adjacency paths audited.
2. Current sign handling for each relevant experiment.
3. Whether negative coefficients actually occur above the threshold in each relevant dataset/artifact.
4. Exact graph changes caused by switching from positive-only to absolute-magnitude support, where measurable.
5. Which of A/B/C must be rerun.
6. Which existing graph artifacts must be regenerated.
7. Which historical `W_est` files were moved/isolated.
8. Exact files modified.
9. Exact commands for the future full experiments.
10. Confirmation that no full experiment was executed.
11. GO/NO-GO status for A, B, and C.

Most importantly:

**Do not execute the expensive experiments in this stage.**

After this audit, stop and provide the report. The next stage will perform the actual A/B/C runs using the finalized graph-construction policy.

