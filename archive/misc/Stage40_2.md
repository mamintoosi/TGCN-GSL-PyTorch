You are continuing work on the current `TGCN-GSL-PyTorch` repository after Stage 40.1.

Execute **Stage 40.2: Counterpart Equivalence and Graph Provenance Audit**.

Do NOT run the full Stage 40 experiment matrix and do NOT start long GPU training. Only code inspection, graph inspection, small smoke tests, and artifact validation are allowed.

The purpose of this stage is to resolve one important scientific requirement before Stage 40:

> Corresponding GCN and T-GCN variants must use the same underlying DAGMA graph construction. If GCN-MultiGSL is presented as the counterpart of T-GCN-MultiGSL, the difference must be the backbone/graph consumption mechanism, not a different DAGMA graph.

## 1. Define the required counterpart relationships

Audit and enforce these conceptual pairs:

* `GCN` ↔ `T-GCN`
* `GCN-NoSpatial` ↔ `T-GCN-NoSpatial`
* `GCN-GSL` ↔ `T-GCN-GSL`
* `GCN-cGSL` ↔ `T-GCN-cGSL`
* `GCN-MultiGSL` ↔ `T-GCN-MultiGSL`

For every pair, document:

* graph source
* graph construction
* threshold
* signed/absolute support rule
* self-loop handling
* directed/symmetric status
* whether the graph is static or lag-specific
* the exact architectural difference

The graph-generation protocol must be identical wherever the counterpart relationship implies it.

## 2. GCN-MultiGSL must use the same DAGMA multi-lag artifacts

The required construction is:

DAGMA produces:

`A_1, A_2, ..., A_L`

Then:

`GCN-MultiGSL:`

`A_union = OR(A_1, A_2, ..., A_L)`

followed by standard GCN using `A_union`.

`T-GCN-MultiGSL:`

uses the same `A_1, A_2, ..., A_L` separately according to the existing multi-lag timestep assignment.

Do NOT generate a separate DAGMA graph for GCN-MultiGSL.

Do NOT use contemporaneous GSL output for GCN-MultiGSL.

Do NOT use a different threshold.

Do NOT invent a new graph-learning procedure.

The only intended difference is how the same multi-lag DAGMA information is consumed by the two backbones.

## 3. Verify exact graph equality

For each dataset:

* Los-loop
* SZ-Taxi

and each relevant PH:

* PH=1,2,3,4

inspect the actual graph files used by:

* `GCN-MultiGSL`
* `T-GCN-MultiGSL`

Verify that both ultimately originate from the same DAGMA multi-lag artifacts.

Because Stage 40.1 reported that the multi-lag DAGMA blocks are PH-independent, explicitly verify that this is actually true in the files.

Compute/check:

* number of edges per lag
* union edge count
* matrix shape
* diagonal handling
* exact binary equality between independently loaded versions of the same graph

Do not merely compare filenames.

## 4. Verify GCN-MultiGSL implementation

Inspect the actual implementation.

It should conceptually be:

```python
A_union = np.logical_or.reduce([A_lag1, A_lag2, A_lag3])
```

or an exactly equivalent operation.

Then the standard GCN must receive that single `N x N` graph.

Verify that:

* no recurrent mechanism is introduced
* no per-timestep graph assignment is introduced
* no additional trainable parameters are introduced
* the GCN architecture itself remains unchanged

A short forward-pass smoke test is sufficient.

## 5. Verify T-GCN-MultiGSL implementation

Confirm that it consumes the same lag-specific matrices before they are combined.

Verify the exact mapping:

`graph_idx = (T-1-t) % n_graphs`

and determine precisely which input timestep receives which lag graph.

Document this mapping in the report.

Do not change it unless it is demonstrably inconsistent with the intended mathematical definition.

## 6. Check the Weighted and Mix variants

For:

* `T-GCN-MultiGSL-Weighted`
* `T-GCN-MultiGSL-Mix`

verify that they use exactly the same underlying lag-specific DAGMA matrices as `T-GCN-MultiGSL`.

The difference must be:

* Fixed: direct lag-specific use
* Weighted: learned global combination
* Mix: learned per-node/per-timestep combination

They must not silently use a different graph set.

## 7. Verify single-GSL counterpart equivalence

For each dataset and PH:

`GCN-GSL` and `T-GCN-GSL`

must use the same contemporaneous DAGMA graph.

Likewise:

`GCN-cGSL` and `T-GCN-cGSL`

must use the same GSL graph followed by exactly the same symmetrization rule.

Check whether the current loader actually guarantees this.

Pay special attention to:

* DAGMA `w_threshold`
* consumer threshold
* absolute-value thresholding
* diagonal removal
* symmetrization

## 8. Important threshold issue

Do not change the existing scientific protocol merely for cosmetic consistency.

However, explicitly report the fact that:

* multi-lag DAGMA artifacts use raw weights followed by `|W| > 0.1`
* contemporaneous Los DAGMA artifacts use the DAGMA internal threshold of 0.3

If these thresholds are scientifically intentional, preserve them.

If any counterpart currently uses a different threshold or graph transformation, flag it as a correctness issue.

## 9. Check the Stage 40 experiment manifest

The manifest must clearly encode:

* graph source
* graph construction
* threshold
* counterpart relationship

Avoid ambiguous labels such as simply `multi_gsl`.

Prefer explicit metadata such as:

* `graph_source: dagma_multilag`
* `graph_combine: union`
* `graph_use: static`
* `graph_use: lag_specific`

where appropriate.

Do not unnecessarily redesign the entire manifest.

## 10. Check whether any existing result can safely be reused

For every existing result that Stage 40 plans to reuse, verify that:

* the graph source is the same
* the graph construction is the same
* the model architecture is the same
* the training protocol is the same
* the dataset split is the same
* the seed corresponds to the training seed
* the DAGMA graph seed/provenance is the same where relevant

If a result is not demonstrably equivalent, mark it `NOT SAFE TO REUSE`.

Do not discard results merely because their directory/name is old.

## 11. Check the 12-variant matrix

The intended matrix is currently:

### T-GCN family

1. T-GCN
2. T-GCN-NoSpatial
3. T-GCN-GSL
4. T-GCN-cGSL
5. T-GCN-MultiGSL
6. T-GCN-MultiGSL-Weighted
7. T-GCN-MultiGSL-Mix

### GCN family

8. GCN
9. GCN-NoSpatial
10. GCN-GSL
11. GCN-cGSL
12. GCN-MultiGSL

Do not add arbitrary GCN-Weighted or GCN-Mix variants. Those are not required counterparts because their mechanisms depend on how temporal recurrence consumes lag-specific graphs.

The important counterpart is specifically:

`GCN-MultiGSL` = union of the same DAGMA multi-lag graphs used separately by `T-GCN-MultiGSL`.

## 12. Do not fit missing SZ contemporaneous DAGMA yet

Stage 40.1 identified missing SZ-Taxi contemporaneous DAGMA graphs.

Do NOT spend the 6–10 hours fitting them in Stage 40.2.

First complete this audit and ensure that the graph pipeline is correct.

## 13. Required deliverable

Create:

`reports/stage40.2_counterpart_equivalence_audit.md`

The report must contain:

1. Executive verdict
2. Counterpart-equivalence table
3. Exact graph provenance for every counterpart
4. GCN-MultiGSL implementation verification
5. T-GCN-MultiGSL implementation verification
6. Weighted/Mix graph-source verification
7. Threshold comparison
8. Existing-result reuse audit
9. Required code fixes, if any
10. Final Stage 40 readiness status

For any problem found, classify it as:

* FAIL = scientifically or technically incorrect
* WARNING = defensible but requires explicit documentation
* PASS = verified

If code changes are necessary, make only the minimum structural fixes required for correctness.

Do not run long experiments.

At the end print:

`STAGE 40.2 VERDICT: READY`

or

`STAGE 40.2 VERDICT: NOT READY`

Do not claim READY unless the counterpart graph provenance has actually been verified from the code and artifacts.
