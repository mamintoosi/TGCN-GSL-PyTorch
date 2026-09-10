# Stage 38 — Scientific Results & Reviewer-Response Mapping Audit

## Context

Stages 34–36 finalized the experimental protocol and DAGMA graph policy. Stage 37 executed all three final experiments A/B/C successfully and produced the final numerical results.

Do NOT run any expensive experiment in this stage.

Do NOT modify the manuscript yet.

The goal of this stage is to perform a rigorous scientific audit of the Stage 37 results and determine exactly how they should be used in the revised paper and in the response to the reviewers.

The repository and its history are available to you. Inspect the actual code, result JSON files, relevant historical artifacts, manuscript, and reviewer comments rather than relying only on this prompt.

Important methodological decisions already finalized:

* DAGMA support is defined by absolute magnitude after the DAGMA threshold:
  `A = 1(|W| > 0)`.
* DAGMA `w_threshold = 0.3` is explicitly specified for the canonical single-graph GSL experiment.
* Negative coefficients are retained by the support rule if they survive the threshold.
* No negative coefficient above the threshold was observed in the current or historical traffic artifacts examined so far.
* The graph is binary; coefficient magnitude is NOT used as a message-passing edge weight.
* Diagonal entries are removed from the learned adjacency; self-loops are introduced only through the existing `A + I` Laplacian/message-passing mechanism.
* Method names should remain stable and recognizable as an evolution of the submitted paper. Do not rename the established methods merely because the implementation has been refined.

Historical submission artifacts have been moved to:

`archive/historical_submission/`

The historical W_est files are archival references only and must not silently enter the new experiments.

---

## 1. Read the reviewer comments and manuscript

Locate and inspect:

* `paper/Reviewers-comments.txt`
* `paper/sn-article.tex`

Identify every reviewer comment that is materially addressed by Experiments A, B, or C.

Create a mapping:

| Reviewer issue | Exact scientific concern | Experiment A/B/C | Evidence/result | What the revised paper should say |
| -------------- | ------------------------ | ---------------- | --------------- | --------------------------------- |

Do not invent reviewer concerns. Quote or accurately summarize the actual reviewer comments.

---

## 2. Audit Experiment A scientifically

Use the actual Stage 37 JSON and relevant Stage 26 results.

Confirm:

* 30-edge matched budget;
* CorrTop30;
* RandTop30;
* T-GCN-NoSpatial;
* T-GCN-MultiGSL;
* T-GCN-MultiGSL-Mix;
* exact seed coverage;
* RMSE and MAE;
* mean/std definitions;
* whether the comparisons are genuinely protocol-matched.

Determine precisely what A establishes and what it does NOT establish.

In particular, distinguish:

1. sparsity effect;
2. graph topology/edge placement;
3. DAGMA-derived structure;
4. MultiGSL mechanism;
5. Mix/gating effect.

Do not overclaim that A proves more than its design supports.

Recommend the exact scientific conclusion that can safely be stated in the paper.

---

## 3. Audit Experiment B scientifically

Inspect:

* `stage33_gsl_canonical.py`
* Stage 37 JSON
* fresh B W_est/A_binary artifacts
* historical W_est artifacts
* historical loader in `utils/data/spatiotemporal_csv_data.py`
* relevant original manuscript text

Verify the distinction between:

### Historical submitted protocol

For horizon K:

`X = data[i::K]`, for `i = 0,...,K-1`

followed by K DAGMA fits and union of the resulting supports.

### Canonical revised B protocol

One DAGMA fit per PH using:

`train[0::PH]`

with the explicit threshold and the finalized absolute-magnitude support rule.

Determine exactly which statements in the original paper are graph-for-graph reproduced and which are not.

Pay particular attention to:

* PH1;
* PH2–PH4;
* historical edge counts 28/32/33/39;
* fresh B edge count 28 for every PH;
* support Jaccard;
* W magnitude differences;
* forecasting RMSE/MAE differences.

Clearly distinguish:

* exact reproduction of graph support;
* approximate reproduction of weights;
* reproduction of the original headline PH1 result;
* reproduction of the entire historical multi-PH protocol.

Do NOT describe these as identical if they are not.

---

## 4. Audit Experiment C scientifically

Use the actual Stage 37 results.

Determine:

* whether MultiGSL itself helps;
* whether MultiGSL-Mix helps;
* effect size in percentage terms;
* consistency across seeds;
* PH dependence;
* whether PH4 should be described as stable or unstable;
* whether the correct conclusion is "dataset-dependent", "marginal", or something more conservative.

Calculate the relevant percentage improvements from the actual mean RMSE values.

Do not manufacture statistical significance tests that were not performed.

---

## 5. Reconcile all DAGMA products

Produce one authoritative table distinguishing:

1. Stage 26 multi-lag DAGMA blocks;
2. Stage 37 Experiment B fresh single-graph fits;
3. archived audit fits;
4. historical submission W_est stacks.

For each, record:

* optimization problem / variable dimensionality;
* input construction;
* lambda1;
* threshold;
* sign rule;
* graph construction;
* number of graphs;
* graph edge count;
* intended method;
* whether it is active or archival.

The purpose is to prevent future accidental mixing of these artifacts.

---

## 6. Audit the signed-weight decision

Review the Stage 36 sign audit and Stage 37 results.

Answer explicitly:

1. Is `|W|` thresholding mathematically the correct interpretation of DAGMA's `w_threshold`?
2. Is retaining both positive and negative coefficients a defensible graph-support policy?
3. Since no negative coefficient survives the threshold in the observed traffic data, did this decision change any Stage 37 numerical result?
4. Should the manuscript explicitly discuss this methodological refinement?
5. If yes, formulate a conservative explanation suitable for a response to reviewers.

Important:

Do NOT claim that negative coefficients were empirically important when they were not.

The correct distinction is between a methodological correction/generalization and an observed numerical effect.

---

## 7. Audit the Stage 37 metadata corrections

Verify that the two Stage 37 corrections are scientifically correct:

### Correction 1

C's previous provenance incorrectly counted the `current` block.

Confirm that MultiGSL actually uses only:

* lag_1
* lag_2
* lag_3

and determine the resulting per-PH edge counts.

### Correction 2

B's physical graph previously reported:

`1307`

because the code summed edge weights.

Confirm that the actual number of positive adjacency entries is:

`2833`

and that this is the correct definition of physical graph edge count.

Check whether either error affected any RMSE/MAE calculation. If not, state explicitly that these were provenance/reporting defects only.

---

## 8. Build a final "old → new" results map

Create a table containing every important result that is likely to appear in the revised manuscript.

Columns:

| Manuscript location / claim | Original submitted value | New canonical value | Source | Should replace? | Reason |

Include at least:

* GSL PH1;
* GSL PH2;
* GSL PH3;
* GSL PH4;
* Physical baseline;
* NoSpatial;
* MultiGSL;
* MultiGSL-Mix;
* CorrTop30;
* RandTop30;
* SZ-Taxi results;
* graph edge counts.

Do not replace a historical number merely because a newer number exists. Decide based on methodological relevance.

---

## 9. Decide what remains historical

Explicitly classify existing results into:

### A. Main-text canonical results

### B. Appendix results

### C. Historical submission results

### D. Obsolete / should no longer be cited

Be especially careful with:

* old single-seed numbers;
* historical union-of-offset GSL graphs;
* cGSL results;
* old W_est files;
* Stage 26 multi-lag results.

Do not delete anything.

---

## 10. Evaluate whether another experiment is scientifically necessary

This is an analysis-only decision.

Based on the reviewer comments and Stage 37 results, determine whether any additional experiment is genuinely necessary before manuscript revision.

Possible outcomes:

* no additional experiment;
* optional robustness experiment;
* necessary experiment.

If an experiment is recommended, explain exactly which reviewer concern it addresses and why A/B/C do not already address it.

Do NOT run it.

---

## 11. Prepare a manuscript-integration plan

Without editing the manuscript, identify:

* exact sections that need numerical replacement;
* tables that need modification;
* figures that may need updating;
* paragraphs whose scientific interpretation needs rewriting;
* claims that should be strengthened;
* claims that should be weakened;
* new sentences required to explain the revised protocol.

For each item, provide the relevant section/table/paragraph and the proposed scientific purpose of the change.

---

## 12. Prepare reviewer-response guidance

For every reviewer issue addressed by A/B/C, provide:

1. reviewer concern;
2. action taken;
3. experiment performed;
4. numerical evidence;
5. scientific conclusion;
6. recommended response strategy.

Do not write the final polished rebuttal yet.

This stage is for scientific planning and verification.

---

## 13. Check reproducibility/provenance

Confirm:

* all Stage 37 result JSON files;
* B's eight graph artifacts;
* the relevant Stage 26 graph blocks;
* historical W_est files;
* scripts and commits associated with the experiments.

Report any missing artifact that would prevent a complete future reproduction.

Do not alter or delete artifacts.

---

## 14. Final deliverable

Produce a report:

`Stage38.md`

with the following structure:

1. Executive Summary
2. Reviewer-to-Experiment Mapping
3. Experiment A Scientific Interpretation
4. Experiment B Scientific Interpretation
5. Experiment C Scientific Interpretation
6. DAGMA Product Reconciliation
7. Signed-Weight Policy Audit
8. Stage 37 Metadata Corrections
9. Old-to-New Results Map
10. Main Text / Appendix / Historical Classification
11. Need for Additional Experiments
12. Manuscript Integration Plan
13. Reviewer Response Plan
14. Reproducibility and Provenance Status
15. Final GO/NO-GO

Do not modify `paper/sn-article.tex`.

Do not run A, B, or C again.

Do not perform any multi-hour computation.

If short verification scripts are useful, they may be executed, but the main task is inspection, comparison, calculation, and scientific interpretation.

At the end, give a concise recommendation for Stage 39.

Stage 39 should only begin after we inspect this Stage 38 report.

