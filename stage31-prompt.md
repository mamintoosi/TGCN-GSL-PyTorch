# Stage 31 — Manuscript Reconstruction & Experimental Evidence Audit

## Objective

Perform a **forensic, evidence-based audit of the manuscript evolution** and determine how the original submitted paper and the current revised manuscript can be combined into one coherent, non-redundant scientific story.

This is an **audit and planning stage only**.

### IMPORTANT

* Do NOT modify any `.tex`, source-code, result, figure, or configuration file.
* Do NOT run expensive experiments.
* Do NOT run DAGMA.
* Do NOT retrain models.
* Do NOT commit anything.
* Do NOT silently correct manuscript claims.
* Base every conclusion on actual repository files, experiment outputs, git history, and manuscript text.
* If something cannot be established from the repository, explicitly mark it as **UNKNOWN**.
* Do not invent results.

---

# 1. Identify the Two Manuscript States

We need to compare:

### A. Original submitted manuscript

Use BOTH:

1. Git commit:

`dac89785f47968d4aaf1d728b6ec6556d9874e1e`

2. File (in current state of repo):

`paper/sn-article_original.tex`

Determine whether these represent the same manuscript state. If they differ, explain the differences.

Extract:

* title
* abstract
* research problem
* stated contributions
* proposed method
* all method names
* experimental datasets
* experimental configurations
* all major tables
* all major figures
* major quantitative claims
* conclusions
* limitations
* claims concerning GSL, sparsity, graph structure, temporal structure, and dataset dependence.


Note that in commit `dac89785f47968d4aaf1d728b6ec6556d9874e1e` only the implementaion code was in the repo, not the paper source or PDF. 

---

### B. Current revised manuscript

Use:

`paper/sn-article.tex`

Also inspect all files included/imported by it, especially:

* `paper/sections/*.tex`
* tables
* figure captions
* references to experimental results

Extract the same information.

---

# 2. Review the Reviewer Comments

Locate and inspect the complete reviewer comments in the repository.

Identify the **actual scientific objections**, especially those concerning:

* graph structure learning
* sparsity
* adequacy of learned graphs
* temporal dependencies
* methodology
* comparison with baselines
* justification of the proposed architecture
* experimental validation
* generalization across datasets.

For each major reviewer criticism, create:

| Reviewer criticism | What the original paper did | What the revised paper currently does | Evidence added since submission | Still unresolved? |

Do NOT paraphrase excessively. Preserve the actual scientific meaning of the reviewers' concerns.

---

# 3. Reconstruct the Evolution of the Method

Trace the evolution from the original method to the current method.

In particular determine:

1. What was the original GSL/cGSL approach?
2. What were the original model variants?
3. What did the original results demonstrate?
4. What scientific observation motivated the move toward multi-lag graphs?
5. What is the exact current proposed architecture?
6. Which reviewer criticism does the new architecture answer?
7. Which parts of the original methodology remain scientifically useful?

Pay special attention to the distinction between:

* original/single-graph GSL
* fixed multi-lag graph assignment
* learned graph mixing
* the current proposed `T-GCN-MultiGSL-Mix`

Do not use the term **Adaptive** in proposed method names unless it already occurs in an unavoidable quotation or source filename.

The current authoritative names are:

* `T-GCN-NoSpatial`
* `Physical`
* `T-GCN-MultiGSL`
* `T-GCN-MultiGSL-Mix`

Treat these names as the current manuscript terminology unless the repository evidence shows otherwise.

---

# 4. Audit the Current Results Section

Inspect the current:

`paper/sections/results.tex`

and all referenced experimental artifacts.

For every current subsection, classify it as:

* KEEP
* KEEP BUT REFRAME
* MERGE
* MOVE TO SUPPLEMENTARY MATERIAL
* REMOVE
* REQUIRES RE-RUN
* REQUIRES VERIFICATION

The current Results subsections include, among others:

* Dense Physical Graphs and Oversmoothing
* Multi-Lag Graph Structure
* Multi-Seed Validation
* Parameter-Matched Control
* Lag Ablation
* Prediction Horizons and Dataset Dependence
* Relation to Original GSL/cGSL Results
* Predicted vs. actual visualization

Explain the scientific role of each subsection in the **new story**.

---

# 5. Recover the Original Experimental Evidence

Search git history and repository artifacts for all original experiments and results.

Especially identify the experiments that produced the strong original results.

For each experiment record:

| Experiment | Dataset | Method | PH | Seeds | RMSE | Purpose | Still scientifically useful? |

Determine whether the original result can be retained **without rerunning it**.

If rerunning is necessary, explain exactly why.

Do NOT assume that an old result must be discarded merely because a new architecture exists.

---

# 6. Compare Original and Current Methodological Evidence

Construct a high-level comparison:

| Aspect                | Original submission | Current revision | Scientific role in revised paper |
| --------------------- | ------------------- | ---------------- | -------------------------------- |
| Base GSL              | ...                 | ...              | ...                              |
| cGSL                  | ...                 | ...              | ...                              |
| Sparse learned graph  | ...                 | ...              | ...                              |
| Multi-lag formulation | ...                 | ...              | ...                              |
| Graph mixing          | ...                 | ...              | ...                              |
| Dataset comparison    | ...                 | ...              | ...                              |
| Multi-seed validation | ...                 | ...              | ...                              |

The purpose is to determine whether the revised paper should be presented as:

> a completely new method replacing the old method

OR

> an evolution of the original GSL approach motivated by the limitations revealed by the reviewer concerns and subsequent analysis.

I strongly suspect the second framing may be scientifically stronger, but **do not assume this conclusion**. Determine it from the evidence.

---

# 7. Incorporate Stage 29 Correctly

Inspect:

`results/stage29_los15min/`

and:

`gsl_stage26/stage29_los15min.py`

Also inspect Stage 26 and Stage 27 artifacts.

Establish the verified status of:

### Los-loop 5-minute

Current canonical result:

* T-GCN-NoSpatial ≈ 5.1432
* T-GCN-MultiGSL-Mix ≈ 4.4578
* improvement ≈ 13.33%

### Los-loop 15-minute

Stage 29:

* NoGraph mean = 8.6000 ± 0.2492
* T-GCN-MultiGSL mean = 7.2462 ± 0.2975
* T-GCN-MultiGSL-Mix mean = 6.2402 ± 0.1873
* improvement of T-GCN-MultiGSL-Mix over NoGraph = 27.44%

### SZ-Taxi

Current canonical result:

* NoGraph ≈ 4.1156
* T-GCN-MultiGSL-Mix ≈ 4.1076
* improvement ≈ 0.19%

Verify these numbers against the actual JSON/result files.

Also explicitly document:

* Stage 27 is invalid for forecasting because it used the wrong model class.
* Stage 29 uses the canonical Stage 26 pipeline.
* Stage 29 should therefore be considered the authoritative Los-loop-15min result, if the repository evidence confirms the previous audit.

---

# 8. Determine the Best Scientific Story

Based ONLY on the evidence, propose 2–3 possible narratives for the revised paper.

For each narrative provide:

1. Central research question
2. Role of original method
3. Role of new method
4. Main experiments
5. Main result
6. How reviewer criticisms are answered
7. Potential reviewer objection to this narrative

Then rank the narratives.

I particularly want you to evaluate whether the following narrative is defensible:

> The original GSL experiments demonstrated that replacing dense physical connectivity with sparse learned graph structures can improve T-GCN forecasting. This motivated a deeper investigation of temporal dependency structure. The revised method extends GSL from a single learned graph to multiple lag-specific graphs and learns how to mix them. The resulting method substantially improves Los-loop forecasting across both 5-minute and 15-minute temporal resolutions, while its benefit is marginal on SZ-Taxi, revealing meaningful dataset dependence rather than universal gains.

Do not accept this narrative automatically. Test it against the actual evidence.

---

# 9. Determine Which Original Results Should Stay

This is one of the most important tasks.

Create a table:

| Original result | Keep? | Where in revised paper? | Why? | Need rerun? |
| --------------- | ----- | ----------------------- | ---- | ----------- |

Pay special attention to the possibility that the original results should remain as an important **baseline/evolutionary stage**, rather than being relegated to an appendix.

The goal is:

> revised paper = improved version of the submitted paper

NOT:

> completely different paper that happens to contain a new architecture.

Determine whether the original strong results can naturally introduce and motivate the new architecture.

---

# 10. Simplify the Experimental Story

We want the revised paper to remain compact.

Determine which current experiments are redundant.

In particular evaluate whether the following can be removed or reduced:

* training curves
* excessive PH plots
* redundant visualizations
* duplicate graph visualizations
* repeated seed-level plots
* overly detailed threshold sensitivity plots

The goal is to preserve **high-value scientific evidence**, not maximize the number of experiments.

Propose a compact Results structure with approximately 4–6 subsections.

For each subsection specify:

* purpose
* key table/figure
* key quantitative result
* reviewer criticism addressed

---

# 11. Proposed Revised Results Architecture

Propose an exact structure for the Results section.

For example, consider whether a structure of this general form is appropriate:

1. Baseline: Dense Physical Graph vs. Sparse Learned Graph
2. From Single-Graph GSL to Multi-Lag GSL
3. Effectiveness of the Proposed T-GCN-MultiGSL-Mix
4. Ablation / Control Experiments
5. Robustness Across Prediction Horizons and Temporal Resolution
6. Dataset Dependence

But DO NOT simply copy this structure. Modify it according to the evidence you find.

For each subsection specify exactly which existing table/figure should be used.

---

# 12. Check Quantitative Consistency

Audit every important number that would appear in the revised Results section.

Check:

* RMSE
* MAE
* percentage improvements
* edge counts
* parameter counts
* seed means/std
* PH results
* Los-loop 5-min
* Los-loop 15-min
* SZ-Taxi

Identify any contradictions between:

* manuscript
* JSON files
* logs
* source code
* previous audit reports.

Create:

| Claim | Current manuscript value | Verified value | Status | Required action |

Do not silently fix anything.

---

# 13. Check Terminology Consistency

Audit all names used in:

* manuscript
* captions
* tables
* code
* JSON
* logs

Current authoritative manuscript terminology:

* `T-GCN-NoSpatial`
* `Physical`
* `T-GCN-MultiGSL`
* `T-GCN-MultiGSL-Mix`

Identify all places where alternative names occur, such as:

* `NoGraph`
* `GatedMultiGraphTGCN`
* `MultiGraphTGCN_fixed`

Do NOT modify them yet.

Recommend where a mapping note is needed and where names should simply be standardized.

---

# 14. Critical Scientific Question

Answer this explicitly:

### Can the revised paper legitimately contain BOTH:

1. the strong original GSL/cGSL results, and
2. the new T-GCN-MultiGSL-Mix results,

without making the paper look like two unrelated studies?

If YES:

* explain the logical bridge between them
* propose the exact narrative transition
* identify which original results are essential
* identify which can be removed

If NO:

* explain exactly why.

---

# 15. Required Final Deliverable

Create:

`results/stage31_manuscript_reconstruction/STAGE31_MANUSCRIPT_RECONSTRUCTION.md`

The report must contain:

1. Executive Summary
2. Original Submission Reconstruction
3. Current Manuscript Reconstruction
4. Reviewer-Criticism → Evidence Mapping
5. Method Evolution
6. Stage 26 / 27 / 29 Evidence Audit
7. Original Results Worth Preserving
8. Results That Should Be Removed or Condensed
9. Recommended Scientific Narrative
10. Recommended Results Section Structure
11. Quantitative Consistency Audit
12. Terminology Audit
13. Experiments That Actually Need Rerunning
14. Experiments That Do NOT Need Rerunning
15. Recommended Final Experimental Package
16. Risks / Potential Reviewer Objections
17. Concrete Next Steps for Stage 32

---

# 16. Final Decision Matrix

End the report with this table:

| Question                                                               | Answer |
| ---------------------------------------------------------------------- | ------ |
| Can original strong results be retained?                               | YES/NO |
| Can they be integrated into the main story?                            | YES/NO |
| Is the new architecture necessary to answer reviewers?                 | YES/NO |
| Is Stage 29 valid evidence?                                            | YES/NO |
| Does 15-min Los-loop support the temporal-resolution hypothesis?       | YES/NO |
| Does SZ-Taxi remain a marginal-result dataset?                         | YES/NO |
| Do we need more expensive experiments before manuscript restructuring? | YES/NO |
| Which existing experiments must be rerun?                              | ...    |
| Which existing experiments should be retained unchanged?               | ...    |
| Recommended next stage                                                 | ...    |

## Important scientific constraint

Do NOT try to make the results support a predetermined hypothesis.

In particular, do NOT attempt to resurrect the hypothesis that 15-minute temporal resolution explains SZ-Taxi's weak result unless the actual evidence supports it.

Our objective is to construct the **strongest scientifically defensible revision**, while preserving as much valid evidence from the original submission as possible.
