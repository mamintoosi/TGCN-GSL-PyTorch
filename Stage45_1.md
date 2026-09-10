# Stage 45.1 — Finalize the New Manuscript Architecture

## Objective

Refine the Stage 45 architecture blueprint into the final architecture for the new manuscript.

This is a PLANNING-ONLY stage.

Do NOT:
- edit the manuscript,
- create or replace any manuscript section,
- modify `submitted_version`,
- modify `previous_revision`,
- modify code,
- run experiments,
- change any results,
- create figures or tables.

You may create only the Stage 45.1 planning report.

---

## Important Editorial Principle

The new manuscript must read as a completely coherent scientific paper from the reader's perspective.

The reader should NOT see the development history of the project.

In particular, the final manuscript must NOT contain:
- "historical results"
- "first version of the manuscript"
- "previous version"
- "submitted version"
- "earlier version"
- "original results"
- "re-baselining"
- discussion of how the manuscript or experiments evolved
- explanations of why current numbers differ from previously submitted numbers

Those matters belong only in the confidential revision process / Response to Reviewers, not in the scientific manuscript.

The manuscript should present the final experimental design and final evidence as the scientific study.

---

## Sources of Truth

Use the following, in this priority order:

1. `paper/gsl_stage45/stage45_new_manuscript_architecture.md`
2. `paper/submitted_version/sn-article.tex`
3. `paper/Reviewers-comments.txt`
4. Stage 40 canonical results
5. Stage 41 statistical analysis
6. Stage 42 audit and claim reconciliation
7. Stage 43 results-structure plan
8. Stage 44 protocol reconciliation

You may inspect `paper/previous_revision/` only for useful material or implementation context.

DO NOT use `previous_revision` as the structural basis of the manuscript.

---

# Required Corrections to Stage 45

## 1. Remove the historical-results appendix

The proposed:

    Appendix A — Historical results from the first version of the manuscript

must be completely removed.

Do not replace it with another appendix containing historical comparisons.

The old submitted results remain preserved in:

    paper/submitted_version/

for internal/revision purposes only.

They are NOT part of the new manuscript's scientific evidence.

---

## 2. Remove "historical baseline" as a contribution

Stage 45 Contribution 5 currently contains the idea of:

    "a corrected historical baseline"
    "preserving the first version's results"

This is NOT a scientific contribution.

Delete this contribution.

The final manuscript should have 3–4 genuine scientific contributions, not a contribution about manuscript history.

---

## 3. Remove historical-version language from the scientific story

Review the entire Stage 45 blueprint and eliminate any dependency on statements such as:

- "the submitted paper reported..."
- "the first version showed..."
- "historical results..."
- "previous results..."
- "the earlier pipeline..."
- "re-baselining..."
- "corrected historical baseline..."

If such information is needed for the reviewer response, mark it as:

    RESPONSE-LETTER ONLY

It must not appear in the manuscript architecture.

---

## 4. Do not present the old results as an appendix

The final manuscript should contain only results that support the final scientific story.

The old tables should NOT be:
- reproduced,
- summarized as historical tables,
- placed in an appendix,
- mixed numerically with Stage 40 results.

---

## 5. Reconsider the appendix structure

Propose the cleanest final appendix structure based on what is actually useful to a reader.

A likely structure is:

    Appendix A — Additional Experimental Results
    Appendix B — Graph Statistics and Implementation Details
    Appendix C — Bibliometric Analysis

But do not assume this is mandatory.

Decide whether each appendix is genuinely useful.

If an appendix does not add scientific value, remove it.

The paper should not contain appendices merely to preserve material from earlier versions.

---

## 6. Recheck the scientific contributions

Rewrite the contribution list so that every contribution describes something scientifically useful to the reader.

The contributions should be traceable to the final evidence, especially:

1. the graph-free baseline finding;
2. the distinction between single contemporaneous graphs and multi-lag graphs;
3. the graph-consumption dissociation using the same learned graphs;
4. the conditional benefit of lag-specific consumption / Mix on Los-loop;
5. the dataset-dependent boundary demonstrated by SZ-Taxi;
6. sparsity controls, if judged strong enough to be a contribution rather than merely supporting evidence.

Do NOT force six contributions.

Prefer 3–4 strong contributions.

---

## 7. Recheck the title

Reassess the three Stage 45 title candidates after removing all historical framing.

The title should describe the actual scientific contribution, not the revision process.

Pay particular attention to whether:

    "Graph Structure or Graph Consumption? Dissecting the Benefit of Learned Multi-Lag Graphs for Traffic Forecasting"

is still the strongest title.

Do not change the title merely for stylistic reasons; explain the scientific rationale.

---

## 8. Recheck the TOC

Produce a final recommended TOC.

The manuscript should remain reasonably flat and readable.

Avoid excessive subsection nesting.

The current candidate structure is:

    Abstract
    Keywords

    1 Introduction

    2 Background and Related Work

    3 Problem Formulation and Graph Learning Framework

    4 Experimental Setup

    5 Results

    6 Discussion

    7 Limitations

    8 Conclusion

    Appendices

Reassess the subsection structure, especially Results.

The Results section must tell one coherent scientific story rather than becoming a catalog of experiments.

---

## 9. Recheck Results ordering

The recommended logical progression should be evaluated critically:

    5.1 Physical graph vs graph-free baseline
    5.2 Single contemporaneous GSL/cGSL
    5.3 Sparsity and capacity controls
    5.4 Multi-lag structure and graph-consumption dissociation
    5.5 Multi-lag mixing and multi-seed validation
    5.6 Temporal resolution and dataset dependence

Check whether this is the clearest order.

In particular:

- The negative baseline result should appear early.
- The single-graph result should establish why the paper moves to multi-lag structure.
- The consumption dissociation should be prominent.
- The positive Los-loop result should not be buried.
- SZ-Taxi should clearly establish the boundary of the claim.
- The 15-minute experiment must remain clearly labeled as a temporal-resolution variant, not as a third independent dataset.
- Do not imply that PH5–8 at 5-minute resolution were evaluated.

If a better ordering exists, recommend it.

---

## 10. Recheck main tables

Critically reassess whether Tables 1–5 from Stage 45 are all necessary.

For each proposed main table, state:

- keep / merge / move to appendix / remove
- scientific message
- exact methods
- datasets
- horizons
- whether it risks redundancy

Pay particular attention to whether a separate per-seed table is necessary when a main results table already reports mean±std and Figure 4 shows seed distributions.

Do not overload the main text.

---

## 11. Recheck main figures

Critically reassess Figures 1–5.

For each figure, decide:

- keep / redesign / appendix / remove
- exact scientific question it answers
- whether it duplicates a table
- whether the figure is actually necessary for the reviewer request

In particular:

### Figure 1
Physical vs learned graph structure.

Make sure the graph visualization does not imply causality.

### Figure 2
Sparsity controls.

Confirm that the scope is explicitly Los-loop PH1.

### Figure 3
Graph-consumption dissociation.

This should remain the conceptual centerpiece if the evidence supports that interpretation.

### Figure 4
Per-seed distributions.

Assess whether this adds enough beyond mean±std.

### Figure 5
Predicted vs actual.

Assess whether this is scientifically informative or merely illustrative.

Do not retain figures merely because a reviewer asked for a figure if the figure does not materially improve the paper.

---

## 12. Recheck claims

Produce a final claim policy with three categories:

### KEEP
Claims directly supported by the final evidence.

### REFRAME
Claims that require careful qualification.

### RETIRE
Claims that should disappear entirely.

Pay special attention to:

- GSL > physical
- GSL > NoSpatial
- cGSL > GSL
- MultiGSL > NoSpatial
- universal benefit of learned graphs
- causal interpretation
- temporal interpretation of contemporaneous DAGMA
- adaptive graph claims
- sparsity explanations
- longer-horizon claims
- statistical significance with n=5

Every retained headline claim must have:
- dataset scope,
- model scope,
- horizon scope where relevant,
- statistical qualification where necessary.

---

## 13. Recheck reviewer mapping

Update the reviewer-to-manuscript map so it describes the FINAL manuscript only.

It must not say that a reviewer concern is addressed by an appendix containing historical results.

For partially addressed reviewer requests, explicitly distinguish:

- fully addressed by existing evidence,
- partially addressed,
- limitation / future work.

Especially:

R1-W5:
- sparsified physical graph control was not run;
- lambda/threshold sweep was not run.

R1-W7:
- PH5–8 at 5-minute resolution were not run;
- the 15-minute sampling experiment is a different temporal-resolution analysis.

R1-Q3:
- genuinely time-varying learned graphs remain future work;
- the current gate changes graph usage, not the learned graph itself.

Do not imply experiments were performed when they were not.

---

## 14. Recheck the abstract strategy

The abstract strategy must describe only the final scientific study.

It must NOT mention:
- previous manuscript versions,
- historical results,
- correction of earlier claims,
- re-baselining.

Use only final canonical evidence.

Recheck the headline numerical claims, especially:
- Los-loop improvement of Mix over NoSpatial;
- improvement over physical T-GCN;
- SZ-Taxi boundary;
- matched-sparsity evidence.

Make sure no number from Stage 29 is accidentally presented as if it were the main canonical 5-minute result.

---

## 15. Recheck the role of the 15-minute experiment

The architecture must clearly distinguish:

### Canonical datasets
- Los-loop: 5-minute sampling
- SZ-Taxi: 15-minute sampling

### Additional temporal-resolution variant
- Los-loop resampled to 15-minute intervals

The 15-minute Los-loop experiment must NOT be described as:
- a third dataset,
- evidence for PH5–8 at 5-minute resolution,
- directly comparable PH-by-PH with the 5-minute experiment.

It may support a qualified temporal-resolution observation.

---

## 16. Recheck terminology

The final architecture must consistently use:

- learned graph / learned adjacency
- contemporaneous graph
- multi-lag graph
- lag-specific graph
- statistical dependency
- graph consumption
- per-timestep graph usage
- per-node, per-timestep mixing

Avoid:

- hidden causal structure
- causal graph
- causal relationship
- temporal DAG, when referring to the contemporaneous graph
- graph adapts to traffic
- Adaptive as a method name

Use "Mix" as the method name if that is the canonical result label.

---

# Output

Create:

    paper/gsl_stage45_1/stage45_1_final_architecture.md

The report must contain exactly these major sections:

1. Executive Summary
2. Final Scientific Position
3. Final Title Recommendation
4. Final Table of Contents
5. Final Section-by-Section Plan
6. Final Results Architecture
7. Final Main Tables
8. Final Main Figures
9. Final Scientific Contributions
10. Final Claim Policy
11. Final Reviewer-to-Manuscript Map
12. Final Abstract Strategy
13. Appendix Policy
14. Writing Sequence
15. Remaining Decisions
16. Final Verdict

For every proposed change from Stage 45, briefly explain why.

---

## Final Verdict

End with exactly one of:

    READY FOR MANUSCRIPT WRITING

or

    NOT READY FOR MANUSCRIPT WRITING

The desired outcome is READY, provided that the architecture is internally coherent and no unresolved scientific issue blocks manuscript writing.

Again: this stage is PLANNING ONLY. Do not write or modify the manuscript itself.