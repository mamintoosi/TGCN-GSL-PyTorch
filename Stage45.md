Stage 45 — Design the Architecture of a New Manuscript
=======================================================

Goal
----
Design the structure and scientific narrative of a NEW manuscript from scratch,
based on the submitted manuscript, reviewer comments, and the latest validated
experimental results.

IMPORTANT:
Do NOT edit, rewrite, or replace any manuscript file.
Do NOT modify previous_revision.
Do NOT modify submitted_version.
Do NOT run new experiments.
Do NOT change code or results.

The purpose of this stage is ONLY to produce a detailed blueprint for the new
manuscript that will be written in a later stage.

-------------------------------------------------------
1. SOURCE OF TRUTH
-------------------------------------------------------

Use the following sources, in this priority order:

1. The submitted manuscript:
   paper/submitted_version/sn-article.tex

2. Reviewer comments / response material associated with the submitted paper.

3. The latest validated experimental evidence:
   - Stage40 canonical results
   - Stage41 statistical analysis
   - Stage42 submitted-vs-Stage40 audit
   - Stage42 claim reconciliation
   - Stage43 results-structure plan
   - Stage44 protocol/provenance reconciliation

4. Previous manuscript versions may be inspected for useful material,
   but MUST NOT be treated as the structural basis of the new manuscript.

The new manuscript must be conceptually reconstructed from the submitted
paper plus the latest evidence.

-------------------------------------------------------
2. CENTRAL SCIENTIFIC QUESTION
-------------------------------------------------------

Determine the strongest scientifically defensible central question that the
new manuscript can answer using the validated experiments.

Do not preserve the old thesis merely because it appeared in the submitted
paper.

In particular, critically evaluate whether the evidence supports claims such
as:

- learned graphs improve traffic prediction in general;
- GSL is better than a graph-free model;
- cGSL is generally better than GSL;
- a single learned graph is sufficient;
- MultiGSL is universally beneficial;
- the learned graph represents causal relationships;
- the method adapts to changing traffic conditions.

Retain only claims supported by the latest evidence.

The new manuscript should explicitly reflect the important Stage40–42 finding
that graph-free / NoSpatial models can outperform physical and contemporaneous
learned graphs, while lag-specific MultiGSL consumption is beneficial on
Los-loop but not meaningfully beneficial on SZ-Taxi.

-------------------------------------------------------
3. DESIGN A NEW TABLE OF CONTENTS
-------------------------------------------------------

Propose a complete table of contents for the new manuscript.

For every section and subsection, provide:

- Section number
- Proposed title
- Purpose of the section
- Main scientific question answered
- Key evidence or material to include
- Which reviewer concern(s) it addresses
- Which old manuscript material can potentially be reused
- Which material should NOT be carried over

Do not write the actual manuscript text.

The structure should be appropriate for a journal paper in graph learning /
traffic prediction, not a thesis or tutorial.

Avoid excessive subsectioning.

-------------------------------------------------------
4. PROPOSED HIGH-LEVEL STRUCTURE
-------------------------------------------------------

You may depart from the following structure if the evidence suggests a better
one, but evaluate it explicitly:

1. Introduction

2. Related Work / Background

3. Problem Formulation

4. Method

5. Experimental Setup

6. Results

7. Discussion

8. Limitations

9. Conclusion

Appendix / Supplementary Material

For each section, determine what should actually remain in the final paper.

In particular, do not automatically reproduce the long GCN/T-GCN background
from the submitted manuscript.

-------------------------------------------------------
5. RESULTS ARCHITECTURE
-------------------------------------------------------

Design the Results section carefully.

It must tell a coherent scientific story rather than simply list experiments.

Evaluate the following possible progression:

5.1 Baseline landscape: physical graph vs graph-free models

5.2 Single-graph learned structure:
    GSL and cGSL relative to physical and NoSpatial baselines

5.3 Sparsity and capacity controls

5.4 Multi-lag graph structure

5.5 Graph consumption:
    union/static consumption vs lag-specific consumption

5.6 Adaptive mixing of lag-specific graphs

5.7 Multi-seed validation and statistical evidence

5.8 Longer prediction horizons / 15-minute sampling

5.9 Dataset dependence: Los-loop vs SZ-Taxi

Decide whether these should remain separate subsections, be merged, reordered,
or partly moved to Discussion or Appendix.

The final Results structure should avoid repetition and should make the main
scientific contribution obvious.

-------------------------------------------------------
6. MAIN RESULTS TABLES AND FIGURES
-------------------------------------------------------

Design the main tables and figures.

For each proposed table/figure specify:

- identifier (e.g. Table 1, Figure 1)
- title
- purpose
- methods included
- datasets
- prediction horizons
- metrics
- whether it belongs in the main paper or appendix
- reviewer concern addressed

Use the validated Stage40 results as the numerical basis.

Pay particular attention to:

A. T-GCN family:
   T-GCN
   T-GCN-NoSpatial
   T-GCN-GSL
   T-GCN-cGSL
   T-GCN-MultiGSL
   T-GCN-MultiGSL-Weighted
   T-GCN-MultiGSL-Mix

B. GCN family:
   GCN
   GCN-NoSpatial
   GCN-GSL
   GCN-cGSL
   GCN-MultiGSL

C. Sparsity controls:
   RandTop30
   CorrTop30
   DAGMA sparse graph
   DAGMA + Mix

Do not create unnecessary tables merely to include every result.

-------------------------------------------------------
7. MANUSCRIPT CLAIMS
-------------------------------------------------------

Create three lists:

A. Claims that should remain.

B. Claims that should be reframed.

C. Claims that should be retired.

Base this explicitly on Stage42 claim reconciliation.

For each important claim, explain why.

Pay special attention to:

- causal interpretation;
- temporal interpretation of DAGMA;
- static vs adaptive graph language;
- NoSpatial baseline;
- cGSL;
- MultiGSL;
- dataset dependence;
- statistical significance;
- sparsity;
- scalability.

-------------------------------------------------------
8. CONTRIBUTIONS
-------------------------------------------------------

Propose a new set of 3–5 contribution statements.

These must describe what the new paper actually contributes.

Do NOT simply rewrite the old contribution list.

Each proposed contribution must be traceable to validated evidence.

Avoid promotional language such as:

"novel breakthrough"
"discovers hidden causal structure"
"universally improves"
"significantly outperforms"

unless explicitly justified by the evidence.

-------------------------------------------------------
9. ABSTRACT AND TITLE
-------------------------------------------------------

Propose:

- 3 candidate titles
- one recommended title
- a one-paragraph description of what the new abstract should say

Do NOT write the full abstract yet.

The titles should reflect the actual revised scientific contribution rather
than the old claim that GSL simply improves traffic prediction.

-------------------------------------------------------
10. REVIEWER RESPONSE ALIGNMENT
-------------------------------------------------------

Create a reviewer-to-manuscript map.

For every reviewer comment from Reviewer 1 and Reviewer 2, specify:

- where the issue will be addressed in the new manuscript;
- what evidence will be used;
- whether a new experiment is required;
- whether an existing experiment is sufficient;
- whether the reviewer comment requires only editorial clarification.

Clearly identify any reviewer request that remains only partially addressed
by the existing evidence.

-------------------------------------------------------
11. OLD vs NEW MANUSCRIPT
-------------------------------------------------------

Create a concise migration plan:

KEEP:
material from submitted manuscript that remains valid.

REWRITE:
material whose scientific meaning or interpretation has changed.

REMOVE:
material that is no longer supported.

MOVE TO APPENDIX:
material that remains useful but is too detailed for the main narrative.

ADD:
material required by reviewers or by the new scientific story.

Do NOT modify the actual files.

-------------------------------------------------------
12. FINAL SCIENTIFIC NARRATIVE
-------------------------------------------------------

End the report with a concise "One-paragraph paper story".

This should explain the new paper from beginning to end:

- What problem is being studied?
- What question is asked?
- What baseline reveals an important issue?
- What does single-graph GSL show?
- Why is multi-lag structure introduced?
- Why does graph consumption matter?
- What does the Los-loop result demonstrate?
- What does SZ-Taxi demonstrate as a boundary case?
- What is the final defensible contribution?

The paragraph must be scientifically cautious and must not use causal
interpretations that are unsupported by the experiments.

-------------------------------------------------------
13. OUTPUT
-------------------------------------------------------

Create:

paper/gsl_stage45/stage45_new_manuscript_architecture.md

The report must contain:

1. Executive summary
2. Recommended central research question
3. Recommended title
4. Complete new table of contents
5. Detailed section/subsection plan
6. Results architecture
7. Main tables
8. Main figures
9. Claims: keep / reframe / retire
10. New contributions
11. Abstract strategy
12. Reviewer-to-manuscript map
13. Submitted-to-new migration plan
14. One-paragraph final paper story
15. Open issues / decisions required before writing

Do not make any manuscript edits.

-------------------------------------------------------
14. IMPORTANT FINAL CHECK
-------------------------------------------------------

Before finishing, verify that the proposed manuscript:

- is based primarily on submitted_version, not previous_revision;
- incorporates the latest validated Stage40–42 evidence;
- respects Stage44 protocol/provenance decisions;
- does not resurrect retired claims;
- does not claim causality without evidence;
- does not treat NoSpatial as an insignificant baseline;
- distinguishes contemporaneous single-graph GSL from multi-lag GSL;
- distinguishes graph structure from graph-consumption mechanism;
- clearly represents the Los-loop benefit and SZ-Taxi boundary;
- addresses all major reviewer comments;
- does not require results that have not actually been validated.

At the end, give a clear verdict:

READY FOR MANUSCRIPT WRITING
or
NOT READY — [list the blocking issues].

Do not write any manuscript section in this stage.