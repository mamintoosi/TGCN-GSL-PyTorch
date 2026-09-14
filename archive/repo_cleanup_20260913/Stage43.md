You are working on the major revision of the paper:

"Graph Structure Learning for Traffic Prediction"

Repository/project:
TGCN-GSL-PyTorch

The manuscript is in the `paper/` directory.

IMPORTANT:

* Do NOT delete any existing manuscript, table, figure, appendix, or result file.
* Do NOT modify the manuscript in this stage.
* Do NOT run new experiments.
* Do NOT change source code.
* This stage is analysis and planning only.
* The goal is to design the structure of the revised manuscript based on the completed Stage 40–42 results and audits.

There are now three conceptually different manuscript snapshots:

paper/submitted_version/
The exact version submitted to the journal, including its tables, figures, appendices, etc.

paper/previous_revision/
The intermediate revised manuscript that existed a few days ago, if present.

paper/revised_version/
This directory will eventually contain the new revised manuscript, but DO NOT create or modify it in this stage unless necessary for a planning report.

First inspect the existing manuscript files and the Stage 40–42 reports/results.

Key evidence available from the completed revision analysis:

1. Stage 40 provides a canonical 480-experiment evaluation:
   12 methods × 2 datasets × 4 prediction horizons × 5 seeds.

Canonical method names:

GCN
GCN-NoSpatial
GCN-GSL
GCN-cGSL
GCN-MultiGSL

T-GCN
T-GCN-NoSpatial
T-GCN-GSL
T-GCN-cGSL
T-GCN-MultiGSL
T-GCN-MultiGSL-Weighted
T-GCN-MultiGSL-Mix

Use exactly these names when referring to the canonical Stage 40 methods.

Do NOT use "Physical" as a canonical method name.
Do NOT use "Adaptive" in method names.

2. Stage 40 main numerical results:

Los-loop mean RMSE:

GCN:
8.1445, 8.5386, 8.7647, 8.7648

GCN-NoSpatial:
4.8796, 5.6044, 6.0226, 6.2585

GCN-GSL:
7.8265, 8.4789, 8.4482, 8.9335

GCN-cGSL:
5.7631, 6.3306, 6.6844, 6.8808

GCN-MultiGSL:
9.7800, 10.0497, 10.2410, 10.2720

T-GCN:
7.8771, 8.1329, 8.3676, 8.6594

T-GCN-NoSpatial:
5.2514, 5.7593, 6.1092, 6.5762

T-GCN-GSL:
5.8589, 6.3009, 6.6554, 7.0403

T-GCN-cGSL:
5.8205, 6.3536, 6.6689, 7.0881

T-GCN-MultiGSL:
4.8408, 5.4468, 5.9402, 6.2598

T-GCN-MultiGSL-Weighted:
4.8338, 5.4375, 5.9265, 6.2532

T-GCN-MultiGSL-Mix:
4.4914, 5.0751, 5.5457, 5.8589

SZ-Taxi mean RMSE:

GCN:
5.9596, 5.9752, 5.9893, 6.0011

GCN-NoSpatial:
4.1140, 4.1527, 4.1872, 4.2171

GCN-GSL:
4.8803, 4.9097, 4.9300, 4.9579

GCN-cGSL:
4.6412, 4.6744, 4.7009, 4.7283

GCN-MultiGSL:
4.8234, 4.8541, 4.8782, 4.9044

T-GCN:
5.4493, 5.5515, 5.6016, 5.6326

T-GCN-NoSpatial:
4.1199, 4.1604, 4.1919, 4.2251

T-GCN-GSL:
4.2809, 4.3089, 4.3325, 4.3698

T-GCN-cGSL:
4.3008, 4.3353, 4.3798, 4.4147

T-GCN-MultiGSL:
4.1303, 4.1607, 4.2002, 4.2245

T-GCN-MultiGSL-Weighted:
4.1304, 4.1608, 4.2006, 4.2246

T-GCN-MultiGSL-Mix:
4.1190, 4.1492, 4.1775, 4.2171

The four values in each row correspond to PH1–PH4.

3. Main Stage 40 findings:

Los-loop:
T-GCN-MultiGSL-Mix is the best method at every prediction horizon.

SZ-Taxi:
GCN-NoSpatial is best at PH1 and PH4.
T-GCN-MultiGSL-Mix is marginally best at PH2 and PH3.
Overall, learned graph structure provides little or no practical improvement over the graph-free baseline on SZ-Taxi.

4. Important improvements relative to T-GCN:

Los-loop:
T-GCN-NoSpatial:
33.33%, 29.18%, 26.99%, 24.06%

T-GCN-GSL:
25.62%, 22.53%, 20.46%, 18.70%

T-GCN-cGSL:
26.11%, 21.88%, 20.30%, 18.14%

T-GCN-MultiGSL:
38.55%, 33.03%, 29.01%, 27.71%

T-GCN-MultiGSL-Weighted:
38.63%, 33.14%, 29.17%, 27.79%

T-GCN-MultiGSL-Mix:
42.98%, 37.60%, 33.72%, 32.34%

SZ-Taxi:
T-GCN-NoSpatial:
24.39%, 25.06%, 25.17%, 24.99%

T-GCN-GSL:
21.44%, 22.38%, 22.66%, 22.42%

T-GCN-cGSL:
21.08%, 21.91%, 21.81%, 21.62%

T-GCN-MultiGSL:
24.20%, 25.05%, 25.02%, 25.00%

T-GCN-MultiGSL-Weighted:
24.20%, 25.05%, 25.01%, 25.00%

T-GCN-MultiGSL-Mix:
24.41%, 25.26%, 25.42%, 25.13%

5. Statistical evidence:

Five random seeds were used.

Paired comparisons generally show 5/5 seed wins for the graph-based T-GCN variants against T-GCN, with very small paired t-test p-values.

However:

* n=5 is small.
* Exact two-sided Wilcoxon tests have a minimum attainable p-value of 0.0625 for five paired observations.
* Therefore do NOT make strong claims of definitive statistical significance based only on these tests.
* Phrase statistical evidence cautiously.

6. Graph structure findings:

Physical graphs:

Los-loop:
2833 directed entries including 207 diagonal self-loops.
207 self-loops.
Off-diagonal density approximately 0.0616.
Symmetric.

SZ-Taxi:
532 positive off-diagonal directed entries.
No self-loops.
Off-diagonal density approximately 0.0220.
Not exactly symmetric.

Contemporaneous DAGMA graphs:
Los-loop: 28 edges per PH.
SZ-Taxi: 8 edges per PH.
DAGMA graphs have no self-loops; graph convolution internally adds self-loops through the Laplacian construction.

Multi-lag DAGMA:
Los-loop:
lag 1 = 12 edges
lag 2 = 3 edges
lag 3 = 15 edges
union = 28 distinct off-diagonal edges.

SZ-Taxi:
lag 1 = 0
lag 2 = 0
lag 3 = 2
union = 2 edges.

7. Sparsity control:

On Los-loop PH1, matched 30-edge controls were tested:

RandTop30:
6.10 ± 0.12

CorrTop30:
5.39 ± 0.10

DAGMA multi-lag:
4.84 ± 0.11

DAGMA multi-lag + Mix:
4.49 ± 0.14

This indicates that sparsity alone does not explain the improvement, at least for this controlled Los-loop experiment.

Do not generalize this control to all datasets/horizons.

8. Critical Stage 42 claim reconciliation:

A. "GSL improves over the physical graph":
Direction survives, but the interpretation must be reframed.

All learned contemporaneous graphs beat the physical graph, but NoSpatial beats both.
Therefore the improvement over the physical graph is not evidence that learned graph structure extracted useful predictive structure.

Preferred interpretation:

"Replacing the physical road-network adjacency—learned or identity—consistently reduces error; the learned contemporaneous graph recovers only a fraction of the gap that simply removing the graph recovers."

B. "GSL is better than NoSpatial":
RETIRED.
Do not make this claim.

C. "cGSL is consistently superior to GSL":
REFRAME as a mechanism-dependent observation.

For GCN, cGSL robustly beats GSL.
For T-GCN, the difference is not decisive.

Potential interpretation:
"Symmetrizing the learned adjacency matters specifically when downstream aggregation is symmetric (GCN), whereas the distinction is much less consequential for the temporal model."

Do not claim universal superiority of cGSL.

D. "T-GCN-GSL is the best":
RETIRED.

E. "GCN-cGSL is the best":
RETIRED.

F. "Learned graphs are universally beneficial":
RETIRED.

Replacement:
"The benefit of learned graph structure is dataset-dependent and is most evident when lag-specific graphs are consumed in a temporally aligned manner."

G. MultiGSL is now the central result.

The strongest evidence is:

* T-GCN-MultiGSL beats T-GCN-NoSpatial on Los-loop.
* T-GCN-MultiGSL-Mix improves further.
* GCN-MultiGSL uses the same multi-lag graph artifacts but unions them into one static graph and performs poorly.
* Therefore the results suggest that how the learned graphs are consumed is at least as important as how they are learned.

Do NOT describe this as causal evidence.
Do NOT claim that DAGMA recovered causal relationships.
Use "statistical dependency", "learned graph structure", "lag-specific structure", or similar cautious language.

8. Reviewer-driven requirements that must be reflected in the revised paper:

Reviewer 1:

* reduce excessive GCN/T-GCN background
* explain A versus W notation
* clarify exactly what is given to DAGMA
* distinguish contemporaneous dependency from temporal/lagged structure
* report graph density/degree statistics
* address sparsity confounding
* report multiple seeds and variability
* discuss longer prediction horizons
* explain GSL/cGSL differences
* add limitations
* discuss scalability
* discuss lambda/threshold sensitivity
* improve figures/convergence plots
* avoid repetitive results presentation

Reviewer 2:

* fix abstract numerical claims
* remove unsupported causal/hidden-structure claims
* resolve static-graph versus changing-traffic wording
* define cGSL earlier
* improve convergence plots
* correct typos.

9. Important terminology restrictions:

Do NOT use:

* "Original GSL"
* "Original cGSL"
* "Adaptive" as a method name
* "causal graph"
* "causal relationship"
* "hidden causal structure"

When referring to the previously submitted manuscript in the response to reviewers, use:
"the previous version of the manuscript"
or
"the first version of the manuscript".

10. The revised manuscript should not become an experiment catalog.

The central Results narrative should answer:

(1) Does graph structure help traffic prediction?
(2) Is the benefit due merely to using a sparse graph?
(3) Does the way graph structure is consumed matter?
(4) Is the benefit consistent across datasets?
(5) What does the learned graph actually represent?
(6) What are the limitations?

TASK:

Based on the actual files in `paper/`, Stage 40 results, Stage 41 statistical/graph analyses, and Stage 42 claim reconciliation:

A. Design a proposed section/subsection structure for the revised Results section.

B. Decide which results should appear:

* in the main paper,
* in an appendix/supplement,
* or only in the reviewer response.

C. Propose the minimum number of main tables needed.
For each proposed table, specify:

* purpose,
* rows,
* columns,
* whether values should be mean ± std,
* whether improvement percentages should be included.

D. Propose the minimum number of main figures.
For each figure specify:

* purpose,
* what should be plotted,
* which dataset(s),
* whether it should show all methods or only selected methods.

E. Decide how to integrate the older submitted results with the new Stage 40 results.
The old results must not be silently mixed with the new results.
Propose precise language for describing the relationship between them.

F. Identify any claims currently present in the manuscript that are no longer supported by Stage 40–42.
List them explicitly and propose a replacement interpretation.

G. Check whether the current abstract, introduction, contributions, method description, Results, Discussion, Conclusion, and reviewer-response material will require changes because of the revised scientific story.

H. Recommend the exact order in which the manuscript should be revised.

I. Do not edit the manuscript yet.
Instead produce a planning report:

paper/stage43_results_structure_plan.md

The report must contain:

1. Executive summary
2. Recommended revised scientific story
3. Proposed Results section hierarchy
4. Main tables
5. Main figures
6. Appendix/supplement material
7. Claims to retire/reframe
8. Relationship between submitted and new results
9. Required changes outside Results
10. Recommended revision order
11. Open issues that must be resolved before manuscript rewriting

Before producing the report, inspect the actual manuscript and existing paper files. Base recommendations on what is actually present, rather than assuming that the manuscript has a particular structure.
