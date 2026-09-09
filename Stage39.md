# Stage 39 — Manuscript Revision & Point-by-Point Reviewer Response

## Objective

Revise the manuscript and prepare the complete point-by-point response to the reviewers using the scientifically audited results and integration plan from Stage 38.

This stage is a MANUSCRIPT-REVISION stage.

Do NOT run experiments.
Do NOT regenerate DAGMA graphs.
Do NOT modify historical artifacts.
Do NOT silently change numerical results.
Do NOT invent evidence, experiments, significance tests, or reviewer claims.

The goal is to integrate the validated Stage 26–38 evidence into the manuscript and produce a rigorous, conservative, reviewer-facing revision.

============================================================
1. AUTHORITATIVE INPUTS
============================================================

Use the following as authoritative sources:

- doc/STAGE38_SCIENTIFIC_AUDIT_REPORT.md
- paper/Reviewers-comments.txt
- paper/sn-article.tex
- paper/sections/*.tex
- all relevant Stage 26–37 reports
- Stage 26 result artifacts
- Stage 32 sparse-control results
- Stage 33 canonical GSL results
- Stage 33 SZ multiseed results
- historical_submission archive
- relevant experiment scripts
- git history and commits referenced in Stage 38

Before editing, inspect the actual current manuscript and all relevant sections.

Stage 38 is the authoritative scientific interpretation of A/B/C.

If a number or claim is not supported by Stage 38 or an existing artifact, do not invent it.

============================================================
2. IMPORTANT SCIENTIFIC RULES
============================================================

### 2.1 No experiments

Do NOT run:

- Experiment A
- Experiment B
- Experiment C
- PH5–8 experiments
- lambda sensitivity experiments
- new multiseed experiments
- new DAGMA fits

Do not rerun anything merely to "confirm" Stage 38.

The Stage 38 conclusion is:

GO FOR MANUSCRIPT REVISION.

Optional experiments listed in Stage 38 are NOT part of this stage.

Do not perform the optional sparsified-physical control unless explicitly required by an existing manuscript inconsistency. Prefer proceeding without experiments.

------------------------------------------------------------

### 2.2 Historical results must remain historical

Do not overwrite or delete historical results.

In particular, preserve the original:

- TGCN-GSL values such as 4.818, 5.400, 5.846, 6.257
- original single-seed results
- historical W_est files
- original appendix material

However, clearly label them as historical / original-submission results where necessary.

Do NOT present historical absolute RMSE values as current canonical results.

The current canonical GSL baseline is:

PH1: 5.792 ± 0.196
PH2: 6.328 ± 0.166
PH3: 6.669 ± 0.080
PH4: 6.999 ± 0.084

with Physical:

PH1: 7.772 ± 0.140
PH2: 8.118 ± 0.189
PH3: 8.456 ± 0.047
PH4: 8.554 ± 0.169

and mean improvement approximately 21.7%.

State explicitly that absolute RMSE values are not directly comparable to the original pipeline because the training pipeline was revised.

------------------------------------------------------------

### 2.3 Experiment A interpretation

Integrate the following results:

Random sparse graph:
6.096 ± 0.107

Correlation sparse graph:
5.389 ± 0.088

NoSpatial:
5.234 ± 0.090

MultiGSL:
4.794 ± 0.102

MultiGSL-Mix:
4.452 ± 0.143

Matched edge budget:
30 edges.

Also integrate the existing union control:

same DAGMA-derived edges used as a single union adjacency:
5.928

The manuscript must NOT claim:

"sparsity causes the improvement."

Instead explain that:

- sparsity alone is insufficient;
- random sparse structure can hurt;
- correlation-based sparse structure does not beat NoSpatial;
- the DAGMA lag-specific structure combined with per-lag graph usage is what produces the observed gain;
- the MultiGSL mechanism is important, because the same edges used as one union graph do not reproduce the MultiGSL result.

Use conservative scientific language.

------------------------------------------------------------

### 2.4 Experiment B interpretation

Correctly explain the two protocols.

Historical protocol:

- K offset-stratified DAGMA fits
- data[i::K]
- union of supports
- historical edge counts:
  28 / 32 / 33 / 39 for PH1–PH4

Canonical revised protocol:

- one DAGMA fit per horizon
- train_norm[0::PH]
- 207-variable contemporaneous DAG
- absolute-magnitude thresholding
- 28 edges per horizon

State that the historical slice-0 graph was reproduced:

- PH1: exact support reproduction
- PH2: exact support reproduction
- PH4: exact support reproduction
- PH3: 26/28 support agreement

Shared-edge weights agree within approximately 0.016.

Do not claim bit-identical weight reproduction.

The key conceptual correction is:

The original single-graph GSL implementation was contemporaneous, not explicitly lagged.

The explicit temporal/lagged construction is provided by the MultiGSL formulation:

Z = [x(t-L), ..., x(t)]

and the lag-specific graph blocks.

This distinction must be reflected consistently throughout the manuscript.

------------------------------------------------------------

### 2.5 Experiment C interpretation

Use the 5-seed SZ results:

PH1:
NoSpatial 4.1192 ± 0.0069
MultiGSL 4.1302 ± 0.0152
Mix 4.1091 ± 0.0052

PH2:
NoSpatial 4.1623 ± 0.0055
MultiGSL 4.1600 ± 0.0036
Mix 4.1515 ± 0.0026

PH3:
NoSpatial 4.1884 ± 0.0011
MultiGSL 4.1988 ± 0.0121
Mix 4.1804 ± 0.0052

PH4:
NoSpatial 4.2196 ± 0.0013
MultiGSL 4.2273 ± 0.0092
Mix 4.2159 ± 0.0073

Interpretation:

- MultiGSL alone does not improve SZ.
- Mix provides only marginal improvements at PH1–PH3.
- PH4 is not stable and should be described as within noise / not stable.
- The result is dataset-dependent.
- Do NOT claim statistical significance.
- Do NOT introduce p-values.
- Do NOT call the PH4 result a "dip" or claim stable improvement.

Use win counts where useful:

- Mix wins 5/5 seeds at PH1–PH3.
- Mix wins 3/5 at PH4.

------------------------------------------------------------

### 2.6 Signed-weight policy

Add a short methodological clarification.

The correct rule is:

an edge survives if its estimated DAGMA coefficient survives the library's absolute-magnitude threshold, regardless of sign.

However:

- no negative coefficient survived the threshold in the examined traffic artifacts;
- therefore this clarification caused no numerical change to reported results.

Do NOT imply that negative coefficients were empirically important.

------------------------------------------------------------

## 3. MANUSCRIPT INTEGRATION TASKS

Execute the following in order.

### Task 1 — Oversmoothing section

Locate:

tab:oversmoothing

and its surrounding discussion.

Add:

- RandTop30
- CorrTop30

with the Stage 38 values.

Add the matched-budget interpretation.

Add the union-control result (5.928).

Ensure that the conclusion is not "sparsity improves performance."

Use the safe scientific conclusion from Stage 38.

------------------------------------------------------------

### Task 2 — Single-graph GSL baseline

Add a short subsection or coherent extension of the relevant results section.

Suggested conceptual title:

"The single-graph GSL baseline revisited"

Include:

- Physical vs GSL
- PH1–PH4
- five-seed mean ± std
- 21.7% mean improvement
- 25.5% PH1 improvement
- reproduction relationship to the historical submission

Clearly explain that this is the revised canonical protocol.

------------------------------------------------------------

### Task 3 — SZ multiseed table

Locate:

tab:sz_multiph

Replace the seed-42 rows with the five-seed values from Experiment C.

Add the MultiGSL row.

Rewrite the accompanying discussion to say:

- gated Mix gives marginal gains at PH1–PH3;
- ungated MultiGSL does not help;
- PH4 is not stable / within noise;
- this demonstrates dataset dependence.

Do not overstate the effect.

------------------------------------------------------------

### Task 4 — Main multihorizon table

Inspect:

tab:multiph

If scientifically and structurally appropriate, add the canonical GSL baseline alongside Physical and the proposed method.

Do not create redundancy if the new dedicated GSL subsection already presents the same information more clearly.

Choose the cleaner manuscript structure.

------------------------------------------------------------

### Task 5 — Temporal interpretation

Inspect every paragraph discussing the temporal meaning of the learned graph.

Correct any statement implying that the original single-graph DAGMA implementation directly learned temporal lag relationships.

Replace it with the validated interpretation:

- original single-graph GSL: contemporaneous structure;
- MultiGSL: explicit lag-stacked formulation;
- per-lag graphs provide the temporal structural representation.

Make sure terminology is consistent throughout Sections 3, 5, Discussion, and Conclusion.

------------------------------------------------------------

### Task 6 — Static graph vs changing traffic

Address Reviewer R2-3.

Inspect the relevant item in Section 3.2.

Clarify:

- the learned graph is fixed after training;
- the GRU/T-GCN component models temporal dynamics;
- the learned graph represents functional spatial dependencies rather than a graph that continuously changes during inference.

Remove any apparent contradiction between "static graph" and "adapts to changing traffic patterns."

------------------------------------------------------------

### Task 7 — Interpretability / causal language

Address R2-2 carefully.

Search the entire manuscript for:

- causal
- causality
- cause
- causal structure
- hidden causal structure
- explanation
- interpretable

Do not claim causal discovery or causal validation unless directly supported.

The graph analysis should be described as:

- descriptive structural analysis;
- learned dependency structure;
- graph topology;
- degree/edge statistics;
- functional dependencies.

The graph visualizations support structural interpretation, not causal proof.

------------------------------------------------------------

### Task 8 — Limitations

Strengthen the limitations section.

Include:

1. DAGMA scalability:
   - approximately 16 min/PH for 207 nodes using 4 CPU cores;
   - approximately 52–76 min/fit for archived 156-node SZ audits;
   - multi-lag 828-variable fit approximately 4 h.

2. lambda sensitivity:
   - no systematic lambda sweep was performed;
   - do not imply that sensitivity was measured.

3. backbone diversity:
   - experiments use only the studied T-GCN/GCN-family backbones;
   - broader backbone validation remains future work.

4. forecast horizon:
   - main 5-min experiments cover PH1–PH4;
   - the 15-min-resolution experiment provides evidence up to 60 minutes ahead;
   - PH5–PH8 in the 5-min setting remain future work.

5. dataset dependence:
   - SZ results show only marginal benefit for the gated model and no clear benefit for ungated MultiGSL.

Keep the limitations honest and concise.

------------------------------------------------------------

### Task 9 — Reproducibility paragraph

Add the signed-weight clarification and reproducibility information where appropriate.

Mention:

- DAGMA determinism audit;
- seed policy;
- five-seed mean ± std;
- threshold semantics;
- historical reproduction.

Do not overload the main text. Put detailed provenance in the appendix where appropriate.

------------------------------------------------------------

### Task 10 — Historical appendix

Update the original GSL appendix.

Explain:

- historical protocol used union-of-K-offset fits;
- edge counts 28/32/33/39;
- revised canonical protocol uses one DAG per horizon;
- slice-0 graphs were reproduced;
- original appendix values remain historical single-seed results.

Do not delete historical results.

Do not replace historical tables with current numbers unless the existing appendix structure requires a clearly labeled additional table.

------------------------------------------------------------

### Task 11 — Abstract

Inspect the abstract.

Ensure that the obsolete:

21.6% / 24.7%

framing is absent.

The current headline numbers should remain consistent with the validated revised manuscript, especially:

- 14.9%
- 27.4%

If useful, add a concise statement that the original GSL-vs-physical improvement was independently reproduced at approximately 21.7% under the revised protocol.

Do not overcrowd the abstract.

------------------------------------------------------------

### Task 12 — Figures

Inspect existing:

- fig:graph_comparison
- fig:pred_vs_actual

Do not create new figures unless strictly necessary.

Do not remove useful existing figures.

If captions contain claims inconsistent with Stage 38, correct them.

------------------------------------------------------------

## 4. REVIEWER RESPONSE LETTER

After revising the manuscript, create or update the reviewer response document.

Use:

paper/Reviewers-comments.txt

as the authoritative reviewer source.

Prepare a point-by-point response covering:

### Reviewer 1

At minimum:

- R1-W4
- R1-W5
- R1-W6
- R1-W7
- R1-Q3
- R1-W9

Also address any related reviewer items actually present in the source.

For each:

1. Quote or accurately paraphrase the reviewer concern.
2. State what was changed.
3. Give the relevant scientific evidence.
4. Identify where the manuscript was changed.
5. Be precise about what was NOT tested.

For R1-W5 explicitly state:

- matched 30-edge controls were added;
- random sparse graph does not explain the gain;
- correlation graph does not explain the gain;
- DAGMA lag-specific MultiGSL does better;
- the same DAGMA edges used as one union graph do not reproduce the MultiGSL gain;
- the reviewer's suggested sparsified-physical control was not run, if this remains true.

Do not claim that the R1-W5 issue is mathematically "proved"; say that the matched controls substantially address the sparsity confound.

For R1-W6 explicitly state:

- five seeds;
- mean ± std;
- seeds 42–46;
- DAGMA determinism;
- no significance testing was performed.

For R1-W7:

- do not pretend PH5–8 were run;
- explain the PH1–PH4 scope;
- mention the 15-minute sampling experiment as the longer wall-clock horizon evidence;
- state PH5–8 as future work if appropriate.

For R1-Q3:

- give the measured DAGMA runtime evidence;
- explain that sliding-window graph learning would multiply fitting cost approximately with the number of windows.

For R1-W9:

- explicitly acknowledge no lambda sweep;
- mention scalability;
- mention backbone limitation.

### Reviewer 2

At minimum:

- R2-1
- R2-2
- R2-3
- R2-4
- R2-5
- R2-6

Use the exact reviewer concerns from Reviewers-comments.txt.

For R2-1:

Explain the correction of the abstract numbers and the new 21.7% five-seed reproduction.

For R2-2:

Explain that graph analysis is descriptive rather than causal.

For R2-3:

Explain the static graph / dynamic temporal modeling distinction.

For R2-4/Q4-type concerns:

Explain the contemporaneous original GSL construction and the explicit lagged MultiGSL construction.

Do not claim more than the evidence supports.

------------------------------------------------------------

## 5. RESPONSE-LETTER STYLE

The response should be:

- professional;
- respectful;
- specific;
- evidence-based;
- non-defensive;
- concise but sufficiently detailed;
- suitable for journal submission.

Use language such as:

"We thank the reviewer..."
"We agree that..."
"To address this concern..."
"Our audit/reproduction showed..."
"We have clarified..."
"We have tempered this claim..."
"We did not perform..."
"Accordingly, we now state..."

Avoid:

- argumentative language;
- blaming reviewers;
- claiming "the reviewer was wrong";
- unsupported claims of significance;
- claiming that an experiment was performed when it was not.

------------------------------------------------------------

## 6. GLOBAL CONSISTENCY AUDIT

After all edits, perform a full manuscript consistency audit.

Search for all occurrences of:

- 21.6
- 24.7
- 21.8
- 26.9
- 4.818
- 5.400
- 5.846
- 6.257
- 1307
- "5 edges"
- 5 edges/PH
- causal
- temporal graph
- dynamic graph
- adaptive graph
- changing traffic patterns
- sparsity
- lag
- contemporaneous

For every occurrence determine whether it is:

1. current canonical result,
2. historical result,
3. explicitly discussed obsolete value,
4. or an error.

Fix accidental mixing.

IMPORTANT:

Historical values may remain if properly labeled and scientifically useful.

Do not blindly replace every occurrence.

------------------------------------------------------------

## 7. NUMERICAL CONSISTENCY CHECK

Verify that all current manuscript tables/text use the authoritative values from Stage 38.

### Los-loop multiseed

NoSpatial:
5.234 ± 0.090

MultiGSL:
4.794 ± 0.102

Mix:
4.452 ± 0.143

### Sparse controls

RandTop30:
6.096 ± 0.107

CorrTop30:
5.389 ± 0.088

Union DAGMA graph:
5.928

### Canonical GSL

Physical:

7.772 ± 0.140
8.118 ± 0.189
8.456 ± 0.047
8.554 ± 0.169

GSL:

5.792 ± 0.196
6.328 ± 0.166
6.669 ± 0.080
6.999 ± 0.084

### SZ

PH1:
NoSpatial 4.1192 ± 0.0069
MultiGSL 4.1302 ± 0.0152
Mix 4.1091 ± 0.0052

PH2:
NoSpatial 4.1623 ± 0.0055
MultiGSL 4.1600 ± 0.0036
Mix 4.1515 ± 0.0026

PH3:
NoSpatial 4.1884 ± 0.0011
MultiGSL 4.1988 ± 0.0121
Mix 4.1804 ± 0.0052

PH4:
NoSpatial 4.2196 ± 0.0013
MultiGSL 4.2273 ± 0.0092
Mix 4.2159 ± 0.0073

Check all derived percentages independently.

Do not introduce rounding inconsistencies.

------------------------------------------------------------

## 8. FILES TO MODIFY / CREATE

Before editing, identify the actual manuscript structure.

Likely files:

- paper/sn-article.tex
- paper/sections/*.tex

Modify only the files necessary for the manuscript revision.

Create/update a reviewer-response file, preferably:

paper/Response-to-Reviewers.md

unless an existing response file already exists, in which case use its established location and format.

Also create:

doc/STAGE39_MANUSCRIPT_REVISION_REPORT.md

This report must document:

1. files changed;
2. sections changed;
3. tables changed;
4. reviewer comments addressed;
5. historical results retained;
6. numerical consistency checks;
7. any claims weakened/tempered;
8. any remaining limitations;
9. whether compilation was attempted;
10. compilation result, if attempted;
11. git diff summary;
12. final GO/NO-GO status.

------------------------------------------------------------

## 9. COMPILATION / LATEX CHECK

After edits:

- inspect the LaTeX structure;
- run a compilation check if the environment supports it;
- fix LaTeX errors caused by your edits;
- do not make unrelated formatting changes.

If compilation is not possible, explicitly report why.

Check:

- table labels;
- references;
- citations;
- undefined references;
- duplicate labels;
- malformed math;
- accidental Unicode/encoding problems;
- overly long table cells where practical.

Do not modify the scientific content merely to make compilation easier.

------------------------------------------------------------

## 10. GIT SAFETY

Before editing:

- inspect git status;
- inspect current branch;
- do not discard unrelated user changes.

After editing:

- inspect git diff;
- verify that historical archives were not modified;
- verify that experiment result files were not altered;
- verify that only intended manuscript/response/report files changed.

Do NOT commit unless explicitly instructed.

------------------------------------------------------------

## 11. FINAL QUALITY CRITERIA

Stage 39 is successful only if:

1. Every substantive reviewer concern addressed by Stage 38 has a corresponding manuscript change and response-letter entry.

2. No unsupported scientific claim was introduced.

3. Historical and canonical protocols are clearly separated.

4. Contemporaneous vs lagged DAG interpretation is correct everywhere.

5. Sparsity is not presented as the explanation for the gain.

6. SZ dataset dependence is honestly reported.

7. No significance claim or p-value is introduced.

8. Limitations are explicitly stated.

9. Abstract numbers are consistent.

10. All numerical tables are internally consistent.

11. Historical artifacts remain untouched.

12. The manuscript compiles, or any compilation limitation is explicitly reported.

13. The reviewer response is submission-quality.

14. A final Stage 39 report documents all changes.

============================================================
12. FINAL OUTPUT

At the end, print a concise final summary containing:

- Manuscript revision: COMPLETE / INCOMPLETE
- Reviewer response: COMPLETE / INCOMPLETE
- Numerical consistency: PASS / FAIL
- Historical artifact integrity: PASS / FAIL
- LaTeX compilation: PASS / FAIL / NOT AVAILABLE
- Remaining scientific issues
- Remaining optional experiments
- Recommended next stage
- Commit & push

Recommended next stage if all criteria pass:

"Stage 40 — Final Manuscript & Reviewer-Response Quality Audit"

Do not run new experiments in Stage 39.
