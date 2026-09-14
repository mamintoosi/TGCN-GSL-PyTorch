Start **Stage 47 (Results)**.

Use the finalized manuscript structure and the verified methodology from **Stage 46**. Before writing, inspect the Stage 40–46 artifacts and the canonical experimental results, especially the exact mean±std RMSE values, improvement percentages, graph statistics, and statistical comparisons.

Write only:

`paper/gsl_stage47/sections/results.tex`

and

`paper/gsl_stage47/stage47_writing_report.md`

Do not modify any other manuscript or source files.

Use the finalized Results structure:

* 5.1 The physical graph is not a sound default
* 5.2 A single learned graph does not close the gap
* 5.3 Sparsity and capacity controls
* 5.4 Multi-lag graphs: consumption matters more than the graph
* 5.5 Gated mixing on Los-loop: the positive result
* 5.6 Boundaries: temporal resolution and dataset dependence

Use the canonical Stage 40 results (5 seeds, 42–46) as the primary quantitative evidence.

Important:

* Do not discuss manuscript history or “previous/original” results.
* Do not claim that GSL generally outperforms NoSpatial; the canonical results show otherwise.
* Do not claim causal relationships from DAGMA.
* Do not call the learned graph “adaptive” or claim that it changes over time.
* Clearly distinguish the **learned graph structure** from the **mechanism used to consume it**.
* Treat the Los-loop T-GCN-MultiGSL-Mix result as the main positive result, and SZ-Taxi as an important boundary/null case.
* Report exact numerical values from the verified artifacts; do not recompute or invent numbers.
* Use relative RMSE reduction carefully and consistently.
* For statistical claims, remember that n=5 is small; do not overstate significance.
* Do not claim that sparsity has been fully controlled in all datasets/horizons; the matched sparsity experiment is Los-loop PH1 only.
* The 15-minute Los-loop experiment should be presented only as a temporal-resolution / longer-horizon variant, not as direct evidence for PH5–8 at 5-minute resolution.
* Keep Results primarily descriptive. Reserve deeper interpretation and causal/mechanistic explanations for Section 6 (Discussion).

Use the manuscript's established terminology and notation from Stage 46.

After writing, perform a consistency check against the canonical Stage 40/41/42 artifacts and report:

1. files created/modified,
2. tables/figures referenced,
3. every numerical result used,
4. any remaining inconsistency or missing evidence,
5. a final READY/NOT READY verdict.
