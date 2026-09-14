# Point-by-Point Response to Reviewers

**Manuscript:** *Graph Structure Learning for Traffic Prediction*
**Journal:** International Journal of Data Science and Analytics

Dear Editor and Reviewers,

We thank the editor and both reviewers for their careful reading of the manuscript and for constructive comments that materially improved the work. In the revision we (i) corrected an error in the abstract's reported numbers, (ii) added five-seed variance reporting (mean ± std, seeds 42–46) to all main results, (iii) ran matched-edge-budget sparse-graph controls that substantially address the sparsity confound raised by Reviewer 1, (iv) re-established the single-graph GSL baseline under a fully documented revised protocol, reproducing the original submission's learned graphs at the support level, and (v) rewrote the interpretation of the learned graphs to correctly distinguish contemporaneous from lagged constructions, and descriptive structural analysis from causal discovery.

All numbers quoted below were independently recomputed from the experiment artifacts during a dedicated audit pass before this revision. Sections and tables refer to the revised manuscript.

---

## Reviewer 1

### R1-Summary / R1-S4 (GSL vs. cGSL asymmetry)

We thank the reviewer for recognizing the GSL/cGSL asymmetry as a genuinely interesting empirical finding. During the revision we established, from the archived artifacts and the original data-loading code, that the original single-graph construction was **contemporaneous** (DAGMA applied to simultaneous per-sensor observations), so the asymmetry could not be explained by a temporal reading of that DAG. This analysis is now presented in Section 5.3 ("The single-graph GSL baseline revisited", Table `tab:gsl_baseline`) and Appendix A, and the temporal interpretation now rests on the explicitly lagged multi-lag formulation (Sections 3.3 and 5.2; see also our response to R1-W4 and R1-Q4 below).

### R1-W1 (Bibliometric analysis underused in the main text)

We thank the reviewer for this suggestion. We used one concise bibliometric observation in the Introduction (paragraph 2) to directly motivate the shift from heuristic graph construction to learned structure, and retained the full analysis in the appendix. We did not expand the bibliometric material further, preferring to keep the additional space for the new experimental evidence requested below.

### R1-W2 (GCN/T-GCN background too long)

We agree. Both subsections were condensed to short treatments with equation references, freeing space for the new analyses (variance reporting, matched-budget controls, and the single-graph GSL baseline subsection).

### R1-W3 (Unexplained A → W notation switch)

We have clarified this in the main text of Section 3.2 rather than a footnote: a convention sentence now appears immediately where W is introduced ("Throughout this paper, we follow the DAGMA convention where W[i,j] represents a directed dependency from variable i to variable j").

### R1-W4 (Temporal interpretation asserted rather than demonstrated)

We thank the reviewer for this comment, which was load-bearing for the revision. The reviewer's suspicion was correct: **the original single-graph construction learned a DAG over simultaneous per-road observations, not over lagged variable blocks.** During the revision we re-derived the original construction from the data-loading code and confirmed this by exactly reproducing the original slice-0 DAGMA fits from contemporaneous input: the revised protocol's first-offset fits reproduce the historical slice-0 graph support exactly at PH = 1, 2, and 4 (28/28 edges) and 26/28 at PH = 3, with shared-edge weights agreeing to within approximately 0.016 (Section 5.3; Appendix A). We have therefore removed the temporal interpretation from the single-graph DAG entirely.

The interpretation the reviewer found missing is now carried by an explicit construction rather than a post-hoc reading: the multi-lag formulation feeds DAGMA lag-stacked blocks Z = [x(t−L), …, x(t)] (828 variables for Los-loop), whose weight-matrix blocks are lag-specific graphs (Section 3.3). An ablation confirms that each lag carries complementary predictive information (~10% individually, 13.3% jointly, Table `tab:lag_ablation`). We did not add a further DAGMA ablation "feeding genuinely simultaneous vs. lagged data" as separate new experiments, because the revision contains precisely that contrast in its two constructions: the contemporaneous baseline (proven contemporaneous by exact reproduction of the historical fits) and the lagged multi-lag formulation, whose block structure is a direct consequence of the lagged input.

### R1-W5 (Sparser learned graphs; are gains attributable to reduced oversmoothing from sparsity?)

We thank the reviewer for identifying this confound; addressing it properly required new controlled experiments, which we have run and added to Section 5.1 (Table `tab:oversmoothing`, bottom block). We trained T-GCN on two sparse control graphs at exactly the same 30-edge budget as the learned graphs, over five seeds:

| Control (30 edges) | RMSE (mean ± std, 5 seeds) | vs. T-GCN-NoSpatial (5.234 ± 0.090) |
|---|---|---|
| Random sparse graph (RandTop30) | 6.096 ± 0.107 | 16.5% **worse** |
| Top-correlation graph (CorrTop30) | 5.389 ± 0.088 | no better |
| DAGMA lag-block edges (T-GCN-MultiGSL) | 4.794 ± 0.102 | 8.4% better |
| DAGMA lag-block edges (T-GCN-MultiGSL-Mix) | 4.452 ± 0.143 | 14.9% better |

So: a random sparse graph does **not** explain the gain (it is worse than no graph), and a correlation-based sparse graph does **not** explain the gain either. Two further controls sharpen the conclusion: the *same* DAGMA edges assembled into a single union adjacency with plain T-GCN yield 5.928 — worse than no graph — and a thresholded single-DAG graph (60 edges) also underperforms. We therefore now attribute the improvement to the combination of DAGMA-learned lag-specific edges and their per-lag use in the multi-graph architecture, and we state explicitly in the revised text that sparsity alone is insufficient. We agree with the reviewer that this does not amount to a mathematical proof; the matched controls substantially address the sparsity confound, but they do not close every conceivable baseline variant.

One control the reviewer suggested — a sparsification of the physical graph itself — was **not** run. We acknowledge this explicitly: the added controls answer the "is any 30-edge graph enough?" version of the question, but the "fairer baseline" version (thresholding or kNN-sparsifying the physical adjacency to 30 edges) remains untested, and we have not claimed otherwise.

### R1-W6 (No significance testing, variance, or multiple seeds)

We agree and have addressed this throughout. All main results now report mean ± std over five seeds (42–46): the Los-loop multi-seed table (Table `tab:multiseed`), the revised single-graph GSL baseline (Table `tab:gsl_baseline`, both Physical and T-GCN-GSL rows), and the SZ-Taxi table (Table `tab:sz_multiph`, three methods × four horizons). The seed-to-seed variance isolates training-related randomness; DAGMA itself is deterministic, which we verified in a dedicated determinism audit (zero initialization, no internal RNG; bit-identical outputs across repeated processes, agreement to approximately one ULP across thread counts with identical thresholded support). Regarding the std convention: standard deviations are population standard deviations over the five seeds. We did **not** perform significance testing (no p-values or hypothesis tests are reported anywhere); we report means, standard deviations, and paired win counts only, and where differences are within seed variability (SZ-Taxi PH4) we say so explicitly.

### R1-W7 (Results only up to horizon 4)

We were not able to add horizons PH5–PH8 in this revision: each additional horizon in the 5-minute setting requires a new DAGMA fit plus model trainings, and we chose not to extend the reported scope with experiments we could not fully validate within the revision. Instead we have (i) added an explicit scope limitation (Section 7, item 9) stating that results cover PH = 1–4, and (ii) pointed to the 15-minute-sampling experiment as the available longer-*wall-clock*-horizon evidence: at 15-minute resolution, PH = 4 corresponds to 60 minutes ahead, and the relative improvement *grows* with horizon (27.4% at PH = 1 vs. 14.9% at 5-minute sampling; Section 5.6). This is indirect evidence that graph-structure quality matters at least as much at longer horizons, but we agree it does not substitute for directly evaluated PH5–PH8, which we now state as future work.

### R1-W8 (Section 5 repetitive; not tied back to opening claims)

We restructured the results into distinct subsections (oversmoothing; single-graph baseline revisited; multi-seed validation and parameter control; lag ablation and multi-horizon; temporal resolution; dataset dependence) and the discussion now explicitly connects the findings back to the physical-proximity-versus-functional-dependency framing of Section 1 (Sections 6.3 and 6.4).

### R1-W9 (No limitations discussion)

We agree and added a dedicated Limitations section (Section 7) covering exactly the reviewer's three concerns plus additional ones:

- **Scalability:** measured DAGMA runtimes are now reported (Section 7 item 3 and Appendix C): approximately 16 minutes per horizon for the 207-variable single-graph fits on 4 CPU cores, 52–76 minutes for archived 156-variable fits, and approximately 4 hours for the 828-variable multi-lag fit.
- **Sensitivity to λ:** we state explicitly that no systematic λ sweep was performed; λ₁ was fixed (0.02 for the single-graph protocol, 0.01 for the multi-lag fits), sensitivity was not measured, and the reported graphs are one operating point rather than an optimized configuration. We are careful not to imply that sensitivity was studied.
- **Backbones:** we state that only the studied T-GCN/GCN-family backbones were evaluated and that broader backbone validation (e.g., ST-GCN, Graph WaveNet) is future work.
- Additional items: dataset dependence, fixed lag window, static learned graphs, no causal identification, and the PH = 1–4 horizon scope.

### R1-W10 (Metric definitions unnecessary)

We condensed the RMSE/MAE definitions to a single two-equation display in Section 4.5 and kept the space for the new qualitative material.

### R1-W11 (Dense convergence plots, Figures 5–8)

We agree. The per-epoch curve grids were removed from the main text; a compact convergence summary (two panels) now appears in Appendix C, and the main text carries the final-value comparisons with five-seed error reporting instead. (Reviewer 2's R2-5 raises the same point; both are addressed by the same change.)

### R1-W12 (Citation style: bare "[number]")

We standardized the flagged occurrences so that citations read naturally as "Author et al. [number]" where the sentence benefits from attribution (e.g., Background and Method sections). All bibliography entries and in-text citations were checked for consistency with the venue's numeric style.

### R1-Q1 (Side-by-side visualization of physical vs. learned graph)

We added exactly this: Figure `fig:graph_comparison` shows the physical adjacency matrix, the multi-lag DAGMA union graph, and the node-degree distributions side by side (physical: 2833 edges, mean degree 12.7; DAGMA union: 28 distinct off-diagonal edges, mean degree 0.14). The figure makes the "physical proximity ≠ functional dependency" claim concrete.

### R1-Q2 (Predicted vs. actual time series)

We added Figure `fig:pred_vs_actual`, one-step-ahead predictions for the three most variable Los-loop nodes over a 100-step test window, with per-node windowed correlations given in the text, complementing the aggregate RMSE/MAE tables.

### R1-Q3 (Time-varying graph extension: concrete plan and overhead)

Our concrete plan is sliding-window DAGMA: refit the graphs on a trailing window of training data every W steps and reuse the graphs between refits, with the multi-lag formulation unchanged. We can now state measured overhead rather than estimates: the 207-variable single-graph fits take approximately 16 minutes each (4 CPU cores), and the 828-variable multi-lag fit takes approximately 4 hours; forecasting-model retraining is minutes. A sliding-window scheme therefore multiplies the per-fit DAGMA cost approximately by the number of windows (e.g., daily refitting of the 5-minute data adds roughly 16 minutes per day for the single-graph variant, and correspondingly more for the multi-lag variant), which is modest relative to continuous training but not negligible for the multi-lag formulation at larger networks. This is stated in Appendix C and the Conclusion, and we keep the extension plan qualitative rather than claiming it has been implemented.

### R1-Q4 (Direct evidence: genuinely simultaneous vs. lagged data into DAGMA)

Please see our response to R1-W4. In short: the historical construction is now *demonstrated* to be contemporaneous (exact support-level reproduction of its slice-0 fits from simultaneous observations), and the lagged alternative is provided as an explicit construction — lag-stacked input blocks — whose per-lag graphs are directly measurable and predictive (lag ablation: ~10% per lag, 13.3% jointly). We agree with the reviewer that the original interpretation was inferred post hoc from the GSL/cGSL performance split; the revised manuscript no longer relies on that inference.

---

## Reviewer 2

### R2-1 (Abstract numbers look mixed up: 21.6% and 24.7%)

The reviewer is correct, and we thank them for catching it. The original abstract's "21.6% for GCN and 24.7% for T-GCN" misattributed values that both belong to the **GCN-cGSL** results on the original single-seed tables (multi-horizon means over PH = 1–4: 21.6% on SZ-taxi and 24.7% on Los-loop), while the T-GCN best average on Los-loop was 21.8%. The revised abstract no longer quotes these numbers. The abstract now reports the revised, five-seed headline results: 14.9% mean RMSE improvement over a no-graph baseline on Los-loop at 5-minute sampling, and 27.4% at 15-minute sampling. For continuity with the original claim, we additionally re-ran the original Physical-vs-GSL comparison under the revised protocol at five seeds: the GSL baseline improves over the physical graph by 21.7% mean across PH = 1–4 (25.5% at PH = 1), independently reproducing the magnitude of the original 21.8% single-seed claim (Section 5.3, Table `tab:gsl_baseline`). The appendix now labels the derivation of the historical percentages precisely (means over PH = 1–4, single seed), so the provenance of every number is unambiguous.

### R2-2 ("Explicit insights into hidden causal structure" unsupported)

We agree with the reviewer's reading and have removed or tempered the causal language throughout. The manuscript no longer claims insights into causal structure: the Introduction and Conclusion now describe the learned graphs as descriptive structural analyses of learned functional dependencies, and the limitations section states explicitly that the DAGMA formulation discovers temporal functional dependencies, not causal relationships, and that no causal validation is claimed. To support structural interpretation, the revision adds the reviewer-requested analysis: the physical-vs-learned visualization with degree statistics (Figure `fig:graph_comparison`) and graph statistics for the learned DAGs (28 edges per horizon, weight ranges, zero negative survivors at threshold). We did not add a comparison against known traffic corridors, and we do not claim one; the visualizations support structural interpretation, not causal proof.

### R2-3 (Static graph vs. "adapt to changing traffic patterns" contradiction)

The reviewer's diagnosis is exactly right, and their proposed reframing is what we adopted. The learned graphs are computed once from training data and remain fixed during inference; what varies over time is the model's use of them. Section 3.2's item 3 was rewritten to say that, unlike fixed proximity graphs, the learned structure captures real functional dependencies, and the GRU then models how those dependencies play out dynamically; Section 3.4 now states this distinction explicitly ("the learned graphs themselves are static: they are estimated once from training data and remain fixed during inference. What adapts over time is the model's *use* of them"). The Limitations section retains the honest statement that the graphs do not adapt to changing traffic conditions.

### R2-4 (cGSL defined in Section 5.3 but evaluated in Section 4)

Addressed by restructuring rather than relocation: in the revised manuscript, the main text presents the multi-lag framework, and the GSL/cGSL results — including the symmetrization formula A_cGSL = A_GSL + A_GSLᵀ — now appear together in the appendix, where the historical protocol is also fully described (Appendix A). The main text introduces them only through the "single-graph GSL baseline revisited" subsection (Section 5.3), after the method section has already defined the contemporaneous construction, so no result is evaluated before its definition.

### R2-5 (Convergence plots with 16 subplots hard to read)

We agree. The dense per-epoch grids were removed; a compact two-panel convergence summary now appears in Appendix C, and the main text reports final values with five-seed variance rather than full curve grids.

### R2-6 (Typo "avergae")

Fixed; the affected passage was rewritten during the revision and the typo no longer occurs anywhere in the manuscript.

---

## Additional changes made during revision (not requested, disclosed for transparency)

1. **Abstract:** now states that the original submission's central result was reproduced under the revised protocol (21.7% mean, five seeds) and that the sparse-control experiment was added; all headline numbers are five-seed values.
2. **Figure `fig:graph_comparison`:** the edge-count and degree annotations were corrected during a numerical audit (edge counts are now counted as edges rather than weight sums; self-loops are excluded consistently with the published graph definition).
3. **Reproducibility subsection** (Section 4.5): documents the DAGMA determinism audit, the seed policy, the threshold semantics (absolute-magnitude rule, matching the DAGMA library's own semantics; on the studied datasets no negative coefficient survived the threshold, so this clarification changed no reported numbers), and the historical reproduction levels.

We believe the manuscript is substantially strengthened by the reviewers' comments, and we hope the revision is now suitable for publication.

Sincerely,
The Authors
