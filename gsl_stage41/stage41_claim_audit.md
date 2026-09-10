# Stage 41 — Claim Audit (Claims to Avoid)

**Purpose:** itemized guard-rails for manuscript language, derived strictly from the Stage 40 results and the stored artifacts. Each entry: claim → verdict → what the data actually support → safe replacement language.

| # | Claim | Verdict | Evidence | Safe replacement |
|---|---|---|---|---|
| 1 | “DAGMA discovers the causal structure of traffic” | **Do not make** | DAGMA-linear assumes linear SEM + Gaussian noise; no interventional/validation evidence in the repo. Stage 38 already mandated tempering causal language | “a sparse statistical dependency structure learned from training data” |
| 2 | “The learned graph is temporal/causal-in-time” (contemporaneous graph) | **Do not make** | `T-GCN-GSL` graphs are fitted on contemporaneous snapshots `train_norm[0::PH]` (Stage 33 provenance); no lag information | “a contemporaneous dependency graph fitted per prediction horizon” |
| 3 | “Multi-lag graphs encode lag-specific temporal dependencies” | **Use with care** | Construction (stacked-lag blocks `Z=[x(t−L)..x(t)]`) supports the *statistical* reading, and per-lag consumption demonstrably matters (Los-loop: T-GCN-MultiGSL 4.01–4.94 RMSE better than the union-graph GCN); but the lag semantics were never directly validated (no within-Stage-40 lag ablation) | “graphs constructed from explicit lag blocks; consistent with, but not validated as, lag-specific dependencies” |
| 4 | “GSL improves forecasting” (universal) | **Do not make** | `T-GCN-GSL` is 7.1–11.6% (Los) / 3.4–3.9% (SZ) *worse* than `T-GCN-NoSpatial` at all 8 cells; SZ learned-graph variants are within 0.34% of NoSpatial | “multi-lag learned graphs improve Los-loop forecasting by up to 14.5% over the no-graph baseline; effects on SZ-Taxi are negligible” |
| 5 | “Multi-lag modeling is consistently useful” | **Do not make** | SZ-Taxi null (MultiGSL [1, 3, 0, 3] wins/5, Mix [4, 4, 5, 4] wins/5, |Δ| ≤ 0.34%); Los-loop strong (5/5 seeds, 9.2–14.5%) | “consistently useful on Los-loop; dataset-dependent benefit” |
| 6 | “Sparsity explains the gains” | **Do not make** | Matched 30-edge controls (Los PH1): RandTop30 6.05 ± 0.11, CorrTop30 5.39 ± 0.08 — both worse than NoSpatial 5.25 ± 0.03; DAGMA placement 4.49 (Mix) | “sparsity alone does not explain the Los-loop gains; the learned edge placement does” (scope: Los-loop, PH1) |
| 7 | “Results are statistically significant” (blanket) | **Do not make** | n = 5; exact Wilcoxon floor p = 0.0625. Only large-gap comparisons (vs physical graph; Mix vs baselines on Los-loop) have paired-t p < 0.01 | “consistent across all five seeds (5/5 wins); formal significance testing is limited by n = 5” |
| 8 | “The method adapts to changing traffic patterns” | **Do not make** | All graphs are static, fitted once from the training split (Stage 26/33 provenance); GRU models dynamics on the fixed structure | “the graph is fixed after training; the recurrent backbone models temporal dynamics over that fixed structure” |
| 9 | “More learned edges improve results” | **Do not make** | GCN-MultiGSL (28-edge union) is the worst Los-loop method (9.78–10.27); T-GCN with 2,833 physical edges is worst in its family | “edge placement and consumption pattern, not edge count, determine performance” |
| 10 | “cGSL symmetrization improves/changes behavior meaningfully” | **Do not make** | max |Δ| = 1.13% of NoSpatial RMSE vs GSL at all 8 cells, direction inconsistent | “symmetrization is immaterial (≤ 1.13% RMSE)” |
| 11 | “T-GCN-MultiGSL-Weighted shows the value of adaptive weighting” | **Do not make** | Weighted ≈ MultiGSL within 0.014 RMSE at every cell (learned global weights collapse toward a fixed mix) | “global learned weighting does not improve on the fixed assignment” (supplementary) |
| 12 | “Improvements generalize to longer horizons” | **Do not make** | PH ≤ 4 only (20 min at 5-min resolution) | “results cover horizons of 5–20 minutes; longer horizons were not evaluated” |

## Statistical-power notes for the manuscript

* 5 seeds → exact Wilcoxon signed-rank two-sided minimum p = 2/2⁵ = 0.0625; paired-t can reach small p only when the per-seed differences are nearly uniformly signed and large relative to their SD (which happens for comparisons against the physical-graph baseline and, on Los-loop, for Mix vs the NoSpatial/MultiGSL baselines).
* Recommended wording: report mean ± std (sample SD, ddof = 1), wins out of 5 seeds, and paired-t p-values with an explicit n = 5 caveat; avoid the word “significant” without qualification.

*Stage 41 constraints honored: no new training, no new DAGMA fitting, no modification of experimental code. All statistics computed from the 480 stored Stage 40 result JSONs and already-stored graph artifacts.*
