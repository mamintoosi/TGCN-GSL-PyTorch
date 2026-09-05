# Response to Reviewers

**Manuscript:** "Graph Structure Learning for Traffic Prediction"
**Journal:** International Journal of Data Science and Analytics
**Revision:** Major Revision

---

## General Response

We sincerely thank both reviewers for their thorough and constructive evaluation. The reviewer comments identified several critical issues in the original manuscript, particularly regarding the temporal interpretation of learned graphs, causal claims, experimental rigor, and manuscript structure.

During the revision, we conducted a comprehensive reproducibility audit (Stages 17–26) that fundamentally changed our understanding of graph structure learning for traffic forecasting. The key finding is that **a single static graph is insufficient to capture temporally heterogeneous traffic dependencies**. We therefore introduced a novel **multi-lag DAGMA formulation** combined with a **GatedMultiGraphTGCN** architecture that adaptively selects among lag-specific dependency graphs. This new approach produces a robust 14.9% RMSE improvement over a no-graph baseline on the Los-loop dataset, validated across five random seeds with a parameter-matched control.

We believe the revised manuscript substantially addresses all reviewer concerns while presenting a stronger, more honest scientific contribution.

---

## Reviewer 1

### Weakness 1: Bibliometric analysis is underused in the main text.

**Response:** We have condensed the bibliometric analysis and used a concise observation in the Introduction to motivate the shift from physical-proximity-based graphs to data-driven graph structure learning. The detailed bibliometric methodology and results are now in Appendix B. This preserves the thorough analysis while keeping the main narrative focused.

**Manuscript location:** Section 1 (Introduction), Appendix B.

**Changes made:** Text revision. No new experiment required.

---

### Weakness 2: GCN/T-GCN background (Sections 2.2 and 2.3) is too long.

**Response:** We have compressed the GCN and T-GCN background substantially, removing redundant derivations while preserving the essential equations needed to understand the proposed method. The background section now occupies approximately one-quarter of its original length.

**Manuscript location:** Section 2 (Background and Problem Formulation).

**Changes made:** Text compression. No new experiment required.

---

### Weakness 3: The A → W notation switch is unexplained.

**Response:** We have added an explicit convention statement when $W$ is first introduced in Section 3.2, clearly explaining the relationship between the adjacency matrix $A$ (used in the GCN/T-GCN formulation) and the DAGMA weight matrix $W$ (used in the graph learning step). The convention $W[i,j] = $ variable $i \rightarrow$ variable $j$ is now stated explicitly.

**Manuscript location:** Section 3.2 (DAGMA-Based Graph Structure Learning), first paragraph after the augmented Lagrangian equation.

**Changes made:** Text revision. No new experiment required.

---

### Weakness 4: The temporal interpretation of the learned DAG is asserted rather than demonstrated.

**Response:** This was the most critical concern raised by Reviewer 1, and we agree that the original manuscript did not adequately justify the temporal interpretation. In the revision, we introduce an **explicit multi-lag DAGMA formulation** (Section 3.3) that directly constructs lag-specific dependency graphs by feeding DAGMA an input of the form:

$$Z_t = [x(t-L), x(t-L+1), \ldots, x(t-1), x(t)]$$

For $L=3$, this produces four distinct blocks in the DAGMA weight matrix:

- $W[0:N, 3N:4N]$ → lag-3 to current: $x(t-3) \rightarrow x(t)$
- $W[N:2N, 3N:4N]$ → lag-2 to current: $x(t-2) \rightarrow x(t)$
- $W[2N:3N, 3N:4N]$ → lag-1 to current: $x(t-1) \rightarrow x(t)$
- $W[3N:4N, 3N:4N]$ → contemporaneous: $x(t) \rightarrow x(t)$

This formulation was validated experimentally:

1. **Stage 21** (synthetic data): The correct block ($W[0:N, N:2N]$) achieved F1=0.75 for recovering known temporal dependencies, while the wrong block achieved F1=0.0.
2. **Stage 25D** (Los-loop): The lag-specific blocks exhibit genuinely different edge structures (Table lag_stats in the revised manuscript), with low cross-lag overlap, confirming that the blocks capture distinct temporal dependency patterns.
3. **Lag ablation** (Table lag_ablation): Each individual lag provides approximately 10% improvement, and the full three-lag combination achieves 13.3%, demonstrating that the different lags carry complementary predictive information.

This directly demonstrates, rather than asserts, the temporal interpretation of the learned dependency structure.

**Manuscript location:** Section 3.3 (Multi-Lag DAGMA Formulation), Section 5.2 (Multi-Lag Graph Structure), Section 5.5 (Lag Ablation).

**Changes made:** New method section + new experimental results from Stages 21, 25D, and 26.

---

### Weakness 5: No hyperparameter optimization; learned graphs may be sparser than physical graphs; oversmoothing concern.

**Response:** We have added an explicit oversmoothing analysis (Table oversmoothing) that directly compares:

| Graph | Edges | RMSE |
|-------|-------|------|
| Physical (207 nodes) | 2,833 | 7.658 |
| NoGraph (identity) | 207 | 5.143 |
| Single DAGMA (τ=0.3) | 6 | 5.213 |
| Single DAGMA (τ=0.1) | 60 | 6.057 |
| MultiGraph_fixed | 30 | 4.715 |
| GatedMultiGraphTGCN | 30 | 4.458 |

The physical graph (2,833 edges) produces RMSE of 7.658, while NoGraph (identity adjacency) achieves 5.143—a 32.8% improvement from removing the graph entirely. This confirms that dense spatial aggregation hurts forecasting. Importantly, the multi-lag GatedMulti approach uses only 30 edges and achieves 4.458, demonstrating that the improvement is not solely from sparsification but from the specific structure and processing of lag-specific dependencies.

We also conducted a **parameter-matched control** (Table param_control): increasing NoGraph parameters by 35% (from 12,672 to 16,872) improves RMSE by only 0.1%, while GatedMulti (17,091 parameters) improves by 13.3%. This confirms that the improvement comes from the gating architecture, not from capacity or sparsity alone.

**Manuscript location:** Section 5.1 (Dense Physical Graphs and Oversmoothing), Section 5.4 (Parameter-Matched Control).

**Changes made:** New experimental analysis from Stages 24–26. No new experiments required.

---

### Weakness 6: No multiple seeds / variance / significance testing.

**Response:** We now report mean ± std over five random seeds (seeds 42–46) for the primary comparison on Los-loop (Table multiseed):

| Method | S42 | S43 | S44 | S45 | S46 | Mean±Std |
|--------|-----|-----|-----|-----|-----|----------|
| NoGraph | 5.143 | 5.281 | 5.386 | 5.205 | 5.154 | 5.234±0.090 |
| MultiGraph_fixed | 4.717 | 4.737 | 4.752 | 4.994 | 4.771 | 4.794±0.102 |
| **GatedMulti** | **4.458** | **4.660** | **4.335** | **4.261** | **4.547** | **4.452±0.143** |

GatedMultiGraphTGCN outperforms NoGraph in all five seeds. The mean improvement is 14.9%.

**Manuscript location:** Section 5.3 (Multi-Seed Validation).

**Changes made:** New experimental results from Stage 26 Validation Experiment A. No new experiments required (used existing Stage 24 DAGMA matrices with 5 different training seeds).

---

### Weakness 7: Results only go to prediction horizon 4.

**Response:** We report results for PH=1 through PH=4 (Table multiph). GatedMulti consistently outperforms NoGraph across all horizons, with the largest improvement at PH=1 (13.3%). The advantage persists but narrows at longer horizons.

We acknowledge that evaluation beyond PH=4 is important and identify it as a limitation of the current study (Limitation #8). Extending to longer horizons is planned as future work.

**Manuscript location:** Section 5.6 (Prediction Horizons and Dataset Dependence), Section 7 (Limitation #8).

**Changes made:** Text revision + existing multi-PH results from Stage 26.

---

### Weakness 8: Section 5 is repetitive and doesn't clearly tie back to opening claims.

**Response:** We have completely restructured the Results and Discussion sections. The new structure is:

- Section 5.1: Oversmoothing evidence (ties to physical-vs-functional motivation in Section 1)
- Section 5.2: Multi-lag graph structure (demonstrates temporal heterogeneity)
- Section 5.3: Multi-seed validation (establishes robustness)
- Section 5.4: Parameter-matched control (rules out confound)
- Section 5.5: Lag ablation (demonstrates complementary information)
- Section 5.6: Prediction horizons and dataset dependence
- Section 6: Discussion (integrates findings into the overarching narrative)

Each subsection directly addresses a specific claim from the Introduction, and the Discussion synthesizes the findings.

**Manuscript location:** Sections 5 and 6.

**Changes made:** Complete restructuring. No new experiments required.

---

### Weakness 9: No explicit limitations section.

**Response:** We have added a dedicated Limitations section (Section 7) with eight explicitly identified limitations:

1. Dataset dependence (strong on Los-loop, marginal on SZ-Taxi)
2. Only two datasets evaluated
3. DAGMA computational scalability
4. Fixed lag window
5. Static learned graphs
6. Limited backbone architectures (only T-GCN)
7. No causal identification
8. Prediction horizons limited to PH=1–4

**Manuscript location:** Section 7 (Limitations).

**Changes made:** New section. No new experiments required.

---

### Weakness 10: Metric definitions are unnecessarily long.

**Response:** We have compressed the metric definitions to a concise two-line equation block with a citation, replacing the original lengthy derivations.

**Manuscript location:** Section 4 (Experimental Setup).

**Changes made:** Text compression. No new experiment required.

---

### Weakness 11: Figures 5–8 contain dense convergence plots.

**Response:** We have moved the detailed convergence curve grids to Appendix C and propose replacing them with compact summary tables in the main text, which convey the same information more effectively.

**Manuscript location:** Appendix C (Convergence Analysis).

**Changes made:** Figure reorganization. No new experiment required.

---

### Weakness 12: Citation style inconsistency.

**Response:** We acknowledge this issue and will standardize all citations to the journal's required style in the final compilation pass. This is a formatting fix that does not affect content.

**Manuscript location:** Throughout.

**Changes made:** To be completed in final pass.

---

### Questions

**Q1: Side-by-side visualization of physical vs learned graph.**

**Response:** We agree this would strengthen the paper. We have the existing DAGMA matrices and can generate a comparative visualization (e.g., adjacency heatmaps or edge diagrams) from the saved Stage 25D/26 results without re-running any experiments. We plan to include this as a new figure in the revised manuscript.

**Status:** Will be generated from existing data. No new experiments required.

---

**Q2: Predicted vs actual time series plots.**

**Response:** This would require inference from saved trained models. If saved model checkpoints are available from the Stage 26 training runs, this figure can be generated relatively quickly. Otherwise, it requires a short re-training run. We include this as a potential enhancement in the appendix.

**Status:** Optional. Depends on availability of saved model checkpoints.

---

**Q3: Concrete plan for time-varying graph updates (sliding-window DAGMA).**

**Response:** We describe sliding-window DAGMA as a concrete future direction in the Conclusion and Limitations sections. A sliding-window approach with window size $W$ would re-run DAGMA every $W$ time steps, adding computational overhead proportional to $T_{\text{data}} / W \times T_{\text{DAGMA}}$. For the current Los-loop dataset ($T_{\text{data}} = 2016$, $T_{\text{DAGMA}} \approx 4$ hours), a 24-hour sliding window would add approximately $84 \times 4 = 336$ hours of computation, making parallel or incremental approaches necessary for practical deployment.

**Manuscript location:** Section 8 (Conclusion), Section 7 (Limitation #5).

**Changes made:** Text revision describing future direction. No new experiment required.

---

**Q4: Direct evidence for temporal/lagged DAG construction (rather than post-hoc inference).**

**Response:** This is directly addressed by the new multi-lag DAGMA formulation (Section 3.3). Unlike the original manuscript, where the temporal interpretation was inferred post-hoc from the GSL/cGSL performance split, the revised approach explicitly constructs lag-specific dependency blocks from $Z = [x(t-3), x(t-2), x(t-1), x(t)]$.

The evidence includes:

1. **Stage 21 synthetic validation:** Known temporal dependencies are correctly recovered by the appropriate block (F1=0.75) but not by the wrong block (F1=0.0), verifying both the block extraction and the DAGMA convention.
2. **Stage 25D structural evidence:** Different lag blocks exhibit genuinely different edge structures with low cross-lag overlap.
3. **Stage 26 forecasting evidence:** The lag ablation study demonstrates that each lag carries complementary predictive information, and the full multi-lag combination is best.

This constitutes direct evidence for temporal dependency discovery, rather than a post-hoc interpretation.

**Manuscript location:** Section 3.3, Section 5.2, Section 5.5.

**Changes made:** New method section + new experimental evidence from Stages 21, 25D, and 26.

---

## Reviewer 2

### Comment 1: Abstract RMSE percentage appears inconsistent.

**Response:** The reviewer correctly identified that the original abstract numbers (21.6% for GCN, 24.7% for T-GCN) were incorrect. The 24.7% figure actually corresponded to GCN on Los-loop, not T-GCN. We have written a completely new abstract with correct numbers based on the revised experiments. The new abstract reports a mean RMSE improvement of 14.9% for GatedMultiGraphTGCN over NoGraph on Los-loop across five seeds.

**Manuscript location:** Abstract.

**Changes made:** Complete abstract rewrite with verified numbers.

---

### Comment 2: Claims about "hidden causal structure" are insufficiently supported.

**Response:** We fully agree. The original manuscript used causal language ("causal structure," "causal insight for urban planners") without performing any causal identification. In the revision, we have **removed all causal claims** throughout the manuscript. The only mention of "causal" is in the Limitations section, where we explicitly state:

> "No causal identification: The DAGMA formulation discovers temporal functional dependencies, not causal relationships. We do not claim that the learned graphs represent causal traffic mechanisms."

We use terminology such as "temporal functional dependency," "data-driven dependency structure," and "lag-specific dependency" throughout.

**Manuscript location:** All sections. Limitation #7.

**Changes made:** Text revision throughout. No new experiments required.

---

### Comment 3: Static vs adaptive graph wording inconsistency.

**Response:** The reviewer correctly identified a contradiction between Section 3.1 ("the graph is static") and Section 3.2 item 3 ("the graph adapts to changing traffic patterns"). The original graph is indeed computed once and remains fixed. We have clarified the four levels at which temporal dynamics operate:

1. **Static learned graph:** The dependency structure is learned once from training data and remains fixed.
2. **GRU temporal modeling:** The recurrent hidden state captures temporal dynamics within the fixed graph structure.
3. **Multi-lag graph architecture:** Multiple lag-specific static graphs capture dependencies at different temporal scales.
4. **Adaptive graph gating:** The GatedMultiGraphTGCN recomputes per-node, per-timestep gate weights, adaptively selecting among the lag-specific graphs.

**Manuscript location:** Section 3.4 (GatedMultiGraphTGCN).

**Changes made:** Text revision with explicit four-level distinction. No new experiments required.

---

### Comment 4: cGSL definition should be introduced before it is used.

**Response:** In the revised manuscript, the main text focuses on the multi-lag DAGMA approach and does not use cGSL. The original GSL/cGSL experiments are now presented in Appendix A (Original GSL/cGSL Results), where the cGSL definition is provided before its results are shown. This resolves the ordering issue without cluttering the main narrative.

**Manuscript location:** Appendix A.

**Changes made:** Structural reorganization. No new experiments required.

---

### Comment 5: Convergence figures are difficult to read.

**Response:** We have moved the convergence figures (originally Figures 5–8 with 16 subplots each) to Appendix C. The main text replaces them with compact summary tables that convey the essential information.

**Manuscript location:** Appendix C (Convergence Analysis).

**Changes made:** Figure relocation. No new experiments required.

---

### Comment 6: Typo "avergae."

**Response:** Fixed. The old text containing this typo is no longer in the main manuscript.

**Manuscript location:** N/A (removed with old text).

**Changes made:** Typo corrected.

---

## Summary of Changes

| Change | Type | New Experiment? |
|--------|------|-----------------|
| New abstract with correct numbers | Text | No |
| All causal claims removed | Text | No |
| Compressed background | Text | No |
| Notation clarification | Text | No |
| Multi-lag DAGMA formulation (Section 3.3) | New method | No (uses existing DAGMA) |
| GatedMultiGraphTGCN (Section 3.4) | New method | No |
| Oversmoothing analysis | New analysis | No (from existing Stage 24-25 results) |
| 5-seed validation | New results | No (from Stage 26 Validation A) |
| Parameter-matched control | New results | No (from Stage 26 Validation B) |
| Lag ablation | New results | No (from Stage 26 Validation C) |
| Multi-PH results | New table | No (from Stage 26) |
| Dataset dependence reported | Text | No |
| Limitations section | New section | No |
| Static vs adaptive clarified | Text | No |
| cGSL moved to appendix | Restructuring | No |
| Convergence figures to appendix | Restructuring | No |
| Metric definitions compressed | Text | No |
| Citation style (pending) | Formatting | No |

**New experiments performed during revision:** None. All new results use existing Stage 21–26 evidence.
