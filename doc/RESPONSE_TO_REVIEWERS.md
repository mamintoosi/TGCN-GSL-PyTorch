# Response to Reviewers

**Paper:** "Graph Structure Learning for Traffic Prediction"
**Journal:** International Journal of Data Science and Analytics
**Revision:** Major Revision

---

## General Response

We sincerely thank both reviewers for their thorough and constructive feedback. The major revision has led to substantial restructuring of the manuscript, including a new central methodological contribution (multi-lag DAGMA + GatedMultiGraphTGCN), new experimental validation, and removal of all unsupported causal claims.

**Key changes in this revision:**

1. New central contribution: Multi-lag DAGMA + GatedMultiGraphTGCN
2. All causal claims removed (replaced with "temporal functional dependency")
3. 5-seed validation with mean ± std (addresses reviewer 1 comment #6)
4. Parameter-matched control experiment (new)
5. Lag ablation study (new)
6. 7 new publication-quality figures
7. Dataset dependence honestly reported
8. Limitations section added
9. Background section compressed
10. Convergence figures moved to appendix

---

## Reviewer 1

### Weakness 1: Bibliometric analysis is underused in the main text

**Response:** We now use one concise bibliometric observation in the Introduction to motivate the problem. The detailed bibliometric methodology has been moved to Appendix B.

**Location:** Introduction (new text), Appendix B.

### Weakness 2: GCN/T-GCN background is too long

**Response:** We have substantially compressed the background section (Section 2), removing redundant exposition while preserving all technically necessary equations.

**Location:** Section 2 (compressed from ~140 lines to ~30 lines).

### Weakness 3: A → W notation switch is insufficiently explained

**Response:** We now explicitly explain the notation when W is first introduced in Section 3.2.

**Location:** Section 3.2, first paragraph.

### Weakness 4: Temporal interpretation of DAG was asserted but not demonstrated

**Response:** This is now a major strength of the revision. We introduce an explicit multi-lag DAGMA formulation:

Z = [x(t−3), x(t−2), x(t−1), x(t)]

and extract lag-specific blocks from the DAGMA weight matrix. This provides direct, mathematically explicit evidence for lag-specific temporal functional dependencies — exactly what the reviewer requested.

**Location:** Section 3.3 (new), Tables 2, 7, Figures 3, 5, 7.

**New experiments:** Stages 25D, 26, 26 Validation provided the evidence. No new experiments were needed for the revision itself.

### Weakness 5: No density/degree analysis; sparsity/oversmoothing discussion

**Response:** We now include:
- Table 1: Oversmoothing evidence showing physical graph (2833 edges, RMSE 7.658) vs NoGraph (207, RMSE 5.143) vs GatedMulti (30, RMSE 4.458)
- Figure 1: Physical vs DAGMA graph visualization with degree distribution
- Figure 6: Threshold sensitivity showing RMSE vs edge count

**Location:** Section 5.1, Figures 1, 6.

### Weakness 6: No multiple seeds / variance / significance testing

**Response:** We now report 5-seed validation (seeds 42–46) with mean ± standard deviation.

Table 3 shows:
- NoGraph: 5.234 ± 0.090
- MultiGraph_fixed: 4.794 ± 0.102
- GatedMulti: 4.452 ± 0.143

GatedMulti beats NoGraph in 5/5 seeds.

**Location:** Section 5.3, Table 3, Figure 3.

### Weakness 7: Results only go to prediction horizon 4

**Response:** We report PH=1–4 results in Table 6. We acknowledge that longer horizons remain future work.

**Location:** Section 5.6, Table 6.

### Weakness 8: Section 5 is repetitive

**Response:** We have completely restructured the Results section with clear subsections: oversmoothing evidence, multi-lag structure, multi-seed validation, parameter control, lag ablation, and dataset dependence.

**Location:** Section 5 (restructured).

### Weakness 9: No explicit limitations section

**Response:** We add a new Section 7 (Limitations) with 8 explicit limitations, including dataset dependence, DAGMA scalability, threshold sensitivity, and the fact that no causal identification was performed.

**Location:** Section 7 (new).

### Weakness 10: Metric definitions are unnecessarily long

**Response:** We compress the metric definitions to a single equation.

**Location:** Section 4.4.

### Weakness 11: Figures 5–8 contain dense convergence plots

**Response:** All convergence figures have been moved to Appendix C. The main text now contains only the 7 new figures described above.

**Location:** Appendix C.

### Weakness 12: Citation style inconsistency

**Response:** We standardize citation style according to the journal's requirements.

**Location:** Throughout manuscript.

### Question 1: Side-by-side visualization of physical vs learned graph

**Response:** Figure 1 now shows physical adjacency heatmap, multi-lag DAGMA union heatmap, and degree distribution comparison.

**Location:** Figure 1.

### Question 2: Predicted vs actual time series plots

**Response:** This requires saving model checkpoints during training, which was not done in the current experiments. We note this as a limitation and plan to include these in future work. The code modification is straightforward (~10 lines) but requires retraining.

**Location:** Section 7 (Limitations), noted as future work.

### Question 3: Concrete plan for time-varying graph updates

**Response:** We describe sliding-window DAGMA as a future research direction in Section 7 (Limitations) and Section 8 (Conclusion).

### Question 4: Direct evidence for temporal/lagged DAG construction

**Response:** This is now the central methodological contribution. Section 3.3 presents the explicit multi-lag formulation, and Section 5.2 demonstrates that different lags produce structurally distinct graphs (low Jaccard overlap, Figure 7).

---

## Reviewer 2

### Comment 1: Abstract RMSE percentage appears inconsistent

**Response:** We have rewritten the abstract entirely with corrected numbers. The new abstract reports 14.9% improvement (mean over 5 seeds) rather than the previous inconsistent percentages.

**Location:** Abstract (rewritten).

### Comment 2: Claims about hidden causal structure are insufficiently supported

**Response:** All causal claims have been removed from the manuscript. We now use "temporal functional dependency" throughout. The only mention of causality is in Section 7 (Limitations), where we explicitly state that causal interpretation is not established.

**Location:** Throughout manuscript.

### Comment 3: Static learned graph versus adaptive/changing graph wording is inconsistent

**Response:** We now clearly distinguish:
- Static learned graph (DAGMA output, fixed after training)
- Multiple lag-specific static graphs (multi-lag DAGMA)
- Adaptive graph gating (GatedMultiGraphTGCN, changes per timestep)

**Location:** Section 3 (clearly defined), Section 7 (Limitations).

### Comment 4: cGSL definition should be introduced before it is used

**Response:** cGSL results are now moved entirely to Appendix A. The main text focuses on the new multi-lag DAGMA + GatedMulti approach.

**Location:** Appendix A.

### Comment 5: Convergence figures are difficult to read

**Response:** All convergence figures are moved to Appendix C. The main text contains only the new publication-quality figures (Figures 1–7).

**Location:** Appendix C, new Figures 1–7.

### Comment 6: Typo "avergae"

**Response:** Fixed.

---

## Summary of Changes

| Change | Status |
|--------|--------|
| Abstract rewritten | ✅ |
| All causal claims removed | ✅ |
| Background compressed | ✅ |
| Notation explained | ✅ |
| Multi-lag DAGMA formulation added | ✅ |
| GatedMultiGraphTGCN described | ✅ |
| 5-seed validation table | ✅ |
| Parameter-matched control | ✅ |
| Lag ablation | ✅ |
| Oversmoothing evidence | ✅ |
| Dataset dependence reported | ✅ |
| Limitations section added | ✅ |
| 7 new figures generated | ✅ |
| Convergence → appendix | ✅ |
| Old GSL/cGSL → appendix | ✅ |
| Citation style standardized | ✅ |
| "avergae" typo fixed | ✅ |
| Predicted vs actual plots | ⏳ Requires retraining (noted in Limitations) |

