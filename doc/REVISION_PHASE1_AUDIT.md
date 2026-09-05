# Revision Phase 1 — Forensic Audit

**Date:** 2026-09-05  
**Manuscript:** paper/sn-article.tex (1166 lines)  
**Status:** Read-only audit — no modifications made

---

## 1. Current Manuscript Structure

| Section | Lines | Content |
|---------|-------|---------|
| Abstract | 121-131 | Claims 21.6% and 24.7% RMSE reduction |
| 1. Introduction | 151-187 | Motivation, GSL vs attention, contributions |
| 2. Background | 188-326 | Problem def, GCN, T-GCN (long) |
| 3. Proposed Method | 327-462 | GSL intuition, DAGMA/NOTEARS, algorithm |
| 4. Experimental Results | 463-873 | Setup, metrics, ablation, tables, figures |
| 5. DAG Analysis | 874-917 | Spatial vs temporal graph interpretation |
| 6. Conclusion | 918-965 | Findings, future work |
| Appendix A | 1004-1060 | Bibliometric methodology |
| Bibliography | — | MyReferences.bib |

### Current experimental tables
- Table 3 (line 562): GCN / GCN-GSL / GCN-cGSL comparison (SZ-Taxi + Los-loop, PH=1-4)
- Table 4 (line 688): TGCN / TGCN-GSL / TGCN-cGSL comparison (SZ-Taxi + Los-loop, PH=1-4)
- Table 2 (line 488): Hardware/hyperparameters

### Current figures
- Fig 1: GCN architecture (line 265)
- Fig 2: Traffic network concept (line 346)
- Fig 3: RMSE GCN comparison (line 676)
- Fig 4: RMSE TGCN comparison (line 800)
- Figs 5-8: Convergence curves (4 metrics × 2 datasets, lines 817-845)
- Fig 9: DAG concept (line 903)
- Figs A1-A3: Bibliometric (lines 1020-1050)

---

## 2. Current Claims/Contributions

### Abstract claims
- "up to 21.6% and 24.7% reduction in RMSE for GCN and T-GCN baselines, respectively"
- "spatial-only models benefit more from cyclic graphs, while temporal models perform best with acyclic"
- "data-driven structure learning as a computationally efficient and interpretable alternative"
- "explicit insights into hidden traffic patterns"

### Introduction claims
- GSL discovers "explicit, persistent dependencies"
- "hidden causal structure of traffic networks"
- "significant error reduction compared to traditional distance-based graphs"
- "GSL is a powerful, plug-and-play module"

### Section 5 claims
- Temporal DAG interpretation: edge j→i means "j at time t predicting i at time t+1"
- DAGs suit T-GCN because "influence flows forward in time"
- Symmetrized cyclic version better for GCN

### Conclusion claims
- "causal dependencies" (multiple occurrences)
- "causal structure learning"
- "causal pathways in traffic networks"
- "causal graphs offer interpretable insights"

---

## 3. Existing Experimental Evidence (Stages 24-26)

### Stage 24: Single-lag temporal DAGMA
- SZ-Taxi: Physical 532 edges, DAGMA 8 edges, cGSL 16 edges
- Los-loop: Physical 2833 edges, DAGMA 28-39 edges
- Key finding: dense physical graphs cause severe oversmoothing

### Stage 25: Multi-PH validation
- Monotonic pattern: fewer edges → better RMSE
- DAGMA ≈ correlation graphs (neither has clear advantage)
- On SZ-Taxi: NoGraph best; on Los-loop: sparse graphs can beat NoGraph

### Stage 25D: Multi-lag DAGMA pilot (N=20 sensors)
- Different lag blocks have genuinely different edge structures
- Cross-lag Jaccard < 0.08 (almost zero overlap)

### Stage 26: Full-sensor multi-lag + GatedMultiGraphTGCN
- **Los-loop PH=1 (5 seeds):**
  - NoGraph: 5.234 ± 0.09
  - MultiGraph: 4.794 ± 0.10
  - GatedMulti: 4.452 ± 0.14 (14.9% improvement)
- **Parameter control:** 99% of improvement from gating, not params
- **Lag ablation:** all 3 lags contribute, combination best (13.3%)
- **SZ-Taxi:** marginal improvement only (0.2-0.9%)

### Stage 26 Validation (latest)
- Experiment A: GatedMulti beats NoGraph in 5/5 seeds
- Experiment B: NoGraph h=74 (16872 params) → 5.137 vs GatedMulti 4.458
- Experiment C: all 3 lags best, individual ~10% each

---

## 4. Reviewer 1 Comments → Evidence Mapping

| # | Comment | Status | Evidence Available |
|---|---------|--------|-------------------|
| 1 | Bibliometric analysis underused | ✅ Fixable | Appendix exists, needs integration |
| 2 | GCN/T-GCN background too long | ✅ Fixable | Condense sections 2.2-2.3 |
| 3 | A→W notation switch unexplained | ✅ Fixable | Rewrite notation |
| 4 | **Temporal interpretation asserted, not demonstrated** | ⚠️ MAJOR | Stage 25D/26 provide explicit multi-lag formulation |
| 5 | **Sparsity/oversmoothing concern** | ⚠️ MAJOR | Stages 24-26 fully address this |
| 6 | **No multiple seeds/variance** | ⚠️ MAJOR | Stage 26: 5 seeds, mean±std available |
| 7 | Results only to PH=4 | ⚠️ MINOR | Current evidence covers PH=1-4 |
| 8 | Section 5 repetitive | ✅ Fixable | Consolidate subsections |
| 9 | **No limitations section** | ⚠️ MODERATE | Must add |
| 10 | Metric definitions too long | ✅ Fixable | Condense |
| 11 | Dense convergence plots | ✅ Fixable | Move to appendix |
| 12 | Citation style inconsistency | ✅ Fixable | Standardize |

### Reviewer 1 Questions
| Question | Answer Available? | Source |
|----------|------------------|--------|
| Physical vs learned graph visualization | ⚠️ Partial | Stage 24 adjacency matrices exist |
| Predicted vs actual time series | ❌ Not yet | Would need new figure |
| Time-varying graph plan | ✅ Yes | Can describe sliding-window DAGMA |
| Direct evidence for temporal DAG | ✅ YES | Stage 25D/26 explicit multi-lag formulation |

---

## 5. Reviewer 2 Comments → Evidence Mapping

| # | Comment | Status | Evidence Available |
|---|---------|--------|-------------------|
| 1 | **Abstract RMSE numbers inconsistent** | ⚠️ MUST FIX | 24.7% is GCN Los-loop, not TGCN |
| 2 | **"Hidden causal structure" unsupported** | ⚠️ MAJOR | Must remove causal claims |
| 3 | **Static vs adaptive graph inconsistency** | ⚠️ MUST FIX | Clarify in text |
| 4 | cGSL defined too late | ✅ Fixable | Move definition earlier |
| 5 | Convergence plots hard to read | ✅ Fixable | Move to appendix |
| 6 | Typo "avergae" | ✅ Fixable | Fix |

---

## 6. Items Requiring Only Textual Revision

These can be fixed by rewriting, no new experiments needed:

1. **Abstract numbers** — correct 24.7% to proper attribution
2. **Remove ALL causal claims** — replace with "temporal functional dependency"
3. **Fix static vs adaptive graph wording** — clarify that learned graph is static, GRU is dynamic
4. **Move cGSL definition before Section 4**
5. **Condense GCN/T-GCN background** (sections 2.2-2.3)
6. **Fix notation A→W** — explain in main text
7. **Add limitations section**
8. **Condense metric definitions**
9. **Move convergence plots to appendix**
10. **Fix citation style**
11. **Fix "avergae" typo**
12. **Consolidate Section 5** (spatial vs temporal graph)

---

## 7. Items Requiring Tables/Figures

These require creating new tables/figures from existing results:

1. **Multi-seed results table** (mean ± std) — data exists in Stage 26 CSV
2. **Parameter-matched control table** — data exists in Stage 26 CSV
3. **Lag ablation table** — data exists in Stage 26 CSV
4. **Graph density comparison** (physical vs DAGMA) — data exists
5. **Oversmoothing evidence table** — data exists across stages
6. **Physical vs learned graph heatmap** — can generate from existing matrices
7. **Predicted vs actual time series** — would need to generate from saved models

---

## 8. Items That Might Require New Experiments

| Item | Required? | Reasoning |
|------|-----------|-----------|
| Multi-seed for SZ-Taxi | Maybe | Current SZ-Taxi improvement is marginal; multi-seed would confirm |
| Longer prediction horizons (PH>4) | Maybe | Reviewer 1 asks; data allows up to PH=8 |
| Physical graph visualization | No | Can use existing adjacency matrices |
| Predicted vs actual plots | Maybe | Would need to re-run inference on saved models |
| Hyperparameter sensitivity sweep | No | Stage 25 already has threshold sensitivity |
| Physical graph sparsification | No | Stage 24-25 already shows this |

---

## 9. Proposed Main-Text vs Appendix Allocation

### Main text (revised)
- Introduction (shortened, corrected claims)
- Background (condensed GCN/T-GCN)
- Proposed Method (multi-lag DAGMA + GatedMultiGraphTGCN)
- Experiments: setup, key results (multi-seed, parameter control, lag ablation)
- Discussion (oversmoothing, dataset dependence, limitations)
- Conclusion

### Appendix
- Original GSL/cGSL results (Tables 3-4)
- Convergence plots (Figs 5-8)
- Bibliometric methodology
- Threshold sensitivity analysis
- SZ-Taxi detailed results
- Physical vs DAGMA graph comparisons
- Full Stage 24-25 evidence

---

## 10. Contradictions Between Manuscript and Current Evidence

| Manuscript Claim | Current Evidence | Resolution |
|------------------|-----------------|------------|
| "21.6% and 24.7% RMSE reduction" | 24.7% is GCN Los-loop, not TGCN | Correct numbers |
| "causal structure" / "causal dependencies" | No causal identification performed | Remove or qualify |
| "adapt to changing traffic patterns" | Graph is static (computed once) | Clarify wording |
| "explicit insights into hidden causal structure" | No visualization of learned graph | Add figure or qualify |
| TGCN-GSL always best | GatedMulti is the new best method | Restructure narrative |
| Single static graph is sufficient | Multi-lag formulation works better | Add multi-lag as main contribution |

---

## 11. Recommended Narrative Shift

### Old narrative
Physical graph → GSL/cGSL learns better graph → outperforms baselines

### New narrative (evidence-supported)
Physical graph → oversmoothing problem → single DAGMA helps but limited → multi-lag DAGMA reveals temporal heterogeneity → GatedMultiGraphTGCN adaptively exploits lag-specific graphs → robust 14.9% improvement on Los-loop (5 seeds)

The old GSL/cGSL results remain valuable as motivation and baseline evidence, but the multi-lag GatedMulti becomes the primary methodological contribution.

---

*Phase 1 audit completed 2026-09-05*
