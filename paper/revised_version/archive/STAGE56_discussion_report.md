# Stage 56 — Discussion, Limitations, Conclusion

**Date:** 2026-09-11  
**Output:** `paper/revised_version/sections/discussion_limitations_conclusion.tex`  
**PDF:** `paper/revised_version/sn-article.pdf` (26 pages)

## Content

**§ Discussion** — why dense physical graphs can hurt; when learned structure helps (multi-lag + lag-aligned use, sparsity control, union failure); what the graph represents (statistical, not causal); dataset dependence (15-min control + edge recovery).

**§ Limitations** — n=5; control scope (Los PH1 only); two datasets / PH≤4; linear DAGMA; static graphs vs Mix; GCN/T-GCN only.

**§ Conclusion** — five findings aligned with Results; future work (horizons, datasets, controls, backbones, attention hybrids, time-varying graphs). **No causal language** from the submitted conclusion.

## Dropped from submitted analysis/conclusion

- Temporal-DAG “j at t predicts i at t+1” for contemporaneous GSL  
- “Causal influence / causal pathways / hidden causal structure”  
- “Significantly outperforms conventional methods” as a blanket claim  
- Standalone acyclicity-paradox section (retired; contemporaneous vs multi-lag already separated in Method)

## Remaining

1. Restore full Appendix A (bibliometric) + optional B (MAE / extra figures) from submitted materials  
2. Flatten stage inputs for submission  
3. Point-by-point **Response to Reviewers** under the final framing  
4. Optional: compile with journal checklist / line numbers  
