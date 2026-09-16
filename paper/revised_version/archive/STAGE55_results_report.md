# Stage 55 — Results Revision Report

**Date:** 2026-09-11  
**Output:** `paper/revised_version/sections/results.tex`  
**PDF:** `paper/revised_version/sn-article.pdf` (27 pages)

**Numbers:** Stage 40 five-seed artifacts only (unchanged from Stage 47 audit).  
**Framing:** GSL for traffic prediction; graph-free as control; multi-lag + use as *conditions*, not a new paper identity.

---

## 1. Narrative arc (main claim)

1. **Physical is weak** — motivates learning $\mathbf{A}$ and requires a graph-free control.  
2. **Single contemporaneous GSL/cGSL** — helps vs Physical, not vs NoSpatial.  
3. **Sparsity control** — 30-edge random/corr graphs worse than no graph; DAGMA placement + use better (Los PH1 only).  
4. **Multi-lag GSL** — strongest Los-loop story: Mix **+14.5% vs NoSpatial**, **+43.0% vs Physical**; $5/5$ seeds; Weighted≈Fixed; static union on GCN fails on the same artifacts.  
5. **Boundaries** — SZ-Taxi null (2 edges); 15-min Los variant keeps large relative gains.

Summary box at end of section (four bullets).

---

## 2. Tables and figures

| ID | Content |
|----|---------|
| `tab:tgcn-main` | T-GCN family both datasets PH1–4 |
| `tab:gcn-main` | GCN family both datasets |
| `tab:controls` | Sparsity + capacity (Los PH1) |
| `tab:los15` | 15-min variant |
| `fig:dissociation` | Union vs per-timestep (same multi-lag graphs) |
| `fig:perseed` | Per-seed Mix ladder |
| `fig:graphstruct` | Physical vs multi-lag union structure |

All figure PDFs already in `paper/revised_version/figures/`.

---

## 3. Reviewer comments vs this section

| Comment | Status |
|---------|--------|
| R1-W5 sparsity | **Applied** — dedicated subsection + Table controls |
| R1-W6 seeds/variance | **Applied** — mean±std, $5/5$ wins, stats pointer |
| R1-W7 longer horizons | **Partial** — 15-min variant with explicit non-equivalence; PH5–8 @5-min not run → response letter |
| R1-W11 dense convergence grids | **Not in main Results** — response letter / appendix |
| R1-Q2 predicted vs actual | **Not in main Results** — appendix / letter |
| R2-1 abstract numbers | **Aligned** — 14.5% / 43.0% as in author abstract |
| R2 GSL vs cGSL “who is best” | **Rephrased** — not universal; GCN vs T-GCN observation only |

Older claims that GSL always beats NoSpatial / Physical are **not** used.

---

## 4. For your review

Check:
1. Whether the story order (physical → GSL → sparsity → multi-lag → boundary) is what you want readers to see.  
2. Whether Mix should be named more prominently as “proposed method” earlier in the section.  
3. Whether Figure 1 (dissociation) should move earlier (e.g. after multi-lag intro) or stay after sparsity.

After your check: Discussion + Limitations + Conclusion, then appendices and response letter.
