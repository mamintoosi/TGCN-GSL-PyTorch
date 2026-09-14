# Stage 47 — Writing Report: Section 5 Results

**Date:** 2026-09-10  
**Scope:** writing stage — `paper/gsl_stage47/sections/results.tex` and this report. No other manuscript or source files modified; no experiments run; no numbers invented.

---

## 1. Files created / modified

| Path | Action |
|------|--------|
| `paper/gsl_stage47/sections/results.tex` | **Created** — Section 5 (§5.1–§5.6) + Tables 1–4 |
| `paper/gsl_stage47/stage47_writing_report.md` | **Created** — this report |

Not modified: Stage 46 sections, `submitted_version/`, `previous_revision/`, code, results artifacts.

---

## 2. Structure used

Per `Stage45_1` / `Stage47.md`:

| Subsection | Label | Message |
|------------|-------|---------|
| 5.1 | `sec:res-physical` | Physical graph is not a sound default |
| 5.2 | `sec:res-single` | A single learned graph does not close the gap |
| 5.3 | `sec:res-controls` | Sparsity and capacity controls |
| 5.4 | `sec:res-dissociation` | Multi-lag graphs: consumption matters |
| 5.5 | `sec:res-mix` | Gated mixing on Los-loop: the positive result |
| 5.6 | `sec:res-boundaries` | Temporal resolution and dataset dependence |

Cross-refs to Stage 46: `sec:method`, `sec:setup`, `sec:stats`, `sec:controls`, `sec:variant15`, `sec:gsl_contemp`, `sec:gsl_multilag`, `sec:consumption`, `sec:results`.

---

## 3. Tables and figures referenced

### Tables defined in `results.tex`

| Manuscript ID | LaTeX label | Content | Source |
|---------------|-------------|---------|--------|
| Table 1 | `tab:tgcn-main` | T-GCN family, both datasets, PH1–4, mean (std) | `gsl_stage41/stage41_summary.csv` |
| Table 2 | `tab:gcn-main` | GCN family, both datasets, PH1–4 | same |
| Table 3 | `tab:controls` | Los PH1 sparsity + capacity (capacity seed-42 labeled) | Stage 41 sparse summary + `stage26_validation_B` + Stage 40 NoSpatial/Mix |
| Table 4 | `tab:los15` | 15-min variant, 3 methods × PH1–4 + improvements | Stage 44 §8 (from Stage 29 JSON) |

### Figure placeholders (commented; not generated)

| ID | Label | Role |
|----|-------|------|
| F1 | `fig:dissociation` | GCN-MultiGSL vs T-GCN-MultiGSL on identical graphs |
| F2 | `fig:perseed` | Per-seed distributions Los-loop (means must match Table 1) |
| F3 | `fig:graphstruct` | Physical vs multi-lag union structure |

Figures are left as commented placeholders for a later stage (Stage 45.1 diet: 3 main figures).

---

## 4. Every numerical result used (traceability)

### 4.1 T-GCN family (Table 1) — `stage41_summary.csv`

**Los-loop RMSE mean (std):**

| Method | PH1 | PH2 | PH3 | PH4 |
|--------|-----|-----|-----|-----|
| T-GCN | 7.877 (0.284) | 8.133 (0.085) | 8.368 (0.174) | 8.659 (0.163) |
| NoSpatial | 5.251 (0.186) | 5.759 (0.084) | 6.109 (0.074) | 6.576 (0.117) |
| GSL | 5.859 (0.206) | 6.301 (0.060) | 6.655 (0.116) | 7.040 (0.095) |
| cGSL | 5.821 (0.228) | 6.354 (0.120) | 6.669 (0.112) | 7.088 (0.123) |
| MultiGSL | 4.841 (0.115) | 5.447 (0.136) | 5.940 (0.028) | 6.260 (0.033) |
| Weighted | 4.834 (0.113) | 5.437 (0.133) | 5.926 (0.026) | 6.253 (0.031) |
| Mix | 4.491 (0.140) | 5.075 (0.149) | 5.546 (0.156) | 5.859 (0.067) |

**SZ-Taxi RMSE mean (std):**

| Method | PH1 | PH2 | PH3 | PH4 |
|--------|-----|-----|-----|-----|
| T-GCN | 5.449 (0.109) | 5.552 (0.082) | 5.602 (0.048) | 5.633 (0.057) |
| NoSpatial | 4.120 (0.004) | 4.160 (0.006) | 4.192 (0.006) | 4.225 (0.008) |
| GSL | 4.281 (0.051) | 4.309 (0.027) | 4.332 (0.012) | 4.370 (0.021) |
| cGSL | 4.301 (0.033) | 4.335 (0.016) | 4.380 (0.025) | 4.415 (0.014) |
| MultiGSL | 4.130 (0.016) | 4.161 (0.005) | 4.200 (0.006) | 4.224 (0.003) |
| Weighted | 4.130 (0.016) | 4.161 (0.005) | 4.201 (0.006) | 4.225 (0.003) |
| Mix | 4.119 (0.020) | 4.149 (0.005) | 4.178 (0.002) | 4.217 (0.010) |

*Displayed in the manuscript rounded to two decimals (matching Stage 45.1 message numbers).*

### 4.2 GCN family (Table 2) — `stage41_summary.csv`

**Los-loop:** Physical 8.145/8.539/8.765/8.765; NoSpatial 4.880/5.604/6.023/6.258; GSL 7.826/8.479/8.448/8.933; cGSL 5.763/6.331/6.684/6.881; MultiGSL 9.780/10.050/10.241/10.272.

**SZ-Taxi:** Physical 5.960/5.975/5.989/6.001; NoSpatial 4.114/4.153/4.187/4.217; GSL 4.880/4.910/4.930/4.958; cGSL 4.641/4.674/4.701/4.728; MultiGSL 4.823/4.854/4.878/4.904.

### 4.3 Improvements (Mix) — `stage41_improvements.csv`

| Reference | PH1 | PH2 | PH3 | PH4 |
|-----------|-----|-----|-----|-----|
| vs T-GCN (Los) | 42.98% | 37.60% | 33.72% | 32.34% |
| vs NoSpatial (Los) | 14.47% | 11.88% | 9.22% | 10.91% |
| vs NoSpatial (SZ) | 0.02% | 0.27% | 0.34% | 0.19% |

Manuscript uses **14.5 / 11.9 / 9.2 / 10.9%** vs NoSpatial and **43.0 / 37.6 / 33.7 / 32.3%** vs Physical (R2: “relative RMSE reduction”).

NoSpatial vs Physical (Los): 33.3 / 29.2 / 27.0 / 24.1%.  
NoSpatial vs Physical (SZ): 24.4 / 25.1 / 25.2 / 25.1%.

### 4.4 Paired statistics — `stage41_paired_tests.csv`

**Los Mix vs NoSpatial:** wins 5/5 all PHs; mean diffs 0.760 / 0.684 / 0.563 / 0.717; paired-$t$ p = 0.00148 / 0.00151 / 0.00077 / 0.000016.

**Los Mix vs MultiGSL:** wins 5/5; diffs 0.349 / 0.372 / 0.394 / 0.401.

**SZ Mix vs NoSpatial:** wins 4/5, 4/5, 5/5, 4/5; diffs ≤ 0.014; p as large as 0.91 (PH1).

### 4.5 Sparsity controls — `stage41_summary.json` → `sparse_controls_stage32`

| Method | mean | std | n |
|--------|------|-----|---|
| RandTop30 | 6.0956 | 0.1191 | 5 |
| CorrTop30 | 5.3891 | 0.0979 | 5 |
| NoSpatial (Los PH1) | 5.2514 | 0.1863 | 5 |
| MultiGSL | 4.8408 | 0.1146 | 5 |
| Mix | 4.4914 | 0.1403 | 5 |

Manuscript: 6.10 / 5.39 / 5.25 / 4.84 / 4.49.

### 4.6 Capacity control (seed 42 only) — Stage 44 §7 + Stage 26 validation B

| Config | RMSE | Params |
|--------|------|--------|
| NoSpatial h=64 | 5.1432 | 12,672 |
| NoSpatial h=74 | 5.1373 | 16,872 |
| Mix | 4.4578 | 17,091 |

Labeled single-seed in §5.3 and Table 3.

### 4.7 15-minute variant — Stage 44 §8 (Stage 29 JSON)

| Method | PH1 | PH2 | PH3 | PH4 |
|--------|-----|-----|-----|-----|
| NoSpatial | 8.600 (0.249) | 9.311 (0.197) | 10.048 (0.074) | 10.460 (0.196) |
| MultiGSL | 7.246 (0.298) | 8.571 (0.327) | 9.112 (0.077) | 9.756 (0.210) |
| Mix | 6.240 (0.187) | 7.322 (0.338) | 8.063 (0.203) | 8.777 (0.326) |
| Mix vs NoSpatial | −27.4% | −21.4% | −19.8% | −16.1% |
| Mix wins | 5/5 | 5/5 | 5/5 | 5/5 |

### 4.8 Graph statistics cited in prose

| Fact | Source |
|------|--------|
| Los multi-lag: 12/3/15 slots, union 28 | Stage 41/44, Stage 46 §3.3 |
| SZ multi-lag: 0/0/2, union 2 | same |
| Union holds 28/30 slots | Stage 46 / Stage 44 |

---

## 5. Consistency check vs Stage 40/41/42

| Check | Status |
|-------|--------|
| All Table 1–2 cells match `stage41_summary.csv` (rounded) | ✅ |
| Improvements match `stage41_improvements.csv` | ✅ |
| Win counts / paired diffs match `stage41_paired_tests.csv` | ✅ |
| Sparse controls match Stage 41 JSON (not the slightly different claim-audit rounding 6.05/0.03) | ✅ used JSON |
| Capacity numbers match Stage 26 B artifact + Stage 44 | ✅ |
| 15-min numbers match Stage 44 §8 exactly | ✅ |
| No claim that GSL beats NoSpatial | ✅ |
| No causal language; dissociation uses R1 wording | ✅ |
| No “adaptive”; Mix described as changing consumption only | ✅ |
| Sparsity scope = Los PH1 only | ✅ |
| 15-min labeled as resolution variant; PH indices not equated | ✅ |
| Results kept descriptive; mechanism left to Discussion | ✅ |
| Stage 46 method names / notation | ✅ |
| History language absent | ✅ |

---

## 6. Remaining inconsistencies / missing evidence

1. **Figures F1–F3 not generated.** Placeholders only; Stage 47 does not create image files. Next writing stage must produce dissociation, per-seed, and graph-structure figures whose means match Table 1 exactly.
2. **15-min series length.** Stage 46 setup states $T=672$ (artifact-verified D1). Stage 44 §8 prose still says $T=992$. Results numbers are from Stage 29 JSON and are unaffected; manuscript assembly should follow **Stage 46 (672)**.
3. **Seed-42 vs five-seed Mix in capacity block.** Table 3 capacity rows use seed-42 Mix (4.46) beside five-seed Mix (4.49) in the sparsity block. Explicitly labeled; no confusion if captions retained.
4. **MAE not in main Results.** Per Stage 45.1, MAE belongs in Appendix A (later stage).
5. **Hard `\ref{fig:...}` to undeclared figure environments.** Text uses `Figure~\ref{fig:perseed}` in §5.5; until figures are added, LaTeX will warn. Acceptable for a draft section file; fix at assembly.
6. **Weighted / cGSL still in main tables.** Stage 45.1 keeps them in Tables 1–2 with nulls stated in text (done). Optional later demotion to appendix if page budget requires.
7. **No new evidence gaps** for the claims written: every quantitative sentence traces to Stage 40/41/44/29 artifacts.

---

## 7. Checklist

- [x] Structure 5.1–5.6 as specified  
- [x] Exact Stage 40 five-seed numbers  
- [x] Relative RMSE reduction phrasing (R2)  
- [x] Dissociation not over-claimed (R1)  
- [x] n=5 / Wilcoxon floor respected  
- [x] Controls scope-labeled  
- [x] 15-min variant correctly framed  
- [x] No history / no causal / no adaptive  
- [x] Only `paper/gsl_stage47/` files written  

---

## 8. Verdict

**STAGE 47 COMPLETE — READY FOR DISCUSSION (STAGE 48) AND FIGURE GENERATION.**

`results.tex` is consistent with Stage 40–44 canonical artifacts and Stage 46 terminology. The section is intentionally descriptive; mechanistic interpretation belongs in Section 6. Figures remain to be built in a subsequent stage.
