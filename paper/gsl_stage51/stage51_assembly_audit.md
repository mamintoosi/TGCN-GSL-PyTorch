# Stage 51 — Manuscript Assembly Audit

**Date:** 2026-09-10  
**Scope:** numeric/claim/cross-ref audit of draft section files produced Stages 46–51. No experimental reruns.

---

## 1. Inventory of draft files

| Stage | Path | Content |
|-------|------|---------|
| 46 | `paper/gsl_stage46/sections/method.tex` | §3 |
| 46 | `paper/gsl_stage46/sections/setup.tex` | §4 |
| 47 | `paper/gsl_stage47/sections/results.tex` | §5 + Tables 1–4 + Figs 1–3 |
| 47 | `paper/gsl_stage47/figures/fig{1,2,3}_*.{pdf,png}` | Main figures |
| 48 | `paper/gsl_stage48/sections/discussion.tex` | §6 |
| 49 | `paper/gsl_stage49/sections/limitations_conclusion.tex` | §7–8 |
| 50 | `paper/gsl_stage50/sections/intro_background.tex` | §1–2 |
| 51 | `paper/gsl_stage51/abstract.tex` | Abstract + keywords |

Not yet assembled into a single `sn-article.tex` master file.

---

## 2. Numeric audit (headline claims)

| Claim | Value | Verified against |
|-------|-------|------------------|
| Mix vs NoSpatial Los PH1–4 | 14.5 / 11.9 / 9.2 / 10.9% | `stage41_improvements.csv` |
| Mix vs Physical Los PH1–4 | 43.0 / 37.6 / 33.7 / 32.3% | same |
| Physical vs NoSpatial Los PH1 | 33.3% | same |
| GCN-MultiGSL vs T-GCN-MultiGSL PH1 | 9.78 vs 4.84 | `stage41_summary.csv` |
| Union / lag slots | 28 / 30 (12+3+15) | Stage 41/44; recomputed from npy with \|W\|>0.1, diag=0 |
| Sparse controls Los PH1 | 6.10 / 5.39 / 5.25 / 4.84 / 4.49 | Stage 41 JSON sparse block |
| Capacity h74 | 5.14 seed-42 | Stage 26 validation B |
| 15-min Mix vs NoSpatial | −27.4 / −21.4 / −19.8 / −16.1% | Stage 44 §8 |
| SZ Mix vs NoSpatial | ≤0.34% | improvements CSV |
| Fig2 means | match Table 1 to 4 decimals | regenerated from Stage 40 training JSONs |
| Fig3 degrees | 12.69 vs 0.14 | recomputed |

**No numeric inconsistency found** between Results tables, figure scripts, Stage 41 audit, and Abstract/Intro/Discussion quotes.

---

## 3. Claim-policy audit

| Rule | Status |
|------|--------|
| No GSL generally beats NoSpatial | ✅ |
| No causal language | ✅ |
| No “adaptive” graphs | ✅ |
| R1 consumption wording (not sole causal locus) | ✅ Abstract, §5.4, §6.2 |
| R2 relative RMSE reduction | ✅ |
| Sparsity scope Los PH1 | ✅ §5.3, §7 |
| 15-min = resolution variant | ✅ §5.6, §7 |
| n=5 / Wilcoxon floor | ✅ §4.7, §5, §7 |
| History language absent | ✅ |

---

## 4. Cross-reference status

| Label | Declared in | Used in |
|-------|-------------|---------|
| `sec:method` … `sec:consumption` | method.tex | setup, results, discussion |
| `sec:setup` … `sec:stats`, `sec:controls`, `sec:variant15` | setup.tex | results, discussion |
| `sec:results`, `sec:res-*` | results.tex | discussion |
| `tab:tgcn-main`, `tab:gcn-main`, `tab:controls`, `tab:los15` | results.tex | results, discussion |
| `fig:dissociation`, `fig:perseed`, `fig:graphstruct` | results.tex floats | results |
| `sec:discussion`, `sec:disc-*` | discussion.tex | — |
| `sec:limitations`, `sec:conclusion` | limitations_conclusion.tex | — |
| `sec:intro`, `sec:background` | intro_background.tex | textual forward refs from method/discussion |

**To fix at assembly:** convert textual “Sect.~2 / Sect.~5 / Introduction” in method.tex and discussion.tex to `\ref{sec:background}` / `\ref{sec:results}` / `\ref{sec:intro}`.

---

## 5. Figure scripts

| Script | Outputs |
|--------|---------|
| `paper/gsl_stage47/figures/make_fig1_dissociation.py` | fig1 |
| `paper/gsl_stage47/figures/make_fig2_fig3.py` | fig2, fig3 |

Fig2 means recomputed and printed; match Stage 41 to 4 decimals.

---

## 6. Remaining work (not blocking scientific draft)

1. Single master `sn-article.tex` with preamble, author block, bibliography.
2. Appendices A–C (MAE tables, graph statistics, bibliometric).
3. Compile check with `sn-jnl`.
4. Response-to-Reviewers letter (separate; next).
5. Optional: commit figure assets if `results*/` stays gitignored — keep small PDFs under `paper/gsl_stage47/figures/`.

---

## 7. Verdict

**DRAFT SCIENTIFIC CORE READY.** Method → Setup → Results (+figures) → Discussion → Limitations/Conclusion → Intro/Background → Abstract are internally consistent with Stage 40–44 artifacts and Stage 45.1 claim policy. Assembly into the journal master file and the response letter remain.
