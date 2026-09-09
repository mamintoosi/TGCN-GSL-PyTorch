# Stage 39 — Manuscript Revision & Point-by-Point Reviewer Response Report

**Date:** 2026-09-09 · **Mode:** manuscript-revision only (no experiments run)
**Authoritative inputs:** `doc/STAGE38_SCIENTIFIC_AUDIT_REPORT.md` (interpretation), Stage 37
result JSONs (numbers), `paper/Reviewers-comments.txt` (reviewer source), Stage 26–37 reports.

---

## 1. Executive Summary

All 12 manuscript integration tasks from Stage39.md were executed, all 22 reviewer items
(R1-Summary + W1–W12 + Q1–Q4; R2-1–6) are addressed in a new point-by-point response
letter, the global consistency audit passes, every integrated number was independently
recomputed from the raw Stage 37/26 artifacts before insertion, and the manuscript
compiles cleanly (28 pages, 0 undefined references, no duplicate labels).

**Verdict: GO for Stage 40 — Final Manuscript & Reviewer-Response Quality Audit.**

---

## 2. Files Changed

| File | Change |
|---|---|
| `paper/sections/abstract.tex` | Task 11: added reproduction sentence (21.7%, five seeds; slice-0 graphs recovered at support level) |
| `paper/sections/introduction.tex` | Task 5: single-graph GSL introduced as *contemporaneous special case* (L=0), forward-referencing the new baseline subsection |
| `paper/sections/method.tex` | Tasks 2.6/9: threshold-semantics paragraph (signed-weight policy); static-graph/dynamic-use clarification (R2-3); contemporaneous-vs-lagged distinction paragraph (Task 5) |
| `paper/sections/experiments.tex` | Task 9: new `\subsection{Reproducibility}` (`sec:reproducibility`): DAGMA determinism audit, seed policy (42–46, population std), threshold semantics, historical reproduction levels |
| `paper/sections/results.tex` | Tasks 1/2/3/4/5/12: matched-budget controls + union-control paragraph; new §5.3 "The single-graph GSL baseline revisited" with 5-seed `tab:gsl_baseline` + reproduction paragraphs; `tab:multiph` + T-GCN-GSL row; `tab:sz_multiph` → 5-seed × 3 methods; temporal-interpretation fixes; `fig:graph_comparison` caption corrected |
| `paper/sections/discussion.tex` | Tasks 2.3/2.5: SZ subsection rewritten (5-seed numbers, MultiGSL-doesn't-help observation); new §"Sparsity Versus Learned Structure"; GSL/cGSL connection subsection updated (contemporaneous finding + 21.7% reproduction) |
| `paper/sections/limitations.tex` | Task 8: expanded to 9 items — measured runtimes (16 min/PH 207-node 4 CPU cores; 52–76 min archived 156-node; ~4 h multi-lag), no-λ-sweep (explicitly not measured), backbone scope, PH5–8 future work with 15-min proxy, dataset dependence with ungated-variant fact |
| `paper/sections/conclusion.tex` | Task 2.3/2.5: dataset-dependence item rewritten (win counts, not "dips"); new sparsity-vs-structure item; reproduction paragraph; future work adds λ sensitivity and longer horizons |
| `paper/sections/background.tex` | R1-W12: citation style ("Kipf and Welling", "Zhao et al.") |
| `paper/appendix/original_gsl_results.tex` | Task 10: historical-labeling preamble strengthened; union-of-K-offset-fits protocol (28/32/33/39 edges) + positive-only historical rule documented; slice-0 reproduction levels stated; 21.6/24.7/21.8/26.9 derivations made explicit (multi-horizon means, single seed) |
| `paper/appendix/additional_diagnostics.tex` | Task 8/9: measured single-graph runtime datapoints + sliding-window cost multiplier added to DAGMA-cost subsection |
| `paper/generate_figures.py` | Task 12: fig1 counting fixed — physical edges = positive entries (not weight sum 1307); DAGMA union self-loops removed before counting (30 lag edges / 28 distinct); degrees exclude self-connections on both graphs |
| `paper/figures/fig1_graph_comparison.{pdf,png}` | Regenerated from fixed script (reads existing artifacts only; no retraining). PDF now prints "(2833 edges, 207 nodes)", "(28 edges, threshold=0.1)", "Physical (mean=12.7)", "DAGMA (mean=0.1)" — consistent with caption and Stage 38 |
| `paper/sn-article.pdf` | Recompiled (28 pages) |
| **`paper/Response-to-Reviewers.md`** | **Created** — point-by-point response, all R1 + R2 items |
| **`doc/STAGE39_MANUSCRIPT_REVISION_REPORT.md`** | **Created** — this report |

Pre-existing workspace state (not touched): `Stage35.md`/`Stage36.md` deletions were
already present in `git status` before Stage 39 began; `Stage38.md`, `Stage39.md`,
`doc/STAGE38_SCIENTIFIC_AUDIT_REPORT.md` were already untracked.

---

## 3. Reviewer Comments Addressed

| Item | Manuscript change | Response-letter entry |
|---|---|---|
| R1-S4/W4/Q4 | Contemporaneous-vs-lagged distinction (§3.3, §5.2, §5.3, Disc., App. A); exact slice-0 reproduction evidence | ✅ |
| R1-W1 | Bibliometric already in Intro ¶2 (pre-revision) | ✅ (honest: not expanded) |
| R1-W2 | Background already condensed (pre-revision) | ✅ |
| R1-W3 | Convention sentence in §3.2 main text (pre-revision) | ✅ |
| R1-W5 | Matched-budget controls (RandTop30, CorrTop30) + union control (5.928) in `tab:oversmoothing` + dedicated paragraph; sparsified-physical control explicitly acknowledged as **not run** | ✅ (with "substantially addresses, not proves" framing) |
| R1-W6 | 5-seed mean±std everywhere new (seeds 42–46); determinism audit cited; **no significance testing** stated | ✅ |
| R1-W7 | Scope limitation (PH1–4; PH5–8 future work) + 15-min-sampling proxy (27.4% at ≤60 min); **PH5–8 not run** — stated plainly | ✅ |
| R1-W8 | Results/Discussion restructure (pre-revision) + tie-back in Disc. §6.3–6.4 | ✅ |
| R1-W9 | Limitations: scalability (measured), λ (no sweep — stated), backbones (T-GCN/GCN-family only) | ✅ |
| R1-W10 | Metrics condensed (pre-revision) | ✅ |
| R1-W11 | Convergence grids in appendix (pre-revision) | ✅ |
| R1-W12 | Citation style standardized in Background/Method | ✅ |
| R1-Q1 | `fig:graph_comparison` (counts corrected this stage) | ✅ |
| R1-Q2 | `fig:pred_vs_actual` (pre-revision) | ✅ |
| R1-Q3 | Measured runtimes in App. C + response letter (windows × per-fit cost) | ✅ |
| R2-1 | Abstract numbers corrected (pre-revision); derivation of 21.6/24.7/21.8 now explicit in App. A; 21.7% five-seed reproduction added | ✅ (with full diagnosis of the original mix-up) |
| R2-2 | Causal language removed; descriptive-structural framing; limitations item strengthened | ✅ |
| R2-3 | §3.2 item 3 reframed (learned structure = real dependencies; GRU models dynamics); static/dynamic-use paragraph in §3.4 | ✅ |
| R2-4 | cGSL evaluated only in appendix after definition via §5.3 | ✅ |
| R2-5 | Compact convergence appendix (pre-revision) | ✅ |
| R2-6 | Typo gone (pre-revision) | ✅ |

---

## 4. Numbers Integrated (all independently recomputed from artifacts before insertion)

### Experiment A — `results/stage32_sparse_control/stage32_sparse_control.json` (n=5, ddof=0)
| Method | RMSE | MAE | Check |
|---|---|---|---|
| RandTop30 | 6.0956±0.1065 → **6.096±0.107** | 3.756±0.150 | ✅ matches Stage 38 |
| CorrTop30 | 5.3891±0.0875 → **5.389±0.088** | 3.247±0.056 | ✅ |

### Experiment B — `results/stage33_gsl_canonical/stage33_gsl_canonical_results.json` (n=5, sample std)
| PH | Physical | T-GCN-GSL | Improvement (from means) |
|---|---|---|---|
| 1 | 7.772±0.140 | 5.792±0.196 | **25.5%** |
| 2 | 8.118±0.189 | 6.328±0.166 | 22.0% |
| 3 | 8.456±0.047 | 6.669±0.080 | 21.1% |
| 4 | 8.554±0.169 | 6.999±0.084 | 18.2% |

Mean of per-PH improvements: **21.71% → "21.7%"** ✅. Seed-42 rows (5.548/6.156/6.645/6.932)
placed in `tab:multiph`; seed-42 Physical (7.658/8.002/8.512/8.540) bit-identical to the
manuscript's existing rows ✅ (protocol-integrity check from Stage 38 confirmed).
Union control 5.928 verified from `stage26_validation/stage26_results_los_ph1_seed42.csv` ✅.

### Experiment C — `results/stage33_sz_multiseed/stage33_sz_multiseed_results.json` (n=5, sample std)
All 12 cells match Stage 38 exactly (e.g., Mix PH1 4.1091±0.0052; MultiGSL PH3 4.1988±0.0121).
Paired win counts recomputed: Mix vs NoSpatial **5/5 at PH1–3, 3/5 at PH4** ✅; per-seed
delta ranges confirm PH4 straddles zero (−0.0124…+0.0037) → "within noise / not stable" ✅.
Mix-vs-NoSpatial improvements from means: +0.25/+0.26/+0.19/+0.09% ✅.
MultiGSL-vs-NoSpatial: −0.27/+0.05/−0.25/−0.18% ✅ ("does not help").

### Historical appendix derivations (recomputed, single seed)
GCN-cGSL SZ mean 21.6%, GCN-cGSL Los mean 24.7%, TGCN-GSL Los mean 21.8%, PH1 26.9% —
now stated as multi-horizon means with explicit provenance ✅.

### Known knife-edge rounding (documented, not an error)
Stage 26 multiseed full-precision stds: NoSpatial 0.0904→0.090 ✅; MultiGSL 0.10145 →
Stage 38/`tab:multiseed` carry **0.102** (double-rounding of 0.1015) while fig3's
3dp display shows 0.101. The authoritative 0.102 is retained per Stage 39 §7
("do not introduce rounding inconsistencies" relative to Stage 38); fig3 regenerates
from raw CSV at full precision and is not wrong at its own display precision.
No other table/figure discrepancies found (fig3/4/5 embed canonical values).

---

## 5. Claims Weakened / Tempered

1. "Sparsity" no longer presented as the explanation of gains — replaced by matched-budget
   decomposition (sparsity necessary but not sufficient; DAGMA lag-block edges + per-lag
   use = the gain; union-of-same-edges control at 5.928).
2. SZ PH4: "$-0.02\%$ at PH=4 (single seed)" → "not stable, within noise (3/5 seeds)".
3. All causal language removed except the explicit disclaimer.
4. "The learned graph adapts to changing traffic patterns" → static-graph/dynamic-use
   distinction (R2-3).
5. Single-graph DAG temporal interpretation removed everywhere; replaced by
   contemporaneous (demonstrated by exact reproduction) vs explicitly lagged (multi-lag).
6. "up to 21.6%/24.7%" → identified as GCN-cGSL multi-horizon means (R2-1), labeled single-seed historical.
7. Abstract now attributes the 21.7% reproduction to the revised protocol explicitly.

## 6. Claims Strengthened

1. GSL-vs-Physical: now five seeds, protocol-reproduced, graphs recovered at support level.
2. Dataset dependence: now a multi-seed boundary result with win counts (more credible).
3. Reproducibility: determinism audit, threshold semantics, provenance documented.

## 7. What Was NOT Done (by design)

- **No experiments**: A/B/C not rerun; no PH5–8; no λ sweep; no new DAGMA fits; no
  sparsified-physical control (Stage 38 optional item; reviewer R1-W5's "fairer baseline"
  acknowledged as not run in the letter).
- **No new figures** (fig1 regenerated only to fix internally embedded wrong numbers).
- Historical appendix tables untouched (values, structure, single-seed basis preserved).

## 8. Historical Artifacts — Integrity Check

`git status`/`git diff` confirm zero modifications under `archive/`, `data/`, `results/`,
`gsl_stage26/`, `utils/`, and `doc/` stage reports. Historical W_est stacks, historical
appendix numbers (4.818/5.400/5.846/6.257 etc.), and original appendix tables are intact.

## 9. Global Consistency Audit

All Stage 39 §6 search terms swept: 21.6/24.7/21.8/26.9 (historical, now labeled with
derivation), 4.818/5.400/5.846/6.257 (appendix tables + one labeled example), 1307 (none),
"5 edges" (only as true lag_2 count, correct), "causal" (limitations disclaimer only),
"adapt to changing" (limitations honesty statement only), temporal/dynamic/adaptive-graph
(clean), sparsity (now consistently framed as necessary-not-sufficient), lag/contemporaneous
(consistent contemporaneous-vs-lagged usage across §3, §5, Discussion, Conclusion).

## 10. Compilation Check

`pdflatex -interaction=nonstopmode -halt-on-error sn-article.tex` × 3 (reference
resolution): **PASS** — 28 pages, 0 undefined references/citations after resolution,
no duplicate labels, no new warnings attributable to this revision. The regenerated
`fig1` compiles in place. Build artifacts (`sn-article.out`, refreshed aux/log) were
regenerated by compilation; `*.aux`/`*.log` are gitignored, `sn-article.out` appears
as a new untracked build artifact.

## 11. Remaining Limitations / Issues

1. Sparsified-physical control (R1-W5's literal suggestion) remains unrun — acknowledged in
   the letter; Stage 38 lists it as optional (~5 min GPU) if the user wants it.
2. λ sensitivity and PH5–8 remain unmeasured (stated honestly in Limitations).
3. MultiGSL/Mix rows of `tab:multiph` (PH2–4) remain seed-42 only (5-seed upgrade is a
   Stage 38 optional item, ~30–60 min GPU, not run).
4. `tab:multiseed` MultiGSL std displays 0.102 vs fig3's full-precision 0.101
   (see §4; authoritative value retained).
5. `paper/revision_notes.md` (internal working notes from an earlier pass) still reflects
   the pre-Stage-39 state; superseded by the new response letter and this report.

## 12. Final Status

| Criterion | Status |
|---|---|
| Manuscript revision (Tasks 1–12) | **COMPLETE** |
| Reviewer response letter | **COMPLETE** (all 22 items) |
| Numerical consistency | **PASS** (all values recomputed from artifacts) |
| Historical artifact integrity | **PASS** |
| LaTeX compilation | **PASS** (28 pp, clean refs) |
| No unsupported claims / no significance claims | **PASS** |
| Contemporaneous-vs-lagged correctness | **PASS** |
| Sparsity framing | **PASS** |

**GO — Stage 40: Final Manuscript & Reviewer-Response Quality Audit.**
