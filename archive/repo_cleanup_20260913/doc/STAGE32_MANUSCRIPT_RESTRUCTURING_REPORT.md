# Stage 32 — Manuscript Restructuring & Sparse-Control Preparation

**Date:** 2026-09-08
**Scope:** All three followups from the Stage 31 integration plan, in the agreed order:
(1) matched-edge sparse-control experiment prepared (Linux runner provided),
(2) Results restructured into the six-subsection architecture,
(3) contradicted claims corrected — plus consistency updates to every affected section.

The manuscript compiles cleanly: **23 pages, 0 undefined references, 0 overfull hboxes.**

---

## 1. Followup 1 — Sparse-Control Experiment (prepared, run on Linux)

**Why not run here:** The Windows environment (`c:/programs/anaconda3/envs/pth`, CPU torch 2.3.1)
is *functionally correct* — a canary (1 seed, 2 epochs, both controls) passed end-to-end with
sensible numbers — but too slow: ~30 s/epoch ⇒ ~5 h for the full 10-run suite (5 seeds × 2 controls,
50 epochs). The Linux/GPU machine runs it in minutes.

### Deliverables

| File | Purpose |
|---|---|
| `gsl_stage26/stage32_sparse_control.py` | Experiment script, exact canonical Stage 26 pipeline |
| `run_stage32_sparse_control.sh` | Linux runner, mirrors `run_stage29_los15min.sh` conventions |

### To run on the Linux machine

```bash
cd /data/git/mamintoosi/TGCN-GSL-PyTorch
bash run_stage32_sparse_control.sh
# then copy results/stage32_sparse_control/ back to this machine
```

(Adjust `REPO`/`PYTHON` at the top of the script if paths differ.)

### Design (answers Reviewer 1 #5 at matched edge parity)

Both controls use exactly **30 directed edges** — the same stated budget as
T-GCN-MultiGSL / Mix (sum over the three lag graphs: 12+3+15) — and the *same canonical
pipeline* (`SupervisedForecastTask(loss="mse_with_regularizer")`, set_seed, Adam
lr=0.001/wd=1e-4, batch 128, 50 epochs, full-batch test eval, feat_max from training data only):

- **CorrTop30** — top-30 |Pearson| edges computed from *training data only*, plain standard
  TGCN. The strongest non-DAGMA heuristic per the 112-experiment archive.
- **RandTop30** — 30 random off-diagonal directed edges, re-drawn per seed, plain TGCN. Floor control.

**Edge-count bookkeeping (important, discovered during canary):** the three thresholded lag
graphs contain 12+3+15 = 30 edge *slots* (the paper's "30 edges"), but their *union* has 28
distinct edges (two overlaps). The canonical pipeline trains on the 3 *separate* graphs, so the
sum is the paper's stated budget; matched controls at 30 ≥ 28 is conservative in the method's
favor. The JSON records both counts (`multilag_reference.per_lag_counts`, `sum_edges=30`,
`union_edges=28`).

**Interpretation rule (pre-registered):** the gains are attributable to the *learned structure*
rather than sparsity per se if CorrTop30 lands near/below NoGraph (5.143) and far from
MultiGSL-Mix (4.452); a strong CorrTop30 would instead show sparsity alone explains much of the
gain. Either outcome is reportable; a row will then be added to Table `tab:oversmoothing`.

---

## 2. Followup 2 — Results Restructured (six subsections, per Stage 31 plan §8)

`paper/sections/results.tex` rewritten. New architecture:

| § | Subsection | Content | Reviewer |
|---|---|---|---|
| 5.1 | Dense Physical Graphs and Oversmoothing | `tab:oversmoothing` + **new Corr-K8 row** (1656 edges, RMSE 6.915 — dense *learned* graph also fails ⇒ density, not origin, is the harm); fig1 | R1#5 |
| 5.2 | From Single-Graph GSL to Multi-Lag Structure | **Drop-in narrative bridge** (original results as Act I, TGCN-GSL −21.8% summary sentence, Appendix A retained in main document); `tab:lag_stats` + caption note (edge counts include self-loops; cross-sensor counts 12/3/15); fig7 | R1#4, R2#4 |
| 5.3 | Multi-Seed Validation and Parameter Control | `tab:multiseed`, `tab:param_control` merged into one subsection (labels `sec:multiseed`+`sec:param_control` both kept so method.tex cross-refs still resolve); 13.3% vs 14.9% now explicitly labeled "seed-42" vs "five-seed mean"; fig3 | R1#6 |
| 5.4 | Ablation: Lags and Prediction Horizons | `tab:lag_ablation`, `tab:multiph` (Los-loop now scoped in text); fig5; threshold sweep moved to appendix with pointer | R1#7 |
| 5.5 | **Robustness Across Temporal Resolution (NEW)** | `tab:los15min` from Stage 29 verified JSON: 8.600±0.249 → **6.240±0.187 (+27.4%)**, 5/5 seeds, +21.4/+19.8/+16.1% at PH2–4; caption states PH=k = k×15 min and warns RMSE magnitudes aren't comparable across resolutions; 15-min protocol paragraph added to Experiments | R1#7, R2 |
| 5.6 | Dataset Dependence: the SZ-Taxi Boundary | `tab:sz_multiph` **promoted from appendix** (+ Physical row added); verified numbers ≤0.3% / −0.02%; resolution ruled out explicitly; graph-learnability mechanism (2 vs 30–32 edges), hedged language | R1#9, R2#1 spirit |

**Figure count in main text: 9 → 6.** Removed: fig2 (duplicated the table), fig8 (duplicated
Appendix C — now placed in appendix as `fig:convergence_rev`), fig4 (moved to appendix as
`fig:param_control_app`), fig6 (moved to appendix as `fig:threshold`). Kept: fig1, fig3, fig5,
fig7, fig9 (rewritten claims). **Main-text tables: 8** (two new/promoted).

**Structural repair:** the old orphan §5.7 ("Relation to Original GSL/cGSL Results", with its
empty `()` citation) is gone; the bridge text is now §5.2's opening and cites
Appendix~`app:original_gsl`, which is **now actually included in the main document**
(`sn-article.tex` appendices previously omitted `original_gsl_results.tex` — a latent bug found
and fixed).

---

## 3. Followup 3 — Contradicted Claims Fixed (per Stage 31 plan §14)

### 3.1 fig9 "correlation >0.99" — CONTRADICTED → REWRITTEN
Recomputed from the checkpoints (`results/stage26_checkpoint/los_ph1_seed42_{gated_multi,nograph}/y_pred.npy`):
overall Pearson r = **0.947 (Mix) / 0.929 (NoSpatial)**; for the exact window/nodes the figure
script plots (top-3 variance nodes, steps 145–245): node 149 → r = 0.33/0.41 (worst case),
node 163 → 0.95/0.92, node 12 → 0.96/0.94. The old caption's "1–2 step lag" cross-correlation
framing was also unsupported (maximum at shift 0 for the plotted series).

New text (results §5.3, "Predicted vs. actual behaviour"): reports r = 0.95 vs 0.93 overall,
per-node values reaching ≈0.99, characterizes the smoothing/lag qualitatively, and honestly notes
that one displayed node is a low-signal case where both models correlate weakly. Caption aligned.

### 3.2 Discussion "SZ-Taxi 0.2–0.9%" — CONTRADICTED → FIXED
Verified per-PH improvements (seed 42): **+0.19, +0.26, +0.11, −0.02 %**. New wording:
"≤0.3% at PH=1–3 and −0.02% at PH=4 (single seed)". The same verified wording propagates to
Conclusion (finding 6) and the new §5.6.

### 3.3 Additional accuracy fixes applied
- **4.715 vs 4.717** (MultiGSL, same config, two real runs): oversmoothing table now footnotes
  both values ("Stage-26 evaluation run" vs "validation rerun"); other occurrences harmonized.
- **Parameter counts**: method.tex keeps the formula-based 17,091/12,672 and now footnotes the
  checkpoint-bookkeeping values (17,156/12,737, difference identical under either accounting).
- **DAGMA hours**: appendix "≈28 hours" corrected to "≈24 hours for the 8 primary runs" (+ the
  15-min run ≈4 h, consistent with Stage 29 metadata).
- **Abstract / Intro / Conclusion**: now carry the two-resolution claim (+14.9% at 5-min,
  +27.4% at 15-min) and the verified SZ numbers.
- **Limitations**: item 1 gains the resolution-refutation nuance; item 8 rewritten (15–60 min
  covered at 15-min sampling; horizon *counts* beyond 4 not evaluated).
- **sz_multiph table**: Physical row added (5.267/5.406/5.629/5.604 — verified from
  `stage26_results_sz_ph{1..4}_seed42.json`), showing the oversmoothing pattern replicates on SZ.

---

## 4. Files Changed

| File | Change |
|---|---|
| `paper/sections/results.tex` | Rewritten (6 subsections, new tables, corrected claims) |
| `paper/sections/discussion.tex` | SZ paragraph rewritten; Training-Dynamics subsection removed (fig8 → appendix, one-sentence pointer kept) |
| `paper/sections/abstract.tex` | Two-resolution claim added |
| `paper/sections/introduction.tex` | 15-min result + single-lag-special-case framing |
| `paper/sections/conclusion.tex` | Findings 4–6 updated with verified numbers |
| `paper/sections/limitations.tex` | Items 1 and 8 updated |
| `paper/sections/experiments.tex` | New "Temporal-Resolution Protocol" subsection |
| `paper/sections/method.tex` | Parameter-count footnote |
| `paper/appendix/additional_diagnostics.tex` | Threshold sweep + convergence + param-control figures relocated here; DAGMA cost corrected; SZ table removed (promoted) |
| `paper/appendix/original_gsl_results.tex` | Protocol-scope + naming note |
| `paper/sn-article.tex` | **Includes `appendix/original_gsl_results` (was missing)** |
| `gsl_stage26/stage32_sparse_control.py` | New: sparse-control experiment |
| `run_stage32_sparse_control.sh` | New: Linux runner |

## 5. Verification

- `pdflatex` × 2 (+bibtex): **0 errors, 0 undefined references/citations, 0 overfull hboxes ≥10pt, 23 pages.**
- Every number edited into the manuscript was re-verified in this session against its JSON/checkpoint:
  Stage 26 (Los multiseed, param control, lag ablation, multi-PH, SZ per-PH, Corr-K8),
  Stage 29 (all 12 mean±std values), fig9 correlations, original TGCN-GSL table (−21.8%).
- Stage 32 script canary-passed on Windows (edge parity 30/28/30 confirmed in-run; JSON schema output verified).
- No experiment outputs fabricated; the Stage 32 table row awaits the Linux run.

## 6. Remaining Actions

1. **Run `bash run_stage32_sparse_control.sh` on the Linux machine** (~5 min GPU) and copy
   `results/stage32_sparse_control/` back; then optionally add the CorrTop30/RandTop30 row to
   `tab:oversmoothing` (one-line edit + one sentence).
2. Optional (only if reviewer strategy warrants): SZ 5-seed, PH 5–8.
3. Final pass before submission: citation style (R1#12), flatten modular sections for Springer,
   commit/archive the small result JSONs backing the paper (result dirs are gitignored).
4. Update `doc/RESPONSE_TO_REVIEWERS.md` numbers/locations per Stage 31 §19 when convenient.
