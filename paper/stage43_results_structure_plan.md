# Stage 43 — Revised-Manuscript Structure Plan (Results and Beyond)

**Date:** 2026-09-10
**Scope:** analysis and planning only. No manuscript, code, table, figure, or result file was modified, deleted, or created except this report. No experiments were run.
**Inputs inspected:** `paper/submitted_version/` (single-file `sn-article.tex`, historical tables verified digit-for-digit in Stage 42), `paper/previous_revision/` (modular manuscript: `sections/{abstract,introduction,background,method,experiments,results,discussion,limitations,conclusion}.tex`, `appendix/{original_gsl_results,bibliometric,additional_diagnostics}.tex`, `Response-to-Reviewers.md`, `revision_notes.md`), `paper/revised_version/` (empty), `paper/Reviewers-comments.txt`, Stage 40 canonical aggregates (`gsl_stage41/stage41_summary.csv`, `stage41_paired_tests.csv`), Stage 42 audit (`gsl_stage42/stage42_submitted_vs_stage40_audit.md`, `stage42_claim_reconciliation.md`, `stage42_comparison.csv`), and reviewer evidence supplied with Stage 43 (graph statistics, matched-budget sparsity controls, 15-minute-sampling protocol).

---

## 1. Executive summary

The `previous_revision` snapshot already tells the correct story — MultiGSL-Mix as the central contribution, NoSpatial as the decisive baseline, sparsity ruled out by matched-budget controls, dataset dependence reported as a boundary, causal and "adaptive graph" language already removed, and both reviewers' comments addressed in a mature point-by-point response. What it does **not** yet contain is the Stage 40 canonical evidence base:

1. **The numbers predate Stage 40.** Main-text values (e.g., Los PH1: NoSpatial 5.234±0.090, Mix 4.452±0.143) come from an earlier pipeline; the Stage 40 canonical five-seed means differ measurably (5.2514±0.1863, 4.4914±0.1403). SZ-Taxi seed-win counts differ qualitatively at PH1/PH4 (5/5/5/3 previously vs 4/5, 4/5, 5/5, 4/5 canonically).
2. **The canonical method grid is incomplete.** The previous revision's main text contains no GCN-family results, no T-GCN or T-GCN-GSL/cGSL rows in the main tables, and no T-GCN-MultiGSL-Weighted. Stage 40 provides all 12 methods × 2 datasets × 4 horizons × 5 seeds, and its key new structural finding — the GCN-MultiGSL (union) vs T-GCN-MultiGSL (per-timestep) dissociation on identical graphs — is absent.
3. **The submitted-version integration is already solved in design** (contemporaneous re-baselining + appendix preservation) but must be re-executed against Stage 40 aggregates rather than the earlier five-seed runs.

The plan below therefore proposes **evolution, not redesign**: retain the previous revision's section architecture, migrate every main-text number to Stage 40 canonical aggregates, add one main results table (the full T-GCN grid), add one main figure (the consumption dissociation), and restrict all new non-Results claims to what Stage 40–43 evidence supports.

**Recommendation: BUILD ON `previous_revision` — migrate its proven, reviewer-aligned structure and storyline to Stage 40 canonical numbers, rather than drafting a new manuscript in `revised_version/` from scratch.**

**Verdict type: READY TO PLAN REVISION** (planning-stage equivalent of Stage 42's READY WITH IMPORTANT CAVEATS — the story is sound; the numbers must be migrated and a few claims must be scoped before any prose is rewritten).

---

## 2. Recommended revised scientific story

Retain the previous revision's storyline, sharpened by Stage 40/42:

1. **Question.** Does learned graph structure help traffic forecasting — and if so, through what mechanism?
2. **Decisive baseline (already in draft, now canonical).** The graph-free T-GCN-NoSpatial / GCN-NoSpatial configurations beat the dense physical graph by large margins on both datasets (T-GCN, Los PH1: 5.25 vs 7.88 RMSE, +33.3% Stage 40), so the physical reference is the *problem*, not the target. Contemporaneous learned graphs (T-GCN-GSL/cGSL, GCN-GSL/cGSL) beat the physical graph but **do not beat NoSpatial** — the submitted version's central claim, retired in Stage 42, stays retired.
3. **Central result.** T-GCN-MultiGSL-Mix — per-node, per-timestep learned gating over per-timestep-consumed DAGMA multi-lag graphs — is the **only learned-graph configuration that beats NoSpatial, and only on Los-loop** (Stage 40: −0.76/−0.68/−0.56/−0.72 RMSE at PH1–4, 5/5 paired seeds, paired-t p ≤ 0.0015; cautious phrasing per the Stage 43 brief). On SZ-Taxi the same configuration is statistically indistinguishable from NoSpatial (deltas ≤ 0.015 RMSE, 4/5–5/5 mixed win counts, paired-t p 0.006–0.9, exact Wilcoxon floor 0.0625 at n=5) — a **null**, stated plainly.
4. **Mechanism.** The GCN-MultiGSL (union) vs T-GCN-MultiGSL (per-timestep) dissociation on identical graph artifacts localizes the benefit in **graph consumption**, not graph learning: union consumption is catastrophic on Los (9.78–10.27, worse than physical GCN), per-timestep consumption is the best learned configuration (4.84–6.26), and Mix gating adds a further 0.30–0.42. Symmetrization (cGSL vs GSL) provides the same lesson in miniature for the symmetric GCN aggregator only.
5. **Confounds controlled.** Sparsity alone does not explain the gain (matched-budget controls: RandTop30 6.10±0.12 and CorrTop30 5.39±0.10 vs MultiGSL 4.84±0.11 and Mix 4.49±0.14 on Los PH1); capacity does not explain it (parameter-matched control retained); the physical graph's density — not its origin — drives its failure (Corr-K=8 control retained). All controls stay Los-PH1-scoped, exactly as previously drafted.
6. **Boundary.** The benefit is **dataset-dependent**; the SZ-Taxi null is reported as a finding (candidate mechanism: learnability — 2 vs 30 cross-sensor lag edges at the operating threshold), with temporal resolution ruled out by the 15-minute Los-loop experiment (its earlier numbers may be retained as a protocol variant pending the provenance note in §8.4).
7. **What the learned graph represents.** Lag-specific statistical dependency structure between contemporaneous traffic observations — descriptive, no causal language anywhere; static graphs, with only the gate adapting over time.
8. **History.** The submitted version's single-run results are treated as a contemporaneous re-baselining, reproduced in relative terms and preserved in an appendix — never mixed numerically with Stage 40 results.

**Not claimed anywhere:** universal benefit; causal structure; contemporaneous-graph superiority over NoSpatial; significance beyond n=5 caveat framing; sparsity as the operative variable; DAGMA "discovering" traffic causality.

---

## 3. Proposed Results section hierarchy

Retain the previous revision's seven-subsection skeleton (it already answers the six Stage 43 narrative questions in order) with two structural changes: create a dedicated controls subsection (current §5.1's controls block + §5.4's parameter control), and re-scope the final subsection to the full canonical method grid. Numbers per cell: Stage 40 five-seed mean ± std (RMSE primary; MAE in appendix). Methods referenced by exact canonical names.

```
5 Results
├── 5.1 Dense Physical Graphs Oversmooth: The Baseline Landscape
│     Stage 40: 12 methods × 4 PH on both datasets (Tables T1/T2);
│     physical vs NoSpatial vs contemporaneous vs multi-lag.
│     [answers Q1; supersedes current §5.1 with the full grid]
├── 5.2 The Single-Graph Baseline Revisited (contemporaneous vs lagged)
│     T-GCN-GSL/cGSL + GCN-GSL/cGSL from Stage 40; submitted-version
│     reproduction paragraph (relative agreement, Stage 42 §2);
│     contemporaneous construction vs multi-lag construction.
│     [answers Q1; supports Q5; absorbs the reproduction paragraph of current §5.3]
├── 5.3 Controlling for Sparsity and Capacity
│     matched-budget controls (RandTop30 / CorrTop30 vs MultiGSL / Mix),
│     union-consumption evidence, parameter-matched control.
│     [answers Q2; merges current §5.1 controls block + §5.4 control]
├── 5.4 From Single-Graph GSL to Multi-Lag Structure
│     Lag ablation + lag-graph statistics (retained);
│     unified with the controlled-consumption argument of §5.3.
│     [answers Q2/Q3; supersedes current §5.2]
├── 5.5 Multi-Seed Validation on Los-loop
│     Stage 40 five-seed paired data: NoSpatial vs MultiGSL vs Weighted
│     vs Mix (4.8408/4.8338/4.4914 vs 5.2514 at PH1); per-seed table/box plot.
│     [replaces current §5.4's seed table with canonical values]
├── 5.6 Beyond 5 Minutes: The 15-Minute Sampling Variant
│     Earlier-pipeline five-seed results retained provisionally
│     (pending the provenance note in §8.4).
│     [unchanged structurally from current §5.6]
├── 5.7 Dataset Dependence: the SZ-Taxi Boundary
│     Stage 40 canonical SZ grid (T1 SZ half); T-GCN-MultiGSL-Mix deltas
│     −0.0109/−0.0112/−0.0143/−0.0080 vs NoSpatial; win counts 4/5, 4/5,
│     5/5, 4/5; paired-t p 0.064–0.908 (PH4 p=0.32); described as a null.
│     [replaces current §5.7's earlier-pipeline numbers]
```

Six-narrative-question mapping: (1)→5.1/5.2, (2)→5.3, (3)→5.3/5.4/5.6, (4)→5.7, (5)→5.2/5.4, (6)→5.7 + Limitations.

**Reviewer-comment alignment note (for the response letter):** R1-W8 asked for consolidation of the old repetitive Section 5 and explicit tie-back to the Introduction's framing; the dedicated 5.3 (controls) and the explicit Q1–Q6 mapping give each subsection one job, and the Discussion (§9 below) carries the tie-back.

### 3.1 Structural options considered and rejected

1. **Full restructure into (i) Baseline landscape → (ii) Consumption mechanism → (iii) Boundary** — rejected: it would discard the previous revision's reviewer-aligned prose (R1-W8/R1-W9 fixes), force a full rewrite of tables already written against the canonical method grid, and reset the mature point-by-point response.
2. **Results-minimal layout** (main text: T-GCN family only; GCN family to appendix) — rejected: the GCN family is needed in the main text to substantiate the union-vs-per-timestep dissociation (the core mechanism claim, R1-W4/R1-Q4) and the cGSL reframing (R2-4-related), and both reviewers asked for *more* analysis space, not less.

---

## 4. Main tables (minimum: four)

All values **five-seed mean ± std (population std, seeds 42–46)** from Stage 40 canonical aggregates (`gsl_stage41/stage41_summary.csv`). Improvement percentages relative to T-GCN (physical baseline) unless stated. RMSE only in main text (MAE to appendix; RMSE is the stated primary metric).

**T1 — Main results: T-GCN family (Los-loop and SZ-Taxi, PH1–4).** The paper's primary evidence.
- Purpose: establish the full T-GCN evidence base on canonical numbers — the physical-graph failure, the NoSpatial verdict on contemporaneous graphs, and the MultiGSL result.
- Rows: T-GCN, T-GCN-NoSpatial, T-GCN-GSL, T-GCN-cGSL, T-GCN-MultiGSL, T-GCN-MultiGSL-Weighted, T-GCN-MultiGSL-Mix (7 canonical methods).
- Columns: Method | Los PH1–4 (4 cols) | SZ PH1–4 (4 cols). Variant: two half-tables (Los / SZ) if width requires.
- Values: mean ± std (RMSE; Los 3 decimals, SZ 4 decimals). **Include** a compact improvement block vs T-GCN on Los (Mix: 42.98/37.60/33.72/32.34%) and vs T-GCN-NoSpatial (Mix: +14.2/+12.1/+9.2/+10.8%). No per-seed columns in T1 (those live in T3/F3).
- Note: replaces `tab:multiph` and `tab:sz_multiph` with the canonical grid; includes every previously missing method.

**T2 — GCN family (both datasets, PH1–4).**
- Purpose: the consumption-dissociation evidence (GCN-MultiGSL vs T-GCN-MultiGSL on identical graphs) and the cGSL reframing.
- Rows: GCN, GCN-NoSpatial, GCN-GSL, GCN-cGSL, GCN-MultiGSL (5 canonical methods).
- Columns: Method | Los PH1–4 | SZ PH1–4.
- Values: mean ± std. No improvement block (GCN comparisons are discussed, not headlined; the family is supporting evidence).
- The reviewer-requested graph statistics (edges, density, mean degree) go to a standalone appendix table (A2), keeping T2 methods-only.

**T3 — Multi-seed paired validation, Los-loop PH1 (the statistical-evidence table).**
- Purpose: honest per-seed evidence for the central claim, replacing `tab:multiseed`.
- Rows: T-GCN-NoSpatial, T-GCN-MultiGSL, T-GCN-MultiGSL-Weighted, T-GCN-MultiGSL-Mix (4 canonical methods; the earlier draft's table had 3).
- Columns: Method | S42 | S43 | S44 | S45 | S46 | mean±std.
- Values: per-seed RMSE + mean±std; **no p-values** in the table (paired-t p ≤ 0.0015 and the Wilcoxon n=5 floor are discussed in text with the explicit caution that n=5 is small).
- Include one summary line: "T-GCN-MultiGSL-Mix beats T-GCN-NoSpatial in 5/5 paired seeds at every horizon."

**T4 — Sparsity and capacity controls (Los-loop, PH1).** Retained from the previous revision, values largely unchanged.
- Purpose: rule out sparsity and capacity confounds (R1-W5).
- Block A (matched 30-edge budget): RandTop30 6.10±0.12; CorrTop30 5.39±0.10 (both earlier-pipeline, five seeds); T-GCN-MultiGSL 4.8408±0.1146 and T-GCN-MultiGSL-Mix 4.4914±0.1403 (canonical); plus a T-GCN-NoSpatial reference row — canonical 5.2514±0.1863, or earlier-pipeline 5.234±0.090 if the controls are compared within-pipeline (decision in §11.2). Improvement figures vs NoSpatial become 7.9%/14.2% if computed against the canonical reference.
- Block B (capacity): NoSpatial H=64 (12,672 params) 5.143; H=74 (16,872) 5.137; Mix (17,091) 4.458 — seed-42 values, caption must state "single seed" explicitly, as previously drafted.
- No improvement block; the text states the comparisons (e.g., "the random 30-edge graph is worse than no graph").
- Provenance labeling per §8.4 (controls predate the canonical protocol).

**Tables removed from the main text:** `tab:oversmoothing` (subsumed by T1/T2 + T4), `tab:multiseed` (→T3), `tab:multiph` (→T1), `tab:sz_multiph` (→T1). All are preserved in snapshot/git history; nothing is deleted.

**Table count rationale:** T1 (primary) + T2 (mechanism) + T3 (statistical honesty) + T4 (confounds) = the minimum set that answers all six narrative questions with canonical numbers; every existing main-text table maps onto one of them.

### 4.1 Data-migration checklist (earlier pipeline → Stage 40 canonical)

Per-cell mappings that **must** be corrected when migrating (verified against `gsl_stage41/stage41_summary.csv` and `stage41_paired_tests.csv`):

| Location | Earlier-pipeline value | Stage 40 canonical | Action |
|---|---|---|---|
| Los PH1 NoSpatial (T3, F3, text) | 5.234±0.090 | 5.2514±0.1863 | replace |
| Los PH1 MultiGSL | 4.794±0.102 | 4.8408±0.1146 | replace |
| Los PH1 Mix | 4.452±0.143 | 4.4914±0.1403 | replace |
| Los PH1–4 Mix vs NoSpatial deltas | −0.78/… (single-horizon basis) | −0.7600/−0.6843/−0.5635/−0.7173 | replace |
| Los PH1–4 Mix vs NoSpatial improvements | 14.9% (PH1), "7.1%" Mix-over-MultiGSL | 14.2/12.1/9.2/10.8%; MultiGSL-over-NoSpatial 7.9%, Mix-over-MultiGSL 7.2% | replace |
| SZ PH1–4 Mix win counts vs NoSpatial | 5/5/5/3 | 4/5, 4/5, 5/5, 4/5 | replace |
| SZ PH1–4 Mix paired-t p-values | 0.064–0.907 | 0.0639–0.9080 (PH4 p=0.315) | replace |
| Abstract "14.9% → 27.4%" | earlier pipeline | 14.2% (PH1) / 10.5% mean PH1–4 vs NoSpatial; 15-min figure pending §11.3 | replace |
| Abstract "21.7% five-seed repro of submitted" | earlier pipeline | 25.7/22.5/20.5/18.7% (Los, T-GCN-GSL vs T-GCN), mean ≈21.9% | replace |
| "Removing the graph improves 32.8%" | seed-42 earlier | 33.3/29.2/27.0/24.1% (canonical, T-GCN Los) | replace |
| `tab:gsl_baseline` T-GCN-GSL row | 5.792±0.196 … | 5.8589±0.2058 …; improvements 25.7/22.5/20.5/18.7% | replace |

Correct-as-is (verified against Stage 40/41 and the Stage 43 brief): Los lag stats 12/3/15 (union 28 distinct off-diagonal); SZ lag edges 0/0/2 (union 2); parameter counts 12,672/16,872/17,091; DAGMA determinism audit claims; the contemporaneous-vs-lagged construction distinction; the static-graphs/adaptive-gate distinction; graph statistics (Los physical 2833 entries incl. 207 self-loops, off-diag density ≈0.0616, symmetric; SZ 532 off-diagonal entries, no self-loops, density ≈0.0220, not exactly symmetric; contemporaneous DAGMA 28/8 edges per PH; DAGMA graphs self-loop-free, self-loops added internally by graph convolution).

**Protocol reconciliation needed before migration (see §11.1):** current §5.3 says λ₁=0.02 with "ω=0.3 applied inside the fit"; current §4.2 says λ₁=0.01, threshold τ=0.1. Verify which values correspond to the Stage 40 canonical graph artifacts (Stage 40.3 report) and label the τ=0.1/30-edge description as earlier-pipeline if that is what it belongs to.

### 4.2 Figures to reuse vs regenerate

- **Reuse unchanged:** fig1 (graph comparison: 2833 vs 28 distinct off-diagonal edges; mean degree 12.7 vs 0.14); fig7 (lag edge stats, Jaccard overlap, weight distributions); fig9 (predicted-vs-actual, seed-42 inference artifact unaffected by aggregate migration).
- **Regenerate from Stage 40 per-seed values:** fig3 → F3 (multiseed box plot; means must match T3 exactly).
- **Verify before reuse:** appendix convergence summary (`appendix/convergence.tex`) against Stage 40 per-run JSONs; regenerate only if curves differ materially.
- **New:** F4 (dissociation figure, §5) — no existing visual covers it.

---

## 5. Main figures (minimum: five, of which only two require work)

**F1 — Physical vs learned graph structure (Los-loop).** Retained (R1-Q1).
- Purpose: make "physical proximity ≠ functional dependency" concrete.
- Plot: (a) physical adjacency (2833 edges incl. 207 self-loops), (b) multi-lag union (28 distinct off-diagonal edges), (c) node-degree distributions (mean degree 12.7 vs 0.14).
- Dataset: Los-loop. Methods: graphs only, no forecasters.

**F2 — Sparsity controls (Los-loop PH1).** Retained in content, values updated (R1-W5).
- Purpose: rule out the sparsity confound visually.
- Plot: bar chart, mean ± std over seeds 42–46, five bars: RandTop30 6.10±0.12, CorrTop30 5.39±0.10, T-GCN-MultiGSL 4.84±0.11, T-GCN-MultiGSL-Mix 4.49±0.14, T-GCN-NoSpatial 5.25±0.19 (canonical reference).
- Dataset: Los-loop PH1. Methods: the five above only.
- Caveat: mixed provenance (controls earlier-pipeline, NoSpatial canonical) must be annotated per §8.4, or the comparison kept within-pipeline (decision in §11.2).

**F3 — Per-seed distributions (Los-loop, all horizons).** Regenerated from Stage 40 (R1-W6).
- Purpose: show the 5/5 paired-seed consistency of Mix > NoSpatial at every horizon.
- Plot: box/strip plot, RMSE over seeds 42–46 for T-GCN-NoSpatial, T-GCN-MultiGSL, T-GCN-MultiGSL-Weighted, T-GCN-MultiGSL-Mix; four panels (PH1–4); means must match T3 exactly; annotate win counts and "n=5".
- Dataset: Los-loop; the SZ contrast panel goes to the appendix (A4).
- Methods: 4 only — T-GCN-GSL/cGSL excluded (they sit above NoSpatial and would compress the scale).

**F4 — The consumption dissociation (new; the mechanism figure).**
- Purpose: visualize the paper's core mechanistic finding (identical graphs, different consumption → opposite outcomes).
- Plot: bar/box chart, Los-loop PH1 (optionally PH2–4): GCN-MultiGSL (union) 9.78 vs T-GCN-MultiGSL (per-timestep) 4.84, with physical GCN 8.14 / physical T-GCN 7.88 as reference lines or bars; annotate "+Mix: −0.35, 5/5 seeds".
- Dataset: Los-loop primary; SZ panel in appendix A4.
- Methods: selected only — GCN-MultiGSL, T-GCN, T-GCN-MultiGSL, T-GCN-MultiGSL-Mix (+ physical references).
- `previous_revision` has **no** dissociation figure — this is the one genuinely new visual element, and the strongest single piece of evidence for the revised story.

**F5 — Predicted vs actual (Los-loop, PH1, seed 42).** Retained (R1-Q2): three high-variance nodes, 100-step window, r=0.95 vs 0.93, per-node windowed correlations in text.

**Figure count rationale:** F1–F5 minimum; only F3 (regenerate) and F4 (new) require work. Optional demotions: the lag-ablation bar chart (current fig5) may move to the appendix — its content overlaps F1/T-blocks and `tab:lag_ablation` is retained; F2 may become a table block inside T4 without loss.

---

## 6. Appendix/supplement material

Retain the existing appendix layout, with these changes:

- **A1 `original_gsl_results.tex` — keep, with a provenance heading.** All submitted-version tables, figures, protocol, and the GSL/cGSL (symmetrization formula) definition live here. Add heading: "Historical results from the first version of the manuscript" (approved phrase). Inside, label the old numbers as the *contemporaneous re-baselining* (single run per cell, earlier pipeline, historical graphs). Do not mix numerically with Stage 40.
- **A1c — NEW "Relationship to the first version" note** (half page): the language of §8.1 below.
- **A2 — Graph statistics table (NEW).** Reviewer-requested density/degree statistics (values listed in §4.1 "correct-as-is"), plus the multi-lag per-lag breakdown (Los 12/3/15; SZ 0/0/2) and the note that DAGMA graphs have no self-loops while graph convolution adds them internally via the Laplacian.
- **A3 — Convergence summary.** Retained; verify against Stage 40 per-run JSONs; regenerate only if materially different.
- **A4 — Cross-dataset per-seed distributions (NEW, optional).** The SZ-panel box plots (F3 contrast) + the full canonical MAE tables (T1/T2 companion, mean±std), keeping the main text RMSE-only.
- **A5 — Bibliometric appendix.** Retained unchanged (`appendix/bibliometric.tex`; R1-W1 resolution stands).
- **A6 — Additional diagnostics** (`appendix/additional_diagnostics.tex`, incl. the threshold-sensitivity note): retained; the τ/ω protocol reconciliation (§11.1) determines its final wording.
- **The 15-minute variant** (current §5.6): retained in main text (it answers R1-W7 partially and rules out temporal resolution as the dataset-dependence explanation); its tables remain earlier-pipeline pending §8.4 labeling.
- **Not to be added:** per-epoch curve grids (both reviewers rejected them); the full 480-run raw table (repository only); the Stage 40–42 audit reports (internal, repository only — cite availability, do not print).

---

## 7. Claims to retire/reframe (Stage 42 reconciliation applied to the draft)

The previous revision is already clean on the major items (NoSpatial central, no causal claims, no "adaptive graph" language, cGSL demoted to appendix, abstract misattribution fixed). Items needing explicit action:

1. **"Sparsity is necessary but not sufficient" (current results.tex §5.1 paragraph).** Scope to Los PH1 only — the matched-budget controls are Los-PH1-only and must not be generalized (Stage 43 brief). Reframe as: "On Los-loop PH1, at a matched 30-edge budget, neither a random nor a top-correlation placement reproduces the learned-graph gain; sparsity is not the operative variable in this controlled comparison." The seed-42 union-consumption control (5.928) and the 60-edge thresholded control are earlier-pipeline artifacts pending §11.2 verification — either verify them against canonical artifacts or re-scope the sentence to the canonical union-consumption evidence (GCN-MultiGSL) and drop the unverifiable one.
2. **"The advantage persists but narrows at longer horizons" (current results.tex, `tab:multiph` paragraph).** Retire — Stage 40 Los deltas (−0.7600/−0.6843/−0.5635/−0.7173) are non-monotonic (smallest at PH3). Supported phrasing: "the advantage persists at all four horizons (5/5 paired seeds), ranging from 9.2% to 14.2% relative to T-GCN-NoSpatial."
3. **"The improvement is robust: it holds in all five seeds at both sampling intervals…" (conclusion.tex item 5).** Reframe to Los-loop scope, and update the conclusion's SZ phrasing with canonical values (0.26/0.27/0.34/0.19%, win counts 4/5–5/5) and the null framing.
4. **SZ-Taxi "directionally consistent" framing.** Canonical win counts (4/5, 4/5, 5/5, 4/5) weaken the draft's "5/5 at each of PH1–3"; migrate numbers and phrase as "directionally consistent but within seed variability; we report it as a null with respect to practical benefit." Keep the candidate-mechanism paragraph (graph learnability, 2 vs 30 edges) as the honest interpretation.
5. **Conclusion item 3 "Multi-lag DAGMA explicitly discovers these lag-specific dependencies."** Soften "discovers" → "yields" ("Multi-lag DAGMA yields lag-specific dependency estimates…"), aligning with the terminology restrictions.
6. **Conclusion item 7** (sparsity conclusion) is already Los-scoped ("at a matched 30-edge budget") — correct as written; only number migration needed.

**Claims already correct (verified, do not touch):** NoSpatial as decisive baseline; contemporaneous-vs-lagged distinction; static-graphs/adaptive-gate distinction; no causal language (limitations item 8 is the only "causal" occurrence, disclaiming); GSL/cGSL in appendix; "the first version of the manuscript" phrasing; abstract numbers corrected from the submitted version's misattribution (R2-1).

---

## 8. Relationship between submitted and new results

### 8.1 Precise integration language (recommended wording)

For the revised manuscript (adapted from the previous revision's reviewer-approved §5.3 template):

> "The first version of the manuscript evaluated GSL and cGSL under an earlier protocol with a single run per configuration (Appendix~A). We treat those results as a contemporaneous re-baselining: the *relative* effects it reported — learned graphs outperforming the dense physical graph, and the improvement magnitude of T-GCN-GSL over T-GCN on Los-loop — are reproduced under the canonical five-seed protocol to within approximately one percentage point (e.g., 21.8% → 21.9% mean across PH1–4 on Los-loop), while its absolute RMSE values are not comparable across protocols. The historical tables are preserved unchanged in Appendix A; all tables in the main text use the canonical five-seed results."

Variant for the response letter:

> "We preserved the first version's results unchanged in Appendix A as a contemporaneous re-baselining, and independently reproduced its relative claims under a fully documented five-seed protocol (mean improvement of T-GCN-GSL over T-GCN on Los-loop: 21.8% → 21.9%) before extending the evidence with the graph-free and multi-lag baselines that were missing from the first version. Absolute RMSE values are not comparable across protocols, which is why the two sets of numbers are never mixed."

### 8.2 What moves where (relative to `previous_revision`)

- Current §5.3's reproduction paragraph and `tab:gsl_baseline` → §5.2 with canonical values (§4.1 mapping).
- Current §5.4's `tab:multiseed` → T3 with canonical per-seed values.
- Current §5.1's controls block → new §5.3 with T4.
- Current `tab:sz_multiph` → T1 (SZ half), canonical values, null framing.
- Current §5.2 (lag stats) → §5.4, numbers verified canonical (12/3/15, union 28).
- `Response-to-Reviewers.md` — re-verify each quoted number against Stage 40/42 before resubmission (§8.3).

### 8.3 Response-letter numbers that must be corrected

The "14.9%/27.4%" headline → canonical 14.2% (PH1, vs NoSpatial, Los) with the 15-minute figure pending §11.3; SZ win counts "5/5, 5/5, 5/5, 3/5" → "4/5, 4/5, 5/5, 4/5"; "no stable difference at SZ PH4 (3/5)" → "4/5 (not stable; paired-t p=0.32)". The R2-1 misattribution fix is unaffected (submitted-version issue, resolved in the previous revision).

### 8.4 Pipeline-labeling rule for retained earlier-pipeline blocks

The 15-minute variant and the matched-budget controls (RandTop30/CorrTop30, seed-42 union/thresholded/capacity controls) predate the canonical protocol. Every table or figure containing them carries a label; suggested manuscript note:

> "The 15-minute-sampling experiment and the matched-budget sparse-graph controls predate the canonical five-seed protocol and use the earlier training pipeline. Their role is protocol-variant evidence (sampling-interval robustness) and confound control, not headline claims; headline claims rest exclusively on the canonical five-seed protocol."

### 8.5 What does NOT migrate

- Submitted-version numbers stay in Appendix A1, frozen, under the contemporaneous re-baselining label.
- The earlier-pipeline multiseed table (`tab:multiseed`) is replaced by T3; it survives only in snapshot history.
- No earlier-pipeline number appears unlabeled in the main text after migration; every retained earlier-pipeline block is pipeline-labeled in its caption.

---

## 9. Required changes outside Results

- **Abstract:** replace both headline percentages with canonical values ("14.2% at PH1, 10.5% mean across PH1–4, over a no-graph baseline across five random seeds at 5-minute sampling"); the 15-minute figure (currently 27.4%) is retained only with the §8.4 label or recomputed under the canonical protocol before use. Retain the existing abstract's correct elements (DAGMA, lag blockings Z=[x(t−L),…,x(t)], NoSpatial reference, dataset-dependence sentence, no causal language).
- **Introduction:** paragraph 6 (contributions) and paragraph 7 (results preview) numbers → canonical. Structure and citations unchanged; R1-W1 resolution stands.
- **Contributions list (Introduction):** add a third contribution bullet: "a controlled dissociation showing that how learned graphs are consumed matters at least as much as how they are learned (identical multi-lag graphs consumed by union (GCN-MultiGSL) and per-timestep (T-GCN-MultiGSL) mechanisms with opposite outcomes)" — Stage 40's strongest new structural finding, currently absent from the draft.
- **Method (§3.4, Key Properties):** add T-GCN-MultiGSL-Weighted to the variants list (canonical variant, currently missing); verify the "fixed cyclic pattern" description of T-GCN-MultiGSL against the Stage 40.3 report before final wording (§11.4).
- **Experiments (§4.2):** verify λ₁ and threshold values against the Stage 40 protocol (§11.1). **§4.4 method list:** add the GCN-family methods (or state that the GCN family is analyzed in the T2 subsection) and add T-GCN-MultiGSL-Weighted.
- **Experiments (§4.5 Reproducibility):** add one sentence anchoring the canonical protocol: "All main-text results are five-seed (42–46) canonical aggregates; DAGMA-linear is deterministic (dedicated determinism audit, bit-identical across processes; agreement to ~1 ULP across thread counts with identical thresholded support); every reported table states its seed count." The existing historical-reproduction sentence (support-level reproduction, 28/28/26/28, ~0.016 weight agreement) is verified and stays.
- **Discussion:** §6.1 "32.8%" → canonical 33.3% (Los PH1, five-seed means); §6.2 "4.794/4.452" → 4.8408/4.4914 with "7.9% (MultiGSL over NoSpatial) and a further 7.2% (Mix over MultiGSL)"; §6.3 SZ percentages → 0.26/0.27/0.34/0.19% with win counts 4/5–5/5; §6.4 "4.794/4.452" → canonical; §6.5 stands (contemporaneous re-baselining language verified).
- **Limitations:** item 5 (λ sensitivity, "no systematic sweep") stands; item 9 (PH1–4 scope) stands; item 3 (scalability runtimes) stands; add one sentence to item 1: "the multi-lag benefit is absent on SZ-Taxi at the canonical operating point (deltas ≤ 0.015 RMSE, 4/5–5/5 wins, within seed variability)". Item 9's "growing relative improvements with horizon" refers to the 15-minute variant — keep under the §8.4 label.
- **Conclusion:** items 4–7 numbers → canonical (§4.1 mapping); item 5 Los-scoped robustness phrasing (§7 item 3); item 3 "discovers" → "yields" (§7 item 5).
- **Response-to-Reviewers.md:** update all quoted numbers (§8.3); add a "canonical protocol" paragraph; the R1-W5 controls answer stands with §8.4 pipeline labels; R1-W6 gains "canonical five-seed aggregates" phrasing; R1-W7 unchanged; R1-W8 gains the new T1/T2 structure description; R2-1 through R2-6 stand.
- **Keywords, Acknowledgements:** unchanged.

**Terminology audit (previous_revision is already compliant — verify at final pass):** no "Original GSL/Original cGSL/Adaptive" method names; no "causal graph/causal relationship/hidden causal structure"; "the first version of the manuscript" for the submitted version; "statistical dependency"/"lag-specific structure" language; exact canonical method names throughout; table row labels use the "T-GCN (physical graph)" style — "Physical" describes the graph, never a method name.

### 9.1 Pipeline-labeling sentence (for every mixed-provenance table)

> "Values are five-seed mean±std under the canonical protocol; blocks labeled 'earlier pipeline' predate the canonical protocol and are retained for protocol-variant and confound-control evidence."

---

## 10. Recommended revision order

1. **Freeze `previous_revision` as the base** — do not draft from scratch in `revised_version/`; copy the snapshot into `revised_version/` when revision work begins (populating `revised_version/` is the next stage's work, not this one).
2. **Protocol reconciliation first** (§11.1–11.3): verify λ₁, τ/ω, and which graph artifacts the canonical runs used (Stage 40.3 report); decide provenance labeling for the 15-minute variant and the controls.
3. **Migrate T1/T2/T3 numbers** (§4.1 checklist); build T4/F2 provenance labels; regenerate fig3 → F3 from canonical per-seed values; design F4.
4. **Rewrite Results prose** per the §3 hierarchy (§5.1–5.7), applying the §7 claim fixes.
5. **Update the response letter** (§8.3): canonical numbers, canonical-protocol paragraph, pipeline labels for retained earlier-pipeline blocks.
6. **Update abstract/introduction/method/experiments/discussion/limitations/conclusion** (§9) — number migration + claim scoping; no structural change outside Results beyond the added contribution bullet and the Weighted-variant mention.
7. **Terminology pass** (§9 audit list) + citation-style pass (R1-W12) + typo sweep — one combined pass at the end.
8. **Verification pass:** every number in the draft must trace to `gsl_stage41/stage41_summary.csv` (canonical) or a labeled earlier-pipeline source (§8.4); no unlabeled earlier-pipeline number anywhere in the main text.

---

## 11. Open issues that must be resolved before manuscript rewriting

1. **Protocol reconciliation (blocking).** Current §5.3 says λ₁=0.02 with "ω=0.3 applied inside the fit"; current §4.2 says λ₁=0.01, threshold τ=0.1. Which values correspond to the Stage 40 canonical graph artifacts (28-edge Los contemporaneous graphs, 12/3/15 multi-lag)? If the canonical runs used ω=0.3 inside-fit, the draft's τ=0.1/30-edge description belongs to the earlier pipeline and must be labeled as such; also verify hidden dim, epochs, batch size against the Stage 40 training configuration.
2. **Control-experiment provenance (blocking).** Confirm whether RandTop30/CorrTop30 (6.10±0.12 / 5.39±0.10) and the seed-42 union (5.928), thresholded 60-edge, and capacity controls (5.137/4.458) are part of the canonical protocol or earlier-pipeline artifacts. If earlier-pipeline: apply §8.4 labeling and decide the NoSpatial reference for improvement figures (canonical 5.2514 → 7.9%/14.2%; earlier-pipeline 5.234 → 8.4%/14.9% within-pipeline).
3. **15-minute variant provenance (blocking for the abstract).** The 27.4% headline rests on the earlier pipeline. Under the no-new-experiments constraint, the only Stage 43-compliant option is §8.4 labeling; a canonical-protocol rerun can only be proposed as explicitly future work.
4. **T-GCN-MultiGSL architecture description.** Verify the draft's "fixed cyclic pattern" description of T-GCN-MultiGSL against the canonical Stage 40.3 implementation before Method §3.4 wording is finalized (the consumption-mechanism claim depends on describing it correctly).
5. **T1 layout.** A 9-column table (Method | Los PH1–4 | SZ PH1–4) may need two half-tables, `table*`, or sideways placement — formatting decision only, no content impact.
6. **MAE placement.** Confirm appendix-only (A4) and that MAE orderings mirror RMSE for the canonical methods (checked in `gsl_stage41/stage41_summary.csv` for the cells cited in the plan).
7. **Response-letter sweep.** Every quoted number re-verified against Stage 40/41/42 (§8.3 list), and section/table references renumbered if Results subsections change.
8. **Figure tooling.** `previous_revision` contains `generate_figures.py` / `generate_figures_extra.py`; decide whether to extend them for F3/F4 from the Stage 40 per-run JSONs (code changes belong to the revision stage, not this one).
9. **Equivalence targets.** Define numeric acceptance criteria before migration: T3 means equal `stage41_summary.csv` to 4 decimals; per-seed values match the per-run JSONs; F3/F4 annotations match T3/T1 exactly.
10. **`revised_version/` population strategy.** Copy `previous_revision` as the base vs a fresh modular tree — recommend copy-then-migrate to preserve the reviewer-response mapping; final LaTeX flatten for submission per the note in the snapshot's `sn-article.tex` header.

---

*End of Stage 43 planning report. No manuscript, code, table, figure, or result file was modified, created, or deleted in this stage except this report.*
