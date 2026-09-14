# Stage 42 — Submitted-Manuscript vs Stage 40 Canonical Results: Consistency Audit

**Date:** 2026-09-10
**Scope:** read-only scientific consistency audit between the submitted manuscript's reported results and the Stage 40 canonical result set.
**Constraints honored:** no code modified, no experiments rerun, no DAGMA graphs refitted, no manuscript files changed. All Stage 40 numbers below are computed directly from the 480 per-run JSONs in `results/stage40_canonical/` (5 seeds 42–46 per cell), aggregated in `gsl_stage41/stage41_summary.csv`; paired tests are from `gsl_stage41/stage41_paired_tests.csv`. Submitted numbers were verified digit-for-digit against `paper/sn-article_original.tex` and `paper/appendix/original_gsl_results.tex` (all 48 cells match; see `gsl_stage42/stage42_comparison.csv`).

---

## 1. Executive verdict

**The Stage 40 canonical results are a rigorous, seed-replicated refinement of the submitted evidence — not a contradiction of it — but they overturn one specific submitted claim (the Los-loop T-GCN PH1 "GSL ≫ cGSL" ordering) and, more importantly, reveal that the submitted comparison omitted the one baseline that determines the scientific story: NoSpatial.**

Four headline facts:

1. **GCN family, SZ-taxi: essentially identical.** All 16 cells reproduce within ±0.6%, every submitted ranking preserved. The old single-run numbers were simply a seed draw from the same distribution.
2. **GCN family, Los-loop: direction preserved, level shifted.** Stage 40 means run 2–9% higher than the submitted single run, but every submitted ranking and every submitted baseline relation survives. The submitted improvement percentages are reproduced almost exactly.
3. **T-GCN family: the baseline rows shifted the most, and one submitted ordering failed.** Submitted T-GCN baselines are 12–23% lower than the Stage 40 five-seed means (the submitted single runs were unusually favorable draws for the baseline). The Los-loop PH1 "GSL 4.818 < cGSL 6.550" ordering reverses (Stage 40: GSL 5.859 vs cGSL 5.821 — a within-noise tie). PH2–PH4 orderings and the submitted headline gain of T-GCN-GSL over T-GCN (+21.8%) are preserved almost exactly.
4. **The decisive finding is structural, not numerical.** Stage 40 adds GCN-NoSpatial and T-GCN-NoSpatial — identity adjacency, no graph at all — and these *beat every learned-graph variant on both datasets*, including the submitted manuscript's flagship configurations (GCN-cGSL, T-GCN-GSL), often by large margins. The submitted manuscript's central claim — that learned graphs (GSL/cGSL) drive the improvement — cannot survive this comparison: on the current evidence, the gains the submitted manuscript attributed to learned graph structure are attributable to a different mechanism, not to learned graphs (see §4 and §6).

Classification of all 48 submitted-vs-Stage40 cells (see `stage42_comparison.csv`):

| Classification | Count |
|---|---|
| essentially consistent (|Δ| ≤ 0.5% or |Δ| ≤ 1σ) | 20 |
| consistent in qualitative trend but numerically different | 25 |
| materially inconsistent | 3 |
| directly contradictory | 0 |

The three materially inconsistent cells: Los-loop PH1 T-GCN-GSL (submitted rank 1 → Stage 40 rank 2), Los-loop PH1 T-GCN-cGSL (submitted rank 2 → Stage 40 rank 1; the submitted GSL≪cGSL ordering collapses to a tie), and SZ-taxi PH2 T-GCN-cGSL (submitted "worse than T-GCN" → Stage 40 "better than T-GCN").

**Final verdict (§14): READY WITH IMPORTANT CAVEATS** — the revised manuscript can tell a coherent story in which Stage 40 refines the submitted evidence, but only if the central claim is substantially rewritten around graph *consumption* mechanisms rather than learned graphs per se.

---

## 2. Submitted vs Stage 40 numerical comparison

Full machine-readable table: `gsl_stage42/stage42_comparison.csv` (48 rows: 2 datasets × 2 families × 4 PH × 3 methods submitted). Summary by family/dataset:

### GCN family — SZ-taxi

| PH | GCN (sub → S40) | GCN-GSL (sub → S40) | GCN-cGSL (sub → S40) |
|---|---|---|---|
| 1 | 5.958 → 5.960 (+0.03%) | 4.886 → 4.880 (−0.12%) | 4.648 → 4.641 (−0.15%) |
| 2 | 5.983 → 5.975 (−0.13%) | 4.904 → 4.910 (+0.12%) | 4.672 → 4.674 (+0.05%) |
| 3 | 5.991 → 5.989 (−0.03%) | 4.958 → 4.930 (−0.57%) | 4.712 → 4.701 (−0.23%) |
| 4 | 6.002 → 6.001 (−0.02%) | 4.933 → 4.958 (+0.50%) | 4.726 → 4.728 (+0.05%) |

All |Δ| ≤ 0.57%; every submitted rank preserved; every submitted baseline relation preserved. **The submitted SZ GCN table is fully reproduced by Stage 40.**

### GCN family — Los-loop

| PH | GCN (sub → S40) | GCN-GSL (sub → S40) | GCN-cGSL (sub → S40) |
|---|---|---|---|
| 1 | 7.724 → 8.145 (+5.4%) | 7.527 → 7.827 (+4.0%) | 5.440 → 5.763 (+5.9%) |
| 2 | 7.940 → 8.539 (+7.5%) | 7.867 → 8.479 (+7.8%) | 5.806 → 6.331 (+9.0%) |
| 3 | 8.102 → 8.765 (+8.2%) | 8.073 → 8.448 (+4.7%) | 6.171 → 6.684 (+8.3%) |
| 4 | 8.285 → 8.765 (+5.8%) | 9.067 → 8.934 (−1.5%) | 6.745 → 6.881 (+2.0%) |

Direction preserved in every cell. GCN-GSL vs GCN: submitted +4.4%/+0.9%/−0.4%/−9.4% relative change; Stage 40 gives −3.9%/−0.7%/−3.6%/+1.9% — same qualitative story (GSL ≈ GCN on Los GCN family; the submitted PH4 "GSL worse than GCN" relation is preserved: submitted worse, Stage 40 8.933 vs 8.765, still worse). GCN-cGSL remains decisively the family's best variant in all PH.

The systematic +2–9% level shift on Los-loop is consistent with the submitted Los-loop runs having been favorable single draws (Stage 40 per-seed spread on Los is substantial, e.g. GCN-cGSL PH2: 5.940–6.781, σ≈0.32; the submitted 5.806 sits below the Stage 40 minimum for GCN-cGSL PH2 — that single submitted value is outside the Stage 40 seed range, i.e. a lucky run).

### T-GCN family — SZ-taxi

| PH | T-GCN (sub → S40) | T-GCN-GSL (sub → S40) | T-GCN-cGSL (sub → S40) |
|---|---|---|---|
| 1 | 4.866 → 5.449 (+12.0%) | 4.214 → 4.281 (+1.6%) | 4.821 → 4.301 (−10.8%) |
| 2 | 4.506 → 5.552 (+23.2%) | 4.239 → 4.309 (+1.7%) | 4.534 → 4.335 (−4.4%) |
| 3 | 4.685 → 5.602 (+19.6%) | 4.344 → 4.333 (−0.3%) | 4.630 → 4.380 (−5.4%) |
| 4 | 4.934 → 5.633 (+14.2%) | 4.366 → 4.370 (+0.1%) | 4.774 → 4.415 (−7.5%) |

T-GCN-GSL reproduces within ±1.7% — excellent. T-GCN-cGSL moves −4% to −11% (the submitted single runs were poor draws for cGSL). The T-GCN baseline moves +12% to +23% (submitted runs were very favorable draws for the baseline). Note the submitted PH2 anomaly is corrected: submitted had T-GCN 4.506 < cGSL 4.534 ("cGSL worse than baseline" — the manuscript's own PH2 row was already inconvenient); Stage 40 has cGSL 4.335 < T-GCN 5.552, so all four PH now show GSL-family < baseline.

### T-GCN family — Los-loop

| PH | T-GCN (sub → S40) | T-GCN-GSL (sub → S40) | T-GCN-cGSL (sub → S40) |
|---|---|---|---|
| 1 | 6.588 → 7.877 (+19.6%) | 4.818 → 5.859 (+21.6%) | 6.550 → 5.821 (−11.1%) |
| 2 | 6.960 → 8.133 (+16.9%) | 5.400 → 6.301 (+16.7%) | 6.915 → 6.354 (−8.1%) |
| 3 | 7.361 → 8.368 (+13.7%) | 5.846 → 6.655 (+13.8%) | 7.331 → 6.669 (−9.0%) |
| 4 | 7.568 → 8.659 (+14.4%) | 6.257 → 7.040 (+12.5%) | 7.539 → 7.088 (−6.0%) |

Two robust invariants survive:

- **T-GCN-GSL vs T-GCN relative improvement:** submitted +26.9%/+22.4%/+20.6%/+17.3%; Stage 40 +25.7%/+22.5%/+20.5%/+18.7%. Agreement to within ~1 point in each PH — the *headline* submitted claim reproduces almost exactly under 5-seed averaging. (On SZ: submitted +13.4%/+5.9%/+7.3%/+11.5%; Stage 40 +21.4%/+22.4%/+22.7%/+22.4% — Stage 40 actually strengthens the SZ-TGCN claim because the submitted baseline rows were unrepresentative.)
- **The Los PH1 ordering failure:** submitted GSL 4.818 ≪ cGSL 6.550 (submitted margin 1.732, huge); Stage 40 GSL 5.859 vs cGSL 5.821 — difference 0.038, far smaller than the seed σ of both (0.206, 0.228). **The submitted Los PH1 "GSL much better than cGSL" claim is not supported by the canonical results; it was a seed artifact.** This is the single clean numerical contradiction among submitted claims.

Everything else in the T-GCN family is trend-consistent: GSL ≫ T-GCN (5/5 seed wins in all 8 dataset×PH cells, paired tests in `stage41_paired_tests.csv`), cGSL ≈ GSL on Los (never decisively separated), cGSL slightly behind GSL on SZ.

---

## 3. GCN-family analysis

Stage 40 GCN family (SZ / Los, PH1; see `stage41_summary.csv` for all PH):

| Method | SZ PH1 | Los PH1 |
|---|---|---|
| GCN (physical) | 5.960 | 8.145 |
| GCN-NoSpatial (identity) | **4.114** | **4.880** |
| GCN-GSL | 4.880 | 7.827 |
| GCN-cGSL | 4.641 | 5.763 |
| GCN-MultiGSL (union) | 4.823 | 9.780 |

Findings:

- **GCN-NoSpatial is the strongest GCN-family configuration on both datasets in every PH** (SZ: 4.114–4.225 across PH; Los: 4.880–6.258). No learned-graph variant approaches it. GCN-cGSL — the submitted manuscript's GCN flagship — is beaten by the *identity graph* by ~0.5 RMSE on SZ (4.641 vs 4.114) and ~0.9 on Los (5.763 vs 4.880), in every PH.
- **GCN-GSL barely improves on the physical graph** (SZ: 4.880 vs 5.960; Los: 7.827 vs 8.145, and it is *worse* than GCN at PH4 on Los). Under the submitted manuscript's framing ("GSL improves over the physical graph"), this is a weak, dataset-dependent effect — and irrelevant once NoSpatial exists.
- **GCN-MultiGSL is catastrophic on Los** (9.78–10.27, worse than the physical-graph GCN by ~1.5) and no better than GCN-GSL on SZ (4.82 vs 4.88). The multi-lag *union* graph consumed by a single GCN aggregation is a harmful representation on Los.
- What remains true and well-supported in the GCN family: **cGSL (symmetrized) is consistently and substantially better than GSL (directed)** — 4.641 vs 4.880 (SZ PH1), 5.763 vs 7.827 (Los PH1) — with 5/5 paired wins in 7 of 8 dataset×PH cells (the exception: Los PH3 GCN-GSL wins 2/5). Symmetrization helps a *directed* adjacency feed a symmetric aggregation operator; that is a coherent, defensible finding about graph-consumption compatibility.
- **But the GCN family as a whole now functions as a supporting analysis, not a headline.** The honest GCN story is: "the physical graph is much worse than no graph for this shallow aggregator; among learned graphs, symmetrization matters; none of the learned variants recovers the information the identity graph already carries." (Interpretation: with 156–207 nodes and a GCN that must diffuse through the given adjacency, a dense-but-uninformative road-network adjacency actively hurts; the identity graph lets the network rely on node features alone.)

---

## 4. T-GCN-family analysis

Stage 40 T-GCN family (SZ / Los, PH1; see `stage41_summary.csv`):

| Method | SZ PH1 | Los PH1 |
|---|---|---|
| T-GCN (physical) | 5.449 | 7.877 |
| T-GCN-NoSpatial (identity) | **4.120** | **5.251** |
| T-GCN-GSL | 4.281 | 5.859 |
| T-GCN-cGSL | 4.301 | 5.821 |
| T-GCN-MultiGSL | 4.130 | 4.841 |
| T-GCN-MultiGSL-Weighted | 4.130 | 4.834 |
| T-GCN-MultiGSL-Mix | **4.119** | **4.491** |

Findings:

- **T-GCN-NoSpatial beats T-GCN-GSL and T-GCN-cGSL on both datasets in every PH.** SZ PH1: NoSpatial 4.120 vs GSL 4.281 / cGSL 4.301. Los PH1: 5.251 vs 5.859 / 5.821. The margin is small but consistent on SZ (0.15–0.18), large on Los (0.6). This is the structural result that the submitted manuscript could not have seen (NoSpatial was not run) and that rewrites the story: **the contemporaneous learned graph does not add value over no graph at all** for the T-GCN aggregator; if anything it subtracts.
- **T-GCN-MultiGSL and especially T-GCN-MultiGSL-Mix are the only learned-graph configurations that beat NoSpatial — and only on Los.** Los PH1: Mix 4.491 vs NoSpatial 5.251 (−0.760, paired 5/5, p≈0.0015); PH2 −0.684 (p≈0.0015); PH3 −0.563 (p≈0.0008); PH4 −0.717 (p≈0.00002). On SZ the Mix-vs-NoSpatial deltas are +0.0009/−0.0112/−0.0143/−0.0080 — tiny, with 4/5 or 5/5 wins but p-values 0.006–0.9 and Wilcoxon 0.125–0.625. **On SZ, MultiGSL-Mix ≈ NoSpatial; on Los, MultiGSL-Mix > NoSpatial.** This is the dataset-dependence the revised manuscript must be built around.
- **Graph source vs graph consumption are now separable.** GCN-MultiGSL and T-GCN-MultiGSL consume the *same* DAGMA multi-lag graphs; the difference is purely architectural (GCN-MultiGSL merges them into one union adjacency; T-GCN-MultiGSL feeds each lag-specific graph at its matching timestep). On Los, union consumption is disastrous (9.78) while timestep-matched consumption is the best learned result (4.84, and 4.49 with Mix gating). This dissociation is the cleanest scientific finding in the entire Stage 40 evidence base, and it is *about the consumption mechanism*, not the graph source.
- The T-GCN family also confirms the cGSL result from the GCN side: on SZ, cGSL (4.301) is very close to GSL (4.281) — no decisive separation; on Los they are statistically indistinguishable (5.859 vs 5.821). Symmetrization is not the driver of the T-GCN gains.

---

## 5. cGSL analysis

What the evidence supports:

- **Within the GCN family, cGSL ≫ GSL, robustly** (SZ: −0.24 RMSE; Los: −2.06 RMSE at PH1, 5/5 paired wins in 7/8 cells). For a symmetric aggregator, feeding a symmetrized adjacency is mechanistically sensible.
- **Within the T-GCN family, cGSL ≈ GSL** (SZ: +0.02; Los: −0.04 — both within noise). The T-GCN gains do not need symmetrization.
- **cGSL never beats NoSpatial anywhere.** SZ GCN: 4.641 vs 4.114. Los GCN: 5.763 vs 4.880. SZ T-GCN: 4.301 vs 4.120. Los T-GCN: 5.821 vs 5.251.

Verdict on the submitted claim "cGSL is consistently superior to GSL": **partially defensible, family-dependent** — true and strong for GCN, unsupported for T-GCN. The submitted manuscript's stronger framing (cGSL as the centerpiece, "GCN-cGSL is the best method") is no longer defensible in any family.

Recommended status of cGSL in the revised manuscript: **secondary finding / mechanism probe** — evidence that adjacency symmetrization matters specifically when the downstream aggregation operator is symmetric (GCN), used to support the graph-consumption-compatibility argument. It is not a central contribution, and it must never be compared against NoSpatial as if it were a winning configuration.

---

## 6. MultiGSL analysis

- **T-GCN-MultiGSL (fixed per-timestep consumption) on Los** improves over NoSpatial by 0.41–0.72 RMSE per PH (4.841 vs 5.251 at PH1), and **T-GCN-MultiGSL-Mix (per-node gating) improves further** (4.491), for a total learned-graph benefit over NoSpatial of ~0.76 RMSE (PH1) with 5/5 paired wins and p≤0.0015 (with the caveat: five seeds only; the exact Wilcoxon p is 0.0625, the minimum attainable with n=5 — treat significance as indicative, not definitive).
- **On SZ, the same configurations are indistinguishable from NoSpatial** (Mix deltas ≤0.015, mixed win counts, p up to 0.9). There is no evidence of a learned-graph benefit on SZ.
- **The GCN/T-GCN MultiGSL dissociation (same graphs, different consumption)** is the strongest mechanistic evidence available: on Los, per-timestep consumption of lag-specific graphs (T-GCN-MultiGSL: 4.84) vs union consumption (GCN-MultiGSL: 9.78) differs by ~5 RMSE. The graphs are identical; the consumption mechanism is not. Whatever benefit exists is created by *how* the graphs are consumed, in interaction with the temporal architecture that can align graph information with the right lag.
- Therefore the submitted claim "**learned graphs are universally beneficial**" is **directly refuted** in its universal form, but replaced by a more interesting, defensible claim: *learned multi-lag graphs confer a benefit only when (a) the architecture can consume them at matching timesteps, and (b) the dataset has temporal graph structure worth aligning* — which holds on Los-loop and not on SZ-taxi.

---

## 7. Dataset-dependence analysis

| Question | SZ-taxi | Los-loop |
|---|---|---|
| T-GCN-MultiGSL-Mix vs T-GCN-NoSpatial | ≈ (≤0.015, n.s.) | Mix better by 0.56–0.76, 5/5 wins, p≤0.0015 |
| T-GCN-MultiGSL vs T-GCN-NoSpatial | ≈ (−0.01…+0.01) | MultiGSL better by 0.41–0.72 |
| Any learned variant vs NoSpatial | none beats it | Mix and MultiGSL beat it |
| GCN-cGSL vs GCN-GSL | strong cGSL win | strong cGSL win |
| T-GCN-GSL vs T-GCN | large win (both) | large win (both) |

**The conclusion "the benefit of learned multi-lag graph structure is dataset-dependent" is supported.** Precisely: the *only* learned configurations that exceed the identity-graph baseline are T-GCN-MultiGSL / -Weighted / -Mix on Los-loop, with consistent margins (0.4–0.8 RMSE) and 5/5 seed wins across all four horizons. On SZ-taxi the same configurations sit within ±0.015 of NoSpatial — a null result, not a small positive.

Why this is better than the submitted universal claim:

1. It is falsifiable and was almost falsified by half the data (SZ); the submitted claim ignored exactly this kind of cell.
2. It relocates the explanation from "sparsity/graph learning is good" to *interaction*: dataset graph structure × consumption mechanism. Los-loop (207 nodes, denser spatiotemporal dependency) appears to contain lag-specific structure that per-timestep consumption can exploit; SZ (156 nodes) does not — its identity-graph performance is already at the learned-graph ceiling (4.12 vs 4.12).
3. It correctly predicts the pattern in the table: methods that consume graphs in a lag-agnostic way (GCN-GSL, GCN-cGSL, T-GCN-GSL, T-GCN-cGSL — all contemporaneous, i.e. one graph for all lags) show no dataset-dependent advantage over NoSpatial; only the lag-specific consumers (MultiGSL family) show the dataset split. The dataset-dependence is precisely where it should be if the mechanism is temporal alignment of graph information.
4. It avoids causal language: nothing here says DAGMA *discovers* the useful structure, only that per-timestep consumption of the multi-lag graph set correlates with lower error on Los.

---

## 8. Claim-by-claim reconciliation

| # | Submitted claim | Stage 40 evidence | Status |
|---|---|---|---|
| A | "GSL improves over the physical graph" | 8/8 dataset×PH cells show learned < physical (SZ GCN: 4.88 vs 5.96; Los T-GCN: 5.86 vs 7.88; …) | **Defensible in direction, weak in force** — and reframed by NoSpatial: the physical graph is the *problem*, not the reference (both datasets: NoSpatial ≪ physical). Improvement over physical ≠ benefit of learning. |
| B | "GSL is better than NoSpatial" | **Refuted everywhere.** NoSpatial beats every contemporaneous-GSL/cGSL variant on both datasets in all PH; only T-GCN-MultiGSL/-Mix on Los survive | **Not defensible** — the submitted manuscript never ran NoSpatial; when run, the claim fails |
| C | "cGSL is consistently superior to GSL" | GCN: yes, robustly (7/8 cells 5/5 wins). T-GCN: no (deltas ≤0.04, within noise) | **Partially defensible, family-dependent** — retain as a secondary, family-specific finding |
| D | "TGCN-GSL is the best method" | Stage 40: NoSpatial, MultiGSL, Weighted, and Mix all beat it on both datasets; even on the submitted table's own terms it was best only among *submitted* methods | **Not defensible** |
| E | "GCN-cGSL is the best method" | Stage 40: GCN-NoSpatial beats it by 0.5 (SZ) / 0.9 (Los) in every PH | **Not defensible** |
| F | "Learned graphs are universally beneficial" | Refuted on SZ (all learned ≈ or < NoSpatial) and refuted by GCN-MultiGSL on Los (9.78 > 8.14 physical GCN) | **Not defensible in universal form**; replaced by the dataset×consumption interaction claim (§6–7) |
| G | "MultiGSL results provide a stronger, more defensible explanation of the gains" | Yes: the only surviving benefit is T-GCN-MultiGSL(-Mix) on Los; the GCN/T-GCN MultiGSL dissociation (same graphs, different consumption) localizes the mechanism in the consumption path | **Supported — this becomes the paper's core contribution** |

**Bottom line:** the submitted manuscript's central explanatory claim (learned contemporaneous graphs GSL/cGSL drive the improvement) is contradicted by Stage 40; the surviving central claim is the MultiGSL/consumption-mechanism story, which the submitted manuscript did not contain. This is a **substantial revision of the thesis**, not a numerical touch-up.

---

## 9. Which submitted tables/claims must be replaced

1. **All four main results tables** (GCN/T-GCN × SZ/Los): replace submitted numbers with Stage 40 five-seed means ± std, and extend them with the NoSpatial, MultiGSL, MultiGSL-Weighted, MultiGSL-Mix rows and the GCN-MultiGSL row. The submitted tables omit 4 of the 12 canonical methods, including the two that decide the story (NoSpatial, MultiGSL-Mix).
2. **All improvement-percentage statements** computed from the submitted table (e.g. "T-GCN-GSL improves T-GCN by 26.9%"): recompute from Stage 40 (they barely change on Los, change materially on SZ).
3. **The claims "cGSL/GSL are the proposed methods and they beat the baselines"**: replaced by the graph-consumption storyline. GSL/cGSL remain as *contemporaneous-graph baselines* that fail to beat the identity graph.
4. **The Los PH1 GSL-vs-cGSL ordering** (submitted: GSL 4.818 ≪ cGSL 6.550): delete; Stage 40 shows a tie. Any sentence built on it ("cGSL can severely underperform", etc.) must go.
5. **Any statement of statistical significance from the old single-run comparisons**: the submitted manuscript had one run per cell — no significance statement was ever supportable. Use the Stage 41 paired tests with the explicit n=5 caveat.

---

## 10. Which results are safe to reuse

- **SZ GCN-family numbers**: fully reproduced (|Δ| ≤ 0.57%, ranks preserved). Safe as supplementary historical evidence, though the revised tables should still use Stage 40 values for uniformity (five-seed means with std).
- **Relative-improvement narrative "T-GCN-GSL ≫ T-GCN"**: survives at nearly identical magnitude on Los (+17–26% both versions). Usable, but must be re-framed as "learned contemporaneous graph vs *physical* graph", immediately followed by the NoSpatial caveat.
- **The qualitative GCN ordering cGSL < GSL < GCN(physical)** on SZ: reproduced exactly. Usable as a supporting observation about symmetrization × symmetric aggregation.
- **The physical-graph baselines themselves** (GCN, T-GCN rows): Stage 40 supersedes them, but the direction of every submitted comparison involving them is preserved, so no submitted conclusion about "physical vs learned" needs reversal — only reframing against NoSpatial.
- **Protocol, metrics, and pipeline documentation** from the submitted version: unchanged by Stage 40 (Stage 40 provenance reports `stage40.1/40.2/40.3` confirm method-equivalence of the old and new pipelines).

Not reusable as *evidence*: the Los-loop PH1 GSL/cGSL ordering, the "best method" claims, the universal-benefit claim, and any NoSpatial-free comparison table.

---

## 11. Which claims are no longer defensible

1. "GSL/cGSL outperform the corresponding baselines" — fails against NoSpatial (the strongest baseline) everywhere.
2. "TGCN-GSL is the best method" / "GCN-cGSL is the best method" — both fail against NoSpatial (and on Los, against MultiGSL variants).
3. "Learned graphs are universally beneficial" — refuted on SZ and by GCN-MultiGSL on Los.
4. "cGSL is consistently superior" — only true for the GCN family; on T-GCN the difference is within noise.
5. "cGSL/GSL gains stem from sparsity" — the learned graphs are *sparser* (8–56 edges vs 207–2833) yet *lose* to the identity graph in 4 of 4 family×dataset combinations; sparsity cannot be the operative variable for the contemporaneous variants.
6. Los PH1 "GSL ≫ cGSL" — seed artifact; not reproducible.
7. Any causal reading ("DAGMA discovers the traffic structure that helps prediction") — unsupported by design; the identity-graph result actively undermines a causal-structure interpretation, since removing all structure performs as well or better except in the one MultiGSL/Los case.

---

## 12. Recommended scientific storyline for the revised manuscript

1. **Question:** does learned graph structure help traffic prediction, and if so, through what mechanism?
2. **Framing result (new, decisive):** a graph-free ablation (NoSpatial, identity adjacency) outperforms the physical road-network graph *and* every learned contemporaneous-graph variant on both datasets. The contemporaneous-graph gains reported in the submitted version are real but are gains over a *bad* reference, not evidence of extracted structure.
3. **Core contribution (new):** T-GCN-MultiGSL — per-timestep consumption of DAGMA multi-lag graphs — is the only configuration that beats the graph-free baseline, and only on Los-loop (−0.56 to −0.76 RMSE, 5/5 seeds, all horizons). Mix-gated consumption adds a further −0.35 to −0.40.
4. **Mechanistic dissociation (new, strongest evidence):** GCN-MultiGSL vs T-GCN-MultiGSL consume *identical* graphs; union consumption collapses (9.78 on Los), timestep-matched consumption excels (4.84). The benefit lives in the consumption mechanism aligned with temporal structure, not in the graph source. cGSL-vs-GSL (symmetrization matters only for the symmetric GCN aggregator) provides the same dissociation in miniature.
5. **Honest boundary:** on SZ-taxi, nothing beats the identity graph by a meaningful margin; the benefit of learned multi-lag structure is **dataset-dependent**. State this as a finding, not a limitation buried in the discussion.
6. **Statistical hygiene:** five seeds; report means ± std and paired per-seed wins; describe p-values as indicative (exact Wilcoxon floor at 0.0625 with n=5); no language of proven significance.
7. **Positioning of the submitted results:** cite the submitted single-run study as a preliminary version whose qualitative GCN conclusions and T-GCN-vs-physical improvements are confirmed under seed replication, while its "best method" claims are corrected by the extended baseline set.

---

## 13. Remaining evidence gaps

1. **No more than five seeds** — the exact Wilcoxon p floor (0.0625) means even the flagship MultiGSL-Mix-vs-NoSpatial result cannot reach conventional significance. Not established by current experiments: definitive significance. (A seed extension is a rerun — out of scope here.)
2. **Only two datasets.** The dataset-dependence claim rests on a single contrast (SZ null, Los positive). A third dataset would establish whether Los is the rule or the exception.
3. **No hypothesis-driven manipulation of the proposed mechanism.** The GCN/T-GCN MultiGSL dissociation is observational (an architectural natural experiment). E.g. a union-consumption *T-GCN* variant or a timestep-matched *GCN* variant would close the loop; not established by current experiments.
4. **NoSpatial vs MultiGSL on SZ is a null result, not a proven equality** — with σ≈0.01–0.02, deltas of ≤0.015 are within noise; "no benefit on SZ" is the supported statement, not "no difference exists".
5. **Edge-content analysis of the DAGMA graphs** (what the 8/28/30-edge graphs contain, sensitivity to the DAGMA fit) is not part of the canonical results; claims about *which* structure matters remain interpretive.
6. **MAE agrees with RMSE throughout** (checked in `stage41_summary.csv`), but no additional metrics (e.g. MAPE, CRPS) are available; robustness of the ordering to metric choice is not established by current experiments.

---

## 14. Final verdict

**READY WITH IMPORTANT CAVEATS.**

The revised manuscript *can* tell a scientifically coherent story in which Stage 40 is a rigorous extension of the submitted evidence: the submitted numbers were honest single-run draws (20/48 cells essentially consistent; 0 directly contradictory at the cell level), every GCN-family and T-GCN-vs-physical qualitative conclusion survives replication, and the headline Los T-GCN improvement reproduces at nearly identical magnitude. But coherence requires the manuscript to *withdraw* its central claim — that learned contemporaneous graphs (GSL/cGSL) are the source of the improvement — because the extended baseline set (NoSpatial, MultiGSL family) shows the improvement over the physical graph is a poor-reference artifact, and relocates the genuine effect in per-timestep consumption of multi-lag graphs on one dataset. The corrected story (graph-consumption mechanism, dataset-dependent benefit) is more defensible and more interesting than the submitted one, but it is a different thesis: the paper's contribution changes from "our learned graphs win" to "when and how learned graphs can help, and why contemporaneous-graph learning does not."
