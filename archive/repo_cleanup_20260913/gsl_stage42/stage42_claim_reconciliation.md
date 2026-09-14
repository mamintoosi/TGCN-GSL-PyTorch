# Stage 42 — Claim Reconciliation: Submitted Manuscript vs Stage 40 Canonical Evidence

**Date:** 2026-09-10
**Companion documents:** `stage42_submitted_vs_stage40_audit.md` (full audit), `stage42_comparison.csv` (all 48 cells).
**Constraints honored:** read-only audit; no experiments, no refits, no manuscript edits.

Each claim below is stated as the submitted manuscript used it, tested against Stage 40, and given one of four statuses: **RETIRED** (no longer supportable), **REFRAMED** (survives only in altered form), **RETAINED** (survives with at most cosmetic changes), or **PROMOTED** (new claim that becomes central).

---

## A. "GSL improves over the physical graph."

**Status: REFRAMED (direction retained, interpretation rejected).**

- Evidence for: in all 8 dataset×family×PH combinations the learned contemporaneous graph beats the physical graph (e.g. SZ GCN-GSL 4.880 vs GCN 5.960; Los T-GCN-GSL 5.859 vs T-GCN 7.877). Paired seed wins 5/5 in 7 of 8 cells (GCN-GSL exceptions: Los PH2 0/5, Los PH4 1/5 — see `stage41_paired_tests.csv`; the Los GCN-GSL "improvement" is itself fragile).
- Evidence against the submitted interpretation: the submitted manuscript framed this as evidence that learning extracts useful structure. Stage 40's NoSpatial runs show the *identity* graph beats the physical graph by more than any contemporaneous learned graph does (SZ GCN: 4.114 vs physical 5.960; Los T-GCN: 5.251 vs physical 7.877). The physical graph is the worst configuration in every family×dataset combination. Improvement over it is therefore not evidence of extracted structure — it is evidence that the physical adjacency was harming the model.
- Required rewording: "Replacing the physical road-network adjacency — learned or identity — consistently reduces error; the learned contemporaneous graph recovers only a fraction of the gap that simply removing the graph recovers."
- Not permitted: any causal attribution (the graph "captures", "discovers", "adapts"). The static graph is fixed across the evaluation window; nothing adapts.

## B. "GSL is better than NoSpatial."

**Status: RETIRED.**

- The submitted manuscript never included NoSpatial, so this claim was implicit rather than tested. Stage 40 tests it and it fails in all 8 cells: NoSpatial beats T-GCN-GSL and T-GCN-cGSL on both datasets at every horizon (SZ: 4.120 vs 4.281/4.301 at PH1; Los: 5.251 vs 5.859/5.821), and beats GCN-GSL/GCN-cGSL likewise (SZ: 4.114 vs 4.880/4.641; Los: 4.880 vs 7.827/5.763).
- The only learned configurations that beat NoSpatial anywhere are T-GCN-MultiGSL / -Weighted / -Mix on Los-loop (see claim G). No contemporaneous GSL/cGSL variant survives.
- Scientific implication: the contemporaneous-graph results must be repositioned as *baselines that fail*, which becomes diagnostic evidence for the consumption-mechanism argument (a single contemporaneous graph — learned or not — cannot serve all lags).

## C. "cGSL is consistently superior to GSL."

**Status: REFRAMED (family-specific, demoted from central contribution to mechanism probe).**

- GCN family: supported and robust. cGSL beats GSL in every PH on both datasets with large margins (SZ PH1: 4.641 vs 4.880; Los PH1: 5.763 vs 7.827), 5/5 paired wins in 7 of 8 cells.
- T-GCN family: not supported. Deltas are +0.02 (SZ) and −0.04 (Los) at PH1 — within seed noise (σ ≈ 0.05–0.23); no decisive separation in any cell.
- The submitted Los PH1 ordering (GSL 4.818 ≪ cGSL 6.550, margin 1.73) is a seed artifact: Stage 40 gives 5.859 vs 5.821. Any manuscript sentence built on that ordering must be deleted.
- Retained form: "Symmetrizing the learned adjacency matters specifically when the downstream aggregation is symmetric (GCN); it is immaterial for the temporal model." This supports the graph-consumption-compatibility argument and nothing stronger. Symmetrization must not be described as generally beneficial.

## D. "TGCN-GSL is the best method."

**Status: RETIRED.**

- On the submitted table it was best only among the three submitted T-GCN variants. Stage 40's extended family puts it behind T-GCN-NoSpatial on both datasets (SZ PH1: 4.281 vs 4.120; Los PH1: 5.859 vs 5.251) and far behind T-GCN-MultiGSL-Mix on Los (4.491).
- The honest preserved fragment: T-GCN-GSL's improvement over the *physical-graph* T-GCN replicates at almost identical magnitude (submitted +26.9%/+22.4%/+20.6%/+17.3% vs Stage 40 +25.7%/+22.5%/+20.5%/+18.7% on Los). This is a fact about the physical graph, not about the merit of GSL.

## E. "GCN-cGSL is the best method."

**Status: RETIRED.**

- GCN-NoSpatial beats GCN-cGSL in every PH on both datasets (SZ PH1: 4.114 vs 4.641; Los PH1: 4.880 vs 5.763).
- Within-family value survives: GCN-cGSL is the best *learned-graph* GCN variant everywhere, and the cGSL-vs-GSL margin is the GCN family's cleanest finding. That is a supporting role, not a flagship.

## F. "Learned graphs are universally beneficial."

**Status: RETIRED (universal form). Replaced by a specific, supported interaction claim.**

- Refutations: (i) on SZ-taxi no learned variant meaningfully beats NoSpatial and none beats it at all except by ≤0.015 within noise; (ii) GCN-MultiGSL on Los-loop (9.78–10.27) is *worse than the physical graph* (8.14–8.76) — a learned graph actively hurting; (iii) the contemporaneous learned graphs lose to the identity graph in all 4 family×dataset combinations despite being far sparser (8–56 vs 156–2833 edges), which also removes sparsity as the operative variable.
- Replacement claim (supported): "The benefit of learned multi-lag graph structure is dataset-dependent: it appears only when lag-specific graphs are consumed at matching timesteps (T-GCN-MultiGSL family) and only on Los-loop." Justification in §7 of the audit: the dataset split aligns exactly with the consumption mechanism, not with graph source, sparsity, or family.

## G. "MultiGSL provides a stronger and more defensible explanation of the gains."

**Status: PROMOTED — becomes the revised manuscript's core contribution, with explicit dataset and significance caveats.**

- Supporting evidence:
  - T-GCN-MultiGSL-Mix beats T-GCN-NoSpatial on Los in all 4 PH (−0.56 to −0.76 RMSE, 5/5 paired seeds, paired-t p ≤ 0.0015).
  - T-GCN-MultiGSL (fixed) already beats NoSpatial on Los (−0.41 to −0.72); Mix gating adds a further −0.35 to −0.40 (5/5 wins, p ≤ 0.023).
  - The GCN/T-GCN MultiGSL dissociation — *identical* DAGMA multi-lag graphs, union consumption collapses (9.78) while per-timestep consumption excels (4.84) on Los — localizes the effect in the consumption mechanism, not the graph source.
- Mandatory caveats:
  1. **Dataset-dependent:** on SZ-taxi the same configurations are statistically indistinguishable from NoSpatial (deltas ≤ 0.015, mixed win counts, p 0.006–0.9). State the null plainly.
  2. **Significance is indicative only:** n = 5 seeds; the exact Wilcoxon floor is 0.0625, so no comparison can reach conventional significance. Report means ± std and per-seed wins; avoid the word "significant" without qualification.
  3. **No causal language:** the multi-lag DAGMA graphs are statistical summaries of lag-wise co-movement; the paper may say per-timestep consumption *correlates with* lower error, never that DAGMA *discovered* the structure that helps.
  4. **Mechanism is observational:** the consumption dissociation is an architectural natural experiment, not a controlled manipulation (see evidence gaps §13 of the audit).

---

## Disposition summary

| Claim | Disposition | Where it lives in the revised manuscript |
|---|---|---|
| A. GSL improves over the physical graph | REFRAMED | Introduction/motivation, immediately followed by the NoSpatial result |
| B. GSL better than NoSpatial | RETIRED | Becomes a *negative result* reported as such (contemporaneous GSL/cGSL fail) |
| C. cGSL consistently superior | REFRAMED | Mechanism probe within the consumption-compatibility argument; family-specific wording |
| D. TGCN-GSL best method | RETIRED | Replaced by MultiGSL-Mix (Los) as the best configuration, with the SZ null |
| E. GCN-cGSL best method | RETIRED | GCN family demoted to supporting analysis |
| F. Learned graphs universally beneficial | RETIRED | Replaced by the dataset-dependence claim (supported) |
| G. MultiGSL explains the gains | PROMOTED | New central thesis: graph-consumption mechanism, dataset-dependent benefit |

**Old results disposition (audit §9–10 condensed):** SZ GCN-family numbers and the T-GCN-vs-physical relative improvements are safe to reuse as replicated historical evidence; the Los PH1 GSL/cGSL ordering, all "best method" claims, and the universal-benefit claim must be removed; all tables should be replaced by Stage 40 five-seed results (submitted tables omit NoSpatial and the MultiGSL family, the methods that decide the story).

**Claims not established by current experiments (explicit list):** definitive statistical significance (n = 5); generalization beyond the two datasets; a controlled (non-observational) demonstration of the consumption mechanism; metric-robustness of the orderings beyond RMSE/MAE; sensitivity of conclusions to the DAGMA fit.

**Reconciliation verdict:** the submitted manuscript's claims A–F are, at best, reframable fragments; the coherent revised thesis is claim G plus the dataset-dependence boundary. The Stage 40 evidence supports that revised thesis without contradiction, and the submitted results integrate into it as the preliminary, single-run study whose qualitative GCN conclusions and physical-graph comparisons survive replication. **READY WITH IMPORTANT CAVEATS** (full rationale: audit §14).
