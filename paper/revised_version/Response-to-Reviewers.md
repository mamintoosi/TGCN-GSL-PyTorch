# Response to Reviewers

**Manuscript:** Graph Structure Learning for Traffic Prediction  
**Journal:** International Journal of Data Science and Analytics  
**Type:** Major revision

---

## General response

We thank the editor and both reviewers. The revision keeps the paper’s identity as a study of **graph structure learning for traffic forecasting**, while rebuilding the experimental design around the reviewers’ main concerns:

1. A **graph-free (identity) control** is now a first-class baseline, so learned graphs are not credited only for beating a possibly harmful physical graph.
2. The **contemporaneous vs multi-lag** constructions of DAGMA are stated explicitly; we no longer interpret a contemporaneous DAG as a temporal $t\!\to\!t+1$ graph.
3. All main results use **five seeds** with mean $\pm$ sample standard deviation, paired win counts, and an explicit $n=5$ statistical caveat.
4. **Matched-sparsity and capacity controls** (scope-labeled) address the sparsity confound.
5. **Causal language is removed**; learned graphs are statistical dependency estimates.
6. A **Limitations** section and a $15$-minute Los-loop **resolution control** (to test whether SZ-Taxi’s weak gains were due to sampling interval) are included.

Headline quantitative outcome under the new protocol: on Los-loop, multi-lag graphs used in a lag-aligned way inside T-GCN beat the graph-free control (T-GCN-MultiGSL-Mix: **$14.5\%$ relative RMSE reduction at PH1**, $5/5$ seeds; up to **$43.0\%$ vs the physical T-GCN baseline**). On SZ-Taxi all learned variants remain near the graph-free baseline. Sparsity alone does not explain the Los-loop gain on the matched-budget cell.

Below we address each comment. Locations refer to the revised manuscript.

---

## Reviewer 1

### W1 — Bibliometrics underused
One motivating sentence remains in the Introduction; full methodology and figures are in **Appendix A** (`sec:Bibliometric-Methodology`).

### W2 — GCN/T-GCN background too long
Section 2 is condensed; full layer-by-layer exposition is reduced to the propagation equations needed later.

### W3 — A $\to$ W notation
Section 3 states in the main text that $\mathbf{A}$ is the backbone adjacency and $W$ the continuous DAGMA matrix (no footnote-only switch).

### W4 — Temporal interpretation of the learned DAG
We agree this was load-bearing and was not demonstrated. The contemporaneous construction (simultaneous snapshots) is **not** described as temporal. Explicit multi-lag input $Z=[x(t-3),\ldots,x(t)]$ carries the lag structure (Section 3). The old “temporal DAG” analysis section is retired.

### W5 — Sparsity / oversmoothing / controls
Added: graph-free control; matched $30$-edge RandTop30 / CorrTop30 vs DAGMA multi-lag on Los-loop PH1 (Table controls); structure figure (physical vs multi-lag union). We do **not** claim a full $\lambda$ sweep or sparsified physical graph — listed in Limitations.

### W6 — Seeds / variance / significance
Five seeds ($42$–$46$); mean $\pm$ sample std; paired wins; paired $t$-tests for reference; explicit Wilcoxon floor $p=0.0625$ and no definitive significance claims (Setup, Results).

### W7 — Horizons only to 4
Canonical grid remains PH $\le 4$ at native sampling. A **$15$-minute Los-loop resolution control** covers $15$–$60$ minutes wall-clock and tests (and refutes) sampling interval as the explanation of SZ-Taxi’s weak result. PH5–8 at $5$-minutes were not run (Limitations / future work).

### W8 — Repetitive Results; weak tie-back
Results subsections are message-oriented; Discussion ties dense-graph failure back to proximity vs dependency.

### W9 — Limitations
Dedicated Limitations section: $n=5$, control scope, datasets/horizons, linear DAGMA, static graphs, backbone scope.

### W10 — Metric definitions
RMSE/MAE one-liners with short formulas; $R^2$/accuracy not required for the comparisons.

### W11 — Dense convergence grids
Removed from the main text.

### W12 — Citation style
`\citep`/`\citet` throughout the revised text.

### Q1 — Physical vs learned visualization
Figure (graph structure): physical adjacency, multi-lag union, degree histograms.

### Q2 — Predicted vs actual
Not in the main text of this revision; can be supplied as supplementary material if required.

### Q3 — Time-varying graphs
Clarified: Mix changes **use** of static lag graphs, not the edge set; incremental structure learning is future work.

### Q4 — Direct contemporaneous vs lagged evidence
Two constructions specified in Section 3; multi-lag vs static-union comparison in Results (same artifacts; backbone confound stated).

---

## Reviewer 2

### Abstract number mix-up (21.6% / 24.7%)
Removed. The abstract now quotes five-seed canonical figures with clear references (14.5% vs graph-free; up to 43.0% vs physical).

### Causal / “hidden causal structure”
All causal claims retired. Acyclicity is an optimization regularizer; graphs are statistical dependency estimates.

### Static graph vs “adapts to changing traffic”
Fixed: graphs fitted once; Mix varies consumption only.

### cGSL defined late
cGSL defined in Section 3 as binary symmetrization of the same GSL artifact, before Results.

### Convergence plots unreadable
Removed from main text.

### Typo “avergae”
No longer present in the rewritten Results.

---

## Note on protocol differences

The submitted single-run protocol (e.g.\ batch size $64$, one seed) is **not** compared cell-by-cell with the new five-seed protocol in the scientific text. Differences are protocol-level (seeds, batch, normalization scope, graph provenance) and are explained here in the response letter, not as a historical narrative inside the paper.

---

## Checklist

| Request | Where |
|---------|--------|
| Bibliometrics | Intro + App. A |
| Condensed background | §2 |
| A/W notation | §3 |
| Temporal DAG evidence | §3 multi-lag; §6 |
| Sparsity/degree/controls | §5 + Table controls + Fig. structure |
| Multi-seed | §4 + tables |
| Longer horizons | Resolution control + Limitations |
| Less repetition | §5–6 |
| Limitations | §7 |
| Metrics | §4 |
| Convergence | Removed / appendix-ready |
| Graph visualization | Results figure |
| Abstract numbers | Corrected |
| No causal claims | Global |
| Static vs adaptive | §3, §6 |
| cGSL early | §3 |
| Typo | Fixed |

We believe the revision answers the reviewers’ scientific concerns with a single, auditable five-seed protocol while preserving the paper’s focus on graph structure learning for traffic prediction.
