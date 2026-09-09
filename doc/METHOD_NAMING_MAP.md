# Method Naming Map (Stage 33 Harmonization)

Canonical terminology follows the revised manuscript. The registry of record is
`models/multigsl.py` (`METHOD_REGISTRY`); this document is the human-readable
version including historical artifacts.

## 1. Canonical names

| Manuscript name | Canonical id (Python/CLI) | Implementation class | Graph argument |
|---|---|---|---|
| T-GCN-NoSpatial | `no_spatial` | `models.tgcn.TGCN` with identity adjacency | identity (N×N) |
| Physical | `physical` | `models.tgcn.TGCN` with road-network adjacency | single (N×N) |
| T-GCN-MultiGSL | `multi_gsl` | `models.multigsl.MultiGraphTGCNFixed` | list of K lag graphs |
| T-GCN-MultiGSL-Mix | `multi_gsl_mix` | `models.multigsl.GatedMultiGraphTGCN` | list of K lag graphs |
| T-GCN-MultiGSL-Weighted (supplementary ablation only) | `multi_gsl_weighted` | `models.multigsl.WeightedMultiGraphTGCN` | list of K lag graphs |

Conceptual distinction (unchanged): **MultiGSL** assigns lag-specific graphs to
input timesteps in a fixed pattern; **MultiGSL-Mix** learns per-node,
per-timestep mixing over those graphs. The word "Adaptive" is deliberately NOT
used in any externally visible name.

## 2. Legacy name → canonical name → implementation class

| Legacy name (code/JSON/CLI) | Canonical manuscript name | Implementation class | Where it appears |
|---|---|---|---|
| `NoGraph`, `nograph`, `standard`, `NoGraph_h64`, `NoGraph_h74`, `NoSpatial` | T-GCN-NoSpatial | `TGCN` (identity adj) | Stage 26/29 JSON+CSV keys, train_with_logging CLI, checkpoint dir names, figure scripts |
| `Physical`, `phys` | Physical | `TGCN` (physical adj) | Stage 26 JSON keys, appendix tables |
| `MultiGraphTGCN_fixed`, `multi_graph_fixed` | T-GCN-MultiGSL | `MultiGraphTGCNFixed` | Stage 26/29 JSON+CSV keys, CLI, checkpoint dirs |
| `MultiGraphTGCN`, `MultiGraphTGCN_thr0.1`, `MultiGraph` | T-GCN-MultiGSL | `MultiGraphTGCNFixed` (or its earlier `MultiGraphTGCNCell` variant in stage26_evaluate.py) | Stage 26 evaluate JSON keys, figure scripts |
| `GatedMultiGraphTGCN`, `gated_multi`, `GatedMulti_thr0.1`, `GatedMulti` | T-GCN-MultiGSL-Mix | `GatedMultiGraphTGCN` | Stage 26/29 JSON+CSV keys, CLI, checkpoint dirs, figure scripts |
| `WeightedMultiGraphTGCN`, `weighted_multi`, `WeightedMulti_thr0.1`, `WeightedMulti` | T-GCN-MultiGSL-Weighted | `WeightedMultiGraphTGCN` | Stage 26 evaluate JSON keys only |
| `CorrTop30`, `Corr-K8/16/32` | CorrTop30 / Corr-K{k} (heuristic controls, no canonical renaming) | `TGCN` (static corr graph) | Stage 32 script, Stage 26 evaluate JSON |
| `RandTop30` | RandTop30 (random control) | `TGCN` (static random graph) | Stage 32 script |
| `SingleDAG_thr{t}` | single-lag DAGMA at threshold τ=t (descriptive, unchanged) | `TGCN` | Stage 26 evaluate JSON, fig2/fig6 |
| `UnionGraph_thr{t}`, `IntersectGraph_thr{t}`, `AggregatedDAG_thr{t}` | descriptive structure ablations (Stage 26 evaluate only) | `TGCN` | Stage 26 evaluate JSON |

Resolution helper in code: `models.multigsl.normalize_method(name)` returns the
canonical id for any legacy name above (None if unknown).

## 3. Historical artifacts that are deliberately NOT renamed

- Stage names and scripts: `stage20`–`stage32`, `gsl_stage20`…`gsl_stage26`
  directories, `stage26_*.py`, `stage29_los15min.py`, `stage32_sparse_control.py`.
- Result directories: `results/stage26_validation/`, `results/stage29_los15min/`,
  `results/stage32_sparse_control/`, `results/stage26_checkpoint/`, etc.
- JSON/CSV method keys inside historical results (`NoGraph`,
  `MultiGraphTGCN_fixed`, `GatedMultiGraphTGCN`, `GatedMulti_thr0.1`, ...).
- Checkpoint directory suffixes produced by `stage26_train_with_logging.py`
  (`nograph`, `multi_graph_fixed`, `gated_multi`), which the figure scripts read.
- `data/W_est_*` and `data/correlation_*` files from the original submission.

Active/current code uses canonical ids going forward; historical files are
read through `normalize_method` / the table above instead of being rewritten.

## 4. Canonical pipeline entry points

| Purpose | Script | Methods exposed |
|---|---|---|
| Multi-seed / param-control / lag ablation | `gsl_stage26/stage26_validation.py` | no_spatial, multi_gsl, multi_gsl_mix (via `gated_multi`/`multi_graph_fixed` internal ids) |
| 15-minute-resolution experiment (verified) | `gsl_stage26/stage29_los15min.py` | same three |
| Sparse matched-edge controls | `gsl_stage26/stage32_sparse_control.py` | CorrTop30, RandTop30 (+ optional multilag_union) |
| Checkpoint training (convergence + predictions) | `gsl_stage26/stage26_train_with_logging.py` | no_spatial / multi_gsl / multi_gsl_mix (+ legacy aliases) |
| Single-graph GSL canonical rerun (Stage 33, prepared) | `gsl_stage26/stage33_gsl_canonical.py` | T-GCN-GSL / GCN-GSL under the canonical protocol |

Model implementations live once in `models/multigsl.py`; experiment scripts
import from there and must not re-define model classes.
