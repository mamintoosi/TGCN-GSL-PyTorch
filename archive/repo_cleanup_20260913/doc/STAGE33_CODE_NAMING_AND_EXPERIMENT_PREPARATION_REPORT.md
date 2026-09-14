# Stage 33 — Code Naming Harmonization and Experiment Preparation Report

**Date:** 2026-09-09
**Scope:** Goal 1 code naming harmonization · Goal 2 canary verification · Goal 3 manuscript source-reference updates · Goal 4 final-experiment preparation (nothing expensive was run)
**Companion document:** `doc/METHOD_NAMING_MAP.md` (the standing naming reference)

---

## 1. Summary of code naming changes

The four externally visible method names are now defined **once** in a new
canonical module, `models/multigsl.py`, and every active experiment script
imports from it:

* `T-GCN-NoSpatial` (id `no_spatial`) — TGCN with identity adjacency
* `Physical` (id `physical`) — TGCN with the road-network adjacency
* `T-GCN-MultiGSL` (id `multi_gsl`) — fixed assignment of lag-specific graphs (`MultiGraphTGCNFixed`)
* `T-GCN-MultiGSL-Mix` (id `multi_gsl_mix`) — learned per-node, per-timestep mixing (`GatedMultiGraphTGCN`, the proposed method)

Key properties of the harmonization:

1. **Single source of truth.** `GatedMultiGraphTGCN`, `MultiGraphTGCNFixed`, and
   `WeightedMultiGraphTGCN` now live only in `models/multigsl.py`. The three
   previously duplicated class definitions (in `stage26_validation.py`,
   `stage29_los15min.py`, `stage26_train_with_logging.py`) were **bit-identical
   copies** and were replaced by imports — verified numerically by the canary
   (Section 4).
2. **Registry + compatibility mapping.** `METHOD_REGISTRY` maps canonical ids to
   classes and adjacency kinds; `normalize_method(name)` translates *any*
   legacy name (`NoGraph`, `gated_multi`, `GatedMulti_thr0.1`, ...) to its
   canonical id. `build_model()` instantiates any method from one entry point.
3. **Historical artifacts untouched.** JSON/CSV rows written by new runs keep
   their historical `method` keys (`NoGraph`, `MultiGraphTGCN_fixed`,
   `GatedMultiGraphTGCN`) and gain a `canonical_name` field, so old and new
   artifacts remain mutually readable. Historical scripts
   (`stage26_evaluate.py`, `stage26_resolution_experiment.py`) are unchanged
   except for a documentation note; result directories, stage names, and
   checkpoint directory names are unchanged.
4. **No "Adaptive" anywhere.** Verified by grep over all manuscript and code
   files: the proposed method is `T-GCN-MultiGSL-Mix` in every externally
   visible surface.
5. **Deduplication of graph constructors.** `binary_graph`,
   `correlation_topk_graph` (CorrTop30), and `random_edge_graph` (RandTop30)
   also moved into `models/multigsl.py`; `stage32_sparse_control.py` now imports
   them instead of redefining them.
6. **Terminology duplication reduced.** New result rows carry `canonical_name`
   so analysis code no longer needs to know the legacy keys; figure scripts
   carry an explicit `LEGACY_TO_CANONICAL` display map instead of embedding
   translations inline.

## 2. Legacy-to-canonical naming map

Condensed here; the full table (including historical stage scripts, JSON keys,
checkpoint dirs, and data files) is in **`doc/METHOD_NAMING_MAP.md`**.

| Legacy name | Canonical name | Implementation class |
|---|---|---|
| `NoGraph` / `nograph` / `standard` / `NoGraph_h64` / `NoGraph_h74` | T-GCN-NoSpatial | `TGCN` (identity adj) |
| `Physical` / `phys` / `TGCN` (as baseline label) | Physical | `TGCN` (road adj) |
| `MultiGraphTGCN_fixed` / `multi_graph_fixed` / `MultiGraphTGCN` / `MultiGraphTGCN_thr0.1` | T-GCN-MultiGSL | `MultiGraphTGCNFixed` |
| `GatedMultiGraphTGCN` / `gated_multi` / `GatedMulti_thr0.1` / `GatedMulti` | T-GCN-MultiGSL-Mix | `GatedMultiGraphTGCN` |
| `WeightedMultiGraphTGCN` / `weighted_multi` / `WeightedMulti_thr0.1` | T-GCN-MultiGSL-Weighted (supplementary) | `WeightedMultiGraphTGCN` |
| `CorrTop30`, `RandTop30`, `Corr-K8/16/32` | unchanged (descriptive control labels) | `TGCN` (static graph) |
| `SingleDAG_thr{t}`, `UnionGraph_thr{t}`, `IntersectGraph_thr{t}`, `AggregatedDAG_thr{t}` | unchanged (descriptive structure ablations) | `TGCN` (static graph) |

## 3. Files changed

**New files**

| File | Purpose |
|---|---|
| `models/multigsl.py` | canonical model classes, registry, `normalize_method`, `build_model`, graph constructors |
| `doc/METHOD_NAMING_MAP.md` | standing legacy→canonical reference |
| `gsl_stage26/stage33_canary.py` | refactor-verification canary (Section 4) |
| `gsl_stage26/stage33_gsl_canonical.py` | prepared: canonical GSL-baseline rerun (not run) |
| `gsl_stage26/stage33_sz_multiseed.py` | prepared: SZ-Taxi 5-seed validation (not run) |
| `run_stage33A_sparse_control.sh` | Linux runner — sparse controls |
| `run_stage33B_gsl_canonical.sh` | Linux runner — canonical GSL rerun |
| `run_stage33C_sz_multiseed.sh` | Linux runner — SZ-Taxi multi-seed |
| `doc/STAGE33_CODE_NAMING_AND_EXPERIMENT_PREPARATION_REPORT.md` | this report |

**Modified files**

| File | Change |
|---|---|
| `models/__init__.py` | re-exports the canonical classes/registry/builders |
| `gsl_stage26/stage26_validation.py` | imports canonical classes; static-TGCN train/eval split into `train_and_eval_standard`; result rows carry `canonical_name`; `--dataset` now accepts `shenzhen`; docstrings use canonical names |
| `gsl_stage26/stage29_los15min.py` | imports canonical classes (duplicates removed); `canonical_name` added to rows; log lines use canonical names |
| `gsl_stage26/stage26_train_with_logging.py` | imports canonical classes; CLI accepts canonical ids (`no_spatial`, `multi_gsl`, `multi_gsl_mix`) plus legacy aliases; checkpoint dirs keep historical names |
| `gsl_stage26/stage32_sparse_control.py` | graph constructors imported from `models.multigsl`; docstring terminology canonicalized |
| `gsl_stage26/stage26_evaluate.py` | documentation note only (historical script) |
| `gsl_stage26/stage26_resolution_experiment.py` | documentation note only (historical Stage 27 script; invalid-for-forecasting warning added) |
| `paper/generate_figures.py` | `LEGACY_TO_CANONICAL` display map added for old JSON keys |
| `paper/generate_figures_extra.py` | checkpoint-dir naming note added |
| `paper/sections/results.tex` | "MultiGraph" → "T-GCN-MultiGSL" (one sentence) |
| `paper/sections/method.tex` | ablation bullet list renamed to canonical terms (Union of lag graphs / T-GCN-MultiGSL-Weighted / T-GCN-MultiGSL) |
| `paper/sn-article.pdf` | recompiled (Section 6) |

## 4. Canary test — command, result, details

**Command** (Windows, CPU):

```
/c/programs/anaconda3/envs/pth/python.exe gsl_stage26/stage33_canary.py
```

**Result: 10/10 PASS** (exit 0), **duration ≈ 21 s**, no warnings other than
torch's import-time notices. Date: 2026-09-09.

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | module imports (9 modules: models, models.multigsl, models.tgcn/gcn/gru, tasks.supervised, utils.losses, utils.graph_conv, stage26_validation) | PASS | all import cleanly |
| 2 | four model variants instantiate (no_spatial, physical, multi_gsl, multi_gsl_mix) | PASS | K=3, H=8 toy graph (N=12) |
| 3+4 | Mix receives lag graphs as `lap_stack` (3,12,12); MultiGSL registers `lap_0..lap_2` **in input order**, each equal to `calculate_laplacian_with_self_loop` of the corresponding input | PASS | buffer contents match inputs exactly |
| 5+8 | synthetic batch (4,12,12) forward through all four variants; shapes (4,12,8); all finite | PASS | |
| 5b | **faithfulness:** canonical Mix forward == inline re-derivation of the original Stage 26 implementation (same weights) | PASS | max abs diff = 0.00e+00 |
| 5c | **faithfulness:** canonical MultiGSL forward == original Stage 26 implementation | PASS | max abs diff = 0.00e+00 |
| 6+7 | `SupervisedForecastTask(loss="mse_with_regularizer")` loss finite; one Adam step changes weights; regressor present | PASS | loss=0.112 |
| 8b | parameter counts match the manuscript formula: TGCN h=64 → 12,672; Mix h=64,K=3 → 17,091 | PASS | |
| 9 | CLI parsing: `stage26_validation` (incl. new `--dataset shenzhen`), `stage26_train_with_logging` (6 method ids, canonical + legacy), `stage32_sparse_control` | PASS | |
| bonus | `normalize_method` maps 10 legacy names correctly; unknown → None | PASS | |

The refactor is therefore **behaviour-preserving**: identical outputs given
identical weights, identical parameter counts, identical graph handling.
These are canary checks on a toy graph, not scientific results.

## 5. Manuscript source-code reference changes

All files under `paper/` were inspected for code/stage/artifact references
(`grep` for stage names, script names, and every legacy method label). Findings:

* The manuscript already used canonical method names everywhere except two spots, both fixed:
  * `results.tex` (oversmoothing discussion): "the sparse MultiGraph and T-GCN-MultiGSL-Mix approaches" → "the sparse **T-GCN-MultiGSL** and T-GCN-MultiGSL-Mix approaches".
  * `method.tex` (contrasts-with-simpler-approaches list): internal codenames `UnionGraph` / `WeightedMulti` / `MultiGraph` replaced with reader-facing terms — "Union of lag graphs", "**T-GCN-MultiGSL-Weighted**" (now named as a proper ablation, with its benchmark value 4.710 cited from the stage-26 artifact), and "**T-GCN-MultiGSL**".
* A table footnote in `results.tex` was temporarily extended with an explicit
  artifact filename; it caused an overfull box and was reverted (provenance
  remains fully documented in `doc/` and in the result JSONs themselves).
* No figure captions or table notes named historical scripts, so nothing else
  needed changing; historical provenance (e.g. the "Stage-26 evaluation run"
  footnote distinguishing 4.715/4.717) is intentionally **kept**.
* No numerical values were altered.

## 6. Manuscript compilation result

`pdflatex` + `bibtex` + 2×`pdflatex` on `paper/sn-article.tex`:

| Check | Result |
|---|---|
| LaTeX errors | **0** |
| Undefined references | **0** |
| Undefined citations | **0** |
| Overfull boxes | **0** |
| Pages | 23 (unchanged from Stage 32 state) |

## 7. Recommended final status of each method

Decision basis: the final narrative (Section 12), reviewer concerns
(sparsity/structure, oversmoothing, dataset dependence, multi-seed rigor), and
the Stage 19/26/29/30 forensic evidence. Not based on "what was in the old paper".

| # | Method | Final status | Rationale |
|---|---|---|---|
| 1 | **GCN-GSL** | APPENDIX (already there) | Supports the architecture-asymmetry observation (GCN prefers cyclic, T-GCN acyclic) that motivates the multi-lag question. Original-protocol numbers stay in the appendix, clearly labelled; a canonical rerun is **optional** (`run_stage33B` has a commented GCN block) — the asymmetry is qualitative, not a headline number. |
| 2 | **GCN-cGSL** | REMOVE from main story (remains in the existing appendix table only as part of the historical record) | Its role (cyclic-beats-acyclic for GCN) is already captured by the appendix narrative; the main text references the asymmetry, not the cGSL numbers. No rerun. |
| 3 | **T-GCN-GSL** | **RERUN UNDER CANONICAL PROTOCOL → MAIN TEXT** (as the ACT I starting point) | The single-graph GSL result is the empirical starting point of the whole story, so it belongs in the main text — but only with canonical numbers. Protocol differences are material (batch 64→128, wd 0→1e-4, single seed →5 seeds, unprovable W provenance). This is `run_stage33B`. |
| 4 | **T-GCN-cGSL** | REMOVE (appendix historical table only) | Same reasoning as GCN-cGSL; the cyclic variant adds no main-text content once lag-specific graphs exist. |
| 5 | **T-GCN-NoSpatial** | MAIN TEXT (verified) | The critical baseline for the oversmoothing and multi-lag claims; 5-seed verified at both resolutions. |
| 6 | **Physical** | MAIN TEXT (verified, single seed) | The density/oversmoothing contrast (7.658 vs 5.143) requires it; also replicated on SZ-Taxi. A 5-seed rerun is cheap but not required — the qualitative point is robust and appears at both resolutions. |
| 7 | **T-GCN-MultiGSL** | MAIN TEXT (5-seed verified at 5-min and 15-min) | The fixed-assignment ablation; required to attribute the gain to *learned mixing* rather than merely having lag graphs. |
| 8 | **T-GCN-MultiGSL-Mix** | MAIN TEXT (verified, 5 seeds, 2 resolutions) | The proposed method. |

Corr-K8 stays in the oversmoothing table (dense-*learned* control, single seed,
clearly labelled). The two matched-edge controls (CorrTop30/RandTop30) join the
main text after `run_stage33A` completes.

## 8. Recommended experiment suite to run next

Prepared scripts, **none executed in this stage**. Ordered by priority:

| ID | Experiment | Question | Script | Runner | Est. runtime (GPU) | Priority |
|---|---|---|---|---|---|---|
| **A** | Sparse matched-edge controls (CorrTop30, RandTop30), Los-loop PH1, 5 seeds | Is the MultiGSL-Mix gain explained by sparsity alone? (Reviewer 1 W5) | `gsl_stage26/stage32_sparse_control.py` (already canary-tested in Stage 32) | `run_stage33A_sparse_control.sh` | ~5 min | **HIGH** — one line lands in `tab:oversmoothing` |
| **B** | Canonical rerun of T-GCN-GSL (physical vs GSL graphs), Los-loop, PH1–4, 5 seeds | Does the original ACT I result hold under the canonical protocol, making the main-text comparison clean? | `gsl_stage26/stage33_gsl_canonical.py` | `run_stage33B_gsl_canonical.sh` | ~2–3 h (DAGMA ≈1.5–3 h once + forecasting ≈20 min) | **HIGH** — supplies the ACT I main-text row |
| **C** | SZ-Taxi 5-seed validation, PH1–4 (reuses existing SZ DAGMA blocks) | Is the marginal SZ improvement stable? Is the PH=4 dip seed noise? | `gsl_stage26/stage33_sz_multiseed.py` | `run_stage33C_sz_multiseed.sh` | ~30–40 min (no DAGMA) | MEDIUM — upgrades `tab:sz_multiph` to mean±std |
| — | GCN-GSL canonical rerun | Appendix completeness only | same as B (`--models gcn`) | (commented block in runner B) | +~30 min | LOW / optional |
| — | SZ-Taxi canonical GSL rerun | Symmetry of the ACT I result on the urban dataset | same as B (`--dataset shenzhen`) | (commented block in runner B) | +~2 h | LOW / optional |
| — | Horizons PH5–8 | Long-horizon behaviour | existing Stage 26/29 scripts | — | ~1 h | LOW — limitations already state PH≤4 |

Not needed (and why): no DAGMA recomputation for A or C (existing matrices are
reused and were proven bit-identical across Stage 26/27/29 in the Stage 30
audit); no Stage 27 forecasting rerun (superseded by the verified Stage 29);
no cGSL reruns (Section 7).

## 9. Exact commands for each prepared experiment

On the Linux machine (`/data/git/mamintoosi/TGCN-GSL-PyTorch`):

```bash
# A. Sparse controls (CorrTop30 + RandTop30, Los-loop PH1, seeds 42-46) ~5 min
bash run_stage33A_sparse_control.sh
# equivalently:
# /data/python-envs/pytorch/bin/python gsl_stage26/stage32_sparse_control.py --seeds 42 43 44 45 46 --ph 1

# B. Canonical T-GCN-GSL rerun (Los-loop, PH 1-4, seeds 42-46) ~2-3 h
bash run_stage33B_gsl_canonical.sh
# add --cyclic for the cGSL appendix variant; uncomment the gcn/sz blocks as desired

# C. SZ-Taxi multi-seed (PH 1-4, seeds 42-46, reuses existing DAGMA) ~30-40 min
bash run_stage33C_sz_multiseed.sh
```

Both experiment scripts also accept a 1-2 minute smoke-test invocation, e.g.
`python gsl_stage26/stage33_gsl_canonical.py --ph 1 --epochs 2 --seeds 42`.

## 10. Expected output directories and artifacts

| Run | Directory | Artifacts |
|---|---|---|
| A | `results/stage32_sparse_control/` | `stage32_sparse_control.json` (rows: method, seed, n_edges, rmse, mae, n_params, multilag_reference with per-lag counts and sum/union edge distinction), `run.log` |
| B | `results/stage33_gsl_canonical/` | `stage33_gsl_canonical_results.json` (protocol block + per-row method/backbone/variant/edges/rmse/mae/params; protocol keys batch_size=128, weight_decay=1e-4), `los_gsl_ph*_seed42_W_est.npy`, `los_gsl_ph*_seed42_A_binary.npy` (fresh, provenance-clean GSL graphs), `run_losloop_tgcn.log` |
| C | `results/stage33_sz_multiseed/` | `stage33_sz_multiseed_results.json` (legacy `method` keys + `canonical_name`, 5 seeds × 4 PHs × 3 methods), `run.log` |

The JSONs keep the sum-vs-union edge-count distinction for the multi-lag
reference (sum = 30 across lag graphs, union = 28), so matched-budget claims
stay auditable. Remember that `results*/` is gitignored — copy result folders
back to the Windows machine and archive the small JSONs before submission.

## 11. Remaining scientific ambiguity

1. **Single-graph GSL under the canonical protocol** — until run B completes,
   the main text can only cite appendix (original-protocol) GSL numbers or the
   stage-26 single-lag DAGMA rows. Expected, but not guaranteed: T-GCN-GSL
   lands between NoSpatial and MultiGSL. If it does not, the ACT I framing
   ("sparse learned graph beats dense physical graph") still holds via the
   appendix table, but the main-text row would need care.
2. **Sparsity-only explanation** — run A is the decisive test; the direction of
   the answer is expected but open until measured.
3. **SZ-Taxi PH=4 dip** — single-seed −0.02% may be seed noise; run C resolves
   it. Either outcome is publishable (stability strengthens the boundary
   claim; noise removes an awkward negative sign).
4. **GCN hidden width (100 vs 64)** — the original GCN rows used hidden=100;
   if a canonical GCN rerun is ever promoted beyond the appendix, width must be
   fixed (64 is the canonical choice).
5. **Negative DAGMA weights** — the original adjacency rule keeps only W>0. The
   committed W_est files contain no negative entries (verified), but a fresh
   DAGMA fit (run B) could produce them; the choice to keep only positive
   entries preserves original semantics and is documented in the script.

## 12. Proposed high-level Results narrative (recommendation only — manuscript NOT rewritten)

The current manuscript already implements most of this; the recommendation
below states the target state after runs A–C, with the old results presented as
**the empirical starting point**, never as legacy baggage:

* **ACT I — A learned sparse graph beats the dense physical graph.**
  Open the Results with the oversmoothing contrast (Physical 7.658 →
  NoSpatial 5.143; dense *learned* Corr-K8 also fails) and the single-graph
  GSL result as the observation that started the investigation: a sparse
  DAGMA-learned graph substantially outperforms the physical adjacency
  (canonical T-GCN-GSL row from run B; historical tables remain in the
  appendix with their clearly-labelled protocol).
* **ACT II — But what does the learned graph encode?**
  The no-graph control showed part of the GSL advantage was really the harm of
  density; the GCN/T-GCN cyclic-vs-acyclic asymmetry showed a single matrix
  conflates temporal roles. This motivates making time explicit.
* **ACT III — Lag-specific graphs are structurally distinct.**
  Multi-lag DAGMA blocks (Table `tab:lag_stats`, Fig 7): different lags carry
  different dependency structures (association, not causation — wording
  already in place).
* **ACT IV — Learned mixing over lag graphs.**
  Multi-seed (14.9%), parameter control, lag ablation, mixing-vs-fixed (7.1%
  additional gain), all five seeds.
* **ACT V — Robustness and the boundary.**
  15-minute resolution (+27.4%, *larger*, refuting the resolution explanation);
  matched-edge sparse controls (run A) closing the sparsity loophole;
  SZ-Taxi as a dataset-dependent boundary (run C giving mean±std), interpreted
  via graph learnability with resolution explicitly ruled out.

Presentation rules carried through: old numbers never justified by "they were
in the previous version"; protocol-incompatible tables stay only in the
appendix with explicit labels; causal language avoided throughout; the 15-minute
result is never used to explain SZ-Taxi.

---

### Verification summary for this stage

* Canary: 10/10 PASS, 21 s, behaviour-preserving refactor (Section 4).
* Manuscript: compiles clean — 0 errors, 0 undefined references/citations,
  0 overfull boxes, 23 pages (Section 6).
* No DAGMA run, no full training, no scientific numbers changed, no historical
  artifacts renamed; `stage27` results remain quarantined as invalid for
  forecasting.
