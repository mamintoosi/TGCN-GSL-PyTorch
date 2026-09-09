# Stage 36 — DAGMA Signed-Weight Audit and Final Experiment Readiness

**Date:** 2026-09-09
**Scope:** Sign-semantics audit of every DAGMA→graph path; adoption of the absolute-magnitude support rule as canonical; historical-artifact isolation; final GO/NO-GO. No full experiment executed, no manuscript change, no model/protocol redesign.

---

## 1. DAGMA → adjacency paths audited (Goal 1)

Traced from the actual generation code and artifact bytes (not filenames):

| # | Path | `.npy` contents | Threshold | Rule | Negatives | Directed | Diag | `A+I` only in message passing |
|---|---|---|---|---|---|---|---|---|
| 1 | **B**: `stage33_gsl_canonical.py → learn_gsl_graph()` | raw `W_est` saved alongside binary `A` | `w_threshold=0.3` inside `fit()` | **was** `1(W>0)` → **now** `1(|W|>0)` | now retained | yes | removed | yes (`calculate_laplacian_with_self_loop`) |
| 2 | **A** (and multi-lag generally): `stage26_run_dagma.py` saves **raw** lag blocks → `binary_graph()` in `models/multigsl.py` | **raw W** blocks (generator ran `w_threshold=0.0`) | `|W| > 0.1` at consumers | **already `abs`** | already retained (none exist) | yes | removed | yes |
| 3 | **C**: `stage33_sz_multiseed.py → load_multilag_blocks → binary_graph` | same raw blocks | `|W| > 0.1` | **already `abs`** | already retained (none exist) | yes | removed | yes |
| 4 | `stage26_validation.py`, `stage29_los15min.py`, `stage26_evaluate.py`, `stage26_train_with_logging.py`, `stage26_resolution_experiment.py` | same raw blocks | `|W| > τ` | **already `abs`** | already retained | yes | removed | yes |
| 5 | Historical: `utils/data/spatiotemporal_csv_data.py` (`use_gsl>0`, via `main.py`/Colab) | library-thresholded W (0.3 inside `fit()`) | library 0.3, then `W>0` union | positive-only | discarded (none exist) | yes | **not** removed (self-loops re-added later; unchanged historical behaviour) | yes |
| 6 | `models/multigsl.py` helpers: `binary_graph`, `correlation_topk_graph`, `random_edge_graph` | W / corr / random | `|W|>τ` / top-k | **already `abs`** / n/a | retained | yes | removed | yes |

Key file-content findings:
* The Stage 26 multi-lag DAGMA artifacts (`results/stage26_validation/{los,sz}_ph*_seed42_L3_{W_full,current,lag_*}.npy`) store **raw unthresholded W** (generator metadata: `w_threshold: 0.0`). All sign handling therefore lives in the consumers — and every consumer uses `binary_graph` = `abs(W) > τ`.
* The only positive-only rule in the repo was **Experiment B's** reconstruction (and the historical pipeline it mirrors). That is the one changed here.

## 2. Sign handling per experiment (Goal 2, current state)

* **A** — absolute-magnitude support **already** (via `binary_graph`); unchanged; provenance block extended to record the rule explicitly.
* **B** — **switched** from `A = 1(W>0)` to the canonical `A = 1(|W| > 0)` (diagonal removed), threshold unchanged at the explicit original default 0.3; both rules recorded in metadata.
* **C** — absolute-magnitude support **already**; provenance extended.
* The models remain **binary** message-passing (`A+I` inside the Laplacian); the sign is not an edge weight; no signed convolution introduced.

## 3. Do negative coefficients actually occur above threshold? (measured, per artifact)

* **Los-loop Stage 26 lag blocks (A's reference, 12 files):** `|W|>0.1` counts are 12/5/15 per lag — **identical** to `W>0.1` counts; zero negatives. Sum = 30 (union 28) unchanged.
* **SZ-Taxi Stage 26 lag blocks (C, 12 files):** `|W|>0.1` = 0/1/2 per lag = positive-only counts; zero negatives. 5 edges per PH unchanged.
* **Archived raw single-graph fits (B's rule applied to real data, SZ PH1–4):** `|W|≥0.3` = 8 coefficients each, all positive (max |negative| on raw fits = 0.013).
* **Historical committed `W_est` files (8):** zero negative survivors at |W|≥0.3 in any slice.
* **Conclusion:** on every artifact in this repository, negative DAGMA coefficients never reach the threshold; **the revised rule produces graphs identical to the old rule on this data** (verified bit-level: SZ PH1 old 8 = revised 8, identical support). The change is semantic/explicit, not numerical — except in the hypothetical case a future fit produces a negative survivor, which the new rule now retains by design.

## 4. Exact graph changes caused by the rule switch

* A: **none** (rule already absolute-magnitude; edge counts unchanged: 12+3+15=30, union 28).
* C: **none** (edge counts unchanged: 5/PH).
* B: **no measurable change on this data** (rule verified identical on the archived raw fits); the fresh Los-loop fits by the full run will record `n_positive_surviving` and `n_negative_surviving` separately, so if any negative survivor ever appears it will be visible in provenance rather than silently dropped.
* Synthetic unit check (sanity of the new rule): on a crafted signed W, old rule keeps 2 edges, revised keeps 5 (3 extra negative edges), sub-threshold entries dropped, diagonal removed — behaviour exactly as specified.

## 5. Which experiments must be rerun? (classification per Goal 2)

| Experiment | Classification | Action |
|---|---|---|
| **A** | **Unaffected** (already absolute-magnitude support) | None — ready |
| **B** | **Affected; code updated; graph regenerated cheaply as part of the run** | Uses the revised rule; no rerun of anything already executed (nothing was executed) |
| **C** | **Unaffected** (already absolute-magnitude support) | None — ready |

No existing numerical result is invalidated (none exist for A/B/C yet), and no graph artifact needed regeneration: the A/C blocks were already raw-W + absolute-threshold at the consumers.

## 6. Existing graph artifacts to regenerate

**None.** All A/C artifacts are already used under the revised policy. B creates its own fresh graphs at run time (deleted after the smoke test, so the full run starts clean).

## 7. Historical `W_est` isolation (Goal 3)

* Moved with `git mv` (history-preserving renames) from `data/` to **`archive/historical_submission/`**: `W_est_{losloop,shenzhen}_pre_len{1..4}.npy` (8 files). Files untouched otherwise.
* Updated the single loader that referenced them (`utils/data/spatiotemporal_csv_data.py`) to the new path, with a comment; the historical `main.py`/Colab protocol keeps working.
* Verified nothing else in the active tree loads them (`tests/test_gsl_clean.py` only does an existence-conditional print and still passes conceptually; Colab notebooks document the historical behaviour and are not run).
* Effect: the active `data/` path now contains only raw inputs and physical adjacencies — a historical graph estimate can no longer be picked up by any active path, even accidentally.
* The `data/correlation_*` adjunct files were left in place: they have no loader in any active code path and are small heuristic artifacts of the original study; moving them is unnecessary churn.

## 8. Exact files modified

| File | Change |
|---|---|
| `gsl_stage26/stage33_gsl_canonical.py` | Support rule → `A = 1(abs(W) > 0)` (diagonal removed); metadata now records threshold rule, support rule, `n_positive_surviving`, `n_negative_surviving`, `n_coefficients_surviving_abs_threshold`, `n_final_binary_edges`, `freshly_fitted`; docstrings updated to the Stage 36 policy |
| `gsl_stage26/stage32_sparse_control.py` | Provenance: `threshold_rule`, generator provenance, `negative_coefficients_above_threshold` |
| `gsl_stage26/stage33_sz_multiseed.py` | Provenance: `threshold_rule`, generator provenance, per-PH positive/negative/edge counts |
| `utils/data/spatiotemporal_csv_data.py` | Historical `W_est` path → `archive/historical_submission/` (+ comment) |
| `run_stage33B_gsl_canonical.sh` | Header documents the Stage 36 canonical support rule |
| `archive/historical_submission/W_est_*.npy` | 8 files moved here (renames, unmodified content) |

Method names untouched (Goal 7): `T-GCN`, `T-GCN-GSL`, `T-GCN-MultiGSL`, `T-GCN-MultiGSL-Mix` etc. all unchanged; the revision is presented as the corrected/evolved implementation of the submitted methodology.

## 9. Commands for the future full experiments (unchanged)

```bash
# A — Sparse matched-edge controls (~5-10 min)
bash run_stage33A_sparse_control.sh

# B — Canonical T-GCN-GSL rerun (~5-9 h, mostly DAGMA)
bash run_stage33B_gsl_canonical.sh

# C — SZ-Taxi multi-seed (~20-40 min)
bash run_stage33C_sz_multiseed.sh
```

(Optionally prefix B's DAGMA phase with `OPENBLAS_NUM_THREADS=1` for bit-level cross-environment reproducibility of the raw W — see Stage 35 §2.)

## 10. Confirmation

**No full experiment was executed** (A/B/C untouched beyond ≤2-epoch smoke tests with tiny DAGMA iterations; smoke artifacts deleted afterwards so the full B run starts clean). No multi-hour DAGMA fit ran. No scientific result was produced. No manuscript change, no model/loss/optimizer/protocol change (Goal 4 respected).

## 11. GO/NO-GO

```text
GO for A
GO for B
GO for C
```

All three experiments are execution-ready under the finalized canonical graph policy: absolute-magnitude thresholding at 0.3 (explicit form of the original library default), support = nonzero coefficients of either sign, diagonal removed, binary adjacency with `A+I` only inside message passing.
