# Stage 34 — Final Experimental Runs A/B/C: Verification & Execution Report

**Date:** 2026-09-09
**Scope:** Goal 1 inspect/verify the prepared A/B/C experiment code · Goal 2 fix what was broken (Experiment B thresholding) · Goal 3 short smoke tests only · Goal 4 hand over exact commands (no expensive runs executed)
**Basis:** `Stage34.md`; Stage 33 state (commit `e2a2949`) preserved — no refactoring undone, no historical artifacts renamed.

---

## 1. Code verification

### Experiment A — Sparse matched-edge controls (`gsl_stage26/stage32_sparse_control.py`)

| Check | Result |
|---|---|
| Script status | **Correct as prepared** (Stage 32 script, canary-tested then, imports graph constructors from `models.multigsl`) |
| Protocol parity | Matches the canonical pipeline exactly: `SupervisedForecastTask(mse_with_regularizer)`, `set_seed`, Adam(lr=1e-3, wd=1e-4), batch 128, 50 epochs, feat_max = train max (verified numerically: los 70.0 = global max) |
| Edge-budget parity | Asserts lag-graph **sum = 30** (12+3+15) before running; union = 28 recorded separately; controls get k=30 (≥ union) — conservative in favour of the method; convention unchanged |
| Required methods | CorrTop30 + RandTop30 by default. `T-GCN-NoSpatial`, `T-GCN-MultiGSL`, `T-GCN-MultiGSL-Mix` reference numbers already exist under the identical protocol (`results/stage26_validation/`, 5 seeds) — **re-running them is unnecessary**; the JSON comparison combines A's rows with the Stage 26 rows |
| Modification made | Result JSON now embeds a `protocol` block (batch/lr/wd/hidden/loss/optimizer/seq_len/feat_max source) — provenance only, no behavioural change |
| Smoke test | **PASS** — 3 epochs, seed 42, CorrTop30: trains, evaluates, saves JSON, 2.2 s/3 epochs on GPU |
| Overwrite risk | Writes only `results/stage32_sparse_control/stage32_sparse_control.json` (gitignored, dir did not exist before) — **no historical result touched** |
| Ready for full run | **YES** |

### Experiment B — Canonical T-GCN-GSL rerun (`gsl_stage26/stage33_gsl_canonical.py`)

| Check | Result |
|---|---|
| Script status | **Required a scientific fix — applied and verified (see §2)** |
| Data split / normalization | 80/20 temporal split, feat_max = train max — correct; numerically identical to the original global max for both committed datasets (los 70.0, sz 86.4292) |
| DAGMA input | **Fixed:** now `train_norm[0::ph]` (the original per-PH subsampling) instead of the full train matrix for every PH |
| Thresholding | **Fixed:** `w_threshold=0.3` inside `fit()` + project rule `A = 1(W>0)` reproduces the original protocol bit-identically (see §2) |
| Training protocol | batch 128, wd 1e-4, lr 1e-3, 50 epochs, hidden 64 — matches the canonical protocol and Stage 33's specification |
| Seed handling | Seeds propagate to model init, data shuffling, and DAGMA (`np.random.seed`); DAGMA itself is deterministic (zero-init W, numpy, no RNG use) — 5-seed spread comes from the forecasting stage, as intended |
| PH handling | Sequences, training, and evaluation per PH ∈ {1..4}; one GSL graph per PH (offset-0 fit; documented deviation from the original per-offset union, whose committed per-offset edge sets were identical) |
| Model naming | Rows carry `method` (T-GCN / T-GCN-GSL / T-GCN-cGSL) and `canonical_name`; the physical row's `canonical_name` is now **`Physical`** per `doc/METHOD_NAMING_MAP.md` |
| Result recording | JSON contains protocol string, per-graph DAGMA metadata (lambda1, w_threshold, input construction, iterations, rows, edges, runtime), and per-row dataset/PH/seed/backbone/variant/n_edges/RMSE/MAE/n_params/protocol parameters |
| Provenance | Fresh `los_gsl_ph{1..4}_seed42_{W_est,A_binary}.npy` retained in `results/stage33_gsl_canonical/`; graphs cached on disk so forecasting can resume without refitting |
| Smoke test | **PASS** — tiny DAGMA (200/400 iters) + 2 epochs, PH1 seed 42: graph learned (0 edges at tiny iterations — expected), both variants train, JSON saved, ~32 s total |
| Overwrite risk | Writes only into `results/stage33_gsl_canonical/` (new directory; smoke artifacts deleted after the test) — **no historical result touched** |
| Ready for full run | **YES** |

### Experiment C — SZ-Taxi multi-seed (`gsl_stage26/stage33_sz_multiseed.py`)

| Check | Result |
|---|---|
| Script status | **Correct as prepared** |
| Data | `data/sz_speed.csv` / `sz_adj.csv` (N=156), 80/20 split, feat_max = train max (86.4292) — same loader as every canonical script |
| DAGMA reuse | Loads the existing `results/stage26_validation/sz_ph{ph}_seed42_L3_lag_{1,2,3}.npy` blocks, thresholded at |W|>0.1 (2 cross-sensor edges per PH, verified) — **no recomputation** |
| Method parity | All three methods consume identical sequences/protocol; models instantiated from `models.multigsl` |
| Seed propagation | `set_seed(seed)` before each training → model init and shuffling vary by seed (verified in smoke test: 3 methods × 1 seed, all trains ran) |
| PH coverage | PH 1–4 all evaluated; missing-block case prints an error and skips rather than crashing |
| Naming | Legacy JSON keys (`NoGraph`, `MultiGraphTGCN_fixed`, `GatedMultiGraphTGCN`) + `canonical_name` — matches the Stage 33 convention |
| Modification made | Result JSON now embeds `protocol` and `dagma_blocks` provenance blocks (explicit `reused: true`, threshold) — provenance only |
| Smoke test | **PASS** — 3 epochs, PH1 seed 42: NoSpatial 4.25, MultiGSL 4.13, Mix 4.14 (smoke numbers, 3 epochs only — **not scientific results**) |
| Overwrite risk | Writes only `results/stage33_sz_multiseed/stage33_sz_multiseed_results.json` (gitignored, dir did not exist before) — **no historical result touched** |
| Ready for full run | **YES** |

---

## 2. Required modification: Experiment B adjacency rule (critical, verified)

**Problem.** The prepared script fitted DAGMA with `w_threshold=0.0` and then kept `A = 1(W>0)`. But DAGMA applies `w_threshold` **post-hoc inside `fit()`** (`dagma/linear.py:354`: `W_est[|W_est| < w_threshold] = 0`), and the *original* pipeline called `fit(X, lambda1)` relying on the **library default 0.3**. Evidence:

* Archived raw (unthresholded) SZ PH1 fit: `A=1(W>0)` yields **13,704 edges** (mostly |w|<1e-3 noise, cyclic → no DAG); the original rule `1(W>0 & |W|≥0.3)` yields exactly **8 edges** — equal to the committed original graph (`data/W_est_shenzhen_pre_len1.npy`, all nonzero entries in [0.31, 0.62]).
* Committed original `W_est_*` files for both datasets have min|w| > 0.30 (nothing between 0 and 0.3 survived), confirming the library default was active.
* `archive/.../dagma_fresh/GRAPH_CONSTRUCTION_AUDIT.md` (Stage 18-era) documents the same mechanism.
* Stage 33 report §11 had flagged negative-weight handling as open; with the 0.3 threshold, negative entries are excluded a fortiori (max |negative| on archived fits ≈ 0.05), and the rule "keep only W>0" is preserved.

**Fix applied** (minimal, in `learn_gsl_graph` only):

1. `model.fit(X, lambda1, w_threshold=0.3, ...)` — makes the original's *implicit* protocol explicit;
2. `X = train_norm[0::ph]` — the original per-PH input construction;
3. `A = 1(W>0)` + diagonal removal — the original project rule, unchanged;
4. Provenance fields added to the saved graph metadata; docstring/runner headers updated.

**Verification without any DAGMA retraining:** applying the fixed rule to the archived raw SZ fit reproduces the committed original SZ PH1 adjacency **bit-identically** (8/8 edges, identical support). The fresh Los-loop graphs will be re-learned from the training split by the full run — as Stage 34 requires (clean provenance), and the script refuses nothing: it is the documented original protocol, not a new choice.

---

## 3. Exact execution commands

Environment: `/data/python-envs/pytorch/bin/python` (torch 2.13.0+cu130, CUDA on RTX 3090, dagma OK). Repo root: `/data/git/mamintoosi/TGCN-GSL-PyTorch`.

```bash
# A — Sparse matched-edge controls (Los-loop, PH=1, seeds 42-46)   ~5-10 min
bash run_stage33A_sparse_control.sh
# equivalently:
# /data/python-envs/pytorch/bin/python gsl_stage26/stage32_sparse_control.py --seeds 42 43 44 45 46 --ph 1

# B — Canonical T-GCN-GSL rerun (Los-loop, PH 1-4, seeds 42-46)    ~5-9 h (mostly DAGMA)
bash run_stage33B_gsl_canonical.sh
# equivalently:
# /data/python-envs/pytorch/bin/python gsl_stage26/stage33_gsl_canonical.py --dataset losloop --models tgcn --phs 1 2 3 4
# optional appendix extras (uncommented blocks in the runner):
#   add --cyclic                      (cGSL variant)
#   add a second run --models gcn     (GCN-GSL, appendix asymmetry row)

# C — SZ-Taxi multi-seed (PH 1-4, seeds 42-46, existing DAGMA)     ~20-40 min
bash run_stage33C_sz_multiseed.sh
# equivalently:
# /data/python-envs/pytorch/bin/python gsl_stage26/stage33_sz_multiseed.py --seeds 42 43 44 45 46 --phs 1 2 3 4
```

Notes:
* B caches its DAGMA graphs (`los_gsl_ph{ph}_seed42_*.npy`); if a run is interrupted, re-running resumes without refitting finished PHs. **Do not keep 0-edge graphs from tiny-iteration smoke tests** (none are present now).
* The runners `taskset -c 2,3` the processes. DAGMA is CPU-numpy and single-thread-bound, so this mainly affects BLAS; if B's DAGMA phase feels slow, run the direct `python ...` command without taskset.
* Suggested order: **A** (fast, unblocks the table) → **C** (fast) → **B** (overnight).

## 4. Expected outputs

| Run | Directory | Artifacts |
|---|---|---|
| A | `results/stage32_sparse_control/` | `stage32_sparse_control.json` (protocol block; rows: method, seed, n_edges, rmse, mae, n_params, train_time_s; `multilag_reference` with per-lag counts, sum=30 vs union=28), `run.log` |
| B | `results/stage33_gsl_canonical/` | `stage33_gsl_canonical_results.json` (protocol string + per-PH DAGMA graph metadata + rows: dataset, ph, seed, backbone, variant, method, canonical_name, graph, n_edges, rmse, mae, n_params, epochs, hidden_dim, batch_size, weight_decay, loss), `los_gsl_ph{1..4}_seed42_W_est.npy`, `los_gsl_ph{1..4}_seed42_A_binary.npy`, `run_losloop_tgcn.log` |
| C | `results/stage33_sz_multiseed/` | `stage33_sz_multiseed_results.json` (dagma_blocks + protocol provenance; rows: dataset, ph, seed, method, canonical_name, model, n_edges, rmse, mae, n_params; 5 seeds × 4 PHs × 3 methods), `run.log` |

All three directories are gitignored — copy the JSONs (and B's `.npy` graph artifacts) back to the archival machine after the runs.

## 5. Runtime estimates (measured where possible)

| Run | Component | Estimate | Basis |
|---|---|---|---|
| A | 10 trainings (2 methods × 5 seeds), 50 epochs, PH1 | **~5–10 min** GPU | smoke: 2.2 s / 3 epochs → ~35 s per training |
| B | DAGMA × 4 PHs (207 vars, library-default 180k iters, CPU) | **~1–1.5 h per PH → ~5–7 h** (add ~1–2 h if taskset-limited) | archived benchmark: 156-var SZ = 52–76 min/PH at ~47 it/s; smoke: ~43 it/s at 207 vars |
| B | Forecasting 2 variants × 5 seeds × 4 PHs × 50 epochs | **~15–25 min** GPU | smoke: 0.5–0.8 s / 2 epochs per training |
| C | 60 trainings (3 × 5 × 4), 50 epochs, SZ | **~20–40 min** GPU | smoke: ~15 s / 3 epochs per training |

## 6. Scientific role of each experiment

* **A (Sparse matched-edge controls)** answers: *is the T-GCN-MultiGSL-Mix gain just sparsity?* CorrTop30 (strongest heuristic) and RandTop30 (floor) at the same 30-edge budget as the proposed method. Lands as two rows in `tab:oversmoothing` (Section "Dense Physical Graphs and Oversmoothing") — closes Reviewer 1 W5.
* **B (Canonical T-GCN-GSL rerun)** supplies the ACT-I main-text baseline: the single DAGMA-learned graph vs. the physical graph under the canonical protocol (batch 128, wd 1e-4, 5 seeds, provenance-clean graph). Its rows extend `tab:multiph`, letting Section "From Single-Graph GSL to Multi-Lag Structure" cite canonical numbers instead of pointing to the appendix's original-protocol table (which remains as the clearly-labelled historical record). **The ordering NoSpatial < GSL < Mix is hypothesized, not assumed** — the script reports whatever happens.
* **C (SZ-Taxi multi-seed)** determines whether the marginal SZ-Taxi advantage (≤0.3% PH1–3, −0.02% PH4, single seed) is stable or seed noise, upgrading `tab:sz_multiph` to 5-seed mean±std. Either outcome is publishable: stability strengthens the dataset-boundary claim; a noise verdict removes an awkward negative sign.

## 7. Manuscript integration (planned, not yet executed)

* No manuscript text was changed in this stage (only the one `canonical_name` correctness fix in B's output JSON).
* Integration after the runs, per the Stage 33 narrative plan: A → two rows + one sentence in §Oversmoothing; B → GSL rows in the multi-horizon table + rewritten ACT-I paragraph anchored on canonical numbers; C → `tab:sz_multiph` becomes mean±std + the boundary paragraph reports the seed-stability verdict. Old/appendix numbers stay in the appendix, never described as "old" or "kept for continuity" in the main text. `GCN-cGSL`/`T-GCN-cGSL` remain appendix-only.

## 8. Verification summary for this stage

* Smoke tests: A PASS (GPU), B PASS (CPU DAGMA + GPU training), C PASS — all ≤ ~1 min.
* CLI parsing (`--help`): A/B/C PASS. Syntax (`py_compile`): all modified files PASS.
* B adjacency-rule fix verified **bit-identical** against the committed original SZ graph, without any DAGMA retraining.
* No full experiment executed; no scientific result produced or inferred; smoke-test numbers are labelled as smoke tests and their artifacts were deleted.
* Working tree: 4 modified files (`stage33_gsl_canonical.py`, `stage32_sparse_control.py`, `stage33_sz_multiseed.py`, `run_stage33B_gsl_canonical.sh`) + this report; Stage 33 refactoring untouched.
