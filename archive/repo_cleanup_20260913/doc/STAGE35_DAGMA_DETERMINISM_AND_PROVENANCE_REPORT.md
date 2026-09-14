# Stage 35 — DAGMA Determinism, Threshold and Provenance Audit

**Date:** 2026-09-09
**Scope:** Pre-flight audit before the expensive A/B/C runs. No full experiment executed, no manuscript change.
**Environment audited:** `/data/python-envs/pytorch/bin/python` — Python 3.12, numpy 2.5.2, scipy 1.18.1, torch 2.13.0+cu130 (CUDA, RTX 3090), dagma 0.1.0 (`/data/python-envs/pytorch/lib/python3.12/site-packages/dagma/linear.py`), BLAS/LAPACK = scipy-openblas 0.3.34 (4 threads default, `OMP_NUM_THREADS`/`MKL_NUM_THREADS` unset, 4 cores).

---

## 1. DAGMA randomness audit (Goal 1)

**Method:** line-by-line inspection of the installed `dagma/linear.py` (`DagmaLinear`, the only class our scripts use), following the exact call path of `gsl_stage26/stage33_gsl_canonical.py → learn_gsl_graph() → DagmaLinear(loss_type="l2").fit(X, lambda1, w_threshold=0.3, warm_iter, max_iter)`.

**Findings across the complete execution path:**

| Randomness source | Present? | Evidence |
|---|---|---|
| NumPy RNG (`np.random.*`) | **No** | `linear.py` contains no `np.random`, `randn`, `rand`, `RandomState`, `permutation` (grep over the installed file: 0 matches) |
| Python `random` | **No** | no `import random`, no RNG use |
| PyTorch RNG | **No** | `linear.py` imports only numpy/scipy/tqdm; the linear solver never touches torch |
| W initialization | **Deterministic** | `fit()` line 325: `self.W_est = np.zeros((d, d))` — the optimizer starts from the zero matrix, not a random init |
| Adam moments | **Deterministic** | `self.opt_m, self.opt_v = 0, 0` at each `minimize()` call |
| Stochastic optimization | **No** | full-batch deterministic gradients: `cov = X.T@X/n` precomputed once; Adam is a deterministic function of the (deterministic) gradient sequence |
| Random subsampling/permutations | **No** | X is used as given; no shuffling anywhere |
| Randomized linear algebra | **No** | `sla.inv` / `la.slogdet` are deterministic dense LU/QR-based routines; no randomized SVD/eigsh |
| Data-dependent `while` loops | **Yes (not randomness)** | M-matrix domain checks can halve `lr`/raise `s`; these depend on floating-point state, not on any RNG |

**Conclusion:** `DagmaLinear` is **algorithmically deterministic**: same input + same library build ⇒ identical `W_est`, independent of any seed. The per-fit `np.random.seed(seed)` in `learn_gsl_graph()` is a harmless no-op for DAGMA.

Distinction requested:
1. *Algorithmic randomness* — none in DAGMA.
2. *Numerical nondeterminism from parallelism* — none for repeated runs at a fixed thread count (§2, bit-identical cross-process); only sub-ULP variation across *different thread counts*.
3. *Randomness in downstream forecasting* — **yes, by design**: `set_seed(s)` in the canonical scripts seeds Python/NumPy/torch before model construction (Xavier init) and DataLoader shuffling; the 5-seed spread measures exactly this.

## 2. Thread/numerical determinism audit (Goal 2)

**Stack:** numpy 2.5.2 and scipy 1.18.1 both link **scipy-openblas 0.3.34** (verified via `numpy.__config__` and threadpoolctl); `sla.inv`/`la.slogdet` route to OpenBLAS/LAPACK. Default 4 threads on 4 cores.

**Empirical test** (tiny synthetic, d=50, n=400, λ1=0.01, w_threshold=0.3, 800/1600 iters — ran in ~15 s per fit; **not** a traffic fit):

| Comparison | max abs diff in W | Bit-identical? | Adjacency (project rule) identical? |
|---|---|---|---|
| Two processes, same threads (4) | 0.000e+00 | **Yes** | **Yes** (54 edges) |
| 4 threads vs 1 thread | 7.772e-16 (~1 ULP) | No | **Yes** (54 edges, identical support) |

**Conclusion:** repeated runs at the same configuration are **bit-identical**. Changing the BLAS thread count perturbs W at the ~1e-16 level (one float64 ULP); the post-threshold binary adjacency is **unchanged**. This matches OpenBLAS's documented reduction-order behaviour. Practical consequence: if a run were ever split across machines/thread settings, W could differ at machine-epsilon level, but the learned *graph* (edges) and everything downstream of it is stable. Recommended (optional, belt-and-braces): pin `OPENBLAS_NUM_THREADS=1` for the DAGMA phase to make even the raw W bit-reproducible across environments.

## 3. Threshold and sign semantics (Goal 3)

Verified against the installed implementation (`dagma/linear.py` line 354, end of `fit()`) and empirically on the archived raw (unthresholded) SZ PH1 fit:

```text
W = DAGMA.fit(X, lambda1, w_threshold=0.3, ...)        # optimization; W init = zeros
W[|W| < 0.3] = 0          # INSIDE fit(): absolute magnitude, signs preserved
A = (W > 0)               # project rule: POSITIVE entries only
A[i,i] = 0                # diagonal removed (np.fill_diagonal)
```

1. **Can DAGMA return negative weights?** Yes in principle (the threshold uses `np.abs`), but after thresholding only entries with |W| ≥ 0.3 survive. On the archived raw SZ fit: 13,798 positive / 10,538 negative raw entries, **zero** entries of either sign at |W| ≥ 0.3 except 8 positive ones; max |negative| = 0.013. Empirically negative survivors do not occur on our data.
2. **Is `w_threshold` applied by absolute magnitude?** Yes — `W_est[np.abs(W_est) < w_threshold] = 0` (verified also on a synthetic `[0.5, −0.5, 0.29, −0.29]` probe → `[0.5, −0.5, 0, 0]`).
3. **Does the project adjacency keep only positive weights?** Yes — `A = 1(W > 0)`.
4. **Would a negative weight with |W| ≥ 0.3 become an edge?** **No.** It survives DAGMA's threshold but is then discarded by the project rule. This preserves the original submission's semantics exactly (same two-step rule); on real data the case never arises (see 1).
5. **Are diagonal entries removed?** Yes — `np.fill_diagonal(A, 0)` in `learn_gsl_graph()`; additionally the DAG constraint itself drives the raw diagonal to ~0 (max |diag| = 5.1e-06 on the raw fit). Note self-loops are later re-added *inside* the Laplacian (`A+I`), which is the standard message-passing convention and unchanged from all previous stages.

**No implementation bug found; the scientific rule is unchanged** (the Stage 34 fix made the original *implicit* threshold explicit — that is documentation, not a rule change).

## 4. Historical W_est isolation audit (Goal 4)

**Inventory** — all committed under `data/` (git-tracked, dated Mar 2025, original submission):
`W_est_{losloop,shenzhen}_pre_len{1,2,3,4}.npy` (8 files), plus `correlation_*` and `correlation_cgsl_*` files (adjuncts of the original heuristic/cGSL studies).

**Who loads them?**

| Consumer | Loads historical W_est? | Risk to A/B/C |
|---|---|---|
| `utils/data/spatiotemporal_csv_data.py` (`compute_adjacency_matrix`, `use_gsl>0`) | **Yes** — the only active-tree loader | **None**: that path is entered only via `main.py` or the two Colab notebooks (original protocol). No Stage 26/32/33 script imports it |
| `main.py`, `main-{GCN,TGCN}-GSL-Colab.ipynb` | Yes (via the above) | None — historical entry points, not part of A/B/C |
| `gsl_stage26/stage33_gsl_canonical.py` (B) | **No** — fits its own graphs; writes `results/stage33_gsl_canonical/{los,sz}_gsl_ph*_seed42_{W_est,A_binary}.npy` | None. It *documents* the committed files (docstring comparison) but never loads them |
| `gsl_stage26/stage32_sparse_control.py` (A) | No — uses Stage 26 multi-lag blocks + corr/random constructors | None |
| `gsl_stage26/stage33_sz_multiseed.py` (C) | No — uses `results/stage26_validation/sz_ph*_seed42_L3_lag_*.npy` | None |
| `tests/test_gsl_clean.py`, `archive/**` | Yes | None — tests/archived audit scripts |

**Conclusion:** no active A/B/C experiment can silently consume a historical `W_est`. The naming collision worth knowing about: B's fresh artifacts contain `gsl_..._W_est.npy` but live in `results/stage33_gsl_canonical/` with a distinct filename pattern (`los_gsl_ph1_seed42_W_est.npy` vs `data/W_est_losloop_pre_len1.npy`) — no overlap.

**Recommendation (minimal, not executed):** physical relocation is **not necessary** (zero live coupling). If desired for belt-and-braces clarity, `git mv data/W_est_*.npy data/historical_submission/` (+ the `correlation_*` files) with a one-line path update in `spatiotemporal_csv_data.py`, in a *separate commit after* A/B/C complete — doing it now would churn the historical pipeline right before the runs for no safety gain. Keep as-is for this stage.

## 5. Experiment B provenance audit (Goal 5)

The JSON's `gsl_graphs` block now records, per PH: dataset, PH, seed, `lambda1`, `loss_type`, `w_threshold` (0.3, labelled as the original library default), `dagma_input` (`train_norm[0::PH]`), `warm_iter`/`max_iter`, `feat_max`, `train_rows`, **`n_nodes`, `n_input_rows`, `nnz_after_threshold`, `n_positive_edges_kept`, `n_negative_weights_at_or_above_threshold`, `n_diagonal_removed`, `max_abs_weight`**, adjacency-rule and formulation strings, `software` (python/numpy/scipy versions + `dagma.linear.__file__`), and a determinism note pointing at this report. Runtime per fit is recorded. The per-result rows already carry dataset/PH/seed/backbone/variant/method/canonical_name/n_edges/RMSE/MAE/n_params and the full protocol (batch 128, wd 1e-4, lr 1e-3, epochs, hidden 64, loss). **Provenance is now complete**; change was minimal (metadata only, verified by `py_compile` + `--help`).

## 6. DAGMA seeds vs forecasting seeds (Goal 6)

Verified in code and empirically:

* B: the DAGMA fit lives **outside** the seed loop — one fit per PH (cached to `.npy`, refit only if absent), and DAGMA is deterministic anyway (§1); seeds 42–46 affect only `set_seed()` before each forecasting train/eval.
* C: `load_multilag_blocks(ph, seed=42, ...)` loads the Stage 26 blocks **once per PH**, before the seed loop; all three methods share them; `set_seed(seed)` precedes each training.

Empirical confirmation (tiny 2-seed smoke run, 2 epochs, deleted afterwards): DAGMA messages printed once per PH; seeds 42/43 gave different RMSEs for every method (e.g. NoSpatial 4.446 vs 4.517), i.e. seeds propagate to model init/shuffling exactly as designed.

**The implementation matches the intended design** — one deterministic graph per (dataset, PH), five forecasting seeds per method. Smallest fix needed: none.

## 7. Recommendation on canonical reruns (Goal 7)

| Method | Final role | Canonical result exists / needed? | Recommendation |
|---|---|---|---|
| T-GCN-GSL | Main-text ACT-I representative of the old approach | Supplied by run **B** (this is B's purpose) | **Rerun (run B)** |
| T-GCN-NoSpatial | Main-text critical baseline | Yes — 5 seeds, both resolutions (Stage 26/29) | No rerun |
| Physical | Main-text density/oversmoothing contrast | Single-seed at PH1-4 (verified); qualitative effect is large and replicates on SZ | No rerun required; optional cheap 5-seed top-up later if a reviewer asks |
| T-GCN-MultiGSL | Main-text fixed-assignment ablation | Yes — 5 seeds (Stage 26) + 15-min (Stage 29); C adds SZ 5-seed | No rerun |
| T-GCN-MultiGSL-Mix | Proposed method | Yes — 5 seeds, 2 resolutions; C adds SZ 5-seed | No rerun |
| GCN-GSL | Appendix only (architecture asymmetry) | Original protocol only | **Optional** rerun (runner B's commented `--models gcn` block; +~30 min forecasting + GCN-width decision hidden=64) — not needed for the main story |
| T-GCN-cGSL | Appendix historical table only | No | No rerun (removed from main story; adds nothing once lag graphs exist) |
| GCN-cGSL | Appendix historical table only | No | No rerun (same reasoning) |

Principle applied: only methods that carry main-text weight under the final protocol need canonical numbers; that is exactly T-GCN-GSL (run B), with GCN-GSL as an optional appendix completeness item.

## 8. Exact code changes made in this stage

1. `gsl_stage26/stage33_gsl_canonical.py` — extended `learn_gsl_graph()` graph metadata with full provenance (nodes, input rows, nnz after threshold, positive edges kept, negative weights ≥ 0.3, diagonal removed, max |W|, software versions, determinism note). Metadata only; no behavioural change.
2. No other file changed. In particular: no manuscript change, no historical file moved, no scientific rule changed.

## 9. Tests executed and results

| Test | Result |
|---|---|
| DAGMA source grep for RNG across the call path | 0 matches — deterministic |
| Synthetic determinism, 2 processes × 4 threads (d=50, ~15 s/fit) | `max|ΔW| = 0.0`, bit-identical; identical adjacency (54 edges) |
| Synthetic determinism, 4 threads vs 1 thread | `max|ΔW| = 7.8e-16` (~1 ULP); adjacency **identical** |
| Threshold semantics probe (`np.abs` rule, sign preservation) | Confirmed |
| Archived raw-fit sign statistics (SZ PH1) | 0 negative entries at \|W\| ≥ 0.3; max \|negative\| = 0.013 |
| `py_compile` + `--help` on modified B | PASS |
| 2-seed × PH1 × 2-epoch C smoke (seed propagation check) | PASS; artifacts deleted |
| Stage 34 smoke tests (A: 3-epoch GPU; B: tiny DAGMA + 2 epochs; C: 3 methods PH1) | PASS (previous stage); all artifacts cleaned |

No full traffic DAGMA fit, no full A/B/C run, no multi-hour training was executed.

## 10. GO/NO-GO

```text
GO for A
GO for B
GO for C
```

**Required change before starting: none.** The Stage 34 B fix (explicit `w_threshold=0.3`, per-PH input `train[0::PH]`) is confirmed correct against the original protocol; determinism is established (algorithmic: yes; thread-level: sub-ULP, adjacency-stable); provenance in B is complete; seed design is correct; historical artifacts cannot leak into the new runs.

Optional (not required): run B's DAGMA phase with `OPENBLAS_NUM_THREADS=1` for bit-level cross-environment reproducibility of the raw W files; and consider the post-runs archival move of `data/W_est_*.npy` (§4) in its own commit.

Suggested order remains: **A** (~5–10 min) → **C** (~20–40 min) → **B** (~5–9 h, mostly DAGMA).
