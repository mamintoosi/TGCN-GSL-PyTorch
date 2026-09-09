# Stage 40 — Repository Restructuring Report

**Date:** 2026-09-09
**Status:** Complete — pipeline ready for user execution

---

## 1. Initial Repository Structure

The repository had accumulated artifacts across 8+ stages. Key issues:

- **Scattered DAGMA outputs**: Multi-lag blocks in `results/stage26_validation/`, contemporaneous in `results/stage33_gsl_canonical/`, historical in `archive/historical_submission/`
- **Inconsistent naming**: `Physical` vs `T-GCN`, `NoGraph` vs `T-GCN-NoSpatial`, `GatedMultiGraphTGCN` vs `T-GCN-MultiGSL-Mix`
- **No unified experiment pipeline**: Each stage had its own script with different protocols
- **GCN variants not systematically covered**: Only basic GCN existed; no GSL/cGSL/MultiGSL variants
- **Method registry incomplete**: Only T-GCN family entries, no GCN family, no GSL/cGSL entries

---

## 2. Final Structure

```
gsl_stage40/                              # NEW — Stage 40 canonical artifacts
├── dagma/
│   └── dagma_artifacts_manifest.json     # Complete DAGMA inventory
├── scripts/
│   ├── stage40_run_all.py                # Main resumable experiment launcher
│   └── stage40_validate_smoke.py         # Lightweight smoke tests
├── experiment_manifest.json              # Full experiment matrix
├── configs/                              # (empty — configs in main experiment script)

results/stage40_canonical/                # NEW — canonical results location
├── dagma/                                # (for future DAGMA outputs)
├── training/                             # Per-experiment result JSONs
├── figures/                              # Regenerated figures
└── logs/                                 # Run logs

run_stage40_experiments.sh                # NEW — bash launcher
```

### Files Created
| File | Purpose |
|------|---------|
| `gsl_stage40/dagma/dagma_artifacts_manifest.json` | Complete DAGMA inventory with provenance |
| `gsl_stage40/experiment_manifest.json` | Full experiment matrix (11 variants × 2 datasets × 4 PHs × 5 seeds) |
| `gsl_stage40/scripts/stage40_run_all.py` | Resumable experiment launcher |
| `gsl_stage40/scripts/stage40_validate_smoke.py` | Lightweight validation tests |
| `run_stage40_experiments.sh` | Bash wrapper with logging |
| `results/stage40_canonical/` | Canonical results directory tree |

### Files Modified
| File | Changes |
|------|---------|
| `models/multigsl.py` | METHOD_REGISTRY expanded to 11 entries (7 T-GCN + 4 GCN); `build_model()` supports GCN backbone; all entries now have `name`, `adjacency`, `backbone` fields; `physical` name changed from `"Physical"` to `"T-GCN"` |
| `paper/generate_figures.py` | `LEGACY_TO_CANONICAL` updated for canonical names; `COLORS` dict expanded for all variants; figure labels use canonical names |

### Files NOT Modified (historical preservation)
- All files under `archive/` — untouched
- All files under `gsl_stage26/` — untouched
- All existing result files under `results/` — untouched
- Manuscript (`paper/sections/`, `paper/sn-article.*`) — untouched
- `paper/Response-to-Reviewers.md` — untouched
- `doc/` (existing reports) — untouched

---

## 3. DAGMA Output Inventory

### Multi-lag DAGMA (reusable, NO recomputation needed)

| Dataset | Source Dir | Files | Status |
|---------|-----------|-------|--------|
| Los-loop PH 1-4 | `results/stage26_validation/` | `los_ph*_seed42_L3_lag_{1,2,3}.npy` + `W_full.npy` + `metadata.json` | READY_TO_REUSE |
| SZ-Taxi PH 1-4 | `results/stage26_validation/` | `sz_ph*_seed42_L3_lag_{1,2,3}.npy` + `W_full.npy` + `metadata.json` | READY_TO_REUSE |

Provenance: Fitted by `stage26_run_dagma.py`, seed 42, lambda1=0.02 (los) / 0.01 (sz), w_threshold=0.0. Consumers apply `|W| > 0.1`.

Edge counts at threshold 0.1 (Los-loop PH 1): lag_1=12, lag_2=3, lag_3=15, total=30.

### Contemporaneous Single-graph DAGMA

| Dataset | Source Dir | Files | Status |
|---------|-----------|-------|--------|
| Los-loop PH 1-4 | `results/stage33_gsl_canonical/` | `los_gsl_ph*_seed42_A_binary.npy` + `W_est.npy` | READY_TO_REUSE |
| SZ-Taxi PH 1-4 | — | — | **NOT_YET_FITTED** (requires ~1.5-2.5h DAGMA computation) |

Los-loop contiguous DAGMA: 28 edges per PH (all positive, no negatives at threshold 0.3).

### Historical DAGMA (archival only)

| Location | Files | Status |
|----------|-------|--------|
| `archive/historical_submission/` | `W_est_{losloop,shenzhen}_pre_len{1-4}.npy` | ARCHIVAL_ONLY — used by `main.py` legacy pipeline, not suitable for Stage 40 canonical pipeline |

---

## 4. Model Variants Assessment

### T-GCN Family (7 variants) — All Implemented

| Variant | Class | Adjacency | DAGMA Required |
|---------|-------|-----------|---------------|
| T-GCN | `TGCN` | Physical road network | No |
| T-GCN-NoSpatial | `TGCN` | Identity (np.eye) | No |
| T-GCN-GSL | `TGCN` | Contemporaneous single-graph | Yes (contemporaneous) |
| T-GCN-cGSL | `TGCN` | Symmetrized GSL: A + A^T | Yes (contemporaneous) |
| T-GCN-MultiGSL | `MultiGraphTGCNFixed` | 3 lag-specific graphs | Yes (multilag) |
| T-GCN-MultiGSL-Weighted | `WeightedMultiGraphTGCN` | 3 lag-specific graphs | Yes (multilag) |
| T-GCN-MultiGSL-Mix | `GatedMultiGraphTGCN` | 3 lag-specific graphs (gated) | Yes (multilag) |

### GCN Family (4 variants) — Implemented, 3 MultiGSL variants EXCLUDED

| Variant | Class | Adjacency | DAGMA Required |
|---------|-------|-----------|---------------|
| GCN | `GCN` | Physical road network | No |
| GCN-NoSpatial | `GCN` | Identity (np.eye) | No |
| GCN-GSL | `GCN` | Contemporaneous single-graph | Yes (contemporaneous) |
| GCN-cGSL | `GCN` | Symmetrized GSL: A + A^T | Yes (contemporaneous) |

### GCN-MultiGSL/Mix/Weighted: NOT IMPLEMENTED (architectural justification)

The GCN class (`models/gcn.py`) processes the **entire input window** in a single graph-convolution step:

```python
# GCN forward: (batch, seq_len, num_nodes) -> (batch, num_nodes, hidden_dim)
# One Laplacian multiplication covers all timesteps simultaneously
ax = self.laplacian @ inputs_flat  # single graph conv over entire window
outputs = torch.tanh(ax @ self.weights)  # single linear transform
```

This is fundamentally different from TGCN, which processes one timestep at a time through a GRU cell. The three MultiGSL variants are designed for per-timestep processing:

- **MultiGSL** (fixed assignment): assigns lag graph `k` to timestep `t` — meaningless when all timesteps are processed simultaneously
- **MultiGSL-Weighted** (learned scalar weights): combines lag graphs into a single static graph — equivalent to just using the physical graph with different weights
- **MultiGSL-Mix** (per-node per-timestep gating): computes timestep-specific graph mixtures — impossible when GCN doesn't have per-timestep processing

Creating artificial GCN variants would violate the Stage 40 instruction: "Do not invent an artificial implementation merely to obtain naming symmetry."

---

## 5. Existing Results Mapping

### Already Available and Reusable

| Source | Methods | Datasets | PHs | Seeds | Status |
|--------|---------|----------|-----|-------|--------|
| `results/stage33_gsl_canonical/` | T-GCN, T-GCN-GSL | Los-loop | 1-4 | 42-46 | COMPLETE |
| `results/stage33_sz_multiseed/` | T-GCN-NoSpatial, T-GCN-MultiGSL, T-GCN-MultiGSL-Mix | SZ-Taxi | 1-4 | 42-46 | COMPLETE |
| `results/stage26_validation/` | T-GCN-NoSpatial, T-GCN-MultiGSL, T-GCN-MultiGSL-Mix | Los-loop | 1 | 42-46 | COMPLETE |
| `results/stage26_validation/` | T-GCN-NoSpatial, T-GCN-MultiGSL, T-GCN-MultiGSL-Mix | SZ-Taxi | 1-4 | 42 | COMPLETE |
| `results/stage32_sparse_control/` | CorrTop30, RandTop30 | Los-loop | 1 | 42-46 | COMPLETE |
| `results/stage29_los15min/` | T-GCN-NoSpatial, T-GCN-MultiGSL, T-GCN-MultiGSL-Mix | Los-loop (15min) | 1-4 | 42 | COMPLETE |

### Requires Training (Stage 40 pipeline)

| Variant | Datasets | PHs | Seeds | Notes |
|---------|----------|-----|-------|-------|
| T-GCN (physical) | Los-loop, SZ-Taxi | 1-4 | 42-46 | New canonical pipeline run needed |
| T-GCN-cGSL | Los-loop | 1-4 | 42-46 | Needs contemporaneous DAGMA (available for Los) |
| T-GCN-cGSL | SZ-Taxi | 1-4 | 42-46 | **Needs DAGMA fit first** |
| T-GCN-MultiGSL-Weighted | Los-loop, SZ-Taxi | 1-4 | 42-46 | Supplementary ablation |
| GCN | Los-loop, SZ-Taxi | 1-4 | 42-46 | All 4 GCN variants |
| GCN-NoSpatial | Los-loop, SZ-Taxi | 1-4 | 42-46 | All 4 GCN variants |
| GCN-GSL | Los-loop, SZ-Taxi | 1-4 | 42-46 | Needs contemporaneous DAGMA |
| GCN-cGSL | Los-loop, SZ-Taxi | 1-4 | 42-46 | Needs contemporaneous DAGMA |

### Requires DAGMA Computation

| Dataset | Construction | Status | Estimated Time |
|---------|-------------|--------|---------------|
| SZ-Taxi | Contemporaneous (for GSL/cGSL) | MISSING | ~1.5-2.5h |

---

## 6. Experiment Manifest Summary

**Total experiments in the matrix:** 11 variants × 2 datasets × 4 PHs × 5 seeds = **440 experiments**

**Already completed under previous stages (reusable):**
- Stage 33 GSL canonical (Los): 2 methods × 4 PHs × 5 seeds = 40
- Stage 33 SZ multiseed: 3 methods × 4 PHs × 5 seeds = 60
- Stage 26 validation (Los, PH=1): 3 methods × 1 PH × 5 seeds = 15
- Stage 26 validation (SZ): 3 methods × 4 PHs × 1 seed = 12
- Stage 32 sparse controls (Los): 2 methods × 1 PH × 5 seeds = 10

**Not yet run in canonical pipeline:** ~303 experiments (accounting for overlaps)

---

## 7. Correctness Verification

All 6 smoke tests pass:

| Check | Status | Detail |
|-------|--------|--------|
| Module imports | PASS | 9 modules imported successfully |
| All 11 variants instantiate | PASS | 11 variants instantiate + forward pass; registry has 11 entries |
| DAGMA graph files present | PASS | All DAGMA files present with correct shapes |
| Result JSONs readable | PASS | 126 result records readable across 3 source files |
| Registry naming consistency | PASS | 11 entries all have name, adjacency, backbone |
| GCN non-recurrent assessment | PASS | GCN has 96 params (= seq_len × hidden_dim), processes whole window at once |

Additional verification:
- **Skip mechanism works**: Re-running an already-completed experiment correctly skips it
- **MultiGSL-Mix end-to-end**: Successfully ran a 2-epoch smoke test reusing existing multilag DAGMA blocks
- **T-GCN end-to-end**: Successfully ran a 2-epoch smoke test with physical adjacency

---

## 8. User Execution Instructions

### Short Tests (you can run these immediately)

```bash
# Activate the correct Python environment
conda activate pth  # or: export PYTHON=/data/python-envs/pytorch/bin/python

# Run all smoke tests (~30 seconds)
python gsl_stage40/scripts/stage40_validate_smoke.py

# Dry-run to preview all experiments
python gsl_stage40/scripts/stage40_run_all.py --dry-run

# Quick smoke test of the pipeline (2 epochs, single experiment)
python gsl_stage40/scripts/stage40_run_all.py --variants physical --datasets losloop --phs 1 --seeds 42 --max-epochs 2
```

### Missing DAGMA Computation (run first)

```bash
# SZ-Taxi contemporaneous DAGMA (REQUIRED for GCN-GSL/cGSL on SZ-Taxi)
# Estimated time: ~1.5-2.5 hours
python gsl_stage26/stage33_gsl_canonical.py --dataset shenzhen --models tgcn --phs 1 2 3 4
```

### Full Experiment Runs

```bash
# Full T-GCN family, both datasets (estimated: ~8-12 hours)
python gsl_stage40/scripts/stage40_run_all.py --backbone tgcn

# Full GCN family, both datasets (estimated: ~6-10 hours)
python gsl_stage40/scripts/stage40_run_all.py --backbone gcn

# Everything (estimated: ~15-22 hours)
python gsl_stage40/scripts/stage40_run_all.py

# Using bash wrapper with logging
bash run_stage40_experiments.sh --backbone tgcn
bash run_stage40_experiments.sh --backbone gcn
```

### Dataset-Specific Runs

```bash
# Los-loop only
python gsl_stage40/scripts/stage40_run_all.py --datasets losloop

# SZ-Taxi only (after DAGMA is fitted)
python gsl_stage40/scripts/stage40_run_all.py --datasets shenzhen
```

### Variant-Specific Runs

```bash
# Just the proposed method (T-GCN-MultiGSL-Mix)
python gsl_stage40/scripts/stage40_run_all.py --variants multi_gsl_mix

# GSL and cGSL variants
python gsl_stage40/scripts/stage40_run_all.py --variants gsl cgsl

# Supplementary ablation only (MultiGSL-Weighted)
python gsl_stage40/scripts/stage40_run_all.py --variants multi_gsl_weighted
```

### Resumability

The launcher is fully resumable. If interrupted:
1. Re-run the same command — it detects existing results and skips them
2. Each experiment writes results atomically (write to .tmp, then rename)
3. Logs are appended (not overwritten) to `results/stage40_canonical/logs/`

### After All Experiments Complete

```bash
# Regenerate figures (reads from canonical results)
python paper/generate_figures.py
```

---

## 9. Directory Tree After Restructuring

```
gsl_stage40/
├── dagma/
│   └── dagma_artifacts_manifest.json
├── scripts/
│   ├── stage40_run_all.py
│   └── stage40_validate_smoke.py
├── configs/                     (empty — configs in main script)
└── experiment_manifest.json

results/
├── stage26_validation/          (HISTORICAL — untouched)
├── stage26_checkpoint/          (HISTORICAL — untouched)
├── stage27_resolution/          (HISTORICAL — untouched)
├── stage29_los15min/            (HISTORICAL — untouched)
├── stage30_forensic_audit/      (HISTORICAL — untouched)
├── stage31_manuscript_integration/ (HISTORICAL — untouched)
├── stage31_manuscript_reconstruction/ (HISTORICAL — untouched)
├── stage32_sparse_control/      (HISTORICAL — untouched)
├── stage33_gsl_canonical/       (HISTORICAL — DAGMA outputs reused)
├── stage33_sz_multiseed/        (HISTORICAL — results reused)
└── stage40_canonical/           (NEW — canonical pipeline)
    ├── dagma/                   (for future DAGMA outputs)
    ├── training/                (per-experiment result JSONs)
    ├── figures/                 (regenerated figures)
    └── logs/                    (run logs)

models/
├── multigsl.py                  (MODIFIED — 11-method registry)
├── tgcn.py                      (unchanged)
├── gcn.py                       (unchanged)
└── gru.py                       (unchanged)

paper/
└── generate_figures.py          (MODIFIED — canonical names in labels/colors)

run_stage40_experiments.sh       (NEW — bash launcher)
```

---

## 10. Key Design Decisions

1. **Physical → T-GCN naming**: The old `Physical` name is replaced by `T-GCN` in the new pipeline. Historical artifacts retain their original keys.

2. **GCN-MultiGSL excluded**: The GCN architecture is a single-step graph convolution (no per-timestep recurrence), making MultiGSL/Mix/Weighted architecturally meaningless. Only 4 static-graph GCN variants are included.

3. **DAGMA reuse**: All existing DAGMA outputs are reused without recomputation. Only SZ-Taxi contemporaneous DAGMA is genuinely missing.

4. **Result atomicity**: Each experiment result is written to a .tmp file then renamed, preventing corruption from interrupted runs.

5. **Historical preservation**: All existing result directories are untouched. New results go exclusively to `results/stage40_canonical/`.

6. **Protocol consistency**: All new experiments use the canonical protocol: batch 128, weight_decay 1e-4, feat_max from train split only, Adam optimizer, 50 epochs.

---

## 11. What's NOT Done (by design)

- **No expensive experiments were run** — only lightweight smoke tests
- **No manuscript edits** — Stage 40 is code/organization only
- **No historical artifacts deleted or modified**
- **No DAGMA recomputation** — all existing outputs reused
- **No scientific claims** based on unrun experiments
