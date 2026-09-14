# Repository cleanup report

**Date:** 2026-09-13  
**Commits:** `0ffedd4`, `6c10530`, and this cleanup pass.

## Goal

Keep only what is needed to (1) run the paper experiments, (2) rebuild
figures, and (3) compile the manuscript. Archive historical stage material.

## Live tree (after cleanup)

```
models/ utils/ tasks/ data/ configs/
src/                          # all experiment + figure scripts
results/                      # Stage 26/29/32/40/58 artifacts used by paper
paper/revised_version/        # sn-article.tex + figures + letter
run_experiments.sh            # single dispatcher
README.md LICENSE requirements.txt main.py
```

### `src/` (new)

| Script | Role |
|--------|------|
| `run_multilag_dagma.py` | Multi-lag DAGMA fit (Los/SZ) |
| `run_gsl_canonical.py` | Contemporaneous GSL/cGSL graphs |
| `run_canonical_matrix.py` | 12-method Stage 40 matrix |
| `aggregate_stage40_results.py` | Stage 41 five-seed summary |
| `run_sparse_controls.py` | Stage 32 RandTop30/CorrTop30 |
| `run_sparsity_sweep.py` / `analyze_sparsity_sweep.py` | Stage 58 edge budgets |
| `run_los15_resolution.py` | 15-min Los-loop variant |
| `train_physical_for_figures.py` | Physical T-GCN y_pred + train loss |
| `make_results_figures.py` / `make_pred_vs_actual.py` | Paper figures |

### Removed from root / live tree

- Google Colab notebooks (`main-*-Colab.ipynb`)
- Per-stage `gsl_stage26/40/41/58` trees (scripts copied to `src/`)
- Extra root `run_stage*.sh` runners (folded into `run_experiments.sh`)
- `doc/` stage-report corpus, Stage 42–48 briefs, `changes.bundle`
- `paper/gsl_stage44–51`, `gsl_master`, `previous_revision` (under archive)

### Archived

`archive/repo_cleanup_20260913/` — historical briefs, old runners, previous
paper drafts, forensic results (stage27/30/31/57), stage42/57 reports.

## What was **not** deleted

- `results/stage40_canonical` — main JSON results  
- `results/stage26_validation` — multi-lag DAGMA `.npy`  
- `results/stage26_checkpoint` — `y_pred` / train-loss for figures  
- `results/stage29_los15min`, `stage32_sparse_control`, `stage58_sparsity_sweep`  
- `models/`, `utils/`, `tasks/`, `data/`, `configs/`  
- `paper/revised_version` (manuscript + letter + figure PDFs)

## How to run experiments

See root `README.md` and:

```bash
bash run_experiments.sh help
bash run_experiments.sh canonical
bash run_experiments.sh sweep
bash run_experiments.sh figures
```

## Note

`paper/submitted_version/` may still exist on Windows if `sn-article.pdf`
is open in another program. After closing it:

```powershell
Move-Item paper\submitted_version archive\repo_cleanup_20260913\paper\submitted_version
```
