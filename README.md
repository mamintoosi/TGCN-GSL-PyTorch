# Graph Structure Learning for Traffic Prediction

PyTorch implementation of **Graph Structure Learning for Traffic Prediction**.

The study asks whether the adjacency used by GCN / T-GCN should be the
physical road-network matrix or a graph estimated from traffic data
(DAGMA). It evaluates contemporaneous and multi-lag learned graphs on
**Los-loop** and **SZ-Taxi**, with a physical baseline, a graph-free
identity control, five-seed training, and matched-sparsity controls.

**Main claim (short):** on Los-loop, lag-specific learned graphs used in a
lag-aligned way inside T-GCN reduce RMSE by up to **43% relative to the
physical adjacency** (PH1–4). Matched 30-edge random/correlation graphs do
not recover that gain, so sparsity alone is not the explanation.

---

## Requirements

- Python ≥ 3.10
- PyTorch (CPU is enough for small runs; GPU optional)
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Repository layout

```
data/                 # speed CSVs + physical adjacencies
models/               # GCN, T-GCN, GRU, multi-lag GSL (multigsl.py)
utils/                # graph Laplacian, losses, logging
tasks/                # SupervisedForecastTask (training + eval)
src/                  # all experiment and figure scripts
configs/              # YAML configs (legacy main.py)
results/              # experiment artifacts used by the paper
paper/revised_version/ # manuscript (sn-article.tex), figures, response letter
run_experiments.sh    # single entry point for experiments and figures
```

Historical stage reports, old runners, and previous paper drafts live under
`archive/repo_cleanup_20260913/` and are **not** required to reproduce the
paper. See `archive/repo_cleanup_20260913/CLEANUP_REPORT.md`.

---

## Datasets

| Dataset | Path | Sensors | Interval | Timesteps |
|---------|------|---------|----------|-----------|
| Los-loop | `data/los_speed.csv`, `data/los_adj.csv` | 207 | 5 min | 2016 |
| SZ-Taxi | `data/sz_speed.csv`, `data/sz_adj.csv` | 156 | 15 min | 2976 |

Chronological 80/20 split; features divided by **training-split max** only.

---

## Experiments

All runners go through **`run_experiments.sh`** (set `PYTHON` if needed).

```bash
bash run_experiments.sh help
bash run_experiments.sh canonical    # 12-method matrix (main tables)
bash run_experiments.sh aggregate    # five-seed mean±std from JSONs
bash run_experiments.sh dagma        # multi-lag DAGMA (long)
bash run_experiments.sh gsl          # contemporaneous GSL/cGSL graphs
bash run_experiments.sh los15        # 15-min Los-loop variant
bash run_experiments.sh sparse       # Stage 32 matched-30-edge controls
bash run_experiments.sh sweep        # Stage 58 edge-budget sweep
bash run_experiments.sh physical     # Physical T-GCN preds + train loss
bash run_experiments.sh figures      # regenerate all paper figures
```

Equivalent Python entry points are under `src/`
(e.g. `python src/run_canonical_matrix.py --help`).

**Outputs**

| Command | Artifacts |
|---------|-----------|
| `canonical` | `results/stage40_canonical/training/*.json` |
| `aggregate` | `gsl_stage41` summary (if present) / printed tables |
| `dagma` / `gsl` | DAGMA graphs under `results/stage26_validation/`, `stage33_gsl_canonical/` |
| `sweep` | `results/stage58_sparsity_sweep/full.csv` |
| `figures` | `paper/revised_version/figures/*.pdf` |

---

## Manuscript figures

```bash
bash run_experiments.sh figures
```

---

## Paper

- Source: `paper/revised_version/sn-article.tex` (+ `commands.tex`, `sn-jnl.cls`, `MyReferences.bib`)
- Compile from `paper/revised_version/`:

```bash
pdflatex sn-article.tex
bibtex sn-article
pdflatex sn-article.tex
pdflatex sn-article.tex
```

---

## Citation

```bibtex
@article{amintoosi2026graph,
  title={Graph Structure Learning for Traffic Prediction},
  author={Amintoosi, Mahmood},
  journal={Submitted},
  year={2026}
}
```

Preliminary conference version:

```bibtex
@inproceedings{amintoosi1403aimc55,
  title={Urban traffic prediction with learning based graph convolutional networks},
  author={Amintoosi, Mahmood},
  booktitle={Proceedings of the 55th Annual Iranian Mathematics Conference},
  year={2024}
}
```

---

## License

MIT — see `LICENSE`.
