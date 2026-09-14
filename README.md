# Graph Structure Learning for Traffic Prediction

PyTorch implementation of **Graph Structure Learning for Traffic Prediction**.

This repository asks whether the adjacency used by GCN / T-GCN should be the
physical road-network matrix or a graph estimated from traffic data (DAGMA).
It evaluates contemporaneous and multi-lag learned graphs on **Los-loop** and
**SZ-Taxi**, with a physical baseline, an identity (no-graph) control,
five-seed training, and matched-sparsity controls.

**Main result (short):** on Los-loop, lag-specific learned graphs used in a
lag-aligned way inside T-GCN reduce RMSE by up to **43% relative to the
physical adjacency** (PH1–4). Matched 30-edge random/correlation graphs do
not recover that gain, so sparsity alone is not the explanation. On SZ-Taxi
the recovered multi-lag structure is nearly empty and improvements over the
physical graph are much smaller.

---

## Requirements

- Python ≥ 3.10
- PyTorch (CPU is enough for the published forecasting runs; GPU optional)
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Repository layout

```
data/                  # speed CSVs + physical adjacencies
models/                # GCN, T-GCN, GRU, multi-lag GSL (multigsl.py)
utils/                 # graph Laplacian, losses, logging
tasks/                 # SupervisedForecastTask (training + eval)
src/                   # experiment and figure scripts
configs/               # YAML configs (legacy main.py)
results/               # experiment artifacts used by the paper
paper/revised_version/ # manuscript, figures, and response letter
run_experiments.sh     # entry point for experiments and figures
```

Historical stage reports and older drafts live under
`archive/repo_cleanup_20260913/` and are **not** required to reproduce the
paper. See `archive/repo_cleanup_20260913/CLEANUP_REPORT.md`.

---

## Datasets

| Dataset | Path | Sensors | Interval | Timesteps |
|---------|------|---------|----------|-----------|
| Los-loop | `data/los_speed.csv`, `data/los_adj.csv` | 207 | 5 min | 2016 |
| SZ-Taxi | `data/sz_speed.csv`, `data/sz_adj.csv` | 156 | 15 min | 2976 |

Chronological 80/20 split; features divided by the **training-split max** only.

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
bash run_experiments.sh sparse       # matched-30-edge controls
bash run_experiments.sh sweep        # edge-budget sweep
bash run_experiments.sh physical     # Physical T-GCN preds + train loss
bash run_experiments.sh figures      # regenerate all paper figures
```

Equivalent Python entry points are under `src/`
(e.g. `python src/run_canonical_matrix.py --help`).

**Outputs**

| Command | Artifacts |
|---------|-----------|
| `canonical` | `results/stage40_canonical/training/*.json` |
| `aggregate` | printed five-seed tables / summary CSVs |
| `dagma` / `gsl` | DAGMA graphs under `results/stage26_validation/`, `stage33_gsl_canonical/` |
| `sparse` | matched-sparsity control results |
| `sweep` | `results/stage58_sparsity_sweep/full.csv` |
| `figures` | `paper/revised_version/figures/*.pdf` |

---

## Reproducibility

### Hardware and environment

| Component | Value |
|-----------|--------|
| GPU | NVIDIA GeForce RTX 3090 (24 GB) |
| CPU | Intel Core i3 (9th Gen) |
| System RAM | 16 GB |
| Framework | PyTorch |
| OS | Windows |
| Python | ≥ 3.10 |

Offline DAGMA structure learning is the heavier stage; CPU training of the
forecasting models is sufficient for the published runs.

### Seeds

- Forecasting training seeds: `{42, 43, 44, 45, 46}`.
- Random seeds are fixed for PyTorch and NumPy so reported numbers are
  replicable. Training still involves stochastic elements (e.g. weight
  initialization), which is why results are reported as mean ± std over five
  seeds.
- DAGMA is deterministic for a given dataset, construction, and
  hyperparameters: the graph is fitted once and shared across forecasting
  seeds.

### Hyperparameters

| Setting | Value |
|---------|--------|
| Historical window $T$ | 12 |
| Prediction horizons | 1, 2, 3, 4 |
| Optimizer | Adam |
| Learning rate | $10^{-3}$ |
| Weight decay | $10^{-4}$ |
| Batch size | 128 |
| Max epochs | 50 |
| Hidden dimension | 64 |
| Loss (GCN) | MSE |
| Loss (T-GCN) | $\ell_2$ with weight penalty $\lambda_{\mathrm{reg}}=1.5\times 10^{-3}$ |
| Structure learning | DAGMA (contemporaneous and multi-lag) |

Training protocol details are in `paper/revised_version/sn-article.tex`
§4.3 (*Training Protocol*).

### Approximate cost

- Contemporaneous DAGMA: tens of minutes per horizon with $O(10^2)$ sensors.
- Multi-lag DAGMA on Los-loop ($828$ variables): a few hours on CPU.
- Exact wall-clock times depend on the machine.

---

## Paper

- Source: `paper/revised_version/sn-article.tex`
  (plus `commands.tex`, `sn-jnl.cls`, `MyReferences.bib`)
- Figures: `paper/revised_version/figures/`
- Point-by-point response letter: `paper/revised_version/R1/Response-to-Comments.tex`

Compile from `paper/revised_version/`:

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
