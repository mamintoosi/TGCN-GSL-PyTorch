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

### Notes for the revised manuscript

These notes are referenced from the point-by-point response to reviewers
(`paper/revised_version/R1/Response-to-Comments.tex`).

1. **Numbers vs. the submitted version.** The evaluation pipeline and especially
   the DAGMA graph-learning code were substantially revised. Absolute RMSE values
   and relative percentages therefore differ in places from the originally
   submitted tables. The experimental protocol (SZ-Taxi and Los-loop, PH1–4,
   physical baseline) is unchanged, but the current artifacts under `results/`
   and the revised tables are authoritative.

2. **Removed from the submitted manuscript.**
   - *Table: Hardware specifications and hyperparameter settings* — removed
     from the revised paper; training settings are described compactly in
     Experimental Setup. The full details are preserved below for
     reproducibility.
   - *Algorithm 1 (NOTEARS)* and the detailed DAGMA algorithm box — removed.
     The Method cites the original papers and describes only the constructions
     used in this study (contemporaneous and multi-lag).
   - Dense multi-panel per-epoch convergence grids — replaced by compact
     final-metric tables and a short appendix training-loss figure.

3. **Hardware, environment, and hyperparameters**

   The hardware and software description below was in the submitted manuscript
   and is kept here so it is not lost from the revision record.

   **Hardware and environment (as used for the paper runs)**

   | Component | Value |
   |-----------|--------|
   | GPU | NVIDIA GeForce RTX 3090 (24 GB) |
   | CPU | Intel Core i3 (9th Gen) |
   | System RAM | 16 GB |
   | Framework | PyTorch |
   | OS | Windows |
   | Python | ≥ 3.10 (see `requirements.txt`) |

   Offline DAGMA structure learning is the heavier stage; CPU training of the
   forecasting models is sufficient for the published runs (see paper
   Limitations §6.5).

   **Reproducibility / seeds**

   - Forecasting training seeds: `{42, 43, 44, 45, 46}` (revised paper).
   - Random seeds are fixed for PyTorch and NumPy so reported numbers are
     replicable; training still involves stochastic elements (e.g. weight
     initialization), which is why the revised paper reports mean ± std over
     five seeds rather than a single run.
   - DAGMA is deterministic for a given dataset, construction, and
     hyperparameters: the graph is fitted once and shared across forecasting
     seeds.

   **Hyperparameters**

   | Setting | Submitted manuscript | Revised manuscript (authoritative) |
   |---------|----------------------|------------------------------------|
   | Historical window $T$ | 12 | 12 |
   | Prediction horizons | 1, 2, 3, 4 | 1, 2, 3, 4 |
   | Optimizer | Adam | Adam |
   | Learning rate | 0.001 | $10^{-3}$ |
   | Weight decay | — | $10^{-4}$ |
   | Batch size | 64 | 128 |
   | Max epochs | 50 | 50 |
   | Hidden dimension | — | 64 |
   | Loss (GCN) | MSE | MSE |
   | Loss (T-GCN) | MSE | $\ell_2$ + weight penalty $\lambda_{\mathrm{reg}}=1.5\times 10^{-3}$ |
   | Structure learning | DAGMA | DAGMA (contemporaneous and multi-lag) |

   Notes:

   - The revised experimental protocol uses **batch size 128**; the submitted
     text listed 64. Treat the revised Setup section and this table’s
     “Revised manuscript” column as the values used for the current tables.
   - Training settings for the main matrices are also described in
     `paper/revised_version/sn-article.tex` §4.3 (*Training Protocol*).

   **Rough cost profile (paper runs)**

   - Contemporaneous DAGMA: on the order of tens of minutes per horizon with
     $O(10^2)$ sensors.
   - Multi-lag DAGMA on Los-loop ($828$ variables): on the order of a few
     hours on CPU.
   - Exact wall-clock times depend on the machine.

4. **Reproducing the revised tables.** Use `run_experiments.sh` as above.
   Canonical main-table training writes under `results/stage40_canonical/`;
   sparsity sweep under `results/stage58_sparsity_sweep/`; figures under
   `paper/revised_version/figures/`.

5. **What changed relative to the submitted manuscript** (for reviewers).
   - Primary framing is learned adjacency vs the physical road-network
     graph; the identity (graph-free) model is a control baseline only.
   - Multi-lag vs contemporaneous constructions are separated; the earlier
     temporal-DAG claim is not used.
   - Five-seed reporting, matched-sparsity and capacity controls, an
     edge-budget sweep, structure figures, and a 15-minute Los-loop
     resolution control were added.
   - Absolute RMSE values differ from the submitted tables because the
     evaluation and DAGMA code were revised; the current tables are
     authoritative.

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
