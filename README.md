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
configs/              # YAML configs (legacy main.py / Colab)
gsl_stage26/          # multi-lag DAGMA fit + Stage 29/32 helpers
gsl_stage40/          # canonical 12-method experiment runner
gsl_stage41/          # aggregation of Stage 40 JSONs → summary CSV
gsl_stage58_sparsity/ # edge-budget sparsity sweep
results/              # experiment artifacts (see below)
paper/revised_version/ # current manuscript (sn-article.tex) + figure scripts
run_*.sh              # Linux-style runners (see Experiments)
```

Archived material (old stage briefs, previous paper drafts, forensic
logs) lives under `archive/repo_cleanup_20260913/` and is **not** required
to reproduce the paper.

---

## Datasets

| Dataset | Path | Sensors | Interval | Timesteps |
|---------|------|---------|----------|-----------|
| Los-loop | `data/los_speed.csv`, `data/los_adj.csv` | 207 | 5 min | 2016 |
| SZ-Taxi | `data/sz_speed.csv`, `data/sz_adj.csv` | 156 | 15 min | 2976 |

Chronological 80/20 split; features divided by **training-split max** only.

---

## Experiments (current)

Assume repo root as cwd. On Windows use
`C:\programs\anaconda3\envs\pth\python.exe` (or your env).

### 1. Multi-lag DAGMA graphs (once per dataset)

Fits lag-stacked DAGMA on the training split; artifacts under
`results/stage26_validation/`.

```bash
# see gsl_stage26/stage26_run_dagma.py for CLI
# (already fitted graphs are typically present in results/)
```

### 2. Canonical 12-method matrix (main paper tables)

Physical, NoSpatial, GSL, cGSL, MultiGSL family on GCN/T-GCN; both
datasets; PH1–4; seeds 42–46.

```bash
bash run_stage40_experiments.sh
# or
python gsl_stage40/scripts/stage40_run_all.py --help
```

JSON per run: `results/stage40_canonical/training/{dataset}_ph{ph}_seed{seed}_{variant}.json`.

Aggregate to five-seed mean±std:

```bash
python gsl_stage41/scripts/stage41_audit.py
```

### 3. Matched-sparsity edge-budget sweep (Los-loop PH1)

```bash
bash run_sparsity_edge_sweep.sh          # smoke
MODE=full bash run_sparsity_edge_sweep.sh
```

Outputs: `results/stage58_sparsity_sweep/full.csv`.

### 4. T-GCN vs GCN fairness audit (loss isolation)

```bash
bash run_tgcn_gcn_audit.sh
MODE=full bash run_tgcn_gcn_audit.sh
```

### 5. Physical T-GCN predictions for figures (optional)

```bash
bash run_physical_for_figures.sh
```

---

## Manuscript figures

```bash
python paper/revised_version/scripts/make_results_figures.py
python paper/revised_version/scripts/make_pred_vs_actual.py
python paper/revised_version/scripts/make_figs_258.py   # optional extras
```

Figures land in `paper/revised_version/figures/`.

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
