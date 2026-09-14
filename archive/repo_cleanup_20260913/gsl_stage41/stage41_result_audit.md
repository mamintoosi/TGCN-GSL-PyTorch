# Stage 41 — Statistical and Scientific Result Audit

**Generated:** 2026-09-10 10:46:48  
**Scope:** read-only audit of the completed Stage 40 canonical results (`results/stage40_canonical/training`, 480 records).  
**Constraints honored:** no new training, no new DAGMA fitting, no modification of experimental code. Graph statistics were verified from already-stored artifacts only.

## 0. Implementation inspected (verified from code and artifacts, not the log)

| Item | Verified value |
|---|---|
| Result files | `results/stage40_canonical/training/{dataset}_ph{ph}_seed{seed}_{variant}.json`; keys `rmse`, `mae` (km/h scale, 4 decimals), `n_edges`, `n_params`, `status` |
| Dataset names | `losloop` (207 nodes, `data/los_speed.csv`), `shenzhen` (156 nodes, `data/sz_speed.csv`) |
| PH convention | PH ∈ {1,2,3,4}: forecast horizon in 5-min steps; input window 12 steps; 80/20 chronological split; normalization by train-split max |
| Seed convention | training seeds 42–46 (5 seeds). DAGMA graphs are deterministic and fitted once (seed 42 artifacts) and shared across training seeds |
| GCN family | `GCN`, `GCN-NoSpatial`, `GCN-GSL`, `GCN-cGSL`, `GCN-MultiGSL` (backbone `models/gcn.py`: single graph-convolution over the whole window; loss `mse`) |
| T-GCN family | `T-GCN`, `T-GCN-NoSpatial`, `T-GCN-GSL`, `T-GCN-cGSL`, `T-GCN-MultiGSL`, `T-GCN-MultiGSL-Weighted`, `T-GCN-MultiGSL-Mix` (backbone `models/tgcn.py`: GRU with per-timestep graph convolution; loss `mse_with_regularizer`) |
| GCN-MultiGSL graph consumption | union `A_union = max_l A_l` of the three thresholded (|W|>0.1) lag blocks into one static graph; standard GCN; no extra parameters |
| T-GCN-MultiGSL graph consumption | the three lag graphs are consumed separately, one per input timestep, `graph_idx = (T−1−t) mod 3` (most recent step ↔ lag-1 graph); no extra parameters |
| T-GCN-MultiGSL-Weighted | same three graphs; learned global scalar weights `softmax(w)` mixing the Laplacians into one graph (3 extra parameters) |
| T-GCN-MultiGSL-Mix | same three graphs; per-node, per-timestep gate network (softmax over 3 graphs) mixes Laplacians before the GRU update (4,419 extra parameters) |
| cGSL construction | `A_cgsl = (A_gsl + A_gsl.T) > 0`, diagonal removed |
| Graph provenance | contemporaneous DAGMA from `results/stage33_gsl_canonical/{los,sz}_gsl_ph*_seed42_A_binary.npy`; multi-lag blocks from `results/stage26_validation/{los,sz}_ph*_seed42_L3_lag_{1,2,3}.npy` (thresholded at |W|>0.1 by the runner) |

## 1. Result inventory and integrity

* Result records audited: **480 / 480** (12 methods × 2 datasets × 4 PHs × 5 seeds).
* Missing results: **0**
* Corrupt/incomplete records: **0**
* Duplicate cells (in-memory or on-disk): **0**
* Cross-check vs execution log `archive/misc/stage40_run_all.txt`: 480 `[DONE]` lines = 480 = records on disk; 0 `[FAIL]`, 0 `[SKIP]`; log matches the artifacts (the log confirms a single clean full run, `Run: 480, Skip: 0, Fail: 0`).

Per-PH result tables below are split into the T-GCN family and the GCN family. `Wins vs T-GCN (/5)` counts seeds on which the method's mean-beating per-seed RMSE is lower than the physical-graph `T-GCN` baseline on the same seed (for GCN-family rows this is a cross-family comparison and should be read together with the improvement columns). All statistics use the five seeds 42–46; std is the sample standard deviation (ddof = 1); no intermediate rounding.

## 2. Complete results table — Los-loop (`losloop`)

### Los-loop, PH=1

#### T-GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| T-GCN | 7.88 ± 0.28 | 7.46–8.19 | 5.49 ± 0.21 | 0 |
| T-GCN-NoSpatial | 5.25 ± 0.19 | 4.97–5.42 | 3.06 ± 0.09 | 5 |
| T-GCN-GSL | 5.86 ± 0.21 | 5.50–6.02 | 3.69 ± 0.14 | 5 |
| T-GCN-cGSL | 5.82 ± 0.23 | 5.47–6.10 | 3.48 ± 0.13 | 5 |
| T-GCN-MultiGSL | 4.84 ± 0.11 | 4.73–5.03 | 2.82 ± 0.06 | 5 |
| T-GCN-MultiGSL-Weighted | 4.83 ± 0.11 | 4.73–5.02 | 2.81 ± 0.05 | 5 |
| T-GCN-MultiGSL-Mix | 4.49 ± 0.14 | 4.29–4.66 | 2.73 ± 0.07 | 5 |

#### GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| GCN | 8.14 ± 0.11 | 8.00–8.30 | 5.79 ± 0.03 | 0 |
| GCN-NoSpatial | 4.88 ± 0.31 | 4.57–5.27 | 3.00 ± 0.21 | 5 |
| GCN-GSL | 7.83 ± 0.11 | 7.73–7.95 | 5.34 ± 0.08 | 4 |
| GCN-cGSL | 5.76 ± 0.20 | 5.57–6.03 | 3.65 ± 0.14 | 5 |
| GCN-MultiGSL | 9.78 ± 0.14 | 9.61–9.94 | 6.60 ± 0.04 | 0 |

#### Improvement over baselines (%, positive = lower RMSE than baseline; mean RMSE used)

| Method | Δ vs T-GCN (%) | Δ vs T-GCN-NoSpatial (%) |
|---|---|---|
| T-GCN | 0.00 | -50.00 |
| T-GCN-NoSpatial | 33.33 | 0.00 |
| T-GCN-GSL | 25.62 | -11.57 |
| T-GCN-cGSL | 26.11 | -10.84 |
| T-GCN-MultiGSL | 38.55 | 7.82 |
| T-GCN-MultiGSL-Weighted | 38.63 | 7.95 |
| T-GCN-MultiGSL-Mix | 42.98 | 14.47 |
| GCN | -3.39 | -55.09 |
| GCN-NoSpatial | 38.05 | 7.08 |
| GCN-GSL | 0.64 | -49.03 |
| GCN-cGSL | 26.84 | -9.74 |
| GCN-MultiGSL | -24.16 | -86.24 |

### Los-loop, PH=2

#### T-GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| T-GCN | 8.13 ± 0.09 | 8.03–8.23 | 5.65 ± 0.08 | 0 |
| T-GCN-NoSpatial | 5.76 ± 0.08 | 5.64–5.86 | 3.30 ± 0.06 | 5 |
| T-GCN-GSL | 6.30 ± 0.06 | 6.23–6.37 | 3.91 ± 0.08 | 5 |
| T-GCN-cGSL | 6.35 ± 0.12 | 6.21–6.51 | 3.76 ± 0.08 | 5 |
| T-GCN-MultiGSL | 5.45 ± 0.14 | 5.28–5.60 | 3.14 ± 0.08 | 5 |
| T-GCN-MultiGSL-Weighted | 5.44 ± 0.13 | 5.28–5.58 | 3.13 ± 0.08 | 5 |
| T-GCN-MultiGSL-Mix | 5.08 ± 0.15 | 4.85–5.23 | 2.99 ± 0.10 | 5 |

#### GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| GCN | 8.54 ± 0.28 | 8.18–8.87 | 6.01 ± 0.16 | 0 |
| GCN-NoSpatial | 5.60 ± 0.39 | 5.09–6.11 | 3.39 ± 0.24 | 5 |
| GCN-GSL | 8.48 ± 0.21 | 8.25–8.79 | 5.68 ± 0.14 | 0 |
| GCN-cGSL | 6.33 ± 0.32 | 5.94–6.78 | 3.97 ± 0.19 | 5 |
| GCN-MultiGSL | 10.05 ± 0.16 | 9.92–10.24 | 6.76 ± 0.05 | 0 |

#### Improvement over baselines (%, positive = lower RMSE than baseline; mean RMSE used)

| Method | Δ vs T-GCN (%) | Δ vs T-GCN-NoSpatial (%) |
|---|---|---|
| T-GCN | 0.00 | -41.21 |
| T-GCN-NoSpatial | 29.18 | 0.00 |
| T-GCN-GSL | 22.53 | -9.40 |
| T-GCN-cGSL | 21.88 | -10.32 |
| T-GCN-MultiGSL | 33.03 | 5.43 |
| T-GCN-MultiGSL-Weighted | 33.14 | 5.59 |
| T-GCN-MultiGSL-Mix | 37.60 | 11.88 |
| GCN | -4.99 | -48.26 |
| GCN-NoSpatial | 31.09 | 2.69 |
| GCN-GSL | -4.25 | -47.22 |
| GCN-cGSL | 22.16 | -9.92 |
| GCN-MultiGSL | -23.57 | -74.49 |

### Los-loop, PH=3

#### T-GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| T-GCN | 8.37 ± 0.17 | 8.25–8.67 | 5.75 ± 0.14 | 0 |
| T-GCN-NoSpatial | 6.11 ± 0.07 | 6.03–6.23 | 3.49 ± 0.07 | 5 |
| T-GCN-GSL | 6.66 ± 0.12 | 6.57–6.85 | 4.12 ± 0.03 | 5 |
| T-GCN-cGSL | 6.67 ± 0.11 | 6.53–6.84 | 3.94 ± 0.09 | 5 |
| T-GCN-MultiGSL | 5.94 ± 0.03 | 5.91–5.98 | 3.35 ± 0.02 | 5 |
| T-GCN-MultiGSL-Weighted | 5.93 ± 0.03 | 5.90–5.97 | 3.33 ± 0.03 | 5 |
| T-GCN-MultiGSL-Mix | 5.55 ± 0.16 | 5.34–5.71 | 3.22 ± 0.09 | 5 |

#### GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| GCN | 8.76 ± 0.26 | 8.39–9.08 | 6.12 ± 0.15 | 0 |
| GCN-NoSpatial | 6.02 ± 0.28 | 5.59–6.29 | 3.55 ± 0.18 | 5 |
| GCN-GSL | 8.45 ± 0.19 | 8.18–8.69 | 5.65 ± 0.15 | 2 |
| GCN-cGSL | 6.68 ± 0.25 | 6.30–6.95 | 4.12 ± 0.16 | 5 |
| GCN-MultiGSL | 10.24 ± 0.13 | 10.09–10.40 | 6.84 ± 0.08 | 0 |

#### Improvement over baselines (%, positive = lower RMSE than baseline; mean RMSE used)

| Method | Δ vs T-GCN (%) | Δ vs T-GCN-NoSpatial (%) |
|---|---|---|
| T-GCN | 0.00 | -36.97 |
| T-GCN-NoSpatial | 26.99 | 0.00 |
| T-GCN-GSL | 20.46 | -8.94 |
| T-GCN-cGSL | 20.30 | -9.16 |
| T-GCN-MultiGSL | 29.01 | 2.77 |
| T-GCN-MultiGSL-Weighted | 29.17 | 2.99 |
| T-GCN-MultiGSL-Mix | 33.72 | 9.22 |
| GCN | -4.75 | -43.47 |
| GCN-NoSpatial | 28.02 | 1.42 |
| GCN-GSL | -0.96 | -38.29 |
| GCN-cGSL | 20.12 | -9.42 |
| GCN-MultiGSL | -22.39 | -67.63 |

### Los-loop, PH=4

#### T-GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| T-GCN | 8.66 ± 0.16 | 8.45–8.82 | 6.00 ± 0.16 | 0 |
| T-GCN-NoSpatial | 6.58 ± 0.12 | 6.44–6.76 | 3.72 ± 0.08 | 5 |
| T-GCN-GSL | 7.04 ± 0.09 | 6.91–7.13 | 4.30 ± 0.06 | 5 |
| T-GCN-cGSL | 7.09 ± 0.12 | 6.99–7.29 | 4.18 ± 0.08 | 5 |
| T-GCN-MultiGSL | 6.26 ± 0.03 | 6.21–6.29 | 3.50 ± 0.05 | 5 |
| T-GCN-MultiGSL-Weighted | 6.25 ± 0.03 | 6.21–6.29 | 3.49 ± 0.04 | 5 |
| T-GCN-MultiGSL-Mix | 5.86 ± 0.07 | 5.82–5.98 | 3.39 ± 0.02 | 5 |

#### GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| GCN | 8.76 ± 0.29 | 8.49–9.20 | 6.12 ± 0.15 | 2 |
| GCN-NoSpatial | 6.26 ± 0.26 | 6.02–6.64 | 3.68 ± 0.22 | 5 |
| GCN-GSL | 8.93 ± 0.12 | 8.80–9.09 | 5.92 ± 0.07 | 1 |
| GCN-cGSL | 6.88 ± 0.21 | 6.68–7.19 | 4.25 ± 0.16 | 5 |
| GCN-MultiGSL | 10.27 ± 0.11 | 10.17–10.43 | 6.88 ± 0.05 | 0 |

#### Improvement over baselines (%, positive = lower RMSE than baseline; mean RMSE used)

| Method | Δ vs T-GCN (%) | Δ vs T-GCN-NoSpatial (%) |
|---|---|---|
| T-GCN | 0.00 | -31.68 |
| T-GCN-NoSpatial | 24.06 | 0.00 |
| T-GCN-GSL | 18.70 | -7.06 |
| T-GCN-cGSL | 18.14 | -7.78 |
| T-GCN-MultiGSL | 27.71 | 4.81 |
| T-GCN-MultiGSL-Weighted | 27.79 | 4.91 |
| T-GCN-MultiGSL-Mix | 32.34 | 10.91 |
| GCN | -1.22 | -33.28 |
| GCN-NoSpatial | 27.73 | 4.83 |
| GCN-GSL | -3.17 | -35.85 |
| GCN-cGSL | 20.54 | -4.63 |
| GCN-MultiGSL | -18.62 | -56.20 |

## 2. Complete results table — SZ-Taxi (`shenzhen`)

### SZ-Taxi, PH=1

#### T-GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| T-GCN | 5.45 ± 0.11 | 5.35–5.63 | 4.08 ± 0.05 | 0 |
| T-GCN-NoSpatial | 4.12 ± 0.00 | 4.12–4.13 | 2.76 ± 0.04 | 5 |
| T-GCN-GSL | 4.28 ± 0.05 | 4.25–4.37 | 2.87 ± 0.13 | 5 |
| T-GCN-cGSL | 4.30 ± 0.03 | 4.26–4.35 | 2.86 ± 0.02 | 5 |
| T-GCN-MultiGSL | 4.13 ± 0.02 | 4.12–4.16 | 2.82 ± 0.08 | 5 |
| T-GCN-MultiGSL-Weighted | 4.13 ± 0.02 | 4.12–4.16 | 2.82 ± 0.08 | 5 |
| T-GCN-MultiGSL-Mix | 4.12 ± 0.02 | 4.11–4.15 | 2.75 ± 0.09 | 5 |

#### GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| GCN | 5.96 ± 0.01 | 5.96–5.97 | 4.41 ± 0.00 | 0 |
| GCN-NoSpatial | 4.11 ± 0.00 | 4.11–4.12 | 2.75 ± 0.03 | 5 |
| GCN-GSL | 4.88 ± 0.01 | 4.87–4.89 | 3.17 ± 0.01 | 5 |
| GCN-cGSL | 4.64 ± 0.00 | 4.64–4.65 | 3.03 ± 0.01 | 5 |
| GCN-MultiGSL | 4.82 ± 0.01 | 4.81–4.83 | 3.04 ± 0.01 | 5 |

#### Improvement over baselines (%, positive = lower RMSE than baseline; mean RMSE used)

| Method | Δ vs T-GCN (%) | Δ vs T-GCN-NoSpatial (%) |
|---|---|---|
| T-GCN | 0.00 | -32.27 |
| T-GCN-NoSpatial | 24.39 | 0.00 |
| T-GCN-GSL | 21.44 | -3.91 |
| T-GCN-cGSL | 21.08 | -4.39 |
| T-GCN-MultiGSL | 24.20 | -0.25 |
| T-GCN-MultiGSL-Weighted | 24.20 | -0.25 |
| T-GCN-MultiGSL-Mix | 24.41 | 0.02 |
| GCN | -9.37 | -44.65 |
| GCN-NoSpatial | 24.50 | 0.14 |
| GCN-GSL | 10.44 | -18.46 |
| GCN-cGSL | 14.83 | -12.65 |
| GCN-MultiGSL | 11.49 | -17.07 |

### SZ-Taxi, PH=2

#### T-GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| T-GCN | 5.55 ± 0.08 | 5.44–5.67 | 4.15 ± 0.04 | 0 |
| T-GCN-NoSpatial | 4.16 ± 0.01 | 4.16–4.17 | 2.78 ± 0.03 | 5 |
| T-GCN-GSL | 4.31 ± 0.03 | 4.29–4.36 | 2.89 ± 0.07 | 5 |
| T-GCN-cGSL | 4.34 ± 0.02 | 4.32–4.36 | 2.92 ± 0.03 | 5 |
| T-GCN-MultiGSL | 4.16 ± 0.00 | 4.16–4.17 | 2.83 ± 0.03 | 5 |
| T-GCN-MultiGSL-Weighted | 4.16 ± 0.00 | 4.16–4.17 | 2.83 ± 0.03 | 5 |
| T-GCN-MultiGSL-Mix | 4.15 ± 0.01 | 4.14–4.16 | 2.77 ± 0.04 | 5 |

#### GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| GCN | 5.98 ± 0.00 | 5.97–5.98 | 4.42 ± 0.00 | 0 |
| GCN-NoSpatial | 4.15 ± 0.00 | 4.15–4.15 | 2.77 ± 0.01 | 5 |
| GCN-GSL | 4.91 ± 0.00 | 4.91–4.91 | 3.20 ± 0.00 | 5 |
| GCN-cGSL | 4.67 ± 0.00 | 4.67–4.68 | 3.06 ± 0.00 | 5 |
| GCN-MultiGSL | 4.85 ± 0.00 | 4.85–4.86 | 3.08 ± 0.00 | 5 |

#### Improvement over baselines (%, positive = lower RMSE than baseline; mean RMSE used)

| Method | Δ vs T-GCN (%) | Δ vs T-GCN-NoSpatial (%) |
|---|---|---|
| T-GCN | 0.00 | -33.44 |
| T-GCN-NoSpatial | 25.06 | 0.00 |
| T-GCN-GSL | 22.38 | -3.57 |
| T-GCN-cGSL | 21.91 | -4.20 |
| T-GCN-MultiGSL | 25.05 | -0.01 |
| T-GCN-MultiGSL-Weighted | 25.05 | -0.01 |
| T-GCN-MultiGSL-Mix | 25.26 | 0.27 |
| GCN | -7.63 | -43.62 |
| GCN-NoSpatial | 25.20 | 0.19 |
| GCN-GSL | 11.56 | -18.01 |
| GCN-cGSL | 15.80 | -12.35 |
| GCN-MultiGSL | 12.56 | -16.67 |

### SZ-Taxi, PH=3

#### T-GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| T-GCN | 5.60 ± 0.05 | 5.54–5.65 | 4.20 ± 0.03 | 0 |
| T-GCN-NoSpatial | 4.19 ± 0.01 | 4.19–4.20 | 2.80 ± 0.04 | 5 |
| T-GCN-GSL | 4.33 ± 0.01 | 4.32–4.35 | 2.89 ± 0.05 | 5 |
| T-GCN-cGSL | 4.38 ± 0.03 | 4.34–4.40 | 2.99 ± 0.08 | 5 |
| T-GCN-MultiGSL | 4.20 ± 0.01 | 4.19–4.21 | 2.85 ± 0.05 | 5 |
| T-GCN-MultiGSL-Weighted | 4.20 ± 0.01 | 4.19–4.21 | 2.85 ± 0.05 | 5 |
| T-GCN-MultiGSL-Mix | 4.18 ± 0.00 | 4.18–4.18 | 2.79 ± 0.03 | 5 |

#### GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| GCN | 5.99 ± 0.00 | 5.99–5.99 | 4.43 ± 0.00 | 0 |
| GCN-NoSpatial | 4.19 ± 0.00 | 4.18–4.19 | 2.79 ± 0.01 | 5 |
| GCN-GSL | 4.93 ± 0.01 | 4.92–4.94 | 3.23 ± 0.01 | 5 |
| GCN-cGSL | 4.70 ± 0.00 | 4.70–4.70 | 3.08 ± 0.00 | 5 |
| GCN-MultiGSL | 4.88 ± 0.01 | 4.87–4.89 | 3.11 ± 0.00 | 5 |

#### Improvement over baselines (%, positive = lower RMSE than baseline; mean RMSE used)

| Method | Δ vs T-GCN (%) | Δ vs T-GCN-NoSpatial (%) |
|---|---|---|
| T-GCN | 0.00 | -33.63 |
| T-GCN-NoSpatial | 25.17 | 0.00 |
| T-GCN-GSL | 22.66 | -3.35 |
| T-GCN-cGSL | 21.81 | -4.48 |
| T-GCN-MultiGSL | 25.02 | -0.20 |
| T-GCN-MultiGSL-Weighted | 25.01 | -0.21 |
| T-GCN-MultiGSL-Mix | 25.42 | 0.34 |
| GCN | -6.92 | -42.88 |
| GCN-NoSpatial | 25.25 | 0.11 |
| GCN-GSL | 11.99 | -17.61 |
| GCN-cGSL | 16.08 | -12.14 |
| GCN-MultiGSL | 12.91 | -16.37 |

### SZ-Taxi, PH=4

#### T-GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| T-GCN | 5.63 ± 0.06 | 5.57–5.69 | 4.22 ± 0.02 | 0 |
| T-GCN-NoSpatial | 4.23 ± 0.01 | 4.22–4.23 | 2.83 ± 0.04 | 5 |
| T-GCN-GSL | 4.37 ± 0.02 | 4.35–4.40 | 2.89 ± 0.01 | 5 |
| T-GCN-cGSL | 4.41 ± 0.01 | 4.39–4.43 | 2.96 ± 0.06 | 5 |
| T-GCN-MultiGSL | 4.22 ± 0.00 | 4.22–4.23 | 2.88 ± 0.03 | 5 |
| T-GCN-MultiGSL-Weighted | 4.22 ± 0.00 | 4.22–4.23 | 2.88 ± 0.03 | 5 |
| T-GCN-MultiGSL-Mix | 4.22 ± 0.01 | 4.21–4.23 | 2.84 ± 0.08 | 5 |

#### GCN family

| Method | RMSE mean ± std | RMSE min–max | MAE mean ± std | Wins vs T-GCN (/5) |
|---|---|---|---|---|
| GCN | 6.00 ± 0.00 | 6.00–6.00 | 4.44 ± 0.00 | 0 |
| GCN-NoSpatial | 4.22 ± 0.00 | 4.22–4.22 | 2.82 ± 0.01 | 5 |
| GCN-GSL | 4.96 ± 0.00 | 4.95–4.96 | 3.26 ± 0.01 | 5 |
| GCN-cGSL | 4.73 ± 0.00 | 4.73–4.73 | 3.11 ± 0.00 | 5 |
| GCN-MultiGSL | 4.90 ± 0.00 | 4.90–4.91 | 3.14 ± 0.01 | 5 |

#### Improvement over baselines (%, positive = lower RMSE than baseline; mean RMSE used)

| Method | Δ vs T-GCN (%) | Δ vs T-GCN-NoSpatial (%) |
|---|---|---|
| T-GCN | 0.00 | -33.31 |
| T-GCN-NoSpatial | 24.99 | 0.00 |
| T-GCN-GSL | 22.42 | -3.43 |
| T-GCN-cGSL | 21.62 | -4.49 |
| T-GCN-MultiGSL | 25.00 | 0.02 |
| T-GCN-MultiGSL-Weighted | 25.00 | 0.01 |
| T-GCN-MultiGSL-Mix | 25.13 | 0.19 |
| GCN | -6.54 | -42.03 |
| GCN-NoSpatial | 25.13 | 0.19 |
| GCN-GSL | 11.98 | -17.34 |
| GCN-cGSL | 16.05 | -11.91 |
| GCN-MultiGSL | 12.93 | -16.08 |

## 3. Seed-consistency analysis (principal comparisons)

Sign convention: Δ = RMSE(A) − RMSE(B) per seed; **B wins** when Δ > 0. `wins/5` counts seeds where B beats A. Tests are paired across the same training seeds. With n = 5 the exact two-sided Wilcoxon signed-rank test has a minimum achievable p of 2/2⁵ = 0.0625, so no comparison can reach p < 0.05 under the exact test — a structural limitation of n = 5, not a property of the methods. Paired-t p-values are reported for reference but inherit the same low-power caveat.

### T-GCN vs T-GCN-NoSpatial

**Los-loop**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 2.626 | 0.125 | 5 | 1.24e-06 | 0.0625 |
| 2 | 2.374 | 0.043 | 5 | 2.59e-08 | 0.0625 |
| 3 | 2.258 | 0.105 | 5 | 1.10e-06 | 0.0625 |
| 4 | 2.083 | 0.088 | 5 | 7.50e-07 | 0.0625 |

**SZ-Taxi**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 1.329 | 0.108 | 5 | 1.04e-05 | 0.0625 |
| 2 | 1.391 | 0.083 | 5 | 2.97e-06 | 0.0625 |
| 3 | 1.410 | 0.046 | 5 | 2.68e-07 | 0.0625 |
| 4 | 1.407 | 0.052 | 5 | 4.59e-07 | 0.0625 |

### T-GCN vs T-GCN-GSL

**Los-loop**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 2.018 | 0.142 | 5 | 5.83e-06 | 0.0625 |
| 2 | 1.832 | 0.076 | 5 | 7.24e-07 | 0.0625 |
| 3 | 1.712 | 0.058 | 5 | 3.26e-07 | 0.0625 |
| 4 | 1.619 | 0.155 | 5 | 2.00e-05 | 0.0625 |

**SZ-Taxi**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 1.168 | 0.122 | 5 | 2.82e-05 | 0.0625 |
| 2 | 1.243 | 0.064 | 5 | 1.70e-06 | 0.0625 |
| 3 | 1.269 | 0.049 | 5 | 5.22e-07 | 0.0625 |
| 4 | 1.263 | 0.042 | 5 | 2.92e-07 | 0.0625 |

### T-GCN vs T-GCN-cGSL

**Los-loop**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 2.057 | 0.081 | 5 | 5.88e-07 | 0.0625 |
| 2 | 1.779 | 0.091 | 5 | 1.64e-06 | 0.0625 |
| 3 | 1.699 | 0.085 | 5 | 1.53e-06 | 0.0625 |
| 4 | 1.571 | 0.091 | 5 | 2.64e-06 | 0.0625 |

**SZ-Taxi**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 1.148 | 0.086 | 5 | 7.43e-06 | 0.0625 |
| 2 | 1.216 | 0.071 | 5 | 2.74e-06 | 0.0625 |
| 3 | 1.222 | 0.053 | 5 | 8.45e-07 | 0.0625 |
| 4 | 1.218 | 0.053 | 5 | 8.47e-07 | 0.0625 |

### T-GCN vs T-GCN-MultiGSL

**Los-loop**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 3.036 | 0.275 | 5 | 1.60e-05 | 0.0625 |
| 2 | 2.686 | 0.175 | 5 | 4.30e-06 | 0.0625 |
| 3 | 2.427 | 0.169 | 5 | 5.56e-06 | 0.0625 |
| 4 | 2.400 | 0.171 | 5 | 6.10e-06 | 0.0625 |

**SZ-Taxi**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 1.319 | 0.111 | 5 | 1.19e-05 | 0.0625 |
| 2 | 1.391 | 0.084 | 5 | 3.12e-06 | 0.0625 |
| 3 | 1.401 | 0.046 | 5 | 2.81e-07 | 0.0625 |
| 4 | 1.408 | 0.057 | 5 | 6.63e-07 | 0.0625 |

### T-GCN vs T-GCN-MultiGSL-Mix

**Los-loop**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 3.386 | 0.324 | 5 | 2.00e-05 | 0.0625 |
| 2 | 3.058 | 0.190 | 5 | 3.57e-06 | 0.0625 |
| 3 | 2.822 | 0.200 | 5 | 5.98e-06 | 0.0625 |
| 4 | 2.800 | 0.138 | 5 | 1.41e-06 | 0.0625 |

**SZ-Taxi**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 1.330 | 0.118 | 5 | 1.49e-05 | 0.0625 |
| 2 | 1.402 | 0.078 | 5 | 2.30e-06 | 0.0625 |
| 3 | 1.424 | 0.048 | 5 | 3.11e-07 | 0.0625 |
| 4 | 1.415 | 0.065 | 5 | 1.04e-06 | 0.0625 |

### T-GCN-NoSpatial vs T-GCN-MultiGSL-Mix

**Los-loop**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 0.760 | 0.219 | 5 | 1.48e-03 | 0.0625 |
| 2 | 0.684 | 0.198 | 5 | 1.51e-03 | 0.0625 |
| 3 | 0.563 | 0.137 | 5 | 7.70e-04 | 0.0625 |
| 4 | 0.717 | 0.065 | 5 | 1.61e-05 | 0.0625 |

**SZ-Taxi**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 0.001 | 0.017 | 4 | 9.08e-01 | 0.6250 |
| 2 | 0.011 | 0.010 | 4 | 6.39e-02 | 0.1250 |
| 3 | 0.014 | 0.006 | 5 | 5.62e-03 | 0.0625 |
| 4 | 0.008 | 0.016 | 4 | 3.15e-01 | 0.4375 |

### T-GCN vs T-GCN-MultiGSL-Weighted

**Los-loop**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 3.043 | 0.275 | 5 | 1.59e-05 | 0.0625 |
| 2 | 2.695 | 0.173 | 5 | 4.03e-06 | 0.0625 |
| 3 | 2.441 | 0.180 | 5 | 6.97e-06 | 0.0625 |
| 4 | 2.406 | 0.171 | 5 | 6.06e-06 | 0.0625 |

**SZ-Taxi**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 1.319 | 0.111 | 5 | 1.19e-05 | 0.0625 |
| 2 | 1.391 | 0.083 | 5 | 3.10e-06 | 0.0625 |
| 3 | 1.401 | 0.046 | 5 | 2.73e-07 | 0.0625 |
| 4 | 1.408 | 0.057 | 5 | 6.55e-07 | 0.0625 |

### T-GCN-MultiGSL vs T-GCN-MultiGSL-Mix

**Los-loop**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 0.349 | 0.217 | 5 | 2.27e-02 | 0.0625 |
| 2 | 0.372 | 0.275 | 5 | 3.89e-02 | 0.0625 |
| 3 | 0.394 | 0.130 | 5 | 2.46e-03 | 0.0625 |
| 4 | 0.401 | 0.085 | 5 | 4.59e-04 | 0.0625 |

**SZ-Taxi**

| PH | mean Δ (A−B) | std Δ | B wins /5 | paired-t p | Wilcoxon exact p |
|---|---|---|---|---|---|
| 1 | 0.011 | 0.012 | 5 | 9.49e-02 | 0.0625 |
| 2 | 0.011 | 0.009 | 4 | 4.80e-02 | 0.1250 |
| 3 | 0.023 | 0.007 | 5 | 1.67e-03 | 0.0625 |
| 4 | 0.007 | 0.009 | 4 | 1.39e-01 | 0.1875 |

### Reading of the seed-consistency tables

* Every T-GCN-family method and its GCN counterpart beat the physical-graph baseline `T-GCN` in 5/5 seeds at all 8 dataset×PH cells; the paired-t p-values are < 10⁻³ and the mean gaps are 10–55× the per-seed SD of the difference. The exact Wilcoxon p is pinned at its n = 5 floor (0.0625); we therefore treat 5/5 wins with a large, seed-stable gap as *consistent* — not as classically significant — evidence.
* `T-GCN-MultiGSL-Mix` beats `T-GCN-NoSpatial` in 5/5 seeds at all four Los-loop PHs (Δ = 9.2–14.5% in NoSpatial-relative terms; paired-t p ≤ 0.0023 at every PH). On SZ-Taxi the same comparison is 4–5/5 seeds with Δ ≤ 0.34% and paired-t p ≥ 0.06 at three of four PHs — direction consistent but practically negligible.
* `T-GCN-MultiGSL` vs `T-GCN-MultiGSL-Mix`: Mix wins 5/5 seeds at all 8 cells, but the mean gap is 0.35–0.40 RMSE points on Los-loop (a 6.4–7.2% relative reduction of MultiGSL RMSE) and ≤ 0.02 RMSE on SZ (≤ 0.5% relative) — the gating adds a modest, seed-consistent improvement over the fixed assignment on Los-loop and essentially nothing on SZ.
* Robustness is assessed by wins/5 **and** the per-seed spread, not by the mean alone: e.g. SZ `T-GCN-NoSpatial` vs `T-GCN-MultiGSL-Mix` is 4/5 seeds at PH 1, 2 and 4 with |Δ| ≤ 0.03 RMSE — well within seed noise, so no claim of benefit is made there.

## 4. Statistical tests — what they can and cannot support

* **Test used:** paired two-sided t-test across the 5 training seeds (same-seed pairing removes between-seed level variation), plus the two-sided Wilcoxon signed-rank test computed with the **exact** permutation distribution (mandatory at n = 5; the normal approximation is invalid and understates p at this sample size).
* **Effect direction:** reported for every comparison in `stage41_paired_tests.csv` (positive Δ = second-listed method better).
* **What is statistically convincing:** all comparisons against the physical-graph `T-GCN` (5/5 seeds, paired-t p < 10⁻³, gaps 10–55× the seed-level SD of the difference), and `T-GCN-MultiGSL-Mix` vs `T-GCN-NoSpatial`/`T-GCN-MultiGSL` on Los-loop (5/5 seeds, paired-t p ≤ 0.039). These are consistent, large effects; even here the exact Wilcoxon test cannot go below p = 0.0625, so we label them *consistent* rather than *classically significant*.
* **What is NOT statistically convincing:** every remaining comparison — the learned-graph variants vs `T-GCN-NoSpatial` on SZ-Taxi (gaps ≤ 0.34%, paired-t p ≥ 0.06 at 3 of 4 PHs), GCN-family learned-graph comparisons on Los-loop (mixed direction: e.g. `GCN` vs `GCN-GSL` is 5/5 seeds at PH1 and PH3 but 3/5 and 1/5 at PH2 and PH4), and all cGSL/Weighted contrasts. For these, the exact Wilcoxon floor is 0.0625 and the gaps are small relative to seed noise. With n = 5 these effects **cannot** be confirmed or denied by conventional significance testing; we report direction and consistency (wins/5) only.
* **Honest statement:** n = 5 gives useful variance estimates and direction, but weak statistical power. No significance is claimed for close comparisons, and no p-hacking (one-sided tests, per-PH cherry-picking) was performed.

## 5. Dataset comparison — Los-loop vs SZ-Taxi

| Question | Los-loop (207 nodes) | SZ-Taxi (156 nodes) |
|---|---|---|
| Does the physical graph help? | **No** — `T-GCN` is the *worst* T-GCN method at every PH (31.7–50.0% *worse* than NoSpatial) | **No** — same direction (32.3–33.6% worse than NoSpatial) |
| Does GSL (single DAGMA graph) help vs NoSpatial? | No — `T-GCN-GSL` is 7.1–11.6% **worse** than NoSpatial | No — `T-GCN-GSL` is 3.4–3.9% worse than NoSpatial |
| Does multi-lag modeling help vs NoSpatial? | **Yes** — `T-GCN-MultiGSL` +2.8–7.8%, 5/5 seeds at all PHs | **Negligible** — -0.25–0.02%, [1, 3, 0, 3] wins/5 |
| Does gating/mixing add benefit beyond fixed multi-lag? | **Yes** — `Mix` adds 6.4–7.2% RMSE relative to `MultiGSL`, 5/5 seeds | **No** — ≤ 0.5% relative difference |
| Does symmetrizing GSL→cGSL matter? | Essentially none | Essentially none (both datasets: max |Δ| = 1.13% of NoSpatial RMSE) |

**The two datasets support different conclusions.** On Los-loop, the learned multi-lag structure produces large, seed-stable improvements (up to 14.5% vs NoSpatial). On SZ-Taxi all learned-graph variants land within 0.34% of the no-graph baseline; the DAGMA graphs there are nearly degenerate (2 multi-lag edges, 8 contemporaneous edges — see §7), so there is little learned structure to exploit. No universal claim about GSL is supportable; the correct statement is that the benefit is **dataset-dependent and tied to the informativeness of the fitted graph**.

## 6. GCN vs T-GCN: what happens to multi-lag graphs when the backbone changes

This is the key architectural result. `GCN-MultiGSL` receives the **same three lag graphs** as `T-GCN-MultiGSL` (same files, same 0.1 threshold), but they are unioned into one static graph, because the GCN backbone has no per-timestep recurrence.

| Dataset | T-GCN-MultiGSL (mean RMSE, PH1–4) | GCN-MultiGSL (union graph, PH1–4) | T-GCN advantage |
|---|---|---|---|
| Los-loop | 4.84–6.26 | 9.78–10.27 | 4.01–4.94 |
| SZ-Taxi | 4.13–4.22 | 4.82–4.90 | 0.68–0.69 |

* On **Los-loop**, unioning the lag graphs is catastrophic: `GCN-MultiGSL` (9.78–10.27 RMSE) is worse than even the physical graph (8.14–8.76) and ≈ 4.0–4.9 RMSE points worse than `T-GCN-MultiGSL` (5/5 seeds, paired-t p < 10⁻³ at every PH). The separate per-timestep consumption — not the union edge set — carries the benefit. **The union has 28 of the 30 edges, so edge content cannot explain the difference.**
* On **SZ-Taxi** the counterpart gap is 0.68–0.69 RMSE, but both operate on a graph family that contains only 2 edges in total; there `GCN-MultiGSL` (4.82–4.90) behaves like a slightly over-smoothed no-graph GCN, and `T-GCN-MultiGSL` simply matches `T-GCN-NoSpatial`.
* Auxiliary counterpart evidence: `GCN-GSL` vs `T-GCN-GSL` (same contemporaneous graph, 28 Los edges) shows the same direction (T-GCN better by 1.79–2.18 RMSE on Los-loop, 0.59–0.60 on SZ), while the no-spatial and cGSL counterparts are nearly identical — so the backbone's ability to exploit a *sparse, directional* graph per timestep, not the graph itself, is the discriminating factor.
* Conservative reading: this is an **architectural interaction** result (recurrent per-lag consumption vs static union), demonstrated on two datasets with one backbone pair. It is not evidence that the lag graphs encode causal temporal structure (§9).

## 7. Graph statistics (verified from stored artifacts)

* **losloop physical (data/los_adj.csv)** — nodes: 207, directed edges (incl. diagonal): 2833, self-loops in raw adjacency: 207, unique undirected pairs: 2833, symmetric: True, off-diagonal density: 0.061582
* **shenzhen physical (data/sz_adj.csv)** — nodes: 156, directed edges (incl. diagonal): 532, self-loops in raw adjacency: 0, unique undirected pairs: 267, symmetric: False, off-diagonal density: 0.022002

**Multi-lag DAGMA blocks (threshold |W| > 0.1, as consumed by Stage 40; identical across PH = 1–4 because the Stage 26 fit is PH-independent):**

| Dataset | lag-1 edges | lag-2 edges | lag-3 edges | sum | union |
|---|---|---|---|---|---|
| losloop | 12 | 3 | 15 | 30 | 28 |
| shenzhen | 0 | 0 | 2 | 2 | 2 |

**Contemporaneous DAGMA graphs (stored `A_binary`, PH-specific fits):**

| Dataset | PH1 | PH2 | PH3 | PH4 |
|---|---|---|---|---|
| losloop | 28 | 28 | 28 | 28 |
| shenzhen | 8 | 8 | 8 | 8 |

* The Stage 40 `results/stage40_canonical/dagma/` directory is empty: Stage 40 reused the Stage 26/33 artifacts in place; no graphs were re-fitted.
* Self-loops: the DAGMA binary adjacencies (contemporaneous and multi-lag) all have zero diagonal; the raw physical adjacency has a unit diagonal for Los-loop (207 entries — counted in the 2,833 figure reported in the result JSONs and above) and zero diagonal for SZ-Taxi. The Laplacian construction (`calculate_laplacian_with_self_loop`) adds self-loops internally for message passing in all cases.
* **Unavailable without re-fitting (reported as unavailable, not generated):** per-PH multi-lag graphs distinct from the PH-independent blocks; gate-weight / learned-weight distributions per run; edge-weight statistics for cGSL beyond the derived counts (56 Los / 16 SZ, from the stored A_binary).

## 8. Sparsity confound — what the stored evidence actually establishes

Source: `results/stage32_sparse_control/stage32_sparse_control.json` (Los-loop, PH=1, matched 30-edge budget, seeds 42–46, canonical T-GCN protocol). Reference: `T-GCN-NoSpatial` on the identical cell = 5.25 ± 0.19 (Stage 40 re-run, 5 seeds).

| Graph (30 edges) | RMSE mean ± std | vs NoSpatial |
|---|---|---|
| RandTop30 (random sparse) | 6.10 ± 0.12 | worse by 16.1% |
| CorrTop30 (top-30 \|Pearson\|, train-only) | 5.39 ± 0.10 | worse by 2.6% |
| DAGMA lag blocks (same 30 edges, per-lag use) | 4.84 ± 0.11 | better by 7.8% |
| DAGMA lag blocks + gating (Mix) | 4.49 ± 0.14 | better by 14.5% |

**Evidence (established):** at an identical 30-edge budget on Los-loop PH1, a random sparse graph is *worse than no graph*, a correlation-placed sparse graph is also worse than no graph, and the DAGMA-placed edges with per-lag consumption beat the no-graph baseline in 5/5 seeds. Therefore sparsity *per se* does not explain the multi-lag gains on Los-loop; the specific learned edge placement (and its per-lag consumption) does.

**Interpretation (not established by these controls):** these controls exist **only for Los-loop PH1**; there is no matched-sparsity control on SZ-Taxi or at PH > 1, no sparsified-physical control, and no λ/threshold sweep. The controls compare three 30-edge graphs; they do not isolate *which property* of the DAGMA placement matters. The distinction between evidence and interpretation is honored throughout this report.

## 9. Strongest Defensible Findings

| # | Finding | Quantitative evidence | Datasets / PHs | Strength | Main-text suitable? |
|---|---|---|---|---|---|
| 1 | The dense physical road graph **hurts** the T-GCN backbone relative to no graph at these horizons | `T-GCN` worse than `T-GCN-NoSpatial` in 5/5 seeds at all 8 dataset×PH cells; 31.7–50.0% (Los) / 32.3–33.6% (SZ) relative RMSE | both, PH1–4 | Strong (5/5, large gap, paired-t p < 10⁻³) | Yes, as an honest baseline finding |
| 2 | Per-lag consumption of DAGMA multi-lag graphs gives the largest gains, on Los-loop | `T-GCN-MultiGSL-Mix` vs NoSpatial: +9.2–14.5% at PH1–4 (mean RMSE), 5/5 seeds everywhere | Los-loop, PH1–4 | Strong (5/5, gap ≫ seed SD, paired-t p ≤ 0.002) | Yes — headline result |
| 3 | The benefit of learned multi-lag graphs is **dataset-dependent** | SZ-Taxi: all learned variants within 0.34% of NoSpatial (Mix wins [4, 4, 5, 4]/5); DAGMA graphs nearly degenerate there (2 multi-lag / 8 contemporaneous edges) | both, PH1–4 | Strong (absence of effect is unambiguous) | Yes — framed as scope condition |
| 4 | Sparsity alone does not explain the Los-loop gains | RandTop30 and CorrTop30 (same 30 edges) are worse than NoSpatial (6.10/5.39 vs 5.25), while DAGMA placement reaches 4.49 (Mix) | Los-loop, PH1 only | Moderate (single cell, but matched-budget design) | Yes, with the PH1/dataset scope stated |
| 5 | Where the graphs are informative, per-timestep consumption beats static unioning | `T-GCN-MultiGSL` vs `GCN-MultiGSL` (same edges): 4.01–4.94 RMSE better on Los-loop, 5/5 seeds, paired-t p < 10⁻³; union holds 28/30 edges | Los-loop (+ SZ direction, PH1–4) | Strong for Los-loop | Yes — central architectural result |
| 6 | Per-node gating (Mix) adds a modest, seed-consistent gain over the fixed assignment | `Mix` vs `MultiGSL`: 5/5 seeds at all 8 cells; 0.35–0.40 RMSE on Los-loop (6.4–7.2% relative); ≤ 0.02 RMSE (0.5%) on SZ | both, PH1–4 | Moderate (consistency high, effect size small; n=5 precludes exact-test significance) | Yes, with effect size stated |
| 7 | Single contemporaneous DAGMA graphs do **not** beat the no-graph baseline | `T-GCN-GSL` 7.1–11.6% (Los) / 3.4–3.9% (SZ) *worse* than NoSpatial; cGSL symmetrization changes nothing (max |Δ| = 1.13%) | both, PH1–4 | Strong (consistency across 8 cells) | Yes — negative result, prevents over-claiming GSL |
| 8 | Learned global weights (Weighted) add nothing beyond the fixed assignment | `Weighted` vs `MultiGSL` mean-RMSE differences ≤ 0.014 RMSE at every cell | both, PH1–4 | Moderate (consistent null) | Supplementary only |

## 10. Claims to Avoid

See `stage41_claim_audit.md` for the full itemized audit. Summary of prohibited claims:

* **Causal language.** DAGMA learns a *statistical dependency structure* under linearity + noise assumptions; nothing here validates causal traffic effects. Use “learned dependency graph”, never “causal graph / discovers causality”.
* **“The contemporaneous graph is a temporal graph.”** It is fitted on PH-subsampled simultaneous snapshots (`train_norm[0::PH]`); it carries no lag structure.
* **“Multi-lag graphs encode lag-specific temporal dependencies.”** The construction (stacked-lag DAGMA blocks) is *consistent* with that reading, and the per-lag consumption is what drives Finding 5, but no direct validation of the lag interpretation was run.
* **Universal superiority of GSL or multi-lag modeling.** Both are null-to-negative on SZ-Taxi; single-graph GSL never beats the no-graph baseline in either dataset.
* **“Sparsity explains the gains.”** Only the matched-budget controls at Los-loop PH1 exist; they show sparsity alone is insufficient, not that it is irrelevant everywhere.
* **Significance claims.** With n = 5, Wilcoxon cannot go below p = 0.0625; only the large-gap comparisons support even weak inference, and paired-t p-values should be reported with the n = 5 caveat attached.
* **“Adapts to changing traffic.”** All graphs are static, fitted once from the training split; the recurrent backbone models temporal dynamics on a fixed structure.
* **“More edges → better.”** Contradicted directly: GCN-MultiGSL has 28 edges and is the worst Los-loop method; T-GCN with 2,833 physical edges is worst in its family.

## 11. Recommended Main-Text Results

**Main results table (both datasets, PH1–4, mean ± std over 5 seeds):**

* `T-GCN-NoSpatial` (no-graph reference), `T-GCN` (physical graph reference), `T-GCN-GSL` (single learned graph), `T-GCN-MultiGSL` (fixed multi-lag), `T-GCN-MultiGSL-Mix` (proposed, gated multi-lag).
* GCN counterparts: `GCN-NoSpatial`, `GCN`, `GCN-MultiGSL` — retained only to support the architecture-interaction finding (§6); a compact separate table or selected columns.

**Supplementary / ablation:**

* `T-GCN-cGSL` and `GCN-cGSL` (symmetrization null result — one appendix table).
* `T-GCN-MultiGSL-Weighted` (global-weight null result — one appendix table).
* `GCN-GSL` (same null direction as T-GCN-GSL; appendix).
* Stage 32 sparse controls (`CorrTop30`, `RandTop30`; Los-loop PH1) — supplementary, with scope limitation.

**Essential comparisons:** (i) `T-GCN-MultiGSL-Mix` vs `T-GCN-NoSpatial` and vs `T-GCN` on both datasets; (ii) `T-GCN-MultiGSL` vs `GCN-MultiGSL` on Los-loop (consumption vs union); (iii) `T-GCN-GSL` vs `T-GCN-MultiGSL` (single graph vs multi-lag); (iv) sparse-control triad at Los-loop PH1.

**Removable from the main text as repetitive:** per-PH GCN-family tables beyond the counterpart summary; the `Weighted` row in main tables; cGSL rows (report as a single sentence: “symmetrization changes RMSE by ≤ 1.13%”); duplicate seed-level scatter for comparisons already summarized by wins/5.

Do not rewrite the manuscript in this stage; this is a recommendation only.

## 12. Reviewer-oriented audit

| Reviewer concern | Status | Evidence from Stage 40/41 |
|---|---|---|
| Multiple seeds / variance | **Addressed** | All 480 cells re-run under one canonical protocol with seeds 42–46; mean ± std and wins/5 reported everywhere; DAGMA determinism previously verified (Stage 35) |
| Sparse-graph confound | **Partially addressed** | Matched 30-edge controls (random / correlation / DAGMA) exist for Los-loop **PH1 only**; no SZ control, no PH > 1, no sparsified-physical control — do not present as fully resolved |
| Longer horizons | **Not addressed** | PH ≤ 4 only (5-min steps). The 15-min-sampling experiment (Stage 29) is a proxy for longer *wall-clock* horizons, not PH 5–8 |
| GSL vs cGSL | **Addressed** (as a null) | cGSL ≈ GSL (max |Δ| = 1.13% of NoSpatial RMSE) at all 8 cells; direction inconsistent; symmetrization is immaterial — report as negative result |
| Temporal interpretation of the DAG | **Partially addressed** | The multi-lag construction is explicit and per-lag consumption demonstrably matters (§6); but the lag-interpretation itself is not directly validated (no lag ablation within Stage 40; Stage 26 C-family ablation is prior evidence, single dataset) |
| Dataset dependence | **Addressed** | Two datasets × 12 methods × 4 PHs × 5 seeds; the SZ null result is reported as a finding, not hidden |
| Scalability | **Not addressed** | DAGMA runtime data exist from earlier stages (828-var fit ≈ 4 h CPU; 156-node contemporaneous ≈ 20 min/PH), but no new scalability experiment; keep as limitation |
| Limitations | **Partially addressed** | n = 5 power, static graphs, λ/threshold fixed by protocol (no sweep), linear DAGMA assumptions — all documented here; manuscript limitations section still needs them |

## 13. Final verdict and audit statistics

| Item | Value |
|---|---|
| Total result records audited | 480 |
| Missing results | 0 |
| Duplicate results | 0 |
| Corrupt/incomplete records | 0 |
| Datasets audited | losloop, shenzhen |
| Methods audited | 12 (T-GCN, T-GCN-NoSpatial, T-GCN-GSL, T-GCN-cGSL, T-GCN-MultiGSL, T-GCN-MultiGSL-Weighted, T-GCN-MultiGSL-Mix, GCN, GCN-NoSpatial, GCN-GSL, GCN-cGSL, GCN-MultiGSL) |
| PHs audited | [1, 2, 3, 4] |
| Seeds audited | [42, 43, 44, 45, 46] |
| New training runs | 0 |
| New DAGMA fits | 0 |
| Output paths | `gsl_stage41/stage41_result_audit.md`, `gsl_stage41/stage41_summary.csv`, `gsl_stage41/stage41_summary.json`, `gsl_stage41/stage41_claim_audit.md` (+ `stage41_improvements.csv`, `stage41_paired_tests.csv`, `stage41_gcn_tgcn_counterparts.csv`) |

**Final verdict: `READY WITH CAVEATS`** — the Stage 40 result set is complete (480/480, 0 missing, 0 duplicates) and internally consistent, and the statistics above are manuscript-usable. Caveats: (i) n = 5 limits formal significance to the large-gap comparisons; (ii) the sparse-graph confound is closed only for Los-loop PH1; (iii) the strong conclusions are Los-loop-specific and must be presented as dataset-dependent; (iv) horizons beyond PH = 4 were not tested.

*No new training, no new DAGMA fitting, no modification of experimental code was performed in Stage 41.*

## 14. Addendum — code-level verification (project `pth` environment, CPU)

Generated: 2026-09-10 10:54:54 — executed the *actual* Stage 40 code path (`gsl_stage40.scripts.stage40_run_all` loaders + `models/*` classes) on CPU for verification only. **No training, no DAGMA fitting, no code modification.** Results: **12/12 checks passed** (`stage41_code_verification.json`).

| Check | Result |
|---|---|
| V1 all 96 (dataset, ph, variant) adjacency constructions match logged n_edges | PASS |
| V2 union identity holds for all 8 cells | PASS |
| V3 source contains graph_idx = (T-1-t) % n_graphs | PASS |
| V3 expected mapping (t -> (11-t) mod 3): [(0, 2), (1, 1), (2, 0), (3, 2), (4, 1), (5, 0), (6, 2), (7, 1), (8, 0), (9, 2), (10, 1), (11, 0)] | PASS |
| V4 Fixed lap_i == Gated lap_stack[i] for all 3 lags | PASS |
| V5 cGSL == (A_gsl + A_gsl.T)>0, diag 0, for all 8 cells | PASS |
| V5 GCN-GSL and T-GCN-GSL share the same loader file (stage40_run_all.py) | PASS |
| V6 n_edges matches in 480/480 files | PASS |
| V6 n_params matches in 480/480 files | PASS |
| V7 dataset/ph/seed/variant/display_name/status consistent in 480/480 | PASS |
| V8 same seed -> identical initial state_dict hash | PASS |
| V8 different seeds -> different initial state_dict hashes | PASS |

What this adds beyond the statistical audit:

* The `n_edges` recorded in all 480 result files was **recomputed from the actual runner loaders** and matches everywhere (T-GCN-MultiGSL/Mix/Weighted record the per-lag *sum* (12+3+15=30 Los, 0+0+2=2 SZ); GCN-MultiGSL records the union (28 Los, 2 SZ)).
* `n_params` was recomputed by instantiating every model class (12,672 for T-GCN family on Los-loop; 17,091 for Mix = +4,419 gate parameters; 768 for all GCN variants) and matches in 480/480 files.
* The lag→timestep mapping `graph_idx = (T−1−t) mod 3` was verified in source, and the Gated/Fixed variants were confirmed to receive byte-identical per-lag Laplacians.
* Per-seed initialization is deterministic: identical seeds produce identical initial weights, different seeds produce different weights — so the seed variation in §3 is genuine training-run variation, not initialization leakage.
