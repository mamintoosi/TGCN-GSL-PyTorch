# Stage 54 — Experimental Setup Revision Report

**Date:** 2026-09-11  
**Output:** `paper/revised_version/sections/setup.tex`  
**Master:** `\input{sections/setup}`  
**PDF:** `paper/revised_version/sn-article.pdf` (28 pages)

**Base:** `submitted_version` experimental setup (datasets, split, horizons, GCN/T-GCN focus, metrics) + verified Stage 40 protocol (not invented).

---

## 1. Kept from submitted

- SZ-Taxi / Los-loop descriptions and complementary urban/highway framing  
- Chronological 80/20 split  
- Horizons PH 1–4 with wall-clock mapping  
- Focus on GCN and T-GCN as foundational backbones (with citations)  
- Window $T=12$; Adam; DAGMA for structure learning  
- RMSE/MAE (equations retained; R²/accuracy de-emphasized per R1-W10)  
- Hardware/PyTorch mention; public code URL; preliminary conference citation  

## 2. Updated to canonical protocol

| Item | Submitted | Revised |
|------|-----------|---------|
| Batch size | 64 | **128** |
| Weight decay | not stated | **1e-4** |
| Hidden dim | not in table | **64** |
| Seeds | one fixed seed | **5 seeds (42–46)** |
| Normalization | not specified as train-max | **train-split max only** |
| Configurations | Physical / GSL / cGSL | + **NoSpatial**, multi-lag family, GCN union |
| Loss | MSE only | T-GCN: MSE + weight $\ell_2$; GCN: MSE |
| Series length | not given | Los **2016** (1612/404); SZ **2976** |

## 3. Added (needed by final study / R1-W5, W6)

- Graph-free control in the configuration list  
- Statistical reporting paragraph ($n=5$, Wilcoxon floor)  
- Matched-sparsity and capacity controls (scoped to Los PH1)  
- Expanded ablation list (physical vs learned vs identity; GSL/cGSL; multi-lag use; 15-min)  
- 15-minute resolution variant subsection  

## 4. Softened

- Ablation bullet that “learned structures improve accuracy” is now framed as a **comparison including identity**, not an assumed win  
- No claim that GSL beats NoSpatial  

## 5. Labels for later sections

`sec:setup`, `sec:datasets`, `sec:methods`, `sec:metrics`, `sec:stats`, `sec:controls`, `sec:ablations`, `sec:variant15` — match interim Results refs.

## 6. Next

**Results** — rewrite from `submitted_version` structure + Stage 40 numbers, under this Setup (not Stage-47 “consumption-first” voice).
