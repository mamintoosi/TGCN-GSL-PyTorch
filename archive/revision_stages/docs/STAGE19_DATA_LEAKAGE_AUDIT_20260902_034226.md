# Stage 19 — Data Leakage and Experimental Validity Audit

**Audit date:** 2026-09-02 03:42 UTC  
**Scope:** Read-only forensic audit. No experimental-pipeline code, configuration, or result was changed.

## 1. Executive Summary

**Confirmed critical findings:** (1) normalization calculates one maximum over the complete raw dataset before the chronological split; this is definite train/test leakage by code path, even though the observed global and training maxima happen to be equal in both supplied datasets; (2) DAGMA receives contemporaneous sensor snapshots, not lagged variables; and (3) the reported Stage 18 runner is single-seed only. It has no `--seeds` parser or seed loop, so the quoted five-seed command silently executes the fixed seed 42 experiment once.

Stage 17 selected threshold 0.3 using held-out test/“validation” RMSE. It is therefore an exploratory test-set sensitivity/tuning result, not independent confirmation. No further quantitative experiments should be treated as final before corrections and reruns.

## 2. Repository/Git State

**Confirmed.** Main source: `utils/`, `models/`, `tasks/`, and `main.py`; data are in `data/`; configurations are in `configs/`; audit/reimplementation scripts are in `gsl_audit/` and `gsl_clean/`; manuscript sources are in `paper/`; prior audits and Stage 17/18 reports are in `doc/`.

The shallow history available contains `572a33c` (Stage 18-A audit) and `9b2f804` (Stage 18-B/C/D runner/report). Cached DAGMA matrices (`data/W_est_*`), physical/correlation/sparse graph files, and earlier reports were added in the available base commit. Their exact generation command, source revision, and provenance cannot be reconstructed from this shallow history.

## 3. Data Split

**Confirmed.** `utils/data/functions.py:15-34` loads `(T,N)` CSV features, computes `train_size=int(T*0.8)`, then takes `data[:train_size]` and `data[train_size:time_len]`. The split is chronological, 80/20, and no pre-split shuffle occurs. Training batches are shuffled later (`main.py:35`), after windows exist.

CSV headers mean effective observations are SZ-Taxi `(T,N)=(2976,156)` and Los-loop `(2016,207)`, yielding train lengths 2380 and 1612. The code calls the 20% partition “validation” (`main.py`, `SpatioTemporalCSVData`) but it is the sole held-out evaluation partition, not a separately held-out test set.

## 4. Sliding Windows

**Confirmed.** Windows are made *after* splitting in `utils/data/functions.py:25-32`: for start `i`, `X[i]=partition[i:i+L]` and `Y[i]=partition[i+L:i+L+PH]`, with `L=seq_len=12`. The loop ends at `len(partition)-L-PH` (strictly excluding the final possible window). Train and held-out windows are generated independently, so no window crosses the split boundary and no raw timestamp participates in both window sets. Apart from the global normalization statistic below, held-out/future values do not enter training features.

## 5. Normalization

**Confirmed — Critical.** `utils/data/functions.py:18-23` performs `max_val=np.max(data); data=data/max_val` **before** the split. `gsl_clean/data_pipeline.py:84-89` reproduces the same order; `gsl_clean/generate_baselines.py:45-51` also divides training rows by a maximum computed from the full feature matrix.

Thus the statistic observes the complete future/held-out segment and is used for both partitions: definite train/test leakage. A small read-only calculation found global max = training max for the committed files (SZ: 86.42919888; Los: 70), so the leakage causes no numerical scaling difference for these particular artifacts. That coincidence does not remove the invalid protocol. All existing normalized forecasting/DAGMA/correlation experiments using this path are affected.

## 6. Physical Graph

**Confirmed.** Physical matrices are directly loaded from `data/sz_adj.csv` or `data/los_adj.csv` in `utils/data/spatiotemporal_csv_data.py:45-48` and `gsl_clean/data_pipeline.py:27-32`; no construction function or edge-semantic metadata exists in this repository.

For SZ, the matrix has 156 nodes, 532 positive off-diagonal entries, no self-loops, and is not exactly symmetric: 265 reciprocal unordered pairs plus 2 one-way pairs = 267 unordered pairs. Consequently “266” is simply `532/2`, an invalid undirected conversion for this slightly asymmetric graph; 532 is the stored directed-entry count. For Los, it has 207 nodes, 2626 positive off-diagonal entries comprising 1313 reciprocal pairs, plus 207 existing diagonal self-loops. GCN/TGCN add identity again in `utils/graph_conv.py:3-10`.

## 7. PhysSparse / PhysSparseDir

**Confirmed for the generated-baseline implementation.** `gsl_clean/generate_baselines.py:103-143` ranks physical upper-triangle entries by their stored adjacency weight, keeps `n_edges`, and writes a symmetric binary graph. It uses topology/physical weights only; no traffic/correlation data, no horizon-specific score, and no traffic leakage. `n_edges` is set to the positive DAGMA entry count (`:180-220`), not selected by an explicit validation procedure.

Stage 18 instead defines `PhysSparse` as 16 top upper-triangle physical-weight pairs symmetrized (`gsl_audit/run_hybrid_experiment.py:101-113,209-210`) and `PhysSparseDir` as eight top upper-triangle pairs retained only in the upper direction (`:87-98,200-201`). K=8/16 were manually fixed and later compared on held-out RMSE; their final selection/use is exploratory test-set comparison, not validated model selection.

## 8. Correlation Graphs

**Confirmed.** `gsl_clean/generate_baselines.py:56-100` and `gsl_audit/run_hybrid_experiment.py:69-84` use `np.corrcoef(train_data.T)`, replace NaNs, rank absolute Pearson correlations on the upper triangle, choose top K, and store a symmetric binary graph. The score is unsigned absolute Pearson r. The input rows are chronological training rows only, but use the global normalization maximum; correlation itself is invariant to positive scaling. Graphs are static, not horizon-specific in data/scoring (only filenames vary).

`Corr-K8` means eight unordered pairs / 16 matrix entries; `Corr-K16` means sixteen unordered pairs / 32 entries. Stage 18 compares these K values on the held-out partition, so the apparent winner cannot be presented as independently selected.

## 9. DAGMA Input

**Confirmed — Critical.** In the original path, `SpatioTemporalCSVData.get_datasets()` creates `self.train_data=[x[0] for x in train_dataset]` (`utils/data/spatiotemporal_csv_data.py:50-63`). `compute_adjacency_matrix()` redundantly extracts `data=[x[0] for x in self.train_data]` (`:93-105`), giving the first timestamp from each length-12 training window. For offset `r=0,...,PH-1`, it fits `DagmaLinear(loss_type="l2").fit(data[r::PH], lambda1)` (`:98-105`).

The clean audit's equivalent is explicit in `gsl_clean/data_pipeline.py:111-167`. No response labels, later in-window timestamps, or held-out windows are supplied to DAGMA.

## 10. Temporal Dependency

**Confirmed.** DAGMA has N contemporaneous variables only. It cannot distinguish `x_i(t-1) -> x_j(t)` from same-time association because neither `x_i(t-1)` nor a sensor-lag variable is a column. A DAG constraint makes the estimated contemporaneous statistical dependency graph acyclic; it does not identify a temporal, predictive, or causal graph. Forecasting models subsequently use temporal windows (and TGCN uses a recurrent mechanism), but that does not change what DAGMA learned.

## 11. DAGMA Data Leakage

**Confirmed.** DAGMA rows originate from training windows only, so test values are not directly supplied. **Confirmed leakage remains:** their normalization has already used all observations. Cached `W_est_*` matrices were committed as artifacts and are loaded without metadata (`utils/data/spatiotemporal_csv_data.py:85-91`); their exact split, normalization, DAGMA version/defaults, and generation command cannot be established independently. Source code supports the intended same-split/global-normalization/current-input claim, not artifact provenance.

## 12. DAGMA Thresholding

**Confirmed.** Original `fit` calls omit `w_threshold` (`utils/data/spatiotemporal_csv_data.py:103-104`); prior audit evidence identifies DAGMA's library default as 0.3. Cached W is therefore already library-thresholded, but source cannot establish the installed library version from the artifact alone. After fitting/loading, the project applies a positive-only presence rule: 3D W is merged by `np.any(W_est_all > 0, axis=2)` (`:111-114`), then binary adjacency is constructed (`:116-128`). Negative weights are discarded. The per-offset Ws are merged per *experiment PH*, not across PH=1..4 files. `use_gsl=2` symmetrizes; `use_gsl=1` stays directed; loops are later added by `utils/graph_conv.py:3-10`.

## 13. Stage 17 Validity

**Confirmed.** `doc/DAGMA_THRESHOLD_SENSITIVITY_REPORT_20260901.md` reports thresholds 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, and 0.30; SZ only; GCN/TGCN; PH 1–4; seed 42; 50 epochs. `gsl_audit/run_threshold_sensitivity.py:160+` evaluates held-out “val” RMSE, and the report calls 0.3 best by that RMSE. Therefore Stage 17 is both a test-set sensitivity analysis and post-hoc hyperparameter tuning if 0.3 is retained as final. The valid claim is descriptive: on this leaked-normalization, single-seed held-out sweep, 0.3 had the lowest observed RMSE among the seven tested values—not that it is independently validated optimal.

## 14. Stage 18 Seed Handling

**Confirmed — Critical.** `gsl_audit/run_hybrid_experiment.py` has no `argparse`, no `--seeds`, no `--graph-subset`, and no seed loop. `SEED=42` is fixed (`:39`), passed at `:321`, and reset in every training invocation (`:47-52,219`). Its loops are 4 PH × 8 graphs × 2 models = 64 runs (`:296-321`). CSV/JSON filenames are fixed (`:343,377`); aggregation is merely a per-PH display table, not multi-seed aggregation.

Accordingly, the stated command with `--seeds 42 43 44 45 46 --graph-subset ...` is accepted as unused Python argv and runs the complete fixed-seed-42 suite once. Stage 18 is genuine one-seed output, not five-seed output.

## 15. Stage 18 Scientific Validity

All categories below are **C — Requires rerun**, for both GCN and TGCN: `GSL`, `GSL+Phys`, `GSL+Corr`, `GSL+PhysC`, `PhysSparseDir`, `Corr-K8`, `Corr-K16`, and `PhysSparse`. Reasons: the data pipeline has definite global-normalization leakage; Stage 18 has one seed only; and K/threshold/hybrid decisions were assessed using held-out RMSE. These are not fundamentally nonsensical graphs (so not D), but no final quantitative conclusion is valid. They remain exploratory descriptive evidence only.

## 16. Hyperparameter Selection

**Confirmed.** `lambda1` is hard-coded by dataset (SZ .01, Los .02) in `utils/data/spatiotemporal_csv_data.py:101-104`. Threshold .3 was inherited as an omitted library default in the original implementation, then Stage 17 explicitly compared it against six alternatives using held-out RMSE. Correlation and physical K/budgets (8/16 and hybrids targeting 16 additions) are fixed in Stage 18 code and their relative quality reported from held-out RMSE. Model settings vary by original YAML; Stage 17/18 use fixed scripts with 50 epochs and fixed learning settings. No validation partition distinct from the final held-out partition is implemented.

Reporting fixed alternatives on the test set is sensitivity analysis. Choosing the final threshold/K/combination because it has the lowest same-test RMSE is test-set selection.

## 17. Code vs Paper Consistency

**Confirmed major mismatch.** `paper/sn-article.tex` claims hidden causal structure and describes road A influencing road B's future state; its commented temporal-DAG description explicitly refers to cross-time layers. The source only feeds contemporaneous `X ∈ R^(M×N)` to DAGMA. The paper's causal, temporal, “predicts at t+1,” and corresponding interpretability claims are stronger than implementation support. The paper also does not document the inherited threshold 0.3 or global pre-split normalization. These claims require revision before publication; this audit does not revise them.

## 18. Current DAGMA Mathematical Specification

Let raw observations be `v(t) ∈ R^N`, `t=0,...,T-1`; let `q=floor(.8T)`, `L=12`, and `PH∈{1,2,3,4}`. Current normalization is `u(t)=v(t)/max_{0≤s<T,i}v_i(s)`. Training-window starts are `s=0,...,M-1`, where `M=q-L-PH` due to the strict loop bound. The extracted matrix is `D[s,:]=u(s,:)`.

For each offset `r=0,...,PH-1`, DAGMA receives:

`X_r = [D[r+k·PH,:]]_{k=0}^{ceil((M-r)/PH)-1} ∈ R^(M_r×N)`.

N is 156 (SZ) or 207 (Los); T is 2976 or 2016; rows are one simultaneous all-sensor observation at window start; columns are individual sensors; row spacing is PH timestamps. There are no lagged variables, no sensor-lag columns, and no labels. `lambda1=.01` (SZ) or `.02` (Los); loss type is l2; `w_threshold=.3` is an inherited library default in the original call. Per-offset W matrices are positive-only union-merged, binarized, optionally symmetrized for cGSL, and then receive identity self-loops plus graph normalization before GCN/TGCN.

PH changes sample spacing, offset partitions, row count, and forecast target length; it does not introduce temporal dependency information into DAGMA. E.g., offset 0 uses `t0,t1,t2,...` for PH=1; `t0,t2,t4,...` for PH=2; and `t0,t4,t8,...` for PH=4.

## 19. Critical Problems

### Global normalization before split
**Evidence:** `utils/data/functions.py:18-23`; clean and baseline equivalents.  
**Scientific consequence:** held-out data determine a training preprocessing statistic.  
**Affected experiments:** all current forecasting, DAGMA, and baseline outputs using these paths.  
**Severity:** Critical.  
**Required correction:** fit scaling only on training observations and apply it to held-out observations.  
**Rerun requirement:** regenerate graphs and rerun all quantitative experiments.

### Contemporaneous DAGMA represented as temporal/causal
**Evidence:** `utils/data/spatiotemporal_csv_data.py:60-63,96-105`; manuscript claims.  
**Scientific consequence:** causal/lagged/predictive edge claims are unsupported.  
**Affected experiments:** all DAGMA/GSL/cGSL and hybrid interpretation/results.  
**Severity:** Critical.  
**Required correction:** revise claims now; separately design and validate a temporal formulation before making temporal claims.  
**Rerun requirement:** a new temporal method requires a distinct experiment campaign.

### Stage 18 claimed multi-seed invocation is one seed
**Evidence:** `gsl_audit/run_hybrid_experiment.py:39,279-384`.  
**Scientific consequence:** no uncertainty, stability, or five-seed claim is supported.  
**Affected experiments:** all Stage 18 comparisons.  
**Severity:** Critical.  
**Required correction:** implement/verify seed parsing, seed loop, isolated result recording, and aggregation in a future stage.  
**Rerun requirement:** all Stage 18 variants under corrected protocol.

### Test-set selection
**Evidence:** Stage 17 report and `gsl_audit/run_threshold_sensitivity.py`; Stage 18 report/runner.  
**Scientific consequence:** selected threshold/K/graph results are optimistically biased.  
**Affected experiments:** Stage 17/18 choices and any paper conclusions relying on them.  
**Severity:** Major.  
**Required correction:** reserve final test set; select only with training/validation.  
**Rerun requirement:** all final comparisons.

## 20. Required Corrections

Do not conduct further confirmatory experiments on the current pipeline. First correct pre-split normalization, establish an explicit train/validation/test protocol, make DAGMA parameters/provenance explicit, implement verified multi-seed execution, and define selection rules without the final test set. Then regenerate DAGMA/correlation/sparse artifacts and rerun baselines, ablations, Stage 17/18 comparisons, and final tables.

## 21. Recommended Stage 20

Investigate—not implement here—a temporal representation with either lag-expanded variables `z(t)=[x(t),...,x(t-L)]`, an explicit past-to-current dependency formulation, or an alternative scalable temporal dependency model. Define temporal semantics, acyclicity constraints, graph projection, and validation protocol before choosing a method.

## Evidence Status

All source-path statements above are **Confirmed** by direct inspection. Artifact provenance (exact cached-W generation environment/command) is **Unresolved** because cached files contain no metadata and available history is shallow. The inference that every reported score is unsuitable as a final result is **Strong inference** from the confirmed leakage, selection, and seed findings.
