# Stage 30 — Forensic Audit of Stage 27 vs Stage 29

You are performing a **forensic scientific/reproducibility audit** of the TGCN-GSL-PyTorch repository.

## Main Objective

Determine **exactly why Stage 27 and Stage 29 produce radically different forecasting results for Los-Loop at 15-minute temporal resolution**, and determine whether the Stage 29 results are valid and suitable for use in the manuscript.

This is an **AUDIT ONLY** stage.

**Do NOT modify the manuscript.
Do NOT commit changes.
Do NOT silently fix code.**

You may create audit scripts or temporary diagnostic files if necessary, but clearly identify them.

---

# 1. Background

The original Stage 27 experiment attempted to investigate whether the poor performance observed on SZ-Taxi at 15-minute resolution could be explained by temporal resolution.

Stage 27 resampled Los-Loop from 5-minute to 15-minute intervals.

Its reported PH=1 result was:

* Los-Loop-15min NoGraph: RMSE = 8.7772
* Los-Loop-15min MultiGSL-Mix: RMSE = 9.0531
* Improvement = -3.14%

However, the Stage 28 audit found serious experimental-control problems in Stage 27, including:

* different training code path from Stage 26;
* plain MSE instead of the Stage-26 MSE+L1 regularized loss;
* only one seed;
* different training implementation;
* potential reproducibility issues;
* Jaccard analysis including self-loops.

Therefore Stage 29 was created using the established forecasting pipeline and 5 seeds.

Stage 29 reports for Los-Loop-15min (Please check and verify these values from the saved results of stage 29):

### PH=1

NoGraph:

* seed 42: 8.3457
* seed 43: 8.6211
* seed 44: 9.0528
* seed 45: 8.5805
* seed 46: 8.3998
* mean = 8.6000
* std = 0.2492

MultiGSL:

* mean = 7.2462
* std = 0.2975

MultiGSL-Mix:

* seed 42: 6.2022
* seed 43: 6.4048
* seed 44: 6.2239
* seed 45: 5.9199
* seed 46: 6.4500
* mean = 6.2402
* std = 0.1873

Thus:

**MultiGSL-Mix improvement = 27.44%**

Stage 29 also reports:

* PH=2: +21.36%
* PH=3: +19.76%
* PH=4: +16.09%

The important scientific question is therefore:

> Is Stage 29 a valid correction of the flawed Stage 27 experiment, or is Stage 29 itself using a different preprocessing/model/data setup that makes the comparison invalid?

---

# 2. First Task — Extract the Exact Implementations

Inspect the repository and identify the exact code used by:

1. Stage 26
2. Stage 27
3. Stage 29

Especially inspect:

* `gsl_stage26/stage26_resolution_experiment.py`
* `gsl_stage26/stage29_los15min.py`
* `run_stage29_los15min.sh`
* Stage 27 shell script
* the common T-GCN implementation
* `SupervisedForecastTask`
* loss functions
* data loading/preprocessing utilities
* graph loading utilities
* normalization
* train/test splitting
* sequence construction
* random seeding
* optimizer initialization
* batch construction/shuffling
* model initialization
* DAGMA graph generation/loading
* graph thresholding
* MultiGSL implementation
* MultiGSL-Mix implementation

Also inspect the exact Stage 27 and Stage 29 JSON/log artifacts under:

`results/`

Do not rely only on previous audit reports. Verify directly from the current repository.

---

# 3. Construct a COMPLETE Stage 27 vs Stage 29 Comparison

Create a table covering every variable that could affect RMSE.

At minimum compare:

### Data

* source CSV
* raw dataset
* resampling procedure
* number of timesteps
* number of nodes
* feature values
* normalization
* normalization statistics
* train/test split
* exact train/test indices
* sequence length
* prediction horizon
* number of training samples
* number of test samples

### Graph

* DAGMA input
* DAGMA parameters
* DAGMA seed
* L
* lambda
* threshold
* self-loop handling
* number of graphs
* exact W matrices
* exact lag-specific matrices
* graph normalization
* adjacency normalization

### Model

* exact class
* hidden dimension
* number of layers
* activation functions
* graph convolution implementation
* GRU implementation
* MultiGSL implementation
* MultiGSL-Mix implementation
* graph mixing mechanism
* initialization

### Training

* loss function
* regularization
* regularization coefficient
* optimizer
* learning rate
* weight decay
* batch size
* epochs
* scheduler
* gradient clipping
* dropout
* early stopping
* shuffling
* seed initialization
* PyTorch deterministic settings
* NumPy seed
* Python random seed
* CUDA seed

### Evaluation

* prediction target
* RMSE implementation
* normalization/denormalization
* aggregation over nodes/time
* best epoch vs final epoch
* checkpoint selection
* averaging across seeds
* any post-processing

---

# 4. Determine the Exact Source of the Numerical Difference

Do NOT stop at saying:

> "Stage 27 used a different code path."

That is already known.

We need to determine **which differences actually explain the approximately 31% relative difference in the proposed method's RMSE**:

Stage 27:

`9.0531`

Stage 29:

`6.2022`

For NoGraph:

Stage 27:

`8.7772`

Stage 29:

`8.3457`

The proposed method changed dramatically, while NoGraph changed much less.

Therefore investigate specifically:

### A. Loss function

Determine whether removing/restoring L1 regularization can explain the difference.

### B. Training implementation

Determine whether Stage 27's inline training loop differs materially from the canonical Stage-26 implementation.

### C. Randomness

Determine whether the seed handling differs.

### D. Data construction

Verify that Stage 27 and Stage 29 use **exactly the same Los-Loop-15min data**.

Compare hashes/statistics where possible.

### E. Sequence construction

Verify exact samples.

For example:

* Does both use the same `seq_len=12`?
* Same PH?
* Same target indices?
* Same train/test boundary?

### F. Graphs

Verify that Stage 27 and Stage 29 use the **same DAGMA graph matrices**.

If not, determine why.

### G. MultiGSL-Mix

Verify that both stages instantiate the same model and pass the same graphs in the same order.

This is particularly important.

### H. Evaluation

Verify whether one result is normalized and the other denormalized, or whether any other evaluation discrepancy exists.

---

# 5. Reproduce Stage 27 Exactly

If possible, execute the Stage 27 PH=1 experiment exactly as originally implemented.

Use its original:

* data
* graph
* seed
* training code
* loss
* hyperparameters

Do NOT modify it first.

The goal is to independently reproduce:

`NoGraph ≈ 8.7772`

and

`MultiGSL-Mix ≈ 9.0531`

If exact reproduction is not possible, identify precisely why.

Report:

* expected result
* reproduced result
* absolute difference
* relative difference

---

# 6. Reproduce Stage 29 Exactly

Likewise verify Stage 29.

At minimum reproduce seed 42:

* NoGraph ≈ 8.3457
* MultiGSL ≈ 7.1031
* MultiGSL-Mix ≈ 6.2022

If possible, verify all five seeds from the saved artifacts.

---

# 7. Perform Controlled Ablation Diagnostics

If the source of the discrepancy is not immediately obvious, run targeted diagnostic experiments.

Do NOT launch unnecessary large experiments.

Use PH=1 and preferably seed=42 first.

Construct a minimal matrix such as:

| Variant | Stage-27 data | Stage-27 training | Stage-29 training | Loss     | Expected diagnostic             |
| ------- | ------------- | ----------------- | ----------------- | -------- | ------------------------------- |
| A       | yes           | yes               | no                | Stage-27 | reproduce Stage 27              |
| B       | yes           | no                | yes               | Stage-27 | isolate training implementation |
| C       | yes           | no                | yes               | Stage-29 | isolate loss                    |
| D       | Stage-29      | Stage-29          | yes               | Stage-29 | canonical result                |

Add further variants only if necessary.

The objective is **causal attribution**, not a large hyperparameter search.

---

# 8. Check Whether Stage 29 Actually Matches the Main Paper Pipeline

This is critical.

Compare Stage 29 against the code used to generate the main manuscript results.

Determine whether Stage 29 is genuinely using the same:

* T-GCN backbone
* loss
* training procedure
* graph normalization
* graph mixing
* preprocessing
* evaluation protocol

If Stage 29 differs from the main experimental pipeline in any material way, flag it.

---

# 9. Verify the Meaning of "Los-Loop as 15-Minute Dataset"

The intended scientific experiment is NOT:

> compare 5-minute PH=1 against 15-minute PH=1.

Do not make that interpretation.

The intended experiment is:

> Treat Los-Loop as an independent dataset whose temporal sampling interval is 15 minutes, analogous to SZ-Taxi's 15-minute temporal resolution.

Therefore:

* Los-Loop-15min should be treated as a 15-minute dataset.
* PH=1 means 15 minutes ahead.
* PH=2 means 30 minutes ahead.
* PH=3 means 45 minutes ahead.
* PH=4 means 60 minutes ahead.

The experiment does NOT need to match the physical horizon of the original 5-minute Los-Loop experiment.

This is important for the final scientific interpretation.

---

# 10. Check the Resampling Procedure

Verify exactly how Los-Loop was converted from 5-minute to 15-minute resolution.

Confirm:

* three consecutive observations are averaged;
* no temporal leakage;
* exact number of samples;
* exact train/test split;
* whether resampling was performed before splitting;
* whether timestamps are relevant;
* whether the resulting data are identical across Stage 27 and Stage 29.

Report any discrepancy.

---

# 11. Statistical Assessment of Stage 29

Verify the five-seed statistics independently.

For each PH and each method calculate:

* mean RMSE
* standard deviation
* improvement over NoGraph
* per-seed improvement
* optionally 95% confidence interval

Do not perform significance testing unless it is statistically meaningful with n=5; if you do, clearly state the limitation.

The main question is whether the large PH=1 improvement is robust across seeds.

---

# 12. Scientific Interpretation

After the forensic analysis, answer the following questions explicitly:

### Q1

Is Stage 27 invalid, partially invalid, or valid?

### Q2

Is Stage 29 valid?

### Q3

What exact implementation difference explains the discrepancy?

### Q4

Can the Stage 29 result:

> Los-Loop-15min +27.44% at PH=1

be trusted?

### Q5

Can we use Stage 29 in the manuscript?

### Q6

Does Stage 29 support or contradict the hypothesis that SZ-Taxi's weak result is caused by its 15-minute temporal resolution?

The answer must distinguish:

* "temporal resolution alone"
  from
* "dataset-specific factors"

Do NOT claim causality unless the experiment supports it.

---

# 13. Important Expected Scientific Possibility

Do not assume beforehand that Stage 27 or Stage 29 is correct.

There are three possible outcomes:

### Outcome A

Stage 27 is demonstrably flawed and Stage 29 correctly uses the canonical pipeline.

Then conclude:

> Stage 27's −3.14% result should be discarded. Stage 29 provides the valid Los-Loop-15min result.

### Outcome B

Stage 29 contains a hidden methodological difference.

Then identify it and determine whether Stage 29 must be rerun.

### Outcome C

Both are technically correct but measure different experimental protocols.

Then clearly explain the distinction and determine which protocol is appropriate for the manuscript.

---

# 14. Do NOT Modify the Manuscript

This stage is an audit only.

Do NOT edit:

* `paper/*.tex`
* manuscript tables
* manuscript text
* figures

Do NOT commit anything.

If diagnostic scripts are created, place them under an explicitly named audit/diagnostic location and report them.

---

# 15. Produce a Detailed Audit Report

Create:

`results/stage30_forensic_audit/STAGE30_FORENSIC_AUDIT.md`

The report must contain:

1. Executive Summary
2. Stage 27 Configuration
3. Stage 29 Configuration
4. Complete Difference Table
5. Reproduction Results
6. Controlled Ablation Results
7. Root Cause of the Numerical Discrepancy
8. Statistical Verification
9. Assessment of Stage 27
10. Assessment of Stage 29
11. Implications for the Manuscript
12. Recommended Next Step

The executive summary must end with a clear verdict:

**VERDICT:**

* Stage 27: VALID / INVALID / PARTIALLY VALID
* Stage 29: VALID / INVALID / REQUIRES RERUN
* Recommended Los-Loop-15min result for manuscript: ...
* Is Stage 29 suitable for manuscript use? YES / NO / AFTER RERUN
* Does 15-minute resolution explain SZ-Taxi's weak performance? YES / NO / NOT ESTABLISHED

---

# 16. Final Constraint

This is a scientific audit, not a software cleanup task.

Do not "fix" Stage 27 and then call the fixed version Stage 27.

Preserve the original implementation sufficiently to reproduce its result.

If a rerun is required, clearly distinguish:

* original Stage 27
* diagnostic reproduction
* corrected experiment
* Stage 29

The ultimate objective is to establish a defensible scientific chain:

**Stage 27 → audit → identified flaw(s) → Stage 29 → verification → manuscript decision.**
