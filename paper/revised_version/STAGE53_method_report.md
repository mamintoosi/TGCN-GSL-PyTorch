# Stage 53 — Method Section Revision Report

**Date:** 2026-09-11  
**Output:** `paper/revised_version/sections/method.tex`  
**Master:** `paper/revised_version/sn-article.tex` now `\input{sections/method}`  
**PDF:** `paper/revised_version/sn-article.pdf` (28 pages)

**Base:** `submitted_version` §3 (Proposed Method), not the Stage-46 “consumption paper” draft.

---

## 1. Structure (aligned with submitted identity)

| Subsection | Content |
|------------|---------|
| §3 title | *The Proposed Method: Estimating the Adjacency Matrix with Graph Structure Learning* |
| Motivation | Proximity vs statistical association; keep `dummy-network` figure, soften “influence” |
| Integration | Static learned graph + temporal backbone; A/W convention in main text (R1-W3); graph normalization; physical + graph-free controls |
| Continuous GSL (NOTEARS) | Keep submitted continuous-optimization narrative, theorem sketch, augmented Lagrangian |
| **DAGMA** | Actual estimator: log-det acyclicity, linear SEM + $\ell_1$; non-causal caveat |
| Contemporaneous GSL + **cGSL** | Original single-graph idea; cGSL defined **before** results (R2-4) |
| Multi-lag GSL | Extension for lag-specific structure; PH-independent fit; consumer threshold 0.1 |
| Using graphs in GCN/T-GCN | Single-graph use; MultiGSL / Weighted / Mix; GCN union |
| Summary table | Graph sources at a glance |

---

## 2. Kept vs changed vs dropped

**Kept from submitted**
- Section identity and GSL framing  
- Motivation + conceptual figure  
- NOTEARS continuous program and $h(W)=\tr(e^{W\circ W})-d$ (as prior method)  
- A→W explanation **in main text**, not only a footnote  
- Integration of static structure with GRU temporal modeling  

**Added (required by final protocol / reviewers)**
- DAGMA as the working estimator  
- Exact contemporaneous construction (`train_norm[0::PH]`, $\lambda_1$, threshold 0.3)  
- cGSL as same-artifact symmetrization  
- Multi-lag construction and edge budgets  
- Consumption mechanisms (fixed / weighted / Mix / GCN union)  
- Graph-free and physical references  

**Dropped / rephrased**
- “Hidden causal structure”, “causal relationships” for DAGs in traffic  
- “Our method can adapt to changing traffic patterns” as time-varying graph  
- Closing claim that experiments already “demonstrate GSL outperforms the physical graph” without qualification  
- Long list item “influence” language in the conceptual figure  

---

## 3. Reviewer points covered here

| Comment | How |
|---------|-----|
| R1-W3 A/W notation | Explicit paragraph at start of §3.1 |
| R1-W4 temporal reading | Contemporaneous vs multi-lag separated; no $t\to t+1$ claim for contemporaneous DAG |
| R2-4 cGSL late | cGSL in §3.4 before Results |
| R2-2 / R2-3 causal / static | Acyclicity = regularizer; Mix changes *use*, not edge set |

---

## 4. Labels

`sec:method`, `sec:gsl_contemp`, `sec:gsl_multilag`, `sec:consumption` match interim Setup/Results `\ref`s.

---

## 5. Next stage

**Experimental Setup** (from `submitted_version` §4 + Stage 40 protocol): datasets, split, seeds, metrics, controls — same framing rules.

No claim in Method requires further code checks beyond Stage 46 verification already recorded.
