# Stage 52 — Introduction & Background Revision Report

**Date:** 2026-09-11  
**Output:** `paper/revised_version/sn-article.tex` (flat working draft)  
**Scope of this stage:** **Introduction** and **Background and Related Work only.**  
**Identity preserved:** *Graph Structure Learning for Traffic Prediction* (title and abstract taken verbatim from the author-final framing in `paper/gsl_master/sn-article.tex`).

---

## 1. What changed

| Part | Action |
|------|--------|
| Title | **Kept** as in `gsl_master` / original paper |
| Abstract | **Copied from `gsl_master`** (author-final; not rewritten) |
| **Introduction** | **Revised from `submitted_version`**, not from the Stage-50 rewrite |
| **Background** | **Revised from `submitted_version`** (structure and citation base preserved; GCN/T-GCN condensed per R1-W2; new short GSL subsection) |
| Method–Conclusion | **Not rewritten this stage**; temporarily `\input` from stage drafts so the PDF compiles |
| Bibliography | Full `MyReferences.bib` from `gsl_master` (55 entries; submitted had ~43 in use) |

Compile output: `paper/revised_version/sn-article.pdf` (28 pages).

---

## 2. Most important changes vs submitted Introduction

1. **Identity / motivation kept:** graph knowledge discovery, spatial–temporal dependence, physical vs functional dependency, attention vs explicit structure.
2. **Softened causal / universal language** (R2 + Stage 41 claim policy):
   - Removed “hidden causal structure”, “one road influences another”, “significantly improve / outperforms in general”.
   - Learned graph described as **statistical dependency** under the estimator’s assumptions.
3. **Added the two evaluation axes** needed by the new experiments without changing the paper’s topic:
   - graph-free control as an important reference (not the paper’s main idea);
   - multi-lag construction and consumption of lag-specific graphs as how GSL is studied, not a new paper topic.
4. **Dataset dependence** stated: gains strongest on Los-loop with lag-aligned use; SZ-Taxi limited / sparse structure.
5. **Sparsity as a methodological question** (R1-W5): one sentence in Intro; no matched-budget numbers in Intro.
6. **Contribution list updated** to the final study (GSL + physical + graph-free; contemporaneous vs multi-lag; consumption; five-seed + sparsity control) without overselling.
7. **Roadmap** updated to current section labels (`sec:method`, `sec:setup`, …) instead of the old acyclicity-section pointer.

---

## 3. Most important changes vs submitted Background

| Change | Reason |
|--------|--------|
| Problem definition / task equation **kept** | Baseline identity |
| Overview of approaches **kept** with the same citation families | R1-W12 / preserve refs |
| Bibliometric pointer **kept** (one sentence + App. C) | R1-W1 lean use |
| GCN subsection **condensed**: one figure, two equations, fewer itemized bullets | R1-W2 |
| T-GCN subsection **kept short** | R1-W2 |
| **New** §2.4 Graph Structure Learning: NOTEARS/DAGMA, thresholding, non-causal caveat | Needed for final method; was buried in proposed-method before |
| **New** §2.5 Position of this work | Ties physical / graph-free / multi-lag / consumption without claiming a new paper identity |
| Softened “influence” wording in traffic/GCN prose | R2 causal-language concern |

---

## 4. Reviewer coverage from these two sections

| Comment | Addressed here? |
|---------|-----------------|
| R1-W1 bibliometrics underused | Yes — brief main-text use + App. C pointer |
| R1-W2 GCN/T-GCN too long | Yes — condensed |
| R1-W3 A/W notation | Partially (Background uses $\mathbf{A}$; full convention in Method) |
| R1-W5 sparsity question | Yes — raised in Intro; details remain in Results |
| R1-W12 citation style | Yes — `\citep`/`\citet` throughout revised sections |
| R2 causal / static-vs-adaptive wording | Yes — Intro + GSL subsection |
| R2 abstract numbers | N/A — author-final abstract already corrected |

Not addressed in this stage (later sections): R1-W4 multi-lag construction detail, W6 seeds, W7 horizons, W8 Results structure, W9 Limitations, Q1–Q4.

---

## 5. Old claims removed or rephrased

| Old claim (submitted) | Status |
|----------------------|--------|
| “hidden causal structure” / causal influence language | **Removed** |
| “significant error reduction” as a general validation of structure discovery | **Rephrased** to conditional, dataset-dependent usefulness |
| Implicit “learned always beats physical” | **Not asserted**; physical vs graph-free vs learned all appear |
| “adapts to changing traffic” (if read as time-varying graph) | **Not used** in Intro/Background |
| Direct temporal-DAG reading of contemporaneous fit | **Not asserted**; multi-lag is the temporal construction |

---

## 6. Citations

- Revised Intro/Background retain the submitted citation base (classical, deep, graph/temporal graph, attention, GCN/T-GCN, GSL/DAGMA/NOTEARS, bibliometrics, reviews).
- Bibliography file has **55** entries; no mass deletion.
- Keys used in the two revised sections are present in `MyReferences.bib` (verified compile).

---

## 7. Needs further code/results check?

| Item | Need? |
|------|--------|
| Exact wording of oversmoothing claim in Intro | No — qualitative; supported by Stage 40 physical vs NoSpatial, not quoted numerically in Intro |
| Whether App. C label `sec:Bibliometric-Methodology` still exists when appendix is assembled | Yes — when appendices are restored from `submitted_version` |
| Interim Method–Conclusion `\input` alignment with new Intro framing | Yes — next writing stage should revise those sections under the *submitted-paper* identity, not the Stage-45 “consumption paper” voice |
| Any number in Intro | Only 14.5% / 43.0% live in the **author abstract**, not re-derived in Intro |

---

## 8. Verdict

**Introduction and Background revised on the submitted-version base; paper identity remains GSL for traffic prediction.**  
PDF: `paper/revised_version/sn-article.pdf` (28 pages).  
Next stage: revise Method / Setup / Results / Discussion / Limitations / Conclusion under the same framing, then restore appendices from `submitted_version`.
