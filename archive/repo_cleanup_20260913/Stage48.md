# Stage 48 — Write Manuscript Section 6 (Discussion)

## Objective

Write `paper/gsl_stage48/sections/discussion.tex` (Section 6 only) and
`paper/gsl_stage48/stage48_writing_report.md`.

Do not modify Results, Method, Setup, submitted/previous revisions, code, or
numbers. Do not write Introduction, Background, Limitations, Conclusion, or
Abstract.

## Governing principle

One self-contained study. No manuscript history. No causal claims about
traffic. Learned graphs are **static statistical dependency estimates**;
only their **consumption** may vary per timestep.

## Sources (priority)

1. `paper/gsl_stage45_1/stage45_1_final_architecture.md` §6
2. `paper/gsl_stage47/sections/results.tex` + `stage47_writing_report.md`
3. `paper/gsl_stage46/sections/method.tex` (structure vs consumption; A/W)
4. Stage 41 claim policy (no causal language; n=5 caveats)

## Accepted refinements (R1/R2)

- **R1:** Do not write that the benefit is *located* in consumption rather than the edge set as a pure causal fact. Prefer: *the results indicate that the utility of the learned multi-lag structure depends critically on how the graphs are consumed* / *provides evidence that graph consumption is a major determinant*. Note backbone confound of GCN vs T-GCN.
- **R2:** Improvements phrased as *relative RMSE reduction*; headline remains Mix vs NoSpatial on Los-loop (not only vs physical).

## Required structure

```
\section{Discussion}\label{sec:discussion}

\subsection{Why Dense Graphs Fail and What Learned Structure Adds}\label{sec:disc-dense}
\subsection{Structure Versus Consumption}\label{sec:disc-consumption}
\subsection{What the Learned Graph Represents}\label{sec:disc-interpretation}
```

### §6.1
- Oversmoothing / harmful-default reading of §5.1–5.2: dense physical graph (Los ~2833 entries including self-loops) worse than identity; single learned graphs intermediate but still worse than identity.
- Tie to proximity-vs-dependency framing (textual forward ref to Introduction; no `\ref{sec:intro}` unless declared).
- Do not claim universal failure of graph-based forecasting — scoped to these backbones, datasets, horizons.

### §6.2 (interpretation centerpiece)
- Dissociation: identical multi-lag artifacts; GCN-MultiGSL union (28/30 slots) 9.78 vs T-GCN-MultiGSL 4.84 (Los PH1) — opposite outcomes.
- Weighted ≈ Fixed → global scalars insufficient; Mix adds per-node/per-timestep selection (+0.35–0.40 RMSE on Los).
- cGSL vs GSL as aggregation-compatibility miniature: robust help for GCN, near-null for T-GCN.
- Language: *consistent with*, *cannot be explained by the graph alone*, *depends critically on how the graphs are consumed*. Explicit backbone confound sentence.

### §6.3
- Lag-specific statistical dependency structure; lag reading consistent with construction, **not independently validated**.
- Static graphs; gate changes usage, not the edge set (R2-3).
- SZ null + 2-edge graphs as boundary of informativeness; 15-min variant as robustness of the *relative* Los gain, not PH5–8 evidence.
- No causal language; acyclicity = fitting regularizer.

## Forbidden
- History / previous version / re-baselining
- “Discovers causal relationships”, “hidden causal structure”
- “Adaptive graphs” / graphs change over time
- Definitive significance from n=5
- Repeating full results tables (cite Table/Figure labels from Stage 47)
- Claiming sparsity controlled outside Los PH1

## Output paths
- `paper/gsl_stage48/sections/discussion.tex`
- `paper/gsl_stage48/stage48_writing_report.md`

## Done when
Discussion subsections written, R1/R2 applied, checklist in report complete, no files outside `Stage48.md` + `paper/gsl_stage48/` modified.
