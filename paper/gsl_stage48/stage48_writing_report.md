# Stage 48 — Writing Report: Section 6 Discussion

**Date:** 2026-09-10  
**Scope:** `discussion.tex` + this report only.

---

## 1. Files

| Path | Action |
|------|--------|
| `Stage48.md` | Created (stage brief) |
| `paper/gsl_stage48/sections/discussion.tex` | Created — §6.1–6.3 |
| `paper/gsl_stage48/stage48_writing_report.md` | Created |

---

## 2. Structure

| Subsection | Label | Role |
|------------|-------|------|
| 6.1 | `sec:disc-dense` | Oversmoothing / harmful-default reading of §5.1–5.2 |
| 6.2 | `sec:disc-consumption` | Dissociation; Weighted≈Fixed; Mix; cGSL/GSL aggregation compatibility |
| 6.3 | `sec:disc-interpretation` | Static statistical dependency; conditional benefit; SZ + 15-min boundary |

---

## 3. Numbers cited (all already in Stage 47 Results)

| Value | Where in Discussion | Source section |
|-------|---------------------|----------------|
| Physical dense Los (~10³ entries) | §6.1 | Stage 45.1 / Stage 41 graph stats |
| Physical vs NoSpatial large gap | §6.1 | §5.1 |
| GSL/cGSL intermediate, never beat NoSpatial | §6.1 | §5.2 |
| Sparsity controls Los PH1 | §6.1 | §5.3 |
| GCN-MultiGSL 9.78 vs T-GCN-MultiGSL 4.84 | §6.2 | §5.4 |
| Union 28/30 slots | §6.2 | §3.3 / §5.4 |
| Weighted ≈ Fixed (≤0.01 RMSE) | §6.2 | §5.4–5.5 |
| Mix − MultiGSL 0.35–0.40, 5/5 | §6.2 | §5.5 |
| cGSL 5.76 vs GSL 7.83; T-GCN 5.82 vs 5.86 | §6.2 | §5.2 |
| SZ 2-edge multi-lag; null | §6.3 | §5.6 |
| 15-min Mix vs NoSpatial −27.4%…−16.1% | §6.3 | §5.6 / Table los15 |

No new experimental numbers. Improvements phrased as relative RMSE reduction where used.

---

## 4. Claim-policy compliance

| Rule | Status |
|------|--------|
| R1: consumption as major determinant, not sole causal locus | ✅ explicit backbone-confound sentence in §6.2 |
| R2: relative RMSE reduction; Los Mix headline context | ✅ |
| No causal traffic language | ✅ “statistical dependency”, “consistent with” |
| No “adaptive” / graphs change over time | ✅ gate changes *use* only |
| No history language | ✅ |
| No definitive n=5 significance | ✅ |
| Sparsity scope Los PH1 | ✅ |
| 15-min as resolution variant only | ✅ |
| cGSL asymmetry as aggregation-compatibility observation | ✅ (no temporal-DAG apparatus) |
| No full table repetition | ✅ cites Table/Figure labels only |

---

## 5. Open items

1. `Figure~\ref{fig:perseed}` / `tab:los15` / `tab:tgcn-main` resolve only when Results floats are in the same compilation unit.
2. Introduction tie-back is textual (`proximity-versus-dependency`); convert to `\ref{sec:intro}` at assembly.
3. Discussion does **not** cover Limitations (Stage 49) or future work (Conclusion).

---

## 6. Verdict

**STAGE 48 COMPLETE — READY FOR LIMITATIONS (§7) AND FRAMING SECTIONS.**
