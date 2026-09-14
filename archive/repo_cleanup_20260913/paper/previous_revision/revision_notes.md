# Revision Notes — Point-by-Point Reviewer Response Map

## Reviewer 1

### #1: Bibliometric analysis is underused
- **Changed:** Used one concise bibliometric observation in Introduction (para 2)
- **Location:** sections/introduction.tex, lines 5-8
- **Evidence:** Appendix A (appendix/bibliometric.tex)
- **New experiment:** No

### #2: GCN/T-GCN background too long
- **Changed:** Compressed from ~140 lines to ~35 lines
- **Location:** sections/background.tex
- **Evidence:** Existing knowledge
- **New experiment:** No

### #3: A→W notation switch unexplained
- **Changed:** Added explicit convention note when W is first introduced
- **Location:** sections/method.tex, Section 3.2
- **Evidence:** N/A
- **New experiment:** No

### #4: Temporal interpretation asserted, not demonstrated
- **Changed:** Introduced explicit multi-lag DAGMA formulation (Section 3.3)
- **Location:** sections/method.tex, Section 3.3
- **Evidence:** Stage 25D/26 results (Table lag_stats in results.tex)
- **New experiment:** No — uses existing Stage 26 DAGMA matrices

### #5: Sparsity/oversmoothing
- **Changed:** Added explicit oversmoothing analysis (Section 5.1)
- **Location:** sections/results.tex, Table oversmoothing
- **Evidence:** Stage 24-25 results
- **New experiment:** No

### #6: No multiple seeds
- **Changed:** Added 5-seed validation table (Section 5.3)
- **Location:** sections/results.tex, Table multiseed
- **Evidence:** Stage 26 Validation Experiment A
- **New experiment:** No — uses existing seed 42-46 results

### #7: Results only to PH=4
- **Changed:** Added multi-PH table (Section 5.6)
- **Location:** sections/results.tex, Table multiph
- **Evidence:** Stage 26 multi-PH results
- **New experiment:** No

### #8: Section 5 repetitive
- **Changed:** Replaced old Section 5 with new Results + Discussion structure
- **Location:** sections/results.tex, sections/discussion.tex
- **Evidence:** N/A
- **New experiment:** No

### #9: No limitations
- **Changed:** Added explicit Limitations section (Section 7)
- **Location:** sections/limitations.tex
- **Evidence:** N/A
- **New experiment:** No

### #10: Metric definitions too long
- **Changed:** Compressed to 2-line combined equation
- **Location:** sections/experiments.tex
- **Evidence:** N/A
- **New experiment:** No

### #11: Dense convergence plots
- **Changed:** Moved to appendix (appendix/convergence.tex)
- **Location:** appendix/convergence.tex
- **Evidence:** Existing figures
- **New experiment:** No

### #12: Citation style
- **Changed:** Will standardize in final compilation pass
- **Location:** Throughout
- **New experiment:** No

### Questions:

**Physical vs learned graph visualization:** Will generate from existing DAGMA matrices in a future step.

**Predicted vs actual time series:** Left as future work — would require inference from saved models.

**Time-varying graph plan:** Described as future work in Conclusion and Limitations.

**Direct evidence for temporal DAG:** Provided by multi-lag DAGMA formulation (Section 3.3) and lag block statistics (Table lag_stats).

---

## Reviewer 2

### #1: Abstract RMSE inconsistent
- **Changed:** Wrote new abstract with correct numbers (14.9% Los-loop)
- **Location:** sections/abstract.tex
- **Evidence:** Stage 26 Validation
- **New experiment:** No

### #2: "Hidden causal structure" unsupported
- **Changed:** Removed ALL causal claims throughout. Only mention of "causal" is in Limitations (explicitly disclaiming it).
- **Location:** All sections
- **Evidence:** N/A
- **New experiment:** No

### #3: Static vs adaptive graph inconsistency
- **Changed:** Explicitly distinguished: static learned graph, GRU temporal modeling, multi-lag graph architecture, adaptive gating
- **Location:** sections/method.tex, Section 3.4
- **Evidence:** Code inspection
- **New experiment:** No

### #4: cGSL definition too late
- **Changed:** cGSL now only appears in Appendix (original results). Main text focuses on multi-lag approach.
- **Location:** appendix/original_gsl_results.tex
- **Evidence:** N/A
- **New experiment:** No

### #5: Convergence plots hard to read
- **Changed:** Moved to appendix
- **Location:** appendix/convergence.tex
- **Evidence:** Existing figures
- **New experiment:** No

### #6: Typo "avergae"
- **Changed:** Removed (old text no longer in main text)
- **Location:** N/A
- **New experiment:** No
