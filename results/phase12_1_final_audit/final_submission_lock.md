# CTG Fetal Distress Prediction Project — Final Submission Lock

**Status:** `SUBMISSION LOCKED`  
**Phase:** 12.1 — Final Audit & Submission Lock  
**Date of Lock:** September 6, 2026  
**Auditor / Protocol:** Automated & Agentic Verification Protocol (Google DeepMind Antigravity)  
**Governing Rule:** *"Audit, reconcile, document, and lock — do not improve the model."*

---

## 1. Executive Statement of Lock

The intrapartum cardiotocography (CTG) fetal acidemia early-warning prediction project has completed all experimental phases (Phases 1 through 12.1). 

Every reported metric, comparison, confidence interval, and scientific claim has been audited against locked experimental artifacts, underlying prediction files, and reproducible evaluation pipelines. All 15 pre-specified quality, statistical, and methodological audit gates have **PASSED** with zero unresolved numerical discrepancies or unsupported claims.

**The codebase, data artifacts, trained model weights, figures, tables, and thesis synthesis reports are hereby permanently LOCKED for thesis submission and publication.**

---

## 2. Audit Gate Certification (15/15 PASS)

| Gate ID | Gate Name | Verification Criteria | Status | Evidence / Artifact |
| :--- | :--- | :--- | :---: | :--- |
| **Gate 1** | **Cohort Verification** | 547 patients, 110 pH $\le 7.15$, 41 pH $\le 7.05$, 8,517 20-min rolling windows, 4 Hz sampling | **PASS** | `data/processed_signals/`, `phase8_causal_rolling_predictions.csv` |
| **Gate 2** | **Endpoint Verification** | Primary: $\text{pH} \le 7.15$; Secondary: $\text{pH} \le 7.05$. Zero mixing with BDecf or unvalidated composites | **PASS** | `scripts/phase1_preprocessing.py`, `scripts/evaluate_phase6.py` |
| **Gate 3** | **Discrepancy Resolution** | Resolved Phase 2 (0.6593 vs 0.6842) and Phase 3 (0.6701 vs 0.6915). Authoritative AUROCs locked | **PASS** | `results/phase12_1_final_audit/numerical_discrepancy_log.csv` |
| **Gate 4** | **Phase 7 Error Purge** | Discarded historical $+0.0881$ signal-branch artifact 100% eliminated from all reports and figures | **PASS** | `reports/phase12_final_scientific_synthesis.md`, `phase7_1_audit_report.md` |
| **Gate 5** | **Statistical Superiority** | Every claim of "better" or "superior" supported by $B=2,000$ patient bootstrap 95% CI strictly excluding 0 | **PASS** | `results/phase11_benchmark/phase11_statistical_comparison.csv` |
| **Gate 6** | **Causal Integrity** | Strictly causal rolling inference: $\tau \le t$. Zero future CTG leakage in features, states, or evaluation | **PASS** | `results/phase8_causal_audit/causal_verification_report.md` |
| **Gate 7** | **Prior-Art Integrity** | P6 superiority bounded over P1 and P3; P6 acknowledged as comparable to P2 ($p = 0.697$) | **PASS** | `results/phase12_1_final_audit/prior_art_claim_audit.csv` |
| **Gate 8** | **Attribution Integrity** | 78% multidomain gain explicitly documented as path-dependent under pre-specified ordering | **PASS** | `results/phase11_5_attribution/attribution_stepwise_ladder.csv` |
| **Gate 9** | **Horizon Integrity** | Clear distinction between near-delivery discrimination ($0.6872$) and $\ge 30$m early warning ($0.5857$) | **PASS** | `reports/figures_phase12/fig3_warning_horizon_degradation.png` |
| **Gate 10** | **Clinical Governance** | Operational alert policies labeled strictly as research operating points, not validated bedside tools | **PASS** | `reports/phase10_clinical_decision_analysis.md` |
| **Gate 11** | **Artifact Consistency** | 100% numerical consistency across Master CSVs, JSON summaries, Figures 1–8, and MD reports | **PASS** | `results/phase12_1_final_audit/cross_artifact_consistency.csv` |
| **Gate 12** | **Reproducibility** | Full trace from thesis claim $\rightarrow$ report $\rightarrow$ result table $\rightarrow$ experiment script $\rightarrow$ seed/config | **PASS** | `results/phase12_1_final_audit/reproducibility_audit.csv` |
| **Gate 13** | **Exploratory Isolation** | Exploratory analyses (e.g. FIGO, unwindowed runs) quarantined from primary performance tables | **PASS** | `results/phase12_1_final_audit/final_result_provenance.csv` |
| **Gate 14** | **Repository Integrity** | Final Git branch `final_synthesis_models` clean, tracked, and synchronized with locked artifacts | **PASS** | Git Commit Log & Working Tree Audit |
| **Gate 15** | **Claim Matrix** | All 13 core scientific claims mapped directly to empirical evidence and approved claim wording | **PASS** | `results/phase12_1_final_audit/claim_audit_matrix.csv` |

---

## 3. Authoritative Locked Headline Metrics

The following metrics constitute the definitive, immutable performance baseline for this project:

```
====================================================================================================
Phase / Model Name               Delivery AUROC [95% CI]       >=30m AUROC [95% CI]    Locked Status
====================================================================================================
Phase 2 (1D Raw FHR CNN)         0.6593 [0.6011, 0.7180]                 --             LOCKED
Phase 3 (Fixed P90 Pooling)      0.6701 [0.6111, 0.7259]                 --             LOCKED
Phase 4 (Logit Prior Mod.)       0.7361 [0.6793, 0.7878]                 --             LOCKED
Phase 6 (Continuous Huber)       0.7426 [0.6876, 0.7931]                 --             LOCKED
Phase 8 (Rolling Causal Huber)   0.6941 [0.6382, 0.7490]       0.5699 [0.5081, 0.6312]  LOCKED
----------------------------------------------------------------------------------------------------
Phase 11 Head-to-Head Benchmark (Common Cohort, N=547):
  P1 (Compact CTG Baseline)      0.5090 [0.4501, 0.5703]       0.5727 [0.5102, 0.6348]  LOCKED
  P2 (Sequential Event Baseline) 0.6774 [0.6198, 0.7331]       0.5983 [0.5372, 0.6589]  LOCKED
  P3 (Snapshot Baseline)         0.5976 [0.5371, 0.6569]       0.5461 [0.4851, 0.6062]  LOCKED
  P4 (Snapshot + State)          0.5945 [0.5342, 0.6538]       0.5459 [0.4849, 0.6059]  LOCKED
  P5 (Snapshot + Trajectory)     0.6142 [0.5541, 0.6732]       0.5355 [0.4745, 0.5958]  LOCKED
  P6 (Full Proposed Framework)   0.6872 [0.6301, 0.7429]       0.5857 [0.5248, 0.6459]  LOCKED
====================================================================================================
```

---

## 4. Discrepancy Resolution Summary

* **Phase 2 & Phase 3 Audit Findings**:
  - Investigation confirmed that `0.684218` and `0.691559` appearing in the preliminary Phase 12 synthesis were individual patient prediction array values in intermediate files (`phase3_results.json` and `phase4_results.json`) that were mistakenly transcribed as dataset-level AUROCs.
  - The true, locked 5-fold cross-validated AUROCs are **0.6593** (Phase 2) and **0.6701** (Phase 3).
  - All Phase 12 synthesis tables, summary JSON files, markdown reports, and figures (Figure 2) have been updated and synchronized with the authoritative values.

---

## 5. Submission Freezing Directives

1. **Zero New Modeling**: No retraining, hyperparameter adjustments, temporal context changes, or architecture alterations are permitted.
2. **Deterministic Reproducibility**: All evaluation runs must be invocable via `python scripts/phase12_synthesis.py` and `python scripts/phase12_figures.py` reproducing the exact audited numerical tables and figures.
3. **Scientific Traceability**: Any future reader or reviewer can trace every sentence in the thesis back to `results/phase12_1_final_audit/final_result_provenance.csv`.

---

**Certified and Locked by:** Antigravity AI & Research Team  
**Repository Branch:** `final_synthesis_models`  
**Workspace:** `e:\Maha\CTG-Fetal-Distress-Prediction`
