# Model 3 + Parity Multimodal Hybrid Experiment

This directory contains the complete artifact suite, experimental results, leakage audit, and scientific documentation for the **Model 3 + Parity Late-Fusion Hybrid** investigation.

## Investigation Overview
- **Objective:** Evaluate whether fusing Model 3 (CTG temporal-position causal attention risk sequence) with maternal parity (admission-level clinical covariate) provides incremental predictive value beyond Model 3, Parity Fusion, P6, Max pooling, and P90 pooling.
- **Cohort:** Cleaned CTU-UHB intrapartum cohort ($N=547$ patients, 8,517 causal windows).
- **Primary Endpoint:** Umbilical arterial $\text{pH} \le 7.15$ ($N=110$ positives, prevalence 20.1%).
- **Evaluation Protocols:** Patient-grouped 5-fold cross-validation and 83-patient held-out internal test partition.
- **Classification Status:** **Promising but Inconclusive (Internal Research Candidate)** — delivery gain over Model 3 is positive ($\Delta = +0.0108$) but not CV-significant ($p = 0.172$); early-warning gains over Model 3 at $\ge 10$m, $\ge 20$m, and $\ge 30$m are statistically significant ($p < 0.05$); operational false-alert rate is the lowest across all models (65.2% vs 69.3% for Model 3 and 83.1% for P6).

---

## Directory Index

| File | Description |
| :--- | :--- |
| [`config.yaml`](config.yaml) | Machine-readable experiment configuration with all locked parameters |
| [`data_feature_audit.json`](data_feature_audit.json) | Machine-readable data audit verifying cohort counts, invariants, and horizons |
| [`data_feature_audit.md`](data_feature_audit.md) | Narrative report of Experiment E0 data audit |
| [`reference_reproduction.json`](reference_reproduction.json) | Exact verification of reference models (P6, Max, P90, Model 3, Parity Fusion) |
| [`reference_reproduction.md`](reference_reproduction.md) | Formatted table comparing reproduced vs. locked historical AUROC numbers |
| [`patient_level_predictions.csv`](patient_level_predictions.csv) | Full patient-level prediction table matching Section 18 schema ($N=547$) |
| [`fold_assignments.csv`](fold_assignments.csv) | Patient-level canonical fold assignments and internal test flags |
| [`fusion_coefficients.csv`](fusion_coefficients.csv) | Fitted logistic fusion parameters ($\beta_0, \beta_1, \beta_2$) across folds |
| [`delivery_metrics.csv`](delivery_metrics.csv) | Delivery-horizon discrimination metrics, per-fold AUROC, paired tests |
| [`early_warning_metrics.csv`](early_warning_metrics.csv) | Early-warning performance at $\ge 10$m, $\ge 20$m, and $\ge 30$m |
| [`operational_metrics.csv`](operational_metrics.csv) | Operational alert metrics at 80% delivery target sensitivity |
| [`calibration_metrics.csv`](calibration_metrics.csv) | Brier score, calibration intercept, calibration slope, risk distributions |
| [`ablation_table.csv`](ablation_table.csv) | Systematic ablation ladder (inputs, transformations, interactions) |
| [`incremental_value_analysis.csv`](incremental_value_analysis.csv) | Bidirectional incremental value ($\Delta_{\text{parity} \mid \text{M3}}$ and $\Delta_{\text{M3} \mid \text{parity}}$) |
| [`prediction_correlation.csv`](prediction_correlation.csv) | Pearson and Spearman prediction correlations across model pairs |
| [`error_disagreement_analysis.csv`](error_disagreement_analysis.csv) | Net reclassification and disagreement cases (corrected vs. harmed) |
| [`duration_sensitivity.csv`](duration_sensitivity.csv) | Observation-duration correlations and window-count tertile analysis |
| [`robustness_results.csv`](robustness_results.csv) | Random bootstrap seed sensitivity and 5 independent fold-resplits |
| [`permutation_control_results.csv`](permutation_control_results.csv) | 30-replicate patient-level parity permutation control results |
| [`leakage_audit.json`](leakage_audit.json) | Formal leakage verification and data boundary sign-off |
| [`test_results.txt`](test_results.txt) | Automated assertion suite results (27 tests, 0 failures) |
| [`environment.txt`](environment.txt) | Python, PyTorch, and CUDA environment specifications |
| [`git_commit.txt`](git_commit.txt) | Git commit hash for complete audit traceability |
| [`model_card.md`](model_card.md) | Standardized Model Card document |
| [`limitations.md`](limitations.md) | Detailed limitation disclosure per Section 20.5 |
| [`final_summary.md`](final_summary.md) | Full final scientific synthesis report per Section 20 |

---

## Reproduction Commands

To reproduce the audit, pipeline execution, and automated assertions from the repository root:

```bash
# 1. Run Data and Feature Audit (E0)
.\venv\Scripts\python.exe scripts/model3_parity_hybrid/data_audit.py

# 2. Run Full Hybrid Pipeline (E1 - E4, Robustness, Permutations)
.\venv\Scripts\python.exe scripts/model3_parity_hybrid/hybrid_engine.py

# 3. Run Automated Assertion Suite
.\venv\Scripts\python.exe scripts/model3_parity_hybrid/test_assertions.py
```
