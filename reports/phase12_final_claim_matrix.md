# Phase 12 — Final Master Claim Matrix & Language Governance

## CTU-UHB Fetal Acidemia Prediction & Early-Warning Project

### 1. Overview & Purpose

To ensure strict scientific integrity, prevent claim inflation, and establish unambiguous vocabulary for publication and thesis defense, this document establishes the **mandatory claim governance rules and claim matrix**.

---

## 2. Mandatory Vocabulary & Claim Rules

| Term / Phrasing | Permissible Context & Criteria | Prohibited Usage |
| :--- | :--- | :--- |
| **"Outperformed" / "Superior"** | Use **ONLY** when $\Delta \text{AUROC} > 0$ with $95\%$ bootstrap CI strictly excluding zero ($p < 0.05$). | Never use for numerical point differences where CI overlaps zero (e.g., P6 vs P2). |
| **"Higher Point Estimate"** | Permissible when point estimate is higher but CI includes zero; must explicitly state non-significance. | Do not imply statistical superiority or generalizable advantage. |
| **"Comparable" / "Statistically Indistinguishable"** | Mandatory when $95\%$ bootstrap CI includes zero (e.g., P6 vs P2 across horizons). | Do not claim parity as proof of equivalence; report confidence interval bounds. |
| **"Associated with"** | Mandatory for observational relationships (e.g., progression vs. reversal and subsequent acidemia risk). | Never use causal terms such as "causes", "prevents", or "cures". |
| **"Biological Compensation / Recovery"** | **STRICTLY PROHIBITED** from retrospective CTU-UHB data. | Do not interpret trajectory reversals as physiological homeostasis mechanisms. |
| **"Retrospective Warning Lead Time"** | Permissible when reporting simulated first alert timing before delivery. | Never claim "proved clinical effectiveness" or "reduced neonatal morbidity". |

---

## 3. Final Master Claim Matrix

| Claim Item | Experimental Source | Statistical Evidence ($B=2,000$) | Allowed Formulation | Prohibited Formulation | Thesis Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **P6 vs P1 (Compact CTG)** | Phase 11 Head-to-Head Benchmark | $\Delta \text{AUROC} = +0.1801$ [$+0.1111, +0.2475$], $p < 0.001$ | "P6 significantly outperformed the compact CTG baseline near delivery." | "P6 is superior to all published compact algorithms across all data." | **CONFIRMED** |
| **P6 vs P3 (Snapshot Huber)** | Phase 11 Head-to-Head Benchmark | $\Delta \text{AUROC} = +0.0896$ [$+0.0349, +0.1470$], $p < 0.001$ | "P6 significantly outperformed the locked snapshot baseline near delivery." | "Snapshot models are useless in clinical obstetrics." | **CONFIRMED** |
| **P5 vs P3 (Trajectory Increment)** | Phase 11 & Phase 11.5 LOCO Ablation | $\Delta \text{AUROC} = +0.0165$ [$+0.0006, +0.0319$], $p = 0.041$ | "Temporal trajectory dynamics provide statistically significant incremental discrimination beyond snapshot risk." | "Trajectory dynamics are the sole cause of total system accuracy." | **CONFIRMED** |
| **Multidomain Severity Fusion** | Phase 11.5 Component Ladder (Step 6) | $\Delta \text{AUROC} = +0.0676$ [$+0.0192, +0.1292$], $p < 0.001$ | "Under the pre-specified ladder, multidomain physiological fusion accounted for ~78% of the total system gain." | "78% of true causal physiological risk resides in multidomain features." | **CONFIRMED** |
| **Progression vs Reversal Risk** | Phase 11.5-D Matched-State Analysis | $\text{Prevalence}: 23.5\% \text{ vs } 20.5\%$ ($\text{OR} = 1.19$) | "Within matched states, progressing trajectories were associated with higher subsequent acidemia rates." | "Reversal trajectories prove fetal biological recovery and protective compensation." | **CONFIRMED (Associative)** |
| **P6 vs P2 (Sequential Event Baseline)** | Phase 11 Multi-Horizon Evaluation | $\Delta = +0.0098$ ($p=0.697$) at delivery; $\Delta = -0.0126$ ($p=0.485$) at $\ge 30$m | "P6 is statistically comparable to P2, exhibiting complementary temporal operating regimes across horizons." | "P6 outperformed the Vargas-Calixto sequential baseline." | **NOT CONFIRMED (Comparable)** |
| **Early Warning at $\ge 30$ Minutes** | Phase 8, 11, 11.5 Multi-Horizon Suite | $\text{AUROC}_{\ge 30\text{m}} = 0.54 - 0.59$ across all models | "Discrimination is constrained at $\ge 30$ min due to compensated autonomic baseline before acute decompensation." | "The framework reliably predicts acidemia hours before delivery." | **BOUNDED / REJECTED** |
| **Operational Warning Lead Time** | Phase 11.5-G & 11.5-H Threshold Sweep | $17.5$ min median at locked threshold; $16.0$ min at matched $\text{FAR}=0.50$/hr | "P6 achieved a longer retrospective warning lead time than prior-art baselines at comparable false alert rates." | "The framework proved clinical utility and reduced neonatal brain injury in trials." | **SUPPORTED (Operational)** |
| **Prospective Clinical Utility** | Entire Project Retrospective Scope | Retrospective cohort simulation ($N=547$) | "Retrospective operational simulation indicates favorable warning characteristics; prospective trials needed." | "The model is clinically validated and ready for unsupervised hospital deployment." | **STRICTLY PROHIBITED** |

---

## 4. Governance Summary

This claim matrix governs all doctoral dissertation chapters, journal manuscripts, conference presentations, and oral viva responses. Any claim not marked as **CONFIRMED** or **SUPPORTED** in this matrix is strictly excluded from the thesis conclusions.
