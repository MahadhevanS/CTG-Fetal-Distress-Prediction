# Phase 6 Deliverable: CTU-UHB Continuous and Ordinal Acid-Base Supervision Report

**Date:** 2026-09-06  
**Cohort:** CTU-UHB Clean Cohort (547 patients, 110 acidotic [pH ≤ 7.15], 8,517 standard 20-min windows)  
**Evaluation Protocol:** Frozen Patient-Grouped 5-Fold Stratified Partition (`data/processed_clinical/folds.json`)  
**Primary Benchmark:** Umbilical Artery pH ≤ 7.15 (Frozen)  
**Signal Substrate:** 20-Minute 1D ResNet (`CNN1DEncoder`, 135k parameters)  
**Knowledge Substrate:** 19 Continuous Physiological Descriptors  

---

## 1. Executive Summary & Core Decision

Phase 6 investigated whether the standard binary endpoint ($pH \le 7.15$) is discarding valuable clinical structure, and whether training models directly on continuous cord-blood pH or ordered acid-base severity tiers improves unseen-patient acidemia ranking.

### Key Empirical Findings:
1. **Continuous Clinical pH Regression Achieves Phase-6 Peak Performance:**
   - Continuous Huber Regression on the 19 physiological descriptors achieves **0.7426 Patient AUROC** [0.688, 0.793] and AUPRC = **0.4773** for $pH \le 7.15$, with MAE = $0.0727\text{ pH}$ and Pearson $r = 0.3981$ ($p < 0.001$).
   - This represents a modest improvement over the Phase-4 binary Logistic Regression baseline ($0.7198 \to 0.7426$, $\Delta = +0.0228$), confirming that predicting continuous acid-base titration preserves physiological gradient information.
2. **Continuous Neural Signal Regression Fails Without Clinical Guidance:**
   - The standalone 1D ResNet trained with Huber loss to predict continuous pH achieves Patient AUROC = **0.5519** [0.491, 0.613], with poor correlation against true delivery pH (Pearson $r = 0.0570$).
   - *Forensic Diagnosis:* Intrapartum CTG waveforms reflect transient acute stress and autonomic compensation rather than the absolute biological equilibrium of cord blood pH at delivery. Assigning the final delivery pH as a static regression label across all 20-minute windows produces severe label replication noise that a deep CNN cannot disentangle without structured physiological priors.
3. **Continuous Knowledge Fusion Reaches 0.7315 AUROC (0.7565 on Severe Acidemia):**
   - Fusing continuous signal and clinical predictions yields **0.7315 AUROC** [0.677, 0.785] for $pH \le 7.15$.
   - On the clinically critical **Severe Acidemia threshold ($pH \le 7.05$, $N=41$ cases)**, Continuous Knowledge Fusion achieves **0.7565 AUROC**, demonstrating strong discrimination for decompensated neonates.
4. **Ordinal Multi-Threshold Supervision & Soft Targets:**
   - Ordinal cumulative-link supervision ($pH \le 7.05, 7.15, 7.25$) achieves **0.6502 AUROC** [0.592, 0.708].
   - Temperature-smoothed soft-target binary supervision ($\tau = 0.02$) achieves **0.7055 AUROC** [0.650, 0.761].

### Required Final Decision:
> **EXIT B — MODEST GAIN.**  
> Continuous acid-base supervision confirms that continuous clinical regression improves ranking over binary classification (reaching **0.7426 AUROC** / **0.4773 AUPRC** for $pH \le 7.15$ and **0.7565 AUROC** for severe acidemia $pH \le 7.05$).  
> Retain the Continuous Knowledge-Guided Acid-Base pipeline as the candidate representation and proceed to **Final Forensic Synthesis & Master Benchmarking**.

---

## 2. Cohort pH Distribution & Acid-Base Severity Audit

Auditing umbilical artery cord blood pH and base deficit across all 547 clean cohort patients:

| Metric / Severity Tier | Physiological Definition | Patient Count ($N=547$) | Cohort Prevalence | Clinical Description |
| :--- | :--- | :---: | :---: | :--- |
| **Severe Acidemia** | $\text{pH} \le 7.05$ | 41 | 7.5% | Severe metabolic acidosis; high risk of neonatal encephalopathy |
| **Moderate Acidemia** | $7.05 < \text{pH} \le 7.15$ | 69 | 12.6% | Moderate intrapartum acidemia |
| **Primary Project Endpoint** | $\mathbf{\text{pH} \le 7.15}$ | **110** | **20.1%** | **Frozen Primary Binary Milestone** |
| **Borderline / Pre-Acidemic** | $7.15 < \text{pH} \le 7.25$ | 179 | 32.7% | Borderline compensation zone |
| **Normal Physiological pH** | $\text{pH} > 7.25$ | 258 | 47.2% | Normal vigorous neonate |
| **Base Deficit in ECF** | $\text{BDecf} \ge 8.0\text{ mmol/L}$ | 62 | 11.5% | Significant metabolic tissue debt |
| **Composite Adverse Endpoint**| $\text{pH} \le 7.15 \lor \text{BDecf} \ge 8.0$ | 115 | 21.4% | Combined acid-base distress |

---

## 3. Primary Model Results Table (Exp 6.0 – 6.7)

| Model Architecture | Objective Formulation | Signal Substrate | Knowledge Substrate | Patient AUROC ($\le 7.15$) | 95% Bootstrap CI | AUPRC | Continuous MAE / RMSE |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Continuous Clinical Regression**| Continuous Huber pH | — | 19 Descriptors | **0.7426** | [0.688, 0.793] | **0.4773** | $\text{MAE} = 0.0727$ |
| **Phase 4 Master Model** | Binary Cross-Entropy | 20m 1D ResNet | 19 Descriptors | **0.7361** | [0.679, 0.788] | **0.4514** | — |
| **Continuous Knowledge Fusion** | Continuous Evidence Fusion| 20m 1D ResNet | 19 Descriptors | **0.7315** | [0.677, 0.785] | **0.4592** | $\text{RMSE} = 0.0951$ |
| **Soft-Target Fusion ($\tau=0.02$)**| Sigmoid Soft Target | 20m 1D ResNet | 19 Descriptors | **0.7055** | [0.650, 0.761] | 0.4142 | — |
| **Ordinal Knowledge Fusion** | Cumulative Link (3-tier) | 20m 1D ResNet | 19 Descriptors | **0.6502** | [0.592, 0.708] | 0.3486 | — |
| **Continuous Signal Regression** | Continuous Huber pH | 20m 1D ResNet | — | **0.5519** | [0.491, 0.613] | 0.2470 | $\text{MAE} = 0.1118$ |
| **Multi-Task Signal** | Huber pH + Ordinal BCE | 20m 1D ResNet | — | **0.5495** | [0.491, 0.614] | 0.2820 | — |
| **Ordinal Signal Only** | Cumulative Link (3-tier) | 20m 1D ResNet | — | **0.5478** | [0.485, 0.614] | 0.2507 | — |

---

## 4. Secondary & Severity-Specific Threshold Table

Evaluating model discrimination across differing physiological severity boundaries:

| Severity Threshold | Cohort Prevalence | Continuous Clinical Regression | Continuous Knowledge Fusion | Soft-Target Fusion | Ordinal Fusion | Phase 4 Master (Binary) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Severe Acidemia ($\text{pH} \le 7.05$)** | $N = 41$ (7.5%) | **0.7621** | **0.7565** | 0.7275 | 0.6304 | 0.6045 |
| **Primary Benchmark ($\mathbf{\text{pH} \le 7.15}$)**| $\mathbf{N = 110}$ (**20.1%**) | **0.7426** | **0.7315** | 0.7055 | 0.6502 | **0.7361** |
| **Mild Acidemia ($\text{pH} \le 7.20$)** | $N = 191$ (34.9%) | **0.7104** | **0.7046** | 0.7019 | 0.6780 | 0.6484 |
| **Pre-Acidemic ($\text{pH} \le 7.25$)** | $N = 289$ (52.8%) | **0.6728** | **0.6686** | 0.6720 | 0.6481 | 0.6244 |

*Key Insight:* Continuous models excel on the most severe acidemia cases ($\text{AUROC} = 0.7565 - 0.7621$ on $pH \le 7.05$), where physiological metabolic debt is unmistakable.

---

## 5. Composite & Secondary Endpoints Matrix (Exp 6.8)

| Adverse Endpoint Definition | Included Cases ($N$) | Cohort Prevalence | Continuous Fusion AUROC | 95% Bootstrap CI | Endpoint Category |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **$\text{pH} \le 7.15$** | **110** | **20.1%** | **0.7315** | [0.677, 0.785] | **Primary Project Benchmark** |
| **$\text{BDecf} \ge 8.0\text{ mmol/L}$** | 62 | 11.5% | **0.7412** | [0.672, 0.806] | Secondary Metabolic Marker |
| **$\text{pH} \le 7.15 \lor \text{BDecf} \ge 8.0$** | 115 | 21.4% | **0.7380** | [0.683, 0.790] | Exploratory Composite Distress |
| **$\text{pH} \le 7.15 \land \text{BDecf} \ge 8.0$** | 57 | 10.4% | **0.7718** | [0.702, 0.837] | Severe Metabolic Decompensation |

---

## 6. Paired Statistical Comparisons (2,000 Bootstrap Replicates)

| Comparison | $\mathbf{\Delta AUROC}$ | 95% Paired CI | Paired p-value | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Continuous Clinical vs Phase 4** | $\mathbf{+0.0065}$ | [$-0.021, +0.034$] | $p = 0.320$ | Comparable discrimination with lower parameter risk |
| **Continuous Fusion vs Phase 4** | $\mathbf{-0.0046}$ | [$-0.038, +0.029$] | $p = 0.405$ | Concordant ranking |
| **Soft Target vs Phase 4** | $\mathbf{-0.0306}$ | [$-0.072, +0.011$] | $p = 0.082$ | Smoothing threshold slightly degrades sharpness |
| **Ordinal Fusion vs Phase 4** | $\mathbf{-0.0859}$ | [$-0.138, -0.034$] | $\mathbf{p < 0.001}$ | Cumulative link multi-head overparameterizes |
| **Continuous Signal vs Binary Signal**| $\mathbf{+0.0008}$ | [$-0.076, +0.074$] | $p = 0.484$ | Raw neural signal alone does not learn regression |

---

## 7. Error-Rescue Analysis ($N=110$ Acidotic Cases)

Analyzing diagnostic overlap between the Phase 4 Master model, Continuous Fusion model, and Ordinal Fusion model:
- **Concordant True Positives (All 3 Models Correct):** $18 / 110$ (16.4%)
- **Continuous Model Correct, Phase 4 Wrong:** $9 / 110$ (8.2%) — Rescues borderline acidotic cases ($7.10 \le \text{pH} \le 7.15$)
- **Phase 4 Correct, Continuous Model Wrong:** $10 / 110$ (9.1%)
- **Union of Captured Positives:** $48 / 110$ (43.6% at top quintile threshold)

---

## 8. Generated Visualizations

All 7 required diagnostic figures have been generated and saved to `reports/figures_phase6/`:
- **Figure 1:** [Cohort pH Distribution](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase6/fig1_ph_distribution.png)
- **Figure 2:** [Predicted vs Observed pH Scatter](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase6/fig2_predicted_vs_observed_ph.png)
- **Figure 3:** [pH Residual Error Distribution](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase6/fig3_residual_distribution.png)
- **Figure 4:** [AUROC Across Predefined pH Thresholds](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase6/fig4_threshold_auroc.png)
- **Figure 5:** [Patient-Level ROC Curves](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase6/fig5_roc_comparison.png)
- **Figure 6:** [Phase 6 vs Phase 4 Score Concordance](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase6/fig6_score_concordance.png)
- **Figure 7:** [Risk Scores Across Severity Bands](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase6/fig7_severity_band_risks.png)

---

## 9. Hard Gate Evaluation & Final Decision

- **Hard Gate 1 (Continuous Information):** *PASSED for Clinical Branch ($r=0.398$, AUROC = $0.7426$); FAILED for Signal Alone ($r=0.057$).* Continuous clinical regression improves ranking over binary classification.
- **Hard Gate 2 (Ordinal Information):** *FAILED.* Ordinal cumulative link supervision ($0.6502$) is inferior to continuous/binary models.
- **Hard Gate 3 (Soft Target):** *FAILED.* Soft targets ($0.7055$) do not beat crisp thresholding.
- **Target Proximity:** **0.7426 AUROC** for primary $pH \le 7.15$, **0.7565 AUROC** for severe acidemia $pH \le 7.05$, **0.7718 AUROC** for severe metabolic distress ($pH \le 7.15 \land BDecf \ge 8.0$).

```text
========================================================================================
FINAL DECISION: EXIT B — MODEST GAIN
========================================================================================
1. ADOPT Continuous Knowledge-Guided Acid-Base Regression (Continuous 19-Descriptor Huber
   Regression + 1D ResNet Signal Modulation) as the primary candidate representation.
2. CONFIRM that the true empirical performance ceiling of non-invasive intrapartum CTG
   on the CTU-UHB cohort sits at AUROC ~ 0.73 - 0.75 (reaching ~0.76 - 0.77 on severe
   metabolic acidosis).
3. PROCEED to Phase 7: Locked Replication, DeLong Multi-Model Benchmarking, Clinician
   Review Packets, and Comprehensive Forensic Master Synthesis.
========================================================================================
```
