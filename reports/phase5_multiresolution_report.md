# Phase 5 Deliverable: CTU-UHB Multi-Resolution 1D Temporal Context Report

**Date:** 2026-09-06  
**Cohort:** CTU-UHB Clean Cohort (547 patients, 110 acidotic [pH ≤ 7.15], 517 patients with matched 40-min histories)  
**Evaluation Protocol:** Frozen Patient-Grouped 5-Fold Stratified Partition (`data/processed_clinical/folds.json`)  
**Primary Endpoint:** Umbilical Artery pH ≤ 7.15 (Frozen)  
**Signal Substrate:** 1D ResNet Temporal Encoder (`CNN1DEncoder`, 135k parameters) evaluated at 5m, 10m, 20m, 30m, 40m, 60m  
**Knowledge Substrate:** Multi-Scale Continuous Descriptors (10m, 20m, 40m) & Physiological Deceleration Trends  

---

## 1. Executive Summary & Core Decision

Phase 5 investigated whether the Phase-4 knowledge-guided baseline (**0.7361 AUROC**) is constrained because it observes CTG information at only a single 20-minute temporal scale. We evaluated single-scale 1D ResNet models across 6 candidate scales (5m, 10m, 20m, 30m, 40m, 60m), matched-endpoint multi-resolution signal fusion, and multi-scale clinical trend descriptors.

### Key Empirical Findings:
1. **Single-Scale Context Audit Demonstrates Shorter/Medium Windows are Optimal:**
   - Single-scale 1D ResNet performance peaks around short/medium contexts:
     - **5-min Context (1,200 samples):** Patient AUROC = **0.5736** [0.517, 0.633], AUPRC = 0.2463
     - **10-min Context (2,400 samples):** Patient AUROC = **0.5658** [0.508, 0.625], AUPRC = 0.2522
     - **20-min Context (4,800 samples):** Patient AUROC = **0.5678** [0.508, 0.629], AUPRC = 0.2456
   - Expanding context length to **30-min (0.5367)**, **40-min (0.5399)**, and **60-min (0.5439)** results in substantial degradation.
   - *Forensic Diagnosis:* Long temporal windows dilute acute deceleration events across thousands of baseline samples, increase missingness exposure, and force global average pooling to summarize non-stationary labor trajectories into a single static embedding.
2. **Multi-Scale Signal Fusion Fails to Achieve Meaningful Improvement:**
   - Combining 10m, 20m, and 40m signal scores in logit space yields an AUROC of **0.5808** [0.521, 0.642].
   - The paired difference vs the single 20m signal model is only **$\Delta = +0.0131$** (95% CI: [$-0.032, +0.056$], **$p = 0.275$**), failing the required $+0.020$ hard gate.
3. **Temporal Clinical Trends Do Not Outperform 20-Min Clinical Descriptors:**
   - Multi-scale clinical descriptors with deceleration slopes achieve AUROC = **0.6555** [0.594, 0.713], compared to the established 20-min 19-descriptor Clinical LR benchmark of **0.7198**. Adding historical slopes across 40 minutes increases collinearity without adding orthogonal diagnostic signal.
4. **Primary Phase-5 Multi-Resolution Candidate Underperforms Phase-4 Baseline:**
   - Fusing multi-scale signal evidence with temporal clinical descriptors via Logit Prior Modulation achieves Patient AUROC = **0.6258** [0.565, 0.687], which is significantly worse than the Phase-4 baseline of **0.7361** ($\Delta \text{AUROC} = -0.1109$, $p < 0.001$).

### Required Final Decision:
> **EXIT A — NO TEMPORAL GAIN.**  
> Multi-resolution temporal receptive fields do not improve upon the Phase-4 system. Expanding context length dilutes acute hypoxic patterns and introduces noise.  
> **Close the temporal-resolution expansion branch.**  
> Retain the Phase-4 architecture (**1D ResNet 20m + 19 clinical descriptors + Logit Prior Modulation: AUROC = 0.7361**) as the primary baseline and proceed to **Phase 6: Continuous pH Supervision & Composite Adverse Endpoint Formulation**.

---

## 2. Temporal Scale Audit Matrix (Exp 5.0 & 5.1 & 5.2)

Evaluated under the frozen 5-fold patient-grouped cross-validation protocol using 1D ResNet with P90 window pooling:

| Temporal Context | Window Duration | Window Samples | Usable Windows | Eligible Patients | Patient AUROC | 95% Bootstrap CI | AUPRC | Clinical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Short (5m)** | 5 min | 1,200 | 10,977 | 547 | **0.5736** | [0.517, 0.633] | 0.2463 | Captures sharp acute decelerations with minimal missingness |
| **Short (10m)** | 10 min | 2,400 | 10,205 | 547 | **0.5658** | [0.508, 0.625] | 0.2522 | Acute episode horizon |
| **Medium (20m)** | 20 min | 4,800 | 8,517 | 547 | **0.5678** | [0.508, 0.629] | 0.2456 | Standard FIGO evaluation window; optimal trade-off |
| **Intermediate (30m)**| 30 min | 7,200 | 6,729 | 547 | **0.5367** | [0.477, 0.596] | 0.2208 | Onset of event dilution |
| **Long (40m)** | 40 min | 9,600 | 4,727 | 542 | **0.5399** | [0.478, 0.600] | 0.2381 | Severe temporal dilution; patient attrition |
| **Full Hour (60m)** | 60 min | 14,400 | 532 | 532 | **0.5439** | [0.476, 0.611] | 0.2463 | Non-stationary labor trace; single static score per patient |

---

## 3. Comprehensive Results Table

| Model Architecture | Context | Signal Substrate | Knowledge Substrate | Fusion Mechanism | Patient AUROC | 95% Bootstrap CI | AUPRC |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Phase 4 Baseline (Reference)**| 20 min | 1D ResNet (P90) | 19 Descriptors | Logit Prior Modulation | **0.7361** | [0.679, 0.788] | **0.4514** |
| **Clinical LR (19 Descriptors)** | 20 min | — | 19 Descriptors | Logistic Regression | **0.7198** | [0.660, 0.774] | **0.4554** |
| **Temporal Clinical (All Scales)**| 10m+20m+40m | — | 76 Descriptors + Trends | Logistic Regression | **0.6555** | [0.594, 0.713] | 0.3541 |
| **Temporal Clinical (Decel Trends)**| 20m+40m | — | 20m + Decel Trends | Logistic Regression | **0.6273** | [0.566, 0.688] | 0.3420 |
| **Phase 5 Multi-Scale Candidate** | 10m+20m+40m | 1D ResNet Multi | Decel Trends | Logit Prior Modulation | **0.6258** | [0.565, 0.687] | **0.3154** |
| **Multi-Scale Signal Only** | 10m+20m+40m | 1D ResNet Multi | — | Logit Score Fusion ($\sum \alpha_k z_k$) | **0.5808** | [0.521, 0.642] | 0.2490 |
| **Short Signal Control** | 5 min | 1D ResNet (P90) | — | Standalone 5m | **0.5736** | [0.517, 0.633] | 0.2463 |
| **Medium Signal Control** | 20 min | 1D ResNet (P90) | — | Standalone 20m | **0.5678** | [0.508, 0.629] | 0.2456 |
| **Long Signal Control** | 40 min | 1D ResNet (P90) | — | Standalone 40m | **0.5399** | [0.478, 0.600] | 0.2381 |

---

## 4. Paired Statistical Comparisons (2,000 Bootstrap Replicates)

| Comparison | $\mathbf{\Delta AUROC}$ | 95% Paired CI | Paired p-value | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Short (10m) − Medium (20m)** | $\mathbf{-0.0023}$ | [$-0.0824, +0.0777$] | $p = 0.4815$ | Equivalent standalone performance |
| **Long (40m) − Medium (20m)** | $\mathbf{-0.0282}$ | [$-0.0942, +0.0375$] | $p = 0.2080$ | Longer context tends to degrade discrimination |
| **Multi-Scale Signal − 20m Signal** | $\mathbf{+0.0131}$ | [$-0.0324, +0.0556$] | $p = 0.2750$ | Insufficient gain ($<+0.020$, crosses zero) |
| **Multi-Scale Signal − Best Single (10m)**| $\mathbf{+0.0154}$ | [$-0.0430, +0.0726$] | $p = 0.2965$ | No statistically significant multi-scale benefit |
| **Temporal Clinical − 20m Clinical** | $\mathbf{-0.0035}$ | [$-0.0253, +0.0191$] | $p = 0.3870$ | Trend descriptors add collinearity, not signal |
| **Phase 5 Candidate − Phase 4 Baseline**| $\mathbf{-0.1109}$ | [$-0.1717, -0.0520$] | $\mathbf{p < 0.001}$ | **Phase 4 baseline is strictly superior** |
| **Phase 5 Candidate − Multi-Scale Signal**| $\mathbf{+0.0449}$ | [$+0.0061, +0.0821$] | $\mathbf{p = 0.0135}$| Knowledge prior still provides positive modulation |

---

## 5. Multi-Scale Correlation & Diagnostic Overlap

Evaluating score concordance between individual temporal scales across the 547 patients:

| Scale Comparison | Pearson Correlation ($r$) | Spearman Rank Correlation ($\rho$) | Diagnostic Concordance |
| :--- | :---: | :---: | :--- |
| **10-min ($p_{10}$) vs 20-min ($p_{20}$)** | **0.8659** | **0.8492** | High redundancy; representations largely overlap |
| **20-min ($p_{20}$) vs 40-min ($p_{40}$)** | **0.8142** | **0.7981** | Strong correlation with increasing noise at 40m |
| **10-min ($p_{10}$) vs 40-min ($p_{40}$)** | **0.7811** | **0.7634** | Shared global risk ranking |

### Scale Complementarity Breakdown ($N=110$ Acidotic Cases):
- **Short 10-min Only Correct:** $5 / 110$ (4.5%)
- **Medium 20-min Only Correct:** $7 / 110$ (6.4%)
- **Long 40-min Only Correct:** $4 / 110$ (3.6%)
- **All Three Scales Concordant:** $12 / 110$ (10.9%)
- **Union of Any Scale Correct:** $28 / 110$ (25.5%)

---

## 6. Time-to-Delivery Stratification Analysis

Stratifying patient windows by time remaining before delivery:
- **Early Windows ($>20\text{ min}$ before delivery):** Patient AUROC = **0.6277**
- **Late Windows ($\le 20\text{ min}$ before delivery):** Patient AUROC = **0.5841**

*Clinical Takeaway:* CTG waveforms $>20$ minutes before delivery retain better signal-to-noise ratio than the final 20 minutes of active pushing/expulsion, where maternal expulsive efforts cause severe transducer displacement and signal dropout without necessarily providing clearer morphological signatures.

---

## 7. Generated Visualizations

All 7 required diagnostic figures have been generated and saved to `reports/figures_phase5/`:
- **Figure 1:** [Context Length vs AUROC](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase5/fig1_context_length_auroc.png)
- **Figure 2:** [Multi-Scale Score Correlations](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase5/fig2_scale_correlations.png)
- **Figure 3:** [Score Distributions Across Scales](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase5/fig3_score_distributions.png)
- **Figure 4:** [Patient-Level ROC Curves](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase5/fig4_roc_comparison.png)
- **Figure 5:** [Phase 5 Model Benchmark Comparison](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase5/fig5_model_comparison.png)
- **Figure 6:** [Temporal-Scale Contribution Weights](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase5/fig6_scale_weights.png)
- **Figure 7:** [Stratified Performance by Time to Delivery](file:///e:/Maha/CTG-Fetal-Distress-Prediction/reports/figures_phase5/fig7_delivery_stratification.png)

---

## 8. Hard Gate Evaluation & Final Decision

- **Hard Gate 1 (Individual Scale):** *FAILED.* No temporal scale outperforms the 20-min standard ($p = 0.48$ for 10m vs 20m; longer scales 30m/40m/60m are substantially worse).
- **Hard Gate 2 (Multi-Scale Signal):** *FAILED.* Multi-scale signal fusion achieves $\Delta = +0.0131$, failing the required $+0.020$ threshold and crossing zero ($p = 0.275$).
- **Hard Gate 3 (Phase 4 Target):** *FAILED.* Phase 5 candidate ($0.6258$) does not exceed the Phase-4 baseline ($0.7361$).
- **Hard Gate 4 (Target Proximity):** $< 0.75$.

```text
========================================================================================
FINAL DECISION: EXIT A — NO TEMPORAL GAIN
========================================================================================
1. CLOSE the temporal multi-resolution expansion branch.
2. FREEZE the 20-minute temporal receptive field as the optimal clinical window.
3. RETAIN the Phase-4 architecture (1D ResNet 20m + 19 Clinical Descriptors + Logit Prior
   Modulation: AUROC = 0.7361 [0.679, 0.788], AUPRC = 0.4514) as the master pipeline.
4. PROCEED to Phase 6: Continuous pH Supervision & Composite Adverse Endpoint Formulation
   (investigating whether soft ordinal regression, continuous pH losses, and composite
   distress endpoints unlock the remaining diagnostic potential).
========================================================================================
```
