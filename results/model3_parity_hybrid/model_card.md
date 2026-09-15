# Model Card: Model 3 + Parity Multimodal Late-Fusion Hybrid

## Model Details
- **Model Name:** Model 3 + Parity Multimodal Hybrid (Late-Fusion Logistic)
- **Model Version:** 1.0.0
- **Model Date:** 2026-09-13
- **Model Type:** Late-fusion regularized logistic regression: $\sigma(\beta_0 + \beta_1 z_{\text{M3}} + \beta_2 z_{\text{parity}})$
- **Inputs:** 
  1. $z_{\text{M3}} = \text{logit}(\text{Model 3 risk score})$ from causal temporal-position attention over P6
  2. $z_{\text{parity}} = \text{logit}(p_{\text{parity}})$ from univariate training-fitted parity logistic regression
- **Upstream Encoders:** Phase 12.1 P6 frozen window encoder; Phase 16 Model 3 causal attention scorer
- **Regularization:** L2 penalty ($C=0.1$, scikit-learn LogisticRegression)
- **Frameworks:** Python 3.10, PyTorch 2.11.0+cu128, scikit-learn 1.4+, NumPy 1.26+, Pandas 2.2+

---

## Intended Use
- **Primary Intended Use:** Internal research benchmark candidate to evaluate multimodal information complementarity between continuous intrapartum CTG signals and maternal parity.
- **Out-of-Scope & Prohibited Uses:**
  - Real-time clinical decision support or bedside alarm generation.
  - Clinical diagnostic decisions regarding operative delivery or cesarean section.
  - Deployment in any hospital, labour ward, or obstetric setting.
  - Retraining or modifying the locked upstream P6 model.

---

## Training and Evaluation Data
- **Cohort:** Cleaned CTU-UHB intrapartum cardiotocography database ($N=547$ unique patient recordings, 8,517 causal 20-minute rolling windows with 2.5-minute stride).
- **Target Endpoints:**
  - Primary: Umbilical arterial $\text{pH} \le 7.15$ ($N=110$ positive, 20.1% prevalence).
  - Secondary severe: Umbilical arterial $\text{pH} \le 7.05$ ($N=41$ positive, 7.5% prevalence).
- **Validation Partitioning:**
  - 5-Fold Patient-Grouped Stratified Cross-Validation ($N=547$ total; $N=464$ in train/val pool).
  - Held-Out Internal Test Partition ($N=83$ patients, 17 positives).
  - Zero patient overlap between training and evaluation splits across all folds.

---

## Quantitative Performance Summary

| Horizon | Evaluation Split | Hybrid AUROC | Model 3 AUROC | P6 AUROC | $\Delta$ vs. Model 3 (p-value) | $\Delta$ vs. P6 (p-value) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Delivery** | 5-Fold CV | **0.7324** | 0.7216 | 0.6872 | +0.0108 ($p=0.172$) | **+0.0452** ($p=0.004$) |
| **Delivery** | Internal Test | 0.6916 | 0.6729 | 0.6497 | **+0.0187** ($p=0.030$) | **+0.0419** ($p=0.030$) |
| $\ge 10$m | 5-Fold CV | 0.6894 | 0.6637 | 0.6823 | **+0.0257** ($p=0.044$) | +0.0071 ($p=0.634$) |
| $\ge 20$m | 5-Fold CV | 0.6490 | 0.5882 | 0.6113 | **+0.0608** ($p < 0.001$) | **+0.0377** ($p=0.061$) |
| $\ge 30$m | 5-Fold CV | 0.6551 | 0.5811 | 0.5695 | **+0.0740** ($p < 0.001$) | **+0.0856** ($p=0.001$) |

### Operational Behavior (80% Target Sensitivity)
- Achieved Sensitivity: 88.2%
- False Alert Rate (FAR): **65.2%** (lowest across all evaluated models; Model 3 FAR = 69.3%, P6 FAR = 83.1%)
- Median Lead Time: **35.0 minutes** (IQR: 35.0 min)
- Detection Rates: 70.1% detected $\ge 20$ min before delivery; 59.8% detected $\ge 30$ min before delivery.

---

## Subgroup & Fairness Factors
- **Maternal Parity:**
  - Nulliparous mothers ($N=373$, 68.2% of cohort): higher distress incidence (24.7% positive).
  - Multiparous mothers ($N=174$, 31.8% of cohort): lower distress incidence (10.3% positive).
- **Subgroup Discrimination:** Parity provides a robust prior adjustment. Shuffled-parity controls confirm 0/30 random permutations achieve the observed true gain.

---

## Ethical Considerations & Governance
- **Internal Research Candidate Only:** Under no circumstances should this model be described as clinically validated or ready for deployment.
- **Mandatory Next Step:** Formal evaluation on genuinely independent external cohorts (e.g. Oxford, STAN) is strictly required before considering clinical translation.
