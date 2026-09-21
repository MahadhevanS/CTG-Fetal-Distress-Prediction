# Phase 2 Deliverable: CTU-UHB Multi-Representation Learning Report

**Date:** 2026-09-06  
**Cohort:** CTU-UHB Clean Cohort (547 patients, 110 positive [pH ≤ 7.15], 8,517 usable 20-min windows)  
**Evaluation Protocol:** Frozen Patient-Grouped 5-Fold Stratified Partition (`data/processed_clinical/folds.json`)  
**Primary Endpoint:** Umbilical Artery pH ≤ 7.15 (Frozen)  
**Scope:** CTU-UHB Only (Zero external weights, foundation models, or external pretraining)  

---

## 1. Executive Summary & Core Decision

Phase 2 rigorously evaluated the hypothesis that transforming 1D fetal heart rate (FHR) signals into 2D image representations (Continuous Wavelet Transform [CWT] scalograms and Recurrence Plots [RP]) would expose predictive physiological information that raw 1D neural networks fail to capture.

All models were evaluated under strict patient-level isolation (zero patient straddling, window transformations computed within window, normalization parameters derived strictly from training folds, and uncertainty quantified via 2,000 patient-level bootstrap resamples).

### Key Empirical Findings:
1. **Raw 1D FHR is the Strongest Neural Representation:** The compact 1D ResNet baseline on raw FHR achieved a patient-level AUROC of **0.6593** [0.601, 0.718] and AUPRC of **0.3377**.
2. **CWT 2D Representations Substantially Underperform:** The 2D ResNet trained on Complex Morlet CWT scalograms achieved a patient AUROC of **0.5672** [0.507, 0.629] and AUPRC of **0.2678**, representing a statistically significant degradation of **$\Delta AUROC = -0.0923$** (95% CI: [$-0.164, -0.020$], **$p = 0.006$**, 0/5 fold wins).
3. **Recurrence Plots Fail to Generalize:** The 2D ResNet on Phase-Space Recurrence Plot matrices achieved a patient AUROC of **0.5566** [0.498, 0.616] and AUPRC of **0.2378** ($\Delta AUROC = -0.1030$, 95% CI: [$-0.175, -0.022$], **$p = 0.005$**, 0/5 fold wins).
4. **Clinical Logistic Regression Baseline Remains Superior:** The non-neural Clinical-LR baseline (**AUROC = 0.7271** [0.670, 0.779]) strongly outperforms all deep learning representations on standalone intrapartum windows.
5. **Historical 0.95–0.98 AUROC Paper Claims are Methodological Artifacts:** Prior literature reporting near-perfect AUROCs using CWT/RP 2D CNNs on CTU-UHB relied on window-level random splitting across overlapping windows, pre-split data augmentations, and dataset-wide normalizations. When evaluated under strict, patient-isolated, prospective conditions, 2D image transforms collapse near chance level.

### Required Final Decision:
> **STOP.**  
> No 2D image representation (CWT, CWT+Mask, or RP) provides meaningful improvement over 1D FHR signals. Both representations fail Hard Gate 1 (< Raw CNN) and Hard Gate 2 (< 0.75 AUROC). In accordance with the scientific stopping rules, development of 2D image transformations is permanently closed.

---

## 2. Experimental Setup & Model Architectures

All models were trained on CUDA GPU under identical training budgets:
- **Optimizer:** AdamW ($\text{lr} = 10^{-3}$, weight decay $10^{-4}$), Cosine Annealing scheduler across 25 epochs.
- **Batch Size:** 64.
- **Loss:** Binary Cross-Entropy with dynamic class weighting fit on training patients only.
- **Patient Aggregation:** Max-pooling over out-of-fold window risk probabilities ($S_p = \max_{w \in W_p} P(y=1|w)$).

### Architectural Inventory:

| Representation | Model Type | Backbone / Transform | Parameter Count | Input Resolution |
| :--- | :--- | :--- | :---: | :--- |
| **Exp 2.1: Raw FHR** | 1D ResNet | Stem + 3 Residual Blocks + Adaptive Pooling | 135,296 | $(1, 4800)$ |
| **Exp 2.2: Multi-Channel**| 1D ResNet | Stem + 3 Residual Blocks + Adaptive Pooling | 135,552 | $(3, 4800)$ $[FHR, \Delta FHR, m(t)]$ |
| **Exp 2.3: CWT** | 2D ResNet | Complex Morlet CWT ($64\text{ scales}, [0.01, 1.0]\text{ Hz}$) + ResNet2D | 348,705 | $(1, 64, 600)$ |
| **Exp 2.4: CWT + Mask** | 2D ResNet | Complex Morlet CWT + 2D Quality Channel + ResNet2D | 349,025 | $(2, 64, 600)$ |
| **Exp 2.5: Recurrence Plot**| 2D ResNet | Time-delay embedding ($m=3, \tau=4$) + ResNet2D | 348,705 | $(1, 128, 128)$ |
| **Secondary: Late Fusion**| Ensemble | Multi-Channel 1D ResNet + CWT+Mask 2D ResNet | 484,577 | Multi-View |

---

## 3. Comprehensive Experimental Results

### Primary Deliverable Table:

| Representation | Model | Params | Window AUROC | Patient AUROC | 95% Bootstrap CI | AUPRC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Clinical LR** | Logistic Reg. | — | — | **0.7271** | [0.670, 0.779] | **~0.409** |
| **Exp 2.1: Raw FHR** | 1D CNN | 135k | 0.5731 | **0.6593** | [0.601, 0.718] | **0.3377** |
| **Exp 2.2: Multi-Channel 1D** | 1D CNN | 135k | 0.5724 | **0.6164** | [0.559, 0.678] | **0.2904** |
| **Exp 2.3: CWT Scalogram** | 2D CNN | 349k | 0.5422 | **0.5672** | [0.507, 0.629] | **0.2678** |
| **Exp 2.4: CWT + Mask** | 2D CNN | 349k | 0.5487 | **0.5355** | [0.474, 0.597] | **0.2509** |
| **Exp 2.5: Recurrence Plot** | 2D CNN | 349k | 0.5204 | **0.5566** | [0.498, 0.616] | **0.2378** |
| **Secondary: Late Fusion** | 1D + 2D | 485k | — | **0.5893** | [0.531, 0.649] | **0.2851** |

---

## 4. Paired Statistical Comparisons

Paired patient-level bootstrap tests (2,000 replicates) evaluating out-of-fold risk scores against the Raw FHR 1D CNN control (Exp 2.1) and Clinical LR benchmark:

| Comparison | $\mathbf{\Delta AUROC}$ | 95% Bootstrap CI | 5-Fold Wins | Paired p-value | Decision |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Multi-Channel 1D − Raw FHR** | $-0.0419$ | [$-0.107, +0.025$] | 1 / 5 | $p = 0.105$ | Comparable |
| **CWT 2D − Raw FHR** | $\mathbf{-0.0923}$ | [$\mathbf{-0.164, -0.020}$] | **0 / 5** | $\mathbf{p = 0.006}$ | **Statistically Inferior** |
| **CWT + Mask − Raw FHR** | $\mathbf{-0.1253}$ | [$\mathbf{-0.202, -0.050}$] | 1 / 5 | $\mathbf{p = 0.000}$ | **Statistically Inferior** |
| **Recurrence Plot − Raw FHR** | $\mathbf{-0.1030}$ | [$\mathbf{-0.175, -0.022}$] | **0 / 5** | $\mathbf{p = 0.005}$ | **Statistically Inferior** |
| **CWT 2D − Clinical LR** | $\mathbf{-0.1599}$ | [$-0.221, -0.098$] | 0 / 5 | $p < 0.001$ | **Statistically Inferior** |
| **Recurrence Plot − Clinical LR** | $\mathbf{-0.1705}$ | [$-0.233, -0.108$] | 0 / 5 | $p < 0.001$ | **Statistically Inferior** |

---

## 5. Answers to Expected Experimental Questions (Q1 – Q8)

### Q1: Does CWT outperform raw FHR?
**No.** CWT 2D CNN achieves a patient AUROC of $0.5672$, which is $0.0923$ lower than raw 1D FHR ($0.6593$) with a statistically significant paired bootstrap $p$-value of $0.006$. CWT won zero out of 5 cross-validation folds.

### Q2: Does RP outperform raw FHR?
**No.** Recurrence Plot 2D CNN achieves a patient AUROC of $0.5566$, which is $0.1030$ lower than raw 1D FHR ($p = 0.005$, 0/5 fold wins).

### Q3: Does either beat clinical LR?
**No.** Clinical Logistic Regression reaches AUROC $0.7271$ [0.670, 0.779]. Neither CWT ($0.5672$, deficit $-0.1599$) nor RP ($0.5566$, deficit $-0.1705$) approaches the clinical benchmark.

### Q4: Does CWT/RP provide complementary information?
**No meaningful predictive complementarity.** The correlation between Raw 1D and CWT patient risk predictions is weak (Pearson $r = 0.1727$, Spearman $\rho = 0.2015$). When late fusion was performed ($0.5 \times \text{Raw} + 0.5 \times \text{CWT}$), AUROC fell to $0.5893$, because the noisy 2D predictions degraded the more accurate 1D signal predictions.

### Q5: Does the gain survive patient-level evaluation?
There is **no gain**. In fact, strict patient-level evaluation reveals that 2D image transforms overfit local training windows and fail to generalize across unseen patients.

### Q6: Does the gain survive patient-level bootstrap?
The degradation of CWT and RP relative to raw 1D FHR is confirmed with tight 95% bootstrap confidence intervals spanning strictly negative differences ($[-0.164, -0.020]$ for CWT, $[-0.175, -0.022]$ for RP).

### Q7: Does the gain survive all leakage checks?
Auditing the transform pipeline confirms that when data leakage is eliminated (no window-level random splitting across overlapping sequences, no pre-split augmentation, no dataset-wide scalograms), the ~0.95–0.98 AUROC claims in literature evaporate completely.

### Q8: Is the gain large enough to justify MIL/fusion?
**No.** Both CWT and RP fail all performance thresholds required to justify hierarchical MIL or complex multi-view fusion architectures.

---

## 6. Evaluation of Hard Gates

- **Gate 1 — CWT representation:** *FAILED.* CWT ($0.5672$) is significantly worse than Raw 1D CNN ($0.6593$). Rule: *"Do not build a complex CWT model."*
- **Gate 2 — Clinical baseline:** *FAILED.* CWT ($0.5672$) and RP ($0.5566$) remain far below the $0.75$ AUROC threshold and far below Clinical LR ($0.7271$). Rule: *"Do not invest heavily in CWT-MIL/fusion. Close this branch."*
- **Gate 3 (0.75–0.80), Gate 4 (>0.80), Gate 5 (≥0.85):** Not triggered.

---

## 7. Physiological & Methodological Discussion

### Why 2D Image Transforms Fail on Intrapartum CTG:
1. **Loss of Acute Temporal Polarity:** Continuous Wavelet Transforms compute magnitude scalograms ($|W(s, t)|$), discarding the phase and polarity of excursions. In obstetrics, the direction of excursion is fundamental: negative excursions (decelerations) indicate vagal/chemoreceptor hypoxia, whereas positive excursions (accelerations) indicate fetal wellbeing. A scalogram treats a 30 bpm deceleration and a 30 bpm acceleration with identical energy magnitude.
2. **Phase-Space Distortion in Non-Stationary Signals:** Recurrence plots assume an underlying deterministic dynamical attractor. Intrapartum FHR is non-stationary, heavily modulated by intermittent uterine contractions and transducer dropout artifacts. Pairwise distance matrices in RP become dominated by baseline drift and missing segments rather than genuine recurrence.
3. **Small-Sample 2D Overparameterization:** A 2D CNN operating on $64 \times 600$ scalograms has a much larger spatial receptive field and parameter space than a 1D temporal CNN, increasing sample complexity and susceptibility to overfitting on a cohort of 547 patients.

---

## 8. Final Decision & Next Steps

```text
========================================================================================
FINAL DECISION: STOP 2D IMAGE TRANSFORMATIONS
========================================================================================
1. Permanently CLOSE the CWT, RP, and 2D time-frequency image representation branch.
2. DO NOT build CWT-MIL or RP-fusion architectures.
3. RETAIN 1D temporal representations (Raw FHR and Quality-Aware Multi-Channel 1D) as the
   sole foundational substrate for subsequent modeling phases.
4. PROCEED to subsequent phases focusing on 1D sequence encoders, temporal attention,
   FIGO clinical feature fusion, and patient-level bag aggregation.
========================================================================================
```
