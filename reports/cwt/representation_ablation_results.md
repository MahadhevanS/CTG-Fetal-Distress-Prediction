# CTU-UHB 2D Representation Ablation Report: Continuous Wavelet Transform (CWT) & Recurrence Plots (RP)

**Date**: 2026-09-05  
**Cohort**: 547 patients (110 positive, pH $\le 7.15$), 8,517 20-minute windows (4 Hz, 2.5-min stride)  
**Evaluation Protocol**: Frozen 5-fold Patient-Grouped Cross-Validation (`folds.json`, repeat 0)  
**Primary Aggregation**: Patient-level Max Aggregation  
**Uncertainty**: 2,000 Patient-Level Bootstrap Resamples  

---

## 1. Executive Summary

This study conducted a strict, leakage-free empirical investigation of **2D time-frequency (Continuous Wavelet Transform / CWT)** and **2D phase-space (Recurrence Plot / RP)** representations for fetal acidemia prediction on the CTU-UHB intrapartum benchmark.

The investigation was motivated by high-performing prior-art literature (e.g., DeepFHR reporting 94–98% AUC) to determine whether 2D representations capture clinically predictive signal that raw 1D encoders miss under a strict patient-level protocol.

### Key Finding
> **2D time-frequency and recurrence representations fail to outperform raw 1D waveforms and severely underperform the 19-descriptor clinical baseline.**
>
> - **Clinical 19-Descriptor Logistic Regression**: **0.7268 [0.670, 0.779]**
> - **Raw 1D CNN Baseline**: **0.6147 [0.556, 0.671]** ($\Delta = -0.1112$, $p = 1.23 \times 10^{-82}$)
> - **CWT 2D CNN (FHR only)**: **0.5301 [0.469, 0.587]** ($\Delta = -0.1978$, $p = 3.38 \times 10^{-25}$)
> - **CWT Dual 2D CNN (FHR + UC)**: **0.5666 [0.505, 0.624]** ($\Delta = -0.1609$, $p = 6.73 \times 10^{-10}$)
> - **Recurrence Plot 2D CNN**: **0.5732 [0.513, 0.633]** ($\Delta = -0.1533$, $p = 5.69 \times 10^{-18}$)

---

## 2. Comprehensive Results Table

All models evaluated on identical patient folds with inner-validation model selection (zero test-set selection):

| Model | Input | Params | Patient AUROC (Max) [95% CI] | AUPRC | Paired $\Delta$ vs Clinical LR [95% CI] | Wilcoxon $p$-value | Mean Fold AUROC $\pm$ Std |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **19-Descriptor Clinical LR** | 19 Features | 19 | **0.7268 [0.670, 0.779]** | **0.4093** | *0.0000 (Reference)* | — | $0.7271 \pm 0.0450$ |
| **Raw 1D CNN (Baseline)** | 1D FHR | 252,321 | **0.6147 [0.556, 0.671]** | 0.2865 | $-0.1112$ $[-0.171, -0.052]$ | $1.23 \times 10^{-82}$ | $0.6403 \pm 0.0375$ |
| **CWT 2D CNN (FHR)** | CWT Scalogram (64 $\times$ 600) | 633,313 | **0.5301 [0.469, 0.587]** | 0.2214 | $-0.1978$ $[-0.272, -0.122]$ | $3.38 \times 10^{-25}$ | $0.5634 \pm 0.0665$ |
| **CWT Dual 2D CNN (FHR+UC)** | 2-Ch CWT (64 $\times$ 600) | 633,601 | **0.5666 [0.505, 0.624]** | 0.2381 | $-0.1609$ $[-0.240, -0.082]$ | $6.73 \times 10^{-10}$ | $0.5873 \pm 0.0848$ |
| **Recurrence Plot 2D CNN** | RP Matrix (128 $\times$ 128) | 633,313 | **0.5732 [0.513, 0.633]** | 0.2543 | $-0.1533$ $[-0.222, -0.084]$ | $5.69 \times 10^{-18}$ | $0.6017 \pm 0.0374$ |

---

## 3. Patient Aggregation Sensitivity Analysis

Comparing aggregation functions across all out-of-fold window predictions:

| Model | Max Aggregation | p90 Aggregation | Top-3 Mean Aggregation | Mean Aggregation |
| :--- | :---: | :---: | :---: | :---: |
| **Clinical LR Baseline** | **0.7268** | 0.7174 | 0.7152 | 0.6741 |
| **Raw 1D CNN** | **0.6147** | 0.6091 | 0.6020 | 0.5948 |
| **CWT 2D CNN (FHR)** | **0.5301** | 0.5268 | 0.5227 | 0.5161 |
| **CWT Dual 2D CNN (FHR+UC)**| **0.5666** | 0.5422 | 0.5418 | 0.5251 |
| **Recurrence Plot 2D CNN** | **0.5732** | 0.5568 | 0.5574 | 0.5376 |

*Finding*: Max aggregation consistently produces the highest AUROC across all models, confirming that intrapartum acidemia risk is best indexed by peak episodic abnormality rather than time-averaged signal.

---

## 4. Evaluation of Pre-Registered Hard Gates

- **Gate 1 (CWT vs Raw CNN)**: 
  *Criterion*: CWT must outperform Raw 1D CNN by a meaningful margin ($\Delta > +0.02$).
  *Result*: **FAILED**. CWT 2D CNN (0.5301) underperformed Raw 1D CNN (0.6147) by $-0.0846$ AUROC.
- **Gate 2 (AUROC $\ge 0.75$)**: 
  *Criterion*: CWT must reach $\ge 0.75$ AUROC before proceeding to MIL or fusion.
  *Result*: **FAILED**. CWT reached only $0.5301 - 0.5666$.
- **Gate 5 (Stopping Rule)**: 
  *Criterion*: If no 2D representation approaches $\ge 0.80$, terminate the 2D representation branch.
  *Result*: **TRIGGERED**. Close representation branch.
