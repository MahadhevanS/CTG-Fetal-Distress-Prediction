# Preprocessed Dataset Summary

This document outlines the structure, statistics, and clinical dynamics of the generated PyTorch datasets for the CTG Fetal Distress Prediction framework.

## 1. Global Overview

The raw PhysioNet CTU-CHB Intrapartum dataset contained **552 clinical recordings**. Clinical metadata, including umbilical artery pH and Apgar scores, were parsed from the WFDB headers and linked to each recording, generating time-series windows strictly mapped to these physiological outcomes.

### Patient-Level Stratified Splitting
To prevent data leakage, the dataset was strictly split at the patient level before window extraction, ensuring no overlapping physiological data exists across the evaluation boundaries.

- **Total Patients**: 552
- **Training Set**: 386 patients (~70%)
- **Validation Set**: 83 patients (~15%)
- **Testing Set**: 83 patients (~15%)

---

## 2. Dataset Statistics & Imbalance Mitigation

Due to the extreme class imbalance (Pathological fetal distress is rare compared to Normal outcomes), an overlap-based minority generation strategy was enforced exclusively on the training split.

### Window Extraction Rules
- **Clinically Relevant Horizon**: Windows were strictly extracted from the **last 60 minutes** of the recording prior to delivery. If a recording was shorter than 60 minutes, the entire available duration was utilized. This prevents early, healthy stages of labor from being incorrectly labeled with a terminal pathological outcome.
- **Training Stride**: 
  - Distress cases: 1-minute stride (heavy overlap/oversampling).
  - Normal cases: 10-minute stride (sparse overlap).
- **Validation/Test Stride**: Fixed 10-minute stride for all cases to ensure metrics reflect true, unbiased real-world distributions.

### Final Window Counts
| Split | Normal Patients | Distress Patients | Total 20-Min Windows | pH Normal | pH Distress (Acidemia) | Dynamics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | ~316 | ~70 | **2,534** | 1,236 | 1,298 | Reduced distress stride (2-min) achieves a perfect ~50:50 balance. |
| **Validation** | ~68 | ~15 | **322** | 263 | 59 | Unbiased clinical snapshot (~4.5:1 Normal to Distress). |
| **Test** | ~68 | ~15 | **329** | 269 | 60 | Unbiased clinical snapshot (~4.5:1 Normal to Distress). |

---

## 3. Clinical Consistency & Baseline Validation

A post-extraction consistency audit was performed to validate the programmatic feature extraction and clinical rule engine against the ground truth pH outcomes (Validation & Test Sets):

- **High Negative Predictive Value**: When the rule engine classified a window as *FIGO Normal*, the terminal outcome was overwhelmingly normal (e.g., 27 Normal vs 1 Distress in the Test set).
- **Limited Positive Predictive Value (Moderate Specificity)**: Although FIGO Pathological windows were associated with increased fetal risk, a substantial proportion still corresponded to normal terminal pH outcomes (e.g., 35 Normal vs 14 Distress). This reflects the known limited specificity of rule-based CTG interpretation.
- **Research Justification**: This correlation perfectly mirrors real-world obstetric challenges—standard clinical guidelines (FIGO) are highly sensitive but poorly specific, leading to unnecessary interventions. This validates the necessity of our deep learning framework, which utilizes these rules merely as auxiliary guidance (`y_figo`) while forcing the Temporal Encoder to discover deeper physiological correlations directly linked to the terminal pH outcome (`y_primary`). The FIGO labels are **not** used during inference; they serve solely as auxiliary supervision during training.

---

## 3. Data Structure and Shapes

The generated files (`train_dataset.pt`, `val_dataset.pt`, `test_dataset.pt`) are PyTorch dictionaries containing three distinct tensors per split.

### A. Input Tensor (`X`)
- **Shape**: `(N, 2, 4800)`
  - `N`: Number of windows in the split.
  - `2`: Channels (Channel 0: Baseline-Corrected FHR, Channel 1: UC).
  - `4800`: Temporal sequence length (20 minutes $\times$ 60 seconds $\times$ 4 Hz sampling rate).
- **Preprocessing Applied**:
  - 30% Missing Signal Threshold Rejection (SQA).
  - $<15$ sec gaps interpolated via Cubic Splines.
  - **4th-Order** Butterworth Low-Pass Filter (cutoff ~1.5 Hz).
  - *Baseline Subtraction* applied to the FHR channel to stabilize network gradients.

### B. Primary Target Tensor (`y_primary`)
- **Shape**: `(N,)`
- **Data Type**: `torch.long` (Integer)
- **Meaning**: Binary Terminal Acidemia Outcome. (Derived from terminal pH, utilized as the model's primary task).
  - `0`: Normal / pH > 7.15
  - `1`: Distress / Acidemia / pH $\leq$ 7.15

### C. Knowledge-Guided Supervisory Targets (`y_figo` and `y_features`)

To preserve semantic purity, the auxiliary targets are decoupled:

1. **`y_figo`**
- **Shape**: `(N,)`
- **Data Type**: `torch.long`
- **Meaning**: Categorical FIGO Class (0 = Normal, 1 = Suspicious, 2 = Pathological).

2. **`y_features`**
- **Shape**: `(N, 8)`
- **Data Type**: `torch.float32`
- **Meaning**: Continuous physiological features and deceleration counts.
- **Indices Map**:
  - `[0]` - **Baseline FHR**: Iterative calculation (bpm).
  - `[1]` - **STV**: Short-Term Variability (Computed as beat-to-beat delta bpm).
  - `[2]` - **LTV**: Long-Term Variability (Computed as variance in bpm).
  - `[3]` - **Accelerations**: Count per 20-minute window ($\geq$ 15 bpm for $\geq$ 15 sec).
  - `[4]` - **Early Decelerations**: Count per 20-minute window (Nadir correlated with UC peak).
  - `[5]` - **Late Decelerations**: Count per 20-minute window (Delayed nadir relative to UC peak).
  - `[6]` - **Variable Decelerations**: Count per 20-minute window (Rapid descent $<$ 30 sec).
  - `[7]` - **Prolonged Decelerations**: Count per 20-minute window (Lasting $\geq$ 2 minutes).

### D. Identifiers (`metadata`)
A tuple of `(record_id, window_start_idx, window_end_idx)` corresponding to each window is stored to enable precise post-hoc analysis (e.g., mapping Grad-CAM attention weights directly back to the specific timestamps within clinical recordings).

---

## 4. Methodological Highlights

1. **Leakage Prevention**: The dataset was strictly split at the patient level before any window extraction, ensuring absolutely no overlapping physiological data exists across training and evaluation boundaries.
2. **Clinical Authenticity**: Zero data synthesis (e.g., SMOTE) was performed on the physiological signals. All tensors represent real physiological waveforms, satisfying the strictest requirements of medical AI research.
3. **Knowledge Supervision**: By extracting the features into `y_features` and establishing a separate FIGO target, the neural network is encouraged to learn clinically meaningful physiological representations rather than purely memorizing the binary pH mapping.
4. **Explainability**: Each prediction is associated with its corresponding physiological descriptors (baseline FHR, STV, LTV, accelerations, deceleration types, and uterine contraction characteristics), enabling direct inspection of the learned clinical reasoning during post-hoc analysis.

---

## 5. Preprocessing Validation Report (Internal Reference)

| Metric | Value |
| :--- | :--- |
| **Total recordings** | 552 |
| **Total accepted windows** | 3,185 |
| **Rejection rate (SQA > 30% loss)** | ~5-8% (varies by split) |
| **Average windows/patient (Train)** | ~6.5 (due to 50:50 overlap tuning) |
| **Mean recording length** | Truncated to max 60 mins |
| **Class distribution (patients)** | ~452 Normal / 100 Distress |
| **Class distribution (train windows)** | 1,236 Normal / 1,298 Distress |
| **FIGO Test Correlation** | FIGO Normal effectively rules out Distress |

---

## 6. Pre-Training Sanity Checks & Dataset Validation

Before initiating model training, a final dataset validation notebook/script should be utilized to perform the following checks, ensuring absolute data integrity:

1. **Tensor Integrity**: Verify `X`, `y_primary`, `y_figo`, and `y_features` shapes. Confirm the absence of `NaN`s, infinities, and verify correct `dtypes`.
2. **Feature Distributions**: Plot histograms for Baseline FHR, STV, LTV, and deceleration counts to ensure they reflect clinically reasonable distributions.
3. **Data Leakage Verification**: Mathematically confirm that `Train IDs ∩ Validation IDs = ∅` and `Train IDs ∩ Test IDs = ∅`.
4. **Random Sample Visualization**: Randomly inspect 20–30 windows (plotting FHR, UC, Baseline, and Decelerations) to visually catch any subtle preprocessing bugs.
5. **Correlation Sanity Check**: Compute correlations (e.g., pH ↔ STV, pH ↔ Late Decelerations) to validate that the extracted physiological features behave as expected.

---

## 7. Dataset Freeze Protocol

With preprocessing officially complete, the dataset is now **Frozen (v1.0)**. 
- **Do not regenerate the dataset** unless the preprocessing pipeline is intentionally altered.
- All future experiments, ablation studies, and architectural comparisons will use these exact `.pt` files to ensure fair and reproducible benchmarking.

---

## 8. Next Phase: Multi-Task Architecture

The pipeline now shifts from data engineering to model architecture and learning. The frozen dataset will be ingested by the following framework:

```text
Preprocessed Dataset (v1.0)
        │
        ▼
Temporal Encoder (1D CNN / Transformer)
        │
        ▼
Shared Latent Representation
        │
        ├────────────► Distress Head (Binary Acidemia)
        │
        ├────────────► FIGO Head (Auxiliary Knowledge)
        │
        └────────────► Clinical Feature Head (Physiological Regression)
        │
        ▼
Multi-Task Loss
```
