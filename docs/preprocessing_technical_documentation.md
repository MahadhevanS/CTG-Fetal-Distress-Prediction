# CTG Fetal Distress Prediction: Preprocessing Technical Documentation

This document explains the data preprocessing pipeline designed for the **CTG Fetal Distress Prediction** framework. It details the clinical logic, data stream parameters, and algorithmic formulas used to transform raw cardiotocography (CTG) recordings into structured inputs for the machine learning models.

---

## 1. Pipeline Architecture Overview

The preprocessing pipeline ingests raw time-series data and clinical descriptors, filters out high-frequency noise and missing segments, extracts key clinical features, assigns clinical pseudo-labels based on expert rules, and outputs structured PyTorch tensors.

```text
Raw CTU-CHB Signals & Headers
        │
        ▼
Signal Extraction: FHR & UC
        │
        ▼
Signal Quality Assessment (SQA) ──► [Reject if >30% missing data]
        │
        ▼
Cubic Spline Interpolation (for gaps <= 15 seconds)
        │
        ▼
4th-Order Low-Pass Butterworth Filter (Cutoff: 1.5 Hz)
        │
        ▼
Iterative FHR Baseline Estimation
        │
        ├───────────────────────────────┐
        ▼                               ▼
Baseline Subtraction & Stacking    Clinical Feature Extraction (STV, LTV, Accel/Decel)
        │                               │
        ▼                               ▼
Normalized Input (2, 4800)         Programmatic FIGO Classification Engine
        │                               │
        └───────────────┬───────────────┘
                        ▼
           Patient-Level Stratified Split
                        │
                        ▼
          Imbalance Handling (Dynamic Stride)
                        │
                        ▼
             Final PyTorch Datasets
```

---

## 2. Dataset Characteristics & Ingestion

The framework operates on the **CTU-CHB Intrapartum Dataset** as its primary time-series repository.

### 2.1 File Ingestion
* **Format**: `.dat` (binary signal data) and `.hea` (text headers) for each record.
* **Signals Extracted**:
  * **Fetal Heart Rate (FHR)**: Captured in beats per minute (bpm). Channel index `0`.
  * **Uterine Contractions (UC)**: Measured in mmHg. Channel index `1`.
  * **Sampling Frequency (fs)**: 4 Hz (4 samples per second).
* **Metadata Parsing**: The `.hea` text comments are parsed to extract umbilical artery pH values and Apgar scores.
  * **Source Code**: [ingestion.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/ingestion.py)
  * **Key Functions**:
    * [load_ctu_chb_record](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/ingestion.py#L7)
    * [load_clinical_metadata](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/ingestion.py#L36)

### 2.2 Clinical Target Definition
* **Primary Target (y_primary)**: Umbilical artery pH <= 7.15. This is the clinical threshold indicating moderate-to-severe fetal acidemia/distress.
  * `y_primary = 1` if pH <= 7.15 (Distress / Acidemia)
  * `y_primary = 0` if pH > 7.15 (Normal)

---

## 3. Signal Quality Assessment & Windowing

### 3.1 Sliding Window Configuration
* **Window Size**: 20 minutes. At 4 Hz, this translates to:
  `20 minutes * 60 seconds/minute * 4 Hz = 4,800 samples`
  This duration matches obstetric guidelines (FIGO/NICE), which require a minimum of 20 minutes of continuous tracing to evaluate CTG patterns.
* **Prediction Horizon**: Only signals from the **last 60 minutes** of a recording prior to delivery are selected. This ensures that early, healthy stages of labor are not labeled with a terminal distress class (preventing weak supervision).

### 3.2 Signal Quality Assessment (SQA)
Before interpolating, each candidate window must pass an SQA audit to avoid hallucinating physiological data:
* **Missing Ratio** = (Number of samples where FHR == 0.0) / (Total samples in window)
* **Rejection Rule**: If the Missing Ratio is greater than 30% (> 0.30), the window is discarded.
  * **Source Code**: [signal_quality.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/signal_quality.py)
  * **Key Functions**:
    * [assess_signal_quality](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/signal_quality.py#L3)
    * [get_valid_windows](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/signal_quality.py#L23)

---

## 4. Signal Filtering & Interpolation

### 4.1 Missing Data Interpolation
* **Target**: Continuous missing data blocks (gaps) of 15 seconds or less (<= 60 samples at 4 Hz).
* **Method**: **Cubic Spline Interpolation**.
  * A cubic spline is fitted on all valid (non-zero) data points in the window and used to fill the missing gaps.
  * Gaps longer than 15 seconds are left as 0.0 to prevent artifact generation from long dropouts.
  * **Source Code**: [interpolate_missing](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/filtering.py#L5)

### 4.2 High-Frequency Noise Filtering
* **Method**: Zero-phase low-pass Butterworth filter.
* **Parameters**:
  * **Order**: 4
  * **Cutoff Frequency**: 1.5 Hz
  * **Implementation**: Forward-backward filtering via `scipy.signal.filtfilt` to prevent temporal phase shift.
  * **Source Code**: [apply_lowpass_filter](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/filtering.py#L68)

---

## 5. FHR Baseline Estimation

To detect deviations (accelerations/decelerations) and normalize signals, a local FHR baseline must be estimated. The project implements two baseline methods:

### 5.1 Iterative Mean Baseline (Clinical Standard)
1. **Initial Estimate**: Compute the median FHR value across the 20-minute window:
   `Baseline_0 = median(FHR)`
2. **Refinement Iterations**: For 5 iterations, compute a new mean excluding significant deviations (outside the +/- 15 bpm FIGO threshold):
   * Select samples where: `(Baseline_prev - 15) <= FHR_sample <= (Baseline_prev + 15)`
   * Compute `Baseline_new = mean(selected_samples)`
3. **Clinical Rounding**: Round the final baseline value to the nearest 5 bpm increment (e.g., 132 bpm rounds to 130 bpm):
   `Final_Baseline = 5.0 * round(Baseline_5 / 5.0)`
* **Source Code**: [calculate_iterative_baseline](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/baseline.py#L33)

### 5.2 Asymmetric Least Squares (ALS) Baseline (Advanced Robust Method)
A second-order regularization framework that smooths the signal while ignoring positive peaks (accelerations) and negative valleys (decelerations) by balancing asymmetry weights:
* Minimizes: `sum( w[i] * (y[i] - z[i])^2 ) + lambda * sum( (z[i] - 2*z[i-1] + z[i-2])^2 )`
  Where:
  * `y` is the raw FHR signal.
  * `z` is the estimated baseline.
  * `lambda` is the smoothness parameter (set to 1e5).
  * `w` is the asymmetry weight vector recomputed iteratively (10 iterations):
    `w[i] = p` if `y[i] > z[i]` (above baseline)
    `w[i] = 1 - p` if `y[i] < z[i]` (below baseline)
    For FHR baseline estimation, `p = 0.5` balances symmetric peak/valley rejection.
* **Source Code**: [asymmetric_least_squares_baseline](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/baseline.py#L5)

---

## 6. Clinical Feature Extraction

A set of clinical descriptors are extracted from each 20-minute window. These parameters represent the standard metrics used by obstetricians.

### 6.1 Short-Term Variability (STV)
STV measures the beat-to-beat variability (microscale fluctuations). It is computed as the mean absolute difference of successive FHR values:
`STV = mean( | FHR[t+1] - FHR[t] | )`
* **Source Code**: [calculate_variability](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/features.py#L5)

### 6.2 Long-Term Variability (LTV)
LTV measures broader changes over minutes (macroscale fluctuations).
1. Divide the 20-minute signal into 1-minute non-overlapping epochs (240 samples per epoch).
2. Compute the peak-to-peak amplitude (range) for each epoch `j`:
   `Range[j] = max(FHR_epoch_j) - min(FHR_epoch_j)`
3. Compute the overall LTV as the mean of these 20 epoch ranges:
   `LTV = mean(Range[0], Range[1], ..., Range[19])`
* **Source Code**: [calculate_variability](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/features.py#L24)

### 6.3 Accelerations
* **Definition**: An increase in FHR above the estimated baseline by 15 bpm or more, lasting for a continuous duration of 15 seconds or more (>= 60 samples at 4 Hz).
* **Source Code**: [detect_accelerations](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/features.py#L39)

### 6.4 Decelerations
* **Definition**: A decrease in FHR below the estimated baseline by 15 bpm or more, lasting for a continuous duration of 15 seconds or more.
* **Categorization Rules**:
  1. **Prolonged Deceleration**: Duration of the drop is 2 minutes or more (>= 120 seconds).
  2. **Variable Deceleration**: Rapid descent phase where the time from onset to nadir (lowest FHR value) is less than 30 seconds.
  3. **Early Deceleration**: Gradual descent phase (onset to nadir >= 30 seconds) where the nadir occurs in phase with the uterine contraction (UC) peak (delay between nadir and UC peak <= 15 seconds).
  4. **Late Deceleration**: Gradual descent phase (onset to nadir >= 30 seconds) where the nadir is delayed relative to the UC peak (delay between nadir and UC peak > 15 seconds).
* **Uterine Contraction Peak Detection**: Computed using `scipy.signal.find_peaks` on the filtered UC signal with a minimum peak distance of 30 seconds (120 samples) and a minimum peak prominence of 10 mmHg.
* **Source Code**: [detect_decelerations](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/features.py#L71)

---

## 7. Knowledge Rule Engine (Programmatic FIGO Classification)

To guide the network with auxiliary clinical representations, the extracted baseline, LTV, accelerations, and deceleration counts are passed through a programmatic logic engine mapping to the standard **FIGO 2015 Guidelines**:

* **Normal (Class 0)**:
  * Baseline FHR is between 110 bpm and 160 bpm (inclusive).
  * LTV is between 5 bpm and 25 bpm (inclusive).
  * No late decelerations and no prolonged decelerations present.
* **Pathological (Class 2)**:
  * Baseline FHR is less than 100 bpm.
  * Or, any prolonged decelerations are present.
  * Or, any late decelerations are present while variability is reduced (LTV < 5 bpm).
* **Suspicious (Class 1)**:
  * Lacks one normal characteristic but does not meet pathological criteria (e.g., baseline slightly outside 110–160 bpm, LTV slightly outside 5–25 bpm, or presence of variable/late decelerations without reduced variability).

* **Source Code**: [figo.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/knowledge/figo.py)
* **Key Function**: [classify_figo](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/knowledge/figo.py#L3)

---

## 8. Splitting, Imbalance Handling, & Normalization

### 8.1 Patient-Level Stratified Split
To avoid data leakage, records are divided at the patient level before window extraction.
* **Splits**: Train 70%, Validation 15%, Test 15%.
* **Leakage Protection**: Ensures no overlapping windows from the same patient appear across the Train, Validation, and Test sets.
* **Stratification**: Done based on the patient-level final clinical outcome (pH <= 7.15) to preserve the target ratio in all sets.

### 8.2 Class Imbalance Mitigation (Training Only)
The dataset exhibits severe class imbalance (fetal distress is rare). During training window generation, a **dynamic stride** is applied:
* **Distress Patients** (pH <= 7.15): Extracted using a **2-minute stride** (480 samples) to generate highly overlapping minority windows (oversampling).
* **Normal Patients** (pH > 7.15): Extracted using a **10-minute stride** (2400 samples) to reduce majority window generation.
* **Validation/Testing Splits**: Extracted using a fixed **10-minute stride** for all patients, ensuring evaluation metrics remain mathematically unbiased.

### 8.3 Input Channel Normalization
To prevent neural networks from learning absolute offset biases and to stabilize gradients, the FHR channel is baseline-corrected:
`FHR_Normalized[i] = FHR[i] - Baseline[i]`
This centers the signal around 0.0, highlighting local dynamic acceleration and deceleration behaviors. The input tensor is constructed by stacking these channels:
* **Channel 0**: Baseline-corrected FHR.
* **Channel 1**: Filtered UC.
* **Final Input Shape**: `(2, 4800)`

* **Source Code**: [pipeline.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/pipeline.py)
* **Key Function**: [process_pipeline](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/pipeline.py#L17)

---

## 9. Output Tensor Specifications

The preprocessing pipeline writes three `.pt` files (`train_dataset.pt`, `val_dataset.pt`, `test_dataset.pt`) containing dictionary objects:

```python
{
    'X': torch.Tensor,          # shape (N, 2, 4800), dtype=float32 (Normalized Input channels)
    'y_primary': torch.Tensor,    # shape (N,), dtype=long (Binary outcome: 0 = Normal, 1 = Distress)
    'y_figo': torch.Tensor,       # shape (N,), dtype=long (FIGO class: 0 = Normal, 1 = Suspicious, 2 = Pathological)
    'y_features': torch.Tensor,   # shape (N, 8), dtype=float32 (Continuous clinical descriptors)
    'metadata': List[Tuple]       # List of (record_id, start_idx, end_idx)
}
```

### Feature Vector Index Map (`y_features`):
* `y_features[:, 0]`: **Baseline FHR** (bpm)
* `y_features[:, 1]`: **STV** (bpm)
* `y_features[:, 2]`: **LTV** (bpm)
* `y_features[:, 3]`: **Accelerations Count**
* `y_features[:, 4]`: **Early Decelerations Count**
* `y_features[:, 5]`: **Late Decelerations Count**
* `y_features[:, 6]`: **Variable Decelerations Count**
* `y_features[:, 7]`: **Prolonged Decelerations Count**

---

## 10. Verification and Consistency Auditing

A consistency check verifies the logical mapping between the programmatically extracted features and the resulting tensors:
1. Validates that no patient IDs intersect between train, val, and test splits (absolute leak prevention).
2. Verifies that all pathological FIGO classifications (`y_figo == 2`) have matching diagnostic features (e.g., baseline < 100, prolonged decelerations, or late decelerations combined with reduced variability).
* **Source Code**: [consistency_audit.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/consistency_audit.py)
* **Key Function**: [run_consistency_audit](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/consistency_audit.py#L5)
