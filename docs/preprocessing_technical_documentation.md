# CTG Fetal Distress Prediction: Preprocessing Technical Documentation

This document explains the data preprocessing pipeline designed for the **CTG Fetal Distress Prediction** framework. It details the clinical logic, data stream parameters, and algorithmic formulas used to transform raw cardiotocography (CTG) recordings and tabular clinical datasets into structured inputs for the machine learning models.

---

## 1. Pipeline Architecture Overview

The preprocessing pipeline ingests raw time-series data and clinical descriptors, filters out high-frequency noise, removes unphysiological transducer spikes, interpolates short missing segments, extracts key clinical features, assigns clinical pseudo-labels based on expert rules, applies a strict prediction horizon, normalises input channels with leak-free Z-scoring, and outputs structured PyTorch tensors alongside tabular ML baselines.

```text
Raw CTU-CHB Signals & Headers                   UCI SisPorto Dataset (.xls / .csv)
        │                                                     │
        ▼                                                     ▼
Signal Extraction: FHR & UC                         Drop Missing / Impute Medians
        │                                                     │
        ▼                                                     ▼
Sampling Rate Validation / Polyphase Resample (4 Hz)  Stratified Patient-Level Split (70/15/15)
        │                                                     │
        ▼                                                     ▼
Signal Quality Assessment (SQA) ──► [Reject >30% missing] Fit StandardScaler on TRAIN ONLY
        │                                                     │
        ▼                                                     ▼
Spike Removal (Rate of change > 25 bpm/sec)         Save UCI Tensors & uci_scaler.joblib
        │
        ▼
Cubic Spline Interpolation (Gaps <= 15 sec)
        │
        ▼
4th-Order Low-Pass Butterworth Filter (1.5 Hz)
        │
        ▼
Iterative FHR Baseline Estimation
        │
        ├───────────────────────────────┐
        ▼                               ▼
Baseline Subtraction & Stacking    Clinical Feature Extraction (STV, LTV, Accel/Decel)
        │                               │
        ▼                               ▼
Channel Stack (2, 4800)             Programmatic FIGO Classification Engine
        │                               │
        └───────────────┬───────────────┘
                        ▼
           Patient-Level Stratified Split
                        │
                        ▼
         Strict 30-Min Prediction Horizon
                        │
                        ▼
       Imbalance Handling (Dynamic Stride)
                        │
                        ▼
Per-Channel Z-Score Normalization (Fit TRAIN ONLY)
                        │
                        ▼
      Save CTU Tensors & ctu_signal_scaler.npz
```

---

## 2. Dataset Characteristics & Ingestion

The framework operates on two distinct dataset modalities:
1. **CTU-CHB Intrapartum Dataset**: Primary time-series repository for deep temporal learning.
2. **UCI SisPorto Cardiotocography Dataset**: Auxiliary tabular dataset for baseline classical ML (Random Forest, XGBoost) and external validation.

### 2.1 CTU-CHB Ingestion & Resampling (GAP 4 FIX)
* **Format**: `.dat` (binary signal data) and `.hea` (text headers) for each record.
* **Signals Extracted**:
  * **Fetal Heart Rate (FHR)**: Captured in beats per minute (bpm). Channel index `0`. Missing/dropout values set to `0.0`.
  * **Uterine Contractions (UC)**: Measured in mmHg. Channel index `1`.
  * **Sampling Frequency (fs)**: Standardised to **`TARGET_FS = 4.0 Hz`**.
* **Sampling Rate Validation & Resampling**:
  * If a record's header indicates $fs \neq 4.0\text{ Hz}$, the pipeline automatically resamples both FHR and UC signals using integer-ratio polyphase filtering (`scipy.signal.resample_poly`), preventing spectral distortion and ensuring uniform 4,800-sample 20-minute windows.
  * If $fs == 4.0\text{ Hz}$, an explicit assertion verifies rate integrity.
* **Metadata Parsing**: `.hea` text comments are parsed to extract umbilical artery pH values and Apgar scores.
  * **Source Code**: [ingestion.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/ingestion.py)
  * **Key Functions**:
    * [load_ctu_chb_record](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/ingestion.py#L19)
    * [load_clinical_metadata](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/ingestion.py#L107)

### 2.2 Clinical Target Definition
* **Primary Target (y_primary)**: Umbilical artery pH $\leq$ 7.15. This is the clinical threshold indicating moderate-to-severe fetal acidemia/distress.
  * `y_primary = 1` if pH $\leq$ 7.15 (Distress / Acidemia)
  * `y_primary = 0` if pH > 7.15 (Normal)

---

## 3. Signal Quality Assessment & Windowing

### 3.1 Sliding Window & Prediction Horizon Configuration (GAP 2 FIX)
* **Window Size**: 20 minutes ($20 \text{ min} \times 60 \text{ sec} \times 4 \text{ Hz} = 4,800 \text{ samples}$). Matches FIGO/NICE guidelines requiring at least 20 minutes to evaluate baseline and variability.
* **Clinically Relevant Horizon**: Analysis is restricted to the final **60 minutes** of recording prior to delivery (`LAST_HOUR_SAMPLES = 14,400`).
* **Strict 30-Minute Prediction Horizon (GAP 2 FIX)**:
  * Fetal distress and acidemia progress dynamically during active labor. Labeling early windows from a distress patient as "Distress" introduces weak supervision label noise.
  * **Rule**: Only windows starting within the final **30 minutes** before delivery (`PREDICTION_HORIZON_SAMPLES = 7,200`) receive `y_primary = 1` for distress patients. Windows starting earlier than 30 minutes before delivery receive `y_primary = 0`.

### 3.2 Signal Quality Assessment (SQA)
Before interpolating, each candidate window must pass an SQA audit:
* **Missing Ratio** = $\frac{\text{Count}(FHR == 0.0)}{\text{Total Window Samples}}$
* **Rejection Rule**: If Missing Ratio > 30% (> 0.30), the window is discarded.
  * **Source Code**: [signal_quality.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/signal_quality.py)

---

## 4. Signal Filtering & Artifact Cleaning

### 4.1 Spike Removal (GAP 1 FIX)
* **Physiological Limit**: Human fetal heart rate changes cannot exceed 25 bpm/second ($6.25 \text{ bpm}$ per sample at 4 Hz).
* **Method**: `remove_spikes()` detects sample-to-sample deltas $> 6.25 \text{ bpm}$. Spike samples are zeroed out (treated as missing) so they are smoothly reconstructed by cubic spline interpolation, preventing artificial acceleration/deceleration peak detection.
  * **Source Code**: [remove_spikes](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/filtering.py#L9)

### 4.2 Missing Data Interpolation
* **Target**: Continuous missing data gaps $\leq 15 \text{ seconds}$ ($\leq 60 \text{ samples}$ at 4 Hz).
* **Method**: **Cubic Spline Interpolation** (`interpolate_missing`). Preserves first and second derivatives (smooth cardiac transitions). Gaps > 15 seconds remain $0.0$.
  * **Source Code**: [interpolate_missing](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/filtering.py#L60)

### 4.3 High-Frequency Noise Filtering
* **Method**: Zero-phase 4th-order low-pass Butterworth filter (`apply_lowpass_filter`) with cutoff frequency 1.5 Hz applied via `scipy.signal.filtfilt` (zero phase shift).
  * **Source Code**: [apply_lowpass_filter](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/filtering.py#L140)

---

## 5. FHR Baseline Estimation

### 5.1 Iterative Mean Baseline (Clinical Standard)
1. Initial baseline = median(FHR).
2. Refine over 5 iterations by taking the mean of samples within $\pm 15 \text{ bpm}$ of the previous baseline estimate.
3. Round final baseline to nearest 5 bpm increment.
  * **Source Code**: [calculate_iterative_baseline](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/baseline.py#L33)

### 5.2 Asymmetric Least Squares (ALS) Baseline
Regularized baseline estimation minimizing smoothness ($\lambda=10^5$) with asymmetric weights ($p=0.5$).
  * **Source Code**: [asymmetric_least_squares_baseline](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/baseline.py#L5)

---

## 6. Clinical Feature Extraction

* **STV (Short-Term Variability)**: Mean absolute beat-to-beat difference $\text{mean}(|FHR_{t+1} - FHR_t|)$.
* **LTV (Long-Term Variability)**: Mean peak-to-peak amplitude across 1-minute epochs.
* **Accelerations**: FHR $\geq$ baseline + 15 bpm for $\geq 15 \text{ seconds}$.
* **Decelerations**: Categorized into Early, Late, Variable (onset-to-nadir < 30 sec), and Prolonged ($\geq 2 \text{ minutes}$).
  * **Source Code**: [features.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/preprocessing/features.py)

---

## 7. Knowledge Rule Engine (FIGO 2015 Classification)

Programmatic rule engine mapping extracted descriptors to FIGO categories:
* **Normal (Class 0)**: Baseline 110–160 bpm, LTV 5–25 bpm, no late/prolonged decelerations.
* **Suspicious (Class 1)**: Lacks one normal metric without meeting pathological criteria.
* **Pathological (Class 2)**: Baseline < 100 bpm, prolonged decelerations present, or late decelerations with LTV < 5 bpm.
  * **Source Code**: [figo.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/knowledge/figo.py)

---

## 8. Splitting, Imbalance Handling, & Normalization

### 8.1 Patient-Level Stratified Split
Records are split 70% Train / 15% Val / 15% Test at the patient level before window extraction to guarantee zero data leakage.

### 8.2 Class Imbalance Mitigation (Training Only)
Dynamic sliding window stride during training:
* **Distress Patients**: 2-minute stride (minority oversampling).
* **Normal Patients**: 10-minute stride (majority undersampling).
* **Val/Test Splits**: Fixed 10-minute stride (unbiased evaluation).

### 8.3 Per-Channel Z-Score Normalization (GAP 5 FIX)
After baseline subtraction ($FHR_{\text{norm}} = FHR - \text{Baseline}$), input channels are Z-score normalized across batch and temporal dimensions:
$$X_{\text{norm}}[:, c, :] = \frac{X[:, c, :] - \mu_c}{\sigma_c}$$
* **Leak-Free Protocol**: Mean $\mu_c$ and std $\sigma_c$ are computed **exclusively on the training split** and applied to val/test sets.
* **Scaler Persistence**: Saved to `data/processed/ctu_signal_scaler.npz`.

---

## 9. UCI SisPorto Tabular Pipeline (GAP 3 FIX)

Implemented in `src/preprocessing/uci_pipeline.py` for classical ML baselines:
* Loads 2,126 records with 21 SisPorto features (`LB`, `AC`, `FM`, `UC`, `DL`, `DS`, `DP`, `ASTV`, `MSTV`, etc.).
* Drops missing NSP rows, applies median imputation for feature NaNs.
* Stratified 70/15/15 patient-level split based on 3-class NSP.
* Fits `StandardScaler` on training split only, saved to `data/processed/uci_scaler.joblib`.
* Outputs `uci_train_dataset.pt`, `uci_val_dataset.pt`, `uci_test_dataset.pt`.

---

## 10. Output Tensor Specifications

Located in `data/processed/`:

### CTU-CHB Time-Series Tensors (`train_dataset.pt`, `val_dataset.pt`, `test_dataset.pt`):
```python
{
    'X': torch.Tensor,          # shape (N, 2, 4800), dtype=float32 (Z-score normalized FHR-diff & UC)
    'y_primary': torch.Tensor,  # shape (N,), dtype=long (Binary outcome: 0 = Normal, 1 = Distress)
    'y_figo': torch.Tensor,     # shape (N,), dtype=long (FIGO class: 0 = Normal, 1 = Suspicious, 2 = Pathological)
    'y_features': torch.Tensor, # shape (N, 8), dtype=float32 (Continuous clinical descriptors)
    'metadata': List[Tuple]     # List of (record_id, start_idx, end_idx)
}
```

### Scaler Artifacts:
* `ctu_signal_scaler.npz`: Contains `mean` and `std` arrays of shape `(2,)`.
* `uci_scaler.joblib`: Fitted `StandardScaler` object for 21 SisPorto features.

---

## 11. Verification and 10-Point Consistency Audit

The audit script `src/preprocessing/consistency_audit.py` performs 10 automated checks:
1. **Tensor Shape & Dtype Audit**: Verifies `X` is `(N, 2, 4800)` `float32`, `y_primary` is `long`, `y_features` has 8 columns.
2. **NaN & Inf Audit**: Zero tolerance for missing or non-finite values in tensors.
3. **Label Range Audit**: Validates `y_primary` $\in \{0, 1\}$ and `y_figo` $\in \{0, 1, 2\}$.
4. **Train Balance Audit**: Confirms training distress ratio is within 35–65%.
5. **FIGO vs pH Cross-Tabulation**: Verifies FIGO Normal contamination by pH Distress is $< 10\%$.
6. **Pathological Feature Consistency**: Ensures $\ge 60\%$ of FIGO Pathological windows have explicit late/prolonged decelerations or baseline $<100 \text{ bpm}$.
7. **Z-Score Verification**: Confirms training set $FHR$ and $UC$ channels have $\mu \approx 0$ and $\sigma \approx 1$.
8. **Leakage Audit**: Verifies $\text{Train IDs} \cap \text{Val IDs} = \emptyset$, $\text{Train IDs} \cap \text{Test IDs} = \emptyset$, $\text{Val IDs} \cap \text{Test IDs} = \emptyset$.
9. **Scaler Artifact Audit**: Verifies existence and shape of `ctu_signal_scaler.npz` and `uci_scaler.joblib`.
10. **UCI Split Audit**: Validates shapes, feature names, and label distributions for `uci_*.pt` files.
