# Preprocessed Dataset Summary (v1.0 Frozen)

This document outlines the structure, statistics, and clinical dynamics of the generated PyTorch datasets and scaler artifacts for the CTG Fetal Distress Prediction framework.

## 1. Global Overview

The raw PhysioNet CTU-CHB Intrapartum dataset contains **552 clinical recordings**. Clinical metadata, including umbilical artery pH and Apgar scores, were parsed from the WFDB headers and linked to each recording, generating time-series windows strictly mapped to these physiological outcomes. The auxiliary UCI SisPorto dataset contains **2,126 records** of pre-extracted 21 SisPorto features for classical ML baselines.

### Patient-Level Stratified Splitting
To prevent data leakage, both datasets were strictly split at the patient level before window extraction/scaling, ensuring no overlapping physiological data exists across evaluation boundaries.

- **CTU-CHB Total Patients**: 552
  - **Training Set**: 386 patients (~70%)
  - **Validation Set**: 83 patients (~15%)
  - **Testing Set**: 83 patients (~15%)
- **UCI SisPorto Records**: 2,126
  - **Training Split**: 1,488 records (~70%)
  - **Validation Split**: 319 records (~15%)
  - **Testing Split**: 319 records (~15%)

---

## 2. Dataset Statistics & Imbalance Mitigation

Due to severe class imbalance (Pathological fetal distress is rare compared to Normal outcomes), an overlap-based minority generation strategy was enforced exclusively on the CTU-CHB training split.

### Window Extraction & Labeling Rules
- **Clinically Relevant Horizon**: Windows extracted from the **last 60 minutes** prior to delivery (`LAST_HOUR_SAMPLES = 14,400`).
- **Strict 30-Minute Prediction Horizon (GAP 2 FIX)**: Only windows starting within the final **30 minutes** before delivery (`PREDICTION_HORIZON_SAMPLES = 7,200`) receive `y_primary = 1` for distress patients. Windows starting earlier receive `y_primary = 0` to eliminate weak supervision label noise.
- **Training Stride**: 
  - Distress cases: 30-second (0.5 minute) stride (heavy overlap/oversampling for minority class, creating ~1,200 distress windows).
  - Normal cases: 10-minute stride (sparse overlap, creating ~2,000 normal windows).
- **Validation/Test Stride**: Fixed 10-minute stride for all cases to ensure metrics reflect true, unbiased real-world distributions.

---

## 3. Data Structure and Shapes

Located in `data/processed/`:

### A. CTU-CHB Time-Series Input Tensor (`X` in `train_dataset.pt`, `val_dataset.pt`, `test_dataset.pt`)
- **Shape**: `(N, 2, 4800)`
  - `N`: Number of windows in the split.
  - `2`: Channels (Channel 0: Baseline-Corrected FHR diff, Channel 1: Filtered UC).
  - `4800`: Temporal sequence length (20 minutes $\times$ 60 seconds $\times$ 4 Hz sampling rate).
- **Preprocessing Applied**:
  - Polyphase resampling to exact 4.0 Hz (`resample_poly`).
  - 30% Missing Signal Threshold Rejection (SQA).
  - **Spike Removal**: Rate-of-change $> 25 \text{ bpm/sec}$ ($>6.25 \text{ bpm/sample}$) zeroed out.
  - $<15$ sec gaps interpolated via **Cubic Splines**.
  - **4th-Order** Butterworth Low-Pass Filter (cutoff 1.5 Hz, zero-phase `filtfilt`).
  - *Baseline Subtraction* applied to the FHR channel.
  - **Per-Channel Z-Score Normalization**: Fitted on training set only, persisted in `ctu_signal_scaler.npz`.

### B. Primary Target Tensor (`y_primary`)
- **Shape**: `(N,)`, Data Type: `torch.float32` (stored as `torch.long` in `.pt`, cast to `float32` for `BCEWithLogitsLoss`).
- **Meaning**: Binary Terminal Acidemia Outcome (0 = Normal / pH > 7.15, 1 = Distress / Acidemia / pH $\leq$ 7.15 within 30-min horizon).

### C. Knowledge-Guided Supervisory Targets (`y_figo` and `y_features`)
- **`y_figo`**: `(N,)` `torch.long` — Categorical FIGO Class (0 = Normal, 1 = Suspicious, 2 = Pathological).
- **`y_features`**: `(N, 8)` `torch.float32` — Continuous physiological features:
  - `[0]`: Baseline FHR (bpm)
  - `[1]`: STV (bpm)
  - `[2]`: LTV (bpm)
  - `[3]`: Accelerations Count
  - `[4]`: Early Decelerations Count
  - `[5]`: Late Decelerations Count
  - `[6]`: Variable Decelerations Count
  - `[7]`: Prolonged Decelerations Count

### D. UCI SisPorto Tensors (`uci_train_dataset.pt`, `uci_val_dataset.pt`, `uci_test_dataset.pt`)
- **`X`**: `(N, 21)` `torch.float32` — Z-score normalized pre-extracted SisPorto 2.0 features (scaler saved in `uci_scaler.joblib`).
- **`y_figo`**: `(N,)` `torch.long` — 3-class NSP label (0 = Normal, 1 = Suspect, 2 = Pathological).
- **`y_binary`**: `(N,)` `torch.long` — Binary outcome (0 = Normal, 1 = Pathological; Suspect rows set to -1).

---

## 4. Methodological Highlights & Scaler Artifacts

1. **Leakage Prevention**: Patient-level splitting before window extraction or scaler fitting guarantees zero data leakage across training, validation, and test boundaries.
2. **Clinical Authenticity**: Zero synthetic signal generation (no SMOTE on waveforms).
3. **Persisted Scaler Artifacts**:
   - `data/processed/ctu_signal_scaler.npz`: Contains per-channel `mean` and `std` vectors (shape `(2,)`).
   - `data/processed/uci_scaler.joblib`: Scikit-learn `StandardScaler` fitted on 21 SisPorto features.
4. **10-Point Consistency Audit**: Enforced via `src/preprocessing/consistency_audit.py` covering tensor shape/dtype, NaN/Inf checks, Z-score validation, patient leakage isolation, and clinical FIGO feature consistency.

---

## 5. Next Phase: Multi-Task Architecture

The pipeline now shifts from data engineering to model architecture and learning:

```text
Preprocessed Dataset (v1.0 Frozen)
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
Multi-Task Loss (Cross-Entropy + Knowledge Loss)
```
