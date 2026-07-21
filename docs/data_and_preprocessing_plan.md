# Dataset Organization and Preprocessing Plan

## 1. Dataset Organization

The two main datasets are extracted and organized within the `data/raw/` directory to maintain original integrity before any processing.

**Target Structure:**
```
data/
├── raw/
│   ├── cardiotocography/
│   │   └── CTG.xls / cardiotocography.csv (UCI SisPorto tabular data)
│   └── ctu-chb-intrapartum/
│       ├── .dat, .hea PhysioNet files
│       └── clinical_metadata.csv
└── processed/
    ├── train_dataset.pt, val_dataset.pt, test_dataset.pt
    ├── ctu_signal_scaler.npz
    ├── uci_train_dataset.pt, uci_val_dataset.pt, uci_test_dataset.pt
    └── uci_scaler.joblib
```

## 2. Preprocessing Pipeline (`src/preprocessing/`)

Preprocessing transforms the raw signals and tabular clinical data into normalized, windowed tensors ready for PyTorch deep learning models and classical ML baselines. All outputs are saved to `data/processed/`.

### Step 1: Data Ingestion & Resampling
- **Time-Series (CTU-CHB)**: Read continuous FHR and UC signals using `wfdb`. Validate sampling rate against `TARGET_FS = 4.0 Hz` and apply polyphase integer-ratio resampling (`scipy.signal.resample_poly`) if necessary (`ingestion.py`).
- **Tabular Features (UCI SisPorto)**: Parse pre-extracted SisPorto 2.0 features using `pandas` (`uci_pipeline.py`).

### Step 2: Signal Quality & Filtering
- **Signal Quality Assessment (SQA)**: Discard any 20-minute window with >30% missing signal (`signal_quality.py`).
- **Spike Removal**: Detect and zero-out unphysiological rate-of-change spikes (>25 bpm/sec, i.e., >6.25 bpm per sample at 4 Hz) using `remove_spikes()` (`filtering.py`).
- **Missing Value Interpolation**: Reconstruct gaps $\le 15$ seconds (60 samples) using **Cubic Spline Interpolation** (`interpolate_missing`). Gaps > 15 seconds remain 0.0.
- **Noise Filtering**: Apply a zero-phase 4th-order low-pass Butterworth filter at 1.5 Hz cutoff (`apply_lowpass_filter`).

### Step 3: Baseline Extraction & Feature Engineering
- **Baseline Extraction**: Compute dynamic FHR baseline via iterative mean excluding $\pm 15$ bpm excursions, rounded to the nearest 5 bpm (`baseline.py`).
- **Feature Engineering**: Compute Short-Term Variability (STV), Long-Term Variability (LTV), Accelerations, and Early/Late/Variable/Prolonged Decelerations (`features.py`).
- **FIGO Pseudo-Labels**: Assign 3-class FIGO categories (Normal=0, Suspicious=1, Pathological=2) via programmatic rule engine (`knowledge/figo.py`).

### Step 4: Sliding Window Generation & Prediction Horizon
- Extract 20-minute consecutive windows ($4,800$ samples at 4 Hz) from the final 60 minutes of labor.
- **Strict 30-Minute Prediction Horizon**: Assign `y_primary = 1` only to windows starting within the final 30 minutes before delivery for distress patients ($\text{pH} \le 7.15$). Earlier windows receive `y_primary = 0`.
- **Training Imbalance Handling**: Apply 2-minute stride for distress cases and 10-minute stride for normal cases during training. Fixed 10-minute stride for val/test sets.

### Step 5: Normalization & Stratified Splitting
- Perform a **Patient-Level Stratified Split** (Train 70% / Val 15% / Test 15%) to prevent data leakage.
- **Per-Channel Z-Score Normalization**: Subtract mean and divide by std per channel (Baseline-corrected FHR diff and UC). Fit scaler **exclusively on the training split** and persist as `ctu_signal_scaler.npz`.
- **UCI Tabular Normalization**: Fit `StandardScaler` on training split only and persist as `uci_scaler.joblib`.

### Step 6: 10-Point Consistency Audit
- Execute `src/preprocessing/consistency_audit.py` to verify tensor shapes, NaN/Inf absence, Z-score unit variance, patient leakage isolation, FIGO-pH cross-tabulation, and scaler artifact existence.
