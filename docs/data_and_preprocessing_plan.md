# Dataset Organization and Preprocessing Plan

## 1. Dataset Organization

The two main datasets will be extracted and organized within the `data/raw/` directory to maintain original integrity before any processing.

**Target Structure:**
```
data/
└── raw/
    ├── cardiotocography/
    │   ├── raw files (.xls, .csv, etc.)
    │   └── dataset_documentation/
    └── ctu-chb-intrapartum/
        ├── signals/ (.dat, .hea PhysioNet files)
        └── clinical_metadata/
```

## 2. Preprocessing Pipeline (`src/preprocessing/`)

Preprocessing transforms the raw signals and tabular clinical data into normalized, windowed tensors ready for the PyTorch temporal encoder. All outputs will be saved to `data/processed/`.

### Step 1: Data Ingestion & Parsing
- **Time-Series (CTU-CHB)**: Read the continuous Fetal Heart Rate (FHR) and Uterine Contraction (UC) signals using the `wfdb` library.
- **Tabular Features (UCI)**: Load the pre-extracted morphological features and clinical outcomes using `pandas`.

### Step 2: Signal Quality & Filtering
- **Missing Value Handling**: Identify signal dropout (often recorded as 0s). Use spline or polynomial interpolation for short gaps (< 15s). Flag or split windows containing prolonged gaps.
- **Artifact Removal**: Apply a low-pass Butterworth filter to smooth the FHR and UC signals, removing high-frequency noise caused by maternal movement or sensor displacement.
- **Baseline Extraction**: Compute the dynamic FHR baseline using a rolling median window, which is critical for defining accelerations and decelerations (FIGO rules).

### Step 3: Feature Engineering & Normalization
- Calculate domain-specific features (e.g., Short Term Variability (STV), Long Term Variability (LTV)) based on the extracted baseline.
- **Normalization**: Apply Z-score standardization (zero mean, unit variance) independently to FHR and UC signals to stabilize deep learning gradients. Ensure the scaler is fit *only* on the training set.

### Step 4: Sliding Window Generation
- Convert the continuous CTU-CHB time-series into discrete temporal windows (e.g., 20-minute windows with a 5-minute stride).
- Align each temporal window with its corresponding clinical outcome label (e.g., fetal distress indicated by pH < 7.15 or Apgar < 7).

### Step 5: Stratified Splitting & Export
- Perform a **Patient-Level Split** (Train 70% / Val 15% / Test 15%). It is crucial to split by patient ID, not by window, to prevent data leakage.
- Export the final, structured tensors to `data/processed/X_train.pt`, `y_train.pt`, etc., for seamless loading by PyTorch `DataLoader`s.
