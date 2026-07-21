# CTG Fetal Distress Prediction: Project Workflow and Plan

## 1. Project Overview
This project aims to build a Knowledge-Infused Multi-Task Temporal Deep Learning Framework for CTG-based fetal distress prediction. The architecture enforces strict separation of concerns, isolating preprocessing, model definition, domain knowledge (FIGO/NICHD rules), and the training pipeline.

## 2. Multi-Task Architecture & Knowledge Infusion

A core design decision is that the temporal encoder **does not classify FIGO classes directly** as its primary goal. Instead, the encoder learns a rich physiological representation, and the multi-task heads predict clinically meaningful outputs, guided by a single, internationally recognized clinical framework (FIGO).

### Proposed Multi-Task Classification
- **Task 1 (Primary - Early Fetal Distress Risk)**: Binary prediction (Normal fetus vs. Early fetal distress / acidemia risk based on pH $\leq$ 7.15 within a strict 30-minute terminal delivery horizon).
- **Task 2 (Clinical Feature Prediction)**: Predicts individual components (Baseline FHR, Variability, Accelerations, and specific Decelerations like Early, Late, Variable, Prolonged).
- **Task 3 (FIGO CTG Classification)**: 3-Class prediction (Normal / Suspicious / Pathological).

### Architecture Flow
```text
Temporal Encoder
        │
        ├───────────────► Distress Head (Binary)
        │
        ├───────────────► Clinical Feature Head (Regression + Binary)
        │
        └───────────────► FIGO Head (3-Class)
```

The outputs of the Clinical Feature Head are mapped against the FIGO Rule Engine to formulate the Knowledge Loss, creating a tight feedback loop that forces the neural features to align with clinical logic.

### Loss Formulation
The total training loss becomes:
$$L = L_{\text{distress}} + \lambda_1 L_{\text{clinical}} + \lambda_2 L_{\text{FIGO}} + \lambda_3 L_{\text{knowledge}}$$

### Role of NICE and SisPorto
To prevent conflicting supervisory signals and redundant learning:
- **NICE Guidelines**: Removed from training entirely. NICE will be used exclusively in the evaluation phase to demonstrate model robustness across different international interpretation frameworks.
- **SisPorto Dataset**: Processed separately via `src/preprocessing/uci_pipeline.py`. Reserved for validating automated feature extraction, providing classical machine learning baselines (Random Forest, XGBoost), and external validation.

## 3. Directory Structure and Responsibilities

- **`src/preprocessing/`**: Handles all data preparation (signal filtering, spike removal, 4 Hz polyphase resampling, cubic spline interpolation, low-pass Butterworth filtering, baseline subtraction, STV/LTV feature extraction, FIGO pseudo-labeling, strict 30-min prediction horizon, per-channel Z-score normalization, UCI SisPorto tabular scaling, and 10-point consistency auditing).
- **`src/models/`**: Contains neural network architectures (temporal encoder, multi-task heads, clinical prior network).
- **`src/training/`**: Houses the training loop, validation, and checkpointing (`train.py`).
- **`src/knowledge/`**: Encodes clinical guidelines (FIGO rules) and knowledge-guided loss functions (`figo.py`).
- **`configs/`**: Stores environment configurations (`local.yaml`, `colab.yaml`).
- **`data/`**: Stores raw and locally processed datasets (`data/processed/`).
- **`checkpoints/`**: Stores trained model weights.

## 4. Workflow Specification

### Phase 1: Local Preprocessing & Validation (COMPLETE ✅)
1. **Data Ingestion & Extraction**: `extract_datasets()` extracts `ctu-chb-intrapartum-cardiotocography-database-1.0.0.zip` and `cardiotocography.zip` to `data/raw/`.
2. **Metadata Aggregation**: `generate_metadata_from_headers()` parses WFDB headers into `clinical_metadata.csv`.
3. **CTU-CHB Pipeline Execution**: `process_pipeline()` performs 4 Hz polyphase resampling, spike removal (>25 bpm/sec), cubic spline interpolation, low-pass filtering, iterative baseline estimation, feature extraction, FIGO pseudo-labeling, patient-level 70/15/15 splitting, strict 30-min prediction horizon labeling, training-only dynamic stride, and per-channel Z-score normalization (persisting `ctu_signal_scaler.npz`).
4. **UCI SisPorto Pipeline Execution**: `preprocess_uci()` parses tabular CTG records, fits `StandardScaler` on training split only (persisting `uci_scaler.joblib`), and outputs `uci_train/val/test_dataset.pt`.
5. **10-Point Consistency Audit**: `consistency_audit.py` validates tensor shapes, NaN/Inf absence, Z-score unit variance, patient leakage isolation, FIGO-pH cross-tabulation, and scaler artifact existence.

### Phase 2: Cloud Sync & Training Preparation
1. **Upload**: The contents of `data/processed/` are uploaded to a designated Google Drive directory or Colab workspace.
2. **Environment Setup**: Google Colab mounts Google Drive to access the processed dataset tensors (`.pt`) and scaler artifacts (`.npz`, `.joblib`).

### Phase 3: Model Training (Colab Environment)
1. **Execution**: Training is initiated on Google Colab using:
   ```bash
   python src/training/train.py --config configs/colab.yaml
   ```
2. **Rules of Engagement**: 
   - `train.py` contains NO Colab-specific code (e.g., `google.colab.drive.mount`). Drive mounting occurs in a separate Colab cell *before* invoking the script.
   - All dataset paths and hyperparameters are injected via `configs/colab.yaml`.
3. **Checkpointing**: The training loop writes model weights and logs directly to the `checkpoints/` directory.

### Phase 4: Evaluation and Post-Processing
1. **Sync Back**: Checkpoints and metrics are copied from Google Drive back to the local `checkpoints/` directory.
2. **Local Inference/Analysis**: Model evaluation, visualization, and further analysis are performed locally using the downloaded weights.

## 5. Immediate Next Steps (Roadmap)
1. **[DONE] Extract and Preprocess Datasets**: Complete `run_all.py` and `consistency_audit.py`.
2. **Phase 2: Draft Model Architecture**: Implement Temporal Encoders (1D CNN, PatchTST / Temporal Transformer) and Multi-task heads in `src/models/`.
3. **Phase 3: Encode Domain Knowledge Loss**: Connect `figo_rule_loss` in `src/knowledge/figo.py` with multi-task prediction outputs.
4. **Phase 4: Develop Training Pipeline**: Finalize `src/training/train.py` to support multi-task loss optimization, validation metrics tracking (AUROC, AUPRC, Sensitivity @ 90% Specificity), and checkpoint saving.
