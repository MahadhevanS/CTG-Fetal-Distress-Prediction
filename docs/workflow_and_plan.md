# CTG Fetal Distress Prediction: Project Workflow and Plan

## 1. Project Overview
This project aims to build a Knowledge-Infused Multi-Task Temporal Deep Learning Framework for CTG-based fetal distress prediction. The architecture enforces strict separation of concerns, isolating preprocessing, model definition, domain knowledge (FIGO/NICHD rules), and the training pipeline.

## 2. Multi-Task Architecture & Knowledge Infusion

A core design decision is that the temporal encoder **does not classify FIGO classes directly** as its primary goal. Instead, the encoder learns a rich physiological representation, and the multi-task heads predict clinically meaningful outputs, guided by a single, internationally recognized clinical framework (FIGO).

### Proposed Multi-Task Classification
- **Task 1 (Primary - Early Fetal Distress Risk)**: Binary prediction (Normal fetus vs. Early fetal distress / acidemia risk based on pH < 7.15).
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
- **SisPorto Dataset**: Not used as a knowledge source. It is reserved for validating the automated feature extraction, providing classical machine learning baselines (Random Forest, XGBoost), and external validation.

## 3. Directory Structure and Responsibilities

- **`src/preprocessing/`**: Handles all data preparation (signal filtering, missing-value handling, normalization, sliding window generation). Must be executed locally.
- **`src/models/`**: Contains neural network architectures (temporal encoder, multi-task heads, clinical prior network).
- **`src/training/`**: Houses the training loop, validation, and checkpointing (`train.py`).
- **`src/knowledge/`**: Encodes clinical guidelines (FIGO rules, NICHD rules) and knowledge-guided loss functions.
- **`configs/`**: Stores environment configurations (`local.yaml`, `colab.yaml`).
- **`data/`**: Stores raw and locally processed datasets.
- **`checkpoints/`**: Stores trained model weights.

## 4. Workflow Specification

### Phase 1: Local Preprocessing
1. **Data Ingestion**: Raw datasets (`cardiotocography.zip`, `ctu-chb-intrapartum-cardiotocography-database-1.0.0.zip`) are extracted to `data/raw/`.
2. **Preprocessing Pipeline**: `src/preprocessing/` modules process the raw signals.
3. **Artifact Generation**: Cleaned, normalized, and windowed datasets are saved to `data/processed/`.

### Phase 2: Cloud Sync & Training Preparation
1. **Upload**: The contents of `data/processed/` are uploaded to a designated Google Drive directory.
2. **Environment Setup**: Google Colab mounts the Google Drive to access the processed dataset.

### Phase 3: Model Training (Colab Environment)
1. **Execution**: Training is initiated on Google Colab using:
   ```bash
   python src/training/train.py --config configs/colab.yaml
   ```
2. **Rules of Engagement**: 
   - `train.py` contains NO Colab-specific code (e.g., `google.colab.drive.mount`). Drive mounting must happen in a separate Colab cell *before* invoking the script.
   - All dataset paths and hyperparameters are injected via `configs/colab.yaml`.
3. **Checkpointing**: The training loop writes model weights and logs directly to the `checkpoints/` directory (which should be synced/mounted to Drive).

### Phase 4: Evaluation and Post-Processing
1. **Sync Back**: Checkpoints and metrics are copied from Google Drive back to the local `checkpoints/` directory.
2. **Local Inference/Analysis**: Model evaluation, visualization, and further analysis are performed locally using the downloaded weights.

## 5. Immediate Next Steps (Roadmap)
1. **Extract and Explore Datasets**: Unzip and perform EDA on the raw CTG datasets.
2. **Implement Preprocessing**: Build robust signal filtering and sliding window generation in `src/preprocessing/`.
3. **Draft Model Architecture**: Implement the Temporal Encoder and Multi-task heads in `src/models/`.
4. **Encode Domain Knowledge**: Translate FIGO/NICHD rules into programmatic constraints or loss penalties in `src/knowledge/`.
5. **Develop Training Loop**: Finalize `src/training/train.py` to ensure it seamlessly supports both local debugging (`local.yaml`) and Colab execution (`colab.yaml`).
