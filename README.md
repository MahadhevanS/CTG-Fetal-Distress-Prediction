# Knowledge-Infused Multi-Task Temporal Deep Learning for CTG Fetal Distress Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Abstract & Clinical Motivation

Intrapartum fetal distress due to hypoxia and fetal acidemia ($\text{pH} \le 7.15$) is a leading cause of preventable neonatal morbidity and mortality. Cardiotocography (CTG), which records continuous Fetal Heart Rate (FHR) and Uterine Contractions (UC), is the global clinical standard for intrapartum monitoring. However, conventional visual CTG interpretation suffers from high inter-observer variability and high false-alarm rates ($>60\%$), driving unnecessary emergency cesarean deliveries.

This project introduces a **Knowledge-Infused Multi-Task Deep Learning Framework** that combines state-of-the-art temporal encoders (1D CNN, BiLSTM, GRU, TCN, Multi-Scale LSTM, PatchCTG, and PatchTST) with clinical domain knowledge. By unifying continuous waveform modeling, FIGO diagnostic rule engines, and physiological feature extraction (STV, LTV, accelerations/decelerations), the system delivers transparent, highly sensitive fetal distress predictions while maintaining strict GE Patent US12094611B2 non-infringement boundary conditions (end-to-end signal representation without bounding boxes or shape-matching correlation loops).

---

## 🔬 Benchmark Results Summary (Phase 3)

All 7 temporal encoders were evaluated under Stratified 5-Fold Patient-Level Cross-Validation (6,917 total windows across 546 unique patients) with dynamic class weighting ($N_{\text{neg}} / N_{\text{pos}} \approx 5.08 - 5.29$) to strictly eliminate patient data leakage and address severe class imbalance.

| Model | Architecture Type | Params | 5-Fold CV AUROC | 5-Fold CV AUPRC | 5-Fold CV F1 | Recall / Sens (%) | Specificity (%) | Sens @ 90% Spec (%) | Key Clinical Trait |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN1D** | 1D Residual CNN | 135K | 0.6860 ± 0.0503 | 0.2560 ± 0.0544 | 0.3304 ± 0.0511 | 42.09 ± 11.19 | 78.68 ± 7.85 | 21.09 ± 10.76 | Ultra-fast local convolution stem |
| **BiLSTM** | Bidirectional LSTM | 159K | 0.6544 ± 0.0409 | 0.2697 ± 0.0426 | 0.3435 ± 0.0338 | 61.55 ± 7.10 | 61.87 ± 7.70 | 24.19 ± 6.92 | Global sequential tracking (High Recall) |
| **GRU** | Gated Recurrent Unit | 179K | 0.6881 ± 0.0627 | 0.2812 ± 0.0839 | 0.3027 ± 0.1080 | 34.00 ± 19.05 | 85.44 ± 9.22 | 25.05 ± 9.96 | Gated recurrence + high specificity |
| **TCN** | Temporal Conv Network | 363K | 0.7154 ± 0.0797 | 0.2846 ± 0.0840 | 0.2413 ± 0.1079 | 21.28 ± 11.05 | **90.57 ± 3.90** | 25.65 ± 13.36 | Causal dilated conv (Peak fold AUROC 0.843) |
| **MS-LSTM** | Multi-Scale BiLSTM | 584K | 0.7263 ± 0.1103 | 0.3140 ± 0.0962 | 0.3668 ± 0.0666 | **69.68 ± 24.77** | 61.61 ± 12.62 | 29.94 ± 12.13 | Multi-resolution temporal receptive fields |
| **PatchCTG** | Joint-Channel Patch Trans. | 479K | 0.6668 ± 0.0161 | 0.2872 ± 0.0334 | 0.3322 ± 0.0418 | 43.19 ± 11.69 | 77.85 ± 8.34 | 28.63 ± 5.06 | Joint-channel patch transformer baseline |
| **PatchTST** 🏆 | Channel-Ind. Patch Trans. | 685K | **0.7504 ± 0.0378** | **0.3820 ± 0.0800** | **0.4102 ± 0.0446** | 63.74 ± 12.12 | 71.54 ± 9.69 | **35.09 ± 6.65** | **SOTA Winner**: Highest AUROC, AUPRC & Sens@90%Spec |

*Note: PatchTST achieved the highest 5-Fold Patient-Level CV AUROC (0.7504 ± 0.0378), AUPRC (0.3820 ± 0.0800), F1 Score (0.4102 ± 0.0446), and Sensitivity at 90% Specificity (35.09% ± 6.65%), establishing it as the official temporal encoder backbone for the Phase 4 Knowledge-Infused Multi-Task Framework.*

---

## 📊 Dataset Citations & Links

1. **PhysioNet CTU-CHB Intrapartum CTG Database**:
   - 552 intrapartum CTG recordings (sampling rate 4.0 Hz) paired with clinical outcomes (umbilical artery pH, Apgar scores, delivery mode).
   - **DOI**: [10.13026/C2188R](https://doi.org/10.13026/C2188R)
   - **Citation**: Chudáček V. et al., *Open access intrapartum CTG database*, BMC Pregnancy and Childbirth, 2014.

2. **UCI Machine Learning Repository — Cardiotocography Dataset**:
   - 2,126 pre-extracted 21-feature SisPorto 2.0 records used for classical baseline comparisons.
   - **Link**: [UCI Cardiotocography Repository](https://archive.ics.uci.edu/dataset/193/cardiotocography)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/MahadhevanS/CTG-Fetal-Distress-Prediction.git
cd CTG-Fetal-Distress-Prediction
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Windows (PowerShell / Command Prompt)
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt

# Linux / macOS / Google Colab
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Automated Preprocessing Pipeline (v1.0 Frozen)
To execute the automated end-to-end signal processing, baseline estimation, spike removal, and patient-stratified window extraction:
```bash
python src/preprocessing/run_all.py
```
To run the 10-point mathematical consistency audit on generated tensors:
```bash
python src/preprocessing/consistency_audit.py
```

---

## 💻 Training & Evaluation Protocol

### Local / CLI Execution
Run patient-stratified 5-fold cross-validation for any encoder using `src/training/train.py`:

```bash
# Dry-run shape & gradient contract validation for all models
python src/training/train.py --model all --dry_run

# 5-Fold Stratified Patient CV for PatchTST
python src/training/train.py --model patchtst --epochs 15 --batch_size 32

# 5-Fold Stratified Patient CV for CNN1D
python src/training/train.py --model cnn1d --epochs 15 --batch_size 32

# 5-Fold Stratified Patient CV for BiLSTM
python src/training/train.py --model bilstm --epochs 15 --batch_size 32
```

### ☁️ Google Colab GPU Execution (Accelerated)
Run the commands directly in Google Colab (T4 / V100 GPU):
```bash
!git clone https://github.com/MahadhevanS/CTG-Fetal-Distress-Prediction.git
%cd CTG-Fetal-Distress-Prediction
!pip install -r requirements.txt

!python src/training/train.py --model cnn1d --epochs 15
!python src/training/train.py --model bilstm --epochs 15
```

---

## 📂 Project Structure

```text
CTG-Fetal-Distress-Prediction/
│
├── configs/                  # YAML environment configs (local.yaml vs colab.yaml)
├── checkpoints/              # Saved model weights per fold (best AUROC epoch)
├── data/
│   ├── raw/                  # Raw PhysioNet CTU-CHB & UCI SisPorto zips
│   └── processed/            # PyTorch dataset tensors (v1.0 Frozen)
├── docs/                     # 17 technical documentation files
│   ├── dataset_summary.md    # Tensor shapes, preprocessed stride specifications
│   ├── model_evaluation_plan.md # Clinical safety metrics protocol
│   └── model_inferences_log.md # Centralized single source of truth for benchmark metrics
├── notebooks/                # Jupyter EDA notebooks
├── src/
│   ├── models/               # Standardized temporal encoders (CNN1D, BiLSTM, GRU, TCN, MS-LSTM, PatchCTG, PatchTST)
│   ├── preprocessing/        # Filtering, baseline extraction, SQA, splitting
│   ├── knowledge/            # FIGO rule engine & clinical feature target generators
│   └── training/             # Universal 5-fold patient-stratified cross-validation loop (train.py)
├── AI_AGENT_RULES.md         # Mandatory contributor guidelines & patent boundaries
└── SANITY_CHECK_REVIEW.md    # Comprehensive expert audit & resolution roadmap
```

---

## 📜 License & Patent Boundaries

This repository is licensed under the MIT License. All model implementations maintain strict adherence to non-infringement boundary conditions for **GE Patent US12094611B2** by projecting continuous multi-channel signals directly to latent space $\mathbb{R}^{128}$ without bounding boxes or shape-matching correlation loops.
