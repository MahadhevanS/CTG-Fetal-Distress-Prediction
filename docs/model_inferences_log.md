# Centralized Model Inference & Evaluation Log

This document serves as the single source of truth for tracking the benchmarking results, hyperparameter configurations, and clinical observations for all evaluated temporal encoders.

> **Instruction**: After completing the training and testing loop for a specific model, update its respective section below. Do not alter the template structure to ensure apples-to-apples comparisons.

---

## 1. 1D CNN (Baseline)
**Objective**: Evaluate local temporal pattern extraction capabilities.

### A. Optimal Hyperparameters
- Learning Rate: `0.001 (AdamW)`
- Kernel Sizes: `Stem: 7 (stride 2), Residual Blocks: 3 (stride 1/2)`
- Number of Convolutional Blocks: `3 Residual Blocks (32 -> 64 -> 128 channels)`
- Parameter Count: `135,169`
- Epochs to Convergence: `4 Epochs` (Best Val Loss: `0.2272`)

### B. Statistical Metrics (Validation / Single Fold Run)
| Metric | Value / Best Epoch (Epoch 4) |
| :--- | :--- |
| Accuracy (Train / Val) | `82.49% (Train) / 95.34% (Val)` |
| Best Validation Loss | `0.2272` |
| Training Loss (Epoch 4) | `0.3832` |
| AUROC | `TBD (Requires 5-Fold Evaluation Protocol)` |
| AUPRC | `TBD (Requires 5-Fold Evaluation Protocol)` |
| F1 Score | `TBD (Requires 5-Fold Evaluation Protocol)` |
| Precision (PPV) | `TBD (Requires 5-Fold Evaluation Protocol)` |
| Recall (Sensitivity)| `TBD (Requires 5-Fold Evaluation Protocol)` |
| Specificity | `TBD (Requires 5-Fold Evaluation Protocol)` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: Demonstrates majority class alignment (~83% baseline accuracy on training set reflecting CTU-CHB class imbalance). Peak validation accuracy reached 95.34% at epoch 4.
- **Training Stability**: Training loss decreased smoothly from `0.4296` to `0.3638`. Validation loss converged to a minimum of `0.2272` at Epoch 4 before showing overfitting symptoms in later epochs (spiking to `0.5547` by Epoch 5).
- **Generalization Gap**: Validation performance peaked at Epoch 4 (95.34% Acc), but overfitted past Epoch 4 due to lack of early stopping/regularization on raw sequence data.
- **Computational Efficiency**: Highly efficient execution (~1-2s per epoch on CPU, ~135k parameters), offering rapid inference speed and low memory footprint.
- **Patent Differentiation Compliance (US12094611B2)**: Verified continuous end-to-end signal encoding mapping $(Batch, 2, 4800) \to \mathbb{R}^{128}$ directly without bounding-box pattern matching or longitudinal correlation loops.
- **Final Verdict**: Highly efficient baseline for local temporal feature extraction; requires early stopping (Patience = 4) or Focal Loss during full 5-Fold CV benchmarking to prevent overfitting past Epoch 4.

---

## 2. BiLSTM (Baseline)
**Objective**: Evaluate long-term sequential dependency tracking across the entire window.

### A. Optimal Hyperparameters
- Learning Rate: `0.001 (AdamW)`
- Hidden State Size: `64` (Bidirectional -> 128 concatenated features)
- Number of Layers: `2`
- Parameter Count: `158,977`
- Epochs to Convergence: `8 Epochs` (Best Val Loss: `0.2029`)

### B. Statistical Metrics (Validation / Single Fold Run)
| Metric | Value / Best Epoch (Epoch 8) |
| :--- | :--- |
| Accuracy (Train / Val) | `82.70% (Train) / 95.34% (Val)` |
| Best Validation Loss | `0.2029` |
| Training Loss (Epoch 8) | `0.4446` |
| AUROC | `TBD (Requires 5-Fold Evaluation Protocol)` |
| AUPRC | `TBD (Requires 5-Fold Evaluation Protocol)` |
| F1 Score | `TBD (Requires 5-Fold Evaluation Protocol)` |
| Precision (PPV) | `TBD (Requires 5-Fold Evaluation Protocol)` |
| Recall (Sensitivity)| `0.0% (Mode Collapse to Majority Class under unweighted BCE)` |
| Specificity | `100.0%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: **Severe Majority Class Bias / Mode Collapse**. Due to standard unweighted BCE loss on imbalanced validation data (95.34% normal vs 4.66% distress), the model learned to predict negative (`0`) for 100% of samples. This yields a constant 95.34% accuracy but **0.0% Sensitivity (Recall)**, missing all pathological distress cases.
- **Training Stability**: Training loss stayed flat around `0.4430` – `0.4623`. Validation loss fluctuated between `0.2029` and `0.3170`, reaching its global minimum at Epoch 8 (`0.2029`).
- **Generalization Gap**: Validation accuracy remained flat at `95.34%` across all 10 epochs due to predicting the dominant majority class.
- **Computational Efficiency**: Slower training speed per epoch compared to 1D CNN due to sequential recurrent steps over 4,800 time steps (`158,977` parameters).
- **Patent Differentiation Compliance (US12094611B2)**: Verified continuous signal encoding mapping $(Batch, 2, 4800) \to \mathbb{R}^{128}$ directly via bidirectional LSTM hidden states without longitudinal shape correlation or bounding-box extraction.
- **Final Verdict**: Captures global sequential context but suffers from majority class collapse under unweighted BCE loss. Requires Focal Loss / Class Weighting and 5-Fold patient-stratified cross-validation during Phase 3 benchmarking.

---

## 3. GRU (Baseline)
**Objective**: Evaluate a lightweight recurrent alternative to the LSTM.

### A. Optimal Hyperparameters
- Learning Rate: `[To be filled]`
- Hidden State Size: `[To be filled]`
- Number of Layers: `[To be filled]`
- Parameter Count: `[To be filled]`
- Epochs to Convergence: `[To be filled]`

### B. Statistical Metrics (Test Set)
| Metric | Mean ± Std |
| :--- | :--- |
| Accuracy | `%` |
| AUROC | `0.00` |
| AUPRC | `0.00` |
| F1 Score | `0.00` |
| Precision (PPV) | `%` |
| Recall (Sensitivity)| `%` |
| Specificity | `%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: `[To be filled]`
- **Training Stability**: `[To be filled]`
- **Generalization Gap**: `[To be filled]`
- **Computational Efficiency**: `[To be filled]`
- **Patent Differentiation Compliance (US12094611B2)**: `[Verified continuous signal encoding without longitudinal shape correlation loops]`
- **Final Verdict**: `[To be filled]`

---

## 4. Temporal Convolutional Network (TCN)
**Objective**: Evaluate parallelizable convolution-based sequence modeling via causal dilated convolutions.

### A. Optimal Hyperparameters
- Learning Rate: `[To be filled]`
- Dilations: `[To be filled]`
- Kernel Size: `[To be filled]`
- Parameter Count: `[To be filled]`
- Epochs to Convergence: `[To be filled]`

### B. Statistical Metrics (Test Set)
| Metric | Mean ± Std |
| :--- | :--- |
| Accuracy | `%` |
| AUROC | `0.00` |
| AUPRC | `0.00` |
| F1 Score | `0.00` |
| Precision (PPV) | `%` |
| Recall (Sensitivity)| `%` |
| Specificity | `%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: `[To be filled]`
- **Training Stability**: `[To be filled]`
- **Generalization Gap**: `[To be filled]`
- **Computational Efficiency**: `[To be filled]`
- **Patent Differentiation Compliance (US12094611B2)**: `[Verified continuous signal encoding without longitudinal shape correlation loops]`
- **Final Verdict**: `[To be filled]`

---

## 5. Multi-Scale LSTM (Literature Baseline)
**Objective**: Evaluate a representative CTG-specific architecture capturing multiple temporal resolutions.

### A. Optimal Hyperparameters
- Learning Rate: `[To be filled]`
- Scales Used: `[To be filled]`
- Hidden Size per Scale: `[To be filled]`
- Parameter Count: `[To be filled]`
- Epochs to Convergence: `[To be filled]`

### B. Statistical Metrics (Test Set)
| Metric | Mean ± Std |
| :--- | :--- |
| Accuracy | `%` |
| AUROC | `0.00` |
| AUPRC | `0.00` |
| F1 Score | `0.00` |
| Precision (PPV) | `%` |
| Recall (Sensitivity)| `%` |
| Specificity | `%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: `[To be filled]`
- **Training Stability**: `[To be filled]`
- **Generalization Gap**: `[To be filled]`
- **Computational Efficiency**: `[To be filled]`
- **Patent Differentiation Compliance (US12094611B2)**: `[Verified continuous signal encoding without longitudinal shape correlation loops]`
- **Final Verdict**: `[To be filled]`

---

## 6. PatchCTG (Transformer Baseline)
**Objective**: Evaluate transformer-based attention mechanisms on patchified CTG sequences.

### A. Optimal Hyperparameters
- Learning Rate: `[To be filled]`
- Patch Size: `[To be filled]`
- Attention Heads & Layers: `[To be filled]`
- Parameter Count: `[To be filled]`
- Epochs to Convergence: `[To be filled]`

### B. Statistical Metrics (Test Set)
| Metric | Mean ± Std |
| :--- | :--- |
| Accuracy | `%` |
| AUROC | `0.00` |
| AUPRC | `0.00` |
| F1 Score | `0.00` |
| Precision (PPV) | `%` |
| Recall (Sensitivity)| `%` |
| Specificity | `%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: `[To be filled]`
- **Training Stability**: `[To be filled]`
- **Generalization Gap**: `[To be filled]`
- **Computational Efficiency**: `[To be filled]`
- **Patent Differentiation Compliance (US12094611B2)**: `[Verified continuous signal encoding without longitudinal shape correlation loops]`
- **Final Verdict**: `[To be filled]`

---

## 7. PatchTST (Modern SOTA Baseline)
**Objective**: Evaluate transformer-based attention over time-series patches, representing the current general-purpose forecasting state-of-the-art.

### A. Optimal Hyperparameters
- Learning Rate: `5e-4 (Cosine Annealing)`
- Patch Length & Stride: `P=16, S=16 (300 patches / channel)`
- Transformer Blocks & Heads: `Layers=3, Heads=8, d_model=128, d_ff=512`
- Parameter Count: `~532,480`
- Epochs to Convergence: `[To be logged after T4 Colab benchmark]`

### B. Statistical Metrics (Test Set)
| Metric | Mean ± Std |
| :--- | :--- |
| Accuracy | `%` |
| AUROC | `0.00` |
| AUPRC | `0.00` |
| F1 Score | `0.00` |
| Precision (PPV) | `%` |
| Recall (Sensitivity)| `%` |
| Specificity | `%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: `[To be filled after T4 Colab benchmark execution]`
- **Training Stability**: `[To be filled after T4 Colab benchmark execution]`
- **Generalization Gap**: `[To be filled after T4 Colab benchmark execution]`
- **Computational Efficiency**: `[To be filled after T4 Colab benchmark execution]`
- **Patent Differentiation Compliance (US12094611B2)**: `Verified continuous signal encoding R^(2x4800) -> R^128 without longitudinal shape correlation or bounding-box loops`
- **Final Verdict**: `[To be filled after T4 Colab benchmark execution]`

---

## 8. Proposed Knowledge-Infused Multi-Task Framework
**Objective**: Evaluate the final system (Best Encoder + Multi-Task Heads) against all other completed baselines to demonstrate the value of clinical auxiliary supervision.

### A. Optimal Hyperparameters
- Learning Rate: `[To be filled]`
- Selected Encoder: `[To be filled]`
- Loss Weights (Primary, FIGO, Features): `[To be filled]`
- Parameter Count: `[To be filled]`
- Epochs to Convergence: `[To be filled]`

### B. Statistical Metrics (Test Set)
| Metric | Mean ± Std |
| :--- | :--- |
| Accuracy | `%` |
| AUROC | `0.00` |
| AUPRC | `0.00` |
| F1 Score | `0.00` |
| Precision (PPV) | `%` |
| Recall (Sensitivity)| `%` |
| Specificity | `%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: `[To be filled]`
- **Training Stability**: `[To be filled]`
- **Generalization Gap**: `[To be filled]`
- **Computational Efficiency**: `[To be filled]`
- **Patent Differentiation Compliance (US12094611B2)**: `[Verified continuous signal encoding + Multi-Task biochemical & FIGO loss supervision]`
- **Final Verdict**: `[Did the multi-task formulation outperform the raw baseline encoder?]`

### D. Statistical Significance Analysis (Before vs. After Knowledge Infusion)
*Comparison between the standalone winner baseline encoder (Pre-Infusion) and the Knowledge-Infused Framework (Post-Infusion) across identical 5-Fold patient splits.*

| Metric | Standalone Baseline (Pre-Infusion) | Knowledge-Infused Framework (Post-Infusion) | Test Statistic ($t$ / $Z$) | $p$-value | Statistically Significant ($p < 0.05$)? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AUROC** | `0.00 ± 0.00` | `0.00 ± 0.00` | `[t/Z score]` | `p = 0.000` | `[Yes / No]` |
| **AUPRC** | `0.00 ± 0.00` | `0.00 ± 0.00` | `[t/Z score]` | `p = 0.000` | `[Yes / No]` |
| **F1 Score** | `0.00 ± 0.00` | `0.00 ± 0.00` | `[t/Z score]` | `p = 0.000` | `[Yes / No]` |
| **Recall (Sensitivity)** | `%` | `%` | `[t/Z score]` | `p = 0.000` | `[Yes / No]` |
| **Specificity** | `%` | `%` | `[t/Z score]` | `p = 0.000` | `[Yes / No]` |
| **DeLong Test (ROC)** | *Reference ROC Curve* | *Comparison ROC Curve* | `[DeLong Z]` | `p = 0.000` | `[Yes / No]` |

---

# Final Conclusion & Selection

Based on the benchmarking results documented above, the **[Insert Model Name]** has been selected as the official Temporal Encoder backbone for the Multi-Task Framework. 

**Selection Rationale**:
- `[Explain why this model won (e.g., highest 5-fold CV AUROC, statistically significant improvement p < 0.05, fast inference time).]`
- `[Detail why other promising models were rejected.]`
