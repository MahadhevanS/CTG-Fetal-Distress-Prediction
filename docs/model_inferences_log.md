# Centralized Model Inference & Evaluation Log

This document serves as the single source of truth for tracking the benchmarking results, hyperparameter configurations, and clinical observations for all evaluated temporal encoders.

> **Instruction**: After completing the training and testing loop for a specific model, update its respective section below. Do not alter the template structure to ensure apples-to-apples comparisons.

---

## 1. 1D CNN (Baseline)
**Objective**: Evaluate local temporal pattern extraction capabilities.

### A. Optimal Hyperparameters
- Learning Rate: `[To be filled]`
- Kernel Sizes: `[To be filled]`
- Number of Convolutional Blocks: `[To be filled]`
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
- **False Positives vs False Negatives**: `[Analyze whether the model over-predicted distress or missed critical pathological cases.]`
- **Training Stability**: `[Did the loss converge smoothly? Were there spiking gradients?]`
- **Generalization Gap**: `[Difference between Validation AUROC and Test AUROC.]`
- **Computational Efficiency**: `[Training time per epoch, inference speed.]`
- **Patent Differentiation Compliance (US12094611B2)**: `[Verified continuous signal encoding without longitudinal shape correlation loops]`
- **Final Verdict**: `[Strengths, weaknesses, and suitability for the final framework.]`

---

## 2. BiLSTM (Baseline)
**Objective**: Evaluate long-term sequential dependency tracking across the entire window.

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

## 3. GRU (Baseline)
**Objective**: Evaluate a lightweight recurrent alternative to the LSTM.

### A. Optimal Hyperparameters
- Learning Rate: `5e-4 (Cosine Annealing)`
- Hidden State Size: `64 (Bidirectional, output size 128)`
- Number of Layers: `2`
- Parameter Count: `179,328 (~179.3K parameters)`
- Epochs to Convergence: `14 epochs`

### B. Statistical Metrics (5-Fold Patient-Level CV)
| Metric | Mean ± Std |
| :--- | :--- |
| Accuracy | `82.20% ± 2.00%` |
| AUROC | `0.7099 ± 0.0471` |
| AUPRC | `0.2797 ± 0.0432` |
| F1 Score | `0.1584 ± 0.1460` |
| Precision (PPV) | `29.40% ± 13.59%` |
| Recall (Sensitivity)| `13.18% ± 13.94%` |
| Specificity | `95.29% ± 4.37%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: High specificity (95.29% ± 4.37%) with low sensitivity (13.18% ± 13.94%) at standard 0.5 decision threshold, producing minimal false alarms but missing subtle distress cases.
- **Training Stability**: Smooth convergence across 100 epochs per fold using Cosine Annealing learning rate schedule.
- **Generalization Gap**: Stable cross-validation performance (AUROC 0.7099 ± 0.0471) with zero patient data leakage across folds.
- **Computational Efficiency**: Gated recurrent architecture (~179.3K parameters) with fast GPU training throughput.
- **Patent Differentiation Compliance (US12094611B2)**: Verified continuous signal encoding R^(2x4800) -> R^128 without longitudinal shape correlation loops or bounding box extraction.
- **Final Verdict**: Functional lightweight recurrent baseline; decision threshold calibration / focal loss weighting recommended to boost sensitivity during multi-task integration.

---

## 4. Temporal Convolutional Network (TCN)
**Objective**: Evaluate parallelizable convolution-based sequence modeling via causal dilated convolutions.

### A. Optimal Hyperparameters
- Learning Rate: `5e-4 (Cosine Annealing)`
- Dilations: `d = [1, 2, 4, 8, 16, 32] (Causal dilated convolutions)`
- Kernel Size: `k = 3`
- Parameter Count: `363,560 (~363.5K parameters)`
- Epochs to Convergence: `12 epochs`

### B. Statistical Metrics (5-Fold Patient-Level CV)
| Metric | Mean ± Std |
| :--- | :--- |
| Accuracy | `82.71% ± 1.64%` |
| AUROC | `0.6962 ± 0.0431` |
| AUPRC | `0.2716 ± 0.0403` |
| F1 Score | `0.0471 ± 0.0607` |
| Precision (PPV) | `31.03% ± 27.05%` |
| Recall (Sensitivity)| `3.12% ± 4.44%` |
| Specificity | `97.98% ± 2.34%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: Extremely high specificity (97.98% ± 2.34%) with low recall (3.12% ± 4.44%) at standard 0.5 decision threshold due to class imbalance; avoids false alarms but requires threshold calibration to catch subtle distress.
- **Training Stability**: Fast and stable parallelized GPU execution across 100 epochs per fold using Cosine Annealing and PyTorch AMP mixed precision.
- **Generalization Gap**: Consistent cross-validation performance across patient splits (AUROC 0.6962 ± 0.0431) with zero patient data leakage.
- **Computational Efficiency**: Superior training speed and throughput via 1D causal dilated convolutions (~363.5K parameters) combined with strided temporal downsampling.
- **Patent Differentiation Compliance (US12094611B2)**: Verified continuous signal encoding R^(2x4800) -> R^128 without longitudinal shape correlation loops or bounding box extraction.
- **Final Verdict**: Highly efficient parallel temporal encoder baseline; positive class reweighting / threshold tuning recommended for Phase 2 multi-task framework integration.

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
