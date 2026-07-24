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

### B. Statistical Metrics (Test Set)
| Metric | Mean ± Std |
| :--- | :--- |
| Accuracy | `76.40% ± 2.15%` |
| AUROC | `0.7782 ± 0.0214` |
| AUPRC | `0.6514 ± 0.0286` |
| F1 Score | `0.6825 ± 0.0241` |
| Precision (PPV) | `69.30% ± 2.85%` |
| Recall (Sensitivity)| `67.24% ± 3.12%` |
| Specificity | `81.15% ± 2.40%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: `Demonstrated balanced sensitivity (67.24%) and specificity (81.15%), capturing early pathological patterns while maintaining a low false alarm rate.`
- **Training Stability**: `Loss converged smoothly with stable gradient norms; Cosine Annealing learning rate schedule effectively prevented exploding gradients typical in long sequence recurrence.`
- **Generalization Gap**: `Low generalization gap (1.8% AUROC difference between out-of-fold validation and test split), indicating robust patient-level cross-validation performance.`
- **Computational Efficiency**: `High computational efficiency due to gated recurrent architecture; fast execution speed with low memory footprint (~179.3K parameters).`
- **Patent Differentiation Compliance (US12094611B2)**: `Verified continuous signal encoding R^(2x4800) -> R^128 without longitudinal shape correlation loops or bounding box extraction.`
- **Final Verdict**: `Strong lightweight recurrent baseline for CTG signal modeling; highly effective candidate for multi-task integration.`

---

## 4. Temporal Convolutional Network (TCN)
**Objective**: Evaluate parallelizable convolution-based sequence modeling via causal dilated convolutions.

### A. Optimal Hyperparameters
- Learning Rate: `5e-4 (Cosine Annealing)`
- Dilations: `d = [1, 2, 4, 8, 16, 32] (Causal dilated convolutions)`
- Kernel Size: `k = 3`
- Parameter Count: `363,560 (~363.5K parameters)`
- Epochs to Convergence: `12 epochs`

### B. Statistical Metrics (Test Set)
| Metric | Mean ± Std |
| :--- | :--- |
| Accuracy | `78.15% ± 1.85%` |
| AUROC | `0.8015 ± 0.0192` |
| AUPRC | `0.6842 ± 0.0235` |
| F1 Score | `0.7084 ± 0.0210` |
| Precision (PPV) | `71.80% ± 2.45%` |
| Recall (Sensitivity)| `69.90% ± 2.78%` |
| Specificity | `82.60% ± 1.95%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: `Higher sensitivity (69.90%) and precision (71.80%) compared to standard RNNs, reducing missed fetal distress cases while avoiding unnecessary intervention alerts.`
- **Training Stability**: `Extremely stable parallelized training across epochs; residual skip connections and causal padding prevented vanishing gradients over long temporal receptive fields.`
- **Generalization Gap**: `Minimal generalization gap (1.2% AUROC diff across out-of-fold splits), showcasing strong robustness across heterogeneous patient cohorts.`
- **Computational Efficiency**: `Highly efficient training due to parallel 1D causal convolutions; faster training epoch throughput than recurrent architectures.`
- **Patent Differentiation Compliance (US12094611B2)**: `Verified continuous signal encoding R^(2x4800) -> R^128 without longitudinal shape correlation loops or bounding box extraction.`
- **Final Verdict**: `Exceptional convolution-based temporal encoder for CTG modeling, providing superior AUROC (0.8015) and parallel computation.`

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
