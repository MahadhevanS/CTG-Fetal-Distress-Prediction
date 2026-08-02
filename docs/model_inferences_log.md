# Centralized Model Inference & Evaluation Logs

This document serves as the single source of truth for tracking the benchmarking results, hyperparameter configurations, and clinical observations for all evaluated temporal encoders under 5-Fold Stratified Patient-Level Cross Validation.

---

## 1. 1D CNN (Baseline)
**Objective**: Evaluate local temporal pattern extraction capabilities via 1D residual convolutional blocks.

### A. Optimal Hyperparameters
- Learning Rate: `0.0005 (AdamW)`
- Kernel Sizes: `Stem: 7 (stride 2), Residual Blocks: 3 (stride 1/2)`
- Number of Convolutional Blocks: `3 Residual Blocks (32 -> 64 -> 128 channels)`
- Parameter Count: `135,169`
- Batch Size: `32` | Epochs: `15` | Loss: `BCEWithLogitsLoss (Dynamic pos_weight ~5.08–5.29)`

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `72.75% ± 5.28%` | `63.04% – 77.05%` |
| **AUROC** | `0.6860 ± 0.0503` | `0.6084 – 0.7586` |
| **AUPRC** | `0.2560 ± 0.0544` | `0.1976 – 0.3244` |
| **F1 Score** | `0.3304 ± 0.0511` | `0.2707 – 0.4089` |
| **Precision (PPV)** | `28.19% ± 5.14%` | — |
| **Recall (Sensitivity)**| `42.09% ± 11.19%` | `30.06% – 60.74%` |
| **Specificity** | `78.68% ± 7.85%` | `63.45% – 84.50%` |
| **Sens @ 90% Specificity** | `21.09% ± 10.76%` | `7.93% – 34.45%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: Balanced performance (78.68% Specificity, 42.09% Recall). Dynamic positive class weighting effectively prevented majority class collapse.
- **Training Stability**: Consistent convergence across all 5 folds without divergence. Fold 4 achieved peak AUROC of 0.7586.
- **Generalization Gap**: Patient-level cross validation demonstrates robust zero-leakage generalization (AUROC 0.6860 ± 0.0503).
- **Computational Efficiency**: Ultra-fast execution (~1-2s per epoch, 135K parameters).
- **Patent Differentiation Compliance (US12094611B2)**: Continuous end-to-end signal representation $(Batch, 2, 4800) \to \mathbb{R}^{128}$ directly without bounding-box pattern matching or longitudinal shape correlation loops.
- **Final Verdict**: solid, computationally lightweight convolutional baseline for local feature extraction.

---

## 2. BiLSTM (Baseline)
**Objective**: Evaluate long-term sequential dependency tracking across continuous 4,800 time steps.

### A. Optimal Hyperparameters
- Learning Rate: `0.0005 (AdamW)`
- Hidden State Size: `64` (Bidirectional -> 128 concatenated features)
- Number of Layers: `2`
- Parameter Count: `158,977`
- Batch Size: `32` | Epochs: `15` | Loss: `BCEWithLogitsLoss (Dynamic pos_weight ~5.08–5.29)`

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `61.83% ± 5.84%` | `54.38% – 68.98%` |
| **AUROC** | `0.6544 ± 0.0409` | `0.5903 – 0.7090` |
| **AUPRC** | `0.2697 ± 0.0426` | `0.2275 – 0.3362` |
| **F1 Score** | `0.3435 ± 0.0338` | `0.3119 – 0.4084` |
| **Precision (PPV)** | `24.03% ± 3.06%` | — |
| **Recall (Sensitivity)**| `61.55% ± 7.10%` | `53.23% – 69.83%` |
| **Specificity** | `61.87% ± 7.70%` | `52.68% – 71.67%` |
| **Sens @ 90% Specificity** | `24.19% ± 6.92%` | `14.11% – 33.19%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: High sensitivity (61.55% ± 7.10%) with moderate specificity (61.87% ± 7.70%). Dynamic class weighting successfully resolved the mode collapse (0% recall) observed in earlier unweighted runs.
- **Training Stability**: Low variance across folds (AUROC std ±0.0409), Fold 2 reached peak AUROC of 0.7090.
- **Generalization Gap**: Consistent sequential performance across all 5 patient splits.
- **Computational Efficiency**: 158,977 parameters; sequential recurrent updates across 4,800 steps require higher compute than 1D CNN.
- **Patent Differentiation Compliance (US12094611B2)**: Continuous sequence encoding mapping $(Batch, 2, 4800) \to \mathbb{R}^{128}$ directly via bidirectional LSTM hidden states without bounding boxes or shape correlation loops.
- **Final Verdict**: Strong baseline for sensitivity/recall performance (61.55%), establishing recurrent sequence tracking efficacy.

---

## 3. GRU (Baseline)
**Objective**: Evaluate lightweight gated recurrent architecture for temporal sequence modeling.

### A. Optimal Hyperparameters
- Learning Rate: `0.0005 (AdamW)`
- Hidden State Size: `64` (Bidirectional -> 128 output features)
- Number of Layers: `2`
- Parameter Count: `179,328`
- Batch Size: `32` | Epochs: `15` | Loss: `BCEWithLogitsLoss (Dynamic pos_weight ~5.08–5.29)`

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `77.14% ± 5.05%` | `68.07% – 82.75%` |
| **AUROC** | `0.6881 ± 0.0627` | `0.5770 – 0.7564` |
| **AUPRC** | `0.2812 ± 0.0839` | `0.1689 – 0.4260` |
| **F1 Score** | `0.3027 ± 0.1080` | `0.0913 – 0.3869` |
| **Precision (PPV)** | `31.38% ± 11.34%` | — |
| **Recall (Sensitivity)**| `34.00% ± 19.05%` | `6.75% – 66.12%` |
| **Specificity** | `85.44% ± 9.22%` | `68.42% – 93.76%` |
| **Sens @ 90% Specificity** | `25.05% ± 9.96%` | `12.88% – 43.15%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: High specificity (85.44% ± 9.22%) with moderate recall (34.00% ± 19.05%). Fold 4 reached 0.7564 AUROC and 0.4260 AUPRC.
- **Training Stability**: Good overall convergence; fold variance reflects varying signal quality across patient cohorts.
- **Generalization Gap**: Zero patient data leakage across all splits.
- **Computational Efficiency**: 179,328 parameters with faster step execution than BiLSTM.
- **Patent Differentiation Compliance (US12094611B2)**: Continuous signal encoding $\mathbb{R}^{2 \times 4800} \to \mathbb{R}^{128}$ without bounding boxes or longitudinal correlation loops.
- **Final Verdict**: Reliable gated recurrent baseline with strong specificity.

---

## 4. Temporal Convolutional Network (TCN)
**Objective**: Evaluate parallelized causal dilated convolutions for extended receptive field modeling.

### A. Optimal Hyperparameters
- Learning Rate: `0.0005 (AdamW)`
- Dilations: `d = [1, 2, 4, 8, 16, 32] (Causal dilated convolutions)`
- Kernel Size: `k = 3`
- Parameter Count: `363,560`
- Batch Size: `32` | Epochs: `15` | Loss: `BCEWithLogitsLoss (Dynamic pos_weight ~5.08–5.29)`

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `79.47% ± 2.55%` | `76.64% – 82.75%` |
| **AUROC** | `0.7154 ± 0.0797` | `0.5949 – 0.8431` |
| **AUPRC** | `0.2846 ± 0.0840` | `0.1818 – 0.4364` |
| **F1 Score** | `0.2413 ± 0.1079` | `0.1125 – 0.3750` |
| **Precision (PPV)** | `30.13% ± 11.74%` | — |
| **Recall (Sensitivity)**| `21.28% ± 11.05%` | `7.44% – 36.55%` |
| **Specificity** | `90.57% ± 3.90%` | `84.50% – 95.54%` |
| **Sens @ 90% Specificity** | `25.65% ± 13.36%` | `9.82% – 49.19%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: Highest specificity among baseline convolutional models (90.57% ± 3.90%), minimizing false alarms. Fold 4 produced an exceptional single-fold AUROC of 0.8431 and AUPRC of 0.4364.
- **Training Stability**: Parallelized execution with stable loss reduction.
- **Generalization Gap**: Strong overall CV performance (0.7154 ± 0.0797 AUROC).
- **Computational Efficiency**: Fast parallel GPU computation via 1D dilated convolutions (363,560 parameters).
- **Patent Differentiation Compliance (US12094611B2)**: Continuous end-to-end dilated convolution encoding without pattern bounding boxes or shape correlation loops.
- **Final Verdict**: Highly efficient parallel convolutional encoder with exceptional specificity and high peak fold performance.

---

## 5. Multi-Scale LSTM (Literature Baseline)
**Objective**: Evaluate multi-resolution temporal recurrent branches capturing short- and long-term CTG dynamics.

### A. Optimal Hyperparameters
- Learning Rate: `0.0005 (AdamW)`
- Scales Used: `Multi-Scale Temporal Recurrent Branches`
- Hidden Size per Scale: `hidden_size=64 (Bidirectional)`
- Parameter Count: `583,904`
- Batch Size: `32` | Epochs: `15` | Loss: `BCEWithLogitsLoss (Dynamic pos_weight ~5.08–5.29)`

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `62.95% ± 6.88%` | `52.52% – 71.97%` |
| **AUROC** | `0.7263 ± 0.1103` | `0.5250 – 0.8261` |
| **AUPRC** | `0.3140 ± 0.0962` | `0.1633 – 0.4234` |
| **F1 Score** | `0.3668 ± 0.0666` | `0.2625 – 0.4460` |
| **Precision (PPV)** | `25.73% ± 3.78%` | — |
| **Recall (Sensitivity)**| `69.68% ± 24.77%` | `38.33% – 97.58%` |
| **Specificity** | `61.61% ± 12.62%` | `43.21% – 78.96%` |
| **Sens @ 90% Specificity** | `29.94% ± 12.13%` | `16.56% – 51.26%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: Highest overall recall (69.68% ± 24.77%), achieving up to 97.58% sensitivity in Fold 4. AUROC reached 0.8261 (Fold 4) and 0.8236 (Fold 5).
- **Training Stability**: Multi-scale recurrent fusion captures multi-resolution patterns effectively, though with higher variance across patient splits.
- **Generalization Gap**: 2nd highest mean validation AUROC (0.7263 ± 0.1103) among all 7 encoders.
- **Computational Efficiency**: 583,904 parameters; multi-branch recurrent operations require moderate computation time.
- **Patent Differentiation Compliance (US12094611B2)**: Continuous multi-scale sequence encoding mapping $(Batch, 2, 4800) \to \mathbb{R}^{128}$ directly without shape matching or bounding boxes.
- **Final Verdict**: Outstanding literature baseline for distress sensitivity (69.68% recall) and 2nd best overall CV AUROC (0.7263).

---

## 6. PatchCTG (Transformer Baseline)
**Objective**: Evaluate joint-channel patch-based self-attention over continuous CTG recordings.

### A. Optimal Hyperparameters
- Learning Rate: `0.0005 (AdamW)`
- Patch Size: `P=16, Stride S=16`
- Transformer Blocks & Heads: `Layers=3, Heads=8, d_model=128, d_ff=512`
- Parameter Count: `670,720`
- Batch Size: `32` | Epochs: `15` | Loss: `BCEWithLogitsLoss (Dynamic pos_weight ~5.08–5.29)`

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `55.53% ± 17.65%` | `23.44% – 73.81%` |
| **AUROC** | `0.6533 ± 0.0781` | `0.5472 – 0.7556` |
| **AUPRC** | `0.2963 ± 0.0829` | `0.2328 – 0.4603` |
| **F1 Score** | `0.3351 ± 0.0687` | `0.2520 – 0.4310` |
| **Precision (PPV)** | `23.62% ± 5.30%` | — |
| **Recall (Sensitivity)**| `68.46% ± 21.20%` | `28.57% – 87.73%` |
| **Specificity** | `53.38% ± 23.73%` | `12.37% – 82.69%` |
| **Sens @ 90% Specificity** | `24.36% ± 11.15%` | `11.76% – 42.74%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: High sensitivity (68.46% ± 21.20%), but joint channel tokenization creates wider performance variation across patient folds. Peak fold AUROC of 0.7556 and AUPRC of 0.4603 (Fold 4).
- **Training Stability**: Multi-head joint self-attention converges smoothly per fold.
- **Generalization Gap**: Zero patient data leakage; fold 3 exhibited sensitivity to extreme imbalance in patient distribution.
- **Computational Efficiency**: 670,720 parameters.
- **Patent Differentiation Compliance (US12094611B2)**: Continuous end-to-end patch transformer encoding mapping $(Batch, 2, 4800) \to \mathbb{R}^{128}$ without bounding boxes or shape correlation loops.
- **Final Verdict**: Functional patch transformer baseline; channel-independent formulation (PatchTST) outperforms joint-channel tokenization.

---

## 7. PatchTST (Modern SOTA Baseline - WINNER 🏆)
**Objective**: Evaluate channel-independent patch-based transformer modeling over time-series sequences.

### A. Optimal Hyperparameters
- Learning Rate: `0.0005 (AdamW)`
- Patch Length & Stride: `P=16, S=16 (300 patches per channel)`
- Transformer Blocks & Heads: `Layers=3, Heads=8, d_model=128, d_ff=512`
- Parameter Count: `685,056`
- Batch Size: `32` | Epochs: `15` | Loss: `BCEWithLogitsLoss (Dynamic pos_weight ~5.08–5.29)`

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `70.23% ± 6.61%` | `58.97% – 78.26%` |
| **AUROC** 🏆 | `0.7504 ± 0.0378` | `0.6980 – 0.8151` |
| **AUPRC** 🏆 | `0.3820 ± 0.0800` | `0.2928 – 0.5272` |
| **F1 Score** 🏆 | `0.4102 ± 0.0446` | `0.3299 – 0.4665` |
| **Precision (PPV)** | `31.29% ± 5.44%` | — |
| **Recall (Sensitivity)**| `63.74% ± 12.12%` | `45.97% – 82.82%` |
| **Specificity** | `71.54% ± 9.69%` | `57.29% – 84.93%` |
| **Sens @ 90% Specificity** 🏆 | `35.09% ± 6.65%` | `27.61% – 44.93%` |

### C. Clinical & Computational Inferences
- **False Positives vs False Negatives**: Outstanding clinical balance—achieves high sensitivity (63.74% ± 12.12%) and solid specificity (71.54% ± 9.69%), while dominating clinical safety metric Sens@90%Spec (35.09% ± 6.65%). Peak fold AUROC reached 0.8151 and AUPRC reached 0.5272 (Fold 2).
- **Training Stability**: Exceptionally low standard deviation across all 5 folds (AUROC std ±0.0378), demonstrating superior patient-level generalization.
- **Generalization Gap**: Zero patient data leakage; highest consistency across all 7 evaluated architectures.
- **Computational Efficiency**: 685,056 parameters; channel-independent patch attention enables parallelized feature extraction across FHR and UC.
- **Patent Differentiation Compliance (US12094611B2)**: Continuous signal encoding $\mathbb{R}^{2 \times 4800} \to \mathbb{R}^{128}$ directly via patchified self-attention without bounding boxes or longitudinal shape matching loops.
- **Final Verdict**: **STATE-OF-THE-ART WINNER**. PatchTST achieves the highest overall AUROC (0.7504), AUPRC (0.3820), F1 Score (0.4102), and Sens@90%Spec (35.09%), making it the official temporal encoder backbone selected for Phase 4 Knowledge-Infused Multi-Task Framework integration.

---

## 8. Master 5-Fold Patient-Level CV Comparison Table

| Model | Architecture Type | Params | Accuracy (%) | AUROC | AUPRC | F1 Score | Recall / Sens (%) | Specificity (%) | Sens @ 90% Spec (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN1D** | 1D Residual CNN | 135K | 72.75 ± 5.28 | 0.6860 ± 0.0503 | 0.2560 ± 0.0544 | 0.3304 ± 0.0511 | 42.09 ± 11.19 | 78.68 ± 7.85 | 21.09 ± 10.76 |
| **BiLSTM** | Bidirectional LSTM | 159K | 61.83 ± 5.84 | 0.6544 ± 0.0409 | 0.2697 ± 0.0426 | 0.3435 ± 0.0338 | 61.55 ± 7.10 | 61.87 ± 7.70 | 24.19 ± 6.92 |
| **GRU** | Gated Recurrent Unit | 179K | 77.14 ± 5.05 | 0.6881 ± 0.0627 | 0.2812 ± 0.0839 | 0.3027 ± 0.1080 | 34.00 ± 19.05 | 85.44 ± 9.22 | 25.05 ± 9.96 |
| **TCN** | Temporal Conv Network | 363K | 79.47 ± 2.55 | 0.7154 ± 0.0797 | 0.2846 ± 0.0840 | 0.2413 ± 0.1079 | 21.28 ± 11.05 | 90.57 ± 3.90 | 25.65 ± 13.36 |
| **MS-LSTM** | Multi-Scale BiLSTM | 584K | 62.95 ± 6.88 | 0.7263 ± 0.1103 | 0.3140 ± 0.0962 | 0.3668 ± 0.0666 | **69.68 ± 24.77** | 61.61 ± 12.62 | 29.94 ± 12.13 |
| **PatchCTG** | Joint Patch Transformer | 671K | 55.53 ± 17.65 | 0.6533 ± 0.0781 | 0.2963 ± 0.0829 | 0.3351 ± 0.0687 | 68.46 ± 21.20 | 53.38 ± 23.73 | 24.36 ± 11.15 |
| **PatchTST** 🏆 | Channel-Ind. Patch Trans. | 685K | 70.23 ± 6.61 | **0.7504 ± 0.0378** | **0.3820 ± 0.0800** | **0.4102 ± 0.0446** | 63.74 ± 12.12 | 71.54 ± 9.69 | **35.09 ± 6.65** |

---

## 9. Proposed Knowledge-Infused Multi-Task Framework (Phase 4 Target)
**Objective**: Evaluate the final system (PatchTST Encoder + Multi-Task Heads) against all completed baselines to demonstrate clinical auxiliary supervision value.

### A. Target Architecture Specifications
- Selected Encoder: `PatchTSTEncoder (685K params)`
- Auxiliary Task Heads:
  1. Primary Distress Logit ($\text{pH} \le 7.15$) via BCE Loss with dynamic class weight.
  2. Auxiliary FIGO Diagnostic Categorization (Normal, Suspicious, Pathological) via CrossEntropy Loss.
  3. Continuous Clinical Feature Regressor (Baseline FHR, STV, LTV, Accel/Decel counts) via MSE Loss.
- Multi-Task Loss Weighting: $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{primary}} + \lambda_2 \mathcal{L}_{\text{FIGO}} + \lambda_3 \mathcal{L}_{\text{features}}$

### B. Planned Pre- vs. Post-Infusion Statistical Significance Protocol
- **Statistical Tests**: Paired $t$-test / Wilcoxon signed-rank test across identical 5 patient folds + DeLong test for ROC curve comparisons.
- **Null Hypothesis ($H_0$)**: Knowledge Infusion produces no significant change in AUROC ($p \ge 0.05$).
- **Alternative Hypothesis ($H_1$)**: Knowledge Infusion produces statistically significant improvement ($p < 0.05$).

---

# Final Conclusion & Selection

Based on the standardized 5-Fold Stratified Patient-Level Cross-Validation benchmarking across 6,917 CTG sequence windows and 546 unique patients, **PatchTST** is officially selected as the winning Temporal Encoder backbone.

**Selection Rationale**:
1. **Highest Predictive Discriminative Power**: PatchTST achieves the top mean AUROC (**0.7504**), AUPRC (**0.3820**), and F1 Score (**0.4102**) among all 7 temporal architectures.
2. **Superior Clinical Safety Metric**: Delivers the highest Sensitivity at 90% Specificity (**35.09% ± 6.65%**), outperforming all baseline models in early fetal distress detection at strict false-alarm thresholds.
3. **Lowest Fold Variance**: Demonstrates the lowest AUROC standard deviation ($\pm 0.0378$) across patient splits, proving exceptional out-of-fold generalization without patient data leakage.
4. **Channel-Independent Tokenization**: Separating FHR and UC tokenization allows the transformer to learn distinct temporal dynamics for fetal cardiac reactivity versus uterine contractility before cross-feature projection.
