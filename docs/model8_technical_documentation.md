# CTG Fetal Distress Prediction: Model 8 Technical Documentation
**(Knowledge-Infused Multi-Task Temporal Deep Learning Framework)**

> **Author**: Member 4 (Lead Developer)
> **Repository**: CTG Fetal Distress Prediction
> **Target Branch**: `models_set_4`
> **Phase**: Phase 4 — Knowledge-Infused Multi-Task Integration
> **Document Status**: Implementation Complete — Awaiting Colab Benchmark Run

This document provides a comprehensive technical, mathematical, architectural, and empirical reference for **Model 8: the Knowledge-Infused Multi-Task Framework (KI-MTF)** — the proposed novelty architecture of this research project.

---

## 1. Executive Summary

Model 8 is the culminating system of this research. It does not introduce a new temporal encoder. Instead, it **augments the winning Phase 3 encoder (PatchTST)** with three clinically-grounded task heads and a composite loss function that enforces FIGO 2015 clinical consistency during training.

### Core Hypothesis
> Jointly supervising the shared PatchTST latent space on three clinically meaningful tasks — binary fetal distress, FIGO classification, and physiological feature regression — forces the encoder to learn representations that are more physiologically faithful, yielding statistically significant improvements over the standalone baseline.

### Universal Encoder Signature (Preserved from Phase 3)
- **Input**: $\mathbf{X} \in \mathbb{R}^{B \times 2 \times 4800}$ (Channel 0: Baseline-corrected FHR, Channel 1: UC at 4 Hz for 20 minutes)
- **Latent**: $\mathbf{z} \in \mathbb{R}^{B \times 128}$ — universal shared representation from PatchTST
- **Patent Compliance (GE US12094611B2)**: End-to-end continuous signal mapping without bounding boxes or shape-matching correlation loops

---

## 2. Architectural Design

### 2.1 Theoretical Motivation

The fundamental limitation of standalone single-task temporal encoders is the **underdetermined supervision problem**: mapping a raw 4800-sample waveform directly to a single binary pH label is an extremely ill-conditioned optimization. The encoder can find many latent representations that minimize the binary cross-entropy loss without learning physiologically meaningful features.

**Knowledge Infusion** addresses this by adding three sources of explicit clinical supervision:

1. **FIGO 2015 Classification Head** — Forces the latent space to encode the same features obstetricians use to classify CTG traces. If the encoder learns the right physiological structure, FIGO classification becomes a natural by-product.

2. **Physiological Feature Regression Head** — Directly penalizes the encoder if its latent space cannot reconstruct known clinical features (Baseline FHR, STV, LTV, deceleration counts). This prevents the encoder from relying on spurious correlations.

3. **Differentiable FIGO Rule Loss** — A novel soft penalty that directly penalizes predictions that violate FIGO 2015 clinical rules during backpropagation. This is the core novelty: FIGO guidelines are embedded as a **differentiable training constraint**, not just as a label source.

### 2.2 Novelty Position in Literature

| Closest System | Key Approach | What We Do Differently |
| :--- | :--- | :--- |
| **McCoy et al. (2024)** AJOG | CNN → binary pH | Multi-task; FIGO knowledge loss |
| **Khan et al. (2025)** PatchCTG | Joint-channel patch transformer | Channel-independent; intrapartum; knowledge loss |
| **CTG-CrossFormer (2025)** | CNN-SE + cross-attention FHR-UC | No FIGO auxiliary supervision; no clinical rule loss |
| **PRISM-CTG (2025)** | Self-supervised PatchTST pre-training | No multi-task heads; no FIGO constraints |
| **Zhang et al. (2022)** | Multimodal CTG + EEG fusion | Requires multi-modal input; no FIGO rule penalty |

**Core novelty claim**: This is the first system to combine a channel-independent patch transformer with three simultaneous clinical supervision heads and a **differentiable FIGO 2015 rule consistency penalty** under a strict 30-minute intrapartum prediction horizon with patient-level data isolation.

---

### 2.3 Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ INPUT: Raw CTG Signal X ∈ ℝ^(B × 2 × 4800)                            │
│   Channel 0: Baseline-Corrected FHR   Channel 1: Uterine Contractions  │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SHARED BACKBONE: PatchTSTEncoder                                        │
│   patch_len=16, stride=16, d_model=128, n_heads=4 (tuned), n_layers=4  │
│   Channel-Independent Patchification → Transformer → Pool              │
│                                                                         │
│   X (B,2,4800)                                                          │
│       │                                                                 │
│       ├── CH0: FHR ── Patch(300 tokens) ──┐                            │
│       │                                   ├── Transformer(L=4) ──┐     │
│       └── CH1: UC  ── Patch(300 tokens) ──┘                      │     │
│                                                                   │     │
│                                         Global Avg Pool ──────────┘     │
│                                         Concat + Project                │
│                                         z ∈ ℝ^(B×128)                  │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       │ z: (B, 128)
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
     ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐
     │  DISTRESS HEAD   │  │    FIGO HEAD     │  │  CLINICAL FEATURE HEAD  │
     │  (Primary Task)  │  │ (Auxiliary Task) │  │    (Auxiliary Task)     │
     │                  │  │                  │  │                         │
     │ Linear(128→64)   │  │ Linear(128→64)   │  │  Linear(128→64)         │
     │ LayerNorm(64)    │  │ LayerNorm(64)    │  │  LayerNorm(64)          │
     │ GELU             │  │ GELU             │  │  GELU                   │
     │ Dropout(0.2)     │  │ Dropout(0.2)     │  │  Dropout(0.2)           │
     │ Linear(64→1)     │  │ Linear(64→3)     │  │  Linear(64→8)           │
     │                  │  │                  │  │  ├─[0] Baseline FHR     │
     │ ŷ_distress (B,1) │  │ ŷ_figo (B,3)    │  │  ├─[1] STV              │
     │ BCEWithLogits    │  │ CrossEntropy     │  │  ├─[2] LTV              │
     │ + pos_weight     │  │ L_FIGO           │  │  ├─[3] Accels ←softplus │
     │ L_distress       │  │                  │  │  ├─[4] Early Decels     │
     └────────┬─────────┘  └────────┬─────────┘  │  ├─[5] Late Decels     │
              │                     │             │  ├─[6] Var Decels      │
              │                     │             │  └─[7] Prolonged Decels│
              │                     │             └────────────┬────────────┘
              │                     │                          │
              │                     │             ┌────────────┴────────────┐
              │                     │             │  MSE Loss   +           │
              │                     │             │  figo_rule_loss_norm()  │
              │                     │             │  L_features + L_know    │
              │                     │             └────────────┬────────────┘
              └─────────────────────┴──────────────────────────┘
                                        │
                                        ▼
     ┌─────────────────────────────────────────────────────────────────────┐
     │ COMPOSITE MULTI-TASK LOSS                                          │
     │ L_total = L_distress + λ₁·L_FIGO + λ₂·L_features + λ₃·L_knowledge │
     │   λ₁ = 0.3  (FIGO CE)                                              │
     │   λ₂ = 0.2  (Feature MSE)                                          │
     │   λ₃ = 0.1  (FIGO Rule Penalty)                                    │
     └─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulation

### 3.1 Backbone: PatchTST Encoder (Tuned Configuration)

The input CTG signal $\mathbf{X} \in \mathbb{R}^{B \times C \times L}$ with $C=2$, $L=4800$ is processed through six sequential stages (see `models_set_4_technical_documentation.md` §2.2 for full derivation):

$$\mathbf{z} = \text{PatchTSTEncoder}(\mathbf{X}) \in \mathbb{R}^{B \times 128}$$

**Tuned Configuration** (from hyperparameter search — Trial 1 winner):

| Parameter | Tuned Value | Default | Rationale |
| :--- | :--- | :--- | :--- |
| Patch Length ($P$) | `16` samples | 16 | 4-second physiological context window |
| Patch Stride ($S$) | `16` samples | 16 | Non-overlapping tokenization |
| Num Patches ($N$) | `300` per channel | 300 | $(4800-16)/16 + 1$ |
| $d_{model}$ | `128` | 128 | Token representation dimension |
| $n_{heads}$ | **`4`** | 8 | $d_k = 128/4 = 32$; tuned winner |
| $n_{layers}$ | **`4`** | 3 | Deeper encoder for long-range patterns |
| Dropout | **`0.2`** | 0.1 | Tuned for CTG dataset size |
| Latent Dim | `128` | 128 | Universal encoder output |

---

### 3.2 Multi-Task Heads

All three heads share the same architectural template:

$$\text{Head}(\mathbf{z}) = \mathbf{W}_2 \cdot \text{Dropout}_{0.2}\left(\text{GELU}\left(\text{LayerNorm}\left(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1\right)\right)\right) + \mathbf{b}_2$$

with $\mathbf{W}_1 \in \mathbb{R}^{64 \times 128}$, output dimensions differing by head:

| Head | Output | Activation | Loss |
| :--- | :--- | :--- | :--- |
| **DistressHead** | $(B, 1)$ | None (logit) | `BCEWithLogitsLoss` + dynamic `pos_weight` |
| **FIGOHead** | $(B, 3)$ | None (logits) | `CrossEntropyLoss` |
| **ClinicalFeatureHead** | $(B, 8)$ | Linear for $[0:3]$; `F.softplus` for $[3:8]$ | `MSELoss` |

**ClinicalFeatureHead Output Mapping** $(B, 8)$:

| Index | Feature | Unit | Activation |
| :--- | :--- | :--- | :--- |
| `[0]` | Baseline FHR | bpm | Linear |
| `[1]` | Short-Term Variability (STV) | bpm | Linear |
| `[2]` | Long-Term Variability (LTV) | bpm | Linear |
| `[3]` | Acceleration Count | count | F.softplus |
| `[4]` | Early Deceleration Count | count | F.softplus |
| `[5]` | Late Deceleration Count | count | F.softplus |
| `[6]` | Variable Deceleration Count | count | F.softplus |
| `[7]` | Prolonged Deceleration Count | count | F.softplus |

> **Design note**: `F.softplus` is applied to count outputs (indices 3–7) rather than `F.relu` to enforce non-negativity while maintaining smooth, non-zero gradients everywhere. `F.relu(x) = 0` for all $x < 0$, creating dead gradients when the network predicts negative counts. `F.softplus(x) = \ln(1 + e^x) > 0$ always, ensuring valid gradients throughout training.

---

### 3.3 Composite Multi-Task Loss

$$\mathcal{L}_{total} = \mathcal{L}_{distress} + \lambda_1 \mathcal{L}_{FIGO} + \lambda_2 \mathcal{L}_{features} + \lambda_3 \mathcal{L}_{knowledge}$$

**Component definitions:**

$$\mathcal{L}_{distress} = \text{BCEWithLogitsLoss}\left(\hat{y}_{distress},\ y_{primary};\ w_{pos} = \frac{N_{neg}}{N_{pos}}\right)$$

$$\mathcal{L}_{FIGO} = \text{CrossEntropyLoss}\left(\hat{y}_{figo},\ y_{figo}\right)$$

$$\mathcal{L}_{features} = \frac{1}{8} \sum_{j=0}^{7} \left(\hat{f}_j - f_j\right)^2 \quad \text{(MSE on Z-normalized targets)}$$

$$\mathcal{L}_{knowledge} = \text{figo\_rule\_loss\_normalized}\left(\hat{\mathbf{f}}_{norm},\ \hat{y}_{figo},\ \boldsymbol{\mu}_{feat},\ \boldsymbol{\sigma}_{feat}\right)$$

**Default $\lambda$ values** (tunable, see §6):

| Symbol | Default | Role |
| :--- | :--- | :--- |
| $\lambda_1$ | `0.3` | FIGO classification supervision |
| $\lambda_2$ | `0.2` | Physiological feature regression |
| $\lambda_3$ | `0.1` | FIGO rule consistency penalty |

---

### 3.4 Differentiable FIGO Rule Loss (`figo_rule_loss_normalized`)

This is the **core methodological novelty** of the framework. The function takes Z-normalized feature predictions and un-normalizes them to clinical units before applying four soft FIGO 2015 rule penalties:

$$\hat{\mathbf{f}} = \hat{\mathbf{f}}_{norm} \odot \boldsymbol{\sigma}_{feat} + \boldsymbol{\mu}_{feat} \quad \text{(un-normalize to clinical units)}$$

**Rule A — Baseline Deviation Penalty** *(penalizes Normal prediction when baseline is outside 110–160 bpm)*:
$$\text{Pen}_A = \left(\frac{\text{ReLU}(110 - \hat{f}_0)}{50} + \frac{\text{ReLU}(\hat{f}_0 - 160)}{50}\right) \cdot \hat{p}_{Normal}$$

**Rule B — LTV Variability Penalty** *(penalizes Normal prediction when LTV is outside 5–25 bpm)*:
$$\text{Pen}_B = \left(\frac{\text{ReLU}(5 - \hat{f}_2)}{20} + \frac{\text{ReLU}(\hat{f}_2 - 25)}{20}\right) \cdot \hat{p}_{Normal}$$

**Rule C — Pathological Deceleration Penalty** *(late or prolonged decels suppress Normal prediction)*:
$$\text{Pen}_C = \left(\text{softplus}(\hat{f}_5) + 2 \cdot \text{softplus}(\hat{f}_7)\right)_{\leq 5} \cdot \hat{p}_{Normal}$$

**Rule D — Pathological Baseline Penalty** *(baseline < 100 or prolonged decels suppress non-Pathological)*:
$$\text{Pen}_D = \left(\frac{\text{ReLU}(100 - \hat{f}_0)}{50} + \text{softplus}(\hat{f}_7)_{\leq 5}\right) \cdot (1 - \hat{p}_{Pathological})$$

$$\mathcal{L}_{knowledge} = \lambda_{cons} \cdot \overline{\text{Pen}_A + \text{Pen}_B + \text{Pen}_C + \text{Pen}_D}$$

> **Implementation note**: All penalties use normalized linear (not quadratic) deviation terms and clamp count-based terms to a maximum of 5.0 to prevent cold-start gradient explosions when the network has not yet learned to predict valid physiological ranges.

---

## 4. Training Protocol

### 4.1 5-Fold Stratified Patient-Level Cross-Validation
Identical to Phase 3 benchmarking:
- `StratifiedGroupKFold(n_splits=5)` on `patient_id` groups
- **All windows from one patient belong exclusively to one fold** — no patient data leakage
- Training: dynamic stride windows (~3,200 per fold) | Validation: fixed 10-min stride (~1,400 per fold)

### 4.2 Optimizer & Scheduler

| Parameter | Value |
| :--- | :--- |
| Optimizer | `AdamW` |
| Learning Rate | `1e-4` *(tuned)* |
| Weight Decay | `1e-4` |
| Scheduler | `CosineAnnealingLR` ($T_{max}$ = 50 epochs) |
| Gradient Clipping | `max_norm = 1.0` |
| Batch Size | `64` *(tuned)* |
| Mixed Precision | `torch.amp.autocast` (CUDA only) |

### 4.3 Checkpoint Strategy
- Best model weights saved per fold at epoch achieving **maximum validation AUROC**
- Output: `checkpoints/model8/model8_full_fold{k}_best.pth`
- Per-fold metrics exported to: `checkpoints/model8/results/model8_full_cv_results.json`

---

## 5. Ablation Study Design

Four sequential variants are trained to isolate the contribution of each knowledge component:

```
                Ablation Hierarchy
                ─────────────────
    distress_only         L = L_distress
         │
         ▼
    + plus_figo           L = L_distress + λ₁·L_FIGO
         │
         ▼
    + plus_features       L = L_distress + λ₁·L_FIGO + λ₂·L_features
         │
         ▼
    + full (Model 8)      L = L_distress + λ₁·L_FIGO + λ₂·L_features + λ₃·L_knowledge
```

| Variant | Research Question Answered |
| :--- | :--- |
| `distress_only` | Does PatchTST standalone = Phase 3 AUROC (0.7504)? Sanity check. |
| `plus_figo` | How much does FIGO auxiliary supervision contribute? |
| `plus_features` | How much does physiological feature regression contribute on top? |
| `full` | Does the FIGO rule consistency penalty further improve over supervised heads alone? |

---

## 6. Statistical Significance Testing

Per `AI_AGENT_RULES.md` §2.2, Model 8 must be validated against the standalone PatchTST baseline through paired statistical tests:

### 6.1 Wilcoxon Signed-Rank Test
Applied to AUROC and Sens@90%Spec distributions across the **5 paired out-of-fold scores**:
- **Null hypothesis ($H_0$)**: No difference in metric distribution between `distress_only` (Model 7 reproduction) and `full` (Model 8).
- **Alternative ($H_1$)**: Model 8 metric is higher.
- **Significance threshold**: $\alpha = 0.05$; a $p < 0.05$ confirms statistical significance.

### 6.2 DeLong AUC Comparison Test
Compares the two ROC curves directly using the DeLong (1988) nonparametric method. Produces a $Z$-statistic and $p$-value for AUROC difference.

**Implementation**: `src/training/statistical_tests.py`
```python
from src.training.statistical_tests import run_all_significance_tests
report = run_all_significance_tests(baseline_fold_results, model8_fold_results)
```

---

## 7. Parameter Count

| Component | Parameters |
| :--- | :--- |
| PatchTST Backbone (n_layers=4, n_heads=4) | ~685,056 |
| DistressHead (128→64→1) | 8,321 |
| FIGOHead (128→64→3) | 8,515 |
| ClinicalFeatureHead (128→64→8) | 8,712 |
| **Total (KnowledgeInfusedFramework)** | **~909,260** |

---

## 8. Dry-Run Validation Results

All 4 ablation variants have been validated via dry-run (forward + backward pass on dummy data):

```
distress_only  → l_distress: 2.77                                          ✓ PASS
plus_figo      → l_distress: 3.36  | l_figo: 1.10                          ✓ PASS
plus_features  → l_distress: 1.75  | l_figo: 1.26  | l_features: 1.45     ✓ PASS
full           → l_distress: 1.63  | l_figo: 0.98  | l_features: 1.04
                 l_knowledge: 2.60                                          ✓ PASS
```

All loss components are on the same ~1–3 scale. Shapes validated:
- `distress_logit`: `(8, 1)` ✓
- `figo_logits`:    `(8, 3)` ✓
- `feature_preds`:  `(8, 8)` ✓

---

## 9. Patent Non-Infringement (GE US12094611B2)

Model 8 satisfies all three non-infringement boundaries:

| Boundary | GE Patent | Our System |
| :--- | :--- | :--- |
| **Signal Representation** | Bounding box extraction of FHR graphical shapes | End-to-end continuous signal encoding $(B, 2, 4800) \to \mathbb{R}^{128}$ — no bounding boxes |
| **Pattern Matching** | Cross-temporal shape correlation confirmation loops | Single-pass patch transformer without post-hoc shape-matching loops |
| **Supervisory Signal** | Derived graphical parameters for clinician notification | Terminal fetal pH $\leq 7.15$ biochemical outcome + FIGO 2015 rule regularization |

---

## 10. Training Commands (Phase 4 Colab)

```bash
# 1. Dry-run validation (run first — ~10 seconds)
python src/training/train_knowledge_infused.py --dry_run --ablation all

# 2. Baseline reproduction — PatchTST standalone (~1 hour on T4)
python src/training/train_knowledge_infused.py --ablation distress_only

# 3. Full ablation study — all 4 variants sequentially (~4 hours on T4)
python src/training/train_knowledge_infused.py --ablation all

# 4. Full Model 8 using config file
python src/training/train_knowledge_infused.py --config configs/model8_config.yaml

# 5. λ override for hyperparameter tuning
python src/training/train_knowledge_infused.py --lambda_figo 0.5 --lambda_knowledge 0.05
```

**Colab recommended execution order:**
```
Step 1: --dry_run --ablation all          (10 sec — verify GPU environment)
Step 2: --ablation distress_only          (~1 hr  — baseline metrics)
Step 3: --ablation all                    (~4 hrs — complete ablation study)
Step 4: λ sweep if needed                 (~2 hrs per sweep)
```

---

## 11. File Reference

| File | Purpose |
| :--- | :--- |
| [`src/models/knowledge_infused_framework.py`](file:///d:/projects/CTG-Fetal-Distress-Prediction/src/models/knowledge_infused_framework.py) | `KnowledgeInfusedFramework`, `DistressHead`, `FIGOHead`, `ClinicalFeatureHead` |
| [`src/models/patchtst.py`](file:///d:/projects/CTG-Fetal-Distress-Prediction/src/models/patchtst.py) | `PatchTSTEncoder` — backbone |
| [`src/training/train_knowledge_infused.py`](file:///d:/projects/CTG-Fetal-Distress-Prediction/src/training/train_knowledge_infused.py) | 5-fold CV training script; 4 ablation variants; CLI |
| [`src/training/multi_task_dataset.py`](file:///d:/projects/CTG-Fetal-Distress-Prediction/src/training/multi_task_dataset.py) | `MultiTaskCTGDataset` loading X, y_primary, y_figo, y_features |
| [`src/training/statistical_tests.py`](file:///d:/projects/CTG-Fetal-Distress-Prediction/src/training/statistical_tests.py) | Wilcoxon + DeLong significance testing |
| [`src/knowledge/figo.py`](file:///d:/projects/CTG-Fetal-Distress-Prediction/src/knowledge/figo.py) | `figo_rule_loss_normalized()`, `vectorized_classify_figo()` |
| [`configs/model8_config.yaml`](file:///d:/projects/CTG-Fetal-Distress-Prediction/configs/model8_config.yaml) | Canonical training config: λ values, backbone, paths |
| [`configs/tuned_hyperparameters.yaml`](file:///d:/projects/CTG-Fetal-Distress-Prediction/configs/tuned_hyperparameters.yaml) | Tuned PatchTST backbone hyperparameters |
| [`docs/model_inferences_log.md`](file:///d:/projects/CTG-Fetal-Distress-Prediction/docs/model_inferences_log.md) | Single source of truth for all benchmark metrics |

---

## 12. Benchmark Results (To Be Updated Post-Training)

> *This section will be populated after the Phase 4 Colab training run completes.*

### 12.1 Ablation Results (5-Fold CV Mean ± Std)

| Variant | AUROC | AUPRC | F1 Score | Sens@90%Spec |
| :--- | :--- | :--- | :--- | :--- |
| `distress_only` (PatchTST Repro) | — | — | — | — |
| `plus_figo` | — | — | — | — |
| `plus_features` | — | — | — | — |
| **`full` (Model 8)** | — | — | — | — |

### 12.2 Final Model 8 vs. All Baselines

| Model | AUROC | AUPRC | F1 | Sens@90%Spec |
| :--- | :--- | :--- | :--- | :--- |
| CNN1D | 0.6860 ± 0.0503 | 0.2560 ± 0.0544 | 0.3304 ± 0.0511 | 21.09% ± 10.76% |
| BiLSTM | 0.6544 ± 0.0409 | 0.2697 ± 0.0426 | 0.3435 ± 0.0338 | 24.19% ± 6.92% |
| GRU | 0.6881 ± 0.0627 | 0.2812 ± 0.0839 | 0.3027 ± 0.1080 | 25.05% ± 9.96% |
| TCN | 0.7154 ± 0.0797 | 0.2846 ± 0.0840 | 0.2413 ± 0.1079 | 25.65% ± 13.36% |
| MS-LSTM | 0.7263 ± 0.1103 | 0.3140 ± 0.0962 | 0.3668 ± 0.0666 | 29.94% ± 12.13% |
| PatchCTG | 0.6668 ± 0.0161 | 0.2872 ± 0.0334 | 0.3322 ± 0.0418 | 28.63% ± 5.06% |
| PatchTST (Model 7) | 0.7504 ± 0.0378 | 0.3820 ± 0.0800 | 0.4102 ± 0.0446 | 35.09% ± 6.65% |
| **Knowledge-Infused (Model 8)** 🎯 | **—** | **—** | **—** | **—** |

### 12.3 Statistical Significance (Post-Training)

| Test | Metric | W-statistic | p-value | Significant? |
| :--- | :--- | :--- | :--- | :--- |
| Wilcoxon | AUROC | — | — | — |
| Wilcoxon | Sens@90%Spec | — | — | — |
| DeLong | AUROC | — | — | — |
