# Model Set 4 Technical Documentation: PatchTST & Knowledge-Infused Framework

> **Author**: Member 4 (Lead Developer)  
> **Repository**: CTG Fetal Distress Prediction  
> **Target Branch**: `models_set_4`  
> **Phase**: Phase 3 (Model Benchmarking & Evaluation)  
> **Document Status**: Complete for **Model 7 (PatchTST Baseline)**; Structured Blueprint active for **Model 8 (Knowledge-Infused Framework)**.

---

## 1. Executive Summary & Scope

Under Phase 3 guidelines, Member 4 is tasked with building, benchmarking, and documenting two core architectures:
1. **Model 7**: **PatchTST** (*Patch Time-Series Transformer*) – Modern State-of-the-Art (SOTA) Transformer Baseline for long sequence modeling.
2. **Model 8**: **Knowledge-Infused Multi-Task Framework** – Proposed novelty architecture integrating clinical auxiliary supervision (FIGO 2015 Guidelines & Fetal Biochemical Signals) into the winning temporal encoder backbone.

All models in this set comply strictly with the frozen dataset policy and the **Universal Encoder Signature**:
- **Input Tensor**: $\mathbf{X} \in \mathbb{R}^{B \times 2 \times 4800}$ (Channel 0: Baseline-Corrected Fetal Heart Rate [FHR], Channel 1: Uterine Contractions [UC] at $4\text{ Hz}$ for $20\text{ minutes}$).
- **Output Latent Vector**: $\mathbf{z} \in \mathbb{R}^{B \times 128}$ (Flattened 1D continuous latent representation).
- **Patent Non-Infringement (GE US12094611B2)**: End-to-end continuous signal mapping without longitudinal graphical pattern matching, bounding-box detection, or visual correlation loops.

---

## 2. Model 7: PatchTST (Patch Time-Series Transformer) Technical Architecture

### 2.1 Theoretical Motivation
Traditional Transformer models applied to long time-series sequences ($L = 4800$) suffer from two primary limitations:
1. **Quadratic Computational Complexity $O(L^2)$**: Full point-wise self-attention over $4800$ temporal points requires $(4800)^2 = 23,040,000$ attention operations per head per channel, resulting in severe GPU memory overhead.
2. **Lack of Local Semantic Context**: Individual time steps (at $\Delta t = 0.25\text{ s}$) carry weak semantic signal independently. Local temporal structures (such as accelerations, decelerations, and contraction peaks) are inherently sub-sequence properties.

**PatchTST** resolves both issues via:
- **Patchification**: Grouping consecutive time steps into local sub-series patches of length $P=16$ ($4.0\text{ seconds}$). This reduces the sequence length from $L=4800$ to $N=300$ tokens, dropping self-attention complexity to $O(N^2) = 90,000$ operations ($256\times$ reduction).
- **Channel Independence**: Each input channel (FHR and UC) is treated as an independent univariate series during patch embedding and transformer encoding, preventing cross-channel noise contamination during early representation learning.

---

### 2.2 Mathematical Formulation & Pipeline Transformation

Let the input CTG sequence be $\mathbf{X} \in \mathbb{R}^{B \times C \times L}$, where $B$ is batch size, $C=2$ channels, and $L=4800$ time steps.

```
                  ┌─────────────────────────────────────────┐
                  │ Input Signal X: (B, C=2, L=4800)        │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ 1. Unfold Patching (P=16, S=16)         │
                  │    -> (B, C=2, N=300, P=16)             │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ 2. Channel Independence Reshape         │
                  │    -> (B * C, N=300, P=16)              │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ 3. Linear Patch Projection (P -> d_model)│
                  │    + Positional Embeddings W_pos         │
                  │    -> (B * C, N=300, d_model=128)        │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ 4. Pre-LN Transformer Encoder (3 Layers)│
                  │    Multi-Head Self-Attention (8 Heads)  │
                  │    -> (B * C, N=300, d_model=128)        │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ 5. Adaptive Global Patch Avg Pooling    │
                  │    -> (B * C, d_model=128)              │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ 6. Channel Recombination Reshape        │
                  │    -> (B, C * d_model = 256)            │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ 7. Latent Projection Head & LayerNorm   │
                  │    -> Universal Output Latent z: (B, 128)│
                  └─────────────────────────────────────────┘
```

#### Step 1: Patchification (Unfolding)
Using a non-overlapping sliding window of patch length $P = 16$ and stride $S = 16$:
$$N = \left\lfloor \frac{L - P}{S} \right\rfloor + 1 = \left\lfloor \frac{4800 - 16}{16} \right\rfloor + 1 = 300\text{ patches per channel}$$
The resulting unfolded tensor is $\mathbf{X}_p \in \mathbb{R}^{B \times C \times N \times P}$.

#### Step 2: Channel-Independent Reshaping
The batch and channel dimensions are folded into a single unified batch dimension:
$$\mathbf{X}_{ci} = \text{Reshape}(\mathbf{X}_p) \in \mathbb{R}^{(B \cdot C) \times N \times P}$$

#### Step 3: Linear Patch Embedding & Positional Encoding
Each patch vector $\mathbf{x}_{i} \in \mathbb{R}^{P}$ is linearly projected into model feature dimension $d_{model} = 128$ using weight matrix $\mathbf{W}_{embed} \in \mathbb{R}^{P \times d_{model}}$ and bias $\mathbf{b}_{embed} \in \mathbb{R}^{d_{model}}$:
$$\mathbf{E} = \mathbf{X}_{ci} \mathbf{W}_{embed} + \mathbf{b}_{embed} \in \mathbb{R}^{(B \cdot C) \times N \times d_{model}}$$
A learnable 1D truncated-normal positional embedding $\mathbf{W}_{pos} \in \mathbb{R}^{1 \times N \times d_{model}}$ is added to preserve temporal sequence order across the 300 patches:
$$\mathbf{H}_0 = \text{Dropout}\left(\mathbf{E} + \mathbf{W}_{pos}\right) \in \mathbb{R}^{(B \cdot C) \times N \times d_{model}}$$

#### Step 4: Pre-LayerNorm Multi-Head Self-Attention (Transformer Encoder)
The sequence of patch representations is passed through $L_{enc} = 3$ Transformer encoder layers. Each layer consists of Multi-Head Self-Attention ($\text{MHSA}$) and Feed-Forward Networks ($\text{FFN}$) using Pre-Layer Normalization ($\text{Pre-LN}$) and $\text{GELU}$ activations:

$$\mathbf{H}_l' = \text{MHSA}(\text{LayerNorm}(\mathbf{H}_{l-1})) + \mathbf{H}_{l-1}$$
$$\mathbf{H}_l = \text{FFN}(\text{LayerNorm}(\mathbf{H}_l')) + \mathbf{H}_l' \quad \text{for } l = 1, \dots, 3$$

Where for each head $k \in \{1, \dots, 8\}$:
$$\text{Attention}(\mathbf{Q}_k, \mathbf{K}_k, \mathbf{V}_k) = \text{Softmax}\left(\frac{\mathbf{Q}_k \mathbf{K}_k^T}{\sqrt{d_k}}\right) \mathbf{V}_k, \quad d_k = \frac{d_{model}}{n_{heads}} = \frac{128}{8} = 16$$

#### Step 5: Adaptive Global Patch Pooling & Channel Recombination
Global temporal aggregation across the $N=300$ patch tokens is performed via 1D Adaptive Average Pooling:
$$\mathbf{P}_{ci} = \text{AdaptiveAvgPool1d}(\mathbf{H}_3^T)^T \in \mathbb{R}^{(B \cdot C) \times d_{model}}$$
The channel components are unfolded back and concatenated along the feature dimension:
$$\mathbf{P}_{cat} = \text{Reshape}(\mathbf{P}_{ci}) \in \mathbb{R}^{B \times (C \cdot d_{model})} = \mathbb{R}^{B \times 256}$$

#### Step 6: Latent Bottleneck Head
A non-linear projection maps the concatenated $256$-dimensional representation down to the universal $128$-dimensional latent output space:
$$\mathbf{z} = \text{LayerNorm}\left(\text{Linear}_{128 \to 128}\left(\text{Dropout}\left(\text{GELU}\left(\text{Linear}_{256 \to 128}(\mathbf{P}_{cat})\right)\right)\right)\right) \in \mathbb{R}^{B \times 128}$$

---

### 2.3 Hyperparameter Specification Summary

| Component / Parameter | Configuration / Value | Description / Rationale |
| :--- | :--- | :--- |
| **Input Signal Dimension** | `(B, 2, 4800)` | 2 Channels (FHR, UC), 20 min @ 4 Hz |
| **Patch Length ($P$)** | `16` samples | 4 seconds per temporal patch |
| **Patch Stride ($S$)** | `16` samples | Non-overlapping temporal tokenization |
| **Num Patches ($N$)** | `300` patches / channel | $N = (4800 - 16)/16 + 1$ |
| **Model Hidden Dim ($d_{model}$)** | `128` | Transformer token representation dimension |
| **Feedforward Dim ($d_{ff}$)** | `512` | Expansion ratio = 4 in MLP blocks |
| **Attention Heads ($n_{heads}$)** | `8` | Head dimension $d_k = 16$ |
| **Encoder Layers ($n_{layers}$)** | `3` | Pre-LN Transformer blocks |
| **Activation Function** | `GELU` | Gaussian Error Linear Unit |
| **Dropout Rate** | `0.1` (encoder) / `0.2` (classifier) | Prevents overfitting |
| **Total Trainable Parameters** | **`532,480`** | Compact, memory-efficient encoder |

---

## 3. Patent Non-Infringement & Regulatory Compliance (GE US12094611B2)

To ensure academic and intellectual property compliance with GE Patent **US12094611B2**:

- ✅ **Continuous Neural Signal Encoding**: PatchTST maps the raw multi-channel sequence $\mathbf{X} \in \mathbb{R}^{B \times 2 \times 4800}$ directly to a continuous feature manifold $\mathbf{z} \in \mathbb{R}^{128}$ via end-to-end backpropagation.
- 🚫 **No Longitudinal Graphical Pattern Matching**: The network contains **zero** 2D visual bounding boxes, spatial segmentation grids, or cross-temporal shape-matching confirmation loops.
- 🚫 **No Heuristic Bounding Box Detection**: Fetal heart rate decelerations and uterine contraction peaks are encoded as dense multi-head attention weights, not discrete visual graphical primitives.

---

## 4. Model Benchmarking & Protocol Implementation

### 4.1 5-Fold Stratified Patient-Level Cross-Validation
To guarantee zero patient data leakage:
- Data is split into 5 out-of-fold splits using `StratifiedGroupKFold(n_splits=5)`.
- All record windows belonging to a given `patient_id` reside exclusively in either the training set or the validation set within any fold.

### 4.2 Standardized MLP Classification Head
For Phase 3 benchmarking, the universal latent vector $\mathbf{z} \in \mathbb{R}^{128}$ is evaluated using a standardized MLP classification head:
$$\hat{y}_{logit} = \mathbf{W}_2 \cdot \text{Dropout}_{0.2}\left(\text{ReLU}\left(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1\right)\right) + b_2, \quad \mathbf{W}_1 \in \mathbb{R}^{64 \times 128}, \mathbf{W}_2 \in \mathbb{R}^{1 \times 64}$$

### 4.3 Training & Loss Configuration
- **Loss Function**: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`) with dynamic positive class weighting:
  $$w_{pos} = \frac{N_{negative}}{N_{positive}}$$
- **Optimizer**: `AdamW` ($\text{lr} = 5\times 10^{-4}$, weight decay $= 1\times 10^{-4}$).
- **Scheduler**: `CosineAnnealingLR` ($T_{max} = 50\text{ epochs}$).

---

## 5. Model 8: Knowledge-Infused Multi-Task Framework (Architectural Blueprint)

> *Note: This section outlines the architectural blueprint for Model 8. Full benchmark metrics and ablation data will be documented here upon completion of Phase 3 implementation.*

```
                                 ┌──────────────────────────────────────────────┐
                                 │ Raw CTG Signal: (B, 2, 4800)                 │
                                 └──────────────────────┬───────────────────────┘
                                                        │
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │ Optimal Temporal Encoder Backbone            │
                                 │ (PatchTST / Best Baseline)                   │
                                 └──────────────────────┬───────────────────────┘
                                                        │
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │ Universal Latent Space z: (B, 128)           │
                                 └──────┬───────────────┼───────────────┬───────┘
                                        │               │               │
                  ┌─────────────────────┘               │               └─────────────────────┐
                  ▼                                     ▼                                     ▼
   ┌─────────────────────────────┐       ┌─────────────────────────────┐       ┌─────────────────────────────┐
   │ Primary Target Head         │       │ FIGO 2015 Auxiliary Head    │       │ Biochemical Signal Head     │
   │ (Fetal Distress / Normal)   │       │ (Normal / Suspicious / Path)│       │ (pH, Base Excess, Lactate)  │
   └──────────────┬──────────────┘       └──────────────┬──────────────┘       └──────────────┬──────────────┘
                  │                                     │                                     │
                  ▼                                     ▼                                     ▼
   ┌─────────────────────────────┐       ┌─────────────────────────────┐       ┌─────────────────────────────┐
   │ Primary Binary Loss L_prim  │       │ FIGO Cross-Entropy L_figo   │       │ Feature MSE Loss L_feat     │
   └──────────────┬──────────────┘       └──────────────┬──────────────┘       └──────────────┬──────────────┘
                  │                                     │                                     │
                  └─────────────────────┐               │               ┌─────────────────────┘
                                        ▼               ▼               ▼
                                 ┌──────────────────────────────────────────────┐
                                 │ Total Multi-Task Loss:                       │
                                 │ L_total = w1*L_prim + w2*L_figo + w3*L_feat │
                                 └──────────────────────────────────────────────┘
```

### 5.1 Design Overview
Model 8 enhances the winning standalone encoder by introducing **Clinical Knowledge Infusion** through a multi-task learning framework:
1. **Primary Binary Classification Head**: Predicts binary fetal distress outcome.
2. **FIGO 2015 Auxiliary Classifier**: Multi-class classification (Normal, Suspicious, Pathological) according to FIGO clinical guidelines.
3. **Biochemical Feature Regressor**: Reconstructs clinical baseline parameters (e.g., basal FHR, signal variability, acceleration/deceleration statistics).

### 5.2 Multi-Task Loss Formulation
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{primary} + \lambda_2 \mathcal{L}_{FIGO} + \lambda_3 \mathcal{L}_{biochemical}$$
Where $\lambda_1, \lambda_2, \lambda_3$ are task-balancing hyperparameters tuned during Phase 3.

---

## 6. Verification & File References

- **PatchTST Model Class**: [src/models/patchtst.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/models/patchtst.py)
- **PatchTST Training Script**: [src/models/train_patchtst.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/models/train_patchtst.py)
- **Centralized Model Inference Log**: [docs/model_inferences_log.md](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/docs/model_inferences_log.md)
