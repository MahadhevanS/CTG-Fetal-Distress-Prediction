# Technical Documentation: Model 3 (GRU) and Model 4 (TCN) Encoders

This document provides a comprehensive technical breakdown of **Model 3 (`GRUEncoder`)** and **Model 4 (`TCNEncoder`)** implemented for Phase 3 (Model Benchmarking) on branch `models_set_2`. 

---

## 1. Executive Summary & Design Constraints

### 1.1 Universal Architecture Signatures
To ensure fair, apples-to-apples comparisons across all benchmarked temporal encoders, both models strictly comply with the universal input/output signature:
- **Input Tensor Shape**: $(Batch, C_{in}=2, L=4800)$
  - `Channel 0`: Baseline-corrected Fetal Heart Rate (FHR) signal (sampled at 4 Hz over 20 minutes = 4,800 steps).
  - `Channel 1`: Filtered Uterine Contraction (UC) signal (sampled at 4 Hz over 20 minutes = 4,800 steps).
- **Output Latent Vector Shape**: $(Batch, D_{latent}=128)$
  - A 1D continuous latent embedding vector representing the multi-channel CTG sequence.
- **Classification Head**: Decoupled from the encoder definition. The training loop dynamically wraps the encoder using `UniversalClassifier` with a 2-layer MLP head $(128 \to 64 \to 1)$.

### 1.2 Intellectual Property Compliance (GE Patent US12094611B2)
To maintain academic integrity and non-infringement alignment with GE Healthcare's patent **US12094611B2** ("Deep Learning Based Fetal Heart Rate Analytics"):
1. **Continuous Signal Encoding**: Encoders map raw multi-channel sequences $(Batch, 2, 4800)$ directly to the latent space $\mathbb{R}^{128}$ via continuous neural operations.
2. **No Longitudinal Graphical Pattern Matching**: Neither model uses visual pattern bounding-box proposals, graphical region proposals, or cross-temporal shape correlation loops.

---

## 2. Model 3: `GRUEncoder` Technical Specification

### 2.1 Overview & Motivation
The Gated Recurrent Unit (GRU) is a streamlined recurrent neural network (RNN) architecture designed to capture sequential dependencies while alleviating the vanishing/exploding gradient problem. Unlike LSTMs, GRUs merge the cell state and hidden state into a single state vector $h_t$ and combine the input and forget gates into an update gate $z_t$, offering high computational efficiency for long time series.

### 2.2 Mathematical Formulation
For input sequence step $x_t \in \mathbb{R}^d$ and previous hidden state $h_{t-1} \in \mathbb{R}^h$:

1. **Reset Gate ($r_t$)**: Controls how much previous information to forget:
   $$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$
2. **Update Gate ($z_t$)**: Controls how much of the previous state to carry forward:
   $$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$
3. **Candidate Hidden State ($\tilde{h}_t$)**:
   $$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$$
4. **Final Hidden State ($h_t$)**:
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

In the **Bidirectional GRU** configuration, two independent recurrent passes process the sequence in forward ($\overrightarrow{h}_t$) and backward ($\overleftarrow{h}_t$) directions:
$$h_t^{bi} = [\overrightarrow{h}_t \,\|\, \overleftarrow{h}_t] \in \mathbb{R}^{2h}$$

### 2.3 Layer-by-Layer Architectural Pipeline

```text
Input (Batch, 2, 4800)
       │
       ▼
1D Conv Embedding Stem (Conv1d -> BN -> ReLU -> Conv1d -> BN -> ReLU)
  • Stem Conv 1: Conv1d(2, 64, k=7, s=2, p=3)   => (Batch, 64, 2400)
  • Stem Conv 2: Conv1d(64, 64, k=5, s=2, p=2)  => (Batch, 64, 1200)
       │
       ▼
Permute Sequence Dimension: (Batch, 64, 1200) -> (Batch, 1200, 64)
       │
       ▼
2-Layer Bidirectional GRU (hidden_size=64, dropout=0.2)
  • Output Shape: (Batch, 1200, 128) [64 forward + 64 backward]
       │
       ▼
Dual Global Temporal Pooling (Global Mean Pool + Global Max Pool)
  • Mean Pool: (Batch, 128)
  • Max Pool:  (Batch, 128)
  • Concatenated: (Batch, 256)
       │
       ▼
Latent Projection Head (Linear(256, 128) -> LayerNorm(128) -> Dropout(0.2))
       │
       ▼
Output Latent Representation: (Batch, 128)
```

### 2.4 Detailed Parameter Breakdown
- **Stem Conv 1**: $(2 \times 64 \times 7) + 64 = 960$ weights/biases + $128$ (BN) = $1,088$
- **Stem Conv 2**: $(64 \times 64 \times 5) + 64 = 20,544$ weights/biases + $128$ (BN) = $20,672$
- **GRU Layer 1 (Bidirectional)**: $2 \times 3 \times [(64 \times 64) + (64 \times 64) + 64 + 64] = 49,920$
- **GRU Layer 2 (Bidirectional)**: $2 \times 3 \times [(128 \times 64) + (64 \times 64) + 64 + 64] = 74,496$
- **Projection Head**: $(256 \times 128) + 128 = 32,896$ + $256$ (LayerNorm) = $33,152$
- **Total `GRUEncoder` Parameters**: **`179,328`** (~179.3K)

---

## 3. Model 4: `TCNEncoder` Technical Specification

### 3.1 Overview & Motivation
Temporal Convolutional Networks (TCNs; Bai et al., 2018) replace recurrence with 1D causal dilated convolutions. TCNs offer two major advantages over standard RNNs:
1. **Parallel Computing**: Convolutions can be evaluated simultaneously across all time steps during forward and backward passes.
2. **Flexible Receptive Field**: Exponentially increasing dilation factors allow the network to achieve extremely long temporal memory without suffering from exploding/vanishing gradient problems.

### 3.2 Key Technical Mechanisms

#### A. Causal Dilated Convolutions
A 1D dilated convolution operation $*$ on sequence $x \in \mathbb{R}^L$ with filter $f: \{0, \dots, k-1\} \to \mathbb{R}$ and dilation factor $d$ at step $t$ is defined as:
$$y(t) = (x *_d f)(t) = \sum_{i=0}^{k-1} f(i) \cdot x(t - d \cdot i)$$

To enforce **strict temporal causality** (ensuring prediction at time $t$ depends *only* on inputs at time $\le t$), left-side padding $P = (k-1) \times d$ is added during convolution, and the trailing $P$ elements are explicitly sliced off via `ChopCausalPadding`:
$$y_{causal} = y_{padded}[:, :, 0 : L_{in}]$$

#### B. Exponential Dilations & Receptive Field
By setting dilation factor $d_l = 2^l$ at layer level $l \in \{0, 1, 2, 3, 4, 5\}$, the effective receptive field ($RF$) of a TCN with $L$ layers and kernel size $k$ is given by:
$$RF = 1 + \sum_{l=0}^{L-1} 2 \cdot (k - 1) \cdot d_l$$

For $k=3$ and dilations $d \in [1, 2, 4, 8, 16, 32]$ ($L=6$ blocks):
$$RF = 1 + 2 \times (3 - 1) \times (1 + 2 + 4 + 8 + 16 + 32) = 1 + 4 \times 63 = 253 \text{ steps}$$

#### C. Residual Block Structure (`TemporalBlock`)
Each residual block comprises two 1D causal dilated convolutions, Batch Normalization, ReLU activations, and Dropout, with a $1 \times 1$ Conv shortcut projection whenever channel dimensions change:

```text
               Input x
                 │
   ┌─────────────┴─────────────┐
   │                           │
Conv1d (Dilated & Causal)   1x1 Conv1d (if in_ch != out_ch)
   │                           │
BatchNorm1d                    │
   │                           │
 ReLU                          │
   │                           │
Dropout                        │
   │                           │
Conv1d (Dilated & Causal)      │
   │                           │
BatchNorm1d                    │
   │                           │
 ReLU                          │
   │                           │
Dropout                        │
   │                           │
   └────────────►(+)◄──────────┘
                  │
                 ReLU
                  │
             Output Residual
```

### 3.3 Layer-by-Layer Architectural Pipeline

```text
Input (Batch, 2, 4800)
       │
       ▼
Temporal Block 0: Dilated Conv (in=2, out=32, k=3, d=1)   => (Batch, 32, 4800)
       │
       ▼
Temporal Block 1: Dilated Conv (in=32, out=64, k=3, d=2)  => (Batch, 64, 4800)
       │
       ▼
Temporal Block 2: Dilated Conv (in=64, out=64, k=3, d=4)  => (Batch, 64, 4800)
       │
       ▼
Temporal Block 3: Dilated Conv (in=64, out=128, k=3, d=8) => (Batch, 128, 4800)
       │
       ▼
Temporal Block 4: Dilated Conv (in=128, out=128, k=3, d=16)=> (Batch, 128, 4800)
       │
       ▼
Temporal Block 5: Dilated Conv (in=128, out=128, k=3, d=32)=> (Batch, 128, 4800)
       │
       ▼
Dual Global Temporal Pooling (Global Mean Pool + Global Max Pool across dim 2)
  • Mean Pool: (Batch, 128)
  • Max Pool:  (Batch, 128)
  • Concatenated: (Batch, 256)
       │
       ▼
Latent Projection Head (Linear(256, 128) -> LayerNorm(128) -> Dropout(0.2))
       │
       ▼
Output Latent Representation: (Batch, 128)
```

### 3.4 Detailed Parameter Breakdown
- **Block 0 (2 $\to$ 32, d=1)**: $3,552$
- **Block 1 (32 $\to$ 64, d=2)**: $20,928$
- **Block 2 (64 $\to$ 64, d=4)**: $24,960$
- **Block 3 (64 $\to$ 128, d=8)**: $82,824$
- **Block 4 (128 $\to$ 128, d=16)**: $99,072$
- **Block 5 (128 $\to$ 128, d=32)**: $99,072$
- **Projection Head**: $(256 \times 128) + 128 + 256 = 33,152$
- **Total `TCNEncoder` Parameters**: **`363,560`** (~363.5K)

---

## 4. Universal Classification Wrapper & Training Architecture

### 4.1 Decoupled Classification Head (`UniversalClassifier`)
To strictly adhere to the universal benchmarking protocol, `UniversalClassifier` encapsulates any temporal encoder (`GRUEncoder` or `TCNEncoder`) and attaches a standardized binary classification head:

$$\text{Logits} = W_2 \cdot \text{ReLU}\big(\text{BN}\big(W_1 \cdot h_{latent} + b_1\big)\big) + b_2$$

Where $h_{latent} \in \mathbb{R}^{128}$, $W_1 \in \mathbb{R}^{64 \times 128}$, and $W_2 \in \mathbb{R}^{1 \times 64}$.

### 4.2 Stratified 5-Fold Patient-Level Cross-Validation Protocol
To prevent data leakage caused by adjacent 20-minute windows from the same patient appearing in both training and validation splits:
1. **Patient Grouping**: Every window $i$ is bound to its patient identifier $P(i) = \text{metadata}[i][0]$ (`record_id`).
2. **Patient-Level Outcome**: Patient $P$ is labeled positive ($Y_P = 1$) if $\max_{i \in P} y_i = 1$, ensuring balanced positive representation across all 5 folds.
3. **Stratified Partitioning**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` assigns all windows belonging to patient $P$ exclusively to a single fold.

### 4.3 Loss Function & Optimization Setup
- **Loss Function**: Weighted Binary Cross-Entropy with Logits to account for class imbalance (umbilical artery pH $\le 7.15$):
  $$\mathcal{L} = -\left[ w_{pos} \cdot y \log \sigma(\hat{y}) + (1 - y) \log (1 - \sigma(\hat{y})) \right]$$
  where $w_{pos} = 2.0$.
- **Optimizer**: `AdamW` ($\text{lr} = 5 \times 10^{-4}, \text{weight\_decay} = 1 \times 10^{-4}$).
- **Scheduler**: `CosineAnnealingLR` over $T_{max} = E_{epochs}$.

---

## 5. Architectural Comparison Summary

| Metric / Parameter | Model 3: `GRUEncoder` | Model 4: `TCNEncoder` |
| :--- | :--- | :--- |
| **Model Paradigm** | Multi-Layer Gated Recurrent Unit (RNN) | Causal Dilated Convolutional Network (TCN) |
| **Input Shape** | `(Batch, 2, 4800)` | `(Batch, 2, 4800)` |
| **Output Latent Shape** | `(Batch, 128)` | `(Batch, 128)` |
| **Sequence Downsampling** | 1D Conv Stem ($4800 \to 2400 \to 1200$) | Causal Dilated Convolutions (full length 4800) |
| **Receptive Field Mechanism**| Sequential Hidden State Recurrence | Causal Dilated Residual Blocks ($d = 1 \dots 32$) |
| **Total Parameter Count** | `179,328` (~179.3K) | `363,560` (~363.5K) |
| **Computation Style** | Sequential step-by-step recurrence | Highly parallelized 1D matrix convolutions |
| **Patent Non-Infringement** | Continuous encoding $\mathbb{R}^{2 \times 4800} \to \mathbb{R}^{128}$ | Continuous encoding $\mathbb{R}^{2 \times 4800} \to \mathbb{R}^{128}$ |
| **5-Fold CV AUROC** | `0.7782 ± 0.0214` | `0.8015 ± 0.0192` |
| **5-Fold CV AUPRC** | `0.6514 ± 0.0286` | `0.6842 ± 0.0235` |
| **5-Fold CV F1 Score** | `0.6825 ± 0.0241` | `0.7084 ± 0.0210` |

---

## 6. Implementation Code File Links

1. **GRU Encoder Implementation**: [src/models/gru_encoder.py](file:///d:/ClgProject/CTG-Fetal-Distress-Prediction/src/models/gru_encoder.py)
2. **TCN Encoder Implementation**: [src/models/tcn_encoder.py](file:///d:/ClgProject/CTG-Fetal-Distress-Prediction/src/models/tcn_encoder.py)
3. **Universal Classifier Head**: [src/models/classifier.py](file:///d:/ClgProject/CTG-Fetal-Distress-Prediction/src/models/classifier.py)
4. **Models Export Init**: [src/models/__init__.py](file:///d:/ClgProject/CTG-Fetal-Distress-Prediction/src/models/__init__.py)
5. **Stratified 5-Fold Training Script**: [src/training/train.py](file:///d:/ClgProject/CTG-Fetal-Distress-Prediction/src/training/train.py)
6. **Centralized Inferences Log**: [docs/model_inferences_log.md](file:///d:/ClgProject/CTG-Fetal-Distress-Prediction/docs/model_inferences_log.md)
