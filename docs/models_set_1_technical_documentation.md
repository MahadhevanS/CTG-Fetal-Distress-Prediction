# Technical Documentation: Baseline Temporal Encoders (1D CNN & BiLSTM)

## Executive Summary

This document provides a comprehensive technical reference for the implementation, architectural specifications, tensor transformations, and execution workflows of the first two baseline temporal encoders evaluated in Phase 3 of the **Knowledge-Infused Multi-Task Temporal Deep Learning Framework for CTG Fetal Distress Prediction**:

1. **1D CNN Encoder** (`CNN1DEncoder`) – Local temporal pattern extraction via residual convolutional blocks.
2. **Bidirectional LSTM Encoder** (`BiLSTMEncoder`) – Long-term sequential dependency tracking across the continuous CTG time window.

Both architectures strictly adhere to the universal project constraints:
- **Universal Input Signature**: $(Batch, 2, 4800)$ where Channel 0 is Fetal Heart Rate (FHR) and Channel 1 is Uterine Contractions (UC) sampled over a continuous 20-minute window (4 Hz, 4,800 time steps).
- **Universal Latent Output Signature**: $(Batch, 128)$ – a 128-dimensional dense continuous sequence embedding vector $\mathbf{h} \in \mathbb{R}^{128}$.
- **Patent US12094611B2 Differentiation**: Continuous end-to-end signal representation directly to latent space $\mathbb{R}^{128}$ without longitudinal shape matching, pattern bounding-box detection, or template correlation loops.

---

## 1. Universal Input & Output Specification

```text
                                  ┌─────────────────────────────┐
                                  │   Continuous CTG Window     │
                                  │    Shape: (Batch, 2, 4800)  │
                                  └──────────────┬──────────────┘
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         ▼                                               ▼
         ┌───────────────────────────────┐               ┌───────────────────────────────┐
         │     1D CNN Temporal Encoder   │               │     BiLSTM Temporal Encoder   │
         │  (Local Feature Extraction)   │               │ (Sequential Dependency Tracking)│
         └───────────────┬───────────────┘               └───────────────┬───────────────┘
                         │                                               │
                         └───────────────────────┬───────────────────────┘
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │   Latent Embedding Vector   │
                                  │     Shape: (Batch, 128)     │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │    Temporal Classifier MLP  │
                                  │      Shape: (Batch, 1)      │
                                  └─────────────────────────────┘
```

| Parameter | Specification | Description |
| :--- | :--- | :--- |
| **Input Shape** | `(Batch_Size, 2, 4800)` | Channel 0 = FHR (bpm), Channel 1 = UC (mmHg). |
| **Encoder Latent Output** | `(Batch_Size, 128)` | Compact latent sequence embedding. |
| **Classifier Output** | `(Batch_Size, 1)` | Logit for binary fetal distress ($\text{pH} \le 7.15$). |
| **Data Normalization** | Per-channel Z-Score | Persistent channel mean and std fitted on training set. |

---

## 2. 1D CNN Encoder (`CNN1DEncoder`)

### 2.1 Architectural Design

The 1D CNN encoder ([cnn1d_encoder.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/models/cnn1d_encoder.py)) is designed to capture multi-scale local temporal variations, high-frequency oscillations, accelerations, and decelerations in the CTG signals using 1D convolutional residual blocks.

```text
Input (B, 2, 4800)
    │
    ▼
Stem Convolution (Conv1d k=7, s=2, p=3) ──> BatchNorm1d ──> ReLU ──> MaxPool1d (k=3, s=2, p=1)
    │  [Shape: (B, 32, 1200)]
    ▼
Residual Block 1 (In: 32, Out: 32, Stride: 1)
    │  [Shape: (B, 32, 1200)]
    ▼
Residual Block 2 (In: 32, Out: 64, Stride: 2) + 1x1 Conv Shortcut
    │  [Shape: (B, 64, 600)]
    ▼
Residual Block 3 (In: 64, Out: 128, Stride: 2) + 1x1 Conv Shortcut
    │  [Shape: (B, 128, 300)]
    ▼
Adaptive Global Average Pooling (AdaptiveAvgPool1d(1)) ──> Squeeze
    │  [Shape: (B, 128)]
    ▼
Linear Projection Layer (Linear(128, 128))
    │
    ▼
Output Embeddings (B, 128)
```

### 2.2 Detailed Layer Specifications

1. **Stem Module**:
   - `Conv1d(in_channels=2, out_channels=32, kernel_size=7, stride=2, padding=3, bias=False)`
   - `BatchNorm1d(32)` + `ReLU(inplace=True)`
   - `MaxPool1d(kernel_size=3, stride=2, padding=1)`
   - *Function*: Aggregates raw 4,800 steps into 1,200 feature steps while expanding channel depth from 2 to 32.

2. **Residual Blocks (`ResidualBlock`)**:
   Each residual block contains two 1D convolution layers with batch normalization, non-linear activation, and a residual shortcut path:
   $$\mathbf{y} = \text{ReLU}(\mathcal{F}(\mathbf{x}, \{W_i\}) + \mathcal{W}_s(\mathbf{x}))$$
   - **Layer 1**: 32 channels $\rightarrow$ 32 channels, stride 1 (Preserves sequence length 1,200).
   - **Layer 2**: 32 channels $\rightarrow$ 64 channels, stride 2 (Downsamples sequence length to 600).
   - **Layer 3**: 64 channels $\rightarrow$ 128 channels, stride 2 (Downsamples sequence length to 300).

3. **Global Temporal Aggregation & Projection**:
   - `AdaptiveAvgPool1d(1)` compresses sequence length from 300 to 1 across all 128 feature maps.
   - `Linear(128, 128)` projects pooled features into the final latent space.

### 2.3 Parameter Count & Complexity

- **Trainable Parameters**: `135,169`
- **Memory Footprint**: Low (~0.5 MB model weights).
- **Execution Speed**: Extremely fast on CPU (~1-2 seconds per epoch for 16-sample batches).

---

## 3. Bidirectional LSTM Encoder (`BiLSTMEncoder`)

### 3.1 Architectural Design

The BiLSTM encoder ([bilstm_encoder.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/models/bilstm_encoder.py)) models forward and backward temporal dependencies across the complete sequence window to capture contextual relationships between uterine contractions and fetal heart rate responses over time.

```text
Input (B, 2, 4800)
    │
    ▼
Permute Dimension (B, 4800, 2)  [batch_first=True]
    │
    ▼
2-Layer Bidirectional LSTM (Hidden Size = 64, Dropout = 0.2)
    │  Forward Hidden State  h_forward  (B, 64)
    │  Backward Hidden State h_backward (B, 64)
    ▼
Concatenation [h_forward; h_backward] (B, 128)
    │
    ▼
Linear Projection Layer (Linear(128, 128))
    │
    ▼
Output Embeddings (B, 128)
```

### 3.2 Detailed Layer Specifications

1. **Input Permutation**:
   - PyTorch LSTM expects sequence step dimension in the second position when `batch_first=True`:
     $$\mathbf{x}_{\text{permuted}} = \text{permute}(0, 2, 1) \quad \implies \quad (Batch, 2, 4800) \to (Batch, 4800, 2)$$

2. **Bidirectional LSTM Core**:
   - `nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)`
   - Computes forward sequence representations $\overrightarrow{h}_t$ and backward sequence representations $\overleftarrow{h}_t$ simultaneously across all 4,800 steps.

3. **Hidden State Extraction & Fusion**:
   - Extracts the last time-step hidden state of the top layer for both directions:
     $$\mathbf{h}_{\text{forward}} = h_{N}^{(L, \rightarrow)} \in \mathbb{R}^{64}, \quad \mathbf{h}_{\text{backward}} = h_{1}^{(L, \leftarrow)} \in \mathbb{R}^{64}$$
   - Concatenates both vectors into a single fused temporal descriptor:
     $$\mathbf{h}_{\text{fused}} = [\mathbf{h}_{\text{forward}} \;\|\; \mathbf{h}_{\text{backward}}] \in \mathbb{R}^{128}$$

4. **Linear Projection Layer**:
   - `Linear(128, 128)` maps the fused state to the final 128-dimensional output space.

### 3.3 Parameter Count & Complexity

- **Trainable Parameters**: `158,977`
- **Memory Footprint**: Light (~0.6 MB model weights).
- **Execution Speed**: Sequential recurrent execution over 4,800 steps requires higher computational time per epoch compared to 1D CNN.

---

## 4. Universal Classification Head (`TemporalClassifier`)

To evaluate both models fairly without altering their pure encoder signatures, [src/training/train.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/training/train.py) wraps each encoder inside a standardized multi-layer perceptron (MLP) classification head:

$$\hat{y}_{\text{logit}} = W_2 \cdot \text{Dropout}\left(\text{ReLU}\left(W_1 \cdot \mathbf{h} + b_1\right), p=0.2\right) + b_2$$

```python
class TemporalClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, latent_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.encoder(x)   # Shape: (Batch, 128)
        logits = self.classifier(features) # Shape: (Batch, 1)
        return logits.squeeze(-1)
```

---

## 5. Comparative Architectural Matrix

| Architectural Feature | 1D CNN Encoder (`CNN1DEncoder`) | BiLSTM Encoder (`BiLSTMEncoder`) |
| :--- | :--- | :--- |
| **Primary Mechanism** | 1D Residual Convolutions | Recurrent Bidirectional Gates |
| **Input Shape** | `(Batch, 2, 4800)` | `(Batch, 2, 4800)` |
| **Output Shape** | `(Batch, 128)` | `(Batch, 128)` |
| **Parameter Count** | `135,169` | `158,977` |
| **Temporal Receptive Field** | Local hierarchy via residual strides | Global window-wide sequential history |
| **Computational Bottleneck** | Parallelizable convolution ops | Sequential step-by-step unrolling (4,800 steps) |
| **Convergence Behavior** | Fast convergence (Best Val Loss at Epoch 4) | Steady convergence (Best Val Loss at Epoch 8) |
| **Sensitivity to Imbalance** | Overfits past epoch 4 | Collapses to majority class under unweighted BCE |

---

## 6. Execution & CLI Interface

The training and evaluation pipeline ([train.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/training/train.py)) provides CLI flags to execute each model separately or together:

### Command Syntax

```bash
# Run 1D CNN Encoder
python src/training/train.py --model cnn1d

# Run BiLSTM Encoder
python src/training/train.py --model bilstm

# Run Both Models
python src/training/train.py --model all

# Dry-run validation check (tensor shape verification without dataset)
python src/training/train.py --model cnn1d --dry_run
```

---

## 7. Intellectual Property & Patent Differentiation (US12094611B2)

To maintain strict non-infringement alignment with GE Healthcare patent **US12094611B2**:

1. **Continuous Signal Encoding**: Both `CNN1DEncoder` and `BiLSTMEncoder` encode continuous multi-channel signals $(Batch, 2, 4800)$ directly into a latent vector $\mathbf{h} \in \mathbb{R}^{128}$.
2. **No Longitudinal Pattern Bounding Boxes**: Neither model uses visual pattern bounding boxes, image segmentation, or discrete graphical decelerations/acceleration extraction.
3. **No Cross-Temporal Correlation Loops**: Neither model performs post-hoc shape-matching confirmation loops across distant time windows.

---

## 8. Summary & Maintenance

- **Source Code Locations**:
  - 1D CNN Encoder: [src/models/cnn1d_encoder.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/models/cnn1d_encoder.py)
  - BiLSTM Encoder: [src/models/bilstm_encoder.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/models/bilstm_encoder.py)
  - Model Registry: [src/models/__init__.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/models/__init__.py)
  - Training Script: [src/training/train.py](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/training/train.py)
- **Inference Log**: Evaluation logs and hyperparameters are documented in [docs/model_inferences_log.md](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/docs/model_inferences_log.md).
