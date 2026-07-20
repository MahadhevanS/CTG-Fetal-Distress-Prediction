# Model Comparison & Evaluation Plan

## Objective
Evaluate representative deep learning architectures on the finalized CTG dataset to identify the most suitable temporal encoder for the proposed **Knowledge-Infused Multi-Task Temporal Deep Learning Framework**. 

The objective is **not** to develop a new baseline architecture, but to rigorously identify the strongest temporal backbone that will subsequently be integrated into the finalized multi-task framework.

---

## 1. Comparison Models & Experimental Progression

The evaluation is structured into two strict phases to prevent unfair comparisons (e.g., evaluating a raw encoder against a multi-task framework).

### Phase 1: Temporal Encoder Benchmark
Identify the most capable temporal backbone by evaluating them purely on the primary classification task.

| Category | Model | Purpose |
| :--- | :--- | :--- |
| **Classical** | 1D CNN | Learns local temporal patterns and high-frequency features from CTG signals. |
| **Classical** | BiLSTM | Captures long-term sequential dependencies across the entire window. |
| **Classical** | GRU | Lightweight recurrent baseline with lower computational cost and faster convergence. |
| **Classical** | TCN (Temporal Convolutional Network) | Strong convolution-based temporal sequence model, offering parallelization. |
| **Literature** | Multi-Scale LSTM (Rao et al., 2024) | Reproduction of a representative CTG-specific architecture. |
| **Literature** | PatchCTG (Khan et al., 2025) | Transformer-based CTG model representing recent attention-based approaches. |
| **Modern SOTA**| PatchTST | Singular representative for modern general-purpose time-series forecasting. |

**Outcome:** The single strongest temporal encoder is selected.

### Phase 2: Final Framework Comparison
Compare the complete proposed system against the strongest standalone systems.

| Model | Description |
| :--- | :--- |
| **Best Baseline Encoder** | The winner from Phase 1. |
| **Multi-Scale LSTM** | Literature baseline. |
| **PatchCTG** | Literature baseline. |
| **PatchTST** | Modern SOTA baseline. |
| **Knowledge-Infused Multi-Task Framework** | **The Proposed Framework** (Best Encoder + Multi-Task Heads). |

---

### 1.1 Intellectual Property & Patent Differentiation (GE US12094611B2 Alignment)

To ensure the selected models and evaluation rerun remain strictly non-infringing upon GE Healthcare's patent **US12094611B2** ("Deep Learning Based Fetal Heart Rate Analytics"), all benchmarked architectures must comply with three core design boundaries:

1. **Continuous Signal Encoding vs. Graphical Pattern Matching**:
   - GE Patent: Identifies bounding boxes / graphical shapes of FHR patterns (accelerations, decelerations).
   - Our Model Rerun: Encodes raw continuous 20-minute multi-channel signals $(Batch, 2, 4800)$ directly into a latent vector $h \in \mathbb{R}^{128}$ without explicit graphical pattern bounding box extraction.

2. **Continuous Representation vs. Longitudinal Correlation Confirmation**:
   - GE Patent: Runs cross-temporal correlation matching to confirm whether a pattern at time $t_1$ matches a pattern at time $t_2$.
   - Our Model Rerun: Processes the temporal window as a unified continuous representation via standard temporal backbones (1D CNN, BiLSTM, GRU, TCN, Transformers) without post-hoc shape-matching loops.

3. **Terminal Biochemical Target & Multi-Task Knowledge Supervision**:
   - GE Patent: Predicts derived graphical parameters for clinician notification.
   - Our Model Rerun: Supervises training with terminal fetal outcome (umbilical artery pH $\le 7.15$) and regularizes network parameters via FIGO 2015 clinical rules ($L_{total} = L_{\text{distress}} + \lambda_1 L_{\text{clinical}} + \lambda_2 L_{\text{FIGO}} + \lambda_3 L_{\text{knowledge}}$).

---

## 2. Model Building & Training Protocol

To ensure a fair and scientifically rigorous comparison, all models will be subjected to the following strict constraints:

### Model Building Constraints
- **Universal Input Signature**: Every temporal encoder must accept exactly `(Batch, Channels=2, Sequence_Length=4800)` as input.
- **Universal Output Signature**: Every temporal encoder must output a `(Batch, Hidden_Dim)` latent representation. For this benchmarking phase, `Hidden_Dim` is fixed (e.g., 128) to ensure the classification head receives the exact same capacity of information from every model.
- **Identical Classification Head**: A standardized MLP head will be attached to the output of each encoder purely to evaluate its latent representation on the primary binary task.

### Training Protocol
- **Identical Dataset**: All models will be trained on the exact same Frozen Preprocessed Dataset (v1.0).
- **Identical Splits**: Strict adherence to the predefined Train/Validation/Test patient-level splits.
- **Standardized Loss**: Binary Cross-Entropy (BCE) with Focal Loss to handle the remaining class nuance.
- **Statistical Significance**: Each model will be trained multiple times (e.g., 3 runs) with varying random seeds. Results will be reported as Mean ± Standard Deviation.

### Hyperparameter Tuning Strategy
To ensure architectures are evaluated fairly (and not penalized by poor default parameters), a structured hyperparameter tuning protocol will be enforced:
1. **Global Fixed Parameters**: Epochs (e.g., 50), Early Stopping Patience (e.g., 10), Batch Size (e.g., 32), and Optimizer (AdamW) will remain identical across all models.
2. **Architecture-Specific Tuning**: A predefined grid-search will be conducted for each model's specific hyperparameters:
   - *CNNs/TCN*: Kernel size {3, 5, 7}, Dilation rates, Channel depths.
   - *RNNs (LSTM/GRU)*: Number of layers {1, 2, 3}, Hidden state size {64, 128, 256}, Dropout {0.2, 0.4}.
   - *Transformers (PatchCTG)*: Number of heads {4, 8}, Patch size {10, 20}, Feedforward dimension.
3. **Learning Rate**: A brief learning rate sweep (e.g., $1e^{-3}$ to $1e^{-5}$) using a Cosine Annealing scheduler will be performed for each architecture to find its optimal convergence path.

---

## 3. Evaluation Metrics

Each model will be comprehensively evaluated using both standard machine learning and clinical metrics:

### Classification Metrics
- **Accuracy**
- **Precision (PPV)**
- **Recall (Sensitivity)**
- **Specificity**
- **F1 Score**
- **AUROC** (Area Under the Receiver Operating Characteristic Curve)
- **AUPRC** (Area Under the Precision-Recall Curve)

### Clinical Metrics
- **False Negative Rate** (Critical for fetal safety)
- **False Positive Rate** (Important for reducing unnecessary interventions)
- **Early Fetal Distress Detection Capability**
- **Generalization Robustness** (Consistency across validation and test datasets)

---

## 4. Experimental Objectives

The model comparison aims to answer four primary research questions:

1. **Which temporal architecture best captures CTG dynamics?** Determine which architectural paradigm (convolutions, recurrences, or attention) learns fetal heart rate patterns most effectively.
2. **Which architecture generalizes best?** Evaluate whether performance remains consistent across independent test sets without overfitting.
3. **What is the computational efficiency?** Compare training time, inference time, parameter count, and memory consumption to assess suitability for future real-time clinical deployment.
4. **What is the clinical reliability?** Assess the sensitivity vs. specificity trade-off, specifically monitoring dangerous false negatives and disruptive false alarms.

---

## 5. Expected Outcome

At the conclusion of this benchmarking stage, we will definitively identify the:
1. **Best Performing Encoder**
2. **Most Stable Encoder**
3. **Most Computationally Efficient Encoder**

The winning encoder will officially become the core backbone of the Knowledge-Infused framework.

---

## 6. Next Phase: Integration

Once the optimal encoder is selected, the project shifts exclusively to the Multi-Task integration:

```text
       Best Temporal Encoder
                 │
                 ▼
Knowledge-Infused Multi-Task Framework
                 │
                 ├── Distress Prediction Head (Primary Task)
                 ├── FIGO Knowledge Head (Auxiliary)
                 └── Clinical Feature Head (Auxiliary)
```

**Crucial Note**: No architectural redesign of the encoder will be performed after this stage. Only the finalized multi-task framework will be trained, tuned, and evaluated.

---

## 7. Inferences to Record

During the benchmarking phase, the following metadata must be documented for every run to support the final paper's *Results* and *Discussion* sections:
- Overall performance (Accuracy, F1, AUROC, AUPRC)
- Sensitivity vs. Specificity trade-off curves
- Convergence behavior and training stability
- Training and inference time (seconds/epoch)
- Number of trainable parameters
- Generalization gap (Validation vs. Test performance differential)
- Qualitative clinical observations (e.g., tendency toward false positives or false negatives)

---

## 8. Final Research Workflow

```text
             Preprocessed CTG Dataset (v1.0)
                            │
                            ▼
           Phase 1: Temporal Encoder Benchmarking
 (CNN, BiLSTM, GRU, TCN, MS-LSTM, PatchCTG, PatchTST)
                            │
                            ▼
             Selection of Best Temporal Encoder
                            │
                            ▼
    Integration into the Knowledge-Infused Multi-Task Framework
                            │
                            ▼
             Training & Hyperparameter Optimization
                            │
                            ▼
            Phase 2: Final Framework Comparison
 (Proposed vs Best Baseline, MS-LSTM, PatchCTG, PatchTST)
                            │
                            ▼
                      Ablation Studies
                            │
                            ▼
           Explainability & Clinical Validation
                            │
                            ▼
             Final Research Results & Paper
```
