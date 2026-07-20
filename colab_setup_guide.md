# Team Execution & Colab Setup Guide 🚀

This document provides step-by-step instructions for our **4-member team** to implement, train, and evaluate our assigned temporal models on **Google Colab (T4 GPU)**.

Each team member is assigned **2 models** and a dedicated **Git branch**. Follow the instructions below to set up your environment and use the pre-formatted **AI Agent Prompts** to build and benchmark your models seamlessly.

---

## 1. 👥 Team Workload & Branch Allocation

| Member | Assigned Models | Git Branch | Key Task |
| :--- | :--- | :--- | :--- |
| **Member 1** | 1. 1D CNN<br>2. BiLSTM | `models_set_1` | Implement & benchmark local convolution & bidirectional recurrence baselines. |
| **Member 2** | 3. GRU<br>4. TCN | `models_set_2` | Implement & benchmark gated recurrence & causal dilated convolution baselines. |
| **Member 3** | 5. Multi-Scale LSTM<br>6. PatchCTG | `models_set_3` | Implement & benchmark multi-resolution LSTM & CTG patch-transformer baselines. |
| **Member 4**| 7. PatchTST<br>8. Knowledge-Infused Framework | `models_set_4` | Implement PatchTST & combine winning encoder into the multi-task framework. |

---

## 2. 🚨 Universal Constraints (Must Follow)

Every AI agent and developer working in this repository **must** strictly adhere to these rules:

1. **Frozen Dataset**: Do **NOT** modify files in `data/processed/` or `src/preprocessing/`.
2. **Universal Input Signature**: `(Batch, Channels=2, Sequence_Length=4800)` where Channel 0 = FHR and Channel 1 = UC.
3. **Universal Latent Output**: Every encoder outputs `(Batch, Hidden_Dim=128)`. Classification heads are attached outside the encoder.
4. **Patent Differentiation (US12094611B2)**: Continuous end-to-end signal representation directly to latent space $\mathbb{R}^{128}$ without longitudinal graphical pattern bounding boxes or shape-matching correlation loops.
5. **Metric Logging**: Always record results (Mean ± Std over 3 seeds) and patent compliance notes in `docs/model_inferences_log.md`.

---

## 3. ☁️ Google Colab Execution Setup (5-Minute Walkthrough)

### Step 1: Open Colab & Enable GPU
1. Navigate to [colab.research.google.com](https://colab.research.google.com/) $\rightarrow$ **New Notebook**.
2. Click **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Select **T4 GPU**.

### Step 2: Mount Shared Google Drive
Paste and run in Colab Cell 1:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Clone Repo & Checkout Your Branch
Paste and run in Colab Cell 2:
```bash
!git clone https://github.com/MahadhevanS/CTG-Fetal-Distress-Prediction.git
%cd CTG-Fetal-Distress-Prediction
!git checkout <YOUR_BRANCH_NAME>   # e.g., models_set_1
!pip install -r requirements.txt
```

### Step 4: Run Training Script via CLI
Paste and run in Colab Cell 3:
```bash
!python src/training/train.py \
  --config configs/colab.yaml \
  --model <YOUR_MODEL_NAME> \
  --data_dir /content/drive/MyDrive/CTG_Project/data/processed/ \
  --save_dir /content/drive/MyDrive/CTG_Project/checkpoints/
```

---

## 4. 🤖 Ready-to-Use AI Agent Prompts

Give the appropriate prompt below directly to your AI coding assistant (Antigravity / Cursor / Copilot) to implement your assigned models.

---

### 🟢 Member 1 Prompt (`models_set_1`)
> **Task**: Implement Model 1 (1D CNN) and Model 2 (BiLSTM).  
> **Copy & Paste this to your AI Agent**:
> ```text
> I am Member 1 working on branch `models_set_1`. 
> Please implement two PyTorch temporal encoders in `src/models/`:
> 1. `CNN1DEncoder`: 1D CNN with residual blocks accepting input (Batch, 2, 4800) and returning (Batch, 128).
> 2. `BiLSTMEncoder`: Bidirectional LSTM network accepting input (Batch, 2, 4800) and returning (Batch, 128).
> 
> Strict Constraints:
> - Do not modify `data/processed/`.
> - Conform strictly to input shape (Batch, 2, 4800) and output shape (Batch, 128).
> - Maintain continuous signal encoding to comply with GE Patent US12094611B2 non-infringement.
> - Ensure models can be instantiated and selected in `src/training/train.py`.
> - Update `docs/model_inferences_log.md` sections 1 and 2 after training.
> ```

---

### 🔵 Member 2 Prompt (`models_set_2`)
> **Task**: Implement Model 3 (GRU) and Model 4 (TCN).  
> **Copy & Paste this to your AI Agent**:
> ```text
> I am Member 2 working on branch `models_set_2`. 
> Please implement two PyTorch temporal encoders in `src/models/`:
> 1. `GRUEncoder`: Multi-layer Gated Recurrent Unit network accepting input (Batch, 2, 4800) and returning (Batch, 128).
> 2. `TCNEncoder`: Temporal Convolutional Network with causal dilated convolutions accepting input (Batch, 2, 4800) and returning (Batch, 128).
> 
> Strict Constraints:
> - Do not modify `data/processed/`.
> - Conform strictly to input shape (Batch, 2, 4800) and output shape (Batch, 128).
> - Maintain continuous signal encoding to comply with GE Patent US12094611B2 non-infringement.
> - Ensure models can be instantiated and selected in `src/training/train.py`.
> - Update `docs/model_inferences_log.md` sections 3 and 4 after training.
> ```

---

### 🟣 Member 3 Prompt (`models_set_3`)
> **Task**: Implement Model 5 (Multi-Scale LSTM) and Model 6 (PatchCTG).  
> **Copy & Paste this to your AI Agent**:
> ```text
> I am Member 3 working on branch `models_set_3`. 
> Please implement two PyTorch temporal encoders in `src/models/`:
> 1. `MultiScaleLSTMEncoder`: Multi-scale temporal pooling + parallel LSTM paths accepting input (Batch, 2, 4800) and returning (Batch, 128).
> 2. `PatchCTGEncoder`: Patchified sequence Transformer encoder accepting input (Batch, 2, 4800) and returning (Batch, 128).
> 
> Strict Constraints:
> - Do not modify `data/processed/`.
> - Conform strictly to input shape (Batch, 2, 4800) and output shape (Batch, 128).
> - Maintain continuous signal encoding to comply with GE Patent US12094611B2 non-infringement.
> - Ensure models can be instantiated and selected in `src/training/train.py`.
> - Update `docs/model_inferences_log.md` sections 5 and 6 after training.
> ```

---

### 🔴 Member 4 Prompt (`models_set_4`)
> **Task**: Implement Model 7 (PatchTST) and Model 8 (Knowledge-Infused Multi-Task Framework).  
> **Copy & Paste this to your AI Agent**:
> ```text
> I am Member 4 working on branch `models_set_4`. 
> Please implement:
> 1. `PatchTSTEncoder`: Channel-independent patch time-series transformer accepting input (Batch, 2, 4800) and returning (Batch, 128).
> 2. `KnowledgeInfusedMultiTaskFramework`: Proposed framework wrapping the winning Phase 1 encoder with:
>    - Distress Prediction Head (Binary classification for pH <= 7.15)
>    - Clinical Feature Head (Regression for baseline, STV, LTV, decelerations)
>    - FIGO Knowledge Head (3-class classification + FIGO rule loss from `src/knowledge/figo.py`)
> 
> Strict Constraints:
> - Do not modify `data/processed/`.
> - Ensure Patent US12094611B2 non-infringement differentiation is logged in `docs/model_inferences_log.md` sections 7 and 8.
> ```

---

## 5. 📤 End-of-Task Git Protocol (Pushing Your Work)

Once your models have been trained and logged in `docs/model_inferences_log.md`:

```bash
git checkout <YOUR_BRANCH_NAME>
git add src/models/ docs/model_inferences_log.md
git commit -m "feat: implement and benchmark assigned temporal models"
git push origin <YOUR_BRANCH_NAME>
```

> ⚠️ **Important**: Do **not** merge directly to `main`. 
