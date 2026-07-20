# AI Agent & Developer Guidelines: Phase 3 Model Building

> **🚨 CRITICAL INSTRUCTION FOR ALL AI AGENTS AND DEVELOPERS 🚨**
> This repository is currently in Phase 3 (Model Benchmarking). The data engineering phase is absolutely frozen. Any AI agent or developer assisting in this repository **MUST** adhere to the following strict rules to ensure scientific integrity and prevent unfair comparisons.

## 1. 🛑 STRICT DATASET FREEZE 
- **DO NOT** modify, regenerate, or alter any files in the `data/processed/` directory.
- **DO NOT** modify the `src/preprocessing/pipeline.py` or any scripts related to dataset generation.
- The dataset is strictly `(Batch, 2, 4800)`. If your model throws a shape error, **fix the model architecture**, do not change the dataset.

## 2. 🏗️ UNIVERSAL ENCODER SIGNATURE
All models built during this phase must act purely as Temporal Encoders conforming to a strict input/output signature:
- **Input Shape**: `(Batch, 2, 4800)` (Channel 0: FHR, Channel 1: UC)
- **Output Shape**: `(Batch, 128)` (A 1D flattened latent representation)
- **Rule**: Do not attach a classification head inside the encoder definition. The universal training loop will dynamically attach a standard MLP head to evaluate the `128`-dim latent vector.

## 2.1 🛡️ PATENT DIFFERENTIATION RULE (GE US12094611B2)
To maintain academic integrity and non-infringement alignment with GE patent US12094611B2:
- **Continuous End-to-End Encoding**: Encoders must map the raw sequence $(Batch, 2, 4800)$ directly to the latent space $\mathbb{R}^{128}$.
- **No Longitudinal Graphical Pattern Matching**: Do **NOT** implement visual pattern bounding-box detection or cross-temporal shape-matching confirmation loops.

## 2.2 📊 5-FOLD CV & STATISTICAL SIGNIFICANCE TESTING
- **5-Fold Cross-Validation**: Evaluate all models across 5 Stratified Patient-Level Folds. Log metrics as Mean ± Std across out-of-fold validation splits.
- **Pre- vs. Post-Knowledge Infusion Testing**: In Phase 2, perform statistical hypothesis testing (Paired t-test / Wilcoxon signed-rank test and DeLong test for AUROC) to evaluate whether the Knowledge-Infused Framework yields a statistically significant improvement ($p < 0.05$) over the standalone baseline encoder.

## 3. 📝 LOGGING INFERENCES
- After training a model, you **MUST** update `docs/model_inferences_log.md`.
- Fill out the specific template for your assigned model. 
- Do not modify the markdown structure of the template, only fill in the bracketed placeholder values.

## 4. 👥 TEAM ALLOCATION & GIT WORKFLOW
The benchmarking workload is distributed among 4 team members (2 models per member). You must strictly follow this Git branch mapping:

| Member | Assigned Models | Git Branch Name |
| :--- | :--- | :--- |
| **Member 1** | 1. 1D CNN<br>2. BiLSTM | `models_set_1` |
| **Member 2** | 3. GRU<br>4. TCN | `models_set_2` |
| **Member 3** | 5. Multi-Scale LSTM<br>6. PatchCTG | `models_set_3` |
| **Member 4 (Lead)**| 7. PatchTST<br>8. Knowledge-Infused Framework | `models_set_4` |

### ✅ End-of-Task Git Protocol
Once you have successfully built, trained, and logged the inferences for your assigned models, the AI agent must perform the following:
1. `git checkout -b <assigned_branch_name>` (if not already on it)
2. `git add src/models/ docs/model_inferences_log.md`
3. `git commit -m "feat: implement and benchmark models X and Y"`
4. `git push origin <assigned_branch_name>`

**No agent is allowed to merge directly to `main`.** All branches will be reviewed by the Lead (Member 4) before Model 8 is implemented.

---

## 5. ☁️ UNIVERSAL COLAB EXECUTION WORKFLOW
To ensure a uniform compute environment and fair time-based benchmarking, **ALL models must be trained on Google Colab (T4 GPU)**. Do not train models locally on CPU.

**The standard Colab execution flow:**
1. **Mount Drive**: The preprocessed datasets (`train_dataset.pt`, `val_dataset.pt`, `test_dataset.pt`) are hosted on a shared Google Drive.
2. **Clone & Checkout**: In a Colab cell, clone this repository and checkout your assigned branch:
   ```bash
   !git clone <repository_url>
   %cd CTG
   !git checkout <assigned_branch_name>
   !pip install -r requirements.txt
   ```
3. **Train via CLI**: Execute your model's training script, pointing to the mounted Drive data:
   ```bash
   !python src/models/train_yourmodel.py --data_dir /content/drive/MyDrive/CTG_Project/data/processed/
   ```
4. **Save Artifacts**: Ensure your training script saves the final model weights (`.pth`) directly back to the mounted Google Drive to prevent data loss when the Colab instance disconnects.
