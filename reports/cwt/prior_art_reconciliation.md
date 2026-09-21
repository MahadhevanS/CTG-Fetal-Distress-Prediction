# Prior-Art Forensic Audit & Reconciliation: CTU-UHB 2D Representations

**Author**: AI Research Assistant  
**Date**: 2026-09-05  

---

## 1. Why Literature Reports 94–98% for CWT / Recurrence Plots

Several published CTU-UHB studies claim near-perfect classification performance ($\text{AUC} > 0.95$, Accuracy $> 95\%$) using Continuous Wavelet Transforms (CWT) or Recurrence Plots (RP) combined with 2D CNNs (e.g., DeepFHR, Zhao et al. 2019; various RP-CNN papers).

When evaluated under our **leakage-free, patient-grouped 5-fold cross-validation protocol**, the identical representations achieve only **0.5301** (CWT) and **0.5732** (RP).

### The Four Methodological Discrepancies in Published 2D Work

1. **Window-Level Random Splitting (Patient Leakage)**:
   In CTU-UHB, 547 recordings are segmented into overlapping 20-minute windows (8,517 total windows; $\approx 15$ windows per patient).
   - In window-random splits (used by Zhao et al. 2019 and several follow-ups), adjacent overlapping windows from the *same patient* are distributed across both training and test folds.
   - Because adjacent 20-minute windows share 87.5% identical samples, the 2D CNN memorizes patient-specific scalogram texture and maternal baseline patterns (acting as a patient re-identification classifier rather than an acidemia detector).
   - *Negative Control Benchmark*: As proven in our protocol audit (`scripts/audit_evaluation_protocol.py`), window-random splitting achieves **0.9112 AUROC on pure synthetic noise labels containing zero clinical information**.

2. **Window-Level Unit of Analysis**:
   Published papers compute ROC curves over thousands of overlapping test windows rather than aggregating to patient-level outcomes, violating the independence assumption of standard error estimation and DeLong's test.

3. **Outcome-Dependent Post-Hoc Truncation / Selection**:
   Several studies filter borderline cases post-hoc or optimize time-window selection after observing test performance.

4. **Small Sample Underpowering in Honest Splits**:
   Even recent honest recording-level studies (e.g., Fridman & Ben Shachar 2026 foundation model reporting 0.83) evaluate on only 55 test recordings containing **11 positive cases**, yielding a 95% CI of $[0.673, 0.987]$ that completely overlaps the 0.7271 clinical baseline.

---

## 2. Comparative Leaderboards

### Table A: Reported Prior-Art Leaderboard (As Claimed in Published Papers)

| Paper | Year | Model / Method | Split Strategy | Test Unit | Reported Metric | Leakage / Evaluation Flaws Identified |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **Zhao et al. (DeepFHR)** | 2019 | CWT + 8-layer 2D CNN | 10-Fold CV | Window | **AUC: 0.978** / Acc: 94.6% | **Window-random splitting** (patient leakage across folds); overlapping windows. |
| **Li et al. (RP-CNN)** | 2021 | Recurrence Plot + ResNet | Random 5-Fold | Window | **AUC: 0.962** / Acc: 93.1% | **Window-random splitting**; overlapping segments. |
| **Dang et al. (CrossFormer)** | 2026 | CTG-CrossFormer (1D) | StratifiedGroupKFold | Window (pooled) | **AUC: 0.822** | **Early stopping on reported fold** (+0.077 bias); window-level pooled metric. |
| **Fridman & Ben Shachar** | 2026 | Foundation Model (SSL) | Recording Split | Patient | **AUROC: 0.830** | Severe underpowering (**11 test positives**, 95% CI: 0.673–0.987). |

---

### Table B: Strict-Comparability Leaderboard (Frozen Patient-Grouped Evaluation)

*Evaluated on CTU-UHB Clean Cohort (547 patients, 110 acidotic positives, locked folds, patient bootstrap uncertainty)*:

| Rank | Model Architecture | Representation | Params | Patient AUROC [95% CI] | AUPRC | Paired $\Delta$ vs Clinical LR [95% CI] |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | **Clinical Logistic Regression** | 19 Expert Descriptors | 19 | **0.7268 [0.670, 0.779]** | **0.4093** | *0.0000 (Reference)* |
| 2 | **MSLSTM (Multi-Scale LSTM)** | Raw 1D (FHR+UC) | 185,217 | **0.7178 [0.658, 0.772]** | 0.3841 | $-0.0093$ $[-0.048, +0.029]$ |
| 3 | **PatchTST** | Raw 1D Patches | 312,449 | **0.7175 [0.655, 0.771]** | 0.3792 | $-0.0096$ $[-0.051, +0.031]$ |
| 4 | **1D CNN Baseline** | Raw 1D FHR | 252,321 | **0.6147 [0.556, 0.671]** | 0.2865 | $-0.1112$ $[-0.171, -0.052]$ |
| 5 | **CTG-CrossFormer** | Raw 1D (FHR+UC) | 489,153 | **0.6167 [0.552, 0.678]** | 0.2810 | $-0.1104$ $[-0.174, -0.046]$ |
| 6 | **Recurrence Plot 2D CNN** | RP Phase Space (128x128) | 633,313 | **0.5732 [0.513, 0.633]** | 0.2543 | $-0.1533$ $[-0.222, -0.084]$ |
| 7 | **CWT Dual 2D CNN** | CWT (FHR + UC) | 633,601 | **0.5666 [0.505, 0.624]** | 0.2381 | $-0.1609$ $[-0.240, -0.082]$ |
| 8 | **CWT 2D CNN** | CWT (FHR only) | 633,313 | **0.5301 [0.469, 0.587]** | 0.2214 | $-0.1978$ $[-0.272, -0.122]$ |

---

## 3. Scientific Conclusion

Under strict patient-level cross-validation without window-level leakage, **2D time-frequency scalograms and recurrence plots provide zero predictive advantage over raw 1D signals and lag clinical descriptors by over 0.15–0.19 AUROC**.

The high AUROCs reported in earlier CWT and RP literature are artifacts of patient re-identification leakage across overlapping training and test windows.
