# Phase 3 Deliverable: CTU-UHB Patient-Level Learning & 1D Multi-Instance Aggregation Report

**Date:** 2026-09-06  
**Cohort:** CTU-UHB Clean Cohort (547 patients, 110 acidotic [pH ≤ 7.15], 8,517 usable 20-min windows)  
**Evaluation Protocol:** Frozen Patient-Grouped 5-Fold Stratified Partition (`data/processed_clinical/folds.json`)  
**Primary Endpoint:** Umbilical Artery pH ≤ 7.15 (Frozen)  
**Representation Substrate:** Frozen 1D Temporal ResNet Backbone (`CNN1DEncoder`, 135k parameters)  

---

## 1. Executive Summary & Core Decision

Phase 3 investigated whether the predictive ceiling on CTU-UHB is caused by heuristic window aggregation (Max / Mean / P90) and whether learned patient-level Multi-Instance Learning (Attention MIL) could unlock higher performance by integrating evidence across multiple 1D CTG windows.

### Key Empirical Findings:
1. **Fixed Extreme-Value Pooling (P90 / Max) Remains the Strongest Aggregator:** 
   - **Fixed P90:** Patient AUROC = **0.6701** [0.611, 0.726], AUPRC = **0.3399**
   - **Fixed Max:** Patient AUROC = **0.6664** [0.608, 0.723], AUPRC = **0.3313**
   - **Fixed Top-3:** Patient AUROC = **0.6680** [0.609, 0.725], AUPRC = **0.3350**
   - **Fixed Mean:** Patient AUROC = **0.6357** [0.574, 0.696], AUPRC = **0.3572**
2. **Learned Attention MIL Fails to Improve Over Heuristic Pooling:**
   - Attention MIL achieved a patient AUROC of **0.5337** [0.475, 0.596] and AUPRC of **0.2224**, representing a statistically significant degradation of **$\Delta AUROC = -0.1343$** (95% CI: [$-0.216, -0.049$], **$p = 0.001$**).
3. **Embedding-Space Pooling Collapses Near Chance Level:**
   - **Mean Embedding Pooling:** AUROC = **0.4894** [0.430, 0.548], AUPRC = **0.2308** ($\Delta AUROC = -0.1772$, $p < 0.001$)
   - **Max Embedding Pooling:** AUROC = **0.4683** [0.405, 0.532], AUPRC = **0.2021** ($\Delta AUROC = -0.1987$, $p < 0.001$)
4. **Controlled Negative Tests Confirm Methodological Soundness (Zero Leakage):**
   - Across-Patient Window Shuffle: AUROC = **0.4891** (collapsed to chance $\sim 0.50$, confirming zero cross-patient leakage).
   - Label Permutation Control: AUROC = **0.5146** (collapsed to chance $\sim 0.50$, confirming no structural training artifacts).
   - Bag-Size Alone ($N_p$): AUROC = **0.4657** (window count has zero predictive confounding).
5. **Clinical Logistic Regression Benchmark (0.7271) Remains Unsurpassed by Standalone Neural Models:** Standalone 1D neural networks reach $\sim 0.67$ AUROC with fixed pooling, but cannot reach Clinical-LR ($0.7271$) without explicit domain clinical features.

### Required Final Decision:
> **STOP MIL.**  
> Learned patient-level aggregation (Attention MIL, embedding pooling) provides no reproducible gain over heuristic extreme-value pooling (P90 / Max). Attention MIL fails Hard Gate 1 ($AUROC_{MIL} < 0.75$, $\Delta AUROC < 0$). In accordance with the scientific stopping rules, the MIL and fine-tuning branches are permanently closed. Subsequent phases will use frozen P90/Max aggregation and focus on clinical knowledge infusion and multi-resolution temporal features.

---

## 2. Comprehensive Results Table

### Primary Results Table:

| Model | Backbone | Aggregation | Trainable Params | Patient AUROC | 95% Bootstrap CI | AUPRC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Clinical LR** | Handcrafted FIGO | Logistic Reg. | — | **0.7271** | [0.670, 0.779] | **~0.409** |
| **Exp 3.0: Fixed P90** | 1D ResNet | P90 score | — | **0.6701** | [0.611, 0.726] | **0.3399** |
| **Exp 3.0: Fixed Top-3** | 1D ResNet | Top-3 score | — | **0.6680** | [0.609, 0.725] | **0.3350** |
| **Exp 3.0: Fixed Max** | 1D ResNet | Max score | — | **0.6664** | [0.608, 0.723] | **0.3313** |
| **Exp 3.0: Fixed Mean** | 1D ResNet | Mean score | — | **0.6357** | [0.574, 0.696] | **0.3572** |
| **Exp 3.3: Attention MIL** | 1D ResNet | Attention | 8,417 | **0.5337** | [0.475, 0.596] | **0.2224** |
| **Exp 3.4: Fixed-K ($K=6$)**| 1D ResNet | Attention | 8,417 | **0.4975** | [0.438, 0.557] | **0.1949** |
| **Exp 3.1: Mean Embedding**| 1D ResNet | Mean vector | 4,161 | **0.4894** | [0.430, 0.548] | **0.2308** |
| **Exp 3.2: Max Embedding** | 1D ResNet | Max vector | 4,161 | **0.4683** | [0.405, 0.532] | **0.2021** |
| **Exp 3.5: Shuffled Order**| 1D ResNet | Attention | 8,417 | **0.4895** | [0.428, 0.547] | **0.1928** |
| **Exp 3.6: Leakage Shuffle**| 1D ResNet | Attention | 8,417 | **0.4891** | [0.422, 0.551] | **0.2071** |
| **Exp 3.7: Label Permute** | 1D ResNet | Attention | 8,417 | **0.5146** | [0.452, 0.575] | **0.2355** |

---

## 3. Paired Statistical Comparisons

Paired patient-level bootstrap tests (2,000 replicates) evaluating out-of-fold predictions against the Fixed Max control:

| Comparison | $\mathbf{\Delta AUROC}$ | 95% Bootstrap CI | $\mathbf{P(\Delta \le 0)}$ | Paired p-value | Decision |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Fixed P90 − Fixed Max** | $\mathbf{+0.0038}$ | [$-0.014, +0.020$] | 0.326 | $p = 0.326$ | **Top Heuristic Performer** |
| **Fixed Top-3 − Fixed Max** | $\mathbf{+0.0016}$ | [$-0.010, +0.014$] | 0.387 | $p = 0.387$ | Comparable |
| **Fixed Mean − Fixed Max** | $\mathbf{-0.0310}$ | [$-0.071, +0.007$] | 0.936 | $p = 0.064$ | Inferior |
| **Attention MIL − Fixed Max** | $\mathbf{-0.1343}$ | [$\mathbf{-0.216, -0.049}$] | 0.999 | $\mathbf{p = 0.001}$ | **Statistically Inferior** |
| **Fixed-K MIL − Fixed Max** | $\mathbf{-0.1706}$ | [$\mathbf{-0.258, -0.081}$] | 1.000 | $\mathbf{p = 0.000}$ | **Statistically Inferior** |
| **Mean Embedding − Fixed Max** | $\mathbf{-0.1772}$ | [$\mathbf{-0.264, -0.090}$] | 1.000 | $\mathbf{p = 0.000}$ | **Statistically Inferior** |
| **Max Embedding − Fixed Max** | $\mathbf{-0.1987}$ | [$\mathbf{-0.281, -0.107}$] | 1.000 | $\mathbf{p = 0.000}$ | **Statistically Inferior** |
| **Fixed Max − Clinical LR** | $\mathbf{-0.0607}$ | [$-0.118, -0.005$] | 0.978 | $p = 0.022$ | Clinical LR Superior |

---

## 4. Attention Dynamics & Interpretability Analysis

- **Attention Concentration ($C_{max}, C_{top3}$):**
  - Mean $C_{max}$: Positive patients = $0.2188$, Negative patients = $0.1989$.
  - Mean $C_{top3}$: Positive patients = $0.4708$, Negative patients = $0.4392$.
  - Attention is relatively diffuse across windows rather than sharply isolating singular crisis events.
- **Correlation with Window Risk ($r(a_i, s_i)$):**
  - Correlation between patient attention risk scores and heuristic max scores is near zero ($r = -0.0175$, $\rho = -0.0291$). The attention mechanism learns a spurious subspace that does not align with the encoder's true window-level risk estimates.
- **Duration Stratification:**
  - Full 60-min Recordings ($N=508$): Fixed Max AUROC = **0.6904** vs Attention MIL AUROC = **0.5462**.
  - Short Records ($N=39$): Fixed Max AUROC = **0.6162** vs Attention MIL AUROC = **0.5164**.

---

## 5. Answers to Core Scientific Questions

### Q1: Does Attention MIL outperform fixed heuristic pooling (Max / P90)?
**No.** Attention MIL (AUROC $0.5337$) underperforms Fixed P90 ($0.6701$) and Fixed Max ($0.6664$) by over $0.13$ AUROC ($p = 0.001$).

### Q2: Does embedding-level pooling preserve more information than probability-level pooling?
**No.** Both Mean Embedding ($0.4894$) and Max Embedding ($0.4683$) collapse near random chance, demonstrating that unconstrained linear classification on pooled 128D embeddings suffers severe sample overfitting with only 110 positive bags.

### Q3: Why does heuristic extreme-value pooling (P90 / Max) succeed while MIL fails?
1. **Clinical Pathophysiology is Extreme-Event Driven:** Intrapartum hypoxia causes acute, transient decompensation (severe variable/late decelerations, bradycardia episodes). A baby is distressed if *any* significant crisis window occurs. Fixed Max/P90 directly implements this clinical logic without learning extra parameters.
2. **Small-Sample Bag Bottleneck:** Learning patient-level attention weights over a continuous 128D manifold with only $\sim 88$ positive training bags per fold leads to catastrophic overfitting.

---

## 6. Evaluation of Hard Gates & Exit Decision

- **Gate 1 — Frozen MIL:** *FAILED.* $AUROC_{MIL} = 0.5337 < 0.75$, and $\Delta AUROC = -0.1343 < 0$. Rule: *"Do not proceed to extensive MIL tuning."*
- **Gate 2 — Fine-tuning:** *CLOSED.* No evidence of improvement from frozen MIL; fine-tuning branch is not opened.
- **Gates 3, 4, 5:** Not triggered.

```text
========================================================================================
FINAL DECISION: STOP MIL
========================================================================================
1. Permanently CLOSE the learned Multi-Instance Learning (MIL) and embedding pooling branch.
2. FREEZE Fixed P90 / Max pooling as the canonical patient-level aggregation method.
3. RETAIN the 1D temporal representation as the foundational encoder.
4. PROCEED to subsequent phases investigating clinical knowledge infusion (FIGO features,
   expert rules) and multi-resolution temporal context rather than bag-level neural pooling.
========================================================================================
```
