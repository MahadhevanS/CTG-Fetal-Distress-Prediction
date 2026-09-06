# Phase 11 — Prior-Art Methodological Review & Benchmark Formulation

## 1. Executive Summary

This report establishes the contextual literature foundation and exact implementation parameters for the comparative prior-art baselines evaluated in **Phase 11 — Prior-Art Head-to-Head Benchmark & Superiority Validation**.

---

## 2. Why Same-Cohort Benchmarking Is Essential

In literature comparisons, published CTG prediction performance figures vary wildly due to differences in:
1. **Clinical Endpoints**: Umbilical artery pH $\le 7.05$, $\le 7.15$, $\le 7.20$, Base Deficit $\ge 12\text{ mmol/L}$, or Apgar $\le 7$.
2. **Preprocessing & Artifact Handling**: Interpolation strategies, filtering bandwidths, and recording selection windows.
3. **Causal Horizon Anchoring**: Evaluating the entire labor trace retrospectively vs causal rolling windows vs delivery snapshots.
4. **Statistical Grouping**: Patient-level clustering vs random window-level splits.

To make an honest, publication-defensible claim of performance differences, **all methods must be evaluated under identical causal data boundaries, patient folds, endpoints, and statistical bootstrap protocols on the CTU-UHB cohort ($N=547$)**.

---

## 3. Evaluated Prior-Art Baseline Formulations

### 3.1 P1: DeepCTG-Inspired Compact CTG Baseline
* **Conceptual Source**: Published engineered morphological approaches (e.g., DeepCTG and related FIGO descriptor pipelines) demonstrating strong acidemia prediction using compact feature representations.
* **Feature Representation (4 Features)**:
  1. Minimum FHR Baseline (bpm)
  2. Maximum FHR Baseline (bpm)
  3. Acceleration Total Area (bpm $\cdot$ s)
  4. Deceleration Total Area (bpm $\cdot$ s)
* **Classifier**: L2-regularized logistic regression ($C=0.1$) trained out-of-fold under patient-stratified 5-fold CV.
* **Terminology Designation**: **"DeepCTG-inspired compact CTG baseline"** (methodological reproduction under common evaluation conditions).

### 3.2 P2: Vargas-Calixto-Inspired Sequential / Event Baseline
* **Conceptual Source**: Sequential event monitoring and temporal persistence frameworks (e.g., Vargas-Calixto et al., 2021) combining epoch-based deceleration descriptors with temporal persistence filtering.
* **Feature Representation (7 Features)**:
  1. Baseline FHR
  2. STV (ms)
  3. LTV (bpm)
  4. Variable Deceleration Frequency
  5. Deceleration Max Depth (bpm)
  6. Deceleration Total Burden (%)
  7. Contraction Frequency (UC count / 20 min)
* **Sequential Classifier & Persistence Operator**:
  - Epoch classifier generating continuous posterior probabilities per 20-minute window.
  - Temporal persistence filter ($\text{EWMA}_{\alpha=0.5}$) accumulating qualifying stress windows across time.
* **Terminology Designation**: **"Vargas-Calixto-inspired sequential / event baseline"**.

### 3.3 P3: Locked Snapshot Baseline (Internal Benchmark)
* Continuous Huber regression on 19 FIGO descriptors operating causally on 20-minute rolling windows (locked delivery $\text{AUROC} = 0.7426$).

### 3.4 P4: Snapshot + Physiological State
* Continuous Huber risk combined with discrete research state index ($R_t + S_t$).

### 3.5 P5: Snapshot + Deterioration Trajectory
* Continuous Huber risk combined with temporal trajectory dynamics ($R_t + V_t + P_t + A_t + R_t + N_t$).

### 3.6 P6: Full Proposed Physiology-Guided System
* Complete multi-domain physiological context, trajectory velocity, acceleration, persistence, and reversals.

# End of Prior-Art Review
