# Phase 10 — Final System Definition & Architecture Report

## 1. Executive Summary

This report defines the final system architectures and frozen mathematical formulations evaluated in **Phase 10 — Final Clinical Decision & Contribution Analysis**. The primary objective is to formalize the candidate systems under strict causal constraints (zero future look-ahead) and patient-stratified 5-fold cross-validation.

---

## 2. Candidate System Configurations

| System Identifier | Mathematical Representation | Feature Components | Functional Purpose |
| :--- | :--- | :--- | :--- |
| **Model A** (Snapshot Baseline) | $P(Y \mid R_t)$ | Continuous Huber proxy $R_t$ | Principal benchmark; instantaneous 20-min risk. |
| **Model B** (Snapshot + State) | $P(Y \mid R_t, S_t)$ | Snapshot risk $R_t$ + Research state $S_t$ | Tests if discrete physiological state adds value beyond continuous risk. |
| **Model C** (Snapshot + State + Trajectory) | $P(Y \mid R_t, S_t, V_t, P_t, A_t, R_t, N_t)$ | Snapshot $R_t$ + State $S_t$ + Velocity $V_t$ + Persistence $P_t$ + Acceleration $A_t$ + Reversals $R_t$ + Concurrence $N_t$ | Proposed complete deterioration framework. |
| **Model D** (Snapshot + Trajectory) | $P(Y \mid R_t, V_t, P_t, A_t, R_t, N_t)$ | Snapshot $R_t$ + Dynamics $(V, P, A, R, N)$ | Tests dynamic deterioration features without discrete state index. |
| **Model E** (Full Physiological Fusion) | $P(Y \mid X_t, \text{Traj}_t, R_t)$ | 19 Descriptors + 6 Domain Severities + 7 Trajectory Dynamics + 5 Occupancies + 2 Predictions | Full multi-dimensional physiological context. |

---

## 3. Causal Feature Extraction Pipeline

Every 20-minute rolling window $W_t = [t - 20\text{ min}, t]$ sampled at 4 Hz ($4,800$ FHR and UC points) is processed through a causally locked pipeline:

```text
                     RAW 4 Hz CTG (FHR + UC)
                                │
                                ▼
                      Causal 20-min Window
                                │
                                ▼
                     19 Clinical Descriptors
         (Baseline, STV, LTV, Decelerations, UC, Lag)
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
Continuous Clinical Huber                   6 Physiological Domains
        Model                                (Baseline, STV, Accel,
          │                                     Decel, UC, Interaction)
          ▼                                           │
  Snapshot Risk R(t)                                  ▼
          │                                 Physiological State S(t)
          │                                           │
          │                                           ▼
          │                                 Trajectory Dynamics
          │                              (Velocity V, Persistence P,
          │                               Acceleration A, Reversal R)
          │                                           │
          └─────────────────────┬─────────────────────┘
                                ▼
                 Final Integrated Decision Layer
                                │
                                ▼
                   Clinical Alert Engine (Policy 5)
```

---

## 4. Cross-Validation & Normalization Protocol

* **Cross-Validation**: Patient-stratified 5-fold cross-validation locked to `data/processed_clinical/folds.json`.
* **Fold Independence**: Feature standard scalers and classifier parameters are fit strictly on training folds and applied out-of-fold. Zero test-fold information enters model training.
* **Causal Audit**: Synthetic future perturbation testing confirms $\max |\Delta X| = 0.000000000000$ across all 8,517 evaluation windows.

# End of Final System Report
