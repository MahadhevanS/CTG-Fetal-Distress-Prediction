# Phase 9C: State Trajectory & Transition Dynamics Formulation Report

## Executive Summary

Phase 9C formalizes the representation of intrapartum cardiotocography as a dynamic sequence of discrete physiological deterioration states ($S_t \in \{0, 1, 2, 3, 4\}$) and continuous trajectory operators (progression velocity $V_t$, acceleration $A_t$, state persistence $L_k$, state reversals $R_t$, multidomain concurrence $N_t, C_t$, and cumulative occupancy proportions $P_0 \dots P_4$).

All feature definitions, state assignments, and trajectory derivatives were computed strictly causally ($X_{\le t}$) and verified under synthetic future-perturbation auditing.

---

## 1. Mathematical Formulation of Trajectory Dynamics

For every 20-minute causal window ending at timestamp $t$:

1. **Current Research State ($S_t \in \{0, 1, 2, 3, 4\}$)**:
   - **State 0 (Stable)**: $S_{\text{max}} \le 0.3$, no persistent worsening.
   - **State 1 (Emerging)**: Transient abnormality in single domain ($S_{\text{dom}} > 0.3$ or $\Delta S > 0.1$), $P < 2$.
   - **State 2 (Persistent)**: Single domain persistent abnormality ($P \ge 2$).
   - **State 3 (Progressive)**: Concurrent multidomain deterioration ($N_t \ge 2$, active slope $s > 0.05$ or $C_t \ge 2$).
   - **State 4 (Severe)**: High severity ($S_{\text{dom}} > 1.0$) with established multidomain decompensation ($N_t \ge 2, C_t \ge 2$).

2. **Progression Velocity ($V_t$)**:
   - 1-step instantaneous velocity: $V_t = S_t - S_{t-1}$
   - 4-step multi-window rate: $V_{4, t} = \frac{S_t - S_{t-3}}{3}$

3. **Progression Acceleration ($A_t$)**:
   - $A_t = V_t - V_{t-1} = (S_t - S_{t-1}) - (S_{t-1} - S_{t-2})$

4. **State Persistence ($L_{k, t}$)**:
   - Number of consecutive windows continuously spent in the current state $k$.

5. **State Reversal Indicator ($R_t$)**:
   - $R_t = \mathbb{I}(S_t < S_{t-1})$, capturing active physiological recovery from a higher-severity state.

6. **Multidomain Concurrence ($N_t, C_t$)**:
   - $N_t = \sum_{d=1}^{6} \mathbb{I}(S_{d, t} > 0.3)$ (domains concurrently abnormal).
   - $C_t = \text{consecutive windows with } N_t \ge 2$.

7. **Cumulative State Occupancy ($P_k(t)$)**:
   - Proportion of total observed monitoring time up to timestamp $t$ spent in state $k$: $P_k(t) = \frac{1}{t}\sum_{i=1}^{t} \mathbb{I}(S_i = k)$.

---

## 2. Causal Invariance & Perturbation Audit

- **Audit Protocol**: Synthetic Gaussian noise ($\mathcal{N}(0, 10.0)$) was injected into all CTG samples after timestamp $t_0$.
- **Result**: Maximum absolute difference across all 40 trajectory dimensions between clean and perturbed recordings was $\max |\Delta X| = 0.000000000000$.
- Zero future look-ahead was confirmed across the entire trajectory pipeline.
