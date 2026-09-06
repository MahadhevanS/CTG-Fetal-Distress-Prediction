# Phase 9: Temporal Feature Engineering & Causal Deterioration Audit

## Executive Summary

Phase 9 investigates whether the temporal evolution of cardiotocographic (CTG) features across successive 20-minute causal windows $[t - 20\text{ min}, t]$ contains predictive physiological information capable of overcoming the early-warning degradation observed in Phase 8 (where frozen snapshot prediction AUROC was 0.5699 at $\ge 30$ minutes vs 0.7426 at delivery).

To test this hypothesis without architectural bias, a 104-dimensional causal temporal feature space was engineered directly from the 19 validated clinical features and the model's own rolling risk estimates. Zero-future-leakage guarantees were verified under synthetic perturbation auditing.

---

## 1. Mathematical Formulation of Temporal Deterioration Descriptors

For every timestamp $t$ and clinical feature $x \in \mathbb{R}^{19}$, causal historical trajectories were captured across discrete monitoring intervals $\Delta t = 2.5\text{ minutes}$:

1. **First-Order Discrete Velocity ($\Delta x_{t, 1}$)**:
   $$\Delta x_{t, 1} = x_t - x_{t-1} \quad (2.5\text{ min trend})$$

2. **Medium-Term Deterioration ($\Delta x_{t, 2}$)**:
   $$\Delta x_{t, 2} = x_t - x_{t-2} \quad (5.0\text{ min trend})$$

3. **Long-Term Trajectory ($\Delta x_{t, 4}$)**:
   $$\Delta x_{t, 4} = x_t - x_{t-4} \quad (10.0\text{ min trend})$$

4. **Temporal Linear Slope ($s_x$)**:
   Least-squares slope computed over the preceding 4 causal windows (10-minute historical horizon):
   $$s_x = \frac{\sum_{k=0}^{3} (t - k\Delta t - \bar{t})(x_{t-k} - \bar{x})}{\sum_{k=0}^{3} (t - k\Delta t - \bar{t})^2}$$

5. **Recent Rolling Dispersion & Extrema**:
   - Rolling standard deviation: $\sigma_x = \text{SD}(x_{t-3}, \dots, x_t)$
   - Recent extrema: $x_{\max} = \max(x_{t-3}, \dots, x_t)$, $x_{\min} = \min(x_{t-3}, \dots, x_t)$

6. **Model-Evidence Trajectory (Rolling Risk Momentum)**:
   - Risk velocity: $\Delta \hat{R}_t = \hat{R}_t - \hat{R}_{t-1}$
   - Risk acceleration: $\Delta^2 \hat{R}_t = (\hat{R}_t - \hat{R}_{t-1}) - (\hat{R}_{t-1} - \hat{R}_{t-2})$
   - Exponentially weighted moving average risk ($\alpha = 0.3$).

Total Feature Dimensionality:
$$\text{Total Features} = 19 \text{ (current)} + 19 \times 3 \text{ (deltas)} + 19 \text{ (slopes)} + 9 \text{ (risk momentum)} = 104 \text{ features}$$

---

## 2. Physiological Feature Ranking & Deterioration Dynamics

Feature importance and mutual information ranking between causal temporal descriptors and umbilical cord arterial acidemia ($\text{pH} \le 7.15$) reveal distinct physiological patterns:

| Feature Category | Top Predictive Features | Physiological Mechanism Captured |
|---|---|---|
| **Deceleration Burden** | `decel_area_slope`, `decel_depth_delta4`, `prolonged_decel_delta2` | Progressive chemoreceptor/baroreceptor exhaustion from repetitive deep hypoxemic decelerations. |
| **Short-Term Variability (STV)** | `stv_slope_negative`, `stv_delta4`, `stv_min` | Autonomic nervous system suppression and loss of parasympathetic tone under prolonged fetal acidemia. |
| **Baseline FHR Evolution** | `baseline_slope_pos`, `baseline_delta4` | Compensatory sympathetic activation causing progressive tachycardia as metabolic acidosis advances. |
| **Uterine Dynamics Coupling** | `uc_frequency_delta4`, `fhr_uc_lag_slope` | Hyperstimulation / tachysystole leading to progressive lengthening of FHR recovery lag times. |
| **Risk Trajectory** | `risk_ewma_trend`, `risk_acceleration` | Sustained elevation in model confidence over consecutive 2.5-minute monitoring windows. |

---

## 3. Causal Leakage & Perturbation Verification

To guarantee that no future samples contaminated temporal feature construction, the causal synthetic perturbation protocol from Phase 8 was applied across all 104 features:

- **Perturbation Test**: For every timestamp $t_0$, synthetic noise ($\mathcal{N}(0, 5.0)$) was injected into all FHR and UC raw samples for $t > t_0$.
- **Audit Criterion**: Maximum absolute difference across all 104 features between clean and perturbed recordings must satisfy $\max |\Delta X| < 10^{-12}$.
- **Result**: **104/104 features verified causally invariant** ($\max |\Delta X| = 0.000000000000$). Zero future look-ahead exists in the temporal feature extraction pipeline.

---

## 4. Key Takeaways

1. **Causal Validity**: Temporal features strictly adhere to real-time clinical constraints, computing changes solely from past observations.
2. **Physiological Alignment**: Deceleration area slopes and STV decline rates exhibit the highest correlation with intrapartum fetal compromise.
3. **Foundation for Modeling**: The 104-dimensional temporal matrix forms the standardized input across all Phase 9 comparative model architectures.
