# Phase 9C: Physiological Deterioration Timing & Lead-Time Analysis Report

## Executive Summary

Phase 9C evaluated the temporal timing of first entry into deterioration states ($W_k = T_{\text{delivery}} - T_{\text{entry}, k}$) and investigated matched-current-state risk stratification to determine whether progression history adds predictive value when current state is held constant.

---

## 1. First State-Entry Timing Before Delivery (Exp 3)

For each deterioration state $k \in \{1, 2, 3, 4\}$, the timing of the first recorded transition was tracked relative to delivery across all acidemic fetuses:

| Research State | Acidemic Fetuses Entering State ($N=110$) | Entry Rate (%) | Median Lead Time | IQR Lead Time | Mean Lead Time | Detected $\ge 30\text{ min}$ | Detected $\ge 20\text{ min}$ | Detected $\ge 10\text{ min}$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **State 1: Emerging Abnormality** | 96 / 110 | **87.27%** | **27.5 min** | 22.5 min | 29.8 min | **46.36%** | 60.91% | 80.00% |
| **State 2: Persistent Abnormality** | 75 / 110 | **68.18%** | **17.5 min** | 15.0 min | 21.2 min | **24.55%** | 39.09% | 58.18% |
| **State 3: Progressive Deterioration** | 49 / 110 | **44.55%** | **12.5 min** | 12.5 min | 15.4 min | **10.91%** | 19.09% | 34.55% |
| **State 4: Severe Multidomain** | 20 / 110 | **18.18%** | **7.5 min** | 7.5 min | 9.6 min | **1.82%** | 4.55% | 14.55% |

### Key Timing Takeaways:
1. **Emerging Perturbations Occur Early**: Fetuses entering State 1 (Emerging) do so with a median lead time of **$27.5\text{ minutes}$**, with **$46.4\%$** detected $\ge 30\text{ minutes}$ before delivery.
2. **Progression and Critical Decompensation are Concentrated Near Delivery**:
   - Entry into State 3 (Progressive) occurs at a median of **$12.5\text{ minutes}$** before delivery.
   - Entry into State 4 (Severe Multidomain) occurs at a median of **$7.5\text{ minutes}$** before delivery.
   - This provides direct physiological confirmation of why CTG discrimination is heavily concentrated within the final 20–30 minutes of labor.

---

## 2. Matched-Current-State Risk Analysis (Exp 9 — Testing Hypothesis H1)

To test Hypothesis H1 (*"Trajectory contains information beyond current state"*), windows sharing the **exact same current research state** were stratified by preceding 4-window trajectory velocity ($V_4$):

| Current State Stratum | Trajectory Dynamic | Total Windows ($N$) | Unique Patients ($N$) | Observed Acidemia ($\text{pH} \le 7.15$) Prevalence (%) | Severe Acidemia ($\text{pH} \le 7.05$) Prevalence (%) |
|---|---|:---:|:---:|:---:|:---:|
| **State 1 (Emerging)** | **Progressing ($V_4 > 0$)** | 712 | 168 | **18.26%** | **4.21%** |
| | **Stable ($V_4 \approx 0$)** | 1,145 | 224 | **12.49%** | **2.62%** |
| | **Reversing ($V_4 < 0$)** | 489 | 122 | **9.61%** | **1.84%** |
| **State 2 (Persistent)** | **Progressing ($V_4 > 0$)** | 438 | 118 | **27.63%** | **10.50%** |
| | **Stable ($V_4 \approx 0$)** | 682 | 154 | **20.09%** | **7.48%** |
| | **Reversing ($V_4 < 0$)** | 291 | 82 | **14.09%** | **4.12%** |
| **State 3 (Progressive)** | **Progressing ($V_4 > 0$)** | 314 | 76 | **42.36%** | **19.75%** |
| | **Stable ($V_4 \approx 0$)** | 398 | 92 | **31.16%** | **13.57%** |
| | **Reversing ($V_4 < 0$)** | 142 | 41 | **21.83%** | **8.45%** |

### Confirmation of Hypothesis H1:
- Within every single research state stratum (State 1, State 2, and State 3), **progressing trajectories ($V_4 > 0$) have nearly double the acidemia risk of reversing trajectories ($V_4 < 0$)**:
  - In State 2: $27.63\%$ (progressing) vs $14.09\%$ (reversing), Odds Ratio $= 2.33$ ($p < 0.001$).
  - In State 3: $42.36\%$ (progressing) vs $21.83\%$ (reversing), Odds Ratio $= 2.62$ ($p < 0.001$).
- **Scientific Impact**: This provides rigorous statistical confirmation that fetal risk is determined not merely by the instantaneous CTG state, but by the **direction and velocity of state transition**.
