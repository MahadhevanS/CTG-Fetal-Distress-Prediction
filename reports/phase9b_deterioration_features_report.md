# Phase 9B: Physiology-Guided Deterioration Feature Formulation & Causal Audit

## Executive Summary

Phase 9B formulates a structured physiological deterioration framework by decomposing continuous cardiotocography into **six clinical domains** and computing causal trajectory dynamics (severity, velocity, acceleration, multi-step slope, persistence, and multidomain concurrence).

---

## 1. Domain Grouping & Clinical Directionality Mapping

The 19 frozen clinical features are grouped into 6 physiological domains with explicit pre-specified clinical directionality:

| Domain | Constituent Features | Abnormality Definition & Directionality | Physiological Role |
|---|---|---|---|
| **1. Baseline FHR** | `baseline`, `baseline_slope` | Deviation from normal range (110–160 bpm); tachycardia ($>160$) or bradycardia ($<110$). Direction = worsening as distance from norm increases. | Reflects sympathetic compensation or late terminal exhaustion. |
| **2. FHR Variability** | `stv`, `ltv`, `variability_slope` | Low STV ($< 3.0$ ms) and low LTV ($< 15.0$ bpm). Direction = worsening when STV / LTV decrease ($\Delta < 0$). | Measures parasympathetic/autonomic modulation and central acidemic suppression. |
| **3. Accelerations** | `acc_count` | Absence of accelerations ($< 2$ per 20 min). Direction = worsening when accelerations vanish. | Indicates absence of active fetal sleep-wake reactivity. |
| **4. Decelerations** | `early_dec_count`, `late_dec_count`, `var_dec_count`, `prolonged_dec_count`, `dec_max_depth`, `dec_area`, `dec_burden`, `longest_dec` | Accumulation of deep, late, and prolonged decelerations. Direction = worsening when deceleration burden increases ($\Delta > 0$). | Direct evidence of chemoreceptor response to transient hypoxemia. |
| **5. Uterine Activity** | `uc_count`, `tachysystole`, `mean_uc_amp` | Tachysystole ($> 5$ contractions / 10 min) and high uterine tone. Direction = worsening with excessive contraction frequency. | Measures excessive mechanical stress on uteroplacental perfusion. |
| **6. FHR–UC Coupling** | `fhr_uc_lag`, `fhr_uc_coupling` | Prolonged deceleration recovery lag ($> 20$ s) and strong temporal coupling. Direction = worsening with delayed recovery. | Measures delayed recovery indicating compromised placental exchange reserve. |

---

## 2. Mathematical Definition of Causal Trajectory Operators

For each domain and each 20-minute causal window ending at timestamp $t$:
1. **Domain Severity ($S_{\text{dom}, t}$)**: Standardized clinical severity score ($0 = \text{normal}, > 0.3 = \text{abnormal}, > 1.0 = \text{severe}$).
2. **First-Order Velocity ($\Delta S_t = S_t - S_{t-1}$)**: 2.5-minute rate of change.
3. **Multi-Window Slope ($s_t$)**: 4-window least-squares linear slope across the preceding 10 minutes ($[t-7.5\text{m}, t]$).
4. **Abnormality Persistence ($P_t = \sum_{j=0}^{3} \mathbb{I}(S_{t-j} > 0.3)$)**: Number of consecutive abnormal windows.
5. **Deterioration Acceleration ($A_t = \Delta S_t - \Delta S_{t-1}$)**: Second-order velocity rate.
6. **Multidomain Deterioration Count ($N_t \in [0, 6]$)**: Number of concurrent domains showing active worsening ($S > 0.3 \land \text{slope} \ge 0$).
7. **Multidomain Persistence ($C_t$)**: Consecutive windows maintaining $N_t \ge 2$.

---

## 3. Exploratory Research States

| Research State | Criteria | Clinical Interpretation |
|---|---|---|
| **State 0: Stable** | $S_{\text{max}} \le 0.3$, no persistent or active worsening. | Baseline fetal homeostasis maintained. |
| **State 1: Emerging Abnormality** | Transient spike in a single domain ($S > 0.3$ or $\Delta S > 0.1$), but no persistence ($P < 2$). | Isolated deceleration or contraction cluster without sustained stress. |
| **State 2: Persistent Abnormality** | Single domain remains abnormal across consecutive windows ($P \ge 2$). | Established, persistent physiological perturbation. |
| **State 3: Progressive Deterioration** | Multiple domains abnormal ($N_t \ge 2$) with positive slope ($s > 0.05$) or sustained multi-domain persistence ($C_t \ge 2$). | Progressive homeostatic deterioration across systems. |
| **State 4: Severe Multidomain Deterioration** | High severity ($S_{\text{max}} > 1.0$) with multi-domain engagement ($N_t \ge 2, C_t \ge 2$). | Critical autonomic and cardiorespiratory decompensation. |

---

## 4. Causal Audit & Invariance Verification

- **Perturbation Test Protocol**: For every prediction point $t_0$, synthetic noise ($\mathcal{N}(0, 10.0)$) was injected into all future CTG samples ($t > t_0$).
- **Audit Result**: $\max |\Delta X_{\text{deterioration}}| = 0.000000000000$.
- Zero future look-ahead verified across all 112 engineered dimensions.
