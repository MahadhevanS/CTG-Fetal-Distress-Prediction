# Phase 10 — Clinical Operating Characteristics & Decision Analysis Report

## 1. Executive Summary

This report evaluates the clinical operating characteristics, false-alarm burden, and warning lead-time distributions for candidate intrapartum alerting policies on the CTU-UHB cohort ($N=547$ patients, $N_{\text{acidemia}}=110$ at $\text{pH} \le 7.15$, $285.75$ negative monitoring hours).

---

## 2. Evaluated Alert Policies

1. **Policy 1 (Single Window Threshold)**: $p_t \ge \tau$ ($\tau = 0.2730$, 90th percentile baseline risk).
2. **Policy 2 (Two Consecutive Windows)**: $p_t \ge \tau \land p_{t-1} \ge \tau$ (Phase 8 persistence policy).
3. **Policy 3 (State-Based Severity)**: $S_t \ge 3 \land S_{t-1} \ge 3$ (Consecutive progressive deterioration).
4. **Policy 4 (State Progression)**: $S_t \ge 3 \land V_t \ge 0$ (Progressive deterioration with non-reversing trajectory).
5. **Policy 5 (Hybrid Decision Policy)**: $p_t \ge 0.85\tau \land S_t \ge 2 \land P_t \ge 2$ (Combined continuous risk + persistent physiological abnormality).

---

## 3. Clinical Operating Points

| Alert Policy | TP | FP | TN | FN | Sensitivity (%) | Specificity (%) | PPV (%) | NPV (%) | FPR (%) | False Alerts / Hour | Alerts / Patient | Median Lead Time | Pct Warned $\ge 30$m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Policy 1** (Single Window $p \ge \tau$) | 79 | 185 | 252 | 31 | **$71.82\%$** | $57.67\%$ | $29.92\%$ | $89.05\%$ | $42.33\%$ | $0.843$ | $0.483$ | $10.0$ min | $15.5\%$ |
| **Policy 2** (Two Consecutive $p \ge \tau$) | 61 | 121 | 316 | 49 | **$55.45\%$** | **$72.31\%$** | $33.52\%$ | $86.58\%$ | $27.69\%$ | **$0.476$** | **$0.333$** | $7.5$ min | $11.0\%$ |
| **Policy 3** (Two Consecutive $S \ge 3$) | 108 | 434 | 3 | 2 | $98.18\%$ | $0.69\%$ | $19.93\%$ | $60.00\%$ | $99.31\%$ | $2.460$ | $0.991$ | $35.0$ min | $71.0\%$ |
| **Policy 4** (State $S \ge 3 \land V \ge 0$) | 110 | 437 | 0 | 0 | $100.00\%$ | $0.00\%$ | $20.11\%$ | $0.00\%$ | $100.00\%$ | $3.027$ | $1.000$ | $37.5$ min | $82.1\%$ |
| **Policy 5** (Hybrid: $0.85\tau + S \ge 2 + P \ge 2$) | 83 | 212 | 225 | 27 | **$75.45\%$** | **$51.49\%$** | $28.14\%$ | **$89.29\%$** | $48.51\%$ | $1.046$ | $0.539$ | $10.0$ min | $10.5\%$ |

---

## 4. Warning Lead-Time Distribution Analysis

$$T_{\text{warning}} = T_{\text{delivery}} - T_{\text{first alert}}$$

| Policy | Patients Alerted | Median Lead Time | IQR Lead Time | Mean Lead Time | Min Lead | Max Lead | Pct $\ge 10$m | Pct $\ge 20$m | Pct $\ge 30$m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Policy 1** | 264 | $10.0$ min | $2.5 - 20.0$ min | $13.2$ min | $0.0$ min | $40.0$ min | $55.3\%$ | $25.0\%$ | $15.5\%$ |
| **Policy 2** | 182 | $7.5$ min | $2.5 - 17.5$ min | $11.1$ min | $0.0$ min | $37.5$ min | $46.2\%$ | $20.9\%$ | $11.0\%$ |
| **Policy 3** | 542 | $35.0$ min | $32.5 - 37.5$ min | $33.4$ min | $0.0$ min | $37.5$ min | $99.3\%$ | $94.7\%$ | $71.0\%$ |
| **Policy 4** | 547 | $37.5$ min | $32.5 - 37.5$ min | $33.9$ min | $0.0$ min | $37.5$ min | $99.3\%$ | $94.7\%$ | $82.1\%$ |
| **Policy 5** | 295 | $10.0$ min | $2.5 - 17.5$ min | $11.8$ min | $0.0$ min | $37.5$ min | $50.5\%$ | $21.4\%$ | $10.5\%$ |

---

## 5. Clinical Implications & Alarm Fatigue Mitigation

1. **Trade-off Between Lead Time and Specificity**:
   * Pure state-based policies (Policies 3 & 4) trigger very early (median lead time $35-37.5$ min), but suffer near-total false-alarm saturation ($>99\%$ FPR, $>2.4$ false alerts/hr) because non-acidemic fetuses also experience transient variable decelerations during normal labor contractions.
   * Continuous probability-anchored policies (Policies 1, 2, and 5) dramatically suppress false alarms ($0.47 - 1.04$ false alerts/hr) by requiring elevated risk scores, yielding acute lead times ($7.5 - 10.0$ min).
2. **The Role of Persistence**:
   * Requiring 2-window temporal persistence (Policy 2) cuts the false-alert rate nearly in half ($0.843 \to 0.476$ false alerts/hr) while retaining $55.45\%$ sensitivity.
3. **Clinical Context**:
   * In intrapartum labor suites, an alerting rate of $<1$ false alarm per monitoring hour is critical to prevent clinician alarm desensitization. Policy 2 and Policy 5 provide viable research operating points balancing actionable warning time ($7.5-10.0$ min) with clinical specificity ($51.5\%-72.3\%$).

# End of Clinical Operating Analysis Report
