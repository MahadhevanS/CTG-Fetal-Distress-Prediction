# Phase 9C: State Transition & Trajectory Dynamics Analysis Report

## Executive Summary

Phase 9C performed comprehensive empirical evaluations of the transition dynamics, state occupancy distributions, persistence durations, reversals, and progression velocity across the 547 CTU-UHB patients.

---

## 1. State-Risk Gradient & Trend Validation (Exp 1)

| Research State | Maximum State Reached ($N$) | Acidemia ($\text{pH} \le 7.15$) Cases | Observed Prevalence (%) [95% Wilson CI] | Severe Acidemia ($\text{pH} \le 7.05$) Cases | Severe Prevalence (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **State 0: Stable** | 162 | 14 | **8.64%** [5.18%, 13.98%] | 2 | **1.23%** |
| **State 1: Emerging** | 148 | 21 | **14.19%** [9.47%, 20.67%] | 5 | **3.38%** |
| **State 2: Persistent** | 114 | 26 | **22.81%** [15.99%, 31.39%] | 10 | **8.77%** |
| **State 3: Progressive** | 81 | 29 | **35.80%** [26.04%, 46.85%] | 13 | **16.05%** |
| **State 4: Severe** | 42 | 20 | **47.62%** [33.38%, 62.29%] | 11 | **26.19%** |

- **Monotonic Gradient**: Acidemia risk increases strictly monotonically with research state severity ($8.64\% \to 14.19\% \to 22.81\% \to 35.80\% \to 47.62\%$).
- **Severe Acidemia Risk**: Risk of severe metabolic acidemia increases **$21.3\times$** from State 0 ($1.23\%$) to State 4 ($26.19\%$).

---

## 2. State Occupancy & Duration Analysis (Exp 2)

| Research State | Mean Minutes (Normal Fetuses) | Median Minutes (Normal) | Mean Minutes (Acidemic Fetuses) | Median Minutes (Acidemic) | Mann-Whitney $U$ | $p$-value |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **State 0: Stable** | **23.4 min** | 20.0 min | **14.2 min** | 10.0 min | 30421.5 | **$< 0.0001$** |
| **State 1: Emerging** | 9.8 min | 7.5 min | 10.4 min | 7.5 min | 23512.0 | $0.684$ |
| **State 2: Persistent** | 5.2 min | 2.5 min | **9.6 min** | 5.0 min | 18450.0 | **$0.0002$** |
| **State 3: Progressive** | 2.8 min | 0.0 min | **7.4 min** | 2.5 min | 16210.5 | **$< 0.0001$** |
| **State 4: Severe** | 0.9 min | 0.0 min | **3.8 min** | 0.0 min | 17945.0 | **$< 0.0001$** |

### Key Findings:
- Normal fetuses spend **$65\%$ more time in State 0 (Stable)** than acidemic fetuses ($23.4$ vs $14.2$ minutes, $p < 0.0001$).
- Acidemic fetuses spend **$2.6\times$ more time in Progressive State 3** ($7.4$ vs $2.8$ min, $p < 0.0001$) and **$4.2\times$ more time in Severe State 4** ($3.8$ vs $0.9$ min, $p < 0.0001$).

---

## 3. Reversals, Velocity, and Multidomain Persistence (Exps 4–8)

| Trajectory Metric | Normal Fetuses (Mean) | Acidemic Fetuses (Mean) | Mann-Whitney $U$ | $p$-value |
|---|:---:|:---:|:---:|:---:|
| **Exp 5: State Reversal Count ($R_t$)** | 1.84 reversals | 2.65 reversals | 19850.0 | **$0.0042$** |
| **Exp 6: Maximum Velocity ($V_{4, t}$)** | 0.42 state steps | **0.78 state steps** | 16540.0 | **$< 0.0001$** |
| **Exp 8: Multidomain Duration ($C_t$)** | 0.88 windows | **1.72 windows** | 17210.0 | **$< 0.0001$** |

- Acidemic fetuses exhibit significantly higher upward progression velocity ($V_4 = 0.78$ vs $0.42$, $p < 0.0001$) and nearly double the consecutive duration of concurrent multi-domain deterioration ($C_t = 1.72$ vs $0.88$ windows, $p < 0.0001$).
