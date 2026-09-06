# Phase 9B: Deterioration Pattern Ablation & State Transition Analysis

## Executive Summary

Phase 9B evaluated the incremental contribution of each component in the progression from instantaneous CTG snapshot to multidomain progressive deterioration, and examined the empirical acidemia risk gradient across the five research states.

---

## 1. Deterioration Pattern Component Ablation (Exps 1 to 7)

Seven nested configurations were benchmarked to identify the exact source of predictive value across horizons:

| Experiment | Feature Content | Dimensionality | $\ge 30\text{ min}$ AUROC | Delivery ($0\text{ min}$) AUROC | Severe Acidemia ($\text{pH} \le 7.05$) Delivery AUROC |
|---|---|:---:|:---:|:---:|:---:|
| **Exp 1: Current State Only** | Instantaneous 19 clinical features ($X_t$) | 19 | 0.5512 | 0.7241 | 0.7512 |
| **Exp 2: State + Change** | $X_t + \Delta X_t$ (1-step velocity) | 38 | 0.5564 | 0.7289 | 0.7601 |
| **Exp 3: State + Persistence** | $X_t + P_t$ (abnormality duration) | 25 | 0.5621 | 0.7305 | 0.7645 |
| **Exp 4: State + Progression** | $X_t + \Delta X_t + \text{slope}_t$ | 57 | 0.5582 | 0.7310 | 0.7680 |
| **Exp 5: Full Deterioration Signature** | $X_t + \text{Deltas} + \text{Slopes} + \text{Accels}$ | 76 | 0.5598 | 0.7315 | 0.7763 |
| **Exp 6: Domain-Level Deterioration** | $X_t + \text{Domain Severities, Deltas, Persistences}$ | 52 | **0.5684** | **0.7342** | **0.7812** |
| **Exp 7: Full Trajectory + Multidomain** | Full 112-D Matrix ($X_t + D_t + N_t + C_t + \text{Traj}$) | 112 | **0.5695** | **0.7368** | **0.7845** |

### Key Ablation Insights:
1. **Domain Structuring Outperforms Generic Deltas**: Grouping features into physiological domains and computing domain persistence (Exp 6) outperforms unstructured 76-D generic deltas (Exp 5), improving near-delivery discrimination to AUROC **0.7342** and severe acidemia to **0.7812**.
2. **Persistent vs Transient Value**: Adding persistence (Exp 3) provides a consistent improvement over pure change (Exp 2), showing that the duration of an abnormality is more informative than its instantaneous velocity.
3. **$\ge 30\text{ Min}$ Barrier**: Across all 7 experiments, $\ge 30$-minute AUROC remains bound between $0.5512$ and $0.5695$, confirming that morphological CTG changes do not reliably distinguish eventual acidemia $> 30$ minutes before delivery.

---

## 2. Transient vs Persistent vs Progressive State Analysis

Each patient was classified according to the highest research state reached during continuous monitoring:

| Research State | Operational Definition | $N$ Patients | Acidemia ($\text{pH} \le 7.15$) Cases | Acidemia Prevalence (%) | Severe Acidemia ($\text{pH} \le 7.05$) Cases | Severe Prevalence (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **State 0: Stable** | No sustained abnormality | 162 | 14 | 8.64% | 2 | 1.23% |
| **State 1: Emerging Abnormality** | Transient single spike ($P < 2$) | 148 | 21 | 14.19% | 5 | 3.38% |
| **State 2: Persistent Abnormality** | Sustained single domain abnormality ($P \ge 2$) | 114 | 26 | 22.81% | 10 | 8.77% |
| **State 3: Progressive Deterioration** | Multidomain worsening ($N_t \ge 2$, active slope) | 81 | 29 | 35.80% | 13 | 16.05% |
| **State 4: Severe Multidomain** | Severe multidomain decompensation ($S > 1.0, C \ge 2$) | 42 | 20 | **47.62%** | 11 | **26.19%** |

### Key State Findings:
- **Monotonic Risk Gradient**: Fetal acidemia prevalence rises steeply from **8.64% in State 0** up to **47.62% in State 4** ($5.5\times$ risk increase).
- **Severe Acidemia Concentration**: Fetuses reaching State 4 exhibit a **$21.3\times$ higher prevalence of severe acidemia** ($\text{pH} \le 7.05$) compared to State 0 ($26.19\%$ vs $1.23\%$).
- **Clinical Implication**: While individual CTG features fluctuate transiently, the progression into multi-domain persistence (State 3/4) serves as a robust physiological marker of advancing fetal distress.
