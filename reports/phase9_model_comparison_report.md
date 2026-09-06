# Phase 9: Multi-Model Horizon Benchmark & Comparative Analysis

## Executive Summary

Phase 9 evaluated five distinct temporal modeling strategies against the frozen Phase 8 continuous clinical Huber snapshot baseline across 6 warning horizons ($\ge 60\text{ min}$, $\ge 45\text{ min}$, $\ge 30\text{ min}$, $\ge 20\text{ min}$, $\ge 10\text{ min}$, and Delivery $0\text{ min}$) on the CTU-UHB cohort ($N = 547$ patients, $N_{\text{acidemia}} = 110$, $N_{\text{severe}} = 41$).

The evaluated candidate architectures represent a systematic progression in temporal model complexity:
1. **Frozen Phase 8 Baseline**: Continuous Clinical Huber snapshot model (AUROC 0.5699 at $\ge 30\text{m}$, 0.7426 at delivery).
2. **Model 1 (Temporal LR)**: Regularized Logistic Regression with 104 causal temporal features.
3. **Model 2 (Temporal GBM)**: Gradient-Boosted Trees capturing non-linear interactions between current state and deterioration rates.
4. **Model 3 (Trajectory EWMA)**: Exponentially weighted moving average risk filter smoothing transient artifacts.
5. **Model 4 (Sequential GRU)**: Recurrent neural network modeling sequences of successive 20-minute clinical feature vectors.
6. **Model 5 (Multi-Horizon Net)**: Multi-task neural network jointly optimized with auxiliary horizon-specific prediction heads ($\ge 10\text{m}, \ge 20\text{m}, \ge 30\text{m}$).

---

## 1. Primary Endpoint Performance Across Warning Horizons ($\text{pH} \le 7.15$)

| Warning Horizon | Phase 8 Baseline | Model 1: Temporal LR | Model 2: Temporal GBM | Model 3: Trajectory EWMA | Model 4: Sequential GRU | Model 5: Multi-Horizon | Target |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\ge 60\text{ min}$** | 0.5360 | 0.5594 | **0.6503** | 0.5360 | 0.6015 | 0.5709 | $\ge 0.75$ |
| **$\ge 45\text{ min}$** | 0.5360 | 0.5594 | **0.6503** | 0.5360 | 0.6165 | 0.5875 | $\ge 0.78$ |
| **$\ge 30\text{ min}$ (Primary)** | **0.5699** | **0.5598** | **0.5510** | **0.5614** | **0.5709** | **0.4979** | **$\ge 0.80$** |
| **$\ge 20\text{ min}$** | 0.6290 | **0.6346** | 0.6274 | 0.6247 | 0.6061 | 0.5777 | $\ge 0.82$ |
| **$\ge 10\text{ min}$** | 0.6941 | 0.6911 | 0.6822 | **0.6920** | 0.6524 | 0.6105 | $\ge 0.85$ |
| **Delivery ($0\text{ min}$)** | **0.7426** | **0.7315** | 0.6875 | 0.6787 | 0.6955 | 0.6299 | — |

---

## 2. Severe Acidemia Discrimination ($\text{pH} \le 7.05$)

| Warning Horizon | Model 1: Temporal LR | Model 2: Temporal GBM | Model 3: Trajectory EWMA | Model 4: Sequential GRU | Model 5: Multi-Horizon |
|---|:---:|:---:|:---:|:---:|:---:|
| **$\ge 60\text{ min}$** | 0.5567 | 0.6166 | 0.5886 | 0.5971 | 0.5914 |
| **$\ge 45\text{ min}$** | 0.5567 | 0.6166 | 0.5886 | 0.6289 | 0.6116 |
| **$\ge 30\text{ min}$** | 0.5642 | 0.5662 | **0.6362** | 0.5827 | 0.5068 |
| **$\ge 20\text{ min}$** | 0.6464 | 0.5466 | **0.6612** | 0.6335 | 0.5383 |
| **$\ge 10\text{ min}$** | 0.6792 | 0.7045 | **0.6805** | 0.6390 | 0.5985 |
| **Delivery ($0\text{ min}$)** | **0.7763** | 0.7144 | 0.6632 | 0.7147 | 0.6700 |

---

## 3. Statistical Significance & Paired Bootstrap Analysis vs Phase 8

Paired patient-level bootstrap analysis ($B = 2,000$ iterations) evaluated whether differences in AUROC between Phase 9 models and the Phase 8 baseline were statistically significant:

- **At $\ge 30\text{ Minutes}$**:
  - Model 1 (Temporal LR) vs Baseline: $\Delta\text{AUROC} = -0.0101$ [95% CI: $-0.042, +0.021$], $p = 0.518$ (Not significant).
  - Model 2 (Temporal GBM) vs Baseline: $\Delta\text{AUROC} = -0.0189$ [95% CI: $-0.056, +0.019$], $p = 0.334$ (Not significant).
  - Model 4 (Sequential GRU) vs Baseline: $\Delta\text{AUROC} = +0.0010$ [95% CI: $-0.038, +0.041$], $p = 0.962$ (Not significant).
- **At Delivery ($0\text{ Minutes}$)**:
  - Model 1 (Temporal LR) vs Baseline: $\Delta\text{AUROC} = -0.0111$ [95% CI: $-0.032, +0.010$], $p = 0.306$ (Comparable).
  - Model 1 on Severe Acidemia ($\text{pH} \le 7.05$): Achieved AUROC **0.7763**, exhibiting strong discrimination near delivery.

---

## 4. Key Scientific Findings

1. **Failure of Early Signal Amplification**: None of the five temporal model families succeeded in elevating $\ge 30$-minute AUROC toward the pre-specified target of $0.80$ (best model achieved $0.5709$).
2. **Biological Reality of Late Deterioration**: Cardiotocographic manifestations of metabolic acidemia (repetitive late decelerations, progressive loss of STV, tachycardia) emerge primarily during the second stage of labor within 20–30 minutes of delivery. Prior to this window, fetal autonomic compensation masks distress in the CTG tracing.
3. **Severe Acidemia Signal**: For severe acidemia ($\text{pH} \le 7.05$), trajectory EWMA and temporal LR maintain higher early discrimination ($\text{AUROC} = 0.6362$ at $\ge 30\text{m}$ and $0.7763$ at delivery), indicating that severe physiological insults manifest earlier and more intensely.
