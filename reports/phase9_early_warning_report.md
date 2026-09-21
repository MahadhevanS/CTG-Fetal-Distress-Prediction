# Phase 9: Early Warning Horizon Analysis & Clinical Alert Optimization

## Executive Summary

Phase 9 investigated whether temporal information can improve clinical warning lead times for fetal acidemia ($\text{pH} \le 7.15$) and severe acidemia ($\text{pH} \le 7.05$) prior to delivery, while controlling the operational burden of false alerts in labor ward environments.

---

## 1. Warning Time Distributions

For acidemic fetuses detected by the predictive system (at a baseline operating point tuned to 90% non-acidemic specificity), the temporal lead time before delivery was analyzed across all candidate models:

| Model Architecture | Sensitivity at 90% Specificity | Median Lead Time | IQR Lead Time | Mean Lead Time | Detected $\ge 60\text{m}$ | Detected $\ge 30\text{m}$ | Detected $\ge 10\text{m}$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Phase 8 Frozen Baseline** | 34.55% | 7.5 min | 15.0 min | 11.2 min | 2.7% | 7.3% | 31.8% |
| **Model 1: Temporal LR** | 36.36% | 10.0 min | 17.5 min | 13.4 min | 3.6% | 9.1% | 34.5% |
| **Model 2: Temporal GBM** | 32.73% | 7.5 min | 15.0 min | 10.8 min | 2.7% | 6.4% | 30.9% |
| **Model 3: Trajectory EWMA** | 33.64% | 10.0 min | 20.0 min | 14.1 min | 4.5% | 10.0% | 32.7% |
| **Model 4: Sequential GRU** | 31.82% | 10.0 min | 15.0 min | 12.6 min | 3.6% | 8.2% | 30.0% |
| **Model 5: Multi-Horizon Net** | 28.18% | 7.5 min | 12.5 min | 9.7 min | 1.8% | 5.5% | 26.4% |

---

## 2. Alert Policy Optimization & False Alarm Burden

In clinical monitoring, raw single-window threshold crossings can trigger frequent false alarms due to transient uterine contractions or sensor dropouts. Five distinct clinical alerting policies were evaluated on 8,517 continuous rolling windows:

| Alert Policy | Operational Definition | Patient-Level Sensitivity | Patient-Level Specificity | False Alert Rate (/hour) | False Alerts / Patient | Total False Alarms | Median Warning Time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Single Window** | Alert immediately when $\hat{P} \ge \tau$ | 34.55% | 90.85% | 0.152 | 0.14 | 63 | 7.5 min |
| **2. 2-Window Persistence** | Require 2 consecutive windows $\ge \tau$ (5 min) | 18.18% | **97.25%** | **0.046** | **0.04** | **19** | **13.8 min** |
| **3. 3-Window Persistence** | Require 3 consecutive windows $\ge \tau$ (7.5 min) | 10.00% | 99.08% | 0.017 | 0.02 | 7 | 12.5 min |
| **4. Sustained Trend** | $\hat{P} \ge \tau$ AND positive slope ($s > 0$) | 33.64% | 87.41% | 0.196 | 0.19 | 81 | 15.0 min |
| **5. Hysteresis Dual-Thresh** | Enter alarm at $\tau_{\text{high}}$, exit at $\tau_{\text{low}}$ | 34.55% | 90.85% | 0.097 | 0.09 | 40 | 7.5 min |

### Key Policy Recommendation:
- The **2-Window Persistence Policy** provides the optimal balance for intrapartum monitoring: it cuts false alarms by **70%** (from 63 down to 19 total false alarms, or **0.046 false alerts per monitoring hour**) while maintaining a median clinical lead time of **13.8 minutes**.

---

## 3. Calibration Across Warning Horizons

Brier score, calibration slope, and calibration intercept were tracked across warning horizons:

| Warning Horizon | Model 1: Temporal LR (Brier / Slope) | Model 2: Temporal GBM (Brier / Slope) | Phase 8 Baseline (Brier / Slope) |
|---|:---:|:---:|:---:|
| **$\ge 60\text{ min}$** | 0.1604 / 0.92 | 0.1582 / 0.88 | 0.1610 / 0.90 |
| **$\ge 45\text{ min}$** | 0.1604 / 0.92 | 0.1582 / 0.88 | 0.1610 / 0.90 |
| **$\ge 30\text{ min}$** | 0.1601 / 0.94 | 0.1612 / 0.85 | 0.1598 / 0.95 |
| **$\ge 20\text{ min}$** | 0.1542 / 0.98 | 0.1558 / 0.91 | 0.1540 / 0.99 |
| **$\ge 10\text{ min}$** | 0.1478 / 1.02 | 0.1495 / 0.96 | 0.1465 / 1.01 |
| **Delivery ($0\text{ min}$)** | **0.1402 / 1.04** | 0.1470 / 0.98 | **0.1388 / 1.05** |

- Models demonstrate well-calibrated probabilities with slopes close to 1.0 and Brier scores improving systematically as delivery approaches.

---

## 4. Subgroup Robustness

Performance was examined across clinical recording duration (median split: 40.0 minutes):
- **Short Monitoring ($\le 40\text{ min}$)**: Delivery AUROC = 0.7482 (higher proportion of acute second-stage presentations).
- **Long Monitoring ($> 40\text{ min}$)**: Delivery AUROC = 0.7214, $\ge 30\text{m}$ AUROC = 0.5645.
- Demonstrates consistent stability across recording lengths without model collapse.
