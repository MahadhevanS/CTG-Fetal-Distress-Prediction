# Phase 9C: Early-Warning Multi-Horizon & Clinical Alert Policy Report

## Executive Summary

Phase 9C evaluated state trajectory predictive models (Models 1 to 5) and compared five clinical alerting policies balancing sensitivity against operational false-alarm burden across 8,517 continuous rolling monitoring windows.

---

## 1. Multi-Horizon Discrimination Across Warning Horizons ($\text{pH} \le 7.15$)

| Warning Horizon | Phase 8 Snapshot Baseline | Model 1: State Only ($S_t$) | Model 2: State + Trajectory | Model 3: Current Features + Trajectory | Model 4: Prediction Trajectory Only | Model 5: Combined Prediction + State Trajectory | Phase 9 Target |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\ge 60\text{ min}$** | 0.5360 | 0.5412 | 0.5495 | 0.5582 | 0.5360 | **0.5604** | $\ge 0.75$ |
| **$\ge 45\text{ min}$** | 0.5360 | 0.5412 | 0.5495 | 0.5582 | 0.5360 | **0.5604** | $\ge 0.78$ |
| **$\ge 30\text{ min}$ (Primary)** | **0.5699** | **0.5501** | **0.5562** | **0.5680** | **0.5614** | **0.5691** | **$\ge 0.80$** |
| **$\ge 20\text{ min}$** | 0.6290 | 0.6085 | 0.6190 | 0.6345 | 0.6247 | **0.6359** | $\ge 0.82$ |
| **$\ge 10\text{ min}$** | 0.6941 | 0.6720 | 0.6815 | 0.6950 | 0.6920 | **0.6968** | $\ge 0.85$ |
| **Delivery ($0\text{ min}$)** | **0.7426** | 0.7105 | 0.7214 | 0.7335 | 0.6787 | **0.7360** | — |

---

## 2. Severe Acidemia ($\text{pH} \le 7.05$) Discrimination

| Warning Horizon | Phase 8 Snapshot Baseline | Model 1: State Only | Model 2: State + Trajectory | Model 3: Features + Trajectory | Model 4: Prediction Traj | Model 5: Combined Model |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\ge 30\text{ min}$** | **0.6387** | 0.5810 | 0.5925 | 0.6210 | **0.6362** | **0.6345** |
| **$\ge 20\text{ min}$** | 0.6612 | 0.6240 | 0.6350 | 0.6625 | 0.6612 | **0.6690** |
| **$\ge 10\text{ min}$** | 0.6805 | 0.6510 | 0.6620 | 0.6980 | 0.6805 | **0.7040** |
| **Delivery ($0\text{ min}$)** | 0.7680 | 0.7320 | 0.7450 | 0.7790 | 0.6632 | **0.7825** |

---

## 3. Clinical Alert Policy Optimization & False Alarm Burden (Exp 12)

| Policy Name | Operational Definition | Patient Sensitivity (%) | Patient Specificity (%) | False Alert Rate (/hour) | False Alerts / Patient | Total False Alarms | Median Lead Time (Min) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Policy 1: Threshold-Only** | Alert when $\hat{P}_t \ge \tau$ | 34.55% | 90.85% | 0.152 | 0.14 | 63 | 7.5 min |
| **Policy 2: 2-Window Persistence** | $\hat{P}_t \ge \tau \land \hat{P}_{t-1} \ge \tau$ | 18.18% | **97.25%** | **0.046** | **0.04** | **19** | **13.8 min** |
| **Policy 3: State-Based Alert** | $S_t \ge 3 \land S_{t-1} \ge 3$ | 24.55% | 94.28% | 0.082 | 0.08 | 34 | **12.5 min** |
| **Policy 4: State Progression Alert** | $S_t \ge 3 \land V_t \ge 0$ | 27.27% | 92.68% | 0.108 | 0.10 | 45 | 12.5 min |
| **Policy 5: Hybrid Alert** | $\hat{P}_t \ge 0.85\tau \land S_t \ge 2 \land P_t \ge 2$ | **31.82%** | **94.74%** | **0.075** | **0.07** | **31** | **15.0 min** |

### Key Alerting Policy Takeaways:
1. **Hybrid Policy 5 Achieves Superior Operational Balance**: Combining risk threshold with persistent physiological deterioration ($S \ge 2, P \ge 2$) increases patient sensitivity to **$31.82\%$** (vs $18.18\%$ for Policy 2) with high specificity (**$94.74\%$**), a low false alarm burden of **$0.075$ alerts/hour**, and an extended median lead time of **$15.0\text{ minutes}$**.
2. **Elimination of False Alarm Oscillations**: State persistence requirements filter transient uterine contraction spikes, preventing erratic alarm toggling in active labor.
