# Phase 12 — Prior-Art Methodological Differentiation & Comparative Analysis

## CTU-UHB Head-to-Head Prior-Art Benchmark (547 Patients, 8,517 Windows)

### 1. Overview & Benchmark Suite Architecture

To ensure an uncompromised scientific benchmark, all comparative methods were evaluated under identical conditions:
* **Cohort**: 547 CTU-UHB intrapartum recordings.
* **Endpoints**: Primary $\text{pH} \le 7.15$ ($110$ cases); Secondary Severe $\text{pH} \le 7.05$ ($41$ cases).
* **Cross-Validation**: Patient-stratified 5-fold CV ($clean\_pids$).
* **Statistical Inference**: Vectorized paired patient-level bootstrap ($B=2,000$).

```
Figure Reference: reports/figures_phase12/fig3_prior_art_head_to_head_comparison.png
```

---

## 2. Head-to-Head Comparative Summary

| Model Identifier | Methodological Concept | Representation Inputs | $\text{AUROC}_{\ge 30\text{m}}$ | $\text{AUROC}_{\text{Del}}$ | Median Warning Lead Time | Comparison vs Proposed Framework (P6) |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **P1** | **Compact CTG Baseline** *(DeepCTG-Inspired)* | 4 compact morphological descriptors (Min/Max Baseline, Accel/Decel Area) | 0.5727 | 0.5090 | 12.5 min | **P6 significantly superior at delivery** ($\Delta = +0.1801, p < 0.001$) |
| **P2** | **Sequential / Event Baseline** *(Vargas-Calixto-Inspired)* | Deceleration depth, duration, event frequency + EWMA temporal persistence | **0.5983** | 0.6774 | 12.5 min | **Statistically comparable across horizons** (P2 leads at $\ge 30$m, P6 leads at delivery) |
| **P3** | **Locked Snapshot Baseline** *(Continuous Huber Regression)* | 19 FIGO descriptors + Continuous Huber acid-base supervision | 0.5461 | 0.5976 | 10.0 min | **P6 significantly superior at delivery** ($\Delta = +0.0896, p < 0.001$) |
| **P4** | **Snapshot + State** | Continuous Huber risk $R_t$ + Research State $S_t$ | 0.5459 | 0.5945 | 10.0 min | State alone without dynamics provides limited gain |
| **P5** | **Snapshot + Trajectory** | Continuous Huber risk $R_t$ + Dynamic operators $(V, P, A, R, N)$ | 0.5355 | 0.6142 | 12.5 min | **Trajectory adds significant incremental value** ($\Delta_{\text{P5}-\text{P3}} = +0.0165, p = 0.041$) |
| **P6** | **Full Proposed Framework** | 40-D: 19 Descriptors + 6 Domain Severities + States + Trajectory + Occupancies | 0.5857 | **0.6872** | **17.5 min** | **Proposed Full System (Highest Delivery AUROC & Longest Warning Lead Time)** |

---

## 3. Deep-Dive Differentiation by Comparator Class

### Class 1: Differentiation from Compact Morphological Prior Art (P1 — DeepCTG-Inspired)
* **P1 Formulation**: Reduces 20-minute CTG traces into 4 summary statistics (minimum baseline, maximum baseline, acceleration area, deceleration area) feeding regularized logistic regression.
* **Observed Limitation of P1**: While compact features achieve modest discrimination early in labor ($\text{AUROC} = 0.5727$ at $\ge 30$m), they collapse near delivery ($\text{AUROC} = 0.5090$). Compact features fail to capture complex deceleration morphology (late vs. variable vs. prolonged), short-term variability depression, or uterine tachysystole.
* **Thesis Differentiation**: P6 outperforms P1 by **$+0.1801$ AUROC ($95\%$ CI: $[+0.1111, +0.2475]$, $p < 0.001$)** at delivery, proving that compact 4-feature representations are inadequate for capturing intrapartum decompensation.

### Class 2: Differentiation from Static Snapshot Prior Art (P3 — Continuous Huber Regression)
* **P3 Formulation**: Uses all 19 FIGO descriptors evaluated independently per 20-minute causal window without temporal state transitions or trajectory dynamics.
* **Observed Limitation of P3**: Evaluates each window in isolation. It cannot distinguish an isolated, recovering deceleration in an otherwise healthy fetus from repetitive, non-recovering decelerations occurring under progressive hypoxemia.
* **Thesis Differentiation**:
  1. Trajectory dynamics alone (P5) add **$+0.0165$ AUROC ($p = 0.041$)** over P3.
  2. Full Multidomain Framework (P6) outperforms P3 by **$+0.0896$ AUROC ($95\%$ CI: $[+0.0349, +0.1470]$, $p < 0.001$)** at delivery and extends median warning lead time from $10.0$ min to $17.5$ min.

### Class 3: Differentiation from Sequential / Event-Based Prior Art (P2 — Vargas-Calixto-Inspired)
* **P2 Formulation**: Extracts univariate deceleration depth, duration, and frequency per epoch, applying exponential moving average (EWMA) smoothing and temporal threshold persistence.
* **Detailed Horizon Comparison**:
  - **At $\ge 30$ minutes**: $\text{AUROC}_{\text{P2}} = 0.5983$ vs $\text{AUROC}_{\text{P6}} = 0.5857$ ($\Delta = -0.0126$, $95\%$ CI: [$-0.0478, +0.0219$], $p = 0.485$, not statistically significant).
  - **At Delivery (0m)**: $\text{AUROC}_{\text{P6}} = 0.6872$ vs $\text{AUROC}_{\text{P2}} = 0.6774$ ($\Delta = +0.0098$, $95\%$ CI: [$-0.0392, +0.0593$], $p = 0.697$, not statistically significant).
* **Physiological Operating Regimes**:
  - **P2 Advantage Regime**: P2 excels in cases of gradual, isolated autonomic baseline shifts ($>138$ bpm) with preserved variability and low uterine activity.
  - **P6 Advantage Regime**: P6 strongly dominates in acute multi-channel decompensation characterized by uterine tachysystole ($31.8\%$ vs $12.2\%$), severe deceleration depth ($45.2$ vs $32.1$ bpm), and delayed autonomic recovery lag ($24.6$ vs $14.2$ s).

```
Figure Reference: reports/figures_phase12/fig6_horizon_dependent_dynamics.png
```

---

## 4. Prior-Art Positioning Summary

1. **Definite Superiority**: The proposed framework demonstrates statistically significant superiority over compact morphological representations (P1) and static snapshot models (P3) near delivery ($p < 0.001$).
2. **Comparable Complementary Regimes**: The proposed framework is statistically comparable to sequential event modeling (P2), with P2 capturing early autonomic baseline drift and P6 capturing acute multi-domain decompensation near delivery.
3. **Operational Lead Time**: P6 achieves the most favorable warning-time-to-false-alert profile across all tested models ($16.0$ min median lead at $\text{FAR}=0.50$/hr).
