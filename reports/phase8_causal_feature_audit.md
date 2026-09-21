# Phase 8 Gate 1: Causal Feature Audit Report

**Audit Objective**: Programmatic verification of temporal causality and zero future look-ahead across all 19 clinical physiological descriptors.  
**Inference Window**: Rolling 20-minute interval $[t - 20\text{min}, t]$ (4,800 samples at 4 Hz).  
**Sampling Rate**: 4.0 Hz.  
**Audit Status**: **PASSED (19/19 Features Causally Verified)**.

---

## 1. Causal Feature Inventory & Verification Table

| # | Descriptor Name | Physiological Category | Input Domain | Look-Ahead | Causal Unit Test | Algorithmic Definition & Boundary Behavior |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | `baseline` | Morphology | `[t-20m, t]` | None | **PASS** | Iterative localized baseline fit solely within the 20-min window. |
| 2 | `stv` | Variability | `[t-20m, t]` | None | **PASS** | Mean absolute adjacent sample differences (diff) within window. |
| 3 | `ltv` | Variability | `[t-20m, t]` | None | **PASS** | 1-minute epoch percentile ranges within window, excluding decelerations. |
| 4 | `acc_count` | Accelerations | `[t-20m, t]` | None | **PASS** | Count of accelerations (>=15 bpm for >=15s) starting within window. |
| 5 | `early_dec_count` | Decelerations | `[t-20m, t]` | None | **PASS** | Decelerations synchronous with contraction peaks in window. |
| 6 | `late_dec_count` | Decelerations | `[t-20m, t]` | None | **PASS** | Decelerations with nadir lagging contraction peak in window. |
| 7 | `var_dec_count` | Decelerations | `[t-20m, t]` | None | **PASS** | Variable decelerations with steep descent (<30s) in window. |
| 8 | `prolonged_dec_count` | Decelerations | `[t-20m, t]` | None | **PASS** | Decelerations lasting between 2 and 5 minutes in window. |
| 9 | `dec_max_depth` | Decelerations | `[t-20m, t]` | None | **PASS** | Maximum amplitude drop below baseline across window decelerations. |
| 10 | `dec_area` | Decelerations | `[t-20m, t]` | None | **PASS** | Integrated bpm*seconds area for all decelerations in window. |
| 11 | `dec_burden` | Decelerations | `[t-20m, t]` | None | **PASS** | Proportion of window duration occupied by active decelerations. |
| 12 | `longest_dec` | Decelerations | `[t-20m, t]` | None | **PASS** | Duration in seconds of the single longest deceleration in window. |
| 13 | `baseline_slope` | Trends | `[t-20m, t]` | None | **PASS** | Linear slope of baseline within the 20-min window (bpm/hour). |
| 14 | `variability_slope` | Trends | `[t-20m, t]` | None | **PASS** | Linear slope of 1-min epoch STV/LTV across the window. |
| 15 | `uc_count` | Uterine Activity | `[t-20m, t]` | None | **PASS** | Count of qualifying uterine contraction peaks within window. |
| 16 | `tachysystole` | Uterine Activity | `[t-20m, t]` | None | **PASS** | Binary flag for contraction frequency > 5 per 10 minutes in window. |
| 17 | `mean_uc_amp` | Uterine Activity | `[t-20m, t]` | None | **PASS** | Average peak amplitude of uterine contractions in window. |
| 18 | `fhr_uc_lag` | FHR-UC Coupling | `[t-20m, t]` | None | **PASS** | Average time lag from contraction peak to nearest FHR nadir in window. |
| 19 | `fhr_uc_coupling` | FHR-UC Coupling | `[t-20m, t]` | None | **PASS** | Normalized cross-correlation between FHR and UC in window. |

---

## 2. Temporal Boundary Rules & Unit Test Methodology

1. **Strict Look-Ahead Prohibition**: For any prediction timestamp $t$, the input tensor is sliced as $\mathbf{x}_t = \mathbf{X}[t - 20\text{min} : t]$.
2. **Synthetic Future Perturbation Test**: A 40-minute continuous CTG recording was generated. The future segment $(t > 20\text{min})$ was corrupted with non-physiological spikes ($220$ bpm FHR, $100$ a.u. UC). Re-extraction of the 19 features at $t = 20\text{min}$ yielded **bitwise identical results** (Max $\Delta = 0.00000000$).
3. **No Centralized Filtering / Post-Hoc Normalization**: Feature scaling utilizes fold-specific scalers fitted strictly on training data; window baseline estimation operates locally within the 20-minute window.

---

## 3. Gate 1 Conclusion

**Gate 1 is officially PASSED**. All 19 descriptors are certified rolling-safe and causal. The pipeline may proceed to Gate 2 (Real-Time Early-Warning Horizon Evaluation).
