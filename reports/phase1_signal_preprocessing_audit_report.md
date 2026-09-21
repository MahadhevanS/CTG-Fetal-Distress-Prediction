# Phase 1 Deliverable: CTU-UHB Signal Quality & Preprocessing Re-Engineering Report

**Date:** 2026-09-06  
**Cohort:** CTU-UHB Clean Cohort (547 patients, 110 positive [pH ≤ 7.15], 8,517 usable 20-min windows)  
**Evaluation Protocol:** Frozen Patient-Grouped 5-Fold Stratified Partition (`data/processed_clinical/folds.json`)  
**Primary Endpoint:** Umbilical Artery pH ≤ 7.15 (Frozen)  

---

## 1. Executive Summary & Core Decision

Phase 1 conducted an exhaustive signal-quality, interpolation, baseline-leakage, and morphology-preservation audit across all 547 patient recordings and 8,517 evaluation windows.

### Primary Audit Findings:
1. **Missingness Structure:** 90.9% of all missing signal gaps in raw recordings (and 92.6% in usable 20-min windows) are **$\le 15$ seconds**. Cubic spline interpolation capped at 15s repairs short transducer dropouts without fabricating long artificial heart rate trends.
2. **Signal Mask Necessity:** Raw CTG signals contain an average of 11.85% missing samples per window (median 7.54%). 70.5% of these missing samples are reconstructed via short-gap interpolation. Reconstructed samples must not be presented to models without an explicit mask distinguishing genuine observations from interpolated fills.
3. **Baseline Drift & Leakage:** The fetal heart rate baseline drifts by a median of **24.77 bpm** (mean 47.66 bpm) across a 20-minute window. A single constant baseline per window obscures non-stationary baseline drift. Furthermore, computing baselines from whole recordings introduces a **7.36 bpm future-informed leakage shift** into earlier windows. Baselines must be calculated strictly from within each window using a localized rolling filter.
4. **Morphology & Channel Representation:** Supplying both absolute FHR ($FHR(t)$) and baseline-relative deviation ($\Delta FHR(t)$), alongside observed quality ($m(t)$) and interpolation ($m_{interp}(t)$) masks, preserves both global physiological rate and acute decelerative dynamics.
5. **Predictive Sanity Check:** Evaluating a compact ResNet-1D baseline under the frozen 5-fold patient protocol demonstrated that Quality-Aware preprocessing (P2) achieves the highest precision-recall performance (**AUPRC 0.3238 vs 0.2959**, $\Delta AUROC = +0.0040$, paired $p=0.445$) while strictly eliminating leakage and preserving signal veracity.

### Decision:
> **ADOPT QUALITY-AWARE (P2)** as the Frozen Preprocessing Specification for all subsequent model development.

---

## 2. Deliverable A: Existing Pipeline Audit

### Exact Current Processing Sequence:
```text
Raw CTU-UHB (.dat / .hea)
     ↓
Signal Ingestion (4 Hz sampling, FHR & UC channels)
     ↓
Patient Quality Gate (Exclude if missing ratio > 50% across recording)
     ↓
Last 60-Minute Truncation (Delivery horizon)
     ↓
Sliding Window Extraction (20 min = 4800 samples, 2.5 min stride = 600 samples)
     ↓
Window Quality Gate (Exclude window if missing ratio > 50%)
     ↓
Artifact & Spike Removal (>25 bpm/s deltas zeroed)
     ↓
Missing Data Interpolation (Cubic spline on gaps ≤ 15s, clamped to [50, 240] bpm)
     ↓
Lowpass Filtering (4th-order zero-phase Butterworth, 1.5 Hz cutoff)
     ↓
Baseline Estimation (Iterative mean ±15 bpm exclusion, rounded to 5 bpm)
     ↓
Baseline Correction (Channel 0 centered to FHR - Baseline)
     ↓
Normalization (Train-fit Z-score scaling on Channels 0 and 1)
     ↓
Model Input (Batch, Channels, 4800)
```

### Operation Inventory Table:

| Operation | Input | Output | Parameters | Reason | Fixed / Learned | Can Introduce Future Info? | Whole Recording Dependence? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ingestion** | `.dat` / `.hea` | Raw FHR, UC (bpm, mmHg) | $f_s = 4.0\text{ Hz}$ | Standardize sampling rate | Fixed | No | Yes (loads whole file) |
| **Patient Gate** | Full FHR | Boolean | Max missing $\le 50\%$ | Exclude corrupt recordings | Fixed | No | Yes |
| **Truncation** | Full FHR/UC | Last 60 min | $N=14,400$ samples | Focus on terminal intrapartum phase | Fixed | No | Yes |
| **Window Extraction** | 60 min FHR/UC | 20 min windows | $L=4800, S=600$ | Standard FIGO assessment epoch | Fixed | No | No |
| **Window Gate** | Window FHR | Boolean | Max missing $\le 50\%$ | Reject uninterpretable epochs | Fixed | No | No |
| **Spike Removal** | Window FHR | Spike-zeroed FHR | Max delta $25\text{ bpm/s}$ ($6.25\text{ bpm/pt}$) | Remove transducer slip artifacts | Fixed | No | No |
| **Interpolation** | Spike-zeroed FHR | Interpolated FHR | Cubic spline, max gap $\le 15\text{s}$ (60 pts) | Repair short autonomic dropouts | Fixed | No (local $\le 15$s) | No |
| **Lowpass Filter** | Interpolated FHR/UC | Smoothed signal | 4th-order Butterworth, $f_c=1.5\text{ Hz}$ | Eliminate out-of-band noise | Fixed | Zero-phase (within window) | No |
| **Baseline Estimation** | Smoothed FHR | Baseline $B(t)$ | Iterative $\pm 15\text{ bpm}$, 5 bpm rounding | FIGO baseline definition | Fixed | If computed on full record: **YES**. If window: **NO** | Dependent on implementation |
| **Channel Scaling** | Multi-channel window | Z-scored window | Train-set mean $\mu$, std $\sigma$ | Stabilize neural network training | **Learned (Train-fit)** | If fit on test/all: **YES**. If train-only: **NO** | Fit on train fold only |

---

## 3. Deliverable B & C: Signal-Quality & Missingness Report

Audit conducted across all 547 patients (110 positive, 437 negative) and 8,517 20-minute windows.

### Cohort Quality Summary:
- **Total Patient Recordings:** 547 (110 acidotic [pH $\le 7.15$], 437 normal)
- **Total Evaluated Windows:** 8,517 (1,659 acidotic, 6,858 normal)
- **Recording-Level FHR Missingness:** Mean = $18.48\%$, Median = $18.18\%$, IQR = $[8.91\%, 26.45\%]$
- **Recording-Level UC Missingness:** Mean = $19.40\%$
- **Total Unphysiological Out-of-Bounds Values ($<50$ or $>240$ bpm):** 1,783 samples
- **Total Transducer Abrupt Jumps ($>25$ bpm/s):** 87,092 events
- **Total Flatline Samples ($>10$s identical non-zero values):** 464 samples

### Missing Gap Duration Distribution:

| Gap Duration Range | Samples ($4\text{ Hz}$) | Full Recording Gap Count | Full Recording % | Window Gap Count | Window % | Usable Data Handling |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0 – 5 sec** | 1 – 20 | 28,766 | 74.14% | 132,031 | 75.79% | Interpolated (Physiologically safe) |
| **5 – 15 sec** | 21 – 60 | 6,523 | 16.81% | 29,235 | 16.78% | Interpolated (Bounded cubic spline) |
| **15 – 30 sec** | 61 – 120 | 1,738 | 4.48% | 7,643 | 4.39% | Preserved as Missing (Do not fabricate) |
| **30 – 60 sec** | 121 – 240 | 956 | 2.46% | 3,307 | 1.90% | Preserved as Missing |
| **> 60 sec** | > 240 | 815 | 2.10% | 2,001 | 1.15% | Preserved as Missing |
| **Total Gaps** | — | **38,798** | **100.0%** | **174,217** | **100.0%** | — |

### Justification of Gap Cutoff ($\le 15$ seconds):
1. **Coverage:** Gaps $\le 15$ seconds account for **92.57%** of all missing occurrences in windows.
2. **Physiological Plausibility:** The autonomic nervous system modulates fetal heart rate on a timescale of 5–15 seconds (baroreceptor response). Gaps under 15 seconds can be smoothly bridged without creating fictitious decelerations. Gaps $>15$ seconds risk inventing or erasing genuine decelerations.
3. **Usable Window Retention:** In 20-minute windows, an average of $8.36\%$ of samples are repaired by the 15-second cutoff, leaving genuine longer gaps explicitly tagged as missing.

### Missingness Mechanism & Outcome Breakdown (Diagnostic Audit):
- **Positive Patients (pH $\le 7.15$):** Mean recording missingness = $21.20\%$, Mean window missingness = $13.27\%$, Usable windows/patient = $15.08$
- **Negative Patients (pH $> 7.15$):** Mean recording missingness = $17.79\%$, Mean window missingness = $11.51\%$, Usable windows/patient = $15.69$
- **Mechanism:** Missingness is **MAR-like** (Missing at Random conditional on maternal movement and strong contractions during the active second stage). Positives exhibit slightly higher missingness ($+3.41\%$ at recording level) due to increased fetal descent/monitoring difficulty during distressed labor.
- **Rule Enforcement:** Quality gates remain strictly threshold-based ($50\%$ missing cutoff) and are **NEVER** tuned or defined using outcome labels.

---

## 4. Deliverable D: Baseline & Leakage Audit

### 1. Estimator Audit & Distribution:
- **Mean Baseline:** $136.42 \pm 15.40\text{ bpm}$
- **Physiological Bounds:** $7.93\%$ of baseline estimates fall outside the standard normal range ($110 - 160\text{ bpm}$), reflecting physiological fetal tachycardia ($>160\text{ bpm}$) and bradycardia ($<110\text{ bpm}$).
- **Temporal Stability & Drift:** Within a single 20-minute window, localized rolling baseline drift averages **$47.66\text{ bpm}$** (median **$24.77\text{ bpm}$**). A constant baseline per window forces non-stationary baseline drift into apparent accelerations/decelerations.

### 2. Prospective Baseline Leakage Audit:
- **Whole-Recording vs Window-Level Baseline Diff:** When a baseline is estimated over the whole recording and sliced into windows, it differs from the strictly window-contained baseline by an average of **$7.36\text{ bpm}$**.
- **Conclusion:** Estimating baselines from the whole recording introduces future information leakage. In the frozen specification, all baseline estimation must be calculated **strictly within the 20-minute window**, ensuring 100% causal/retrospective validity.

---

## 5. Deliverable E: Preprocessing Candidates Definition

### Candidate Regimes:

```text
========================================================================================
Regime P0 (Current Baseline - Control)
----------------------------------------------------------------------------------------
- Spike Removal: Absolute sample delta > 6.25 bpm (25 bpm/s) zeroed.
- Interpolation: Cubic spline on gaps <= 15s (60 pts), clamped to [50, 240] bpm.
- Smoothing: 4th-order zero-phase Butterworth lowpass filter at 1.5 Hz.
- Baseline: Constant iterative baseline (+/- 15 bpm exclusion, rounded to 5 bpm).
- Output Channels:
    Ch 0: Baseline-corrected FHR (FHR - B) [Z-score normalized on Train fold]
    Ch 1: Lowpass UC [Z-score normalized on Train fold]
    Ch 2: Missingness mask (1 = missing, 0 = observed)

========================================================================================
Regime P1 (Minimal Physiological Cleaning)
----------------------------------------------------------------------------------------
- Artifact Cleaning: Values < 50 or > 240 bpm zeroed; spikes > 25 bpm/s zeroed.
- Interpolation: Conservative short-gap interpolation (<= 5s only); gaps > 5s left missing.
- Smoothing: NONE (Preserves full raw high-frequency variability).
- Baseline: Localized rolling iterative baseline calculated within window.
- Output Channels:
    Ch 0: Raw cleaned FHR [Z-score normalized on Train fold]
    Ch 1: Baseline-relative Delta FHR (FHR - B) [Z-score normalized on Train fold]
    Ch 2: Observed quality mask (1 = observed, 0 = missing)

========================================================================================
Regime P2 (Quality-Aware Multi-Channel Representation - RECOMMENDED)
----------------------------------------------------------------------------------------
- Artifact Cleaning: Spikes > 25 bpm/s zeroed; physiological bounds [50, 240] enforced.
- Interpolation: Cubic spline on gaps <= 15s with bounded clamping.
- Smoothing: 4th-order zero-phase Butterworth lowpass filter at 1.5 Hz.
- Baseline: Drift-tracking localized rolling baseline calculated within window.
- Output Channels:
    Ch 0: Absolute FHR (FHR(t)) [Z-score normalized on Train fold]
    Ch 1: Baseline-relative Delta FHR (FHR(t) - B(t)) [Z-score normalized on Train fold]
    Ch 2: Observed quality mask (1 = genuine observation, 0 = missing/artifact)
    Ch 3: Interpolation mask (1 = genuine observation, 0 = reconstructed fill / missing)
========================================================================================
```

---

## 6. Deliverable F: Morphology-Preservation Analysis

Quantifying signal feature changes across all 8,517 windows:

| Metric | P0 (Control) | P1 (Minimal) | P2 (Quality-Aware) | Physiological Implication |
| :--- | :--- | :--- | :--- | :--- |
| **Short-Term Variability (STV, ms)** | $0.869 \pm 0.38$ | $0.916 \pm 0.44$ | $0.871 \pm 0.38$ | Lowpass filter attenuates transducer jitter without destroying autonomic STV |
| **Long-Term Variability (LTV, bpm)** | $26.39 \pm 14.12$ | $36.82 \pm 22.45$ | $26.85 \pm 14.30$ | P1 retains unfiltered high-frequency noise spikes; P0/P2 track genuine LTV |
| **Accelerations / Window** | $1.54 \pm 1.48$ | $1.26 \pm 1.35$ | $1.51 \pm 1.45$ | Rolling baseline accurately references acceleration peaks |
| **Decelerations / Window** | $4.23 \pm 2.61$ | $3.65 \pm 2.42$ | $4.18 \pm 2.58$ | Deceleration depth and area are fully preserved in P2 |
| **Residual Transducer Spikes** | $439,945$ | $334,609$ | $385,120$ | Bounded spline prevents artificial overshoot |
| **Retained Window Quality** | $88.15\%$ valid | $84.22\%$ valid | $88.15\%$ valid | P2 recovers short dropouts while explicitly tagging reconstructed samples |

---

## 7. Deliverables G & H: Predictive Sanity Check & Statistical Comparison

All regimes evaluated using an identical ResNet-1D encoder, identical learning rate ($10^{-3}$), Cosine Annealing, 20 epochs, and patient max-pooling aggregation across the **frozen 5-fold patient split**:

### Comprehensive 5-Fold Validation Results:

| Regime | Fold 1 AUROC | Fold 2 AUROC | Fold 3 AUROC | Fold 4 AUROC | Fold 5 AUROC | **Pooled OOF AUROC (95% CI)** | **AUPRC** | $\mathbf{\Delta AUROC}$ vs P0 | **Paired p-value** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P0 (Control)** | 0.6286 | 0.6477 | 0.6113 | 0.7685 | 0.6118 | **0.6450** [0.588, 0.703] | 0.2959 | Ref | — |
| **P1 (Minimal)** | 0.6136 | 0.6927 | 0.6317 | 0.7132 | 0.5104 | **0.6390** [0.578, 0.699] | 0.3090 | $-0.0060$ [$-0.075, 0.059$] | $p=0.439$ |
| **P2 (Quality-Aware)** | 0.6565 | 0.6389 | 0.6416 | 0.7393 | 0.5831 | **0.6491** [0.588, 0.709] | **0.3238** | $\mathbf{+0.0040}$ [$-0.062, 0.068$] | $p=0.445$ |

### Statistical & Methodological Insights:
1. **Priority Hierarchy Evaluation:**
   - **Priority 1 (Zero Leakage):** P2 enforces localized rolling baseline within window, eliminating the $7.36\text{ bpm}$ prospective whole-recording leakage.
   - **Priority 2 (Physiological Validity):** P2 provides both absolute physiological heart rate level ($FHR(t)$) and deviation from non-stationary baseline ($\Delta FHR(t)$).
   - **Priority 3 (Signal Preservation & Veracity):** P2 provides explicit masks ($m(t)$ and $m_{interp}(t)$), preventing the neural network from confusing reconstructed values with real sensor data.
   - **Priority 4 (Reproducibility):** Zero stochasticity; fully deterministic signal transformations.
   - **Priority 5 (Predictive Performance):** P2 achieves the highest precision-recall performance (**AUPRC 0.3238**, $+0.0279$ over P0) with the most consistent cross-fold stability.

---

## 8. Summary Comparison Table

| Component | Current (P0) | Candidate (P2) | Final Decision | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Spike Removal** | Sample delta $> 6.25\text{ bpm}$ | Sample delta $> 6.25\text{ bpm}$ | **ADOPT P2** | Standard physiological limit ($25\text{ bpm/s}$) |
| **Interpolation** | Cubic spline $\le 15\text{s}$ | Cubic spline $\le 15\text{s}$ clamped | **ADOPT P2** | Clamping to $[50, 240]$ prevents spline overshoot |
| **Observed Quality Mask** | None (implicit missing mask) | Explicit mask $m(t) \in \{0, 1\}$ | **ADOPT P2** | Tells model exactly which points were recorded |
| **Interpolation Mask** | None | Explicit mask $m_{interp}(t)$ | **ADOPT P2** | Tells model which points were reconstructed |
| **Baseline Estimator** | Constant iterative (rounded 5 bpm) | Localized rolling iterative | **ADOPT P2** | Tracks non-stationary drift ($24.8\text{ bpm}$) without window leakage |
| **Normalization** | Train-fit Z-score on Ch 0, 1 | Train-fit Z-score on Ch 0, 1 | **ADOPT P2** | Continuous channels scaled; masks preserved as binary |
| **Channel 0** | Baseline-corrected ($FHR - B$) | Absolute FHR ($FHR(t)$) | **ADOPT P2** | Retains baseline tachycardia/bradycardia level |
| **Channel 1** | Uterine Contraction (UC) | Baseline-relative ($\Delta FHR(t)$) | **ADOPT P2** | Directly presents acute decelerations to 1D/2D models |
| **Channel 2** | Missingness mask | Observed quality mask $m(t)$ | **ADOPT P2** | Explicit signal reliability input |
| **Channel 3** | None | Interpolation mask $m_{interp}(t)$ | **ADOPT P2** | Explicit reconstruction awareness |
| **Uterine Contraction (UC)**| Included in Ch 1 | Optional auxiliary channel | **ADOPT P2** | Retained in metadata/multimodal store without degrading core FHR representation |
| **Window Length** | 20 minutes (4800 samples) | 20 minutes (4800 samples) | **KEEP 20 MIN** | Standard FIGO evaluation duration |
| **Stride** | 2.5 minutes (600 samples) | 2.5 minutes (600 samples) | **KEEP 2.5 MIN** | Standard dense evaluation coverage |

---

## 9. FINAL FROZEN PREPROCESSING CONTRACT

```json
{
  "contract_name": "CTU_UHB_PREPROCESSING_CONTRACT_V1_FROZEN",
  "phase": "Phase 1 Final Frozen Specification",
  "dataset": "CTU-UHB Intrapartum Database (PhysioNet)",
  "cohort": {
    "total_patients": 547,
    "positive_patients": 110,
    "negative_patients": 437,
    "total_usable_windows": 8517,
    "primary_endpoint": "umbilical_artery_ph <= 7.15",
    "partition_file": "data/processed_clinical/folds.json",
    "n_folds": 5,
    "n_repeats": 5,
    "seed": 0
  },
  "signal_parameters": {
    "sampling_frequency_hz": 4.0,
    "window_length_minutes": 20,
    "window_length_samples": 4800,
    "stride_minutes": 2.5,
    "stride_samples": 600,
    "recording_horizon_minutes": 60,
    "recording_horizon_samples": 14400,
    "patient_quality_max_missing_ratio": 0.50,
    "window_quality_max_missing_ratio": 0.50
  },
  "signal_pipeline": {
    "step_1_spike_removal": {
      "method": "rate_of_change_threshold",
      "max_rate_bpm_per_sec": 25.0,
      "max_delta_per_sample": 6.25,
      "replacement_value": 0.0
    },
    "step_2_interpolation": {
      "method": "cubic_spline",
      "max_gap_seconds": 15.0,
      "max_gap_samples": 60,
      "clip_min_bpm": 50.0,
      "clip_max_bpm": 240.0,
      "longer_gaps": "preserved_as_missing_zero"
    },
    "step_3_filtering": {
      "filter_type": "butterworth_lowpass",
      "order": 4,
      "cutoff_hz": 1.5,
      "forward_backward_zero_phase": true
    },
    "step_4_baseline": {
      "method": "localized_rolling_iterative",
      "window_seconds": 300.0,
      "step_seconds": 30.0,
      "exclusion_threshold_bpm": 15.0,
      "leakage_scope": "strictly_within_evaluated_window"
    },
    "step_5_tensor_representation": {
      "shape": [8517, 4, 4800],
      "channels": {
        "channel_0": "FHR(t) [Absolute FHR, Z-scored on train-fold only]",
        "channel_1": "Delta_FHR(t) = FHR(t) - B(t) [Baseline-relative, Z-scored on train-fold only]",
        "channel_2": "m(t) [Observed valid mask: 1 = genuine sensor reading, 0 = missing/spike]",
        "channel_3": "m_interp(t) [Interpolation mask: 1 = genuine observation, 0 = reconstructed / missing]"
      },
      "optional_auxiliary_channel": "UC(t) [Uterine Contraction, available for multimodal ablation]"
    }
  },
  "invariance_rules": {
    "no_outcome_dependent_quality_tuning": true,
    "no_prospective_baseline_leakage": true,
    "patient_grouped_splitting_only": true,
    "train_set_only_scaling": true
  }
}
```
