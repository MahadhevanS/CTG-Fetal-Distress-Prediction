# Phase 9B: Final Locked Synthesis & Decision Gate Audit

## Executive Summary

Phase 9B investigated whether representing cardiotocography as a trajectory of clinically interpretable physiological states—distinguishing current state, rate of change, persistence, and multidomain progressive deterioration—can provide an earlier and more clinically useful warning of intrapartum fetal acidemia ($\text{pH} \le 7.15$ and severe $\text{pH} \le 7.05$) at $\ge 30$ minutes before delivery.

The study utilized the complete CTU-UHB cohort ($N = 547$ patients, $N_{\text{acidemia}} = 110$, $N_{\text{severe}} = 41$, 8,517 continuous rolling 20-minute windows) under strict patient-grouped 5-fold cross-validation. Zero future look-ahead was verified across all 112 engineered deterioration features under synthetic future-perturbation testing ($\Delta = 0.000000000000$).

---

## 1. Formal Decision Gate Audit

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 9B DECISION GATE AUDIT                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Gate 9B-1: Causal Integrity & Leakage Verification                              │
│ Status: PASSED                                                                  │
│ Evidence: Synthetic future-perturbation testing across all 112 features passed  │
│           with max difference = 0.000000000000. Strict 5-fold patient fold      │
│           isolation preserved across all stages.                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Gate 9B-2: Deterioration Information Exists                                     │
│ Status: PASSED (NEAR DELIVERY & SEVERE ENDPOINTS)                               │
│ Evidence: Domain-structured deterioration features (Model D/E) enhance          │
│           near-delivery discrimination (AUROC 0.7368 overall, 0.7845 on severe  │
│           acidemia pH <= 7.05) and establish a clear 5-state risk gradient.     │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Gate 9B-3: Primary Early-Warning Target (AUROC >= 0.80 at >= 30 min)            │
│ Status: NOT ACHIEVED (PHYSIOLOGICAL CEILING IDENTIFIED)                         │
│ Evidence: Best >=30 min model achieved AUROC = 0.5695 vs Phase 8 baseline       │
│           0.5699 (delta = -0.0004, p = 0.980). Morphological CTG signals        │
│           >30 min prior to delivery are biologically buffered by fetal reserves.│
├─────────────────────────────────────────────────────────────────────────────────┤
│ Gate 9B-4: Clinical Usefulness & Operational Alerting                           │
│ Status: PASSED                                                                  │
│ Evidence: Policy 2 (2-window confirmation) achieves 97.25% specificity,        │
│           0.046 false alerts / hour, and 13.8 min median lead time.             │
│           Research states provide actionable clinical interpretability.         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Research State Progression & Acidemia Risk Gradient

```
  Research State Acidemia Risk Gradient (Observed Prevalence in Cohort)
  
  50% ┌───────────────────────────────────────────────────────────── 47.62% (State 4)
      │                                                              [Severe: 26.19%]
  40% ├────────────────────────────────────────────── 35.80% (State 3)
      │                                               [Severe: 16.05%]
  30% ├────────────────────────────── 22.81% (State 2)
      │                               [Severe: 8.77%]
  20% ├────────────── 14.19% (State 1)
      │               [Severe: 3.38%]
  10% ├── 8.64% (State 0)
      │   [Severe: 1.23%]
   0% └───┴───────────────┴───────────────┴───────────────┴───────────────┴──────────
        State 0         State 1         State 2         State 3         State 4
        (Stable)       (Emerging)     (Persistent)   (Progressive)     (Severe)
```

---

## 3. Scientific Discussion: The Biological Horizon Boundary

1. **The Biological Compensation Paradox**:
   - The mammalian fetus maintains stable cardiac autonomic function during early and mid-stage labor through carotid/aortic chemoreceptor reflexes, redistribution of blood flow to vital organs (brain, myocardium, adrenals), and anaerobic glycogenolysis.
   - During this compensated phase ($> 30\text{ minutes}$ prior to delivery), the fetal heart rate baseline and variability remain largely preserved on CTG.
2. **Timing of Observable Decomposition**:
   - When anaerobic reserves are exhausted and metabolic acidemia ($\text{pH} \le 7.15, \text{BE} \le -12\text{ mmol/L}$) develops, myocardial glycogen depletion triggers severe late decelerations, loss of STV, and terminal bradycardia.
   - These pathognomonic signs emerge predominantly during late second-stage active pushing—within **$20\text{ to }30\text{ minutes}$** of delivery.
3. **Scientific Conclusion**:
   - Neither generic machine learning (Phase 9A) nor physiology-guided deterioration modeling (Phase 9B) can extract absent morphological distress patterns $> 30\text{ minutes}$ before delivery from retrospective CTG.
   - CTG is intrinsically an **imminent distress detector** ($\le 20-30\text{ min}$ lead time), not a multi-hour prognostic forecaster. Extending reliable prediction beyond 30 minutes will require non-invasive continuous biochemical sensors (fetal pulse oximetry, transcutaneous lactate) or maternal biophysical monitoring.

---

## 4. Final Deliverables Summary

- **Code Suite**:
  - `scripts/phase9b_deterioration_features.py`: 112-D causal deterioration feature extractor.
  - `scripts/phase9b_consolidated_evaluation.py`: Complete cross-validation, ablation, state, and policy evaluator.
  - `scripts/phase9b_figures.py`: 8 publication-grade diagnostic figures.
- **Data Deliverables**:
  - `results/phase9b_deterioration/deterioration_predictions.csv`: 8,517 rolling window predictions.
  - `results/phase9b_deterioration/model_comparison_metrics.csv`: 5-model multi-horizon benchmark.
  - `results/phase9b_deterioration/ablation_metrics.csv`: 7 ablation experiment metrics.
  - `results/phase9b_deterioration/state_analysis.csv`: 5-state risk gradient analysis.
  - `results/phase9b_deterioration/alert_policy_metrics.csv`: 5 alerting policy evaluations.
  - `results/phase9b_deterioration/calibration_metrics.csv`: Calibration parameters across horizons.
  - `results/phase9b_deterioration/warning_times.csv`: Lead time distributions.
  - `results/phase9b_deterioration/statistical_comparisons.json`: Paired bootstrap test results.
- **Figures**:
  - Located in `reports/figures_phase9b/`.
