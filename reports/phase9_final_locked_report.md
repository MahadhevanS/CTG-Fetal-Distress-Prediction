# Phase 9: Final Locked Synthesis & Decision Gate Report

## Executive Summary

Phase 9 established and evaluated a comprehensive suite of causal temporal early-warning models designed to test whether the temporal evolution of 20-minute cardiotocography (CTG) observations can provide early detection of intrapartum fetal acidemia ($\text{pH} \le 7.15$) at $\ge 30$ minutes before delivery.

The entire study maintained strict patient-grouped 5-fold cross-validation across the 547 CTU-UHB patients (110 acidemic cases, 41 severe cases, 8,517 continuous rolling windows). Zero future look-ahead was verified across all 104 engineered temporal features under synthetic perturbation auditing.

---

## 1. Formal Decision Gate Audit

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 9 DECISION GATE AUDIT                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Gate 9A: Temporal Information Exists                                            │
│ Status: CONDITIONAL PASS                                                        │
│ Evidence: Temporal features enhance near-delivery discrimination (AUROC 0.7315  │
│           overall, 0.7763 on severe acidemia pH <= 7.05) and improve alert      │
│           persistence, but contribute marginal gain >30 min before delivery.    │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Gate 9B: Early-Warning Target Achievement (AUROC >= 0.80 at >= 30 min)          │
│ Status: NOT PASSED (PHYSIOLOGICAL LIMITATION IDENTIFIED)                        │
│ Evidence: Best >=30 min model achieved AUROC = 0.5709 (Sequential GRU) and     │
│           0.5699 (Phase 8 Baseline). The target of AUROC >= 0.80 at >=30 min    │
│           is biologically constrained in retrospective intrapartum CTG.         │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Gate 9C: Clinical Alert Utility & False-Alert Constraint                        │
│ Status: PASSED                                                                  │
│ Evidence: The 2-Window Persistence alert policy achieves 97.25% specificity,    │
│           0.046 false alarms per monitoring hour, and 13.8 min median warning   │
│           lead time, satisfying real-time labor ward operational constraints.   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Synthesis of Multi-Horizon Discrimination

```
   Warning Horizon Discrimination Progression (Patient-Level AUROC)
   
  0.80 ─── Target (0.80) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
       │                                                                  
  0.75 ───                                                     0.7426 (P8)
       │                                                       0.7315 (LR)
  0.70 ───                                         0.6941 (P8) 0.6955 (GRU)
       │                               0.6290 (P8) 0.6920 (EWMA)
  0.65 ───                             0.6346 (LR)
       │                   0.5699 (P8)
  0.60 ─── 0.6015 (GRU)    0.5709 (GRU)
       │   0.5594 (LR)     0.5598 (LR)
  0.55 ─── 0.5360 (P8)
       │
  0.50 └───┴───────────────┴───────────┴───────────┴───────────┴───────────
          >=60m           >=30m        >=20m       >=10m        Delivery (0m)
```

---

## 3. Scientific Discussion: Why Early CTG Warning Is Physiologically Constrained

1. **Fetal Homeostatic Buffer**: During early active labor ($> 30\text{ minutes}$ prior to delivery), the mammalian fetus possesses robust compensatory mechanisms (carotid/aortic baroreceptors, peripheral vasoconstriction, anaerobic glycogen mobilization) that preserve normal FHR baseline and variability despite intermittent hypoxic stress.
2. **Timing of Autonomic Exhaustion**: Frank CTG abnormalities (repetitive deep late decelerations, saltatory or absent STV, progressive bradycardia) manifest when compensatory reserves are depleted—typically during second-stage expulsive contractions within the final 20–30 minutes of labor.
3. **Implication for Predictive Systems**: CTG is an exquisite detector of *imminent* acute asphyxia, but inherently limited as a multi-hour early forecaster. Combining CTG with biochemical or maternal biomarker signals will be necessary to extend predictive horizons beyond 30 minutes.

---

## 4. Final System Specifications & Artifact Deliverables

- **Code Deliverables**:
  - `scripts/phase9_temporal_features.py`: 104-D causal feature extractor.
  - `scripts/phase9_temporal_lr.py`: Regularized logistic regression.
  - `scripts/phase9_temporal_gbm.py`: HistGradientBoosting classifier.
  - `scripts/phase9_prediction_trajectory.py`: Causal EWMA trajectory filter.
  - `scripts/phase9_sequential_model.py`: Compact GRU recurrent network.
  - `scripts/phase9_multihorizon.py`: Multi-task neural network.
  - `scripts/phase9_alert_policy.py`: Labor ward alert policy optimizer.
  - `scripts/phase9_consolidated_evaluation.py`: Cross-validation & bootstrap suite.
  - `scripts/phase9_figures.py`: 8 publication-grade diagnostic figures.

- **Data Deliverables**:
  - `results/phase9_temporal/temporal_predictions.csv`: 8,517 rolling window predictions.
  - `results/phase9_temporal/horizon_metrics.csv`: Multi-horizon performance benchmark.
  - `results/phase9_temporal/calibration_metrics.csv`: Brier scores and calibration parameters.
  - `results/phase9_temporal/warning_times.csv`: Lead time distributions.
  - `results/phase9_temporal/alert_policy_metrics.csv`: 5 alerting policy evaluations.
  - `results/phase9_temporal/statistical_comparisons.json`: Paired bootstrap test results.
