# Phase 22 — G-A: trained model and full evaluation

Spec: `docs/phase22_ga_final_evaluation_spec.md` (frozen before any metric was computed). Script: `scripts/phase22_ga_final_evaluation.py`.
G-A = URM pipeline whose window model is a ridge regression on standardised −pH (nested-CV α ∈ {10,100,1000}) + Platt step, TAM-architecture pooler (3 seeds), parity fusion.
**Exploratory model. It failed the pre-registered adoption rule (Phase 21); the frozen URM remains the locked reference. Nothing below promotes G-A.**
Baseline B0 = the locked P6 recipe through the identical harness (reproduces the frozen URM: 0.7334 / 0.6957 / 0.6677 / 0.6646).

## 1. Discrimination — canonical 5-fold CV, all 547 patients (110 positives)
| Horizon | G-A AUROC (95% CI) | B0 AUROC (95% CI) | Δ (G-A − B0), p | G-A AUPRC | B0 AUPRC |
|---|---|---|---|---|---|
| Delivery | **0.7368** (0.684–0.787) | 0.7334 (0.679–0.786) | +0.0034, p .72 | 0.456 | 0.443 |
| ≥10 min | **0.6986** (0.644–0.754) | 0.6957 (0.641–0.750) | +0.0029, p .77 | 0.391 | 0.380 |
| ≥20 min | **0.6737** (0.617–0.730) | 0.6677 (0.614–0.725) | +0.0060, p .61 | 0.360 | 0.348 |
| ≥30 min | 0.6574 (0.599–0.716) | 0.6646 (0.606–0.723) | −0.0072, p .54 | 0.341 | 0.342 |
| **Mean (M)** | **0.6916** | 0.6903 | **+0.0013** (CI −0.015, +0.018), p .89 | | |

DeLong at delivery: p = .71. Frozen URM (Phase 18) for reference: 0.7335 / 0.6921 / 0.6638 / 0.6631.
AUROC vs lead time (0, 5, …, 40 min): G-A .737 .732 .699 .679 .674 .670 .657 .679 .672; B0 .733 .724 .696 .673 .668 .669 .665 .679 .679.

## 2. Held-out test partition — trained on 464, scored on 83 (17 positives); reproduces Phase 21b exactly
| Horizon | G-A AUROC (95% CI) | B0 AUROC (95% CI) | AUPRC G-A / B0 |
|---|---|---|---|
| Delivery | 0.773 (0.653–0.879) | 0.753 (0.629–0.869) | .434 / .416 |
| ≥10 | 0.798 (0.692–0.891) | 0.718 (0.589–0.836) | .435 / .386 |
| ≥20 | 0.849 (0.754–0.928) | 0.781 (0.653–0.886) | .598 / .525 |
| ≥30 | 0.809 (0.691–0.909) | 0.781 (0.645–0.895) | .585 / .510 |
| **M** | **0.807** | 0.758 | ΔM **+0.049** (CI +0.0004, +0.109), p .05 |

## 3. Robustness
- **10 independent resplits of all 547** (ΔM, G-A − B0): +.0056 +.0110 +.0179 +.0056 +.0185 (screening seeds) and +.0117 +.0099 +.0189 +.0142 +.0112 (fresh seeds).
  **10/10 positive, mean +0.0124**, p ≤ .05 in 3/10. The canonical split (+0.0013) is the weakest of the 11 splits.
- Pooler seed spread on the canonical split: G-A 0.6915–0.6918, B0 0.6871–0.6904 (stable).
- Ridge α held fixed (descriptive, post-hoc): M = 0.6971 / 0.6971 / 0.6972 for α = 10 / 100 / 1000, versus 0.6916 with the nested selection (selection cost ≈ 0.005 on this split).

## 4. Where the gain comes from (ablation, CV 547, mean M; per-horizon in brackets)
| Component | G-A | B0 | Control (binary label + Platt) |
|---|---|---|---|
| Parity only | — | 0.5755 | — |
| Latest window only | **0.6577** (.693/.698/.656/.584) | 0.6449 (.687/.686/.624/.583) | 0.6449 |
| TAM pooled window scores only | **0.6620** | 0.6471 | 0.6473 |
| Fused (URM form) | 0.6916 | 0.6903 | 0.6883 |

The graded target improves the CTG-only scores by ≈ +0.013–0.015 in M (delivery, ≥10 and ≥20 min; nothing at ≥30 min), but after fusion with parity most of that
disappears on the canonical split. The control (recalibration only) does nothing, so the window-level gain is attributable to the graded target.

## 5. Calibration (delivery horizon; no recalibration applied)
| | Brier | Mean prediction vs prevalence | Cal. intercept / slope | ECE (10 bins) |
|---|---|---|---|---|
| G-A, CV 547 | 0.1479 | 0.213 vs 0.201 | 2.03 / 2.73 | 0.054 |
| B0, CV 547 | 0.1488 | 0.212 vs 0.201 | 1.93 / 2.63 | 0.044 |
| G-A, test (83) | 0.1486 | 0.223 vs 0.205 | 1.77 / 2.63 | 0.080 |

Both models are **under-dispersed**: a slope of ≈ 2.7 means scores are too compressed (bin 0.1–0.2 observes 7.8%, bin 0.3–0.4 observes 59%). The overall level is right;
what they offer is ranking, not trustworthy probability values. (Brier of a prevalence-only model: 0.160.)

## 6. Operating point — 80% target sensitivity, training-only threshold, patient-level "ever alerted" (CV 547)
| | Sensitivity | False-alert rate | PPV | Median lead | ≥10 min | ≥20 min | ≥30 min |
|---|---|---|---|---|---|---|---|
| G-A | 85.5% (78–92) | **60.6%** (56–65) | 26.2% | 32.5 min | 84.0% | 71.3% | 55.3% |
| B0 | 86.4% (80–93) | 62.5% (58–67) | 25.8% | **40.0 min** | 91.6% | **83.2%** | **70.5%** |

Frozen URM (Phase 18): 86.4% / 62.5% / 40 min / 81.1% ≥20 min / 69.5% ≥30 min.
G-A raises slightly fewer false alerts (−1.9 points) but **warns later**, with overlapping CIs. Sensitivity at fixed specificity (delivery): G-A 52.7% at 80% spec and 33.6% at 90%; B0 53.6% / 34.5%.

## 7. Clinical utility — decision curve at delivery (net benefit)
Threshold 0.15 / 0.20 / 0.25 / 0.30 / 0.35 / 0.40 — G-A: .078 / .059 / .045 / .031 / .010 / .002; B0: .076 / .054 / .051 / .026 / .008 / .000; treat-all: .060 / .001 / −.065 / −.141 / −.229 / −.332.
Both beat treat-all above ≈ 0.15 and are indistinguishable from each other.

## 8. Subgroups (delivery-time AUROC, G-A vs B0; wide intervals, descriptive)
Vaginal 0.725 vs 0.723 (n 504) · Caesarean 0.826 vs 0.805 (n 43, 15 positives) · Primiparous 0.701 vs 0.699 · Multiparous 0.712 vs 0.682 (n 174, 19 positives) ·
Last window in 1st stage 0.739 vs 0.742 · **2nd stage 0.648 vs 0.639 (weakest)** · Long recordings 0.750 vs 0.750 · Short 0.698 vs 0.672 (n 81).
Severity: pH ≤ 7.05 vs normal **0.799** vs 0.793 (≥30 min 0.765 vs 0.749) · BDecf ≥ 8 vs normal 0.825 vs 0.824 · borderline pH 7.10–7.15 vs normal 0.744 vs 0.757.
No subgroup is distinguishable from the overall result; the model is strongest on clear acidemia and weakest on borderline cases and second-stage recordings.

## 9. Interpretability
Top ridge features (same as P6): accelerations (`raw_3`), baseline severity, LTV (`raw_2`), decel burden (`raw_10`), decel area (`raw_9`), FHR–UC coupling (`raw_18`, negative), EWMA risk,
variability severity, baseline, state persistence (negative). All 10 signs agree with the P6 logistic model; window-score rank correlation with P6 = 0.90.
G-A is a strongly shrunk re-weighting of the same model, not a different one.

## 10. Trained model (all 547 patients) and integrity
Artefacts: `results/phase22_ga_final/frozen_model/` (`model.json`, `pooler_seed42/43/44.pt`) and the standalone scorer `scripts/phase22_ga_predict.py`.
Standalone scorer vs in-memory pipeline on all 547 patients: max difference 2.8e-17 (PASS). In-sample AUROC (OPTIMISTIC, not a performance estimate): 0.773 / 0.728 / 0.706 / 0.694.
No held-out data exist for this final fit; the honest estimates are Sections 1–3.

## 11. What to conclude
- **Best single-number estimate of G-A vs baseline: about +0.01 in M** (canonical 547 +0.001; resplits +0.006 to +0.019; test partition +0.049 on 83 patients). The direction is positive in 10/10 resplits and on the test partition.
- **Not established:** the improvement is not statistically supported on 547 (p .89 canonical; 3/10 resplits ≤ .05; test lower bound +0.0004), and the resplits reuse the same patients.
- **Trade-offs:** operationally G-A warns **later** than the baseline/URM; both are badly calibrated (compressed scores); neither has been validated outside CTU-UHB.
- The frozen URM stays the reference. G-A's window-level gain (+0.013) mostly disappears after parity fusion on the canonical split; external validation is what could settle whether the remaining difference is real.
