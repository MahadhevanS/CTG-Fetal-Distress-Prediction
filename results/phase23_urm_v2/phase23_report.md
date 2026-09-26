# Phase 23 — URM-v2 composite: results

Protocol: `docs/phase23_urm_v2_composite_protocol.md` (frozen before any code). Scripts: `scripts/phase23_urm_v2_composite.py`, `scripts/phase23b_matched_far.py` (post-hoc).
Composite V2E = window score ½(P6 + G-A with ridge α = 1000 fixed) + soft-label TAM-architecture pooler + parity fusion + per-fold recalibration.
Evaluation: all 547 patients, canonical split + 10 fresh resplits (seeds 121–220, never used before). Baseline B0 = locked P6 through the identical harness.
The 83-patient test partition (opened twice already) was **not used**.

## Pre-registered verdict: **EXPLORATORY NO SUPPORT** (fails only the lead-time guard L)
| Criterion | Result | Met |
|---|---|---|
| D1 mean ΔM over 10 fresh resplits ≥ +0.005 and ΔM > 0 in ≥ 8/10 | **+0.0108, 10/10 positive** (range +.0079 to +.0162) | yes |
| D2 mean per-horizon Δ ≥ −0.010 at every horizon | +0.0132 / +0.0100 / +0.0110 / +0.0087 (0 / 10 / 20 / 30 min) | yes |
| L lead-time guard (canonical **and** fresh mean): Δ share warned ≥ 20 min ≥ −0.03, Δ median lead ≥ −2.5 min, Δ sensitivity ≥ −0.03, Δ false-alert rate ≤ +0.03 | share ≥ 20 min: **−0.084** canonical, **−0.044** fresh mean; median lead −2.5 / −1.0 min; sensitivity 0.0 / −0.014; false-alert rate −0.032 / −0.049 (better) | **no** |
| E recalibration "works" (slope in [0.80, 1.25], Brier no worse, AUROC drop ≤ 0.005) | slope **1.31 / 1.29** (just above 1.25); Brier −0.006 / −0.005 (better); AUROC −0.0014 / −0.0016 | **no** (by the letter) |

Statistics: canonical 547 ΔM = +0.0012 (95% CI −0.009, +0.011; p .83); fresh-resplit pooled p = .006 (all resplits reuse the same 547 patients, so this is split variability, not new patients).

## Discrimination — V2E vs B0 (mean AUROC over horizons; fresh-resplit means)
| Horizon | B0 | V2E | Δ |
|---|---|---|---|
| Delivery | 0.7180 | 0.7313 | +0.0132 |
| ≥10 min | 0.6731 | 0.6831 | +0.0100 |
| ≥20 min | 0.6514 | 0.6625 | +0.0110 |
| ≥30 min | 0.6412 | 0.6499 | +0.0087 |
| **M** | **0.6709** | **0.6817** | **+0.0108** (canonical +0.0012) |

## Ablation (descriptive, no decision weight): what each piece does
| Arm | Fresh-mean M | ΔM vs B0 | positive / 10 | Canonical ΔM |
|---|---|---|---|---|
| C — G-A window (ridge α = 1000 fixed), standard pooler | **0.6891** | **+0.0182** | 10/10 (min +0.0117) | +0.0069 |
| B — ½ (P6 + G-A) window, standard pooler | 0.6820 | +0.0110 | 10/10 | +0.0043 |
| A — P6 window, **soft-label pooler** | 0.6710 | +0.0000 | 4/10 | +0.0001 |
| V2 — B + A | 0.6833 | +0.0123 | 10/10 | +0.0026 |
| V2E — V2 + recalibration | 0.6817 | +0.0108 | 10/10 | +0.0012 |
- The **soft-label pooler (idea A) contributes nothing.** The ensemble (B) adds about +0.011; the fixed-α G-A window alone (C) adds the most, +0.018, better than the composite.
- Averaging with P6 (B) dilutes G-A's gain rather than adding to it.
- The fixed α = 1000 was chosen after seeing earlier canonical results, but these resplits are new seeds, so C's advantage is out-of-sample with respect to the splits (not with respect to patients).

## Operating point (80% target sensitivity, training-only threshold, patient-level "ever alerted")
| | Sensitivity | False-alert rate | PPV | Median lead | ≥ 20 min | ≥ 30 min |
|---|---|---|---|---|---|---|
| B0 canonical / fresh mean | 86.4% / 86.7% | 62.5% / 64.7% | 25.8% / 25.2% | 40.0 / 40.0 | 83.2% / 83.7% | 70.5% / 71.3% |
| V2E canonical / fresh mean | 86.4% / 85.4% | **59.3% / 59.7%** | 26.8% / 26.5% | 37.5 / 39.0 | 74.7% / 79.3% | 64.2% / 66.7% |
| C canonical / fresh mean | 86.4% / 85.5% | **58.8% / 56.7%** | 27.0% / 27.5% | 30.0 / 34.4 | 66.3% / 72.2% | 52.6% / 59.4% |
Under a fixed-sensitivity threshold, the G-A-type models raise **fewer false alerts (−3 to −8 points)** but warn **later**; C most, V2E less.

## Post-hoc check (not pre-registered, descriptive): matched false-alert rate
Threshold set so 62.5% of training negatives are ever alerted (B0's canonical rate), then sensitivity and lead time are compared (`matched_far.csv`):
| | Sensitivity | False-alert rate | Median lead | ≥ 20 min | ≥ 30 min |
|---|---|---|---|---|---|
| B0 canonical / fresh mean | 87.3% / 86.1% | 63.2% / 62.9% | 40.0 / 39.9 | 83.3% / 81.7% | 71.9% / 68.7% |
| V2 canonical / fresh mean | 89.1% / 87.5% | 61.1% / 62.7% | 40.0 / 39.8 | 82.7% / 81.5% | 71.4% / 68.4% |
| C canonical / fresh mean | 87.3% / 88.8% | 63.2% / 62.4% | 33.8 / 37.5 | 71.9% / 78.5% | 58.3% / 65.5% |
At the same false-alert rate the composite V2 **keeps B0's lead times** and gains about +1.4 points of sensitivity (fresh mean); C gains +2.7 points of sensitivity but still warns somewhat later.
So V2E's lead-time shortfall in the pre-registered check is mostly an operating-point effect of its better specificity, not a loss of early-warning ability. This reading is post-hoc and does not change the pre-registered label.

## Calibration (delivery horizon; V2E is recalibrated, others are raw)
| | Calibration slope | Brier | ECE (10 bins) |
|---|---|---|---|
| B0 canonical / fresh mean | 2.63 / 2.43 | 0.149 / 0.150 | 0.044 / 0.044 |
| V2 (before recalibration) | 2.77 / 2.62 | 0.149 / 0.149 | 0.046 / 0.046 |
| **V2E** | **1.31 / 1.29** | **0.143 / 0.144** | **0.037 / 0.039** |
Recalibration halves the compression (slope 2.7 → 1.3) and improves Brier by ≈ 0.006, at a cost of ≈ 0.0015 AUROC. It narrowly misses the pre-registered slope window (≤ 1.25).

## What to conclude
- **The composite is not supported under its own pre-registered rule**, only because of the lead-time guard; discrimination gains are consistent (10/10 fresh resplits, +0.011) but small and rest on the same 547 patients.
- **Two of the four ideas do the work:** the graded-target window model with fixed regularisation (C, +0.018) and, separately, recalibration (probability quality). The soft-label pooler is useless and ensembling with P6 gives back part of C's gain.
- The clearest practical gain is **calibration**: the final probabilities become much more usable (slope 1.3, Brier −0.006).
- Neither the composite nor C has any independent validation; the frozen URM remains the reference. The natural follow-up is a new pre-registered candidate of the simplest kind (C plus recalibration) evaluated on independent data.
