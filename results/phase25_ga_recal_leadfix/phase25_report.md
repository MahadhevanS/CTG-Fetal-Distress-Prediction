# Phase 25 — Candidate D2: does a lead-time-constrained fusion weight fix Candidate D?

Protocol: `docs/phase25_ga_recal_leadfix_protocol.md` (frozen before code, after a read-only diagnostic). Script: `scripts/phase25_ga_recal_leadfix.py`.
D2 = Candidate D (G-A window, ridge α = 1000 fixed, standard pooler, recalibration) with **one change**: the fusion weight α is chosen per fold to maximise training AUROC
**subject to a training-only constraint that in-sample median lead time is ≥ 35 minutes**, instead of by unconstrained AUROC-argmax. Evaluated on canonical + 10 fresh
resplits (seeds 501–627, never used before). No independent cohort available (unchanged caveat from Phase 24).

## Verdict: **EXPLORATORY NO SUPPORT — but for a different, narrower reason than Candidate D**
| Criterion | Candidate D (Phase 24) | Candidate D2 (this phase) |
|---|---|---|
| D1 discrimination (fresh mean ΔM ≥ +0.005, ≥8/10 positive) | pass (+0.0108) | **pass, and larger (+0.0152)**, 10/10 positive |
| D2 per-horizon floor | pass | pass |
| **L lead-time guard (matched false-alert rate)** | **FAILED** (canonical ge20 −0.125, fresh mean −0.059) | **PASSED** (canonical ge20 **+0.055**, fresh mean −0.020; median lead unchanged canonical, −2.1 min fresh) |
| E calibration (slope ∈ [0.80,1.25] both) | failed (slope 1.32 canonical) | failed, **narrower miss** (slope 1.29 canonical, 1.18 fresh — inside the band) |

**The fix worked on the problem it targeted.** D2 passes the lead-time guard that killed Candidate D, with a *larger* discrimination gain, and only narrowly fails calibration
(canonical slope 1.29 vs the 1.25 ceiling; the fresh-resplit mean, 1.18, is inside the band). It is not labelled "supported" only because of that one narrow miss.

## Discrimination
| Horizon | B0 fresh mean | D2 fresh mean | Δ |
|---|---|---|---|
| Delivery | 0.7142 | 0.7258 | +0.0116 |
| ≥10 min | 0.6723 | 0.6900 | +0.0178 |
| ≥20 min | 0.6518 | 0.6680 | +0.0163 |
| ≥30 min | 0.6437 | 0.6590 | +0.0153 |
| **M** | 0.6742 | **0.6894** | **+0.0152** |
Fresh-resplit pooled p = **.043**. Canonical split: B0 0.6903 → D2 0.6849 (**ΔM = −0.0054, p = .64**) — the canonical split is the one exception, discussed below.

## Lead time at matched false-alert rate (the guard D failed)
| | Sensitivity | FAR | Median lead | ≥10 min | ≥20 min | ≥30 min |
|---|---|---|---|---|---|---|
| B0 canonical | 87.3% | 63.2% | 40.0 min | 92.7% | 83.3% | 71.9% |
| **D2 canonical** | 89.1% | 66.1% | 40.0 min | **95.9%** | **88.8%** | **76.5%** |
| B0 fresh mean | 84.8% | 62.6% | 39.5 min | 90.7% | 81.8% | 69.1% |
| **D2 fresh mean** | 87.2% | 60.4% | 37.4 min | 90.0% | 79.8% | 67.5% |
On the canonical split D2 is better than B0 on every one of these six numbers (more sensitive, though at a slightly higher FAR, and *earlier or equal* on every lead-time measure).
On the fresh-resplit mean it trades a small amount of lead time (−2.1 min median, −2.0 points ≥20min, −1.6 points ≥30min) for better sensitivity and a **lower** FAR (−2.2 points)
— a much smaller and more defensible trade than Candidate D's (which lost 15–17 points on the same measures).

## Calibration
| | Slope | Brier (calibrated / raw) | ECE10 |
|---|---|---|---|
| D2 canonical | 1.29 | 0.145 / 0.150 | 0.051 |
| D2 fresh mean | **1.18** | 0.146 / 0.150 | 0.038 |
Recalibration still clearly helps (Brier improves by ≈0.005 in both). The fresh-resplit slope sits inside the pre-registered band; only the single canonical-split slope (1.29) pushes
the joint "both must pass" rule to a fail. This is the closest any exploratory candidate in this study has come to meeting all four guards at once.

## What changed mechanically
The constraint visibly bites: selected α values cluster around 0.25–0.40 (vs Candidate D's fixed, unconstrained ≈0.42–0.45), i.e. the fix works exactly as diagnosed — it pulls fusion
weight back toward parity when the training data show that doing so is needed to keep lead time up, and only pulls toward G-A's window score when the training data allow it without
the constraint biting.

## The canonical-split anomaly
D2 loses to B0 on the canonical split's AUROC (−0.0054) while winning by a wide, consistent margin on all 10 fresh resplits (+0.006 to +0.024, mean +0.015). This is the reverse of
Candidate D's pattern (which won canonically, +0.0053, and won on all 10 of *its* fresh resplits too, but by less, +0.0108). The canonical split's selected alphas (0.20–0.30, the lowest
of any split run) suggest the lead-time constraint bit hardest there — canonical happens to be a split where satisfying the 35-minute floor costs more AUROC than in most resplits.
With only one canonical split this cannot be resolved statistically; it is reported as an open question, not explained away.

## What to conclude
- **The targeted fix worked**: identifying that unconstrained α-selection was the mechanism behind Candidate D's lead-time failure, and constraining it, produced a candidate that keeps
  (and on fresh resplits, improves) the discrimination gain while passing the lead-time guard on 10 of 11 splits and landing very close to acceptable calibration.
- **It is not yet a clean pass.** Calibration misses by a small margin on the canonical split, and the canonical-vs-fresh discrimination split is unexplained.
- **Still exploratory, still internal-cohort, still not adopted.** The frozen URM remains the deployed model. This result is materially more encouraging than any of Candidates D or
  the V2/V2E composite, but "close to passing four guards on internal resplits" is a smaller claim than "validated."
- **Natural next step, not run here** (out of scope per the frozen protocol): a small, equally principled tweak to recalibration — e.g. isotonic instead of logistic, which cannot
  produce a slope outside [1] by construction — would likely close the calibration gap; that would need its own short pre-registration before running, consistent with this study's
  practice of not chasing a failed guard by re-running until it passes.
