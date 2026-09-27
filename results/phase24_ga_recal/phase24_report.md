# Phase 24 — Candidate D (G-A fixed α + recalibration): results

Protocol: `docs/phase24_ga_recal_protocol.md` (frozen before any code). Script: `scripts/phase24_ga_recal.py`.
Candidate D = G-A window (ridge α = 1000 fixed, no selection) + standard binary-label pooler + parity fusion + per-fold recalibration.
Evaluated on all 547 patients: canonical split + 10 fresh resplits (seeds 301–433, never used in Phases 20–23). The 83-patient test partition was **not used**.

> **No independent cohort exists in this environment.** Everything below still reuses the same 547 CTU-UHB patients; "fresh resplits" means new random partitions of the same people,
> not new people. This evaluation can show whether the candidate is consistent across a wider set of internal splits than Phase 23 used — it cannot show that the candidate generalises.

## Pre-registered verdict: **EXPLORATORY NO SUPPORT** (D1, D2 met; lead-time guard L failed; calibration guard E narrowly failed)
| Criterion | Result | Met |
|---|---|---|
| D1 mean ΔM over 10 fresh resplits ≥ +0.005, ≥8/10 positive | **+0.0148, 10/10 positive** (range +.0072 to +.0229) | yes |
| D2 mean per-horizon Δ ≥ −0.010 | +0.0158 / +0.0174 / +0.0141 / +0.0121 (0/10/20/30 min) | yes |
| L lead-time guard at matched false-alert rate (canonical **and** fresh mean): Δ sensitivity ≥ −0.02, Δ median lead ≥ −2.5 min, Δ share ≥20 min ≥ −0.03 | canonical: sens 0.0, lead **−7.5 min**, share≥20 **−0.125**; fresh mean: sens +0.006, lead −2.1 min, share≥20 **−0.059** | **no** (canonical fails badly; fresh mean fails on share≥20) |
| E calibration (slope ∈ [0.80,1.25] both, Brier no worse) | slope canonical **1.32** (just above 1.25), fresh mean 1.23 (inside); Brier improves in both (canonical 0.147→0.141, fresh 0.150→0.144) | **no** (canonical slope alone) |

Statistics: canonical ΔM = +0.0053 (95% CI −0.013, +0.024; p = .57); fresh-resplit pooled p = .067.

## Discrimination — fresh-resplit means
| Horizon | B0 | D | Δ |
|---|---|---|---|
| Delivery | 0.7120 | 0.7278 | +0.0158 |
| ≥10 min | 0.6698 | 0.6871 | +0.0174 |
| ≥20 min | 0.6520 | 0.6661 | +0.0141 |
| ≥30 min | 0.6420 | 0.6541 | +0.0121 |
| **M** | 0.6707 | **0.6855** | **+0.0148** |
Canonical: B0 0.6903 → D 0.6957 (+0.0053). This is the largest, most consistent discrimination gain of any candidate tried in Phases 19–24 (10/10 resplits positive, every horizon positive on every split checked).

## Lead time at matched false-alert rate (the pre-registered guard, this phase's main question)
| | Sensitivity | False-alert rate | Median lead | ≥10 min | ≥20 min | ≥30 min |
|---|---|---|---|---|---|---|
| B0 canonical | 87.3% | 63.2% | 40.0 min | 92.7% | 83.3% | 71.9% |
| **D canonical** | 87.3% | 63.2%* | **32.5 min** | 85.4% | **70.8%** | 56.3% |
| B0 fresh mean | 84.6% | 62.2% | 40.0 min | 93.1% | 85.2% | 71.3% |
| **D fresh mean** | 85.2% | 59.8% | 37.9 min | 88.5% | **79.3%** | 67.0% |
(*threshold is fixed from B0 by construction, so B0's own FAR is reproduced exactly; D's FAR differs slightly because its α differs — see protocol §3.)

Phase 23b's post-hoc finding (that a matched-FAR comparison would show G-A-type models keep the baseline's lead times) **does not replicate for this simpler candidate**: even at a matched
operating point, D still warns later than B0, most clearly on the canonical split (median lead 32.5 vs 40 min, share warned ≥20 min 70.8% vs 83.3%). The 80%-target-sensitivity operating
point (reported for continuity, no decision weight) shows the same pattern, slightly worse: canonical median lead 30.0 vs 40.0 min, ≥20 min 66.3% vs 83.2%.

## Calibration (delivery horizon)
| | Slope | Brier (calibrated / raw) | ECE10 |
|---|---|---|---|
| D canonical | 1.32 | 0.141 / 0.147 | 0.053 |
| D fresh mean | 1.23 | 0.144 / 0.150 | 0.041 |
Recalibration clearly helps (Brier improves by ≈ 0.006 in both, slope moves from ≈2.7 in the raw fused score to ≈1.2–1.3) but the canonical slope sits just outside the pre-registered
[0.80, 1.25] window; by the letter of the rule this guard fails, though the practical calibration quality is similar to — and better than — every other model evaluated so far.

## What to conclude
- **Discrimination is the strongest and most consistent result across Phases 19–24**: +0.015 mean AUROC, positive on every one of 21 splits now tried across Phases 23 and 24 combined.
- **It comes with a real cost.** At a false-alert rate matched to the baseline, this simpler candidate (unlike the earlier composite) still warns noticeably later, especially on the
  canonical split. The earlier post-hoc reading — that the lead-time loss was just an operating-point artefact — does not hold up once actually pre-registered and tested on more splits.
- **Calibration is good but not quite inside the pre-registered band on the canonical split**, while clearly improving both splits' Brier scores.
- **No claim of generalisation is supported by this phase**; every result here is internal-cohort. The consistent discrimination gain across 21 total splits is the strongest evidence
  produced in this study for *some* real signal in the graded-pH window model, but the lead-time trade-off means it is not a straightforward improvement over URM, and nothing here
  substitutes for external validation.
