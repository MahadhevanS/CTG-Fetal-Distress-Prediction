# Phase 21 — Graded-outcome supervision: screening result

Protocol: `docs/phase21_graded_supervision_protocol.md` (frozen before any run). Diagnostics: `results/phase21_diagnostics/`.
Development set D464, canonical + 5 resplits, baseline = Phase 20 B0 (M 0.6671 canonical). Test partition **not touched**; confirmation stage **not run**
(no candidate met the screening rule).

## Why the earlier redesign found nothing (diagnostics)
- 51% of positives (47/93) lie within 0.05 pH of the 7.15 cut-off; the same score reaches AUROC 0.794 for pH≤7.05 vs normal but 0.724 for borderline positives.
- The learning curve is flattening (0.598 → 0.619 → 0.634 → 0.639 from 92 → 371 training patients); one descriptor (accelerations) alone reaches 0.667.
- Subgroups (delivery type, stage at end, recording length, parity): no distinguishable differences (intervals wide).
=> the binary label mixes real severity signal with near-threshold noise, and the feature side is already near what these patients support.

## Screening (ΔM vs baseline B0; control B0-P = binary label + same Platt step)
| Candidate | Canonical ΔM (p) | Resplit ΔM (5) | vs B0-P (6 splits) | Tier |
|---|---|---|---|---|
| Control B0-P | −0.0002 | −.0002 +.0004 −.0002 +.0032 +.0009 | — | (recalibration alone does nothing) |
| **G-A** ridge on −pH | **+0.0054** (p .61; Holm 1.0) | +.0125 +.0042 +.0170 +.0270 +.0093 | all 6 positive | SUGGESTIVE |
| G-B multi-threshold | +0.0022 (p .83) | +.0103 −.0029 +.0120 +.0180 −.0040 | 4/6 | NOT SUPPORTED |
| G-C soft label | +0.0032 (p .69) | +.0095 +.0005 +.0082 +.0204 −.0081 | 4/6 | SUGGESTIVE |

- **G-A is the only consistent lead**: positive in all 6 splits, all 6 against the recalibration control, and it raises the window-level PRS-only score in every split too
  (+0.006 to +0.024). The mean over the 5 resplits is +0.014.
- **It is not adopted.** ADOPT required a Holm-adjusted p < .05 canonically; the canonical bootstrap p is .61 (CI roughly ±0.02), so on these 464 patients the canonical gain
  cannot be told apart from sampling noise. This is a sample-size limit, not a multiplicity problem (K = 1 would not change p).
- G-A selected the strongest regularisation offered (α = 1000, the grid edge) in nearly every fold, which suggests the useful part of the pH signal is low-dimensional.
- The six splits reuse the same 464 patients, so their agreement is evidence about split variability, not additional patients.

---
# Exploratory follow-up 21b (protocol §7; rules frozen before the run)

Status: **EXPLORATORY.** G-A failed the §4 rule, so it is never called adopted or confirmed. Files: `confirm_verdict.json`, `followup_21b.json`, `followup_21b.log`.
(`"CONFIRMED": false` in `confirm_verdict.json` is the code applying the §5 rule, which needs p < .05; it is consistent with the exploratory status.)

## 1. Fresh resplits (seeds 66/77/88/99/111 on D464; never used in selection)
| Arm | ΔM per resplit | mean ΔM | positive | worst mean per-horizon Δ | pooled p |
|---|---|---|---|---|---|
| **G-A** (primary) | +.0098 +.0077 +.0074 +.0042 +.0149 | **+0.0088** | 5/5 | +0.0077 | .29 |
| G-A-wide (post-hoc, α up to 1e5) | +.0068 +.0052 +.0093 +.0044 +.0187 | +0.0089 | 5/5 | +0.0064 | .36 |

Every fresh split favours G-A, at about +0.009 (screening resplits: mean +0.014, canonical +0.005). No horizon is negative on average. The pooled p is not significant.

## 2. Test partition, opened once (83 patients, 17 positives; trained on all D464)
| Arm | M | 0m / 10m / 20m / 30m | ΔM vs baseline (95% CI) | p |
|---|---|---|---|---|
| Baseline B0 | 0.7582 | .753 / .718 / .781 / .781 | — | — |
| **G-A** | **0.8073** | .773 / .798 / .849 / .809 | **+0.049** (+0.0004, +0.109) | .050 |
| G-A-wide | 0.8144 | .786 / .824 / .846 / .802 | +0.056 (−0.018, +0.146) | .158 |
Frozen URM test AUROC for reference (Phase 18): 0.743 / 0.699 / 0.756 / 0.775.

## Label (mechanical, primary arm): **EXPLORATORY SUPPORT**
(fresh mean ΔM +0.0088 ≥ .005; 5/5 fresh resplits positive; worst horizon +0.0077; test ΔM +0.049 > 0.)

## How far this can be trusted
- **What it shows:** a consistent direction across 11 splits and the untouched test partition, always against a baseline computed the same way, and the recalibration-only control did nothing.
- **What it does not show:** improvement in the statistical sense. All resplits reuse the same 464 patients, so they measure split variability, not sampling variability; the fresh pooled p is .29 and the test-partition interval has a lower bound of +0.0004 on 17 positives.
- G-A-wide was defined after seeing the screening result and carries no decision weight; it behaves like G-A, so the grid edge is not what limits the result.
- The frozen URM is unchanged and remains the recommendation. Confirming G-A needs independent patients (external validation), not more splits.
