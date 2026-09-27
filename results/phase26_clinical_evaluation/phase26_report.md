# Phase 26 — Full clinical evaluation of the locked URM (Problems 3–7)

Protocol: `docs/phase26_full_clinical_evaluation_protocol.md`. Scripts: `scripts/phase26_clinical_evaluation.py`, `scripts/phase26_parity_ablation.py`.
**URM is unchanged.** Everything here is a new report on the existing, locked model.

**Important correction made during this phase:** `results/phase18_fusion_ablation/stage1_patient_level_predictions.csv` — the file that looked like
the obvious source of per-patient URM predictions — turned out to be an earlier, exploratory Stage-1 snapshot (dated 2026-09-15) whose own AUROC
(0.7382/0.6913/0.6659/0.6470) does not match the frozen headline URM numbers (0.7335/0.6921/0.6638/0.6631) this whole study reproduces and gates
against. It was not used. Instead, the exact URM sequence was regenerated from the frozen artifacts (per-fold TAM checkpoints, `fit_parity_lr`,
`fit_deployable_bundle` imported directly, not reimplemented) and **gated against the frozen headline AUROC on both CV and the held-out test
partition before anything below was computed** — both gates passed exactly (0.7335/0.6921/0.6638/0.6631 CV; 0.7433/0.6988/0.7558/0.7754 test).

## Problem 3 — Calibration

| Split | Horizon | Brier | Cal. slope | Cal. intercept | ECE (10 bins) |
|---|---|---|---|---|---|
| CV | Delivery | 0.149 | 2.67 | 1.98 | 0.038 |
| CV | ≥10m | 0.152 | 2.08 | 1.43 | 0.041 |
| CV | ≥20m | 0.154 | 1.85 | 1.17 | 0.037 |
| CV | ≥30m | 0.155 | 1.69 | 0.95 | 0.046 |
| Test | Delivery | 0.151 | 2.19 | 1.31 | 0.063 |
| Test | ≥10m | 0.154 | 2.50 | 2.01 | 0.046 |
| Test | ≥20m | 0.149 | 3.66 | 3.57 | 0.060 |
| Test | ≥30m | 0.149 | 3.80 | 3.74 | 0.080 |

Reliability table, CV delivery (full table saved to `reliability_tables.json`):

| Predicted-probability bin | n | Mean predicted | Observed rate |
|---|---|---|---|
| 0.0–0.1 | 15 | 0.082 | 0.067 |
| 0.1–0.2 | 190 | 0.158 | 0.090 |
| 0.2–0.3 | 310 | 0.238 | 0.239 |
| 0.3–0.4 | 32 | 0.328 | **0.563** |

**Reading:** URM's own probabilities are poorly calibrated — slope well above 1 at every horizon (worse at the later horizons and on test), meaning
scores are compressed into too narrow a range for the separation the model actually has. The 0.3–0.4 bin is a clear failure: the model says ~33%
risk, the true rate there is 56%. Brier is similar across horizons (0.149–0.155), consistent with prevalence-driven floor effects rather than a
genuinely well-separated probability. **This directly answers the user's MIRF-Net comparison point** (MIRF-Net Brier 0.2537): URM's Brier
(0.149–0.155) is numerically better, but Brier alone hides the calibration failure shown above — a literature comparison on Brier alone would be
misleading in URM's favour without also reporting the slope/intercept, which is why both are now included.

## Problem 4 — Classification metrics, PPV/NPV, false-alert burden (URM's existing 80%-target operating point, canonical CV)

| Metric | Value |
|---|---|
| Sensitivity | 86.4% |
| Specificity | 38.4% |
| PPV | 26.1% |
| NPV | 91.8% |
| F1 | 0.401 |
| False-alert rate | 61.6% (published operational table: 62.5% — a 0.9-point difference, from an independent alpha-grid tie-break across separate script runs; both numbers reported, not reconciled by re-tuning) |
| **False alerts per patient screened** | **0.49** |
| **False alerts per hour monitored** | **0.51** (269 false alerts over 527.2 total monitored hours across all 547 patients) |

The per-hour figure is the number most directly meaningful to a bedside clinician: roughly one false alert every two hours of monitoring, on top
of the true alerts.

## Problem 5 — Threshold-performance curve (canonical CV, delivery-horizon threshold applied patient-level "ever alerted")

| Target sensitivity | Sensitivity | Specificity | PPV | NPV | F1 | FAR | Alerts/patient | Alerts/hour | Median lead | ≥20min | ≥30min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50% | 63.6% | 72.8% | 0.370 | 0.888 | 0.468 | 27.2% | 0.218 | 0.226 | 21.3 min | 51.4% | 37.1% |
| 60% | 75.5% | 61.6% | 0.331 | 0.909 | 0.460 | 38.4% | 0.307 | 0.319 | 22.5 min | 54.2% | 42.2% |
| 70% | 80.9% | 52.6% | 0.301 | 0.916 | 0.438 | 47.4% | 0.378 | 0.393 | 32.5 min | 65.2% | 53.9% |
| **80% (current)** | **86.4%** | **38.4%** | **0.261** | **0.918** | **0.401** | **61.6%** | **0.492** | **0.510** | **40.0 min** | **80.0%** | **68.4%** |
| 90% | 91.8% | 23.1% | 0.231 | 0.918 | 0.369 | 76.9% | 0.614 | 0.637 | 40.0 min | 92.1% | 80.2% |
| 95% | 95.5% | 12.1% | 0.215 | 0.914 | 0.351 | 88.0% | 0.702 | 0.664 | 40.0 min | 91.4% | 81.0% |

Full curve (with confusion counts) saved to `threshold_performance_curve.csv`. This is descriptive — no threshold is chosen here; it lets a
clinician see, e.g., that dropping the target from 90% to 70% roughly halves both the false-alert rate (77%→47%) and alerts/hour (0.64→0.39),
at a cost of about 11 points of sensitivity and a slightly shorter lead time (32.5 vs 40 min).

## Problem 6 — Does parity carry independent information?

New experiment (canonical CV, frozen per-fold TAM checkpoints, same fusion mechanism as URM):

| Arm | M (mean AUROC) | Δ vs CTG-only | Δ vs URM | Selected α per fold |
|---|---|---|---|---|
| CTG only (TAM alone) | 0.6472 | — | −0.0431 | (n/a, α=1 by construction) |
| Parity only (MCM alone) | 0.5755 | −0.0717 | −0.1148 | (n/a, α=0) |
| **CTG + parity (= URM)** | **0.6903** | **+0.0431** | — | 0.30–0.35 |
| CTG + shuffled parity (null control) | 0.6415 | **−0.0057** | −0.0488 | 0.05–1.0 (mostly pushed to 1.0, i.e. ignore the covariate) |
| CTG + random N(0,1) variable (null control) | 0.6408 | **−0.0064** | −0.0495 | 0.1–1.0, no consistent value |
| CTG + additional maternal vars. (age, gravidity, gest. weeks) | 0.6789 | +0.0317 | −0.0114 | 0.30–0.40 |

**Reading:** both null controls (a covariate with the true patient–value link destroyed, and pure random noise) come out at or slightly *below*
CTG alone once fused — the alpha-selection mechanism correctly detects they carry no signal and pushes weight away from them (α often selected
at 1.0, i.e. "ignore this covariate"). This is the key check: **the fusion mechanism does not manufacture an apparent gain from an arbitrary
covariate.** Parity's own contribution (+0.043 mean AUROC, and per the already-frozen, gated per-horizon figures in
`deployable_fusion_results.csv`: Δ = +0.012 delivery (p = .47), +0.024 at ≥10m (p = .23), **+0.062 at ≥20m (p = .001)**, **+0.065 at ≥30m
(p < .001)**) is far larger than either null control's, and grows sharply at the horizons furthest from delivery — exactly where the CTG signal
is weakest and a static, always-available covariate should help most. Adding still more maternal variables beyond parity does **not** help
further (−0.011 vs URM), confirming Phase 20's independent finding (§C4-S1, NOT SUPPORTED, worse in 5/5 splits) on a completely different
pipeline basis. **Conclusion: parity is very likely acting as a genuine — if crude — proxy for obstetric risk (multiparity is an established
correlate of labour progress and intrapartum risk), not an artefact of the fusion procedure, but this is inferred from internal evidence only;
it has not been tested against a directly recorded risk-severity variable, which would be the more direct external test if hospital data provides one.**

## Problem 7 — Fusion weight (already answered by existing frozen results, restated here)

URM's α was **not** hand-picked: it was chosen by grid search over α ∈ {0, 0.05, …, 1} inside development CV, independently per fold (30
fold-fits across the canonical split and 5 resplits: mean 0.333, median 0.35, range 0.30–0.40 — `deployable_fusion_FROZEN_params.json`), then
**frozen once** and evaluated only once, untouched, on the held-out test partition. A learned fusion function was also tried and rejected on
evidence, not by assumption:

| Model | CV AUROC (Delivery/10/20/30m) | Test AUROC (Delivery/10/20/30m) |
|---|---|---|
| **A3 (linear α, = URM)** | 0.7335 / 0.6921 / 0.6638 / 0.6631 | **0.7433 / 0.6988 / 0.7558 / 0.7754** |
| F2 (logistic on [logit(TAM), logit(parity)]) | 0.7291 / 0.6890 / 0.6466 / 0.6449 | 0.7077 / 0.6417 / 0.6970 / 0.7371 |
| F7 (F2 + interaction term) | 0.7329 / 0.6907 / 0.6494 / 0.6472 | 0.7094 / 0.6435 / 0.7014 / 0.7353 |

The simple linear fusion beats both learned alternatives on CV **and** on the untouched test set, at every horizon. No new run was needed; the
evidence already existed and directly answers the question.
