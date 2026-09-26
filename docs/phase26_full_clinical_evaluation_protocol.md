# Phase 26 — Full clinical evaluation of the locked URM (calibration, PPV/NPV, threshold curve, parity ablation)
## Pre-Registration Protocol

> **Status (2026-09-26): executed.** Calibration/PPV-NPV/threshold-curve computed on the gate-verified exact URM sequence (a data-source correction was made mid-phase -- see report); parity ablation run with two null controls; fusion-weight review compiled from existing frozen evidence. Results: results/phase26_clinical_evaluation/phase26_report.md. Explainability interface (Problem 9): results/phase26_clinical_evaluation/explainability_interface.html.

**Frozen 2026-09-26, before any Phase 26 code was written or run.** This phase changes **nothing** about the locked model (PRS, TAM, MCM, URM,
α = 0.35). It only computes reporting metrics that were missing, and runs one new ablation. Corresponds to the user's Problems 3–6 (calibration,
PPV/NPV/false-alert-burden, threshold-performance curve, parity contribution) from the 2026-09-26 review. Problems 1/2/8 (external validation,
test-set governance) and 9–14 (explainability, framing, temporal modelling, signal quality) are handled as separate documents, not experiments,
since none of them involve a new metric on held-out data.

## 1. Data source (no retraining)
All of Problems 3–5 use the **already-frozen** per-patient, per-horizon predicted probabilities in
`results/phase18_fusion_ablation/stage1_patient_level_predictions.csv` (`pred_A3_prob_avg_selected` = URM; `split='cv'` is the canonical 547-patient
out-of-fold prediction that the frozen headline numbers (0.7335/0.6921/0.6638/0.6631) were computed from; `split='test'` is the 83-patient held-out
set). These are the exact frozen predictions; nothing is recomputed except the metrics themselves.

## 2. Problem 3 — Calibration
For URM, at each of the 4 horizons, on the canonical CV predictions (primary) and the test predictions (confirmatory, reported not gated):
Brier score; calibration intercept and slope (logistic regression of the binary outcome on logit(predicted probability), unregularized);
a 10-bin reliability table (predicted-probability bin → n, mean predicted, observed rate) and its plot. No recalibration is applied — these are the
model's own probabilities, reported as delivered.

## 3. Problem 4 — Classification metrics, PPV/NPV, false-alert burden
At URM's existing operating point (Phase 18: 80% target sensitivity, per-fold training-only threshold, patient-level "ever alerted", delivery
horizon) already on file (`deployable_fusion_operational_metrics.csv`, `A3_selected` row): compute the two numbers that file does not carry —
**specificity**, **PPV**, **NPV**, **F1** (from the same confusion counts: TP/FP/TN/FN already implicit in achieved sensitivity and FAR × n) — and
two clinician-facing burden measures:
- **false alerts per patient** = FP / n_patients screened (not just among positives);
- **false alerts per hour of monitoring** = FP / (total monitored hours across all patients in that split), using each patient's own retained
  recording length (`results/phase8_rolling/rolling_predictions.csv`, patient's last window's end sample).

## 4. Problem 5 — Threshold-performance curve
Sweep the delivery-horizon URM score over its own decile thresholds (10 points, canonical CV) plus the two named operating points already in use
(80% and 90% target sensitivity). At each threshold: sensitivity, specificity, PPV, NPV, FAR, F1, alerts/patient, alerts/hour, median lead time and
share of positives alerted ≥20/≥30 min before delivery (patient-level "ever alerted", training-only threshold selection convention, canonical CV).
Report as one table; no threshold is "selected" here — this is descriptive, for a clinician to pick from, not a new adoption decision.

## 5. Problem 6 — Does parity carry independent information, or is it a proxy?
**New experiment**, evaluated on the canonical 547-patient CV split, using the **frozen TAM scores** (identical per-fold checkpoints used for URM
itself: `Model_3_magnitude_position_fold{0..4}`, loaded via `load_completed_units`/`load_scorer_checkpoint`, exactly as `phase18_deployable_fusion.py`
does) so the "CTG" component in every arm is the literal deployed component, not a harness reproduction. Fusion mechanism unchanged from URM: α
selected per fold by sample-weighted pooled training AUROC, grid 0.05.

| Arm | Covariate model | Purpose |
|---|---|---|
| **CTG only** | none (TAM alone) | reference floor |
| **Parity only** | MCM (LR, parity) | reference ceiling for the covariate alone |
| **CTG + parity (= URM)** | MCM | the deployed model |
| **CTG + shuffled parity** | MCM refit on parity values **randomly permuted across training patients within each fold** (breaks the true patient–parity link; preserves the value distribution and the exact fitting/fusion procedure) | null control: does fusion + free α-selection manufacture an apparent gain from a covariate with no real link to outcome? |
| **CTG + random demographic-like variable** | MCM refit on an i.i.d. `N(0,1)` variable, one fixed draw per patient (seed 42), in place of parity | second, independently-constructed null control |
| **CTG + additional maternal variables** | LR (C=1.0) on {parity, maternal age, gravidity, gestational weeks}, training-fold median imputation for missing gravidity | does going beyond parity add anything, using the same late-fusion recipe? (a confirmatory rerun of Phase 20 §C4-S1's already-negative finding, this time against the frozen TAM/URM basis directly, for a self-contained table) |

**Statistics**: paired patient bootstrap (B = 2000, seed 42) of each arm's ΔM (mean AUROC over the 4 horizons) vs **both** "CTG only" and
"CTG + parity"; 95% CI and two-sided p reported for every arm. **Reading rule (descriptive, not adopt/reject):** parity is read as carrying
independent information if (a) CTG+parity beats CTG-only by a margin that is *not* matched by either null control, and (b) the null controls'
own ΔM vs CTG-only is not significantly positive. This is an explanation, not a new candidate; it cannot change URM's locked status either way.

## 6. Reporting
One consolidated report `results/phase26_clinical_evaluation/phase26_report.md` covering Problems 3–6, plus the existing frozen evidence that already
answers two more of the reviewer's points without new computation:
- **Problem 7 (fusion form)**: URM's α was already chosen by a full grid search over α ∈ {0, 0.05, …, 1} inside development CV, then frozen once
  (`deployable_fusion_FROZEN_params.json`: 30 independent fold-fits, mean 0.333, range 0.30–0.40) and evaluated only once, untouched, on the held-out
  test partition. A learned fusion function was also tried (`F2_deployable`, logistic on [logit(TAM), logit(parity)]; `F7_deployable`, the same plus
  an interaction term) and **both underperform the simple linear-α fusion on CV and on the untouched test set at every horizon**
  (`deployable_fusion_results.csv`). No new run is needed; this is restated here because it directly answers the question asked.

## 7. Out of scope
Any change to PRS, TAM, MCM, URM, α, or the operating-point threshold; new candidates; the exploratory G-A/D/D2 line (Phases 21–25, unaffected).
