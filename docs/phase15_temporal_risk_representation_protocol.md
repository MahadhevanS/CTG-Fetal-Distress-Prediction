# Phase 15 — Temporal Risk Representation: Exploratory Methodology Study
## Pre-Registration Protocol

**Frozen 2026-09-11, before the final analysis script was written.** Governed
by the same discipline as `docs/phase13_protocol.md` / `docs/phase14_protocol.md`.

> **Governance rule (unchanged).** Phase 12.1 is locked and authoritative
> regardless of any result in this phase. No model is retrained. No
> production convention changes. This is a methodology study of what the
> already-frozen P6 window-level predictions contain, not a search for a
> replacement model.

## 0. Pre-implementation verification (completed before this protocol was written)

- `results/phase13/audit/p6_predictions.npz` confirmed as the frozen source:
  `patient_ids`/`time_before_delivery` arrays match `results/phase8_rolling/rolling_predictions.csv`
  row-for-row exactly (8,517/8,517 rows, exact match, no reordering).
- No NaN in CV predictions; range [0.045, 0.928], consistent with a
  probability output.
- Test-partition predictions are populated for exactly the 1,274 windows
  belonging to the 83 held-out test patients and zero elsewhere — confirms
  no cross-partition leakage in how the array was constructed.
- `folds.json`: 547 patients, 5 folds, unchanged from every prior phase.
- Causal boundary on the underlying window-level scores was already
  certified in Phase 8 (synthetic future-perturbation test, 0.00000000
  delta) and the scores themselves were bit-reproduced independently in
  Phase 13.0A against `results/phase13_information_density/step_predictions.csv`
  (max difference 0.000000 across all 8,517 windows). This phase does not
  repeat those audits; it cites them and adds no new model, so nothing new
  could reintroduce a causal violation.
- No pH, delivery time, or post-window information is used by anything in
  this protocol — every representation below is a function of the
  window-level score sequence and its own timestamps only.

## 1. Hypothesis (broader than any single aggregator)

Does the longitudinal distribution/structure of causal P6 window-level risk
scores contain patient-level information that single-window selection
discards? P90 (Phase 13/14) is one candidate answer, not the object of
study — the object of study is the sequence itself.

## 2. Epistemic framing (binding on all reporting in this phase)

Same 547-patient cohort as every prior phase. No claim in this phase may
use the words "independently validated," "replicated on an independent
dataset," "confirmed," or "externally validated." Permitted vocabulary:
*exploratory, hypothesis-generating, internally evaluated, preliminary,
descriptive.* The P90 point estimate was already observed in Phase 13 and
formally (non-)tested in Phase 14 — nothing in this phase re-litigates that
verdict as if it were unseen evidence.

## 3. Frozen inputs

`results/phase13/audit/p6_predictions.npz` (window-level CV/test scores),
`data/processed_clinical/folds.json`, `data/processed_clinical/test_dataset.pt`,
`results/phase8_rolling/rolling_predictions.csv` (for `time_before_delivery_min`,
`start_sample`, labels). No retraining anywhere in this phase.

## 4. Core representations (Section 6 of the request — five only, no expansion)

| Code | Representation | Definition | Intended information |
|---|---|---|---|
| SW | Single-window | `get_patient_scores_at_horizon_corrected` (Phase 13/14 convention) | Risk at one representative time point |
| P90 | 90th percentile | `numpy.percentile(scores, 90)` over the causally-eligible pool — **Phase 14's exact estimator**, not Phase 13's weighted-quantile function, chosen for continuity with the already-reported Phase 14 primary test and to avoid introducing a third P90 definition | Upper-tail / recurrent high-risk behavior |
| Max | Maximum | `max()` over the eligible pool | Peak observed risk |
| Mean | Arithmetic mean | `mean()` over the eligible pool | Overall risk burden |
| Median | 50th percentile | `numpy.percentile(scores, 50)` over the eligible pool | Typical risk, reduced sensitivity to isolated peaks |

Eligible pool at horizon *h*: `{i : t_i ≥ h}` (all windows if `h=0`).
Nothing beyond these five is added to the core comparison.

## 5. Horizons

Delivery (0m) and ≥30m reported as the two horizons of interest (neither
promoted to sole "primary" status in this phase — Section 9 of the request
explicitly asks that this not be framed as another confirmatory P90 test).
≥10m/≥20m computed and reported for completeness, exploratory only.
Corrected-horizon convention only (Phase 13.0A already settled
existing-vs-corrected; not reopened here). Both conventions are not
silently mixed.

## 6. Temporal persistence/structure descriptors (Section 7 of the request)

Four descriptors, each with a fixed, pre-specified definition — none swept,
none chosen after seeing how they perform:

- **Persistence(τ)** = fraction of eligible windows with score > τ.
- **Longest run(τ)** = longest consecutive run (in chronological order)
  of eligible windows with score > τ.
- **Slope** = OLS slope of window score vs. window chronological index
  over the eligible pool (per-minute units via known stride).
- **Recent − earlier** = mean of the later half of the eligible pool minus
  mean of the earlier half (patient's own within-recording split, no
  cross-patient comparison).

**Threshold τ, fixed before any outcome is examined:** the 75th percentile
of all window-level P6 scores pooled across that fold's *training* patients
only (fit on train, applied to held-out — same discipline as every other
statistic in this phase). This is a population-relative, purely
distributional definition — it uses no label and no per-patient outcome
information, so it does not leak, but it is explicitly **not** a
clinically-validated risk cutpoint (P6's score scale has no established
clinical meaning at any absolute value). This limitation is to be restated
in the final report, not resolved by inventing a "better" threshold.

## 7. Statistics

Patient-level AUROC + AUPRC, patient-level clustered bootstrap (B=2000),
paired on identical patients against the SW baseline, DeLong as a secondary
check. CV (primary evidence source, larger sample) and held-out test
(reported separately, never used to pick a winner) both reported for every
representation at every horizon. No representation's hyperparameter (the
one exception, τ) is selected by outcome-maximization anywhere in this
phase — τ is a fixed population statistic, not swept.

## 8. Operational metrics

For SW, P90, Max, Mean, Median at both horizons: sensitivity/specificity at
a fixed target-sensitivity operating point (80%, matching Phase 13's
convention), false-alert rate, median warning lead time, % detected ≥20min,
% detected ≥30min.

## 9. Recording-duration analysis

Per patient: eligible-window count at each horizon, and its correlation
(Pearson + Spearman) with each representation's aggregated value and with
the SW-vs-representation delta. Descriptive only — no duration adjustment
is applied to any reported AUROC (that would itself be an unplanned,
outcome-blind-only-in-theory transformation risking a new source of
post-hoc tuning).

## 10. P90 vs. Max descriptive comparison (Section 14 of the request)

Per patient, `P90 − Max` (always ≤ 0 by construction). Reported
descriptively: distribution of this difference, and whether patients with
`P90 − Max ≈ 0` (P90 behaving like a softened max) differ systematically
from patients with a larger gap, split by outcome label. No new hypothesis
test; no percentile re-tuning based on this observation.

## 11. Decision framework (fixed before results are seen)

Each of the five core representations is classified independently:

- **Category 1 — no evidence of useful additional information**: AUROC
  delta ≈0 or negative, no operational advantage.
- **Category 2 — hypothesis-generating signal**: consistent positive
  direction across CV and test and/or a believable operational trend, CI
  crosses zero. This is the ceiling this phase can award given the
  same-cohort constraint (Section 20 of the request) — nothing here can
  reach a stronger category without a confirmatory test on data this
  cohort cannot provide.
- **Category 3 — strong internal signal**: reserved for a result significant
  on CV, directionally confirmed on test, and robust under the recording-
  duration check — even then reported as exploratory, never as grounds to
  replace P6.

## 12. Reporting rules

The complete planned set of five representations (plus the four descriptors)
is reported regardless of which looks best — no result is dropped or
promoted to lead position based on its own performance. If no representation
reaches formal significance, that null is reported as the result, not as a
reason to alter the estimator, horizon, threshold, or fold split
post hoc. Phase 12.1's locked numbers (delivery 0.6872, ≥30m 0.5857 under
the original horizon convention) are cited for reference only and are not
altered by anything computed here.

## 13. Amendments

None yet.
