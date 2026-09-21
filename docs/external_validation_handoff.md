# External Validation Handoff & Protocol
## Candidates: Model 3 (Phase 16) and Parity Fusion (Phase 13D)

Written 2026-09-12. This document is self-contained — written for whoever
executes this validation, whether or not they have context on the
preceding 16-phase internal investigation.

> **Governance, carried forward unchanged.** Phase 12.1 (547 patients,
> pH ≤ 7.15 / ≤ 7.05, delivery AUROC 0.6872, ≥30m 0.5857) is locked,
> untouched, and remains the sole authoritative production model
> regardless of anything in this document or any result that follows from
> it. A successful external validation is a prerequisite for a future
> promotion conversation — it is not itself a promotion decision.

---

## 1. What is being handed off, and why these two

Six phases of post-lock internal investigation (Phases 13–16, summarized in
full in `reports/phase13_16_closure_summary.md`) tested a broad space of
ways to extract more information from P6's causal window-level risk
sequence than the current single-window production convention uses. Most
directions closed cleanly. Two did not, and both cleared their own
pre-registered bars well enough, and were verified rigorously enough, to be
worth testing on data this project has never seen:

- **Model 3** (`docs/phase16_protocol.md`) — a trainable causal-attention
  aggregator over the sequence of P6's own window-level scores, weighted by
  magnitude and temporal position. Internal result: delivery CV AUROC
  0.7216 vs. P6's 0.6872 (Δ+0.0344, p=.014, 95% CI entirely positive),
  survived two independent verification passes (epoch-budget extension,
  fold-resplit sensitivity) designed to break it before it was trusted.
- **Parity fusion** (`docs/phase13_protocol.md` Option D) — P6 fused via
  logit-prior modulation with a univariate model on maternal parity
  (admission-time, causally safe). Internal result: delivery test AUROC
  0.6497 → 0.7148 (Δ+0.0651, p=.025), passed seed-robustness,
  fold-resplit-robustness, recording-duration-confound, and
  conditional-analysis checks.

**Not part of this handoff, closed with reasonable confidence:**
recency-weighted P90, window stride change, Huber-derived-risk fusion, the
P90+Parity hybrid (unstable under verification, not promoted), Model 2
(rediscovers fixed max-pooling, no improvement), Model 4 (redundant with
Model 3). See the closure summary for why each closed.

**A combination never cleanly tested:** Model 3 + parity together. They
test different, plausibly complementary hypotheses (learned temporal
aggregation of CTG-derived risk vs. independent admission-time context),
but the only hybrid attempted (fixed P90 pooling + parity) used the fixed
aggregator, not Model 3. If external data supports both individually, this
combination is a natural, well-motivated addition to test — not before.

---

## 2. Frozen artifacts (fit once on all 547 CTU-UHB patients, not per-fold)

No prior phase persisted an actual fitted classifier object — only
cross-validated predictions on CTU-UHB. These are new, produced
specifically for this handoff (`scripts/prepare_external_validation_artifacts.py`):

| Artifact | File | Note |
|---|---|---|
| P6-final classifier | `models/external_validation_handoff/p6_final_classifier.joblib` + `p6_final_scaler.joblib` | `LogisticRegression(C=0.05)` on the 40-D state-trajectory feature vector, fit on all 547 patients |
| Model 3-final | `models/external_validation_handoff/model3_final_scorer.pt` | `AttentionScorer(in_dim=2, hidden=8)`, trained on P6-final's own (in-sample) window scores |
| Parity-final | `models/external_validation_handoff/parity_final_model.joblib` + `parity_final_scaler.joblib` | Univariate `LogisticRegression(C=1.0)` on raw parity; fusion `lambda=0.80` (fit on all 547 — the 5 per-fold lambdas were 1.3–1.5; this single-fit value is lower, expected since it's optimizing one global value against more, more homogeneous data rather than 5 separately-tuned folds — reported as-is, not re-tuned) |
| Frozen Huber ensemble | `models/continuous_clinical_huber/huber_fold_{0-4}.joblib` + `scaler_fold_{0-4}.joblib` | **Not refit.** For a new cohort, use the mean of these 5 fold models' predictions as the Huber-derived risk input feature — the same ensembling convention this project already uses elsewhere for deployment, rather than attempting a from-scratch Huber refit whose exact original fitting procedure isn't independently re-verifiable from this handoff alone |

Full provenance and exact application instructions:
`models/external_validation_handoff/manifest.json`.

**In-sample reference number** (P6-final, fit and evaluated on the same
547 patients): delivery AUROC 0.7189. This is *not* a validation metric —
it is expected to look better than the honest out-of-fold 0.6872 precisely
because it was not held out. Do not cite it as evidence of anything; it
exists only so the final-fit artifact's basic sanity (it isn't badly
broken) can be checked before shipping it.

---

## 3. Pipeline to apply to a new patient (raw CTG → prediction)

```
Raw CTG signal (FHR + UC, target 4 Hz)
        │
Preprocessing -- MUST match src/preprocessing/pipeline_clinical.py exactly:
        │  spike removal, cubic-spline gap interpolation, lowpass filter,
        │  20-minute window / 2.5-minute stride, iterative baseline
        ▼
19 clinical descriptors + 11 extended descriptors (src/knowledge/extended_features.py)
        │
        ├──► mean of 5 frozen Huber-ensemble predictions ──► Huber-derived risk feature
        │
        ▼
40-D P6 state-trajectory feature vector
  (domain severities, FIGO state, trajectory operators, occupancy
   proportions, Huber risk, EWMA risk -- exact construction in
   scripts/phase9b_deterioration_features.py + phase9c_state_trajectory.py)
        │
        ▼
p6_final_scaler.transform(...) -> p6_final_classifier.predict_proba(...)
        │
        ▼
   window-level P6 risk score r_t  (one per 20-min window, causal)
        │
        ├──► [Candidate: Model 3] causal sequence (r_t, elapsed_t) per patient
        │     -> model3_final_scorer (src/models/phase16_causal_attention.py:
        │        score_full_sequence + pooled_prediction_from_logits, using
        │        the causally-eligible prefix at whatever horizon is being
        │        evaluated)
        │     -> patient-level prediction
        │
        └──► [Candidate: Parity fusion] delivery-horizon r_T (last window)
              + parity_final_scaler/model on the patient's admission parity
              -> logit(p_fused) = logit(p_P6_final) + 0.80 * logit(p_parity_final)
              -> patient-level prediction
```

Every step through the 40-D feature vector is identical to what Phases
8–16 already do on CTU-UHB — nothing about the feature *construction* is
new. What's new is only the final classifier/attention/fusion weights,
now fit once for portability instead of living only inside a CV loop.

---

## 4. What was checked before writing this document — do not re-derive

**No existing dataset in this repository can serve as the external cohort.**
Checked directly, not assumed:

- `data/raw/fhrma/` (`CTGDL_FHEMA_metadata.csv`) — FHR morphological-analysis
  dataset. Its metadata is entirely signal-quality/characteristics (recording
  length, mean FHR, missingness rates) — **no clinical outcome field of any
  kind.** Could conceivably support a *different* future study (validating
  this project's own descriptor-extraction algorithms against expert
  morphological annotations), but cannot support acidemia-outcome validation.
- `data/raw/cardiotocography/CTG.xls` (UCI Cardiotocography, 2,126 records,
  the pre-extracted-feature dataset already cited in this project's README)
  — labels are `NSP` (Normal/Suspect/Pathologic FHR pattern) and `CLASS`
  (10-class morphology). **No umbilical pH, no acid-base outcome, no
  neonatal outcome field.** Same conclusion: unusable for this purpose.

A genuinely new, outcome-labeled cohort is required. This project has no
data-sourcing or web-browsing capability from within this session — locating
and obtaining that cohort is outside what could be done here and is the
first concrete action item for whoever picks this up.

---

## 5. Requirements for the external cohort

- Continuous intrapartum FHR + UC signal, ideally near 4 Hz (or resampleable
  to it) — matching CTU-UHB's acquisition characteristics closely enough
  that the frozen preprocessing pipeline's fixed constants (filter cutoffs,
  spike-removal thresholds, baseline algorithm parameters) remain meaningful.
  Different acquisition equipment/protocols may require re-validating (not
  re-fitting) these constants against the new signal's characteristics first.
- Umbilical arterial pH at delivery (primary: ≤7.15 threshold; secondary,
  if available: ≤7.05 for the severe endpoint) — or a documented, principled
  translation if the new cohort records a different acid-base measure.
- Patient-level linkage between signal and outcome (not window-level).
- Admission-time parity recorded, for the parity-fusion candidate specifically.
- Recording length sufficient to support at least one causal 20-minute window
  before the horizons being evaluated (delivery, ideally also ≥30m).

### Rough power consideration

CTU-UHB (110 positive / 547 total, ~20% prevalence) produced p-values in the
0.01–0.25 range for the effect sizes reported above (Δ 0.03–0.07 AUROC) —
i.e., this cohort's size sits right at the edge of reliably detecting these
effects, and several results were significant on one data split but not
another purely from this power limitation (documented explicitly in Phase
16's fold-resplit check). As a rough, non-rigorous planning guide: an
external cohort with **materially fewer than ~100 positive cases** is
unlikely to improve on CTU-UHB's own power to confirm effects of this size;
something in the 150–300+ positive-case range would meaningfully strengthen
the ability to distinguish a real +0.03–0.07 effect from noise. This is a
planning heuristic, not a formal power calculation — treat it as a floor to
aim above, not a guarantee.

---

## 6. Statistical plan (pre-register before looking at any external data)

Same discipline as every internal phase, applied to the new cohort instead
of CTU-UHB:

- **Primary confirmatory test 1:** Model 3 (external) vs. single-window P6
  (external), delivery, patient-level AUROC, paired bootstrap (B≥2000) + DeLong.
- **Primary confirmatory test 2:** Parity fusion (external) vs. single-window
  P6 (external), delivery, same statistical machinery.
- **Secondary:** both candidates at ≥30m (and ≥10/20m if the cohort supports
  those horizons), same pairing.
- **Exploratory, only if both primaries show a positive signal:** Model 3 +
  parity combined (the untested combination named in Section 1) — labeled
  exploratory regardless of outcome, per this project's own established
  rule that a hybrid is never confirmatory on its first test.
- **No refitting or recalibration against the external labels** as part of
  this validation — applying the frozen artifacts as-is and measuring what
  happens is the entire point. A recalibration/transfer-learning study is a
  legitimate, valuable, *different* follow-up, not a substitute for this one.
- **Report the result regardless of outcome.** A null or negative result
  here is real information (it would suggest the CTU-UHB signal is
  population-specific or driven by the recording-duration confound
  documented in Phase 15) and must be reported with the same weight as a
  positive one — mirroring the discipline every phase since 13 has held to.

## 7. Decision framework for the external result itself (fixed now, before data)

| Outcome | Interpretation |
|---|---|
| Significant, CI positive, magnitude comparable to CTU-UHB's own estimate (Δ ≈0.02–0.07) | Genuine external support — warrants a promotion conversation, still not automatic promotion |
| Positive but not significant | Consistent with, not stronger than, the internal evidence — continue treating as promising, not confirmed |
| Null or negative | Informative non-replication — investigate whether the recording-duration confound (Phase 15) or a population/equipment difference explains it, before concluding the internal signal was entirely spurious |

## 8. What not to do

Mirrors every prior phase's own closing discipline: do not refit the frozen
models on the external cohort and call it validation; do not expand the
candidate set (new percentiles, new fusion architectures, new covariates)
if the first result disappoints; do not report only the horizon or
candidate that happens to look best; do not treat a positive external
result as grounds to bypass a proper deployment/promotion review of Phase
12.1 itself.

## 9. Immediate next actions

1. Source an external, outcome-labeled intrapartum CTG cohort (outside this
   session's capability — the first concrete task for whoever continues this).
2. Verify the new cohort's signal format against Section 5's requirements
   before attempting to apply the frozen pipeline.
3. Write and freeze that cohort's own pre-registration (Section 6 above is
   a template, not a substitute for one specific to the actual data obtained).
4. Apply the artifacts in `models/external_validation_handoff/` exactly as
   documented in `manifest.json` — no refitting.
5. Report the result with the same rigor, and the same willingness to report
   a null, as every phase in `reports/phase13_16_closure_summary.md`.
