# Phase 13 — Controlled Post-Lock Information Extension

**Frozen 2026-09-11, before any Phase 13 experiment code was written or run.**
This protocol governs everything under `scripts/phase13_*.py` and
`results/phase13/`. Nothing in this document changes without an explicit,
versioned amendment noted below it — the same discipline `PROTOCOL.md` /
`src/training/protocol.py` already enforce elsewhere in this project.

> **Governance rule.** Phase 13 is an exploratory extension of the locked
> Phase 12.1 submission. The Phase 12.1 model, results, claims, and
> repository lock remain unchanged. Phase 13 cannot retroactively alter the
> primary thesis results. Any Phase 13 improvement must be independently
> validated before being presented as a final model. All Phase 13 artifacts
> are uncommitted exploratory work until explicitly promoted.

## 1. Scientific objective

Phase 13 does not ask "can we find a model with higher AUROC." It asks:

> Does P6 leave useful predictive information unexploited in three specific,
> orthogonal dimensions — temporal recency, continuous acid-base severity,
> and admission-time maternal context?

## 2. Frozen cohort

547 patients, 110 primary positives (pH ≤ 7.15), 41 severe positives
(pH ≤ 7.05), 8,517 canonical rolling windows, CTU-UHB, existing
preprocessing (`data/processed_clinical/`, `data/phase1_candidates/`,
`results/phase8_rolling/`, `results/phase9c_state_trajectory/`). Frozen
5-fold patient-grouped assignment: `data/processed_clinical/folds.json`.
Frozen held-out test partition: `data/processed_clinical/test_dataset.pt`
(83 patients).

## 3. Endpoints

Primary: `y_primary = 1[pH ≤ 7.15]`. Secondary: `y_severe = 1[pH ≤ 7.05]`.
No other endpoint is introduced in Phase 13.

## 4. Evaluation horizons — two conventions, kept explicitly separate

**Existing convention** (`get_patient_scores_at_horizon`, used for every P6
number reported this session, including the locked 0.6872/0.5857):
first chronological window satisfying `t_i ≥ h`. For `h > 0` this is close
to "the patient's earliest window," not the window nearest `h`.

**Corrected convention** (`get_patient_scores_at_horizon_corrected`, new):

```
i* = argmin_i |t_i - h|   subject to   t_i ≥ h
```

Both are computed side by side for every Phase 13 experiment. The corrected
convention is the primary scientific interpretation for any horizon-specific
claim (≥10/20/30 min); the existing convention is retained as a
compatibility/sensitivity check against every historical number in this
project. Delivery (`h=0`) is unaffected by this distinction (both
conventions select the last window).

## 5. Statistical contract

- Patient-level evaluation only; windows are never treated as independent.
- Canonical 5-fold patient-grouped CV (`folds.json`), unchanged.
- No relabeling, no experiment-specific preprocessing, no new causal
  boundary.
- Paired comparison on identical patients: `ΔAUROC = AUROC_new − AUROC_baseline`.
- Uncertainty: patient-level clustered bootstrap (B=2000) — never window-level
  resampling (already known from this project's own audit to understate
  variance ~2.7x). Report AUROC, 95% CI, ΔAUROC, Δ 95% CI, bootstrap p,
  DeLong p, AUPRC.
- Operational metrics where applicable: sensitivity, specificity, PPV, NPV,
  FAR/hour, median warning time, % detected ≥10/20/30 min.

## 6. Hyperparameter rule

> **No hyperparameter may be selected using the final evaluation/test
> patients, and no hyperparameter may be selected using the same outer-CV
> fold it is subsequently reported on.**

Pattern for anything requiring a parameter (recency half-life, any fusion λ):
`outer training patients → parameter selection → frozen parameter → outer
validation/test`. This is written into the protocol because the first
information-density-weighting implementation violated exactly this rule
(selected β/span on the same 5 outer folds later reported as results) before
it was caught and fixed with nested selection.

## 7. Baseline reconciliation (locked into this protocol so it is never
re-litigated as if the numbers contradict each other)

| Label | AUROC (delivery) | What it is |
|---|---|---|
| `P6_canonical` | 0.6872 | Unweighted P6 (`sample_weight=None`), matches published Phase 12.1 |
| `P6_patient_normalized` | 0.6843 | P6 with per-patient weight normalization applied — used deliberately as the stride experiment's baseline arm so windows-per-patient differences between stride arms wouldn't bias that comparison |

Both are legitimate; they answer different questions. `P6_canonical` (0.6872)
is the reference baseline for every Option A/C/D comparison in this
document. `P6_patient_normalized` is retained only for stride, where both
arms use it symmetrically.

## 8. Experiment branches

| Branch | Changes | Question |
|---|---|---|
| A — Recency-weighted P90 | Patient-level aggregation | Does recency of window-level risk add information beyond the upper-tail magnitude? |
| B — Stride | Window sampling density | Does temporal resolution improve trajectory/state estimation? |
| C — Huber-derived risk fusion | Adds `risk_prob_proxy` | Does continuous acid-base severity add information beyond P6's binary representation? |
| D — Parity fusion | Adds admission-time covariate | Does independent maternal context add information CTG cannot capture? |

No new endpoint, no shared trainable trunk, no retraining of P6 or the
frozen Huber models in any branch.

## 9. Survivor gate (predefined before any result is seen)

A branch is a:

- **🟢 Survivor** — `ΔAUROC > 0` with `95% CI(Δ) > 0` (strong evidence), OR
  a meaningful operational improvement (warning time / ≥20/30-min detection
  / FAR) with no causal-integrity violation.
- **🟡 Interesting** — positive point estimate, CI crosses zero, but a
  believable operational trend. Labeled *promising*, never *superior*.
- **🔴 Closed** — CI spans zero and no operational advantage.

Only 🟢/🟡 branches are eligible for hybrid fusion. 🔴 branches are closed,
full stop — not carried into hybrids "because they're available."

## 10. Hybrid rules

Hybrids are `logit(p_fused) = logit(p_P6) + Σ λ_k·logit(p_k)` over surviving
branches only, λ selected per fold via the same nested, train-only rule as
§6. Pre-authorized candidate (contingent on both surviving independently):
`P6 + Huber + Parity` — different information sources (continuous severity
vs. admission-time context), the cleanest complementarity hypothesis. Every
other pairing (Recency+Parity, Recency+Huber, all three) is admissible if
its components independently survive, but is exploratory, not confirmatory.
Stride does not enter hybrids unless it independently shows an operational
(not necessarily AUROC) benefit under the corrected horizon — in which case
it becomes a preprocessing configuration for the strongest surviving model,
not a fourth fused logit term.

## 11. Confirmatory vs. exploratory classification (fixed before results)

**Confirmatory** (one test each): `P90_recency − P90_plain` (Option A's own
primary comparison, not vs. P6); `P6+Huber − P6`; `P6+Parity − P6`;
`P6(1.0min) − P6(2.5min)` under patient-normalized weighting (already run).

**Exploratory**: everything else — secondary horizons, severe endpoint, any
hybrid (including the pre-authorized one), operational-metric comparisons,
post-hoc attribution decompositions.

## 12. Amendments

**2026-09-11 — Option E added (P90 pooling vs. single-window selection).**
Emerged from Option A's exploratory secondary comparison
(`results/phase13/recency_p90/`: plain P90 of P6's own window scores scored
+0.0382 CV / +0.0419 test above P6's production single-window score at
delivery — the largest point estimate anywhere in the Phase 13 program, but
not pre-registered, not significant, and not consistent across all
horizons). See Section 13 for the full specification.

**Honesty note on pre-registration order.** The underlying point estimates
were already visible from Option A's exploratory pass before this amendment
was written — this is not a blind pre-registration in the strictest sense.
What is fixed here, before any *new* number is computed, is: the exact test
specification, which comparisons count as confirmatory vs. exploratory, the
horizons, the pooling operators, and the statistical protocol — and a
commitment to report whatever the confirmatory result is. No new model
fitting or hyperparameter search happens in Option E (P90/max/mean are all
parameter-free pooling operators over the already-frozen P6 window-level
scores computed in 13.0A) — the only thing this branch does that wasn't
already done is compute paired bootstrap/DeLong statistics against the
correct baseline and check consistency across all four horizons, which
Option A's secondary comparison did not do rigorously.

## 13. Option E — P90 pooling vs. single-window selection

### 13.1 Scientific question

Independent of any new information source or reweighting scheme: does
**how P6 aggregates evidence across a patient's windows** — single-window
horizon-selection (current production convention) vs. P90 pooling across
all causally-eligible windows (Phase 3's historical convention) — change
discrimination, holding the underlying window-level P6 model fixed?

### 13.2 What's frozen

The exact window-level P6 predictions already computed and cached in
`results/phase13/audit/p6_predictions.npz` (13.0A) — no retraining, no new
LogisticRegression fit. Same cohort, folds, labels, causal window-eligibility
rule (`t_i >= h`) as every other branch.

### 13.3 Confirmatory test

`AUROC(P90_pooled) − AUROC(P6_single-window, corrected convention)`, at
**delivery (h=0)** and **≥30m**, patient-grouped CV and held-out test, paired
bootstrap (B=2000) + DeLong. `q=0.90` is fixed by the project's own Phase 3
convention, not swept — introducing no new hyperparameter-selection surface.

### 13.4 Exploratory comparisons (not confirmatory)

- Same pooling comparison at h=10 and h=20.
- Max-pooling and mean-pooling as additional aggregation operators (also
  parameter-free), included because `src/training/protocol.py`'s own
  documented history found max beat both P90 and mean for a different
  (signal-only) model — worth having as context for whether that ordering
  replicates here, without treating it as a new hyperparameter search.

### 13.5 Survivor criteria

Identical to Section 9, applied uniformly: 🟢 requires `ΔAUROC>0` with
`95% CI(Δ)>0`; 🟡 is a positive point estimate with CI crossing zero,
labeled *promising*, never *superior*; 🔴 otherwise. This is a genuinely new
confirmatory test, not a re-report of Option A's exploratory number — it
will be judged by the same bar as A–D.

## 14. Hybrid 1 — P90-pooled P6 + Parity (2026-09-11)

With two branches now at 🟡 (D — parity; E — P90 pooling), the pairing
between them becomes eligible under Section 10's hybrid rule. This was not
the pre-authorized combination (that was `P6+Huber+Parity`, not reached
since Huber closed) — this is a new pairing, pre-registered here before it
is run.

**Important tiering note, stated explicitly so it is not blurred later:**
per Section 11, *every* hybrid is exploratory, not confirmatory, regardless
of pre-registration. Pre-registering this test fixes its specification
before results are computed, so nothing about horizon choice or metric gets
selected after the fact — but it does not upgrade the test to confirmatory
status. A hybrid combines two components' researcher-degrees-of-freedom
(which two branches, in what combination), which A–E's individual tests did
not carry.

### 14.1 Scientific question

Does the aggregation upgrade (Option E) and the independent admission-time
covariate (Option D) provide complementary value when combined — i.e., is
"how P6 pools its own evidence" orthogonal to "information CTG cannot see,"
the way the theory motivating both branches separately predicted?

### 14.2 Construction — this is a base-swap + fusion, not a 3-term sum

`P90_pooled_P6` (Option E, `q=0.90` fixed, the base score) replaces
`P6_single_window` as the base being fused with parity — it does not add a
third logit term alongside it. Formally:

```
logit(p_hybrid) = logit(P90_pooled_P6) + lambda * logit(p_parity)
```

`p_parity` is the same per-fold univariate LogisticRegression(parity) used
in Option D, unchanged. `lambda` selected by the identical Phase 4
train-fold-AUROC-maximization rule (Section 6) — no new hyperparameter
surface beyond what D already validated. No new model is fit on raw
features; this reuses the frozen `p6_predictions.npz` window-level scores,
Option E's pooling function, and Option D's parity-fusion machinery as-is.

### 14.3 Registered comparisons (both exploratory, per 14's tiering note)

- **(a) Attribution**: `AUROC(hybrid) − AUROC(P90_pooled_P6 alone)` — does
  parity add anything on top of the already-better aggregation?
- **(b) Total effect**: `AUROC(hybrid) − AUROC(P6_single_window)` — the
  locked 0.6872/0.6497 starting point — does the combination beat where
  Phase 13 started, end to end?

Both at delivery and ≥30m (h=10/20 reported for context only, as elsewhere).
Same statistical protocol as every other branch: patient-grouped 5-fold CV,
held-out test, paired bootstrap (B=2000) + DeLong.

**2026-09-11 — PHASE 13 CLOSED.** Final verdict, lambda-boundary diagnostic,
and fold-resplit robustness check for this hybrid are recorded in
[reports/phase13_final_closure_report.md](../reports/phase13_final_closure_report.md),
the authoritative closing document. No further hybrid searches, lambda-grid
expansions, covariate additions, alternative encodings, or fusion
architectures are to be run under this protocol. Any further work requires
a new, independently pre-registered protocol (see the closure report's
Phase 14 recommendation).

### 14.4 What would make this worth carrying forward

Given its exploratory tier, no CI-based criterion here should be read as a
final validation the way Sections 9/13.5 are for a first-order branch — a
positive result here is a lead for a dedicated, independently-designed
confirmatory follow-up (its own pre-registration, ideally with a fresh look
at whether the effect replicates), not a result to promote directly to
"final candidate." This section exists to keep that expectation explicit
before the numbers are seen.
