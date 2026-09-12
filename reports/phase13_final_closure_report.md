# Phase 13 — Final Closure Report and Hybrid Verification Addendum

**Status: Phase 13 is now CLOSED.** Written 2026-09-11. This is the
authoritative closing document for Phase 13 and supersedes the interim
[reports/phase13_survivor_gate_report.md](phase13_survivor_gate_report.md)
as the final record — that report is retained unmodified, not overwritten,
per the file-discipline requirement below.

**Phase 12.1 is untouched.** Nothing in this document, or in any Phase 13
script, modified `data/processed_clinical/`, `results/phase8_rolling/`,
`results/phase9c_state_trajectory/`, the frozen Huber models, the locked
P6 model definition, or any file on the `final_synthesis_models` branch's
locked history. All Phase 13 artifacts are new, additive files under
`docs/phase13_protocol.md`, `scripts/phase13_*.py`, `src/aggregation/`,
`src/evaluation/`, and `results/phase13/`. Nothing has been committed.

---

## A. Hybrid specification (P90 + Parity)

Exact, as implemented in `scripts/phase13_hybrid_p90_parity.py` and governed
by `docs/phase13_protocol.md` Section 14.

**P90 definition.** For patient *j* at evaluation horizon *h*, the causally
eligible window pool is `{i : t_i ≥ h}` (windows already observed by the
time horizon *h* is reached; all windows if `h=0`/delivery). The patient
score is the **plain, unweighted 90th percentile** (`q=0.90`, fixed by
Phase 3's historical convention, not swept) of P6's own frozen window-level
probabilities over that pool, via the standard weighted-percentile
estimator with uniform weights (`src/aggregation/recency_weighted_p90.py`,
`half_life_min=np.inf`).

**Parity representation.** Raw `parity` value (integer count, 0–7) from
`clinical_metadata.csv`, standardized and passed through a per-fold
univariate `LogisticRegression(C=1.0)` fit on that fold's training patients
only, producing `p_parity` for every patient (in-fold for training patients,
out-of-fold for held-out patients) — identical construction to Option D.

**Fusion equation:**

```
logit(p_hybrid) = logit(P90_pooled_P6) + lambda * logit(p_parity)
```

This is a base-swap-plus-fusion, not a three-term sum: `P90_pooled_P6`
replaces `P6_single_window` as the base being fused with parity; it is not
an additional logit term alongside it.

**Fold-specific training.** For the 5-fold CV metric: for each outer fold,
the parity model is fit on that fold's training patients; `lambda` is
selected by maximizing training-fold AUROC over a fixed 30-point grid
`linspace(0.1, 3.0, 30)` (Phase 4's original fusion-selection rule,
`src/evaluation/phase13_common.py:select_lambda_trainfold`); both are then
applied, frozen, to that fold's held-out patients. For the held-out test
metric: the parity model and `lambda` are fit once on all 464 train+val
patients and applied once, frozen, to the 83 test patients. No parameter is
ever selected using the partition it is reported on.

**Evaluation horizons.** Delivery (0m) and ≥30m are the two horizons this
hybrid's numbers are discussed at, matching every other Phase 13 branch;
h=10/20 were also computed and are in the raw result file for completeness
but are not discussed further here.

**Patient-level inference / causal constraints.** All scoring is
patient-level (never window-level independence assumptions). The P6
window-level probabilities being pooled were themselves produced by a model
trained under Phase 8/12.1's certified causal boundary (prediction at time
*t* uses only CTG ≤ *t*); this hybrid adds no new signal source and
therefore inherits that boundary unchanged. No pH, delivery time, or any
post-window information enters either the P90 pooling or the parity model
at inference time.

---

## B. Raw hybrid result — exploratory only, not a confirmed effect size

| Horizon | CV: P6→P90→Hybrid | CV Δ(hybrid−P90), p | CV Δ(hybrid−P6orig), p | Test: P6→P90→Hybrid | Test Δ(hybrid−P90), p | Test Δ(hybrid−P6orig), p |
|---|---|---|---|---|---|---|
| Delivery | 0.6872→0.7254→0.7373 | +0.0119, p=0.425 | +0.0501, p=0.073 | 0.6497→0.6916→0.7549 | +0.0633, **p=0.001** | +0.1052, **p=0.031** |
| ≥30m | 0.5828→0.6117→0.6418 | +0.0302, p=0.290 | +0.0590, p=0.068 | 0.6078→0.7014→0.8048 | +0.1034, **p=0.007** | +0.1970, **p=0.004** |

**These test-column numbers must not be read as reliable effect estimates,
for two independently-verified reasons (Section C, D below).** They are
reported here in full, not suppressed, because Section 5's discipline
requires the raw finding to be on record — but every subsequent section of
this closure exists specifically to explain why they are not trustworthy as
stated.

---

## C. Lambda-boundary diagnostic

| Horizon | CV lambdas (5 folds) | Test lambda | Parity-only lambdas at same horizon (reference) |
|---|---|---|---|
| Delivery | 1.0, 1.3, 1.1, 1.2, 0.9 | 0.9 | 1.5, 1.4, 1.5, 1.4, 1.3 |
| ≥10m | 2.0, 2.0, 2.4, 2.0, 2.0 | 2.6 | — |
| ≥20m | 2.0, 2.6, 2.6, 2.1, 2.1 | **3.0** | — |
| ≥30m | 2.3, **3.0**, 2.9, 2.2, 2.2 | **3.0** | — |

Grid: `linspace(0.1, 3.0, 30)`, fixed before this hybrid was run
(Section 14.2), **not expanded post hoc** at any point during or after
this diagnostic.

At ≥20m and ≥30m, multiple folds and the test-partition selection landed
exactly on the grid's upper bound (3.0). At ≥10m, selections (2.0–2.6) sit
close to the same boundary. This is qualitatively different from the
delivery-horizon lambdas (0.9–1.3) and from the parity-only fusion's own
lambdas at delivery (1.3–1.5) — both comfortably interior to the grid.

A hyperparameter search that repeatedly saturates its boundary indicates
the training-fold objective wants a larger value than the (fixed,
pre-registered) grid permits — i.e., the selection procedure is
under-constrained at those horizons. This is a standard overfitting/
instability signature: it means the fold-specific fusion is leaning as
hard as it is permitted to on a single weak univariate predictor (parity's
own standalone AUROC was 0.411 on the full cohort), which is far more
consistent with the optimizer chasing fold-specific noise than with a
genuinely strong, generalizable complementary signal. Per Section 14.2,
the correct response to this finding is to document it, not to widen the
grid and re-run — doing the latter would be exactly the kind of post-hoc
hybrid fishing this closure is meant to foreclose.

---

## D. Fold-resplit robustness (delivery horizon)

Complete results, four independent patient-grouped 5-fold re-splits
(`StratifiedKFold`, seeds 11/22/33/44) plus the canonical partition,
identical hybrid procedure applied to each:

| Fold source | Hybrid AUROC | Δ vs. P90 alone | Δ vs. original P6 |
|---|---|---|---|
| Canonical | 0.7373 | +0.0119 | +0.0501 |
| Resplit seed 11 | 0.7131 | **−0.0123** | +0.0259 |
| Resplit seed 22 | 0.7154 | **−0.0099** | +0.0283 |
| Resplit seed 33 | 0.7183 | **−0.0070** | +0.0312 |
| Resplit seed 44 | 0.7315 | +0.0061 | +0.0443 |

**Key analysis:**

- **Hybrid vs. original P6**: positive in all five fold assignments
  (+0.026 to +0.050). This is the one part of the result that is directionally
  stable — the *combined system*, however it is attributed internally, does
  not fall below the original locked P6 baseline under any tested partition.
- **Hybrid vs. P90 alone**: **sign-unstable** — negative in 3 of 4
  independent resplits, barely positive in the 4th, positive only in the
  canonical split. This is the direct test of "does parity add anything on
  top of P90-pooling," and it does not survive independent re-partitioning.
- **Conclusion**: the claim "parity contributes information beyond P90" is
  **not supported** by this evidence. The much larger effect sizes seen in
  the single held-out test partition (Section B) are best explained by
  small-sample interaction with the boundary-saturated lambda selection
  (Section C) rather than a real effect of that magnitude — the fold-resplit
  CV evidence (a more reliable, larger-sample source) caps the plausible
  total effect at roughly +0.03 to +0.05 AUROC, not +0.10 to +0.20.

---

## E. Final verdict

| Branch | Status |
|---|---|
| A — Recency-weighted P90 | **CLOSED.** No significant gain over plain P90 at any horizon; CV and test disagree in sign at delivery and h=10. |
| B — Stride (2.5min vs 1.0min) | **CLOSED.** No significant AUROC gain either horizon convention. Operational trend is a trade-off (better lead time/detection, materially worse false-alert rate), not a clean improvement. |
| C — Huber fusion | **CLOSED.** Null at both confirmatory horizons, consistent with `X_19 ⊂ P6` redundancy. `risk_prob_proxy` correctly understood as a Huber-*derived risk proxy* (sigmoid transform of predicted pH), not a calibrated probability. |
| D — Parity fusion | **PROMISING / NOT CONFIRMED.** Test-significant at delivery (p=0.025); CV positive but CI crosses zero (p=0.172). Passed seed-sensitivity, fold-resplit, recording-duration-confound, and conditional-analysis checks. Best-supported single addition in the program, but does not meet the pre-registered 🟢 bar (CV CI>0). |
| E — Plain P90 pooling | **PROMISING / NOT CONFIRMED.** Positive in all four CV/test × delivery/≥30m comparisons; largest and most directionally consistent point estimates in the program; none individually significant. Pre-registered per `docs/phase13_protocol.md` Section 13 after being noticed during Option A's exploratory work. |
| P90 + Parity hybrid | **EXPLORATORY / INCONCLUSIVE.** Not a confirmed improvement; not a definitive failure. The combined system beats original P6 by a fold-resplit-stable +0.03 to +0.05 at delivery, but parity's incremental contribution beyond P90 is sign-unstable, and the fusion coefficient repeatedly saturated its pre-registered grid boundary. The strong initial held-out-test numbers are considered unstable and are explicitly **not** to be used as an effect-size estimate. **Not promoted to survivor or final-model status.** |

---

## Hypotheses preserved for future work (not current conclusions)

**Hypothesis 1.** Admission parity may provide independent contextual
information about fetal acidemia risk beyond CTG-derived physiology.
*Status: suggestive, multiply-robustness-checked, not confirmed.*

**Hypothesis 2.** The distribution of sequential causal P6 window-level
risk scores may contain clinically useful temporal information that is
lost when a patient's recording is reduced to one selected window.
*Status: suggestive exploratory signal (Option E), not confirmed.* This is
flagged as the more scientifically important of the two open threads: it
concerns patient-level aggregation methodology itself, not an added
information source, and its effect sizes (Section 13.5 of the protocol)
were the largest and most horizon-consistent observed anywhere in Phase 13.

Neither hypothesis is to be treated as confirmatory evidence. The P90 point
estimate in particular was already observed once (during Option A's
exploration) before being pre-registered as Option E — any future test of
it needs an independently fresh design, not a re-report of this number.

---

## Recommended Phase 14 direction (documented only — not executed)

**"Patient-level temporal aggregation study."** A dedicated, freshly and
independently pre-registered protocol comparing, on frozen P6 window-level
scores:

1. P6's existing single-window selection (current production convention)
2. Plain P90 pooling
3. Maximum window score
4. Mean window score
5. Other robust distributional summaries, only if scientifically justified
   in advance of seeing any new data

**Research question:** does preserving the distribution of sequential
causal CTG risk scores improve patient-level discrimination compared with
selecting a single representative window?

This must be pre-registered independently of Phase 13 precisely because the
P90 point estimate has already been observed once here — using it to
motivate Phase 14 is legitimate hypothesis generation; using it as Phase
14's own confirmatory evidence would not be. No Phase 14 code has been
written and none should be until that separate protocol is frozen.

---

## Reproducibility

| Artifact | Path |
|---|---|
| Frozen protocol (all amendments) | `docs/phase13_protocol.md` |
| Interim survivor-gate report (unmodified, retained) | `reports/phase13_survivor_gate_report.md` |
| This closure report | `reports/phase13_final_closure_report.md` |
| Shared horizon/fusion primitives | `src/evaluation/phase13_common.py` |
| P90/recency pooling primitives | `src/aggregation/recency_weighted_p90.py` |
| 13.0A audit (`python scripts/phase13_evaluation_audit.py`) | `results/phase13/audit/` |
| Option A (`python scripts/phase13_recency_p90.py`) | `results/phase13/recency_p90/` |
| Option B (`python scripts/phase13_stride.py`) | `results/phase13/stride/` |
| Option C (`python scripts/phase13_huber_fusion.py`) | `results/phase13/huber/` |
| Option D (`python scripts/phase13_parity_robustness.py`) | `results/phase13/parity/`, `results/parity_fusion/` |
| Option E (`python scripts/phase13_p90_pooling.py`) | `results/phase13/p90_pooling/` |
| Hybrid (`python scripts/phase13_hybrid_p90_parity.py`) | `results/phase13/hybrids/` |
| Master survivor table | `results/phase13/statistics/master_survivor_table.csv` |

All scripts read only from frozen, already-existing Phase 8–13 artifacts
(`data/processed_clinical/folds.json`, `results/phase8_rolling*/`,
`results/phase9c_state_trajectory*/`, `results/phase13/audit/p6_predictions.npz`)
and the raw `clinical_metadata.csv`; none retrains P6 or the Huber models.
Re-running any script above reproduces its cited numbers exactly (all use
fixed `random_state=42` / `seed=42` throughout).

---

## Closing confirmations

- **Phase 12.1 unchanged.** No file under the locked pipeline was read
  destructively or written to. The locked numbers (delivery ≈0.6872, ≥30m
  ≈0.5857 under the original horizon convention) are unaltered and are not
  superseded by anything in this document.
- **No further hybrid searches were performed.** No new lambda grid, no
  additional covariates, no alternative parity encodings, no max/mean/median
  combinations with parity, no new fusion architectures, no threshold
  tuning, no P90-percentile tuning. The lambda-boundary finding (Section C)
  was documented, not resolved by re-running with a wider grid.
- **Phase 13 is closed as of this report.** Any further work (Phase 14 or
  otherwise) requires its own, separately pre-registered protocol.
