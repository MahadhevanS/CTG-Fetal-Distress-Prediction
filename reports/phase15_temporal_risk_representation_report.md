# Phase 15 — Temporal Risk Representation: Exploratory Methodology Study

Written 2026-09-11, executed exactly per the pre-registered
[docs/phase15_temporal_risk_representation_protocol.md](../docs/phase15_temporal_risk_representation_protocol.md),
frozen before this script ran. Phase 12.1 is untouched and authoritative
throughout; nothing here is a confirmatory claim (Section 2 of the
protocol — no result in this report is described as "validated,"
"replicated," or "confirmed").

## A. Motivation

Phases 13–14 left one thread open: a P90 pooling of P6's own window-level
scores showed a consistent, if unconfirmed, positive pattern relative to
single-window selection. Rather than re-test P90 a third time on the same
cohort, this phase asks the broader question that P90 was only one answer
to — does the *sequence* of causal risk scores a patient's recording
produces carry information that reducing it to one window discards, and if
so, what about that sequence (its magnitude, its distribution, its
direction) is doing the work?

## B. Frozen protocol

See the protocol document in full. Summary of what was fixed before this
run: five core representations (single-window, P90, max, mean, median, none
added or dropped after seeing results), four temporal descriptors with a
threshold defined as a training-fold population statistic (75th percentile
of pooled training-window scores, never outcome-optimized), two reported
horizons (delivery, ≥30m; ≥10/20m computed for completeness only), patient-
level bootstrap + DeLong throughout, and a decision framework that caps
every representation at "hypothesis-generating" regardless of numbers,
because the cohort is unchanged from Phases 1–14.

## C. Data and causal verification

Confirmed before the protocol was written (protocol Section 0): the frozen
window-level P6 scores (`results/phase13/audit/p6_predictions.npz`) align
row-for-row with `results/phase8_rolling/rolling_predictions.csv` (8,517/8,517
exact match on patient ID and `time_before_delivery_min`); no NaN; test-set
predictions populate exactly the 1,274 windows of the 83 held-out patients
and are zero elsewhere; `folds.json` unchanged (547 patients, 5 folds). The
underlying causal boundary on these scores was certified in Phase 8
(synthetic future-perturbation test) and the scores were independently
bit-reproduced in Phase 13.0A. No model was retrained anywhere in Phase 15.

## D. Aggregation definitions

| Code | Definition | Estimator note |
|---|---|---|
| Single-window | `get_patient_scores_at_horizon_corrected` | Unchanged Phase 13/14 convention |
| P90 | `numpy.percentile(scores, 90)` | **Phase 14's estimator**, not Phase 13's weighted-quantile function — chosen deliberately for continuity, documented per protocol Section 4 |
| Max | `max()` over eligible pool | — |
| Mean | `mean()` over eligible pool | — |
| Median | `numpy.percentile(scores, 50)` | — |
| Persistence(τ) | fraction of eligible windows > τ | τ = 75th percentile of pooled **training-fold** window scores, fit per fold |
| Longest run(τ) | longest consecutive chronological run > τ | same τ |
| Slope | OLS slope of score vs. chronological window index | — |
| Recent − earlier | mean(later half) − mean(earlier half) of eligible pool | — |

Eligible pool at horizon *h*: windows with `time_before_delivery ≥ h`.

## E. Results table — core representations

**Delivery and ≥30m** (the two reported horizons; neither promoted to sole
primary status per protocol Section 5):

| Representation | Delivery CV AUROC | Δ vs SW (p) | Delivery Test AUROC | Δ vs SW (p) | ≥30m CV AUROC | Δ vs SW (p) | ≥30m Test AUROC | Δ vs SW (p) |
|---|---|---|---|---|---|---|---|---|
| Single-window | 0.6872 | reference | 0.6497 | reference | 0.5828 | reference | 0.6078 | reference |
| P90 | 0.7178 | +0.0307 (.225) | 0.6622 | +0.0125 (.797) | 0.6090 | +0.0261 (.216) | 0.6979 | +0.0900 (.189) |
| Max | 0.7210 | +0.0338 (.140) | 0.7130 | +0.0633 (.080) | 0.6126 | +0.0298 (.176) | 0.7130 | +0.1052 (.108) |
| Mean | 0.6926 | +0.0054 (.865) | 0.6488 | −0.0009 (.956) | 0.5956 | +0.0127 (.430) | 0.6631 | +0.0553 (.374) |
| Median | 0.6536 | −0.0336 (.283) | 0.6061 | −0.0437 (.553) | 0.5824 | −0.0004 (.951) | 0.6515 | +0.0437 (.548) |

**≥10m and ≥20m** (exploratory only — CV and test *disagree in sign* for
every representation at both horizons, e.g. P90 at ≥10m: CV −0.0423 (p=.055)
vs. test +0.0561 (p=.367)). Full table with CI/DeLong: `results/phase15/phase15_core_representations.csv`.

### Persistence/structure descriptors (delivery, ≥30m)

| Descriptor | Delivery CV Δ vs SW (p) | Delivery Test AUROC | ≥30m CV Δ vs SW (p) | ≥30m Test AUROC |
|---|---|---|---|---|
| Persistence(τ) | +0.0023 (.953) | 0.7059 | −0.0007 (.969) | 0.7228 |
| Longest run(τ) | −0.0021 (.922) | 0.6849 | +0.0022 (.939) | 0.7077 |
| Slope | **−0.0729 (.012)** | 0.4661 | **−0.1131 (.004)** | 0.4657 |
| Recent − earlier | **−0.0751 (.016)** | 0.4492 | **−0.1089 (.008)** | 0.4398 |

**These are the only formally significant results anywhere in this study —
both negative.** Slope and recent-minus-earlier (pure trend/direction
descriptors, blind to magnitude) discriminate significantly *worse* than
single-window selection at both horizons, and their test AUROCs fall below
0.5 (worse than chance on the held-out set). Persistence and longest-run
(threshold-crossing descriptors) are statistically indistinguishable from
single-window in both directions.

## F. Test-set results

Reported in full in Sections E above — not used to select a winner.
Directionally, test agrees with CV for P90/Max/Mean at delivery and ≥30m
(all positive except Mean's near-zero delivery point estimate), disagrees
with CV at ≥10m/≥20m for all four core representations, and agrees with CV
on the negative verdict for slope/recent-minus-earlier at both reported
horizons.

## G. Operational results (delivery / ≥30m, 80% target sensitivity)

| Representation | Median lead (min) | % detected ≥20min | % detected ≥30min | False alert rate (delivery) |
|---|---|---|---|---|
| Single-window | 32.5 | 64.5% | 53.6% | 56.8% |
| P90 | 17.5 | 49.1% | 39.1% | 52.9% |
| Max | **12.5** | **38.2%** | **25.5%** | 51.5% |
| Mean | 40.0 | 75.5% | 67.3% | 58.6% |
| Median | **40.0** | **82.7%** | **72.7%** | 62.9% |

**This inverts the AUROC ranking, and it matters.** Max has the best
delivery-horizon discrimination and the *worst* lead time and detection
rate; median has the worst discrimination and the *best* lead time and
detection rate. Mechanistically plausible explanation, offered descriptively
and not as an established finding: max/P90 are dominated by the single
highest-risk window in a recording, which tends to occur close to actual
deterioration (and thus close to delivery), so their calibrated alert
threshold is crossed late; mean/median are smoother, lower-magnitude signals
that cross their own (lower) calibrated threshold earlier in the recording.
**A representation that discriminates better is not automatically the one
that warns earlier** — exactly the caution the protocol's Section 8 was
written to surface.

## H. Recording-duration analysis (delivery horizon)

| Representation | r(value, n_windows) | r(Δ vs SW, n_windows) |
|---|---|---|
| P90 | −0.055 (p=.198) | **−0.190 (p<.001)** |
| Max | +0.007 (p=.864) | **−0.123 (p=.004)** |
| Mean | **−0.148 (p=.001)** | **−0.232 (p<.001)** |
| Median | **−0.177 (p<.001)** | **−0.232 (p<.001)** |

**Every representation's advantage over single-window shrinks (or reverses)
as recording length grows.** The pooling benefit, where present, is
concentrated in patients with *shorter* recordings (fewer eligible windows),
not evenly distributed across the cohort. Two candidate explanations, both
consistent with this data and not distinguished by it: (1) genuinely more
informative aggregation for shorter, less-observed recordings, where a
single window is a comparatively poor representative; or (2) a small-sample
statistical-estimation artifact (percentile/extremum estimates are noisier
with fewer windows, which can inflate apparent differences from a single
fixed comparator without reflecting real information). This is reported as
an open confound, not resolved — no duration adjustment was applied to any
AUROC above (protocol Section 9 explicitly rules that out to avoid a new
source of post-hoc tuning).

## I. Interpretation

**Ordering.** Across both reported horizons, Max ≥ P90 > Mean > Median in
directional strength — a clean, consistent pattern (not individually
significant) supporting the sub-hypothesis that acidemia-related signal
concentrates in a subset of high-risk windows rather than being evenly
distributed across a recording (Section 13/18 framing: "if max/P90
outperform mean/median..."). Median being the single worst discriminator at
every horizon — sometimes below single-window itself — is the cleanest
piece of evidence for this: "typical" risk specifically discards what
matters.

**Direction vs. magnitude.** The one place this study found statistical
significance was negative: slope and recent-minus-earlier, which encode
*only* trajectory direction and discard magnitude, are significantly worse
than single-window selection at both horizons, with test AUROCs below
chance. This narrows Hypothesis 2 usefully: the evidence here points toward
**peak/upper-tail magnitude**, not trajectory shape, as the more promising
candidate mechanism. A patient uniformly at high risk the whole recording
and a patient uniformly at low risk both have slope≈0 — exactly the
information max/P90 do NOT discard and magnitude-blind descriptors do.

**Horizon instability.** ≥10m/≥20m show CV and test disagreeing in sign for
every representation — a pattern already seen in Phase 13's Huber analysis
at the same horizons, likely reflecting genuinely small/volatile eligible-
window pools at those intermediate points rather than anything specific to
this study. Reported, not resolved.

**AUROC vs. operational utility diverge sharply** (Section G) — this is
arguably the single most actionable finding in the report for anyone
designing a deployed alerting rule: the representation that best separates
classes is not the one that warns earliest, and choosing an aggregator
requires deciding what trade-off is being made, not just reading an AUROC
column.

## J. Final status — classification per protocol Section 11

| Representation | Category |
|---|---|
| Max | **Category 2** — hypothesis-generating. Strongest, most consistent directional signal of any core representation, but not significant, materially confounded by recording duration, and operationally the *worst* aggregator by lead time/detection despite the best AUROC. |
| P90 | **Category 2** — hypothesis-generating. Same pattern as Max, slightly weaker; consistent with (not independent of) Phase 13/14's own prior characterization. |
| Mean | **Category 1** — no evidence of useful additional information. Point estimates near zero at delivery, inconsistent sign elsewhere. |
| Median | **Category 1**, trending negative. Worst or near-worst discriminator at every horizon; best operational lead time/detection among core representations — a genuinely different trade-off, not a discrimination win. |
| Persistence(τ) | **Category 1** — statistically indistinguishable from single-window in both directions. |
| Longest run(τ) | **Category 1** — same. |
| Slope | **Category 1, negative** — significantly worse than single-window at both reported horizons; test AUROC below chance. |
| Recent − earlier | **Category 1, negative** — same pattern as slope. |

**No representation reaches Category 3.** Per protocol Section 11, nothing
in this cohort could reach that bar regardless of the numbers obtained.

## K. Future validation

Max and P90 are the two candidates worth carrying into an external-cohort
validation design, should one become available — not for promotion here.
Any such design should account explicitly for the recording-duration
confound (Section H) and should evaluate operational lead-time/detection
alongside AUROC from the outset, given how sharply the two diverged in this
cohort (Section G). Slope-type trend descriptors, on this evidence, are not
worth carrying forward in their current univariate form.

---

## Answers to the seven closing questions

1. **What was implemented.** A fresh, pre-registered protocol
   (`docs/phase15_temporal_risk_representation_protocol.md`) and one study
   script (`scripts/phase15_temporal_representation_study.py`) covering: five
   core patient-level aggregations of the frozen P6 window-level scores
   (single-window, P90, max, mean, median) across four horizons with paired
   patient-level bootstrap + DeLong on CV and held-out test; four temporal
   persistence/trend descriptors with a training-fold-only, non-outcome
   threshold; operational metrics (sensitivity, specificity, FAR, lead time,
   ≥20/30min detection) per representation; a recording-duration confound
   check; and a descriptive P90-vs-Max comparison. No model was retrained.

2. **What was pre-specified.** The five core representations (no others
   added), the four descriptors and their exact definitions, the τ threshold
   as a training-fold population statistic (never outcome-tuned), both
   horizons reported without one promoted to sole primary status, the full
   statistical/operational protocol, and the decision-category ceiling
   (nothing above Category 2/hypothesis-generating is reachable on this
   cohort) — all frozen before the study script was run.

3. **What results were obtained.** Full tables in Sections E–H above. In
   short: Max and P90 show the most consistent positive pattern (not
   significant); Mean is flat; Median is flat-to-negative; slope and
   recent-minus-earlier are *significantly negative* (the only formally
   significant findings in the study); every representation's advantage
   shrinks with longer recordings; and AUROC ranking inverts operational
   ranking (Max best AUROC, worst lead time; Median worst AUROC, best lead
   time).

4. **Which aggregation patterns were consistent.** Max ≥ P90 > Mean > Median
   held at both delivery and ≥30m on both CV and test (Mean's delivery-test
   point estimate the only near-exact-zero exception). Slope and
   recent-minus-earlier were consistently negative at both horizons on both
   splits. ≥10m/≥20m showed no consistent pattern at all (CV/test disagree
   in sign throughout) and are not used to support any conclusion.

5. **Whether the evidence supports the hypothesis that single-window
   selection loses information.** Partially and directionally, not
   confirmatorily. The consistent Max/P90 > Mean/Median > (trend
   descriptors) ordering, replicated across two horizons and two data
   splits, is more evidence than a single point estimate would be — but
   none of it clears statistical significance on this 547-patient cohort,
   and the recording-duration analysis shows the apparent benefit is not
   uniform across patients. The honest reading: the *type* of information
   plausibly being lost is magnitude/peak-risk, not trajectory direction —
   the hypothesis is narrowed, not confirmed.

6. **Whether any method deserves future external validation.** Max, and to
   a slightly lesser degree P90 — consistent with, not stronger than, Phase
   13/14's prior characterization of P90 alone. Neither is ready for
   promotion; both are worth carrying into an independent-cohort design
   specifically because internal re-analysis of this cohort has now been
   run twice (Phase 14, single test) and broadened once (Phase 15, five
   representations plus four descriptors) with converging, non-significant,
   directionally consistent results — the ceiling of what this data can
   say has plausibly been reached.

7. **Whether Phase 12.1 remains completely untouched.** Yes. No file under
   the locked pipeline was read destructively or modified. The locked
   numbers (delivery ≈0.6872, ≥30m ≈0.5857 under the original horizon
   convention) are unaltered and are not superseded by anything in this
   report. All Phase 15 artifacts are new, additive files under
   `docs/phase15_*`, `scripts/phase15_*`, and `results/phase15/`; nothing
   has been committed.

## Reproducibility

`python scripts/phase15_temporal_representation_study.py` — reads only
`results/phase13/audit/p6_predictions.npz`, `data/processed_clinical/folds.json`,
`data/processed_clinical/test_dataset.pt`, `results/phase8_rolling/rolling_predictions.csv`.
Fixed `seed=42` throughout. Outputs: `results/phase15/phase15_core_representations.csv`,
`phase15_operational_metrics.csv`, `phase15_persistence_descriptors.csv`,
`phase15_duration_analysis.csv`, `phase15_p90_vs_max_descriptive.json`.
