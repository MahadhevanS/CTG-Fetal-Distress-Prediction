# Phase 14 — Patient-Level Temporal Aggregation Study: Results

Written 2026-09-11, executed exactly per the pre-registered
[docs/phase14_protocol.md](../docs/phase14_protocol.md), frozen before this
script ran. Phase 12.1 and the closed Phase 13 threads (A–D, the P90+Parity
hybrid) are untouched and out of scope here.

## Primary confirmatory result

**P90 vs. single-window, delivery, 5-fold CV** (the sole pre-registered
primary test, Section 4):

```
delta = +0.0307   p = 0.225   95% CI = [-0.0189, +0.0822]
test point estimate: +0.0125 (agrees in direction, not itself powered to be significant)
```

**Pre-registered verdict, applied mechanically per Section 7's decision
table: NOT REPLICATED AT THE PRE-REGISTERED BAR.** CV does not reach
p<0.05 and the CI crosses zero — the strict "internally replicated" branch
requires both, plus test agreement, and only the last of the three holds.
Per Section 7, this keeps single-window as the production convention; P90
pooling remains promising, not confirmed, and is not promoted.

**One transparency note on why this delta (+0.0307) differs slightly from
Phase 13's raw exploratory number (+0.0382).** This script used `numpy`'s
standard linear-interpolation percentile; Phase 13's Option E used a
different (Hazen-style) weighted-quantile estimator
(`src/aggregation/recency_weighted_p90.py`). Both are legitimate
definitions of "the 90th percentile" and neither is more "correct" — the
difference is reported here rather than silently reconciled, because the
fact that a minor, defensible implementation choice moves the point
estimate this much (and the CV p-value from 0.121 to 0.225) is itself
informative: it's a small piece of independent evidence that the original
estimate was not highly stable, consistent with the "not yet confirmed"
verdict rather than contradicting it.

## Full results

| Horizon | Candidate | Tier | CV Δ (p, CI) | Test Δ (p) |
|---|---|---|---|---|
| 0m | P90 | **PRIMARY** | +0.0307 (p=.225, [-.019,+.082]) | +0.0125 (p=.797) |
| 0m | Max | exploratory | +0.0338 (p=.140, [-.008,+.080]) | +0.0633 (p=.080) |
| 0m | Mean | exploratory | +0.0054 (p=.865, [-.049,+.060]) | −0.0009 (p=.956) |
| 30m | P90 | secondary-confirmatory | +0.0261 (p=.216, [-.016,+.069]) | +0.0900 (p=.189) |
| 30m | Max | exploratory | +0.0298 (p=.176, [-.014,+.074]) | +0.1052 (p=.108) |
| 30m | Mean | exploratory | +0.0127 (p=.430, [-.020,+.045]) | +0.0553 (p=.374) |
| 10m, 20m | all | exploratory | mixed sign, none significant | see `results/phase14/phase14_results.csv` |

Full table with h=10/20 and DeLong p-values: `results/phase14/phase14_results.csv`.

## Reading the pattern honestly

Every P90/max comparison at delivery and ≥30m is directionally positive on
both CV and test — twelve non-independent looks at the same underlying
signal, all pointing the same way — but none individually clears p<0.05,
and the CV confidence intervals are wide enough (roughly ±0.04 to ±0.05) to
be consistent with a true effect anywhere from mildly negative to
moderately positive. Max pooling tracks P90 closely throughout and is
occasionally stronger on the test point estimate (e.g. delivery test:
max +0.0633 vs. P90 +0.0125) — noted as an exploratory observation only,
since P90 (not max) was the pre-registered primary candidate and swapping
to whichever candidate looks best after seeing results is exactly the
practice this pre-registration exists to prevent. Mean pooling is the
weakest candidate throughout, consistent with the established finding
(Phase 3 onward) that acidemia-related signal concentrates in a minority
of high-risk windows rather than being evenly distributed.

## Interpretation

A stricter, single-primary-test, fully pre-committed protocol — applied to
essentially the same evidence base as Phase 13's exploratory pass —
returns the same epistemic status Phase 13 already assigned: **promising,
not confirmed.** That agreement is itself a meaningful (if modest) result:
it means Phase 13's caution about not promoting the P90 finding was
justified, not merely procedurally correct — a genuinely independent
analytical pass, designed before seeing this run's numbers, did not
manufacture the significance a less disciplined process might have found
room to claim.

## What would actually settle this

Per protocol Section 0 and 8: nothing further on this cohort will
settle it. The 547-patient ceiling has now been evaluated under an
exploratory pass (Phase 13) and a strict single-primary-test pass
(Phase 14) with converging conclusions. The only escalation that could
change this verdict is **external validation on an independent CTG
cohort** — already the project's own highest-priority identified future
work, independent of and prior to this thread. No further internal
re-analysis of the same 547 patients (different percentile definitions,
different horizon choices, different candidate aggregators) should be
expected to produce a different answer than the two already obtained.

## Reproducibility

`python scripts/phase14_temporal_aggregation.py` — reads only
`results/phase13/audit/p6_predictions.npz` (frozen, Phase 13.0A),
`data/processed_clinical/folds.json`, `data/processed_clinical/test_dataset.pt`,
`results/phase8_rolling/rolling_predictions.csv`. No model retrained. Fixed
`seed=42` throughout; re-running reproduces the table exactly.

## Status

**Phase 14 primary hypothesis: not confirmed, not closed — same status as
entering the study, now on stricter evidence.** No promotion. No further
tuning, candidate expansion, or re-analysis is planned under this protocol.
Phase 12.1 remains the locked, authoritative model; nothing in Phases 13 or
14 supersedes it.
