# Phase 25 — Candidate D2: fixing Candidate D's lead-time failure
## Pre-Registration Protocol

> **Status (2026-09-23): executed on 11 internal splits — EXPLORATORY NO SUPPORT, but the lead-time guard L PASSED** (the failure that killed Candidate D). Discrimination held up (fresh mean +0.0152, 10/10 positive, pooled p .043); only the calibration guard E narrowly failed (canonical slope 1.29 vs the 1.25 limit; fresh-resplit mean 1.18, inside the band). Canonical-split AUROC was actually below baseline (-0.0054) even though all 10 fresh resplits were positive -- an unresolved anomaly, reported not explained away. Full results: results/phase25_ga_recal_leadfix/phase25_report.md. Protocol text unchanged.

**Frozen 2026-09-23, before any Phase 25 code was written or run, but after the diagnostic in section 1 (a read-only analysis of already-cached Phase 22/24 scores on the canonical split).**
Per this study's convention (exactly how Candidate D itself arose from Phase 23's ablation), a diagnostic run may inform which hypothesis is tested; it never substitutes for testing it. The confirmatory run below uses **resplit
seeds never touched by the diagnostic or by any prior phase.**

> **Governance.** Locked artefacts (P6/PRS, TAM, MCM, URM, Phase 12–24 results) are never modified. New outputs go to `results/phase25_ga_recal_leadfix/`. Status is **exploratory** regardless of outcome; the frozen URM
> remains the reference and deployed model; the 83-patient test partition (opened in Phases 21b and 22) is **not used**.

## 1. Diagnosis: why Candidate D warns later (read-only, on cached Phase 22/24 scores, canonical split only)
- G-A's *raw window-level* scores are not concentrated near delivery — they beat P6's at nearly every time-before-delivery bucket (e.g. AUROC 0.646 vs 0.633 at 10–20 min, 0.594 vs 0.582 at 20–30 min). The window model itself is not the problem.
- G-A's *pooled* trajectory (after the TAM-style pooler) is slightly more back-loaded than P6's: at 50% of the way through a recording it has reached 77.3% of its eventual delivery-time score, versus 79.6% for P6 — a small effect.
- **The fusion weight α is the amplifier.** α is selected, in every phase so far, to maximise pooled training AUROC with no term for *when* the score crosses a threshold. Because G-A's score is more discriminative overall,
  this selection pushes α higher for G-A (mean ≈0.42, up to 0.45 per fold) than URM's own selection ever does for P6 (mean 0.333, max 0.40) — moving weight away from parity, which is flat and available from the first window,
  onto a CTG score that is (slightly) more back-loaded. A controlled sweep holding the G-A window/pooler score fixed and varying only α (canonical split) confirms this is the dominant effect:

| α (fixed, no selection) | M | Sensitivity | FAR | Median lead | ≥20 min | ≥30 min |
|---|---|---|---|---|---|---|
| 0.25 | 0.688 | 86.4% | 59.7% | **40.0 min** | **82.1%** | 68.4% |
| 0.30 | 0.693 | 85.5% | 58.8% | 35.0 min | 73.4% | 61.7% |
| 0.33 | 0.695 | 85.5% | 59.3% | 35.0 min | 71.3% | 59.6% |
| **0.35 (URM's own value)** | 0.696 | 84.5% | 58.3% | 32.5 min | 69.9% | 57.0% |
| 0.40 | 0.700 | 85.5% | 57.4% | 31.25 min | 67.0% | 53.2% |
| 0.45 (≈ what selection actually picks) | 0.701 | 87.3% | 58.3% | 30.0 min | 64.6% | 52.1% |

Reusing URM's own α (0.35) only partially fixes the problem — lead time is still well below URM's actual 40 min / 81.1% / 69.5%. The window model's own mild back-loading means G-A needs **more**, not the same, weight on
parity to match URM's early-warning profile. This motivates a selection rule that trades a little AUROC for lead time, rather than reusing a fixed number picked by eye.

## 2. Candidate D2 (fixed, one change from Candidate D)
Identical to Candidate D (`docs/phase24_ga_recal_protocol.md` §2: G-A window, ridge α = 1000 fixed, Platt step; standard binary-label pooler; MCM; per-fold recalibration) **except** the fusion weight is chosen by:

> **Lead-time-constrained α selection.** For each outer training fold, compute, for every α on the grid {0.05, 0.10, …, 0.95}, (i) the pooled training AUROC (exactly as today, in-sample on that fold's training patients — the
> existing convention, not newly nested) and (ii) the **in-sample training median lead time**: fit the 20th-percentile-of-training-positives delivery-score threshold on that same training set, find each training positive's own
> first-alert time, take the median. Select **α = argmax(training AUROC) subject to training median lead time ≥ 35 minutes.** If no α on the grid satisfies the constraint, use the α with the largest training median lead time
> (documented, not expected to trigger). The 35-minute floor is a fixed constant, chosen once, in every fold and every split — informed by the diagnostic table above (it sits between D's own worst case and URM's actual 40 min)
> but **not** re-derived from any held-out result; it is one plausible choice, stated as such.

This changes nothing about the window model, the pooler, or the recalibration step — the only change is how one scalar per fold is chosen.

Baseline **B0** = locked P6 recipe, unconstrained α selection (unchanged, since B0 does not need the constraint).

## 3. Evaluation
Canonical 5-fold CV on all 547, plus **10 fresh resplits with seeds never used in any prior phase**: 501, 514, 528, 541, 557, 569, 583, 598, 612, 627.
Same metrics as Phase 24: M and per-horizon AUROC; calibration (Brier, slope, intercept, ECE10) at delivery; **OP-far** (threshold fixed from B0's own training negatives to match B0's canonical 62.5% false-alert rate, applied
identically to D2 and B0 — the fair, matched-specificity comparison); OP-sens (80% target, each model's own threshold) reported for continuity only, no decision weight.

## 4. Decision rule (mechanical, fixed now) — identical structure to Phase 24 §4, so the two candidates are judged on the same bar
**EXPLORATORY SUPPORT** iff all hold:
- **D1** mean ΔM (D2 − B0) over the 10 fresh resplits ≥ +0.005 and > 0 in ≥ 8 of 10;
- **D2crit** mean per-horizon Δ ≥ −0.010 at every horizon;
- **L** at OP-far, on the canonical split **and** the fresh-resplit mean: Δ sensitivity ≥ −0.02, Δ median lead ≥ **−2.5 min**, Δ(share ≥ 20 min) ≥ **−0.03** (unchanged from Phase 24 — this candidate is held to the same guard it
  failed, not a relaxed one);
- **E** calibration slope of D2 in [0.80, 1.25] on canonical and fresh-resplit mean, Brier no worse than D2's own uncalibrated score.
Otherwise **EXPLORATORY NO SUPPORT**, reporting which criterion failed. This is not "confirmed" or "adopted" either way; the frozen URM is unaffected.

## 5. Reporting
Canonical ΔM with CI/p; fresh-resplit pooled p; both operating-point tables; calibration table; the α values actually selected per fold (to show how much the constraint bites); explicit comparison against Candidate D on the
same splits and metrics, so it is clear whether this is a genuine improvement over D or just a different trade-off point. Repeats the standing caveat: no independent cohort exists in this environment; every split reuses the
same 547 patients.

## 6. Out of scope
Changing the 35-minute floor, the α grid, the window model, the pooler or the recalibration step after seeing results; new candidates; the test partition.
