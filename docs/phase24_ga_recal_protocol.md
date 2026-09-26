# Phase 24 — Candidate D: G-A (fixed α) + recalibration, matched-false-alert-rate lead-time guard
## Pre-Registration Protocol

> **Status (2026-09-23): executed on 21 total internal splits (canonical + 10 fresh resplits) — EXPLORATORY NO SUPPORT.** D1/D2 (discrimination) passed strongly (fresh mean +0.0148, 10/10 positive); the lead-time guard L failed (candidate warns later even at matched false-alert rate, especially on the canonical split); the calibration guard E narrowly failed (canonical slope 1.32 vs the 1.25 limit, though Brier improved). Full results: results/phase24_ga_recal/phase24_report.md. No independent cohort was available; see the caveat in section 0/protocol governance. Protocol text unchanged.

**Frozen 2026-09-23, before any Phase 24 code was written or run.**

> **Governance.** Locked artefacts (P6/PRS, TAM, MCM, URM, Phase 12–23 results, the Phase 22 frozen G-A model) are never modified. New outputs go to `results/phase24_ga_recal/`.
> **No independent/external cohort is available in this environment.** The user asked to evaluate this candidate "on independent data"; since none exists here, this phase instead uses
> **fresh resplit seeds never used in any prior phase** of this study, which is the closest approximation available and is explicitly weaker evidence than independent patients — every
> split below still reuses the same 547 CTU-UHB patients. This is stated in every reported result, not just here. Status is **exploratory** regardless of outcome; the frozen URM remains
> the reference; the 83-patient test partition (opened in Phases 21b and 22) is **not used**.

## 1. Why this candidate (from Phases 21–23)
Phase 23's ablation found the full V2 composite's gain came almost entirely from one piece — the G-A window score with its ridge strength **fixed** at α = 1000 (no nested selection) — fresh-resplit ΔM +0.018,
10/10 positive; the soft-label pooler added nothing (4/10 positive) and averaging with P6 diluted the gain. Phase 23's pre-registered lead-time guard failed at a **sensitivity-matched** operating point, but a
post-hoc check at a **false-alert-rate-matched** operating point (phase23b) found the composite kept the baseline's lead times. **Candidate D** therefore drops the soft-label pooler and the P6 averaging, keeps
only the fixed-α G-A window and recalibration, and re-tests the lead-time question at the operating point the post-hoc check used — but pre-registered this time, not post-hoc.

## 2. The candidate (fixed, nothing tuned)
1. **Window score** = G-A: Ridge on standardised −pH, **α = 1000 fixed** (no selection), then the Phase 21 Platt step. Locked 40-D features; pH used only to build training targets.
2. **Pooler** = the harness TAM architecture and training exactly as the baseline (binary label, pooled BCE, seeds 42–44, mean of running scores) — **not** the soft-label version (Phase 23 showed it adds nothing).
3. **MCM and fusion α** exactly as the harness (parity LR C = 1.0; α by sample-weighted pooled training AUROC, grid 0.05).
4. **Recalibration**: per outer fold, logistic recalibration (C = 1e6) of the binary label on logit(fused score), fitted on the training patients' pooled causal-prefix fused scores with 1/T weights, applied to
   held-out patients — identical to Phase 23's component E.

Baseline **B0** = locked P6 recipe through the identical harness (reproduces the frozen URM). Cohort, endpoint, canonical folds, horizons, eligible-prefix rule, M, paired bootstrap (B = 2000, seed 42): unchanged.

## 3. Evaluation
- Canonical 5-fold CV on all 547, plus **10 fresh resplits with seeds never used in Phases 20–23**: 301, 313, 327, 341, 359, 372, 386, 401, 417, 433.
- Metrics per split: M and per-horizon AUROC; calibration (Brier, slope, intercept, ECE10) at delivery; two operating-point protocols:
  - **OP-sens** (Phase 18 convention, for continuity): 80% target sensitivity, per-fold training-only threshold, patient-level "ever alerted": sensitivity, false-alert rate, PPV, median lead, share ≥10/20/30 min.
  - **OP-far** (the guard this phase decides on): per fold, threshold = the value at which **B0's own training-negative "ever alerted" rate on that fold equals its canonical false-alert rate of 62.5%** (i.e. threshold fixed
    from B0, applied identically to D and B0, so both systems are read at the *same* operating point rather than each choosing its own). Same metrics reported.

## 4. Decision rule (mechanical, fixed now)
**EXPLORATORY SUPPORT** iff all hold:
- **D1** mean ΔM over the 10 fresh resplits ≥ +0.005 and ΔM > 0 in ≥ 8 of 10 (paired bootstrap, same convention as Phase 23);
- **D2** mean per-horizon Δ ≥ −0.010 at every horizon (fresh-resplit mean);
- **L** at OP-far, on the canonical split **and** the fresh-resplit mean: Δ sensitivity ≥ −0.02, Δ median lead ≥ −2.5 min, Δ(share ≥ 20 min) ≥ −0.03;
- **E** calibration slope of D in [0.80, 1.25] on canonical and fresh-resplit mean, and Brier no worse than the uncalibrated version of D.
Otherwise **EXPLORATORY NO SUPPORT**, reporting which criterion failed. OP-sens is reported for continuity but carries no decision weight (Phase 23 already showed it disadvantages any model with better specificity).
Neither label is "confirmed" or "adopted"; the frozen URM is unaffected either way.

## 5. Reporting
Canonical ΔM with CI and p; fresh-resplit pooled p (same resamples across resplits, stated as measuring split- not patient-variability); both operating-point tables; calibration table.
Explicitly restated in the results: this evaluation has no external cohort and cannot establish generalisation; it can only show whether the candidate is consistent across a wider set of internal resplits than Phase 23 used.

## 6. Out of scope
Any further tuning of α, the pooler, or the recalibration; new candidates; the test partition; claims of "validated" or "generalises".
