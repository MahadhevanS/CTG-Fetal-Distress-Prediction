# Phase 23 — URM-v2: a single composite built from the G-A lessons
## Pre-Registration Protocol

> **Status (2026-09-21): executed — EXPLORATORY NO SUPPORT (D1, D2 met; lead-time guard L failed; recalibration slope 1.29-1.31 just outside the 0.80-1.25 window).** Full results: results/phase23_urm_v2/phase23_report.md. A post-hoc matched-false-alert-rate check (phase23b, not pre-registered) is reported separately. Protocol text unchanged.

**Frozen 2026-09-21, before any Phase 23 code was written or run.**

> **Governance.** The locked artefacts (P6/PRS, TAM checkpoints, MCM, URM, Phase 12–22 results, the Phase 22 frozen G-A model) are never modified. New outputs go to `results/phase23_urm_v2/`.
> This is **one composite hypothesis with no search**: nothing below is tuned, and the composite is not chosen from among alternatives. Its status is **exploratory**; it cannot be called adopted or
> confirmed, and the frozen URM remains the reference whatever happens. The 83-patient test partition has been opened twice (Phases 21b and 22) and is **not used in this phase**.

## 1. Motivation (from Phases 21–22)
G-A's graded pH target improved the CTG-only scores by ≈ +0.013 (mean AUROC over horizons) but, after parity fusion on the canonical 547 split, only +0.001; its nested choice of ridge strength cost ≈ 0.005; its warnings came **later** (median
lead 32.5 vs 40 min); and both models are badly calibrated (slope ≈ 2.7). URM-v2 combines the four things those findings suggest.

## 2. The composite (fixed)
1. **[B + C] Window score = ½ (P6 probability + G-A probability).** P6 = locked recipe (StandardScaler + LogisticRegression C = 0.05, binary pH ≤ 7.15 label, unweighted). G-A = Ridge on standardised −pH with the **ridge strength fixed at α = 1000
   (no nested selection)** followed by the Platt step exactly as in Phase 21. Both on the locked 40-D features; pH only builds training targets.
2. **[A] Pooler** = the harness TAM architecture and training (2→8→1 tanh, Adam lr .01, wd 1e-4, ≤ 200 epochs, patience 10, inner-val 15%, pooled over every causal truncation, 1/T normalisation, seeds 42–44, mean of running scores) **with the
   soft label q = σ((7.15 − pH)/0.03) as the cross-entropy target** in place of the binary label (also for its early-stopping loss).
3. **MCM and fusion α** exactly as the harness: parity LR (C = 1.0) fitted on the split's training patients; α by sample-weighted pooled training AUROC on the **binary** label, grid step 0.05.
4. **[E] Recalibration**: within each outer fold, a logistic recalibration (C = 1e6) of the binary label on logit(fused score), fitted on the **training patients' pooled causal-prefix fused scores with 1/T weights**, applied to the held-out
   patients' scores. It is monotone within a fold (so it cannot change within-fold rankings, thresholds or lead times) and can change pooled AUROC only through fold-to-fold differences.

Baseline **B0** = the locked P6 recipe through the identical harness (reproduces the frozen URM). Same cohort (all 547), endpoint (pH ≤ 7.15), canonical folds, horizons 0/10/20/30, eligible-prefix rule, primary metric
M = mean patient AUROC over horizons, paired patient bootstrap (B = 2000, seed 42).

## 3. Evaluation
- Canonical 5-fold CV on all 547, plus **10 fresh resplits** (StratifiedKFold seeds 121, 132, 143, 154, 165, 176, 187, 198, 209, 220 — never used before).
- For every split: M and per-horizon AUROC; Brier, calibration slope/intercept and ECE at the delivery horizon; the Phase 18 operational protocol (80% target sensitivity, per-fold training-only threshold from training positives' delivery score,
  patient-level "ever alerted"): sensitivity, false-alert rate, PPV, median lead, share alerted ≥ 10/20/30 min before delivery.
- **Descriptive ablation on the same splits** (no decision weight): B0; C (G-A window, α = 1000 fixed, standard pooler); B (ensemble window, standard pooler); A (P6 window, soft-label pooler); V2 (= B + A, before recalibration); V2+E (final composite).

## 4. Decision rules (mechanical, fixed now) — for V2+E versus B0
Let ΔM be measured on each fresh resplit. **EXPLORATORY SUPPORT** iff all hold:
- **D1** mean ΔM over the 10 fresh resplits ≥ +0.005 **and** ΔM > 0 in ≥ 8 of 10;
- **D2** the mean per-horizon Δ over the fresh resplits is ≥ −0.010 at every horizon;
- **L (lead-time guard)**, on the canonical split **and** on the mean of the fresh resplits: Δ(share of alerted positives warned ≥ 20 min ahead) ≥ −0.03, Δ(median lead) ≥ −2.5 min, Δ(sensitivity) ≥ −0.03, Δ(false-alert rate) ≤ +0.03.
Otherwise **EXPLORATORY NO SUPPORT**, reporting which criteria failed. Neither label changes the status of the frozen URM.

**Recalibration component E** is labelled separately: it **WORKS** iff on the canonical split and on the fresh-resplit mean the calibration slope of V2+E lies in [0.80, 1.25], its Brier score is no worse than V2 before recalibration, and its mean per-horizon AUROC is no lower than V2's by more than 0.005.

## 5. Reporting
Everything is reported whatever the outcome: canonical 547 ΔM with paired-bootstrap CI and p, the fresh-resplit pooled p (same resamples across resplits), per-resplit ΔM, ablation table, operational and calibration tables.
The word "improved" is reserved for a result that meets D1 with independent evidence, which this cohort cannot provide; a support label is not a claim of improvement.

## 6. Out of scope
Changing any component's definition or the soft-label scale (0.03) or α = 1000 after seeing results; other covariates; test-partition evaluation; retraining the locked artefacts.
