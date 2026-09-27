# Phase 21 — Graded-outcome supervision of the window model
## Pre-Registration Protocol

> **Status (2026-09-21): screening executed — none met the ADOPT rule (G-A SUGGESTIVE, G-C SUGGESTIVE, G-B NOT SUPPORTED); confirmation and test partition not run.** See `results/phase21_graded/phase21_report.md`. Protocol text unchanged.
> **21b exploratory follow-up executed (§7): EXPLORATORY SUPPORT for G-A** — fresh resplits mean ΔM +0.0088 (5/5 positive, pooled p .29); test partition (opened once, 83 patients / 17 positives) ΔM +0.049 (95% CI +0.0004 to +0.109). Not adopted or confirmed; the frozen URM stands. See `results/phase21_graded/phase21_report.md`.

**Frozen 2026-09-21, before any Phase 21 candidate was run.** Same discipline as `docs/phase13_protocol.md` … `docs/phase20_redesign_protocol.md`.

> **Governance.** Locked artefacts (P6/PRS predictions, TAM checkpoints, MCM, URM, all Phase 12–20 results) are never modified.
> Everything new is written to `results/phase21_graded/`. Endpoint, cohort, folds, harness and metric are exactly Phase 20's, so
> results are directly comparable. The endpoint stays **pH ≤ 7.15**; only the *training signal* changes.

## 1. Why this, and what the evidence is

Phase 20 found no gain from more features (C2, C5a–c), other learners (C3), window weighting (C1), extra covariates (C4) or a GRU (C6a).
Exploratory diagnostics on the baseline (`scripts/phase21_failure_diagnostics.py`, `results/phase21_diagnostics/`) point at the label:

| Diagnostic (baseline URM at delivery, D464 canonical) | Result |
|---|---|
| Positives within 0.05 pH of the cut-off | **51%** (47/93) |
| AUROC vs pH≤7.15 / ≤7.05 / BDecf≥8 / BDecf≥12 | 0.710 / 0.743 / 0.753 / 0.805 (n_pos 93 / 31 / 53 / 12) |
| Deep positives (pH≤7.05) vs normal (>7.20) / borderline (7.10–7.15] vs normal | **0.794 / 0.724** |
| Spearman(score, −pH), all patients | +0.394 |
| Learning curve, 92 → 371 training patients per fold | 0.598 → 0.619 → 0.634 → 0.639 (flattening) |
| Best single descriptor at delivery | accelerations 0.667 (baseline 0.651 inverted, LTV 0.623) |

Reading: the window model already extracts most of what a handful of classical descriptors carry; the binary label discards graded
information and injects near-threshold noise. **Hypothesis H:** supervising the *same* window model with a graded target (pH severity)
instead of the binary 7.15 label improves the end-to-end URM evaluated on the unchanged pH ≤ 7.15 endpoint.

Prior art, stated: hand-crafted down-weighting of near-threshold windows in the CrossFormer (commit `10ce497`) gave CV +0.019 but worse
test, not adopted; Phase 6 tried continuous/ordinal supervision on the old deep-encoder pipeline (signal-only arms ≈ 0.55). Neither tested
graded supervision on the current 40-D window model with the trajectory harness. **Caveat, declared:** H was formed after looking at D1–D4
on D464 canonical scores, so adoption requires fresh-resplit confirmation (§5) and the test partition is used once.

## 2. Candidates (fixed; all use the locked 40-D features, nothing else changes)

Window-level training target is a function of the patient's umbilical-artery pH, which is used **only to build training targets**, never as an input.
Each candidate's window score is converted to the same probability scale by a **Platt step** (1-D logistic fit of the binary pH≤7.15 label on the
model's *in-sample training* scores, applied to held-out scores) so that pooler, α-fusion and MCM see the scale they always did.

| ID | Training signal | Model | Inner grid |
|---|---|---|---|
| **G-A** regression | standardised `−pH` (train-fold stats) on every window | Ridge on standardised features | α ∈ {10, 100, 1000} |
| **G-B** multi-threshold | four binary targets `1[pH ≤ τ]`, τ ∈ {7.20, 7.15, 7.10, 7.05}; score = mean of the four predicted probabilities | 4 LogisticRegression, same C | C ∈ {0.01, 0.05, 0.2} |
| **G-C** soft label | soft target `q = σ((7.15 − pH)/0.03)` (fixed scale 0.03 ≈ pH measurement error), realised as each window duplicated with label 1 weight q and label 0 weight 1−q | LogisticRegression | C ∈ {0.01, 0.05, 0.2} |
| **Control B0-P** (not a candidate) | binary pH≤7.15 label, as P6 (C = 0.05) **plus the same Platt step** | — | — |

Inner selection (regularisation only) uses inner patient-grouped 4-fold CV inside each outer training set with criterion = inner-OOF mean over
horizons of single-window patient AUROC **against the binary pH≤7.15 label** (as in Phase 20). Outer held-out folds never influence any choice.
B0-P isolates the effect of the label from the effect of the Platt step.

## 3. Evaluation (identical to Phase 20)

Development set D464; canonical folds + 5 resplits (11/22/33/44/55); common harness (window OOF → pooler seeds 42–44 → MCM → α → horizon scoring);
horizons 0/10/20/30; primary metric M = mean patient AUROC over horizons; paired patient bootstrap B = 2000 seed 42; baseline = the Phase 20
B0 scores (`results/phase20_redesign/stage0_baseline_D464_scores.npz`, M canonical 0.6671).

## 4. Decision rule for screening (mechanical, fixed now)

Holm family K = 3 (G-A, G-B, G-C). A candidate is **ADOPT-to-confirmation** iff all hold:
(A1) canonical ΔM vs B0 ≥ +0.005 with Holm-adjusted p < .05; (A2) ΔM > 0 in all 5 resplits; (A3) no per-horizon Δ worse than −0.020 canonically or in the
resplit mean; (A4) ΔM vs the control B0-P is > 0 canonically and in ≥ 4 of 5 resplits (so the gain is attributable to graded supervision, not to recalibration).
SUGGESTIVE = canonical ΔM > 0 and ≥ 4/5 resplits positive; NOT SUPPORTED otherwise. Only ADOPT-to-confirmation candidates proceed.

## 5. Confirmation (only for candidates that pass §4)

On **fresh resplits** (seeds 66/77/88/99/111, D464, never used in any Phase 20/21 selection): **CONFIRMED** iff canonical-fresh mean ΔM ≥ +0.005 (mean over the 5
fresh resplits) with paired-bootstrap p < .05 on the pooled fresh scores, ΔM > 0 in ≥ 4 of 5, and no per-horizon Δ worse than −0.020 in the mean. Then, once,
train on all D464 → 83-patient test partition; a test ΔM < −0.020 is a contradiction flag that blocks the word "improved". If nothing passes §4 the test
partition is not touched and the frozen URM stands.

## 6. Out of scope

Tuning the soft-label scale or the τ set; using pH/BDecf as model inputs; retraining TAM/pooler architecture; any change to endpoint, folds, harness or the
Phase 20 candidates; claims from the diagnostics alone.

## 7. Exploratory follow-up 21b (frozen 2026-09-21, after the screening result was known, before any follow-up run)

**Status of this section: EXPLORATORY.** G-A did not pass §4 (canonical Holm-adjusted p ≥ .05), so under §5 it would not be confirmed. The user asked
for it to be run anyway as an exploratory follow-up. This section fixes what is run and how it may be described; whatever happens, **G-A is never called
ADOPTED or CONFIRMED, the frozen URM remains the recommendation, and no claim of improvement is made from this section alone.**

Arms (both use the locked 40-D features, pH only for training targets, same Platt step, same harness as §3):
- **G-A (primary)** exactly as screened: Ridge on standardised −pH, α ∈ {10, 100, 1000}.
- **G-A-wide (secondary, exploratory)**: identical but α ∈ {100, 1000, 10000, 100000}, because screening chose the grid edge (α = 1000). It was defined after seeing
  the screening result, so it is labelled a post-hoc variant and carries no decision weight.

Comparison baseline: B0 (the Phase 20 baseline) recomputed in the same run on each split.

1. **Fresh resplits** (seeds 66/77/88/99/111 on D464, never used in Phase 20/21 selection): report ΔM per split and per horizon, mean ΔM, number of positive splits, worst
   mean per-horizon Δ, and the paired-bootstrap p pooled over the five resplits (same resamples across resplits; p reported, not gated).
2. **Test partition, once** (83 patients, ≈17 positives): train on all D464, evaluate B0 and each arm; the arm's window-model hyper-parameter is chosen by inner 4-fold CV on D464,
   the pooler is trained on D464 out-of-fold window scores (canonical D464 folds; inner-val split seed 123, seeds 42–44), MCM and α are fitted on D464. Report ΔM with a
   paired bootstrap CI. The frozen URM's test AUROC (Phase 18) is shown for reference only.

Descriptive label (mechanical, primary arm G-A only):
- **EXPLORATORY SUPPORT** iff fresh mean ΔM ≥ +0.005, ΔM > 0 in ≥ 4 of 5 fresh resplits, worst mean per-horizon Δ ≥ −0.020, **and** test ΔM > 0.
- **EXPLORATORY NO SUPPORT** otherwise.
Neither label changes §4/§5 status. The test partition is opened once for both arms and not again in Phase 21. Even EXPLORATORY SUPPORT could not be upgraded without
independent data (external validation), because every split here reuses the same patients.

