# Phase 19 — Improving the Adaptive Temporal Pooling (TAM)
## Pre-Registration Protocol

> **Status: executed 2026-09-21 — A1, A2 and A3 are all NOT SUPPORTED.**
> Neither reliability cues (Route 1) nor a ranking objective (Route 2)
> improved on the TAM control; TAM itself is indistinguishable from
> single-window PRS when averaged over horizons. Full results and the
> consequence for next steps: `results/phase19_pooler/phase19_report.md`.
> The protocol below is unchanged from when it was frozen.

**Frozen 2026-09-21, before any arm code was written or run.** Governed by the
same discipline as `docs/phase13_protocol.md` through `docs/phase18_*`.

> **Governance (unchanged).** PRS (P6, Phase 12.1) is locked and is never
> retrained; all PRS window scores are the frozen canonical out-of-fold scores
> (`results/phase13/audit/p6_predictions.npz`). Nothing here is promoted or
> clinically usable. Every result is reported regardless of outcome.

---

## 1. What the diagnostics established (why these two routes)

Phase 19 Steps 1–3 (`scripts/phase19_attention_diagnostics.py`,
`phase19_parametric_pooling.py`, `phase19_pooling_landscape.py`,
`phase19_information_profile.py`; results in `results/phase19_attention/`):

1. TAM's learned attention is dominated by **recency** (logit rises ≈0.15/min
   of elapsed time; weights monotone in time for essentially every patient),
   with only a mild positive tilt toward higher-risk windows.
2. The window-level PRS signal is **strongest 2.5–7.5 min before delivery
   (AUROC ≈0.71) and weaker at the final window (0.687)**, which is also the
   noisiest (window-to-window jump 0.047 vs ≈0.026 at ≥20 min). Smoothing over
   the last few windows therefore helps at the delivery-time query.
3. At ≥10m and ≥20m the newest window is the best available evidence and
   averaging older windows dilutes it: TAM is *below* single-window PRS there
   (−0.018, −0.022), and **the best fixed weighting differs by horizon**
   (moderate recency at delivery, pure recency at 10–20m, risk-tilted at 30m).
   TAM is a compromise, not a per-horizon optimum.
4. A 2-parameter recency × risk rule fit under TAM's BCE objective does not
   reproduce TAM at delivery (0.702 vs 0.722) and picks a sharper decay than
   TAM — the BCE objective and the deployed metric (ranking) are misaligned.

**Route 1 hypothesis (H1).** Giving the attention *observable reliability
cues* lets it trust the newest window when the trace is calm and smooth when
it is volatile, resolving the horizon compromise without knowing the horizon.
**Route 2 hypothesis (H2).** Training on a ranking objective averaged over the
lead-time grid (instead of pooled BCE) aligns training with the deployed
metric and improves horizon-averaged AUROC.

Routes 3 (parity as attention input) and 4 (seed ensembling as a stand-alone
route) are out of scope here; seed-ensembling is applied to *every* arm
equally (§4) so it cannot confound the comparison.

## 2. Arms (all trained with the identical pipeline; only inputs/objective differ)

| Arm | Per-window inputs | Objective |
|---|---|---|
| **A0** (control) | `r_t`, `elapsed_t/60` (exactly TAM) | pooled BCE over every causal truncation, per-patient 1/T normalization (exactly TAM) |
| **A1** (Route 1) | A0 inputs + 4 cues (§3) | as A0 |
| **A2** (Route 2) | A0 inputs | horizon-grid ranking loss (§3) |
| **A3** (both) | A1 inputs | as A2 |

## 3. Frozen definitions

**Cues (all causal, computed from the window sequence up to `t` or from the
frozen 40-D PRS feature row of window `t`):**
`d_t = |r_t − r_(t−1)|` (0 for the first window); `s3_t` = std of
`r_(t−2..t)` (over available windows); `dom_sev_uterine_t` and
`dom_sev_coupling_t` from `results/phase9c_state_trajectory/state_trajectory_features.npz`
(named columns of `X_state_trajectory`, row-aligned to the PRS scores —
alignment verified). The four cues are standardized with the mean/std of the
**training-fold windows only**; `r_t` and `elapsed_t/60` are left unstandardized
as in TAM.

**Ranking objective.** For each lead-time grid point `h ∈ {0,5,10,…,35}` min,
take each training patient's score at its eligible prefix for `h`
(`eligible_prefix_length`, the project-wide fallback convention). Loss_h =
mean over all (positive, negative) training-patient pairs of
`softplus(−(logit z_pos − logit z_neg))`; objective = mean over `h`. The
retrospective lead time is used **only to choose which truncation of a
training patient enters the loss**; inference never uses it.

**Fixed, untuned hyperparameters (nothing is searched):** MLP `in_dim→8→1`
with tanh; Adam, lr 0.01, weight-decay 1e-4; full-batch; max 200 epochs;
early stopping patience 10 (min-delta 1e-5) on the arm's *own objective*
evaluated on an inner-validation split (15% of the outer-training patients,
stratified, seed `42+fold`); ranking-loss scale 1; grid step 5 min.
Network seeds {42, 43, 44}.

## 4. Evaluation

- Cohort/endpoint/folds: 547 patients, pH ≤ 7.15, canonical `folds.json`;
  held-out internal test partition (83 patients) evaluated once per arm/seed
  by a train+val model. Horizons 0/10/20/30 min via the eligible-prefix rule;
  **N = 547 (CV) / 83 (test) at every horizon, no patient dropped.**
- **An arm's score = the mean of its three seeds' pooled scores** (seed
  ensemble), applied identically to A0. Single-seed spread is also reported.
- **Primary metric M** = mean over the four horizons of patient-level AUROC.
- Statistics: paired patient bootstrap (B=2000, seed 42, the
  `scripts/phase11_bootstrap.py` convention) on ΔM and on each per-horizon Δ,
  vs A0 (primary) and vs single-window PRS (secondary).
- **Fold-resplit check:** 5 independent `StratifiedKFold` resplits (seeds
  11/22/33/44/55); PRS window scores stay frozen; only the pooler is retrained
  per resplit (project precedent, `scripts/phase16_model3_verification.py`).

## 5. Gates and decision rule (mechanical, fixed now)

**Sanity gate (before any arm is read).** A0, seed 42, canonical CV must
reproduce frozen TAM (0.7216/0.6683/0.6015/0.5976 at 0/10/20/30 min) within
±0.006 per horizon. If it does not, stop and investigate; do not adjust arms.

For each arm X ∈ {A1, A2, A3} vs A0:

- **CONFIRMED** iff all hold: (C1) canonical ΔM ≥ +0.005 with bootstrap
  p < .05; (C2) ΔM > 0 in **all 5** resplits and p < .05 in **≥ 3 of 5**;
  (C3) no per-horizon regression worse than −0.020 vs A0, canonically and
  in the resplit mean.
- **SUGGESTIVE** iff not CONFIRMED but canonical ΔM > 0 and ΔM > 0 in ≥ 4 of 5
  resplits.
- **NOT SUPPORTED** otherwise.

Secondary, reported with no decision attached: per-horizon Δ vs A0 and vs
PRS; whether each arm reaches single-window PRS at ≥10m and ≥20m (TAM's
shortfall); test-partition ΔM and its direction; epochs used and any
epoch-cap hits; seed spread. Three arms are compared to one control; the
fold-resplit requirement (C2), not a p-value correction, is the guard
against a chance canonical win, and this is stated rather than hidden.

## 6. Conditional follow-up (not run now)

Only if an arm is CONFIRMED: fuse its pooled score with parity by the frozen
URM procedure (α selected once, pooled truncations, `docs/phase18_deployable_fusion_protocol.md`)
and compare against frozen URM under a separate, later amendment.

## 7. Out of scope

Tuning any hyperparameter; new features beyond §3; parity as an attention
input; horizon-conditioned models (unusable — horizon is unknown in real
time); retraining PRS; any claim from the exploratory diagnostics alone.
