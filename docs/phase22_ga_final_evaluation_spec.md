# Phase 22 — G-A: final training and full evaluation
> **Status (2026-09-21): executed.** All aspects computed; test-reproduction and frozen-model integrity gates passed. Results: results/phase22_ga_final/phase22_report.md.

## Evaluation specification (frozen 2026-09-21, before the model was trained on all data or any metric below was computed)

> **Status of the model.** G-A is the URM pipeline whose window model is trained on a graded pH target (Ridge on standardised −pH; α ∈ {10, 100, 1000} chosen by nested CV; Platt step)
> instead of the binary pH ≤ 7.15 label. It has **exploratory support only** (`docs/phase21_graded_supervision_protocol.md` §7): it failed the pre-registered adoption rule. This phase
> trains it on all data and evaluates it from every angle **without changing its definition** — no new tuning, no new candidates. The frozen URM (Phase 18) remains the locked
> reference, and this evaluation cannot promote G-A; it can only describe it.

## 1. What is trained

- **Final G-A model** fitted on all 547 patients: ridge window model (α by inner 4-fold CV on all 547) + Platt, TAM-architecture pooler (3 seeds, trained on out-of-fold window scores from
  the canonical 547 folds, inner-validation seed 123), MCM (parity LR, C = 1.0), fusion α by sample-weighted pooled training AUROC (grid 0.05). Artefacts and a standalone scorer are saved
  under `results/phase22_ga_final/frozen_model/`. No held-out data exists for this fit; every honest number below comes from cross-validation or the earlier held-out fit.
- **Same-pipeline baseline B0** (P6 recipe through the identical harness) is computed alongside for every metric. The frozen URM (Phase 18 tables) is quoted for reference.

## 2. Aspects evaluated (all on out-of-fold scores unless stated)

1. **Discrimination** — canonical 5-fold CV over all 547 patients (the basis of the frozen URM numbers): AUROC and AUPRC at delivery / ≥10 / ≥20 / ≥30 min with patient-bootstrap 95% CIs (B = 2000),
   mean M; a finer AUROC-vs-lead-time curve (0–40 min in 5-min steps).
2. **Paired comparison** vs B0: ΔAUROC per horizon and ΔM with paired patient bootstrap and DeLong (delivery horizon).
3. **Calibration** — Brier score, calibration intercept and slope, 10-bin ECE, reliability table (delivery horizon), for G-A and B0. No recalibration is applied to headline numbers.
4. **Operating points** — sensitivity at 80% and 90% specificity; and the Phase 18 operational protocol (80% target sensitivity, per-fold training-only threshold, patient-level "ever alerted"):
   achieved sensitivity, false-alert rate, PPV, lead-time median/IQR and share alerted ≥ 10/20/30 min before delivery.
5. **Clinical utility** — decision-curve net benefit at delivery for thresholds 0.10–0.50 versus treat-all and treat-none.
6. **Held-out test partition** — the frozen definition trained on the 464 train+val patients, scored on the 83 test patients (this fit was already run once in Phase 21b; it is re-run here only to obtain more metrics
   and must reproduce the earlier mean AUROC 0.8073 within 0.001, otherwise the run is stopped). Same discrimination and calibration metrics; no threshold-based metrics (n too small).
7. **Robustness** — 10 independent resplits over all 547 (seeds 11/22/33/44/55/66/77/88/99/111): ΔM vs B0 per resplit; per-seed pooler results (42/43/44) on the canonical split;
   sensitivity to the ridge α held fixed at 10 / 100 / 1000 (no selection) — descriptive only.
8. **Subgroups** (delivery-time AUROC and ≥30 min AUROC, with CIs, n and positives shown, small groups flagged): delivery type, parity, stage of labour at the last window, recording length,
   and outcome-severity strata (pH ≤ 7.05, BDecf ≥ 8, borderline pH 7.10–7.15, all vs normal pH > 7.20).
9. **Ablation / decomposition** — parity alone, window-only (single latest window), TAM-only, and fused, for G-A vs B0 vs the binary-label-plus-Platt control.
10. **Interpretability** — standardised ridge coefficients (top 10) versus the P6 logistic coefficients, and rank correlation of the two window scores.
11. **Frozen-model integrity** — the standalone scorer reproduces the pipeline's scores for the training patients, and its in-sample AUROC is reported clearly labelled as optimistic.

## 3. Reporting rules

- Every number carries its n and (where computable) a 95% CI. Comparisons with B0 are paired. No result is described as "improved" unless the pre-registered rule in the Phase 21 protocol is met — which it is not.
- Subgroup and calibration results are descriptive; nothing is selected on them.
- Limitations stated with the results: 110 positives; one dataset; retrospective; second-level stacking convention shared with every earlier phase; test partition previously opened once (Phase 21b).
