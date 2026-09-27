# Phase 18 — M3 + Parity Fusion Ablation Study
## Pre-Registration Protocol (Stage 1)

> **Stage 1 status: DONE, 2026-09-15, including stability checks.**
> `scripts/phase18_m3_parity_fusion_ablation.py` implements Sections 1–2
> exactly as frozen below; `scripts/phase18_stability_checks.py` runs the
> bootstrap-seed and fold-resplit checks the main pass deferred. Results:
> `results/phase18_fusion_ablation/stage1_report.md` (full writeup, now
> revised after the stability checks), `stage1_main_results.csv`,
> `stage1_coefficient_stability.csv`, `stage1_patient_level_predictions.csv`,
> `stability_bootstrap_seed_sensitivity.csv`, `stability_fold_resplit_sensitivity.csv`,
> plus the F2/F4 regularization sensitivity curves.
>
> **Superseded by the deployable-fusion correction (Stage 1e/1f) — read
> that, not the horizon-specific finding below, for the actual
> recommendation.** The horizon-specific sweep summarized in this box
> until 2026-09-15 recommended *switching fusion mechanisms per horizon*.
> That is not deployable: during actual labor, minutes-before-delivery is
> unknown, so a real-time system cannot select coefficients based on which
> horizon it's "in." Every fusion model up to that point — Stage 1's 16,
> and the earlier Parity Fusion / Hybrid work before this study — had been
> fit separately per retrospective horizon, which is exactly the flaw this
> assumes away. Fixed in Stage 1e by mirroring Model 3's own convention
> (one model, trained pooled across every causal truncation, evaluated at
> multiple retrospective horizons via prefix truncation — never told the
> horizon). Stage 1f re-ran the fold-resplit stability check on this
> corrected, deployable version.
>
> **Final, deployable, resplit-verified recommendation: adopt
> `A3_selected_once`** — probability-space averaging with a single α
> (≈0.30–0.35, selected once per fold, never per horizon) — **as the fusion
> mechanism.** At ≥20m/≥30m it robustly beats M3-only across all 6
> independent splits (worst-case p=.002–.003) — the strongest, most
> completely verified result in this study. Near delivery/≥10m its edge is
> real in direction but not yet resplit-significant; a fixed α=0.75 is the
> one alternative that is resplit-robust there, with a smaller effect.
> Three real bugs were caught and fixed across this whole line of work
> (an interaction-term mismatch, a parity-refit leak in two separate
> resplit scripts) — all documented in
> `results/phase18_fusion_ablation/stage1_report.md`. Stage 2 (temporal
> representation) and Stage 3 (nonlinear meta-models) remain not started.

**Frozen 2026-09-15, before any ablation code was written.** Adapted directly
from the user-supplied ablation plan (pasted in full to this session), which
is treated as the source specification — this document scopes it into
staged, executable work and maps every symbol onto this project's actual
frozen artifacts. Governed by the same discipline as every prior phase
protocol (`docs/phase13_protocol.md` through `docs/phase16_protocol.md`).

> **Governance rule (unchanged).** Phase 12.1 (P6) is locked and
> authoritative. Nothing here retrains or modifies P6, Model 3, or the
> parity model. This phase asks only whether a different way of combining
> their two already-frozen outputs does better than the existing hybrid —
> exactly the question the user's plan poses in its closing line:
> *"Does parity provide independent, stable, and clinically useful
> information beyond temporal CTG risk, and does treating parity as a
> contextual modifier improve the model over treating it as another
> additive scalar predictor?"*

---

## 0. Staging decision (binding)

The source plan specifies ~40 model variants across five axes (score
transform, fusion algorithm, temporal representation, parity encoding,
regularization) plus a five-figure/five-table reporting structure. Its own
§13–14 stage this into three tiers and calls Stage 1 *"the minimum
experiments needed to answer the current conceptual question."*
**This phase implements Stage 1 only.** Stage 2 (temporal representation:
R3 window-statistics, R4 sequence-encoder, R5 attention-weight features)
and Stage 3 (nonlinear meta-models: F10–F13 trees/forests/boosting/neural)
are deferred to separate, later phases — both require materially different
data plumbing (raw per-window M3 sequences/attention weights, held out here
only as an artifact field, not yet modeled) and, for Stage 3, justification
that Stage 1's simpler models actually leave residual signal on the table
before spending model capacity chasing it.

**What Stage 1 answers:** of the plan's three framing questions —
(1) does parity add information beyond M3, (2) which fusion mechanism
combines the two best, (3) is any improvement stable — Stage 1 answers (1)
and (2) at the score level, and puts the stability checks (resampling,
early-warning horizons, untouched test set) on every model in scope, not
deferred.

---

## 1. Experimental foundation (per plan §1)

### 1.1 Cohort and endpoint
- 547 patients, canonical `data/processed_clinical/folds.json`, patient-grouped 5-fold CV.
- 83-patient held-out internal test partition (`data/processed_clinical/test_dataset.pt`).
- Primary endpoint: pH ≤ 7.15 (110 positive, 20.1%). Severe endpoint (pH ≤ 7.05) not evaluated in Stage 1 — out of scope, flagged for a later pass if requested.
- **Canonical patient-inclusion rule, no exceptions:** every model is evaluated over the full N (547 CV / 83 test) at every horizon. No patient is ever dropped for lacking an eligible window — this is the exact bug found and fixed in the Model3+Parity Hybrid investigation (`results/model3_parity_hybrid/RECONCILIATION_AUDIT.md`); Stage 1's evaluator reuses the same canonical fallback logic (`src.evaluation.phase13_common.get_patient_scores_at_horizon_corrected`, `eligible_prefix_length`) throughout, by construction, not by a later audit.
- All metrics are derived programmatically from saved prediction arrays (per plan §1.3), never manually transcribed.

### 1.2 Frozen upstream components (never retrained, never re-tuned)
- **P6**: `results/phase13/audit/p6_predictions.npz` (frozen OOF/test window scores).
- **Model 3 (M3)**: the five fold checkpoints + test-model checkpoint at `results/phase16/checkpoints/Model_3_magnitude_position_*.pt`, scored via `predict_at_horizon_for_patients` — identical procedure used throughout Phases 16–17.
- **Parity model procedure**: per-fold `StandardScaler` + `LogisticRegression(C=1.0)` on raw parity, fit on training patients only — identical procedure used in `scripts/parity_fusion_test.py` and `scripts/model3_parity_hybrid/hybrid_engine.py`.
- **CTG preprocessing, windowing, fold assignment**: unchanged from every prior phase.

Only the **fusion layer** varies across Stage 1's models. Nothing upstream of `(m3_score, parity_score)` per patient per horizon is touched.

### 1.3 Prediction artifact schema
Every model's per-patient, per-horizon predictions are saved with the fields the plan specifies:
`patient_id, fold_id, split, outcome, parity_raw, parity_standardized, parity_probability, parity_logit, m3_score, m3_logit, prediction_horizon`, plus per fusion model: `fusion_method, fusion_hyperparameters, fusion_prediction, fusion_logit`.
(`m3_window_scores`, `m3_window_times`, `m3_attention_weights` are Stage-2 fields — not populated in Stage 1's flat score-level artifact, since no Stage-1 model consumes them.)
**Every fusion method receives exactly the same upstream `(m3_score, parity_score)` pair for the same patient at the same horizon** — enforced by construction (one shared upstream-scoring pass, reused by every fusion model).

---

## 2. Stage 1 model set (11 items, per plan §13 Stage-1 table)

| ID | Model | Fit | Regularization |
|---|---|---|---|
| B0 | Prevalence-only | training-fold event rate, constant per fold | — |
| B1 | Parity-only | `p_parity` (frozen parity model) | — |
| B2 | M3-only | `m3_score` (frozen M3) | — |
| F2 | **Current Hybrid (reference)** | `logit(p) = β0+β1·z_M3+β2·z_parity`, standardized inputs | L2, C=0.1 (+ full grid {0.001,...,100} for the sensitivity curve, §8) |
| A3 | Probability-space fusion | `p = α·p_M3+(1−α)·p_parity` | α ∈ {0.25, 0.5, 0.75, train-selected} |
| A4 | Logit-space fusion | `logit(p)=γ0+γ1·ℓ_M3+γ2·ℓ_parity` on **raw** (unstandardized) logits | unconstrained / non-negative / sum-to-1 / equal-weight variants |
| F1 | Unregularized logistic | same form as F2, standardized inputs | C=1e6 (≈ unregularized) |
| F3 | L1 logistic | same form as F2, standardized inputs | L1, C=0.1 |
| F4 | Elastic-net logistic | same form as F2, standardized inputs | l1_ratio ∈ {0, 0.25, 0.5, 0.75, 1}, C=0.1 |
| F7 | Interaction logistic | `logit(p)=β0+β1·z_M3+β2·z_parity+β3·z_M3·z_parity` | L2, C=0.1 |
| P3 | Categorical-parity fusion | z_M3 + one-hot{parity=0, 1, 2, ≥3} (fixed, data-independent buckets) | L2, C=0.1 |

F8 (hierarchical baseline-plus-modifier) is noted but not separately fit: its additive form is algebraically identical to F2 and its interaction form is algebraically identical to F7 (the plan itself notes this: *"mathematically similar... interpretation is different"*) — Stage 1 reports F2/F7's numbers under both framings rather than re-fitting an identical model twice.

**Primary scientific comparison (plan §17):** M3-only (B2) vs. M3+parity additive fusion (F2) vs. M3+parity interaction fusion (F7) — every other Stage-1 model is a secondary/exploratory comparison against this spine.

---

## 3. Horizons, splits, statistics

- Horizons: delivery (h=0), ≥10m, ≥20m, ≥30m — the four already used project-wide. 40/45/60m deferred (plan marks these "if feasible"; not run in Stage 1).
- 5-fold patient-grouped CV (primary evidence) + held-out internal test partition (confirmatory only, never used for any model or hyperparameter selection).
- Any hyperparameter (α, C, l1_ratio) is selected via training-fold-only grid search (`select_lambda_trainfold`-style: maximize training-fold AUROC, apply unchanged to that fold's held-out patients) — never on the reported fold, never on the test partition, matching this project's standing rule (`src/training/protocol.py`).
- Statistics: patient-level paired bootstrap (B=2000, seed 42) + DeLong, exactly as every prior phase. Primary comparisons: each Stage-1 model vs. B2 (M3-only) and vs. F2 (current hybrid reference).

---

## 4. Threshold / operational protocol

Deferred in Stage 1's primary report (AUROC/AUPRC/calibration only). Per-fold, training-only, 80%-target-sensitivity operational metrics (plan §11) will be computed for the primary spine (B2, F2, F7) only, reusing the exact convention already established in `scripts/model3_parity_hybrid/hybrid_engine.py`'s (bug-fixed) operational evaluator — not re-derived from scratch.

## 5. Reporting

Stage 1 produces: Table 1 (model definitions), Table 2 (AUROC by horizon, all 11 models, CV+test), Table 4 (incremental parity contribution — B2 vs. every fusion model), Table 5 (coefficient stability across folds for every linear model). Figures are not produced (this project's convention throughout has been CSV + markdown tables, not plots) — Table 2/4 in CSV form serve the same purpose and are more directly checkable.

## 6. What would count as a meaningful result (plan §15, adopted unchanged)

A fusion method is judged promising only if it clears *most* of: improves over M3-only across multiple horizons; improves over parity-only especially at early horizons; is not dependent on one arbitrary regularization setting; is stable across folds/bootstrap; has a CI supporting real improvement; does not worsen calibration; holds on the untouched internal test partition. A single best-of-11 AUROC number is explicitly **not** sufficient — this is stated up front, before any model is fit, so it cannot be relaxed after seeing results.
