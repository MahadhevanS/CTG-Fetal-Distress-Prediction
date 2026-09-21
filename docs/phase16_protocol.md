# Phase 16 — Trainable Patient-Level Temporal Aggregation
## Pre-Registration Protocol

**Frozen 2026-09-11, before any model code was written.** Governed by the
same discipline as `docs/phase13_protocol.md` / `phase14_protocol.md` /
`phase15_temporal_risk_representation_protocol.md`.

> **Governance rule (unchanged).** Phase 12.1 is locked and authoritative
> regardless of any result here. P6 is not retrained, not modified, not
> replaced. This phase asks whether a small trainable component *on top of*
> P6's already-frozen window-level scores can learn what Phases 13–15 could
> only detect by fixed aggregation — it does not touch P6 itself.

## 0. What changed from the proposal, and why (read before implementing)

Three corrections were made to the originally discussed design before
freezing this protocol. Each is binding.

**(a) `x_t` is scalar, not a rich embedding.** The only frozen per-window
artifact available is P6's window-level probability `r_t`
(`results/phase13/audit/p6_predictions.npz`). There is no learned per-window
embedding anywhere in the current pipeline to attend over — P6's 40-D
feature vector is hand-engineered, not a hidden state. Every model in this
protocol operates on `x_t = (r_t, elapsed_t)` only, where `elapsed_t` is the
gap-aware elapsed monitoring time already established in Phase 13
(`start_sample / (fs*60)`, offset per patient). No model in this phase is
given access to the 40-D feature vector. This is a capacity decision, not a
simplification of convenience — see (c).

**(b) Strategy B (end-to-end fine-tuning) is out of scope for Phase 16.**
P6's construction — descriptor formulas, FIGO state rules, trajectory
arithmetic, the frozen `HuberRegressor`, EWMA smoothing — contains no
differentiable path from raw CTG to risk score. "End-to-end fine-tuning"
would require first rebuilding large parts of P6 as differentiable
approximations, which is a separate, substantially larger undertaking, not
a natural continuation gated on this phase's results. If Phase 16 succeeds,
that rebuild is a candidate for its own, independently-scoped future phase
— not assumed here.

**(c) Direct precedent, stated as the explicit falsification target.**
Phase 3 already tried trainable attention-based patient-level aggregation
on this cohort — over learned signal embeddings, not scalar scores — and it
failed badly: **AUROC 0.5337**, against fixed P90's 0.6701, attributed
explicitly to insufficient patient-bag count for learning flexible
aggregation (`reports/final_project_synthesis_report.md` §5). Phase 16's
working hypothesis is that a *much* lower-capacity model (attention over
2-dimensional scalars, not high-dimensional embeddings) can avoid that
failure mode. If Phase 16 reproduces Phase 3's collapse, that is a
meaningful, reportable result — evidence the earlier Max/P90 signal is more
likely a fixed-statistic phenomenon than genuinely learnable structure —
not a bug to be engineered around.

## 1. Scientific objective

Can a trainable, causally-constrained, low-capacity aggregator extract
predictive information from the sequence `r_1, ..., r_T` of frozen P6
window-level scores that fixed aggregation (single-window, P90, max, mean —
Phase 15) could not confirm?

## 2. Epistemic framing (binding, unchanged from Phase 15 §2)

Same 547-patient cohort. No language claiming "validated," "confirmed," or
"externally validated" anywhere in reporting. A positive result here means
*worth external validation*, never *ready to replace P6*.

## 3. Frozen inputs, no retraining of P6

`results/phase13/audit/p6_predictions.npz` (`pred_unweighted_cv`,
`pred_unweighted_test`), `data/processed_clinical/folds.json`,
`data/processed_clinical/test_dataset.pt`,
`results/phase8_rolling/rolling_predictions.csv` (for `start_sample`,
`time_before_delivery_min`, labels). P6 itself is never re-fit; only the
new aggregator's parameters are trained.

## 4. Causal construction — training data augmentation via chronological truncation

For patient *i* with causally-ordered windows `r_1, ..., r_{T_i}`
(ascending chronological order = descending `time_before_delivery`, per
Phase 13's established convention), every truncation length
`k = 1, ..., T_i` is a separate training example:
`({r_1..r_k}, {elapsed_1..elapsed_k}) → y_i` (the patient's actual outcome,
identical across all of that patient's truncations). This:

1. Guarantees causality by construction — the model never receives an
   array containing `r_{k+1..T_i}` when producing a prediction meant to be
   valid at truncation `k`, because the array itself is truncated before
   it reaches the model, not masked inside a fixed-length buffer.
2. Makes a single trained-per-fold model usable at *every* horizon: since
   `time_before_delivery ≥ h` selects a chronological prefix (monotonic by
   construction), evaluating at horizon `h` is exactly evaluating the model
   on the prefix ending at that horizon's eligible window.
3. Multiplies effective training examples (up to `T_i`, capped at 17, per
   patient) — directly addressing the small-cohort constraint that Phase
   3's attention MIL foundered on.

## 5. Model family

| Model | Description | Trainable parameters |
|---|---|---|
| 1 (baseline) | Single-window (`get_patient_scores_at_horizon_corrected`) — already computed, Phase 13/14/15's control | none (reused, not re-run) |
| 2 | Magnitude-only causal attention: `e_t = f(r_t)`, `α = softmax(e)` over the truncated prefix, `ŷ = Σ α_t r_t` | tiny MLP `f`: 1→8→1 (~25 params) |
| 3 | Magnitude + temporal position: `e_t = f(r_t, elapsed_t)`, same pooling | tiny MLP `f`: 2→8→1 (~33 params) |
| 4 | Peak-aware fusion — gated on 2 or 3 showing signal (§9) | logistic regression on `[max(prefix), p90(prefix), mean(prefix), z_from_model_3]` (4 params) |
| 5 (Strategy B) | End-to-end fine-tuning | **out of scope**, §0(b) |

`ŷ = Σ α_t r_t` is deliberately chosen over adding a further classifier
head `g(z)` — the pooled output is already a probability-weighted average
of probabilities, staying in `[0,1]` without an extra learned layer, and it
directly generalizes every Phase 15 aggregator as a special case (uniform
`α` = mean; `α` concentrated on one window = single-window or max). This
keeps every model in this family lower-capacity than Phase 3's failed
attempt by construction, not just by input dimensionality.

Model 4's fusion layer is a plain logistic regression on four scalars, not
a deep fusion network — deliberately, given this project's repeated finding
(Phase 4, Phase 13) that flexible fusion architectures lose to low-capacity
ones here.

## 6. Training procedure (fixed, not tuned)

- Loss: binary cross-entropy between `ŷ` and `y_primary`, summed/averaged
  over all truncated-prefix examples across training patients.
- Architecture: hidden dim 8, one hidden layer, tanh activation — fixed,
  not swept, given the parameter count is already small enough that
  sweeping would risk more overfitting from the search than from the model.
- Optimizer: Adam, lr=0.01, weight_decay=1e-4 — fixed.
- Early stopping: patience 10 epochs, monitored on an **inner validation
  split carved from that fold's training patients only** (never the outer
  held-out fold, never the test partition) — same "no model selection on
  the reported fold" rule `src/training/protocol.py` already establishes
  project-wide.
- Patient-grouped 5-fold CV on the canonical `folds.json` assignment,
  identical to every prior phase. Held-out test: trained once on all 464
  train+val patients (inner validation carved from them), evaluated once
  on the 83 test patients.
- No hyperparameter grid, no post-hoc architecture search. If Model 2/3
  shows a signal worth pursuing, a *separately scoped* robustness pass
  (seed sensitivity, architecture sensitivity) is the appropriate next
  step — not folded into this first pass.

## 7. Evaluation

Delivery and ≥30m reported as the two horizons of interest (neither sole
primary, consistent with Phase 15); ≥10m/≥20m computed for completeness,
exploratory only. At each horizon, the trained model is queried on the
causally-truncated eligible prefix for that horizon (§4.2) — not on a
separately retrained model. AUROC, AUPRC, patient-level bootstrap
(B=2000) paired against Model 1, DeLong, on both CV and held-out test.
Operational metrics (sensitivity/specificity, FAR, median lead time, %
detected ≥20/30min) computed identically to Phase 15 §8.

## 8. Decision framework — explicitly not a p<0.05 gate

Per the discussion preceding this protocol: statistical significance
quantifies uncertainty here, it does not gate whether a model is judged
worth carrying forward. A model is assessed holistically on:

1. **Direction and magnitude** — CV point estimate vs. Model 1, with
   bootstrap CI reported (not thresholded).
2. **CV/test agreement** — does the held-out test point estimate agree in
   direction with CV, at both delivery and ≥30m.
3. **Mechanistic interpretability** — do learned attention weights `α_t`
   concentrate near the same window(s) that Max/P90 pooling would select
   (Phase 15's finding that magnitude, not trajectory, carries the
   signal)? This is checked directly by correlating each patient's
   arg-max-`α_t` window with their arg-max-`r_t` window.
4. **Overfitting signature** — train-vs-inner-validation loss gap, and
   whether the five outer folds' results are wildly inconsistent (a red
   flag independent of the aggregate point estimate — this is exactly what
   caught the P90+Parity hybrid's instability in Phase 13).

**Explicit falsification condition:** if Model 2/3's CV AUROC is
materially *below* Model 1 (reproducing Phase 3's attention-collapse
pattern) and/or attention weights show no relationship to peak-window
location, that is reported as a meaningful negative result — evidence the
Max/P90 signal is more likely a fixed-statistic phenomenon than something a
trainable aggregator can recover — not a reason to keep tuning until it
looks better.

## 9. Gating between models

Model 3 runs regardless of Model 2's result (it's the natural ablation —
does adding temporal position help beyond magnitude alone). Model 4 runs
only if Model 2 or 3 clears criteria 1–3 of §8 with no overfitting red flag
under criterion 4. If neither Model 2 nor 3 shows a holistically positive
read, Model 4 is not run and Phase 16 closes with Models 2/3's null as the
result.

## 10. What this phase explicitly is not

Not a hyperparameter search, not a combinatorial architecture search, not a
re-opening of Phases 13/14/15's closed threads (A/B/C from Phase 13 remain
closed; this phase only extends the Max/P90/single-window thread). Not a
claim that any resulting model is ready for deployment or promotion over
P6. If a model here shows a genuinely positive, holistically-read signal,
the recommendation is external-cohort validation — same as Phase 15's own
closing recommendation, not a new conclusion invented here.

## 11. Amendments

None yet.
