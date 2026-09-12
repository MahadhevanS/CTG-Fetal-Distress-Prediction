# Information-density adaptive weighting — design document

Written 2026-09-11, before any code changes. This is a design spec for
implementation, not a results report — no model has been retrained against
this mechanism yet. Every number cited below is already reproducible from
`data/processed_clinical/` via the two diagnostics referenced in Section 1.

**Decision recorded here:** this mechanism will be implemented and validated
to completion *before* the window-extraction stride sweep is run. Section 6
explains why stride comparisons would otherwise be confounded by exactly the
redundancy effect this mechanism corrects.

---

## 0. Research question

Two related problems were identified empirically (not assumed) on the current
`data/processed_clinical/` cohort (547 patients, 110 primary positives,
20-minute windows, 2.5-minute stride):

1. **Uneven information density across windows.** Consecutive windows are, on
   average, 97–98% cosine-similar — but similarity is lower (i.e. windows
   carry more distinct information) closer to delivery, and this gradient is
   roughly twice as steep in distress-positive patients specifically.
2. **Uneven patient representation in training.** All patients get windows at
   the same fixed stride regardless of recording length, so a patient with 17
   windows contributes ~2–6x more (highly autocorrelated) training examples
   to the loss than a patient with 3–9 windows, with no correction anywhere
   in the current training scripts.

The question: can a single weighting mechanism, derived from measurements
already taken on this cohort rather than a hand-picked decay function, correct
(2) and productively exploit (1) at the same time — without repeating the
generalization failure of the earlier hand-crafted label-confidence-weighting
experiment (commit `10ce497`: CV AUROC +0.019, held-out test worse, not
adopted)?

---

## 1. Empirical grounding

### 1.1 Bag-size / count-leak audit (`scripts/audit_bag_size_leak.py`)

Run against `data/processed_clinical/` on 2026-09-10:

| Split | Patients | Windows | AUROC(window count → label) | Verdict |
|---|---|---|---|---|
| train | 381 | 5,931 | 0.4576 | PASS |
| val | 83 | 1,312 | 0.4973 | PASS |
| test | 83 | 1,274 | 0.4675 | PASS |

No count-based label leak (unlike the deprecated `data/processed/` pipeline,
which scored 0.9947 due to label-dependent stride). Window count is *not* a
label proxy here — but it still creates a within-fold weighting imbalance,
which is a different problem from leakage (Section 1.2).

### 1.2 Consecutive-window redundancy (19-descriptor + extended feature vector, train split)

- 5,550 consecutive-window pairs across 381 patients (all have ≥2 windows).
- Mean Pearson r = 0.971, median = 0.996.
- Mean cosine similarity = 0.977; **67.4%** of pairs exceed 0.99 cosine
  similarity; **86.1%** exceed 0.95.
- Baseline (first window vs. last window per patient, maximum possible
  temporal separation): mean cosine similarity 0.886 — still high, since
  physiology genuinely changes slowly, but consecutive windows are
  substantially more similar than this ceiling, confirming genuine
  stride-driven oversampling rather than just slow physiology.

### 1.3 Redundancy vs. time-before-delivery

Using each patient's last window's end as the delivery-aligned reference
(consistent with `time_before_delivery_min` / `w_minutes_before_end` elsewhere
in the pipeline):

| Horizon band (min before delivery) | Pairs | Mean cosine sim | % pairs >0.99 |
|---|---|---|---|
| 0–5 | 759 | 0.9747 | 60.7% |
| 5–10 | 749 | 0.9740 | 64.5% |
| 10–15 | 741 | 0.9749 | 66.3% |
| 15–20 | 722 | 0.9772 | 70.8% |
| 20–30 | 1,372 | 0.9780 | 67.6% |
| 30–45 | 1,207 | 0.9795 | 71.7% |

Pearson correlation (time-before-delivery vs. cosine similarity) = **+0.043**
overall (weak but consistent in direction — similarity decreases as delivery
approaches). Split by outcome:

- All patients: far (>20 min) mean cosine 0.9787 vs. near (≤20 min) 0.9752 (Δ≈0.0035)
- **Distress-positive patients only**: far 0.9753 vs. near 0.9687 (**Δ≈0.0066** — roughly 2x the pooled effect)
- Distress windows are less self-similar than normal windows overall (0.9717 vs. 0.9780), consistent with unstable physiology changing faster than a stable trace.

**Implication:** redundancy is pervasive at every horizon (even 30–45 min out,
71.7% of pairs are near-identical), so this is not solely a near-delivery
phenomenon and correcting it should not be scoped as such. But the region
that matters most clinically — distress-positive patients near delivery — is
also where windows are least redundant, i.e. uniform weighting currently
under-uses exactly the windows carrying the most real information.

---

## 2. Mechanism overview

Three components, combined multiplicatively, applied per window:

```
w_i = class_weight_i  ×  patient_normalization_i  ×  novelty_weight_i
```

`class_weight_i` is whatever imbalance correction is already in use
(`pos_weight` in `FocalLoss`, `class_weight='balanced'` in the GBM) and is
**not replaced** — the new terms are multiplied on top of it so redundancy
correction and class-imbalance correction don't fight each other.

### 2.1 Patient normalization (structural — no free parameters)

```
patient_normalization_i = K / n_windows(patient(i))
```

Rescales so every patient contributes a fixed total vote to the loss
regardless of recording length or window count. `K` is a constant chosen so
the mean weight across the training set is 1.0 (keeps overall loss magnitude
comparable to the unweighted baseline for a clean before/after comparison).

This term alone directly corrects the imbalance measured in Section 1.1/1.2 —
it requires no novelty computation and has no tunable behavior beyond the
normalization constant.

### 2.2 Novelty score (causal, per window)

For window *i*, the *j*-th window of patient *p* (*j* = 1, 2, ... in
chronological order):

```
novelty_i = 1 - cosine_similarity(X_i, X_{i-1})       for j > 1
novelty_i = fold_median_novelty                        for j = 1 (no predecessor)
```

`X_i` is the existing 19-descriptor (+ extended) feature vector, each
dimension z-scored by its global training-fold std before the distance is
computed, so no single noisy descriptor (e.g. `fhr_uc_lag`) dominates.

**Smoothing:** raw single-pair novelty is noisy (STV/LTV are already
known-noisy statistics on 20-minute windows). Apply a short causal EWMA over
the last 2–3 novelty values:

```
novelty_i_smoothed = EWMA(novelty_i, novelty_{i-1}, novelty_{i-2}; span=3)
```

### 2.3 Elapsed-time empirical prior (shrinkage, fold-internal)

Rather than assuming a decay shape, reuse the measurement in Section 1.3
directly. Within each **training** fold only (never touching val/test, to
avoid leakage):

```
fold_bin_avg(elapsed_i) = mean novelty_smoothed for windows in the same
                           elapsed-time / horizon bin as window i,
                           computed strictly on that fold's training partition
```

Bin boundaries mirror Section 1.3's table (0–5, 5–10, 10–15, 15–20, 20–30,
30–45+ min) or, if elapsed-since-monitoring-start is used instead of
time-before-delivery (see Section 2.4 on causality), an analogous binning on
window index *j*.

```
novelty_i_final = beta * novelty_i_smoothed + (1 - beta) * fold_bin_avg(elapsed_i)
novelty_weight_i = novelty_i_final / mean(novelty_i_final over training fold)
```

`beta` is the one real hyperparameter in this mechanism (how much to trust an
individual window's own signal vs. the fold's empirical horizon pattern) and
must be selected via the standard CV protocol (Section 5), not hand-set.

### 2.4 Causality note — two legitimate variants, pick one deliberately

`time_before_delivery_min` (used above for *analysis*, since it's available
offline) is **not causally observable during live monitoring**. Two ways to
resolve this for anything that touches training or a live weight:

- **Preferred:** bin by *elapsed monitoring time since recording start*
  (`window_index × stride`, or the existing `w_minutes_before_end`-complement
  if a monitoring-start reference is available), which is always causal.
  Section 1.3's horizon-band pattern is used only to *motivate* that this
  binning is worth doing, not as a literal input — the actual fold_bin_avg
  is recomputed against whichever causal time axis is chosen.
- **Analysis-only:** `time_before_delivery_min` may still be used for
  post-hoc evaluation/stratification (as Phases 8–12 already do throughout),
  just never as a training feature or live weight input.

This mirrors the resolution reached earlier in this design discussion:
elapsed-monitoring-time is not treated as an approximation of "closeness to
delivery" (which would require knowing something unobservable) but as its own
legitimate, causal, clinically plausible signal — cumulative labor/monitoring
exposure — independently motivated regardless of the delivery-time analysis.

---

## 3. Feature exposure (do this alongside, not instead of, the weight)

Per the earlier design discussion, exposing `novelty_i` and elapsed-time as
plain **input features** (added to the existing 104-dim vector built in
`scripts/phase9_temporal_features.py`, alongside the current 1/2/4-step
deltas and slopes) is lower-risk than only using them as loss weights — it
lets the model learn a nonlinear use of the signal rather than imposing one.
Both the feature and the weight should be implemented, but validated as
separable steps (Section 5) so any effect can be attributed correctly.

---

## 4. Wiring into existing infrastructure

No new training-loop infrastructure is required — both hooks already exist:

- **Classical models** (`HuberRegressor`, `HistGradientBoostingClassifier` in
  the Phase 6/9 scripts): `sample_weight=w` is already a supported
  constructor/`.fit()` argument.
- **Deep crossformer**: `criterion(logits, y_batch, w_batch)` is already
  wired in `train_epoch()` at
  [src/models/train_ctg_crossformer.py:187](../src/models/train_ctg_crossformer.py#L187),
  and `FocalLoss` already accepts and applies a `sample_weight` term
  ([src/models/train_ctg_crossformer.py:79](../src/models/train_ctg_crossformer.py#L79)).

Implementation work is therefore a weight-computation module upstream of
training (Sections 2.1–2.3), not a change to any model architecture or
training loop.

---

## 5. Validation ladder

Matches the project's established Phase 11.5 component-ladder methodology —
incremental, each step attributable, each step checked against held-out test
(not CV alone), given the precedent of a weighting scheme that improved CV
but regressed on test.

| Step | Change | Isolates |
|---|---|---|
| 0 | Current Full System (P6) baseline, unweighted | — |
| 1 | + novelty/elapsed features as model input only, weights unchanged | Whether the model can use the signal at all |
| 2 | + patient-normalization weight only (Section 2.1), novelty untouched | The redundancy-correction effect alone |
| 3 | + novelty-based weighting on top of step 2 (Section 2.2) | The information-density effect alone |
| 4 | + elapsed-time shrinkage prior blended in (Section 2.3), `beta` tuned via CV | The time-prior refinement |

Per step: patient-grouped 5-fold CV, paired bootstrap (B=2,000) and DeLong
test against the previous step, on both the primary pH≤7.15 endpoint and the
early-warning operational metrics (median lead time, false-alert rate)
already established in Phases 10–12. A step is adopted only if it holds on
the held-out test set, not just CV — the standard this project has applied
since Phase 7.1.

All binning/shrinkage statistics (fold_bin_avg, per-feature z-score stats,
`K`) must be fit strictly within each training fold and applied unchanged to
that fold's validation/test partition, to avoid leaking cross-patient
statistics across the same boundary Phase 1 established for the raw data.

---

## 6. Why this must precede the stride sweep

At the current 2.5-minute stride, patients average ~17 windows each (capped
by the 60-minute truncation). A finer stride (e.g. 1 minute) would produce
roughly 2.5x more windows per patient; a coarser stride (e.g. 5 minutes)
roughly half. Comparing stride arms without the patient-normalization
correction (Section 2.1) in place means the finer-stride arm would
mechanically carry a larger per-patient redundancy imbalance than the
coarser-stride arm — reintroducing a bag-size-shaped confound through a
different door than the one `scripts/audit_bag_size_leak.py` was built to
catch. Any AUROC difference observed between stride arms would be entangled
with this count effect rather than isolating the effect of stride itself.

Implementing and locking this mechanism first — rather than only the minimal
Section 2.1 normalization — also avoids recalibrating twice: the novelty and
elapsed-time components (Sections 2.2–2.3) are themselves sensitive to
stride, since finer strides push consecutive-window similarity even higher
and coarser strides push it lower. Finalizing the full mechanism against the
current, already-measured 2.5-minute-stride cohort, then re-running the same
validation ladder once a new stride is chosen, is the ordering adopted here.

---

## 7. Open parameters (to be resolved via CV during implementation, not hand-set)

| Parameter | Role | Resolution method |
|---|---|---|
| `beta` (Sec. 2.3) | individual novelty vs. fold horizon-bin average | CV sweep, held-out test confirmation |
| EWMA span (Sec. 2.2) | novelty smoothing window | small grid (2–4), CV sweep |
| Horizon/elapsed-time bin edges (Sec. 2.3) | granularity of the empirical prior | start from Section 1.3's bands, sensitivity-check |
| Distance metric (Sec. 2.2) | cosine vs. z-scored L2 vs. Mahalanobis-style | start with z-scored cosine (matches diagnostics already run), revisit if step 3 underperforms |
| `K` (Sec. 2.1) | patient-normalization scale | fixed analytically (mean weight = 1.0), not tuned |

---

## 8. Risks and mitigations

- **Chasing measurement noise.** STV/LTV and related descriptors are
  known-noisy on individual 20-minute windows; a raw novelty score could
  up-weight noise rather than true physiological change. Mitigated by the
  EWMA smoothing (2.2) and the shrinkage toward the fold's empirical prior
  (2.3), and checked directly by the step-by-step held-out test gate (5).
- **Repeating the label-confidence-weighting failure mode.** That scheme
  used an untested hand-crafted function and passed CV but failed test. This
  mechanism differs in kind — its shape comes from measurements already
  taken on this cohort (Section 1.3), recomputed strictly per training fold
  — but the same test-set discipline applies regardless of how principled
  the design looks on paper.
- **Interaction with existing class weighting.** Patient-normalization and
  novelty weighting are applied multiplicatively alongside `pos_weight` /
  `class_weight='balanced'`, not as a replacement — verify at step 2 that
  class balance in the effective (post-weighting) training distribution
  hasn't drifted before attributing any change to redundancy correction.
- **First-window edge case.** No predecessor exists for a patient's first
  window; resolved by falling back to the fold's median novelty (Section
  2.2), not zero or an arbitrary constant.
- **Leakage through fold-fit statistics.** All calibration statistics
  (`fold_bin_avg`, z-score stats, `K`) are fit on the training partition of
  each fold only and applied to that fold's val/test partition — same
  boundary already enforced for scalers elsewhere in this pipeline.

---

## 9. Expected implementation touch points (for scoping only — not yet changed)

- `scripts/phase9_temporal_features.py` — extend `FEATURE_NAMES` /
  `expanded_feature_names` with `novelty`, `novelty_smoothed`, and an
  elapsed-monitoring-time feature.
- New weight-computation module (upstream of training) implementing Sections
  2.1–2.3, fit per fold.
- `src/training/train_knowledge_infused.py` / the Phase 6 Huber training
  script — accept and pass through `sample_weight`.
- `src/models/train_ctg_crossformer.py` — populate `w_batch` from the new
  module instead of its current (class-imbalance-only) source, multiplicatively.
- A new Phase 13-style evaluation script mirroring the Phase 11.5 component
  ladder (`scripts/phase11_5_*` pattern) implementing the 5-step validation
  ladder in Section 5.
