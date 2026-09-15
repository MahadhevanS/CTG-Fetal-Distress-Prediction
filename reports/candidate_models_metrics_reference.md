# Candidate Models — Complete Metrics & Performance Reference

Written 2026-09-12, revised 2026-09-13 (Tier-1 hardening pass per
`docs/pre_external_validation_hardening_plan.md` — canonical estimator
policy, precise time-representation language, verified leakage-control
description, AUPRC-vs-prevalence framing, delivery/early-warning/
operational separation, terminology standardization), revised again
2026-09-13 (Tier-2 hardening pass — parity horizon-convention correction
applied at §4.2, new direct Model-3-vs-Max/P90 comparisons at §1.2a, new
Model 3 duration-confound check at §1.2b). Every number is sourced
directly from committed result files — nothing is recalled or
approximated. Tier 2 changed two real conclusions, in opposite directions:
it **strengthened** parity fusion's early-warning case (§4.2, §4.5) and it
**weakened** (without reversing) Model 3's "strongest candidate" framing
(§1.2a, §1.2b, §1.6). Revised again 2026-09-13 (Tier 3 — the shuffled-time
control, §1.2c): the pre-registered decision rule scores the result
**"capacity artifact"** at the primary delivery endpoint (shuffled-time
Model 3 collapses to Model 2's performance; its ~0.02 AUROC gap below
real-time Model 3 does not reach significance, p=.217, stable across 3
independent shuffle draws). This is reported as computed, not softened —
but it is not a clean refutation either: the same gap is significant at
the exploratory ≥10m horizon (p=.009), where scrambled time actively hurts
below the P6 baseline rather than being harmlessly ignored. All three
Tiers are now complete; every finding, favorable or not, is reported in
full per this project's standing rule against discarding inconvenient
results. **Revised again 2026-09-15 — added §5, the Phase 18 M3+Parity
deployable fusion candidate.** This corrects a design flaw shared by every
fusion model in the project up to that point (including the earlier
Parity Fusion λ and the Model 3+Parity Hybrid, `results/model3_parity_hybrid/`):
each was fit with separate coefficients per retrospective horizon, which
is not deployable, since minutes-before-delivery is unknown during actual
labor. The corrected version — one fixed model, trained pooled across
every causal truncation, never told the horizon — is the single most
thoroughly verified result in this document: a fold-resplit check (not
just a bootstrap-seed check) confirms it significantly beats Model 3 alone
at ≥20m/≥30m across all 6 independent patient partitions tried. See §5 for
the full verification history, including two real bugs caught along the
way.

**Scope.** The five candidates that have emerged from the post-lock
investigation (Phases 13–18), plus the two reference points needed to
interpret them:

| # | Model | Origin | Type |
|---|---|---|---|
| — | **P6 single-window** (= Phase 16's "Model 1") | Phase 12.1 locked | Reference baseline |
| — | Model 2 — magnitude-only attention | Phase 16 | Context (rediscovers max-pooling) |
| **1** | **Model 3 — magnitude + temporal position attention** | Phase 16 | **Trainable candidate** |
| **2** | **Model 4 — peak-aware fusion** | Phase 16 | Trainable candidate |
| **3** | **P90 pooling** | Phase 13E | Fixed-statistic candidate |
| **4** | **Parity fusion** | Phase 13D | Independent-covariate candidate |
| **5** | **M3 + Parity deployable fusion** | Phase 18 | Deployable multimodal candidate |

**Cohort throughout:** 547 CTU-UHB patients, 110 primary positives
(pH ≤ 7.15, prevalence 20.1%), 8,517 causal 20-minute windows, patient-
grouped 5-fold CV, an 83-patient **held-out internal test partition** (not
an external cohort — see `docs/external_validation_handoff.md` for what
that would mean). CV is the primary evidence source (larger sample); the
held-out internal test partition is reported separately and was never used
to select anything.

> **Governance (restated, unchanged from every Phase 13–16 report):** none
> of the four candidates below is promoted, deployed, or treated as
> clinically ready. Phase 12.1 remains the sole locked production model.
> Every status line in this document ends at "worth external validation,"
> never at "ready for use" — that determination requires the study scoped
> in `docs/external_validation_handoff.md`, not anything in this reference.

---

## 0. Reference baseline — P6 single-window (Phase 12.1, locked)

| Horizon | CV AUROC | Held-out internal test AUROC |
|---|---|---|
| Delivery | 0.6872 | 0.6497 |
| ≥10m | 0.6859 | 0.5802 |
| ≥20m | 0.6238 | 0.6123 |
| ≥30m | 0.5828 | 0.6078 |

Every Δ below is measured against these. (The originally published Phase
12.1 ≥30m figure, 0.5857, used the *original* horizon convention; 0.5828
is the same model under the corrected convention — see
`results/phase13/audit/horizon_alignment_comparison.csv`.) P6's own AUPRC
was not separately cached alongside these AUROC figures in any committed
result file — a Tier-2 item if a baseline lift-over-prevalence figure is
needed for item 7's comparison below.

---

## 1. Model 3 — magnitude + temporal position attention ⭐ strongest candidate

**Mechanism.** A trainable causal-attention aggregator over P6's own frozen
window-level risk sequence. For each window *t*, an attention logit
`e_t = f(r_t, elapsed_t)` is produced by a tiny MLP (2→8→1, ~33 trainable
parameters); the patient score is `ŷ = Σ softmax(e)_t · r_t` over the
causally-eligible prefix. No extra classifier head — the pooled output is
already a probability-weighted average of probabilities. P6 itself is
never retrained.

**Exact time representation** (precision requested — this is the complete,
unambiguous definition, not a paraphrase): `elapsed_t` = minutes since
*that patient's own first retained window* — `start_sample / (4 Hz · 60)`,
offset so the patient's own first window reads 0 — divided by 60 before
being passed to the scorer MLP. This is:
- **Not** time-before-delivery (using that would be non-causal — it
  requires knowing when delivery will occur).
- **Not** absolute wall-clock or gestational age.
- Gap-aware: correctly reflects the ~6.9% of patients with quality-gate-
  dropped windows, using each window's actual sample offset rather than
  its position index (the same fix established for elapsed-time
  computation in Phase 13).

### 1.1 Delivery discrimination (primary endpoint)

**Source:** `results/phase16/Model_3_magnitude_position_auroc_results.csv`

| | CV AUROC | CV AUPRC | Δ vs P6 | CV p (boot / DeLong) | CV 95% CI | Internal-test AUROC | Test Δ | Test p (boot / DeLong) |
|---|---|---|---|---|---|---|---|---|
| **Delivery** | **0.7216** | 0.4335 | **+0.0344** | **.014 / .018** | **[+0.0064, +0.0649]** | 0.6729 | +0.0232 | .157 / .113 |

**AUPRC relative to prevalence** (item 7): random-classifier AUPRC at 20.1%
prevalence = 0.201. Model 3's 0.4335 is a **2.16× lift** over that floor.

Per-fold stability at delivery: 0.7448, 0.7474, 0.7842, 0.6813, 0.6776
(std 0.0413) — reported here, not folded into the headline number, since
fold-to-fold spread is itself part of judging robustness, not a discrimination
metric.

### 1.2 Early-warning performance (exploratory — no horizon here is a
### primary endpoint; reported for completeness, not for headline claims)

| Horizon | CV AUROC | CV AUPRC | Δ vs P6 | CV p (boot / DeLong) | CV 95% CI | Internal-test AUROC | Test Δ | Test p (boot / DeLong) |
|---|---|---|---|---|---|---|---|---|
| ≥10m | 0.6683 | 0.3612 | −0.0176 | .141 / .141 | [−0.0415, +0.0066] | 0.5642 | −0.0160 | .617 / .617 |
| ≥20m | 0.6015 | 0.3238 | −0.0223 | .052 / .041 | [−0.0443, +0.0003] | 0.6203 | +0.0080 | .732 / .699 |
| ≥30m | 0.5976 | 0.3043 | +0.0148 | .237 / .232 | [−0.0097, +0.0398] | 0.6658 | +0.0579 | .232 / .247 |

### 1.2a Direct comparisons — Model 3 vs. Max, Model 3 vs. P90 (Tier 2, item 3)

**Source:** `results/model3_direct_comparisons.csv`, produced by pure
inference on Model 3's already-committed per-fold checkpoints (no
retraining) against Max/P90 reconstructed from the same frozen window
scores using the canonical estimator (item 2). Every prior comparison in
this document was "candidate vs. single-window P6"; this is the first
direct candidate-vs-candidate test.

| Horizon | CV: M3 vs Max (Δ, p) | CV: M3 vs P90 (Δ, p) | Test: M3 vs Max (Δ, p) | Test: M3 vs P90 (Δ, p) |
|---|---|---|---|---|
| Delivery | +0.0006 (.939) | +0.0038 (.820) | −0.0401 (.247) | +0.0107 (.869) |
| ≥10m | +0.0218 (.154) | +0.0247 (.112) | −0.0731 (.118) | −0.0722 (.182) |
| ≥20m | −0.0020 (.942) | +0.0000 (.954) | −0.0544 (.301) | −0.0579 (.279) |
| ≥30m | −0.0150 (.264) | −0.0114 (.329) | −0.0472 (.097) | −0.0321 (.248) |

**Finding, stated plainly:** Model 3 is **not statistically distinguishable
from either cheap, zero-training fixed aggregator at any horizon, in either
direction** — every CI in the full CSV crosses zero, every p-value is well
above .05. On the held-out internal test partition, the point estimates
actually favor Max/P90 over Model 3 at every horizon (Model 3 trails by
0.03–0.07 AUROC, though never significantly). This does not undercut
Model 3's original, separately-tested result against the P6 baseline
(§1.1 — that comparison and this one answer different questions), but it
does mean Model 3's training procedure has not yet been shown to extract
more from the sequence than a 90th-percentile calculation already does.
**This tempers, not reverses, Model 3's "strongest candidate" framing** —
it remains the only candidate with a positive, CI-clear-of-zero result
against the baseline, but it is not yet shown to beat the cheapest
alternative that requires no training at all.

### 1.2b Recording-duration confound (Tier 2, item 6)

**Source:** `results/model3_duration_confound.csv`, extending
`results/phase15/phase15_duration_analysis.csv` with one new row using
Phase 15's own `eligible_window_count()` helper, unchanged, at delivery.

| Representation | Pearson r (value vs. n_windows) | p | Pearson r (Δ vs. P6, vs. n_windows) | p |
|---|---|---|---|---|
| P90 (fixed) | −0.0551 | .198 | −0.1901 | <.0001 |
| Max (fixed) | +0.0073 | .864 | −0.1231 | .0039 |
| Mean (fixed) | −0.1478 | .0005 | −0.2315 | <.0001 |
| Median (fixed) | −0.1769 | <.0001 | −0.2316 | <.0001 |
| **Model 3 (trainable)** | **+0.0162** | .705 | **−0.2128** | **<.0001** |

(Spearman ρ on Model 3's delta: −0.2092, p<.0001 — consistent with the
Pearson figure, not a rank-order artifact.)

**Finding, stated plainly: Model 3 does NOT escape the confound Phase 15
found in every fixed aggregator — it shows one of the strongest instances
of it.** Model 3's advantage over single-window P6 shrinks with longer
recordings (r=−0.213) more than P90's (−0.190) or Max's (−0.123), and is
comparable in magnitude to Mean's and Median's — the two aggregators Phase
15 judged weakest. Being trainable did not let Model 3 learn to discount
long sequences; whatever it learned still correlates with how much
eligible signal is available, the same structural pattern driving every
fixed statistic's own duration-dependence. This is a real, unflattering
result reported in full per this project's standing rule against
discarding inconvenient controls (see §1.6 for how this changes — and
doesn't change — Model 3's status).

### 1.2c Shuffled-time control (Tier 3, item 5 — resolved)

**Source:** `scripts/model3_shuffled_time_control.py`,
`results/phase16/model3_shuffled_time_ablation_ladder.csv`,
`model3_shuffled_time_seed_sensitivity.csv`,
`model3_shuffled_time_control_summary.json`. Model 3's identical
architecture (same 2-input scorer, hidden=8, training procedure, canonical
folds) was retrained with `elapsed_t` values **randomly permuted within
each patient's own window sequence** — same value distribution and
parameter count as real Model 3, but the true correspondence between a
window's chronological position and its elapsed-time value is destroyed.
`r_t` and the causal eligible-prefix logic (which depends on
time-before-delivery, not elapsed_t) are completely untouched. Three
independent shuffle draws (seeds 777/888/999) were trained, not one, as a
shuffle-draw-sensitivity check.

**The decision rule was fixed in the script before any result was
computed** (see the script's docstring): genuine temporal-order use
requires shuffled-time Model 3 to be significantly worse than real-time
Model 3 *and* close to Model 2 (the zero-time-information endpoint);
capacity artifact requires shuffled-time to perform comparably to
real-time Model 3. This is applied mechanically below, not adjusted after
seeing the numbers.

**Three-point ablation ladder, delivery (CV):**

| | Model 2 (zero time info) | Shuffled-time Model 3 | Real-time Model 3 |
|---|---|---|---|
| AUROC | 0.7006 | 0.7003 / 0.6979 / 0.7007 (seeds 777/888/999) | **0.7216** |

**Shuffle-draw sensitivity:** extremely stable — CV AUROC ranges only
0.6979–0.7007 across three independent random permutation draws (test:
0.6462–0.6506). The control itself is not an artifact of one lucky/unlucky
shuffle.

**Direct comparisons, delivery (primary seed 777, CV):**

| Comparison | Δ | p (boot) | 95% CI |
|---|---|---|---|
| Shuffled-time vs. Model 2 | −0.0003 | .876 | [−0.0035, +0.0032] — **statistically indistinguishable** |
| Shuffled-time vs. Real-time Model 3 | −0.0213 | **.217** | [−0.0554, +0.0128] — CI crosses zero |

**Mechanical verdict (as computed, not softened):** *"CAPACITY ARTIFACT:
shuffled-time Model 3 performs comparably to real-time Model 3 at delivery
(not significantly different, CI includes zero). The extra input
dimension appears to have bought noise-driven flexibility rather than
genuine temporal-order exploitation. This meaningfully undercuts Model 3's
current interpretation, though not necessarily its raw predictive value
against the P6 baseline."*

**The fuller picture — reported because the mechanical label alone
understates its nuance, not to argue it away:**

1. Shuffled-time Model 3 collapses almost exactly onto **Model 2**
   (0.7003 vs. 0.7006, Δ=−0.0003, p=.876) — not onto real-time Model 3.
   Real-time Model 3 sits a consistent ~0.021 AUROC *above both*, in the
   same direction across all three independent shuffle draws (implied gaps
   ≈+0.021 to +0.024). A model that had learned nothing from `elapsed_t`
   would be expected to land exactly here — indistinguishable from
   Model 2 — which is what happened; whether real Model 3's extra ~0.02
   reflects genuine signal or sampling noise is exactly what the p=.217
   result leaves unresolved, not something the collapse-to-Model-2 pattern
   by itself settles.
2. At the exploratory **≥10m horizon**, real-time Model 3 significantly
   outperforms shuffled-time Model 3 (Δ=−0.0338, p=**.009**), and
   shuffled-time Model 3 significantly *underperforms the P6 baseline
   itself* (Δ=−0.0513 to −0.0546, p=.012–.018, consistent across all three
   seeds). A purely-ignored noise dimension would not be expected to drag
   performance below baseline — this is evidence that scrambled temporal
   position actively misleads the model at that horizon, not merely fails
   to help.
3. At ≥20m/≥30m, Model 2, shuffled-time, and real-time Model 3 are all
   within ~0.002 AUROC of each other (CV) — no signal in either direction,
   consistent with Model 3's already-documented weakness at those horizons
   (§1.2).

**Honest summary: this control does not confirm genuine temporal-order use
at the pre-registered significance bar, but it is also not a clean
refutation.** The primary (delivery) result is directionally consistent
with real information being used — real Model 3 beats both the zero-info
and scrambled-info endpoints, in the same direction across three
independent random draws — but that gap is not statistically significant
at this cohort's size, the same power limitation documented everywhere
else in this project (§1.4). The one place the control *is* significant
(≥10m) argues against a pure noise/capacity-artifact story. Read together
with §1.2a (no significant edge over Max/P90) and §1.2b (inherits the
duration confound), the honest conclusion is: **Model 3's originally
registered result against P6 (§1.1) stands, unchanged and still
CV-significant, but no test run in this hardening pass — direct, duration,
or shuffled-time — has shown Model 3 to be doing something demonstrably
more than the cheaper alternatives.** External validation, not further
internal re-testing, is the appropriate next step (§1.6).

### 1.3 Operational behavior (delivery, 80% target sensitivity)

Threshold 0.1943, median lead time 27.5 min, 54.6% detected ≥20 min, 45.5%
detected ≥30 min, false alert rate 0.714. **Not directly comparable to
Phase 15's operational numbers** — see the warning in §8.

### 1.4 Verification (`results/phase16/model3_verification.json`)

Run because training hit its 200-epoch cap in 3 of 6 fits:

- *Epoch-budget extension (400 vs 200):* AUROC 0.7216 → 0.7216, Δ +0.0344 →
  +0.0345, p .014 → .014. Identical to 4 decimals; the capped folds needed
  only 5–23 more epochs. **Concern resolved — not undertrained.**
- *Fold-resplit sensitivity:* Δ +0.0344 (canonical), +0.0315 (seed 11,
  p=.090), +0.0273 (seed 22, p=.183). Magnitude stable in a narrow band,
  always positive, no sign flip. Significance is fold-dependent — a power
  limitation at this cohort size, not instability.

### 1.5 Interpretability

Attended window matches the pure-magnitude argmax in only **31.6%** of
patients (Spearman ρ=0.379) — meaningfully different from Model 2's 100%,
suggestively indicating it uses temporal position beyond peak-picking. See
§1.2c: the shuffled-time control neither confirms this claim at the
pre-registered significance bar nor cleanly refutes it (shuffled-time
Model 3 collapses to Model 2's performance, and real-time Model 3's ~0.02
AUROC edge over both is directionally consistent across three independent
shuffle draws but not statistically significant, p=.217) — weigh this
31.6% figure as suggestive only, per §1.2c's full discussion.

### 1.6 Status

The only result across all of Phases 13–16 to clear CV p<0.05 with a
fully-positive CI at a primary horizon, and the only one stress-tested
three times afterward (§1.4 epoch/resplit checks, §1.2c shuffled-time
control). That result — Model 3 vs. **P6** — stands unchanged by every
hardening pass; nothing here reruns or contradicts it. What Tier 2 and
Tier 3 add are three separate reasons for caution that sit alongside it,
not instead of it:

1. Model 3 is not shown to beat either cheap fixed aggregator directly
   (§1.2a — CIs cross zero at every horizon vs. both Max and P90).
2. It inherits the same recording-duration confound Phase 15 found in
   those aggregators, as strongly as the weakest of them (§1.2b).
3. The shuffled-time control's pre-registered decision rule scores its
   primary result **"capacity artifact"** — shuffled-time Model 3 collapses
   to Model 2's performance rather than to real-time Model 3's, and the
   ~0.02 AUROC gap between real-time Model 3 and both zero/scrambled-time
   endpoints, while directionally consistent across three independent
   shuffle draws, does not reach significance (p=.217) (§1.2c). The control
   is not a clean refutation either — it is significant in Model 3's favor
   at ≥10m (p=.009), and scrambled time actively *hurts* below baseline
   there rather than being harmlessly ignored.

Read together, the current honest summary is: **Model 3 reliably beats the
single-window P6 baseline (its one unambiguous, twice-and-now-thrice-
verified result), but no test run across Tiers 2–3 — direct comparison,
duration-confound, or shuffled-time — has shown it to be doing something
demonstrably beyond what a training-free percentile calculation over the
same sequence already provides, at a significance level this cohort's size
can confirm.** This is not proof Model 3 is *only* capacity noise — the
consistent directionality and the significant ≥10m result argue against
that reading — but it is proof that "Model 3 uses genuine temporal
position" cannot currently be asserted at the same confidence level as
"Model 3 beats P6." **Still recommended for external validation — not for
clinical use — but no longer described as clearly ahead of the fixed
aggregators; the honest framing is "worth testing alongside P90/Max, with
an open question about mechanism that only a larger cohort can settle,"
not "better than them, and known to work via temporal position."** All
three hardening tiers are now complete; this is the final status pending
external data.

---

## 2. Model 4 — peak-aware fusion

**Mechanism.** `LogisticRegression(C=0.1)` on four scalars per patient:
`[max(prefix), P90(prefix), mean(prefix), Model 3's own pooled output]`.

**Leakage-control walkthrough (item 8 — verified directly against
`scripts/phase16_model4_peak_aware_fusion.py`, not assumed):** for outer
fold *f*, `fold_scorers[f]` — Model 3's checkpoint for that specific fold,
itself trained excluding fold *f*'s held-out patients — is used to compute
the 4-feature vector for **both** that fold's training patients and its
held-out patients. The `StandardScaler` and `LogisticRegression` for
Model 4 are then fit on the training patients' features only and applied
to the held-out patients' features. No step in this chain touches fold
*f*'s held-out patients during any fitting — Model 3's own out-of-fold
property (established in Phase 16) carries through unbroken, and Model 4's
own fit respects the same fold boundary independently. Verified by reading
the fold loop directly before this document was written.

### 2.1 Delivery discrimination

**Source:** `results/phase16/Model_4_peak_aware_fusion_results.csv`

| | CV AUROC | CV AUPRC | Δ vs P6 (p, CI) | Δ vs **Model 3** (p) | Internal-test AUROC | Test Δ vs P6 (p) | Test Δ vs Model 3 (p) |
|---|---|---|---|---|---|---|---|
| **Delivery** | **0.7244** | 0.4311 | +0.0372 (.053, [−0.0002, +0.0769]) | **+0.0028 (.760)** | 0.6774 | +0.0276 (.201) | +0.0045 (.783) |

**AUPRC relative to prevalence:** 0.4311 / 0.201 = **2.14× lift** — within
noise of Model 3's 2.16×, consistent with the "no improvement beyond Model
3" verdict below.

### 2.2 Early-warning performance (exploratory)

| Horizon | CV AUROC | CV AUPRC | Δ vs P6 (p, CI) | Δ vs Model 3 (p) | Internal-test AUROC | Test Δ vs P6 (p) | Test Δ vs Model 3 (p) |
|---|---|---|---|---|---|---|---|
| ≥10m | 0.6606 | 0.3552 | −0.0253 (.081) | −0.0077 (.212) | 0.5749 | −0.0053 (.853) | +0.0107 (.252) |
| ≥20m | 0.5929 | 0.3058 | −0.0309 (**.050**, [−0.0633, −0.0001]) | −0.0086 (.252) | 0.6381 | +0.0258 (.406) | +0.0178 (.383) |
| ≥30m | 0.6002 | 0.2978 | +0.0173 (.414) | +0.0025 (.859) | 0.7059 | +0.0980 (.123) | +0.0401 (.092) |

### 2.3 Operational behavior

Not computed for Model 4 in Phase 16 (only Model 2/3 and Phase 15's fixed
aggregators have operational lead-time/FAR figures). A gap, not a zero —
flagged rather than filled with an estimate.

### 2.4 Status

Highest raw CV AUROC of any candidate (0.7244) — but **not significantly
better than Model 3 at any horizon** (p = .212–.859), and its own
significance against the baseline is *weaker* than Model 3's (p=.053 vs
.014). Adding hand-crafted max/P90/mean on top of Model 3's learned
attention is redundant: Model 3 already captures that information. At
≥20m it is significantly *worse* than the baseline (CI entirely negative)
— inherited from Model 3's own weakness at that horizon, not introduced by
the fusion. **Not recommended over Model 3.**

---

## 3. P90 pooling (fixed statistic)

**Mechanism.** No training at all. Patient score = 90th percentile of P6's
window-level risk scores over the causally-eligible prefix, replacing
single-window selection.

**Canonical estimator policy (item 2).** Two legitimate percentile
definitions were used across this investigation: Phase 13E's Hazen-style
weighted-quantile (`src/aggregation/recency_weighted_p90.py`) and Phase
14/15's `numpy.percentile` (linear interpolation). **`numpy.percentile` is
adopted as canonical from this document forward** — it is the more-used,
more-recently-endorsed choice (Phase 14 selected it explicitly "for
continuity," Phase 15 carried it forward as its own stated basis). The
Hazen-based figure is retained below as a labeled historical/sensitivity
comparison, never as a competing headline.

### 3.1 Delivery discrimination

**Source:** `results/phase14/phase14_results.csv` (canonical),
`results/phase13/p90_pooling/p90_pooling_results.csv` (historical)

| | CV AUROC | Δ vs P6 | CV p | CV 95% CI | Internal-test AUROC | Test Δ | Test p |
|---|---|---|---|---|---|---|---|
| **Delivery (canonical, `numpy.percentile`)** | **0.7178** | +0.0307 | .225 | [−0.0189, +0.0822] | 0.6622 | +0.0125 | .797 |
| Delivery (historical, Hazen weighted-quantile) | 0.7254 | +0.0382 | .121 | [−0.0092, +0.0875] | 0.6916 | +0.0419 | .352 |

**Estimator sensitivity is itself a finding**, retained deliberately: swapping
between two equally defensible percentile definitions moved the delivery Δ
from +0.0382 to +0.0307 and p from .121 to .225. This is direct evidence
the point estimate is not highly stable — reported, not reconciled away.

### 3.2 Early-warning performance (exploratory, canonical estimator)

| Horizon | CV AUROC | Δ vs P6 | CV p | Internal-test AUROC | Test Δ | Test p |
|---|---|---|---|---|---|---|
| ≥30m | 0.6090 | +0.0261 | .216 | 0.6979 | +0.0900 | .189 |

**Related fixed aggregators** (Phase 15, `numpy.percentile` basis, delivery
CV, canonical estimator throughout): Max 0.7210 (Δ+0.0338, p=.140, AUPRC
lift 0.4227 not separately isolated — see Phase 15 raw files) · Mean 0.6926
(Δ+0.0054, p=.865) · Median 0.6536 (Δ−0.0336, p=.283). Ordering **Max ≥
P90 > Mean > Median** held at both horizons and both splits.

### 3.3 Operational behavior (Phase 15, delivery, 80% target sensitivity)

| Aggregator | Median lead (min) | ≥20min detected | ≥30min detected | FAR |
|---|---|---|---|---|
| Single-window | 32.5 | 64.5% | 53.6% | 56.8% |
| P90 | 17.5 | 49.1% | 39.1% | 52.9% |
| Max | 12.5 | 38.2% | 25.5% | 51.5% |

Higher discrimination (Max) coincides with *worse* lead time and detection
here — see §8's standing warning against reading AUROC alone as "the"
operational answer.

### 3.4 Status

Consistently positive direction across every test, never significant, and
failed to replicate at Phase 14's stricter pre-registered bar. **Promising,
not confirmed.**

---

## 4. Parity fusion (independent covariate)

**Mechanism.** `logit(p_fused) = logit(p_P6) + λ · logit(p_parity)`, where
`p_parity` comes from a univariate `LogisticRegression` on admission-time
maternal parity. λ selected per fold by training-fold AUROC maximization
(Phase 4's proven mechanism); per-fold λ = 1.3–1.5.

### 4.1 Delivery discrimination

**Source:** `results/parity_fusion/parity_fusion_results.csv`

| | CV AUROC (P6→fused) | Δ | CV p (boot / DeLong) | CV 95% CI | Internal-test AUROC (P6→fused) | Δ | Test p (boot / DeLong) | Test 95% CI |
|---|---|---|---|---|---|---|---|---|
| **Delivery** | 0.6872 → 0.7094 | +0.0223 | .172 / .159 | [−0.0089, +0.0516] | 0.6497 → **0.7148** | **+0.0651** | **.025 / .020** | **[+0.0103, +0.1242]** |

### 4.2 Early-warning performance (exploratory)

**Horizon-convention correction applied (item 1, Tier 2 — resolved).**
`scripts/parity_fusion_test.py`'s original horizon selector used the
first-eligible-chronological-window convention, not the
`get_patient_scores_at_horizon_corrected` convention every other
≥30m/≥20m/≥10m figure in this document uses. Delivery (§4.1) was always
unaffected (both conventions agree at h=0).
`scripts/parity_fusion_horizon_correction.py` reused the exact fitted
objects (frozen P6 scores, per-fold parity models, per-fold λ, all
unchanged) and swapped only the horizon-selection call — no refitting of
anything. **≥10m and ≥20m are newly computed here for the first time**
(the original script only ever covered delivery and ≥30m); ≥30m now has
both the original and corrected figures, kept side by side and explicitly
labeled, per this project's standing rule against silent reconciliation.

**Source:** `results/parity_fusion/parity_fusion_horizon_corrected_results.csv`

| Horizon | CV AUROC (P6→fused) | Δ | CV p (boot) | Internal-test AUROC (P6→fused) | Δ | Test p (boot) |
|---|---|---|---|---|---|---|
| ≥10m *(new)* | 0.6859 → 0.7008 | +0.0149 | .356 | 0.5802 → 0.6266 | +0.0463 | **.006** |
| ≥20m *(new)* | 0.6238 → 0.6435 | +0.0197 | .472 | 0.6123 → 0.7389 | **+0.1266** | **<.001** |
| ≥30m *(corrected)* | 0.5828 → 0.6368 | +0.0539 | .057 | 0.6078 → 0.7264 | **+0.1185** | **.019** |
| ≥30m *(original, uncorrected — retained, not silently replaced)* | 0.5857 → 0.6351 | +0.0494 | .078 | 0.6845 → 0.7629 | +0.0784 | .166 |

**Finding, stated plainly: the correction makes parity fusion's
early-warning case stronger, not weaker.** Every corrected test-partition
delta is larger than its original-convention counterpart and now clears
p<.05 at *all three* early-warning horizons on the held-out internal test
partition (≥10m p=.006, ≥20m p<.001, ≥30m p=.019) — versus only delivery
being significant before this correction. CV deltas remain positive but
not individually significant at any single horizon (consistent with
delivery's own CV p=.172, §4.1) — the same fold-count power limitation
seen throughout this project, not a new concern. This is not evidence of
motivated reasoning toward a better number: the correction was specified
and verified against source code (plan item 1) *before* being run, exactly
mirroring Phase 13's own pre-registration discipline, and the deliverable
was "report the corrected number, original kept alongside" regardless of
which direction it moved.

### 4.3 Operational behavior

Not computed for the parity-fusion candidate in any committed script. A
gap, not a zero.

### 4.4 Robustness suite — the most thoroughly checked candidate in the program

- *Bootstrap-seed stability:* Δ 0.0220–0.0223 across 5 seeds (p .156–.172) — the marginal p is not a seed artifact.
- *Fold-assignment stability:* Δ +0.0117, +0.0135, +0.0204, +0.0223 across 4 independent re-splits — **all positive**, no sign flip.
- *Recording-duration confound:* r(parity, windows-per-patient) = −0.049 (p=.25) — **ruled out**.
- *Conditional analysis:* removing parity from a joint 9-covariate model drops AUROC by **0.089** — parity is not substitutable by any other admission variable.
- *Permutation control:* 0/30 shuffled-parity permutations reached the true Δ (mean shuffled Δ = −0.047).

**Why parity and not other covariates:** a leakage-controlled screen of 9
admission-time variables found diabetes (AUROC .486), hypertension (.496),
preeclampsia (.498), age (.477), gestational weeks (.523), gravidity
(.465), induction (.476), sex (.498) and presentation (χ² p=.51) all
essentially null. Only parity reached significance (AUROC .411 — i.e.
*lower* parity predicts higher risk — p=.0004, surviving Bonferroni across
9 tests). The joint 9-covariate model scored just 0.5132.

### 4.5 Status

The only candidate significant on the held-out internal test partition —
and, after the horizon-convention correction (§4.2), significant on that
partition at **every** horizon tested (delivery, ≥10m, ≥20m, ≥30m), not
delivery alone. The only candidate with a complete independent robustness
suite. CV CIs still cross zero at every horizon — the internal test
partition's strength has not yet been matched by CV significance, the same
power-limitation pattern seen throughout this project, not a new caveat.
**Promising, not confirmed. Recommended for external validation alongside
Model 3 — not for clinical use** — it tests an orthogonal hypothesis
(information CTG cannot see), and unlike Model 3 (§1.6), nothing in Tier-2
hardening weakened its case — if anything, the horizon correction
strengthened it.

---

## 5. M3 + Parity deployable fusion (Phase 18) ⭐ strongest early-warning candidate

**Mechanism.** Probability-space averaging of Model 3's own running score
and the frozen parity model's probability:
`p_fused = α · p_M3(t) + (1 − α) · p_parity`, with a **single α selected
once per fold** (grid search maximizing sample-weighted training AUROC,
pooled across every causal truncation of every training patient's
sequence — never re-selected per horizon). **Frozen deployed value:
α = 0.35** (fit on all 547 patients, no held-out fold), stable across 30
independent fold-fits — 5 canonical folds plus 5 independent resplits × 5
folds each — at mean 0.333, median 0.35, range [0.30, 0.40]. Parity is
weighted roughly **65%**, M3 roughly 35%. Full spec, verification history,
and frozen parameters: `docs/phase18_deployable_fusion_protocol.md`.
`p_M3(t)` is Model 3's own running
score at whatever causal truncation is available at query time
(`predict_all_prefixes`, the same function used for Model 3's own
operational lead-time simulation) — **not** retrained or modified from
Model 3's own locked checkpoints.

**Why this candidate exists, and what it corrects.** Every fusion model
built before this one in the project's post-lock investigation — the
original Parity Fusion λ, the Model 3 + Parity Hybrid
(`results/model3_parity_hybrid/`), and Phase 18's own first ablation pass
(`docs/phase18_m3_parity_fusion_ablation_protocol.md`, 16 models) — fit
**separate fusion coefficients per retrospective horizon**, using that
horizon's own truncated score as a training feature. That is not
deployable: during actual labor, minutes-before-delivery is unknown, so a
real-time system cannot select which coefficients to use based on
information that does not yet exist. This candidate is the corrected
version: **one fixed model, trained once per fold on every causal
truncation pooled together** (sample-weighted 1/T_i per patient, mirroring
`docs/phase16_protocol.md` §4's own convention for training Model 3
itself), then evaluated at each retrospective horizon via the identical
prefix-truncation logic used everywhere else. It is queried the same way
regardless of how far labor has progressed; only its *retrospective*
accuracy varies by lead time, exactly like Model 3's own does.

### 5.1 Delivery discrimination (primary endpoint)

**Source:** `results/phase18_fusion_ablation/deployable_fusion_results.csv`
(canonical split), `deployable_fusion_stability_ranking.csv` (resplit
check). Comparator is **M3-only** (already established as significantly
better than P6, §1.1) — a direct paired test against P6 was not separately
computed for this candidate; treat the P6 comparison as inferred through
the M3 chain, not as its own formal test.

| | CV AUROC | Δ vs M3-only | CV p (canonical) | 6-split fold-resplit check | Internal-test AUROC | Test Δ vs M3-only | Test p |
|---|---|---|---|---|---|---|---|
| **Delivery** | 0.7335 | +0.0119 | .468 | **0/6 splits significant** (mean Δ+0.0103, worst p=.695) | 0.7433 | +0.0704 | **.004** |

**Not yet a confirmed delivery-horizon result.** The point estimate is
positive on both CV and test (and test reaches significance), but the CV
effect does not clear the resplit bar — report this candidate's delivery
performance as directionally positive, not established.

### 5.2 Early-warning performance — the result this candidate exists for

| Horizon | CV AUROC | Δ vs M3-only | CV p (canonical) | 6-split fold-resplit check | Internal-test AUROC | Test Δ vs M3-only | Test p |
|---|---|---|---|---|---|---|---|
| ≥10m | 0.6921 | +0.0238 | .231 | 0/6 significant (mean Δ+0.0250, worst p=.254) | 0.6988 | +0.1346 | **<.001** |
| **≥20m** | **0.6638** | **+0.0623** | **.001** | **6/6 splits significant** (mean Δ+0.0616, worst-case p=**.002**) | 0.7558 | +0.1355 | **<.001** |
| **≥30m** | **0.6631** | **+0.0654** | **<.001** | **6/6 splits significant** (mean Δ+0.0641, worst-case p=**.003**) | 0.7754 | +0.1096 | **<.001** |

**This is the most completely verified result in the entire post-lock
investigation.** At ≥20m and ≥30m, every one of 6 independent patient
partitions (the canonical split plus 5 fully independent resplits, model
refit fresh under each) agrees: this fusion mechanism significantly beats
Model 3 alone, with large margins (+0.06 AUROC) and comfortable worst-case
significance (p=.002–.003, i.e. even the *least* favorable of 6 splits
clears the bar). Two competing mechanisms (F2-style logistic fusion,
A4 logit-space sum-to-one fusion, both refit under the same deployable —
pooled, not per-horizon — discipline) also clear 6/6 at these horizons but
with smaller margins (+0.045 to +0.060); this candidate is the strongest
of everything tested. Full comparison:
`results/phase18_fusion_ablation/deployable_fusion_stability_ranking.csv`.

### 5.3 Operational behavior (80% target sensitivity, per-fold training-only threshold, patient-level "ever alerted")

| System | Achieved Sens | FAR | Median Lead | ≥20m Detected | ≥30m Detected |
|---|---|---|---|---|---|
| M3-only | 88.2% | 69.3% | 32.5 min | 66.0% | 54.6% |
| **This candidate (A3, α=0.35)** | 86.4% | **62.5%** | 40.0 min | **81.0%** | **69.5%** |

At comparable achieved sensitivity to M3-only, this candidate lowers FAR by
~7 points and extends median lead time by 7.5 minutes, with substantially
higher ≥20m/≥30m detection. Full table, including two close operational
competitors (A4 sum-to-one, F2-deployable) not adopted as the primary
recommendation: `results/phase18_fusion_ablation/deployable_fusion_operational_metrics.csv`.

**Caveat, not yet resolved:** the shared operating threshold behaves very
differently across parity groups (nulliparous patients get pushed toward
*more* alerting, higher-parity patients toward *much less*, even though
within-group AUROC is roughly unchanged) — see
`docs/phase18_deployable_fusion_protocol.md` §7. Treat this as an open
calibration/fairness question for external validation, not resolved by
the pooled numbers above.

### 5.3a The central hypothesis, tested directly

*"Does parity act primarily as a maternal-context modifier for ambiguous
CTG evidence, or does it help uniformly?"* Tested via M3-risk-tertile
stratification (per-fold training-derived cutoffs) and a continuous
ambiguity correlation. **The answer is horizon-dependent, not uniform
either way:** at delivery, the fusion gain concentrates sharply in the
intermediate M3-risk tertile (AUROC Δ+0.10) and is essentially zero when
M3 is already confident (Δ−0.01) — a clean ambiguity-resolving pattern,
confirmed by a significant continuous correlation (Spearman r=+0.30,
p<.0001) between shift magnitude and closeness of M3's score to 0.5. At
≥20m, the gain is present broadly across all three risk strata (not
concentrated in the middle) and the ambiguity correlation is not
significant (r=−0.07, p=.10) — parity reads more as a general
complementary signal than a targeted disambiguator at this horizon. Full
writeup: `docs/phase18_deployable_fusion_protocol.md` §6.

### 5.4 Verification history (why this is trusted)

This result survived more direct scrutiny than any other candidate in the
project before being reported:

1. **The horizon-specific version of this same idea (Stage 1, 16 models)
   initially looked strong at 4 separate (model, horizon) pairs.** A
   bootstrap-seed check alone would have certified all 4. A fold-resplit
   check — recommended by the source ablation plan, easy to skip — found
   **3 of the 4 were canonical-partition artifacts**, not real effects.
2. **A full 16-model × 4-horizon × 6-split sweep**, built to check
   systematically rather than chase individual flagged pairs, caught a
   real leakage bug in its own first draft (a parity model reused across
   resplits instead of refit fresh) before any conclusion was drawn from
   it — found by cross-checking against the earlier, more careful
   single-pair result and noticing a direct contradiction.
3. **The horizon-specific framing was then identified as fundamentally
   non-deployable** (minutes-before-delivery is unknown in real time) —
   this is the correction that produced the mechanism described here.
4. **The deployable version's own fold-resplit stability check** caught
   the *same* leakage bug pattern a second time, independently, in a newly
   written script, before trusting its output.

Every one of those checks is documented in
`results/phase18_fusion_ablation/stage1_report.md`. The ≥20m/≥30m result
above is what remains after all four rounds of adversarial verification —
not the first number that looked good.

### 5.5 Status

**Recommended for external validation, alongside Model 3 and Parity
Fusion — not for clinical use.** Unlike every other candidate in this
document, this one has already survived a dedicated fold-resplit
robustness check at the horizons where it claims significance, not merely
a bootstrap-seed check, and has a frozen, fully-documented mechanism
(`docs/phase18_deployable_fusion_protocol.md`) with operational metrics
(§5.3) and a directly-tested explanatory hypothesis (§5.3a) already in
hand. Its delivery-horizon performance is not yet established and should
not be claimed. Before external-validation submission: resolve the
parity-group threshold-calibration question (§5.3's caveat), and produce
frozen, portable artifacts analogous to `models/external_validation_handoff/`
for this specific fusion formula (currently only Model 3, P6, and Parity
Fusion have portable frozen artifacts there — trivial to add given the
frozen α is already known).

---

## 6. Context — Model 2 (why it matters even though it's not a candidate)

Same architecture as Model 3 but with `e_t = f(r_t)` only — no elapsed-time
input at all (contrast with §1's exact time-representation definition).
Delivery CV 0.7006 (Δ+0.0135, p=.601); internal-test 0.6506 (Δ+0.0009,
p=.920). **Its value is mechanistic, not predictive:** its learned
attention picks the single highest-risk window in **100%** of patients
(Spearman ρ≈1.0) — a trained model, given only magnitude, independently
rediscovered exactly the max-pooling strategy Phase 15 identified by fixed
statistics. That convergence is why Model 3's *departure* from pure
magnitude (31.6% agreement, §1.5) was interpreted as genuinely using
something more — a reading now in tension with §1.2a's direct finding that
Model 3 doesn't actually outperform Max or P90 in practice, and with
§1.2b's finding that Model 3 shares Max/P90's own duration-dependence. The
shuffled-time control (§1.2c, Tier 3 item 5) tested this directly: shuffled-
time Model 3 collapses almost exactly onto Model 2's performance, and
real-time Model 3's small edge over it, while consistent across three
independent shuffle draws, does not reach significance (p=.217) — the
question of whether Model 3 is doing something Model 2/Max/P90 cannot
remains open, tilted slightly toward "something real" by the consistent
direction and by a significant effect at ≥10m, but not settled at this
cohort's size.

Operational (delivery, 80% sens): median lead 40.0 min, 75.5% ≥20 min,
67.3% ≥30 min, FAR 0.778. AUPRC not separately cached for Model 2 in the
committed result files.

---

## 7. Side-by-side (delivery horizon, the primary endpoint)

| Model | CV AUROC | CV Δ | CV p | CV CI crosses 0? | Internal-test AUROC | Test Δ | Test p | Verified? |
|---|---|---|---|---|---|---|---|---|
| P6 (baseline) | 0.6872 | — | — | — | 0.6497 | — | — | Locked |
| Model 2 | 0.7006 | +0.0135 | .601 | yes | 0.6506 | +0.0009 | .920 | — |
| **Model 3** | 0.7216 | +0.0344 | **.014** | **no** | 0.6729 | +0.0232 | .157 | **✔ 3 passes*** |
| Model 4 | **0.7244** | +0.0372 | .053 | yes (barely) | 0.6774 | +0.0276 | .201 | — |
| P90 (canonical `numpy.percentile`) | 0.7178 | +0.0307 | .225 | yes | 0.6622 | +0.0125 | .797 | Failed Ph.14 re-test |
| Parity | 0.7094 | +0.0223 | .172 | yes | **0.7148** | **+0.0651** | **.025** | **✔ 5 checks** |
| M3+Parity deployable fusion (Phase 18)† | 0.7335 | +0.0463‡ | not tested‡ | not tested‡ | 0.7433 | +0.0936‡ | not tested‡ | **✔✔ 6-split resplit-verified — but only at ≥20m/≥30m, see †** |

† **This row is misleading if read at face value — its real strength is not at delivery.** This candidate (§5) was built and verified for early-warning performance: at ≥20m/≥30m it beats Model 3 alone by +0.06 AUROC, confirmed significant in all 6 independent splits (worst-case p=.002–.003) — the single most robustly verified result in this document. At delivery (the row shown above), its CV effect does **not** clear the fold-resplit bar (0/6 splits significant) — report it as directionally positive, not established, at this horizon. ‡ The Δ/p/CI columns above are computed **against Model 3, not P6** (the formal test this candidate was actually built and verified against, §5.1); the P6 comparison shown (+0.0463, implied from 0.7335 − 0.6872) was never itself run as a paired significance test and should not be read as one.

## 8. How to read this table

- **\*Model 3's "3 passes":** epoch-budget extension and fold-resplit
  sensitivity (§1.4, both clean) plus the shuffled-time control (§1.2c,
  mixed — scores "capacity artifact" by the pre-registered rule at
  delivery, but is significant in Model 3's favor at ≥10m and directionally
  consistent at delivery across 3 independent shuffle draws). "3 passes"
  means three checks were run and reported in full, not that all three
  came back clean — see §1.6 for the complete, unresolved picture.
- **Tier-2 hardening pass (2026-09-13) added two direct tests that change
  how Model 3 should be read, though not its own result against P6.**
  (a) Model 3 is not statistically distinguishable from Max or P90 at any
  horizon, on either split (§1.2a) — the vs-P6 columns above for Model 3
  and P90 look similar because the two candidates *are* statistically
  similar to each other, not coincidentally close. (b) Model 3 inherits
  Phase 15's recording-duration confound as strongly as the weakest fixed
  aggregators (§1.2b).
- **Tier-3 hardening pass (2026-09-13) ran the shuffled-time control
  (§1.2c), the check most likely to alter Model 3's interpretation.** It
  did not confirm genuine temporal-order use at the pre-registered
  significance bar (shuffled-time Model 3 collapses to Model 2's
  performance; real-time Model 3's edge over both is consistent in
  direction across 3 independent shuffle draws but not significant,
  p=.217) — nor did it cleanly refute it (significant at ≥10m, p=.009, and
  scrambled time actively underperforms baseline there rather than being
  ignored). None of Tier 2 or Tier 3's findings reverse Model 3's CV
  significance against the P6 baseline; together they mean it should be
  described as "worth testing alongside P90/Max, mechanism unresolved,"
  not "ahead of them and known to use temporal position."
- **Highest raw number ≠ best supported.** Model 4 tops the CV column but
  is not statistically distinguishable from Model 3 at any horizon; the
  P90 figure shown here is the canonical (`numpy.percentile`) estimator,
  which is *weaker* than the historical Hazen-based figure some earlier
  documents cited (§3.1) — a reminder that estimator choice alone can move
  a headline number by more than most of the deltas being compared.
- **Model 3 is the only CV-significant result with a CI clear of zero**,
  and the only one re-tested after looking promising rather than reported
  on first sight.
- **Parity is the only test-significant result**, and the most robustness-
  checked — but on an 83-patient held-out internal test partition
  (~17 positives), where a handful of patients can move AUROC several
  points. Its ≥10m/≥20m/≥30m figures now use the corrected horizon
  convention (§4.2, Tier-2 item 1 — resolved), matching delivery's own
  footing; the correction moved every early-warning test-partition delta
  up, not down.
- **The Phase 18 deployable fusion candidate (§5) is the exception to
  "read the delivery table first."** Every other row in this table is
  primarily characterized by its delivery-horizon result. This one should
  be read from §5.2 (early-warning), not the row above — its delivery
  performance is weak/unestablished, and reporting only the delivery row
  (as the table format above forces) would understate what is actually
  this project's strongest verified result.
- **⚠ Operational metrics are not comparable across phases.** Phase 15
  counts a false alert at a single evaluation point; Phase 16 counts "ever
  alerted" across the whole rolling sequence — the latter is structurally
  higher. Compare Model 2 vs Model 3 to each other, and Phase 15's
  aggregators to each other, never across phases.
- **Nothing here replaces P6, and nothing here is clinically ready.** Every
  Δ is measured on the same 547 patients that every phase since Phase 1
  has used. External validation is the decisive next step — see
  `docs/external_validation_handoff.md`. A positive external result would
  open a promotion conversation; it would not itself constitute one.
