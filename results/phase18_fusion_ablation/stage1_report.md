# Phase 18 Stage 1 — M3 + Parity Fusion Ablation: Results

**Date:** 2026-09-15, revised 2026-09-15 after stability checks. Protocol:
`docs/phase18_m3_parity_fusion_ablation_protocol.md`.
16 models × 4 horizons × (5-fold CV + held-out internal test partition), all
consuming the identical frozen `(m3_score, parity_score)` pair per patient
per horizon. Full data: `stage1_main_results.csv`,
`stage1_coefficient_stability.csv`, `stage1_patient_level_predictions.csv`,
`f2_regularization_sensitivity.csv`, `f4_elasticnet_sensitivity.csv`,
`stability_bootstrap_seed_sensitivity.csv`, `stability_fold_resplit_sensitivity.csv`.

> **Revision notice.** The original version of this report (below) flagged
> four (model, horizon) pairs as significantly beating the current hybrid
> (F2). The bootstrap-seed and fold-resplit stability checks explicitly
> deferred at the time have now been run (see **§ Stability checks**,
> added below). **Three of those four findings do not survive an
> independent fold resplit** — they were specific to the one canonical
> fold partition, not general. Only one — A3 with a train-selected α at
> ≥20m — replicates robustly across both 5 bootstrap seeds and 4 of 5
> independent resplits. Read the headline finding below with that
> correction in mind; the stability-checks section is now the more
> reliable summary of what actually holds up.

## Headline finding

**The currently-deployed fusion mechanism (F2, standardized-logit L2
logistic regression, C=0.1) is not the best-performing fusion mechanism
among those tested.** Two simpler alternatives beat it, one of them by a
wide and statistically significant margin at multiple horizons:

- **Probability-space averaging (A3)** — literally `p = α·p_M3 + (1−α)·p_parity`,
  no logistic fitting at all beyond a 1-D grid search for α — outperforms F2
  at every one of the four horizons, and does so **significantly** at ≥10m
  (fixed α=0.5: Δ+0.0066, p=**.048**) and ≥20m/≥30m (train-selected α:
  Δ+0.0195, p=**<.001** at ≥20m; fixed α=0.25: Δ+0.0188, p=**<.001** at
  ≥30m). At delivery it is directionally ahead but not significant
  (Δ+0.0065, p=.174). **Of these three, only the ≥20m train-selected-α
  result survives the fold-resplit check below — see the revision notice
  above.**
- **Interaction logistic fusion (F7)** — the same additive form as F2 plus
  one `z_M3·z_parity` term — significantly beats F2 at delivery specifically
  (Δ+0.0034, p=**.036**), with a consistently negative interaction
  coefficient across all 5 folds (−0.288 to −0.417, never flips sign) —
  see "does parity modify M3" below. **This delivery result is stable
  across bootstrap seeds (p stays in [.032,.049] across all 5) but does
  not reach significance under any of 5 independent fold resplits
  (p ranges [.052,.234]) — see the stability-checks section.** The
  negative, sign-stable interaction coefficient itself is a separate,
  more durable finding, unaffected by whether the AUROC gain over F2 is
  significant on any particular fold partition.

Neither of these findings should be read as "swap in A3 and ship it" without
the caveats below (calibration cost, test-partition volatility at the small
N this cohort has) — but they directly answer the plan's second framing
question ("which fusion mechanism combines the two sources most
effectively") with an answer that isn't the one currently in production.

## Question 1 — Does parity add information beyond M3?

**Yes, robustly — every fusion mechanism tested beats M3-only (B2) at every
horizon**, and the margin is CV-significant (p<.05 vs. B2) for most of them
at ≥20m/≥30m and for several at ≥10m. M3-only itself: 0.7216 / 0.6683 /
0.6015 / 0.5976 at delivery/10/20/30m (matches the locked canonical Model 3
numbers exactly, confirming the upstream scoring is unchanged from every
prior phase). Parity-only (B1) is flat at 0.5755 across all horizons (it
carries no CTG information, so horizon has no effect on it) — well below
M3 alone, but clearly above the 0.4986 prevalence-only floor, and its
addition consistently helps regardless of *how* it's combined.

## Question 2 — Which fusion mechanism combines the two best?

Ranked by CV AUROC, aggregated impression across all four horizons (full
numbers in `stage1_main_results.csv`):

1. **A3 (probability averaging)** — best or near-best at every horizon;
   the *only* family that is simultaneously simple (no regularization
   hyperparameter to justify), significant vs. F2 at 2–3 of 4 horizons, and
   requires no learned coefficients at all in its fixed-α variants.
2. **F7 (interaction logistic)** — best at delivery specifically, with the
   most interpretable and most stable coefficient of any model tested (see
   below).
3. **A4 logit-space variants** (nonneg, unconstrained, equal, sum-to-one)
   — all close to F2, none significantly different from it, but their
   fitted weights reveal something F2 doesn't show on its own (see
   "what the weights say" below).
4. **F2 (current hybrid)** — solidly ahead of M3-only everywhere, but
   consistently the median-to-lower performer among the fusion family,
   not the best.
5. **F3 (L1)** and **P3 (categorical parity)** — both slightly *below* F2
   at every horizon. L1's own regularization curve (below) shows why.

### Regularization sensitivity (F2's own C, delivery)

| C | 0.001 | 0.01 | 0.1 (current) | 1 | 10 | 100 |
|---|---|---|---|---|---|---|
| CV AUROC | 0.7322 | 0.7328 | 0.7323 | 0.7327 | 0.7329 | 0.7329 |

**Flat across 5 orders of magnitude.** The current C=0.1 is not doing any
identifiable work — nothing in this sweep suggests it was a load-bearing
choice, nor that a different C would have changed the reported result
materially. This is a reassuring finding for F2's own robustness, even
though F2 itself isn't the top performer.

### Elastic-net sweep (delivery)

| l1_ratio | 0 (pure L2) | 0.25 | 0.5 | 0.75 | 1 (pure L1) |
|---|---|---|---|---|---|
| CV AUROC | 0.7323 | 0.7318 | 0.7305 | 0.7290 | 0.7268 |

**Monotonically decreasing as L1 weight increases** — pushing toward
eliminating one branch costs AUROC steadily, all the way to F3's pure-L1
number. With only two predictors, L1 has nothing useful to prune: **both
branches carry real, non-redundant signal**, which is itself informative
evidence for the plan's central hypothesis, independent of which fusion
algorithm is ultimately preferred.

### What the fitted weights say (coefficient stability, `stage1_coefficient_stability.csv`)

- **F2**: coef_m3 ∈ [0.66, 0.74], coef_parity ∈ [0.22, 0.29] across the 5
  folds — both positive and stable, but M3 gets roughly 2.5× parity's
  weight.
- **A4_logit_sum_to_one** (the two weights constrained to add to 1, fit by
  train-fold AUROC): selects **γ_M3 ≈ 0.55–0.60, γ_parity ≈ 0.40–0.45** in
  every fold — meaningfully more weight on parity than F2's own fit implies,
  once the two are forced onto a directly comparable scale. This is a
  genuinely different picture of how much parity should count than the
  currently-deployed model gives it.
- **A4_logit_nonneg**: unconstrained-but-clamped-at-zero fit lands at
  γ_M3 ≈ 1.18–1.32, γ_parity ≈ 0.58–0.78 (raw, unstandardized logits) — the
  non-negativity constraint never binds; both coefficients want to be
  positive on their own, with no fold showing a sign flip.
- **F7 interaction term**: **−0.288 to −0.417 in every one of the 5 folds** —
  never crosses zero. Per the plan's own interpretation guide (§F7):
  β₃<0 means *"one source may attenuate the effect of the other"* — high
  parity-risk and high M3-risk are not simply additive; their combined
  effect is somewhat sub-additive. This is consistent with (though not
  proof of) the plan's own E5 hypothesis that parity may matter most when
  CTG evidence is ambiguous rather than when M3 is already extreme in
  either direction — Stage 2/E5 stratified analysis, not run here, would
  be the direct test of that.
- **P3 categorical parity**: the parity=1-vs-0 coefficient is consistently
  negative and the largest in magnitude (−0.35 to −0.52, all 5 folds) —
  confirms the established Phase 13 finding (lower parity → higher risk)
  concentrates in the 0-vs-≥1 distinction. The parity=2 coefficient is
  small and **flips sign across folds** (−0.073 to +0.057) — unstable,
  consistent with that category holding only ~29 patients. Parity=3+
  is small and consistently negative but based on only 6 patients total —
  not a reliable estimate either way. **Bucketing parity does not
  out-perform the continuous encoding** (P3 is at or slightly below F2 at
  every horizon) and trades away signal in the sparse categories for no
  visible gain.

## Question 3 — Is any improvement stable across horizons, resampling, and the untouched test set?

Mixed, and this is the most important caveat on the headline finding:

- **A3's significance is horizon-dependent, not uniform.** It's
  significant at ≥10m/≥20m/≥30m (at least one α variant each) but not at
  delivery. F7's significance is the reverse — significant at delivery
  only. Neither dominates at every horizon; a reader should not conclude
  "A3 is simply better" without the horizon qualifier.
- **Calibration is not free.** A3's Brier scores are *worse* than F2's at
  every horizon (e.g. delivery: 0.1460 vs. F2's 0.1404) despite better
  discrimination — exactly the ranking-vs-calibration split the plan's §3/A5
  anticipated. F7, by contrast, has the *best* Brier score of any model at
  delivery (0.1401) alongside its AUROC win — a cleaner result on that
  axis.
- **Test-partition behavior is noisy, as expected at N=83 (~17 positives).**
  A3 variants' test AUROC swings widely across horizons (0.6586 → 0.7175 →
  0.7460 as horizon lengthens for the fixed-0.5 variant) — directionally
  consistent with the CV story at ≥20/30m, inconsistent in magnitude, and
  this is exactly the sampling-noise pattern this project has flagged
  repeatedly at this test-partition size, not a new concern specific to
  this study.
- **Now checked (below): most of the flagged significant findings do not
  survive an independent fold resplit.** This is the single most important
  correction to this report.

## Stability checks (2026-09-15) — bootstrap-seed and fold-resplit sensitivity

Scripts: `scripts/phase18_stability_checks.py`. Data:
`stability_bootstrap_seed_sensitivity.csv`, `stability_fold_resplit_sensitivity.csv`.
Two checks, both following this project's own established conventions
exactly (`scripts/model3_parity_hybrid/hybrid_engine.py` §14,
`scripts/phase16_model3_verification.py`):

1. **Bootstrap-seed sensitivity** (5 seeds: 42, 1, 7, 123, 2024) — the
   underlying fitted predictions are fully deterministic; this checks
   whether the *reported p-value* is itself an artifact of one resampling
   draw, holding the model and the fold partition fixed.
2. **Fold-resplit sensitivity** (5 independent `StratifiedKFold` resplits,
   seeds 11/22/33/44/55) — the canonical, frozen M3 OOF scores are **not**
   retrained (matching `hybrid_engine.py`'s own precedent — M3 is not what
   this ablation tests); only the parity model and the fusion layer are
   refit under each new fold structure, and compared against a
   resplit-matched F2 refit under the identical resplit (a fair
   "does A3/F7 beat F2 under this resplit" test, not a comparison against
   a stale canonical F2).

### Results, the four flagged (model, horizon) pairs

| Model | Horizon | Bootstrap-seed p range (vs. F2) | Fold-resplit p range (vs. resplit-matched F2) | Verdict |
|---|---|---|---|---|
| A3, α=0.5 fixed | ≥10m | [.036, .048] — stable, always <.05 | [.025, .321] — significant in only 2/5 resplits | **Does not replicate** |
| A3, α train-selected | ≥20m | **[.000, .002]** — stable, always highly significant | **[.000, .005] in 4/5 resplits, .639 in the 5th** | **Replicates — the one robust finding** |
| A3, α=0.25 fixed | ≥30m | [.000, .003] — stable, always <.05 | [.065, .401] in 4/5, .004 in 1/5 | **Does not replicate** |
| F7 (interaction) | Delivery | [.032, .049] — stable, always <.05 | [.052, .234] — never <.05 in any of 5 resplits | **Does not replicate** |

**Three of the four findings that looked significant on the canonical fold
partition — and stayed significant across every bootstrap-resampling
seed tried on that same partition — evaporate under a genuinely different
fold partition.** This is exactly the failure mode bootstrap-seed
sensitivity alone cannot catch (it only resamples *within* the fixed
partition) and exactly why the plan's own stability requirement asked for
both checks, not one. A result that is "robust to bootstrap seed" here
turned out to mean only "robust to how you resample this one particular
5-way split of 547 patients" — not "robust to which 5-way split you use."

**A3 with a train-selected α at ≥20m is the one finding that holds up
under both checks — stable across resampling and stable across resplits
(4 of 5; the 5th resplit's much smaller delta, +0.0042 vs. +0.012–0.018 in
the other four, is itself informative: even the one robust finding has a
resplit-dependent magnitude, not just a resplit-dependent significance
label).** This is now the single most credible lead to come out of Stage 1
— more credible than the delivery-horizon F7 result the original version
of this report led with.

**What does *not* change:** F7's negative, sign-stable interaction
coefficient across all 5 canonical folds (§ "what the fitted weights say"
above) is a separate claim from whether its AUROC edge over F2 is
significant, and is not addressed or undermined by this resplit check —
that check only re-fits and re-splits at the AUROC-comparison level, it
does not refit F7 under new resplits and re-examine the coefficient's
sign there. That remains a plausible, interpretable, but *not yet
resplit-verified* secondary finding — a natural next check if this line
is pursued further.

## Stage 1d — full stability sweep, all 16 models × 4 horizons × 6 splits (2026-09-15)

Motivated directly by the single-pair check above: only one flagged
comparison survived, out of four that looked significant on the canonical
split. Rather than chase individual pairs further, this sweep asks the
resplit question of **every** Stage-1 model at **every** horizon at once.
Script: `scripts/phase18_full_stability_sweep.py`. Data:
`full_stability_sweep.csv`, `stability_ranking.csv`.

**One real bug caught and fixed before trusting this.** The first version
of the sweep reused a parity model fit once under the canonical fold
structure and sliced it for every resplit too — silently leaking canonical
train-fold information into resplit "held-out" patients (a resplit test
patient's parity score could come from a model trained on other members of
that same resplit test fold, just not on that patient's canonical
fold-mates). This inflated resplit-significance counts and was caught by
cross-checking against the previous section's numbers: A3(α=0.25)@≥30m had
shown only 1/5 resplits significant vs. F2 there, but 5/6 in the buggy
sweep — a direct contradiction that shouldn't exist for the same model,
horizon, and resplit seeds. Fixed by fitting the parity model fresh, per
split, inside the per-fold fitting function (see that function's docstring
in the script). **One small, explained, non-leakage discrepancy remains**
after the fix: A3 with a train-selected α reproduces the canonical
delivery AUROC at 0.7340 here vs. 0.7382 in the original Stage-1 run — a
difference traced to how the α-selection step's training-side parity score
is computed (in-sample-on-the-fold here vs. nested-out-of-fold in the
original), which occasionally shifts which α the grid search lands on.
Both conventions are legitimate "train-fold-only" hyperparameter selection;
this affects only this one variant's exact number, not the held-out
evaluation's leakage-safety, and the sanity check confirmed all other 15
models reproduce exactly.

### Results

| Horizon | Models beating M3-only in all 6 splits | Best vs. current hybrid (F2) |
|---|---|---|
| Delivery | Only A3(α=0.75) — small effect (+0.011) | **Nothing reliably beats F2** — every model's best split-count vs. F2 is 0/6 |
| ≥10m | Only A3(α=0.75) (+0.015) | **Nothing reliably beats F2** — best is A3(α=0.5) at 2/6 |
| ≥20m | **12 of 14 non-trivial models** — parity's contribution here is essentially mechanism-agnostic | **A3, train-selected α: 5/6** — the clear, singular standout, also the highest mean AUROC of all 16 models (0.6636) |
| ≥30m | **13 of 14 non-trivial models** — same story | Best is A4 logit-fusion (weights sum to 1): 3/6; A3(α=0.25): 2/6; no clean winner |

**The headline, now much better-supported than the single-pair check
alone could show:** at **delivery and ≥10m, no fusion mechanism tested —
including the currently-deployed hybrid — shows a resplit-robust
improvement over M3 alone.** This isn't a fusion-algorithm problem to
solve; it's evidence that parity's marginal contribution at these horizons
is genuinely too weak or noisy for any of the 16 mechanisms tried to
reliably capture across different patient partitions. At **≥20m and
≥30m, parity's contribution is robust and mechanism-agnostic** — almost
any reasonable way of combining the two scores captures it, including the
current hybrid. Within that, one mechanism is a doubly-verified,
substantially better performer than the status quo: **probability-space
averaging with a train-fold-selected α, at ≥20m specifically** (5/6
resplits significantly beat F2, confirmed independently by both this sweep
and the earlier single-pair check's 4/5).

## Stage 1e — deployable fusion (one model per fold, not one per horizon) (2026-09-15)

**Everything above this section has a real design flaw, caught by the
user, not found internally: every fusion model — Stage 1's 16, and the
earlier Parity Fusion / Model3+Parity Hybrid work before this study — was
fit separately per retrospective horizon, using that horizon's own
truncated M3 score as a training feature. That is not deployable.** During
actual labor, "minutes before delivery" is unknown — a system cannot select
which coefficients to use based on information that does not exist yet.
Model 3 itself never had this problem (one trained network, truncated at
eval time for retrospective scoring); the fusion layer never inherited
that discipline until now.

**Fix**, mirroring Model 3's own established convention exactly
(`docs/phase16_protocol.md` §4 — "every prefix length is a separate
training example"): one fusion model per fold, trained on every causal
truncation of every training patient's sequence pooled together
(sample-weighted 1/T_i per patient), then evaluated at each retrospective
horizon via the same prefix-truncation logic used everywhere else. Script:
`scripts/phase18_deployable_fusion.py`. Models: A3 fixed α (0.25/0.5/0.75
— need no fitting beyond the parity model), A3 α selected once (pooled,
not per horizon), F2/F7/A4(sum-to-one) deployable (one logistic fit each,
pooled).

### Results (before resplit checking)

Every deployable model beat M3-only at every horizon, including delivery
and ≥10m — a materially better picture than the horizon-specific study,
where nothing reliably beat M3-only near delivery. CV significance was
inconsistent near delivery (small-sample power, the same pattern seen
everywhere in this project) but the held-out test partition was
significant almost everywhere in the same direction.

## Stage 1f — fold-resplit stability check on the deployable models (2026-09-15)

Same discipline as every previous check in this study: canonical split +
5 independent `StratifiedKFold` resplits, M3's per-patient running-score
sequence frozen (never retrained), only parity and the fusion layer refit
per split. Script: `scripts/phase18_deployable_fusion_stability.py`, data:
`deployable_fusion_resplit_sensitivity.csv`, `deployable_fusion_stability_ranking.csv`.

### Results — decisive, and cleanly split by horizon

| Horizon | Models significant in all 6 splits | Best performer |
|---|---|---|
| Delivery | **Only A3(α=0.75)** — small effect (+0.011), but its worst split still has p=.019 | Nothing else is robust — A3(α=0.5)'s strong-looking canonical number (0.7388) does **not** survive resplitting (0/6, max p=.22) |
| ≥10m | **Only A3(α=0.75)** — small effect (+0.015), worst-split p=.031 | Same pattern — every other variant's larger canonical-split gain evaporates under resplitting |
| ≥20m | **All 7 deployable models, 6/6** | **A3, α selected once: mean Δ+0.062, worst-split p=.002** — the clear best, with A4(sum-to-one) a close second (Δ+0.056, p=.007) |
| ≥30m | **All 7 deployable models, 6/6** | **A3, α selected once: mean Δ+0.064, worst-split p=.003** — again the best, A4(sum-to-one) close second (Δ+0.060, p=.007) |

**This is the strongest, most completely verified result in the whole
study.** At ≥20m and ≥30m, parity's contribution is not only real and
mechanism-agnostic (as Stage 1d/§ full stability sweep already showed) but
now confirmed **deployable** — one fixed model, never told the horizon,
evaluated identically at every retrospective lead time, still wins
robustly across every independent patient partition tried.

### The final, deployable recommendation

**Adopt `A3_selected_once` — probability-space averaging with a single α
(≈0.30–0.35, i.e. weighting parity roughly 65–70%, selected once per fold
by pooled training AUROC, never re-selected per horizon) — as the fusion
mechanism.** It is one fixed model, run continuously, with no horizon
input required. Its honest performance profile:

- **Near delivery (delivery/≥10m): a real but statistically unconfirmed
  edge over M3 alone** — always directionally positive, sizeable point
  estimates (+0.010 to +0.025 mean AUROC), but not yet resplit-significant
  at this cohort's size. Do not claim a confirmed win here.
- **Early warning (≥20m/≥30m): a large, robustly and repeatedly confirmed
  win over M3 alone** — the exact regime an early-warning system exists to
  serve, and where this result is as solid as anything else in this
  project's post-lock investigation.

If a training-free fallback is preferred at delivery/≥10m specifically, a
fixed α=0.75 (75% M3, 25% parity) is the one alternative that is also
resplit-robust there, though its effect size is small. `A3_selected_once`
and fixed-α=0.75 need not be mutually exclusive candidates for a follow-up
external-validation comparison — both are now honestly characterized,
deployable, and resplit-verified, which nothing in this line of work was
before this correction.

## What Stage 1 does *not* answer (explicitly out of scope here)

Per the frozen protocol, Stage 1 is score-level only. It does **not** test:
whether M3's single pooled scalar is too compressed (R3/R4/R5 — needs
window-level sequences/attention weights, deferred to Stage 2); nonlinear
meta-models (F10–F13, deferred to Stage 3, and per the protocol's own
staging rule, only worth pursuing if Stage 1's linear/simple models leave
visible residual signal — arguably A3's win over F2 without any nonlinear
capacity at all is a mild argument *against* rushing to Stage 3, not for
it); operational (sensitivity/FAR/lead-time) metrics for anything beyond
what CV/test AUROC and Brier already show; or subgroup/residual analysis
(E4/E5/E6) beyond the coefficient-stability evidence above.

## Bottom line, against the plan's own "what counts as meaningful" bar (§15/§6), updated after the full stability sweep

**Question 1 (does parity add information beyond M3) — settled, and
strengthened by the full sweep**, not just the four originally-flagged
pairs: at ≥20m and ≥30m, 12–13 of 14 non-trivial fusion mechanisms
robustly beat M3-only across all 6 independent splits. This is
mechanism-agnostic and highly robust. At delivery/≥10m, parity's
contribution is real in direction but too weak or noisy for any tested
mechanism to reliably clear significance across resplits — an honest
horizon-dependent limit, not a fusion-algorithm failure.

**Question 2 (which fusion mechanism combines them best) — answered, with
a horizon-dependent recommendation, not a single universal winner:**

- **Delivery / ≥10m:** stick with the current hybrid (F2) or the current
  practice generally — no tested alternative shows a resplit-robust edge
  over it, and none over M3-only either. There is nothing to switch to
  here; the honest conclusion is that added complexity buys nothing at
  these horizons on this cohort's size.
- **≥20m: switch the fusion mechanism to probability-space averaging with
  a train-fold-selected α.** This is the one candidate in the entire
  16-model, 4-horizon, 6-split sweep that robustly and by a wide margin
  beats the current hybrid (5/6 splits) — independently confirmed by two
  separately-built scripts, one of which had a real bug caught and fixed
  before being trusted.
- **≥30m: no clean single winner**, but A4 logit-space fusion (weights
  constrained to sum to one) is the most consistent runner-up (3/6 vs. F2)
  and A3(α=0.25) has the highest raw mean AUROC (0.6585) with 6/6 vs.
  M3-only — either is a more defensible choice than the current hybrid at
  this horizon specifically, though neither clears the same bar the ≥20m
  candidate does.

**Question 3 (is it stable) — answered, and the answer is a genuine
caution about this whole line of work, not just this study:**
bootstrap-seed invariance, checked in isolation, would have certified
three findings that turned out to be fold-partition artifacts. Only the
fold-resplit check — recommended in the original plan, easy to skip, and
almost skipped here too after the first (buggy) sweep pass looked
plausible — caught both that and a second, independent bug in the
resplit-checking code itself. **The practical rule this leaves behind:
never report a fusion-mechanism comparison on this cohort without a
fold-resplit check, and never trust a resplit check without first
confirming it reproduces the canonical-split numbers it's supposed to be
extending.**

**Overall recommendation for "a stable fusion technique to combine M3 and
parity":** there is no single mechanism that is uniformly best. The
evidence supports a **horizon-specific choice** — keep the current
practice at delivery/≥10m, adopt train-selected-α probability averaging at
≥20m, and prefer the sum-to-one logit fusion or α=0.25 probability
averaging at ≥30m over the current hybrid. This is now a specific,
resplit-verified, actionable recommendation — a meaningfully stronger
deliverable than the single flagged lead this report started with.
