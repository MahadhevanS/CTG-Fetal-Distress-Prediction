# Pre-External-Validation Hardening Plan

Written 2026-09-13. This is a **plan**, mirroring the protocol-before-code
discipline used throughout Phases 13–17 — nothing in this document has
been executed. It addresses twelve specific gaps identified in a review of
`reports/candidate_models_metrics_reference.md` before the four candidates
(Model 3, Model 4, P90 pooling, parity fusion) go to external validation.

**Honest framing up front:** some of these checks (#5, #6 in particular)
could weaken, not just polish, Model 3's current "strongest candidate"
status. This plan is written to find that out, not to produce a better-
looking document around an unchanged conclusion. Two items are already
verified as real, not hypothetical — see #1 and #8 below, checked directly
against source code before this plan was written, not assumed from memory.

---

## 1. Fix the parity ≥30-minute horizon-convention inconsistency [DONE 2026-09-13]

**Verified, not hypothetical.** `scripts/parity_fusion_test.py`'s
`get_patient_scores_at_horizon()` uses the **original** first-eligible-
chronological-window convention:

```python
eligible = np.where(t_pts >= h_val)[0]
chosen = idx[eligible[0]] if len(eligible) > 0 else idx[-1]
```

— not `get_patient_scores_at_horizon_corrected()` (argmin-nearest, subject
to `t_i ≥ h`), which every Phase 13-audit, Phase 14, Phase 15, and Phase 16
comparison uses. Delivery (`h=0`) is unaffected — both conventions select
the last window there — but **parity's ≥30m figures (CV Δ+0.0494, test
Δ+0.0784) are not on the same horizon-convention footing as Model 3/Model
4/P90's ≥30m figures**, and the reference doc's side-by-side table
currently implies they are.

**Plan:** write `scripts/parity_fusion_horizon_correction.py` — reuse
`parity_fusion_test.py`'s exact fitted objects (frozen P6 scores, per-fold
and train-val parity models, per-fold λ) unchanged, swap only the horizon-
selection call to `get_patient_scores_at_horizon_corrected`, recompute
≥10m/≥20m/≥30m (delivery is already correct and is not recomputed), rerun
the paired bootstrap + DeLong tests. No refitting of anything — this is a
horizon-selection substitution only, using the same estimator discipline
established in Phase 13.0A.

**Deliverable:** a corrected ≥30m row for parity in the reference doc,
with the original (inconsistent) number kept alongside and explicitly
labeled, not silently replaced — matching this project's own standing rule
against silent reconciliation (see the P90-estimator handling in item #2).

**Result:** `scripts/parity_fusion_horizon_correction.py`,
`results/parity_fusion/parity_fusion_horizon_corrected_results.csv`. The
correction *strengthened* parity's early-warning case: held-out internal
test partition deltas grew at ≥10m (+0.0463, p=.006), ≥20m (+0.1266,
p<.001), and ≥30m (+0.1185, p=.019 vs. the original convention's +0.0784,
p=.166) — parity fusion is now test-significant at every horizon, not just
delivery. See `reports/candidate_models_metrics_reference.md` §4.2/§4.5.

---

## 2. Adopt one canonical P90 estimator for the headline comparison table

Two legitimate definitions exist: Phase 13E's Hazen-style weighted-quantile
(`src/aggregation/recency_weighted_p90.py`) and Phase 14/15's
`numpy.percentile` (linear interpolation). Phase 14 chose the latter "for
continuity," and Phase 15 explicitly adopted it as canonical going forward.

**Plan:** no new computation — a documentation decision. Adopt
`numpy.percentile` (delivery CV 0.7178, Δ+0.0307, p=.225) as the number
quoted in every headline comparison from this point forward, since it is
already the more-used, more-recently-endorsed choice (2 of 3 derivations).
Retain the Hazen-based number (0.7254, Δ+0.0382, p=.121) as an explicitly
labeled historical/sensitivity figure in a footnote, never as a competing
headline — exactly as the reference doc's §3 already does; this item makes
that the stated *policy* rather than an implicit choice, so future work
doesn't reopen it.

---

## 3. Add direct paired comparisons: Model 3 vs. Max, Model 3 vs. P90 [DONE 2026-09-13]

**Verified gap.** `results/phase16/phase16_models_2_3_summary.json` stores
only aggregate metrics (`auroc_table`, `per_fold_auroc_h0`, interpretability
fields) — no per-patient prediction vector for Model 2 or Model 3 was ever
saved. Every existing comparison is "candidate vs. single-window P6"; a
direct Model-3-vs-Max and Model-3-vs-P90 test has never been run.

**Model 3 vs. Model 4 already exists** — `Model_4_peak_aware_fusion_results.csv`
has `cv_delta_vs_model3` / `cv_p_vs_model3` columns computed and reported
(reference doc §2). No new work needed there; this item is scoped to the
two genuinely missing pairs only.

**Plan:** write `scripts/model3_direct_comparisons.py` — reload Model 3's
committed per-fold checkpoints (`results/phase16/checkpoints/Model_3_magnitude_position_fold*.pt`,
`..._testmodel.pt`), reconstruct its per-patient predictions at all four
horizons via the existing `predict_at_horizon_for_patients()` (no
retraining — pure inference), reconstruct Max/P90 per-patient scores from
the same frozen `results/phase13/audit/p6_predictions.npz` window scores
using the canonical estimator from item #2, then run paired bootstrap
(B=2000) + DeLong for Model3-vs-Max and Model3-vs-P90 at every horizon, CV
and held-out test.

**Deliverable:** `results/model3_direct_comparisons.csv` + an addendum
section in the reference doc placed alongside the existing Model3-vs-P6
and Model3-vs-Model4 rows.

**Result:** Model 3 is **not** statistically distinguishable from Max or
P90 at any horizon, on either split — every CI crosses zero, every p-value
well above .05, and on the held-out internal test partition the point
estimates actually favor Max/P90 by 0.03–0.07 AUROC (not significantly).
This does not reverse Model 3's own CV-significant result against the P6
baseline, but it means Model 3 has not been shown to add anything beyond
what a training-free percentile calculation already provides. See
`reports/candidate_models_metrics_reference.md` §1.2a/§1.6.

---

## 4. Pin down Model 3's exact time representation, in writing

Not a new experiment — a precision-of-language fix. Verified from
`src/models/phase16_causal_attention.py` / `scripts/phase16_temporal_attention_model.py`:

`elapsed_t` = minutes since **that patient's own first retained window**
(`start_sample / (fs·60)`, offset so the patient's first window reads 0),
divided by 60 before being passed to the scorer MLP. This is:

- **Not** time-before-delivery (that would be non-causal).
- **Not** absolute wall-clock or gestational time.
- Gap-aware — correctly reflects the ~6.9% of patients with quality-gate-
  dropped windows (the same fix established in Phase 13's elapsed-time bug).

**Plan:** add one unambiguous paragraph, worded exactly as above, to the
top of Model 3's section in the reference doc and to `docs/phase16_protocol.md`'s
own Section 0(a), so "temporal position" is never left to reader inference
again.

---

## 5. Shuffled-time control (genuinely new — the most important item here) [DONE 2026-09-13]

**Why this matters more than a polish item.** Model 3's evidence that it
uses "something beyond magnitude" currently rests on one fact: attention
argmax agrees with the pure-magnitude argmax in only 31.6% of patients
(vs. Model 2's 100%). That's suggestive, not dispositive — a model with an
extra, uninformative input dimension could also show reduced argmax
agreement purely from added noise, without genuinely exploiting temporal
order.

**Plan — the actual control:** retrain Model 3's identical architecture
(same 2-input scorer, same hidden size, same training procedure, same
folds) with `elapsed_t` **values randomly permuted within each patient's
own window sequence** (breaking true temporal correspondence while
preserving the exact same value distribution and parameter count as real
Model 3) — directly analogous to the permutation control already run for
parity in Phase 13 (0/30 shuffled permutations exceeded the true effect).

- If shuffled-time Model 3 performs comparably to real-time Model 3 →
  the gain is a capacity artifact, not genuine temporal exploitation —
  **this would meaningfully undercut Model 3's current interpretation**,
  though not necessarily its raw predictive value.
- If shuffled-time Model 3 collapses toward Model 2's performance →
  strong, clean evidence real temporal ordering is what's being used.

**Plan — time-ablation, for completeness:** also report Model 2's existing
numbers explicitly as the "zero time information" ablation endpoint
(already computed, just needs to be framed as part of this same ablation
ladder rather than a separate side-note) — Model 2 → shuffled-time Model 3
→ real-time Model 3 is the complete, three-point ablation curve.

**Deliverable:** `scripts/model3_shuffled_time_control.py` (checkpointed,
same resumable pattern as `phase16_model3_verification.py`), 5-fold CV +
held-out test, same statistical treatment as the original Model 3 run.
Reported regardless of outcome, per this project's standing rule against
running a control and discarding it if inconvenient.

**Result — run with three independent shuffle draws (seeds 777/888/999),
not one, as an added shuffle-draw-sensitivity check beyond what this item
originally specified.** Delivery, CV: Model 2=0.7006, shuffled-time Model 3
= 0.7003/0.6979/0.7007 across the three seeds (extremely stable — the
shuffled result is not an artifact of one draw), real-time Model 3=0.7216.
The pre-registered decision rule scores this **"capacity artifact"**:
shuffled-time is statistically indistinguishable from real-time Model 3
(Δ=−0.0213, p=.217, CI=[−0.0554,+0.0128]) and collapses almost exactly
onto Model 2 (Δ=−0.0003, p=.876). That verdict is reported as computed,
not softened. It is not a clean refutation, though: real-time Model 3's
edge over both endpoints is directionally consistent across all three
independent shuffle draws, and the shuffled-time vs. real-time gap *is*
significant at the exploratory ≥10m horizon (Δ=−0.0338, p=.009), where
shuffled-time also significantly underperforms the P6 baseline itself
(p=.012–.018 across all three seeds) — scrambled time actively misleads
the model there rather than being harmlessly ignored. Net: Model 3's own
CV-significant result against P6 (unaffected, not retested here) stands;
the claim that it works *via genuine temporal-position use* is now neither
confirmed nor refuted at this cohort's size. Full detail:
`results/phase16/model3_shuffled_time_ablation_ladder.csv`,
`model3_shuffled_time_seed_sensitivity.csv`,
`model3_shuffled_time_control_summary.json`, and
`reports/candidate_models_metrics_reference.md` §1.2c/§1.6.

---

## 6. Test whether Model 3's gain survives adjustment for recording duration [DONE 2026-09-13]

**Verified gap.** Phase 15 ran exactly this check for every *fixed*
aggregator and found the advantage shrinks with longer recordings for all
of them (r = −0.12 to −0.23, all p<.01) — `results/phase15/phase15_duration_analysis.csv`.
The identical check was never run for Model 3, a *trainable* aggregator,
which could plausibly behave differently.

**Plan:** reuse Phase 15's `eligible_window_count()` helper directly, at
delivery, correlate patient-level window count against (Model 3 prediction
− P6 single-window prediction), Pearson + Spearman, same reporting format
as `phase15_duration_analysis.csv`'s existing rows so the two are directly
comparable in one combined table.

**Deliverable:** one new row appended to a copy of that duration-analysis
table, headlined in the reference doc as "does Model 3 inherit the same
confound Phase 15 found in every fixed aggregator, or not."

**Result:** it does — Model 3's delta vs. P6 correlates with eligible
window count at r=−0.213 (p<.0001, Spearman ρ=−0.209, p<.0001), stronger
than P90's (−0.190) or Max's (−0.123) and comparable to Mean's/Median's
(the two aggregators Phase 15 judged weakest). Being trainable did not let
Model 3 escape the confound. See
`results/model3_duration_confound.csv` and
`reports/candidate_models_metrics_reference.md` §1.2b/§1.6.

---

## 7. AUPRC relative to prevalence

Documentation-only. Primary-endpoint prevalence is 110/547 = 20.1% — the
AUPRC a random classifier would score. Every candidate's AUPRC already
exists in the committed result files; this item adds one column
("lift over prevalence" = AUPRC / 0.201) to every AUPRC figure already in
the reference doc. No new computation.

**Plan:** e.g. Model 3 delivery CV AUPRC 0.4335 → lift ×2.16 over
prevalence; compute the same ratio for Model 4, P90, parity, and the P6
baseline, and add as a column to reference-doc §6.

---

## 8. Expand the Model 4 leakage-control description

**Verified precisely** by reading `scripts/phase16_model4_peak_aware_fusion.py`'s
fold loop directly (not assumed) before writing this plan: for outer fold
`f`, `fold_scorers[f]` — Model 3's checkpoint for that fold, itself trained
excluding fold `f`'s held-out patients — is used to build the 4-feature
vector for **both** `tr_pids` and `te_pids` of that same fold. The
`StandardScaler` and `LogisticRegression` are fit on `X_tr` only and applied
to `X_te`. No step touches fold `f`'s held-out patients during any fitting.

**Plan:** no new computation — write this exact chain into the reference
doc's Model 4 section and `docs/phase16_protocol.md` Section 5, replacing
the current one-line "reuses Model 3's frozen checkpoints" note with the
verified walkthrough above.

---

## 9. Separate delivery discrimination, early-warning performance, and operational false-alert behavior into distinct sections

Documentation restructuring. The reference doc currently threads AUROC-
at-4-horizons and operational metrics (lead time, FAR, ≥20/30min detection)
through the same per-model sections. Reorganize the next revision into
three explicit, separately-headed blocks per candidate:

1. **Delivery discrimination** (the primary endpoint, single number the
   headline table leads with).
2. **Early-warning performance** (≥10/20/30m AUROC, explicitly labeled
   exploratory per every phase's own tiering rule).
3. **Operational behavior** (sensitivity/specificity/FAR/lead-time/detection
   at the calibrated operating point) — with the reference doc's existing
   §7 caveat (Phase 15 vs. Phase 16 operational metrics are not directly
   comparable to each other) promoted to a visible warning box, not a
   trailing bullet.

No new computation — a presentation fix, but one that directly serves the
external-validation audience this document is ultimately for.

---

## 10. Standardize on "held-out internal test partition"

Editorial, global. Every occurrence of bare "test," "test set," or
"held-out test" across `reports/candidate_models_metrics_reference.md`,
`reports/phase13_16_closure_summary.md`, `docs/phase16_protocol.md`, and
`docs/external_validation_handoff.md` gets replaced with "held-out
**internal** test partition" — precisely to prevent a reader from
conflating CTU-UHB's internal 83-patient partition with the external
cohort discussed in the handoff document. Mechanical find-and-replace pass,
file by file, no numbers change.

---

## 11. Carry Max and P90 forward as mandatory baselines in external validation

Currently `docs/external_validation_handoff.md` names Model 3 and parity
fusion as the two candidates going external, with Max/P90 discussed only as
internal CTU-UHB context. This item amends the handoff protocol's
statistical plan (Section 6) to require Max and P90 pooling be computed and
reported on the external cohort **alongside** every Model 3 / parity
comparison — not as candidates for promotion, but as the cheap, zero-
training reference points that any claimed external improvement must beat,
consistent with this project's own repeated finding that simple fixed
statistics are a serious baseline, not a formality.

**Plan:** add one paragraph to `docs/external_validation_handoff.md`
Section 6, no other change to that document's scope or governance.

---

## 12. Reaffirm: no promotion without external validation

Already the standing rule in every report since Phase 13 — restated here
as confirmation, not a new task. This plan changes evidence and
presentation; it does not change the governance conclusion. Model 3,
Model 4, P90, and parity fusion remain "worth external validation," not
"ready for clinical use," regardless of how items 1–9 resolve.

---

## Sequencing

| Tier | Items | Why this order |
|---|---|---|
| **Do first — cheap, mechanical, no new risk to the headline conclusion** | 2, 4, 7, 8, 9, 10, 11, 12 | Pure documentation/policy; zero chance of changing any number — **[DONE 2026-09-13]** |
| **Do second — cheap new computation, reuses existing frozen artifacts** | 1, 3, 6 | Real fixes/additions, but low effort (inference-only or direct reuse of committed checkpoints), and 1/6 are corrections to internal consistency, not re-tests of Model 3 itself — **[DONE 2026-09-13]** |
| **Do last, most carefully — genuinely new training, could change the conclusion** | 5 | The shuffled-time control is the one item that could meaningfully alter how Model 3's result should be interpreted; it should be run and reported after everything else is settled, with the same two-sided commitment to report it "regardless of outcome" that governed every control in Phases 13–16 — **[DONE 2026-09-13]** |

**All three tiers are now complete.** Their combined effect on the
headline conclusion was mixed and reported in full at every step, exactly
as this plan committed to: parity fusion's early-warning case got
*stronger* (item 1), and Model 3's "strongest candidate" framing got
*weaker* across three independent lines of evidence — no significant edge
over Max/P90 (item 3), a shared duration confound (item 6), and a
shuffled-time control the pre-registered rule scores "capacity artifact"
at the primary horizon, though not cleanly (significant in Model 3's favor
at ≥10m, consistent direction at delivery across three shuffle draws, item
5). None of this reopens or contradicts Model 3's own original
CV-significant result against P6 — that specific test was never rerun,
only supplemented. The honest net position, unchanged from what this plan
set out to determine honestly regardless of which way it went: Model 3,
Model 4, P90, and parity fusion are all still "worth external validation,"
not "ready for clinical use" (item 12) — and Model 3 specifically should
now be presented as "worth testing, mechanism unresolved," not as clearly
ahead of the cheaper alternatives it was originally framed against. No
further internal re-testing is planned; external validation
(`docs/external_validation_handoff.md`) is the next actual step.
