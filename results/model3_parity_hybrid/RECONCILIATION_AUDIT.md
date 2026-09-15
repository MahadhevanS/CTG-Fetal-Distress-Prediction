# Reconciliation Audit: Model 3 + Parity Hybrid

> ## ✅ STATUS UPDATE 2026-09-13: ALL THREE CODE BUGS FIXED, PIPELINE RERUN, RESULTS CROSS-VERIFIED
> `scripts/model3_parity_hybrid/hybrid_engine.py` has been fixed at all
> three sites identified below (the horizon patient-drop, the hardcoded
> Parity Fusion λ=1.4, and the degenerate duration-tertile split) and the
> full pipeline was rerun end to end. The regenerated
> `early_warning_metrics.csv`, `incremental_value_analysis.csv`, and
> `duration_sensitivity.csv` now match this audit's independently-derived
> `canonical_horizon_table.csv` (produced by
> `canonical_horizon_reconciliation.py`, a separately-written script) to
> 4 decimal places at every horizon for every model — the fix is
> confirmed correct by two independent implementations agreeing, not by
> one script's self-report. `final_summary.md` has been fully rewritten
> to reflect the corrected numbers and the narrower, more accurate
> interpretive claims; see its own "CORRECTED AND RERUN" banner. The
> calibration and duration-tertile *narrative* mismatches described in
> §7 below (numbers that never matched their own source CSVs) have also
> been corrected in `final_summary.md` by re-quoting directly from
> `calibration_metrics.csv` and the fixed `duration_sensitivity.csv` —
> no further code change was needed there since `calibration_metrics.csv`
> itself was always correct; only its transcription into prose was wrong.
> The classification remains **Promising but Inconclusive**, on somewhat
> narrower evidence than originally reported (see `final_summary.md`
> §20.6 for the updated rationale). The analysis below is preserved as
> the original diagnostic record.

**Audit date:** 2026-09-13
**Audit type:** Independent reconciliation — no retraining, no model changes, no new hyperparameters. This audit re-evaluates `hybrid_engine.py`'s own already-fitted, already-frozen predictions under a corrected patient-inclusion rule, and separately cross-checks every number quoted in `final_summary.md` against the CSV/JSON artifacts that were supposed to produce them.
**Verdict up front:** The walkthrough's central discrepancy is real, fully explained, and confined to one specific bug. Beyond that bug, three additional numeric mismatches between the narrative and the saved artifacts were found during this audit and are reported below — none of them were on the original checklist, but all were checked because the checklist's discipline ("do not accept either set of values without explaining the difference") was applied to everything, not just the flagged item.

---

## 1. Reconcile Model 3 early-warning values — ROOT CAUSE FOUND, 100% EXPLAINED

**Cause: a different patient-inclusion rule, not a horizon selector, checkpoint, cohort, fold, timestamp, or seed difference.**

Direct inspection of `scripts/model3_parity_hybrid/hybrid_engine.py` lines 373–401 (and the same pattern repeated at lines 449–450 for the incremental-value analysis) shows:

```python
if h == 0:
    elig_mask_cv = np.ones(n_patients, dtype=bool)
else:
    elig_mask_cv = np.array([np.any(df_rolling[df_rolling["patient_id"] == p]["time_before_delivery_min"] >= h) for p in clean_pids])
...
y_h_cv = y_pat[elig_mask_cv]
score_m_cv = models_dict_cv[m_name][h][elig_mask_cv]
```

For horizons above delivery, this **drops every patient who has no window with `time_before_delivery_min >= h`** before computing AUROC. This is a genuinely different rule from the one used everywhere else in this project — `src.evaluation.phase13_common.get_patient_scores_at_horizon_corrected` and `src.models.phase16_causal_attention.eligible_prefix_length` (which Model 3's own `predict_at_horizon_for_patients` uses) both **fall back to the patient's last available window and keep the patient in the evaluation set**. No patient is ever dropped in the canonical convention, at any horizon. This fallback is exactly what `scripts/model3_direct_comparisons.py`, `scripts/phase16_temporal_attention_model.py`, and every other Tier-1/2/3 hardening script already do, and it is what reproduces the locked numbers.

**This is checked, not assumed.** I wrote `scripts/model3_parity_hybrid/canonical_horizon_reconciliation.py`, which regenerates `hybrid_engine.py`'s identical per-patient scores (same checkpoints, same folds, same fusion fit, same seeds — verbatim reuse, nothing retrained) and evaluates them under **both** conventions side by side:

| Horizon | N (buggy convention) | Model 3 AUROC (buggy convention, reproduced) | Model 3 AUROC (walkthrough reported) | Model 3 AUROC (locked canonical) |
|---|---|---|---|---|
| Delivery | 547 | 0.7216 | 0.7216 | 0.7216 |
| ≥10m | 545 | 0.6637 | 0.6637 | 0.6683 |
| ≥20m | 534 | 0.5882 | 0.5882 | 0.6015 |
| ≥30m | 492 | 0.5811 | 0.5811 | 0.5976 |

**The reproduced buggy-convention numbers match the walkthrough's reported numbers exactly, at all three affected horizons, to 4 decimal places.** The bug is the complete and sole explanation — not a partial one, not a coincidence. Every other candidate cause the user listed (horizon selector, timestamp, cohort, fold assignment, checkpoint, pooling, preprocessing, outcome definition, seed, "a legitimate rerun compared against stale values") is ruled out: the per-patient *scores* were always computed with the canonical fallback logic already built into `predict_at_horizon_for_patients`; only the final evaluation-time patient filter was non-canonical.

**Scope of the bug: it is not Model-3-specific.** The same `elig_mask` was applied identically to P6, Max, P90, Parity Fusion, and Hybrid in `early_warning_metrics.csv` and `incremental_value_analysis.csv`. Every non-delivery number in the walkthrough's Table 2 is on a shrunken, non-canonical N and is not comparable to any previously locked figure. Table 1 (delivery) and Table 3 (operational) are **not affected** — delivery uses `elig_mask = all patients` unconditionally, and the operational evaluation uses a separate code path that was never filtered this way (confirmed below).

**Note on the automated test suite:** `test_results.txt` records `[PASS] Identical patient inclusion across all models at >=10m/>=20m/>=30m` (27/27 assertions passed). That assertion checked that the *same* patients were dropped for every model at each horizon — which is true, and passed — but it never checked that the resulting N matched the project's canonical/locked N. It validated internal self-consistency of the bug, not correctness against the canonical convention. This is why 27/27 green did not catch the discrepancy; the gap was in test *coverage*, not in test *execution*.

---

## 2. Recompute all horizon metrics with one canonical evaluator

Done — `scripts/model3_parity_hybrid/canonical_horizon_reconciliation.py`, output `results/model3_parity_hybrid/canonical_horizon_table.csv`. Same patient set (N=547 CV / N=83 test) used for **every** model at **every** horizon, matching the canonical fallback convention exactly.

### Required table (delivery + all three early-warning horizons, canonical convention)

| Horizon | P6 | Model 3 | Hybrid | Hybrid − Model 3 (CV, p) |
|---|---|---|---|---|
| Delivery | 0.6872 | 0.7216 | 0.7324 | +0.0108 (p=.172) |
| ≥10m | 0.6859 | **0.6683** | 0.6896 | +0.0213 (p=.097) |
| ≥20m | 0.6238 | **0.6015** | 0.6481 | +0.0467 (p=**.005**, CI [+0.0119,+0.0810]) |
| ≥30m | 0.5828 | **0.5976** | 0.6451 | +0.0475 (p=**.013**, CI [+0.0109,+0.0827]) |

(Model 3's bolded values are its canonical, locked figures — restored, not the walkthrough's incorrect 0.6637/0.5882/0.5811.)

**What survives correction and what doesn't:**
- **≥10m's claimed significance does NOT survive.** Walkthrough reported Δ+0.0257, p=.044 (significant) on N=545. Canonical: Δ+0.0213, p=.097 (not significant) on N=547. This specific claim — "statistically significant improvement at all three pre-delivery horizons" — is **false** under the canonical evaluator; it is two of three, not three of three.
- **≥20m and ≥30m significance DOES survive**, though at meaningfully smaller magnitude than claimed (+0.0467 vs. claimed +0.0608; +0.0475 vs. claimed +0.0740). These are genuine, still-positive findings — just smaller ones, on the correct N.
- **Model 3 was never as catastrophically weak as claimed.** Canonical Model 3 at ≥30m (0.5976) is actually slightly *above* P6 (0.5828), not "worse than baseline P6" as §20.4 asserted — that claim was itself an artifact of the same patient-dropping bug depressing Model 3's own number. This is consistent with, and now further confirmed by, the Tier-2 hardening finding already on record (`reports/candidate_models_metrics_reference.md` §1.2a) that canonical Model 3 is statistically indistinguishable from Max/P90 at every horizon — it was never an outlier-weak candidate at ≥20/30m, it was always comparable to the other fixed aggregators.
- **Full CSV** (`canonical_horizon_table.csv`) additionally reports Hybrid vs. Max and Hybrid vs. P90 at every horizon: Hybrid beats P90 significantly at ≥20m (p=.023) but not ≥30m (p=.097); the Max comparison is never significant at any horizon (p≥.05 throughout, ≥20m borderline at p=.050).
- Patient counts, positive/negative counts, and patient IDs at each horizon are the canonical `clean_pids`/`test_pids` lists (547/83, 110/27 positive at delivery) — unchanged and identical across all six models, as they must be under the canonical rule. AUROC, AUPRC, CI, and paired-bootstrap differences are all in `canonical_horizon_table.csv`.

---

## 3. Reconcile internal-test rankings — CONFIRMED, HYBRID DOES NOT RANK FIRST

Values reproduce exactly (delivery_metrics.csv matches final_summary.md Table 1 to 4 decimals): Hybrid=0.6916, Parity Fusion=0.7148, Max=0.7130, Model 3=0.6729, P90=0.6622, P6=0.6497.

**On the held-out internal test partition, the ranking is Parity Fusion (0.7148) > Max (0.7130) > Hybrid (0.6916) > Model 3 (0.6729) > P90 (0.6622) > P6 (0.6497). The Hybrid ranks third, not first.** It does rank first on CV (0.7324, highest in the table). The correct statement, replacing the walkthrough's "strongest overall model": *the Hybrid achieves the highest point estimate on cross-validation but does not outperform Parity Fusion or Max pooling on the held-out internal test partition; CV and internal-test rankings disagree, which is itself worth treating as a signal of estimation noise at this sample size (N=83, ~17 positives) rather than resolved in either table's favor.*

---

## 4. Clarify resplit reporting — CONFUSION CONFIRMED, PLUS A SMALL NUMBER MISMATCH

`robustness_results.csv` already stores hybrid AUROC, delta-vs-Model-3, p-value, and CI **separately per resplit** — the underlying computation was never missing this, only the prose in `final_summary.md` collapsed it into "AUROC remained within [0.7289, 0.7341]" and described that range as if it were the range of *improvements*.

| Resplit seed | Hybrid AUROC | Δ vs. Model 3 | p | 95% CI |
|---|---|---|---|---|
| 11 | 0.7286 | +0.0070 | .420 | [−0.0101, +0.0234] |
| 22 | 0.7318 | +0.0102 | .231 | [−0.0066, +0.0264] |
| 33 | 0.7302 | +0.0086 | .310 | [−0.0083, +0.0243] |
| 44 | 0.7306 | +0.0090 | .266 | [−0.0070, +0.0248] |
| 55 | 0.7308 | +0.0092 | .287 | [−0.0082, +0.0258] |

**Two separate problems, both now fixed by this table:** (1) as the user identified, "AUROC range" and "improvement range" were conflated in the prose; (2) independently, the walkthrough's quoted range **[0.7289, 0.7341]** does not match the CSV's actual range **[0.7286, 0.7318]** — 0.7341 does not appear anywhere in the file. Neither the correct nor the incorrect range should be read as "stability of the *improvement*": every single resplit's Δ-vs-Model-3 confidence interval **crosses zero** (none of the five is significant). The honest statement is: *the Hybrid's raw AUROC is stable across five independent resplits (0.7286–0.7318), and its point-estimate improvement over Model 3 is consistently positive (+0.0070 to +0.0102) but not statistically significant in any single resplit.* A narrow AUROC range is not evidence of a stable improvement — the CIs are the relevant evidence, and they are uniformly inconclusive.

---

## 5. Audit operational comparisons

Verified directly from `operational_metrics.csv`, which **does** match `final_summary.md` Table 3 exactly (no fabrication here).

1. **Target:** 0.80, implemented as "at least" in effect — the threshold is set at the 20th percentile of training-fold positive patients' delivery scores, so it targets exactly 80% *of the training distribution*, not a guaranteed 80% on held-out data.
2. **Selected per fold:** Yes — `fold_thresholds` computed inside the `for f_idx in range(5)` loop, a new threshold per outer fold.
3. **Training patients only:** Yes — `pos_tr_scores = tr_scores[y_pat[tr_mask] == 1]`, `tr_mask` excludes that fold's held-out patients. No leakage.
4. **Patient-level or window-level thresholding:** **Patient-level.** The threshold is set on each training patient's single delivery-horizon score, then applied against a **window-level rolling sequence** (`seq_scores`, one value per prefix) for the alert simulation. This mixed convention is not a new invention — it is identical to `compute_lead_time_for_model` in `scripts/phase16_temporal_attention_model.py`, already used for Model 2/Model 3's own reference-doc operational numbers.
5. **Why achieved sensitivities differ (88.2–92.7%):** a 20th-percentile cut on ~22 training-fold positives per fold is a coarse, quantized threshold; the empirical hit rate on the different held-out fold naturally drifts from exactly 80% by an amount that depends on each model's own score distribution near that percentile. This is sampling noise inherent to the method, not a bug, but it does mean —
6. **False alerts are counted per patient** ("ever alerted across the whole rolling sequence" — `patient_alerted[p_idx] = True` if any window crosses threshold), matching Phase 16's own convention (Model 2/Model 3), **not** Phase 15's single-evaluation-point convention.
7. **Alert episodes are not separately collapsed** — a patient is alerted or not; there is no multi-episode counting.
8. **Same operational evaluator was used for all four rows in Table 3** (P6, Model 3, Parity Fusion, Hybrid) — this table's internal comparisons are apples-to-apples with each other.
9. **Not comparable to Phase 15.** This confirms the reference doc's own standing warning (§7): Phase 15's Max/P90 operational figures use a different, single-evaluation-point FAR definition. **§20.4 point 4 of the walkthrough directly violates this** by comparing the Hybrid's Phase-16-convention FAR (65.2%) against "Max had a 51.5% FAR in Phase 15" — those two numbers are not measuring the same thing and the comparison should be deleted, not merely caveated.

**One additional, smaller inconsistency found in this section:** the Parity Fusion row's operational sequence hardcodes `1.4 * to_logit(p_par_val)` for the fusion lambda (line 600), rather than using that fold's own tuned λ from `select_lambda_trainfold` (which is what the AUROC-table Parity Fusion numbers correctly use, and which ranges ~1.3–1.5 per fold per the reference doc). 1.4 sits inside that range, so the resulting numbers are plausible, but this is a different procedure from the one used for Parity Fusion's own AUROC figures elsewhere in the same document — worth noting as an inconsistency, not necessarily a materially wrong number.

**Required correction to the claim "lowest FAR across the entire repository":** delete it. The corrected statement is exactly what the user proposed: *under the operational evaluation implementation used in this experiment (Phase-16-style, patient-level, ever-alerted convention), the Hybrid produced a lower reported false-alert rate than P6, Model 3, and Parity Fusion (all evaluated identically in this same table) — but at 88.2% achieved sensitivity, tied with Model 3 and below Parity Fusion's 90.9% and P6's 92.7%, so the FAR comparison is not sensitivity-controlled, and no comparison to Phase 15's Max/P90 figures is valid under any convention used in this project.*

---

## 6. Reframe permutation-control interpretation

`permutation_control_results.csv` is verified: 30 rows, `exceeds_real_delta` is `False` in every row — **0/30 confirmed**, not fabricated. The real delta being tested is +0.0108 (Hybrid − Model 3 at delivery), and the permutation shuffles `parity_pat` across all 547 patients before refitting per fold — a genuine patient-level permutation, matching the procedure description.

**Convention:** with 0/30 exceedances and no plus-one correction applied in the script, the reported empirical p-value convention is **0/30 = 0** (not 1/31 ≈ .032). The audit brief is right that this should be stated explicitly rather than left implicit, since the two conventions differ by a factor that matters at this replicate count. Recommended going forward: report both, e.g. "0/30 (equivalently, p<1/31 under a plus-one-corrected convention)."

**Required wording correction, exactly as the user specified:** replace "confirming that the observed gain is driven by true biological information rather than flexible parameter noise" with: *none of the 30 patient-level parity permutations exceeded the observed improvement. This supports the presence of a non-random parity-associated signal under the specified permutation procedure — it does not establish biological causation, and cannot rule out cohort-specific confounding, site/practice-pattern effects, data-collection artifacts, or residual model-selection effects, none of which a within-cohort permutation test is designed to detect.* Also note: 30 replicates only resolves down to a minimum detectable exceedance rate of ~3% (1/30) — enough to say "not obviously explained by permutation noise," not enough to make a strong quantitative claim about how rare the true effect is under the null.

---

## 7. Complete complementarity analysis

Already computed, not missing — but two of the figures quoted in `final_summary.md`'s prose do not match their own saved artifacts (see the dedicated section below). What does check out:

- **Prediction correlation** (`prediction_correlation.csv`): Hybrid–Model 3 Pearson r=0.9651, Spearman ρ=0.942; Hybrid–Parity Fusion r=0.8845, ρ=0.9034; Model 3–Parity Fusion r=0.8177, ρ=0.7807; Model 3–P6 r=0.9148; Parity Fusion–P6 r=0.8887. The Hybrid is much more correlated with Model 3 than with Parity Fusion, consistent with Model 3 carrying more weight in the fusion at delivery.
- **Fusion coefficients** (`fusion_coefficients.csv`): Model 3 coefficient positive in all 6 fits (5 folds + test model), range [+0.658, +0.758] — matches the walkthrough's claimed range exactly. Parity coefficient positive in all 6 fits, range [+0.178, +0.301] — also matches exactly. **Sign stability for both features is confirmed, not merely claimed.**
- **Error disagreement** (`error_disagreement_analysis.csv`): Hybrid corrects 32 of Model 3's errors and introduces 26 new ones (net +6, of which +4/+4 split between positives/negatives corrected vs. harmed) — a real but modest, roughly-balanced trade rather than one-sided improvement. This number was not itself quoted incorrectly in the walkthrough (it wasn't quoted at all), but it belongs in any complete complementarity picture and is included here for completeness.
- **Calibration-only sensitivity analysis:** not run. Given the calibration numbers below are unreliable, this should be re-run rather than approximated from the current (mismatched) figures.

### Calibration and duration-tertile: numbers in the narrative do not match their own artifacts

These were **not** on the user's original checklist — they were found by applying the same "verify every quoted number against its source artifact" discipline to the rest of the document, per the audit brief's own instruction not to accept any value without checking it.

**Calibration** (`calibration_metrics.csv`, unmodified, produced by `hybrid_engine.py` itself):

| Model | Brier (actual CSV) | Brier (walkthrough §20.3 Table 5) | Slope (CSV) | Intercept (CSV) | Slope/Intercept (walkthrough, Hybrid only) |
|---|---|---|---|---|---|
| Model 3 | 0.1424 | 0.1362 | 1.288 | 0.1051 | — |
| Parity Fusion | 0.1708 | 0.1415 | 0.7922 | 1.0009 | — |
| Hybrid | 0.1403 | 0.1345 | 1.1453 | 0.1749 | 0.9942 / −0.0086 |

None of the six numbers quoted in the walkthrough match the corresponding cell in the artifact the walkthrough cites as its source. The direction of the claim (Hybrid has the best/lowest Brier of the three) happens to still be correct under both the quoted and the actual numbers, but the specific values, and the specific slope/intercept pair claimed for "near-ideal calibration," are not what `calibration_metrics.csv` contains. **This section of the walkthrough should be treated as unverified until it is re-quoted directly from the CSV** (values above) or the CSV is regenerated and shown to match a corrected narrative.

**Duration/window-count tertile stratification** (`duration_sensitivity.csv`, unmodified): the file contains **one row**, not three — `Low (<17 windows), n_patients=547, n_positives=110, model3_auroc=0.7216, hybrid_auroc=0.7324` — i.e., every single patient fell into the "Low" stratum, and the reported AUROCs are simply the whole-cohort delivery AUROCs, not a stratified subset. The tertile-cut logic (`np.percentile(w_counts_pat, [33.33, 66.67])` compared via strict `>`) produced a degenerate split, almost certainly because window counts are heavily concentrated at the maximum value (3–17 windows per patient per `data_feature_audit.json`, with 17 as a hard ceiling many patients share) — a strict `>` comparison against a percentile that lands on a common value puts most or all patients in the lowest bin. **The three-stratum table quoted in §20.4 point 6 of the walkthrough (Low 0.712/0.701, Mid 0.741/0.732, High 0.728/0.719) does not exist in any saved artifact and should be treated as unverified pending a fixed stratification (e.g. `pd.qcut` with duplicate-handling, or rank-based tertiles).**

**Taken together, these two findings mean the walkthrough contains numbers not traceable to any script output in at least two more places beyond the horizon bug the user flagged**, plus a smaller mismatch in the resplit range (item 4). None of these appear to be adversarial — they read as transcription or memory errors during report-writing, consistent with values that are plausible and directionally correct but not exactly sourced — but they mean **no number in this walkthrough should be treated as reliable until independently checked against its artifact**, which is the standard this audit has now applied.

---

## 8. Final classification

**Promising internal research candidate, pending reconciliation of the canonical horizon and operational evaluations; eligible for external-validation consideration if all discrepancies are resolved** — using the exact tightened wording the audit brief specified, because that is what the reconciled evidence supports:

- **What holds up after correction:** Hybrid vs. P6 is significant at delivery (CV p=.004, test p=.03) and vs. Model 3 on the internal test partition at delivery (p=.03). Hybrid vs. Model 3 at ≥20m and ≥30m remains CV-significant under the canonical evaluator (p=.005, p=.013), at reduced but real magnitude (+0.047, +0.048 rather than the claimed +0.061/+0.074). The parity permutation control (0/30) and fusion-coefficient sign-stability (positive for both features in all 6 fits) are both independently confirmed, not fabricated.
- **What does not hold up:** the ≥10m "significant improvement" claim (p=.097 canonical, not .044). The claim that the Hybrid is "the strongest overall model" (it is 3rd on the internal test partition, behind Parity Fusion and Max). The claim that Model 3 "collapses" below P6 at early horizons (it does not, canonically). The "lowest FAR across the entire repository" claim (invalid cross-phase, cross-convention comparison). The "proves biological causation" language for the permutation control. The specific calibration numbers and the three-stratum duration table (unverified, likely mismatched to their own source files).
- **A finding not previously visible in the buggy report:** the Hybrid's early-warning advantage over Model 3 (≥20m/≥30m) is **not** matched by a significant advantage over Parity Fusion alone at those same horizons (all p>.35, `canonical_horizon_table.csv`) — Parity Fusion by itself already captures most of the early-warning signal (0.6435/0.6368 vs. Hybrid's 0.6481/0.6451). The cleaner scientific reading is *"Parity carries early warning, Model 3 carries delivery-time discrimination, and the fused model roughly inherits each component's strength in its own regime"* rather than "the hybrid synthesizes and exceeds both."

**This is not "Strong candidate"** (delivery CV significance vs. Model 3 is unresolved at p=.172, ≥10m significance evaporates under correction, and it does not lead the internal-test ranking) **and not "Not supported"** (≥20/30m CV significance over Model 3 and delivery significance over P6 both survive independent re-derivation) **and not "Not interpretable"** (the mechanism — logit-fusion of two already-understood components — is fully transparent and the leakage audit's core claims about fold-disjointness and training-only fitting check out from direct code inspection). **"Promising but inconclusive" is the correct classification, on the corrected evidence, not merely as a hedge.**

### Before this can be promoted to external validation

1. Regenerate `early_warning_metrics.csv`, `incremental_value_analysis.csv`, and `duration_sensitivity.csv` from `hybrid_engine.py` with the `elig_mask` patient-drop removed (use `canonical_horizon_table.csv` as the reference; the fix is deleting lines 380–382/449–450's filtering, not touching any model-fitting code).
2. Re-derive the calibration table by reading `calibration_metrics.csv` directly rather than re-quoting from memory, and fix the tertile stratification (e.g. rank-based cut) before re-reporting duration sensitivity.
3. Remove the cross-phase/cross-convention FAR claim and the "strongest overall model" / "proves biological causation" language, replacing them with the corrected statements given above.
4. Re-run the resplit-range and calibration numbers through an automated "narrative matches artifact" check before the next write-up — the test suite's 27/27 pass rate did not, and structurally could not, catch any of the four issues found in this audit, because none of them were assertions it made.
5. No claim of clinical readiness, deployment readiness, biological causality, or "solved" fetal distress prediction is supported by any evidence produced here, before or after correction.

**External-validation panel, per the audit brief:** P6, Max pooling, P90 pooling, Model 3, Parity Fusion, and Model 3 + Parity Hybrid should all be carried forward once corrected; Model 4 remains supplementary. This is an addition to, not a replacement of, `docs/external_validation_handoff.md`'s existing two-candidate scope (Model 3, Parity Fusion) — that document should be updated to add the Hybrid as a third candidate only after item 1 above is done and the corrected numbers are stable.
