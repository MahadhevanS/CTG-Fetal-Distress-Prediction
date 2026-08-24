# Improvement Plan (Aug 2026): Idea 2 detail + 4-person split across Idea 1 & Idea 2

**Goal**: raise the delivered CRP-Crossformer (`checkpoints/ctg_crossformer_crp/`,
AUROC 0.8124) past a ~80% AUROC benchmark and/or fix its specificity at a
clinically usable sensitivity, without disturbing "crossformer is the fixed
architecture for the final proposal."

**Standing decision**: Idea 2 (calibrate → analyze errors → patch) is the
primary track — it's cheaper, mostly-built already, and targets a defect
that's already root-caused. Idea 1 (retrain other backbones under KI/CRP) runs
side-by-side as a parallel-track validation/ablation, not a replacement track.
See `docs/calibration_test_plan.md`, `docs/model8_crossformer_run_history.md`,
and `docs/models_1_to_7_inferences_log.md` for the evidence this plan is built
on.

---

## Part 1 — Idea 2 plan (misclassification → patch)

### Phase A — Wire in the calibration fix that already exists (no GPU)

The model over-predicts risk 3.5x and the hardcoded 0.30 threshold isn't
transferable across seeds (`docs/calibration_test_plan.md`, T3/T6). A fix
(per-fold Platt scaling + validation-derived threshold) is already
implemented in `scripts/calibrate_crp.py` and already plumbed as an opt-in
`--calibrated` flag in `scripts/run_clinical_review.py` — it has just never
been turned on by default or checked into the patient-facing report.

- **A1.** Run `scripts/calibrate_crp.py` for the delivered checkpoint (and
  confirm `checkpoints/ctg_crossformer_crp/calibration.json` is committed).
  Sweep `--target_sens` at 0.70 / 0.75 / 0.80 / 0.90 and record the resulting
  test specificity for each — this becomes the honest "here's what's
  actually achievable" table for the proposal, replacing the misleading
  0.30-threshold numbers.
- **A2.** Make `--calibrated` the default in `run_clinical_review.py` and in
  `src/explainability/patient_report.py` (currently mid-edit on this repo —
  check that diff before starting). Re-generate
  `docs/clinician_review/clinician_packet.html` with calibrated output.
- **A3.** Re-run `scripts/eval_all_models.py --json` so `docs/model_metrics.json`
  reflects calibrated operating points, not just the raw 0.30 threshold.

Output: a stable, reproducible sensitivity/specificity table, and the
patient-facing report showing numbers that mean what they say.

### Phase B — Build & run the misclassification analysis (no GPU)

`src/training/error_analysis.py`'s `ErrorAnalyzer` already exists but was
only ever wired into the old KI-MTF trainer (`train_knowledge_infused.py`),
never run against the delivered crossformer_crp model. Ground-truth clinical
features are already available per-window in
`data/processed_mil/{val,test}_extended_features.npy` — so this doesn't need
the model's own (nonexistent) prediction heads, just ground truth.

- **B1.** Write `scripts/error_analysis_crp.py`: load the delivered ensemble,
  run inference on val+test, and produce one dataframe per window with:
  `patient_id, fold, y_true, p_raw, p_calibrated, flagged, extended_features[18], figo_tier`.
  (`figo_tier` needs deriving from ground-truth features via the existing
  rule engine in `src/knowledge/` — reuse, don't re-derive the thresholds.)
- **B2.** Adapt `ErrorAnalyzer.analyze()` (or a light copy of it) to accept
  ground-truth `figo_tier`/features instead of model-predicted ones. Run it
  on the calibrated predictions from Phase A. Break FN/FP down by:
  - FIGO tier (does the old Model-8 finding — FNs cluster in FIGO-Suspicious,
    not Pathological — replicate on the current model? See
    `docs/fold4_anomaly_diagnostic.md` for the precedent.)
  - fold (is there one bad fold driving most errors, as in the old fold-4/fold-2
    issues documented in `docs/fold4_anomaly_diagnostic.md` and
    `docs/calibration_test_plan.md` T3's "41% of alarms depend on fold 2")
  - clinical feature ranges (STV/LTV/deceleration stats among errors vs. correct)
  - confidence (near-threshold misses vs. confidently-wrong)
  - patient-level clustering (do a handful of patients account for most FNs —
    a MIL/bag-composition effect — or is it spread evenly?)
- **B3.** Qualitative pass: pull the 5–10 worst false negatives and false
  positives through `scripts/demo_patient_report_single.py` /
  `run_clinical_review.py --explain all` and read them as a clinician would.
  Cross-check against `docs/clinician_review/`.
- **B4.** Write up `docs/error_analysis_findings.md`: state plainly whether a
  dominant, actionable pattern exists, or whether errors are diffuse (which
  is itself a real, useful finding — it would mean per-case root-causing is a
  dead end and argues more weight toward Idea 1).

### Phase C — Design and implement the patch (GPU only if the chosen patch needs retraining)

The concrete patch depends on what B4 finds. Don't pre-commit to one; decide
after B4 lands, from this menu:

| If B4 finds... | Candidate patch | GPU? |
|---|---|---|
| FNs cluster in FIGO-Suspicious (borderline) zone | Two-stage decision: a borderline-zone-specific secondary rule/threshold, or oversample borderline windows harder in fine-tuning | Only if fine-tuning; a post-hoc decision rule is CPU-only |
| Errors concentrated in 1–2 folds / patients | Investigate signal quality (SQA) for those patients; consider fold-aware calibration | **No** (diagnostic + calibration-level fix) |
| Errors are diffuse, not clustered anywhere | No cheap patch exists — this is evidence *for* pursuing Idea 1 (the backbone itself is the bottleneck) | N/A |
| Confidently-wrong (not just near-threshold) errors dominate | Suggests a representation gap, not a calibration gap — feed this directly into Idea 1's backbone comparison | N/A |

- **C1.** Implement the chosen patch.
- **C2.** Validate with the project's own bar: paired comparison against the
  Phase-A-calibrated baseline, replicate on val before touching test, and if
  it involves retraining, run it across the same 3 seeds (42/1/7) the CRP
  work used before trusting the sign of the effect — a 1-seed win has already
  burned this project once (`docs/calibration_test_plan.md`'s combiner
  test, `docs/model8_crossformer_run_history.md`'s "selective layer" rejection).
- **C3.** Update `docs/model_metrics.json`, `docs/calibration_test_plan.md`
  (close out the T7 "open decision"), and the final proposal draft.

---

## Part 2 — 4-person split (Idea 1 run side-by-side with Idea 2)

Two people on Idea 2 (Phase A/B, the priority track — no GPU), two people on
Idea 1 (GPU pilots). Idea 1 has a prerequisite the original ask didn't
account for: **`scripts/run_clinical_relational_pretrain.py` and
`src/training/clinical_relational_pretrain.py` are hardcoded to
`CTGCrossformerEncoder`** — CRP is *architecturally* backbone-agnostic (every
encoder emits the same 128-dim latent per the project's Universal Encoder
Signature rule) but is not yet *wired* to accept another encoder. That's real
code work, not a config change — it's called out explicitly below so it
isn't underestimated.

| Member | Track | Task | GPU? | Depends on |
|---|---|---|---|---|
| **M1** | Idea 2 | Phase A (A1–A3): wire calibration into inference/report path, produce the honest sens/spec table | No — CPU, ~160ms/window, test set runs in minutes | — |
| **M2** | Idea 2 | Phase B (B1–B3): build `error_analysis_crp.py`, adapt `ErrorAnalyzer`, run the FN/FP breakdown, qualitative review | No | Needs M1's calibrated predictions for the "real" error set (can start against raw 0.30-threshold predictions in parallel, re-run once A lands) |
| **M3** | Idea 1 | Generalize CRP pretraining to accept a pluggable encoder (add `--backbone` to `run_clinical_relational_pretrain.py` / `clinical_relational_pretrain.py`, matching the `(B,2,4800) → (B,128)` signature already enforced project-wide). Then run: (a) plain KI-MTF baseline on **MS-LSTM** using the existing `configs/model8_mslstm_config.yaml` (no CRP, quick sanity check first), (b) CRP-pretrain + fine-tune MS-LSTM once CRP is generalized, seed 42 only (single-seed pilot, not the full 3-seed protocol) | **Yes — Colab T4** (all training happens on Colab per `AI_AGENT_RULES.md` §5; CRP pretrain is 60 epochs, KI fine-tune is 100 epochs) | Should coordinate with M4 on the `--backbone` generalization so it isn't built twice |
| **M4** | Idea 1 | Same as M3 but for **CNN1D** (`configs/model8_cnn1d_config.yaml` already exists). Also owns collecting both backbones' results into one apples-to-apples table against crossformer_crp's own numbers, run through the *same* patient-level split + calibration protocol (not the older 5-fold-CV numbers in `docs/models_1_to_7_inferences_log.md`, which use a different eval protocol and can't be compared directly) | **Yes — Colab T4** | Same as M3 |

### Sync points

1. **After Phase A + B land (M1, M2)** — group review of `docs/error_analysis_findings.md` before anyone starts Phase C. This is the gate that decides what patch (if any) gets built, per the table in Phase C above.
2. **After M3/M4's single-seed pilots land** — compare against crossformer_crp on the *same* test set. Only invest in the full 3-seed replication (the project's own bar for trusting a result) if a candidate clears crossformer by a margin that looks bigger than the seed-to-seed noise already documented for this dataset (per-fold AUROC swings 0.68–0.87 per `docs/model8_crossformer_run_history.md`).
3. **Joint decision point**: even if Idea 1 produces a stronger backbone, treat it as an appendix/ablation result ("we checked alternatives; here's why we kept crossformer" or "here's the honest tradeoff") rather than swapping the delivered architecture this late — unless the gain is large and stable enough to justify redoing the calibration + error-analysis work in Part 1 for the new backbone too.

### Explicit GPU flags

- **GPU required**: M3's and M4's entire track (any KI-MTF training run, with or without CRP, per project convention — Colab T4, never local CPU).
- **GPU not required**: everything in Phase A and Phase B (Idea 2's core), and Phase C *unless* the chosen patch requires retraining/fine-tuning rather than a post-hoc decision rule or calibration change.
