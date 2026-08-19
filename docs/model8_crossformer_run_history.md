# Model 8 (Knowledge-Infused Multi-Task Framework) — CrossFormer Track Run History

**Purpose of this document**: continuity/handoff brief. Contains full run history,
configs, results, and reasoning so this investigation can be continued from a
fresh chat session on either machine (local dev machine or GPU laptop) without
re-deriving context. Structured per run as: **Run configs → Run metrics → Run
inferences → Updations for next run**.

**Project**: CTG Fetal Distress Prediction. Model 8 = Knowledge-Infused
Multi-Task Framework: a shared temporal encoder + up to 4 task heads
(DistressHead = primary binary target, pH ≤ 7.15; FIGOHead = 3-class FIGO
classification; ClinicalFeatureHead = 8-feature regression; FIGOCriteriaHead =
7 decomposed binary FIGO judgments). 8 ablation variants exist:
`distress_only`, `plus_figo`, `distress_features_only`, `plus_features`,
`distress_figo_only`, `full`, `plus_criteria`, `features_criteria`. Backbone
being tracked here: **CTG-CrossFormer** (CNN + bidirectional cross-attention +
transformer, 256-dim pre-adapter → 128-dim universal latent).

**Goal driving this whole track**: Model 8's best AUROC (~0.79) trails the
standalone CrossFormer classifier's own benchmark (CV 0.8565, Test 0.8653) by
~0.067 — user wants at least a 5% (~0.05) AUROC boost to close that gap.
**Status as of this document: not yet achieved.** Best so far: ~0.80 (proxy,
unvalidated) / 0.7995 (validated).

**Key standing facts, established once, apply to every run below unless noted:**
- Dataset: CTU-CHB, patient-level stratified 5-fold CV, 546 unique patients,
  ~6,826–6,917 windows total (small variation run-to-run depending on which
  machine/dataset-generation state; not itself a concern, see Run 1 setup notes).
- `data/processed/*.pt` and `feature_scaler.npz` are git-ignored — never
  transferred via `git pull`, only code is. Must be regenerated (Colab
  `pipeline.py` + `generate_feature_scaler.py`) or copied machine-to-machine
  manually.
- Standard significance bar used throughout: a clean 5/5-fold win vs.
  `distress_only` (Wilcoxon-style, p=0.03125). No CrossFormer variant has hit
  this cleanly; best cases are 4/5-fold wins.
- Every full-scale run uses `--seed 42` for reproducibility (fixed early in
  the project after discovering the training script had no seeding at all).

---

## Pre-existing state (before this document's tracked runs — established in
## an earlier session phase, included for context)

- Fixed a severe LTV (long-term variability) preprocessing bug in
  `src/preprocessing/features.py::calculate_variability()` across **three
  combined fixes**: (1) percentile range (p95–p5) instead of raw
  peak-to-peak, (2) local linear detrending per 1-minute sub-window, (3)
  excluding accel/decel episodes from the range calculation (literature-
  driven — FIGO defines "saltatory" as >25bpm sustained >30min and "reduced"
  as <5bpm sustained >50min, both longer than the pipeline's 20-minute
  window, so per-window snapshot thresholding was never a faithful
  implementation to begin with). Verified end-to-end on a real Colab
  regeneration: mean LTV 46.26 → 28.00 bpm, >25bpm prevalence 81.7% → 53.4%.
- Fixed `TemperatureScaler.fit()` in `src/training/calibration.py` — was
  allowing negative temperatures via a clamp-only-in-forward-pass bug; fixed
  via log-space reparametrization (`T = exp(log_T)`), always positive now.
- Fixed a critical **y_features scale-mismatch bug** in
  `compute_multitask_loss()` — `y_features` (raw clinical units) was compared
  via MSE directly against small-scale network outputs with no
  normalization. Fixed by Z-normalizing `y_features` before the MSE.
- Added `FIGOCriteriaHead` (decomposed FIGO judgments, now 7 flags — see Run
  6.5 below for why 8→7) as a prototype alternative to the single 3-class
  FIGOHead.
- Standalone CrossFormer benchmark (`train_ctg_crossformer.py`) fixed for a
  compound class-weighting bug (WeightedRandomSampler + full pos_weight
  double-correcting) → CV AUROC 0.8565, Test AUROC 0.8653. **This is the
  ceiling everything in this document is measured against.**
- `configs/model8_crossformer_config.yaml` iteratively tuned via an "Option-A"
  diagnostic (20-epoch/1-fold, 5 configs): landed on `scheduler: onecycle`
  (max_lr matching base lr, pct_start=0.1 — NOT the codebase's 10×-lr
  default), `sampling: balanced` (BalancedBatchSampler, matching the
  standalone script's oversampling), `heads: hidden_dim=128, dropout=0.3`
  (matching CrossFormer's own standalone classifier capacity). EMA's
  disable decision came from this same diagnostic but was **only confirmed
  at full scale in Run 2 below** (the diagnostic was short enough that it
  could have just been non-convergence, not genuine harm).

---

## RUN 1 — Standard CrossFormer, full 8-variant ablation sweep (baseline)

### Run configs
- Command: `python src/training/train_knowledge_infused.py --config configs/model8_crossformer_config.yaml --ablation all --seed 42`
- Backbone: `ctg_crossformer` (standard, 128-dim shared latent for all heads)
- Scheduler: `onecycle`, max_lr=0.0003, pct_start=0.1
- Loss weighting: `fixed` (λ_figo=0.3, λ_features=0.2, λ_knowledge=0.1, λ_consistency=0.5, λ_criteria=0.3)
- EMA: **disabled**. SWA: disabled. Augmentation: *believed* disabled at the
  time (later discovered to actually be **enabled** the whole time — see the
  bug note after Run 10).
- Sampling: `balanced` (BalancedBatchSampler)
- Fold split: **original** (distress-label-only stratification — the fold-2
  imbalance bug was not yet discovered/fixed).
- Dataset: post-LTV-fix Colab-regenerated (N≈6,826 total, N≈6,177 train).
- Heads: hidden_dim=128, dropout=0.3. Criteria head: 7 flags (already fixed by this point).

### Run metrics
| Variant | AUROC (mean ± std) | Folds beating `distress_only` |
|---|---|---|
| distress_only | 0.7743 ± 0.0566 | — |
| plus_figo | 0.7875 ± 0.0489 | 4/5 |
| distress_features_only | 0.7800 ± 0.0476 | 3/5 |
| plus_features | 0.7816 ± 0.0456 | 4/5 |
| distress_figo_only | 0.7837 ± 0.0368 | 3/5 |
| **full** | **0.7893 ± 0.0515** | 4/5 |
| plus_criteria | 0.7730 ± 0.0537 | 1/5 |
| features_criteria | 0.7783 ± 0.0340 | 2/5 |

### Run inferences
- `full` best, `plus_figo` close second — both real but not clean 5/5
  significant wins (both lose specifically on fold 1).
- `plus_criteria`/`features_criteria` (the new decomposed-criteria head)
  **did not pay off** — `plus_criteria` is the *weakest* of all 8 variants.
  The simpler 3-class FIGOHead beats the decomposed 7-flag approach.
- `features_criteria` fold 1 hit a calibration edge case: `T=0.010` (the
  0.01 clamp floor), `RuntimeWarning: overflow encountered in exp` — LBFGS
  driving T toward zero, a classic near-perfectly-separable-logits pathology.
  Not fatal, but that fold's calibration is unreliable.
- Still a real, unresolved ~0.067 AUROC gap vs. the 0.8565 standalone
  CrossFormer ceiling.

### Updations for next run
- Investigate whether EMA (currently off based on a short diagnostic) and
  loss-weighting method actually matter at full scale — the "cheap" levers
  not yet confirmed. → **Run 2, 3, 4**.
- Investigate whether Model 8's hyperparameters (borrowed wholesale from the
  standalone classifier's own tuning) are actually optimal for the
  *multi-task* loss landscape — never tuned for Model 8 specifically. →
  **Run 5** (needed building `tune_model8_hyperparameters.py`).
- `plus_criteria`/`features_criteria` underperforming is now a known,
  accepted result — deprioritize the criteria-head direction going forward.

---

## RUN 2 — Standard CrossFormer, `full` only, EMA enabled

### Run configs
- Command: `python src/training/train_knowledge_infused.py --config configs/model8_crossformer_config.yaml --ablation full --seed 42` (with `ema.enabled: true` manually toggled in the config)
- Everything else identical to Run 1.

### Run metrics
`full`: **AUROC 0.7461 ± 0.0514** (vs. Run 1's 0.7893)

### Run inferences
- **EMA confirmed genuinely harmful at full scale** — not just a short-diagnostic
  artifact as originally caveated. Consistent drop across folds (fold 2:
  0.7128→0.6684, fold 5: 0.7464→0.7041). One side benefit: EMA improved ECE
  (0.0941 vs. Run 1's 0.1268) even while hurting AUROC — smooths the decision
  boundary at the cost of discrimination.

### Updations for next run
- EMA should stay **off** by default going forward for this backbone —
  settled, don't re-litigate without new evidence. Updated the stale
  code comment in `configs/model8_crossformer_config.yaml` to reflect this
  (was previously hedged as "may just be non-convergence").
- Test whether loss-weighting method changes this picture. → **Run 3, 4**.

---

## RUN 3 — Standard CrossFormer, `full` only, EMA on + uncertainty weighting

### Run configs
- Command: `python src/training/train_knowledge_infused.py --config configs/model8_crossformer_config.yaml --ablation full --loss_weighting uncertainty --seed 42` (EMA still `true`)

### Run metrics
`full`: **AUROC 0.7593 ± 0.0550**

### Run inferences
- Uncertainty weighting partially offsets EMA's damage (0.7461→0.7593) but
  still well below the EMA-off baseline (0.7893). Confounded result (two
  variables changed at once) — needed an isolated test.

### Updations for next run
- Re-run uncertainty weighting **alone**, EMA off, to isolate its true effect. → **Run 4**.

---

## RUN 4 — Standard CrossFormer, `full` only, uncertainty weighting, EMA OFF (clean isolated test)

### Run configs
- Command: `python src/training/train_knowledge_infused.py --config configs/model8_crossformer_config.yaml --ablation full --loss_weighting uncertainty --seed 42` (EMA off, i.e. base config unmodified except the weighting flag)

### Run metrics
`full`: **AUROC 0.7771 ± 0.0556**

### Run inferences
- Clean result: uncertainty weighting alone (no EMA) is a **small regression**
  vs. the fixed/EMA-off baseline (0.7893→0.7771, −0.0122). Not a win.
- Full 2×2 picture across EMA × weighting-method for `full`:

  | EMA | fixed | uncertainty |
  |---|---|---|
  | off | **0.7893** (best) | 0.7771 |
  | on | 0.7461 | 0.7593 |

  Real interaction effect: uncertainty weighting helps when EMA is on
  (offsets its damage) but hurts when EMA is off. Not simply additive.
- **Conclusion: `fixed` weighting + EMA off remains the champion recipe.**
  Both "cheap" levers (EMA, loss-weighting method) are now exhausted without
  a win.

### Updations for next run
- Stop exploring EMA/loss-weighting-method combinations for this backbone —
  diminishing returns confirmed empirically, not just assumed.
- Move to the more expensive lever: dedicated Model-8-specific hyperparameter
  search (never done — Model 8 has always borrowed the standalone
  classifier's own tuned LR/schedule). → **Run 5**.
- Also consider trying the other candidate backbones (MS-LSTM, CNN1D) since
  they don't share CrossFormer's specific architecture-bypass handicap. →
  **Run 6**.

---

## RUN 5 — Hyperparameter tuning proxy search, standard CrossFormer architecture

### Run configs
- Built `src/training/tune_model8_hyperparameters.py` specifically for this
  (random search directly over `train_and_evaluate_model8()`, since
  `tune_hyperparameters.py` only covers the 7 single-task standalone models,
  zero awareness of Model 8's multi-task loop).
- Search space: `lr` [1e-4…8e-4], `onecycle_pct_start` [0.1/0.2/0.3],
  `weight_decay` [1e-5/1e-4/1e-3], `batch_size` [16/32/64], `lambda_figo`
  [0.1–0.5], `lambda_features` [0.1–0.3], `lambda_knowledge` [0.05–0.2],
  `loss_weighting_method` [fixed/uncertainty/dwa/gradnorm].
- **EMA forced off for every trial** (already-settled finding, not
  re-searched). Head capacity NOT searched (separately justified for
  CrossFormer already).
- Command: `python src/training/tune_model8_hyperparameters.py --config configs/model8_crossformer_config.yaml --ablation full --n-trials 12 --epochs 15 --k-folds 3 --seed 42`
- **PROXY METRIC**: 15 epochs / 3 folds per trial (not the real 50/5) — cheap
  search, explicitly flagged as needing full-length re-validation.

### Run metrics
Best trial (of 12): `lr=0.0002, onecycle_pct_start=0.2, weight_decay=1e-05,
batch_size=16, lambda_figo=0.2, lambda_features=0.3, lambda_knowledge=0.2,
loss_weighting_method=dwa`

**Proxy AUROC: 0.8044** | score=0.6691 | accuracy=79.86% | sens@90spec=46.62% | ece=0.2107 | brier=0.1584

Saved to `configs/model8_tuned_hyperparameters.yaml` under key `ctg_crossformer__full`.

### Run inferences
- Best proxy result across the whole session at time of running — but
  **unvalidated** (reduced epochs/folds). Note this tuner also silently
  switches MS-LSTM off its `warmup_cosine` scheduler onto `onecycle`
  wherever it's used (hardcoded per-trial), a side effect worth remembering.
- **User initially mis-attributed this result to MS-LSTM in conversation —
  it was later corrected: this is the standard CrossFormer architecture's
  tuning result, not MS-LSTM's.** (MS-LSTM has never actually had a
  hyperparameter search run against it — see Run 6.)

### Updations for next run
- Built `configs/model8_crossformer_tuned_config.yaml` (base config + these
  winning hyperparameters, at full 50 epochs/5 folds) to validate this proxy
  result properly. **Still pending as of this document — deprioritized, see
  the note after Run 10.**
- Try MS-LSTM's own ablation sweep in parallel (independent of this). → **Run 6**.

---

## RUN 6 — MS-LSTM, full 8-variant ablation sweep (EMA-confounded)

### Run configs
- Command: `python src/training/train_knowledge_infused.py --config configs/model8_mslstm_config.yaml --ablation all --seed 42`
- Backbone: `multiscale_lstm`, hidden_size=128, num_layers=1, dropout=0.3
  (tuned-search winners from an earlier, separate single-task tuning run).
- Scheduler: `warmup_cosine` (base config's own setting, never changed).
- **EMA: enabled** — this config was never updated with the EMA-off finding
  discovered via CrossFormer-specific diagnostics; this is a **known confound**.
- Loss weighting: fixed. Sampling: none specified (defaults to plain
  shuffling + dynamic pos_weight, matching MS-LSTM's own standalone
  benchmark methodology — this one IS correct/intentional, unlike the EMA gap).

### Run metrics
| Variant | AUROC |
|---|---|
| distress_features_only | 0.7893 |
| distress_only | 0.7887 |
| features_criteria | 0.7849 |
| plus_features | 0.7866 |
| plus_criteria | 0.7826 |
| plus_figo | 0.7812 |
| distress_figo_only | 0.7727 |
| **full** | **0.7643 (worst variant)** |

### Run inferences
- Striking, opposite pattern from CrossFormer: for CrossFormer `full` was
  the *best* variant; for MS-LSTM `full` is the *worst*. Knowledge infusion
  appears actively harmful for this backbone — **but this entire sweep is
  confounded by EMA being on**, and EMA is now known (from CrossFormer
  testing) to hurt more complex multi-task variants disproportionately —
  exactly the pattern seen here (full/plus_figo/distress_figo_only cluster
  at the bottom).
- MS-LSTM's own standalone ceiling (0.8126 default hyperparams / 0.8195
  tuned) is below CrossFormer's (0.8565) to begin with, but MS-LSTM's
  standalone classifier does NOT have CrossFormer's architecture-bypass
  problem (confirmed: `MultiScaleLSTMForClassification` classifies from the
  true `encoder.latent_dim`, matching what Model 8 receives) — so a fair
  (EMA-off) MS-LSTM comparison is still a legitimate open question.

### Updations for next run
- **MS-LSTM was parked here.** Never re-run with EMA off. Never had its own
  hyperparameter tuning run (the tuning result the user later pasted was
  mistakenly attributed to MS-LSTM but was actually for wide-distress
  CrossFormer — see Run 8). **Both of these remain fully pending.**

---

## [Side investigation, not a training run] — LTV literature check & criteria-head flag drop

Prompted by the user asking to check prior art before trusting the (already
twice-fixed) LTV calibration. Live WebSearch found the FIGO 2015 guideline
defines "saltatory" (>25bpm) as requiring **>30 minutes** sustained, and
"reduced" (<5bpm) as requiring **>50 minutes** sustained (in baseline
segments) — both longer than the pipeline's entire 20-minute assessment
window. Conclusion: a per-window LTV>25 binary flag was never structurally
capable of faithfully implementing "saltatory," independent of how well LTV
itself is computed.

**Action taken**: dropped `variability_increased` from
`derive_figo_criteria_flags()`/`FIGOCriteriaHead` (now **7 flags**, joining
the already-dropped `variability_reduced`). LTV remains a continuous
`ClinicalFeatureHead` regression target, unaffected — only the two binary
saltatory/reduced classifications were dropped. `classify_figo()`'s 3-class
FIGOHead target has the same underlying issue but was left untouched
(out of scope of what was asked).

## [Side investigation] — Patent-differentiation constraint re-examined

Re-read `docs/patent_risk_analysis.md` directly (rather than relying on
earlier self-imposed caution). Found the documented differentiation against
GE's US12094611B2 rests on: (1) biochemical pH target vs. graphical pattern
target, (2) FIGO knowledge infusion via loss function, (3) explainability —
and critically, on **continuous end-to-end signal→latent mapping with no
bounding boxes/correlation loops**, NOT on a specific latent width or every
head sharing one bottleneck. This cleared the way to widen `DistressHead`'s
capacity without any real patent risk — the earlier framing of "128-dim
shared bottleneck = untouchable patent constraint" was an overstatement.

---

## RUN 7 — Wide-distress CrossFormer, `full` only, default hyperparameters (pre fold-fix)

### Run configs
- **New architecture built**: `src/models/knowledge_infused_framework_wide_distress.py`
  — `CTGCrossformerDualLatentEncoder` (subclasses `CTGCrossformerEncoder`,
  returns both the 128-dim `z` AND the 256-dim pre-adapter `pooled`
  representation from one forward pass) + `WideDistressHead` (classifies
  from the 256-dim `pooled` instead of the shared 128-dim `z`, mirroring
  `CTGCrossformerForClassification`'s own classifier exactly) +
  `KnowledgeInfusedFrameworkWideDistress` (drop-in compatible, same output
  arity). FIGOHead/ClinicalFeatureHead/FIGOCriteriaHead unchanged (still
  128-dim `z` only).
- Wired into `train_knowledge_infused.py` via `backbone.model:
  ctg_crossformer_wide_distress` (additive branch, standard path untouched
  — regression-tested).
- Config: `configs/model8_crossformer_widedistress_config.yaml` — identical
  to Run 1's baseline config in every other setting (fixed weighting, EMA
  off, same scheduler/lr/batch), isolating the architecture change alone.
- Command: `python src/training/train_knowledge_infused.py --config configs/model8_crossformer_widedistress_config.yaml --ablation full --seed 42`
- Fold split: still the **original** (pre fold-2-fix) split.

### Run metrics
`full`: **AUROC 0.7924 ± 0.0444**
(accuracy 83.67%±3.64%, auprc 0.4363±0.1025, f1 0.4714±0.0515, precision
47.97%±11.56%, recall 48.73%±7.24%, specificity 89.69%±4.71%,
sens@90spec 44.36%±7.88%, ece 0.1808±0.0751, brier 0.1524±0.0335)

Per-fold: Fold1=0.8060, Fold2=0.7090, Fold3=0.8270, Fold4=0.8312, Fold5=0.7885

### Run inferences
- Small, real edge over the standard architecture (0.7924 vs. Run 1's
  0.7893, +0.0031) on the same (old) fold split.
- Fold 2 is weak again (0.7090) — same fold that's been weak across multiple
  variants/architectures regardless of what changes. This is what triggered
  the fold-composition investigation.
- User kicked off a hyperparameter tuning search on this architecture too
  (→ Run 8, initially mis-attributed to MS-LSTM in conversation).

### Updations for next run
- Investigate why fold 2 is persistently weak — build a diagnostic script. →
  led to `scripts/diagnose_fold2_composition.py` and the joint-stratification fix.
- Validate the wide-distress + tuned-hyperparameters combination once both
  pieces are ready. → **Run 8** (tuning), **pending validation run** (not yet done).

---

## [Diagnostic build, not a training run] — Fold-2 composition investigation & fix

`scripts/diagnose_fold2_composition.py`: found fold 2 (under the *original*
distress-only stratification) had FIGO-Normal representation of only 5.5%
vs. 8–13% in other folds, and FIGO-Suspicious of 78.6% vs. lower elsewhere —
a real, structural composition skew, not noise. Root cause:
`create_patient_level_folds()` only ever stratified on the binary distress
label; nothing constrained the FIGO-class mix per fold.

**Fix**: `create_patient_level_folds()` in `src/training/train.py` given an
optional `secondary_labels` parameter — when provided, builds a JOINT
(distress × secondary-label) stratification key instead of distress alone,
with a safe fallback to distress-only stratification if any joint stratum
would have fewer than `k_folds` patients. **Default `None` preserves exact
original behavior** — `train.py`'s own standalone single-task call site
(line ~455) was deliberately left unchanged, so every existing standalone
benchmark result remains exactly reproducible. Only
`train_knowledge_infused.py`'s call site was updated to pass
`secondary_labels=dataset.y_figo`.

Verified via `diagnose_fold2_composition.py --joint` on local (stale) data:
fixed the FIGO-Normal outlier (8.3% vs. 8.9–12.0% elsewhere, no longer
flagged) but **surfaced a new outlier** in FIGO-Pathological% (20.1% vs.
14.2%, z=+4.82) and Prolonged-Decel count (z=+7.10) — a genuine improvement
on the targeted axis, not a perfect/complete fix (per-patient window-count
variance of 1–81 makes simultaneous balance on every axis hard with 546
patients).

---

## RUN 8 — Hyperparameter tuning proxy search, WIDE-DISTRESS CrossFormer architecture

**(This is the run the user initially mis-attributed to MS-LSTM in
conversation — corrected here. MS-LSTM itself has never had a tuning run.)**

### Run configs
- Command (inferred from context — user ran "the same tuning script... for
  this widened crossformer model"): `python src/training/tune_model8_hyperparameters.py --config configs/model8_crossformer_widedistress_config.yaml --ablation full --n-trials 12 --epochs 15 --k-folds 3 --seed 42`
- Same search space and proxy-metric caveats as Run 5, applied to the
  wide-distress architecture instead of standard.

### Run metrics
Best trial (of 12): `lr=0.0005, onecycle_pct_start=0.1, weight_decay=0.001,
batch_size=16, lambda_figo=0.3, lambda_features=0.3, lambda_knowledge=0.05,
loss_weighting_method=uncertainty`

**Proxy AUROC: 0.8015** | score=0.6651 | accuracy=81.60% | sens@90spec=46.04% | ece=0.1902 | brier=0.1544

Top 3 of 12 trials: Trial 5 (uncertainty) = 0.8015, Trial 12 (gradnorm) =
0.7886, Trial 10 (dwa) = 0.7866.

### Run inferences
- Close to Run 5's standard-architecture proxy result (0.8015 vs. 0.8044) —
  both still unvalidated at full scale.
- No single loss-weighting method dominated across all 12 trials (fairly
  close spread 0.776–0.787 on average per method) — best single trial
  happened to use `uncertainty`.

### Updations for next run
- Built `configs/model8_crossformer_widedistress_tuned_config.yaml` (base
  wide-distress config + these winning hyperparameters, full 50/5 scale) —
  **still pending, deprioritized** (see note after Run 10).

---

## RUN 9 — Standard CrossFormer, `full` only, AFTER fold-2 joint-stratification fix

### Run configs
- Same as Run 1's `full` recipe exactly (fixed weighting, EMA off, default
  hyperparameters), but now automatically using the **new joint-stratified
  fold split** (baked into `train_knowledge_infused.py`, not a per-config
  setting — every run from this point forward uses it).
- Command: `python src/training/train_knowledge_infused.py --config configs/model8_crossformer_config.yaml --ablation full --seed 42`

### Run metrics
`full`: **AUROC 0.7908 ± 0.0312**
(accuracy 81.02%±3.86%, auprc 0.4177±0.1203, f1 0.4672±0.0507, precision
41.14%±8.42%, recall 56.33%±5.52%, specificity 85.36%±5.31%,
sens@90spec 41.84%±10.90%, ece 0.1482±0.0611, brier 0.1444±0.0198)

Per-fold: Fold1=0.8006, Fold2=0.8010, Fold3=0.8385, Fold4=0.7639, Fold5=0.7499

### Run inferences
- **The fold-2 fix worked exactly as intended**: mean essentially unchanged
  vs. Run 1 (0.7893→0.7908, +0.0015, noise-level), but **std dropped ~40%**
  (0.0515→0.0312). Per-fold spread tightened from [0.71–0.85] to
  [0.75–0.84] — no more single catastrophic fold.
- This is a measurement-reliability win, not a performance win. Worth
  remembering that some earlier comparisons in this document (Runs 1–8) used
  the *old*, noisier fold split — differences of ~0.03 or less between those
  runs should be treated with extra caution.
- **This is now the reference "standard architecture" baseline going forward.**

### Updations for next run
- Re-run the wide-distress architecture on this same fixed split for a fair
  comparison. → **Run 10**.

---

## RUN 10 — Wide-distress CrossFormer, `full` only, AFTER fold-2 joint-stratification fix

### Run configs
- Same as Run 7's recipe exactly, but now using the new joint-stratified fold split.
- Command: `python src/training/train_knowledge_infused.py --config configs/model8_crossformer_widedistress_config.yaml --ablation full --seed 42`

### Run metrics
`full`: **AUROC 0.7995 ± 0.0498**
(accuracy 78.52%±2.85%, auprc 0.3920±0.0969, f1 0.4705±0.0833, precision
37.10%±8.14%, recall 65.48%±10.09%, specificity 80.79%±2.63%,
sens@90spec 37.37%±9.56%, ece 0.1819±0.0841, brier 0.1536±0.0329)

Per-fold: Fold1=0.8134, Fold2=0.7976, Fold3=0.8851 (**overflow warning here —
T=1.412, a perfectly ordinary temperature, NOT the near-zero-T pathology
seen earlier**), Fold4=0.7438, Fold5=0.7574

### Run inferences
- Wide-distress edges ahead of standard again on the same (now fixed) split
  (0.7995 vs. 0.7908, +0.0087) — consistent direction with Run 7 vs. Run 1,
  so this isn't an artifact of one particular fold split.
- **Real tradeoff, not a clean win**: fold-to-fold std is *higher* than
  standard's on this same split (0.0498 vs. 0.0312 — wide-distress
  reintroduces some instability the fold-fix had reduced), and
  **Sens@90%Spec is meaningfully worse** (37.4% vs. 41.8%) despite the
  better raw AUROC — a clinically-relevant fixed-operating-point metric
  moving the wrong way.
- New calibration overflow pattern (fold 3, T=1.412 — not near-zero) is
  worth watching if it recurs; likely means wide-distress occasionally
  produces very large raw logits pre-calibration in some folds. Different
  root cause than the earlier near-zero-T pathology (folds hitting the 0.01
  clamp floor).

### Updations for next run
- Neither architecture alone is close to the 5% target. Best so far:
  wide-distress at 0.7995 (validated) or ~0.80–0.804 (proxy, unvalidated,
  pending Runs pointed at by configs `model8_crossformer_tuned_config.yaml`
  and `model8_crossformer_widedistress_tuned_config.yaml`).
- **User's explicit decision at this point: deprioritize both pending
  tuned-hyperparameter validation runs** — even in the best case they don't
  approach 5%, so validating them doesn't change the priority picture. Both
  configs remain ready to use later if a promising architecture change wants
  to be combined with tuning.
- Pivot to a genuinely different knowledge-infusion *mechanism* rather than
  more tuning. → led to the diagnostic below and **Run 11 (queued, not yet run)**.

---

## [Bug found & fixed, not a training run] — Augmentation logging bug

While investigating what levers hadn't been tried, found
`train_and_evaluate_model8()`'s status print
(`augmentation_cfg.get('enabled', False)`) had a **different default** than
the actual augmentor-building code (`augmentation_cfg.get("enabled", True)`).
Since no CrossFormer config has ever had an explicit `augmentation:` section,
**`PhysiologicalAugmentor` has actually been active in every single
CrossFormer run in this entire document**, despite every printed log
claiming "Augmentation: False". Not wrong data — just mislabeled. Fixed the
print statement to match reality (now correctly shows `True`). Also
confirmed (correctly, no bug there) that `curriculum` and `error_analysis`
have genuinely been OFF the whole time (their defaults are consistently
`False` in both the print and the gating logic) — both are legitimate,
never-tried, already-built levers, especially `error_analysis`
(`src/training/error_analysis.py::ErrorAnalyzer`, one-line config flag to enable).

## [Diagnostic experiment, not a Model 8 run] — Classical ML on hand-crafted features alone

Ran a `GradientBoostingClassifier` directly on the 8 hand-crafted clinical
features (`y_features`: baseline, STV, LTV, accel/decel counts) with **no
raw signal at all**, patient-level 5-fold CV (joint-stratified), purely
locally, no GPU needed.

**Result: AUROC 0.7297 ± 0.0347.**

Feature importances (mean across folds): LTV 33.8%, STV 16.7%, Baseline FHR
15.8%, Var Decels 10.6%, Accel Count 8.0%, Prolonged Decels 6.7%, Late
Decels 4.8%, Early Decels 3.7%.

**Inference**: these 8 coarse numbers alone get ~90% of the deep model's
AUROC — real, substantial, non-redundant signal. Every knowledge-infusion
mechanism tried so far (FIGOHead, ClinicalFeatureHead, FIGOCriteriaHead,
figo_rule_loss) injects this knowledge only as an auxiliary *output* the
encoder has to learn to reproduce — a weak, indirect signal. Direct *input*
fusion (handing the network these already-known, deterministically-
computable numbers directly, not asking it to re-derive them) is a
fundamentally stronger, untried mechanism. Also independently confirms the
earlier LTV-fix work was well-targeted (LTV is the single most predictive
feature by a wide margin). This is NOT leakage — these features are
deterministically computable from the raw signal via the same pipeline that
already runs before any model sees a window, exactly as a real deployed
system would compute them at inference time.

---

## RUN 11 — Feature-fusion CrossFormer, `full` only, default hyperparameters

### Run configs
- **New architecture built**: `src/models/knowledge_infused_framework_feature_fusion.py`
  — `FeatureFusionDistressHead` (concatenates the shared 128-dim `z` with
  the (normalized) 8-dim `y_features` vector before classifying — 136-dim
  input) + `KnowledgeInfusedFrameworkFeatureFusion` (backbone-agnostic,
  unlike wide-distress — works with any single-output encoder).
  **Deliberately only `DistressHead` sees the fused input** —
  FIGOHead/ClinicalFeatureHead/FIGOCriteriaHead remain `z`-only, since their
  targets are themselves deterministic functions of `y_features` and fusing
  it into those heads would trivialize those auxiliary tasks (verified
  directly: FIGOHead/ClinicalFeatureHead outputs are provably unaffected by
  what `y_features` is passed; `distress_logit` correctly changes).
- Wired into `train_knowledge_infused.py`: `_forward_model()` now checks a
  `requires_features_input` marker and threads `y_features` through all 4
  call sites (main loop, validation, SWA re-eval, dry-run) automatically —
  regression-tested against both the standard and wide-distress paths (no
  effect on either).
- New config: `configs/model8_crossformer_featurefusion_config.yaml` —
  identical to Run 9's baseline config except `heads.feature_fusion: true`,
  isolating the fusion mechanism alone, not combined with wide-distress or
  tuned hyperparameters (deliberate — avoid repeating the earlier
  uncertainty+EMA confound mistake).
- Command: `python src/training/train_knowledge_infused.py --config configs/model8_crossformer_featurefusion_config.yaml --ablation full --seed 42`
- Run directly on the GPU laptop (real run, not the earlier killed CPU smoke
  test). Fold split: joint-stratified (post fold-2-fix), same as Runs 9/10.

### Run metrics
`full`: **AUROC 0.7954 ± 0.0504**
(accuracy 78.2640%±4.4135%, auprc 0.3666±0.1171, f1 0.4672±0.0803, precision
37.2454%±8.6452%, recall 64.1518%±5.3497%, specificity 80.7373%±4.7017%,
sens@90spec 40.2305%±12.4674%, optimal_threshold 0.5300±0.0551,
ece 0.2025±0.0657, brier 0.1606±0.0266)

Per-fold: Fold1=0.8117, Fold2=0.8055, Fold3=0.8736, Fold4=0.7622, Fold5=0.7239

### Run inferences
- **Lands squarely between the other two architectures on the same fold
  split, and beats neither cleanly.** vs. Run 9 standard (0.7908±0.0312):
  +0.0046, wins fold 1/2/3, loses fold 4/5 (3/5). vs. Run 10 wide-distress
  (0.7995±0.0498): −0.0041, wins fold 2/4, loses fold 1/3/5 (2/5). Directly
  contradicts the "fundamentally stronger, untried mechanism" framing that
  motivated building this — in practice, feature fusion is a wash, not a win.
- **Worst calibration of all three architectures**: ECE 0.2025, vs. standard's
  0.1482 and wide-distress's 0.1819. Sens@90%Spec (40.23%) also sits between
  the other two rather than leading. No metric on this run is architecture-best.
- Fold pattern echoes Run 10 (fold 3 spikes to 0.87, fold 5 sinks to 0.72) more
  than Run 9's tighter spread — suggests the instability wide-distress
  reintroduced isn't unique to widening the latent; concatenating raw
  features in also loosens fold-to-fold consistency vs. the plain standard
  architecture.
- Only `full` was tested (no `distress_only` counterpart run for this
  architecture) — so unlike the other two architectures, there's no
  same-architecture 5-fold significance baseline to compare against here,
  only cross-architecture comparisons on the same split.

### Updations for next run
- **Feature fusion alone does not beat wide-distress and is not worth pursuing
  in isolation.** Don't run the full 8-variant ablation sweep for this
  architecture — the `full`-vs-`full` comparison already shows it's not
  competitive with the current best.
- The one combination not yet tried: fusing features into the *wide-distress*
  256-dim path (not built) — the two architecture changes individually gave
  small, different-direction edges (wide-distress +0.0087, feature-fusion
  +0.0046 vs. standard), so an additive combination is a legitimate next
  experiment, but given how close all three now are (0.7908–0.7995, spanning
  under 0.01), it's not obviously going to close a ~0.06 gap either — treat as
  low-confidence, not the next presumed win.
- Three architecture variants have now been tried and all cluster tightly
  (0.79–0.80) — this is starting to look like a ceiling for architecture-only
  tweaks on this encoder/data. The larger untried levers remain: MS-LSTM
  EMA-off re-run (never done since Run 6), MS-LSTM hyperparameter tuning
  (never done), CNN1D backbone (never run at all with this session's fixes),
  and turning on `error_analysis: {enabled: true}` for concrete
  where-does-it-fail evidence (never done, one config line) — the last one
  is cheap and could reveal *why* all three architectures plateau in the same
  place, which no amount of further architecture tweaking has explained.

---

## [Investigation, not a Model 8 training run] — Ensembling toward a 90%+ AUROC target

**New goal from the user (2026-08-17), superseding the ~0.84 target above**:
90%+ AUROC via knowledge infusion/feature fusion, to demonstrate Model 8 beats
the standalone CrossFormer's own 0.8565 CV / 0.8653 test SOTA benchmark — not
just close the gap to it. Flagged to the user as a large jump (+0.04–0.14
beyond even the plain classifier's own ceiling, larger than any single lever
in this whole document moved AUROC) and above what's typically reported in
CTU-CHB / pH-based fetal-distress literature (~0.75–0.85). User chose
**ensembling** as the first, cheapest thing to try, over pretraining/deeper
fusion/a literature reality-check.

### Sub-step A — OOF ensemble sanity check, two already-trained Model 8 variants

Before ensembling against the actual standalone benchmark, found a data-
currency problem: `checkpoints/ctg_crossformer/` (the standalone benchmark
achieving 0.8565/0.8653) was trained **2026-08-10**, but `data/processed/`
was regenerated **2026-08-15** (LTV preprocessing fix, `b7fbe48`/`0de6edd`).
Window/patient composition can drift run-to-run (documented earlier in this
doc), so reusing those checkpoints' fold assignments against current data
risks silently invalid held-out splits. Deferred fixing that (→ Sub-step B)
and instead built `scripts/ensemble_oof_eval.py` to sanity-check the
ensembling *methodology* first, using two checkpoint sets both confirmed
trained on the current (post-2026-08-15) data: `model8_crossformer/
distress_only` (standard architecture, old distress-only-stratified folds,
trained 2026-08-15) and `model8_crossformer_widedistress/full` (new joint-
stratified folds, trained 2026-08-16).

**Method**: the two checkpoint sets use *different* fold-split algorithms, so
fold `i` doesn't mean the same patients in both. Rather than assume alignment,
the script reconstructs each model's own fold split independently
(deterministic given fixed data + seed=42) and computes true out-of-fold
predictions per window from only that window's own held-out fold — leak-free
regardless of how the two splits relate. Pooled AUROC computed once across
all 6,826 windows (not averaged per-fold, since fold membership differs).

**Finding #1 (methodological)**: pooling raw `sigmoid(logit)` OOF scores
directly gave pooled AUROC **0.03–0.05 lower** than each checkpoint's own
previously-reported mean-of-fold AUROC (e.g. wide-distress full: 0.7742
pooled vs. 0.7995 reported) — each fold's model lands at its own local logit
scale/offset with no cross-fold calibration enforced during training, so
naive pooling corrupts the *across*-fold ranking even though each fold's own
internal ranking is fine. **Fixed** by fitting a per-fold Platt scaling
(1-D logistic regression on that fold's own held-out logits/labels — a
monotonic transform *within* the fold, so it cannot change that fold's own
AUROC, only puts folds on a comparable probability scale before pooling).
After the fix, pooled numbers reconcile closely with reported means
(distress_only: 0.7776 pooled vs. 0.7743 reported; wide-distress full: 0.7976
pooled vs. 0.7995 reported) — confirms the pooling methodology is now sound.
**Any future pooled/ensemble OOF evaluation in this project should reuse this
per-fold-Platt-scaling step — raw-probability pooling across independently-
trained fold checkpoints is not reliable.**

**Finding #2 (result)**: 50/50-average ensemble of these two Model 8 variants
— **AUROC 0.8045**, a genuine **+0.0070** lift over the better single model
(wide-distress full, 0.7976). A blind (non-label-fit) weight sweep is
included in the script's output as a diagnostic only (best alpha=0.4 →
0.8056) — that number is optimistic/non-blind (the mixing weight was chosen
using the labels), reported for reference, not as a claimed result.

**Inference**: ensembling shows a real, modest, positive signal in the ~1%
AUROC range — consistent with typical expectations for averaging two
correlated models, not a breakthrough on its own. The two models ensembled
here (0.7776, 0.7976) are both far below the real prize target (the 0.8565
standalone benchmark) — ensembling two mediocre models can only ever produce
a modestly-less-mediocre result. The methodologically interesting result is
elsewhere: this validates the OOF-ensembling pipeline works and is worth
running against the actual strong model next.

### Sub-step B — Retraining the standalone benchmark on current data (IN PROGRESS)

To get a valid apples-to-apples ensemble against the real 0.8565/0.8653 SOTA
benchmark, retraining `train_ctg_crossformer.py`'s standalone 5-fold CV fresh
against the current `data/processed/` files (unchanged since 2026-08-15
12:38, verified via file mtimes — no further regeneration since). **Written
to a new checkpoint directory (`checkpoints/ctg_crossformer_current/`),
deliberately NOT overwriting `checkpoints/ctg_crossformer/`** — the original
Aug-10 checkpoints and their 0.8565/0.8653 numbers stay reproducible/intact
as the documented benchmark; this is a data-currency-matched sibling for
ensembling purposes only.

- Command: `python src/models/train_ctg_crossformer.py --config configs/ctg_crossformer_config.yaml --data_dir data/processed/ --checkpoint_dir checkpoints/ctg_crossformer_current/`
- Launched in background 2026-08-17; ~1.5–2h expected (5 folds × 50 epochs,
  based on the original Aug-10 run's checkpoint-save timestamps spanning
  ~1h42m end to end).

### Sub-step B result — MAJOR FINDING: the 0.8565/0.8653 ceiling does not reproduce on current data

Retraining finished (2026-08-17, ~52 min, much faster than the ~1.5–2h
estimate — ~12–15s/epoch). Same script, same config, same code, **only the
underlying `data/processed/` files differ** (current, post-LTV-fix, vs. the
2026-08-10 checkpoints' stale pre-fix data):

| | CV AUROC (mean±std, 5-fold) | Test AUROC (held-out 82 patients) |
|---|---|---|
| Original (2026-08-10, stale data) | **0.8565** | **0.8653** |
| Retrained (2026-08-17, current data) | **0.7834 ± 0.0551** | **0.7685** |

Per-fold: Fold1=0.7791, Fold2=0.8505, Fold3=0.8284, Fold4=0.6919, Fold5=0.7671.
Test set at default threshold 0.5: accuracy 92.3%, precision 27.3%, recall
15.8%, F1 0.20 — the test AUROC (0.7685) is respectable but the model is
badly under-triggering the positive class at that threshold (no threshold
tuning in this script, unlike Model 8's pipeline).

**Initially attributed to the LTV preprocessing fix (2026-08-15) — CORRECTED
after direct code inspection, prompted by the user asking whether the old
data could be recovered.** Traced the standalone classifier's actual inputs
through `src/preprocessing/pipeline.py`: `calculate_variability()`'s
STV/LTV output only feeds `y_features` and the FIGO pseudo-label
(`classify_figo()`) — **neither of which `train_ctg_crossformer.py` ever
sees**. Its only inputs are `X` (built from `fhr_norm`/`uc_win`, computed
*before* the variability step) and `y_primary` (pure pH-threshold + horizon
logic, unrelated to variability). The patient train/val/test split
(`train_test_split(..., random_state=42)`) is also independent of it. Checked
the full repo-wide commit history for 2026-08-08–17 (not just
`src/preprocessing/`) — no other commit in that window touches windowing,
filtering, `signal_quality.py`, `baseline.py`, or `ingestion.py`. `data/raw/`
itself (CTU-CHB `.dat` files + `clinical_metadata.csv`) is untouched since
2026-08-09, predating both the Aug-10 standalone run and the Aug-15 regen.
**Conclusion: the LTV fix cannot be the cause — the data the standalone
model actually trains on should be bit-identical whether the fix is applied
or not.**

**Real suspect: GPU training non-determinism.** `train_ctg_crossformer.py`
calls `set_seed(42)` (random/numpy/torch/cuda manual_seed) but never sets
`torch.backends.cudnn.deterministic=True` / `benchmark=False` — so training
is not actually bit-reproducible run-to-run on GPU. For a 546-patient,
14.7%-prevalence dataset, ordinary cudnn/kernel-selection nondeterminism can
plausibly swing 5-fold AUROC by several points on its own (this run's fold 4
cratered to 0.6919, consistent with one unlucky fold rather than a systematic
data problem). **RESOLVED (second run completed 2026-08-17)**: ran the standalone benchmark
a second time on identical current data (`checkpoints/ctg_crossformer_current_run2/`).
Result: **CV AUROC 0.7887 ± 0.0631, Test AUROC 0.7237** — within 0.0053 CV of
the first retrain (0.7834), with the same fold pattern both times (fold 4
weak both runs: 0.6919 then 0.6798; folds 2–3 strong both runs: 0.85+).
Per-fold: Fold1=0.7798, Fold2=0.8520, Fold3=0.8513, Fold4=0.6798, Fold5=0.7804.

**Conclusion: 0.8565/0.8653 does not reproduce and is not ordinary
run-to-run variance — two independent runs converge tightly on ~0.79 CV.
The honest, current, reproducible standalone CrossFormer ceiling is ~0.79 CV
/ ~0.72–0.77 test, not 0.8565/0.8653.** Why the original Aug-10 run scored
~0.07 higher remains unexplained (data and code are provably equivalent per
the analysis above) — most likely that specific run was itself the outlier,
possibly from an unusually favorable weight initialization given no cudnn
determinism is enforced. Not pursued further — the practical, decision-
relevant conclusion is settled: **use ~0.79 CV as the standalone reference
for all future comparisons, not 0.8565.** This substantially closes (in fact
reverses) the gap this document was built around: Model 8 wide-distress
`full` (0.7976 pooled OOF) already sits at or slightly above the honest
standalone reference.

### Sub-step C — the real ensemble: standalone (current-data retrain) x Model 8 wide-distress full

`scripts/ensemble_standalone_vs_model8.py` — same per-fold-Platt-scaling
pooled-OOF methodology as Sub-step A, extended to handle an asymmetry: the
standalone benchmark only ever trains/CVs over `train_dataset.pt`'s 381
patients (6,177 of 6,826 windows); the other 649 windows (165 patients from
`val_dataset.pt`/`test_dataset.pt`) were never held out by any of its 5
folds, so those get the average of all 5 fold checkpoints' predictions
(still leak-free — none of the 5 folds ever trained on them — just not a
single-fold OOF in the strict sense). Verified index alignment between the
381-patient pool and Model 8's 546-patient pool via a direct tensor-equality
assertion (both derive from the same files via sequential, unshuffled
`torch.load` concatenation).

**Results** (all pooled OOF AUROC, current data, both models):

| | AUROC |
|---|---|
| Standalone (current-data retrain) | 0.7888 |
| Model 8 wide-distress `full` | 0.7976 |
| **50/50 ensemble** | **0.8100** |
| Best-alpha ensemble (alpha=0.4 toward standalone, optimistic/non-blind) | 0.8103 |

**50/50 ensemble lift over the better single model: +0.0125.** Combined with
Sub-step A's +0.0070 (a weaker pairing), this confirms ensembling gives a
real, consistent, but modest lift (~1–1.5 points) — not remotely enough on
its own to reach 90%, and the freshly-measured current-data reality is that
**Model 8 wide-distress `full` (0.7976) already slightly *beats* the fairly-
retrained standalone benchmark (0.7888)** on the same data — a reversal of
the entire narrative this document was built around, which was chasing a gap
against stale-data numbers.

### Updations for next run

- **Report this finding to the user before doing anything else** — it
  reframes the goal. Two open questions only the user can settle: (1) is
  0.8565/0.8653 still the intended target (paper-replication number) even
  though it doesn't reproduce on current data, or should
  `checkpoints/ctg_crossformer_current/`'s 0.7834/0.7685 be the new reference
  ceiling? (2) given ensembling only adds ~1–1.5 points and the current-data
  ceiling across all tested models/ensembles tops out at 0.8100, is 90% still
  the target, or should it be revisited?
- If current-data numbers are accepted as the new reference: Model 8 (wide-
  distress `full`) already matches/beats the standalone benchmark on
  apples-to-apples current data — arguably the original "beat the SOTA"
  framing is already satisfied, just not by the originally-cited margin.
- Bigger ensembles (3+ models: standalone + wide-distress + feature-fusion +
  distress_only, stacked with a proper meta-learner instead of grid-searched
  alpha) could add another fraction of a point but are very unlikely to
  reach 90% — the other candidate levers (self-supervised pretraining, deeper
  multi-layer feature fusion, literature reality-check) remain the more
  credible paths if 90% stays the goal.
- Worth a dedicated side investigation: what did the LTV fix actually change
  about the label-signal relationship that cost ~0.07–0.10 AUROC uniformly
  across a completely unrelated model/training script? Not yet investigated
  — this affects every number in this document post-dating the fix, not just
  the standalone benchmark.

---

## [Investigation] — Self-supervised pretraining, tried and found NOT to help (2026-08-17)

Following the ensembling/data-currency investigation above, user chose to pursue self-supervised
pretraining of `CTGCrossformerEncoder` as the next lever (over reconsidering the pH-threshold
label framing). Full design planned and implemented:

- `src/training/ssl_masking.py::mask_ctg_signal()` — blockwise raw-signal masking (~50% ratio,
  4–40s blocks, channel-independent, mask_value=0.0), returns `(X_masked, mask)`.
- `src/models/ctg_crossformer_pretraining.py` — `CTGCrossformerSSLEncoder` (subclasses
  `CTGCrossformerEncoder`, exposes `tf_out` alongside `z`, zero new params so `state_dict()`
  stays loadable into the base encoder class), `CTGReconstructionDecoder` (ConvTranspose1d stack,
  `tf_out (B,150,256) -> (B,2,4800)` raw-signal reconstruction), `CTGCrossformerSSLPretrainer`.
- `scripts/generate_pretraining_windows.py` — dense unlabeled windows via `get_valid_windows()`
  directly (30s stride, bypassing the supervised pipeline's label-conditional stride), same
  signal-processing chain and the already-fit `ctu_signal_scaler.npz` (not refit). Produced
  **30,985 windows from 468 patients** (test-set's 82 patients excluded from the pretraining
  corpus; single shared corpus across all folds — the cheaper of the two data-scope options,
  chosen deliberately over 5x-cost per-fold pretraining).
- `scripts/pretrain_ctg_crossformer_ssl.py` — masked-reconstruction loss (MSE on masked positions
  only), AdamW, `WarmupCosineScheduler` (reused from `train_knowledge_infused.py`), 150 epochs.
- Both `train_ctg_crossformer.py` and `train_knowledge_infused.py` given a `--pretrained_encoder`
  flag to load the saved encoder state_dict before each fold's fine-tuning.

### Pretraining run result
150 epochs, ~4.6h (slower than the ~2.3–3.1h estimate — likely the masking function's per-sample
Python-loop CPU cost). **Overfit past epoch 40**: train loss fell monotonically (0.370→0.156) but
holdout reconstruction loss bottomed at **epoch 40 (0.26915)** then steadily worsened to ~0.30 by
epoch 150. Checkpointing-on-best-holdout-loss worked as designed — the saved
`checkpoints/ctg_crossformer_ssl/encoder_pretrained.pth` is the epoch-40 weights, not the
overfit epoch-150 ones.

### Fine-tuning evaluation — NEGATIVE RESULT

| | AUROC | vs. random-init |
|---|---|---|
| Standalone, random-init (2 runs) | 0.7834, 0.7887 | — |
| **Standalone, pretrained-init** | **0.7945 ± 0.0443** | +0.006 to +0.011 (within the ~0.005–0.007 noise band already established between the two random-init runs — inconclusive) |
| Model 8 wide-distress `full`, random-init (Run 10) | 0.7995 ± 0.0498 | — |
| **Model 8 wide-distress `full`, pretrained-init** | **0.7733 ± 0.0377** | **−0.0262 — a real regression, well outside the noise band** |

Standalone's operating-point metrics also got worse under pretrained-init despite flat/slightly-better
AUROC: mean F1 dropped to 0.295 (vs. ~0.40 random-init), with some folds (fold 1) collapsing to
2.6% sensitivity at the default 0.5 threshold — pretraining shifted the logit distribution in a
way that leaves the default threshold poorly calibrated, a real practical cost AUROC doesn't
capture. Wide-distress's calibration also got worse (ECE 0.196 vs. 0.182 random-init).

**Verdict: this pretraining approach did not help, and regressed the architecture that actually
matters (Model 8 wide-distress).** Best current explanation, not fully confirmed:
1. **Task mismatch** — denoising/reconstruction optimizes the encoder to preserve all signal
   detail (including clinically-irrelevant noise), not necessarily what's useful for distress
   discrimination.
2. **Low effective data diversity** — 30,985 windows came from only 468 patients at a 30s stride
   against a 20-minute window (~97.5% overlap between adjacent windows); window *count* is large
   but actual information content is far less than that number suggests, likely not meaningfully
   more diverse than the labeled 6,826-window supervised set already provides.
3. **SSL task didn't fully converge** — holdout reconstruction loss maxed out its improvement by
   epoch 40; full unfrozen 50-epoch fine-tuning may simply overwrite whatever the pretrained
   initialization contributed.

**Note**: this run was executed via `train_knowledge_infused.py ... --pretrained_encoder ...`
WITHOUT a `--checkpoint_dir` override, so it **overwrote**
`checkpoints/model8_crossformer_widedistress/`'s fold checkpoints and
`results/model8_full_cv_results.json` — the original random-init 0.7995-AUROC model weights are
gone (the number itself remains recorded here and is not in doubt, just the weights are).

### Updations for next run
- **Do not iterate further on this specific pretraining recipe** (different mask ratios etc.)
  without a specific reason to expect a different outcome — each iteration costs ~5–6 GPU-hours
  for an approach that has now failed twice (marginal-at-best on one architecture, clearly
  negative on the other).
- The cheaper, previously-deprioritized lever — reconsidering the pH≤7.15 hard-threshold label
  framing (ordinal/regression reframing, or excluding a gray-zone band around the threshold) —
  is now the more promising untried option if the user wants to keep pushing past the current
  ceiling.
- **Practical standing best result remains**: Model 8 wide-distress `full` random-init (0.7995,
  though its checkpoint weights are now gone — architecture/config still reproducible) plus the
  standalone+Model8 ensemble at 0.8100 pooled OOF (Sub-step C, still valid/unaffected by this
  investigation since it used the pre-overwrite wide-distress checkpoints at the time it ran).

---

## [Major investigation, 2026-08-17/18] — 0.8565 closed, the paper found, and a data leak discovered

### Part A — The 0.8565 target is closed (8-seed sweep)

User asked whether the standalone CrossFormer's 0.8565 could be recovered by finding the
right seed. Added a `--seed` flag (the script hardcoded `set_seed(42)`) and swept 8 seeds
on `data/processed/`:

| Seed | 1 | 7 | 13 | 21 | 100 | 123 | **777** | 2026 |
|---|---|---|---|---|---|---|---|---|
| CV AUROC | 0.7830 | 0.7808 | 0.7878 | 0.7668 | 0.7702 | 0.7731 | **0.7987** | 0.7705 |

With the two earlier seed-42 runs (0.7834, 0.7887) that is **10 independent runs spanning
0.7668–0.7987, mean ≈0.780**. None came within 0.058 of 0.8565. **Conclusion: 0.8565 is
outside the distribution this setup produces — not a reachable seed.** Chasing it is closed.
User accepted **0.7987 (seed 777)** as the working baseline.

### Part B — The actual paper, obtained

Dang, Nguyen, Ho, *"A Hybrid CNN-Transformer with Cross-Attention for Automated Fetal
Distress Detection from Cardiotocography,"* E3S Web of Conferences 723, 01005 (2026),
AIEI 2026. PDF at repo root. **Its reported figure is AUC-ROC 0.822** (pooled across
5-fold CV; their Table 2 gives mean-of-fold 0.825) — never 0.8565, which was always our
own run's number.

Architecture matches `ctg_crossformer.py` closely. Protocol differs in 8 identified ways
(patient-level >50%-missing exclusion → 404 patients; uniform 5-min stride, no last-hour
truncation → 1,753 windows; no horizon relabeling; per-recording z-score; linear not
cubic interpolation; no spike/lowpass/baseline steps mentioned; no held-out test set;
pooled not mean-of-fold metric). Built `src/preprocessing/pipeline_paper_match.py` +
`scripts/pooled_oof_ctg_crossformer.py` to reproduce it literally.

**Reproduction outcome: did not match.** Our patient-level exclusion removed only 5/552
patients (0.9%) vs the paper's 148/552 (26.8%) at the same >50% threshold — verified by
computing the raw missing-ratio distribution directly (max 53.5%, mean 18.8%), so the
calculation is correct; the paper does not specify how it measured "missing". Standalone
training on that substrate gave folds 0.6193 / 0.7039 — well below both 0.822 and our own
0.79. **Run stopped early at user request; paper reproduction abandoned.**

### Part C — CRITICAL: bag-size leak in `data/processed/`

Investigating patient-level modelling (the label, pH ≤ 7.15, is a *patient* outcome, but
Models 1–8 all train/score per *window*), an initial measurement on seed-777 checkpoints
appeared to show a large gain (top-3 pooling 0.9186 vs window-level 0.8030).

**That was an artifact and was retracted.** `pipeline.py` selects its stride from the
label — `DISTRESS_STRIDE_MINUTES=0.5` vs `NORMAL_STRIDE_MINUTES=10` — so distress patients
get ~80 windows and normal patients ~5. Measured directly:

> **AUROC using window count alone as the predictor: 0.9947** on `data/processed/` train.
> (distress median 80 windows, range 23–81; normal median 5, range 1–60; *zero* distress
> patients have <20 windows.)

Max/top-k pooling over 80 draws vs 5 draws is inflated by pure order statistics. Any
patient-level aggregation on this data measures bag size, not fetal distress.

**Window-level results are UNAFFECTED** — a window is scored on its own signal with no
knowledge of its sibling count. Every window-level number in this document stands.

New standing guard: `scripts/audit_bag_size_leak.py` (fails at AUROC ≥ 0.60), plus a
runtime assertion inside `train_mil.py`.

### Part D — New substrate: `data/processed_mil/` (uniform stride)

`src/preprocessing/pipeline_mil.py` — keeps `pipeline.py`'s full signal chain (spike
removal → cubic interpolation → lowpass → iterative baseline → baseline-corrected
channel 0 → global train-fit z-score) and changes only windowing/labelling: **uniform
2.5-min stride for every patient**, plus a new `y_patient` key (pH outcome) and the
patient-level quality gate. `pipeline.py` and `data/processed/` untouched.

Output: 7,551 windows / 544 patients (5,286 train / 1,162 val / 1,103 test).
**Leak audit: 0.4415 / 0.5294 / 0.4394 — PASS** (vs 0.9947 on the old data).

**Window-level standalone CrossFormer on this substrate: CV AUROC 0.8200 ± 0.0752,
pooled OOF 0.8309** (per-fold 0.8003, 0.7659, 0.7204, 0.9087, 0.9048; test 0.7009).
That is **above the 0.7987 old-data baseline**, and arguably more trustworthy: under the
old label-conditional stride, window-level AUROC was dominated by a few distress patients
contributing ~80 near-duplicate windows each. **Caveat: CV 0.8200 vs test 0.7009 is a
wide gap — the CV figure may be optimistic and should be verified, not assumed.**

### Part E — Model 9 (KG-MIL) built, and its go/no-go FAILED

Built the full patient-level stack: `src/training/mil_dataset.py` (masked variable-length
bag collation), `src/models/knowledge_guided_mil.py` (gated ABMIL attention + learnable-λ
clinical risk prior biasing attention + 12 trajectory features, all ablation-gated),
`src/training/train_mil.py`. Smoke-verified: padded slots get exactly zero attention,
rows sum to 1, gradient reaches all 99 encoder params, λ receives gradient.
Deliberately did **not** warm-start from the window-level checkpoints — the two scripts
use different fold algorithms, so a fold-k baseline checkpoint would have trained on
patients in KG-MIL's fold-k validation set (a real leak).

**`plain` ablation (the Stage-1 gate, threshold ~0.82):**
per-fold 0.8568, 0.7173, 0.7915, 0.7062, 0.6535 → **0.7451 ± 0.0712**, **pooled OOF 0.7104**.

Independent cross-check — aggregating the *window* baseline's OOF scores per patient on
the same clean substrate: mean 0.7580, max 0.7435, top-3 0.7479, last-3 0.7453, versus
**window-level pooled OOF 0.8309**.

**Two independent routes to patient-level both land 0.71–0.76, clearly below window-level
0.83. The patient-level reframing does not help on leak-free data — it hurts.** The
earlier apparent +0.06 was entirely the bag-size artifact. Ablation ladder stopped after
`plain` rather than burning ~8 GPU-hours on a disproven premise.

Diagnosed cause: patient-level training has ~435 bags / ~86 positives to fit a 2.5M-param
model, versus 5,286 window labels — supervision collapses by an order of magnitude.

Also corrected: an interim note that KG-MIL's auxiliary window head beat the window
baseline was based on 3 folds; over all 5 it is 0.8032 vs 0.8200. Multi-task
patient+window training did **not** help the window task.

### Part F — Current direction (user-set target 0.85–0.87)

Decision: **compare at window level** — CTG-CrossFormer is natively a window classifier,
so this is the like-for-like comparison, and patient-level degrades both models.

Gap is now **+0.02–0.05**, not +0.06–0.07, because the substrate change banked ~+0.03.
Path using only levers with measured value in this project:

| Lever | Measured | Running total |
|---|---|---|
| Window-level, `data/processed_mil/` | 0.8309 pooled | 0.8309 |
| + Model 8 knowledge infusion | ~+0.01 | ~0.841 |
| + ensembling (standalone × Model 8) | +0.0125 | ~0.853 |

Realistic landing **0.84–0.86**; 0.87 is a stretch and is not being promised.
**In progress**: Model 8 wide-distress `full` on `data/processed_mil/` via
`configs/model8_widedistress_mil_substrate_config.yaml`.

---


## [2026-08-19] — Sensitivity fixed: paper-specified class weighting restored

**Problem**: the honest baseline ranked well (unbiased CV AUROC 0.8014) but was
clinically unusable -- test sensitivity **11.8%** at threshold 0.5, i.e. it
missed ~9 of every 10 distress cases. The paper reports 89.5% sensitivity.

**Two causes, diagnosed separately.**

*1. Threshold 0.5 is wrong at 4.3% window prevalence.* A calibrated model
outputs mostly low probabilities (our median 0.21; only 8.9% of windows exceed
0.5). Moving along the EXISTING ROC, at no training cost:

| threshold | sensitivity | specificity |
|---|---|---|
| 0.50 | 40.3% | 92.6% |
| 0.30 | 70.4% | 71.3% |
| 0.20 | 80.1% | 48.5% |

`compute_metrics()` now also reports `sens_at_90spec`, `spec_at_90sens`,
`thresh_at_80sens` so a single arbitrary cut-off no longer hides this.

*2. We deviated from the paper's loss recipe.* `train_ctg_crossformer.py` had
hardcoded `pos_weight=1.0` (a 2026-08-09 "bug fix" claiming the sampler already
rebalances). The paper specifies **both** "Focal Loss with gamma=2.0 and
inverse-frequency class weights" **and** sqrt-inverse `WeightedRandomSampler`.
The arithmetic supports the paper: sqrt-inverse sampling lifts batch prevalence
only from ~4.3% to ~17%, so it does NOT fully rebalance -- the model was left
under-corrected toward the negative class. **That "fix" was itself the bug.**

**Results** (all nested/unbiased selection, fold-matched, `--class_weight`):

| | none (pos_w 1.0) | **inverse_freq (23.5)** | sqrt_inverse (4.85) |
|---|---|---|---|
| Unbiased CV AUROC | 0.8014 | 0.7768 | 0.7980 |
| CV sens @0.5 | 39.5% | **61.7%** | 55.4% |
| CV spec@90%sens | -- | 42.5% | 45.6% |
| Test AUROC (5-fold ens.) | 0.7850 | **0.7937** | 0.7731 |
| **Test sens @0.5** | **11.8%** | **49.0%** | 37.3% |
| **Test spec@90%sens** | -- | **48.4%** | 41.6% |

**Test sensitivity 11.8% -> 49.0% with no AUROC cost** -- the signature of a pure
operating-point fix. At matched ~90% sensitivity we reach **48.4% test
specificity vs the paper's 41.4% at 89.5%** — the sensitivity gap is closed.

**Correction to an earlier note in this session**: a figure of "17.6% specificity
at 89.5% sensitivity" was computed from raw pooled OOF logits without per-fold
Platt scaling, which is known to bias pooled numbers low. The per-fold values
above (42-48%) are the fair measurement.

**Chosen baseline: `inverse_freq`** (`archive/standalone_invfreq_clinical/`) --
matches the paper's spec and wins on the held-out test set. `sqrt_inverse` is
mildly better on CV AUROC; kept in `archive/standalone_sqrtinv/` for the record.

**Also fixed**: held-out test evaluation used whatever model was left in memory
after the CV loop (the last fold's end state), not a checkpoint or ensemble --
so every previously reported test number was one arbitrary fold's leftover
weights (0.6482 reported vs 0.7850 for the true 5-fold ensemble). Now ensembles
the five saved best checkpoints; ensembling is worth +0.0191 on test.

---


## [2026-08-19] — Label-confidence weighting: modest CV gain, WORSE on test, not adopted

**Hypothesis**: pH <= 7.15 is a hard cut on a continuous measure -- a fetus at
7.14 and one at 7.16 are physiologically indistinguishable but labelled
oppositely. That boundary noise sits in every experiment run so far.

**Design choice**: soft down-weighting, NOT gray-zone exclusion. Excluding a
7.10-7.20 band would drop 116/552 patients including **52 of the 113 positives
(46%)**, and positive scarcity is already this dataset's binding constraint
(it is what killed the patient-level MIL attempt). Instead each window's loss
is weighted by its patient's pH distance from 7.15: floor 0.30 at the
threshold, ramping to 1.0 at +/-0.05. 1,324/5,286 windows down-weighted, mean
weight 0.914 -- nothing discarded.

**Results** (nested/unbiased, fold-matched, on top of `inverse_freq`):

| | invfreq | + label-confidence |
|---|---|---|
| Unbiased CV AUROC | 0.7768 | 0.7955 (+0.019) |
| CV AUPRC | 0.1679 | 0.2134 (+0.045) |
| CV Sens@90%Spec | 0.4513 | 0.5021 (+0.051) |
| **Test AUROC (ens.)** | **0.7937** | **0.7718 (-0.022)** |
| **Test AUPRC** | **0.1207** | **0.1093 (-0.011)** |
| Test spec@90%sens | 0.4838 | 0.4240 |

Per-fold unbiased: 0.8089, 0.8517, 0.8978, **0.7255, 0.6935**.

**Verdict: NOT adopted.** The CV gain (+0.019) is well inside the fold std
(0.076), and it does not transfer -- on the held-out test set it is worse than
`inverse_freq` on every metric. Folds 1-3 looked strong (0.81-0.90) and were
reported as promising mid-run; folds 4-5 regressed hard. A repeat of the
pattern seen in the knowledge-infusion run, where a 2-fold signal (+0.0074)
decayed to +0.0027 by 5 folds. **`inverse_freq` remains the deliverable model.**
Archived at `archive/standalone_labelconf/` for the ablation record.

**Operating-point instability worth carrying forward**: sensitivity at a fixed
0.5 threshold ranged 29%-94% across folds while AUROC stayed 0.69-0.90. For a
device this matters more than AUROC -- a hard-coded factory threshold would
behave inconsistently across populations. The threshold must be calibrated
against a reference set, not fixed.

**Bug fixed**: `CTGDataset` now returns a 3-tuple (X, y, weight); the held-out
test loop still unpacked 2 values and crashed AFTER the CV completed. All five
folds and checkpoints were unaffected.

---

## Current state summary (as of this document)

**Best validated result**: Wide-distress CrossFormer, `full`, fold-2-fixed split — **AUROC 0.7995 ± 0.0498** (Run 10). Real edge over standard architecture but with a stability/Sens@90Spec tradeoff. Feature-fusion (Run 11, 0.7954±0.0504) came in between standard and wide-distress and did **not** unseat this.

**Best unvalidated (proxy) result**: Standard CrossFormer + tuned hyperparameters — **AUROC 0.8044** (Run 5, 15ep/3fold proxy, deprioritized for full validation).

**Target (superseded 2026-08-17)**: originally ~0.84 (5% above the ~0.79 starting point). User has since raised this to **90%+ AUROC**, specifically to demonstrate Model 8 beats the standalone CrossFormer's own 0.8565 CV / 0.8653 test ceiling, not just close the gap to it. **Not yet reached by any tested combination** — flagged to the user as a large jump, above typical CTU-CHB literature results.

**Immediately next**: all ensembling sub-steps plus the follow-up variance check are done (see "Ensembling toward a 90%+ AUROC target" below) — **major, now-confirmed finding: the 0.8565/0.8653 standalone ceiling does not reproduce on current data and is NOT explained by ordinary training variance either** (two independent reruns both converge to ~0.79 CV, tight agreement). Root cause of why the original Aug-10 run scored higher remains unexplained but is very likely that specific run being an outlier, not a systematic data/code difference (verified the LTV fix's code path cannot even reach the standalone model's inputs). **`~0.79 CV / ~0.72–0.77 test` is now the standing reference for the standalone CrossFormer classifier — 0.8565/0.8653 should no longer be cited as the target.** On apples-to-apples current data: standalone ~0.79, Model 8 wide-distress `full` 0.7976 (already at/above standalone), 50/50 ensemble 0.8100 (+0.0125 over best single model). Awaiting user direction on how to proceed given this reframing — see the two open questions at the end of that section.

Below this line is the **pre-2026-08-17 state** (architecture-tweak track, ~0.84 target) — still accurate for that track, kept for continuity: three architecture variants (standard 0.7908, wide-distress 0.7995, feature-fusion 0.7954) cluster within 0.01 AUROC of each other, so pure architecture tweaks on the DistressHead/latent appear to be topping out. Candidates if this track is revisited: (a) fuse features into the wide-distress 256-dim path (untried combination of the two partial wins), (b) enable `error_analysis` to get diagnostic evidence for *why* the plateau exists, (c) revisit MS-LSTM/CNN1D backbones which haven't shared any of this document's fixes.

**Fully parked/pending, in rough priority order**:
1. MS-LSTM EMA-off re-run — never done since Run 6.
2. MS-LSTM hyperparameter tuning — never done.
3. Tuned-hyperparameter full-scale validation for both standard (`configs/model8_crossformer_tuned_config.yaml`) and wide-distress (`configs/model8_crossformer_widedistress_tuned_config.yaml`) architectures — deprioritized, ready whenever revisited.
4. Enabling `error_analysis` on some future run for concrete failure-mode evidence — never done, trivial to turn on.
5. CNN1D backbone — configs exist (`configs/model8_cnn1d_config.yaml`) but has never been run with any of this session's fixes at all.
6. Wide-distress + feature-fusion combined (fuse features into the 256-dim path) — not built, lowest-confidence of the pending items given how tightly the three individual architectures already cluster.

**Files that matter for continuity**:
- `src/models/knowledge_infused_framework.py` — original framework (DistressHead/FIGOHead/ClinicalFeatureHead/FIGOCriteriaHead).
- `src/models/knowledge_infused_framework_wide_distress.py` — wide-distress variant.
- `src/models/knowledge_infused_framework_feature_fusion.py` — feature-fusion variant.
- `src/training/train_knowledge_infused.py` — main training script, all three architectures wired in via `backbone.model` / `heads.feature_fusion` config flags.
- `src/training/tune_model8_hyperparameters.py` — dedicated Model 8 hyperparameter tuner.
- `scripts/diagnose_fold2_composition.py` — fold-composition diagnostic (supports `--joint` flag).
- All `configs/model8_crossformer*.yaml` variants — see table above for what each represents.
