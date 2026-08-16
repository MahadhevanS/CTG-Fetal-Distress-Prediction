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

## RUN 11 — Feature-fusion CrossFormer, `full` only, default hyperparameters (QUEUED, NOT YET RUN)

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
- Command (to run): `python src/training/train_knowledge_infused.py --config configs/model8_crossformer_featurefusion_config.yaml --ablation full --seed 42`
- **Status: a local CPU smoke test (1 epoch, `distress_only`, this sandbox
  only) was started to sanity-check the wiring, showed no crash through
  fold-1 setup, then was killed at the user's request** — the user will run
  the real thing directly on the GPU laptop instead. **No real results yet.**

### Run metrics
*(pending — not yet run)*

### Run inferences
*(pending)*

### Updations for next run
- Compare against Run 9's **0.7908 ± 0.0312** (standard architecture,
  current best validated baseline).
- If promising: consider combining with wide-distress (fuse features into
  the 256-dim path too — not built yet) and/or with tuned hyperparameters
  (the two deprioritized-but-ready configs from Runs 5/8) — one variable at
  a time, as established practice this whole document.
- Also still fully open/pending: MS-LSTM EMA-off re-run, MS-LSTM
  hyperparameter tuning (never done, parked after Run 6), turning on
  `error_analysis: {enabled: true}` on a future run to get concrete
  where-does-it-fail evidence (never done, cheap, one config line).

---

## Current state summary (as of this document)

**Best validated result**: Wide-distress CrossFormer, `full`, fold-2-fixed split — **AUROC 0.7995 ± 0.0498** (Run 10). Real edge over standard architecture but with a stability/Sens@90Spec tradeoff.

**Best unvalidated (proxy) result**: Standard CrossFormer + tuned hyperparameters — **AUROC 0.8044** (Run 5, 15ep/3fold proxy, deprioritized for full validation).

**Target**: ~0.84 (5% above the ~0.79 starting point) or ideally closing toward the 0.8565 standalone ceiling. **Not yet reached by any tested combination.**

**Immediately next**: Run 11 (feature-fusion) — user is about to run this directly on the GPU laptop.

**Fully parked/pending, in rough priority order**:
1. Feature-fusion full run (Run 11) — imminent.
2. MS-LSTM EMA-off re-run — never done since Run 6.
3. MS-LSTM hyperparameter tuning — never done.
4. Tuned-hyperparameter full-scale validation for both standard (`configs/model8_crossformer_tuned_config.yaml`) and wide-distress (`configs/model8_crossformer_widedistress_tuned_config.yaml`) architectures — deprioritized, ready whenever revisited.
5. Enabling `error_analysis` on some future run for concrete failure-mode evidence — never done, trivial to turn on.
6. CNN1D backbone — configs exist (`configs/model8_cnn1d_config.yaml`) but has never been run with any of this session's fixes at all.

**Files that matter for continuity**:
- `src/models/knowledge_infused_framework.py` — original framework (DistressHead/FIGOHead/ClinicalFeatureHead/FIGOCriteriaHead).
- `src/models/knowledge_infused_framework_wide_distress.py` — wide-distress variant.
- `src/models/knowledge_infused_framework_feature_fusion.py` — feature-fusion variant.
- `src/training/train_knowledge_infused.py` — main training script, all three architectures wired in via `backbone.model` / `heads.feature_fusion` config flags.
- `src/training/tune_model8_hyperparameters.py` — dedicated Model 8 hyperparameter tuner.
- `scripts/diagnose_fold2_composition.py` — fold-composition diagnostic (supports `--joint` flag).
- All `configs/model8_crossformer*.yaml` variants — see table above for what each represents.
