# Per-fold calibration in the CRP ensemble — findings

Status: **all tests run** (2026-08-21). Selection was done on
`data/processed_mil/val_dataset.pt`, which `train_ctg_crossformer.py` never
loads (it reads train + test only), so it is a clean selection set: 83 patients,
1162 windows, 61 positives (5.2%).

## Verdict

| Test | Question | Answer |
|------|----------|--------|
| T1/T2 | Change the combiner? | **No.** No combiner replicates 3/3 on validation; DeLong p ≈ 0.51 |
| T3 | Is the 0.30 operating point sound? | **No — this is the real defect** |
| T4 | Does the spread replicate? | **Yes, in all 5 ensembles including baselines** |
| T5 | Why does it happen? | **Double class-rebalancing in training** (already documented in code) |
| T6 | Safe to show as probabilities? | **No — over-predicts 3.5×. Platt scaling fixes it, 3/3 seeds** |
| T7 | Wire into inference? | **Open decision — see below** |

## T1/T2 — the combiner should not change

Selected on validation, averaged over the three CRP seeds:

| strategy | val AUROC | val AUPRC | beats mean-prob (AUROC / AUPRC) |
|---|---|---|---|
| mean prob (current) | 0.7826 | 0.2026 | — |
| median prob | 0.7708 | 0.1850 | 0/3 · 1/3 |
| **rank average** | 0.7872 | 0.2065 | 2/3 · 2/3 |
| logit average | 0.7846 | 0.1950 | 1/3 · 0/3 |
| z-score average | 0.7833 | 0.2032 | 2/3 · 2/3 |

Nothing clears the 3/3 paired-replication bar this project used to accept the
CRP gain itself. Logit average — which looked best on *test* (0.8205) — falls to
1/3 on validation, a clean demonstration of why the earlier test-set ranking was
not usable. DeLong on test: logit vs mean **p = 0.512**, rank vs mean
**p = 0.515**. With 51 positive windows this set cannot resolve ~0.008 AUROC.

**Keep mean-probability averaging.**

## T3 — the operating point is the real defect

A fixed 0.30 does not mean the same thing on two held-out sets, or across seeds:

| seed | val sens | val spec | test sens | test spec | **sens drift** |
|---|---|---|---|---|---|
| crp_s42 (delivered) | 0.344 | 0.871 | 0.490 | 0.879 | **+0.146** |
| crp_s1 | 0.852 | 0.529 | 0.843 | 0.544 | −0.009 |
| crp_s7 | 0.803 | 0.598 | 0.824 | 0.590 | +0.020 |

The delivered model at 0.30 sits at **49% sensitivity on test but 34% on
validation**. The same nominal threshold puts seed 1 at 84% sensitivity. 0.30 is
not a transferable operating point; it is an artefact of seed 42's particular
mix of fold offsets.

Deriving the threshold on validation instead transfers well (crp_s42, raw mean
prob):

| target sens | thr (val) | test sens | test spec | drift |
|---|---|---|---|---|
| 0.70 | 0.1795 | 0.725 | 0.737 | +0.021 |
| 0.75 | 0.1667 | 0.745 | 0.719 | −0.009 |
| 0.80 | 0.1511 | 0.784 | 0.679 | −0.019 |
| 0.90 | 0.1244 | 0.863 | 0.594 | −0.039 |

Drift of ±0.02–0.04 versus **+0.146** for the hardcoded value.

**How much of the alarm is one fold?** Of 152 windows flagged at 0.30 on test,
fold 2 supplies 33.6% of the flagged score (equal share would be 20%) and fold 5
supplies 3.7%. Removing fold 2 drops the flag count from 152 to 89 — **41% of
alarms depend on fold 2's offset**, not on agreement between folds.

## T4 — the spread replicates everywhere

Per-fold mean probability, validation:

| ensemble | fold means |
|---|---|
| crp_s42 | 0.108, **0.462**, 0.105, 0.138, 0.021 |
| crp_s1 | 0.022, **0.491**, **0.483**, 0.101, **0.493** |
| crp_s7 | 0.260, 0.342, **0.449**, 0.065, **0.402** |
| base_s1 | **0.514**, **0.509**, **0.457**, 0.084, 0.098 |
| base_s7 | **0.510**, 0.476, **0.489**, 0.185, 0.095 |

Every ensemble — CRP and pre-CRP baseline alike — splits into folds sitting near
~0.45–0.51 and folds near ~0.02–0.19. This is a property of the training recipe,
not of seed 42 or of clinical-relational pretraining. Val and test fold means
agree to ~0.01, so each fold's offset is a stable property of that model, which
is why a calibrator fit on validation transfers.

## T5 — root cause: double class-rebalancing

Training applies **two** imbalance corrections at once:

1. `WeightedRandomSampler` with sqrt-inverse frequency oversampling
   (`train_ctg_crossformer.py:379`), lifting batch prevalence ~4.3% → ~17%
2. `pos_weight = n_neg/n_pos` (≈22) in the focal loss, when
   `--class_weight inverse_freq` — which the delivered model uses

This is already documented in the code. The 2026-08-09 note at
`train_ctg_crossformer.py:419` set `pos_weight = 1.0` precisely because stacking
both "double-counts the class imbalance correction, pushing the model toward
extreme positive confidence during training that doesn't match the true class
balance at inference time." The 2026-08-19 change re-enabled it to match the
benchmarked paper's specified recipe and recover sensitivity.

The measured over-prediction is exactly the predicted side effect. Folds differ
in how far training pulled them back toward natural prevalence, which produces
the bimodal spread in T4. **The trade-off was made deliberately; only the
calibration consequence was left unaddressed.**

## T6 — the numbers shown to clinicians do not mean what they say

Test-set reliability of the delivered raw ensemble (mean p 0.163 vs prevalence
0.046 — **over-predicts 3.5×**, ECE 0.117):

| predicted bin | n | predicted | observed |
|---|---|---|---|
| (0.2, 0.3] | 117 | 0.245 | 0.077 |
| (0.3, 0.4] | 67 | 0.338 | 0.119 |
| (0.4, 0.5] | 43 | 0.444 | 0.186 |
| (0.7, 0.8] | 8 | 0.735 | **0.000** |
| (0.8, 0.9] | 4 | 0.845 | 0.250 |

`demo_patient_report.py` prints values like "peak risk 0.907" to a clinician.
A reading of "≈90% chance of acidosis" is not supported: windows scored 0.7–0.8
contained no positives at all.

**Per-fold Platt scaling, fit on validation, fixes this — 3/3 seeds:**

| seed | Brier raw → platt | ECE raw → platt | mean p → | AUROC raw → platt |
|---|---|---|---|---|
| crp_s42 | 0.0610 → **0.0408** | 0.1168 → **0.0139** | 0.163 → 0.053 | 0.8124 → 0.8112 |
| crp_s1 | 0.1266 → **0.0414** | 0.2771 → **0.0154** | 0.323 → 0.055 | 0.7983 → 0.8006 |
| crp_s7 | 0.1304 → **0.0406** | 0.2705 → **0.0160** | 0.317 → 0.057 | 0.8085 → 0.8322 |

Calibrated mean probability lands at 0.053–0.057 against a true prevalence of
0.046. AUROC is essentially unchanged (Platt is monotone per fold), and AUPRC
never gets worse (2/3 improve). It also makes a fixed threshold portable: at
0.30, raw sensitivity spans 0.490–0.843 across seeds (spread 0.353); calibrated,
all three land at 0.059 (spread 0.000).

## Recommended pipeline

Per-fold Platt fit on validation + threshold derived on validation at a target
sensitivity. Implemented in [`scripts/calibrate_crp.py`](../scripts/calibrate_crp.py):

```bash
python scripts/calibrate_crp.py                       # delivered model, 80% sens
python scripts/calibrate_crp.py --ckpt_dir checkpoints/repl_crp_s7 --target_sens 0.90
```

Delivered model, target 80% sensitivity, threshold **0.0487** derived on val:

| | sens | spec | PPV | alarm rate |
|---|---|---|---|---|
| val | 0.803 | 0.654 | 0.114 | 0.370 |
| **test** | **0.824** | **0.672** | 0.109 | 0.351 |

drift +0.020, versus the current deployed 0.30 giving test sens 0.490 / spec
0.879 with drift +0.146. Writes `calibration.json` next to the checkpoints.

The clinical trade is explicit: **0.824 sensitivity at 0.672 specificity**
instead of 0.490 at 0.879. At 0.30 the delivered model misses just over half of
acidotic windows — likely the wrong trade for a screening tool, but that is a
clinical call, not a modelling one.

## T7 — open decision

Nothing in the delivered inference path has been changed. Wiring calibration in
would touch `src/inference/ctg_inference.py`, `scripts/demo_inference.py`,
`scripts/demo_patient_report.py` and `scripts/temporal_alarm_logic.py`, and it
would change what clinicians see. That is a deliberate call to make, not a
silent edit.

`scripts/demo_explainability.py` is unaffected either way: it is single-fold and
its ρ is a Spearman correlation over raw scores, invariant to any monotone
recalibration.

## Not retested

Training was not re-run. Whether removing the double rebalancing (dropping
`--class_weight inverse_freq`, or switching to `sqrt_inverse_freq`) would give a
natively calibrated model **without** losing the sensitivity the 2026-08-19
change bought is untested — it needs three paired seeds to answer honestly, and
post-hoc calibration achieves the same end without touching the recipe the CRP
result was established on.
