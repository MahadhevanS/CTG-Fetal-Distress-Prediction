# Why the model stops at ~0.81 AUROC — root cause analysis

Status: analysis complete (2026-09-02), run on the held-out test split with the
delivered CRP ensemble (seed 42), confirmed on val and train where applicable.

## The headline

**A predictor that never looks at the signal — just the window's position in the
recording — beats the delivered model on the trained task, on every split.**

| split | delivered CRP model | **time alone** (no signal) |
|---|---|---|
| train | — | 0.8567 |
| val | 0.7879 | **0.8633** |
| test | 0.8124 | **0.8406** |

That is the answer to "why can't we get past 0.81." The window label is largely a
statement about *when* the window occurred, not *what is in it*, and the model
cannot see time. It is being asked to predict something its input only weakly
determines.

## Where the label comes from

`pipeline_mil.py:190-192`:

```python
within_horizon = (start >= horizon_start_idx)   # last 30 min of the segment
window_label   = int(is_distress and within_horizon)
```

So a window is positive **only if** the baby was acidotic (pH ≤ 7.15) **and** the
window starts in the final 30 minutes. Three consequences follow, and all three
are measurable.

### 1. The label is a clock as much as a diagnosis

Decomposing what the score actually separates, on test:

| target | delivered model | time alone | 8 FIGO features alone |
|---|---|---|---|
| `y_primary` (the trained task) | 0.8124 | **0.8406** | 0.7805 |
| `y_patient` (**is this baby acidotic?**) | **0.6086** | 0.4984 | **0.6179** |

On the actual clinical question the model scores **0.6086**, barely above chance,
and *slightly worse than eight hand-computed FIGO features*. Time alone scores
0.4984 on that target — exactly chance, confirming the temporal signal lives in
the label, not in the outcome.

Patient-level, aggregating each patient's window scores: mean → AUROC 0.6383,
max → 0.6241 (n=82 patients).

**The 0.81 headline is a blend of a hard clinical task the model does poorly
(0.61) and a timing task that inflates the number.**

### 2. The model has learned "late labour" and applies it to healthy babies

Correlation between a window's position in the recording and its risk score:

| group | ρ(position, score) | mean score, early third → late third |
|---|---|---|
| **normal patients** (every window labelled 0) | **+0.351** (p = 1.8e-27) | 0.112 → 0.222 (**1.98×**) |
| acidotic patients | +0.301 (p = 1.4e-05) | 0.167 → 0.273 (1.63×) |

In babies who were completely fine, the model's risk score still **doubles** from
the start of the hour to the end. It has learned the physiological signature of
late second-stage labour — rising contraction frequency, more decelerations,
falling baseline — which is genuinely predictive of the *label* but happens in
everyone. This is a direct, mechanical explanation for the false-positive rate.

### 3. Adjacent windows carry opposite labels on 87.5% identical input

Windows are 20 minutes long at a 2.5-minute stride. For **every** acidotic patient
in the test set, the last negative window and the first positive window are 2.5
minutes apart — they share **87.5% of their signal** and carry opposite labels.

In 6 of 11 such patients the model scores the *negative* window higher than the
*positive* one. It is not making a mistake; the pair is not separable, and no
model can be.

## The preprocessing problem: the quality gate eats the positive class

`get_valid_windows(..., max_missing_ratio=0.30)` drops windows with >30% missing
samples. Signal quality is worst in late second-stage labour — exactly where the
positive labels live. Audited across all 544 patients:

| group | horizon windows kept | pre-horizon windows kept | **asymmetry** |
|---|---|---|---|
| acidotic | 338/540 (62.6%) | 1117/1296 (86.2%) | **−23.6 pp** |
| normal | 1614/2180 (74.0%) | 4482/5232 (85.7%) | −11.6 pp |

**27 of 108 acidotic patients (25%) lose *every* horizon window** and therefore
contribute zero positive labels. Four are in the test split:

| patient | pH | missing in segment | horizon windows | surviving the gate |
|---|---|---|---|---|
| 1001 | 7.140 | 29.3% | 5 | **0** |
| 1187 | 7.080 | 38.2% | 5 | **0** |
| 1358 | 7.140 | 52.7% | 5 | **0** |
| 2009 | **6.960** | 31.0% | 5 | **0** |

Patient 2009 has pH 6.96 — severe acidosis — and is labelled entirely normal.

The positive class loses 202 of 540 acidotic horizon windows (37%) to the gate.
Given that this project's stated binding constraint is the number of positive
windows, **the gate is destroying over a third of the scarce class, and doing so
non-randomly**: the traces it removes are the noisy, distressed ones.

## The pH threshold is a hard cut through a continuum

| pH band | patients (test) | label |
|---|---|---|
| (7.00, 7.10] | 5 | acidotic |
| (7.10, 7.15] | 5 | acidotic |
| (7.15, 7.20] | 10 | normal |
| (7.20, 7.25] | 17 | normal |
| (7.25, 7.40] | 38 | normal |

**15 of 82 patients (18%) sit within ±0.05 pH of the 7.15 cut.** A baby at 7.16
and one at 7.14 are physiologically indistinguishable and oppositely labelled.
One test patient sits exactly on 7.150.

## A result I computed and then discarded

An earlier pass excluded windows straddling the horizon boundary and reported
AUROC **0.9832**, which looked like a clean "recoverable ceiling". It is an
artefact and must not be used: the exclusion removed **49 of 51 positives (96%)**
while leaving the negative pool nearly intact, so 0.9832 is two positives against
984 negatives. The symmetric version (removing the same band from every patient)
leaves zero positives and is undefined. **There is no evidence for a 0.98
ceiling.** Recorded here so it is not rediscovered and believed.

## What to change, in order of expected payoff

### A. Fix the label — highest payoff, moderate cost

The horizon rule is the root cause. Three options, in increasing conservatism:

1. **Train on `y_patient` (the MIL target) instead.** Removes the time confound
   entirely: every window of an acidotic patient is positive. Costs label noise
   (early windows may genuinely look normal) but the target becomes the clinical
   question. Note the project previously measured patient aggregation as worse
   (0.79 → 0.66); that measurement was of *aggregating window scores*, which is
   a different thing from *training on the patient label*.
2. **Keep the horizon but exclude the boundary band from training.** Drop windows
   within one window-length of the cut so contradictory near-identical pairs
   never both appear. Cheap, and removes the unlearnable pairs.
3. **Give the model time explicitly** as an input feature. If late-labour risk is
   real (it is), stop making the model infer the clock from signal. But then the
   reported AUROC must be interpreted knowing time is an input — and time alone
   already scores 0.84.

**Recommendation: (1) as the primary experiment, (2) as a cheap ablation.** They
answer different questions and both are informative.

### B. Fix the quality gate — high payoff, low cost

Currently a hard 30% missing-sample exclusion. Options:

- Relax the threshold for horizon windows, or raise it globally to 40–50% and let
  the interpolation handle it (the chain already does cubic interpolation).
- Replace hard exclusion with a **quality weight in the loss**, so poor windows
  contribute less rather than vanishing.
- At minimum, **stop dropping patients silently**: 25% of acidotic patients
  currently contribute no positive label at all.

This alone could increase the positive class by up to ~60% (338 → 540 acidotic
horizon windows), directly attacking the binding constraint.

### C. Soften the pH threshold — moderate payoff, low cost

Use continuous pH as a regression or ordinal target, or exclude the ambiguous
7.10–7.20 band from *training* (keeping it in test). 18% of patients currently
receive a near-arbitrary hard label.

### D. What will NOT help

**Swapping the encoder architecture.** The ceiling is in the label, not the
model. The delivered CrossFormer already scores below a clock on this task;
CNN1D, MSLSTM or anything else will hit the same wall. Architecture work is worth
doing to test whether a smaller model matches the CrossFormer (it may, at 1/20th
the parameters — which matters for the embedded device), but it will not move the
ceiling.
