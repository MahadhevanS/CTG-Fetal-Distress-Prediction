# The FHR–UC interaction experiment — result

Written 2026-09-04. Ruleset `figo2015-seq-v3`, protocol
`data/processed_figo/folds.json`, repeat 0. Reproduce with:

```bash
python scripts/figo_run_detection.py          # writes oof.npz per arm
python scripts/figo_eval_detection.py --baseline Z_frozen_smallcnn
```

## The hypothesis, as pre-registered

> Detection sits at AUROC ≈0.75 against a label whose ceiling is provably 1.0.
> Suspicious is 65.8% driven by *repetitive decelerations* — decelerations
> accompanying more than 50% of contractions — so the label depends on a
> **relation between two channels**. Giving a plain CNN the UC channel changes
> nothing. The hypothesis is that this is a *representational* failure, not an
> information failure: a network that concatenates channels at the input has
> no mechanism to encode "this dip corresponds to that contraction".

Falsifiable, and it has now been falsified.

## Result

| arm | params | AUROC | 95% CI | AUPRC | spec@90%sens | balanced se/sp |
|---|---:|---:|---|---:|---:|---|
| **Z** frozen Gate-2 SmallCNN | 69,281 | **0.7559** | 0.737–0.774 | 0.6327 | **0.3980** | 0.688/0.688 |
| **A** FHR only | 221,378 | 0.7369 | 0.715–0.757 | 0.6555 | 0.3247 | 0.673/0.674 |
| **C** FHR+UC cross-attention | 452,066 | 0.7328 | 0.711–0.754 | 0.6421 | 0.3202 | 0.676/0.674 |
| **B** FHR+UC concatenation | 222,818 | 0.7324 | 0.712–0.753 | 0.6368 | 0.3005 | 0.676/0.675 |
| **D** C, UC shuffled | 452,066 | 0.7067 | 0.685–0.729 | 0.6264 | 0.2549 | 0.649/0.649 |
| **E** C, FHR shuffled | 452,066 | 0.5322 | 0.509–0.554 | 0.3883 | 0.1096 | 0.525/0.523 |

Every arm scored through the same harness
(`scripts/figo_eval_detection.py`), on the same folds, with CIs bootstrapped
over patients.

### The four pre-registered comparisons

| comparison | asks | Δ AUROC | answer |
|---|---|---:|---|
| **A → B** | does UC help at all under plain fusion | **−0.0044** | no |
| **B → C** | does an explicit interaction mechanism beat fusion | **+0.0004** | **no** |
| **C → D** | does the model need the *correct* FHR–UC pairing | **+0.0261** | yes |
| **C → E** | reverse control | **+0.2006** | FHR carries the signal |

**B → C is the hypothesis, and it is +0.0004.** Cross-attention is
indistinguishable from concatenating the channels and convolving them
jointly.

### Paired tests — the deltas, not the marginal CIs

Comparing two overlapping marginal CIs is the wrong instrument; both models
score the same patients, so the difference must be tested paired. Bootstrapped
over patients, 4,000 resamples:

| comparison | Δ AUROC | 95% CI | |
|---|---:|---|---|
| **C − B** (the hypothesis) | +0.0004 | [−0.0137, +0.0137] | **crosses 0** |
| C − A | −0.0041 | [−0.0169, +0.0092] | crosses 0 |
| **C − D** (the mechanism) | +0.0261 | [+0.0096, +0.0430] | **excludes 0** |
| **Z − C** (small beats large) | +0.0231 | [+0.0043, +0.0417] | **excludes 0** |

Per fold, C versus B:

| fold | C | B | C−B |
|---|---:|---:|---:|
| 1 | 0.7037 | 0.7281 | −0.0243 |
| 2 | 0.7238 | 0.7442 | −0.0203 |
| 3 | 0.7503 | 0.7530 | −0.0027 |
| 4 | 0.7525 | 0.7455 | +0.0071 |
| 5 | 0.7392 | 0.7217 | +0.0175 |

**C beats B in 2 of 5 folds**, mean delta −0.0045, sd 0.0178, paired Wilcoxon
p = 0.625. The sign flips across folds. There is no effect here to refine.

C versus D goes the other way: **4 of 5 folds**, mean +0.0215. (Wilcoxon on
n = 5 gives p = 0.188 and is simply underpowered at that sample size; the
patient bootstrap above is the instrument to read, and it excludes zero.)

## PHASE 7 GATE

Evaluated on the interaction arm C, per the pre-registered rule — not on the
best-scoring arm, which is a control.

> **C = 0.7328 < 0.75 → KILL CRITERION. Stop architecture exploration.**

No refinement is permitted under the rule as written. Contraction-conditioned
local attention (Phase 8) was contingent on "meaningful evidence" from C, and
+0.0004 is not that.

## What C → D means, and what it does not

D is the one genuinely interesting number. Shuffling UC across epochs — same
marginal distribution, correspondence destroyed — costs C **0.0261 AUROC**,
with barely-overlapping CIs (0.711–0.754 vs 0.685–0.729). So the
cross-attention model **is** using the real temporal relationship between the
two channels. The mechanism works.

It just does not pay. C ≈ A ≈ B means whatever the coupling contributes is
**redundant with what FHR alone already provides**. The model can see the
relationship and the relationship tells it nothing new.

There is a second, duller reading of D that must be stated: mismatched UC is
*worse than no UC* (D 0.7067 < A 0.7369), so some of the 0.0261 is simply the
cost of feeding a network contradictory input. The two readings are not
separable from this experiment alone, and the honest summary is that D
establishes dependence on the pairing without establishing its value.

## An unplanned finding: capacity actively hurts

The best model in the table is the **smallest by a factor of 6.5**. The 69k
SmallCNN beats the 452k cross-attention model by 0.0231 and the 221k FHR
model by 0.0191, and it also has the best `spec@90%sens` (0.3980 vs 0.3202).

Ordering by parameter count reverses the ordering by score almost exactly.
That is the third time this project has measured it — Model 9 (KG-MIL) put
2.5M parameters on ~435 bags and collapsed
(`model8_crossformer_run_history.md`, Part E), and
`temporal_modelling_feasibility.md` reached the same arithmetic independently.
3,497 epochs from 547 patients does not support these models.

*Caveat on Z's number.* It reads 0.7559 here against 0.7518 in
`figo_state_gate1_gate2.md`. Same network, same folds, different training
recipe (40 epochs / AdamW 3e-4 / patience 10, versus the Gate-2 recipe).
The difference is within the fold-to-fold spread and is a recipe difference,
not a discrepancy — the two must not be quoted interchangeably.

## Against the 89/89 target

No arm comes close. Best balanced operating point is **0.688 / 0.688** (Z).
At the pre-specified threshold — chosen inside each training fold at 90%
validation sensitivity and applied blind — the models sit at roughly 88%
sensitivity for **35% specificity**. The ROC does not pass near (0.10, 0.90);
it is not close enough for a threshold choice to rescue it.

## Where the remaining gap actually is

The label is exactly recoverable from the descriptors (tree, AUROC 0.9997).
So the ~0.24 gap between 0.756 and 1.0 is entirely the network's failure to
compute the descriptors from signal. The interaction hypothesis said the
missing piece was FHR–UC coupling. It is not. Remaining candidates, none of
them tested:

1. **The baseline.** Every FIGO criterion is expressed *relative to the
   baseline*, and the baseline is a per-epoch scalar estimated by an iterative
   procedure. A convolutional network has no obvious way to compute
   "the modal level of the least oscillatory segments" and then measure
   everything against it. This is the single largest untested candidate, and
   unlike architecture width it is a *specific* representational gap.
2. **Counting and thresholding.** "More than 50% of contractions" and
   "≥15 bpm for ≥15 s" are counting operations with hard cut-offs. Networks
   approximate those poorly, and the label is built entirely from them.
3. **Cross-epoch persistence.** Three of five pathological criteria are
   defined over 20–50 minutes, i.e. across epochs, and every model here sees
   one 10-minute epoch in isolation.

(3) is the only one that is a *modelling* gap rather than an inductive-bias
gap, and it is also the cheapest to test — but note that this project has
already run the trajectory experiment once on the pH task and got a clean null
(`temporal_modelling_feasibility.md`).

## Recommendation

The pre-registered gate says stop, and it should be honoured. Specifically:

- **Do not** run Phase 8 (contraction-conditioned attention). Its trigger
  condition was not met.
- **Do not** enlarge, re-tune, or sweep. The capacity finding says the
  direction of travel is *smaller*, and this project has already spent a
  phase learning that architecture search does not move this dataset.
- **The honest headline for detection is AUROC 0.756 [0.737–0.774]**, from a
  69k-parameter CNN on FHR alone, against a rule-defined label. Sensitivity
  and specificity at the balanced point are 0.688/0.688, which does not meet
  the 89/89 target and is not within reach of it.

Two defensible continuations, both outside the killed branch:

**(a) Accept detection as measured and write it up.** The result is a
methodologically clean negative on a falsifiable hypothesis, on top of a
label instrument validated against expert consensus. That is publishable
work, and the surrounding findings — the deceleration-fragmentation repair,
the interpolation bug, the LR-on-interval-rules trap, the censoring trap —
are individually useful.

**(b) Move to the early-warning track** with the two-tier target
(Normal → Abnormal within 30 min: 1,251 anchors, 706 positives, 362
patients), which is well powered and has not been touched.

What should **not** happen is another architecture attempt on detection
justified after the fact.
