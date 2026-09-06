# FIGO-state detection and early deterioration — Gates 1 and 2

Written 2026-09-04. Ruleset `figo2015-seq-v3`. Every number is reproducible
from `data/processed_figo/` via the scripts named in each section. This is a
new track: it does not touch `src/preprocessing/pipeline*.py`, `src/models/`
or `data/processed_clinical/`, so every earlier pH result stays reproducible.

**Two questions, deliberately kept apart.**

1. *Detection* — what FIGO state is this CTG in now?
2. *Early warning* — will it deteriorate into a pathological state soon?

## Verdict

**Detection passes both gates.** Raw signal alone reaches **AUROC 0.7518**
against a label whose ceiling is *provably* 1.0 (§5.2). The remaining gap is
attributable to a specific, identified architectural limitation (§5.4), not
to a label ceiling — which is the opposite of the situation in
[auroc_ceiling_analysis.md](auroc_ceiling_analysis.md).

**Early warning fails Gate 1 on statistical power**, and repairing the
measurement instrument made it *worse*: **36 positive anchors from 18
patients** at the 30-minute horizon. This is a base-rate property of CTU-UHB,
not a modelling failure, and no architecture recovers from it.

**The measurement repair (§4) changed the answers, and both directions
matter.** Deceleration F1 against expert consensus went 0.433 → **0.603**;
the state prevalences became clinically canonical; and detection AUROC fell
**0.8915 → 0.7518** because the earlier number was partly the model reading a
label artefact. §7 records the before/after in full.

---

## 0. Why the existing `y_figo` could not be used

`data/processed_clinical/` already carried a `y_figo` column, produced by
`knowledge/figo.py::classify_figo`. It is unusable as a state label for a
reason that is one line of code: it marks a window **Suspicious if it
contains a single variable deceleration**.

FIGO 2015 does not say that. It treats isolated variable decelerations as a
normal finding and escalates only on *repetitive* decelerations. On the
8,517-window substrate:

| clause | fires on |
|---|---:|
| **has variable deceleration** | **81.4%** of windows |
| variability outside 5–25 bpm | 44.6% |
| baseline outside 110–160 | 7.9% |

That single non-guideline clause produced the observed distribution — Normal
12.7%, Suspicious 66.6%, Pathological 20.6% — which is not a plausible
intrapartum distribution. Phase 9A had already measured our decelerations as
over-detected 1.8× against expert consensus, so the clause was thresholding
detector noise.

The rebuild is `src/figo_state/`: `descriptors.py` (measurement), `rules.py`
(**the frozen label**), `epochs.py` (segmentation), `protocol.py`
(evaluation). Measurement and labelling are separate modules on purpose, so
the instrument can be revalidated against expert data without touching the
label definition — which is exactly what §4 does.

### Three structural changes

**Non-overlapping 10-minute epochs, not 20-minute windows at 2.5-min stride.**
`auroc_ceiling_analysis.md` showed the old stride made the last negative and
first positive window of every acidotic patient 2.5 minutes apart, sharing
87.5% of their signal and carrying opposite labels. A temporal label inherits
that directly. 10 minutes is also FIGO's own unit of assessment — "baseline…
estimated over 10 minutes".

**Whole recordings, not the last 60 minutes.** Duration over 552 records is
median 71.7 min (p25 70.0, p95 90.0, min 60.0). Truncation discarded a median
11.7 min from the *start*, which is where a deterioration trajectory begins.

**Persistence evaluated across epochs.** Every pathological
variability/deceleration criterion in FIGO 2015 carries a 20–50 minute
persistence requirement that a single window structurally cannot observe —
which is why `knowledge/figo.py` dropped its variability flags entirely.
`classify_sequence()` walks a recording forward and looks *back* over
trailing epochs, never forward, so the state at epoch *t* is computable in
real time. That is a hard requirement for early warning and the reason the
old horizon label had to be abandoned.

---

## 1. A real preprocessing bug, found and fixed

`interpolate_missing()` repairs gaps up to 15 s and **deliberately leaves
longer ones at 0.0 bpm** so it does not fabricate decelerations. Correct for
a neural net. Catastrophic for a rule engine: a 0.0 sample sits ~140 bpm
below any real baseline, so every long gap reads as a deep, long
deceleration.

Measured on the cleaned FHR channel: **13.2% of samples at or below 50 bpm,
minimum −23.4 bpm** (the low-pass filter smearing the zeros). The consequence
was monotone in signal quality:

| epoch quality | decels/epoch | mean deepest decel | frac. time in decel |
|---|---:|---:|---:|
| 0.50–0.70 | 4.14 | 149.8 bpm | 0.41 |
| 0.70–0.85 | 3.49 | 141.4 | 0.24 |
| 0.85–0.95 | 2.27 | 92.4 | 0.13 |
| **0.95–1.00** | **1.01** | **27.1** | **0.06** |

The fix is a validity mask taken *after* repair and *before* filtering, then
dilated by the filter settling time, and threaded through every measurement:
baseline, variability, accelerations, decelerations, contraction detection.
Runs are broken at gaps rather than bridged, which under-counts long events
rather than inventing them — the conservative direction, and the one that
matters when a single prolonged deceleration is sufficient for Pathological.

Two pathological criteria were being fired **entirely by dropout** and fell
to zero after the fix.

---

## 2. Descriptor calibration

`scripts/figo_audit_labels.py`, 3,497 readable epochs.

| | features.py (20-min) | figo_state v3 | plausible? |
|---|---:|---:|---|
| variability <5 bpm | 0.4% | 6.2% | ✔ |
| variability 5–25 bpm | 55.4% | **92.8%** | ✔ |
| variability >25 bpm | 44.2% | **1.0%** | ✔ |
| baseline 110–160 | 92.1% | 92.2% | ✔ |
| decelerations/epoch | 3.44 *variable alone* | **2.01 total** | ✔ |
| contractions/epoch | — | 4.08 | ✔ (≤5 per 10 min) |

`features.py`'s own docstring flagged the >25 bpm prevalence as a known
unresolved calibration gap. Three changes closed it: excluding accel/decel
samples before measuring; smoothing to 2.5 s (five times above FIGO's 3–5
cycles/min oscillation band, so the oscillation itself is untouched); and
combining 1-minute segments with a **median** rather than a mean, since the
mean let one artefactual minute set the epoch's value.

Saltatory variability at 1.0% is consistent with it being a genuinely rare
finding. The v1 build read 13.8%, and §7 shows that excess was inflating the
Pathological class through a criterion that has since disappeared.

---

## 3. GATE 1 — label validity

### 3.1 State prevalence: PASS on Normal/Suspicious, Pathological is rare

| state | epochs | % of readable | patients | published reference |
|---|---:|---:|---:|---|
| Normal | 2236 | **63.9%** | 530 | ~60–70% ✔ |
| Suspicious | 1214 | **34.7%** | 474 | ~20–30% ✔ |
| Pathological | 47 | **1.3%** | 26 | ~5–10% ✗ |
| *unreadable* | 412 | *10.5% of all* | 288 | — |

Which criterion fires:

| criterion | epochs | patients |
|---|---:|---:|
| baseline <100 bpm | 35 | 18 |
| **repetitive late/prolonged decels >30 min** | **10** | **7** |
| reduced variability sustained >50 min | 2 | 1 |
| single prolonged decel >5 min below 80 bpm | 1 | 1 |
| increased variability sustained >30 min | 0 | 0 |

The deceleration pathway now fires (it did not before §4). Sinusoidal pattern
is **not implemented** — no validated detector exists here and inventing one
would put an unvalidated component inside the label. The Pathological class
is under-inclusive by exactly that criterion, stated rather than hidden.

Why Suspicious fires: repetitive decelerations 65.8%, variability outside
5–25 or unreadable 25.0%, baseline outside 110–160 19.7%. **This composition
is what makes the detection task hard for a channel-concatenating CNN**
(§5.4).

### 3.2 Transitions — genuine, because epochs do not overlap

| from \ to | Normal | Suspicious | Pathological | unreadable |
|---|---:|---:|---:|---:|
| Normal | 68.8% | 23.8% | **0.1% (3)** | 7.2% |
| Suspicious | 42.1% | 45.6% | **3.2% (27)** | 9.1% |
| Pathological | 10.3% | 15.4% | 64.1% | 10.3% |

**Normal → Pathological essentially does not happen.** Deterioration goes
through Suspicious. Any early-warning task at Pathological severity is
therefore a *Suspicious → Pathological* task, and that conclusion comes from
the data rather than from a preference.

### 3.3 A censoring trap, found and closed

The first build scored a positive whenever a pathological epoch was seen,
even on a truncated horizon, and required full observation only for
negatives. That is the natural survival-analysis instinct and it is wrong
here, because it makes **inclusion depend on the outcome**:

| | n | mean epoch index |
|---|---:|---:|
| y = 0 | 1731 | 1.53 |
| y = 1 | 142 | **3.55** |

A predictor consisting of nothing but the epoch index scored **AUROC
0.8321** — the same failure this project diagnosed once already, where window
position alone reached 0.84 and beat every trained model. Censoring both
classes identically dropped it to 0.7503, and after the §4 repair it sits at
**0.6400**. It stays in the audit as a permanent control.

### 3.4 Early warning: FAIL on power

30-minute horizon, anchors where the current state is Normal or Suspicious:

| target | n | positives | patients | prevalence |
|---|---:|---:|---:|---:|
| **Pathological within 30 min** | 1651 | **36** | **18** | 2.2% |
| Abnormal within 30 min, from Normal | 1251 | 706 | 362 | 56.4% |
| Worsens ≥1 category, from Normal or Susp | 1649 | 727 | 367 | 44.1% |

Baselines any model must beat, on the 36-positive task:

| predictor | AUROC |
|---|---:|
| current state (persistence) | 0.6628 |
| epoch index — no signal at all | 0.6400 |

**36 positives from 18 patients cannot support the stated goal.** Five-fold
patient-grouped CV puts ~4 positive patients in each test fold; the bootstrap
interval on AUROC would be roughly ±0.15, wider than any effect worth
detecting. Training a deep model against it would produce a number, not a
result.

---

## 4. Repairing the instrument against FHRMA

`scripts/figo_diagnose_decel_detection.py`,
`scripts/figo_validate_descriptors_fhrma.py`. 1,021 analysable 10-minute
epochs, 1,375 expert-annotated decelerations. FHRMA carries **no pH**, so
nothing tuned here can leak into an acidaemia claim.

### 4.1 Why Phase 9A's conclusion does not transfer

Phase 9A found that repairing the baseline toward expert behaviour **cost
0.0972 patient AUROC** on the pH task, and concluded expert agreement is not
a proxy objective. That finding is correct and does not apply here, for a
precise reason: there, descriptors were *inputs* to a model predicting an
external outcome, so a biased descriptor could still be predictive. Here
descriptors **define the label**. There is no external outcome to be
accidentally right about, so expert agreement *is* the objective.

### 4.2 The cause was fragmentation, not the baseline

A FIGO event has a 15 bpm / 15 s core, but the guideline does not say how to
*bound* it, and clinicians read a deceleration from onset to return to
baseline. Detecting only the part deeper than 15 bpm splits one expert
deceleration into several fragments, each scoring as a false positive.

| delineation | n found | sens | prec | event F1 | Dice | frac. time |
|---|---:|---:|---:|---:|---:|---:|
| core only (as first written) | 2187 | 0.588 | 0.370 | 0.454 | 0.591 | 0.167 |
| edge 2 bpm, merge 30 s | 1905 | 0.751 | 0.542 | 0.629 | 0.605 | 0.382 |
| edge 10 bpm, merge 45 s | 1694 | 0.696 | 0.565 | 0.624 | 0.590 | 0.344 |
| **edge 7.5 bpm, merge 30 s** | **1999** | **0.740** | **0.509** | **0.603** | **0.619** | **0.329** |
| *expert annotation* | *1375* | — | — | — | — | *0.231* |

**A methodological trap worth recording.** Event matching is by *any*
temporal overlap, so one bloated span overlapping one expert deceleration
still scores a true positive — the metric is blind to span inflation.
Optimising event F1 alone selected edge 2 bpm / merge 30 s, whose
decelerations covered **38.2%** of every epoch against the experts' 23.1%.
Sample-level Dice catches this and event F1 cannot. The selection rule was
therefore fixed in advance: maximise event F1 subject to (a) merge gap < the
60 s minimum contraction separation, (b) Dice ≥ 0.60, (c) time-in-deceleration
within 1.5× the expert's 0.231.

Both constants are principled rather than fitted: **7.5 bpm is half the FIGO
event amplitude**, **30 s is half the minimum contraction separation**. A
longer merge fuses decelerations from *adjacent* contractions; a fused span
has one nadir, so it pairs with one contraction and suppresses the
"repetitive decelerations" test that the pathological criterion depends on.

The 15 bpm / 15 s core was **not** tuned. A sweep found 18 bpm / 10 s scored
marginally higher and it was not adopted — those numbers are the guideline's
definition of the event, and fitting them to a corpus would make the label a
fitted object rather than FIGO 2015.

### 4.3 Result

| | Phase 9A (old) | v3 | 
|---|---:|---:|
| acceleration sensitivity | 0.204 | **0.447** |
| acceleration F1 | 0.297 | **0.436** |
| deceleration sensitivity | 0.604 | **0.740** |
| deceleration precision | 0.338 | **0.509** |
| deceleration F1 | 0.433 | **0.603** |
| deceleration Dice | — | 0.619 |
| baseline MAD | 2.28 bpm | 2.86 bpm |

**This retires a number.** Phase 9A reported 0.616 as the *expert-baseline
upper bound* for deceleration F1. We now reach 0.603 using **our own**
baseline, and 0.679 using the expert's. That 0.616 was an artefact of
fragmentation, not a ceiling.

Note that **accelerations are not a criterion in the FIGO 2015 classification
table** — their presence is reassuring, their absence is not classified. So
acceleration F1, the weaker number, does not enter the label at all. Only
deceleration detection does.

---

## 5. GATE 2 — is the state learnable, and from what?

`scripts/figo_gate2_state_detection.py`. Task A1, binary Normal vs Abnormal,
3,497 epochs, 36.1% prevalence, 5 patient-grouped folds, CI bootstrapped over
patients.

| arm | input | AUROC | 95% CI | sens/spec |
|---|---|---:|---|---|
| 0 | rule descriptors → **tree** | 0.9997 | 0.999–1.000 | 0.997 / 0.997 |
| 0b | rule descriptors → **LR** | 0.8371 | 0.810–0.862 | 0.790 / 0.793 |
| 1 | non-rule descriptors → tree | 0.8459 | 0.828–0.862 | 0.772 / 0.773 |
| **2** | **raw FHR + UC + mask → CNN** | **0.7383** | 0.718–0.759 | 0.684 / 0.683 |
| **2a** | **raw FHR + mask → CNN** | **0.7518** | 0.733–0.772 | 0.688 / 0.687 |
| 3 | epoch index alone | 0.5510 | 0.529–0.571 | — |

Arms 0 and 0b are **circular** — the label is a deterministic function of
those columns — and are wiring checks, not results.

### 5.1 Do not use logistic regression as the descriptor baseline

Arms 0 and 0b differ by **0.16 AUROC on identical columns and identical
folds**. The FIGO rule is a conjunction of *intervals* — "110 ≤ baseline ≤
160", "5 ≤ variability ≤ 25". A linear model cannot express an interval; it
pushes each coefficient one way and must fail at one end of every band.

This matters beyond the wiring check. "Logistic regression on the clinical
descriptors" is the natural first baseline to reach for, and here it reads
**0.837 on a label that is exactly recoverable at 0.9997**. Read as a
difficulty estimate it is badly wrong.

### 5.2 The ceiling is 1.0, and that is the point

Arm 0 proves the label is exactly recoverable from the descriptors, and the
descriptors are computed from the signal. So the raw-signal ceiling **is
1.0**, and the 0.7518 → 1.0 gap is entirely the CNN failing to extract
descriptors it provably could.

This is the structural opposite of the pH work, where
`auroc_ceiling_analysis.md` showed the ceiling lived in the label and
concluded *"swapping the encoder architecture will not help."* Here it will.

### 5.3 Against the stated goal

The target is AUROC ≥ 0.85 **and** sensitivity ≥ 85% **and** specificity ≥
85%. Those are not the same bar: a ROC through (0.15, 0.85) has AUROC ≈
**0.92**.

| | achieved | required |
|---|---:|---:|
| AUROC | 0.7518 | 0.85 → really ≈0.92 |
| balanced sens/spec | 0.688 / 0.687 | 0.85 / 0.85 |

### 5.4 The specific reason the CNN falls short

Suspicious is now **65.8% driven by repetitive decelerations**, and
"repetitive" is defined as *decelerations accompanying more than 50% of
contractions*. Computing it requires detecting decelerations, detecting
contractions, pairing them in time, and taking a ratio.

**The UC channel adds nothing** — 0.7383 with it, 0.7518 without. On a label
that now depends critically on FHR–UC coupling, that is diagnostic: a CNN
that concatenates channels at the input and convolves them jointly has no
mechanism to represent "this dip corresponds to that contraction". It is
being asked for a relational computation with a non-relational architecture.

That is a concrete, testable next step rather than a general call for more
capacity, and the repo already contains a cross-channel attention encoder
(`src/models/ctg_crossformer.py`) built for exactly this.

---

## 6. Where this leaves each task

**Detection — proceed.** Both gates pass, the ceiling is proven reachable,
and the gap has an identified cause. Next: an encoder with explicit FHR–UC
cross-attention, tested against the 0.7518 baseline on the frozen folds.

**Early warning — the Pathological-severity target is not viable on this
dataset.** 36 positives / 18 patients, after the instrument was repaired
specifically to improve it. Remaining options:

1. **Two-tier target.** Primary = "abnormal within 30 min from a currently
   Normal epoch" (706 positives, 362 patients, clock control 0.638).
   Secondary = pathological within 30 min, reported with its honest CI.
2. **Report Suspicious → Pathological as an underpowered secondary
   endpoint**, with the interval, never as a headline.
3. **Accept that CTU-UHB cannot answer this question** and say so. 552
   recordings with 26 patients ever reaching a strict FIGO Pathological state
   is a corpus-size problem, not a method problem.

**Do not train an early-warning model on 36 positives and report the AUROC.**

---

## 7. What the instrument repair cost, and why it was still right

The repair in §4 was chosen *specifically* to enrich the Pathological class
by making the deceleration pathway fire. It did make that pathway fire — and
the class got **smaller**, and detection AUROC **fell**.

| | v1 (pre-repair) | v3 (post-repair) |
|---|---:|---:|
| deceleration F1 vs expert | 0.454 | **0.603** |
| variability >25 bpm | 13.8% | **1.0%** |
| Pathological epochs | 63 | 47 |
| — from `path_increased_var` | 31 | **0** |
| — from `path_rep_decels` | 0 | **10** |
| 30-min positives (eligible) | 43 (27 patients) | 36 (18 patients) |
| clock control AUROC | 0.750 | **0.640** |
| **raw-signal detection AUROC** | **0.8915** | **0.7518** |

Reading this honestly:

- **The v1 Pathological class was inflated.** Half of it came from
  `path_increased_var`, which depended on a variability estimate reading
  13.8% saltatory. At 1.0% post-repair that criterion fires zero times. The
  v1 number was measuring an estimator artefact.
- **The genuine deceleration pathway now works** — 10 epochs across 7
  patients, where it previously fired zero times.
- **The v1 detection score of 0.8915 was partly inflated by the same
  artefact.** A CNN can read gross variability from signal morphology far
  more easily than it can compute a deceleration-to-contraction ratio. The
  v1 label was easier because it was partly wrong.
- **The time confound also shrank** (0.750 → 0.640), which is a further sign
  the v3 label is less contaminated.

So the repair traded a larger, partly-spurious positive class and a flattering
score for a smaller, more valid one and an honest score. That is the right
trade for a label definition, and it is why §5's 0.7518 is the number to
build on rather than the 0.8915 an earlier draft reported.
