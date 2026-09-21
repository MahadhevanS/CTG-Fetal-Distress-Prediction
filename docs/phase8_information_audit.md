# Phase 8 — New-information audit

Completed 2026-09-04. Premise accepted from the project owner: **the 0.85 target
stands.** What is rejected is the *current information ladder*, not the goal. This
phase therefore asks one question only:

> What information about fetal distress exists in CTU-UHB that the current
> representation throws away — and is any of it worth ≥ +0.12 AUROC?

Nothing here is architecture search. Every number below is a *representation* or
*information* experiment run on the frozen protocol
([../src/training/protocol.py](../src/training/protocol.py)): 547 patients, 110
positive, patient-grouped folds, repeat 0, patient-level **max** aggregation,
patient-bootstrap CIs. The comparator is the 19-descriptor clinical baseline.
Pre-registered bar: **+0.0642** (the phase-7 MDE), not +0.02.

---

## 1. Executive verdict

**Eight independent information sources were tested against the established
0.7271. The best is +0.0027; none approaches the +0.0642 bar. Seven different
outcome definitions land within ±0.005 of each other.**

The six window-level feature additions are tabulated here; the three structural
routes — contraction-response trajectory, stage-aligned trajectory, and the
full-length recording — are in §8a and §8b, and are also negative.

| experiment | alone | added to the 19 | Δ |
|---|---:|---:|---:|
| **19 clinical descriptors (baseline)** | — | **0.7271** [0.670–0.779] | — |
| + 18 HRV / spectral / nonlinear features | 0.6435 | 0.7119 | **−0.0152** |
| + 14 prospective maternal/clinical covariates | 0.5886 | 0.7096 | **−0.0175** |
| + 6 retrospective covariates (*leaky, reference only*) | 0.6194 | 0.7129 | **−0.0142** |
| + stage-II flag + minutes-before-end | — | 0.7276 | **+0.0005** |
| + 6 missingness-pattern features | 0.5911 | 0.7297 | **+0.0027** |
| + HRV + stage-II flag | — | 0.7180 | −0.0091 |

Not one reaches the bar. Four of six are *negative* — the classic signature of
adding noise features to 110 positives.

This is materially stronger evidence than phase 7 had. Phase 7 could say only
"two information sources remain untested." Both have now been tested, along with
six more. The full ledger is in §9.

---

## 2. What the raw signal actually contains (Q1, Q3, Q4)

### The dataset has no beat-to-beat information to recover

Measured across all 552 records:

| | Doppler US (n=413) | scalp FECG (n=137) |
|---|---:|---:|
| min quantization step | **0.25 bpm** | **0.25 bpm** |
| spectral power above 0.5 Hz | 1.67 % | 2.02 % |
| sample-to-sample sd | 2.84 bpm | **2.08 bpm** |
| missing fraction | 0.205 | **0.096** |

Three consequences, and they are physical, not methodological:

1. **98 % of FHR power lies below 0.5 Hz.** There is almost nothing in the band
   where beat-to-beat morphology would live.
2. **Quantization is 0.25 bpm for every record regardless of sensor.** CTU-UHB
   does not distribute RR intervals; the 4 Hz series is what was published. True
   beat-to-beat detail was removed *before* the data reached this project — no
   preprocessing change can recover it.
3. Doppler records have *higher* sample-to-sample variance than scalp-electrode
   records (2.84 vs 2.08). The extra high-frequency content in the majority of
   the cohort is **autocorrelation artifact, not physiology**.

### And the direct test agrees

18 features the 19 descriptors do not contain — VLF/LF/MF/HF band powers, LF/HF,
Poincaré SD1/SD2, **sample entropy**, **DFA α1 and α2**, Dawes-Redman STV,
skewness, kurtosis, zero-crossing rate — were computed on every window.

Added to the baseline: **−0.0152**. Alone: 0.6435.

The only ones carrying univariate signal are re-measurements of variability
amplitude the baseline already has:

| feature | patient-level AUROC |
|---|---:|
| VLF power | 0.6237 |
| log total power | 0.6202 |
| LTV (min–max) | 0.6199 |
| Dawes-Redman STV | 0.6160 |

**Sample entropy and DFA — the genuinely new nonlinear measures — do not reach
the top eight.** The "beat-to-beat morphology" route is closed, on both physical
and empirical grounds.

---

## 3. The record geometry nobody had decoded (Q9)

`Pos. II.st.` is not an independent annotation. It is **exactly determined**:

```
Pos. II.st.  =  record_length  −  II.stage × 240
```

489 of 506 exact; the other 17 differ by 16–20 samples (rounding). Therefore:

> **Every CTU-UHB record is 60 minutes of first stage followed by the entire
> second stage.** Record length is not a recording choice — it is
> `60 + II.stage` minutes.

This has three consequences the project had not registered.

**(a) The current 60-minute crop mixes two physiologically distinct regimes in a
patient-varying ratio.** Mean second-stage share of the crop is 20.8 %, ranging
0 → 50 %:

| stage-II share of the 60-min crop | patients |
|---|---:|
| < 17 % | 320 |
| 17–34 % | 162 |
| 34–51 % | 70 |

**(b) It explains the phase-6 "time confound."** The finding that risk score
doubles from the early to the late third *even in healthy babies* is the model
detecting the stage I → II transition, which occurs at a different point in every
recording. It was inferring a boundary it was never given.

**(c) Second-stage duration is a real risk gradient** — and it is currently
unused:

| II.stage | n | prevalence | mean pH |
|---|---:|---:|---:|
| 0–5 min | 113 | **5.3 %** | 7.285 |
| 6–10 min | 161 | 17.4 % | 7.247 |
| 11–20 min | 162 | 24.7 % | 7.204 |
| 21–30 min | 70 | **34.3 %** | 7.177 |

ρ(II.stage, pH) = **−0.296** (p = 1.2e-12), AUROC 0.5943 alone.

**That single header scalar has almost the same rank correlation with pH as the
entire deep CTG model** (ρ = −0.309, phase 7).

But it is **not prospectively available** — second-stage duration is known only
once the second stage has ended, i.e. at delivery. It cannot be an input to a
prediction model. Its prospective analogue (*time elapsed in second stage so
far*) is legitimate, and was tested: **+0.0005**.

`pipeline_clinical.py` already computes and stores `w_is_second_stage`. **No
model consumes it.** It is dead code.

---

## 4. Two phase-7 claims that do not survive

### 4.1 The maternal-covariate experiment is negative

Phase 7 recommended adding the 13 maternal/clinical covariates as one half of its
"ONE TARGETED EXPERIMENT". Run: **0.5886 alone, −0.0175 added.**

The reason is visible in the metadata. Of 35 header fields, five are identically
zero for all 552 records — `NICU days`, `Seizures`, `HIE`, `Intubation`,
`Main diag.` are `!NotReadyYet!` placeholders, not data. Several more are too
rare to carry weight against 110 positives:

| field | positive count |
|---|---:|
| Pyrexia | 6 / 552 |
| CK/KP | 14 / 552 |
| Preeclampsia | 17 / 552 |
| Diabetes | 37 / 552 |
| Hypertension | 44 / 552 |

### 4.2 The "more recording helps" inference is inverted

Phase 7 argued that full-length recordings scoring 0.7672 vs 0.6735 (+0.094)
justifies recovering the 19 % of signal discarded by the 60-minute crop.

That does not follow. **Every patient yields at most 17 windows**, because the
crop is a fixed 60 minutes at a 2.5-minute stride. Fewer than 17 means the
*quality gate* removed some. Window count is a **signal-quality** measure, not a
recording-length measure — and the correlation actually runs the wrong way:

```
corr(window count, record length) = −0.2207
```

Longer recordings have *fewer* usable windows, because a longer record means a
longer second stage, and second-stage signal is noisier. The +0.094 stratum
effect is a quality effect measured entirely *within* the identical 60-minute
crop, and says nothing about recovering earlier signal.

Separately: the discarded 19 % is the **earliest** part of the first stage — the
segment furthest in time from the outcome, and physiologically the least
informative. Its stated justification is void. **The experiment was run anyway
(§8b) and is negative.**

---

## 5. The target is not the problem (Q8)

Same 19 features, same patient partition, seven outcome definitions:

| target | AUROC |
|---|---:|
| pH ≤ 7.15 (current) | 0.7271 |
| pH ≤ 7.05 (severe) | 0.7311 |
| BDecf ≥ 12 (metabolic acidaemia) | 0.7294 |
| BDecf ≥ 10 | 0.6701 |
| BDecf ≥ 8 | 0.7287 |
| pH ≤ 7.15 **and** BDecf ≥ 8 (concordant) | 0.7285 |
| **Apgar5 < 7** (neonatal depression) | **0.5423** |

Every acid-base formulation lands at **0.727–0.731**. Switching from pH to base
deficit — which isolates *metabolic* acidosis, the physiology CTG is supposed to
reflect — changes nothing. Requiring pH and BDecf to agree, which removes the
label-noise objection, changes nothing.

Apgar5 collapses to 0.54, confirming the features are specific to acid-base
status rather than generically predicting "bad outcome."

> **~0.73 is a property of the CTG → fetal acid-base channel itself, not of the
> pH ≤ 7.15 threshold.** Label reformulation is closed.

---

## 6. Power — and a correction to the "small dataset" assumption

Hanley–McNeil, at the cohort's 20.1 % prevalence:

| quantity | value |
|---|---:|
| SE at true AUROC 0.85, n=547 | 0.0239 |
| **95 % CI half-width at 0.85** | **0.0469** |
| implied lower bound if 0.85 were achieved | **0.803** |
| n needed to detect a paired +0.12 at 80 % power | **157 patients** |
| n needed to detect a paired +0.02 | 5 623 patients |

**CTU-UHB is not too small to demonstrate 0.85.** If a model genuinely reached
0.85, this cohort would support it with a lower bound above 0.80, and a +0.12
jump is detectable at less than a third of the current cohort.

The small-n problem is real only for *small* effects. It is not what stands
between this project and 0.85. The missing ingredient is signal, not sample size.

This retires "Goal B" as originally framed: the question was never whether
CTU-UHB could *demonstrate* 0.85.

---

## 7. Where 0.85 sits relative to the literature

From [literature_forensic_audit.md](literature_forensic_audit.md): the best
published CTU-UHB results are **0.822–0.83**, and this project's own audit
reconciles them with honest patient-level ~0.73–0.78 through three measurable
methodological choices (epoch selection on the reported fold, window- rather than
patient-level units, and 2.7× understated uncertainty).

So a patient-level 0.85 on CTU-UHB would **exceed every published result on this
dataset**, on a protocol stricter than any of them used. That is not a reason to
abandon the target. It is the correct calibration of what achieving it means.

---

## 8. What remains genuinely untested

Everything in §1 shares three properties: linear models, window-level features,
max aggregation. That bounds the claim. Three things change one of those
properties and have never been run.

| # | route | what is new | prior |
|---|---|---|---|
| **1** | **Per-contraction response** | changes the **unit of analysis** from a 20-min window to a single contraction | **RUN — negative, §8a** |
| **2** | **Stage-aligned trajectory + learned aggregator** | changes the **time axis** and replaces max-aggregation | **RUN in combination — negative, §8a** |
| **3** | Full-length recording | the 19 % discarded signal — cheap, but justification void (§4.2) | **RUN — negative, §8b** |

### Route 1 — per-contraction FHR response (fetal reserve)

The uterine-contraction channel is **usable in 496/552 records (89.9 %)**, median
4.82 contractions per 10 min — roughly 29 contractions per recording. The
baseline spends only 5 of its 19 features on UC, and all five are window-level
aggregates (count, tachysystole, mean amplitude, lag, coupling).

What has never been built is the contraction as the unit: for each contraction,
the FHR response to *that* contraction — depth normalized by contraction
amplitude, lag to nadir, recovery time constant, response area, overshoot — and
then **the trajectory of that response across successive contractions within a
patient**.

This is the physiological definition of fetal reserve: a healthy fetus recovers
fully from each contraction; a compromised one shows progressive deterioration.
The statement *"the 12th contraction produced a deeper, slower-recovering
deceleration than the 3rd"* is **inexpressible** in the current 19 descriptors,
and is not a restatement of anything tested in §1.

Honest prior: **+0.02 to +0.05**, not +0.12. It is the strongest remaining
hypothesis, not a likely route to 0.85 on its own.

### Route 2 — stage-aligned trajectory

Re-align every recording on the stage I → II transition (exactly computable,
§3) instead of on delivery time, and replace max-aggregation with a learned
patient-level aggregator over the sequence of window descriptors.

This is **not** another temporal architecture. The new content is the time axis
and the aggregator. Both are currently discarded: `w_is_second_stage` is dead
code, and max-aggregation throws away all patient-level structure.

Tempering evidence: the stage flag *as a feature* gave +0.0005. An alignment is
not a flag, but the prior should be set accordingly.

---

## 8a. Route 1 — RUN, and negative

Built and tested 2026-09-04:
[`src/preprocessing/contraction_response.py`](../src/preprocessing/contraction_response.py),
[`scripts/probe_contraction_trajectory.py`](../scripts/probe_contraction_trajectory.py),
[`scripts/probe_contraction_stacked.py`](../scripts/probe_contraction_stacked.py).
Features only, no network — the question was whether the *representation*
contains signal.

### The detector is clinically calibrated

Tuned on signal morphology only, never on labels:

| | measured | clinical norm |
|---|---|---|
| contraction duration | median 44 s (IQR 44–57) | 45–90 s |
| contraction rate | median 3.67 / 10 min (IQR 2.83–4.50) | 3–5 / 10 min |
| contractions per patient | median 21 | — |

18 of 547 patients yield no usable trajectory.

### The response measurement is physiologically coherent

Across 2 984 individual contraction responses:

| check | result | expected |
|---|---|---|
| ρ(contraction amplitude, decel depth) | **+0.128** (p = 1.9e-12) | > 0 ✓ |
| ρ(decel depth, recovery time) | **+0.631** (p ≈ 0) | > 0 ✓ |
| mean depth, recovered vs not | 30.6 vs 39.0 bpm | lower ✓ |

The instrument works. What follows is not an artifact of broken extraction.

### It adds nothing to the strongest baseline

Against the established window-level 0.7271, patient-level, frozen folds:

| model | AUROC | Δ |
|---|---:|---:|
| **19 descriptors, window-level + max (established)** | **0.7271** | — |
| baseline score alone (max + mean) | 0.7210 | −0.0061 |
| baseline + 45 trajectory features | 0.6512 | −0.0759 |
| baseline + top-10 trajectory (selected in-fold) | 0.6837 | −0.0434 |
| baseline + trajectory (gradient boosting) | 0.6665 | −0.0606 |
| baseline + stage-split trajectory | 0.6428 | −0.0843 |
| baseline + top-12 of (trajectory + stage-split) | 0.7153 | −0.0118 |
| baseline + `baseline_shift_sd` alone | 0.7204 | −0.0067 |

Complementarity was **tested, not assumed**: the Route 1 × Route 2 combination
(trajectory + stage-split, with in-fold selection) is −0.0118.

### Why — and this is the substantive finding

The deterioration hypothesis was tested directly, with no model involved:
per-patient late-third minus early-third, and per-patient slope across
contraction index, acidotic vs normal.

| deterioration statistic | normal | acidotic | p (MWU) | AUROC |
|---|---:|---:|---:|---:|
| depth, late − early | **+9.665** | **+8.696** | 0.515 | 0.479 |
| depth, trend per contraction | +0.663 | +0.614 | 0.359 | 0.471 |
| recovery time, trend | +0.279 | +0.360 | 0.130 | 0.548 |
| recovered, trend | −0.001 | −0.003 | 0.887 | 0.496 |
| fraction unrecovered | 0.294 | 0.313 | 0.505 | 0.521 |
| longest unrecovered run | 2.000 | 2.000 | 0.696 | 0.512 |

> **Both groups deteriorate, and they deteriorate by the same amount.**
> Decelerations deepen by ~9 bpm from the early to the late third of the hour in
> acidotic *and* normal fetuses alike.

Progressive deceleration deepening across contractions is a property of
**advancing labour**, not of fetal compromise. This is the same confound the
phase-6 audit found at the window level — "the model learned late labour and
applies it to healthy babies" — now measured at the contraction level and shown
to be non-differential. Not one of eleven deterioration statistics reaches
significance.

The trajectory features that *do* separate the groups are magnitude and
variability summaries, not deterioration:

| feature | normal | acidotic | p | AUROC |
|---|---:|---:|---:|---:|
| `baseline_shift_sd` | 13.257 | 17.452 | <0.001 | **0.662** |
| `area_mean` | 936.99 | 1083.22 | 0.001 | 0.608 |
| `depth_worst` | 74.745 | 78.749 | 0.032 | 0.567 |

`baseline_shift_sd` — the variability of the inter-contraction baseline shift —
is the single strongest new feature this project has produced at 0.662 alone.
It still adds **−0.0067**, because it re-measures what `decel_area` and
`decel_max_depth` already carry.

### Bounded claim

The trajectory is observed over the final 60 minutes only. Deterioration
operating on a longer timescale would be invisible here — but CTU-UHB contains
at most 30 additional minutes (§3), all of it earlier first stage.

**Route 1 is closed.** The hypothesis was well-posed, the instrument was
validated, and the specific physiological claim — that compromised fetuses show
*differential* deterioration across repeated contractions — is false in this
cohort.

---

## 8b. Route 3 — RUN, and negative

Built and tested 2026-09-04:
[`scripts/probe_full_recording.py`](../scripts/probe_full_recording.py). The
60-minute crop was removed and every record processed at full length with an
identical signal chain, window geometry, quality gate and descriptor set.

### The re-implementation is faithful

| | AUROC |
|---|---:|
| last-60 crop, max — this script | **0.7245** [0.667–0.777] |
| established `pipeline_clinical` result | 0.7271 |

Within 0.003. Everything below can be believed.

### The full recording does not help

11 378 windows total (8 471 inside the last hour, 366 wholly before it):

| model | AUROC | Δ |
|---|---:|---:|
| **last-60 crop, max (reproduction baseline)** | **0.7245** | — |
| full recording, max | 0.7132 | **−0.0113** |
| full recording, mean | 0.6674 | −0.0571 |
| last-60 crop, mean | 0.6769 | −0.0476 |
| pre-hour windows ONLY, max | 0.6635 | −0.0610 |
| full recording, max, bag fixed at 17 | 0.7086 | −0.0160 |

Only **144 of 547** patients have any window lying wholly before the final hour,
median 3 such windows — because the discarded portion has a median duration of
11.4 minutes against a 20-minute window.

### Using the full recording introduces a confound rather than removing one

Phase 7 reported that window count does not predict the label (AUROC 0.4657).
That held for the *cropped* data, where every patient has at most 17 windows. It
does **not** hold once the crop is removed:

```
windows per patient: median 21, range 8-29
AUROC(window count -> label) = 0.5751
```

Record length is `60 + II.stage` (§3), so at full length the bag size *encodes
second-stage duration* — a retrospective variable not available at prediction
time (§3). Full-length max-aggregation is therefore mildly **optimistic**, and it
is still negative. Fixing the bag at 17 windows removes the leak and the score
falls further, to 0.7086.

**Route 3 is closed**, and the direction of its residual bias means the null is
safe rather than marginal.

---

## 9. Decision

**All three routes have now been run. All three are negative.** Route 1 §8a,
Route 2 in combination with it §8a, Route 3 §8b. The only untested item left on
the board is self-supervised representation learning.

The empirical position, stated precisely:

> **Eight information sources have been tested against the established 0.7271
> on the frozen protocol. The best result is +0.0027. The range is −0.0759 to
> +0.0027. Not one approaches the +0.0642 bar, let alone +0.12.**

That is not the same as proving 0.85 unreachable, and this document does not
claim it. What it does establish is that the hypotheses available at the start of
phase 8 are exhausted, and that a further attempt needs a *new physiological
hypothesis* rather than a new model or a new aggregation of the same signal.

### Summary of everything tested in phase 8

| information source | best Δ vs 0.7271 | § |
|---|---:|---|
| missingness pattern | **+0.0027** | 1 |
| second-stage flag + elapsed time | +0.0005 | 1, 3 |
| contraction-response trajectory (best variant) | −0.0067 | 8a |
| full recording, max | −0.0113 | 8b |
| trajectory × stage-split combined | −0.0118 | 8a |
| retrospective covariates (*leaky*) | −0.0142 | 4.1 |
| HRV / spectral / nonlinear | −0.0152 | 2 |
| maternal + clinical covariates | −0.0175 | 4.1 |

Explicitly closed by this phase, with evidence, and not to be re-opened without
new argument:

- beat-to-beat / spectral / nonlinear HRV representations (§2)
- maternal and clinical covariates (§4.1)
- label and threshold reformulation, including base deficit (§5)
- missingness-pattern modelling (§1)
- second-stage timing as an input feature (§3)
- per-contraction response and its trajectory (§8a)
- the full-length recording / the discarded 19 % (§8b)
- deterioration across repeated contractions — **both groups deteriorate equally** (§8a)
- "the dataset is too small for 0.85" (§6)

### Reproducing this phase

All probes run from the repo root against the frozen protocol; HRV features cache
to `results/phase8/`.

```
python scripts/audit_signal_resolution.py        # §2 sensor / quantization / spectrum
python scripts/audit_uc_channel.py               # §8 UC usability
python scripts/extract_hrv_features.py           # caches results/phase8/hrv_*.npy
python scripts/probe_hrv_information.py          # §2 HRV table
python scripts/probe_metadata_covariates.py      # §4.1 covariates
python scripts/probe_target_formulation.py       # §5 target table
python scripts/probe_missingness_information.py  # §1 missingness row
```

### Caveat bounding every null above

All six probes used **linear** models over **window-level** features with **max**
aggregation. A different unit of analysis or a nonlinear aggregator could behave
differently — which is precisely why Routes 1 and 2 are defined by changing those
properties rather than by changing the encoder.
