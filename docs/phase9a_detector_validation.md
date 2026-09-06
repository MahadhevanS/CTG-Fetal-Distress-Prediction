# Phase 9A — validating the measurement instrument against expert consensus

Completed 2026-09-04. First use of data from outside CTU-UHB in this project.

**Question:** every one of the 19 clinical descriptors underpinning the 0.7271
baseline — and therefore every null result in Phases 5–8 — is produced by this
repo's own detectors, which had never been checked against expert ground truth.
FHRMA (Boudet et al., distributed inside CTGDL v5, open access) provides
expert-consensus baseline / acceleration / deceleration annotations on 156
recordings. Are our detectors right?

**FHRMA carries no pH outcome.** This validates the instrument only. It makes no
claim about acidaemia prediction.

Code: [`scripts/validate_detectors_fhrma.py`](../scripts/validate_detectors_fhrma.py),
[`scripts/compare_baseline_estimators_fhrma.py`](../scripts/compare_baseline_estimators_fhrma.py),
[`scripts/probe_baseline_repair_impact.py`](../scripts/probe_baseline_repair_impact.py).
452 analysable 20-minute windows from 152 recordings.

---

## 1. Executive verdict

**The detectors disagree substantially with expert consensus. Repairing that
disagreement makes outcome prediction markedly worse.**

| | |
|---|---|
| acceleration detection vs expert | sensitivity **0.204**, precision 0.545, **F1 0.297** |
| deceleration detection vs expert | sensitivity 0.604, precision **0.338**, F1 0.433 |
| baseline agreement | median abs diff **3.93 bpm**; within 5 bpm only **70.9 %** |
| repaired baseline → event F1 | acc 0.297 → **0.464**, dec 0.433 → **0.497** |
| **repaired baseline → patient AUROC** | **0.7271 → 0.6300 (−0.0972)** |

So the instrument is measurably not doing what a clinician would do — and making
it more clinician-like **costs ~0.10 AUROC**. Expert agreement and outcome
prediction are **not** aligned objectives on this task.

This *strengthens* Phases 5–8 rather than undermining them: the instrument was
audited, its disagreement quantified, and the disagreement shown not to be
costing predictive performance.

---

## 2. The defect is structural

`calculate_iterative_baseline` ends with:

```python
return np.full_like(fhr, baseline)      # baseline rounded to nearest 5 bpm
```

**One constant per 20-minute window.** Measured against expert annotation, the
baseline moves *within* such a window by:

| statistic | expert baseline movement in 20 min |
|---|---:|
| median range (max − min) | **14.2 bpm** |
| IQR | 6.8 – 29.7 bpm |
| 90th percentile | 56.5 bpm |

| windows where the expert baseline moves by more than… | share |
|---|---:|
| 5 bpm | 78.6 % |
| 10 bpm | 64.5 % |
| **15 bpm — the FIGO event threshold itself** | **47.1 %** |

A constant can represent none of this.

### Consequences measured

| | expert events | ours | sensitivity | precision | F1 |
|---|---:|---:|---:|---:|---:|
| accelerations | 882 | 330 | 0.204 | 0.545 | 0.297 |
| decelerations | 1177 | **2105** | 0.604 | 0.338 | 0.433 |

Accelerations are **under-detected 2.7×**; decelerations are **over-detected
1.8×**.

### The ablation localises the fault to the baseline, not the FIGO rule

Same 15 bpm / 15 s rule, but run on the **expert** baseline:

| | our baseline | expert baseline | Δ |
|---|---:|---:|---:|
| acceleration F1 | 0.297 | **0.598** | **+0.301** |
| deceleration F1 | 0.433 | **0.616** | **+0.182** |

Roughly half the event-detection error comes from the baseline estimator. The
event rule itself is sound.

---

## 3. Two candidate repairs

### 3.1 The unused ALS function is not the answer

`asymmetric_least_squares_baseline` sits in `baseline.py` imported by nothing.
Tested at four smoothness settings:

| estimator | MAD (bpm) | within 5 bpm | acc F1 | dec F1 |
|---|---:|---:|---:|---:|
| current (constant) | **3.93** | **70.9 %** | 0.297 | 0.433 |
| ALS λ=1e4 | 12.00 | 48.9 % | 0.000 | 0.000 |
| ALS λ=1e5 | 11.56 | 50.3 % | 0.000 | 0.000 |
| ALS λ=1e7 | 10.78 | 52.8 % | 0.022 | 0.093 |

With `p = 0.5` it is a **smoother, not a baseline estimator** — it hugs the
signal so closely that excursions never reach 15 bpm and essentially no events
are detected. Do not wire it in.

### 3.2 A rolling version of the existing rule does improve event detection

`rolling_iterative_baseline` (added to `baseline.py`) applies the identical
iterative 15 bpm exclusion in a centred rolling window. Window length chosen on
FHRMA event F1 — **no CTU-UHB outcome label is involved**, so it cannot leak.

| estimator | MAD | within 5 bpm | acc F1 | dec F1 |
|---|---:|---:|---:|---:|
| current (constant) | **3.93** | **70.9 %** | 0.297 | 0.433 |
| **rolling 5 min** | 9.23 | 61.9 % | **0.464** | **0.497** |
| rolling 10 min | 6.04 | 65.4 % | 0.335 | 0.431 |
| rolling 15 min | 4.46 | 66.0 % | 0.274 | 0.477 |
| expert (upper bound) | 0.00 | 100 % | 0.598 | 0.616 |

Note the trade-off, which is real: rolling-5min is **better at events** and
**worse point-wise**. It is not a strict improvement, and it does not reach the
expert bound.

---

## 4. The decisive test — and it goes the other way

Descriptors recomputed on all 8 517 CTU-UHB windows with each estimator. Frozen
protocol, unchanged: 547 patients, `folds.json`, patient-level max aggregation.

| | patient AUROC | window AUROC |
|---|---:|---:|
| **19 descriptors, constant baseline (as shipped)** | **0.7271** [0.670–0.779] | 0.6179 |
| 19 descriptors, rolling 5-min baseline (repaired) | **0.6300** [0.571–0.691] | 0.5957 |
| both concatenated (38 features) | 0.7172 [0.659–0.770] | — |

> **−0.0972.** The estimator that agrees better with expert event annotation is
> a substantially worse predictor of fetal acidaemia. The concatenation of both
> feature sets is also worse than the constant baseline alone, so the repaired
> features add nothing even as a supplement.

The constant-baseline run reproduces 0.7271 exactly, validating the
recomputation.

### Why — the mechanism is visible in one feature

| descriptor | constant mean | rolling mean | correlation |
|---|---:|---:|---:|
| baseline | 136.42 | 129.44 | 0.704 |
| stv | 0.869 | 0.869 | 1.000 |
| ltv | 26.39 | 25.86 | 0.967 |
| accels | 1.540 | 1.433 | 0.696 |
| dec_early | 0.329 | 0.245 | 0.741 |
| dec_late | 0.244 | 0.157 | 0.745 |
| dec_variable | 3.441 | 3.027 | 0.842 |
| **dec_prolonged** | **0.212** | **0.003** | **0.102** |

A drift-tracking baseline **absorbs a sustained FHR drop into the baseline
itself**, so it stops being a deceleration. `dec_prolonged` collapses by 98 %.

But a sustained drop is clinically informative *precisely because* it is a
departure from the earlier level. The constant baseline anchors that reference
and keeps the deviation visible; the rolling baseline follows the fetus down and
erases the evidence.

**What looked like a defect is, for this task, load-bearing.**

---

## 5. What this licenses

**Supportable**

1. The descriptor extractors were validated against expert consensus — the first
   external validation in this project — and disagree substantially
   (acceleration F1 0.297, deceleration over-detection 1.8×).
2. About half that disagreement is attributable to the baseline estimator, not
   the FIGO event rule (expert-baseline ablation: +0.301 / +0.182 F1).
3. Correcting the baseline toward expert behaviour **reduces** patient-level
   AUROC by 0.0972 — far outside the 0.0642 MDE — via a specific, identified
   mechanism (drift absorption erasing prolonged decelerations).
4. Therefore Phases 5–8 are **not** invalidated by detector error; the audit was
   run and the conclusion is that the defect does not cost predictive
   performance.

**Not supportable**

- that our detectors are "correct" — they are not, by expert standards;
- that no better baseline estimator exists — only two repairs were tested, and
  the expert upper bound (acc F1 0.598) shows real headroom remains. What is
  established is that *drift-tracking* repairs backfire, for a reason that would
  apply to any of them;
- any claim about acidaemia prediction derived from FHRMA — it has no pH.

---

## 6. Implication for Phase 9B

FHRMA's annotation value is now spent, and it produced a negative-but-useful
result. Its remaining use is as **+135 recordings of pre-training corpus**
(≈ +24 %), which [phase9_data_availability.md](phase9_data_availability.md)
already prices as unlikely on its own to overturn the August SSL regression of
−0.0262.

An unanticipated finding worth carrying forward: **expert-agreement is not a
proxy objective for outcome prediction here.** Any future auxiliary task built on
expert morphology labels — including multi-task SSL over FHRMA annotations —
should be treated as an open question rather than an obvious gain, because the
one time this project optimised toward expert behaviour, prediction got worse.
