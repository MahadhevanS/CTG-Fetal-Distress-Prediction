# Early warning — Stages 0–4 result

Written 2026-09-04. Ruleset `figo2015-seq-v3`, folds reused unchanged from
the detection track (`data/processed_figo/folds.json`).

```bash
python scripts/figo_ew_baselines.py
```

## Verdict

**The task is not predictable, and §8 establishes why: two of the three
concepts the label depends on are memoryless at a 30-minute horizon.**

Best non-clock arm AUROC 0.5842 against a clock at 0.5681. Given the TRUE
current concepts — perfect perception of the present — the future label is
predicted at **0.4353**, i.e. chance. There is no perception gap left to
close, so raw FHR and a temporal network cannot help.

The Stage 9 gate reads **< 0.70 → weak, do not tune**, and Stages 5–8 (raw
FHR, KI temporal, GRU) were explicitly gated behind "the structured models
demonstrate signal". They do not. The disciplined action is to stop here.

Unlike the pH work, this null is **explained** rather than bare (§4), and the
explanation was tested rather than asserted (§5).

## 0. The frozen task

From an epoch classified **Normal**, does the CTG become **Abnormal**
(Suspicious or Pathological) within the next 30 minutes?

| | |
|---|---:|
| anchors | **1,122** |
| positives | **703 (62.7%)** |
| patients | 454 |
| horizon | 3 × 10-minute epochs |

Three leakage rules, enforced in `src/figo_state/earlywarning.py`:

1. **A negative requires a fully observed future** — all three horizon epochs
   present *and* readable. Otherwise "did not deteriorate" can mean "we
   stopped recording", and recordings end at delivery.
2. **Inclusion does not depend on the outcome** — the same observability test
   is applied to positives. An earlier version of a deterioration target in
   this project censored only negatives, and the epoch index alone then
   scored 0.8321.
3. **No feature comes from after the anchor**, and neither
   `n_epochs_in_record` nor `minutes_before_end` is ever a feature — a labour
   ward does not know when labour will end.

**Correction to the planning figures.** The plan quoted 1,251 anchors / 706
positives / 362 patients and a clock baseline of **0.638**. Those came from a
pre-v3 build. On the current ruleset the cohort is 1,122 / 703 / 454 and the
clock is **0.5681**. The hurdle is lower than expected, which makes the null
below more striking, not less.

## 1–3. Results

Stage 1, full cohort (n = 1,122):

| arm | AUROC | 95% CI | AUPRC |
|---|---:|---|---:|
| current state | *undefined* — every anchor is Normal by construction | | |
| **clock: epoch index alone** | **0.5681** | 0.528–0.608 | 0.6727 |
| anchor descriptors → LR | **0.5812** | 0.542–0.618 | 0.6891 |
| anchor descriptors → GBM | 0.5768 | 0.540–0.613 | 0.6776 |
| anchor descriptors → tree(d=4) | 0.5487 | 0.511–0.587 | 0.6622 |

The best model beats a clock by **+0.013**.

Stages 2–3, on a cohort fixed at ≥2 prior epochs (n = 575, 67.3% positive,
361 patients) so context length is not confounded with anchor position:

| arm | AUROC | 95% CI |
|---|---:|---|
| clock | 0.5644 | 0.520–0.611 |
| anchor descriptors → LR (context 0) | 0.5646 | 0.515–0.616 |
| anchor descriptors → GBM (context 0) | 0.5277 | 0.474–0.580 |
| **history 10 min → LR** | **0.5842** | 0.533–0.633 |
| history 10 min → GBM | 0.5323 | 0.479–0.584 |
| history 20 min → LR | 0.5579 | 0.503–0.612 |
| history 20 min → GBM | 0.5395 | 0.484–0.596 |

## 4. Stage 4 — does trajectory information help? No.

The central hypothesis of the plan was that deterioration is encoded in the
trajectory rather than the instantaneous state. Paired, same cohort, same
folds:

| model | context | Δ vs context 0 | 95% CI | |
|---|---|---:|---|---|
| LR | 10 min | +0.0189 | [−0.0253, +0.0651] | crosses 0 |
| LR | 20 min | −0.0071 | [−0.0554, +0.0408] | crosses 0 |
| GBM | 10 min | +0.0043 | [−0.0471, +0.0563] | crosses 0 |
| GBM | 20 min | +0.0112 | [−0.0454, +0.0682] | crosses 0 |

**All four cross zero.** Level, trend, instability and delta over the
descriptors, plus FIGO-state persistence over the window (fraction of recent
epochs Suspicious/Pathological/unreadable, number of state changes, length of
the current Normal run), add nothing.

This is the third independent time this project has reached that conclusion.
`temporal_modelling_feasibility.md` found every order-invariant statistic
beat every order-dependent one on the pH task, and
`audit_evaluation_protocol.py`'s representation probe found trajectory slopes
*lowered* patient AUROC. The cohort and the target are different here; the
answer is the same.

## 5. Why — the label is a threshold coin-flip

The interesting part. Decomposing the 703 deterioration events by which
normality criterion fails in the first abnormal epoch:

| criterion that fails | share of events |
|---|---:|
| **repetitive decelerations** | **69.8%** |
| variability outside 5–25 bpm | 26.3% |
| baseline outside 110–160 | 11.2% |
| **repetitive decelerations ALONE** | **63.7%** |

Now look at how marginal those crossings are:

* "Repetitive" means decelerations accompanying **more than 50% of
  contractions**. The median epoch has **4 contractions**, and 53% have ≤4.
  On 4 contractions the test is decided by whether **2 or 3** of them carry a
  deceleration — **a single deceleration**.
* Of the variability-driven events, median variability is **4.3 bpm** against
  a 5 bpm threshold, and **58.4% sit within 1 bpm of the 5/25 boundary**.
* **50.9% of Normal → Abnormal transitions revert to Normal in the very next
  epoch** (230 of 452).

So the target is, in the majority of cases, a threshold crossing decided by
one deceleration pairing or not, on a detector measured at precision 0.509
against expert consensus, half of which reverts within ten minutes. AUROC
0.58 is what predicting that looks like.

## 6. The obvious fix was tested, and it does not work

If half the events are transient flickers, requiring the abnormality to
**persist** should sharpen the target — and FIGO's own criteria use
persistence qualifiers throughout, so this is guideline-consistent rather
than label-fitting.

| target | n | positives | prevalence | clock | descriptors → LR |
|---|---:|---:|---:|---:|---:|
| any abnormal epoch (original) | 1,122 | 703 | 62.7% | 0.5681 | **0.5812** |
| abnormal **sustained ≥2 epochs** | 784 | 230 | 29.3% | 0.5330 | **0.5332** |

**Worse, not better** — and now indistinguishable from its own clock. So
transience is not the cause. Sustained deterioration is, if anything, *less*
predictable from the current state than transient deterioration is. The null
survives the most obvious repair, which is what makes it worth reporting.

## 7. What this does and does not license

**Supportable**

1. From a Normal epoch, neither the current FIGO descriptors nor 10–20
   minutes of their trajectory predicts abnormality within 30 minutes better
   than AUROC ≈0.58, barely above a clock at 0.57.
2. Explicit trajectory features (level, trend, instability, delta, state
   persistence) add nothing measurable — four paired comparisons, all
   crossing zero.
3. The dominant deterioration event is a marginal threshold crossing on a
   noisy detector, and 51% of such events revert within one epoch.
4. Requiring persistence does not rescue it.

**Not supportable**

- that fetal deterioration is unpredictable in general. This tests one
  operationalisation on one 552-recording corpus with a rule-derived label,
  not the clinical question.
- that raw FHR would fail too (Stage 5 was not run). But note what it would
  have to do: predict which individual contraction, up to 30 minutes ahead,
  will carry a deceleration. The diagnosis in §5 makes that look irreducible
  rather than merely hard.
- any claim about acidaemia. This target is CTG morphology, not outcome.

## 8. TARGET-INFORMATION AUDIT — why, mechanistically

`python scripts/figo_ew_target_audit.py`. Run before any model, to establish
whether ANY clinically sensible formulation on this corpus carries signal.

### 8.1 The decisive finding: two of the three concepts are memoryless

The label is exactly NOT(baseline_normal AND variability_normal AND
no_repetitive_decels). So the future label is predictable only to the extent
those three concepts persist. Measured across all epoch pairs:

| concept | r at 10 min | r at 20 min | r at 30 min |
|---|---:|---:|---:|
| **baseline_bpm** | **0.860** | **0.759** | **0.703** |
| variability_bpm | 0.214 | 0.149 | **0.100** |
| decel_repetitive | 0.174 | 0.110 | **0.075** |

And the binary flags themselves — does the flag now predict the flag later?

| flag | 10 min | 20 min | 30 min |
|---|---:|---:|---:|
| **baseline_normal** | **0.800** | 0.722 | **0.710** |
| variability_normal | 0.580 | 0.534 | **0.506** |
| no_repetitive_decels | 0.585 | 0.552 | **0.534** |

**Variability and repetitive decelerations are essentially memoryless at a
30-minute horizon — AUROC 0.506 and 0.534, which is chance.** Baseline, the
one concept that does persist (0.710), drives only **11.2%** of deterioration
events. The two that do not persist drive **96%** of them (69.8% repetitive,
26.3% variability).

That is a complete quantitative account of the 0.58 result. The target is
carried almost entirely by quantities that the present does not determine.
This is not a feature-engineering problem and not an architecture problem.

### 8.2 The oracle confirms it

Given the **true** current rule-concepts — a model with perfect perception of
the present — the future label is predicted at **AUROC 0.4353**, i.e. no
better than chance and slightly inverted under a linear model.

There is no perception gap left to close. A model that measured baseline,
variability and repetitiveness perfectly, right now, still could not say what
happens in 30 minutes.

### 8.3 No horizon and no severity definition rescues it

| target | n | positives | clock | descriptors → LR |
|---|---:|---:|---:|---:|
| any abnormal within 10 min | 1,838 | 539 | 0.5732 | 0.5908 |
| any abnormal within 20 min | 1,467 | 730 | 0.5675 | 0.5943 |
| any abnormal within 30 min | 1,122 | 703 | 0.5681 | 0.5812 |
| any abnormal within 40 min | 784 | 565 | 0.5727 | **0.6264** |
| ≥2 normality criteria fail | 1,122 | 72 | 0.5249 | 0.5818 |
| clear excursion (>1 bpm past threshold) | 1,122 | 139 | 0.6127 | 0.4692 |
| Pathological within 30 min | 1,122 | **13** | 0.6022 | 0.3501 *(n_pos<50, not trusted)* |

The strongest trustworthy target is **0.6264** at 40 minutes, still far below
the 0.70 gate, and its prevalence is 72% on the smallest anchor set.
Demanding an *unambiguous* event (≥2 criteria failing) does not help (0.5818);
demanding a *clear excursion* actively hurts (0.4692). Both severity-graded
arms are reported with their surviving class balance because this project has
previously produced a 0.9832 artefact by excluding ambiguous cases until 96%
of the positive class was gone.

### 8.4 Where the event happens

Of 703 positives, the first abnormal epoch is at t+1 in 42.7% of cases, t+2
in 33.0%, t+3 in 24.3%. Scoring each separately gives 0.5908 / 0.5689 /
0.5261 — the further ahead, the closer to chance, exactly as §8.1 predicts.

## 9. Recommendation

**Stop the early-warning track here**, per the pre-registered Stage 9 gate.
Stages 5–8 were conditional on structured models showing signal; they show
0.58 against a clock at 0.57.

The finding to carry forward is §5, and it is not confined to early warning:
**the >50%-of-contractions criterion is decided by a single deceleration in
the majority of epochs.** That same brittleness sits underneath the detection
track's ceiling — `figo_knowledge_infusion.md` §7 found `decel_repetitive`
recovered at only AUROC 0.755 and *unchanged* by making contractions
estimable (R² 0.017 → 0.580 bought +0.021 on the concept). Two tracks, two
independent routes, converging on the same component.

If the project continues on this dataset, the honest next question is not
another model. It is whether a *less brittle* repetitiveness measure — a
continuous deceleration burden rather than a >50% count on ~4 contractions —
produces a label that is both clinically faithful and estimable. That is a
measurement question, and it would need the same FHRMA-style external
validation the deceleration repair got.
