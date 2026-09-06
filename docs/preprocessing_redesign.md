# Preprocessing redesign — building a clinically valid dataset

Written 2026-09-02, after the ceiling analysis in
[auroc_ceiling_analysis.md](auroc_ceiling_analysis.md). That document explains
why the model stops at ~0.81. This one is about the deeper problem it exposed:
**the current target is not a disease.**

---

## 1. The finding that should drive everything

The dataset labels a baby "fetal distress" when umbilical pH ≤ 7.15. That is
113 of 552 deliveries (20.5%). Looking at what actually happened to those 113
babies:

| of the 113 labelled DISTRESSED | value |
|---|---|
| median pH | 7.100 |
| median base deficit (BDecf) | **8.19 mmol/L** — metabolic acidemia needs ≥ 12 |
| median Apgar at 5 min | **8** |
| had BDecf ≥ 12 (true metabolic acidemia) | **18 / 113 (16%)** |
| had Apgar5 < 7 (depressed newborn) | **11 / 113 (10%)** |
| had Apgar5 ≥ 9 — **vigorous, healthy newborn** | **53 / 113 (47%)** |

**Nearly half the positive class walked away completely healthy.** A transient
dip to pH 7.10 during second-stage labour is a normal physiological finding, not
a pathology. The model is being trained to detect something that, in most of its
positive examples, had no clinical consequence at all.

This explains, without needing any modelling argument, why:

- the model scores only **0.6086** on "is this baby acidotic" — the target is
  weakly related to fetal compromise;
- healthy-outcome patients get flagged constantly (patient 1119, pH 7.34, every
  window flagged) — the model learned features that genuinely occur in late
  labour in *everyone*;
- FIGO rules (0.6179) match the deep network on the patient outcome — neither is
  detecting disease, because the label barely marks disease.

**No architecture, loss function, or amount of training fixes a target that
isn't the condition you care about.**

---

## 2. The four problems, ranked by clinical seriousness

| # | Problem | Evidence | Severity |
|---|---|---|---|
| **P1** | The positive class is mostly healthy babies | 47% had Apgar5 ≥ 9; only 16% had BD ≥ 12 | **Critical** |
| **P2** | The label uses information unavailable at inference | horizon = last 30 min *before delivery*; `sig2birth = 0` for all records, so recordings end at birth | **Critical** |
| **P3** | The quality gate destroys the positive class | acidotic patients keep 62.6% of horizon windows vs 86.2% pre-horizon (−23.6pp); 27/108 lose *all* positives | High |
| **P4** | Hard threshold on a physiological continuum | 18% of patients within ±0.05 pH of the 7.15 cut | Moderate |

**P2 deserves emphasis** because it is a correctness bug, not a tuning choice.
`within_horizon = (start >= signal_length - 7200)` is defined relative to the
*end of the recording*, and the recordings end at delivery. A device in a labour
ward does not know how long labour has left to run. **The label is computed from
the future.** Even a model that learned it perfectly could not be deployed,
because the quantity it predicts is undefined at inference time.

---

## 3. What the dataset actually offers

CTU-UHB carries far more outcome information than the pipeline currently uses:

| field | n | range | use |
|---|---|---|---|
| `ph` | 552 | 6.85 – 7.47 | currently the only target |
| `bdecf` | 541 | −3.4 – 26.11 | **base deficit — the metabolic acidemia marker** |
| `be` | 541 | −26.8 – −0.2 | base excess |
| `apgar1`, `apgar5` | 552 | 1–10, 4–10 | **newborn clinical condition** |
| `pos._ii.st.` | **506** | second-stage onset, in samples | **a clinically real time landmark** |
| `ii.stage` | 552 | median 10 min, max 30 | second-stage duration |
| `meconium`, `pyrexia`, `preeclampsia`, `diabetes`, `hypertension` | 552 | 0/1 | risk factors for stratification |

Two of these change the design:

- **`bdecf` and `apgar5` let us define a target that means something.** ACOG's
  criterion for intrapartum hypoxic injury is pH < 7.00 **and** base deficit ≥ 12.
- **`pos._ii.st.` gives a legitimate temporal anchor.** Second-stage onset is
  known prospectively in a labour ward; "30 minutes before delivery" is not.

**Not available:** `nicu_days`, `seizures`, `hie`, `intubation` are all zero —
the CSV marks them `!notreadyyet!`. Hard neonatal outcomes cannot be used, and
that limit should be stated in any writeup.

---

## 4. The redesign

### 4.1 Target — keep pH ≤ 7.15 primary, add a composite secondary

> **Revised 2026-09-02 after the literature review**
> ([literature_preprocessing_comparison.md](literature_preprocessing_comparison.md)).
> An earlier draft of this section said to *replace* the pH target. That was too
> strong. pH < 7.15 giving 113 positives (20.5%) is exactly what the benchmarked
> paper and the 2026 foundation-model paper use, and abandoning it forfeits
> comparability with every published number on this dataset. Report **both**
> targets instead of swapping one for the other.

**Primary (unchanged): `pH <= 7.15`** — for comparability with Dang et al. and
the wider CTU-UHB literature.

**Secondary (new): a composite adverse outcome**

```
y_adverse = (pH <= 7.05) OR (BDecf >= 12) OR (Apgar5 < 7)
```

| definition | patients | % |
|---|---|---|
| pH ≤ 7.15 (current) | 113 | 20.5% |
| pH < 7.00 AND BD ≥ 12 (strict ACOG) | 13 | 2.4% — too few to train |
| **pH ≤ 7.05 OR BD ≥ 12 OR Apgar5 < 7** | **56** | **10.1%** |

Rationale for each arm: pH ≤ 7.05 is moderate-to-severe acidemia; BD ≥ 12 is
metabolic (not respiratory) acidemia, which is what indicates real oxygen debt;
Apgar5 < 7 catches the **8 babies who were clinically depressed despite a normal
pH** — cases the pH-only label misses entirely.

Also emit `ph`, `bdecf`, `apgar5` as continuous columns so the target can later
be made ordinal or regression-based (addresses **P4**), and keep `pH ≤ 7.15` as
a legacy target so existing results stay comparable.

### 4.2 Window labelling — delete the horizon rule

```python
# DELETE:
within_horizon = (start >= horizon_start_idx)
window_label   = int(is_distress and within_horizon)

# REPLACE WITH:
window_label   = int(is_adverse_outcome)          # patient label, every window
```

This removes **P2** entirely. Every window of an adverse-outcome labour is
positive; the model is asked "is this labour heading for a bad outcome," which
is well-posed at any moment and is the actual clinical question.

Keep `minutes_before_end` and `is_second_stage` (from `pos._ii.st.`) **in the
metadata, never as labels** — so the time-confound audit in
`auroc_ceiling_analysis.md` can be re-run on the new dataset to confirm the
confound is gone.

The cost is honest label noise: an early window in a labour that later goes bad
may look genuinely normal. That is what MIL is for — which the substrate was
already built to support (`y_patient` exists in the current tensors).

### 4.3 Quality — stop deleting the positive class

The literature review sharpened this one: prior work drops **recordings** with
>50% missing (Dang et al.; the dataset's own selection used >50% quality per
30-min window). This project drops **windows** at >30%. It is both stricter and
— because it acts per window — biased in time, letting a patient lose only their
late windows. Recording-level filtering is blunter but unbiased within a patient.

Replace the hard `max_missing_ratio=0.30` exclusion with:

1. **A third input channel: the missingness mask** (binary, taken *before*
   interpolation). Currently the pipeline interpolates gaps and hands the model
   a clean-looking trace with no indication which samples were invented. A
   clinician sees signal loss on the strip; the model should too.
2. **Quality as a loss weight, not a filter.** Keep windows up to ~50% missing
   and down-weight them, instead of deleting them.
3. **Never silently drop a patient.** Emit a per-patient flag. Right now 27 of
   108 acidotic patients contribute zero positive windows and nothing says so.

Expected effect: acidotic horizon windows recover from 338 to up to 540 (+60%),
directly attacking the stated binding constraint of too few positives.

### 4.4 Evaluation — report at the patient level

The clinical unit is the patient, not the 20-minute window. Window-level metrics
on overlapping windows (87.5% shared signal at a 2.5-min stride) also badly
overstate the effective sample size. Report patient-level AUROC/AUPRC as primary
with bootstrap CIs over patients; keep window-level as secondary.

### 4.5 Re-split

Re-stratify the patient splits on the new composite target, and stratify on
gestational age and delivery type as well.

---

## 5. The honest trade-off

A clinically valid target roughly **halves the positive class**:

| split | patients | pH ≤ 7.15 | composite |
|---|---|---|---|
| train | 379 | 75 | **34** |
| val | 83 | 17 | **8** |
| test | 82 | 16 | **10** |
| **total** | **544** | **108** | **52** |

Ten positive patients in test is thin. Confidence intervals will be wide and
should be reported as such. **This is a real cost and it must be stated up front
rather than discovered at review.**

But the alternative is worse: continuing to optimise a metric on a label where
47% of the positives were healthy babies produces a number that will not survive
clinical scrutiny, and a device that alarms on normal labours. The AUROC will
very likely *drop* when the target becomes harder and more honest. **That drop is
the project getting more truthful, not less capable** — and it should be framed
that way in the writeup.

Mitigations for the sample-size loss:
- Use the **continuous** pH/BD/Apgar as auxiliary regression targets so the
  gradient signal is not limited to 52 binary labels.
- The quality-gate fix (4.3) recovers windows for the positives that remain.
- Pool across seeds and report paired replication, as this project already does.

---

## 6. Implementation plan

Write `src/preprocessing/pipeline_clinical.py` alongside the existing pipelines
(never in place — `pipeline.py` and `pipeline_mil.py` stay byte-reproducible, as
this project has consistently done). It should:

1. Build the composite target and emit `ph`, `bdecf`, `apgar5` continuously.
2. Drop the horizon rule; label every window with the patient outcome.
3. Emit `minutes_before_end`, `is_second_stage`, `quality` per window as
   metadata.
4. Add the missingness mask as input channel 3 → `X` becomes `(N, 3, 4800)`.
5. Relax the quality gate to 0.50 with weighting; emit dropped-patient flags.
6. Re-stratify splits on the composite target.

Then re-run the ceiling audit on the new tensors: **time-alone AUROC should fall
to ~0.50.** If it does not, the confound is still present and the redesign has
not worked. That is the acceptance test.

Sequence: (1) build the pipeline, (2) run the time-alone audit as a gate, (3)
retrain the delivered recipe, (4) only then compare architectures.
