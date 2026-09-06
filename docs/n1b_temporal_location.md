# N1b — controlled temporal-location experiment: feasibility outcome

Written 2026-09-04. **The experiment as specified cannot be run on this dataset.**
This document records why, quantifies every feasible weakened variant, and states
the decision that follows.

---

## 1. Research question

> Does the CTG segment immediately preceding the final 60 minutes contain
> predictive information about the fetal-distress target that differs from the
> final 60-minute segment?

---

## 2. Exact cohort definition — the blocker

The specified comparison is:

- **N1b-0**: final 60 minutes, `[t_end − 60, t_end]`
- **N1b-1**: previous 60 minutes, `[t_end − 120, t_end − 60]`

N1b-1 requires **at least 120 minutes** of recording per patient.

Measured across all 552 CTU-UHB records (durations computed directly from the
raw `.dat` signals, not from metadata):

| statistic | minutes |
|---|---:|
| minimum | 60 |
| 10th percentile | 65 |
| median | 72 |
| 90th percentile | 85 |
| **maximum** | **90** |

**Patients with ≥ 120 minutes: 0 / 552.**

The required cohort is empty. No amount of careful design recovers it; the
signal simply does not exist in this database.

This also explains a design choice that had looked arbitrary: **60 minutes is
the longest span available for every patient** (the cohort minimum is exactly
60). `pipeline_clinical.py:190` crops to the final hour, and that is the maximal
common window, not a preference.

---

## 3. Feasible weakened variants

The maximum shift available for a patient is `duration − 60`: median 12 min,
maximum 30 min. A shifted 60-minute window at shift *s* overlaps the baseline
window by `(60 − s)/60`.

| shift | patients | positives | overlap with baseline | duration→label AUROC in subset | estimated MDE |
|---:|---:|---:|---:|---:|---:|
| 10 min | 435 | 105 | 83% | 0.5778 | ~0.072 |
| **15 min** | **267** | **75** | **75%** | **0.5235** | **~0.092** |
| 20 min | 160 | 45 | 67% | 0.5536 | ~0.119 |
| 25 min | 82 | 27 | 58% | 0.4411 | ~0.17 |
| 30 min | 39 | 12 | 50% | 0.4444 | ~0.24 |

MDE scaled from the Phase-7 paired-bootstrap estimate (0.0642 at n=547) by
√(547/n); it is an approximation, but the ordering is unambiguous.

### The bind

The two requirements pull in opposite directions:

- **Contrast** requires a large shift — at 15 min the conditions share 45 of
  their 60 minutes, so the comparison is largely between a signal and itself.
- **Power** requires a small shift — at 25–30 min only 39–82 patients remain
  and the MDE exceeds 0.17.

There is no shift at which the experiment is both a meaningful contrast and
adequately powered. The best available compromise (15 min) asks whether moving
the window 15 minutes earlier, while retaining 75% of the same signal, changes
AUROC by more than 0.092.

### One thing the variants do establish

Restricting to the ≥ 75-minute cohort collapses the duration confound:
duration→label AUROC falls from **0.6407** (full cohort) to **0.5235** (267
patients). The confound is a property of the whole-cohort duration spread, not
an intrinsic feature of longer recordings.

---

## 4. Data-integrity checks performed

| check | result |
|---|---|
| durations read from raw signal, not metadata | yes, 552 records |
| cohort for N1b-1 as specified | **empty (0 patients)** |
| duration → label association, full cohort | AUROC 0.6407, Spearman(pH) −0.318, p < 0.001 |
| duration → label association, shift-15 subset | AUROC 0.5235 |
| window length / stride / preprocessing | unchanged, would be identical across conditions |

---

## 5–9. Results

Not produced. The primary comparison has an empty cohort, and no weakened
variant is adequately powered for the contrast it can support. Reporting an
underpowered null here would invite it to be read as evidence of absence, which
is precisely the error Phase 7 identified in this project's earlier +0.02 bars.

---

## 10. Decision

### B — Stop the information search

Treat **~0.73 as the empirical performance region for this cohort and task**,
without claiming an absolute theoretical ceiling.

Reasoning, restricted to evidence produced in this phase and Phase 7:

1. **N1 (more CTG) is confounded.** Recording duration predicts the outcome at
   AUROC 0.6407. Uncropping makes bag size covary with the label and would
   reintroduce the leak that `audit_bag_size_leak.py` currently passes at 0.4657.
2. **N1b (earlier CTG) is infeasible.** The required interval exists for zero
   patients; every runnable variant trades contrast against power with no
   workable setting.
3. **N2 (clinical context) is null.** 11 antepartum covariates reach 0.5423
   alone, and adding them to the CTG features gives **−0.0191**. Every covariate
   is individually near-chance (parity best at 0.5894; meconium 0.5106).
4. **N3 is moot**, both its components having failed.

The information ladder was the project's own stopping condition: *"if everything
remains ~0.73, accept that 0.85 is probably not obtainable from the currently
available information."* All three rungs have now returned blocked, infeasible,
or null.

### What this decision does NOT claim

- Not that 0.73 is a theoretical ceiling. Phase 7's MDE of 0.0642 means effects
  smaller than that remain undetectable, not absent.
- Not that CTG contains no temporal information — only that this dataset cannot
  test the temporal-location question.
- Not that a better model is impossible on a *different* cohort. CTU-UHB is
  single-centre, 552 records, 110 positive, and capped at 90 minutes.

### The one remaining route to a materially higher number

External data. A cohort with longer recordings would make N1b answerable; a
larger cohort would lower the MDE below the effect sizes this project keeps
measuring at +0.01 to +0.04. Both are data-acquisition problems, not modelling
problems, and neither is achievable within this project's scope.

**No further architecture is recommended.** No previously untested information
source has been identified that would justify one.
