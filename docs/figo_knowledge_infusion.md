# Knowledge infusion for FIGO-state detection — result

Written 2026-09-04. Ruleset `figo2015-seq-v3`, protocol
`data/processed_figo/folds.json`, repeat 0.

```bash
python scripts/figo_run_ki.py --arms KI1_concept KI2_rule KI3_combined
python scripts/figo_eval_detection.py --baseline Z_frozen_smallcnn --gate_on KI3_combined
```

## Verdict (after replication across all 5 protocol partitions — see §9)

**Knowledge infusion works. The claim that the reasoning layer specifically
does the work is supported on pooled evidence but FAILS the pre-registered
per-repeat criterion, and is stated accordingly.**

Headline, five patient partitions:

| arm | mean AUROC | sd | Δ vs Z, combined | beats Z |
|---|---:|---:|---|---|
| **KI-3** | **0.7759** | **0.0025** | **+0.0191 [+0.0139, +0.0244]** | 5/5 |
| KI-2 | 0.7753 | 0.0026 | +0.0186 [+0.0127, +0.0244] | 5/5 |
| KI-1 | 0.7679 | 0.0032 | +0.0111 [+0.0074, +0.0147] | 4/5 |
| Z frozen | 0.7568 | **0.0093** | — | — |

Sections 1–6 below report the FIRST repeat only, which is how the experiment
was originally run and written up. **Their single-repeat numbers should not
be quoted as headline** — §9 supersedes them. They are kept because the
concept diagnostics and the refinement analysis were computed on that repeat
and remain valid as diagnostics.

A correction is recorded openly: on the evidence of one repeat this document
originally called the result "robust" and called the reasoning-layer effect
"a very nice mechanistic result". One repeat did not license either
adjective. See §9.

## Why this was a well-posed test

The binary label is not merely correlated with the descriptors. It is
**exactly**

```
Abnormal = NOT (baseline_normal AND variability_normal AND no_repetitive_decels)
```

verified on all 3,497 readable epochs: **100.0000% match, zero mismatches**,
and zero pathological epochs where all three normality flags hold. So KI-2's
soft rule is the label with hard thresholds replaced by sigmoids, and its
ceiling is not an assumption:

| soft-rule temperature | AUROC fed **true** concepts |
|---|---:|
| τ = 0.25 bpm | **0.9999** |
| τ = 1.0 | 0.9992 |
| τ = 2.0 | 0.9974 |
| τ = 5.0 | 0.9902 |

τ is learnable, so KI-2's ceiling is effectively 1.0 and **any shortfall is
attributable to concept estimation alone** — which is measured in §4.

## 1. Result

All arms share arm Z's convolutional backbone verbatim (~78k parameters vs
Z's 69k), so no difference can be attributed to a better encoder.

| arm | params | AUROC | 95% CI | AUPRC | spec@90%sens | balanced se/sp | Brier |
|---|---:|---:|---|---:|---:|---|---:|
| **KI-2** rule only | 78,448 | **0.7771** | 0.759–0.797 | **0.6642** | **0.4423** | **0.707/0.708** | 0.1954 |
| **KI-3** combined | 78,516 | 0.7749 | 0.756–0.794 | 0.6606 | 0.4310 | 0.703/0.703 | 0.1957 |
| **KI-1** concepts only | 78,511 | 0.7667 | 0.748–0.786 | 0.6450 | 0.4083 | 0.690/0.690 | 0.1958 |
| **Z** frozen baseline | 69,281 | 0.7559 | 0.737–0.774 | 0.6327 | 0.3980 | 0.688/0.688 | 0.1987 |
| A FHR (interaction family) | 221,378 | 0.7369 | 0.715–0.757 | 0.6555 | 0.3247 | 0.673/0.674 | 0.2035 |
| C cross-attention | 452,066 | 0.7328 | 0.711–0.754 | 0.6421 | 0.3202 | 0.676/0.674 | 0.2113 |

### Paired tests — the instrument that decides

Both models score the same patients, so the difference must be tested paired,
not by comparing marginal CIs. Bootstrapped over patients, 4,000 resamples:

| comparison | Δ AUROC | 95% CI | P(Δ≤0) | |
|---|---:|---|---:|---|
| **KI-2 − Z** | **+0.0213** | [+0.0118, +0.0313] | 0.0000 | **excludes 0** |
| **KI-3 − Z** | **+0.0191** | [+0.0114, +0.0272] | 0.0000 | **excludes 0** |
| **KI-1 − Z** | **+0.0109** | [+0.0028, +0.0190] | 0.0040 | **excludes 0** |
| **KI-2 − KI-1** | **+0.0104** | [+0.0030, +0.0177] | — | **excludes 0** |
| KI-2 − KI-3 | +0.0022 | [−0.0049, +0.0093] | — | crosses 0 |

Per fold, versus Z:

| arm | f1 | f2 | f3 | f4 | f5 | folds up | mean | Wilcoxon |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| KI-2 | +0.027 | +0.007 | +0.023 | +0.031 | −0.004 | **4/5** | +0.0167 | p=0.125 |
| KI-3 | +0.014 | +0.002 | +0.019 | +0.027 | +0.013 | **5/5** | +0.0148 | p=0.062 |
| KI-1 | +0.003 | −0.002 | +0.018 | +0.022 | −0.014 | 3/5 | +0.0053 | p=0.438 |

Wilcoxon on n=5 folds is underpowered by construction and cannot reach
p<0.05 with 5/5 concordance (its minimum attainable p is 0.0625); the
patient-clustered bootstrap is the instrument to read, and all three exclude
zero.

## 2. The reasoning layer is what helps, not the auxiliary loss

This is the mechanistic result and it is worth stating precisely.

**KI-2 − KI-1 = +0.0104, CI [+0.0030, +0.0177], excludes zero.** KI-1 and
KI-2 receive the *same* concept supervision with the *same* λ grid on the
*same* backbone. They differ only in how the state is produced: KI-1 by a
learned linear head on the pooled feature, KI-2 by pushing the predicted
concepts through the FIGO conjunction.

So the gain is **not** multi-task regularisation. Adding concept supervision
alone (KI-1) buys +0.0109; routing the prediction through the rule buys a
further +0.0104 on top of that — roughly half the total effect comes from the
reasoning structure.

**KI-2 − KI-3 = +0.0022, crosses zero.** Given the rule, the direct
classifier head adds nothing. KI-3 was free to weight the two routes and
gains nothing by having both.

This is the same lesson Gate 2 recorded from the other direction: on
identical descriptor columns and folds, a decision tree scored 0.9997 and
logistic regression 0.8371, because a linear model cannot express an
interval. Here the network is handed the interval-and-conjunction structure
explicitly and improves by exactly the amount that structure is worth.

## 3. The operating point improves more than the AUROC does

The clinically binding direction gains more than the headline:

| | Z | KI-2 | Δ |
|---|---:|---:|---:|
| **specificity @ 90% sensitivity** | 0.3980 | **0.4423** | **+0.044** |
| specificity @ 85% sensitivity | 0.4973 | 0.5228 | +0.026 |
| balanced sens/spec | 0.688/0.688 | **0.707/0.708** | +0.019 |
| Brier | 0.1987 | **0.1954** | −0.003 |
| at the pre-specified threshold | 88.3% / 43.0% | 87.6% / **48.3%** | +5.3 pp spec |

At a fixed 90% sensitivity the false-alarm rate falls from 60.2% to 55.8% of
normal epochs. That is a real reduction, and it is the metric this project
argued matters. **It does not reach the 89/89 target**, and the balanced
point at 0.707/0.708 is not within reach of 0.89 by any threshold choice.

## 4. Where the remaining gap is — now measured, not guessed

Because the soft rule is exact when fed true concepts, the whole 0.777 → 1.0
gap is concept estimation. Out-of-fold concept recovery, KI-2:

| concept | recovery | |
|---|---:|---|
| stv_bpm | R² +0.825 | good |
| **baseline_bpm** | **R² +0.670** | good |
| frac_time_in_decel | R² +0.653 | moderate |
| n_decels | R² +0.552 | moderate |
| longest_decel_s | R² +0.497 | moderate |
| deepest_decel_bpm | R² +0.472 | moderate |
| n_accels | R² +0.323 | weak |
| n_decel_prolonged | R² +0.285 | weak |
| **variability_bpm** | **R² +0.155** | **poor** |
| n_decel_late | R² +0.044 | ~none |
| **n_contractions** | **R² +0.018** | **~none** |
| **decel_repetitive** | **AUROC 0.755** | moderate |
| variability_measurable | AUROC 0.790 | — |
| has_acute_hypoxia_decel | AUROC 0.071 | **meaningless — prevalence 0.03%** |

Now weight those by how much each concept actually decides the label:

| normality flag | AUROC of that flag alone |
|---|---:|
| **no_repetitive_decels** | **0.8220** |
| variability_normal | 0.6277 |
| baseline_normal | 0.6086 |

The picture is unambiguous. **The concept that decides most of the label
(`no_repetitive_decels`, 0.822 alone) is recovered at only AUROC 0.755, and
it is defined in terms of contractions — which are recovered at R² 0.018,
because the KI backbone is arm Z's and sees FHR only.** The best-recovered
concept, baseline, is the one that matters least.

`variability_bpm` at R² 0.155 is the second finding, and a more surprising
one: bandwidth amplitude ought to be readable from FHR morphology, and it is
not being read.

## 5. What must be fixed in the concept set

`has_acute_hypoxia_decel` has **one positive epoch in 3,497** (prevalence
0.03%). Its AUROC of 0.071 is noise and should not be reported as a concept
metric. It is retained in the label rules (a single prolonged deceleration
below 80 bpm is a sufficient pathological criterion and must stay) but should
be **dropped from L_clinical**, where it contributes only gradient noise.

## 6. The one permitted refinement, specified in advance

The gate allows exactly one controlled refinement in the 0.75–0.799 band.
§4 determines what it must be:

> **Give the KI backbone the UC channel, so that `n_contractions` and hence
> `decel_repetitive` become estimable.**

This is **not** a revival of the falsified interaction hypothesis, and the
distinction matters:

* the interaction experiment asked whether FHR–UC coupling helps a
  **classifier** predict the state directly. Answer: no (C−B = +0.0004,
  CI crossing zero).
* this asks whether UC helps **estimate a concept that is defined in terms of
  UC**, which then enters an exact rule. Different claim, and the interaction
  experiment's own control already showed the coupling is learnable:
  destroying it cost arm C 0.0261 with a CI excluding zero (C−D).

Pre-registered, before running:

| outcome | reading |
|---|---|
| `n_contractions` R² rises **and** `decel_repetitive` AUROC rises **and** state AUROC rises | UC matters for concept estimation though not for direct classification |
| concepts improve, state AUROC does not | the rule is not limited by these concepts after all |
| concepts do not improve | contractions are not recoverable from this toco channel at this quality; close the branch |

Bar: **+0.02 AUROC over KI-2 with a paired CI excluding zero**, same
protocol, one configuration, no sweep. If it fails, detection closes at
**0.777** and the project moves to early warning.

## 7. The permitted refinement was run, and it FAILED

`python scripts/figo_run_ki.py --arms KI2_FIX KI2_UC`

Two arms: **KI2_UC**, the refinement (UC added to the backbone as two extra
input channels, +960 parameters, no attention, no extra depth), and
**KI2_FIX**, its matched control (same corrected concept set, no UC) so that
the effect of UC is separable from the effect of dropping
`has_acute_hypoxia_decel`. Both changes landed together; without the control
they would be confounded.

| arm | AUROC | 95% CI | spec@90%sens |
|---|---:|---|---:|
| KI-2 (original) | **0.7771** | 0.759–0.797 | 0.4423 |
| KI2_FIX (concept-set fix only) | 0.7725 | 0.754–0.791 | 0.4307 |
| **KI2_UC (the refinement)** | **0.7634** | 0.745–0.783 | 0.3967 |

| paired comparison | Δ | 95% CI | |
|---|---:|---|---|
| **KI2_UC − KI-2** — *the gate* | **−0.0139** | [−0.0281, +0.0002] | crosses 0 |
| KI2_UC − KI2_FIX — *UC alone* | −0.0093 | [−0.0233, +0.0046] | crosses 0, **1/5 folds up** |
| KI2_FIX − KI-2 — *concept-set fix* | −0.0046 | [−0.0097, +0.0007] | crosses 0 |

**Bar was ≥0.7971 with a paired CI excluding zero. Achieved 0.7634 — missed
by 0.034, and the point estimate moved the wrong way.** The gate closes.

The concept-set fix was neutral (−0.0046, CI crossing zero), so dropping
`has_acute_hypoxia_decel` neither helped nor hurt; it remains correct on the
grounds that supervising a target with one positive epoch in 3,497 is
meaningless regardless.

### Why it failed — capacity competition, measured

UC did exactly what §6 predicted it would do for the concepts that need it:

| concept | no UC | with UC | Δ | |
|---|---:|---:|---:|---|
| **n_contractions** | +0.017 | **+0.580** | **+0.563** | UC-dependent |
| decel_repetitive | 0.753 | 0.773 | +0.021 | UC-dependent |
| n_decel_late | +0.046 | +0.064 | +0.018 | UC-dependent |
| variability_bpm | +0.139 | +0.193 | +0.054 | |
| baseline_bpm | +0.649 | +0.675 | +0.027 | |
| longest_decel_s | +0.492 | +0.483 | −0.009 | |
| frac_time_in_decel | +0.650 | +0.639 | −0.011 | |
| n_decel_prolonged | +0.288 | +0.276 | −0.012 | |
| deepest_decel_bpm | +0.463 | +0.449 | −0.014 | |
| n_accels | +0.317 | +0.294 | −0.023 | |
| variability_measurable | 0.816 | 0.791 | −0.024 | |
| n_decels | +0.560 | +0.513 | −0.047 | |
| **stv_bpm** | +0.806 | **+0.708** | **−0.098** | |

**n_contractions improved 34-fold (R² 0.017 → 0.580).** The intervention did
precisely what it was designed to do. But 5 concepts improved and **8
degraded**: mean +0.200 on the three UC-dependent concepts, mean **−0.016**
on the ten FHR-only ones, with `stv_bpm` losing 0.098.

The trunk is shared and holds ~79k parameters. Capacity spent encoding the
toco channel is capacity taken from representing FHR, and the net effect on
the state was negative. This is the same arithmetic that has now appeared
four times in this project (Model 9's collapse, the temporal-feasibility
count, the interaction experiment's 69k-beats-452k, and now this): 3,497
epochs from 547 patients does not fund additional representational load.

### The sharper finding underneath

**`decel_repetitive` rose only 0.753 → 0.773 even though `n_contractions`
rose 0.017 → 0.580.** Making the contraction denominator estimable did almost
nothing for the repetitiveness judgement itself.

So the bottleneck on the concept that decides most of the label (0.822 alone)
was **never the contraction count**. It is the deceleration-to-contraction
*pairing*, or the >50% threshold over it — a counting-and-matching operation,
not a perception one. That is consistent with, and now independently
corroborates, the interaction experiment: the pairing is learnable (arm C
lost 0.0261 with a CI excluding zero when it was destroyed) but does not
convert into classification value.

**Detection closes at KI-2, AUROC 0.7771 [0.759–0.797].**

## 8. Honest limitations

- **λ was selected per fold on inner-validation patients** from a fixed grid
  {0.1, 0.3, 1.0}. That is protocol rule 3's sanctioned mechanism and never
  touches the reported fold, but it is a degree of freedom Z did not have.
  Selected λ varied by fold (0.1/1.0/1.0/0.3/1.0 for KI-2), so a single fixed
  λ would have given a smaller effect.
- **One repeat, one seed.** The +0.021 is a single pass through repeat 0.
  Before this is written up as a headline, it should be replicated on at
  least one further repeat.
- **Wilcoxon cannot confirm this at n=5 folds.** The bootstrap is the
  evidence; the fold table is descriptive.
- KI-2's gain is not uniform: it *loses* on fold 5 (−0.004) while KI-3 is
  positive on all five. If a single arm has to be carried forward, **KI-3 is
  the more stable choice** despite scoring 0.0022 lower, and that difference
  is not significant.


---

## 9. REPLICATION — all five protocol partitions

`python scripts/figo_ki_replication.py`. Repeats 1–4 re-run identical code
against different patient partitions (`src/figo_state/protocol.py` uses
`seed + rep` per repeat). Nothing was re-tuned and no arm was added.

### 9.1 Per repeat

| arm | r0 | r1 | r2 | r3 | r4 | mean | sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| Z frozen | 0.7559 | 0.7546 | **0.7726** | 0.7484 | 0.7523 | 0.7568 | **0.0093** |
| KI-1 | 0.7667 | 0.7723 | 0.7700 | 0.7662 | 0.7643 | 0.7679 | 0.0032 |
| KI-2 | 0.7771 | 0.7720 | 0.7780 | 0.7731 | 0.7760 | 0.7753 | 0.0026 |
| KI-3 | 0.7749 | 0.7784 | 0.7786 | 0.7730 | 0.7745 | 0.7759 | 0.0025 |

### 9.2 Two estimators, two answers, both reported

**Per-repeat** (the pre-registered criterion: sign consistent across repeats,
spread below the effect):

| comparison | per-repeat deltas | pre-registered verdict |
|---|---|---|
| KI-2 − Z | +0.021, +0.018, +0.006, +0.025, +0.024 | sign holds, **spread exceeds effect** |
| KI-3 − Z | +0.019, +0.024, +0.006, +0.025, +0.022 | sign holds, spread < effect → **passes** |
| KI-1 − Z | +0.011, +0.018, **−0.003**, +0.018, +0.012 | **sign flips → fails** |
| KI-2 − KI-1 | +0.010, **−0.000**, +0.008, +0.007, +0.012 | **sign flips → fails** |

**Combined** — resample patients, and within each draw average the delta over
all five partitions. This is the correct instrument for an effect smaller
than partition noise, because it separates patient sampling from partition
sampling instead of confounding them:

| comparison | Δ | 95% CI | P(Δ≤0) |
|---|---:|---|---:|
| KI-2 − Z | +0.0186 | [+0.0127, +0.0244] | 0.0000 |
| **KI-3 − Z** | **+0.0191** | [+0.0139, +0.0244] | 0.0000 |
| KI-1 − Z | +0.0111 | [+0.0074, +0.0147] | 0.0000 |
| **KI-2 − KI-1** | **+0.0074** | [+0.0030, +0.0121] | 0.0003 |
| **KI-3 − KI-1** | **+0.0080** | [+0.0039, +0.0120] | 0.0007 |

The two disagree on KI-2 − KI-1 and the disagreement is not a contradiction.
Per repeat asks "does this appear reliably in any single 5-fold partition"
— **no**, it is below that noise floor. Combined asks "averaged over
partitions, is it nonzero" — **yes**. The supportable statement is therefore:

> the reasoning layer contributes ~+0.008 beyond concept supervision, real
> but smaller than the partition-to-partition variation of a single
> cross-validation, and detectable only by repeating the partition.

That is materially weaker than this document's original claim and replaces it.

### 9.3 The unanticipated finding: KI stabilises the model

Z's standard deviation across partitions is **0.0093**; every KI arm sits at
**0.0025–0.0032**, a 3.0–3.7x reduction. Z ranges 0.7484–0.7726 while KI-3
ranges 0.7730–0.7786.

This explains repeat 2 completely, which the per-repeat table flags as the
lone failure: it is a **Z outlier** (0.7726, Z's best of five), not a KI
failure — KI-2 posted its own best score (0.7780) on that same partition.
Any criterion built on per-repeat deltas inherits the baseline's variance,
which is why the combined estimator is the right one here.

For a model whose purpose is to generalise to unseen patients, a 3.6x
reduction in partition sensitivity is arguably worth as much as the +0.019 in
mean AUROC, and it is a property of the knowledge infusion rather than of
capacity: KI arms carry ~78k parameters against Z's 69k.

### 9.4 Final position for detection

**KI-3, AUROC 0.7759 ± 0.0025**, +0.0191 [+0.0139, +0.0244] over the frozen
baseline, ahead on 5/5 partitions. KI-3 over KI-2 on variance and fold
consistency; their 0.0006 mean separation is noise.

This does not approach 0.85, and the balanced operating point (~0.707/0.708)
is not within reach of the 89/89 target. Detection is closed here.
