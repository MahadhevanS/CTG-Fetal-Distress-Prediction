# Deceleration-burden redesign study

Written 2026-09-04. A label/measurement study, not a modelling study. No
CTU-UHB outcome, no early-warning target and no pH value was read while
constructing or evaluating anything below.

```bash
python scripts/figo_decel_burden_study.py    # Phases 3-4
python scripts/figo_build_burden.py          # continuous burden for every epoch
```

**Outcome: the redesign is JUSTIFIED as a measurement but CANNOT be completed
as a label. Stopping rule 2 fires.** Phases 5–7 are not run. §8 separates the
four limitation types the brief asks to distinguish.

---

## 1. Existing deceleration pipeline

All in `src/figo_state/descriptors.py` unless stated.

| step | function | how |
|---|---|---|
| contraction detection | `detect_contractions` (L363) | `scipy.find_peaks` on low-passed toco, `distance = 60 s` (`UC_MIN_SEPARATION_S`), `prominence = 10` (`UC_MIN_PROMINENCE`); peaks whose ±30 s neighbourhood is >50% dropout are discarded |
| baseline | `estimate_baseline` (L248) | iterative ±15 bpm exclusion over valid samples, seeded from the previous epoch, **rounded to the nearest 5 bpm**, one constant per 10-min epoch |
| deceleration detection | `delineate_events` (L216) | core where `baseline − FHR ≥ 15 bpm`, extended outward to `DECEL_EDGE_BPM = 7.5`, spans closer than `DECEL_MERGE_GAP_S = 30 s` merged, then `≥ 15 s` duration |
| pairing | `classify_decelerations` (L409) | nearest contraction peak to the deceleration **nadir**, accepted if `|lag| ≤ 120 s` (`DECEL_UC_PAIRING_TOLERANCE_S`); contributes the contraction index to a `set` |
| repetitiveness | `compute_descriptors` (L523) | `n_uc > 0 and len(paired_contractions)/n_uc > 0.50` |
| consumption | `rules.py` L198, L261, L272 | `no_repetitive_decels = not decel_repetitive`; a single False makes the epoch Suspicious |

**Hard-coded thresholds:** 15 bpm, 15 s, 7.5 bpm, 30 s, 120 s, 60 s,
prominence 10, 0.50, 180 s, 300 s, 80 bpm, 110/160/100 bpm, 5/25 bpm.

**Rounded/discretised:** baseline to 5 bpm; contraction and deceleration
counts are integers; the repetitiveness verdict is binary; the FIGO state is
one of three.

**Can a single detection error change the FIGO state?** Yes — measured at
**48.3%** of epochs (§5).

### Two defects found during the audit

1. **Prolonged decelerations can never be repetitive.**
   `classify_decelerations` executes `continue` immediately after
   incrementing `prolonged` (L455), so the pairing block below it never runs
   and a prolonged deceleration is never added to `paired_contractions`.
   FIGO 2015's pathological criterion is "repetitive **late or prolonged**
   decelerations", so the code structurally excludes prolonged decelerations
   from a criterion they are supposed to be able to satisfy. This
   contributes to `path_rep_decels` firing on only 10 epochs.
2. **Zero-contraction epochs are benign by default.** `n_uc > 0` guards the
   expression, so 357 epochs (10.2%) score not-repetitive because the ratio
   is undefined — while **311 of them contain at least one deceleration**,
   averaging 2.45 per epoch.

Neither is a clinical judgement. Both are coding defaults.

---

## 2. Why the >50% rule is brittle

Measured on 3,140 readable epochs with ≥1 contraction, using
`n_contractions_with_decel` recomputed from raw signal (`burden.npz`;
recomputation reproduces the stored v3 verdict on **97.9%** of epochs).

| | |
|---|---:|
| median contractions per epoch | **4** |
| epochs with ≤4 contractions | **52.5%** |
| **epochs flippable by ±1 deceleration** | **48.3%** |
| epochs within 1 event of the boundary | 62.1% |
| median distance from the boundary | **1.00 event** |
| epochs with 0 contractions (criterion undefined) | 10.2% |

By denominator:

| contractions | epochs | flippable | % |
|---:|---:|---:|---:|
| 1 | 132 | 132 | **100.0** |
| 2 | 256 | 181 | 70.7 |
| 3 | 421 | 258 | 61.3 |
| 4 | 671 | 329 | 49.0 |
| 5 | 733 | 352 | 48.0 |
| 6 | 563 | 163 | 29.0 |
| 8 | 76 | 13 | 17.1 |

One deceleration moves the ratio by 1/k — **25 percentage points at the
median k=4** — and can cross 0.50. There is no denominator at which the rule
is stable, because normal labour simply does not produce enough contractions
in 10 minutes for a proportion to be well estimated.

---

## 3. Candidate continuous burden measures

Implemented in `src/figo_state/burden.py`. Each is stated as a formula; none
is thresholded.

**A. Contraction-associated fraction** — the >50% rule's own ratio, kept continuous:

$$\text{frac} = \frac{|\{\text{contractions carrying} \ge 1 \text{ deceleration}\}|}{|\text{contractions}|}$$

**B. Severity-weighted burden per minute**, using the deficit area:

$$\text{area\_per\_min} = \frac{\sum_e \int_e \max(\text{baseline} - \text{FHR},\, 0)\, dt}{\text{observation minutes}} \quad [\text{bpm·s/min}]$$

with the cruder peak-depth surrogate reported alongside for comparability
with the literature form:

$$\text{depth} \times \text{duration per min} = \frac{\sum_e \text{depth}_e \cdot \text{duration}_e}{\text{observation minutes}}$$

**C. Contraction-normalised burden**, so different contraction rates compare:

$$\text{area\_per\_contraction} = \frac{\sum_e \text{area}_e}{|\text{contractions}|}$$

**D. Temporal clustering / repetition** — `max_consecutive_uc_with_decel`
(longest run of consecutive contractions each carrying a deceleration; this
is what "repetitive" means physiologically and it has no denominator),
`decel_interval_cv` (regularity of inter-nadir gaps),
`decel_trend_per_min` and `severity_trend` (is the burden increasing).

**E. Combined** — the 15 quantities in `DecelBurden` span frequency,
severity, duration and clustering without reproducing the >50% rule: the
ratio is retained as one continuous coordinate among several, not as the
organising principle.

---

## 4. FHRMA validation

152 recordings, expert per-sample deceleration annotation, **no pH**. Both
arms use the *same* contraction detection and the *same* baseline, so only
the deceleration representation differs. FHRMA does not annotate
"repetitive"; it is derived by pairing the **expert** deceleration spans to
the same contractions under the same rule.

### 4.1 The binary criterion

| | |
|---|---:|
| our epochs called repetitive | **23.1%** |
| expert-derived repetitive | **13.3%** |
| sensitivity | 0.714 |
| precision | **0.411** |
| F1 | 0.522 |
| raw agreement | 0.826 |
| **Cohen κ** | **0.425** |

We **over-call repetitiveness 1.7×**, and κ 0.425 is moderate agreement on a
binary call that one deceleration can flip.

### 4.2 The continuous measures

| measure | Spearman | Pearson | ours mean | expert mean |
|---|---:|---:|---:|---:|
| **area_per_contraction** | **0.629** | **0.732** | 1261 | 846 |
| **max_consecutive_uc_with_decel** | **0.614** | 0.644 | 1.28 | 0.83 |
| decel_area_per_min | 0.532 | 0.616 | 482 | 306 |
| frac_contractions_with_decel | 0.475 | 0.566 | 0.39 | 0.24 |
| n_decels | 0.344 | 0.416 | 1.95 | 1.26 |

**The severity-weighted measures agree with expert-derived values
substantially better than the count-based ones**, and better than the binary
verdict does.

**A correction to an earlier draft of this document.** It read the pairing of
κ 0.425 (binary) against Pearson 0.732 (continuous) as "the signature of
information destroyed by thresholding rather than by the detector". That
overstates what these numbers support. Correlations of 0.475–0.732 are
meaningful but far from near-perfect, and the systematic over-estimate
persists across *every* measure (ours mean > expert mean throughout, ratios
1.49–1.68). Substantial measurement error is present in both representations.
§6 of the ordering analysis localises it: the worst-ordered epochs are ones
where the expert annotated **no** decelerations and we found some. The
detector is not innocent.

The defensible statement is the narrower one:

> The continuous representations preserve substantially more graded agreement
> with expert-derived deceleration burden than the binary repetitive-
> deceleration decision does, which supports further investigation of
> continuous burden as a measurement representation.

**Gate: stopping rule 1 does not fire.** But "more faithful than the binary
call" is not the same as "faithful", and §4b shows where it stops being true.

---

## 4b. Ordering and severity — the deeper analysis

`python scripts/figo_burden_ordering.py`. Correlation is a weak claim: a
measure can correlate at 0.7 and still order individual epochs badly, and
ordering is precisely what a severity threshold needs.

### 4b.1 Within-record ordering — global correlation was inflated

Can the measure rank **one mother's own epochs**? That is the ordering a
monitor needs, and it removes the between-recording spread that inflates a
global correlation.

| measure | median ρ | IQR | records with ρ ≤ 0 |
|---|---:|---|---:|
| **max_consecutive_uc_with_decel** | **0.667** | [+0.67, +0.67] | 6% |
| **area_per_contraction** | **0.400** | [+0.40, +0.69] | 3% |
| frac_contractions_with_decel | 0.026 | [+0.03, +0.37] | 4% |
| **decel_area_per_min** | **0.000** | [+0.00, +0.66] | **65%** |

**This changes the reading of §4.** `decel_area_per_min` correlated at
Spearman 0.532 globally but orders a single labour's epochs at **ρ = 0.000,
with 65% of recordings at or below zero** — its global correlation was almost
entirely between-recording variation. Only `area_per_contraction` and
`max_consecutive_uc_with_decel` survive this test, and neither strongly.

### 4b.2 Severity tiers

Expert quartiles versus ours:

| measure | exact | within ±1 tier | quadratic-weighted κ |
|---|---:|---:|---:|
| **area_per_contraction** | 0.487 | 0.865 | **0.545** |
| decel_area_per_min | 0.405 | 0.868 | 0.429 |
| max_consecutive_uc_with_decel | 0.676 | 0.989 | 0.346 |
| frac_contractions_with_decel | 0.344 | 0.835 | 0.272 |

Best is moderate (κ 0.545). No measure assigns the expert's severity tier
reliably.

### 4b.3 Extreme discrimination — the one strong result

Can the measure find the expert's **worst** epochs? This is the question a
*Pathological* threshold asks.

| measure | AUROC, expert top decile | top quartile |
|---|---:|---:|
| **area_per_contraction** | **0.982** | **0.924** |
| decel_area_per_min | 0.900 | 0.864 |
| frac_contractions_with_decel | 0.845 | 0.643 |
| max_consecutive_uc_with_decel | 0.778 | 0.814 |

**`area_per_contraction` identifies the expert's worst-decile epochs at AUROC
0.982.** That is a genuinely strong result, and it is the sharpest evidence
in this study.

### 4b.4 The bias is NOT recalibratable

If the over-estimate were a monotone transform, a calibration curve would fix
it and ordering would be intact. Comparing correlation on raw values against
correlation on ranks:

| measure | Pearson raw | Pearson on ranks | ours/expert mean |
|---|---:|---:|---:|
| area_per_contraction | 0.732 | **0.629** | 1.49 |
| decel_area_per_min | 0.616 | 0.532 | 1.57 |
| max_consecutive_uc_with_decel | 0.644 | 0.614 | 1.54 |
| frac_contractions_with_decel | 0.566 | 0.475 | 1.68 |

Rank correlation is **lower** than raw correlation for every measure. The raw
agreement is being carried by high-leverage extreme epochs rather than by
consistent ordering — which is the same fact §4b.3 reports as a strength and
§4b.1 reports as a weakness. **Recalibration would not fix the mid-range.**

### 4b.5 Where it fails, and why

The worst-ordered decile of epochs (by rank error on
`area_per_contraction`), against the best-ordered half:

| | worst decile | best half |
|---|---:|---:|
| **expert burden** | **0.0** | 1120.8 |
| contractions | 4.77 | 4.10 |
| usable fraction | 1.00 | 1.00 |

**The worst failures are epochs where the expert annotated NO deceleration
and we found one**, on fully usable signal. They are detector false
positives, not representation failures. This is the direct evidence that the
detector is not innocent, and it is why §4's original claim was withdrawn.

---

## 4c. Reconciliation of the recomputation (required before any v4)

The Phase 4 recomputation agreed with the stored v3 verdict on 97.9% of
epochs, not 100%. Building a new label on an incompletely reconciled
pipeline would be indefensible, so the 74 mismatches were attributed
exhaustively:

| pairing rule used in the recomputation | agreement with stored v3 | mismatches |
|---|---:|---:|
| `burden.py` — prolonged decelerations **included** | 97.88% | 74 |
| `descriptors.py` semantics — prolonged **excluded** | **100.00%** | **0** |

**All 74 mismatches, and only those, are explained by the prolonged-
deceleration pairing difference. The unexplained residual is 0 epochs
(0.00%).** No baseline, contraction-detection, validity-mask or epoch-
alignment discrepancy exists between the two paths.

This also prices the §1 defect: correcting the `continue` bug so a prolonged
deceleration can satisfy repetitiveness would flip **74 epochs (2.1%)** from
not-repetitive to repetitive. Recorded in
`results/figo_burden/reconciliation.csv`.

## 5. Threshold-instability analysis

§2 is the analysis. Two additions.

**The continuous burden separates the flippable epochs**, i.e. the epochs the
binary rule cannot resolve are not homogeneous — they carry graded burden the
binary call discards:

| measure | flippable epochs | stable epochs |
|---|---:|---:|
| decel_area_per_min | 466.2 | 231.4 |
| area_per_contraction | 1360.8 | 441.1 |
| max_consecutive_uc_with_decel | 1.6 | 1.0 |

**A continuous burden has no discontinuity.** One borderline deceleration
changes an area-based measure by its own area — small when the deceleration
is shallow and short. The measure degrades gracefully exactly where the
binary rule does not.

---

## 6. v3 vs v4 label comparison — NOT PERFORMED

Phase 5 is gated on a *clinically defensible* replacement threshold existing.
It does not.

FIGO 2015 supplies one number for this criterion: **>50% of contractions**.
It supplies no threshold for deceleration area per minute, area per
contraction, or run length, and no literature threshold was found for them.
Any cut-point on a continuous burden would therefore be one of:

1. **arbitrary** — no clinical support, so v4 would be less defensible than
   v3, not more;
2. **fitted to CTU-UHB outcomes** — explicitly forbidden by the brief, and
   the exact circularity this project has spent three tracks avoiding;
3. **calibrated to reproduce v3's prevalence** — which would recreate the
   >50% rule under another name, the failure mode the brief names in Phase 2E.

There is no fourth option available from the evidence in hand.

**Stopping rule 2 fires: the continuous burden cannot be converted into a
clinically defensible state criterion without outcome-driven tuning.**

### What IS defensible, and is recommended instead

The two defects in §1 are corrections to *faithfulness*, not new clinical
claims, and neither needs a new threshold:

* **pair prolonged decelerations** — remove the early `continue` so a
  prolonged deceleration can satisfy the criterion FIGO says it can satisfy;
* **stop scoring zero-contraction epochs as benign** — 10.2% of epochs, 311
  of them carrying decelerations, currently read Normal because a ratio is
  undefined. Marking them *unassessable for this criterion* is faithful;
  marking them benign is not.

Both are v4-worthy on guideline-fidelity grounds alone. Neither would be
expected to change prediction, for the reason in §7.

---

## 7. Prediction comparison — NOT PERFORMED, and why it would not settle anything

Phase 7 is gated on Phase 5. But it is worth recording that the prediction
question is **already answered by evidence in hand**, from two directions:

* **Early warning.** `figo_early_warning.md` §8 measured `decel_repetitive`
  persistence at **AUROC 0.534 over 30 minutes** — chance. And feeding a model
  the **true** current rule-concepts gave **AUROC 0.4353** for the future
  label. Improving how well we *measure* repetitiveness moves a model toward
  that oracle, and the oracle is chance. A better deceleration measure cannot
  rescue early warning.
* **Detection.** The label is *defined by* the measurement. Changing the
  measurement changes the label rather than improving prediction of it, so a
  v4 detection AUROC would not be comparable with v3's 0.7759 — it would be a
  different task. `figo_knowledge_infusion.md` §7 already showed this
  directly: making contractions estimable moved `n_contractions` recovery
  from R² 0.017 to 0.580 and moved `decel_repetitive` only 0.753 → 0.773,
  with state AUROC *falling*.

So the honest position is that the redesign is worth doing **for measurement
fidelity**, and should not be sold as a route to a better AUROC.

---

## 8. Decision: continue or close

**Close this branch**, with the redesign's measurement findings recorded and
the two guideline-fidelity fixes recommended for any future v4.

Separating the four limitation types, as required:

### Detector limitation — REAL, quantified, partly fixable
Deceleration detection over-calls repetitiveness 1.7× (23.1% vs 13.3%) at
precision 0.411 and κ 0.425. Every burden measure over-estimates its expert
counterpart. This is a genuine instrument gap, and §4 shows it is *not* the
dominant one.

### Label-definition limitation — REAL, severe, only partly fixable
**48.3% of epochs are flippable by a single deceleration**, median distance
from the boundary 1.00 event, and 10.2% of epochs have an undefined ratio
scored as benign. The continuous measures are demonstrably more faithful
(Pearson 0.732 vs κ 0.425 on the same events), so thresholding is destroying
real information. **But the fix requires a clinical threshold that does not
exist**, so the defect can be *documented* and two implementation errors
corrected, and no more than that without new external evidence.

### Physiological information limitation — DECISIVE, not fixable here
`decel_repetitive` has **no 30-minute persistence** (AUROC 0.534), and
perfect knowledge of the true current rule-concepts predicts the future label
at **0.4353**. This bounds early warning regardless of how well anything is
measured. It is the binding constraint, and it is a property of the
physiology-plus-label on this corpus, not of the pipeline.

### Model limitation — NOT the binding constraint anywhere
Detection: capacity consistently hurt (69k beat 452k by 0.0231, CI excluding
zero). Cross-attention over FHR–UC bought +0.0004. Early warning: trajectory
features added nothing across four paired comparisons. In no track did model
capacity turn out to be what was limiting.

### What would actually be needed

For the label question, an external corpus with **expert FIGO state
annotations** — not just deceleration spans — would allow a continuous
threshold to be calibrated against clinical judgement rather than against
CTU-UHB outcomes. FHRMA does not provide that. Without it, the >50% rule
stands as the only defensible option despite being demonstrably brittle,
which is itself a reportable finding about the guideline's operationalisation
at 10-minute resolution.
