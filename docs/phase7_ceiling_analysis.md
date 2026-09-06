# Phase 7 — CTG–pH information ceiling and label-coherence analysis

Completed 2026-09-04. All analyses run on the frozen protocol
([../src/training/protocol.py](../src/training/protocol.py)): 547 patients, 110
positive, patient-grouped folds, repeat 0, patient-level max aggregation,
patient-bootstrap uncertainty. No model was trained or tuned for this phase; all
scores are cached out-of-fold predictions.

---

## 1. Executive verdict

**SUGGESTIVE BUT INCONCLUSIVE.**

There is real evidence consistent with a CTG–pH information limit, but the
analysis **cannot exclude modest improvements**, and two concrete information
sources remain untested. Forcing the "strong evidence of a ceiling" conclusion
would not be supported.

The single most important finding of this phase is not about the ceiling at all:

> **The minimum detectable paired AUROC difference in this cohort is 0.0642 at
> 80% power. Every null result in this project used a pre-registered bar of
> +0.02 — three times smaller than the smallest effect the experiment could
> detect.**

Those nulls are therefore **"no detectable effect", not "no effect"**. This must
be stated wherever they are reported. It does not make them worthless — a
pre-registered null is still the correct call at the time — but it bounds how
strongly they can be interpreted.

---

## 2. Evidence table

| Question | Evidence | Result | Interpretation |
|---|---|---|---|
| Do CTG scores track continuous pH? | Spearman ρ(score, pH) | −0.309 CTG, −0.347 clinical, −0.369 top-3 | Weak-to-modest monotone relationship; ~11–14% of rank variance |
| Is the binary threshold hiding a stronger continuous signal? | AUROC across 6 pH thresholds | peaks at **pH ≤ 7.15** for all 3 models | **No.** The chosen cut is where the models do best |
| Do similar-CTG patients have similar pH? | pH spread within top score quintile | **0.520 of the cohort's 0.600 range** | Large discordance |
| Are errors concentrated at the threshold? | error share in pH 7.05–7.25 | 60–63% of errors in a band holding 45.3% of patients | Modest enrichment (1.33–1.39×), **not** dominant |
| Are false negatives closer to the threshold than true positives? | \|pH−7.15\|, FN vs TP | 0.0695 vs 0.0870, **p = 0.217** | **No significant difference** |
| Does the model miss only ambiguous cases? | severe cases pH ≤ 7.05 | **misses 19/41 (46%)** | **No** — it misses clearly acidotic babies |
| Does recording quality matter? | quality quartiles | Q4 0.8299 [0.73–0.91]; top-half vs bottom-half 0.7336 vs 0.7159 | Non-monotonic; quartile result likely fluctuation |
| Does more recording help? | window-count strata | full 17 windows **0.7672** [0.71–0.82] vs <15 windows 0.6735 | **Yes, +0.094**, better powered |
| Are physiologically similar patients labelled alike? | nearest-neighbour in feature space | label disagreement **27.4%** vs 32.1% chance | Barely better than chance |
| How large an effect can we detect? | paired bootstrap | **MDE 0.0642 @ 80% power** | Every +0.02 bar was underpowered |

---

## 3. CTG–pH relationship (Analyses 1, 3)

| model | binary AUROC | ρ(score, continuous pH) |
|---|---:|---:|
| CTG (mslstm E1) | 0.6834 | −0.309 |
| clinical LR (19) | 0.7268 | −0.347 |
| clinical top-3 | 0.7372 | −0.369 |

AUROC against alternative thresholds — the models are **not** better at any other
cut, which answers Analysis 3 directly:

| threshold | n pos | CTG | clinical 19 | top-3 |
|---|---:|---:|---:|---:|
| pH ≤ 7.00 | 20 | 0.6064 | 0.7026 | 0.6753 |
| pH ≤ 7.05 | 41 | 0.6600 | 0.7133 | 0.7192 |
| pH ≤ 7.10 | 58 | 0.6612 | 0.6946 | 0.7209 |
| **pH ≤ 7.15** | 110 | **0.6834** | **0.7268** | **0.7372** |
| pH ≤ 7.20 | 191 | 0.6790 | 0.7099 | 0.7147 |
| pH ≤ 7.25 | 289 | 0.6490 | 0.6631 | 0.6861 |

**There is no evidence that the binary threshold is masking a stronger continuous
relationship.** ρ ≈ −0.35 is a weaker relationship than the binary AUROC implies,
not a stronger one. This rules out the "REVISE THE TASK" option on the grounds
originally proposed.

### pH spread within score quintiles (clinical LR)

| quintile | n | pH mean | pH sd | pH min | pH max | % positive |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 110 | 7.272 | 0.087 | 6.950 | 7.430 | 6.4% |
| Q3 | 109 | 7.242 | 0.088 | 6.980 | 7.470 | 11.9% |
| Q5 | 110 | 7.170 | 0.114 | 6.870 | 7.390 | 42.7% |

The model separates the extremes (6.4% → 42.7% positive) but even the highest-risk
quintile spans pH 6.87–7.39, and **57% of it is not acidotic**.

---

## 4. Error and borderline analysis (Analyses 2, 6)

Operating point set at cohort prevalence (a label-free rule), clinical LR:

| band | N | pos | FP | FN | error rate |
|---|---:|---:|---:|---:|---:|
| clearly above, pH > 7.25 | 258 | 0 | 28 | 0 | 10.9% |
| borderline above, 7.15 < pH ≤ 7.25 | 179 | 0 | 35 | 0 | 19.6% |
| borderline below, 7.05 < pH ≤ 7.15 | 69 | 69 | 0 | 44 | 63.8% |
| clearly below, pH ≤ 7.05 | 41 | 41 | 0 | 19 | **46.3%** |

62.7% of errors fall in the borderline band, which contains 45.3% of patients —
an enrichment of only 1.39×.

**The borderline-ambiguity explanation is not sufficient.** Three findings
contradict it:

1. FN and TP are **not** significantly different in distance from the threshold
   (0.0695 vs 0.0870, p = 0.217).
2. The model misses **19 of 41 (46%)** patients with pH ≤ 7.05 — severe acidemia,
   not label noise.
3. FN and TP are identical in recording quality (0.857 vs 0.857, p = 0.995).

---

## 5. Quality and availability (Analyses 4, 5)

### Quality quartiles — non-monotonic, treat with caution

| stratum | N | pos | CTG | clinical |
|---|---:|---:|---:|---:|
| Q1 lowest | 137 | 35 | 0.7258 | 0.6910 [0.57–0.80] |
| Q2 | 137 | 28 | 0.6504 | 0.7444 [0.65–0.83] |
| Q3 | 136 | 27 | 0.5620 | 0.6470 [0.52–0.76] |
| **Q4 highest** | 137 | 20 | 0.7953 | **0.8299 [0.73–0.91]** |
| top half | 273 | 47 | 0.6713 | 0.7336 |
| bottom half | 274 | 63 | 0.6866 | 0.7159 |

Q4 at 0.8299 is the highest number in the project, but **it should not be
believed as a quality effect**: the trend is non-monotonic (Q1 > Q3), the
top-half/bottom-half contrast is negligible (0.7336 vs 0.7159), the subgroup has
only 20 positives, and quality is itself weakly associated with outcome
(r = −0.098, p = 0.023) so the strata differ in case mix. Exploratory only.

### Recording length — better powered and more credible

| stratum | N | pos | CTG | clinical |
|---|---:|---:|---:|---:|
| < 15 windows | 122 | 31 | 0.6565 | 0.6735 [0.56–0.78] |
| **17 windows (full)** | 381 | 72 | 0.7055 | **0.7672 [0.71–0.82]** |

**+0.094 for complete recordings**, on 381 patients / 72 positives. Window count
does not predict the label (AUROC 0.4657, p = 0.171), so this is not a bag-size
artifact. This is the most credible signal in the phase that **information
availability, not model capacity, is a live constraint**.

---

## 6. Outcome ambiguity (Analysis 7)

For each patient, taking the physiologically most similar other patient in the
19-feature space:

- median |Δ pH| = **0.090** — comparable to the 0.10 separating pH 7.05 from 7.15
- label disagreement = **27.4%**, versus 32.1% expected by chance at this prevalence
- cohort pH sd = 0.103; sd of nearest-neighbour pH differences = 0.083

Physiologically near-identical patients have labels only slightly more concordant
than chance. This is **evidence consistent with** a CTG–pH information limit. It
is not a proof, and no irreducible-noise figure is claimed.

---

## 7. Power (Analysis 8)

| quantity | value |
|---|---:|
| single-model AUROC 95% CI half-width | ±0.0546 |
| paired-difference bootstrap SE | 0.0229 |
| **minimum detectable difference, 80% power** | **0.0642** |
| minimum detectable difference, 90% power | 0.0743 |
| observed clinical − CTG | +0.0435 (below MDE) |

Even the +0.0435 gap between the clinical and CTG models is **below the detection
threshold** for this cohort. Every architecture, knowledge-infusion, temporal and
gating null in this project sits far below 0.0642.

---

## 8. What remains untested

| source | status | note |
|---|---|---|
| beat-to-beat detail beyond the 19 descriptors | **UNTESTED** | 8 architectures tried, but all on the same 20-min windows |
| **CTG before the final 60 minutes** | **AVAILABLE, DISCARDED** | 100% of recordings exceed 60 min; median 70, max 90; **19% of signal discarded** by `pipeline_clinical.py:190` |
| **maternal / clinical covariates** | **UNTESTED** | 13 fields present: gestational weeks, birth weight, sex, age, gravidity, parity, diabetes, hypertension, preeclampsia, pyrexia, meconium, presentation, delivery type |
| second-stage timing (`pos._ii.st.`) | partially used | metadata only, never a feature |
| repeated measurements / external cohort | UNAVAILABLE | single-centre, 552 records |

---

## 9. Decision

### ONE TARGETED EXPERIMENT

Not "stop", because the power analysis shows the nulls cannot exclude effects
below 0.064, and because two genuinely untested information sources exist. Not
"revise the task", because Analysis 3 shows the binary threshold is where the
models perform best — the continuous target is not hiding signal.

**The experiment, with the new information stated explicitly:**

> Add (a) the **19% of each recording currently discarded** by the 60-minute crop,
> and (b) the **13 maternal/clinical covariates** in the metadata, to the existing
> clinical baseline. Evaluate on the frozen protocol against the pre-registered
> bar.

Justification from this phase's evidence, not speculation:

- Full-length recordings score **0.7672** versus 0.6735 for short ones (+0.094,
  the only effect measured in this phase that exceeds the 0.0642 MDE). More
  observation is associated with better prediction, and 19% more is available.
- Maternal covariates are the only untested source of *non-CTG* information;
  every previous experiment recombined the same CTG-derived signal, which
  Analyses 1, 6 and 7 indicate is largely exhausted.

**Pre-registered bar: +0.0642** — the minimum detectable effect — not +0.02.
Using the old bar would guarantee an uninterpretable result.

**Scope caveat to state in any writeup:** adding maternal covariates changes the
task from "CTG-based prediction" to "CTG plus clinical context". That is
clinically legitimate — all 13 are known before delivery — but it is a different
research question and must be reported as such, with the CTG-only model retained
as the comparator.

**Explicitly NOT recommended:** further architecture search, further knowledge
infusion over the existing 19 descriptors, temporal Transformers, or any
experiment whose plausible effect is below 0.0642.
