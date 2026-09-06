# Persistence × severity — final FHRMA-only measurement study

Written 2026-09-04. `python scripts/figo_burden_2d.py`.

FHRMA only. No pH, BDecf, Apgar, early-warning label, CTU-UHB target or
outcome-driven threshold search was read or used. FHRMA carries no outcome,
so nothing here *can* be tuned toward prediction.

## DECISION: B — CONTINUOUS REPRESENTATION ONLY

**v4 is not created.** `area_per_contraction` is retained as a validated
measurement result. **The 2D persistence × severity extension is rejected**:
it does not improve the mid-range, does not preserve extreme discrimination,
and its apparent within-record gain is an artefact of fitting to the
reference it is scored against.

The label-reconstruction branch closes. §8 states which limitation is
binding.

---

## 1. Existing evidence, reproduced and frozen

| measure | within-record ρ | was | top-decile AUROC | was | |
|---|---:|---:|---:|---:|---|
| area_per_contraction | 0.400 | 0.400 | 0.982 | 0.982 | OK |
| max_consecutive_uc_with_decel | 0.667 | 0.667 | 0.778 | 0.778 | OK |

998 epochs, 154 recordings, 149 with ≥4 epochs. Same epochs, same
preprocessing, detector untouched. Reproduction is exact.

### The reference problem — stated before any result

FHRMA annotates deceleration **spans**, not burden. Every "expert burden"
quantity here is *derived* by applying one of our own summaries to the
expert's spans, and that choice is not neutral: against an expert **severity**
reference, `area_per_contraction` is favoured by construction; against an
expert **persistence** reference, `max_consecutive` is. Both references are
therefore reported side by side throughout, and only a claim surviving both
is treated as supported.

**A limitation that constrains the whole study:** the expert references are
coarse.

| reference | zeros | distinct values | tiers a quartile split actually yields |
|---|---:|---:|---:|
| expert area/contraction | 32.7% | 315 | **3** (not 4) |
| expert max-run | 42.6% | **7** | **2** (not 4) |

The brief asks for low / moderate / high / extreme tiers. **The expert
reference cannot define four tiers.** All Phase 4 results below are therefore
on 3 and 2 tiers respectively, and the "extreme" column is undefined — not
omitted for convenience.

---

## 2. The 2D representation — are the dimensions complementary?

Correlation between the two dimensions: **+0.346** in our measures, **+0.444**
in the expert's. So they are related but far from redundant.

Marginal and partial rank association with each reference (partial =
residualise both against the other dimension):

| reference | severity | persistence | severity \| persistence | persistence \| severity |
|---|---:|---:|---:|---:|
| severity ref | 0.629 | 0.463 | **0.539** | **0.250** |
| persistence ref | **−0.007** | 0.614 | **−0.347** | 0.629 |

**Against the severity reference the two are genuinely complementary** —
both retain non-zero partial association (0.539, 0.250).

**Against the persistence reference severity is useless and then harmful** —
marginal −0.007, partial **−0.347**. Once persistence is accounted for,
severity points the *wrong way*.

This is the study's first substantive finding, and it is negative: the two
dimensions do not agree about what "burden" is. Each tracks its own reference
and is uninformative or anti-informative about the other. That is not the
signature of a single latent burden that a 2D space measures better — it is
the signature of two different quantities.

---

## 3. Within-record ordering

| representation | median ρ | IQR | 95% CI | records | ρ ≤ 0 |
|---|---:|---|---|---:|---:|
| **severity reference** | | | | | |
| area_per_contraction alone | 0.400 | [+0.40, +0.68] | [+0.40, +0.40] | 144 | 3% |
| max_consecutive alone | 0.289 | [+0.29, +0.29] | [+0.29, +0.29] | 143 | 6% |
| 2D, LOO-fitted | **0.800** | [+0.80, +0.80] | [+0.80, +0.80] | 144 | 2% |
| 2D, Pareto (no weights) | **0.833** | — | concordance on **60%** of pairs | 143 | — |
| **persistence reference** | | | | | |
| area_per_contraction alone | **−0.866** | [−0.87, 0.00] | [−0.87, −0.87] | 143 | **76%** |
| max_consecutive alone | 0.667 | [+0.67, +0.67] | [+0.67, +0.67] | 143 | 6% |
| 2D, LOO-fitted | **0.866** | [+0.67, +0.87] | [+0.87, +0.87] | 143 | 7% |
| 2D, Pareto (no weights) | 0.804 | — | concordance on **48%** of pairs | 52 | — |

**The LOO-fitted numbers must not be read as a gain.** The combination is
fitted by regression *onto the expert reference it is then scored against*.
Leave-one-record-out stops a recording influencing its own score; it does not
stop the weights being optimised toward that reference in general. An arm
fitted to the target will beat arms that are not, and 0.400 → 0.800 is mostly
that.

**The Pareto result is the honest one**, because it involves no weights at
all: where the 2D representation makes an unambiguous claim (one epoch ≥ the
other on *both* dimensions) it orders correctly 83.3% of the time — but it
makes that claim on only **60%** of within-record pairs and abstains on the
rest. Against the persistence reference, coverage falls to 48% and only 52 of
143 records have enough comparable pairs to score at all.

So: a real but partial ordering improvement, purchased with 40–52%
abstention.

---

## 4. Severity tiers — the criterion the brief made decisive

> "The joint representation should only be considered an improvement if it
> improves the mid-range as well as the extremes."

Expert **severity** reference (3 tiers; per-tier columns are recall):

| representation | exact | ±1 | weighted κ | low | moderate | high |
|---|---:|---:|---:|---:|---:|---:|
| area_per_contraction alone | 0.487 | 0.865 | 0.545 | 0.50 | 0.52 | **0.43** |
| max_consecutive alone | 0.530 | 0.843 | 0.193 | 0.81 | 0.44 | 0.05 |
| **2D, LOO-fitted** | 0.389 | 0.865 | 0.567 | 0.50 | 0.51 | **0.05** |

Expert **persistence** reference (2 tiers):

| representation | exact | ±1 | weighted κ |
|---|---:|---:|---:|
| area_per_contraction alone | 0.307 | 0.521 | −0.006 |
| max_consecutive alone | 0.676 | 0.989 | **0.346** |
| 2D, LOO-fitted | 0.275 | 0.549 | **0.095** |

**The 2D representation fails this criterion outright.** On the severity
reference its weighted κ rises trivially (0.545 → 0.567) while recall in the
**high** tier collapses from **0.43 to 0.05** — it gains a little on the
aggregate by giving up the upper-middle band almost entirely, which is the
opposite of the requirement. On the persistence reference it is far worse
than persistence alone (κ 0.095 vs 0.346).

Note also that even the best single measure reaches only κ 0.545 with 0.43
recall in the high tier. No representation examined assigns expert severity
tiers reliably.

---

## 5. Extreme-burden discrimination

Benchmark to beat: `area_per_contraction` at **0.982** for the expert top decile.

| representation | severity ref, top 25% | top 10% | persistence ref, top 25% | top 10% |
|---|---:|---:|---:|---:|
| area_per_contraction | 0.924 | **0.982** | 0.512 | 0.455 |
| max_consecutive | 0.643 | 0.453 | 0.814 | 0.778 |
| 2D, LOO-fitted | **0.970** | 0.891 | **0.858** | **0.810** |

**The benchmark is not beaten: 0.982 → 0.891.** The 2D arm is more *even*
across the two references, which is what a fitted compromise produces, but it
is worse than the best single measure at the thing that measure is for. Per
the brief, a metric improving elsewhere is not success.

---

## 6. Failure-mode structure

Severity high = ≥ median of non-zero (1433 bpm·s/contraction); persistence
high = ≥2 consecutive contractions; expert burden high = top quartile.

| quadrant | n | % | contractions | mean expert burden |
|---|---:|---:|---:|---:|
| **A** persistence high, expert burden low | 216 | **21.6%** | 4.38 | 800.5 |
| **B** severity high, persistence low | 242 | **24.2%** | 3.77 | 1355.4 |
| **C** both high | 225 | 22.5% | 4.42 | 1422.4 |
| **D** both low, expert burden high | **2** | **0.2%** | 4.00 | 1448.6 |

Reading this honestly:

* **D is essentially empty (n = 2).** The detector almost never misses a
  genuinely high-burden epoch on both dimensions at once. That is a real
  positive finding about the detector.
* **A and B together are 45.8% of epochs.** Nearly half sit in a quadrant
  where the two dimensions disagree. A (persistence high, expert burden low)
  is the false-repetitive mode identified earlier and is large.
* **B and C have almost the same mean expert burden** (1355 vs 1422) despite
  differing in persistence. Adding persistence to severity therefore does
  little to identify high expert burden — which is the same conclusion §2
  reached from the partial correlations, arrived at independently.

The space has *interpretable* structure, but not the structure a joint
criterion would need: the persistence axis does not separate expert burden
once severity is known.

---

## 7. Can a rule be defined without outcome tuning?

Thresholds taken only from the expert/our own distributions — severity cut at
our 75th percentile (1880 bpm·s/contraction), persistence cut at 2
consecutive contractions (the minimal physiological meaning of "repetitive":
two successive contractions responding). Target is the **expert's**
top-quartile burden — a measurement reference, not a clinical state and not
an outcome.

| rule | fires | sens | prec | F1 | κ |
|---|---:|---:|---:|---:|---:|
| high persistence **AND** high severity | 2.8% | 0.096 | 0.857 | 0.173 | 0.129 |
| high persistence **OR** high severity | 57.0% | 0.976 | 0.429 | 0.596 | 0.380 |
| persistence ≥ 2 alone | 34.2% | 0.500 | 0.367 | 0.423 | 0.188 |
| **severity ≥ p75 alone** | 25.7% | 0.572 | 0.559 | **0.565** | **0.418** |
| **current v3 rule (>50% of contractions)** | 23.1% | 0.512 | 0.554 | 0.532 | **0.384** |

**No joint rule beats the current v3 rule meaningfully.** AND fires on 2.8%
of epochs (far too rare to be a Suspicious criterion); OR fires on 57% (far
too often). The best performer is a *single-dimension* rule — severity ≥ p75
at κ 0.418 versus v3's 0.384 — a +0.034 improvement, and one measured against
a severity reference that shares its functional form, so even that is
partly circular.

**Conclusion for Phase 7:** a useful continuous representation exists, but a
defensible categorical FIGO replacement cannot be derived from this dataset.

---

## 8. Final decision

### DECISION B — CONTINUOUS REPRESENTATION ONLY

Against the Decision-A criteria:

| requirement | met? |
|---|---|
| improves within-record ordering | **partially** — only via a fit to the reference; unfitted Pareto gains cover 60%/48% of pairs |
| improves severity-tier agreement | **NO** — high-tier recall collapses 0.43 → 0.05 |
| preserves extreme discrimination | **NO** — 0.982 → 0.891 |
| reduces the binary rule's brittleness | yes, inherently (continuous, no denominator cliff) |
| failure modes clinically interpretable | **partially** — 45.8% of epochs in disagreement quadrants |
| categorical rule without outcome tuning | **NO** — no joint rule beats v3 |

Three requirements fail outright. **Decision A is not available.**

Not Decision C either: `area_per_contraction` is a genuinely validated
measurement (top-decile AUROC 0.982, best tier κ 0.545, within-record ρ 0.400
with only 3% of records non-positive). That result stands and should be kept.
What closes is the **2D joint extension** and the **label-reconstruction**
attempt.

### Which limitation is binding — the distinction the brief asks for

1. **Not a model limitation.** No model was involved in this study, and the
   earlier tracks already established capacity was never the constraint.
2. **Not primarily a detector limitation** — though the detector is
   imperfect. Quadrant D contains 2 epochs: the detector almost never misses
   a high-burden epoch outright. Its errors are false positives (quadrant A,
   21.6%), which is a precision problem and a real one, but it is not what
   blocks v4.
3. **The binding limitation is the absence of a defensible categorical
   clinical threshold.** FIGO 2015 supplies one number for this criterion
   (>50% of contractions). It supplies none for deceleration area,
   area-per-contraction, or run length, and none was found in the
   literature. FHRMA cannot supply one either: its derived burden reference
   is too coarse to define even four severity tiers (32.7% zeros; the
   persistence reference has 7 distinct values). Any threshold would be
   arbitrary, outcome-fitted (forbidden), or calibrated to reproduce v3 —
   which is the >50% rule renamed.
4. **A secondary, genuine limitation is the reference itself.** "Expert
   burden" is derived, not annotated. A study that could settle this needs a
   corpus with expert **FIGO state** annotations, which FHRMA is not.

### What is preserved

* `src/figo_state/burden.py` — 15 continuous burden quantities, and
  `data/processed_figo/burden.npz` computed for all 3,909 CTU-UHB epochs.
* The v3 reconciliation: 100.00% agreement, zero unexplained residual, with
  the 2.1% attributed entirely to prolonged-deceleration pairing.
* Two implementation defects in `descriptors.py`, both documented and both
  corrections to guideline fidelity rather than new clinical claims:
  prolonged decelerations cannot currently satisfy repetitiveness (74 epochs,
  2.1%), and zero-contraction epochs read benign (357 epochs, 10.2%, of which
  311 carry decelerations).

### What is not claimed

That the >50% rule is adequate — §2 of the redesign study measured 48.3% of
epochs as flippable by a single deceleration, and that finding stands. The
rule is demonstrably brittle **and** no better-supported replacement can be
derived from the data available. Those are compatible, and reporting both is
the result.
