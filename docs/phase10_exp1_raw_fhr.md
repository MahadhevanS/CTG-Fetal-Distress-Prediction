# Phase 10 / Experiment 1 — Raw FHR information test

Completed 2026-09-04.

**Question:** does minimally processed raw FHR contain predictive information
that the clinical descriptors do not capture?

**Answer: no. Verdict KILL on both variants, by the pre-registered gate.**

Code: [`src/preprocessing/pipeline_raw_minimal.py`](../src/preprocessing/pipeline_raw_minimal.py),
[`scripts/exp1_raw_fhr_information.py`](../scripts/exp1_raw_fhr_information.py).

---

## 1. Result

| model | representation | AUROC | 95 % CI | AUPRC | Δ vs LR | verdict |
|---|---|---:|---|---:|---:|---|
| **Clinical LR** | 19 descriptors | **0.7271** | [0.670–0.779] | 0.4094 | — | baseline |
| **Clinical LR** | **11 FHR-only descriptors** | **0.7300** | [0.673–0.781] | 0.4160 | +0.0029 | *matched comparator* |
| Clinical LR | 8 UC-derived descriptors | 0.6200 | [0.560–0.678] | 0.2874 | −0.1071 | — |
| E1-A | raw FHR @ 1 Hz | **0.5865** | [0.528–0.645] | 0.2518 | **−0.1406** | **KILL** |
| E1-B | raw FHR @ 4 Hz | **0.6117** | [0.551–0.670] | 0.2893 | **−0.1154** | **KILL** |

Aggregation was held fixed at the baseline's (20-min windows → per-window
probability → patient max), so only the representation differs.

**E1-C (multi-resolution) was not run.** The gate requires PROMOTE (≥ 0.75) and
the best result was 0.6117.

### Secondary metrics

| | window AUROC | fold SD | sens@80spec | spec@90sens | ECE |
|---|---:|---:|---:|---:|---:|
| E1-A (1 Hz) | 0.5708 | 0.0309 | 0.291 | — | 0.551 |
| E1-B (4 Hz) | 0.5761 | 0.0210 | 0.336 | — | 0.507 |

Calibration is worse than any previous arm (ECE 0.51–0.55).

---

## 2. The matched-comparator check — the objection this experiment had to survive

Experiment 1 used **FHR only**; the 19 descriptors include UC-derived features.
Comparing them directly would have been unfair, and a −0.115 gap could have been
"the raw model simply lacks the contraction channel".

Partitioning the 19 by whether UC is required to compute them:

| descriptor set | n | AUROC |
|---|---:|---:|
| all 19 | 19 | 0.7271 |
| **FHR-only** (baseline, STV, LTV, accels, prolonged decels, decel depth/area/burden/duration, baseline & variability slope) | 11 | **0.7300** |
| UC-derived (early/late/variable decels, contraction count, tachysystole, amplitude, lag, coupling) | 8 | 0.6200 |

**The FHR-only descriptors score 0.7300 — slightly better than all 19.** The UC
features contribute nothing, consistent with Phase 8's contraction findings.

So the correct comparison is **0.6117 vs 0.7300, a gap of −0.1183**, and the
absence of UC does not explain it. The objection is closed.

---

## 3. Resolution does matter slightly, and it does not rescue the route

4 Hz beats 1 Hz by **+0.0252** (0.6117 vs 0.5865). Higher temporal resolution
carries a little more usable signal — which is a coherent finding, and consistent
with Phase 8's measurement that 98 % of FHR spectral power lies below 0.5 Hz, so
the extra band is thin. It is nowhere near enough to change the verdict.

Both raw variants also fall below the random-initialised CrossFormer (0.6489)
from Phase 9B, which operates on the filtered, baseline-corrected representation.

---

## 4. Preprocessing audit (Phase 10 §8.1)

| property | value |
|---|---|
| sampling frequency | 4.0 Hz as distributed; 1 Hz variant by 4-sample mean of **measured** samples |
| temporal range | final 60 minutes (14 400 samples) |
| artifact criterion | `remove_spikes`: \|d(FHR)/dt\| > 25 bpm/s → sample zeroed, then interpolated |
| interpolation | cubic spline for gaps ≤ 15 s; linear across longer gaps |
| max interpolated gap | unbounded, but every interpolated sample is flagged in channel 1 |
| smoothing | **none** (pipeline_clinical applies a 1.5 Hz low-pass; this does not) |
| filtering | **none** |
| baseline handling | **none** — absolute bpm retained |
| clipping | [50, 240] bpm |
| normalisation | per-channel z-score, fit on **training-fold windows only** |
| window / stride | 20 min / 2.5 min |
| window quality gate | drop if > 50 % of raw FHR samples missing |
| channels | [0] FHR bpm, [1] missingness mask |
| model | residual 1D CNN, **252 545 parameters** (budget 100k–300k, asserted at runtime) |
| optimiser | AdamW, lr 1e-3, wd 1e-4, batch 64, cosine schedule |
| epoch limit / stopping | 40 / inner-validation AUROC, patience 8 |
| folds | the five frozen patient-grouped folds; fit/val and fit/test patient-disjointness asserted per fold |

### The one deviation from a literal "raw FHR" reading

`pipeline_clinical` leaves gaps > 15 s as 0.0, and **7.05 % of CTU samples sit in
such gaps** (Phase 9B). Feeding those to a CNN as heart rate teaches it an
artifact; zero-filling fabricates a bradycardia, interpolating fabricates a
plausible trace. Both were rejected as sole options: gaps are interpolated **and**
a missingness channel tells the model which samples were measured. The input is
therefore 2-channel, not 1. This is metadata, not physiology, but it is a
deviation and is recorded as one.

---

## 5. Interpretation — Phase 10 §32, Case 1

> **Case 1 — raw FHR < clinical baseline.** The current raw representation does
> not expose additional useful signal.

Stated precisely, and bounded:

**Supportable**

1. Minimally processed raw FHR, at either 1 Hz or 4 Hz, under a fixed
   window→max aggregation and a small CNN, carries **less** usable information
   about fetal acidaemia than 11 hand-computed FHR descriptors (−0.1183).
2. The gap is not explained by the absence of the UC channel: the matched
   FHR-only comparator scores 0.7300.
3. Higher temporal resolution helps marginally (+0.0252 for 4 Hz over 1 Hz) but
   does not approach the baseline.

**Not supportable**

- that raw FHR contains no additional information — only that *this*
  representation, *this* aggregation and *this* model do not extract any. A
  small CNN over 8 477 windows from 110 positive patients is a weak extractor,
  and that is exactly the limitation Experiment 2 (end-to-end patient MIL)
  exists to test;
- that the descriptors are near-optimal — Phase 9A showed they disagree
  substantially with expert annotation while still predicting better.

---

## 6. Gate decision

Per §33, Priority 2 (small end-to-end patient MIL) is conditioned on
**"only if Priority 1 passes"**, and Priority 1 did not pass.

The honest reading is that the *representation* test failed while the
*aggregation* question remains formally untested. These are separable: this
experiment deliberately froze aggregation to isolate the representation. Whether
to spend Priority 2 on a route whose representation just scored 0.61 is a
scoping decision, not a technical one, and is left to the project owner rather
than assumed either way.

---

## 7. Running comparison (Phase 10 §29)

| model | representation | AUROC | 95 % CI | Δ vs LR | status |
|---|---|---:|---|---:|---|
| Clinical LR | 19 descriptors | **0.7271** | [0.670–0.779] | — | **baseline** |
| Clinical LR | 11 FHR-only descriptors | 0.7300 | [0.673–0.781] | +0.0029 | reference |
| CrossFormer | filtered, baseline-corrected + UC | 0.6489 | [0.596–0.701] | −0.0782 | rejected |
| SSL CrossFormer | + CTU+FHRMA pre-training | 0.6494 | [0.591–0.704] | −0.0777 | rejected (Phase 9B) |
| **Raw CNN 4 Hz** | **minimally processed raw FHR** | **0.6117** | **[0.551–0.670]** | **−0.1154** | **KILL** |
| **Raw CNN 1 Hz** | **minimally processed raw FHR** | **0.5865** | **[0.528–0.645]** | **−0.1406** | **KILL** |
| Multi-scale CNN | multi-resolution FHR | — | — | — | not run (gated) |
| Patient MIL | raw FHR | — | — | — | pending decision |
| FHR + UC | response waveform | — | — | — | pending |
| Stage MIL | delivery-aligned | — | — | — | pending |
