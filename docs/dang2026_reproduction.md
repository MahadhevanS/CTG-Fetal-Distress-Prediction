# Reproduction of Dang et al. 2026 (CTG-CrossFormer, AUC 0.822)

Run 2026-09-04. Purpose: determine which documented pipeline differences account
for the gap between the published 0.822 and this project's frozen-protocol
0.7271 — **explanation, not score maximisation**.

Two tracks are kept strictly separate and are never combined into one number:

- **Track A — paper reproduction.** The paper's cohort rule, preprocessing,
  windowing, split, model, checkpoint selection and metric.
- **Track B — frozen project protocol.** Patient-grouped folds from
  `data/processed_clinical/folds.json`, patient-level max aggregation,
  patient-bootstrap uncertainty, no selection on the reported fold. **Track B
  was not modified for this exercise.**

Code: [`src/preprocessing/pipeline_paper_literal.py`](../src/preprocessing/pipeline_paper_literal.py),
[`scripts/reproduce_dang2026.py`](../scripts/reproduce_dang2026.py),
[`scripts/reproduce_dang2026_report.py`](../scripts/reproduce_dang2026_report.py).
Out-of-fold predictions for every configuration are preserved in
`results/phase8/repro/*.npz`, so cohort slices and aggregation changes are
computed post-hoc from identical predictions rather than by retraining.

---

## 1. Paper specification

Every methodological detail stated by Dang, Nguyen & Ho, *"A Hybrid
CNN-Transformer with Cross-Attention for Automated Fetal Distress Detection from
Cardiotocography"*, E3S Web of Conferences **723**, 01005 (2026), AIEI 2026.
Verbatim quotes are from
[`literature_archive/dang_2026_crossformer.txt`](literature_archive/dang_2026_crossformer.txt).

| # | aspect | paper's stated value | source |
|---|---|---|---|
| 1 | dataset | CTU-UHB, 552 recordings, 4 Hz, FHR + UC | §3.1 |
| 2 | label | pH **< 7.15** → Acidosis; pH ≥ 7.15 → Normal, at **recording** level | §3.1 |
| 3 | cohort filter | "recordings with >50% missing FHR values are excluded" → **404 patients** | §3.1 |
| 4 | interpolation | **linear**, gaps ≤ 15 s only; "longer gaps are preserved as-is" | §3.1 |
| 5 | normalisation | "**per-recording z-score** normalization" on both FHR and UC | §3.1 |
| 6 | windowing | 20-min windows (4 800 samples), **5-min stride** → **1 753 windows**, 16.6 % acidosis | §3.1 |
| 7 | truncation | *not mentioned* — "continuous recordings are segmented" | §3.1 |
| 8 | spike removal / low-pass / baseline | *not mentioned* | §3.1 |
| 9 | augmentation | jitter σ=0.03, time-warp σ=0.2, magnitude-warp σ=0.2 (4 knots); UC jitter σ=0.015 | §3.1 |
| 10 | CNN branches | 3 blocks/channel, kernels 7/5/3, channels 32/64/128, pools 4/4/2, ×32 downsample | §3.2.1 |
| 11 | SE module | reduction ratio r = 4 | §3.2.1 |
| 12 | cross-attention | bidirectional, **4 heads**, d_k = 32 | §3.2.2 |
| 13 | transformer | **4 layers**, **8 heads**, d_ff = 512, learnable positional encoding | §3.2.3 |
| 14 | head | GAP → MLP **256 → 128 → 2**, dropout (0.3, 0.15), ReLU | §3.2.4 |
| 15 | loss | **Focal Loss γ = 2.0** + inverse-frequency class weights | §3.3 |
| 16 | sampling | **sqrt-inverse** frequency `WeightedRandomSampler` | §3.3 |
| 17 | optimiser | AdamW, weight decay 1e-5 | §3.3 |
| 18 | schedule | OneCycleLR, 10 % warmup, max_lr 3e-4, cosine → 1.2e-9 | §3.3 |
| 19 | batch / precision | 32, AMP | §3.3 |
| 20 | early stopping | patience 15, **monitors validation AUC-ROC** | §3.3 |
| 21 | split | **5-fold Stratified Group K-Fold**, grouped by patient | §3.3 |
| 22 | inner validation split | **not described** — the CV is the whole evaluation | §3.3 |
| 23 | held-out test set | **none described** | §3.3 |
| 24 | metric | "AUC computed on **pooled predictions across all folds**" | Table 1 |
| 25 | evaluation unit | **window**; no patient-level figure appears anywhere | Table 1 |
| 26 | reported result | **AUC 0.822**, Sens 0.895, Spec 0.414, F1 0.471 | Table 1 |
| 27 | per-fold AUCs | 0.903, 0.802, 0.808, 0.806, 0.808 (mean 0.825) | §4 |
| 28 | significance | DeLong on AUC; McNemar on classification | §3.3 |

### 1.1 Implementation choices where the paper is silent

Recorded so they are visible as degrees of freedom, not hidden as defaults.

| choice | what was done | why |
|---|---|---|
| per-window quality gate | **none** (literal) | the paper specifies a per-recording gate only |
| z-score reference samples | all samples of the recording | literal reading of "per-recording z-score" |
| classification head | 256 → 128 → **1** logit | this repo's `ctg_crossformer.py`; equivalent for AUROC |
| pooling of fold predictions | reported **both** raw and per-fold Platt-scaled | independently trained folds have different logit scales |
| augmentation | **not applied** | isolates protocol from regularisation; noted as a deviation |
| max epochs | 60 | the paper gives patience but no epoch cap |

---

## 2. The two irreducible discrepancies

Both were checked before any training, and neither can be resolved from the
paper's text. They are reported as unexplained degrees of freedom, **not** as
errors.

### 2.1 The cohort of 404 cannot be derived from the stated rule

Measured over all 552 records, six candidate definitions of "missing FHR":

| definition | median | max | records left at >50 % |
|---|---:|---:|---:|
| **(a) FHR zeros / record length** (literal) | 0.183 | **0.535** | **547** |
| (b) FHR zeros / 90-min nominal | 0.334 | 0.649 | 519 |
| (c) FHR or UC zero / length | 0.264 | 1.000 | 515 |
| (d) FHR zeros / 60-min nominal | 0.001 | 0.473 | 552 |
| (e) FHR zeros in last 60 min | 0.166 | 0.700 | 533 |
| (f) UC zeros / length | 0.174 | 1.000 | 531 |

**None yields 404.** The literal reading (a) removes 5 records and leaves 547 —
exactly this project's cohort. To leave 404, the threshold would have to be
≈ 26 %, not 50 %.

### 2.2 The window count is inconsistent with the stated windowing

1 753 windows / 404 patients = **4.34 windows per patient**. At a 20-minute
window and 5-minute stride that implies a **~37-minute** segment per recording.
CTU-UHB records are 60–90 minutes (median 70.6), and the paper describes no
truncation.

Building the paper's stated procedure literally gives:

| | patients | windows | windows/patient | acidosis |
|---|---:|---:|---:|---:|
| **paper states** | 404 | 1 753 | 4.34 | 16.6 % |
| **literal reconstruction** | 547 | 6 435 | 11.76 | 19.5 % |

---

## 2b. Fridman & Ben Shachar 2026 (AUC 0.83) — no reproduction needed

This paper requires a different treatment from Dang et al., because its
evaluation is already **recording-level**, i.e. directly comparable to Track B.
Three facts, all verbatim from
[`literature_archive/foundation_model_2601.06149.txt`](literature_archive/foundation_model_2601.06149.txt):

**(a) The paper itself places properly-evaluated CTU-UHB performance at 0.68–0.75.**

> "Reported AUC values range from 0.68 to over 0.95, but the highest figures
> typically reflect evaluation on private data or methodological issues. The
> public CTU-UHB database remains the only substantial benchmark; **on this
> dataset, properly evaluated methods achieve AUC of 0.68–0.75.**"

This project's frozen-protocol **0.7271 sits inside the range this paper itself
designates as correct**. That is a stronger positioning statement than anything
this project could assert on its own authority.

**(b) Their headline rests on a documented information advantage, not on protocol.**

| | recordings | hours |
|---|---:|---:|
| SSL pre-training corpus (CTGDL) | **984** | **2 444** |
| — CTGDL_CTU_UHB | 552 | |
| — CTGDL_FHRMA | 135 | |
| — CTGDL_SPAM | 297 | |
| fine-tuning (CTU-UHB) | 552 | |

They further **augment the positive class from outside CTU-UHB**: CTGDL_SPAM
caesarean cases (stage-2 duration zero) are added as positives "under the
assumption that intrapartum cesarean delivery typically indicates CTG
abnormalities".

So their advantage is **1.8× more recordings for representation learning plus
extra labelled positives** — not a different evaluation protocol. This is the
single most decision-relevant finding for the project: SSL confined to CTU-UHB
would not reproduce their setup, because the corpus is the point.

**(c) The test set is very small, and the paper is internally inconsistent about it.**

| source | N | acidemia | prevalence | AUC |
|---|---:|---:|---:|---:|
| Table 3, "All test cases" | 55 | **12** | 21.4 % | 0.826 |
| body text | 55 | **11** | 20.0 % | 0.826 |

Hanley–McNeil at 11 positives / 44 negatives: **95 % CI 0.673 – 0.987**, which
overlaps this project's 0.7271 [0.670–0.785] almost entirely. The reported 0.85
figures are *subgroups* (n = 43–50), smaller still.

**Status: not reproducible here and not necessary to reproduce.** The comparison
is settled by (a) and (c) without any claim about their method, and (b) tells us
what would actually be required to beat 0.73.

---

## 3. Reproduction result

Five configurations, each 5 folds of CTG-CrossFormer. Out-of-fold predictions are
preserved in `results/phase8/repro/*.npz`; every slice below is computed from
those same predictions, never by retraining toward a target.

### Track A — the paper's own metric (window-level, pooled across folds)

| run | substrate | selection | **pooled** | +Platt | mean-of-folds | clean-404 |
|---|---|---|---:|---:|---:|---:|
| R1 | paper-literal | `outer_best` (paper's) | **0.6450** | 0.6564 | 0.6482 | 0.6398 |
| R2 | paper-literal | nested (unbiased) | **0.5678** | 0.5988 | 0.5899 | 0.5430 |
| R3 | paper-match (+0.3 window gate) | `outer_best` | 0.6092 | 0.6312 | 0.6284 | 0.6219 |
| R4 | this project's clinical | nested | 0.5922 | 0.6141 | 0.6112 | 0.6191 |
| R5 | paper-literal, gaps filled | `outer_best` | *not run* | | | |
| — | **Dang et al. reported** | `outer_best` | **0.8220** | — | 0.8250 | — |

### Track B — frozen project protocol (patient-level, max-aggregated)

Reported separately and never mixed with Track A. **Track B was not modified for
this exercise.**

| run | patients | AUROC | 95 % CI | clean-404 |
|---|---:|---:|---|---:|
| R1 | 547 | 0.6647 | [0.603–0.727] | 0.6813 |
| R2 | 547 | 0.6021 | [0.539–0.665] | 0.5933 |
| R3 | 546 | 0.6605 | [0.600–0.721] | 0.6566 |
| R4 | 547 | 0.6240 | [0.563–0.685] | 0.6653 |
| **19-descriptor LR (frozen baseline)** | 547 | **0.7271** | [0.670–0.779] | 0.7530 |

> Under every configuration tested, the CrossFormer scores **below** the
> 19-descriptor logistic regression at patient level. That is a statement about
> this reproduction, **not** about the paper's model.

### Harness validity check

R4 places the CrossFormer at window-level 0.5922 on this project's clinical
substrate, beside the 19-descriptor LR's 0.6179 on the identical substrate and
folds. The harness therefore learns where signal is known to exist, and the low
Track A numbers are a property of the substrate rather than a broken loop.

This also resolves an apparent contradiction with this repo's historical
"window-level CV 0.8200" for CrossFormer: that number was measured on the **MIL
substrate**, whose horizon label
[auroc_ceiling_analysis.md](auroc_ceiling_analysis.md) showed to be 84 %
predictable from elapsed time alone. It is not comparable to this paper, which
uses no horizon rule.

---

## 4. Gap decomposition

Each component isolated by changing one thing and holding the rest fixed.

| # | component | measured effect | direction |
|---|---|---:|---|
| 1 | **epoch selection on the reported fold** (R1 − R2) | **+0.0772** | explains part of the gap |
| 2 | per-fold Platt scaling before pooling (R1) | +0.0114 | interpretive, minor |
| 3 | mean-of-folds vs pooled (R1) | +0.0032 | negligible |
| 4 | undocumented 0.3 per-window gate (R3 − R1) | −0.0358 | **hurts** |
| 5 | cohort → cleanest 404 (R1) | −0.0052 | **does not help** |
| 6 | cohort → cleanest 404 (R2) | −0.0248 | **does not help** |
| 7 | this project's preprocessing vs paper-literal (R4 − R2) | +0.0244 | ours slightly better |
| | **total explained** | **≈ +0.077** | |
| | **remaining gap to 0.822** | **≈ 0.177** | **unexplained** |

Component 1 independently reproduces this repo's earlier, separately measured
epoch-selection bias of ~0.077 on different data — two routes, one answer.

### 4.1 Where the paper's stated preprocessing loses signal

Measured *before any training*, as window-level AUROC of plain FHR statistics, so
this is a property of the representation and not of any model:

| representation | sd-AUROC | mean-AUROC |
|---|---:|---:|
| absolute bpm, valid samples only | **0.6128** | 0.5015 |
| paper-literal: per-recording z-score, gaps preserved | **0.5046** | 0.5182 |
| global normalisation, gaps preserved | 0.5438 | 0.5036 |
| global normalisation, gaps filled | 0.5992 | — |

Two stated choices each cost signal:

* **"per-recording z-score normalization"** removes between-recording differences
  in variability amplitude — and FHR variability is the strongest simple
  predictor available in this dataset.
* **"longer gaps are preserved as-is"** leaves raw zeros in the model input;
  after scaling they dominate window variance.

Together they take the strongest simple feature from 0.613 to 0.505. This makes
0.822 *harder* to explain from the stated pipeline, not easier.

### 4.2 An anomaly in the reported per-fold values

The paper reports per-fold AUCs 0.903, **0.802, 0.808, 0.806, 0.808**.

At 1 753 windows over 5 folds — ~350 windows per fold, ~58 positive — the
Hanley–McNeil sampling SE per fold at AUC ≈ 0.805 is **0.0360**.

| | sd |
|---|---:|
| paper, folds 2–5 | **0.00283** |
| expected from sampling noise alone | 0.0360 |
| this reproduction, all 5 folds | 0.0662 |

Folds 2–5 span 0.006 where sampling noise alone predicts ~0.036 per fold — a
spread **13× tighter** than chance permits for disjoint test sets.

**No cause is attributed.** Rounding, a reporting convention, or a mechanism not
described in the paper could each be responsible. It is recorded because anyone
recomputing an interval from these five numbers needs to know it.

---

## 5. Reproduction status

### NOT REPRODUCED

| | AUROC |
|---|---:|
| Dang et al. reported | 0.822 |
| best literal configuration, **using the paper's own selection procedure** | 0.6450 |
| literal configuration, unbiased selection | 0.5678 |
| **shortfall after every documented difference is applied** | **0.177** |

The classification is *not reproduced* rather than *partially reproduced*: the
documented differences account for ~0.077 of a ~0.254 total gap, and the largest
single component has no identified source. Applying the paper's own optimistic
selection procedure — the configuration most favourable to matching its number —
still falls 0.177 short.

**This is not a claim that the paper is erroneous.** §6 lists what would have to
be known to close it.

---

## 6. Remaining unexplained degrees of freedom

Enumerated, not tuned. No parameter was searched toward 0.822.

| # | degree of freedom | what is unknown | could it plausibly be worth 0.177? |
|---|---|---|---|
| 1 | **cohort selection** | how 552 → 404 was actually performed; no tested definition of "missing" reproduces it | measured at −0.005 to −0.025 here, i.e. the wrong sign — but an *outcome-correlated* selection rule could behave differently |
| 2 | **segment extraction** | 4.34 windows/patient implies ~37 min per recording; no truncation is described | plausibly material — it is unknown *which* 37 minutes |
| 3 | **inner validation split** | §3.3 describes none; if one exists, component 1 of §4 does not apply and the gap **grows** | no — it widens the gap |
| 4 | **gap handling in practice** | whether preserved zeros were masked before the encoder | worth ~0.09 on simple features (§4.1). A CrossFormer run (R5) was started and **stopped at fold 4 of 5** by a decision to reprioritise; its four completed folds were 0.6517 / 0.6326 / 0.6576 / 0.6128, i.e. no material recovery over R1's 0.6450 — **indicative only, no pooled figure is claimed** |
| 5 | **z-score reference** | over all samples, or valid samples only | worth ~0.04 on simple features |
| 6 | **augmentation** | jitter / time-warp / magnitude-warp specified but not applied here | regularisation; unlikely to be worth 0.177 alone |
| 7 | **released implementation** | not obtained; the paper's text was the only available source | the decisive artifact |

Item 7 is the honest bottom line: **this reproduction is bounded by the paper's
text.** Nothing here rules out that a correct implementation reaches 0.822.

---

## 7. Final comparison

| Result | Model | Cohort | Split | Evaluation unit | AUROC |
|---|---|---|---|---|---:|
| **Dang et al. 2026 (reported)** | CTG-CrossFormer | 404 (rule not reproducible) | 5-fold Stratified Group K-Fold, patient-grouped | **window**, pooled | **0.822** |
| **Fridman & Ben Shachar 2026 (reported)** | SSL transformer + LR alert stage | 552; 55 test, 11–12 positive | recording-level 441 / 56 / 55 | **recording** | **0.826** [0.673–0.987] |
| **Reproduction, paper's selection** (R1) | CTG-CrossFormer | 547 (literal >50 % rule) | 5-fold Stratified Group K-Fold, patient-grouped | window, pooled | **0.6450** |
| **Reproduction, unbiased selection** (R2) | CTG-CrossFormer | 547 | 5-fold Stratified Group K-Fold, patient-grouped | window, pooled | **0.5678** |
| **Reproduction, patient-level** (R1, Track B) | CTG-CrossFormer | 547 | frozen patient-grouped folds | patient, max | 0.6647 |
| **Our frozen baseline** | 19-descriptor logistic regression | 547 | patient-grouped | patient, max | **0.7271** [0.670–0.779] |

Per the brief's §6 requirement, the 0.6507 window-level logistic-regression figure
from [literature_forensic_audit.md](literature_forensic_audit.md) §7.3 is **not**
used anywhere above as evidence about the paper's CrossFormer. The comparison to
Dang et al. is made only against R1/R2, which are the same architecture.

---

## 8. What this licenses, and what it does not

**Supportable**

1. "The published methodology, implemented as described, did not reproduce the
   published result. Under the paper's own evaluation metric and its own
   checkpoint-selection procedure, we obtain 0.6450 against a reported 0.822."
2. "Epoch selection on the reported fold accounts for +0.077, measured twice by
   independent routes."
3. "The stated cohort rule does not yield the stated cohort, and the stated
   windowing does not yield the stated window count."
4. "The stated preprocessing removes the strongest simple predictor available in
   this signal (window AUROC 0.613 → 0.505)."
5. "Under a frozen patient-level protocol this dataset yields 0.7271 — inside the
   0.68–0.75 range that Fridman & Ben Shachar themselves identify as what
   properly evaluated methods achieve on CTU-UHB."

**Not supportable**

- that the paper is erroneous — the released implementation was never obtained,
  and §6 lists seven live degrees of freedom;
- that window-level evaluation inflates their number — it runs the other way
  ([literature_forensic_audit.md](literature_forensic_audit.md) §7.1);
- that this project solves a different or more prospective task — it does not.
