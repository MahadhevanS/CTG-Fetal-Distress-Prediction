# Phase 10 / Experiment 2C — descriptor trajectory, and the project's position

Completed 2026-09-04.
Code: [`scripts/exp2c_descriptor_trajectory.py`](../scripts/exp2c_descriptor_trajectory.py).

---

## 1. Result — KILL

**Question:** does the *evolution* of the 11 FHR-only clinical descriptors over
the final hour carry information beyond their patient-level summary?

| model (patient-level, frozen folds) | AUROC | 95 % CI | AUPRC |
|---|---:|---|---:|
| window-LR + max aggregation, 11 FHR descriptors (**anchor**) | **0.7300** | [0.673–0.781] | 0.4160 |
| STATIC only — mean / sd / min / max, ordering never used | 0.7172 | [0.655–0.772] | 0.4336 |
| TEMPORAL only — final / trend / final−initial / late−early | 0.7154 | [0.657–0.771] | **0.4613** |
| **STATIC + TEMPORAL** | **0.7118** | [0.651–0.769] | 0.4334 |

| contrast | value |
|---|---:|
| patient-level reformulation, no temporal content, vs anchor | −0.0128 |
| **TEMPORAL CONTRIBUTION** (static+temporal − static) | **−0.0054** |

**Gate verdict: KILL the temporal-descriptor route** (0.7118 < 0.73).

The control mattered. Comparing 0.7118 against the 0.7300 anchor would have
confounded "temporal information" with "patient-level reformulation". Splitting
the summaries shows the reformulation costs −0.0128 and the temporal content
adds **−0.0054** — nothing.

Temporal-only (0.7154) ≈ static-only (0.7172): the temporal summaries are
**redundant with**, not additive to, the static ones. Combining all 88 features
against 110 positives is slightly worse than either half.

### One observation, flagged as post-hoc

Temporal-only reaches **AUPRC 0.4613**, the highest in the project (anchor
0.4160). AUROC is the pre-registered primary metric and it did not pass, so this
does **not** reopen the gate. Noting it because it was noticed after the primary
metric failed is exactly the kind of thing that becomes a false lead if left
unlabelled.

### Which temporal features carry anything univariately

| feature | patient AUROC |
|---|---:|
| `accels_final` | 0.6647 |
| `accels_trend` | 0.6493 |
| `accels_late_minus_early` | 0.6422 |
| `baseline_trend` | 0.6374 |

Acceleration count and baseline *do* move informatively — but that information is
already inside the static summaries, which is why adding it gains nothing.

---

## 2. The full ledger

Every serious representation tried, patient-level AUROC on the identical frozen
folds. **Nothing has ever exceeded 0.7300.**

| representation | AUROC | phase |
|---|---:|---|
| **11 FHR-only clinical descriptors** | **0.7300** | 10 |
| 19 clinical descriptors | 0.7271 | frozen baseline |
| + missingness pattern | 0.7297 | 8 |
| + stage-II flag + elapsed time | 0.7276 | 8 |
| + contraction-response trajectory (best variant) | 0.7204 | 8 |
| descriptor trajectory, static only | 0.7172 | 10 |
| **descriptor trajectory, static + temporal** | **0.7118** | **10** |
| full recording (crop removed) | 0.7132 | 8 |
| + HRV / spectral / nonlinear | 0.7119 | 8 |
| + maternal & clinical covariates | 0.7096 | 8 |
| CrossFormer + SSL (CTU + FHRMA) | 0.6494 | 9B |
| CrossFormer, random init | 0.6489 | 9B |
| baseline estimator "repaired" toward expert | 0.6300 | 9A |
| CrossFormer + SSL (CTU only) | 0.6202 | 9B |
| **raw FHR CNN @ 4 Hz** | **0.6117** | **10** |
| raw FHR CNN @ 1 Hz | 0.5865 | 10 |

Target: **0.85**. Best: **0.7300**. Gap: **0.12**. Detectable effect size in this
cohort: **0.0642**. The largest positive effect ever measured: **+0.0029**.

---

## 3. What the pattern says

Three independent results now point the same way, and none is about model
capacity.

1. **Phase 9A.** Making the event detectors agree better with expert consensus
   (acceleration F1 0.297 → 0.464) *reduced* AUROC by 0.0972. Physiological
   fidelity and endpoint prediction are not the same objective.
2. **Phase 9B.** The SSL arm with by far the best reconstruction loss (0.086 vs
   0.253) had middling downstream performance. Objective quality on the
   pretext task does not transfer.
3. **Phase 10 E1.** Minimally processed raw FHR scores 0.6117 against 0.7300 for
   11 hand-computed descriptors *derived from the same signal*. The descriptors
   are not merely a convenient summary — they are a compression a 252k-parameter
   CNN could not rediscover from 8 477 windows.

Taken with 2C — where the temporal evolution of those same proven quantities adds
−0.0054 — the consistent finding is that **~0.73 is what the CTG→acid-base
channel yields in this cohort**, and that richer representations of the same
signal do not recover more.

Phase 8 reached this conclusion from eight information sources. Phases 9 and 10
have added four more tests, from three genuinely different angles (external
expert labels, external corpus, raw waveform), and all agree.

---

## 4. What this does and does not establish

**Supportable**

- Twelve distinct representations, tested on one frozen protocol with a stated
  MDE, span 0.5865–0.7300, and the best is a linear model on 11 hand-computed
  FHR descriptors.
- The limit is not architecture: a CrossFormer, a residual CNN, SSL-pretrained
  encoders and logistic regression all land below or at the descriptor baseline.
- The limit is not feature poverty: HRV, spectral, nonlinear, contraction-level,
  maternal, missingness and temporal-trajectory features have each been added
  and each contributed within ±0.02.
- It is not the label: seven acid-base outcome definitions land within ±0.005
  (Phase 8 §5).

**Not supportable**

- That 0.85 is impossible on CTU-UHB. Phase 8 §6 showed the cohort is large
  enough to *demonstrate* 0.85 (CI half-width 0.047). Nothing here proves no
  representation exists — only that twelve did not find one.
- That the descriptors are optimal. Phase 9A showed they disagree substantially
  with expert consensus while still predicting better than the alternatives.
- That the aggregation question is closed. Experiment 1 deliberately froze
  aggregation; 2C tested it only through hand-built summaries, not a learned
  aggregator.

---

## 5. The position, stated plainly

The project owner's own criterion was: *"If that fails too, we should make a much
bigger strategic pivot rather than continuing the current model-development
loop."*

It failed. **There is currently no credible route to 0.85 on CTU-UHB**, and the
evidence increasingly indicates this is an information-source problem rather than
a machine-learning problem.

Three honest options, in the order I would rank them.

### A. Reframe the deliverable around what has actually been established

The strongest asset this project now holds is not a model — it is a rigorously
bounded negative result plus three methodological findings that are individually
publishable:

- the record-geometry decoding that explains the field's "late labour" confound
  (Phase 8 §3);
- the demonstration that expert-agreement and endpoint prediction are opposed
  objectives (Phase 9A);
- a documented failed reproduction of a published 0.822, with the gap
  decomposed and the unexplained residual stated (Dang reproduction).

Combined with the fact that Fridman & Ben Shachar themselves place properly
evaluated CTU-UHB performance at **0.68–0.75** — a range containing 0.7300 —
this is a defensible contribution that does not require 0.85.

### B. Change the information source

SPaM'17 access (request drafted, unsent) or an institutional route to OxMat.
This is the only lever with a mechanism behind it: Fridman's advantage was
1.8× the recordings plus external positives, not a better protocol. Timeline is
outside the project's control.

### C. Change the endpoint

Every acid-base definition saturates at ~0.73, but Apgar5 < 7 collapsed to 0.54 —
these are genuinely different constructs. A composite decision-relevant outcome
(e.g. operative delivery for fetal distress *with* acidaemia) has never been
tested and is not the same question as "which pH threshold".

**Not recommended:** another architecture, another feature set, another auxiliary
loss, or a larger version of anything already tried. Twelve representations is
enough evidence that the next one will also land near 0.73.
