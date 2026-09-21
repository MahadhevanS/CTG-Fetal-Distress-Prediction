# Foundation-model transfer — contamination and reproducibility audit

Written 2026-09-04. **No weights were downloaded. No GPU inference was run.**
This audit exists to decide whether that is permissible, and it concludes it
is not.

---

## 9. DECISION: **NO-GO** *(stated first; evidence below)*

Three independent hard stopping rules fire. The first is fatal on its own and
is not repairable by any amount of care:

1. **Every CTU-UHB recording was used in pre-training, so no clean subset
   exists.** Not "cannot be resolved" — resolved, and the answer is total
   exposure.
2. The reported 0.853 is a **restricted cohort of 46 recordings** selected on
   delivery mode and fetal presentation, and the full-cohort 0.83 rests on
   **55 recordings with 12 positives**, whose confidence interval **contains
   our 0.7271**.
3. The repository carrying the weights has **no licence**.

The correct conclusion, in the brief's own words:

> The published foundation model cannot be independently evaluated under this
> project's leakage-free patient-level protocol, because every one of our
> evaluation patients was present in its pre-training corpus.

---

## 1. Primary-source evidence

| item | source |
|---|---|
| paper | [arXiv:2601.06149](https://arxiv.org/abs/2601.06149), full text via [HTML v1](https://arxiv.org/html/2601.06149v1) |
| dataset | [Zenodo 10.5281/zenodo.18034361](https://zenodo.org/doi/10.5281/zenodo.18034361), 103.2 MB |
| code + weights | [github.com/naomifridman/CTGDL](https://github.com/naomifridman/CTGDL) |
| newer dataset record | [Zenodo 19510407](https://zenodo.org/records/19510407) |

The paper's in-text code link is an **unfilled placeholder**
(`https://github.com/[repository]`). The real repository was located by
search, not from the paper. Recorded because it matters for reproducibility
claims: the abstract states "We release standardized dataset splits and model
weights", and the paper as published does not say where.

---

## 2. Exact pre-training corpus

The 2,444 hours are the CTGDL collection, three databases:

| database | recordings | source |
|---|---:|---|
| **CTGDL_CTU_UHB** | **552** | Czech Technical University / University Hospital Brno |
| CTGDL_FHRMA | 135 | Lille Catholic Hospital, France |
| CTGDL_SPAM | 297 | CTG Challenge 2017 (Oxford, Lyon, Brno) |

Per-database hour counts are not broken out; only the 2,444 h total is given.

---

## 3. CTU-UHB exposure status — **CONFIRMED, COMPLETE**

The paper states, of the combined CTGDL collection:

> **"The complete dataset was used for self-supervised pre-training via
> masked prediction."**

CTGDL_CTU_UHB is one of its three components. Therefore **all 552 CTU-UHB
recordings entered pre-training**, including every recording later used for
downstream fine-tuning and testing.

There is no manifest question left to resolve. "The complete dataset"
answers it.

---

## 4. Recording-level contamination status

Our cohort is 547 of those 552 recordings (5 dropped by our own quality
gate). Classifying every patient as the brief requires:

| class | patients | share |
|---|---:|---:|
| definitely **unseen** during pre-training | **0** | 0% |
| definitely **seen** during pre-training | **547** | **100%** |
| unknown | 0 | 0% |

**This is not Case B and not Case C — it is worse than both.** Case B assumes
a clean subset can be constructed from recording IDs; there is no clean
subset, because there are no unseen CTU-UHB patients. Case C assumes exposure
is unresolvable; here it is fully resolved and complete.

### On "it was only unlabeled exposure"

The brief pre-empts this and is right to. The exposure is masked
reconstruction, not label leakage — the model never saw pH during
pre-training, and that is a genuinely weaker form of contamination than
outcome leakage. But the encoder's weights were optimised to reconstruct the
exact signals of our held-out patients. A representation fitted to our test
patients' waveforms is not independent of them, and any AUROC we measured
would be partly a measure of how well it memorised signals it had already
seen.

Stating the limit of that argument honestly: this does not prove the
transferred score would be *inflated*, and self-supervised pre-training on
target-domain data is common practice in many fields. What it does mean is
that the resulting number could not be presented as a **leakage-free
patient-level result**, which is the only kind this project's ≥0.85 claim
would accept. That distinction is the whole point of the audit.

---

## 5. Audit of the reported 0.83 / 0.853

| reported | n | positives | AUROC | SE | 95% CI (Hanley–McNeil) |
|---|---:|---:|---:|---:|---|
| full test set | **55** | **12** | 0.830 | 0.0769 | **[0.679, 0.981]** |
| "Vaginal delivery" | 50 | ~10 | 0.850 | 0.0800 | [0.693, 1.000] |
| + excluding non-cephalic | **46** | ~9 | **0.853** | 0.0837 | **[0.689, 1.000]** |
| **our frozen benchmark** | **547** | **110** | **0.7271** | — | **[0.670, 0.779]** |

*(positive counts for the two subgroups inferred at the stated 20.0%
prevalence; the paper gives 12/55 for the full test set explicitly.)*

Four findings:

**5.1 The intervals contain our baseline.** Both the 0.83 and the 0.853 CIs
include **0.7271**. On their own test-set size, these results are not
statistically distinguishable from the benchmark they are reported as
exceeding. Their own comparison range for prior work — "0.68–0.75" — brackets
our 0.7271 too.

**5.2 The test set has 12 positives.** One of our five folds carries roughly
22. Their entire test set is about half the positive count of one of our
folds.

**5.3 The 0.853 cohort is selected on delivery mode and presentation.**
"Vaginal delivery" (50) then excluding "non-cephalic presentation" (46).
Delivery mode is not knowable while monitoring, and this project's own rules
forbid using it as a predictor. Selecting the *evaluation cohort* on it
admits the same information through a different door — it conditions the
reported performance on how the labour ended.

**5.4 The split is, in fairness, patient-level.** "Stratified sampling to
maintain consistent class distribution" is recording-level, and in CTU-UHB
one recording is one delivery is one patient. So recording-level splitting
*is* patient-level here. This is **not** a defect and should not be listed as
one.

**Unresolved:** whether checkpoint or hyperparameter selection used the test
set. Not established from the sources retrieved; recorded as unknown rather
than assumed either way.

---

## 6. Weight / code / licence availability

| check | status |
|---|---|
| weights actually downloadable | **YES** — `trained_model/` in the GitHub repo holds a pre-trained PatchTST classification model |
| code available | **YES** — `pred_with_trained_PatchTST_and_plot.ipynb`, Colab-runnable |
| processed signals provided | YES — `ctgdl_proc_samples/`, and the Zenodo CSVs |
| Zenodo contains weights | **NO** — data and metadata only (CTU-UHB raw/processed CSV, FHRMA, SPaM metadata + scripts) |
| which recordings trained the released model | **NOT DOCUMENTED** in the repo |
| **licence on code/weights** | **NONE FOUND** — no LICENSE file |
| data licences | CTU-UHB ODC-By-1.0; FHRMA GPL-3.0; SPaM under DUA, must be obtained separately; record overall CC-BY-4.0 |

**A correction to my earlier reading.** From the paper alone I recorded
weights as unavailable, because the in-paper link is a placeholder. That was
wrong: the weights exist and are downloadable from the repository. The
availability blocker is not existence but **licence** — absent a LICENSE
file, default copyright applies and no use rights are granted. That is a
factual blocker for a research claim, whatever common practice may be.

---

## 7. Reproducibility assessment

Against the brief's seven pre-conditions:

| # | condition | status |
|---|---|---|
| 1 | weights downloadable | ✅ |
| 2 | code available | ✅ |
| 3 | preprocessing implementable | likely — CTGDL processed CSVs are published |
| 4 | accepts our CTU-UHB signal format | likely — same source data, 4 Hz FHR+UC |
| 5 | inference runnable per window | likely — PatchTST is patch-based |
| 6 | no proprietary preprocessing | ✅ |
| 7 | **licence permits research use** | ❌ **no licence** |

Six of seven pass. The seventh fails, and condition 4 of Phase 2
(contamination) fails before any of this matters.

---

## 8. Clean-evaluation feasibility

**None.** A clean evaluation would require CTU-UHB patients absent from
pre-training. There are none. The options and why each fails:

* *Evaluate on our folds anyway* — every test patient was seen. Not
  leakage-free.
* *Construct a clean subset* — requires unseen patients. Zero exist.
* *Evaluate on a different corpus* — FHRMA and SPaM were also in
  pre-training, and neither carries pH.
* *Use only the encoder, frozen* — freezing does not undo the fact that the
  encoder's parameters were fitted to our test patients' signals.

The one configuration that would be clean is a model pre-trained on a corpus
provably excluding CTU-UHB. No such released model was found.

---

## 10. Experiment specification — not applicable

The GO condition was not met, so no experiment is specified. Per the brief's
final rule, no GPU inference is run.

---

## What this closes, and what it does not

**Closes:** the external-transfer route *via this released model*. It was the
last route on the Phase-10 board with an unbounded prior, and it is blocked
for a reason that is structural rather than empirical — the only public CTG
foundation model was pre-trained on the entire benchmark it is evaluated on.

**Does not close:** external transfer as a concept. A model pre-trained on a
corpus excluding CTU-UHB would be evaluable cleanly. [PRISM-CTG
(arXiv:2605.02917)](https://arxiv.org/pdf/2605.02917) has not been assessed
and is the obvious next candidate for the *same* audit — with the same first
question, asked before anything is downloaded: **which recordings were in
pre-training?**

**Does not establish** that the foundation model has no useful signal. It may
well have. What is established is that we cannot measure it here without
compromising exactly the property — leakage-free patient-level evaluation —
that would make a ≥0.85 claim worth making.

### One observation worth carrying forward

The strongest published CTU-UHB result available, at 0.83 on 55 recordings
with 12 positives, has a 95% interval of **[0.679, 0.981]** — which contains
our 0.7271. Combined with
[literature_forensic_audit.md](../literature_forensic_audit.md)'s
reconciliation of published 0.822–0.83 results with honest patient-level
0.73–0.78, the external literature does not currently provide evidence that
≥0.85 has been achieved on this benchmark under an evaluation as strict as
ours. It provides evidence that it has been *reported*.
