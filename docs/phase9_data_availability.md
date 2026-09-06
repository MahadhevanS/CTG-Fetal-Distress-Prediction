# Phase 9 gate — external CTG corpus availability

Checked 2026-09-04. This is the gating question for the whole SSL/foundation-model
route, because **masked-reconstruction SSL confined to CTU-UHB has already been
run in this project and was negative**
([model8_crossformer_run_history.md](model8_crossformer_run_history.md),
2026-08-17):

| | AUROC | vs random-init |
|---|---|---|
| Standalone, random-init | 0.7834, 0.7887 | — |
| Standalone, pretrained | 0.7945 ± 0.0443 | +0.006–0.011 (inside noise) |
| Model 8 wide-distress, random-init | 0.7995 ± 0.0498 | — |
| **Model 8 wide-distress, pretrained** | **0.7733 ± 0.0377** | **−0.0262 regression** |

The project's own diagnosis was corpus-limited, not method-limited:

> "Low effective data diversity — 30,985 windows came from only 468 patients at a
> 30 s stride against a 20-minute window (~97.5% overlap between adjacent
> windows); window count is large but actual information content is far less than
> that number suggests."

Fridman & Ben Shachar's advantage is therefore **the corpus** (984 recordings,
2,444 hours) rather than the method. Repeating SSL on CTU-UHB alone would repeat a
known failure with a known cause.

---

## 1. What is actually obtainable

| source | recordings | est. hours | outcome labels | access | status |
|---|---:|---:|---|---|---|
| **CTU-UHB** | 552 | ~660 | umbilical artery pH | PhysioNet, open | **held** |
| **FHRMA** | 135 | ~170 | none — expert-consensus baseline / acceleration / deceleration annotations | CTGDL v5, Zenodo, open (CC-BY-4.0; FHRMA itself GPL-3.0+) | **available now** |
| **SPaM'17** | 297 | ~1,610 | pH (was the challenge target) | Data Use Agreement; the Oxford page states the dataset "**has now been removed**" and access is case-by-case via the organiser, "for its original purpose — to validate existing algorithms" | **BLOCKED** |
| OxMat | 177,211 (≈94 % **antepartum**) | — | acidaemia, stillbirth | not stated publicly; PRISM-CTG describes it as private | institutional collaboration only |
| PRISM-CTG weights | — | — | — | MIT licence, but weights **not released** ("will be added soon"); pretrained on private OXMAT + SPaM; **no CTU-UHB AUROC or evaluation protocol disclosed** | **not usable** |

Sources: [CTGDL v5 (Zenodo 19510407)](https://zenodo.org/records/19510407),
[CTGDL DOI 10.5281/zenodo.17903226](https://zenodo.org/records/17903226),
[CTG Challenge 2017 / SPaM in Labour](https://users.ox.ac.uk/~ndog0178/CTGchallenge2017.htm),
[FHRMA dataset + toolbox](https://github.com/utsb-fmm/FHRMA),
[OxMat (arXiv 2404.08024)](https://arxiv.org/abs/2404.08024),
[PRISM-CTG](https://github.com/sfwon17/Prism-CTG).

### 1.1 CTGDL is open, but does not contain SPaM

CTGDL v5 (22 Dec 2025, 107.9 MB) ships `CTGDL_ctu_uhb_*`, `CTGDL_FHRMA_ano_csv`
and `CTGDL_FHRMA_proc_csv` — but for SPaM only `CTGDL_SPAM_metadata.csv`
(34 kB). The record states SPaM "cannot be uploaded or redistributed" and must be
obtained independently.

---

## 2. The decisive arithmetic

| | recordings | hours | share of Fridman's corpus |
|---|---:|---:|---:|
| Fridman & Ben Shachar | 984 | 2 444 | 100 % |
| **obtainable today (CTU-UHB + FHRMA)** | **687** | **~830** | **~34 %** |
| blocked (SPaM) | 297 | ~1 610 | ~66 % |

> **The single largest component of the corpus — two thirds of the pre-training
> hours — is the one that is no longer publicly available.** It is also the source
> Fridman used for external positive-class augmentation (CTGDL_SPAM caesarean
> cases labelled positive by stage-2 duration of zero).

Their exact setup therefore **cannot be reproduced** from public data as of this
date.

---

## 3. Honest prior on "SSL with FHRMA added"

Corpus grows 552 → 687 recordings (+24 %); pre-training patient count grows
~468 → ~600 (+28 %).

The August experiment regressed by −0.0262 with a diagnosis of insufficient
corpus diversity. A 24 % increase is a real change in the right direction but is
**unlikely on its own to flip a negative result of that size**, and it is far
short of the 4.4× more hours that Fridman had.

It remains worth running, for two reasons: it is cheap, and it would be the first
time this project uses information from outside CTU-UHB at all — which is the only
lever Phase 8 did not close.

---

## 4. The more interesting thing FHRMA offers

FHRMA's value here may not be its raw signal. It carries **expert-consensus
annotations of baseline, accelerations and decelerations** on 155 recordings
(66 shared-consensus training, 90 independent evaluation).

Every descriptor in this project's 0.7271 baseline is produced by *this repo's own*
detectors — `calculate_iterative_baseline`, `detect_accelerations`,
`detect_decelerations` — which have never been validated against expert ground
truth, because none was available. FHRMA is exactly that ground truth.

Two distinct uses, and the second is arguably the stronger:

1. **SSL pre-training corpus** — +24 %, prior above.
2. **Validating and correcting the descriptor extractors themselves.** If the
   baseline or deceleration detectors are systematically off against expert
   consensus, that error propagates into all 19 descriptors and therefore into
   every result in Phases 5–8. This is measurable, cheap, and has never been done.

Use 2 does not require SSL, a new architecture, or the blocked dataset.

---

## 5. Recommendation

| # | action | cost | gate |
|---|---|---|---|
| 1 | Download CTGDL v5; validate this repo's baseline/accel/decel detectors against FHRMA expert consensus | ~1 day | none — do it |
| 2 | Request SPaM access from the organiser, stating the actual purpose (pre-training, not algorithm validation) honestly | 1 email | may be declined; the stated purpose of the DUA is narrower than our use |
| 3 | Re-run masked SSL with CTU-UHB + FHRMA, **changing only the corpus** so the August run is the control arm | ~1 day GPU | Gate 7 ablation is built in by construction |
| 4 | OxMat / institutional collaboration | months | only if 1–3 are insufficient and the project has an appetite for it |

Action 1 is the highest value per unit effort and is independent of everything
else: it tests whether the measurement instrument underneath every Phase 5–8
result is correct.

**Do not** invest in PRISM-CTG until weights are released and CTU-UHB numbers with
a stated evaluation protocol are published; at present neither exists.
