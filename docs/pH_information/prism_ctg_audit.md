# PRISM-CTG — contamination and availability audit

Written 2026-09-04. **No GPU. No inference. No downloads. No fine-tuning.**
Primary sources only.

---

## 1. Executive verdict: **NO-GO**

Two Phase-7 conditions fail:

| # | condition | status |
|---|---|---|
| 1 | CTU-UHB contamination status **resolved** | ❌ **partially unresolved** — direct inclusion is cleanly excluded, but indirect exposure via SPaM cannot be established |
| 3 | exact pretrained checkpoint identified | ❌ **weights are not released** — *"The link to the model's weights will be added soon!"* |

**But this is a different kind of NO-GO from Fridman, and the difference
matters.** Fridman is blocked *structurally* — the complete CTU-UHB corpus
was in pre-training and no clean subset can ever exist. PRISM is blocked
*contingently*: CTU-UHB was **explicitly excluded from pre-training**, and the
two blockers are an unreleased file and an undocumented third-party dataset.
Both could be resolved by the authors without any change to the science.

**PRISM-CTG is the first external CTG representation this project has found
that is not disqualified by design.** It should be re-audited if weights
appear.

---

## 2. Pre-training datasets

| corpus | size | domain | access |
|---|---:|---|---|
| **OXMAT** | 51,336 pregnancies, ~753,352 segments, **>250,000 hours** | antepartum, 5 Oxfordshire institutions, 1990–2024 | **private** — "unavailable due to privacy and ethical reasons" |
| **SPaM** | 300 intrapartum recordings | intrapartum, CTG Challenge 2017 | withdrawn; case-by-case via organiser |

For scale: >250,000 hours against Fridman's 2,444 and our own ~830.

**Objectives** — multi-view SSL with three pretext tasks:
1. random-projected guided masked signal reconstruction;
2. **clinical variable prediction** — targets are *gestational age, time to
   birth, maternal age*;
3. feature classification over 17 handcrafted features (incl. baseline FHR,
   short-term variability) as patch-level targets.

**Objective 2 was checked specifically for outcome leakage.** Umbilical pH is
**not** among the pre-training targets. That matters: a metadata pretext task
*could* have smuggled the downstream label into pre-training, and it did not.
(*"Time to birth"* is outcome-adjacent, but it is an OXMAT antepartum target,
not a CTU-UHB one.)

---

## 3. CTU-UHB contamination

### 3.1 Direct inclusion: **NO** — with primary-source evidence

CTU-UHB appears only as **downstream task T5** and as an **external
validation** set. The paper states intrapartum recordings are **"less than 1%
of the total pre-training data"**, and names that 1% as SPaM.

### 3.2 Indirect exposure via SPaM: **UNRESOLVED** — and this is the real question

SPaM **is** in pre-training. Per this project's own
[phase9_data_availability.md](../phase9_data_availability.md), SPaM'17 is
~297 recordings from **Oxford, Lyon and Brno**.

**CTU-UHB is University Hospital Brno.** So a pre-training corpus containing
Brno intrapartum recordings, evaluated on a Brno intrapartum benchmark,
raises a specific and non-hypothetical overlap question.

Attempts to resolve it:

| source | result |
|---|---|
| PRISM paper | silent on SPaM composition or CTU-UHB overlap |
| [CTG Challenge 2017 page](https://users.ox.ac.uk/~ndog0178/CTGchallenge2017.htm) | no composition details; **"The dataset has now been removed"** |
| CTGDL v5 | ships only `CTGDL_SPAM_metadata.csv` (34 kB); SPaM itself "cannot be uploaded or redistributed" |

**It cannot be resolved from public sources.** Per the brief — *"Do NOT infer
'clean' merely because CTU-UHB is not prominently mentioned"* — this is
**CASE C on the indirect axis**, even though it is Case A on the direct one.

### 3.3 Our 547-patient cohort

| class | patients |
|---|---:|
| definitely unseen (no direct pre-training exposure) | 547 |
| exposure via SPaM overlap | **unknown — 0 to unknown** |

Unlike Fridman (547/547 definitely seen), no patient here is *known* to be
exposed. But "not known to be exposed" is not "known to be unexposed", and
the brief is explicit that the burden runs the other way.

**Resolvable by one question to the authors:** *were any SPaM recordings
sourced from the same University Hospital Brno archive as CTU-UHB, and can a
recording-level manifest be provided?* That is the single question that would
move PRISM from Case C to Case A.

---

## 4. Weights and code

| check | status |
|---|---|
| **weights downloadable** | ❌ **NO** — *"The link to the model's weights will be added soon!"* |
| exact checkpoint identified | ❌ none exists publicly |
| pretraining code | ✅ `run_pretraining.py` |
| downstream/eval code | ✅ `run_linear_probe.py` |
| example data | ✅ `Example_data/` |
| input format documented | ✅ **1 Hz, 20-minute chunks (1200 samples), FHR + TOCO**, plus metadata (gestational age, maternal age, time to birth) |
| self-pretraining as a fallback | ❌ impossible — OXMAT is private |

Both routes are closed: the checkpoint is unreleased, and the corpus needed
to reproduce it is unavailable.

---

## 5. Licensing

| artefact | licence |
|---|---|
| code | **MIT** — permissive, research use explicitly fine |
| weights | n/a (unreleased); presumably MIT if released under the repo |
| OXMAT | private, not redistributable |
| SPaM | DUA, withdrawn |
| CTU-UHB (ours) | ODC-By-1.0 |

Licensing is **not** a blocker here — a genuine improvement over Fridman,
whose weights repository carries no licence at all.

---

## 6. Reproducibility

If weights appeared, the transformation to our data would be:

| | ours | PRISM requires | transformation |
|---|---|---|---|
| sampling rate | 4 Hz | **1 Hz** | decimate 4→1 Hz |
| window | 20 min (4800) | 20 min (**1200**) | same span |
| channels | FHR + UC | FHR + TOCO | direct |
| metadata | — | GA, maternal age, **time to birth** | ⚠️ see below |

The rate change is a fixed, outcome-independent decimation and would **not**
require altering our evaluation protocol — window geometry, folds,
aggregation and bootstrap all stay put.

**One flag requiring resolution before any use:** the repo lists *time to
birth* among required inputs. If that is needed at **inference** rather than
only as a pre-training target, it is a future variable, unavailable
prospectively, and this project forbids it. From the paper it reads as a
pre-training target only, but the repo's phrasing is ambiguous and it must be
settled, not assumed.

---

## 7. Published CTU-UHB results

| item | T5 (in-dataset) | external validation |
|---|---|---|
| dataset | CTU-UHB, 552 patients | train APHP-CTG (450) → test CTU-UHB |
| **AUROC** | **0.883** | **0.806** |
| method | **linear probing** on frozen representation | cross-dataset transfer |
| split | 80/20, **no patient overlap**, averaged over 5 runs | full CTU-UHB as test |
| positives | **not reported** | not reported |
| pH threshold | **not reported** | not reported |
| aggregation (window vs patient) | **not stated** | not stated |
| CIs | **none** | none |
| checkpoint selection | **not described** | not described |

**Credit where due:** the split is explicitly patient-disjoint and results
are averaged over 5 runs. That is better practice than several published
CTU-UHB results this project has audited.

**But four things are undisclosed and each is material:** positive count, pH
threshold, whether AUROC is computed per window or per patient, and how the
checkpoint was selected. Window-level AUROC on overlapping windows is the
single most common inflation mechanism this project has documented, and it
cannot be ruled out from what is published.

Implied uncertainty, which the paper does not give (Hanley–McNeil, 80/20 of
552 → ~110 test patients, ~22 positive at 20.1% prevalence):

| result | implied 95% CI |
|---|---|
| T5 0.883 | **[0.788, 0.978]** |
| external 0.806 (at n=110) | [0.690, 0.922] |
| external 0.806 (if full 552/110 positives) | **[0.754, 0.858]** |

*(Supersedes [phase9_data_availability.md](../phase9_data_availability.md),
which recorded "no CTU-UHB AUROC or evaluation protocol disclosed" — the full
text does disclose both.)*

---

## 8. Comparison with our benchmark

Not a superiority claim — the comparison is not currently possible.

| | AUROC | n | positives | CI | protocol |
|---|---:|---:|---:|---|---|
| **our clinical LR** | **0.7271** | 547 | 110 | [0.670, 0.779] | patient-grouped 5-fold, patient-level max, patient bootstrap |
| PRISM T5 | 0.883 | ~110 test | unreported | [0.788, 0.978]* | 80/20 ×5, aggregation unstated |
| PRISM external | 0.806 | 552 | unreported | [0.754, 0.858]* | cross-dataset |

*computed by us; not reported.

The T5 interval excludes our 0.7271 and the external interval (on the full
cohort) does too. **That is genuinely interesting and is the strongest
external signal this project has encountered.** It is also exactly the point
at which to recall that
[literature_forensic_audit.md](../literature_forensic_audit.md) reconciled
published CTU-UHB results of 0.822–0.83 into honest patient-level 0.73–0.78
via three measurable choices — one of which (window- vs patient-level
evaluation) is *undisclosed here*. The number cannot be taken at face value,
and equally cannot be dismissed.

---

## 9. Clean-experiment feasibility

**Not currently feasible** — no checkpoint exists to evaluate.

If both blockers cleared, the experiment is already specified and is small:

> **Arm A** clinical LR (frozen, 0.7271) · **Arm B** PRISM frozen encoder →
> logistic regression.
> Identical patient folds, identical seeds, patient-level max aggregation,
> patient-bootstrap CI. 4→1 Hz decimation, documented. No fine-tuning, no
> sweeps, no KI, no ensembling, no threshold tuning. One configuration.
> Report per-fold AUROC, pooled OOF AUROC, AUPRC, paired Δ vs 0.7271 with
> bootstrap CI, sign consistency.
> **Gate: paired Δ ≥ +0.03 with CI excluding zero** before any fine-tuning is
> considered. Project target remains **≥0.85**.

---

## 10. Final recommendation

> **NO GPU. NO INFERENCE. STOP.**

Both blockers are outside our control and neither is a scientific defect in
PRISM's design. The proportionate action is **not** to search for a
workaround — the brief forbids that, correctly — but to record two precise,
answerable questions for the authors:

1. **Are the released weights available, and which checkpoint corresponds to
   the published T5 result?**
2. **Does SPaM contain any recordings from the same University Hospital Brno
   archive as CTU-UHB, and is a recording-level manifest available?**

Until (2) is answered, PRISM is Case C and no CTU-UHB evaluation may be run
however attractive 0.883 looks. Until (1) is answered, there is nothing to
run at all.

### Standing position after Phases 10–11

Every identified route to ≥0.85 is now either closed on evidence or blocked
on external availability:

| route | status |
|---|---|
| all eleven internal information sources | closed on evidence, best +0.0027 |
| Fridman foundation model | **structurally** contaminated — 547/547 exposed |
| **PRISM-CTG** | **contingently blocked** — weights unreleased, SPaM overlap unresolved |
| larger labelled corpus (OxMat collaboration) | unavailable |

This is not the same as a demonstrated information ceiling, and this report
does not claim one. It is that the project has exhausted what it can test
alone, and the remaining upside now depends on artefacts other groups have
not yet released.
