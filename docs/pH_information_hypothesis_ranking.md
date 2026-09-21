# Route to ≥0.85 — hypothesis ranking and next-experiment definition

Written 2026-09-04. **No model was trained for this report.** It is the
Phase-10 deliverable: answer Q1–Q5, rank the remaining hypotheses, recommend
exactly one experiment.

Frozen protocol unchanged: 547 patients, 110 positive, patient-grouped folds
in `data/processed_figo`/`data/processed_clinical`, patient-level **max**
aggregation, patient-bootstrap CIs, no selection on the reported fold.
Benchmark: **19-descriptor LR, AUROC 0.7271 [0.670–0.779]**. Target **≥0.85**,
not redefined.

---

## Headline: four of the five proposed routes are already closed

The brief lists beat-to-beat/fine temporal information, the pre-delivery
window, self-supervised representation learning, and external transfer as
"not yet adequately exploited". **Three of those four have been run on the
frozen protocol and are negative, and the fourth is physically impossible on
this corpus.** That evidence is in
[phase8_information_audit.md](phase8_information_audit.md) and
[phase9b_ssl_corpus.md](phase9b_ssl_corpus.md), and this report's main job is
to surface it before compute is spent re-running it.

---

## Q1 — What has already been tested?

Against 0.7271, frozen protocol, pre-registered bar **+0.0642** (the Phase-7
minimum detectable effect), not +0.02:

| information source | best Δ | verdict |
|---|---:|---|
| missingness pattern | **+0.0027** | closed |
| second-stage flag + elapsed time | +0.0005 | closed |
| **SSL (CTU+FHRMA) vs random init** | **+0.0005** | **closed** |
| per-contraction response trajectory | −0.0067 | closed |
| full-length recording (the discarded 19%) | −0.0113 | closed |
| trajectory × stage-split combined | −0.0118 | closed |
| retrospective covariates *(leaky, reference only)* | −0.0142 | closed |
| **HRV / spectral / nonlinear (18 features)** | **−0.0152** | **closed** |
| maternal + clinical covariates | −0.0175 | closed |
| SSL on CTU-UHB only | **−0.0287** | closed, replicated twice |
| external-only SSL (FHRMA, leakage-clean) | −0.0077 | closed |

**Range: −0.0287 to +0.0027 against a +0.0642 bar.** Eleven sources, best is
+0.0027.

### Four closures that specifically answer the brief's proposals

**A. Beat-to-beat / fine temporal information does not exist in this corpus.**
Measured across all 552 records: minimum quantisation step **0.25 bpm** on
*every* record regardless of sensor, **98% of FHR power below 0.5 Hz**, and
CTU-UHB publishes a 4 Hz series, not RR intervals. True beat-to-beat detail
was removed *before the data reached this project*; no preprocessing can
recover it. The direct test agrees: 18 features the descriptors lack — VLF/
LF/HF, Poincaré SD1/SD2, sample entropy, DFA α1/α2, Dawes-Redman STV — score
**−0.0152** added, 0.6435 alone, and sample entropy and DFA do not reach the
top eight univariately.

A further physical detail worth stating: Doppler records have *higher*
sample-to-sample variance than scalp-electrode ones (2.84 vs 2.08 bpm). The
extra high-frequency content in the majority of the cohort is
**autocorrelation artefact, not physiology**.

**B. The pre-delivery window question is answered by the record geometry.**
`Pos. II.st. = record_length − II.stage × 240` holds exactly for 489/506
records. Therefore **every CTU-UHB record is 60 minutes of first stage
followed by the entire second stage.** Consequences:

* the discarded portion has median duration **11.4 minutes**, and only
  **144 of 547** patients have any window lying wholly before the final hour;
* it is the *earliest* first stage — furthest from the outcome;
* running it anyway gives **−0.0113**, and full-length max-aggregation is
  *optimistic* because bag size then encodes second-stage duration (a
  retrospective variable). Fixing the bag at 17 windows drops it to 0.7086.

So the answer to "is the chosen window discarding useful information" is: the
window is not the problem, and there is very little outside it to discard.

**C. Self-supervised representation learning has been run, with the right
control.** `ssl_ctu_fhrma − randinit = +0.0005`. The whole SSL enterprise is
worth nothing here. CTU-only SSL actively *hurts* (−0.0287, replicated across
two substrates and two protocols), and the leakage-clean external-only arm is
−0.0077. Reading the designed contrast (`+0.0293` for adding FHRMA) in
isolation would have been the error the `randinit` arm exists to prevent.

**D. Neither the target nor the sample size is the problem.** Seven outcome
definitions on the same features and folds: pH≤7.15 **0.7271**, pH≤7.05
0.7311, BDecf≥12 0.7294, BDecf≥8 0.7287, pH∧BDecf concordant 0.7285 — all
within 0.004. (Apgar5<7 collapses to 0.5423, confirming the features are
specific to acid-base status.) And Hanley–McNeil at this prevalence: **157
patients** suffice to detect a paired +0.12 at 80% power. We have 547. **The
missing ingredient is signal, not sample size or label definition.**

---

## Q2 — What has NOT been tested?

After eleven closures, the honest list is short.

| # | untested | why it is still open |
|---|---|---|
| **1** | **External pretrained CTG representation with released weights** | never attempted; the only lever that adds information from outside CTU-UHB at scale |
| 2 | Knowledge infusion applied to the **pH** task | KI was built and validated on FIGO, never run against pH |
| 3 | A substantially larger *labelled* corpus (OxMat / institutional) | requires months and a collaboration; nothing to run today |
| 4 | Nonlinear aggregator over window descriptors, without max | the caveat bounding every Phase-8 null: all used linear models + max aggregation |

Item 4 is weaker than it looks: Route 2 of Phase 8 (stage-aligned trajectory
*with* a learned aggregator) was run in combination and gave −0.0118, and the
temporal-feasibility study independently found every order-invariant
statistic beat every order-dependent one.

---

## Q3 — Which remaining source could plausibly give **≥+0.12**?

**On the evidence, none of items 2–4.** Stating the priors honestly:

* **Item 2 (KI on pH): +0.00 to +0.02, and it cannot exceed 0.727 by much in
  principle.** KI's mechanism is teaching a network to compute the clinical
  descriptors and then reason over them. Those descriptors *are* the 0.7271
  baseline. Teaching a network to reproduce a 0.727 representation cannot
  produce 0.85 — at best it approaches 0.727 from below. Its measured FIGO
  gain was +0.019, and its more interesting property (3.6× variance
  reduction) improves *stability*, not ceiling. **This is the most
  tempting-looking leftover and it is a dead end for the target.**
* **Item 4: ≤ +0.02**, per the combination result above.
* **Item 3: unknown, but unavailable today.**

**Item 1 is the only candidate whose prior is not already bounded below +0.12
by our own measurements**, because it is the only one that adds information
from outside this corpus at a scale we have never had: 2,444 hours against
our ~830.

---

## Q4 — Which can be tested without changing the frozen protocol?

Item 1 and item 2 both can: each produces a per-window or per-patient score
that goes through `protocol.py` unchanged. Item 3 cannot (different cohort).
Item 4 can.

---

## Q5 — The smallest falsifying experiment

For item 1, the smallest decisive step is **not compute — it is a
leakage/availability audit**, and it can falsify the hypothesis outright
before a GPU is touched.

---

## The external-transfer candidate, audited as far as public sources allow

[Fridman & Ben Shachar, *A Foundation Model Approach for Fetal Stress
Prediction During Labor from CTG recordings*, arXiv:2601.06149 (Jan
2026)](https://arxiv.org/abs/2601.06149). PatchTST with channel-asymmetric
masking, 2,444 hours unlabelled masked pre-training, fine-tuned on CTU-UHB.
Reports **AUROC 0.83 on the full test set and 0.853 on uncomplicated vaginal
deliveries**, against "previously reported results on this benchmark
(0.68–0.75)" — a range that brackets our 0.7271.

Against the brief's 10-point audit:

| # | question | status |
|---|---|---|
| 1 | exact dataset | 2,444 h; per [phase9_data_availability.md](phase9_data_availability.md) the corpus is 984 recordings of which **SPaM (297 recordings, ~1,610 h, 66% of the hours) is no longer publicly available** |
| 2 | exact CTU-UHB subset | **0.853 is on "uncomplicated vaginal deliveries" — a cohort restriction, not the full benchmark.** 0.83 is the full test set |
| 3 | split protocol | **UNRESOLVED from the abstract** |
| 4 | **was CTU-UHB in pretraining?** | **UNRESOLVED — and decisive.** Phase 9 recorded CTU-UHB as a component of that corpus; if so, our test patients were seen during pre-training |
| 5 | did downstream labels influence pretraining? | UNRESOLVED |
| 6 | preprocessing compatibility | unknown; PatchTST on 4 Hz FHR+UC is compatible in principle |
| 7 | **weights/code released?** | the abstract **states** "we release standardized dataset splits and model weights"; **no repository URL was locatable from the abstract page** |
| 8 | license | unknown |
| 9 | patient-level evaluation possible? | yes *if* weights load and inference runs per window |
| 10 | metric comparable to ours? | **No, as reported.** 0.853 is a restricted cohort; our protocol forbids cohort restriction, uses patient-level max aggregation and patient bootstrap |

**Two of these are potentially disqualifying and both are unresolved (#3,
#4).** If CTU-UHB was in the pre-training corpus, then evaluating those
weights on our CTU-UHB folds is **transductive contamination**: the encoder
saw our held-out patients' signal, without labels, but saw it. That is
exactly the "uncertain or unverifiable data leakage" of **stop condition F**.

There is a secondary paper, [PRISM-CTG (arXiv:2605.02917)](https://arxiv.org/pdf/2605.02917),
which post-dates this project's prior audits and has not been assessed.

---

## Ranking

| rank | hypothesis | expected Δ | feasibility | leakage risk | reproducibility | cost | P(≥0.85) |
|---|---|---:|---|---|---|---|---|
| **1** | **External pretrained representation, evaluated under our protocol** | unknown; the only unbounded prior | audit cheap; use depends on release | **HIGH — unresolved** | depends on weights | ~0 for audit, ~1 GPU-day if usable | low, but non-zero |
| 2 | Nonlinear patient-level aggregator | ≤ +0.02 | high | low | high | hours | ~0 |
| 3 | KI transferred to pH | +0.00 to +0.02 | high | low | high | ~1 GPU-day | **~0 — bounded by the 0.727 representation** |
| 4 | Larger labelled corpus (OxMat) | unknown | **not available** | low | n/a | months | unknown |

---

## RECOMMENDED NEXT EXPERIMENT — exactly one

> **A pre-compute availability-and-contamination audit of the released
> Fridman foundation model, with a pre-registered abort rule.**

**Hypothesis.** A CTG representation pre-trained on ~3× more data than we can
assemble carries acid-base information that no representation derived from
CTU-UHB alone has produced, and that information survives evaluation under
our frozen patient-level protocol.

**Step 1 (no compute, hours).** Obtain the full paper and the release. Resolve
exactly four things:

1. Was **CTU-UHB in the pre-training corpus**?
2. Was the downstream split **patient-level**?
3. What defines "uncomplicated vaginal deliveries", and what is the full-cohort
   patient-level number?
4. Are weights actually downloadable, under what license?

**Pre-registered abort rules — these are the falsification, and they fire
before any training:**

* If **CTU-UHB was in pre-training** → the representation cannot be evaluated
  on our folds without transductive contamination. **Stop condition F.**
  Report it, do not train. (A partial escape exists only if the release
  documents which recordings were used *and* excludes enough of ours to leave
  a clean held-out set — check, do not assume.)
* If **weights are not actually released** → nothing to evaluate. Stop.
* If the reported number depends on **cohort restriction** and no full-cohort
  patient-level figure exists → the headline is not comparable to ours, and
  the honest expectation drops toward the 0.73–0.78 band that
  [literature_forensic_audit.md](literature_forensic_audit.md) already
  reconciles published CTU-UHB results into.

**Step 2, only if all four clear (~1 GPU-day).** Extract per-window
embeddings with the encoder **frozen**, fit the simplest possible head
(logistic regression) through `protocol.py` unchanged, report patient-level
AUROC with patient-bootstrap CI, per-fold AUROC, AUPRC, and the paired delta
against 0.7271. No fine-tuning, no architecture variation, no threshold
tuning. One configuration.

**Success bar:** +0.0642 (the frozen MDE) with a paired CI excluding zero.
Anything less is a null by the existing standard, whatever the paper reports.

---

## What this report does *not* conclude

It does not claim a ~0.73 information ceiling. Per the brief, that is not
established. What the evidence supports is narrower and should be stated
exactly:

> Eleven information sources have been tested against 0.7271 on a frozen
> patient-level protocol; the best is +0.0027 and none approaches +0.0642.
> Beat-to-beat information is **physically absent** from the published
> corpus. The target definition is not the limiting factor (seven
> formulations, all 0.727–0.731). The sample size is not the limiting factor
> (157 patients would suffice for +0.12). Published CTU-UHB results of
> 0.822–0.83 are reconciled with honest patient-level 0.73–0.78 by three
> measurable methodological choices.

One consequence of that last line deserves stating plainly, because it
calibrates the target rather than lowering it: **a patient-level 0.85 on
CTU-UHB would exceed every published result on this dataset, on a protocol
stricter than any of them used.** That is not an argument for stopping. It is
what achieving the target would mean, and it is the reason the external-
transfer audit — the one route that adds genuinely new information — is the
only experiment worth running next.
