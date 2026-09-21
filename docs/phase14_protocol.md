# Phase 14 — Patient-Level Temporal Aggregation Study (Pre-Registration)

**Frozen 2026-09-11, before any Phase 14 script was written or run.**
Governed by the same discipline as `docs/phase13_protocol.md`: nothing below
changes without a dated amendment in Section 9, written before the change
takes effect.

> **Governance rule (unchanged from Phase 13).** Phase 14 is exploratory,
> post-lock work. Phase 12.1 remains untouched and authoritative. Nothing
> here can retroactively alter the locked P6 model or its reported metrics.
> This is a distinct study from Phase 13 — it does not inherit Phase 13's
> parity or Huber threads, which are closed.

---

## 0. What "independently pre-registered" can and cannot mean here

Stated plainly before anything else, because it governs how every result
below must be read: **there is no fresh cohort for this study.** CTU-UHB's
547 patients are the same cohort used in every phase back to Phase 1; the
P90-vs-single-window point estimates this study formalizes were already
observed once, during Phase 13's exploratory work (Option E). "Independent
pre-registration" therefore cannot mean independence from the *data* — that
would require external validation on a cohort this study does not have
access to, which is precisely the gap the project's own final synthesis
(`reports/final_project_synthesis_report.md`, Section 18) already named as
the highest-priority future work, unrelated to Phase 13/14.

What this pre-registration *does* buy, honestly: independence from
*analysis-choice flexibility*. Phase 13's Option E looked at four horizons
and reported all of them; this study commits, before computing anything, to
exactly one primary confirmatory test, one primary horizon, and a decision
rule fixed in advance — removing the "look at several things, discuss
whichever looks best" latitude that (even when handled carefully) Phase 13's
more exploratory process retained. A positive result here should be reported
as **internal replication under a stricter, pre-committed protocol**, never
as independent confirmation, and any promotion decision must say so
explicitly.

## 1. Scientific question

Does preserving the distribution of sequential, causal P6 window-level risk
scores — rather than reducing a patient's recording to one selected window —
improve patient-level discrimination for the primary endpoint?

## 2. Frozen inputs (no retraining, no new feature engineering)

- Window-level P6 probabilities: `results/phase13/audit/p6_predictions.npz`
  (`pred_unweighted_cv`, `pred_unweighted_test`) — the exact scores audited
  and bit-reproduced in Phase 13.0A.
- Cohort, folds, labels: `data/processed_clinical/folds.json` (547 patients,
  canonical 5-fold assignment), `data/processed_clinical/test_dataset.pt`
  (83-patient held-out test), `y_primary = 1[pH ≤ 7.15]`.
- Causal window-eligibility rule: `t_i ≥ h` (unchanged from Phase 13
  Section 4/13.2).

No new model is fit on raw signal or descriptor features anywhere in this
study.

## 3. Candidate aggregators — all deterministic, no hyperparameter selection

| Candidate | Definition |
|---|---|
| Single-window (control) | Current production convention: `get_patient_scores_at_horizon_corrected` |
| Plain P90 | `q=0.90`, fixed by Phase 3's historical convention, uniform weights |
| Max | Maximum window-level score over the eligible pool |
| Mean | Arithmetic mean over the eligible pool |

All four are parameter-free given the frozen window-level scores — this is
a deliberate design choice. Phase 13's hybrid instability traced directly to
a *tuned* fusion coefficient; removing any tunable parameter from this
study's core comparison removes that entire risk category. No fifth
candidate is added — per the Phase 13 closure report, further distributional
summaries require their own advance scientific justification, and none is
offered here.

## 4. Primary confirmatory test (singular, fixed in advance)

**`AUROC(P90) − AUROC(single-window)`, at delivery (h=0), on 5-fold
patient-grouped CV.** This is the sole primary hypothesis test. Delivery is
chosen as the single primary horizon because it is this project's headline
endpoint everywhere from Phase 6 onward (the number every phase reports
first). CV is the primary evidence source, consistent with `protocol.py`'s
own stated rationale (window/fold variance far exceeds the effect sizes in
question); the held-out test point estimate is required only to **agree in
direction** as a secondary corroboration, not to itself reach significance
given n=83 is underpowered on its own.

**Pre-specified significance threshold:** two-sided patient-level bootstrap
p < 0.05 on the primary test, `B=2000`, paired on identical patients.

## 5. Secondary / exploratory comparisons (reported, not confirmatory)

- P90 vs. single-window at ≥30m (co-reported horizon throughout Phase 13,
  kept for continuity, not part of the primary decision rule).
- Max vs. single-window, Mean vs. single-window — both horizons — for
  context on whether P90 specifically is the right summary statistic or
  whether any pooling operator would do.
- h=10 / h=20 for all three candidates — exploratory only, as in Phase 13.

## 6. Statistical contract (unchanged from Phase 13)

Patient-level evaluation, canonical 5-fold CV, held-out test, paired
patient-level bootstrap (B=2000) + DeLong, both horizon conventions'
descendant (corrected convention only — this study does not reopen the
existing-vs-corrected question, already settled in Phase 13.0A).

## 7. Decision rule (fixed before results are seen)

| Outcome of the primary test | Verdict |
|---|---|
| CV p<0.05 **and** test point estimate positive **and** CV CI(Δ) entirely >0 | **Internally replicated** — recommend promoting P90 pooling to the production aggregation convention, explicitly caveated as pending external validation before being treated as clinically final. |
| CV p<0.05 but CI touches zero, or test disagrees in direction | **Not replicated at the pre-registered bar** — report as still-promising-not-confirmed, no promotion, no further tuning of q or the candidate set. |
| CV p≥0.05 | **Closed** — single-window selection remains the production convention. |

No candidate-set expansion, no new percentile sweep, no hyperparameter
search, and no combination with parity/Huber/stride is permitted regardless
of outcome — those threads are closed under Phase 13 and are out of scope
here by construction (Section 3 has no tunable parameter to expand).

## 8. What this study explicitly is not

Not a hybrid search, not a re-opening of Options A–D, not a claim of
external validity. If the primary test is confirmed under Section 7, the
correct next recommendation is external-cohort validation, not immediate
production deployment.

## 9. Amendments

None yet.
