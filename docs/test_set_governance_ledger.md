# Test-set governance ledger and freeze declaration
(User review, Problem 8 — "the distinction between CV → validation → test is unclear")

## 1. What the 83-patient held-out test partition is
`data/processed_clinical/test_dataset.pt`, fixed at the original 70/15/15 patient-level split (`random_state=42`,
`src/preprocessing/pipeline_clinical.py`), 83 of the 547 patients, never in any of the 5 canonical training/validation folds.
It has **not** been regenerated or reshuffled at any point in this project's history.

## 2. Every time it has been touched, in order (honest ledger, not a summary)
| Phase | What was evaluated on it | Selection happened using it? |
|---|---|---|
| 18 | URM (A3_selected_once) and every other deployable-fusion candidate, once, after all fusion-mechanism choices (α, F2, F7, A4) were made on CV only | No — architecture/α/threshold were frozen from CV before this |
| 19b | U_TAM (=URM) and U_PRS (train_val → test), once | No — arms were pre-registered; gate compared against frozen CV numbers first |
| 21 screening | not touched | — |
| 21b (exploratory follow-up) | G-A and G-A-wide (train on D464 → test), once | No selection *among* G-A variants happened after seeing test; but this was the **second** time the partition was opened for a genuinely new model family |
| 22 | G-A final, evaluated on test again (same fit as 21b, additional metrics only) | No new selection; reused the already-fit 21b model |
| 23, 24, 25 (V2/V2E, Candidate D, Candidate D2) | **not touched** — explicitly out of scope in each protocol | — |
| 26 (this phase) | Calibration/PPV-NPV/threshold-curve computed on test **for URM only**, using the already-frozen Phase 18 test fit | No new selection |

**Net assessment:** the partition has been opened for **two distinct model lineages** (URM's own components in Phases 18/19b, and the
exploratory G-A lineage in Phases 21b/22) with no candidate *selection* performed using its results either time — every arm evaluated on it
had already been fixed by CV before the test numbers were seen. That is the correct discipline. But it has been *looked at* four times now
across two lineages, which is more than a single clean "final exam" use, and the user's concern (Problem 8) is legitimate: repeated exposure,
even without selection, erodes how independent a test-set number can be treated as, especially for the exploratory line.

## 3. Freeze, effective now (2026-09-26)
- **The 83-patient test partition is frozen for the exploratory lineage (G-A / D / D2 / any future internal-cohort candidate).** No further
  evaluation on it is permitted until either (a) external hospital data arrives (Phase B, see `external_validation_master_protocol.md`), or
  (b) a candidate is formally proposed for adoption into the locked model, at which point it gets **one** final look, pre-registered, and
  the ledger above is updated.
- **URM itself may continue to be evaluated on it** for reporting purposes only (as in this phase's calibration/PPV-NPV work), since URM is
  locked and not being selected — this does not compromise anything, because URM's parameters cannot react to what is found.
- Any violation of this freeze (a new script that loads `test_dataset.pt` for a candidate still under development) should be treated as a
  protocol breach and flagged before results from it are reported as if independent.

## 4. Going forward: the diagram the user asked for
```
547 patients
     |
     +-- Development set (464 train+val)
     |      +-- patient-grouped CV (canonical folds.json + resplits)
     |             -> architecture, hyperparameters, fusion weight, threshold
     |
     v
 LOCK MODEL  (already done for PRS / TAM / MCM / URM, alpha = 0.35)
     |
     v
 FINAL INTERNAL TEST (83 patients)   <-- frozen as of 2026-09-26 for anything not already locked
     |
     v
 NO MORE MODEL CHANGES on this cohort
     |
     v
 EXTERNAL HOSPITAL DATA (Phase B) -- see external_validation_master_protocol.md
```
This is now the binding rule for every future phase in this project, not just a description of best practice.
