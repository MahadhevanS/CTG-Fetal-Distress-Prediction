# External validation master protocol
(User review, Problems 1 and 2 — small dataset / no external validation)

**Status: planning document, no external data available yet.** This fixes the rules *before* any hospital data arrives, so the analysis
cannot be steered by what that data turns out to show. It supersedes nothing already locked; it governs how new data will be used once it exists.

## Phase A — current CTU-UHB (essentially complete)
- Architecture locked: PRS (P6) → TAM → MCM (parity) → URM, α = 0.35, all frozen.
- Patient-grouped CV: canonical `folds.json`, 5 folds, used throughout.
- Final held-out internal test: the 83-patient partition (see `test_set_governance_ledger.md`), now frozen for anything not already locked.
- **Nothing about Phase A changes because of Phases B/C below.** URM stays exactly as it is unless a future amendment explicitly relocks it.

## Phase B — independent hospital data: tests generalisation, does not improve the model
**The question Phase B answers is "does the locked URM generalise?" — not "how do we make URM better?"** Concretely:
1. Hospital data is collected and processed through the **same signal pipeline** (or the closest faithful equivalent the new source allows —
   documented deviations are a Phase-B finding, not silently absorbed).
2. It is **not** mixed into any training, fold, or hyperparameter selection. It is scored **once**, by the already-frozen URM (and, for
   context, by PRS/TAM/MCM individually), exactly as it is today.
3. No threshold, α, or feature is retuned on it. If URM's calibration is poor on the new data, that is *reported*, not fixed by refitting
   the recalibration step on the new cohort — refitting would belong to Phase C, not Phase B.

### The three possible outcomes — pre-declared, so none of them can be treated as the "expected" one
| Outcome | What it would look like | What it means |
|---|---|---|
| **A. Generalises** | AUROC/calibration/operating-point metrics on the new cohort within roughly the CTU-UHB canonical-vs-resplit noise band (~±0.02–0.03 AUROC; see Phase 25's cross-phase measurement that even *within* CTU-UHB, the canonical split itself sits outside that band relative to fresh resplits — so the bar for "generalises" should be informed by that internal noise floor, not assumed to be zero-tolerance) | Strong evidence the representation is portable |
| **B. Moderate degradation** | Discrimination or calibration drops beyond the internal noise band but the model still separates cases usefully | Still valuable: quantifies the distribution shift and which physiological features it affects (see the checklist below) |
| **C. Major failure** | Near-chance discrimination, or systematically wrong calibration direction | Also valuable — triggers the investigation checklist, not a discard of the whole approach |

**We do not assume A.** The evaluation protocol (metrics, thresholds, splits) is fixed *before* the new data is scored, exactly as every
phase in this project has done, so the outcome cannot retroactively change what counts as success.

### Investigation checklist for outcome B or C (pre-specified now, not improvised after seeing a bad number)
For each of these, check whether it plausibly differs between CTU-UHB and the new source, and by how much, before concluding the model itself
has failed:
- sensor / device differences (manufacturer, sampling rate, FHR-vs-UC channel assignment);
- preprocessing differences (baseline estimation, artifact rejection, gap-filling — anything in `src/preprocessing/`);
- patient population (gestational-age range, risk stratification at admission, referral pattern to that hospital);
- prevalence of the outcome (pH ≤ 7.15 rate; recall CTU-UHB's 20% is already high for a general obstetric population);
- clinical practice (how early monitoring starts, intervention thresholds, how "delivery" is timed relative to the last recorded window);
- pH measurement protocol (cord-blood sampling technique, assay, timing);
- missingness / signal-quality profile (dropout rate, artifact rate — compare against `results/phase21_diagnostics/` if similar checks are wanted for the new cohort);
- feature-distribution shift on the 40 locked descriptors — a direct, checkable comparison (histogram / KS-test per feature, CTU-UHB vs new cohort), the most actionable single diagnostic if discrimination drops.

## Phase C — pooled retraining, gated on Phase B completing first
Only after Phase B's result (A, B, or C) is known and reported does pooled retraining become appropriate, and only if Phase B shows enough of
a gap to justify it (outcome A alone would argue for *no* retraining — if it already generalises, retraining risks overfitting the combined
set to whichever cohort is larger). If undertaken:
- new patient-grouped folds across the pooled cohort, stratified by source hospital as well as outcome;
- the *same* discipline as Phases 19–25: pre-register, gate against the Phase A locked numbers, no model change is "final" until it passes
  its own held-out test on data from **both** sources, never just the new one.

## What this means for "your upcoming hospital data" specifically
When that data arrives, the first script written against it should be a **Phase B scoring script only** — load the frozen URM
(`results/phase22_ga_final/frozen_model/` is the exploratory G-A model, not this; the operative locked model is Phase 18's URM, reproducible via
`scripts/phase18_deployable_freeze_and_operational.py` / `scripts/phase26_clinical_evaluation.py`'s `build_urm_sequences()`), score the new
patients once, and report against this document's pre-declared outcome table — before any exploratory analysis of what might be wrong or how
to improve on it.
