# Post-lock optimization experiments: three directions, one survivor

Written 2026-09-11, on top of the locked Phase 12.1 submission (`final_synthesis_models`
branch). This documents three optimization directions investigated after the
project's formal synthesis: what was implemented, what broke, what was fixed,
and what the honest final numbers say. All work here is exploratory and
**uncommitted** — nothing in this report has been merged into the locked phase
sequence.

**Bottom line up front:** two of the three directions (adaptive/information-
density weighting, and window-extraction stride) were implemented rigorously
and failed to beat the existing P6 baseline after fixing real bugs found
during verification. The third (fusing a single maternal covariate — parity —
into the existing system) produced the first genuinely promising, multiply-
validated result of the three, though it is not yet proven to the standard
this project holds elsewhere.

---

## 1. Adaptive / information-density weighting

### 1.1 Motivation

Two observations from prior analysis: (a) consecutive 20-minute windows at the
existing 2.5-minute stride are ~97–98% cosine-similar on average — heavily
redundant — and (b) that redundancy is measurably lower for distress-positive
patients near delivery, suggesting uneven information density that uniform
per-window weighting ignores. The design (see
[docs/information_density_weighting_design.md](../docs/information_density_weighting_design.md))
proposed a three-part weight: per-patient normalization (`K/n_windows`), a
causal novelty score (1 − cosine similarity to the previous window, EWMA-
smoothed), and an elapsed-monitoring-time empirical shrinkage prior — combined
as `beta*novelty + (1-beta)*fold_bin_avg(elapsed_time)`.

### 1.2 What was implemented

- [src/training/information_density_weighting.py](../src/training/information_density_weighting.py) — the `InformationDensityWeighter` class.
- [scripts/phase9_temporal_features.py](../scripts/phase9_temporal_features.py) — novelty/elapsed features exposed as model inputs.
- [src/models/train_ctg_crossformer.py](../src/models/train_ctg_crossformer.py) — CLI wiring (`--weighting_scheme`) into the deep model's existing `sample_weight` hook.
- [scripts/phase13_information_density_evaluation.py](../scripts/phase13_information_density_evaluation.py) — the 5-step validation ladder (unweighted baseline → features-only → patient-norm → raw novelty → shrinkage).

### 1.3 What verification found

A full audit against the design spec surfaced three real defects before any
result could be trusted:

1. **Hyperparameter-selection leakage (critical).** The original sweep picked
   `(beta, span)` by maximizing OOF AUROC on the *same* 5 outer folds later
   reported as "Step 4 CV performance" — i.e. it selected hyperparameters
   using the exact data being reported on, and picked β=0.0 in most folds
   (meaning the novelty score itself contributed nothing — the "win" was
   coming from raw elapsed-time bucketing, not information density).
2. **Elapsed-time gap bug.** `elapsed_monitoring_min` was computed as
   `window_index × stride`, silently assuming no window was ever dropped by
   the quality gate — wrong for the 38/547 patients (6.9%) with non-uniform
   stride.
3. **Cross-patient leakage in the feature-exposure script.** Z-scoring
   statistics and the first-window novelty fallback were computed once over
   all 547 patients (train+val+test mixed) rather than per fold — contradicted
   the design doc's explicit fold-isolation requirement, despite the original
   walkthrough claiming "zero-leakage" verification for this component.

Also noted: the deep-crossformer integration was never actually exercised
against real outcomes (all reported numbers came from a plain
`LogisticRegression`), and every weighted variant pushed the operational
false-alert rate from baseline's 82.6% up to 93–98% for a few extra minutes
of lead time — a bad trade even before the statistics are considered.

### 1.4 Fix and re-run

Fixed all three issues: nested CV (nightly-selected `(beta, span)` per outer
fold using only that fold's own training patients; a separate nested selection
for the held-out test partition), gap-aware elapsed time from real
`start_sample`, and out-of-fold z-scoring in the feature-exposure script.

**Result — total delta vs. Step 0 baseline (the comparison that matters, not stepwise):**

| Horizon | CV Δ | CV p | Test Δ | Test p |
|---|---|---|---|---|
| Delivery (0m) | +0.0133 | 0.234 | −0.0027 | 0.890 |
| ≥30m | −0.0174 | 0.087 | 0.0000 | 1.000 |

No horizon reached significance either direction. Step 3 (raw, un-shrunk
novelty weighting) was robustly *significantly worse* than baseline at every
horizon ≤30m (p=0.005–0.028) both before and after the fix — the one finding
that held up cleanly, confirming per-window novelty is too noisy to weight by
directly, exactly as the design doc's own risk section anticipated.

### 1.5 Verdict: **not adopted.**

---

## 2. Window-extraction stride

### 2.1 Motivation

The existing pipeline uses a fixed 20-minute window at a 2.5-minute stride.
Given the redundancy finding above, would a finer stride (more overlap, denser
temporal resolution) sharpen the trajectory/velocity features that Phase
11.5 identified as the second-largest driver of the existing system's
performance?

### 2.2 What was implemented

Changing stride required regenerating a full dependency chain, mapped before
any code was written:

`pipeline_clinical.py` (windows + 8 descriptors) → `phase1_candidates_builder.py`
(P2 signal candidate, independently hardcoded the same stride) → a
reconstructed extended-features generator (the script that produced the
committed `*_extended_features.npy` was never committed to the repo) →
`phase8_rolling_inference.py` (frozen Huber inference) → `phase9b`/`phase9c`
(multidomain severity + 40-D state-trajectory features) → evaluation.

Five pilot scripts (`scripts/stride_pilot_01` through `_05`) implement this
chain for a 1.0-minute stride, monkeypatching the frozen modules' path/stride
constants rather than editing them, reusing the frozen Huber models unchanged,
and reusing the canonical `folds.json` so both stride arms are evaluated on
identical patients. Per-patient weight normalization was applied to **both**
arms in the final comparison — established as a prerequisite beforehand,
since a finer stride mechanically produces more (correlated) windows per
patient and would otherwise bias the comparison the same way the project's
own `audit_bag_size_leak.py` was built to catch.

### 2.3 Verification along the way

- Confirmed the new-stride cohort is patient-for-patient identical to the
  canonical 547 (not just equal in count).
- Confirmed window counts matched exactly (20,586) between the two
  independently-run sub-pipelines that must agree for downstream feature
  lookups to work.
- Re-ran the project's own mandatory time-confound gate on the new data —
  passed (0.4954–0.5046).
- Caught and fixed a self-authored bug: the DeLong return-value unpacking in
  the comparison script had `auc_base`/`auc_pilot` swapped relative to the
  call's argument order, flipping the sign of the printed delta. Caught by
  cross-checking against the independently-computed per-arm breakdown before
  reporting anything, fixed, and re-run.

### 2.4 Result

| Horizon | Baseline (2.5min) | Pilot (1.0min) | Δ | Boot p | DeLong p |
|---|---|---|---|---|---|
| Delivery (0m) | 0.6843 | 0.6678 | **−0.0165** | 0.081 | 0.102 |
| ≥10m | 0.5681 | 0.5699 | +0.0018 | 0.744 | 0.778 |
| ≥20m | 0.5733 | 0.5732 | −0.0001 | 0.967 | 0.985 |
| ≥30m | 0.5678 | 0.5729 | +0.0051 | 0.494 | 0.522 |

No horizon significant. At the primary delivery endpoint, ~2.4x denser
sampling (median 41 vs. 17 windows/patient) trended numerically *worse*
(not significant, but the wrong direction to build a case on).

**Side finding, unrelated to stride itself:** the shared
`get_patient_scores_at_horizon()` convention (used throughout Phases 8–13)
picks the first chronological window still satisfying `time_before_delivery
>= h`, not the window closest to `h` — which for most patients is just their
very first recorded window. This is why the horizon-alignment slop barely
moved between stride arms (~9/18/27 min at h=30/20/10, both arms): stride
density has almost no effect on which window gets selected for `h>0` metrics.
Doesn't invalidate the stride comparison (both arms use identical selection
logic — a fair, paired comparison either way), but it means "≥30m AUROC"
project-wide is closer to "risk score at the start of the observed recording"
than "30 minutes before delivery," worth knowing if that number is cited
elsewhere as literally 30 minutes out.

### 2.5 Verdict: **not adopted.**

---

## 3. Maternal / admission-time covariate fusion

### 3.1 Motivation

With both temporal-lever directions exhausted, this tests a mechanistically
different lever: static, admission-time maternal/pregnancy information, which
the project's own [reports/final_project_synthesis_report.md](../reports/final_project_synthesis_report.md)
(Section 18, future work #3) already flagged as unexplored — "integration of
additional maternal, fetal, labour, and delivery variables alongside CTG" —
but never executed in this track.

### 3.2 Scoping and causal safety tiering

`clinical_metadata.csv` has 43 columns, 98–100% complete. Before building
anything, these were split into three tiers:

- **Safe (admission-time, zero leakage risk):** age, gravidity, parity,
  diabetes, hypertension, preeclampsia, gestational weeks, induced, sex,
  presentation.
- **Needs care (can evolve during labor, not fixed at admission):**
  meconium, pyrexia, premature rupture of membranes, failure-to-progress,
  labor-stage durations.
- **Excluded (only known at/after delivery — leakage):** delivery type,
  birth weight, pCO2, base excess, NICU days, seizures, HIE, intubation,
  diagnosis codes.

### 3.3 Stage 0 — cheap check before building anything

Having just spent two rounds of substantial engineering before getting an
answer, a five-minute univariate/joint check ran first, against the same
547-patient cohort and `y_primary` label used throughout:

| Covariate | AUROC | p (Mann-Whitney) |
|---|---|---|
| age | 0.477 | 0.450 |
| gest. weeks | 0.523 | 0.441 |
| gravidity | 0.465 | 0.107 |
| **parity** | **0.411** | **0.0004** |
| diabetes | 0.486 | 0.301 |
| hypertension | 0.496 | 0.798 |
| preeclampsia | 0.498 | 0.798 |
| induced | 0.476 | 0.356 |
| sex | 0.498 | 0.951 |
| presentation (χ²) | — | 0.508 |

**Joint 9-covariate LR, 5-fold CV: AUROC 0.5132 — near chance.** The
clinically plausible risk factors (diabetes, hypertension, preeclampsia)
showed essentially no signal in this cohort for this endpoint. Only parity
survived — significant even after Bonferroni correction across the 9 tests
(α=0.05/9≈0.0056), and clinically coherent (nulliparity is a textbook labor-
complication risk factor). This five-minute check correctly avoided building
a full fusion pipeline around signal that mostly wasn't there.

### 3.4 Parity-only fusion

Implemented via exact replication of Phase 4's already-proven mechanism
([scripts/phase4_fusion_eval.py:416-444](../scripts/phase4_fusion_eval.py#L416)),
not a new design — low-capacity fusion has out-performed every more flexible
alternative tried in this project:

```
logit(p_fused) = logit(p_P6) + lambda * logit(p_parity)
```

`p_P6` is the existing, frozen Full System probability (reused unchanged, not
retrained). `p_parity` is a per-patient univariate LogisticRegression fit per
fold on that fold's training patients only. `lambda` is selected by a 30-point
grid sweep maximizing training-fold AUROC — the same single-scalar, no-
separate-validation-split procedure Phase 4 used, deliberately kept simple.
Implementation: [scripts/parity_fusion_test.py](../scripts/parity_fusion_test.py).

**Result:**

| Horizon | CV: P6 → fused | CV p | Test: P6 → fused | Test p |
|---|---|---|---|---|
| Delivery (0m) | 0.6872 → 0.7094 (+0.0223) | 0.172 / 0.159 | 0.6497 → **0.7148** (+0.0651) | **0.025 / 0.020** |
| ≥30m | 0.5857 → 0.6351 (+0.0494) | 0.078 / 0.074 | 0.6845 → 0.7629 (+0.0784) | 0.166 / 0.144 |

### 3.5 Shuffle-control validation

The selected lambda (1.3–1.6 across folds) was large enough to warrant a
direct check: could the same lambda-optimization procedure manufacture a
similar gain from pure noise? Ran the project's own Phase-3-precedent control
— 30 permutations with parity shuffled across patients, identical pipeline
otherwise:

```
P6 baseline (CV, delivery):        0.6872
P6 + TRUE parity fused:            0.7094   (+0.0223)
30 shuffled-parity controls:       mean -0.0467, max +0.0060
True delta exceeded by:            0/30 permutations (0.0%)
```

Shuffled parity, run through the identical selection machinery, *hurt*
performance on average — the true signal sits far outside that noise
distribution, ruling out the specific concern that the fusion procedure
itself was curve-fitting.

### 3.6 Verdict: **promising, not yet proven.**

Three independent lines of evidence agree in direction (univariate
significance survives correction, permutation control rules out a selection
artifact, consistent sign across both CV and test at both horizons) — a
materially stronger evidence base than either of the first two directions
produced. But the CV-level bootstrap p-value at delivery (0.172) is not
conventionally significant, and the test-set significance (p=0.025) rests on
only 83 patients (~17 positive), where single-patient placement can move
AUROC by several points. Recommended next steps before treating this as
locked-in, roughly in order of effort: seed-robustness check on the CV
estimate; check whether parity's effect survives controlling for anything it
might proxy for (e.g. correlation with recording length); if it holds, write
it up with the same rigor as Phase 4's original fusion work.

---

## 4. Cross-cutting lessons

- **Hyperparameter-selection leakage is an easy, subtle trap.** It appeared
  in the very first draft of the information-density work — selecting
  `(beta, span)` on the same folds later reported as results is the same
  species of mistake as the historical Phase 7 baseline-drift bug, just one
  layer further in.
- **Report total-vs-baseline, not just stepwise-vs-previous-step.** The
  original information-density walkthrough's narrative ("Step 4 rescues
  performance...") was accurate about the *stepwise* comparison but
  materially misleading about whether the *whole mechanism* beat the
  original system — it didn't, at any horizon, on held-out test.
- **Permutation/shuffle controls are cheap and decisive.** The 30-permutation
  check that validated the parity result took a couple of minutes and
  answered a question bootstrap CIs alone couldn't: not just "is this
  delta large," but "could the selection procedure itself produce this from
  noise." Worth making standard practice for any new fusion/aggregation
  mechanism going forward, matching the project's own Phase 3 precedent.
- **Cheap diagnostics before expensive pipelines.** The five-minute Stage 0
  univariate check avoided building a full 9-covariate fusion pipeline
  (comparable effort to either of the first two directions) around signal
  that was mostly absent. Worth doing first, every time, before committing
  to a multi-stage build.
- **Even carefully-written new code has real bugs.** Two self-authored bugs
  were caught in this session's own new scripts (a DeLong label swap in the
  stride comparison; the original nested-CV leakage before it was fixed) —
  both caught by cross-checking against independently-computed numbers
  before reporting results, not by trusting the first printout.
- **The frozen-model discipline held up.** The Huber models were never
  retrained across either experiment; the P6 baseline reproduced to 4
  decimal places against the published Phase 12 numbers every time it was
  re-derived from scratch. That discipline is what made two large negative
  results trustworthy instead of ambiguous.

---

## 5. Artifacts (uncommitted, `final_synthesis_models` branch)

**New/modified source:**
`src/training/information_density_weighting.py`,
`src/models/train_ctg_crossformer.py` (modified),
`scripts/phase9_temporal_features.py` (modified),
`scripts/phase13_information_density_evaluation.py`,
`scripts/stride_pilot_01..05_*.py`,
`scripts/parity_fusion_test.py`.

**Design docs:** `docs/information_density_weighting_design.md`.

**Results:** `results/phase13_information_density/`,
`results/stride_pilot/`, `results/parity_fusion/`,
`results/phase8_rolling_stride1p0/`, `results/phase9b_deterioration_stride1p0/`,
`results/phase9c_state_trajectory_stride1p0/`,
`data/processed_clinical_stride1p0/`.
