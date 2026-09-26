# Phase 20 — Redesign of the window model (PRS-v2) and trajectory model (TAM-v2), starting from URM
## Pre-Registration Protocol

> **Status (2026-09-21): executed — no candidate ADOPTED; no URM-v2; the frozen URM stands.** Stage 0 gates G1–G4 passed; Stage 1 (C1–C5c, C4) and Stage 2 (C6a) all NOT SUPPORTED or SUGGESTIVE-only; Stage 3 and the test partition were not run. Full results: `results/phase20_redesign/phase20_report.md`. Protocol text below unchanged apart from Amendments 1–3 in §7.

**Frozen 2026-09-21, before any Phase 20 code was written or run.** Same discipline
as `docs/phase13_protocol.md` through `docs/phase19b_*`.

> **Governance.** The user authorised redesigning PRS and TAM on 2026-09-21. The locked
> artefacts (PRS/P6 predictions in `results/phase13/audit/p6_predictions.npz`, TAM
> checkpoints in `results/phase16/checkpoints`, MCM, URM and every Phase 12–19 result) are
> **never modified or overwritten**: they remain the baseline. Everything new is written
> under `results/phase20_redesign/` with a `-v2` name. Nothing is promoted to "final" by
> this protocol; it only determines which components are *adopted into the URM-v2 candidate*.
> Every result is reported whatever the outcome.

---

## 1. Starting point and the evidence that constrains this phase

Starting model: **URM** = `α·p_TAM(t) + (1−α)·p_MCM`, α≈0.33 (`docs/phase18_deployable_fusion_protocol.md`).
CV AUROC 0.7335 / 0.6957 / 0.6675 / 0.6646 at delivery / ≥10 / ≥20 / ≥30 min (mean **M = 0.690**).
Phase 19b: about the same fusion without TAM (latest PRS window + parity) is ΔM −0.0065 (p = .26).

Prior negative evidence, stated up front so that a null result is the expected outcome and not a surprise:

| Route touched by this phase | Prior result | Source |
|---|---|---|
| Raw FHR (1/4 Hz) instead of descriptors | KILL, −0.115 | `docs/phase10_exp1_raw_fhr.md` |
| HRV / spectral / nonlinear features | −0.0152, closed | `docs/pH_information_hypothesis_ranking.md` |
| 14 maternal/clinical covariates **inside the window model** | −0.0175, closed | `docs/phase8_information_audit.md` §4.1 |
| Per-contraction trajectory, missingness, SSL, full-length recording | all ≤ +0.003 | same |
| Patient-normalisation / novelty window weights | no gain; nested CV picked β≈0 | `results/phase13_information_density/` |
| Hand-crafted label-confidence window weighting | CV +0.019, held-out test worse, not adopted | commit `10ce497` |
| Attention-pooler redesign (reliability cues, ranking objective) | NOT SUPPORTED; oracle pooler headroom ≈ +0.016 | Phase 19 |
| Parity as a **late** fusion input | worked (this is URM) | Phase 18 |

Consequences built into the design: (i) covariates enter only by **late fusion**, the form that
has worked, and never inside the window model; (ii) spectral/HRV features are **excluded**;
(iii) the adoption bars below are deliberately high and multiplicity-corrected; (iv) with 110
positives the detectable difference is roughly ±0.01–0.03 AUROC, so small wins are expected to be
indistinguishable from noise and are reported as such.

## 2. Data universe and selection-independence (fixes a known weakness)

Earlier phases ran CV over all 547 patients, and the 83-patient test partition is a subset of that,
so the test partition was never independent of model *selection*. This phase removes that:

- **Development set D464** = the 464 train+val patients (`test_dataset.pt` patients excluded).
  **All selection and adoption decisions use only D464.** CV folds = the canonical `folds.json`
  assignment restricted to D464 (fold membership of the 464 patients unchanged), plus 5 resplits
  (`StratifiedKFold`, seeds 11/22/33/44/55, on D464).
- **Fresh confirmation resplits** (never used for any selection): seeds 66/77/88/99/111 on D464.
- **Test partition (83 patients)**: touched **exactly once**, at the very end, for the final
  URM-v2 vs frozen URM (train on all D464 → predict 83). Never used for selection. It is small
  (about 17 positives), so it is reported and can *flag a contradiction* but is not a gate.
- **547-patient canonical CV** of the final composite is reported for continuity with all earlier
  headline numbers; it is descriptive only (it contains the test patients).
- Legacy caveat, stated not hidden: PRS/TAM/parity/URM design choices in Phases 8–19 were made with
  all 547 patients visible. The frozen baseline inherits that; only the new candidates are
  selection-clean.

## 3. Primary metric, horizons, and the common evaluation harness

- Horizons 0/10/20/30 min via the eligible-prefix rule (`eligible_prefix_length`); N = 464 (D464 CV)
  / 83 (test) at every horizon, no patient dropped. Horizons are retrospective evaluation
  constructs only; every model is horizon-agnostic at inference.
- **Primary metric M** = mean over the four horizons of patient-level AUROC of the end-to-end
  deployable score. **ΔM = M(candidate) − M(baseline)**, both computed in the same harness.
- Statistics: paired patient bootstrap, B = 2000, seed 42 (`scripts/phase11_bootstrap.py`
  convention: patient resampling, percentile CI, two-sided empirical p), on ΔM and per-horizon Δ.
- **Common harness (identical for every arm; only the CTG score sequence differs):**
  1. per split, produce out-of-fold window/step scores from the arm's CTG model;
  2. train the pooler (TAM control architecture 2→8→1 tanh, Adam lr .01, wd 1e-4, full-batch,
     ≤200 epochs, patience 10, inner-val 15%, pooled BCE over every causal truncation, 1/T
     normalisation) on those scores, **seeds {42,43,44}, mean of the three seeds' running scores**
     (`scripts/phase19_pooler_experiment.py` trainer, re-used);
  3. fit MCM (StandardScaler + LR C=1.0) on the split's training patients;
  4. select α once per fold by sample-weighted (1/T_i) pooled training AUROC, grid step 0.05;
  5. score held-out patients at each horizon's eligible prefix.
- **Baseline arm B0 ("URM-harness")** = P6 refit per split (LR C = 0.05 on the locked 40-D
  features, unweighted) run through steps 2–5.

## 4. Candidates (fixed list — nothing is added after this point)

Each candidate changes **one thing** relative to B0 (or, for C4, relative to the URM fusion), with
its own small hyper-parameter grid, selected by **inner patient-grouped 4-fold CV inside each outer
training set** using the inner-OOF mean over horizons of single-window patient AUROC. The outer
held-out fold never participates in any choice.

### CTG window-model family (Stage 1; Holm family, K = 6)

| ID | Change | Frozen definition | Inner grid |
|---|---|---|---|
| **C1** proximity-weighted training | per-window loss weight `w = max(exp(−t_del/τ), 0.2)`; `t_del` is used only at training time to weight windows and is never an input or used at inference | | τ ∈ {20, 40, 80} min |
| **C2** patient-relative features | append, for each of the 19 raw descriptors, `raw_t − mean(raw_s, s ≤ t)` over that patient's own causal history (59-D total) | | C ∈ {0.01, 0.05, 0.2} |
| **C3** learner upgrade | replace the LR by the best of: elastic-net LR (saga; l1_ratio {0.3,0.7} × C {0.05,0.2}); shallow HistGradientBoosting (depth 3, lr 0.05, 150 iter, min_samples_leaf 100, l2 1.0); mean of the P6 LR and the HGB probabilities | on the locked 40-D features | the 6 configs listed |
| **C5a** multi-scale | recompute the 11 FHR-only descriptors on the **last 5 min and last 10 min** of each 20-min window (22 new features, 62-D total) | descriptors via `src/figo_state/descriptors.compute_descriptors` on the tail sub-window | C ∈ {0.01, 0.05, 0.2} |
| **C5b** long context | the 11 FHR-only descriptors on the **40-min span** ending at the window (current window plus the window 20 min earlier); where the span is not fully available the 20-min value is used (no availability flag, to avoid an elapsed-time proxy). 51-D total | | C ∈ {0.01, 0.05, 0.2} |
| **C5c** decel morphology | 6 new FHR/UC features per window: mean and max decel **recovery time** (nadir → within 5 bpm of baseline); mean and max **post-decel overshoot** amplitude; **worsening trend** of decel nadir depth across the window's decels; fraction of contractions followed by a decel with recovery > 60 s. 46-D total | events from the existing detectors `delineate_events` / `classify_decelerations` / `detect_contractions` | C ∈ {0.01, 0.05, 0.2} |

Feasibility rules (applied before any result is read, and logged): a feature that needs an event the
existing detectors do not expose is dropped and listed; if C5c cannot be built with the detectors it
is recorded "not run". HRV, spectral, and nonlinear features are excluded (closed, §1).

### Sequence-model family (Stage 2; Holm family, K = 2)

| ID | Change | Frozen definition |
|---|---|---|
| **C6a** causal GRU | replaces *both* the window model and the TAM pooler: a single unidirectional GRU (hidden 16) over the chronological sequence of standardised **locked 40-D** window vectors plus `elapsed/60`, input dropout 0.3, a linear + sigmoid head giving a patient-risk score at every step; trained with the TAM objective (BCE at every causal truncation, per-patient 1/T normalisation), Adam lr 3e-3, wd 1e-3, ≤150 epochs, patience 10 on inner-val BCE (15%, stratified, seed 42+fold), seeds {42,43,44}, mean of the three seeds' running scores. **No hyper-parameter is searched.** Its running score feeds steps 3–5 of the harness in place of TAM's. |
| **C6b** GRU on v2 features | as C6a but on the feature set adopted in Stage 1; **run only if at least one of C2/C5a/C5b/C5c is adopted**, otherwise recorded "not run". |

### Late-fusion covariates (Stage 1, sequential fixed-order testing)

**C4** replaces MCM with a covariate model `p_clin` inside the URM form `α·s(t) + (1−α)·p_clin`, where `s(t)` is the **frozen** TAM running score (so this stage costs no retraining). `p_clin` = StandardScaler + LR with C ∈ {0.1, 1.0} chosen by inner CV; training-fold median imputation (missingness: gravidity 4, presentation 3, all else 0). Only antepartum/in-labour-observable variables are allowed; **post-hoc/leaky variables are excluded**: pH, BDecf, pCO2, BE, Apgar, NICU, seizures, HIE, intubation, diagnoses, weight, sex, first/second-stage durations and flags, no-progress, `ck/kp`, delivery type, `sig2birth`, `pos. ii.st.`.

| Set | Variables | 
|---|---|
| S0 (baseline) | parity |
| S1 | S0 + maternal age + gravidity + gestational weeks |
| S2 | S1 + `any_risk` (diabetes ∨ hypertension ∨ preeclampsia ∨ pyrexia) + `liq. praecox` + meconium |
| S3 | S2 + induced + presentation |

Fixed-sequence testing (no multiplicity correction needed): S1 vs S0; **only if S1 is adopted**, S2 vs S1; **only if S2 is adopted**, S3 vs S2. Each at its own α = .05 with the §5 rules. (Rare flags are merged into `any_risk` because pyrexia and preeclampsia have too few cases to estimate separately.)

### Fixed design choices (no test)

Seed ensemble of 3 network seeds is used for every arm including the baseline (Phase 19 convention),
so it is not a candidate. Item "auxiliary lead-time targets" is **not** included: Phase 19 arm A2
(the horizon-grid ranking objective) already tested it and it was NOT SUPPORTED.

## 5. Stages, gates, and decision rules (mechanical, fixed now)

### Stage 0 — gates (read-only; stop if any fails, do not adjust arms)
- **G1** P6 refit per canonical fold on all 547 reproduces `p6_predictions.npz` (max |diff| < 1e-6).
- **G2** the harness run on the 547-patient canonical folds with B0 reproduces frozen URM within ±0.006
  per horizon (0.7335/0.6957/0.6675/0.6646). The ±0.006 is the Phase 19 gate for a 3-seed pooler
  vs a single-seed frozen TAM.
- **G3** if C5 features are built: recomputing the 19 descriptors on the full 20-min window
  reproduces `X_state_trajectory[:, :19]` within 1e-5; all new features pass a future-perturbation
  test (altering samples after the window end leaves the feature unchanged to 1e-9).
- **G4** the noise floor is measured and reported: SD of M across B0's 6 D464 splits; seed-to-seed SD of M.
- Baseline B0 metrics on D464 (canonical + 5 resplits) are computed here and are the reference for all ΔM.

### Stage 1 — CTG window-model candidates C1, C2, C3, C5a, C5b, C5c, and C4 covariates
For each PRS candidate X vs B0, on D464 canonical CV and the 5 resplits:

- **ADOPT** iff all hold: (A1) canonical ΔM ≥ +0.005 and **Holm-adjusted** p < .05 (family K = 6 canonical
  bootstrap p-values); (A2) ΔM > 0 in **all 5** resplits; (A3) no per-horizon Δ worse than −0.020
  canonically or in the resplit mean.
- **SUGGESTIVE** iff not ADOPT, canonical ΔM > 0 and ΔM > 0 in ≥ 4 of 5 resplits. Reported; **not** carried forward.
- **NOT SUPPORTED** otherwise.

C4 covariate sets follow the fixed-sequence rule with the same ADOPT conditions (unadjusted p < .05, since
the sequence controls the family). Adopted PRS candidates and the adopted covariate set are then combined
**once** (one composite arm, no interaction search). If two adopted candidates modify the same
component (for example C2 and C5b both change the feature vector) the composite includes both; if the
composite's ΔM is below the best single component's, the best single component is used instead.

### Stage 2 — sequence model C6a (and C6b if applicable)
C6a (and C6b) vs the **Stage 1 composite** if anything was adopted, otherwise vs B0; same ADOPT rule with
Holm family K = 2 (K = 1 if C6b is not run). Additionally, C6a is always also compared against B0 and
reported (secondary).

### Stage 3 — final composite and confirmation (decides "URM-v2")
The composite of all adopted components (or, if nothing is adopted, **no URM-v2 exists and the frozen URM stands**).
On the **fresh resplits** (seeds 66/77/88/99/111, D464) and canonical D464 CV, URM-v2 vs the frozen-structure baseline:

- **CONFIRMED** iff (F1) canonical D464 ΔM ≥ +0.005 with paired-bootstrap p < .05; (F2) ΔM > 0 in ≥ 4 of the
  5 fresh resplits; (F3) no per-horizon Δ worse than −0.020 canonically or in the fresh-resplit mean.
- Otherwise **NOT CONFIRMED**; the frozen URM remains the recommendation, and the composite is reported as
  exploratory.
- Then, once: train on all D464 → 83-patient test partition, URM-v2 vs frozen URM. Reported with the
  direction of ΔM; a test ΔM < −0.020 is reported as a **contradiction flag** and blocks the word
  "improved" even if CONFIRMED. Not a gate on its own (n = 83).
- Descriptive extras: 547-patient canonical CV, operational metrics (80% target sensitivity, per-fold
  training-only threshold, patient-level ever-alerted), selected α, per-arm parameter counts.

## 6. Implementation plan

Scripts (all new): `scripts/phase20_stage0_gates.py`, `phase20_stage1_candidates.py`,
`phase20_stage2_sequence_gru.py`, `phase20_stage3_confirmation.py`; shared harness in
`scripts/phase20_harness.py` (built on `phase19_pooler_experiment.py` and `phase19b_fusion_tam_vs_prs.py`
helpers). Results in `results/phase20_redesign/`. Every stage writes its CSVs and a
`verdict_*.json`; verdicts are produced by code applying §5, not by hand.

## 7. Deviations

Any change after this freeze is recorded as a dated amendment appended below and reported in the final
report; results obtained before an amendment are not silently re-labelled.

**Amendment 1 (2026-09-21, during Stage 0, before any candidate was run).** §4 names
`src/figo_state/descriptors.compute_descriptors` as the source of the descriptors for C5a/C5b/C5c. That was
wrong: the 19 raw descriptors in the locked 40-D features are produced by the per-window chain in
`src/preprocessing/pipeline_clinical.py` (remove_spikes → interpolate → lowpass → iterative baseline →
`calculate_variability` / `detect_accelerations` / `detect_decelerations` → 8 clinical features, plus the 11
features of `src/knowledge/extended_features.py`). C5a/C5b/C5c will use **that** chain, so the new features are
built exactly like the existing ones; gate G3(a) verifies this against the stored values. Which features, scales
and definitions are used is unchanged. Detector wording in C5c ("existing detectors") therefore refers to
`src/preprocessing/features.py` and `extended_features.py`; a C5c feature that these do not expose is dropped
and listed, per the existing feasibility rule. Nothing about the decision rules changes.

**Amendment 1 status:** gate G3(a) passed on 2026-09-21 after `wfdb` was installed (40 patients, 608 windows, max |diff| 3.6e-6 ≤ 1e-5). The extended features were found to have been computed from the z-scored float32 on-disk tensors, so the recomputation reproduces that normalisation round trip; this is a property of the existing chain, not a change to the protocol.

**Amendment 2 (2026-09-21, before any candidate was run).** The chain's decel detector (`detect_decelerations`) returns counts only. For C5c the event spans are re-derived with the **identical run rule** (≥15 bpm below baseline for ≥15 s), and the build asserts that the number of spans equals early+late+variable+prolonged of `detect_decelerations` on every window. C5c definitions as listed are unchanged, with two clarifications fixed now: recovery time and overshoot are computed only over events whose recovery is observed inside the window (features are 0 when there are none); a "contraction followed by a decel" is one whose nadir lies 0–90 s after a UC peak found with the chain's own `find_peaks(uc, distance=30 s, prominence=10)`. Decision rules unchanged.

**Amendment 3 (2026-09-21, before any Stage 1 candidate was run).** Three implementation details of C4 that §4 left open or that conflicted with §2:
(i) C4 uses the **B0-harness TAM running scores** (pooler trained inside D464 only) rather than the frozen TAM checkpoints, because the frozen fold scorers were trained on folds that include the 83 test patients, which would let test-patient labels influence a D464 selection (contrary to §2). Everything else in C4 is as written.
(ii) `presentation` is coded as the binary `non-vertex = 1[presentation ≠ 1]` (3 records missing → training-fold median); `induced`, `liq. praecox`, `meconium` are the raw 0/1 flags; parity, age, gravidity and gestational weeks enter as raw numbers (standardised inside the model).
(iii) S1's comparator S0 is the URM's own MCM (LR C = 1.0, fixed); for S2 vs S1 and S3 vs S2 the comparator is the previously adopted set with its inner-CV-selected C.
Decision rules unchanged.

## 8. Out of scope

Retuning the URM α grid or fusion form; spectral/HRV/raw-waveform models; SSL/external data; horizon-conditioned
models; early fusion of covariates inside the window model (closed, §1); re-opening Phase 19 arms A1–A3; changing the cohort, endpoint (pH ≤ 7.15), or canonical folds.
