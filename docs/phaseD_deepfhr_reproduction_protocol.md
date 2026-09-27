# Phase D — DeepFHR Reproduction & URM Comparison
## Pre-Registration Protocol

> **Status (2026-09-26): D1 complete.** Two fixes applied and verified: class-weighted loss (fixed the Se/Sp collapse) and per-channel zero-centering (per the paper's own Table 2 spec, previously missing). Neither closed the AUC gap (pooled AUC 0.66-0.69 vs published 0.978). Diagnosed why: a pixel-level control confirms the leakage signal is real and strong (AUC 0.93); an epoch-count sweep (20/100/300) shows train AUC itself plateaus at ~0.72, ruling out undertraining. The remaining gap is attributed to the disclosed wavelet-basis substitution (SS1), not training configuration. Gate category: inconclusive for the reproduction question, cause now understood and documented. Full results: results/phaseD_deepfhr/phaseD1_report.md. D2 (Amendment 1) executed and SETTLED: on the identical 547 patients, canonical folds, causal horizon rule and bootstrap convention as URM itself, the DeepFHR reconstruction scores 0.537/0.591/0.575/0.567 at 0/10/20/30 min vs URM 0.734/0.692/0.664/0.663 -- URM wins at every horizon, p < .02 everywhere (p < .0001 at delivery), no CI crossing zero. Full results: results/phaseD_deepfhr/phaseD2_report.md.**

**Frozen 2026-09-26, before any Phase D training code was written or run**, after (a) reading the primary source directly (not
relying on this project's own earlier, less precise characterization of it) and (b) confirming what tooling is actually
available. Structure follows the user's plan exactly: D1 (faithful reproduction) must be attempted **before** any leakage
conclusion is drawn, on the paper's own protocol, even though we expect it to be flawed.

> **Governance.** Nothing about the locked PRS/TAM/MCM/URM changes. This is a new, separate investigation. The existing
> `src/cwt/` 2D-representation ablation (`reports/cwt/`, 2026-09-05) is **not** superseded — it answered a different, valid
> question ("do 2D time-frequency representations of this project's own rolling windows beat 1D/clinical descriptors under
> patient-grouped CV?" — no) using a generic complex-Morlet CWT on this project's own 20-min/2.5-min-stride windows. It was
> **not** a faithful reconstruction of Zhao et al.'s specific image-construction scheme, and its 0.5301 AUROC should not be
> quoted as "we reproduced DeepFHR and it collapsed" — that claim is what Phase D1 below is for.

## 0. Primary-source verification (done before this protocol was frozen)

Fetched and read in full: Zhao, Z., Deng, Y., Zhang, Y. et al. **DeepFHR: intelligent prediction of fetal Acidemia using
fetal heart rate signals based on convolutional neural network.** BMC Med Inform Decis Mak 19, 286 (2019),
https://doi.org/10.1186/s12911-019-1007-5 (open access, CC-BY 4.0).

**Verified facts, with the exact source text, replacing this project's earlier second-hand summary:**

| Fact | Verified value | Source text |
|---|---|---|
| Cohort | 552 CTU-UHB recordings, **447 normal / 105 pathological** | "the database contained 447 normal and 105 abnormal FHR recording" |
| Outcome definition | pH **< 7.15** pathological, pH **≥ 7.15** normal (strict less-than, not ≤) | "A pH below 7.15 was agreed as pathological and a pH greater than or equal to 7.15 was classified as normal" |
| Preprocessing | spline-fill gaps ≤15s (else discard), then a custom despiking pass (interpolate where adjacent-sample diff >25 bpm, until 5 consecutive samples differ <10 bpm), then cubic-spline replace values <50 or >200 bpm | direct paraphrase of the Signal preprocessing section |
| **Image construction** | **One 20-minute segment per recording**, transformed by **2 mother wavelets (db2, sym2) × 3 wavelet scales (4, 5, 6) → 6 images per recording**, **552 × 6 = 3312 total** (2682 normal / 630 pathological) | "two mother wavelets of db and sym with an order of 2 and three wavelet scales of 4, 5 and 6 were determined to enrich the database. Thus, the final dataset contained 3312 time-frequency images" |
| Image size | 64×64×3 RGB (best of 4 tested resolutions; largest tested, most compute) | Table 5, "Set4" |
| CNN | 8 layers: input(+random-crop aug) → conv(5×5, 15 filters, stride 1, pad 0) → ReLU → batch-norm → max-pool(2×2, stride 2) → FC(256) → dropout(0.5) → FC(2) → softmax | Fig. 4, Tables 2–4 |
| Training | SGD, momentum 0.9, initial lr 0.01, lr ×0.1 every 10 epochs, L2 1e-4, 20 epochs, batch size 50 | Table 3, "Para3=20, Para4=50" |
| **Split** | **image-level random**: "the total images were randomly separated into 10 segments and 90% ... formed the training set while the remainder (10%) was used to test" | Results §"Experimental setup" — **confirms this project's leakage hypothesis directly, from the primary source, not by inference** |
| Metric | Acc/Se/Sp/QI/AUC averaged over the 10 repeats; **QI = √(Se·Sp)** (geometric mean) | Eq. 4–7 |
| Published result | **Acc 98.34 / Se 98.22 / Sp 94.87 / QI 96.53 / AUC 97.82** (all %) | Abstract, Table 5 "Set4" |

**This confirms the leakage mechanism is more direct than this project's earlier "adjacent overlapping windows" framing**: it is
six near-duplicate images of the *same* 20-minute segment (only the wavelet/scale differ) landing on both sides of a random
split — arguably an even more blatant leak than overlapping-but-distinct windows.

## 1. One unavoidable reproduction ambiguity (declared now, not resolved by trial and error later)

MATLAB's legacy `cwt()` (the version in wide use circa 2017, before the Wavelet Toolbox's 2016+ rewrite to analytic-only
`bump`/`morse`/`amor` wavelets) accepts discrete wavelet families (`db2`, `sym2`, …) as a **real-valued** CWT basis, computed
by direct convolution of the signal with a dilated copy of the wavelet's own reconstruction function at each integer scale.
Neither of the two standard open-source equivalents available here (`pywt.cwt`, which only accepts proper continuous/analytic
wavelets and errors on `db2`/`sym2`; `scipy.signal.cwt`, which takes an arbitrary wavelet *function*) implements that exact
legacy algorithm, and MATLAB itself is not available in this environment. **Resolution, fixed before any training run:**
reconstruct each discrete wavelet's scaling function via `pywt.Wavelet(name).wavefun(level=8)`, dilate it by the paper's
integer scale value (4, 5, or 6), and convolve with the signal via `scipy.signal.cwt`'s callable-wavelet interface — the
closest available reconstruction of "real-valued CWT with a discrete-wavelet basis." This is stated as an assumption, not
hidden; if D1 fails to approach 0.98, this substitution is the first thing to revisit (§5, gate categories already allow for
"partial reproduction; investigate implementation differences").

Everything else in Phase D1 (preprocessing, image size, CNN architecture, training hyperparameters, split procedure, metric
definitions) is implemented exactly as verified in §0, with no substitutions.

## 2. Phase 0 — dataset and config freeze

`data/raw/ctu-chb-intrapartum/` (already on disk, matches the same physionet source). Config file
`docs/deepfhr_original.yaml` records every paper-derived parameter (wavelets, scales, image size, CNN spec, training
hyperparameters, split ratio, repeats) so later phases (D2+) change *one* axis at a time against a fixed reference.
**Cohort caveat, carried forward explicitly:** this project's own canonical cohort is 547 patients / 110 positive
(pH ≤ 7.15, inclusive), 5 fewer patients and a different threshold direction than DeepFHR's 552/105 (pH < 7.15, strict). D1
reproduces the *paper's own* cohort definition (552/105) to match its target; D2 onward switches to this project's own
canonical 547/110 cohort and folds so that the final URM comparison (§8+) is on the same patients URM was evaluated on.

## 3. D1 — faithful, paper-style reproduction (run first, whatever we expect)

Exactly the recipe in §0: 552 recordings → 1 segment each (the **last 20 minutes** before the recording ends, the natural
reading of "20mins in length" in Fig. 2 and standard practice for an intrapartum acidemia label at delivery — not stated
verbatim in the paper, the one segment-choice not pinned down by the text, declared here rather than tuned later) → 6 CWT
images/recording (db2/sym2 × scales 4/5/6, §1 method) → 64×64×3 → the exact 8-layer CNN → **image-level random 10-fold**,
90/10 per fold, **standard 10-fold CV: one random image-level partition, each fold used once as test** (ASSUMPTION:
"the process was repeated 10 times and the final results were averaged" is read as the standard 10-fold-CV description —
10 folds cycled once each — not 10 independent re-partitions; this is the more common reading and keeps training to 10
models rather than 100). Report Acc/Se/Sp/QI/AUC, mean ± std across the 10 folds, next to the published
98.34/98.22/94.87/96.53/97.82.

**Gate categories (fixed now):** AUC ≥ 0.95 → faithful reproduction, proceed treating D1 as the confirmed leaky ceiling.
0.85–0.95 → partial; document which implementation choice (most likely §1's wavelet substitution) plausibly explains the
gap, but still proceed to D2 — a partial reproduction is still sufficient evidence that image-random splitting is what
produces high numbers, which is D1's actual purpose. <0.85 → treat as inconclusive for the "did we reproduce it" question,
but still report D2–D8 since the patient-level/causal/horizon questions stand on their own regardless of whether D1 hits
0.98 exactly.

## 4. D2–D8 — leakage audit, causal reconstruction, early-warning horizons

Run on this project's own canonical 547-patient cohort/folds from here on (§2 caveat), reusing D1's exact CWT+CNN recipe
unchanged except for the axis each stage tests:

- **D2 (patient-level split):** same 6-images-per-recording construction, but assign whole patients to folds
  (`data/processed_clinical/folds.json`, this project's canonical patient-grouped assignment) before generating images, so
  no patient's images cross the train/test boundary. Compare against D1.
- **D3 (causal):** the DeepFHR image is one static 20-minute segment per patient — it has no notion of "at time t" to begin
  with. To make it causal, redefine the segment as *the last 20 minutes available up to a query prefix length*, using this
  project's existing rolling-window infrastructure (`results/phase8_rolling/`, already certified causal by the synthetic
  future-perturbation test, Phase 8) instead of a single fixed end-of-recording segment. This produces one causal image
  per rolling window (2.5-min stride), not one per patient.
- **D4–D6 (horizons ≥10/20/30 min):** score each patient's causal image sequence (D3) at the eligible-prefix window for
  each horizon (`eligible_prefix_length`, this project's standing convention). D7 (≥40/60 min) run only if enough patients
  retain usable signal that far out (checked, not assumed).
- **Patient-level aggregation rule, frozen now, not chosen after seeing test performance:** primary = **sustained-alert**
  (patient positive iff ≥2 consecutive causal-window images exceed the fold's training-derived threshold), matching how
  URM's own "ever alerted" operational convention already works, for direct comparability. Max and mean aggregation are
  computed and reported as secondary, frozen at the same time.

## 5. Common metric panel and comparison with URM

Discrimination: AUROC, AUPRC. Classification (at the sustained-alert threshold, chosen the same way URM's is — 80%-target
sensitivity, per-fold training-only): sensitivity, specificity, PPV, NPV, F1, FAR. Early-warning: coverage at ≥10/20/30 min,
median and mean lead time, false alerts/patient. Calibration: Brier, slope, intercept, reliability table. All bootstrapped
over **patients**, not windows/images (B=2000, seed 42, this project's standing convention). URM's own numbers are read
verbatim from the locked, frozen results (`results/phase18_fusion_ablation/`, `results/phase26_clinical_evaluation/`) —
never rerun or adjusted to make the comparison look a particular way. Paired comparison (DeLong at delivery; patient
bootstrap elsewhere) is computed only where both models are scored on the identical patient set (this project's 547/110,
i.e., from D3 onward, not D1/D2 which use DeepFHR's own 552/105 cohort).

## 6. Reporting

The full degradation chain (D1 → D2 → D3 → D4/5/6) reported whatever it shows, with the four scenarios from the user's own
message (A: DeepFHR survives, becomes a real benchmark; B: moderate drop; C: collapse; D: fine at delivery, poor at
horizon) used as the interpretive frame, decided by the numbers rather than assumed. If D1 does not approach 0.98, that is
reported as a partial reproduction with the wavelet-substitution caveat (§1), not claimed as proof of leakage on its own —
D2's patient-level result is what actually establishes the leakage magnitude, D1 only establishes the ceiling being
compared against.

## 6a. Amendment 1 (2026-09-26, after D1 completed, before any D2+ code was written)

D1's purpose was to establish DeepFHR's own leaky ceiling for context; it does not gate whether D2+ can answer the question
that actually matters for the URM comparison ("what does a faithfully-reconstructed DeepFHR-style model score under the
*same* evaluation discipline URM was built and locked under?"). That question is answered by combining D2 (patient-level
split), D3 (causal rolling-window images, replacing D1's single end-of-recording segment) and D4–D6 (horizon scoring) into
**one run**, rather than as separate stages, since D3's causal image construction is needed before D2's patient-grouped
training is meaningful for a horizon-based comparison at all. Frozen now, before any code:

- **Cohort/folds**: this project's own canonical 547 patients / 110 positive (pH ≤ 7.15), canonical `folds.json` 5-fold
  assignment — the exact patients and folds URM was evaluated on, not DeepFHR's own 552/105.
- **Images**: one CWT image set (6 images: 2 wavelets × 3 scales, §0) per **rolling window** (this project's existing
  20-min/2.5-min-stride windows, `results/phase8_rolling/rolling_predictions.csv`, already certified causal by the
  Phase 8 synthetic future-perturbation test), not one per patient. Same preprocessing chain, same CNN, same two fixes
  (class-weighted loss, zero-centering) verified in D1.
- **Training**: pooled across all training-fold patients' window-images (image is still the training unit, as in the
  paper's own design — no per-patient re-weighting added, to keep this a direct extension of D1, not a new model).
- **Window score** = mean predicted probability across a window's 6 image variants (a fixed, undebated aggregation rule,
  not chosen after seeing results).
- **Patient/horizon score**: `eligible_prefix_length(t_del, h, T)` selects the eligible causal prefix, exactly as URM's
  own evaluation does; patient-level AUROC at h ∈ {0, 10, 20, 30} min, canonical CV (pooled out-of-fold across the 5
  folds).
- **Comparison**: URM's own scores are the exact, gate-verified sequence already used in Phase 26
  (`scripts/phase26_clinical_evaluation.py`'s `build_urm_sequences()`, reproduces 0.7335/0.6921/0.6638/0.6631 to 4
  decimals) — never rerun or adjusted. Paired patient bootstrap (B=2000, seed 42) on ΔAUROC per horizon, this project's
  standing convention.
- **Secondary, no decision weight**: the sustained-alert operational metric (≥2 consecutive causal windows over a
  per-fold training-only threshold), for continuity with URM's own operational reporting.

This is the metric that settles the URM-vs-DeepFHR question: same patients, same folds, same causal horizon rule, same
statistical convention on both sides. Whatever DeepFHR's own literal ceiling number turns out to be (D1, still
inconclusive) does not change what is measured here.

## 7. Out of scope for this phase

Modifying DeepFHR's architecture or hyperparameters to improve it; any change to the locked URM; claims about the paper's
authors' intent (the framing note in `scripts/audit_evaluation_protocol.py` applies equally here: this is a statement about
what the benchmark permits under the described protocol, not an accusation).
