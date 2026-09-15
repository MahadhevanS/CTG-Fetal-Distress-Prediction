# Phase 18 — Deployable M3 + Parity Fusion
## Frozen Mechanism Specification

**Frozen 2026-09-15.** This is the authoritative, standalone specification
of the deployable fusion mechanism — item 1 of the user-directed next-stage
plan ("fully document and freeze the deployable A3 implementation"). It
supersedes every horizon-specific fusion coefficient set produced earlier
in this project's post-lock investigation (Parity Fusion's per-horizon λ,
the Model 3 + Parity Hybrid's per-horizon β, and Stage 1's own first,
non-deployable 16-model ablation) for the purpose of deployment. Those
earlier results remain valid as *what they were tested for* (retrospective
characterization of each horizon in isolation) but none of them describes
a system that could actually run during labor. This one does.

> **Governance (unchanged).** Phase 12.1 (P6) remains locked and
> authoritative. Model 3's own checkpoints are frozen and unmodified. This
> document specifies a fusion layer built entirely on top of already-frozen
> outputs — it does not retrain, and cannot be promoted to production,
> without a successful external validation study
> (`docs/external_validation_handoff.md`).

---

## 1. The mechanism

```
p_fused(t) = α · p_M3(t) + (1 − α) · p_parity
```

- **`p_M3(t)`** — Model 3's own running score at query time `t`: the pooled
  attention prediction over whatever causal prefix of window scores has
  been observed so far (`predict_all_prefixes`, the same function used for
  Model 3's own operational lead-time simulation). Model 3's checkpoints
  are unmodified from their Phase 16 training.
- **`p_parity`** — the frozen univariate parity model's probability
  (`StandardScaler` + `LogisticRegression(C=1.0)` on raw parity), constant
  for a given patient throughout labor.
- **`α`** — a single scalar, **not a function of horizon, elapsed time, or
  anything else that varies during monitoring.** The system is queried the
  same way at every point in time; only its *retrospective* accuracy is
  later characterized by how much lead time happened to remain before
  delivery.

**This is a probability-space convex combination — no logistic regression,
no standardization, no interaction term.** It was selected over five other
mechanisms tested under the same deployable discipline (F2-style logistic
fusion, F7-style interaction fusion, A4 logit-space fusion, and fixed-α
alternatives) because it was the strongest and most robustly verified of
all of them — not chosen for simplicity's own sake, though the simplicity
is a genuine additional advantage (no scaler to version, no coefficients
that could silently drift).

---

## 2. Training procedure

**α is selected once per fold**, via grid search (`np.linspace(0, 1, 21)`,
step 0.05) maximizing **sample-weighted training AUROC, pooled across
every causal truncation of every training patient's sequence**:

- For each training patient with `T_i` retained windows, every prefix
  length `k = 1..T_i` is a separate training example: `(p_M3 at truncation
  k, p_parity, patient's outcome)`.
- Each example is weighted `1/T_i`, so every patient contributes total
  weight 1 regardless of recording length — identical per-patient
  normalization principle to `docs/phase16_protocol.md` §4's
  `patient_avg_bce_loss` and Phase 13's information-density work.

This mirrors Model 3's own training convention exactly (`docs/phase16_protocol.md`
§4: "every prefix length is a separate training example") — the discipline
that makes Model 3 itself deployable was never applied to any fusion layer
before this one.

**No other free parameters.** `p_parity` and `p_M3(t)` are both frozen
inputs; α is the only quantity this training procedure produces.

---

## 3. Frozen α

| | Value |
|---|---|
| **Deployed value** (fit on all 547 patients, no held-out fold — matches the "final artifact" convention of `models/external_validation_handoff/`) | **α = 0.35** |
| Stability across 30 independent fold-fits (5 canonical folds + 5 independent resplits × 5 folds each) | mean 0.333, median 0.35, std 0.033, range [0.30, 0.40] |

**α ≈ 0.35 means parity is weighted roughly 65%, Model 3 roughly 35%** —
notably more weight on parity than any horizon-specific fusion attempt
gave it (those typically landed M3's coefficient at 2–3× parity's). This
is a real, substantive finding in its own right, not an artifact: pooling
training examples across every causal truncation (including very early,
information-poor prefixes where M3 has little signal) shifts the optimal
blend toward the covariate that is informative from the first window
onward.

The grid search resolution (0.05) means the reported stability is close to
the practical floor of what this method can distinguish — 27 of 30 fits
landed on exactly 0.30 or 0.35, none landed outside [0.30, 0.40]. A finer
grid was not run; there is no evidence a finer search would change the
deployed value.

Full data: `results/phase18_fusion_ablation/deployable_fusion_FROZEN_params.json`.

---

## 4. Verification history (why this is trusted)

This mechanism survived more adversarial scrutiny than any other result in
the project's post-lock investigation before being written down here:

1. The horizon-specific version of this idea (Stage 1's 16-model ablation)
   initially showed 4 significant (model, horizon) pairs. A bootstrap-seed
   check alone would have certified all 4.
2. A fold-resplit check — the check the source ablation plan itself
   demanded — found **3 of those 4 were canonical-partition artifacts.**
3. A full 16-model × 4-horizon × 6-split sweep, built to check
   systematically, caught a real leakage bug in its own first draft (a
   parity model reused across resplits instead of refit fresh) before any
   conclusion was drawn — found by a direct numerical contradiction against
   the earlier single-pair result.
4. The user identified that the entire horizon-specific framing — not just
   individual results within it — was undeployable: minutes-before-delivery
   is unknown during actual labor, so a system cannot select coefficients
   by horizon. This produced the mechanism in §1–2.
5. The deployable version's own fold-resplit stability check caught the
   *same* leakage-bug pattern a second time, independently, in a newly
   written script, before its output was trusted.
6. This document's operational-metric and subgroup scripts each re-fit the
   canonical folds fresh and cross-checked against the already-verified CV
   numbers before computing anything further — both checks passed exactly
   (0 mismatches).

Full narrative: `results/phase18_fusion_ablation/stage1_report.md`.

---

## 5. Performance

### 5.1 Discrimination — sharply horizon-dependent, and the pattern itself is informative

| Horizon | CV AUROC | Δ vs. M3-only | CV significance (canonical) | Fold-resplit check (6 independent splits) | Test AUROC | Test Δ |
|---|---|---|---|---|---|---|
| Delivery | 0.7335 | +0.0119 | p=.468 (n.s.) | **0/6 significant** | 0.7433 | +0.0704 (p=.004) |
| ≥10m | 0.6921 | +0.0238 | p=.231 (n.s.) | **0/6 significant** | 0.6988 | +0.1346 (p<.001) |
| ≥20m | 0.6638 | **+0.0623** | **p=.001** | **6/6 significant**, worst-case p=.002 | 0.7558 | +0.1355 (p<.001) |
| ≥30m | 0.6631 | **+0.0654** | **p<.001** | **6/6 significant**, worst-case p=.003 | 0.7754 | +0.1096 (p<.001) |

**Do not claim a confirmed delivery-horizon benefit.** Report near-delivery
performance as directionally positive (consistently so on the internal
test partition) but not established on CV/resplit evidence. Report ≥20m/
≥30m as this project's most robustly confirmed early-warning result.

### 5.2 Operational behavior (80% target sensitivity, per-fold training-only threshold, patient-level "ever alerted")

| System | Achieved Sens | FAR | Median Lead | ≥20m Detected | ≥30m Detected |
|---|---|---|---|---|---|
| M3-only | 88.2% | 69.3% | 32.5 min | 66.0% | 54.6% |
| **A3 (this mechanism)** | 86.4% | **62.5%** | 40.0 min | **81.0%** | **69.5%** |
| A4 sum-to-one (reference) | 85.5% | 63.4% | 40.0 min | 84.0% | 71.3% |
| F2 deployable (reference) | 86.4% | 66.4% | 40.0 min | 75.8% | 66.3% |
| A3 fixed α=0.75 (reference) | 87.3% | 65.0% | 35.0 min | 66.7% | 56.2% |

At a comparable achieved sensitivity to M3-only, this mechanism lowers the
false-alert rate by ~7 points and extends median lead time by 7.5 minutes,
with substantially higher ≥20m/≥30m detection rates. A4 sum-to-one is a
close operational competitor (marginally better detection, marginally
worse FAR) — both are legitimate candidates for external validation; this
document specifies A3 as the primary recommendation because of its
stronger, more completely resplit-verified CV/test discrimination result
(§5.1) and its structural simplicity (§1).

Full table: `results/phase18_fusion_ablation/deployable_fusion_5way_comparison.csv`,
`deployable_fusion_operational_metrics.csv`.

---

## 6. The central hypothesis, tested directly

> *"Parity acts primarily as a maternal-context modifier that improves the
> reliability of earlier CTG-based risk estimation, rather than providing
> a uniformly useful additive contribution at every prediction horizon."*

**Supported, in a more specific form than originally stated — the
mechanism differs by horizon.**

### 6.1 The horizon-dependent half — already established (§5.1)

The fusion gain grows monotonically as lead time increases: +0.012
(delivery) → +0.024 (≥10m) → +0.062 (≥20m) → +0.065 (≥30m). Parity adds
essentially nothing when CTG evidence is mature (delivery) and a large,
robustly confirmed amount when CTG evidence is least mature (≥30m before
delivery, near the start of the observable window). This is the most
direct evidence for "earlier CTG-based risk estimation" being where
parity helps.

### 6.2 The risk-stratification half — tested for the first time here, and the answer is horizon-dependent

Patients were split into Low/Mid/High M3-only-risk tertiles using
**per-fold, training-derived cutoffs** (no leakage), separately at
delivery and ≥20m:

| Horizon | Low M3-risk | Mid M3-risk | High M3-risk |
|---|---|---|---|
| **Delivery** | AUROC Δ +0.064 | **AUROC Δ +0.100 (largest)** | AUROC Δ **−0.008 (~none)** |
| ≥20m | AUROC Δ +0.086 (largest) | AUROC Δ +0.079 | AUROC Δ +0.044 (smallest, still positive) |

A continuous check (Spearman correlation between the magnitude of the
fusion-induced score shift and how close M3's own score sits to 0.5 — an
"ambiguity" proxy) confirms this pattern with p-values, not just a
3-bin table: **at delivery, r=+0.30 (p<.0001)** — the more ambiguous M3's
own read, the larger the fusion shift. **At ≥20m, r=−0.07 (p=.10)** — no
such concentration.

**The honest, combined reading:**

- **At delivery — the hypothesis is sharply confirmed.** Parity functions
  almost exactly as an ambiguity-resolving modifier: it does essentially
  nothing when Model 3 is already confident (High tertile, Δ≈0) and its
  entire measurable effect concentrates in the Mid tertile and the
  continuous ambiguity correlation. This is a clean, interpretable,
  clinically sensible picture — but recall from §5.1 that the *overall*
  delivery-horizon gain is itself unconfirmed by the resplit check, so
  this concentration pattern describes where a small, not-yet-significant
  effect sits, not a large confirmed one.
- **At ≥20m — the hypothesis's "primarily intermediate-risk" framing does
  not hold as cleanly.** The gain is real and large everywhere (§5.1), and
  present in all three risk strata, not concentrated in the middle one —
  smallest (though still clearly positive, +0.044) in the High-risk
  tertile, largest in Low. At this horizon parity reads more like a
  broadly complementary information source than a narrowly-targeted
  disambiguator.

**Revised statement of the mechanism, supported by this data:** parity's
value shifts character with how mature the CTG evidence is. When CTG
evidence is mature and usually decisive (delivery), parity's residual
value is small and specifically fills in where CTG is ambiguous. When CTG
evidence is inherently less mature (≥20–30m before delivery), parity
provides substantial complementary information broadly across the risk
spectrum, not narrowly targeted at ambiguous cases — consistent with
parity being a stable signal available from the first window, functioning
as a partial substitute for CTG evidence that has not yet accumulated,
rather than only as a tie-breaker for borderline CTG reads.

Full data: `results/phase18_fusion_ablation/deployable_m3_risk_tertile_analysis.csv`,
`deployable_ambiguity_correlation.csv`.

---

## 7. Subgroup and residual findings (secondary, but not to be discarded)

- **At delivery, classification-level agreement is roughly a wash, not an
  improvement.** At the 80%-target-sensitivity operating point, fusion
  corrects 59 previously-misclassified patients and harms 63 previously-
  correctly-classified ones (net −4; positives: 7 corrected vs. 7 harmed).
  This is consistent with §5.1's finding that delivery discrimination is
  not yet confirmed — even at a fixed operating point, there is no clear
  classification-level win at this horizon. (`deployable_error_disagreement.json`)
- **The shared operating threshold behaves very differently across parity
  groups — a real calibration consideration, not yet resolved.** At
  delivery, fusion pushes sensitivity *up* and FAR *up* for nulliparous
  patients (parity=0: sens .813→.89, FAR .525→.723) while pushing both
  sharply *down* for higher-parity patients (parity=1: sens .786→.429, FAR
  .395→.048; parity=2: sens .6→.2, FAR .333→.042) — even though within-group
  AUROC is roughly unchanged in every group. Parity is correctly
  recalibrating baseline risk in the direction the earlier Phase 13
  covariate screen established (lower parity → higher risk), but a single
  global threshold applied on top of that recalibration produces starkly
  different classification behavior by parity group. **This should be
  treated as an open fairness/calibration question for external
  validation, not swept into the headline AUROC result.**
  (`deployable_parity_subgroup_analysis.csv`)

---

## 8. Status and next steps

**Recommended for external validation, alongside Model 3 and Parity
Fusion** — this document is the authoritative specification to carry
forward, superseding every earlier fusion coefficient set for deployment
purposes. Before an external-validation submission:

1. Produce frozen, portable artifacts analogous to
   `models/external_validation_handoff/` for this specific mechanism
   (currently only Model 3, P6, and Parity Fusion have portable frozen
   artifacts there) — trivial given §3's frozen α and the already-frozen
   parity model.
2. Resolve the parity-group threshold-calibration question (§7) before
   any single global operating point is recommended for external use.
3. Decide whether A4 sum-to-one (§5.2's close operational competitor)
   should be carried forward as a second candidate alongside A3, given how
   close the two are operationally.

**Stage 2 (representation-level fusion using richer M3 temporal features —
window-level statistics, sequence encoders, attention-weight features)
remains explicitly deferred**, per the user's own prioritization: this
document's results (a small number of well-understood, thoroughly-verified
levers — α, parity encoding, risk-stratified behavior) had not yet been
exhausted, and Stage 2's added model complexity is not justified until they
were. It still is not, pending the items in this section.
