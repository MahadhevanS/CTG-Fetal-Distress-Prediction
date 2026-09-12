# Phase 13–16 Closure Summary: The Post-Lock Investigation

Written 2026-09-12. This is the single, authoritative consolidation of
everything undertaken after the Phase 12.1 lock — four phases, four
independent methodological approaches (fixed aggregation, stricter
replication, broadened representation analysis, trainable modeling), all
converging on the same modest, unconfirmed signal. It supersedes nothing —
each phase's own report remains the detailed record — but it is the
document to read first.

> **Governance, unchanged across all four phases.** Phase 12.1 (547
> patients, pH ≤ 7.15 primary / ≤ 7.05 severe endpoint, delivery AUROC
> 0.6872, ≥30m AUROC 0.5857 under the original horizon convention) is
> locked and untouched. P6 was never retrained anywhere in Phases 13–16.
> Every branch below is exploratory. No language in this document claims
> validation, replication, or external confirmation — that requires a
> cohort this investigation never had access to.

## Executive summary

Four phases asked the same underlying question from four different angles:
**does P6 leave predictive information unexploited in how it aggregates a
patient's sequence of window-level risk scores?** Every independent
methodology converged on the same answer: **plausibly yes, concentrated in
peak/magnitude information specifically, with an effect size consistently
in the +0.03 to +0.04 AUROC range at delivery — but this 547-patient cohort
sits right at the edge of what can be statistically confirmed, and one
result (Phase 16 Model 3) is close enough, and well enough verified, to be
worth external-cohort validation.** Three unrelated hypotheses (temporal
recency-weighting, window stride, continuous-severity fusion) were tested
and closed. Nothing here replaces P6.

---

## Phase 13 — Controlled Post-Lock Information Extension

Four pre-registered, hypothesis-driven branches plus one branch that
emerged from exploration and was itself pre-registered mid-phase.

| Branch | Question | Result (delivery / ≥30m, CV) | Status |
|---|---|---|---|
| A — Recency-weighted P90 | Does window recency add value beyond magnitude? | vs. plain P90: +0.0104 (p=.39) / −0.0007 (p=.59) | 🔴 Closed |
| B — Stride (1.0 vs 2.5min) | Does temporal resolution improve discrimination? | −0.0165 (p=.08) / +0.0095 (p=.35); operational trade-off (better lead time, worse FAR) | 🔴 Closed |
| C — Huber fusion | Does continuous severity add info beyond P6? | +0.0000 (p=.98) / −0.0168 (p=.56) — redundant with P6 by construction | 🔴 Closed |
| D — Parity fusion | Does admission-time context add info? | CV +0.0223 (p=.17); **test +0.0651 (p=.025)**; 4/4 robustness checks clean | 🟡 Promising |
| E — Plain P90 pooling | Does fixed aggregation beat single-window selection? | +0.0382 (p=.12) / +0.0288 (p=.18) — pre-registered after emerging from A's exploratory pass | 🟡 Promising |

**Parity's robustness** (`results/phase13/parity/`): bootstrap-seed-stable
(Δ 0.0220–0.0223 across 5 seeds), fold-resplit-stable (Δ +0.0117 to +0.0223,
*all positive* across 4 independent re-splits), no recording-duration
confound (r=−0.049, p=.25), and a conditional analysis showing parity is
not substitutable — removing it from a joint 9-covariate model costs 0.089
AUROC. The best-supported single addition in the whole investigation.

**Hybrid 1 — P90 + Parity**, tested since two branches reached 🟡: raw
held-out test numbers looked dramatic (delivery 0.6497→0.7549, +0.1052,
p=.031) but two verification checks intervened before this was reported as
real. The fusion lambda repeatedly saturated its pre-registered grid
boundary (2.0–3.0, vs. parity-alone's 1.3–1.5) — a textbook overfitting
signature — and a fold-resplit check showed the "does parity add anything
*beyond* P90" comparison **flipping sign in 3 of 4 independent splits**,
while "does the combination beat original P6" stayed stable (+0.026 to
+0.050). **Verdict: exploratory/inconclusive, not promoted** — neither
confirmed nor a definitive failure. Full diagnostic in
`reports/phase13_final_closure_report.md`.

---

## Phase 14 — Stricter Single-Primary-Test Replication

Phase 13's Option E raised the question; Phase 14 answered it under a
protocol designed to remove Phase 13's multi-horizon, multi-look latitude
— one pre-registered primary test, one horizon, a decision rule fixed
before any number was computed.

**Primary test: P90 vs. single-window, delivery, CV.**
`Δ = +0.0307, p = 0.225, 95% CI = [-0.0189, +0.0822]`. Test point estimate
agrees in direction (+0.0125) but is not independently powered.

**Verdict, applied mechanically: not replicated at the pre-registered bar.**
No promotion. The point worth keeping: a materially different, equally
legitimate percentile estimator (`numpy.percentile`, vs. Phase 13's
weighted-quantile function) moved the point estimate from +0.0382 to
+0.0307 and the p-value from .121 to .225 — reported transparently as
evidence the original estimate wasn't highly stable, not reconciled away.
**A stricter, independently-designed test landed on the same conclusion
Phase 13's own caution already reached** — meaningful agreement, not a
contradiction.

---

## Phase 15 — Broadened Temporal Risk Representation Study

Widened the question from "is P90 specifically better" to "what, if
anything, about the window-score sequence carries information." Five
aggregators (single-window, P90, max, mean, median) plus four temporal
descriptors (persistence, longest-run, slope, recent-minus-earlier), full
operational-metric and recording-duration analysis.

| Finding | Result |
|---|---|
| Aggregator ordering | **Max ≥ P90 > Mean > Median**, consistent at both horizons, both CV/test — supports "signal concentrates in a subset of high-risk windows" |
| Only significant results in the study | **Slope and recent-minus-earlier were significantly *worse* than single-window** (delivery p=.012/.016, ≥30m p=.004/.008; test AUROC below 0.5) — magnitude carries signal, trajectory *direction* alone does not |
| AUROC vs. operational utility | **Inverted.** Max: best AUROC, worst lead time (12.5min) and detection (25.5% ≥30min). Median: worst AUROC, best lead time (40min) and detection (72.7%) |
| Recording-duration confound | Every representation's advantage over single-window **shrinks with longer recordings** (r = −0.12 to −0.23, all p<.01) — the apparent benefit concentrates in shorter recordings, not uniformly |
| P90 vs. Max | P90 ≠ softened max for 84% of patients (gap > 0.01) — captures something distinct from pure peak-picking, just not enough to outperform it |

**No representation reached Category 3** (strong internal signal) — Max and
P90 reached Category 2 (hypothesis-generating); Mean, Median, persistence,
and longest-run reached Category 1 (no evidence); slope and
recent-minus-earlier reached Category 1, negative.

---

## Phase 16 — Trainable Patient-Level Temporal Aggregation

Tests whether a small trainable component can learn what fixed aggregation
could only detect, not confirm. Explicit design corrections made before any
code was written: scalar-only inputs (no rich embedding exists in P6 to
attend over), end-to-end fine-tuning ruled out of scope (P6 has no
differentiable path from CTG to score), and Phase 3's own historical
attention-MIL collapse (0.5337 vs. fixed P90's 0.6701) named as the
explicit falsification target.

| Model | Mechanism | Delivery CV Δ (p, CI) | Status |
|---|---|---|---|
| 1 | Single-window (baseline) | reference | — |
| 2 | Magnitude-only attention | +0.0135 (p=.601) | Rediscovered max-pooling exactly (100% argmax agreement, ρ≈1.0) — corroborates Phase 15, adds nothing new |
| 3 | Magnitude + temporal position | **+0.0344 (p=.014, CI=[+.006,+.065])** | **First fully-significant CV result across the entire program**; verified (see below) |
| 4 | Peak-aware fusion (max/p90/mean + Model 3) | +0.0372 vs. SW (p=.053); **+0.0028 vs. Model 3 (p=.760)** | No improvement beyond Model 3 — redundant, same pattern as Huber-vs-P6 |

**Model 3 verification** (`results/phase16/model3_verification.json`),
run specifically because it hit its training epoch cap in 3 of 6 fits:

- Epoch-budget extension (400 vs. 200): **AUROC 0.7216 → 0.7216, identical
  to 4 decimals** — training had converged, the cap wasn't truncating anything.
- Fold-resplit sensitivity: Δ stayed in a **narrow, always-positive band**
  across three independent fold assignments (canonical +0.0344, resplit 11
  +0.0315, resplit 22 +0.0273) — tighter than even the parity resplit check.
  What moved was significance (only canonical clears p<.05), not direction
  or magnitude — a power signature, not the sign-flipping instability that
  closed the P90+Parity hybrid.

Interpretability: Model 3's attended window agrees with pure-magnitude
argmax in only 31.6% of patients (vs. Model 2's 100%) — it learned to use
temporal position alongside magnitude, not just re-derive max.

**Strategy B** (end-to-end fine-tuning of P6 itself): not attempted, out of
scope by design — no differentiable path exists in P6's current
construction to fine-tune through.

---

## Cross-phase synthesis

**The throughline.** Four independent methodologies — fixed statistics
(Phase 13/15), a stricter single-test replication (Phase 14), and a trained
model (Phase 16) — all converge on the same underlying signal: aggregating
across a patient's window sequence, weighted toward peak/magnitude
information, plausibly beats single-window selection by roughly +0.03 to
+0.04 AUROC at delivery. No single test crosses a clean confirmatory bar on
this cohort, but the *convergence across methodologies*, not just repeated
looks at the same statistic, is itself evidence worth taking seriously.

**What consistently failed, across every phase that touched it.** Trend/
direction-only signals (recency-weighting in Phase 13A, slope and
recent-minus-earlier in Phase 15) were never merely null — they were the
*only* statistically significant negative results in the whole program.
Whatever this cohort rewards, it is not "which way is risk trending," it is
"how high did risk get." Fusion of redundant information sources — Huber
fusion (Phase 13C, redundant with P6's own inputs) and peak-aware fusion
(Phase 16 Model 4, redundant with Model 3's own learned representation) —
failed for the identical structural reason each time.

**The one instability pattern that recurred and was caught both times.**
The P90+Parity hybrid (Phase 13) and Model 3's raw training run (Phase 16)
both initially produced strikingly large numbers on the small held-out test
partition, and both times a concrete diagnostic — a saturated hyperparameter
grid in one case, a saturated epoch budget in the other — flagged the
result as needing verification before being trusted. One (the hybrid) did
not survive that verification; the other (Model 3) did. The discipline of
checking rather than reporting raw numbers is what tells them apart.

**AUROC is not the only axis that matters.** Phase 15's finding that Max
has the best discrimination and the *worst* early-warning lead time, while
Median has the opposite profile, is a standing caution against reading any
single AUROC column as "the" answer for anything meant to operate as an
alerting system.

## Master status table

| Item | Phase | Status |
|---|---|---|
| Recency-weighted P90 | 13A | 🔴 Closed |
| Stride (1.0 vs 2.5min) | 13B | 🔴 Closed |
| Huber fusion | 13C | 🔴 Closed |
| Parity fusion | 13D | 🟡 Promising, best-supported single addition |
| Plain P90 pooling | 13E / 14 | 🟡 Promising, not replicated at strict bar |
| P90 + Parity hybrid | 13 Hybrid 1 | ⚪ Exploratory/inconclusive |
| Max/P90/Mean/Median (fixed) | 15 | Max/P90 🟡; Mean/Median 🔴 |
| Persistence/longest-run | 15 | 🔴 No evidence either direction |
| Slope/recent-minus-earlier | 15 | 🔴 Significantly negative |
| Model 2 (magnitude attention) | 16 | 🔴 Rediscovers Max, no improvement |
| **Model 3 (magnitude+position attention)** | **16** | **🟢 Strongest result in the program — worth external validation** |
| Model 4 (peak-aware fusion) | 16 | 🔴 No improvement beyond Model 3 |
| Strategy B (end-to-end fine-tuning) | 16 | Out of scope, not attempted |

## Recommendation

**Model 3** (Phase 16) and **parity fusion** (Phase 13D) are the two
candidates worth carrying forward — into an independent-cohort validation
design, not into production. They test genuinely different, complementary
hypotheses (learned temporal aggregation of CTG-derived risk vs.
independent admission-time context) and neither has been shown to
interact productively with the other yet (that combination was never
cleanly tested — the P90+Parity hybrid tested fixed pooling, not Model 3,
combined with parity). Everything else in this four-phase investigation is
closed with reasonable confidence, for a legible, mechanistically
consistent reason in every case rather than an unexplained null.

**The next decisive step, unchanged from every phase's own conclusion, is
external validation.** This cohort has now been examined by five distinct
analytical passes; further internal re-analysis (new percentiles, new
horizons, new architectures) is not expected to move the answer.

## Reproducibility index

| Phase | Protocol | Key scripts | Key results |
|---|---|---|---|
| 13 | `docs/phase13_protocol.md` | `scripts/phase13_{evaluation_audit,recency_p90,stride,huber_fusion,parity_robustness,p90_pooling,hybrid_p90_parity}.py` | `results/phase13/`, `reports/phase13_{survivor_gate,final_closure}_report.md` |
| 14 | `docs/phase14_protocol.md` | `scripts/phase14_temporal_aggregation.py` | `results/phase14/`, `reports/phase14_temporal_aggregation_report.md` |
| 15 | `docs/phase15_temporal_risk_representation_protocol.md` | `scripts/phase15_temporal_representation_study.py` | `results/phase15/`, `reports/phase15_temporal_risk_representation_report.md` |
| 16 | `docs/phase16_protocol.md` | `scripts/phase16_{temporal_attention_model,model3_verification,model4_peak_aware_fusion}.py`, `src/models/phase16_*.py` | `results/phase16/`, `reports/phase16_trainable_temporal_aggregation_report.md` |

All scripts read only from frozen, already-audited artifacts
(`data/processed_clinical/folds.json`, `results/phase8_rolling/`,
`results/phase13/audit/p6_predictions.npz`); none retrains P6. Fixed seeds
throughout (`42` for bootstrap/DeLong, `torch.manual_seed(42)` for Phase 16)
make every reported number exactly reproducible. Nothing in Phases 13–16
has been committed; all of it is additive, uncommitted work on top of the
locked `final_synthesis_models` branch.
