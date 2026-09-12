# Phase 13 — Independent Branches Complete: Survivor Gate

Written 2026-09-11, per `docs/phase13_protocol.md` Section 9 ("Step 8" in the
implementation sequence). All four independent branches (A–D) plus the
evaluation audit (13.0A) are complete. **No hybrid has been run** — per the
frozen protocol, hybrids require at least two survivors to combine, and this
round produced at most one. This report is the checkpoint the protocol
requires before deciding whether/how to proceed to hybrids.

---

## 13.0A — Evaluation audit

- Canonical P6 reproduced bit-for-bit against the independently-computed
  `results/phase13_information_density/step_predictions.csv` (max
  difference across 8,517 windows: 0.000000).
- Baseline reconciliation confirmed: `0.6872` = P6 unweighted (reference for
  Options A/C/D), `0.6843` = P6 with patient-normalized weighting (stride's
  baseline arm only). Both legitimate, now documented so they never again
  read as contradictory numbers.
- **The corrected-horizon convention produces materially different numbers
  from the existing one at every horizon except delivery** — e.g. at h=10m,
  CV jumps 0.5814→0.6859 under the corrected convention while test *drops*
  0.6800→0.5802. This confirms fixing horizon selection first, before
  evaluating any branch, was the right call — every ≥10/20/30m number from
  earlier this session was under the old (first-eligible-window) convention.
  Full table: `results/phase13/audit/horizon_alignment_comparison.csv`.

## Branch results (confirmatory horizons only; full tables per branch)

| Branch | Confirmatory test | Delivery (CV / test, p) | ≥30m (CV / test, p) | Status |
|---|---|---|---|---|
| A — Recency-weighted P90 | recency P90 − plain P90 | +0.0104 / −0.0143 (p=.39/.80) | −0.0007 / +0.0018 (p=.59/.73) | 🔴 **Closed** |
| B — Stride | P6(1.0min) − P6(2.5min) | −0.0165 / — (p=.08) | +0.0095 / — (p=.35) | 🔴 **Closed** |
| C — Huber fusion | P6+Huber − P6 | +0.0000 / −0.0053 (p=.98/.55) | −0.0168 / +0.0009 (p=.56/1.0) | 🔴 **Closed** |
| D — Parity fusion | P6+Parity − P6 | +0.0223 / **+0.0651** (p=.17/**.025**) | +0.0494 / +0.0784 (p=.08/.17) | 🟡 **Promising** |

Full per-branch tables: `results/phase13/{recency_p90,stride,huber,parity}/`.
Master table: `results/phase13/statistics/master_survivor_table.csv`.

### A — Recency-weighted P90: closed

The literal, previously-untested hypothesis from the original discussion.
Built as a genuinely parallel aggregation pathway (not compared against
P6's single-window score, to avoid confounding aggregation with horizon
selection). Half-lives (5/10/20 min, ∞ control) selected per fold via
train-fold-only AUROC maximization. Result: no horizon shows a consistent,
significant gain from recency-weighting over plain P90 — CV and test even
disagree in sign at delivery and h=10. **Recency does not add information
beyond magnitude.**

### B — Stride: closed

Re-scored the existing 1.0min/2.5min pipeline outputs under both horizon
conventions plus operational metrics. AUROC: no horizon significant either
convention. Operational metrics show a genuine trade-off, not a clean win:
1.0min gets better median lead time (40.0 vs 32.5 min) and ≥20/30min
detection (77.3%/63.6% vs 67.3%/56.4%), but at a materially worse false
alert rate (92.9% vs already-high 84.7%). Given this project's own stated
motivation (>60% false-alarm rates in current clinical practice as the
problem to solve), that trade isn't a defensible "operational improvement."

### C — Huber fusion: closed, as the a priori reasoning anticipated

`risk_prob_proxy` confirmed via direct code inspection: window-causal,
frozen out-of-fold Huber model, no pH in the inference path, row-aligned to
P6's own windows. Aggregated via the identical horizon-selection function as
P6 (protocol C4). Confirmatory result: null at both delivery and ≥30m,
matching the a priori expectation (`X_19 ⊂ P6`, so Huber's inputs are already
inside P6). One exploratory note: at h=10/20 specifically, **Huber alone**
(not fused) outperforms **P6 alone** under the corrected convention (e.g.
h=20 test: 0.731 vs 0.612) — but the additive logit fusion doesn't capture
this reliably (fused sometimes underperforms both components), and this
wasn't a pre-registered comparison. Worth a look in a future round, not
enough to reopen Option C now.

### D — Parity fusion: promising, and now much better supported

Test-set significant at delivery (p=0.025, CI entirely positive
[+0.010,+0.124]); CV positive but CI crosses zero (p=0.172) — same pattern
as before. What's new this round is the robustness audit
(`results/phase13/parity/`):

- **Bootstrap-seed stable**: delta 0.0220–0.0223 across 5 seeds, p 0.156–0.172.
- **Fold-assignment stable**: delta +0.0117 to +0.0223 across 4 independently-generated 5-fold splits, *all positive* — not an artifact of the canonical partition.
- **No recording-duration confound**: r(parity, windows-per-patient) = −0.049, p=0.25.
- **Conditional analysis**: removing parity from the joint 9-covariate model drops AUROC by 0.089 — parity is doing genuine, non-substitutable work, not standing in for another covariate.
- **Permutation control** (prior round): 0/30 shuffled-parity permutations exceeded the true delta.

Classified 🟡 rather than 🟢 by strict, uniform application of the protocol's
own gate (CV CI crosses zero) — the same bar every other branch was held to.
It is, by a wide margin, the best-supported result in the program.

## Survivor gate outcome

Only **one** branch (D) reached 🟡/eligible status. A, B, C are cleanly
closed. Per protocol Section 10, hybrids require at least two eligible
components — **the pre-authorized `P6+Huber+Parity` hybrid is not
justified**, since Huber did not independently survive. No other pairing is
possible this round. **No hybrid should be run yet.**

## One unplanned finding worth flagging above the others

Building Option A's parallel P90 pathway surfaced something not in any
original hypothesis: **plain P90 pooling of P6's own window-level scores
(no recency weighting at all) outperforms P6's own single-window production
score** — +0.0382 CV / +0.0419 test at delivery, +0.0288 CV / +0.0936 test at
≥30m. This is the largest point estimate anywhere in the program. It is
**not statistically significant** (CV CI [-0.009,+0.088] at delivery, p=0.12;
wider and also non-significant at ≥30m) and **not consistent across all
horizons** (h=10 CV goes the other way, p=0.068). It was also not
pre-registered as a confirmatory test — Section 11 only registered
recency-vs-plain as Option A's test, not plain-P90-vs-P6.

This is a different question from all four original branches: it's not
about adding new information, it's about **how P6 itself pools evidence
across a patient's windows** — currently single-window horizon-selection,
historically (Phase 3) P90 pooling, apparently still competitive or better
even against the richer, trajectory-engineered P6 representation. Worth
pre-registering as its own confirmatory hypothesis in a follow-up round
rather than being treated as a footnote to Option A's (closed) recency
question.

## Recommendation

1. Don't run hybrids this round — no pairing is justified.
2. Parity stands on its own as the strongest surviving candidate; it's
   already been through more validation than anything else in this
   project's post-lock history short of the original locked phases.
3. The P90-pooling-vs-single-window finding deserves a proper, pre-registered
   confirmatory test of its own before it's acted on — it's the most
   promising unexplored thread this round produced, but it needs the same
   discipline every other branch here just went through, not a shortcut
   because the point estimate looks good.

## Artifacts

`docs/phase13_protocol.md` (frozen protocol) · `src/evaluation/phase13_common.py`,
`src/aggregation/recency_weighted_p90.py` (shared infra) ·
`scripts/phase13_evaluation_audit.py`, `phase13_recency_p90.py`, `phase13_stride.py`,
`phase13_huber_fusion.py`, `phase13_parity_robustness.py` ·
`results/phase13/{audit,recency_p90,stride,huber,parity,statistics}/`.
All uncommitted, per the governance rule in `docs/phase13_protocol.md` — the
Phase 12.1 lock is untouched.
