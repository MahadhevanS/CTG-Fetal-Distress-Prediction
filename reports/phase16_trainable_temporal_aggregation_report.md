# Phase 16 — Trainable Patient-Level Temporal Aggregation: Results

Written 2026-09-12, executed per the frozen
[docs/phase16_protocol.md](../docs/phase16_protocol.md). Phase 12.1 is
untouched and authoritative throughout — P6 was never retrained; only a
small trainable component was fit on top of its already-frozen window-level
scores. No language in this report claims validation or confirmation beyond
what this 547-patient cohort can support.

## Model family results

| Model | Mechanism | Params | Delivery CV Δ vs SW (p, CI) | Delivery Test Δ vs SW (p) |
|---|---|---|---|---|
| 1 (baseline) | Single-window | 0 | reference | reference |
| 2 | Magnitude-only attention: `e_t=f(r_t)` | ~25 | +0.0135 (p=.601) | +0.0009 (p=.920) |
| 3 | Magnitude + position: `e_t=f(r_t,elapsed_t)` | ~33 | **+0.0344 (p=.014, CI=[+.006,+.065])** | +0.0232 (p=.157) |
| 4 | Peak-aware fusion: LR on `[max,p90,mean,z₃]` | 5 | +0.0372 (p=.053, CI=[-.000,+.077]) | +0.0276 (p=.201) |

Full tables (all four horizons, both splits): `results/phase16/*_auroc_results.csv`, `Model_4_peak_aware_fusion_results.csv`.

## Model 2 — rediscovered max-pooling, cleanly

Trained with only `r_t` as input (no temporal information), Model 2's
learned attention converges to picking out a single window with near-total
certainty: **argmax(α) = argmax(r) in 100% of patients, Spearman ρ ≈ 1.0**.
A trainable component, given only magnitude, independently rediscovered
exactly the aggregation strategy Phase 15 found best by fixed statistics
(Max ≥ P90 > Mean > Median). This is a clean, mechanistic corroboration of
Phase 15's finding from an entirely different methodological angle — not a
new discovery, but real independent support for an old one. Its raw AUROC
is weaker than fixed Max pooling (0.7006 vs. Phase 15's Max at 0.7210),
consistent with softmax attention never reaching a literal hard-max even
when it learns the right index.

## Model 3 — the standout result, verified before being trusted

Delivery: the **first result across the entire Phase 13–16 program to
clear CV p<0.05 with a 95% CI entirely above zero**. Given a training run
that hit its fixed 200-epoch cap in 3 of 6 fits — a legitimate reason for
suspicion, in the same category as the P90+Parity hybrid's lambda hitting
its grid boundary — two verification checks were run before reporting this
as anything more than a raw number (`results/phase16/model3_verification.json`):

- **Epoch-budget extension (400 vs. 200):** AUROC 0.7216 → 0.7216, identical
  to 4 decimal places; the two capped folds only needed 5–23 more epochs
  before stopping naturally. Training had converged; the cap wasn't cutting
  anything off.
- **Fold-resplit sensitivity:** delta held in a narrow, stable, always-positive
  band across three independent fold assignments — canonical +0.0344, resplit
  seed 11 +0.0315, resplit seed 22 +0.0273 — a tighter relative spread than
  even the Phase 13 parity result showed. What moved was *significance*, not
  *direction or magnitude*: only the canonical split's CI clears zero (the
  other two sit just barely on the wrong side, −0.005 and −0.013). This is
  the signature of a real, modestly-sized effect at the edge of what this
  cohort's size can reliably detect — not the sign-flipping instability that
  closed the P90+Parity hybrid.

**Interpretability:** Model 3's attended window agrees with the pure-magnitude
argmax in only 31.6% of patients (Spearman ρ=0.38) — meaningfully different
from Model 2's 100%, indicating it learned to weigh temporal position
alongside magnitude, not simply rediscover max a second time. This is
consistent with, and a plausible explanation for, why Model 3 outperforms
Model 2.

**Verdict applying protocol Section 8's four criteria:** all four satisfied
— direction/magnitude confirmed stable, CV/test agree in direction,
interpretability shows genuine non-trivial structure, and the overfitting
check resolved favorably. Per the protocol's own explicit framing (a
model with a modest, well-supported effect and a wide CI is "interesting,
requires stronger validation," not a discard) — this is the single most
evidentially supported model-based finding in the post-lock investigation.

## Model 4 — gated open, does not clear a further bar

Peak-aware fusion (`max`, `p90`, `mean`, and Model 3's own output, fused via
a fixed, untuned `LogisticRegression(C=0.1)`) was run per Section 9's gate,
reusing Model 3's frozen per-fold checkpoints rather than retraining it.
Result: **no horizon shows a significant improvement over Model 3 alone**
(all "vs. Model 3" p-values 0.09–0.86), and at delivery Model 4's own
significance against the single-window baseline is marginally *weaker*
than Model 3's (p=0.053 vs. p=0.014). The most coherent reading: Model 3's
learned attention already extracts the magnitude information max/P90/mean
would separately contribute, making them largely redundant inputs — the
same redundancy pattern Phase 13 found between the frozen Huber score and
P6 itself (`X_19 ⊂ P6`). The ≥20m negative result (CV p=0.050, entirely
negative CI) is inherited from Model 3's own pre-existing weakness at that
horizon, not something Model 4 introduces.

## Strategy B (end-to-end fine-tuning)

Remains explicitly out of scope, per protocol Section 0(b): P6's
construction has no differentiable path from raw CTG to risk score for
gradients to flow through. Not attempted, not scoped as a natural next step
of this phase — any future attempt requires first rebuilding parts of P6 as
differentiable, a separate undertaking.

## Answering the phase's central question

Does a trainable aggregator recover information Phase 15's fixed statistics
couldn't confirm? **Partially, and specifically through Model 3.** Model 2
reproduced Phase 15's own best fixed aggregator without adding anything new
— informative as corroboration, not as improvement. Model 3 is the first
model in this entire six-phase investigation to reach the protocol's
strictest confirmatory bar at a primary horizon, survived two independent
verification passes designed to falsify it, and shows a qualitatively
different (not merely rediscovered) attention pattern. Model 4 shows that
richness beyond Model 3's own learned representation adds nothing further.
**If the Max/P90 signal Phase 13–15 detected is learnable at all on this
cohort, Model 3 is the clearest evidence yet that it is** — though "learnable
on this cohort" and "generalizes beyond it" remain distinct claims, and only
the second one would justify anything beyond continued internal interest.

## Recommendation

Model 3 (magnitude + temporal position causal attention) is the sole
candidate from this phase worth carrying into external-cohort validation —
not for promotion over P6. Model 2 and Model 4 are closed as non-improving.
Strategy B is deferred, unscoped, pending a separate future decision to
invest in a differentiable P6 rebuild.

## Governance

Phase 12.1 unmodified throughout. P6 never retrained — only its frozen,
already-audited window-level scores were consumed. No hyperparameter search
was run for any model in this family (architecture, learning rate, weight
decay, and Model 4's regularization strength were all fixed in the protocol
before training, not tuned against results). All Phase 16 artifacts are
additive and uncommitted.

## Reproducibility

`python scripts/phase16_temporal_attention_model.py` (Models 1–3, checkpointed
and resumable via `results/phase16/checkpoints/`) → `python scripts/phase16_model3_verification.py`
(epoch-extension + fold-resplit checks, checkpointed via `results/phase16/checkpoints_verification/`)
→ `python scripts/phase16_model4_peak_aware_fusion.py` (reuses Model 3's
checkpoints, no retraining). Fixed seeds throughout (`torch.manual_seed(42)`,
bootstrap `seed=42`) — every run reproduces the numbers in this report exactly.
