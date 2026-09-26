# Phase 19 — Routes 1 and 2: results

Protocol: `docs/phase19_adaptive_pooling_protocol.md` (frozen before any arm code). Script:
`scripts/phase19_pooler_experiment.py`. Data: `results/phase19_pooler/`.

## Verdict (pre-registered rule, applied mechanically): all three arms NOT SUPPORTED

| Arm | Change vs A0 (TAM control) | Canonical dM (p) | Resplits dM>0 | Tier |
|---|---|---|---|---|
| A1 | + 4 reliability cues | -0.0112 (.118) | 1 of 5 | NOT SUPPORTED |
| A2 | horizon-grid ranking loss | -0.0004 (.811) | 1 of 5 | NOT SUPPORTED |
| A3 | both | -0.0084 (.218) | 1 of 5 | NOT SUPPORTED |

Gates passed first: vectorized pooling equals frozen TAM inference to 1.2e-7; A0 (seed 42) reproduces
frozen TAM at all four horizons within 0.0004 (gate tolerance 0.006).

## Canonical CV AUROC (seed-ensembled; N=547 every horizon)

| Model | Delivery | 10m | 20m | 30m | Mean over horizons (M) |
|---|---|---|---|---|---|
| PRS single window | 0.6872 | 0.6859 | 0.6238 | 0.5828 | 0.6449 |
| A0 (= TAM) | 0.7214 | 0.6684 | 0.6014 | 0.5974 | 0.6472 |
| A1 | 0.7007 | 0.6449 | 0.6034 | 0.5946 | 0.6359 |
| A2 | 0.7185 | 0.6708 | 0.6017 | 0.5958 | 0.6467 |
| A3 | 0.7068 | 0.6487 | 0.6044 | 0.5949 | 0.6387 |

## What this shows

1. **Route 2 (ranking objective) changes nothing.** A2 matches A0 within 0.003 at every horizon and in every
   resplit (dM between -0.0017 and +0.0013). The objective mismatch I saw in the 2-parameter family is not what
   limits TAM; it converges to the same weighting either way. A0 is also essentially seed-independent
   (seed spread < 0.0003), so seed ensembling had nothing to reduce.
2. **Route 1 (reliability cues) does not generalize.** A1/A3 stop after ~50 epochs on average (min 11) versus
   ~125–170 for A0/A2 — the extra inputs fit noise in a 110-event cohort. They lose ~0.02 at delivery and 10m
   canonically (resplit mean at 10m: -0.015) and gain at most +0.005 at 20m. One resplit (55) is positive (+0.008),
   so the size of the loss is partly split-dependent; the sign is not favorable.
3. **The larger finding is about TAM itself.** Averaged over horizons, TAM is indistinguishable from the
   single-window PRS (dM +0.0022, p=.79): its +0.034 at delivery is offset by -0.017 at 10m and -0.022 at 20m.
   The pooling layer trades one lead time for another.
4. **Little headroom in the pooler.** An oracle that knew the horizon and picked the best (lambda, kappa) per
   horizon from the recency x risk family (selected on the evaluation folds, so optimistic) reaches M ~ 0.663,
   only ~+0.016 above A0. Pooling PRS window scores cannot move discrimination much; the information going in
   is the limit.

## Consequence for the optimisation plan

Stop tuning the pooler. The large, verified gains at >=20m/30m came from parity (URM), not from TAM's weighting.
Most promising next questions, in order:
1. **Does URM need TAM at all?** Repeat the deployable fusion (Phase 18 procedure, alpha selected once) with the
   running PRS window score in place of TAM. Given TAM ~= PRS at 10-30m this may match or beat URM and would
   simplify the system. Cheap; uses existing machinery.
2. **Improve early-lead-time information** (window AUROC is only 0.57–0.61 at 20–30 min before delivery):
   new window-level evidence, not a new pooler.
3. Route 3 (parity inside the attention) is now less attractive: the pooler has little headroom, and parity is
   already fused at the score level.

## Deviations / notes (none affect results)
- The first resplit launch did not terminate when expected, so two identical deterministic runs appended to the
  training log; 240 duplicate rows were removed (300 = 4 arms x 3 seeds x 25 folds). Result CSVs had no duplicates.
- Resplit inner-validation seeds use 42+fold+100+resplit_seed (the Model 3 verification precedent).
- Epoch cap (200) was hit in 7/15 (A0) and 9/15 (A2) canonical trainings, 3/3 test models each — as with
  TAM originally (extending to 400 epochs changed nothing there).
