# Phase 19b — Does the deployable fusion need TAM?
## Pre-Registration Protocol

> **Status: executed 2026-09-21 — verdict INCONCLUSIVE** (canonical ΔM −0.0065,
> p = .26; TAM's advantage is concentrated at delivery-time queries). Results:
> `results/phase19b_fusion/phase19b_report.md`. Protocol below unchanged.

**Frozen 2026-09-21, before any code for this check was written or run.**

> **Governance (unchanged).** PRS is locked; TAM's checkpoints and the parity
> model are frozen. Nothing here is promoted; results reported regardless.

## 1. Question and motivation

Phase 19 found TAM is indistinguishable from single-window PRS when averaged
over horizons (ΔM +0.002, p=.79) and is *below* PRS at 10m/20m. The deployable
Unified Risk Model (URM, `docs/phase18_deployable_fusion_protocol.md`) fuses
TAM's running score with parity. **Does URM's gain come from TAM, or from
parity alone?** If the plain latest PRS window works as well, TAM can be
removed from the deployed system.

## 2. Arms (identical fusion procedure; only the CTG score sequence differs)

`p_fused(t) = α · s(t) + (1 − α) · p_parity`, α selected once per fold by
maximizing sample-weighted (1/T_i) training AUROC over every causal truncation
of every training patient (grid step 0.05), exactly the Phase 18 procedure.

| Arm | `s(t)` |
|---|---|
| **U_TAM** (URM, control) | TAM's running score at the query prefix |
| **U_PRS** | the latest PRS window score at the query prefix |

References (no fitting): TAM alone, PRS single window, parity alone.

## 3. Procedure

- One shared per-split function fits the parity model fresh on that split's
  training patients (in-sample for training rows, out-of-fold for held-out
  rows) and both arms' α; **both arms use the same function**, so no convention
  can favor either. TAM's per-patient running scores stay frozen (canonical
  checkpoints), as in every prior resplit check.
- Splits: canonical 5-fold CV, plus 5 independent resplits (StratifiedKFold
  seeds 11/22/33/44/55); held-out internal test partition once (train+val → test).
- Horizons 0/10/20/30 min via the eligible-prefix rule; N=547 CV / 83 test at
  every horizon. **Primary metric M** = mean over horizons of patient AUROC.
- Statistics: paired patient bootstrap (B=2000, seed 42), U_PRS − U_TAM, on ΔM
  and per-horizon Δ.

## 4. Gate (before any comparison is read)

U_TAM canonical CV AUROCs must reproduce the Phase 18 stability script's
CANONICAL rows for `A3_selected_once`
(`results/phase18_fusion_ablation/deployable_fusion_resplit_sensitivity.csv`)
within 0.001 at each horizon; PRS-alone canonical AUROCs must reproduce
0.6872/0.6859/0.6238/0.5828 within 0.001. Otherwise stop and investigate.

## 5. Decision rule (mechanical, fixed now)

For ΔM = M(U_PRS) − M(U_TAM):

- **PRS-BETTER:** canonical ΔM > 0 with p < .05 and ΔM > 0 in all 5 resplits.
- **NON-INFERIOR (TAM removable):** not PRS-BETTER, and (N1) canonical ΔM ≥ −0.005;
  (N2) ΔM ≥ −0.005 in ≥ 4 of 5 resplits; (N3) no per-horizon Δ below −0.020
  canonically or in the resplit mean.
- **TAM-NEEDED:** canonical ΔM < −0.005 with p < .05 and ΔM < −0.005 in ≥ 4 of 5
  resplits.
- **INCONCLUSIVE** otherwise.

Secondary, no decision attached: per-horizon Δ; test-partition ΔM; selected α
per arm; operational metrics (80% target sensitivity, per-fold training-only
threshold, patient-level "ever alerted", canonical folds only).

## 6. Out of scope

New fusion mechanisms, retuning α's grid, new features, changing TAM.
