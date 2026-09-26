# Phase 19b — Does the deployable fusion (URM) need TAM?

Protocol: `docs/phase19b_tam_vs_prs_fusion_protocol.md` (frozen before code). Script: `scripts/phase19b_fusion_tam_vs_prs.py`.
Gate passed: U_TAM reproduces the Phase 18 stability CANONICAL rows and PRS alone reproduces 0.6872/0.6859/0.6238/0.5828 (all to 4 decimals).

## Mechanical verdict: **INCONCLUSIVE**

Canonical ΔM = M(U_PRS) − M(U_TAM) = **−0.0065** (95% CI −0.017 to +0.004, p = .26).
Not PRS-BETTER (ΔM < 0). Not NON-INFERIOR (N1 fails: −0.0065 < −0.005; N3 fails: delivery Δ = −0.019). Not TAM-NEEDED (p = .26, and 0/5 resplits have ΔM < −0.005).

## Canonical CV AUROC

| Model | Delivery | ≥10m | ≥20m | ≥30m | M |
|---|---|---|---|---|---|
| U_TAM (URM) | 0.7335 | 0.6957 | 0.6675 | 0.6646 | 0.6903 |
| U_PRS (latest window + parity) | 0.7141 | 0.6975 | 0.6682 | 0.6554 | 0.6838 |
| TAM alone | 0.7216 | 0.6683 | 0.6015 | 0.5976 | 0.6472 |
| PRS alone | 0.6872 | 0.6859 | 0.6238 | 0.5828 | 0.6449 |
| Parity alone | 0.5755 | 0.5755 | 0.5755 | 0.5755 | 0.5755 |

U_PRS − U_TAM per horizon: delivery −0.0194 (p = .049), ≥10m +0.0018 (p = .82), ≥20m +0.0007 (p = .94), ≥30m −0.0092 (p = .31).

## Stability and test

- Resplits (5): ΔM = −0.0040, −0.0014, −0.0032, −0.0033, −0.0040 (all p > .5; sign consistent, magnitude under the −0.005 line). Delivery Δ is −0.016 to −0.020 in every resplit; ≥10m and ≥20m are +0.004 to +0.007 in 4–5 of 5.
- Test partition (n = 83, once): M(U_TAM) 0.7547 vs M(U_PRS) 0.7348, ΔM = −0.0198 (direction agrees; n is small).
- Selected α: U_TAM 0.30–0.35 per fold, U_PRS 0.30–0.35 (essentially identical).

## Operational (canonical folds, 80% target sensitivity, training-only threshold)

| System | Sens | False-alert rate | Median lead | ≥20 min | ≥30 min |
|---|---|---|---|---|---|
| U_TAM | 86.4% | 62.5% | 40.0 | 83.2% | 70.5% |
| U_PRS | 88.2% | 67.7% | 40.0 | 83.5% | 69.1% |

## Reading

Most of URM's gain over its parts comes from **parity fusion**, not from TAM: U_PRS recovers the entire ≥10m/≥20m fusion result and gives up most of the benefit only at delivery-time and ≥30m queries. TAM's contribution to the fusion is a small, consistent delivery-time advantage (~+0.016–0.020 AUROC in all 6 splits and on the test partition), which reaches p = .049 canonically but does not change the pre-registered verdict. Removing TAM is therefore *not* shown safe by the pre-registered rule; keeping it is *not* shown necessary either. The two arms differ in false-alert rate by ~5 points at matched target sensitivity, in TAM's favour.
