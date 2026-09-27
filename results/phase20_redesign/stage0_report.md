# Phase 20 Stage 0 — gates and noise floor

Protocol: `docs/phase20_redesign_protocol.md` §5. Code: `scripts/phase20_harness.py`, `phase20_stage0_gates.py`,
`phase20_stage0_aa.py`, `phase20_g3_feature_reproduction.py`. Numbers: `stage0_gates.json`, `stage0_baseline_D464.csv`,
`stage0_AA_test.csv`.

## Gates

| Gate | Result |
|---|---|
| **G1** P6 refit reproduces `p6_predictions.npz` | **PASS** — max abs diff 0.0 (CV and train+val→test) |
| **G2** harness B0 on 547 canonical folds reproduces frozen URM / TAM (±0.006) and PRS (±0.001) | **PASS** — URM 0.7334/0.6957/0.6677/0.6646 vs frozen 0.7335/0.6957/0.6675/0.6646; TAM within 0.0003; PRS exact |
| **G3(a)** 19 raw descriptors recomputable from raw records (1e-5) | **PASS** — 40 patients / 608 windows, max abs diff 3.6e-6 (after `pip install wfdb`, a declared requirement). The stored extended features come from the z-scored float32 tensors, so the recomputation emulates that round trip (see protocol Amendment 1). |
| **G4** noise floor | measured, below |

## G4 — baseline B0 on the development set D464 (464 patients)

| Split | M | 0m | 10m | 20m | 30m |
|---|---|---|---|---|---|
| CANONICAL | 0.6671 | 0.7101 | 0.6692 | 0.6484 | 0.6407 |
| resplit 11 | 0.6634 | 0.7085 | 0.6746 | 0.6447 | 0.6260 |
| resplit 22 | 0.6618 | 0.7109 | 0.6714 | 0.6361 | 0.6286 |
| resplit 33 | 0.6535 | 0.7052 | 0.6622 | 0.6352 | 0.6115 |
| resplit 44 | 0.6489 | 0.7095 | 0.6618 | 0.6230 | 0.6011 |
| resplit 55 | 0.6509 | 0.7122 | 0.6548 | 0.6301 | 0.6067 |

- Mean M = **0.6576**, SD across the 6 splits = **0.0075** (range 0.6489–0.6671). This is *lower* than the
  547-patient URM (0.690): D464 excludes the 83 test patients, so all D464 numbers are the reference for Phase 20
  and must not be compared with earlier headline figures.
- Single-seed spread on the canonical split: M 0.6603 / 0.6690 / 0.6689 (SD 0.0050; seed 42, the frozen-TAM seed,
  is the low one at delivery, 0.6885 vs 0.714). The 3-seed mean is used everywhere.
- Epoch-cap hits (200 epochs) are common (7/15 on 547-canonical; 0–7/15 per D464 split) — same behaviour as the frozen
  TAM training; not a new issue, reported.

## A/A test (supplement to G4, no arm changed)

Identical baseline with network seeds {45,46,47} vs {42,43,44}: ΔM = −0.0007, +0.0002, +0.0011, +0.0003, +0.0000, +0.0003
over the six splits (SD 0.0006, max |ΔM| 0.0011, worst per-horizon −0.0018). **Training noise on ΔM is ~0.001**, five times
below the +0.005 ADOPT bar; the bar is therefore not at risk from seed noise. Caution: the paired bootstrap called
one of these no-difference pairs "significant" (ΔM = +0.0011, p = .002), because near-identical score sets have tiny
resampling variance. The magnitude requirement (ΔM ≥ +0.005) is what protects against that, not the p-value alone.
Patient-sampling noise (the split-to-split SD of 0.0075) is controlled by the resplit-consistency rules, not by seeds.

## Convention carried over (not new)

The pooler's training inputs for outer fold f are window scores that were produced out-of-fold *for their own fold*,
so they came from window models that had seen fold f's patients. This is the same second-level stacking convention used
in Phases 16–19 and by the frozen URM; it is kept so that arms are comparable, and it applies equally to every arm.
