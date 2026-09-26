# Phase D2/D3/D4-D6 — The settled URM-vs-DeepFHR comparison

Protocol: `docs/phaseD_deepfhr_reproduction_protocol.md`, Amendment 1. Code: `scripts/phaseD2_patient_causal_comparison.py`
(reuses the exact D1 CWT+CNN pipeline and both verified fixes — class-weighted loss, zero-centering — unchanged).

**What makes this number trustworthy, unlike D1:** both models are scored on **the identical 547 patients**, the **identical
canonical 5-fold patient-grouped split** (`folds.json`), the **identical causal eligible-prefix horizon rule**, and the
**identical paired-bootstrap convention** (B=2000, seed 42). URM's scores are its own exact, gate-verified sequence
(`build_urm_sequences()`, reproduces the frozen 0.7335/0.6921/0.6638/0.6631 to 4 decimals) — never rerun or touched. This
does not depend on ever resolving D1's gap to the paper's own literal ceiling.

## Result

| Horizon | DeepFHR-reconstruction AUROC | URM AUROC | Δ (URM − DeepFHR) | 95% CI | p |
|---|---|---|---|---|---|
| Delivery | 0.537 | **0.734** | **+0.197** | [0.121, 0.272] | <0.0001 |
| ≥10 min | 0.591 | **0.692** | **+0.101** | [0.037, 0.170] | 0.002 |
| ≥20 min | 0.575 | **0.664** | **+0.089** | [0.015, 0.158] | 0.015 |
| ≥30 min | 0.567 | **0.663** | **+0.096** | [0.020, 0.175] | 0.016 |

**URM beats the reconstructed DeepFHR at every horizon, by a statistically significant margin (p < .02 at every horizon,
p < .0001 at delivery), with no confidence interval crossing zero.** This is the trustworthy, apples-to-apples metric
requested — it settles the comparison regardless of DeepFHR's own unconfirmed literal ceiling (D1).

## Why the DeepFHR-reconstruction number itself (0.54–0.59) is so much lower than D1's leaky-split number (0.66–0.70)
This is expected and consistent with two independent pieces of prior evidence:
- This project's own earlier, separate CWT ablation (`reports/cwt/`, a different — generic Morlet — CWT construction,
  already patient-grouped) found **AUROC 0.5301**, in the same range.
- CrossFormer's own patient-level reproduction (this project, `reports/cwt/prior_art_reconciliation.md`) dropped to
  **AUROC 0.6167** once re-scored patient-level instead of window-pooled — the same qualitative pattern: a CWT/CNN or
  transformer architecture that looked strong under a leakier evaluation drops sharply once patient-level independence is
  enforced.

Combined with D1's own diagnostics (a pixel-level control reaching 0.93 AUROC under the *leaky* split, confirming strong
exploitable duplicate-image signal that disappears once patients can't straddle folds), the overall picture is coherent:
**DeepFHR's published performance is very likely substantially inflated by the leaky image-random split it uses**, and
under a fair, patient-independent, causal evaluation — the one that actually matters for this project's early-warning
use case — it does not outperform URM. It underperforms URM clearly.

## What this does and does not establish
- **Does establish**: on this cohort, under the evaluation discipline URM itself is held to, URM is the better model.
  This is now a properly controlled, paired, statistically significant result — not an inference from mismatched published
  numbers.
- **Does not establish**: what DeepFHR's *authors'* own implementation would score under this same discipline (that would
  need their actual code/weights, unavailable). It also does not resolve D1's separate, still-open question (why this
  reconstruction falls short of the paper's own literal 0.978 under its own leaky protocol) — that question no longer
  needs to be resolved for the URM comparison to stand, per Amendment 1's reasoning.
- Window-level sanity AUROC per fold (0.536–0.604) confirms the underlying window classifier is weak but not degenerate
  (better than chance in every fold), consistent with a genuine, if modest, causal signal — matching this project's other
  CWT-representation findings rather than contradicting them.

## Files
`d2_window_images.npz` (51,102 causal rolling-window images, cached), `d2_urm_comparison.csv`/`.json` (this table),
`d2_train.log` (full run log, including per-fold window-level sanity AUROC).
