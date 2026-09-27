# Phase 20 — Redesign of PRS / TAM from URM: results

Protocol: `docs/phase20_redesign_protocol.md` (frozen before any code; Amendments 1–3 recorded before any candidate ran).
All selection on the 464-patient development set D464 (canonical folds + 5 resplits). The 83-patient test partition was
**not touched** (no composite existed to confirm). Baseline B0 (URM harness) on D464: M = 0.6671 canonical,
mean 0.6576, split-to-split SD 0.0075; A/A seed noise on ΔM ≈ 0.001.

## Verdict: **nothing adopted — no URM-v2 exists; the frozen URM stands.**

| ID | Change | Canonical ΔM (95% CI) | p / Holm-adj p | Resplit ΔM (5) | Tier |
|---|---|---|---|---|---|
| C1 | proximity-weighted training | −0.0021 (−0.012, +0.008) | .67 / 1.0 | +.0005 −.0025 +.0027 +.0058 +.0010 | NOT SUPPORTED |
| C2 | patient-relative features | +0.0047 (−0.020, +0.029) | .72 / 1.0 | +.0149 +.0002 +.0022 −.0023 +.0008 | SUGGESTIVE |
| C3 | learner upgrade (EN / HGB / LR+HGB) | +0.0008 (−0.021, +0.022) | .90 / 1.0 | +.0039 +.0071 +.0196 +.0094 −.0246 | SUGGESTIVE |
| C5a | multi-scale (5/10 min) | −0.0020 (−0.009, +0.006) | .60 / 1.0 | −.0091 +.0018 +.0028 +.0090 −.0005 | NOT SUPPORTED |
| C5b | 40-min context | −0.0086 (−0.027, +0.007) | .29 / 1.0 | −.0080 −.0029 −.0099 +.0015 −.0018 | NOT SUPPORTED |
| C5c | decel morphology | +0.0017 (−0.008, +0.012) | .73 / 1.0 | −.0016 +.0078 +.0064 +.0141 −.0022 | NOT SUPPORTED |
| C4 S1 vs S0 | + age, gravidity, gestational weeks (late fusion) | −0.0053 | .64 | −.0012 −.0116 −.0287 −.0027 −.0304 | NOT SUPPORTED (S2, S3 not tested: fixed sequence) |
| C6a | causal GRU replacing PRS + TAM | −0.0135 (p .21) | .21 (K=1) | −.0094 −.0061 −.0025 −.0054 **−.0453** | NOT SUPPORTED |
| C6b | GRU on v2 features | — | — | — | not run (no feature candidate adopted) |

ADOPT required ΔM ≥ +0.005 with Holm-adjusted p < .05, positive in all 5 resplits, and no horizon worse than −0.020.
No candidate came near the first condition: every confidence interval spans zero, and the best canonical ΔM (+0.0047,
C2) has p = .72.

## What the numbers say

- **Feature and learner changes move the window-level score a little, but not the deployed system.** C3 raised the single-window PRS-only M by +0.009 and C5c by +0.007, yet the full URM-form result barely changed (+0.0008 / +0.0017): the pooler and parity fusion absorb what the window model gains.
- **C2/C3 are "suggestive" only in the sense of the pre-registered label** (canonical ΔM > 0, ≥ 4/5 resplits positive). Their canonical p-values are .72 and .90 and both have a resplit at −0.025 or near zero; they are not carried forward and should not be reported as improvements.
- **Individual splits produce apparent wins from noise:** C3 resplit 33 +0.0196 (p .037) and C5c resplit 44 +0.0141 (p .014) sit beside C3 resplit 55 −0.0246 (p .047). This is the split-to-split scatter (SD 0.0075) the protocol was built to guard against.
- **The extra covariates hurt** (S1 vs S0: worse in 5/5 splits, up to −0.030). Consistent with Phase 8's earlier covariate result; parity's contribution to URM does not extend to the other covariates screened here.
- **The GRU is worse than the two-stage design** in all 5 resplits (one at −0.045), consistent with 464 patients being too few to learn the sequence model end to end. It was not tuned (none of its hyper-parameters was searched, by design).
- Nested selection mostly chose the strongest regularisation available (C = 0.01 for C2 and all C5 arms; that is the edge of the frozen grid). Widening the grid was out of scope; it suggests extra features carry little usable signal at this sample size.

## Not established / caveats

- With 110 positives the resampling CIs on ΔM are about ±0.01–0.03, so effects below ~0.01 cannot be excluded; the finding is "no detectable improvement", not "proven equal".
- The baseline B0 inherits the legacy exposure to all 547 patients from Phases 8–19 (protocol §2); the new candidates are selection-clean with respect to the test partition.
- The pooler input convention (window scores out-of-fold for their own fold) is the same second-level stacking as Phases 16–19 and is applied to every arm equally.

## Files

`stage0_report.md`, `stage0_gates.json`, `stage0_G3a.json`, `stage0_G3b.json`, `stage1_window_results.csv`,
`stage1_window_verdict.json`, `stage1_c4_results.csv`, `stage1_c4_verdict.json`, `stage2_c6a_results.csv`,
`stage2_c6a_verdict.json`; scripts `scripts/phase20_*.py`.
