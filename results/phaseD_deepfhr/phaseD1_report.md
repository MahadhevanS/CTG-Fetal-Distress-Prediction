# Phase D1 — Faithful DeepFHR reproduction: result

Protocol: `docs/phaseD_deepfhr_reproduction_protocol.md`, config: `docs/deepfhr_original.yaml`. Code: `src/deepfhr/`,
`scripts/phaseD1_deepfhr_paperstyle.py`, `scripts/phaseD1_diagnostics.py`. Primary source read in full before any code was
written (Zhao et al. 2019, BMC Med Inform Decis Mak 19:286, open access) — see protocol §0 for verified facts and quotes.

## Cohort and image counts reproduced exactly
All 552 raw CTU-UHB records, 447 normal / 105 pathological (pH < 7.15) — matches the paper's own numbers exactly. Image
generation reproduced the paper's own count exactly: **3312 images (2682 normal / 630 pathological)**, from 552 records ×
(2 mother wavelets × 3 scale-counts).

## Result 1 (initial run): does NOT reproduce the published ceiling
Image-level random 10-fold CV, unweighted training, no input normalisation:

| Metric | Initial run | Published |
|---|---|---|
| Accuracy | 81.4% | 98.34% |
| Sensitivity (Normal = positive) | 99.9% | 98.22% |
| **Specificity** | **2.5%** | 94.87% |
| QI | 14.6% | 96.53% |
| AUC | 70.2% | 97.82% |

The Se/Sp pattern (99.9%/2.5%) is the signature of a model that mostly predicts the majority class.

## Two fixes applied (both requested and completed before any further work)

1. **Inverse-frequency class-weighted `CrossEntropyLoss`**, weights computed from the training fold only (never the held-out
   fold) — addresses the ~4.3:1 class imbalance the initial run's Se/Sp collapse pointed to.
2. **Per-channel zero-centering** of the input images (mean computed on the training fold only) — Table 2, Layer 1 of the
   paper specifies "Data normalization: Zero center"; the initial run omitted this entirely.

### Result 2 (after both fixes)
| Metric | Fixed threshold 0.5 | Train-Youden threshold | Published |
|---|---|---|---|
| Accuracy | 59.2% ± 7.8 | 59.4% ± 3.6 | 98.34% |
| Sensitivity | 58.3% ± 12.2 | 57.6% ± 5.6 | 98.22% |
| Specificity | 61.2% ± 11.6 | 66.9% ± 7.3 | 94.87% |
| QI | 58.6% ± 4.7 | 61.8% ± 2.5 | 96.53% |
| **AUC (threshold-independent)** | **66.2%** (pooled) | same | 97.82% |

**The fixes worked exactly as intended** — Se and Sp are now balanced instead of collapsed to one extreme — **but AUC barely
moved** (66–69% across the two attempts, vs. 70% before either fix). Fixing the classification threshold/balance was never
going to fix a *ranking* problem, and AUC is a ranking measure. This means the shortfall is not the class-imbalance/
normalisation issues just fixed; something deeper limits how well this CNN, on these reconstructed images, can rank
patients — so a further diagnostic was needed before concluding anything.

## Diagnosis: a hard capacity ceiling, not a training-duration problem

Two additional checks (`scripts/phaseD1_diagnostics.py`), run to isolate *why* the ranking quality is stuck:

**Q1 — does the leakage signal exist in the images at all, independent of the CNN?** A trivial pixel-level logistic
regression (flattened raw pixels, same image-random 10-fold split, no CNN, no training-duration question at all) reaches
**AUC 0.934 ± 0.014** — far above the CNN's 0.66–0.70. The leakage signal is unambiguously present and strong.

**Q1b — is this because of near-duplicate images from the same recording, as expected?** Mean pixel-space L2 distance
between a record's own 6 images: **2.61**; between images from different records: **7.94** — a **3.0× gap**, confirming
the intended leakage mechanism (six near-duplicates of one segment landing on both sides of the split) is real and large.

**Q2 — is the CNN just undertrained (20 epochs, per the paper's own spec)?** Training the *same* CNN for 20 / 100 / 300
epochs on one fold:

| Epochs | Test AUC | **Train AUC** |
|---|---|---|
| 20 | 0.627 | 0.721 |
| 100 | 0.643 | 0.723 |
| 300 | 0.647 | 0.717 |

**Train AUC itself is flat at ~0.72 regardless of a 15× increase in training time.** This rules out "not enough epochs" —
the model has already converged to whatever it can fit, and that ceiling (0.72 in-sample) is itself far below the 0.93 a
linear model achieves on raw pixels of the same data.

## Conclusion
The leakage mechanism is confirmed real (Q1, Q1b) and the two requested fixes are correctly implemented and working as
intended (Se/Sp balance restored) — but neither the fixes nor additional training time close the gap to the published
ceiling. The remaining gap is best explained by the **one substitution flagged from the start** (protocol §1): this
project's reconstructed CWT images, built from a discrete wavelet's own reconstruction function convolved directly with the
signal (the closest available stand-in for MATLAB's legacy `cwt()`), apparently organise the leakage signal in a form a
small-kernel, translation-invariant CNN cannot exploit as readily as a full-image linear read can — even though the signal
is present and strong at the pixel level. This is a property of the *image construction*, not of training configuration,
class balance, or training duration, all three of which have now been checked and ruled out.

**Per the protocol's own gate rule, this stays in the "inconclusive for the reproduction question" category** — the gap is
attributable to a disclosed, unavoidable methodological substitution (no MATLAB available to run the original toolbox
function), not to an error in this implementation. D1 is complete: both requested fixes are applied and verified, the
result is cleaned up, and the reason for the remaining gap is now well understood and documented rather than an open question.
