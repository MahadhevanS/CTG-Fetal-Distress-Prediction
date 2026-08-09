# Models 1–7 Re-Benchmark Log (Post Signal-Corruption Repair)

**Status**: Models 1–7 complete. **Model 8 (Knowledge-Infused Multi-Task Framework) is not included — deferred pending Colab compute availability.**

## 0. Why This Document Exists

A knowledge-infusion audit (2026-08-07) found that `interpolate_missing()` in `src/preprocessing/filtering.py` filled short signal gaps using an unconstrained cubic spline (`extrapolate=True`, no physiological bounds check), which could overshoot to thousands of bpm on ~56% of windows. This corrupted the FHR channel's Z-score scaler (std inflated to 453 instead of a physiologically sane ~15–30) and the derived clinical feature targets (LTV reaching a max of 20,596 bpm). The bug was fixed at the source (`filtering.py` / `pipeline.py`, physiological clamp of 50–240 bpm) and the CTU-CHB dataset was regenerated from raw `.dat` files with the corrected pipeline.

This document records the **first re-benchmark of Models 1–7 on the corrected dataset**, run on Google Colab (T4 GPU) directly from raw data. It supersedes the equivalent sections of `docs/model_inferences_log.md` for Models 1–7. Model 8's re-benchmark will be logged separately once run.

### Methodology note — read before comparing to the old table

This run built every encoder from **`configs/tuned_hyperparameters.yaml`** applied uniformly across all 7 models, at 100 epochs. Cross-checking against the original `docs/model_inferences_log.md`, only **PatchCTG** and **PatchTST** were documented as having gone through this tuned configuration originally — the other five (CNN1D, BiLSTM, GRU, TCN, MS-LSTM) appear to have used `train.py`'s built-in architecture defaults (e.g. old BiLSTM entry lists hidden size 64 / 2 layers, while `tuned_hyperparameters.yaml` specifies hidden size 32). So for those five models, the comparison below reflects **both** the data fix **and** a hyperparameter change, not the data fix in isolation — the two effects aren't separable from this run alone. PatchCTG and PatchTST are the cleanest apples-to-apples comparisons, since their hyperparameters are unchanged from the original benchmark.

### Dataset

- 6,826 total windows across 546 unique patients (train: 6,177 / val: 337 / test: 312 — see individual fold breakdowns below for train/val split sizes per fold).
- Corrected FHR channel scaler std: 33.49 (was 453.15 pre-repair).
- Stratified 5-Fold Patient-Level Cross-Validation, zero patient leakage across folds (unchanged protocol).

---

## 1. 1D CNN

### A. Hyperparameters Used (`configs/tuned_hyperparameters.yaml → cnn1d`)
- Learning Rate: `0.0001 (AdamW)` | Weight Decay: `0.0001`
- Batch Size: `16` | Epochs: `100`
- Loss: `BCEWithLogitsLoss` (Dynamic pos_weight, 5.55–6.09 across folds)

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `72.68% ± 6.81%` | `62.54% – 82.13%` |
| **AUROC** | `0.8100 ± 0.0573` | `0.7047 – 0.8678` |
| **AUPRC** | `0.4272 ± 0.0899` | `0.2764 – 0.5276` |
| **F1 Score** | `0.4587 ± 0.0509` | `0.3730 – 0.5085` |
| **Precision (PPV)** | `33.44% ± 6.00%` | — |
| **Recall (Sensitivity)** | `77.22% ± 13.78%` | `60.41% – 94.29%` |
| **Specificity** | `71.78% ± 9.38%` | `61.17% – 86.06%` |
| **Sens @ 90% Specificity** | `42.19% ± 10.13%` | `25.27% – 54.95%` |

### C. Inferences
- AUROC jumped from `0.6860` (pre-repair) to `0.8100` — the largest absolute AUROC gain of the seven models re-run here.
- Recall (77.22%) is now the highest among all 7 baselines, at the cost of moderate specificity (71.78%) — a reasonable trade for a distress-screening task.
- Fold 1 (0.8678 AUROC) substantially outperforms Fold 2 (0.7047), a wider spread than most other models — worth a closer look at that fold's patient composition before treating the mean as fully stable.

---

## 2. BiLSTM

### A. Hyperparameters Used (`configs/tuned_hyperparameters.yaml → bilstm`)
- Learning Rate: `0.001 (AdamW)` | Weight Decay: `1e-05`
- Hidden Size: `32` (bidirectional → 64 concat, projected to 128) | Layers: `2`
- Batch Size: `64` | Epochs: `100`

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `70.14% ± 3.15%` | `65.88% – 75.19%` |
| **AUROC** | `0.7540 ± 0.0230` | `0.7356 – 0.7967` |
| **AUPRC** | `0.3342 ± 0.0670` | `0.2528 – 0.4441` |
| **F1 Score** | `0.4078 ± 0.0326` | `0.3667 – 0.4482` |
| **Precision (PPV)** | `28.91% ± 2.47%` | — |
| **Recall (Sensitivity)** | `69.79% ± 7.43%` | `58.24% – 81.22%` |
| **Specificity** | `70.03% ± 4.88%` | `62.72% – 77.65%` |
| **Sens @ 90% Specificity** | `34.78% ± 8.56%` | `23.63% – 46.12%` |

### C. Inferences
- AUROC improved from `0.6544` to `0.7540`, the tightest fold-to-fold AUROC variance of all 7 models (±0.0230) — the most stable re-run in this batch.
- Recall (69.79%) and specificity (70.03%) are now closely balanced, a marked change from the old entry's asymmetric profile.

---

## 3. GRU

### A. Hyperparameters Used (`configs/tuned_hyperparameters.yaml → gru`)
- Learning Rate: `0.0005 (AdamW)` | Weight Decay: `1e-05`
- Hidden Size: `128` (single-layer bidirectional) | Layers: `1`
- Batch Size: `32` | Epochs: `100`

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `76.46% ± 3.31%` | `70.42% – 79.94%` |
| **AUROC** | `0.7971 ± 0.0554` | `0.6926 – 0.8578` |
| **AUPRC** | `0.3876 ± 0.0562` | `0.2980 – 0.4713` |
| **F1 Score** | `0.4460 ± 0.0765` | `0.3283 – 0.5596` |
| **Precision (PPV)** | `34.71% ± 5.95%` | — |
| **Recall (Sensitivity)** | `66.72% ± 19.85%` | `35.71% – 96.98%` |
| **Specificity** | `78.40% ± 6.32%` | `66.44% – 84.45%` |
| **Sens @ 90% Specificity** | `45.47% ± 10.79%` | `27.47% – 60.41%` |

### C. Inferences
- AUROC improved from `0.6881` to `0.7971`. Best Sens@90%Spec (45.47%) of all 7 baselines here.
- Highest recall variance of the batch (±19.85%, ranging 35.71%–96.98% across folds) — Fold 4's 96.98% recall looks like an outlier worth a second look rather than a representative operating point.

---

## 4. Temporal Convolutional Network (TCN)

### A. Hyperparameters Used (`configs/tuned_hyperparameters.yaml → tcn`)
- Learning Rate: `0.001 (AdamW)` | Weight Decay: `1e-05`
- Kernel Size: `3` | Dropout: `0.3`
- Batch Size: `64` | Epochs: `100`

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `77.42% ± 6.60%` | `65.18% – 84.46%` |
| **AUROC** | `0.7820 ± 0.0397` | `0.7136 – 0.8359` |
| **AUPRC** | `0.3714 ± 0.0327` | `0.3326 – 0.4242` |
| **F1 Score** | `0.4080 ± 0.0367` | `0.3545 – 0.4500` |
| **Precision (PPV)** | `36.21% ± 7.75%` | — |
| **Recall (Sensitivity)** | `52.48% ± 11.95%` | `30.46% – 63.32%` |
| **Specificity** | `81.83% ± 9.10%` | `66.18% – 94.22%` |
| **Sens @ 90% Specificity** | `36.75% ± 6.36%` | `29.67% – 44.67%` |

### C. Inferences
- AUROC improved from `0.7154` to `0.7820`. Still the highest-specificity model in the batch (81.83%), consistent with its pre-repair profile, but recall (52.48%) is now the second-lowest of the 7 — TCN continues to trade sensitivity for a low false-alarm rate.

---

## 5. Multi-Scale LSTM (MS-LSTM)

### A. Hyperparameters Used (`configs/tuned_hyperparameters.yaml → multiscale_lstm`)
- Learning Rate: `0.0005 (AdamW)` | Weight Decay: `0.0001`
- Hidden Size: `64` | Layers: `2`
- Batch Size: `32` | Epochs: `100`

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `70.19% ± 10.12%` | `55.00% – 83.28%` |
| **AUROC** | `0.8126 ± 0.0429` | `0.7347 – 0.8474` |
| **AUPRC** | `0.4349 ± 0.0907` | `0.3043 – 0.5480` |
| **F1 Score** | `0.4371 ± 0.0302` | `0.3842 – 0.4682` |
| **Precision (PPV)** | `31.69% ± 4.79%` | — |
| **Recall (Sensitivity)** | `74.87% ± 11.19%` | `57.14% – 87.91%` |
| **Specificity** | `68.98% ± 13.72%` | `48.75% – 87.07%` |
| **Sens @ 90% Specificity** | `44.05% ± 13.47%` | `20.88% – 60.41%` |

### C. Inferences
- AUROC improved from `0.7263` to `0.8126` — **the highest mean AUROC of all 7 baselines re-run here**, narrowly ahead of CNN1D.
- Widest accuracy and specificity variance of the batch (±10.12% / ±13.72%) — Fold 2 (55.00% accuracy, 48.75% specificity) drags the mean down noticeably relative to Folds 4–5 (>77% accuracy).

---

## 6. PatchCTG

### A. Hyperparameters Used (`configs/tuned_hyperparameters.yaml → patchctg` — same tuned config as the original benchmark)
- Learning Rate: `0.0001 (AdamW)` | Weight Decay: `1e-05`
- Patch Length / Stride: `P=8, S=16` | Layers: `2` | Heads: `4` | `d_model=128` | Dropout: `0.2`
- Batch Size: `32` | Epochs: `100` *(original benchmark used 15 epochs — see caveat below)*

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `71.41% ± 8.46%` | `54.91% – 78.95%` |
| **AUROC** | `0.7766 ± 0.0277` | `0.7323 – 0.8200` |
| **AUPRC** | `0.4005 ± 0.0407` | `0.3344 – 0.4607` |
| **F1 Score** | `0.4231 ± 0.0435` | `0.3469 – 0.4765` |
| **Precision (PPV)** | `31.36% ± 5.16%` | — |
| **Recall (Sensitivity)** | `69.54% ± 12.33%` | `55.49% – 91.96%` |
| **Specificity** | `71.95% ± 11.50%` | `49.36% – 80.53%` |
| **Sens @ 90% Specificity** | `41.55% ± 7.10%` | `32.24% – 53.30%` |

### C. Inferences
- AUROC improved from `0.6668` to `0.7766` using the **same tuned hyperparameters** as the original benchmark — this is the cleanest evidence in this batch that the data fix itself, not a hyperparameter change, drove the gain (epoch count differs, 15→100, so treat as strongly suggestive rather than fully isolated).
- Fold 4 shows a real weak point: 54.91% accuracy against 91.96% recall — the model is over-predicting distress in that fold specifically, dragging specificity to 49.36%.

---

## 7. PatchTST

### A. Hyperparameters Used (`configs/tuned_hyperparameters.yaml → patchtst` — same tuned config as the original benchmark, and the Model 8 backbone config)
- Learning Rate: `0.0001 (AdamW)` | Weight Decay: `0.0001`
- Patch Length / Stride: `P=16, S=16 (300 patches/channel)` | Layers: `4` | Heads: `4` | `d_model=128` | Dropout: `0.2`
- Batch Size: `64` | Epochs: `100` *(original canonical benchmark used 50 epochs — see caveat below)*

### B. Statistical Metrics (5-Fold Stratified Patient-Level CV)
| Metric | Mean ± Std | Fold Range (Min – Max) |
| :--- | :--- | :--- |
| **Accuracy** | `74.40% ± 6.49%` | `64.59% – 84.38%` |
| **AUROC** | `0.7939 ± 0.0359` | `0.7426 – 0.8376` |
| **AUPRC** | `0.4094 ± 0.0668` | `0.3295 – 0.4936` |
| **F1 Score** | `0.4278 ± 0.0502` | `0.3640 – 0.5109` |
| **Precision (PPV)** | `33.79% ± 8.19%` | — |
| **Recall (Sensitivity)** | `64.00% ± 13.91%` | `51.10% – 89.45%` |
| **Specificity** | `76.30% ± 9.60%` | `60.87% – 90.00%` |
| **Sens @ 90% Specificity** | `44.45% ± 5.62%` | `37.36% – 53.30%` |

### C. Inferences
- AUROC improved from `0.7504` (old §8 canonical benchmark) to `0.7939`, again using the same tuned hyperparameters as before — another largely data-driven gain (epoch count differs, 50→100).
- **Important**: PatchTST was selected as the Model 8 backbone specifically for being the top-performing baseline pre-repair (0.7504, highest of 7). Post-repair, it now ranks **4th of 7** by AUROC (see §8) — MS-LSTM, CNN1D, and GRU all score higher. This doesn't necessarily mean PatchTST is the wrong backbone (Model 8's own value depends on how well its architecture supports the knowledge-infusion heads, not just standalone AUROC), but it does mean the original backbone-selection rationale should be revisited once Model 8 is re-benchmarked on this same corrected data.

---

## 8. Master Comparison Table — Post-Repair (This Run)

| Model | AUROC | AUPRC | F1 Score | Precision (%) | Recall (%) | Specificity (%) | Sens @ 90% Spec (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MS-LSTM** 🏆 | **0.8126 ± 0.0429** | 0.4349 ± 0.0907 | 0.4371 ± 0.0302 | 31.69 ± 4.79 | 74.87 ± 11.19 | 68.98 ± 13.72 | 44.05 ± 13.47 |
| **CNN1D** | 0.8100 ± 0.0573 | **0.4272 ± 0.0899** | **0.4587 ± 0.0509** | 33.44 ± 6.00 | **77.22 ± 13.78** | 71.78 ± 9.38 | 42.19 ± 10.13 |
| **GRU** | 0.7971 ± 0.0554 | 0.3876 ± 0.0562 | 0.4460 ± 0.0765 | 34.71 ± 5.95 | 66.72 ± 19.85 | 78.40 ± 6.32 | **45.47 ± 10.79** |
| **PatchTST** | 0.7939 ± 0.0359 | 0.4094 ± 0.0668 | 0.4278 ± 0.0502 | 33.79 ± 8.19 | 64.00 ± 13.91 | 76.30 ± 9.60 | 44.45 ± 5.62 |
| **TCN** | 0.7820 ± 0.0397 | 0.3714 ± 0.0327 | 0.4080 ± 0.0367 | 36.21 ± 7.75 | 52.48 ± 11.95 | **81.83 ± 9.10** | 36.75 ± 6.36 |
| **PatchCTG** | 0.7766 ± 0.0277 | 0.4005 ± 0.0407 | 0.4231 ± 0.0435 | 31.36 ± 5.16 | 69.54 ± 12.33 | 71.95 ± 11.50 | 41.55 ± 7.10 |
| **BiLSTM** | 0.7540 ± 0.0230 | 0.3342 ± 0.0670 | 0.4078 ± 0.0326 | 28.91 ± 2.47 | 69.79 ± 7.43 | 70.03 ± 4.88 | 34.78 ± 8.56 |

*(Model 8 excluded — not yet re-benchmarked on corrected data.)*

---

## 9. Delta vs. Pre-Repair Benchmark (`docs/model_inferences_log.md`)

| Model | Old AUROC | New AUROC | Δ AUROC | Hyperparameters Same as Old Run? |
| :--- | :--- | :--- | :--- | :--- |
| CNN1D | 0.6860 ± 0.0503 | 0.8100 ± 0.0573 | **+0.1240** | No — tuned config differs from old defaults |
| BiLSTM | 0.6544 ± 0.0409 | 0.7540 ± 0.0230 | **+0.0996** | No — tuned hidden size (32) differs from old (64) |
| GRU | 0.6881 ± 0.0627 | 0.7971 ± 0.0554 | **+0.1090** | No — tuned config (1 layer, hidden 128) differs from old (2 layers, hidden 64) |
| TCN | 0.7154 ± 0.0797 | 0.7820 ± 0.0397 | **+0.0666** | Uncertain — old entry doesn't fully specify LR/dropout used |
| MS-LSTM | 0.7263 ± 0.1103 | 0.8126 ± 0.0429 | **+0.0863** | Uncertain — old entry doesn't fully specify tuning source |
| PatchCTG | 0.6668 ± 0.0161 | 0.7766 ± 0.0277 | **+0.1098** | **Yes** (epochs differ: 15→100) |
| PatchTST | 0.7504 ± 0.0378 | 0.7939 ± 0.0359 | **+0.0435** | **Yes** (epochs differ: 50→100) |

Every model improved, with no exceptions. The two models run under genuinely unchanged hyperparameters (PatchCTG, PatchTST) still improved by a meaningful margin (+0.11 and +0.04 AUROC respectively), which is the strongest evidence that the signal-corruption fix itself — not just the hyperparameter changes applied to the other five — is doing real work. For CNN1D/BiLSTM/GRU/TCN/MS-LSTM, treat the reported delta as data-fix-plus-hyperparameter-change combined; isolating the two would require a same-hyperparameter re-run against the old (corrupted) dataset, which no longer exists to test against without reverting the fix.

---

## 10. Open Items Before Finalizing

1. **Model 8 re-benchmark** — pending Colab compute. This is the comparison that actually matters for the project's central claim; nothing above should be treated as final until it's done.
2. **PatchTST's backbone selection should be revisited.** It was chosen as Model 8's encoder for being the top pre-repair baseline; post-repair it ranks 4th of 7 on standalone AUROC. Worth at minimum documenting this explicitly when Model 8's results land, and considering whether MS-LSTM or CNN1D deserve a knowledge-infusion trial of their own.
3. **Epoch-count mismatch vs. old canonical numbers** (PatchCTG 15→100, PatchTST 50→100) means the deltas in §9 aren't purely attributable to the data fix even for those two models — a fully controlled re-run (same epochs, same hyperparameters, only data changed) would close this gap if a rigorous causal claim is needed for the writeup.
4. **Update `docs/model_inferences_log.md`** once Model 8 is re-benchmarked, so there's a single current source of truth rather than two documents.
