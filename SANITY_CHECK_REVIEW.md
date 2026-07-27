# CTG Fetal Distress Prediction - Comprehensive Sanity Check Review

**Reviewer**: AI Expert Review (Antigravity)
**Date**: 2026-07-27
**Scope**: Full project review - Phase 1 (Preprocessing) through Phase 3 (Model Benchmarking, 7 encoders)

---

## Executive Summary

This is a well-structured, scientifically serious deep-learning research project for CTG-based fetal distress prediction. The team demonstrates strong understanding of clinical constraints, data leakage prevention, and multi-encoder benchmarking. **The project is in solid shape overall**, but several specific issues require immediate attention before Phase 3 can be considered complete.

**Overall Rating: 7.5 / 10**

---

## Table of Contents

1. Project Structure and Organization
2. Documentation Quality
3. Data and Preprocessing Pipeline
4. Model Architecture Review (All 7 Encoders)
5. Universal Training Loop (train.py)
6. Model Metrics Analysis and Cross-Comparison
7. Critical Inconsistencies Found
8. Git and Workflow Compliance
9. What Is Missing
10. Prioritized Action Items

---

## 1. Project Structure and Organization

**Rating: EXCELLENT**

```
CTG-Fetal-Distress-Prediction/
|- src/
|   |- models/          [GOOD] Encoders only, no training logic leaked in
|   |- preprocessing/   [GOOD] Fully frozen v1.0 pipeline
|   |- training/        [GOOD] Universal training loop
|   |- knowledge/       [MISSING] Referenced in docs but directory does not exist
|- configs/             [GOOD] Environment-separated YAML (local vs colab)
|- checkpoints/         [GOOD] Saved weights per fold
|- docs/                [GOOD] 17 comprehensive technical documentation files
|- notebooks/           [GOOD] EDA notebook present
+-- scripts/            [GOOD] Helper scripts
```

**Strengths:**
- Strict separation between encoder definitions (src/models/) and training logic (src/training/)
- Environment-aware config system (local.yaml vs colab.yaml)
- Clear phase progression: Preprocessing -> Benchmarking -> Knowledge Infusion

**Issues:**
- [WARNING] src/knowledge/ is referenced throughout workflow_and_plan.md as containing
  figo.py, but the directory does not exist under src/. The FIGO rule engine is imported
  in pipeline.py from a relative path. This blocks Phase 4 (Model 8) entirely.
- [WARNING] No tests/ directory. Given clinical stakes of fetal distress prediction,
  basic unit tests for preprocessing and encoder shape contracts are appropriate.
- [WARNING] scratch/ directory at project root should be in .gitignore or documented
  as developer-only.

---

## 2. Documentation Quality

**Rating: EXCELLENT (Best-in-class for a research project)**

| Document                                    | Quality     | Notes                                              |
|---------------------------------------------|-------------|----------------------------------------------------|
| AI_AGENT_RULES.md                           | EXCELLENT   | Clear rules with team allocation matrix            |
| workflow_and_plan.md                        | EXCELLENT   | Phase breakdown with formal loss formulation       |
| dataset_summary.md                          | EXCELLENT   | Precise tensor specs, imbalance rationale          |
| model_evaluation_plan.md                    | EXCELLENT   | Rigorous; covers clinical safety metrics           |
| preprocessing_technical_documentation.md   | EXCELLENT   | Full pipeline with math and code links             |
| models_set_1_technical_documentation.md    | VERY GOOD   | Well-structured with architectural diagrams        |
| models_set_2_technical_documentation.md    | VERY GOOD   | Excellent math: GRU gates, TCN receptive field     |
| models_set_3_technical_documentation.md    | VERY GOOD   | Mermaid diagrams add visual clarity                |
| models_set_4_technical_documentation.md    | EXCELLENT   | Publication-ready math and architecture diagrams   |
| model_inferences_log.md                     | INCOMPLETE  | Sets 1 & 2 have no 5-fold AUROC/AUPRC (TBD)       |
| README.md                                   | CRITICAL    | Placeholder URL, wrong dir name, no results table  |

**Specific Issues:**

1. README.md is an unfinished stub. Needs: abstract, dataset citations (CTU-CHB
   PhysioNet DOI, UCI repo), results table, reproduction steps. Both
   placeholder URL and "cd CTG" remain unresolved.

2. CNN1D and BiLSTM (Set 1) metrics are TBD. AUROC, AUPRC, F1, Sensitivity,
   and Specificity are all marked "TBD (Requires 5-Fold Evaluation Protocol)"
   despite being the first two models completed. This is a direct contradiction
   of the project's own evaluation standard.

3. Model 8 log has default 0.00 values in the statistical significance table,
   which could be misread as actual null results rather than placeholders.

---

## 3. Data and Preprocessing Pipeline

**Rating: EXCELLENT - Strongest component of the project**

### What Is Done Right

| Design Decision                                         | Status                                    |
|---------------------------------------------------------|-------------------------------------------|
| Patient-level 70/15/15 split BEFORE window extraction  | CORRECT - prevents data leakage           |
| Strict 30-min prediction horizon (GAP 2 FIX)           | CORRECT - eliminates weak supervision     |
| Spike removal at 25 bpm/sec physiological limit         | CORRECT physiological threshold           |
| Cubic spline interpolation only for gaps <= 15 seconds  | CORRECT - conservative                    |
| 4th-order Butterworth at 1.5 Hz, zero-phase filtfilt    | CORRECT - preserves phase integrity       |
| Z-score fitted on training set only                     | CORRECT - zero leakage                    |
| 10-Point Consistency Audit (consistency_audit.py)       | OUTSTANDING - rarely seen in research     |
| No SMOTE applied to raw waveforms                       | CORRECT - synthetic waveforms invalid     |
| Dual-dataset strategy (CTU-CHB + UCI SisPorto)          | Strong external validation design         |

### Issues Found

[CRITICAL] Distress stride discrepancy between code and documentation:
  dataset_summary.md: "Distress cases: 2-minute stride"
  pipeline.py line 37: DISTRESS_STRIDE_MINUTES = 0.5  (30-second stride)
  The code creates 4x MORE aggressive oversampling than documented. The total
  number of training windows and effective class balance ratio will differ
  significantly from what is claimed. One source is wrong and must be corrected.

[WARNING] y_primary dtype is misleading:
  dataset_summary.md Section B claims dtype torch.long.
  train.py line 107 immediately casts via y_primary.float() to float32.
  The documentation should state float32 to avoid confusion.

[WARNING] MultiScale LSTM parameter count conflict:
  model_inferences_log.md body text: 583,904 parameters
  Same log benchmark table + models_set_3 doc: 622,464 parameters
  Must be verified with: sum(p.numel() for p in model.parameters())

---

## 4. Model Architecture Review (All 7 Encoders)

### 4.1 CNN1DEncoder  (src/models/cnn1d_encoder.py)
Rating: GOOD (correct implementation, inconsistent API)

Architecture: Stem -> 3 Residual Blocks -> AdaptiveAvgPool1d -> Linear(128, 128)
Sound ResNet-lite pattern with correct 1x1 shortcut projection for channel changes.

ISSUES:
  [WARNING] __init__ takes ZERO arguments: def __init__(self):
    All other encoders accept in_channels, seq_len, latent_dim keyword args.
    Makes CNN1D completely unconfigurable and forces build_encoder() to use
    a fragile empty-kwargs fallback.
  [WARNING] No LayerNorm on output embedding. Models 3-7 all include LayerNorm(128).
    Creates subtle representation distribution mismatch at Universal Classifier head.
  [WARNING] No module-level docstring with math or parameter count.

### 4.2 BiLSTMEncoder  (src/models/bilstm_encoder.py)
Rating: GOOD (correct implementation, underdocumented)

Standard bidirectional LSTM with final hidden state extraction. Correct.

ISSUES:
  [WARNING] File is 1,201 bytes - smallest encoder by far. No math docstring,
    no parameter count comment. Compare to GRU encoder which has full formulations.
  [WARNING] No LayerNorm on output (same issue as CNN1D).

### 4.3 GRUEncoder  (src/models/gru_encoder.py)
Rating: EXCELLENT

Best-implemented classical model. 1D Conv stem downsamples 4800->1200 steps before
the GRU (reducing effective sequence length 4x, alleviating vanishing gradient).
Dual temporal pooling (mean + max concatenation) extracts complementary statistical
moments. LayerNorm on the output. Correct implementation throughout.

### 4.4 TCNEncoder  (src/models/tcn_encoder.py)
Rating: EXCELLENT

ChopCausalPadding for strict temporal causality is correctly implemented. The
exponential dilation sequence [1, 2, 4, 8, 16, 32] yields a receptive field of 253
timesteps (~63 seconds at 4 Hz) - clinically meaningful for spanning a full uterine
contraction cycle. Dual mean+max pooling matches GRU's pattern.

Minor: Log cites 363.5K params; technical doc calculates 366.9K. Verify with torchinfo.

### 4.5 MultiScaleLSTMEncoder  (src/models/multiscale_lstm.py)
Rating: VERY GOOD

Three parallel BiLSTM branches (fine: stride 2, medium: stride 8, coarse: stride 32)
map to CTG physiological timescales: STV, contraction cycle, and LTV respectively.
GELU activations appropriate for network depth. LayerNorm on output correct.

ISSUE:
  [WARNING] assert statements in forward() lines 115-116 raise unhelpful AssertionError.
    Replace with: if not condition: raise ValueError("descriptive message here")

### 4.6 PatchCTGEncoder  (src/models/patchctg.py)
Rating: EXCELLENT

Joint channel patchification (C*P = 32 dimensions per token) deliberately differs from
PatchTST's channel-independence - capturing FHR-UC cross-channel interactions within
each 4-second patch. Pre-LN Transformer blocks and enable_nested_tensor=False for
CUDA compatibility are both correct architectural choices.

### 4.7 PatchTSTEncoder  (src/models/patchtst.py)
Rating: EXCELLENT

Channel-independent patching (folding B*C into unified batch dimension) is the canonical
PatchTST design, shown superior on diverse time-series benchmarks. The latent head
Linear(256->128) -> GELU -> Dropout -> Linear(128->128) -> LayerNorm is publication-quality.

Minor: import math at line 12 is unused. Remove before paper submission.

---

## 5. Universal Training Loop (train.py)

**Rating: GOOD design, but THREE CRITICAL BUGS**

### What Is Done Right

| Feature                                                 | Status  |
|---------------------------------------------------------|---------|
| Stratified patient-level 5-fold CV (StratifiedKFold)    | CORRECT |
| drop_last=True prevents BatchNorm NaN on 1-sample batch | CORRECT |
| AMP (autocast + GradScaler) gated to CUDA only          | CORRECT |
| calculate_metrics() handles zero-division edge cases    | CORRECT |
| Dynamic model registration via MODEL_REGISTRY           | CLEAN   |
| dry_run mode for shape validation                       | USEFUL  |

### CRITICAL BUG 1: pos_weight hardcoded to 2.0 (line ~277)

    pos_weight = torch.tensor([2.0]).to(device)

The model_evaluation_plan.md and models_set_3 documentation BOTH specify that
pos_weight must be computed dynamically per fold as N_neg / N_pos. With the actual
CTU-CHB class ratio (~95:5 in evaluation splits), the correct weight is approximately
19.0, not 2.0. Using 2.0 under-penalizes false negatives by a factor of ~10.

More critically: Models 5-7 (MS-LSTM, PatchCTG, PatchTST) were each trained via
their dedicated scripts (train_multiscale_lstm.py, train_patchctg.py, train_patchtst.py)
which compute dynamic pos_weight. The universal train.py used for Models 1-4 does NOT.
The benchmark comparison is therefore NOT apples-to-apples - directly violating the
central mandate of AI_AGENT_RULES.md Section 2.

### CRITICAL BUG 2: Saves last-epoch checkpoint, not best-epoch (line ~427)

    torch.save(model.state_dict(), save_path)

train_single_fold() tracks best_val_auroc and returns best_metrics, but does NOT save
the best model weights. The torch.save() after the function returns saves whatever state
the model is in at the LAST training epoch, which is not necessarily the best epoch.
All Set 1-2 checkpoints (BiLSTMEncoder_best.pt, CNN1DEncoder_best.pt) are therefore
likely not optimal.

### CRITICAL BUG 3: PatchTSTEncoder absent from src/models/__init__.py

Current __init__.py exports:
    CNN1DEncoder, BiLSTMEncoder, GRUEncoder, TCNEncoder,
    MultiScaleLSTMEncoder, MultiScaleLSTMForClassification,
    PatchCTGEncoder, PatchCTGForClassification
    -- PatchTSTEncoder IS MISSING --

train.py attempts: from src.models import PatchTSTEncoder
Since PatchTSTEncoder is not exported, this import FAILS SILENTLY every run.
PatchTST is never added to MODEL_REGISTRY via the universal script.
Running: python src/training/train.py --model patchtst
Will always error: "Unsupported model_name 'patchtst'"

### Additional Issues

[WARNING] build_encoder() uses a fragile 5-option try/except instantiation cascade.
  A genuine TypeError bug in an encoder constructor will be silently caught and
  the next fallback kwargs tried, potentially instantiating the wrong configuration.

[WARNING] GRU and TCN hyperparameters are hardcoded in build_encoder(), bypassing YAML.
  encoder_cls(hidden_dim=128, gru_hidden=64, num_layers=2, dropout=0.2)
  Changes to model: section of configs/local.yaml have zero effect on GRU and TCN.

---

## 6. Model Metrics Analysis and Cross-Comparison

### 6.1 Complete Benchmark Results

VALIDATION SET (5-fold CV):

Model     | Params | Val AUROC       | Val AUPRC       | Val F1         | Val Recall | Val Spec
----------|--------|-----------------|-----------------|----------------|------------|----------
CNN1D     | 135K   | TBD             | TBD             | TBD            | TBD        | TBD
BiLSTM    | 159K   | TBD             | TBD             | TBD            | 0.0%       | 100.0%
GRU       | 179K   | 0.7099 +/-0.047 | 0.2797 +/-0.043 | 0.158 +/-0.146 | 13.18%     | 95.29%
TCN       | 363K   | 0.6962 +/-0.043 | 0.2716 +/-0.040 | 0.047 +/-0.061 | 3.12%      | 97.98%
MS-LSTM   | 584K   | 0.7464 +/-0.070 | 0.3765 +/-0.106 | 0.383 +/-0.044 | 55.79%     | 73.58%
PatchCTG  | 671K   | 0.6738 +/-0.057 | 0.2899 +/-0.054 | 0.302 +/-0.083 | N/A        | 83.93%
PatchTST  | 685K   | 0.7456 +/-0.044 | 0.3615 +/-0.073 | 0.402 +/-0.074 | 51.17%     | 79.93%
                     ^^^^ WINNER                          ^^^^ WINNER

HELD-OUT TEST SET (Models 5-7 only):

Model     | Test AUROC      | Test AUPRC      | Test F1        | Test Recall | Test Spec
----------|-----------------|-----------------|----------------|-------------|----------
MS-LSTM   | 0.6064 +/-0.069 | 0.1288 +/-0.069 | 0.159 +/-0.062 | 36.84%      | 76.19%
PatchCTG  | 0.6825 +/-0.058 | 0.1044 +/-0.025 | 0.173 +/-0.041 | 40.00%      | 81.68%
PatchTST  | 0.7014 +/-0.021 | 0.1158 +/-0.011 | 0.194 +/-0.051 | 46.32%      | 80.90%
           ^^^^ WINNER

### 6.2 Key Metric Observations

OBSERVATION 1: AUPRC Collapse from Validation to Test (MAJOR RESEARCH CONCERN)

Model     | Val AUPRC | Test AUPRC | Drop
MS-LSTM   | 0.3765    | 0.1288     | -66%
PatchCTG  | 0.2899    | 0.1044     | -64%
PatchTST  | 0.3615    | 0.1158     | -68%

AUROC only drops 8-14% for the same models. This divergence means models maintain
discriminative rank ordering but generate poorly calibrated probability scores,
causing precision to collapse at higher recall thresholds. AUPRC is extremely
sensitive to calibration error in imbalanced datasets.

ACTION REQUIRED: Apply Platt scaling or isotonic regression calibration to raw
model probabilities before computing AUPRC on the test set. This is non-optional
for a clinical AI paper.

OBSERVATION 2: The pos_weight Fairness Problem

Models 1-4 used pos_weight=2.0. Models 5-7 used dynamic N_neg/N_pos weighting.
GRU's AUROC of 0.71 may have been suppressed by inadequate class weighting, not
by architectural limitations. The encoder selection comparison cannot be published
until weighting is equalized and both experiments are rerun.

OBSERVATION 3: Stale Checkpoint JSON Files Are Actively Misleading

checkpoints/bilstm_metrics.json shows val_auprc_mean = 0.800.
A BiLSTM AUPRC of 0.80 on a 95% majority class dataset is near-impossible and
is clearly an artifact of a tiny 2-fold validation run. Meanwhile the inference
log correctly shows the model predicted 0% recall (all negatives). These two
sources directly contradict each other. The JSON files must be deleted.

OBSERVATION 4: PatchTST is the Correct and Well-Supported Winner

Metric               | MS-LSTM | PatchTST | Winner
Val AUROC            | 0.7464  | 0.7456   | TIE
Test AUROC           | 0.6064  | 0.7014   | PatchTST (+0.095)
Val->Test AUROC drop | -0.140  | -0.044   | PatchTST (3.2x better generalization)
Inference speed      | ~1.8s   | ~0.75s   | PatchTST (2.4x faster per epoch)

The log's conclusion selecting PatchTST as the SOTA backbone is well-supported.

OBSERVATION 5: Sensitivity at 0.5 Threshold Is Clinically Unacceptable

In fetal distress detection, false negatives are catastrophic clinical outcomes.

Model          | Sensitivity at 0.5 | Clinical Interpretation
TCN            | 3.12%              | Misses 97 of 100 distress cases - unusable
GRU            | 13.18%             | Misses 87 of 100 distress cases - unacceptable
PatchTST (test)| 46.32%             | Still misses more than half of cases

The model_evaluation_plan.md explicitly requires reporting "Sensitivity @ 90%
Specificity" as the key clinical operating point. This metric appears NOWHERE
in the inference log or documentation. This is a critical gap.

---

## 7. Critical Inconsistencies Found

#  | Location                             | Issue                                              | Severity
---|--------------------------------------|----------------------------------------------------|----------
1  | pipeline.py L37 vs dataset_summary   | Distress stride: code=30sec, docs=2min             | CRITICAL
2  | train.py L277 vs evaluation plan     | pos_weight hardcoded 2.0, not dynamic N_neg/N_pos  | CRITICAL
3  | src/models/__init__.py               | PatchTSTEncoder not exported, silent fail in train  | CRITICAL
4  | train.py L427                        | Saves last-epoch weights, not best-AUROC epoch     | HIGH
5  | model_inferences_log.md              | CNN1D + BiLSTM AUROC/AUPRC TBD despite "complete"  | HIGH
6  | checkpoints/bilstm_metrics.json      | AUPRC=0.80 is stale artifact, statistically invalid | HIGH
7  | model_inferences_log.md vs docs      | MS-LSTM params: body=583,904 vs table=622,464      | MEDIUM
8  | cnn1d_encoder.py                     | __init__ takes no args - inconsistent encoder API  | MEDIUM
9  | workflow_and_plan.md                 | src/knowledge/figo.py referenced but dir missing   | MEDIUM
10 | dataset_summary.md                   | y_primary dtype claimed torch.long, cast to float32| LOW
11 | patchtst.py L12                      | import math unused                                 | LOW
12 | README.md                            | Placeholder URL and directory name unresolved      | LOW

---

## 8. Git and Workflow Compliance

**Rating: PARTIALLY COMPLIANT**

### Compliant

- Model-specific training scripts exist (train_multiscale_lstm.py, train_patchctg.py,
  train_patchtst.py) for Models 5-7
- Checkpoints saved with fold-level granularity (patchtst_fold_3_best.pth, etc.)
- AI_AGENT_RULES.md defines clear branch naming and commit conventions
- Preprocessing pipeline correctly frozen throughout the model benchmarking phase

### Non-Compliant

[CRITICAL] AI_AGENT_RULES.md Section 4 mandates the following branch structure:

  Member 1: CNN1D, BiLSTM     -> required branch: models_set_1
  Member 2: GRU, TCN          -> required branch: models_set_2
  Member 3: MS-LSTM, PatchCTG -> required branch: models_set_3
  Member 4: PatchTST          -> required branch: models_set_4

  Rule explicitly states: "No agent is allowed to merge directly to main."
  All 7 models appear to be committed directly on main. The branch workflow
  was bypassed entirely, making it impossible to audit which member implemented
  which encoder.

[WARNING] Stale early-run artifacts remain in checkpoints/:
  BiLSTMEncoder_best.pt, CNN1DEncoder_best.pt - these are from a pre-5-fold era
  and create ambiguity alongside the proper fold-level checkpoint files.

---

## 9. What Is Missing

Missing Component                     | Impact                                      | Priority
--------------------------------------|---------------------------------------------|----------
5-fold CV metrics for CNN1D, BiLSTM  | Cannot compare fairly with Models 3-7       | CRITICAL
Sensitivity @ 90% Specificity metric | Key clinical metric in plan, never computed | CRITICAL
src/knowledge/ directory with figo.py| Model 8 implementation is completely blocked| CRITICAL
ROC curves and PR curves per model   | Specified in evaluation plan; none exist    | MEDIUM
Probability calibration analysis      | Needed to explain 65% AUPRC collapse        | MEDIUM
Unit tests for preprocessing modules  | No safety net for filtering.py, features.py | MEDIUM
Confusion matrix per fold             | Required for clinical FN/FP analysis        | MEDIUM
Complete README.md                    | First file reviewers see - currently a stub | MEDIUM
Model 8 (Knowledge-Infused Framework)| Core research contribution                  | NEXT PHASE
Wilcoxon + DeLong statistical tests   | Required per AI_AGENT_RULES.md Section 2.2  | NEXT PHASE

---

## 10. Prioritized Action Items

=== MUST FIX: Before Phase 3 can be declared complete ===

1. Add PatchTSTEncoder to src/models/__init__.py
   Action: Add the following line to the file:
     from src.models.patchtst import PatchTSTEncoder, PatchTSTForClassification
   Also add PatchTSTEncoder and PatchTSTForClassification to __all__

2. Fix pos_weight in train.py - make it dynamic per fold
   Remove: pos_weight = torch.tensor([2.0]).to(device)
   Replace with (computed from the current fold's training indices before the epoch loop):
     n_pos = float(y_all[train_idx].sum())
     n_neg = float(len(train_idx)) - n_pos
     pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
   Pass pos_weight as a parameter into train_single_fold()

3. Fix checkpoint saving in train_single_fold() to save the best-AUROC epoch
   Add save_path parameter to the function signature. Inside the epoch validation loop:
     if metrics["auroc"] > best_val_auroc:
         best_val_auroc = metrics["auroc"]
         best_metrics = metrics.copy()
         if save_path:
             torch.save(model.state_dict(), save_path)
   Remove the torch.save() call from train_and_evaluate() after fold completion.

4. Run proper 5-fold CV for CNN1D and BiLSTM under the corrected training loop
   (dynamic pos_weight, best-epoch checkpointing) and populate all 7 metrics
   in model_inferences_log.md for Models 1 and 2.

5. Resolve stride discrepancy between pipeline.py and dataset_summary.md
   Verify which value (30 seconds or 2 minutes) was actually used to generate
   the frozen train_dataset.pt. Update the incorrect source to match reality.

=== SHOULD FIX: Before paper submission ===

6. Add Sensitivity @ 90% Specificity to calculate_metrics() in train.py
   Using sklearn.metrics.roc_curve:
     fpr, tpr, _ = roc_curve(y_true, y_probs)
     specificity_arr = 1.0 - fpr
     idx = np.where(specificity_arr >= 0.90)[0]
     sens_at_90spec = float(tpr[idx[-1]]) if len(idx) > 0 else 0.0

7. Apply probability calibration before reporting test AUPRC
   Use CalibratedClassifierCV(method='isotonic') fit on the validation set
   before evaluating on the held-out test set.

8. Delete stale checkpoint artifacts:
   checkpoints/BiLSTMEncoder_best.pt
   checkpoints/CNN1DEncoder_best.pt
   checkpoints/bilstm_metrics.json
   checkpoints/cnn1d_metrics.json

9. Fix CNN1DEncoder API for consistency:
   Add parameters: in_channels=2, seq_len=4800, latent_dim=128 to __init__()
   Add LayerNorm(latent_dim) after the final fc layer for output normalization.

10. Complete README.md with:
    - Project abstract and clinical motivation
    - Dataset citations with PhysioNet DOI and UCI repository URL
    - Condensed results table (val + test AUROC for all models)
    - Environment setup and reproduction instructions
    - Correct repository URL (replace placeholder)

---

## Final Verdict

Dimension                                              | Score   | Assessment
-------------------------------------------------------|---------|----------------------------------
Clinical Rigor (preprocessing, labeling, leakage)      | 9.0/10  | Exceptional - rarely achieved
Architecture Diversity and Technical Correctness       | 8.0/10  | All 7 encoders sound; API issues
Training Infrastructure                                | 6.0/10  | Three critical bugs in universal loop
Documentation Quality                                  | 8.5/10  | Outstanding; Set 1 completeness gap
Metric Completeness and Validity                       | 5.0/10  | TBD models; AUPRC collapse unexplained
Git and Workflow Compliance                            | 5.0/10  | Branch policy violated for all models
OVERALL                                                | 7.5/10  | Strong foundation - critical fixes needed

=== MOST IMPORTANT FINDING ===

PatchTST's apparent superiority over GRU and TCN may be PARTIALLY AN ARTIFACT of
asymmetric training conditions: PatchTST was trained with dynamic pos_weight
(~N_neg/N_pos per fold) while GRU and TCN used a fixed pos_weight=2.0 - roughly
a 10x difference in effective false-negative penalty. This confound must be
corrected before any encoder selection conclusion can be defended in a research paper.

The three CRITICAL bugs (PatchTST export, dynamic pos_weight, best-epoch checkpointing)
are straightforward code fixes that would significantly strengthen the scientific
integrity of Phase 3. The rest of the codebase is research-quality work.
