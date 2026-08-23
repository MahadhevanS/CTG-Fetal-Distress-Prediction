# Reproducing the delivered CRP model

The delivered model is the clinical-relational-pretrained (CRP) Crossformer,
**seed 42**, 5-fold patient-level ensemble.

## Seed -> directory map (verified against the MANIFESTs, 2026-08-21)

The three CRP replications wrote to **three different** checkpoint directories
and **three different** pretrained encoders. They did not overwrite one another,
and all three sets of weights are live on disk plus archived.

| Seed | Pretrained encoder | Fine-tuned checkpoints | Archive copy |
|------|--------------------|------------------------|--------------|
| 42 (**delivered**) | `checkpoints/clinical_relational/encoder_pretrained.pth` | `checkpoints/ctg_crossformer_crp/` | `archive/crp_seed42/` |
| 1 | `checkpoints/clinical_relational/encoder_s1.pth` | `checkpoints/repl_crp_s1/` | `archive/crp_seed1/` |
| 7 | `checkpoints/clinical_relational/encoder_s7.pth` | `checkpoints/repl_crp_s7/` | `archive/crp_seed7/` |

Paired baselines (no CRP pretraining) for seeds 1 and 7 are in
`checkpoints/repl_base_s1/` and `checkpoints/repl_base_s7/`.

`archive/crp_seed42/checkpoints/` and `archive/crp_seed1/checkpoints/` are
byte-identical (md5) to their live directories; `archive/crp_seed7/` holds a
full copy of the seed-7 weights.

## Reproduce commands

Taken verbatim from `archive/crp_seed{42,1,7}/MANIFEST.json`.

> **Do not reproduce a seed by changing `--seed` alone.** Three flags move
> together: `--seed`, `--out` (the encoder), and `--checkpoint_dir`. Changing
> only `--seed` points every run at seed 42's directory and its encoder, which
> overwrites the delivered model.

```bash
# seed 42 -- THE DELIVERED MODEL. Running this overwrites it.
python scripts/run_clinical_relational_pretrain.py --epochs 60 --seed 42 \
  --out checkpoints/clinical_relational/encoder_pretrained.pth
python src/models/train_ctg_crossformer.py --data_dir data/processed_mil/ \
  --checkpoint_dir checkpoints/ctg_crossformer_crp/ --fold_mode patient_level \
  --early_stop_mode nested --patience 15 --class_weight inverse_freq --seed 42 \
  --pretrained_encoder checkpoints/clinical_relational/encoder_pretrained.pth

# seed 1
python scripts/run_clinical_relational_pretrain.py --epochs 60 --seed 1 \
  --out checkpoints/clinical_relational/encoder_s1.pth
python src/models/train_ctg_crossformer.py --data_dir data/processed_mil/ \
  --checkpoint_dir checkpoints/repl_crp_s1/ --fold_mode patient_level \
  --early_stop_mode nested --patience 15 --class_weight inverse_freq --seed 1 \
  --pretrained_encoder checkpoints/clinical_relational/encoder_s1.pth

# seed 7
python scripts/run_clinical_relational_pretrain.py --epochs 60 --seed 7 \
  --out checkpoints/clinical_relational/encoder_s7.pth
python src/models/train_ctg_crossformer.py --data_dir data/processed_mil/ \
  --checkpoint_dir checkpoints/repl_crp_s7/ --fold_mode patient_level \
  --early_stop_mode nested --patience 15 --class_weight inverse_freq --seed 7 \
  --pretrained_encoder checkpoints/clinical_relational/encoder_s7.pth
```

Headline result across the three paired replications: **+0.0199 AUROC /
+0.0419 AUPRC** mean gain over the paired baseline, 3/3 runs positive.

## Evaluating without retraining

```bash
python scripts/eval_crp_metrics.py              # test-set metrics @ 0.5 and 0.30
python scripts/demo_inference.py 2045           # one raw record, end to end
python scripts/demo_patient_report.py           # TP / FP / TN worked examples
python scripts/demo_explainability.py           # faithfulness, rho=+0.405
```

`demo_explainability.py` defaults to the delivered CRP fold 1. The pre-CRP
inverse_freq baseline (rho=+0.461) is reachable with
`--ckpt checkpoints/ctg_crossformer_invfreq/ctg_crossformer_fold_1_best.pth
--thresh 0.365`.

## Known issue: per-fold calibration spread

The five CRP folds discriminate comparably (test AUROC 0.765-0.813) but emit
probabilities on very different scales (mean 0.018 to 0.466), because training
applies both sqrt-inverse oversampling and a full `n_neg/n_pos` pos_weight. The
raw ensemble over-predicts risk ~3.5x, and the deployed 0.30 threshold is not
stable across splits or seeds (49% sensitivity on test vs 34% on validation for
this model).

Investigated in full in
[`docs/calibration_test_plan.md`](calibration_test_plan.md). The fix — per-fold
Platt scaling fit on validation plus a validation-derived operating point — is
implemented in [`scripts/calibrate_crp.py`](../scripts/calibrate_crp.py) and is
**not** yet wired into the inference path.
