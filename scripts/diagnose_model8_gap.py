"""
Diagnose the Model 8 distress_only vs. Standalone CrossFormer Gap (Option A)
==============================================================================
Across four full training runs -- sampling fix, head-capacity/wiring fix,
FIGO class-weighting fix, knowledge-loss warmup fix -- Model 8's `distress_only`
ablation has stayed stuck ~0.07-0.09 AUROC below the validated standalone
CrossFormer benchmark (0.8565 CV AUROC), despite distress_only sharing the same
encoder and (nominally) the same primary loss. None of the four tested fixes
explain it.

Three remaining, never-isolated differences between the two training setups:
    1. Scheduler:     standalone uses OneCycleLR (max_lr=0.0003, pct_start=0.1);
                       Model 8 defaults to warmup_cosine.
    2. EMA:            Model 8 has it on (decay=0.9995, used for validation);
                        standalone has none at all.
    3. Augmentation:   Model 8 applies physiological signal augmentation;
                        standalone uses none.

Rather than testing these one at a time via full 50-epoch x 5-fold x 6-ablation
runs (expensive, and we've already burned four cycles on single-variable
guesses), this script trains `distress_only` on a SINGLE fold for a SHORT
epoch budget under 5 configurations, reusing the real training code path
(train_single_fold_multitask) so the comparison reflects actual production
behaviour, not a reimplementation:

    0. baseline            -- current Model 8 defaults (warmup_cosine + EMA + aug)
    1. standalone_matched  -- onecycle (standalone's exact max_lr/pct_start) + no EMA + no aug
    2. onecycle_only       -- baseline, scheduler swapped to onecycle
    3. no_ema_only         -- baseline, EMA off
    4. no_aug_only         -- baseline, augmentation off

CAVEAT: EMA (decay=0.9995) has a half-life of roughly 10 epochs at this fold's
batch count -- a short run gives it only ~2 half-lives to converge, so if EMA
looks disadvantageous here it may partly reflect not having converged yet
rather than being disadvantageous over a full 50-epoch run. Read the no_ema_only
vs baseline comparison with that in mind, not as a final verdict.

This does NOT replace a full run -- it's a fast triage to find which factor(s)
are worth carrying into the next full, well-designed run, so we stop guessing
one variable at a time.

Usage:
    python scripts/diagnose_model8_gap.py --config configs/model8_crossformer_config.yaml --epochs 20
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.training.train_knowledge_infused import build_encoder, train_single_fold_multitask
from src.models.knowledge_infused_framework import KnowledgeInfusedFramework
from src.training.multi_task_dataset import load_all_multitask_splits, load_feature_scaler_stats
from src.training.train import create_patient_level_folds
from src.training.samplers import BalancedBatchSampler
from src.training.augmentation import PhysiologicalAugmentor


DIAGNOSTIC_CONFIGS = {
    "baseline": {
        "scheduler_cfg": {"type": "warmup_cosine", "warmup_epochs": 3, "eta_min": 1e-6},
        "ema_cfg": {"enabled": True, "decay": 0.9995, "use_for_validation": True},
        "augmentation": True,
    },
    "standalone_matched": {
        "scheduler_cfg": {"type": "onecycle", "onecycle_max_lr": 0.0003, "onecycle_pct_start": 0.1},
        "ema_cfg": {"enabled": False},
        "augmentation": False,
    },
    "onecycle_only": {
        "scheduler_cfg": {"type": "onecycle", "onecycle_max_lr": 0.0003, "onecycle_pct_start": 0.1},
        "ema_cfg": {"enabled": True, "decay": 0.9995, "use_for_validation": True},
        "augmentation": True,
    },
    "no_ema_only": {
        "scheduler_cfg": {"type": "warmup_cosine", "warmup_epochs": 3, "eta_min": 1e-6},
        "ema_cfg": {"enabled": False},
        "augmentation": True,
    },
    "no_aug_only": {
        "scheduler_cfg": {"type": "warmup_cosine", "warmup_epochs": 3, "eta_min": 1e-6},
        "ema_cfg": {"enabled": True, "decay": 0.9995, "use_for_validation": True},
        "augmentation": False,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model8_crossformer_config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/processed/")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Short epoch budget per diagnostic config (default 20, vs. 50 in a real run)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    backbone_cfg = cfg.get("backbone", {})
    heads_cfg = cfg.get("heads", {})
    training_cfg = cfg.get("training", {})
    sampling_cfg = cfg.get("sampling", {})
    path_cfg = cfg.get("paths", {})

    data_dir = args.data_dir or path_cfg.get("data_dir", "data/processed/")
    scaler_path = path_cfg.get("feature_scaler_path", "data/processed/feature_scaler.npz")
    lr = training_cfg.get("lr", 0.0003)
    weight_decay = training_cfg.get("weight_decay", 1e-5)
    batch_size = training_cfg.get("batch_size", 32)
    gradient_clip = training_cfg.get("gradient_clip", 1.0)

    print(f"Loading dataset from {data_dir} ...")
    dataset, patient_ids = load_all_multitask_splits(data_dir)
    feat_means_np, feat_stds_np = load_feature_scaler_stats(scaler_path)
    feature_means = torch.tensor(feat_means_np, dtype=torch.float32)
    feature_stds = torch.tensor(feat_stds_np, dtype=torch.float32)

    y_all = dataset.y_primary
    folds = create_patient_level_folds(patient_ids, y_all, k_folds=5)
    train_idx, val_idx = folds[0]  # single fold only -- this is a triage, not a full CV
    print(f"Using fold 1/5 only for triage: Train={len(train_idx)}, Val={len(val_idx)}")

    # pos_weight: match the config's own sampling method (kept CONSTANT across all
    # 5 diagnostic configs -- already tested/settled in a prior run, not what's
    # being isolated here)
    sampling_method = sampling_cfg.get("method", "default")
    if sampling_method == "balanced":
        pos_weight = torch.tensor([1.0])
        print("pos_weight: 1.00 (BalancedBatchSampler active, matching production config)")
    else:
        n_pos = float(y_all[train_idx].sum().item())
        n_neg = float(len(train_idx)) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
        print(f"pos_weight: {pos_weight.item():.2f}")

    train_sub = Subset(dataset, train_idx)
    val_sub = Subset(dataset, val_idx)
    if sampling_method == "balanced":
        train_y = dataset.y_primary[train_idx]
        batch_sampler = BalancedBatchSampler(train_y, batch_size=batch_size)
        train_loader = DataLoader(train_sub, batch_sampler=batch_sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {}
    for name, dcfg in DIAGNOSTIC_CONFIGS.items():
        print(f"\n{'='*70}\n[{name}] scheduler={dcfg['scheduler_cfg']['type']} | "
              f"ema={dcfg['ema_cfg']['enabled']} | augmentation={dcfg['augmentation']}\n{'='*70}")

        encoder = build_encoder(backbone_cfg)
        model = KnowledgeInfusedFramework(
            encoder=encoder,
            head_hidden_dim=heads_cfg.get("hidden_dim", 64),
            head_dropout=heads_cfg.get("dropout", 0.2),
        ).to(device)

        augmentor = PhysiologicalAugmentor(p=0.5) if dcfg["augmentation"] else None

        metrics = train_single_fold_multitask(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_indices=train_idx,
            val_dataset=val_sub,
            epochs=args.epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            pos_weight=pos_weight,
            feature_means=feature_means,
            feature_stds=feature_stds,
            lambda_figo=0.0, lambda_features=0.0, lambda_knowledge=0.0, lambda_consistency=0.5,
            gradient_clip=gradient_clip,
            save_path=None,
            ablation="distress_only",
            scheduler_cfg=dcfg["scheduler_cfg"],
            ema_cfg=dcfg["ema_cfg"],
            augmentor=augmentor,
            calibration_cfg={"optimize_threshold": False, "temperature_scaling": False},
            fold_idx=1,
        )
        results[name] = metrics
        print(f"[{name}] AUROC={metrics.get('auroc', 0):.4f} | F1={metrics.get('f1', 0):.4f} | "
              f"Sens@90Spec={metrics.get('sens_at_90spec', 0):.2f}%")

    print(f"\n{'='*70}\nSUMMARY -- {args.epochs}-epoch, single-fold triage (distress_only)\n{'='*70}")
    print(f"{'Config':<22}{'AUROC':>10}{'F1':>10}{'Sens@90Spec':>14}")
    for name, m in results.items():
        print(f"{name:<22}{m.get('auroc',0):>10.4f}{m.get('f1',0):>10.4f}{m.get('sens_at_90spec',0):>13.2f}%")
    print(f"\n(Reference: standalone CrossFormer full-run CV AUROC = 0.8565 -- not directly\n"
          f" comparable at {args.epochs} epochs/1 fold, but the RELATIVE ranking across\n"
          f" these 5 configs is the actual signal this triage is for.)")


if __name__ == "__main__":
    main()
