"""
Recover Lost CTG-CrossFormer Training Output From Saved Checkpoints
======================================================================
train_ctg_crossformer.py only prints metrics to stdout -- it never writes a
results file, so a lost terminal session loses the numbers even though the
per-fold model weights are saved to disk. This script reconstructs those
numbers WITHOUT retraining, by:

  1. Rebuilding the exact same 5-fold split. StratifiedGroupKFold is used
     with the default shuffle=False, which is fully deterministic given the
     same input order -- no seed is even needed to reproduce it, as long as
     data/processed/train_dataset.pt hasn't changed since the run.
  2. Loading each fold's saved best checkpoint
     (checkpoints/ctg_crossformer/ctg_crossformer_fold_{N}_best.pth) and
     evaluating it on that fold's own validation split -- recovering the
     original "Fold N Metrics" and "5-FOLD CV RESULTS" sections.
  3. Evaluating each fold's checkpoint on the held-out test set -- recovering
     the original "HELD-OUT TEST SET EVALUATION" section.

Only works if the checkpoint files from the run you want to recover are
still present and haven't been overwritten by a later run.

Usage:
    python scripts/recover_crossformer_results.py --config configs/ctg_crossformer_config.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification
from src.models.train_ctg_crossformer import CTGDataset, compute_metrics, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ctg_crossformer_config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/processed/")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ctg_crossformer/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    m_cfg = cfg["model"]
    t_cfg = cfg["training"]
    batch_size = t_cfg.get("batch_size", 32)

    train_pt = os.path.join(args.data_dir, "train_dataset.pt")
    test_pt = os.path.join(args.data_dir, "test_dataset.pt")

    print(f"Loading {train_pt} ...")
    data = torch.load(train_pt, weights_only=False)
    X = data["X"].numpy()
    y = data["y_primary"].numpy()
    patient_ids = np.array([m[0] for m in data["metadata"]])
    dataset = CTGDataset(X, y, patient_ids)

    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    splits = list(sgkf.split(X, y, groups=patient_ids))

    def build_model():
        encoder = CTGCrossformerEncoder(
            in_channels=2, seq_len=4800,
            cnn_channels=m_cfg.get("cnn_channels", 128),
            n_heads_cross=m_cfg.get("n_heads_cross", 4),
            n_heads_tf=m_cfg.get("n_heads_tf", 8),
            n_tf_layers=m_cfg.get("n_tf_layers", 4),
            d_ff=m_cfg.get("d_ff", 512),
            dropout=m_cfg.get("dropout", 0.1),
            latent_dim=m_cfg.get("latent_dim", 128),
        )
        return CTGCrossformerForClassification(
            encoder=encoder,
            hidden_dim=m_cfg.get("classifier_hidden_dim", 128),
            dropout=m_cfg.get("classifier_dropout", 0.3),
        ).to(device)

    keys = ["accuracy", "precision", "recall", "specificity", "f1", "auroc", "auprc"]
    fold_metrics = []
    missing_ckpts = []

    print(f"\n--- Recovering {n_splits}-Fold Stratified Patient-Level CV Results ---")
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        ckpt_path = os.path.join(args.checkpoint_dir, f"ctg_crossformer_fold_{fold}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"[MISSING] {ckpt_path} -- cannot recover fold {fold}")
            missing_ckpts.append(fold)
            continue

        val_sub = torch.utils.data.Subset(dataset, val_idx)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

        model = build_model()
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))

        metrics = evaluate(model, val_loader, device, desc=f"Fold {fold} [Val]")
        fold_metrics.append(metrics)
        print(
            f"Fold {fold} Recovered | AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f} | "
            f"Sens: {metrics['recall']:.4f} | Spec: {metrics['specificity']:.4f} | F1: {metrics['f1']:.4f}"
        )

    if missing_ckpts:
        print(f"\n[WARNING] Missing checkpoints for folds {missing_ckpts} -- "
              f"CV summary below is computed only from the {len(fold_metrics)} recovered folds.")

    if fold_metrics:
        print("\n================ RECOVERED 5-FOLD CROSS-VALIDATION RESULTS ================")
        for k in keys:
            vals = [fm[k] for fm in fold_metrics]
            print(f"{k.capitalize():<12}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # --- Held-out test set ---
    if os.path.exists(test_pt):
        print("\n================ RECOVERED HELD-OUT TEST SET EVALUATION ================")
        test_data = torch.load(test_pt, weights_only=False)
        test_ds = CTGDataset(test_data["X"].numpy(), test_data["y_primary"].numpy())
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        test_fold_metrics = []
        for fold in range(1, n_splits + 1):
            ckpt_path = os.path.join(args.checkpoint_dir, f"ctg_crossformer_fold_{fold}_best.pth")
            if not os.path.exists(ckpt_path):
                continue
            model = build_model()
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            tm = evaluate(model, test_loader, device, desc=f"Test Eval Fold {fold}")
            test_fold_metrics.append(tm)

        if test_fold_metrics:
            for k in keys:
                vals = [tm[k] for tm in test_fold_metrics]
                print(f"Test {k.capitalize():<12}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    else:
        print(f"\n[INFO] {test_pt} not found -- skipping held-out test recovery.")


if __name__ == "__main__":
    main()
