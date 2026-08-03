"""
Knowledge-Infused Multi-Task Framework — Training Script (Model 8)
===================================================================
Runs 5-Fold Stratified Patient-Level Cross-Validation for the
KnowledgeInfusedFramework across 4 ablation variants:

  Ablation Variants:
    distress_only  — L = L_distress (Model 7 baseline reproduction)
    plus_figo      — L = L_distress + λ₁·L_FIGO
    plus_features  — L = L_distress + λ₁·L_FIGO + λ₂·L_features
    full           — L = L_distress + λ₁·L_FIGO + λ₂·L_features + λ₃·L_knowledge

Usage:
    # Full Model 8 (all tasks):
    python src/training/train_knowledge_infused.py --config configs/model8_config.yaml

    # Dry-run (shape validation only):
    python src/training/train_knowledge_infused.py --dry_run

    # Single ablation variant:
    python src/training/train_knowledge_infused.py --ablation distress_only

    # All 4 ablation variants sequentially:
    python src/training/train_knowledge_infused.py --ablation all
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.patchtst import PatchTSTEncoder
from src.models.knowledge_infused_framework import KnowledgeInfusedFramework
from src.knowledge.figo import figo_rule_loss_normalized
from src.training.multi_task_dataset import (
    MultiTaskCTGDataset,
    load_all_multitask_splits,
    load_feature_scaler_stats,
)
from src.training.train import (
    calculate_metrics,
    create_patient_level_folds,
)

# Valid ablation variant names
ABLATION_VARIANTS = ["distress_only", "plus_figo", "plus_features", "full"]


# =============================================================================
# Loss Computation
# =============================================================================

def compute_multitask_loss(
    distress_logit: torch.Tensor,       # (B, 1)
    figo_logits: torch.Tensor,          # (B, 3)
    feature_preds_norm: torch.Tensor,   # (B, 8) — Z-normalized model outputs
    y_primary: torch.Tensor,            # (B,) float32
    y_figo: torch.Tensor,               # (B,) long
    y_features: torch.Tensor,           # (B, 8) float32 — normalized targets
    pos_weight: torch.Tensor,           # scalar tensor
    feature_means: torch.Tensor,        # (8,) for un-normalization
    feature_stds: torch.Tensor,         # (8,) for un-normalization
    lambda_figo: float = 0.3,
    lambda_features: float = 0.2,
    lambda_knowledge: float = 0.1,
    lambda_consistency: float = 0.5,
    ablation: str = "full",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the composite multi-task loss for the selected ablation variant.

    Args:
        distress_logit:      (B, 1) — raw binary logit from DistressHead
        figo_logits:         (B, 3) — raw FIGO logits from FIGOHead
        feature_preds_norm:  (B, 8) — Z-normalized predictions from ClinicalFeatureHead
        y_primary:           (B,) float32 — binary distress target
        y_figo:              (B,) long — FIGO class target
        y_features:          (B, 8) float32 — normalized feature targets
        pos_weight:          scalar tensor — dynamic class weight for BCEWithLogitsLoss
        feature_means:       (8,) for un-normalization in knowledge loss
        feature_stds:        (8,) for un-normalization in knowledge loss
        lambda_*:            loss weighting hyperparameters
        ablation:            variant name (see ABLATION_VARIANTS)

    Returns:
        Tuple of (total_loss_tensor, component_losses_dict)
    """
    device = distress_logit.device
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Primary Task: Binary Distress
    l_distress = criterion_bce(distress_logit.squeeze(-1), y_primary)
    component_losses = {"l_distress": l_distress.item()}
    total_loss = l_distress

    if ablation == "distress_only":
        return total_loss, component_losses

    # Auxiliary Task 1: FIGO 3-class Classification
    l_figo = nn.functional.cross_entropy(figo_logits, y_figo)
    component_losses["l_figo"] = l_figo.item()
    total_loss = total_loss + lambda_figo * l_figo

    if ablation == "plus_figo":
        return total_loss, component_losses

    # Auxiliary Task 2: Physiological Feature Regression (MSE on normalized targets)
    l_features = nn.functional.mse_loss(feature_preds_norm, y_features)
    component_losses["l_features"] = l_features.item()
    total_loss = total_loss + lambda_features * l_features

    if ablation == "plus_features":
        return total_loss, component_losses

    # Auxiliary Task 3: FIGO Knowledge Consistency Penalty
    # Uses figo_rule_loss_normalized to un-normalize before applying clinical thresholds.
    l_knowledge = figo_rule_loss_normalized(
        pred_features_norm=feature_preds_norm,
        pred_figo_logits=figo_logits,
        feature_means=feature_means.to(device),
        feature_stds=feature_stds.to(device),
        target_figo=None,  # CE already added above; don't double-count
        lambda_consistency=lambda_consistency,
    )
    component_losses["l_knowledge"] = l_knowledge.item()
    total_loss = total_loss + lambda_knowledge * l_knowledge

    return total_loss, component_losses


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_single_fold_multitask(
    model: KnowledgeInfusedFramework,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    pos_weight: torch.Tensor,
    feature_means: torch.Tensor,
    feature_stds: torch.Tensor,
    lambda_figo: float,
    lambda_features: float,
    lambda_knowledge: float,
    lambda_consistency: float,
    gradient_clip: float,
    save_path: Optional[str],
    ablation: str,
) -> Dict[str, float]:
    """
    Trains Model 8 on one fold and returns best validation metrics (by AUROC).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_auroc = -1.0
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()
        epoch_loss = 0.0
        for X_batch, yp_batch, yf_batch, yfeat_batch in train_loader:
            X_batch = X_batch.to(device)
            yp_batch = yp_batch.to(device)
            yf_batch = yf_batch.to(device)
            yfeat_batch = yfeat_batch.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                distress_logit, figo_logits, feature_preds = model(X_batch)
                total_loss, _ = compute_multitask_loss(
                    distress_logit=distress_logit,
                    figo_logits=figo_logits,
                    feature_preds_norm=feature_preds,
                    y_primary=yp_batch,
                    y_figo=yf_batch,
                    y_features=yfeat_batch,
                    pos_weight=pos_weight,
                    feature_means=feature_means,
                    feature_stds=feature_stds,
                    lambda_figo=lambda_figo,
                    lambda_features=lambda_features,
                    lambda_knowledge=lambda_knowledge,
                    lambda_consistency=lambda_consistency,
                    ablation=ablation,
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item() * len(yp_batch)

        scheduler.step()

        # --- Validation phase (evaluate primary task only: binary distress) ---
        model.eval()
        val_targets, val_probs = [], []
        with torch.no_grad():
            for X_batch, yp_batch, _, _ in val_loader:
                X_batch = X_batch.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    distress_logit, _, _ = model(X_batch)
                    probs = torch.sigmoid(distress_logit.squeeze(-1))
                val_targets.extend(yp_batch.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs)
        metrics = calculate_metrics(val_targets, val_probs)

        if metrics["auroc"] > best_val_auroc:
            best_val_auroc = metrics["auroc"]
            best_metrics = metrics.copy()
            if save_path:
                torch.save(model.state_dict(), save_path)

    return best_metrics


def run_dry_run(device: torch.device, ablation: str = "full") -> None:
    """Executes a forward + backward pass on dummy data for shape validation."""
    print(f"\n{'='*60}")
    print(f"[DRY RUN] Ablation: {ablation} | Device: {device}")
    print("Building KnowledgeInfusedFramework...")

    encoder = PatchTSTEncoder(
        in_channels=2, seq_len=4800, patch_len=16, stride=16,
        d_model=128, n_heads=4, n_layers=4, dropout=0.2, latent_dim=128,
    )
    model = KnowledgeInfusedFramework(encoder=encoder).to(device)
    print(f"Total trainable parameters: {model.param_count:,}")

    B = 8
    X = torch.randn(B, 2, 4800).to(device)
    yp = torch.randint(0, 2, (B,)).float().to(device)
    yf = torch.randint(0, 3, (B,)).long().to(device)
    yfeat = torch.randn(B, 8).to(device)
    pos_weight = torch.tensor([5.0]).to(device)
    feat_means = torch.zeros(8).to(device)
    feat_stds = torch.ones(8).to(device)

    distress_logit, figo_logits, feature_preds = model(X)
    print(f"distress_logit shape: {tuple(distress_logit.shape)}")
    print(f"figo_logits shape:    {tuple(figo_logits.shape)}")
    print(f"feature_preds shape:  {tuple(feature_preds.shape)}")

    total_loss, components = compute_multitask_loss(
        distress_logit, figo_logits, feature_preds,
        yp, yf, yfeat, pos_weight, feat_means, feat_stds,
        ablation=ablation,
    )
    total_loss.backward()
    print(f"Total loss: {total_loss.item():.4f} | Components: {components}")
    print("[OK] Dry run PASSED.")
    print(f"{'='*60}\n")


# =============================================================================
# Main Training Loop
# =============================================================================

def train_and_evaluate_model8(
    data_dir: str,
    scaler_path: str,
    save_dir: str,
    results_dir: str,
    ablation: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    k_folds: int,
    gradient_clip: float,
    lambda_figo: float,
    lambda_features: float,
    lambda_knowledge: float,
    lambda_consistency: float,
    dry_run: bool,
) -> Dict[str, Tuple[float, float]]:
    """Runs full 5-fold patient-level CV for a given ablation variant."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dry_run:
        run_dry_run(device=device, ablation=ablation)
        return {}

    print(f"\n{'='*65}")
    print(f" Model 8 Training | Ablation: {ablation.upper()} | Device: {device}")
    print(f" Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Folds: {k_folds}")
    print(f" λ_figo={lambda_figo}, λ_features={lambda_features}, λ_know={lambda_knowledge}")
    print(f"{'='*65}\n")

    # Load dataset
    try:
        dataset, patient_ids = load_all_multitask_splits(data_dir)
    except FileNotFoundError as e:
        print(f"[WARNING] {e}\n[FALLBACK] Running dry-run on dummy data.")
        run_dry_run(device=device, ablation=ablation)
        return {}

    # Load feature scaler stats for knowledge loss
    feat_means_np, feat_stds_np = load_feature_scaler_stats(scaler_path)
    feature_means = torch.tensor(feat_means_np, dtype=torch.float32)
    feature_stds = torch.tensor(feat_stds_np, dtype=torch.float32)

    y_all = dataset.y_primary
    folds = create_patient_level_folds(patient_ids, y_all, k_folds=k_folds)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    fold_results = {
        m: [] for m in ["accuracy", "auroc", "auprc", "f1", "precision",
                         "recall", "specificity", "sens_at_90spec"]
    }

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{k_folds} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")

        # Dynamic positive class weight
        n_pos = float(y_all[train_idx].sum().item())
        n_neg = float(len(train_idx)) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
        print(f" pos_weight: {pos_weight.item():.2f} (n_pos={int(n_pos)}, n_neg={int(n_neg)})")

        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

        # Build fresh model for each fold
        encoder = PatchTSTEncoder(
            in_channels=2, seq_len=4800, patch_len=16, stride=16,
            d_model=128, n_heads=4, n_layers=4, dropout=0.2, latent_dim=128,
        )
        model = KnowledgeInfusedFramework(encoder=encoder).to(device)
        fold_save = os.path.join(save_dir, f"model8_{ablation}_fold{fold_idx}_best.pth")

        metrics = train_single_fold_multitask(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            pos_weight=pos_weight,
            feature_means=feature_means,
            feature_stds=feature_stds,
            lambda_figo=lambda_figo,
            lambda_features=lambda_features,
            lambda_knowledge=lambda_knowledge,
            lambda_consistency=lambda_consistency,
            gradient_clip=gradient_clip,
            save_path=fold_save,
            ablation=ablation,
        )

        for m_key in fold_results:
            if m_key in metrics:
                fold_results[m_key].append(metrics[m_key])

        print(
            f" Fold {fold_idx} → AUROC: {metrics.get('auroc', 0):.4f} | "
            f"AUPRC: {metrics.get('auprc', 0):.4f} | "
            f"Sens@90%Spec: {metrics.get('sens_at_90spec', 0):.2f}%"
        )

    # Summarise
    summary = {}
    print(f"\n{'='*65}")
    print(f" FINAL {k_folds}-FOLD CV RESULTS | Model 8 — {ablation.upper()}")
    print(f"{'='*65}")
    for m_key, vals in fold_results.items():
        if not vals:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        summary[m_key] = (mean_val, std_val)
        unit = "%" if m_key in ["accuracy", "precision", "recall", "specificity", "sens_at_90spec"] else ""
        print(f" {m_key:<20}: {mean_val:.4f}{unit} ± {std_val:.4f}{unit}")
    print(f"{'='*65}\n")

    # Save results JSON
    results_path = os.path.join(results_dir, f"model8_{ablation}_cv_results.json")
    with open(results_path, "w") as f:
        json.dump({k: {"mean": v[0], "std": v[1]} for k, v in summary.items()}, f, indent=2)
    print(f"[Saved] CV results → {results_path}")
    return summary


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Model 8: Knowledge-Infused Multi-Task Framework Training"
    )
    parser.add_argument(
        "--config", type=str, default="configs/model8_config.yaml",
        help="Path to Model 8 YAML config file",
    )
    parser.add_argument(
        "--ablation", type=str, default="full",
        help=f"Ablation variant to run: {ABLATION_VARIANTS + ['all']}",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda_figo", type=float, default=None)
    parser.add_argument("--lambda_features", type=float, default=None)
    parser.add_argument("--lambda_knowledge", type=float, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Load config
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    train_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    path_cfg = cfg.get("paths", {})

    # CLI overrides take precedence over config
    data_dir = args.data_dir or path_cfg.get("data_dir", "data/processed/")
    epochs = args.epochs or train_cfg.get("epochs", 50)
    batch_size = args.batch_size or train_cfg.get("batch_size", 64)
    lr = args.lr or train_cfg.get("lr", 0.0001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    k_folds = train_cfg.get("k_folds", 5)
    gradient_clip = train_cfg.get("gradient_clip", 1.0)
    lambda_figo = args.lambda_figo or loss_cfg.get("lambda_figo", 0.3)
    lambda_features = args.lambda_features or loss_cfg.get("lambda_features", 0.2)
    lambda_knowledge = args.lambda_knowledge or loss_cfg.get("lambda_knowledge", 0.1)
    lambda_consistency = loss_cfg.get("lambda_consistency", 0.5)
    scaler_path = path_cfg.get("feature_scaler_path", "data/processed/feature_scaler.npz")
    save_dir = path_cfg.get("checkpoint_dir", "checkpoints/model8/")
    results_dir = path_cfg.get("results_dir", "checkpoints/model8/results/")

    # Determine which ablation variants to run
    if args.ablation.lower() == "all":
        variants_to_run = ABLATION_VARIANTS
    elif args.ablation in ABLATION_VARIANTS:
        variants_to_run = [args.ablation]
    else:
        raise ValueError(
            f"Unknown ablation '{args.ablation}'. Options: {ABLATION_VARIANTS + ['all']}"
        )

    all_results = {}
    for variant in variants_to_run:
        print(f"\n{'#'*65}")
        print(f"# Running Ablation Variant: {variant.upper()}")
        print(f"{'#'*65}")
        results = train_and_evaluate_model8(
            data_dir=data_dir,
            scaler_path=scaler_path,
            save_dir=save_dir,
            results_dir=results_dir,
            ablation=variant,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            k_folds=k_folds,
            gradient_clip=gradient_clip,
            lambda_figo=lambda_figo,
            lambda_features=lambda_features,
            lambda_knowledge=lambda_knowledge,
            lambda_consistency=lambda_consistency,
            dry_run=args.dry_run,
        )
        all_results[variant] = results

    if args.ablation.lower() == "all" and not args.dry_run and all_results:
        print("\n" + "="*65)
        print(" ABLATION COMPARISON SUMMARY (AUROC)")
        print("="*65)
        for variant, res in all_results.items():
            if res and "auroc" in res:
                m, s = res["auroc"]
                print(f" {variant:<20}: {m:.4f} ± {s:.4f}")
        print("="*65)


if __name__ == "__main__":
    main()
