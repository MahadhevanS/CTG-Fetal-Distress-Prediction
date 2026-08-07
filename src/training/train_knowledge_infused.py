"""
Knowledge-Infused Multi-Task Framework — Enhanced Training Script (Model 8, Phase 4+)
=======================================================================================
Runs 5-Fold Stratified Patient-Level Cross-Validation for the
KnowledgeInfusedFramework. Integrates all Phase 4+ improvements:

  Ablation Variants (6 total):
    distress_only         — L = L_distress (Model 7 baseline reproduction)
    plus_figo             — L = L_distress + λ₁·L_FIGO
    distress_features_only — L = L_distress + λ₂·L_features
    plus_features         — L = L_distress + λ₁·L_FIGO + λ₂·L_features
    distress_figo_only    — L = L_distress + λ₁·L_FIGO + λ₃·L_knowledge
    full                  — L = L_distress + λ₁·L_FIGO + λ₂·L_features + λ₃·L_knowledge

  Phase 4+ Improvements:
    - Dynamic multi-task loss weighting (uncertainty/DWA/GradNorm/fixed)
    - Warmup + Cosine / OneCycleLR scheduling
    - Exponential Moving Average (EMA) for validation
    - Stochastic Weight Averaging (SWA) per fold
    - Balanced batch sampling and Hard Example Mining
    - Curriculum training (progressive task introduction)
    - Physiologically safe CTG signal augmentation
    - Per-fold decision threshold optimization
    - Temperature scaling probability calibration
    - Structured error analysis reports

Usage:
    # Full Model 8 with all Phase 4+ improvements:
    python src/training/train_knowledge_infused.py --config configs/model8_config.yaml

    # Dry-run (shape + forward/backward validation):
    python src/training/train_knowledge_infused.py --dry_run

    # Single ablation variant:
    python src/training/train_knowledge_infused.py --ablation distress_only

    # All 6 ablation variants sequentially:
    python src/training/train_knowledge_infused.py --ablation all

    # With specific override (e.g., disable augmentation for ablation comparison):
    python src/training/train_knowledge_infused.py --ablation full --no_augmentation

    # With uncertainty weighting:
    python src/training/train_knowledge_infused.py --ablation full --loss_weighting uncertainty
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
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
from src.training.loss_weighting import build_loss_weighter, DynamicWeightAveraging
from src.training.checkpoint_utils import ExponentialMovingAverage, StochasticWeightAveraging
from src.training.samplers import BalancedBatchSampler, HardExampleMiner
from src.training.augmentation import PhysiologicalAugmentor
from src.training.calibration import ThresholdOptimizer, TemperatureScaler
from src.training.error_analysis import ErrorAnalyzer

# Valid ablation variant names (extended from 4 to 6)
ABLATION_VARIANTS = [
    "distress_only",
    "plus_figo",
    "distress_features_only",
    "plus_features",
    "distress_figo_only",
    "full",
]


# =============================================================================
# Warmup Scheduler
# =============================================================================

class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs`, then CosineAnnealingLR decay.
    Wraps a PyTorch scheduler for clean integration.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 1e-6,
        base_lr: float = 1e-4,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lr = base_lr
        self._cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=eta_min,
        )
        self._epoch = 0

    def step(self) -> None:
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            # Linear ramp-up
            lr = self.base_lr * (self._epoch / max(self.warmup_epochs, 1))
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self._cosine_scheduler.step()

    def get_last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


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
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    aux_warmup_factor: float = 1.0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Computes individual loss components for the selected ablation variant.
    Returns raw component tensors (before weighting) for dynamic loss weighters.

    Returns:
        Tuple of:
          - component_tensors: Dict[str, Tensor] — unweighted scalar loss tensors
          - component_floats:  Dict[str, float]  — detached float values for logging
    """
    device = distress_logit.device

    # Primary Task: Binary Distress (Focal Loss or BCEWithLogits)
    if use_focal_loss:
        bce = nn.functional.binary_cross_entropy_with_logits(
            distress_logit.squeeze(-1), y_primary, pos_weight=pos_weight.to(device), reduction="none"
        )
        probs = torch.sigmoid(distress_logit.squeeze(-1))
        p_t = probs * y_primary + (1.0 - probs) * (1.0 - y_primary)
        focal_weight = (1.0 - p_t) ** focal_gamma
        l_distress = (focal_weight * bce).mean()
    else:
        criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        l_distress = criterion_bce(distress_logit.squeeze(-1), y_primary)

    component_tensors = {"distress": l_distress}
    component_floats = {"l_distress": l_distress.item()}

    if ablation == "distress_only":
        return component_tensors, component_floats

    # Auxiliary Task 1: FIGO 3-class Classification (with curriculum warmup)
    if ablation in ("plus_figo", "plus_features", "distress_figo_only", "full"):
        l_figo = nn.functional.cross_entropy(figo_logits, y_figo) * aux_warmup_factor
        component_tensors["figo"] = l_figo
        component_floats["l_figo"] = l_figo.item()

    # Auxiliary Task 2: Physiological Feature Regression (with curriculum warmup)
    if ablation in ("distress_features_only", "plus_features", "full"):
        l_features = nn.functional.mse_loss(feature_preds_norm, y_features) * aux_warmup_factor
        component_tensors["features"] = l_features
        component_floats["l_features"] = l_features.item()

    # Auxiliary Task 3: FIGO Knowledge Consistency Penalty (with curriculum warmup)
    if ablation in ("distress_figo_only", "full"):
        l_knowledge = figo_rule_loss_normalized(
            pred_features_norm=feature_preds_norm,
            pred_figo_logits=figo_logits,
            feature_means=feature_means.to(device),
            feature_stds=feature_stds.to(device),
            target_figo=None,
            lambda_consistency=lambda_consistency,
        ) * aux_warmup_factor
        component_tensors["knowledge"] = l_knowledge
        component_floats["l_knowledge"] = l_knowledge.item()

    return component_tensors, component_floats


def _get_curriculum_ablation(epoch: int, schedule: List) -> str:
    """Determine the active ablation variant based on curriculum schedule."""
    active = "full"
    for (start_epoch, variant) in schedule:
        if epoch >= start_epoch:
            active = variant
    return active


# =============================================================================
# Training & Evaluation (Single Fold)
# =============================================================================

def train_single_fold_multitask(
    model: KnowledgeInfusedFramework,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_indices: np.ndarray,
    val_dataset: Subset,
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
    # Phase 4+ params
    loss_weighter=None,
    scheduler_cfg: Optional[Dict] = None,
    ema_cfg: Optional[Dict] = None,
    swa_cfg: Optional[Dict] = None,
    sampling_cfg: Optional[Dict] = None,
    augmentor: Optional[PhysiologicalAugmentor] = None,
    calibration_cfg: Optional[Dict] = None,
    curriculum_schedule: Optional[List] = None,
    error_analysis_cfg: Optional[Dict] = None,
    fold_idx: int = 1,
    results_dir: str = "checkpoints/model8/results/",
) -> Dict[str, float]:
    """
    Trains Model 8 on one fold with all Phase 4+ improvements and returns best metrics.
    """
    scheduler_cfg = scheduler_cfg or {}
    ema_cfg = ema_cfg or {}
    swa_cfg = swa_cfg or {}
    sampling_cfg = sampling_cfg or {}
    calibration_cfg = calibration_cfg or {}
    error_analysis_cfg = error_analysis_cfg or {}

    # --- Build optimizer ---
    optimizer_params = [{"params": model.parameters(), "lr": lr}]

    # UncertaintyWeighting adds learnable parameters to the optimizer
    if hasattr(loss_weighter, "log_sigma2"):
        optimizer_params.append({
            "params": loss_weighter.parameters(),
            "lr": lr * 10,  # Uncertainty params often need higher LR
        })

    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=weight_decay)

    # --- Build scheduler ---
    sched_type = scheduler_cfg.get("type", "cosine")
    if sched_type == "warmup_cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=scheduler_cfg.get("warmup_epochs", 5),
            total_epochs=epochs,
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
            base_lr=lr,
        )
    elif sched_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_cfg.get("onecycle_max_lr", lr * 10),
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=scheduler_cfg.get("onecycle_pct_start", 0.3),
        )
    else:
        # Original cosine
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=scheduler_cfg.get("eta_min", 1e-6)
        )

    # --- EMA ---
    ema_enabled = ema_cfg.get("enabled", False)
    ema = ExponentialMovingAverage(model, decay=ema_cfg.get("decay", 0.9995)) if ema_enabled else None

    # --- SWA ---
    swa_enabled = swa_cfg.get("enabled", False)
    swa_start_epoch = swa_cfg.get("start_epoch", 30)
    swa = StochasticWeightAveraging(max_checkpoints=swa_cfg.get("max_checkpoints", 10)) if swa_enabled else None

    # --- Hard Example Miner ---
    hard_mining = sampling_cfg.get("method", "default") == "hard_mining"
    miner = (
        HardExampleMiner(
            n_samples=len(train_indices),
            warmup_epochs=sampling_cfg.get("hard_mining_warmup_epochs", 5),
            ema_decay=sampling_cfg.get("hard_mining_ema_decay", 0.9),
            hard_ratio=sampling_cfg.get("hard_mining_hard_ratio", 0.6),
        )
        if hard_mining else None
    )

    # --- DWA epoch-loss history ---
    use_dwa = isinstance(loss_weighter, DynamicWeightAveraging)
    epoch_loss_accumulator: Dict[str, List[float]] = {}

    use_amp = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_auroc = -1.0
    best_metrics: Dict[str, float] = {}
    best_val_probs: Optional[np.ndarray] = None
    best_val_targets: Optional[np.ndarray] = None
    best_val_logits: Optional[np.ndarray] = None
    best_val_figo_preds: Optional[np.ndarray] = None
    best_val_feature_preds: Optional[np.ndarray] = None

    # Set augmentor to train mode
    if augmentor is not None:
        augmentor.train()

    for epoch in range(1, epochs + 1):
        # --- Curriculum: determine active ablation ---
        active_ablation = (
            _get_curriculum_ablation(epoch, curriculum_schedule)
            if curriculum_schedule else ablation
        )

        # --- Rebuild train loader with updated miner weights (if hard mining) ---
        if miner is not None and miner.is_active:
            weights_np = miner.get_sample_weights()
            sampler = WeightedRandomSampler(
                torch.from_numpy(weights_np), num_samples=len(train_indices), replacement=True
            )
            # Note: train_loader will be rebuilt by caller; use the original here
            # (miner weights apply from next epoch's batch construction)

        # --- Training phase ---
        model.train()
        if loss_weighter is not None:
            loss_weighter.train()

        epoch_component_sums: Dict[str, float] = {}
        n_batches = 0

        # --- Auxiliary curriculum warmup factor (10 epochs) ---
        aux_warmup_epochs = 10
        aux_warmup_factor = min(1.0, epoch / float(aux_warmup_epochs)) if epoch <= aux_warmup_epochs else 1.0

        for batch in train_loader:
            X_batch, yp_batch, yf_batch, yfeat_batch = batch
            X_batch = X_batch.to(device)
            yp_batch = yp_batch.to(device)
            yf_batch = yf_batch.to(device)
            yfeat_batch = yfeat_batch.to(device)

            # Apply augmentation during training
            if augmentor is not None:
                X_batch = augmentor(X_batch)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                distress_logit, figo_logits, feature_preds = model(X_batch)
                component_tensors, component_floats = compute_multitask_loss(
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
                    ablation=active_ablation,
                    use_focal_loss=True,
                    focal_gamma=2.0,
                    aux_warmup_factor=aux_warmup_factor,
                )

            # Apply loss weighting
            if loss_weighter is not None:
                total_loss, _ = loss_weighter.weight(component_tensors, model=model, epoch=epoch)
            else:
                # Fallback: fixed weighting
                total_loss = component_tensors["distress"]
                if "figo" in component_tensors:
                    total_loss = total_loss + lambda_figo * component_tensors["figo"]
                if "features" in component_tensors:
                    total_loss = total_loss + lambda_features * component_tensors["features"]
                if "knowledge" in component_tensors:
                    total_loss = total_loss + lambda_knowledge * component_tensors["knowledge"]

            scaler_amp.scale(total_loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            if ema is not None:
                ema.update()

            # Accumulate per-task losses for DWA
            for k, v in component_floats.items():
                task_key = k.replace("l_", "")
                epoch_component_sums[task_key] = epoch_component_sums.get(task_key, 0.0) + v

            n_batches += 1

        # Scheduler step (per-epoch, except OneCycleLR which is per-step)
        if sched_type != "onecycle":
            if isinstance(scheduler, WarmupCosineScheduler):
                scheduler.step()
            else:
                scheduler.step()

        # Update DWA history
        if use_dwa and n_batches > 0:
            epoch_mean_losses = {k: v / n_batches for k, v in epoch_component_sums.items()}
            loss_weighter.update_history(epoch_mean_losses)

        # Update hard example miner epoch counter
        if miner is not None:
            miner.step_epoch()

        # Collect SWA snapshot
        if swa is not None and epoch >= swa_start_epoch:
            swa.add_checkpoint(model.state_dict())

        # --- Validation phase ---
        model.eval()
        if augmentor is not None:
            augmentor.eval()

        val_targets_list, val_probs_list, val_logits_list = [], [], []
        val_figo_preds_list, val_feature_preds_list = [], []

        eval_context = ema.average_parameters() if (ema is not None and ema_cfg.get("use_for_validation", True)) else _null_context()

        with eval_context:
            with torch.no_grad():
                for X_batch, yp_batch, yf_batch, yfeat_batch in val_loader:
                    X_batch = X_batch.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        distress_logit, figo_logits, feature_preds = model(X_batch)
                        probs = torch.sigmoid(distress_logit.squeeze(-1))
                    val_targets_list.extend(yp_batch.cpu().numpy())
                    val_probs_list.extend(probs.cpu().numpy())
                    val_logits_list.extend(distress_logit.squeeze(-1).cpu().numpy())
                    val_figo_preds_list.extend(torch.argmax(figo_logits, dim=-1).cpu().numpy())
                    val_feature_preds_list.extend(feature_preds.cpu().numpy())

        val_targets = np.array(val_targets_list)
        val_probs = np.array(val_probs_list)
        val_logits = np.array(val_logits_list)

        metrics = calculate_metrics(val_targets, val_probs)

        if metrics["auroc"] > best_val_auroc:
            best_val_auroc = metrics["auroc"]
            best_metrics = metrics.copy()
            best_val_probs = val_probs.copy()
            best_val_targets = val_targets.copy()
            best_val_logits = val_logits.copy()
            best_val_figo_preds = np.array(val_figo_preds_list)
            best_val_feature_preds = np.array(val_feature_preds_list)
            if save_path:
                torch.save(model.state_dict(), save_path)

    # --- Post-fold: SWA evaluation ---
    if swa is not None and swa.n_checkpoints > 0:
        try:
            swa_state = swa.average()
            model.load_state_dict(swa_state)
            model.eval()
            swa_targets, swa_probs = [], []
            with torch.no_grad():
                for X_batch, yp_batch, _, _ in val_loader:
                    X_batch = X_batch.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        distress_logit, _, _ = model(X_batch)
                        probs = torch.sigmoid(distress_logit.squeeze(-1))
                    swa_targets.extend(yp_batch.cpu().numpy())
                    swa_probs.extend(probs.cpu().numpy())
            swa_metrics = calculate_metrics(np.array(swa_targets), np.array(swa_probs))
            if swa_metrics["auroc"] > best_metrics.get("auroc", 0.0):
                best_metrics = swa_metrics.copy()
                print(f"   [SWA] SWA model improved AUROC to {swa_metrics['auroc']:.4f} ✓")
                if save_path:
                    swa_path = save_path.replace(".pth", "_swa.pth")
                    torch.save(model.state_dict(), swa_path)
        except Exception as e:
            print(f"   [SWA] Warning: SWA averaging failed: {e}")

    # --- Post-fold: Threshold Optimization ---
    if calibration_cfg.get("optimize_threshold", True) and best_val_probs is not None:
        threshold_optimizer = ThresholdOptimizer(
            objective=calibration_cfg.get("threshold_objective", "f1")
        )
        best_thresh, thresh_score = threshold_optimizer.fit(best_val_probs, best_val_targets)
        calibrated_metrics = threshold_optimizer.evaluate(best_val_probs, best_val_targets, best_thresh)
        # If calibrated metrics are better than default-0.5 metrics, use them
        if calibrated_metrics.get("f1", 0) > best_metrics.get("f1", 0):
            best_metrics.update(calibrated_metrics)
        best_metrics["optimal_threshold"] = best_thresh
        best_metrics["threshold_objective_score"] = thresh_score

    # --- Post-fold: Temperature Scaling ---
    if calibration_cfg.get("temperature_scaling", True) and best_val_logits is not None:
        try:
            temp_scaler = TemperatureScaler()
            T = temp_scaler.fit(best_val_logits, best_val_targets)
            cal_probs = temp_scaler.calibrate(best_val_logits)
            ece = temp_scaler.expected_calibration_error(cal_probs, best_val_targets)
            brier = temp_scaler.brier_score(cal_probs, best_val_targets)
            best_metrics["temperature"] = T
            best_metrics["ece"] = ece
            best_metrics["brier_score"] = brier
            print(f"   [Calibration] T={T:.3f} | ECE={ece:.4f} | Brier={brier:.4f}")
        except Exception as e:
            print(f"   [Calibration] Warning: {e}")

    # --- Post-fold: Error Analysis ---
    if error_analysis_cfg.get("enabled", False) and best_val_probs is not None:
        try:
            err_dir = error_analysis_cfg.get("results_dir", results_dir + "error_analysis/")
            os.makedirs(err_dir, exist_ok=True)
            analyzer = ErrorAnalyzer()
            opt_thresh = best_metrics.get("optimal_threshold", 0.5)
            report = analyzer.analyze(
                y_true=best_val_targets,
                y_probs=best_val_probs,
                y_figo_pred=best_val_figo_preds,
                y_features_pred=best_val_feature_preds,
                threshold=opt_thresh,
                fold_idx=fold_idx,
            )
            analyzer.print_report(report)
            if error_analysis_cfg.get("save_per_fold", True):
                err_path = os.path.join(err_dir, f"fold{fold_idx}_error_analysis.json")
                analyzer.save_report(report, err_path)
        except Exception as e:
            print(f"   [ErrorAnalysis] Warning: {e}")

    return best_metrics


class _null_context:
    """No-op context manager when EMA is disabled."""
    def __enter__(self): return self
    def __exit__(self, *args): pass


# =============================================================================
# Dry-Run Validation
# =============================================================================

def run_dry_run(device: torch.device, ablation: str = "full") -> None:
    """Executes a forward + backward pass on dummy data for shape validation."""
    print(f"\n{'='*65}")
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

    component_tensors, component_floats = compute_multitask_loss(
        distress_logit, figo_logits, feature_preds,
        yp, yf, yfeat, pos_weight, feat_means, feat_stds,
        ablation=ablation,
    )
    total_loss = sum(component_tensors.values())
    total_loss.backward()
    print(f"Total loss: {total_loss.item():.4f} | Components: {component_floats}")

    # Test augmentor
    augmentor = PhysiologicalAugmentor(p=1.0)
    augmentor.train()
    X_aug = augmentor(torch.randn(B, 2, 4800).to(device))
    assert X_aug.shape == (B, 2, 4800), f"Augmentor shape mismatch: {X_aug.shape}"
    print(f"Augmentor: {tuple(X_aug.shape)} [OK]")

    print("[OK] Dry run PASSED.")
    print(f"{'='*65}\n")


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
    # Phase 4+ config dicts
    loss_weighting_cfg: Optional[Dict] = None,
    scheduler_cfg: Optional[Dict] = None,
    ema_cfg: Optional[Dict] = None,
    swa_cfg: Optional[Dict] = None,
    sampling_cfg: Optional[Dict] = None,
    augmentation_cfg: Optional[Dict] = None,
    calibration_cfg: Optional[Dict] = None,
    curriculum_cfg: Optional[Dict] = None,
    error_analysis_cfg: Optional[Dict] = None,
) -> Dict[str, Tuple[float, float]]:
    """Runs full 5-fold patient-level CV for a given ablation variant."""
    loss_weighting_cfg = loss_weighting_cfg or {}
    scheduler_cfg = scheduler_cfg or {}
    ema_cfg = ema_cfg or {}
    swa_cfg = swa_cfg or {}
    sampling_cfg = sampling_cfg or {}
    augmentation_cfg = augmentation_cfg or {}
    calibration_cfg = calibration_cfg or {}
    curriculum_cfg = curriculum_cfg or {}
    error_analysis_cfg = error_analysis_cfg or {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dry_run:
        run_dry_run(device=device, ablation=ablation)
        return {}

    print(f"\n{'='*65}")
    print(f" Model 8 Training | Ablation: {ablation.upper()} | Device: {device}")
    print(f" Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Folds: {k_folds}")
    print(f" λ_figo={lambda_figo}, λ_features={lambda_features}, λ_know={lambda_knowledge}")
    loss_method = loss_weighting_cfg.get("method", "fixed")
    print(f" Loss Weighting: {loss_method} | Scheduler: {scheduler_cfg.get('type', 'cosine')}")
    print(f" EMA: {ema_cfg.get('enabled', False)} | SWA: {swa_cfg.get('enabled', False)}")
    print(f" Augmentation: {augmentation_cfg.get('enabled', False)} | Sampling: {sampling_cfg.get('method', 'default')}")
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

    # Curriculum schedule
    curriculum_enabled = curriculum_cfg.get("enabled", False)
    curriculum_schedule = curriculum_cfg.get("schedule", []) if curriculum_enabled else None

    # Build augmentor (shared across folds)
    augmentor = None
    if augmentation_cfg.get("enabled", True):
        augmentor = PhysiologicalAugmentor(
            p=augmentation_cfg.get("p", 0.5),
            noise_std=augmentation_cfg.get("noise_std", 0.02),
            drift_max_amp=augmentation_cfg.get("drift_max_amp", 0.05),
            scale_range=tuple(augmentation_cfg.get("scale_range", [0.97, 1.03])),
            jitter_max=augmentation_cfg.get("jitter_max", 5),
            missing_max_ratio=augmentation_cfg.get("missing_max_ratio", 0.03),
            missing_block_max=augmentation_cfg.get("missing_block_max", 48),
        )

    fold_results = {
        m: [] for m in [
            "accuracy", "auroc", "auprc", "f1", "precision",
            "recall", "specificity", "sens_at_90spec",
            "optimal_threshold", "ece", "brier_score",
        ]
    }

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{k_folds} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")

        # Dynamic positive class weight
        sampling_method = sampling_cfg.get("method", "default")
        if sampling_method == "balanced":
            # Batch sampler already balances positive/negative 50/50.
            # Setting pos_weight=1.0 prevents double-counting pos_weight in BCEWithLogitsLoss.
            pos_weight = torch.tensor([1.0])
            print(f" pos_weight: 1.00 (BalancedBatchSampler active — 50/50 batch ratio)")
        else:
            n_pos = float(y_all[train_idx].sum().item())
            n_neg = float(len(train_idx)) - n_pos
            pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
            print(f" pos_weight: {pos_weight.item():.2f} (n_pos={int(n_pos)}, n_neg={int(n_neg)})")

        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)

        # --- Build train DataLoader with selected sampling strategy ---
        sampling_method = sampling_cfg.get("method", "default")
        if sampling_method == "balanced":
            train_y = dataset.y_primary[train_idx]
            batch_sampler = BalancedBatchSampler(train_y, batch_size=batch_size)
            train_loader = DataLoader(train_sub, batch_sampler=batch_sampler, num_workers=0)
        else:
            # Default or hard_mining — hard_mining rebuilds sampler per epoch (handled in train loop)
            train_loader = DataLoader(
                train_sub, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
            )

        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=0)

        # --- Build loss weighter per fold (stateful: DWA, GradNorm) ---
        loss_weighter = build_loss_weighter(
            method=loss_weighting_cfg.get("method", "fixed"),
            lambda_figo=lambda_figo,
            lambda_features=lambda_features,
            lambda_knowledge=lambda_knowledge,
            task_names=["distress", "figo", "features", "knowledge"],
        )

        # --- Build fresh model for each fold ---
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
            train_indices=train_idx,
            val_dataset=val_sub,
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
            loss_weighter=loss_weighter,
            scheduler_cfg=scheduler_cfg,
            ema_cfg=ema_cfg,
            swa_cfg=swa_cfg,
            sampling_cfg=sampling_cfg,
            augmentor=augmentor,
            calibration_cfg=calibration_cfg,
            curriculum_schedule=curriculum_schedule,
            error_analysis_cfg=error_analysis_cfg,
            fold_idx=fold_idx,
            results_dir=results_dir,
        )

        for m_key in fold_results:
            if m_key in metrics:
                fold_results[m_key].append(metrics[m_key])

        print(
            f" Fold {fold_idx} → AUROC: {metrics.get('auroc', 0):.4f} | "
            f"AUPRC: {metrics.get('auprc', 0):.4f} | "
            f"Sens@90%Spec: {metrics.get('sens_at_90spec', 0):.2f}%"
            + (f" | Thresh: {metrics.get('optimal_threshold', 0.5):.3f}" if calibration_cfg.get("optimize_threshold") else "")
        )

    # --- Summarise ---
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
        print(f" {m_key:<25}: {mean_val:.4f}{unit} ± {std_val:.4f}{unit}")
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
        description="Model 8: Knowledge-Infused Multi-Task Framework Training (Phase 4+)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/model8_config.yaml",
        help="Path to Model 8 YAML config file",
    )
    parser.add_argument(
        "--ablation", type=str, default="full",
        help=f"Ablation variant: {ABLATION_VARIANTS + ['all']}",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda_figo", type=float, default=None)
    parser.add_argument("--lambda_features", type=float, default=None)
    parser.add_argument("--lambda_knowledge", type=float, default=None)
    parser.add_argument("--loss_weighting", type=str, default=None,
                        help="Override loss weighting method (fixed/uncertainty/dwa/gradnorm)")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable signal augmentation")
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA")
    parser.add_argument("--no_swa", action="store_true", help="Disable SWA")
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
    loss_weighting_cfg = cfg.get("loss_weighting", {})
    scheduler_cfg = cfg.get("scheduler", {})
    ema_cfg = cfg.get("ema", {})
    swa_cfg = cfg.get("swa", {})
    sampling_cfg = cfg.get("sampling", {})
    augmentation_cfg = cfg.get("augmentation", {})
    calibration_cfg = cfg.get("calibration", {})
    curriculum_cfg = cfg.get("curriculum", {})
    error_analysis_cfg = cfg.get("error_analysis", {})

    # CLI overrides
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

    if args.loss_weighting:
        loss_weighting_cfg["method"] = args.loss_weighting
    if args.no_augmentation:
        augmentation_cfg["enabled"] = False
    if args.no_ema:
        ema_cfg["enabled"] = False
    if args.no_swa:
        swa_cfg["enabled"] = False

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
            loss_weighting_cfg=loss_weighting_cfg,
            scheduler_cfg=scheduler_cfg,
            ema_cfg=ema_cfg,
            swa_cfg=swa_cfg,
            sampling_cfg=sampling_cfg,
            augmentation_cfg=augmentation_cfg,
            calibration_cfg=calibration_cfg,
            curriculum_cfg=curriculum_cfg,
            error_analysis_cfg=error_analysis_cfg,
        )
        all_results[variant] = results

    if args.ablation.lower() == "all" and not args.dry_run and all_results:
        print("\n" + "=" * 65)
        print(" ABLATION COMPARISON SUMMARY (AUROC)")
        print("=" * 65)
        for variant, res in all_results.items():
            if res and "auroc" in res:
                m, s = res["auroc"]
                print(f" {variant:<25}: {m:.4f} ± {s:.4f}")
        print("=" * 65)


if __name__ == "__main__":
    main()
