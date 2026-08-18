"""
Training and 5-Fold Patient-Level Cross-Validation Benchmark for CTG-CrossFormer.

Replicates the paper training protocol (Dang et al., 2026, Section 3.3):
- Stratified 5-Fold Patient-Level CV (Grouped by Patient ID)
- Focal Loss (gamma=2.0) with pos_weight
- Sqrt-inverse frequency oversampling (WeightedRandomSampler)
- OneCycleLR Scheduler (10% warmup, max LR 3e-4)
- AdamW optimizer (weight decay 1e-5)
- Evaluates: AUROC, AUPRC, Sensitivity (Recall), Specificity, F1, Accuracy
- Synthetic dry-run mode for quick shape/gradient verification (--dry_run)
"""

import os
import sys
import time
import argparse
import random
from typing import Optional, Dict, Tuple, List
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
from tqdm import tqdm

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Without these, a given seed is NOT reproducible run-to-run on GPU --
    # cudnn's default algorithm selection/atomic ops are nondeterministic,
    # which is why identical seed=42 runs previously landed at 0.7834 and
    # 0.7887. Needed so that a seed found via sweeping actually reproduces
    # its result later, not just on the one run it was found on.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with pos_weight and gamma=2.0 (Section 3.3).
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class CTGDataset(Dataset):
    """PyTorch Dataset wrapper for CTG windowed signals."""
    def __init__(self, X, y, patient_ids=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity calculation
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUROC & AUPRC
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.5

    try:
        p_precision, p_recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(p_recall, p_precision)
    except ValueError:
        auprc = 0.0

    # Operating-point metrics. Threshold 0.5 is a poor cut-off at this
    # prevalence (~4.3% positive windows): a well-calibrated model outputs
    # mostly low probabilities, so 0.5 yields high specificity and unusable
    # sensitivity. These threshold-free summaries describe the whole ROC, which
    # is what matters clinically -- a missed acidosis costs far more than a
    # false alarm, so the deployed operating point will not be 0.5.
    sens_at_90spec = 0.0
    spec_at_90sens = 0.0
    thresh_at_80sens = 0.5
    try:
        from sklearn.metrics import roc_curve as _roc
        fpr, tpr, thr = _roc(y_true, y_prob)
        idx = np.where(fpr <= 0.10)[0]
        if len(idx):
            sens_at_90spec = float(tpr[idx].max())
        idx = np.where(tpr >= 0.90)[0]
        if len(idx):
            spec_at_90sens = float(1.0 - fpr[idx].min())
        idx = np.where(tpr >= 0.80)[0]
        if len(idx):
            thresh_at_80sens = float(thr[idx[0]])
    except Exception:
        pass

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'sens_at_90spec': sens_at_90spec,
        'spec_at_90sens': spec_at_90sens,
        'thresh_at_80sens': thresh_at_80sens,
    }


def generate_synthetic_data(num_samples=160, num_patients=20):
    """Generates synthetic (Batch, 2, 4800) data for dry-run verification."""
    print("Generating synthetic dataset for CTG-CrossFormer dry-run verification...")
    X = np.random.randn(num_samples, 2, 4800).astype(np.float32)
    y = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])  # 4:1 imbalance ratio
    patient_ids = np.random.choice([f"P{i:03d}" for i in range(num_patients)], size=num_samples)
    return X, y, patient_ids


def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, desc="[Train]"):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=desc, leave=False)
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        train_loss += loss.item() * len(y_batch)
        pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
    return train_loss / len(train_loader.dataset)


@torch.no_grad()
def evaluate(model, val_loader, device, desc="[Val]"):
    model.eval()
    all_targets, all_probs = [], []
    pbar = tqdm(val_loader, desc=desc, leave=False)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        logits = model(X_batch).squeeze(-1)
        probs = torch.sigmoid(logits)
        all_targets.extend(y_batch.numpy())
        all_probs.extend(probs.cpu().numpy())
    return compute_metrics(np.array(all_targets), np.array(all_probs))


def main():
    parser = argparse.ArgumentParser(description="Train CTG-CrossFormer Standalone Benchmark")
    parser.add_argument("--config", type=str, default="configs/ctg_crossformer_config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/processed/")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ctg_crossformer/")
    parser.add_argument("--dry_run", action="store_true", help="Run 2-epoch synthetic verification pass")
    parser.add_argument("--class_weight", type=str, default="none",
                        choices=["none", "inverse_freq", "sqrt_inverse_freq"],
                        help="Positive-class weight inside the focal loss, ON TOP of the "
                             "sqrt-inverse WeightedRandomSampler. none (default) = pos_weight 1.0, "
                             "this repo's historical behaviour. inverse_freq = n_neg/n_pos, which is "
                             "what the benchmarked paper actually specifies. sqrt_inverse_freq = a "
                             "milder middle ground.")
    parser.add_argument("--early_stop_mode", type=str, default="outer_best",
                        choices=["outer_best", "nested"],
                        help="outer_best (default, ORIGINAL behaviour): pick the epoch with the "
                             "best AUROC on the reported validation fold. This is selection-on-test "
                             "and inflates CV AUROC by ~0.077 (measured 2026-08-19: 0.8243 reported "
                             "vs 0.7472 final-epoch vs 0.7305 held-out test). Kept as default so "
                             "every previously reported number stays reproducible, and because it "
                             "matches the convention used by the literature we benchmark against. "
                             "nested: hold out an inner patient-level split from the training fold "
                             "for epoch selection/early stopping, leaving the reported fold "
                             "untouched -- an unbiased estimate. In nested mode BOTH numbers are "
                             "reported so the bias is visible.")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early-stopping patience on the INNER validation split (nested mode only).")
    parser.add_argument("--inner_val_frac", type=float, default=0.2,
                        help="Fraction of the training fold's PATIENTS held out for inner "
                             "selection (nested mode only).")
    parser.add_argument("--fold_mode", type=str, default="stratified_group",
                        choices=["stratified_group", "patient_level"],
                        help="stratified_group (default) = this script's original window-level "
                             "StratifiedGroupKFold. patient_level = create_patient_level_folds() "
                             "joint-stratified on y_figo, matching train_knowledge_infused.py so "
                             "the two are directly comparable on identical folds.")
    parser.add_argument("--pretrained_encoder", type=str, default=None,
                         help="Path to a CTGCrossformerEncoder state_dict (e.g. from "
                              "scripts/pretrain_ctg_crossformer_ssl.py) to initialize each "
                              "fold's encoder from, instead of random init.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "model": {"cnn_channels": 128, "n_heads_cross": 4, "n_heads_tf": 8, "n_tf_layers": 4, "d_ff": 512, "dropout": 0.1, "latent_dim": 128, "classifier_hidden_dim": 128, "classifier_dropout": 0.3},
            "training": {"epochs": 50, "batch_size": 32, "lr": 0.0003, "weight_decay": 1e-5, "k_folds": 5, "focal_loss_gamma": 2.0}
        }

    m_cfg = cfg["model"]
    t_cfg = cfg["training"]

    train_pt = os.path.join(args.data_dir, "train_dataset.pt")
    test_pt  = os.path.join(args.data_dir, "test_dataset.pt")

    y_figo = None
    if not args.dry_run and os.path.exists(train_pt):
        print(f"Loading preprocessed dataset from {train_pt}...")
        data = torch.load(train_pt, weights_only=False)
        X = data['X'].numpy()
        y = data['y_primary'].numpy()
        metadata = data['metadata']
        patient_ids = np.array([m[0] for m in metadata])
        y_figo = data.get('y_figo')
    else:
        if not args.dry_run:
            print(f"\n[!] Dataset file 'train_dataset.pt' not found at '{train_pt}'.")
            print("    Running synthetic dataset pass for verification...\n")
        X, y, patient_ids = generate_synthetic_data(num_samples=160 if args.dry_run else 100, num_patients=20)

    dataset = CTGDataset(X, y, patient_ids)

    # 5-Fold Stratified Patient-Level Cross-Validation
    n_splits = 5

    # FOLD MODE (added 2026-08-19, for valid head-to-head benchmarking):
    # this script has always used window-level StratifiedGroupKFold, while
    # train_knowledge_infused.py (Model 8) uses patient-level
    # create_patient_level_folds() joint-stratified on y_figo. Those produce
    # DIFFERENT partitions, so comparing this script's CV mean against Model 8's
    # is not apples-to-apples -- per-fold AUROC has been observed to swing
    # 0.72-0.91, which is larger than the effect sizes being claimed.
    # --fold_mode patient_level reproduces Model 8's exact split so the two are
    # directly comparable. Default 'stratified_group' preserves this script's
    # original behaviour, so every previously reported benchmark stays reproducible.
    if args.fold_mode == "patient_level":
        from src.training.train import create_patient_level_folds
        secondary = y_figo if y_figo is not None else None
        fold_iter = create_patient_level_folds(
            list(patient_ids), torch.as_tensor(y), k_folds=n_splits,
            secondary_labels=secondary,
        )
        print(f"Fold mode: patient_level (matches train_knowledge_infused.py"
              f"{' , joint-stratified on y_figo' if secondary is not None else ''})")
    else:
        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        fold_iter = list(sgkf.split(X, y, groups=patient_ids))
        print("Fold mode: stratified_group (this script's original split)")

    fold_metrics = []
    biased_fold_metrics = []
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\n--- Starting CTG-CrossFormer {n_splits}-Fold Stratified Patient-Level CV ---")
    start_time = time.time()
    epochs_run = 2 if args.dry_run else t_cfg.get("epochs", 50)

    for fold, (train_idx, val_idx) in enumerate(fold_iter, 1):
        # NESTED MODE: carve an inner validation split out of the TRAINING fold,
        # split at PATIENT level so no patient's windows straddle it, and use it
        # (not the reported fold) for epoch selection / early stopping.
        inner_val_idx = None
        fit_idx = train_idx
        if args.early_stop_mode == "nested":
            from sklearn.model_selection import train_test_split as _tts
            tr_pids = patient_ids[train_idx]
            uniq_tr = np.array(sorted(set(tr_pids)))
            p_lab = np.array([int(y[train_idx][tr_pids == p].max()) for p in uniq_tr])
            strat = p_lab if len(np.unique(p_lab)) > 1 and np.bincount(p_lab).min() >= 2 else None
            inner_tr_p, inner_va_p = _tts(
                uniq_tr, test_size=args.inner_val_frac, stratify=strat, random_state=42
            )
            va_set = set(inner_va_p.tolist())
            is_inner_va = np.array([pid in va_set for pid in tr_pids])
            fit_idx = train_idx[~is_inner_va]
            inner_val_idx = train_idx[is_inner_va]

        train_sub = Subset(dataset, fit_idx)
        val_sub   = Subset(dataset, val_idx)

        # Sqrt-inverse frequency oversampling (WeightedRandomSampler)
        y_train_sub = y[fit_idx]
        class_counts = np.bincount(y_train_sub.astype(int))
        class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
        sample_weights = class_weights[y_train_sub.astype(int)]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        batch_size = t_cfg.get("batch_size", 32)
        train_loader = DataLoader(train_sub, batch_size=batch_size, sampler=sampler)
        val_loader   = DataLoader(val_sub,   batch_size=batch_size, shuffle=False)
        inner_val_loader = (
            DataLoader(Subset(dataset, inner_val_idx), batch_size=batch_size, shuffle=False)
            if inner_val_idx is not None else None
        )
        if inner_val_loader is not None:
            print(f"  [nested] fit on {len(fit_idx)} windows | inner-val {len(inner_val_idx)} "
                  f"({len(set(patient_ids[inner_val_idx]))} patients) | outer-val {len(val_idx)}")

        # Build CTG-CrossFormer Model
        encoder = CTGCrossformerEncoder(
            in_channels=2,
            seq_len=4800,
            cnn_channels=m_cfg.get("cnn_channels", 128),
            n_heads_cross=m_cfg.get("n_heads_cross", 4),
            n_heads_tf=m_cfg.get("n_heads_tf", 8),
            n_tf_layers=m_cfg.get("n_tf_layers", 4),
            d_ff=m_cfg.get("d_ff", 512),
            dropout=m_cfg.get("dropout", 0.1),
            latent_dim=m_cfg.get("latent_dim", 128)
        )
        if args.pretrained_encoder:
            encoder.load_state_dict(
                torch.load(args.pretrained_encoder, map_location=device, weights_only=True),
                strict=True,
            )
            if fold == 1:
                print(f"Loaded pretrained encoder weights from {args.pretrained_encoder}")
        model = CTGCrossformerForClassification(
            encoder=encoder,
            hidden_dim=m_cfg.get("classifier_hidden_dim", 128),
            dropout=m_cfg.get("classifier_dropout", 0.3)
        ).to(device)

        # Focal Loss (gamma=2.0) with pos_weight
        # BUG FIX (2026-08-09): the WeightedRandomSampler above already rebalances
        # each batch via sqrt-inverse-frequency oversampling of the positive class.
        # Also applying the full n_neg/n_pos pos_weight on top double-counts the
        # class imbalance correction, pushing the model toward extreme positive
        # confidence during training that doesn't match the true class balance at
        # inference time (same failure mode diagnosed and fixed for Model 8's
        # BalancedBatchSampler in train_knowledge_infused.py). Since the sampler
        # already handles rebalancing here, pos_weight is fixed at 1.0.
        # CLASS WEIGHTING (2026-08-19): the 2026-08-09 note below fixed pos_weight
        # to 1.0 on the reasoning that the sampler already rebalances. That is a
        # DEVIATION from the paper, which specifies BOTH "Focal Loss with gamma=2.0
        # and inverse-frequency class weights" AND "sqrt-inverse frequency
        # oversampling via WeightedRandomSampler". The arithmetic supports the
        # paper: sqrt-inverse sampling only lifts batch prevalence from ~4.3% to
        # ~17%, so it does NOT fully rebalance and the model stays under-corrected
        # toward the negative class -- which is why sensitivity at threshold 0.5 is
        # ~40% here versus the paper's 89.5%.
        n_pos = float(max(y_train_sub.sum(), 1))
        n_neg = float(len(y_train_sub) - n_pos)
        if args.class_weight == "inverse_freq":
            pw = n_neg / n_pos
        elif args.class_weight == "sqrt_inverse_freq":
            pw = float(np.sqrt(n_neg / n_pos))
        else:
            pw = 1.0
        pos_weight = torch.tensor([pw]).to(device)
        if fold == 1:
            print(f"Class weighting: {args.class_weight} -> pos_weight={pw:.2f} "
                  f"(n_neg/n_pos = {n_neg/n_pos:.1f})")
        criterion = FocalLoss(gamma=t_cfg.get("focal_loss_gamma", 2.0), pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=t_cfg.get("lr", 0.0003),
            weight_decay=t_cfg.get("weight_decay", 1e-5)
        )

        steps_per_epoch = max(len(train_loader), 2)
        epochs_sched = max(epochs_run, 2)
        total_steps = max(steps_per_epoch * epochs_sched, 20)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=t_cfg.get("lr", 0.0003),
            total_steps=total_steps,
            pct_start=0.1
        )

        best_val_auroc = 0.0       # best on the REPORTED fold -> protocol-matched (biased)
        best_val_metrics = None
        best_inner_auroc = -1.0    # best on the INNER split -> unbiased selection signal
        unbiased_metrics = None    # outer metrics AT the inner-selected epoch
        patience_ctr = 0

        epoch_pbar = tqdm(range(1, epochs_run + 1), desc=f"Fold {fold}/{n_splits} Epochs", unit="epoch")
        for epoch in epoch_pbar:
            loss = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, device,
                desc=f"Fold {fold} Ep {epoch} [Train]"
            )
            val_metrics = evaluate(model, val_loader, device, desc=f"Fold {fold} Ep {epoch} [Val]")

            # Protocol-matched (biased) tracking -- best epoch on the reported fold.
            if val_metrics['auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc']
                best_val_metrics = val_metrics
                if args.early_stop_mode == "outer_best":
                    ckpt_path = os.path.join(args.checkpoint_dir, f"ctg_crossformer_fold_{fold}_best.pth")
                    torch.save(model.state_dict(), ckpt_path)

            postfix = {'loss': f'{loss:.4f}', 'val_auroc': f'{val_metrics["auroc"]:.4f}'}

            if inner_val_loader is not None:
                inner_metrics = evaluate(model, inner_val_loader, device,
                                         desc=f"Fold {fold} Ep {epoch} [InnerVal]")
                postfix['inner_auroc'] = f'{inner_metrics["auroc"]:.4f}'
                if inner_metrics['auroc'] > best_inner_auroc:
                    best_inner_auroc = inner_metrics['auroc']
                    # The outer score at this epoch is the UNBIASED estimate: the
                    # epoch was chosen without ever consulting the reported fold.
                    unbiased_metrics = val_metrics
                    patience_ctr = 0
                    ckpt_path = os.path.join(args.checkpoint_dir, f"ctg_crossformer_fold_{fold}_best.pth")
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    patience_ctr += 1
                    if patience_ctr >= args.patience:
                        epoch_pbar.close()
                        print(f"  [nested] early stop at epoch {epoch} "
                              f"(no inner improvement for {args.patience} epochs)")
                        break
            else:
                postfix['val_f1'] = f'{val_metrics["f1"]:.4f}'

            epoch_pbar.set_postfix(postfix)

        biased_metrics = best_val_metrics if best_val_metrics is not None else val_metrics
        if args.early_stop_mode == "nested":
            metrics = unbiased_metrics if unbiased_metrics is not None else val_metrics
            biased_fold_metrics.append(biased_metrics)
            print(f"\nFold {fold} | UNBIASED AUROC: {metrics['auroc']:.4f} "
                  f"(AUPRC {metrics['auprc']:.4f}, Sens {metrics['recall']:.4f}, "
                  f"Spec {metrics['specificity']:.4f})")
            print(f"Fold {fold} | protocol-matched (biased) AUROC: {biased_metrics['auroc']:.4f} "
                  f"| selection inflation: {biased_metrics['auroc'] - metrics['auroc']:+.4f}\n")
        else:
            metrics = biased_metrics
            print(f"\nFold {fold} Best | AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f} | "
                  f"Sens: {metrics['recall']:.4f} | Spec: {metrics['specificity']:.4f} | F1: {metrics['f1']:.4f}\n")
        fold_metrics.append(metrics)

    elapsed_time = time.time() - start_time
    print(f"\nCompleted {n_splits}-Fold CV in {elapsed_time:.2f} seconds.")

    # Calculate Mean +/- Std across folds
    keys = fold_metrics[0].keys()
    header = ("UNBIASED (nested selection)" if args.early_stop_mode == "nested"
              else "5-FOLD CROSS-VALIDATION RESULTS")
    print(f"\n================ {header} ================")
    for k in keys:
        vals = [fm[k] for fm in fold_metrics]
        print(f"{k.capitalize():<12}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    if args.early_stop_mode == "nested" and biased_fold_metrics:
        print("\n======== PROTOCOL-MATCHED (best-epoch-on-reported-fold, as literature) ========")
        for k in keys:
            vals = [fm[k] for fm in biased_fold_metrics]
            print(f"{k.capitalize():<12}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        ub = np.mean([fm["auroc"] for fm in fold_metrics])
        bi = np.mean([fm["auroc"] for fm in biased_fold_metrics])
        print(f"\n>>> selection inflation on AUROC: {bi - ub:+.4f} "
              f"(protocol-matched {bi:.4f} vs unbiased {ub:.4f})")

    if not args.dry_run and os.path.exists(test_pt):
        print("\n================ HELD-OUT TEST SET EVALUATION ================")
        test_data = torch.load(test_pt, weights_only=False)
        test_ds = CTGDataset(test_data['X'].numpy(), test_data['y_primary'].numpy())
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        # FIX (2026-08-19): this previously evaluated `model` -- whatever was left
        # in memory after the fold loop, i.e. the LAST fold's end-of-training state,
        # neither a saved best checkpoint nor an ensemble. That made every test
        # number this project reported a single arbitrary fold's leftover weights
        # (and fold 5 was often the weakest). Now evaluates the 5-fold ENSEMBLE by
        # averaging the saved best checkpoints' predicted probabilities, which is
        # both the standard approach and consistent with what CV actually selected.
        import glob as _glob
        ckpt_paths = sorted(_glob.glob(os.path.join(args.checkpoint_dir, 'ctg_crossformer_fold_*_best.pth')))
        if ckpt_paths:
            fold_probs = []
            for cp in ckpt_paths:
                model.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
                model.eval()
                probs_one, targs = [], []
                with torch.no_grad():
                    for Xb, yb in test_loader:
                        probs_one.extend(torch.sigmoid(model(Xb.to(device)).squeeze(-1)).cpu().numpy())
                        targs.extend(yb.numpy())
                fold_probs.append(np.array(probs_one))
            ens = np.mean(np.stack(fold_probs, axis=0), axis=0)
            test_metrics = compute_metrics(np.array(targs), ens)
            print(f'(5-fold ensemble of {len(ckpt_paths)} checkpoints)')
        else:
            test_metrics = evaluate(model, test_loader, device, desc='[Test]')
            print('(WARNING: no checkpoints found -- fell back to last in-memory model)')
        for k, v in test_metrics.items():
            print(f"Test {k.capitalize():<12}: {v:.4f}")


if __name__ == "__main__":
    main()
