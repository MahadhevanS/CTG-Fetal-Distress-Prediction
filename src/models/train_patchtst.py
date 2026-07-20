"""
Training and 5-Fold Patient-Level Stratified Cross-Validation Benchmark for PatchTST.

Conforms to Phase 3 Model Evaluation Protocol:
- Stratified 5-Fold Patient-Level CV
- Evaluates Metrics: Accuracy, Precision, Recall, Specificity, F1, AUROC, AUPRC
- Handles missing dataset gracefully via synthetic dry-run verification mode
"""

import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

from patchtst import PatchTSTEncoder, PatchTSTForClassification


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    
    # Specificity
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

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc
    }


def generate_synthetic_data(num_samples=100, num_patients=20):
    """Generates synthetic (Batch, 2, 4800) data for dry-run verification."""
    print("Generating synthetic dataset for dry-run verification...")
    X = np.random.randn(num_samples, 2, 4800).astype(np.float32)
    y = np.random.randint(0, 2, size=num_samples)
    patient_ids = np.random.choice([f"P{i:03d}" for i in range(num_patients)], size=num_samples)
    return X, y, patient_ids


def train_fold(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs=50, patience=10):
    best_val_auroc = 0.0
    best_val_metrics = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        scheduler.step()

        # Validation Phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(y_batch.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        metrics = compute_metrics(val_targets, val_preds)

        if metrics['auroc'] > best_val_auroc:
            best_val_auroc = metrics['auroc']
            best_val_metrics = metrics
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_val_metrics if best_val_metrics is not None else metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmarking PatchTST Encoder")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to preprocessed data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--patch_len", type=int, default=16, help="Patch length P")
    parser.add_argument("--stride", type=int, default=16, help="Patch stride S")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", help="Run in dry-run synthetic mode for testing")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing PatchTST Benchmark on device: {device}")

    # Load Dataset
    train_pt = os.path.join(args.data_dir, "train_dataset.pt")
    if not args.dry_run and os.path.exists(train_pt):
        print(f"Loading preprocessed dataset from {train_pt}...")
        data = torch.load(train_pt)
        X = data['X'].numpy()
        y = data['y_primary'].numpy()
        metadata = data['metadata']
        patient_ids = np.array([m[0] for m in metadata])
    else:
        X, y, patient_ids = generate_synthetic_data(num_samples=160 if args.dry_run else 100, num_patients=20)

    dataset = CTGDataset(X, y, patient_ids)

    # 5-Fold Stratified Patient-Level Cross-Validation
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    
    fold_metrics = []
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\n--- Starting {n_splits}-Fold Patient-Level Stratified CV ---")
    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=patient_ids), 1):
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)

        train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False)

        # Build PatchTST Encoder & Classifier
        encoder = PatchTSTEncoder(
            in_channels=2,
            seq_len=4800,
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            latent_dim=128
        )
        model = PatchTSTForClassification(encoder, hidden_dim=64, dropout=0.2).to(device)

        pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        epochs_run = 5 if args.dry_run else args.epochs
        metrics = train_fold(
            model, train_loader, val_loader, optimizer, scheduler, 
            criterion, device, epochs=epochs_run, patience=5 if args.dry_run else 10
        )
        fold_metrics.append(metrics)

        # Save Checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"patchtst_fold_{fold}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Fold {fold} | AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f} | F1: {metrics['f1']:.4f}")

    elapsed_time = time.time() - start_time
    print(f"\nCompleted {n_splits}-Fold CV in {elapsed_time:.2f} seconds.")

    # Calculate Mean ± Std across folds
    keys = fold_metrics[0].keys()
    print("\n================ FINAL BENCHMARK RESULTS ================")
    for k in keys:
        vals = [fm[k] for fm in fold_metrics]
        mean_val, std_val = np.mean(vals), np.std(vals)
        print(f"{k.capitalize():<12}: {mean_val:.4f} ± {std_val:.4f}")

    # Count Parameters
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nPatchTST Encoder Total Trainable Parameters: {total_params:,}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
