import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models import GRUEncoder, TCNEncoder, UniversalClassifier

class CTGWindowDataset(Dataset):
    """PyTorch Dataset wrapper for CTG 20-minute signal windows."""
    def __init__(self, X: torch.Tensor, y_primary: torch.Tensor, patient_ids: List[str]):
        self.X = X  # (N, 2, 4800)
        self.y_primary = y_primary.float()  # (N,)
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_primary[idx]


def calculate_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate 7 standard classification metrics."""
    y_pred = (y_probs >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Scikit-learn AUROC & AUPRC if available, else trapezoidal approximation
    try:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        auroc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
        p_vals, r_vals, _ = precision_recall_curve(y_true, y_probs)
        auprc = auc(r_vals, p_vals) if len(np.unique(y_true)) > 1 else 0.0
    except ImportError:
        auroc = 0.5
        auprc = 0.0

    return {
        "accuracy": accuracy * 100.0,
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "specificity": specificity * 100.0
    }


def load_all_dataset_splits(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Loads all split files (.pt) and aggregates X, y_primary, and patient record IDs."""
    X_list, y_list, patient_list = [], [], []

    for split_name in ['train', 'val', 'test']:
        pt_path = os.path.join(data_dir, f'{split_name}_dataset.pt')
        if os.path.exists(pt_path):
            ds = torch.load(pt_path, map_location='cpu', weights_only=False)
            X_list.append(ds['X'])
            y_list.append(ds['y_primary'])
            # metadata elements: (record_id, start, end)
            patients = [str(item[0]) for item in ds['metadata']]
            patient_list.extend(patients)

    if len(X_list) == 0:
        raise FileNotFoundError(f"No .pt dataset files found in {data_dir}")

    X_all = torch.cat(X_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    return X_all, y_all, patient_list


def create_patient_level_folds(patient_ids: List[str], y_all: torch.Tensor, k_folds: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generates Stratified K-Fold indices based on unique patient IDs."""
    unique_patients = np.array(sorted(list(set(patient_ids))))
    
    # Compute max label per patient for stratification
    patient_labels = []
    patient_ids_np = np.array(patient_ids)
    y_all_np = y_all.numpy()
    
    for pid in unique_patients:
        mask = (patient_ids_np == pid)
        p_label = int(np.max(y_all_np[mask]))
        patient_labels.append(p_label)
    patient_labels = np.array(patient_labels)

    try:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        folds = []
        for train_p_idx, val_p_idx in skf.split(unique_patients, patient_labels):
            val_patients = set(unique_patients[val_p_idx])
            
            train_indices = np.where([pid not in val_patients for pid in patient_ids])[0]
            val_indices = np.where([pid in val_patients for pid in patient_ids])[0]
            folds.append((train_indices, val_indices))
        return folds
    except ImportError:
        # Fallback simple split if sklearn not installed
        n_samples = len(patient_ids)
        fold_size = n_samples // k_folds
        indices = np.arange(n_samples)
        folds = []
        for f in range(k_folds):
            val_idx = indices[f * fold_size : (f + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            folds.append((train_idx, val_idx))
        return folds


def train_single_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device
) -> Dict[str, float]:
    """Trains a model on one fold and returns best validation metrics."""
    pos_weight = torch.tensor([2.0]).to(device)  # Focal weighting for positive distress cases
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_auroc = -1.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(y_batch)

        scheduler.step()

        # Validation phase
        model.eval()
        val_targets, val_probs = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(X_batch).squeeze(-1)
                    probs = torch.sigmoid(logits)
                
                val_targets.extend(y_batch.numpy())
                val_probs.extend(probs.cpu().numpy())

        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs)
        metrics = calculate_metrics(val_targets, val_probs)

        if metrics['auroc'] > best_val_auroc:
            best_val_auroc = metrics['auroc']
            best_metrics = metrics

    return best_metrics


def train_and_evaluate(
    model_name: str,
    data_dir: str,
    k_folds: int = 5,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.0005,
    save_dir: str = "checkpoints"
) -> Dict[str, Tuple[float, float]]:
    """Runs 5-Fold Patient-Level CV training loop for a selected model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f" Starting Stratified {k_folds}-Fold Patient-Level CV: {model_name.upper()}")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"=======================================================\n")

    X_all, y_all, patient_ids = load_all_dataset_splits(data_dir)
    print(f"Loaded {len(X_all)} total windows across {len(set(patient_ids))} unique patients.")

    folds = create_patient_level_folds(patient_ids, y_all, k_folds=k_folds)
    fold_results = {m: [] for m in ["accuracy", "auroc", "auprc", "f1", "precision", "recall", "specificity"]}

    os.makedirs(save_dir, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"--- Fold {fold_idx}/{k_folds} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")

        train_ds = CTGWindowDataset(X_all[train_idx], y_all[train_idx], [patient_ids[i] for i in train_idx])
        val_ds = CTGWindowDataset(X_all[val_idx], y_all[val_idx], [patient_ids[i] for i in val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Instantiate fresh model for each fold
        if model_name.lower() == 'gru':
            encoder = GRUEncoder(hidden_dim=128, gru_hidden=64, num_layers=2, dropout=0.2)
        elif model_name.lower() == 'tcn':
            encoder = TCNEncoder(hidden_dim=128, kernel_size=3, dropout=0.2)
        else:
            raise ValueError(f"Unsupported model_name '{model_name}'. Choose 'gru' or 'tcn'.")

        model = UniversalClassifier(encoder=encoder, latent_dim=128).to(device)

        metrics = train_single_fold(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

        for m_key in fold_results:
            fold_results[m_key].append(metrics[m_key])

        print(f" Fold {fold_idx} Metrics -> AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f} | "
              f"F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.2f}% | "
              f"Sens: {metrics['recall']:.2f}% | Spec: {metrics['specificity']:.2f}%")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name.lower()}_fold{fold_idx}.pth"))

    summary = {}
    print(f"\n=======================================================")
    print(f" FINAL {k_folds}-FOLD PATIENT-LEVEL CV RESULTS: {model_name.upper()}")
    print(f"=======================================================")
    for m_key, vals in fold_results.items():
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        summary[m_key] = (mean_val, std_val)
        if m_key in ['accuracy', 'precision', 'recall', 'specificity']:
            print(f" {m_key.capitalize():<12}: {mean_val:.2f}% ± {std_val:.2f}%")
        else:
            print(f" {m_key.upper():<12}: {mean_val:.4f} ± {std_val:.4f}")
    print(f"=======================================================\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train CTG Fetal Distress Temporal Encoders")
    parser.add_argument("--config", type=str, default="configs/local.yaml", help="Path to config yaml")
    parser.add_argument("--model", type=str, choices=['gru', 'tcn', 'all'], default='gru', help="Model architecture")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of patient-level cross validation folds")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to preprocessed .pt files")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Path to save trained weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")

    args = parser.parse_args()

    # Load config file if provided
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        if 'training' in cfg:
            args.batch_size = cfg['training'].get('batch_size', args.batch_size)
            args.epochs = cfg['training'].get('epochs', args.epochs)
            args.lr = cfg['training'].get('learning_rate', args.lr)
            args.save_dir = cfg['training'].get('checkpoint_dir', args.save_dir)

    models_to_train = ['gru', 'tcn'] if args.model == 'all' else [args.model]

    for m in models_to_train:
        train_and_evaluate(
            model_name=m,
            data_dir=args.data_dir,
            k_folds=args.k_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir=args.save_dir
        )

if __name__ == "__main__":
    main()
