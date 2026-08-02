"""
Universal Model Training & Evaluation Module for CTG Fetal Distress Prediction
================================================================================
Supports patient-level stratified 5-fold cross-validation, dynamic model selection,
PyTorch Automatic Mixed Precision (AMP), and standardized metric evaluation.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Registry for dynamic model instantiation
MODEL_REGISTRY = {}

# Dynamically register available temporal encoders
try:
    from src.models import GRUEncoder
    MODEL_REGISTRY["gru"] = GRUEncoder
except ImportError:
    pass

try:
    from src.models import TCNEncoder
    MODEL_REGISTRY["tcn"] = TCNEncoder
except ImportError:
    pass

try:
    from src.models import CNN1DEncoder
    MODEL_REGISTRY["cnn1d"] = CNN1DEncoder
    MODEL_REGISTRY["1dcnn"] = CNN1DEncoder
    MODEL_REGISTRY["cnn"] = CNN1DEncoder
    MODEL_REGISTRY["cnn1dencoder"] = CNN1DEncoder
except ImportError:
    pass

try:
    from src.models import BiLSTMEncoder
    MODEL_REGISTRY["bilstm"] = BiLSTMEncoder
    MODEL_REGISTRY["bilstmencoder"] = BiLSTMEncoder
    MODEL_REGISTRY["lstm"] = BiLSTMEncoder
except ImportError:
    pass

try:
    from src.models import MultiScaleLSTMEncoder
    MODEL_REGISTRY["multiscale_lstm"] = MultiScaleLSTMEncoder
except ImportError:
    pass

try:
    from src.models import PatchCTGEncoder
    MODEL_REGISTRY["patchctg"] = PatchCTGEncoder
except ImportError:
    pass

try:
    from src.models import PatchTSTEncoder
    MODEL_REGISTRY["patchtst"] = PatchTSTEncoder
except ImportError:
    pass


class UniversalClassifier(nn.Module):
    """
    Universal wrapper attaching a standard binary classification head
    to any temporal encoder outputting (Batch, 128).
    """

    def __init__(self, encoder: nn.Module, latent_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        features = self.encoder(x)  # (Batch, 128)
        logits = self.classifier(features)  # (Batch, 1)
        return logits.squeeze(-1)


# Compatibility alias
TemporalClassifier = UniversalClassifier


class CTGWindowDataset(Dataset):
    """PyTorch Dataset wrapper for CTG 20-minute signal windows."""

    def __init__(self, X: torch.Tensor, y_primary: torch.Tensor, patient_ids: List[str] = None):
        self.X = X  # (N, 2, 4800)
        self.y_primary = y_primary.float()  # (N,)
        self.patient_ids = (
            patient_ids if patient_ids is not None else [f"p_{i}" for i in range(len(X))]
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_primary[idx]


def calculate_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate standard classification metrics including Sensitivity @ 90% Specificity."""
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

    try:
        from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

        auroc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
        p_vals, r_vals, _ = precision_recall_curve(y_true, y_probs)
        auprc = auc(r_vals, p_vals) if len(np.unique(y_true)) > 1 else 0.0

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            spec_arr = 1.0 - fpr
            idx = np.where(spec_arr >= 0.90)[0]
            sens_at_90spec = float(tpr[idx[-1]]) * 100.0 if len(idx) > 0 else 0.0
        else:
            sens_at_90spec = 0.0
    except ImportError:
        auroc = 0.5
        auprc = 0.0
        sens_at_90spec = 0.0

    return {
        "accuracy": accuracy * 100.0,
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "specificity": specificity * 100.0,
        "sens_at_90spec": sens_at_90spec,
    }


def load_config(config_path: str) -> dict:
    """Loads configuration parameters from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_all_dataset_splits(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Loads dataset split files (.pt) and aggregates X, y_primary, and patient record IDs."""
    X_list, y_list, patient_list = [], [], []

    for split_name in ["train", "val", "test"]:
        pt_path = os.path.join(data_dir, f"{split_name}_dataset.pt")
        if os.path.exists(pt_path):
            ds = torch.load(pt_path, map_location="cpu", weights_only=False)
            X_list.append(ds["X"])
            y_list.append(ds["y_primary"])
            if "metadata" in ds and ds["metadata"]:
                patients = [str(item[0]) for item in ds["metadata"]]
            else:
                start_id = len(patient_list)
                patients = [f"rec_{start_id + i}" for i in range(len(ds["X"]))]
            patient_list.extend(patients)

    if len(X_list) == 0:
        raise FileNotFoundError(f"No .pt dataset files found in {data_dir}")

    X_all = torch.cat(X_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    return X_all, y_all, patient_list


def create_patient_level_folds(
    patient_ids: List[str], y_all: torch.Tensor, k_folds: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generates Stratified K-Fold indices based on unique patient IDs."""
    unique_patients = np.array(sorted(list(set(patient_ids))))

    patient_labels = []
    patient_ids_np = np.array(patient_ids)
    y_all_np = y_all.numpy()

    for pid in unique_patients:
        mask = patient_ids_np == pid
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
        n_samples = len(patient_ids)
        fold_size = n_samples // k_folds
        indices = np.arange(n_samples)
        folds = []
        for f in range(k_folds):
            val_idx = indices[f * fold_size : (f + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            folds.append((train_idx, val_idx))
        return folds


def build_encoder(model_name: str, **kwargs) -> nn.Module:
    """Instantiates encoder instance based on model_name with optional hyperparameter overrides."""
    name_clean = model_name.lower().replace("-", "").replace("_", "").strip()

    matched_key = None
    for key in MODEL_REGISTRY:
        if key.lower().replace("-", "").replace("_", "") == name_clean:
            matched_key = key
            break

    if not matched_key:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model_name '{model_name}'. Registered options: {available} or 'all'"
        )

    encoder_cls = MODEL_REGISTRY[matched_key]
    
    default_args = {"in_channels": 2, "seq_len": 4800, "latent_dim": 128}

    if matched_key == "gru":
        default_args.update({"hidden_dim": 128, "gru_hidden": 64, "num_layers": 2, "dropout": 0.2})
    elif matched_key == "tcn":
        default_args.update({"hidden_dim": 128, "kernel_size": 3, "dropout": 0.2})
    elif matched_key in ["cnn1d", "1dcnn", "cnn", "cnn1dencoder"]:
        default_args.update({"latent_dim": 128})
    elif matched_key in ["bilstm", "bilstmencoder", "lstm"]:
        default_args.update({"hidden_size": 64, "num_layers": 2, "dropout": 0.2})
    elif matched_key == "multiscale_lstm":
        default_args.update({"hidden_size": 64, "num_layers": 2, "dropout": 0.2})
    elif matched_key in ["patchctg", "patchtst"]:
        default_args.update({"patch_len": 16, "stride": 16, "d_model": 128, "n_heads": 8, "n_layers": 3, "dropout": 0.1})

    default_args.update(kwargs)

    import inspect
    sig = inspect.signature(encoder_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in default_args.items() if k in valid_params}

    return encoder_cls(**filtered_kwargs)


def train_single_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    weight_decay: float = 1e-4,
    pos_weight: torch.Tensor = None,
    save_path: str = None,
) -> Dict[str, float]:
    """Trains a model on one fold and returns best validation metrics."""
    if pos_weight is None:
        pos_weight = torch.tensor([1.0]).to(device)
    else:
        pos_weight = pos_weight.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_auroc = -1.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(X_batch)
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
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(X_batch)
                    probs = torch.sigmoid(logits)

                val_targets.extend(y_batch.cpu().numpy())
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


def run_dry_run(model_name: str, batch_size: int, lr: float, device: torch.device, encoder_kwargs: dict = None):
    """Executes a dry-run forward/backward pass using dummy tensors."""
    print("\n" + "=" * 60)
    print(f"[DRY RUN MODE] Initializing Model: {model_name}")
    encoder = build_encoder(model_name, **(encoder_kwargs or {}))
    model = UniversalClassifier(encoder=encoder, latent_dim=128).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {param_count:,}")
    print("Running forward and backward pass on dummy batch...")

    x_dummy = torch.randn(batch_size, 2, 4800).to(device)
    y_dummy = torch.randint(0, 2, (batch_size,)).float().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    optimizer.zero_grad()
    out = model(x_dummy)
    loss = criterion(out, y_dummy)
    loss.backward()
    optimizer.step()

    print(f"[OK] Dummy batch shape: Input {tuple(x_dummy.shape)} -> Output {tuple(out.shape)}")
    print(f"[OK] Loss: {loss.item():.4f}")
    print("[OK] Dry run completed successfully!")
    print("=" * 60 + "\n")


def train_and_evaluate(
    model_name: str,
    data_dir: str,
    k_folds: int = 5,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.0005,
    weight_decay: float = 1e-4,
    encoder_kwargs: dict = None,
    save_dir: str = "checkpoints",
    dry_run: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """Runs Stratified Patient-Level CV training loop for a selected model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dry_run:
        run_dry_run(model_name, batch_size, lr, device, encoder_kwargs=encoder_kwargs)
        return {}

    print("\n" + "=" * 60)
    print(f" Starting Stratified {k_folds}-Fold Patient-Level CV: {model_name.upper()}")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Weight Decay: {weight_decay}")
    print("=" * 60 + "\n")

    try:
        X_all, y_all, patient_ids = load_all_dataset_splits(data_dir)
        print(f"Loaded {len(X_all)} total windows across {len(set(patient_ids))} unique patients.")
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("Running quick dry-run test on dummy data instead...")
        run_dry_run(model_name, batch_size, lr, device, encoder_kwargs=encoder_kwargs)
        return {}

    folds = create_patient_level_folds(patient_ids, y_all, k_folds=k_folds)
    fold_results = {
        m: [] for m in ["accuracy", "auroc", "auprc", "f1", "precision", "recall", "specificity", "sens_at_90spec"]
    }

    os.makedirs(save_dir, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"--- Fold {fold_idx}/{k_folds} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")

        # Dynamic pos_weight calculation from fold's training labels
        n_pos = float(y_all[train_idx].sum().item())
        n_neg = float(len(train_idx)) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
        print(f" Dynamic pos_weight: {pos_weight.item():.2f} (n_pos={int(n_pos)}, n_neg={int(n_neg)})")

        train_ds = CTGWindowDataset(
            X_all[train_idx], y_all[train_idx], [patient_ids[i] for i in train_idx]
        )
        val_ds = CTGWindowDataset(
            X_all[val_idx], y_all[val_idx], [patient_ids[i] for i in val_idx]
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Instantiate fresh model for each fold
        encoder = build_encoder(model_name, **(encoder_kwargs or {}))
        model = UniversalClassifier(encoder=encoder, latent_dim=128).to(device)

        fold_save_path = os.path.join(save_dir, f"{model_name.lower()}_fold{fold_idx}_best.pth")

        metrics = train_single_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            weight_decay=weight_decay,
            pos_weight=pos_weight,
            save_path=fold_save_path,
        )

        for m_key in fold_results:
            if m_key in metrics:
                fold_results[m_key].append(metrics[m_key])

        print(
            f" Fold {fold_idx} Metrics -> AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f} | "
            f"F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.2f}% | "
            f"Sens: {metrics['recall']:.2f}% | Spec: {metrics['specificity']:.2f}% | "
            f"Sens@90%Spec: {metrics.get('sens_at_90spec', 0.0):.2f}%"
        )

    summary = {}
    print("\n" + "=" * 60)
    print(f" FINAL {k_folds}-FOLD PATIENT-LEVEL CV RESULTS: {model_name.upper()}")
    print("=" * 60)
    for m_key, vals in fold_results.items():
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        summary[m_key] = (mean_val, std_val)
        if m_key in ["accuracy", "precision", "recall", "specificity", "sens_at_90spec"]:
            print(f" {m_key.capitalize():<15}: {mean_val:.2f}% ± {std_val:.2f}%")
        else:
            print(f" {m_key.upper():<15}: {mean_val:.4f} ± {std_val:.4f}")
    print("=" * 60 + "\n")

    return summary


def main():
    available_choices = list(MODEL_REGISTRY.keys()) + ["all"]
    parser = argparse.ArgumentParser(
        description="Universal Model Training & Evaluation for CTG Fetal Distress Prediction"
    )
    parser.add_argument(
        "--config", type=str, default="configs/local.yaml", help="Path to config yaml file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn1d",
        help="Model architecture ('cnn1d', 'bilstm', 'gru', 'tcn', etc., or 'all')",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of patient-level cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory containing .pt files",
    )
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs per fold"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument(
        "--dry_run", action="store_true", help="Run a quick forward/backward dry-run on dummy data"
    )

    args = parser.parse_args()

    # Load YAML config defaults if available
    config = load_config(args.config)
    cfg_data = config.get("data", {})
    cfg_train = config.get("training", {})

    data_dir = args.data_dir or cfg_data.get("processed_path", "data/processed")
    save_dir = args.save_dir or cfg_train.get("checkpoint_dir", "checkpoints")
    epochs = args.epochs or cfg_train.get("epochs", 15)
    batch_size = args.batch_size or cfg_train.get("batch_size", 32)
    lr = args.lr or cfg_train.get("learning_rate", 0.0005)

    if args.model.lower() == "all":
        models_to_train = (
            list(MODEL_REGISTRY.keys()) if MODEL_REGISTRY else ["cnn1d", "bilstm"]
        )
    else:
        models_to_train = [args.model]

    for m in models_to_train:
        train_and_evaluate(
            model_name=m,
            data_dir=data_dir,
            k_folds=args.k_folds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_dir=save_dir,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
