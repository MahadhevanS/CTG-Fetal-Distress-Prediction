import argparse
import os
import sys
import yaml

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import available model encoders
from src.models import CNN1DEncoder, BiLSTMEncoder



class TemporalClassifier(nn.Module):
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


# Model Registry mapping CLI names to classes
MODEL_REGISTRY = {
    "cnn1d": CNN1DEncoder,
    "1dcnn": CNN1DEncoder,
    "cnn": CNN1DEncoder,
    "cnn1dencoder": CNN1DEncoder,
    "bilstm": BiLSTMEncoder,
    "bilstmencoder": BiLSTMEncoder,
    "lstm": BiLSTMEncoder,
}


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def instantiate_model(model_name: str) -> tuple[str, nn.Module]:
    clean_name = model_name.lower().replace("-", "").replace("_", "")
    if clean_name not in MODEL_REGISTRY:
        available = ["cnn1d", "bilstm"]
        raise ValueError(
            f"Unknown model name '{model_name}'. Available options: {available} or 'all'"
        )

    model_class = MODEL_REGISTRY[clean_name]
    display_name = model_class.__name__
    encoder = model_class()
    classifier_model = TemporalClassifier(encoder=encoder, latent_dim=128)
    return display_name, classifier_model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_single_model(
    model_name: str,
    data_dir: str,
    save_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    dry_run: bool = False,
):
    print("\n" + "=" * 60)
    display_name, model = instantiate_model(model_name)
    param_count = count_parameters(model)
    print(f"Initializing Model: {display_name}")
    print(f"Trainable Parameters: {param_count:,}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    if dry_run:
        print("\n[DRY RUN MODE] Running forward and backward pass on dummy batch...")
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
        return

    # Check for processed dataset files
    train_pt = os.path.join(data_dir, "train_dataset.pt")
    val_pt = os.path.join(data_dir, "val_dataset.pt")

    if not os.path.exists(train_pt):
        print(f"\n[WARNING] Dataset file not found at: {train_pt}")
        print("Running forward pass test on random sample instead...")
        x_dummy = torch.randn(4, 2, 4800).to(device)
        out = model(x_dummy)
        print(f"[OK] Model output shape: {tuple(out.shape)}")
        print("To train on real data, place dataset in data/processed/ or pass --data_dir.")
        return

    # Load real dataset
    print(f"Loading training data from {train_pt}...")
    train_data = torch.load(train_pt, weights_only=False)
    X_train = train_data["X"].float()
    y_train = train_data["y_primary"].float()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if os.path.exists(val_pt):
        print(f"Loading validation data from {val_pt}...")
        val_data = torch.load(val_pt, weights_only=False)
        X_val = val_data["X"].float()
        y_val = val_data["y_primary"].float()
        val_ds = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_info = ""
        if val_loader:
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss_sum += loss.item() * X_batch.size(0)
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)

            val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total
            val_info = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(save_dir, f"{display_name}_best.pt")
                torch.save(model.state_dict(), ckpt_path)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}{val_info}"
        )

    print(f"\n[OK] Training completed for {display_name}.")
    if val_loader:
        print(f"Best checkpoint saved to: {os.path.join(save_dir, f'{display_name}_best.pt')}")


def main():
    parser = argparse.ArgumentParser(description="Train CTG Fetal Distress Models Separately or Together")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn1d",
        help="Model to run/train: 'cnn1d', 'bilstm', or 'all' (default: cnn1d)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/local.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Path to processed data directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick dry-run forward pass without training")

    args = parser.parse_args()

    # Load YAML config defaults if available
    config = load_config(args.config)
    cfg_data = config.get("data", {})
    cfg_train = config.get("training", {})

    data_dir = args.data_dir or cfg_data.get("processed_path", "data/processed")
    save_dir = args.save_dir or cfg_train.get("checkpoint_dir", "checkpoints")
    epochs = args.epochs or cfg_train.get("epochs", 5)
    batch_size = args.batch_size or cfg_train.get("batch_size", 16)
    lr = args.lr or cfg_train.get("learning_rate", 0.001)

    if args.model.lower() == "all":
        target_models = ["cnn1d", "bilstm"]
    else:
        target_models = [args.model]

    for m in target_models:
        train_single_model(
            model_name=m,
            data_dir=data_dir,
            save_dir=save_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
