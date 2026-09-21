"""
Self-supervised masked-reconstruction pretraining for CTGCrossformerEncoder
-- see docs/model8_crossformer_run_history.md (2026-08-17 entry) and the
associated plan.

Trains CTGCrossformerSSLEncoder + CTGReconstructionDecoder to reconstruct
masked spans of raw CTG signal (from scripts/generate_pretraining_windows.py's
dense unlabeled window pool). Only the encoder's state_dict is checkpointed --
it loads directly into a plain CTGCrossformerEncoder for downstream
fine-tuning (train_ctg_crossformer.py / train_knowledge_infused.py).

Usage:
    python scripts/pretrain_ctg_crossformer_ssl.py [--smoke_test]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.ctg_crossformer_pretraining import (
    CTGCrossformerSSLEncoder,
    CTGReconstructionDecoder,
    CTGCrossformerSSLPretrainer,
    masked_reconstruction_loss,
)
from src.training.ssl_masking import mask_ctg_signal
from src.training.train_knowledge_infused import WarmupCosineScheduler


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PretrainWindowDataset(Dataset):
    def __init__(self, X: torch.Tensor):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def patient_holdout_split(metadata, holdout_frac=0.05, seed=42):
    rng = np.random.RandomState(seed)
    patients = sorted(set(m[0] for m in metadata))
    n_holdout = max(1, int(len(patients) * holdout_frac))
    holdout_patients = set(rng.choice(patients, size=n_holdout, replace=False))
    train_idx = [i for i, m in enumerate(metadata) if m[0] not in holdout_patients]
    holdout_idx = [i for i, m in enumerate(metadata) if m[0] in holdout_patients]
    return np.array(train_idx), np.array(holdout_idx), len(holdout_patients)


@torch.no_grad()
def evaluate_holdout(pretrainer, loader, device, use_amp):
    pretrainer.eval()
    total_loss, n_batches = 0.0, 0
    for X in loader:
        X = X.to(device)
        X_masked, mask = mask_ctg_signal(X)
        with torch.amp.autocast("cuda", enabled=use_amp):
            recon = pretrainer(X_masked)
            loss = masked_reconstruction_loss(recon, X, mask)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed/pretrain_windows.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ctg_crossformer_ssl/")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true",
                         help="2 epochs on a small slice, to verify shapes/gradients before a full run.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    print(f"Loading {args.data_path}...")
    data = torch.load(args.data_path, map_location="cpu", weights_only=False)
    X_all = data["X"]
    metadata = data["metadata"]
    print(f"Loaded {len(X_all)} windows, {len(set(m[0] for m in metadata))} patients.")

    if args.smoke_test:
        args.epochs = 2
        X_all = X_all[:512]
        metadata = metadata[:512]
        print(f"[SMOKE TEST] Using {len(X_all)} windows, {args.epochs} epochs.")

    train_idx, holdout_idx, n_holdout_patients = patient_holdout_split(metadata)
    print(f"Train windows: {len(train_idx)} | Holdout windows: {len(holdout_idx)} "
          f"({n_holdout_patients} patients)")

    train_ds = PretrainWindowDataset(X_all[train_idx])
    holdout_ds = PretrainWindowDataset(X_all[holdout_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    holdout_loader = DataLoader(holdout_ds, batch_size=args.batch_size, shuffle=False)

    encoder = CTGCrossformerSSLEncoder(
        in_channels=2, seq_len=4800, cnn_channels=128,
        n_heads_cross=4, n_heads_tf=8, n_tf_layers=4, d_ff=512,
        dropout=0.1, latent_dim=128,
    )
    decoder = CTGReconstructionDecoder(d_model=256, dropout=0.1)
    pretrainer = CTGCrossformerSSLPretrainer(encoder, decoder).to(device)

    optimizer = torch.optim.AdamW(
        pretrainer.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs, base_lr=args.lr
    )
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_holdout_loss = float("inf")
    ckpt_path = os.path.join(args.checkpoint_dir, "encoder_pretrained.pth")

    print(f"\n{'='*65}\nSSL Pretraining | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}\n{'='*65}\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        pretrainer.train()
        epoch_loss, n_batches = 0.0, 0
        for X in train_loader:
            X = X.to(device)
            X_masked, mask = mask_ctg_signal(X)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                recon = pretrainer(X_masked)
                loss = masked_reconstruction_loss(recon, X, mask)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), max_norm=args.gradient_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)
        holdout_loss = evaluate_holdout(pretrainer, holdout_loader, device, use_amp)

        improved = holdout_loss < best_holdout_loss
        if improved:
            best_holdout_loss = holdout_loss
            torch.save(encoder.state_dict(), ckpt_path)

        marker = " *" if improved else ""
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.5f} | "
              f"holdout_loss={holdout_loss:.5f} | lr={scheduler.get_last_lr()[0]:.2e}{marker}")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} min. Best holdout loss: {best_holdout_loss:.5f}")
    print(f"Pretrained encoder saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
