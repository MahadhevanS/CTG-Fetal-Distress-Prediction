"""
Pooled out-of-fold AUROC for the standalone CTG-CrossFormer benchmark
(src/models/train_ctg_crossformer.py), matching the paper's own metric
definition: "AUC computed on pooled predictions across all folds" (Table 1
caption, Dang/Nguyen/Ho, E3S Web of Conferences 723, 01005 (2026)) -- not
train_ctg_crossformer.py's own reported mean-of-per-fold AUROC.

Cannot reuse scripts/ensemble_oof_eval.py's compute_oof_array() -- that
function reconstructs Model 8's fold split via
src/training/train.py::create_patient_level_folds() (patient-level
StratifiedKFold(shuffle=True, random_state=42)), which is a DIFFERENT,
incompatible algorithm from what train_ctg_crossformer.py actually uses:
window-level StratifiedGroupKFold(n_splits=5).split(X, y, groups=patient_ids)
(deterministic, no shuffle/seed -- reproducible given the same data load
order). This script reconstructs THAT fold split directly, then reuses the
same per-fold Platt-scaling-before-pooling pattern validated in
ensemble_oof_eval.py (raw-logit pooling across independently-trained fold
checkpoints was found there to bias pooled AUROC low by ~0.03-0.05).
"""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification


def build_standalone_model(device):
    encoder = CTGCrossformerEncoder(
        in_channels=2, seq_len=4800, cnn_channels=128,
        n_heads_cross=4, n_heads_tf=8, n_tf_layers=4, d_ff=512,
        dropout=0.1, latent_dim=128,
    )
    return CTGCrossformerForClassification(
        encoder=encoder, hidden_dim=128, dropout=0.3,
    ).to(device)


@torch.no_grad()
def predict_logits(model, X, idx, device, batch_size=64):
    model.eval()
    logits = np.zeros(len(idx), dtype=np.float32)
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        xb = X[chunk].to(device)
        out = model(xb)
        logits[start:start + len(chunk)] = out.squeeze(-1).cpu().numpy()
    return logits


def main():
    parser = argparse.ArgumentParser(description="Pooled OOF AUROC for the standalone CTG-CrossFormer benchmark")
    parser.add_argument("--data_dir", type=str, default="data/processed/")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ctg_crossformer/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_pt = os.path.join(args.data_dir, "train_dataset.pt")
    data = torch.load(train_pt, map_location="cpu", weights_only=False)
    X = data["X"].numpy()
    y = data["y_primary"].numpy()
    metadata = data["metadata"]
    patient_ids = np.array([m[0] for m in metadata])
    X_t = torch.as_tensor(X, dtype=torch.float32)

    print(f"Loaded {len(X)} windows, {len(set(patient_ids))} unique patients")

    sgkf = StratifiedGroupKFold(n_splits=5)
    folds = list(sgkf.split(X, y, groups=patient_ids))

    oof = np.full(len(y), np.nan, dtype=np.float32)
    for fold_idx, (_, val_idx) in enumerate(folds, 1):
        ckpt_path = os.path.join(args.checkpoint_dir, f"ctg_crossformer_fold_{fold_idx}_best.pth")
        model = build_standalone_model(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

        logits = predict_logits(model, X_t, val_idx, device)
        y_fold = y[val_idx]
        platt = LogisticRegression(C=1e6, max_iter=1000)
        platt.fit(logits.reshape(-1, 1), y_fold)
        oof[val_idx] = platt.predict_proba(logits.reshape(-1, 1))[:, 1]

        fold_auroc = roc_auc_score(y_fold, logits)
        print(f"  Fold {fold_idx}/5: within-fold AUROC={fold_auroc:.4f} (n={len(val_idx)})")

        del model
        torch.cuda.empty_cache()

    assert not np.isnan(oof).any(), "Every window must be held out exactly once across the 5 folds."

    pooled = roc_auc_score(y, oof)
    print(f"\nPooled OOF AUROC (paper's Table 1 metric definition): {pooled:.4f}")
    print(f"Paper's own reported value: 0.822")
    print(f"Gap: {pooled - 0.822:+.4f}")


if __name__ == "__main__":
    main()
