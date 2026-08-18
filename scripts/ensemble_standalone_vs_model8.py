"""
The real ensemble: freshly-retrained standalone CrossFormer benchmark
(checkpoints/ctg_crossformer_current/, trained 2026-08-17 on current
data/processed/, NOT the stale 2026-08-10 checkpoints) x Model 8 wide-distress
`full` (checkpoints/model8_crossformer_widedistress/, current best Model 8
variant). See scripts/ensemble_oof_eval.py for the methodology this reuses
(per-fold Platt scaling before pooling OOF probabilities -- required, see
that script's docstring for why raw-probability pooling is unreliable).

ASYMMETRY vs. ensemble_oof_eval.py: the standalone benchmark
(train_ctg_crossformer.py) only ever trains/CVs over data/processed/
train_dataset.pt's 381 patients (6,177 windows) -- StratifiedGroupKFold over
that pool alone. The other 165 patients (val_dataset.pt 83 + test_dataset.pt
82 = 6,826 - 6,177 = 649 windows) were NEVER part of its CV loop at all, so
none of its 5 fold checkpoints ever trained on them -- meaning ANY of the 5
checkpoints (or an average of all 5) gives a valid, leak-free prediction for
that pool, just not a single-fold "held-out" one in the usual CV sense. This
script:
  - Uses genuine per-window OOF predictions (own held-out fold) for the first
    6,177 windows (index-aligned with train_dataset.pt's own row order --
    guaranteed identical order to the first 6,177 rows of the pooled
    train+val+test array load_all_multitask_splits() builds, since both come
    from the same file via sequential torch.load with no shuffling).
  - Uses the average of all 5 fold checkpoints' predictions for the remaining
    649 windows (indices 6177:6826) -- still leak-free, just not "OOF" in the
    single-fold sense.
Model 8's fold split (joint-stratified, secondary_labels=dataset.y_figo)
already covers all 546 patients with genuine single-fold OOF throughout, so
its side needs no such split-and-average handling.
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
from src.models.knowledge_infused_framework_wide_distress import (
    CTGCrossformerDualLatentEncoder,
    KnowledgeInfusedFrameworkWideDistress,
)
from src.training.multi_task_dataset import load_all_multitask_splits
from src.training.train import create_patient_level_folds


def build_standalone_model(device):
    encoder = CTGCrossformerEncoder(
        in_channels=2, seq_len=4800, cnn_channels=128,
        n_heads_cross=4, n_heads_tf=8, n_tf_layers=4, d_ff=512,
        dropout=0.1, latent_dim=128,
    )
    return CTGCrossformerForClassification(
        encoder=encoder, hidden_dim=128, dropout=0.3,
    ).to(device)


def build_wide_distress_model(device):
    encoder = CTGCrossformerDualLatentEncoder(
        in_channels=2, seq_len=4800, cnn_channels=128,
        n_heads_cross=4, n_heads_tf=8, n_tf_layers=4, d_ff=512,
        dropout=0.1, latent_dim=128,
    )
    return KnowledgeInfusedFrameworkWideDistress(
        encoder=encoder, head_hidden_dim=128, head_dropout=0.3,
        include_criteria_head=False,
    ).to(device)


@torch.no_grad()
def predict_logits(model, X, idx, device, batch_size=64, distress_index=None):
    model.eval()
    logits = np.zeros(len(idx), dtype=np.float32)
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        xb = X[chunk].to(device)
        out = model(xb)
        if distress_index is not None:
            out = out[distress_index]
        logits[start:start + len(chunk)] = out.squeeze(-1).cpu().numpy()
    return logits


def platt_calibrate(logits, y_true):
    platt = LogisticRegression(C=1e6, max_iter=1000)
    platt.fit(logits.reshape(-1, 1), y_true)
    return platt.predict_proba(logits.reshape(-1, 1))[:, 1]


def compute_standalone_oof(data_dir, checkpoint_dir, device, n_total_windows):
    """Returns a length-n_total_windows array: genuine OOF for the first
    len(train_dataset.pt) windows, 5-checkpoint-averaged for the rest."""
    train_pt = os.path.join(data_dir, "train_dataset.pt")
    data = torch.load(train_pt, map_location="cpu", weights_only=False)
    X_train_pool = data["X"]
    y_train_pool = data["y_primary"].numpy()
    patient_ids_pool = np.array([m[0] for m in data["metadata"]])
    n_train_pool = len(X_train_pool)

    sgkf = StratifiedGroupKFold(n_splits=5)
    folds = list(sgkf.split(X_train_pool.numpy(), y_train_pool, groups=patient_ids_pool))

    full_dataset, _ = load_all_multitask_splits(data_dir)
    X_full = full_dataset.X
    assert len(X_full) == n_total_windows
    # sanity: first n_train_pool rows of the pooled array must match train_dataset.pt's own rows
    assert torch.equal(X_full[:n_train_pool], X_train_pool), (
        "First N rows of load_all_multitask_splits() pool must equal train_dataset.pt's own "
        "row order -- otherwise index alignment between standalone and Model 8 is invalid."
    )

    remainder_idx = np.arange(n_train_pool, n_total_windows)

    pooled_train_raw_logits = np.full(n_train_pool, np.nan, dtype=np.float32)
    remainder_logits_per_fold = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        ckpt_path = os.path.join(checkpoint_dir, f"ctg_crossformer_fold_{fold_idx}_best.pth")
        model = build_standalone_model(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

        pooled_train_raw_logits[val_idx] = predict_logits(model, X_train_pool, val_idx, device)
        remainder_logits_per_fold.append(predict_logits(model, X_full, remainder_idx, device))

        del model
        torch.cuda.empty_cache()
        print(f"  standalone fold {fold_idx}/5 done")

    assert not np.isnan(pooled_train_raw_logits).any()

    # Train-pool: per-fold Platt scaling before pooling (see ensemble_oof_eval.py
    # docstring for why raw-logit pooling across independently-trained folds is
    # unreliable) -- fit uses each window's own held-out fold's raw logits.
    oof_train_pool = np.full(n_train_pool, np.nan, dtype=np.float32)
    for _, val_idx in folds:
        oof_train_pool[val_idx] = platt_calibrate(pooled_train_raw_logits[val_idx], y_train_pool[val_idx])

    # Remainder (val+test patients, never held out by any fold): average raw
    # logits across all 5 folds, then apply a single Platt fit trained on the
    # full train-pool's pooled raw logits (the remainder has no single "home
    # fold" of its own to calibrate against).
    remainder_avg_logits = np.mean(np.stack(remainder_logits_per_fold, axis=0), axis=0)
    platt = LogisticRegression(C=1e6, max_iter=1000)
    platt.fit(pooled_train_raw_logits.reshape(-1, 1), y_train_pool)
    remainder_probs = platt.predict_proba(remainder_avg_logits.reshape(-1, 1))[:, 1]

    oof_full = np.full(n_total_windows, np.nan, dtype=np.float32)
    oof_full[:n_train_pool] = oof_train_pool
    oof_full[n_train_pool:] = remainder_probs
    assert not np.isnan(oof_full).any()
    return oof_full


def compute_model8_oof(data_dir, checkpoint_dir, device, dataset, patient_ids):
    folds = create_patient_level_folds(
        patient_ids, dataset.y_primary, k_folds=5, secondary_labels=dataset.y_figo
    )
    y_np = dataset.y_primary.numpy()
    n = len(patient_ids)
    oof = np.full(n, np.nan, dtype=np.float32)
    for fold_idx, (_, val_idx) in enumerate(folds, 1):
        ckpt_path = os.path.join(checkpoint_dir, f"model8_full_fold{fold_idx}_best.pth")
        model = build_wide_distress_model(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        logits = predict_logits(model, dataset.X, val_idx, device, distress_index=0)
        y_fold = y_np[val_idx]
        oof[val_idx] = platt_calibrate(logits, y_fold)
        del model
        torch.cuda.empty_cache()
        print(f"  wide-distress fold {fold_idx}/5 done")
    assert not np.isnan(oof).any()
    return oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed/")
    parser.add_argument("--standalone_checkpoint_dir", type=str, default="checkpoints/ctg_crossformer_current/")
    parser.add_argument("--model8_checkpoint_dir", type=str, default="checkpoints/model8_crossformer_widedistress/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset, patient_ids = load_all_multitask_splits(args.data_dir)
    y_true = dataset.y_primary.numpy()
    n_total = len(dataset)

    print("\n--- Standalone CrossFormer (current-data retrain) OOF ---")
    oof_standalone = compute_standalone_oof(args.data_dir, args.standalone_checkpoint_dir, device, n_total)
    auroc_standalone = roc_auc_score(y_true, oof_standalone)
    print(f"Pooled OOF AUROC (standalone, current data): {auroc_standalone:.4f}")

    print("\n--- Model 8 wide-distress full OOF ---")
    oof_model8 = compute_model8_oof(args.data_dir, args.model8_checkpoint_dir, device, dataset, patient_ids)
    auroc_model8 = roc_auc_score(y_true, oof_model8)
    print(f"Pooled OOF AUROC (Model 8 wide-distress full):  {auroc_model8:.4f}")

    print("\n--- Ensemble sweep ---")
    best_alpha, best_auroc = 0.5, -1.0
    for alpha in np.arange(0.0, 1.01, 0.1):
        probs = alpha * oof_standalone + (1 - alpha) * oof_model8
        auroc = roc_auc_score(y_true, probs)
        marker = ""
        if auroc > best_auroc:
            best_auroc, best_alpha = auroc, alpha
            marker = "  <-- best so far"
        print(f"  alpha(standalone)={alpha:.1f}: AUROC={auroc:.4f}{marker}")

    ensemble_5050 = 0.5 * oof_standalone + 0.5 * oof_model8
    auroc_5050 = roc_auc_score(y_true, ensemble_5050)

    print("\n================ SUMMARY ================")
    print(f"Standalone (current-data retrain) pooled OOF AUROC: {auroc_standalone:.4f}")
    print(f"Model 8 wide-distress full         pooled OOF AUROC: {auroc_model8:.4f}")
    print(f"50/50 ensemble                     pooled OOF AUROC: {auroc_5050:.4f}")
    print(f"best-alpha ensemble                pooled OOF AUROC: {best_auroc:.4f} (alpha={best_alpha:.1f}, optimistic/non-blind)")
    lift = auroc_5050 - max(auroc_standalone, auroc_model8)
    print(f"50/50 ensemble lift over best single model: {lift:+.4f}")


if __name__ == "__main__":
    main()
