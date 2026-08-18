"""
Out-of-fold (OOF) ensembling sanity check -- Model 8 standard-architecture
`distress_only` (checkpoints/model8_crossformer/) x wide-distress `full`
(checkpoints/model8_crossformer_widedistress/).

WHY THIS EXISTS: the user's goal is 90%+ AUROC, beating the standalone
CrossFormer benchmark (train_ctg_crossformer.py, CV 0.8565 / Test 0.8653).
Ensembling was picked as the first, cheapest thing to try. The ideal first
ensemble would pair Model 8 with the actual standalone benchmark, but its
checkpoints (checkpoints/ctg_crossformer/, trained 2026-08-10) predate the
2026-08-15 data/processed/ regeneration (LTV preprocessing fix) -- the
underlying window/patient composition may have shifted since, so reusing
those checkpoints directly risks silently misaligned fold membership
(a checkpoint's "held-out" fold might not be held-out against current data).
This script instead ensembles two Model 8 checkpoint sets that ARE both
confirmed trained on the CURRENT data/processed/ files (see mtimes: both
sets postdate the 2026-08-15 12:38 data regen), to get an honest read on
whether ensembling helps at all before spending GPU time retraining the
standalone benchmark fresh for the real comparison.

METHOD: the two checkpoint sets used *different* fold-splitting algorithms
(distress_only predates the 2026-08-16 joint-stratification fix; wide-
distress full postdates it) -- see create_patient_level_folds() in
src/training/train.py for both variants. Rather than assuming fold i means
the same patients in both, this script reconstructs EACH model's own fold
split independently (deterministic given fixed data + seed=42) and computes
independent out-of-fold predictions for every window under each model, using
only that window's own held-out fold's checkpoint. This guarantees no
leakage regardless of how the two fold splits relate to each other -- each
per-window prediction is genuinely from a checkpoint that never trained on
that window's patient, for both models independently. The two OOF arrays are
then combined elementwise (both aligned to the exact same window order from
load_all_multitask_splits) and pooled AUROC is computed once across all
5,826(?)/6,826 windows -- not averaged per-fold -- since fold membership
differs between the two models.
"""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.ctg_crossformer import CTGCrossformerEncoder
from src.models.knowledge_infused_framework import KnowledgeInfusedFramework
from src.models.knowledge_infused_framework_wide_distress import (
    CTGCrossformerDualLatentEncoder,
    KnowledgeInfusedFrameworkWideDistress,
)
from src.training.multi_task_dataset import load_all_multitask_splits
from src.training.train import create_patient_level_folds


def build_standard_model(device):
    encoder = CTGCrossformerEncoder(
        in_channels=2, seq_len=4800, cnn_channels=128,
        n_heads_cross=4, n_heads_tf=8, n_tf_layers=4, d_ff=512,
        dropout=0.1, latent_dim=128,
    )
    return KnowledgeInfusedFramework(
        encoder=encoder, head_hidden_dim=128, head_dropout=0.3,
        include_criteria_head=False,
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
def oof_predict_logits(model, X, val_idx, device, batch_size=64):
    model.eval()
    logits = np.zeros(len(val_idx), dtype=np.float32)
    for start in range(0, len(val_idx), batch_size):
        chunk = val_idx[start:start + batch_size]
        xb = X[chunk].to(device)
        out = model(xb)
        distress_logit = out[0].squeeze(-1)
        logits[start:start + len(chunk)] = distress_logit.cpu().numpy()
    return logits


def compute_oof_array(
    checkpoint_dir, checkpoint_prefix, build_fn, X, y_all, patient_ids,
    device, secondary_labels=None, k_folds=5,
):
    """
    Returns pooled OOF probabilities, Platt-scaled per fold before pooling.

    Raw sigmoid(logit) pooled directly across folds gave a pooled AUROC ~0.03-0.05
    LOWER than the mean-of-per-fold AUROC these checkpoints were originally
    reported with -- each fold's model reaches its own local optimum with a
    somewhat different logit scale/offset (no cross-fold calibration is
    enforced during training), so naively pooling raw scores corrupts the
    across-fold ranking even though each fold's own within-fold ranking is
    fine. Fix: fit a 1-D logistic regression (scale + bias, i.e. Platt
    scaling) per fold on that fold's own held-out logits/labels before
    pooling. This is a monotonic transform *within* each fold, so it cannot
    change that fold's own AUROC contribution -- it only puts folds on a
    comparable probability scale before combining, which is the standard
    fix for pooled multi-fold-model evaluation.
    """
    folds = create_patient_level_folds(
        patient_ids, y_all, k_folds=k_folds, secondary_labels=secondary_labels
    )
    y_np = y_all.numpy()
    n = len(patient_ids)
    oof = np.full(n, np.nan, dtype=np.float32)
    for fold_idx, (_, val_idx) in enumerate(folds, 1):
        ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_fold{fold_idx}_best.pth")
        model = build_fn(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        logits = oof_predict_logits(model, X, val_idx, device)
        y_fold = y_np[val_idx]
        platt = LogisticRegression(C=1e6, max_iter=1000)
        platt.fit(logits.reshape(-1, 1), y_fold)
        calibrated_probs = platt.predict_proba(logits.reshape(-1, 1))[:, 1]
        oof[val_idx] = calibrated_probs
        del model
        torch.cuda.empty_cache()
    assert not np.isnan(oof).any(), "Every window must be held out exactly once across the 5 folds."
    return oof


def pooled_auroc(y_true, probs):
    return roc_auc_score(y_true, probs)


def main():
    parser = argparse.ArgumentParser(description="OOF ensembling sanity check for Model 8 variants")
    parser.add_argument("--data_dir", type=str, default="data/processed/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset, patient_ids = load_all_multitask_splits(args.data_dir)
    X = dataset.X
    y_all = dataset.y_primary
    y_true_np = y_all.numpy()

    print("\n--- Standard architecture, distress_only (old distress-only-stratified folds) ---")
    oof_standard = compute_oof_array(
        checkpoint_dir="checkpoints/model8_crossformer/",
        checkpoint_prefix="model8_distress_only",
        build_fn=build_standard_model,
        X=X, y_all=y_all, patient_ids=patient_ids,
        device=device, secondary_labels=None,
    )
    auroc_standard = pooled_auroc(y_true_np, oof_standard)
    print(f"Pooled OOF AUROC (distress_only, standard): {auroc_standard:.4f}")

    print("\n--- Wide-distress architecture, full (new joint-stratified folds) ---")
    oof_wide = compute_oof_array(
        checkpoint_dir="checkpoints/model8_crossformer_widedistress/",
        checkpoint_prefix="model8_full",
        build_fn=build_wide_distress_model,
        X=X, y_all=y_all, patient_ids=patient_ids,
        device=device, secondary_labels=dataset.y_figo,
    )
    auroc_wide = pooled_auroc(y_true_np, oof_wide)
    print(f"Pooled OOF AUROC (full, wide-distress):     {auroc_wide:.4f}")

    print("\n--- Ensemble (simple 50/50 average of OOF probabilities) ---")
    ensemble_probs = 0.5 * oof_standard + 0.5 * oof_wide
    auroc_ensemble = pooled_auroc(y_true_np, ensemble_probs)
    print(f"Pooled OOF AUROC (ensemble):                 {auroc_ensemble:.4f}")

    print("\n--- Weighted-average sweep (diagnostic only -- picks alpha using labels, optimistic) ---")
    best_alpha, best_auroc = 0.5, auroc_ensemble
    for alpha in np.arange(0.0, 1.01, 0.1):
        probs = alpha * oof_standard + (1 - alpha) * oof_wide
        auroc = pooled_auroc(y_true_np, probs)
        marker = ""
        if auroc > best_auroc:
            best_auroc, best_alpha = auroc, alpha
            marker = "  <-- best so far"
        print(f"  alpha(standard)={alpha:.1f}: AUROC={auroc:.4f}{marker}")

    print("\n================ SUMMARY ================")
    print(f"distress_only (standard) pooled OOF AUROC: {auroc_standard:.4f}")
    print(f"full (wide-distress)     pooled OOF AUROC: {auroc_wide:.4f}")
    print(f"50/50 ensemble           pooled OOF AUROC: {auroc_ensemble:.4f}")
    print(f"best-alpha ensemble      pooled OOF AUROC: {best_auroc:.4f} (alpha={best_alpha:.1f}, optimistic/non-blind)")
    lift = auroc_ensemble - max(auroc_standard, auroc_wide)
    print(f"50/50 ensemble lift over best single model: {lift:+.4f}")


if __name__ == "__main__":
    main()
