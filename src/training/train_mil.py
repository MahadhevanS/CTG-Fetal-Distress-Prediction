"""
Patient-level (MIL) training + 5-fold CV for Model 9 (KG-MIL).

Primary metric: PATIENT-level AUROC -- one prediction per patient, which is the
question the clinical label (umbilical pH <= 7.15) actually poses. Window-level
AUROC is reported alongside for continuity with Models 1-8.

Ablation ladder (one variable at a time):
    --ablation plain        plain ABMIL, no knowledge  (Stage 1 baseline)
    --ablation knowledge    + knowledge-guided attention
    --ablation trajectory   + clinical trajectory features
    --ablation full         both

HARD GATE: refuses to run on a dataset whose bag size predicts the label
(AUROC >= 0.60). data/processed/ fails this at 0.9947 -- see
scripts/audit_bag_size_leak.py.

Usage:
    python src/training/train_mil.py --ablation plain --seed 42
    python src/training/train_mil.py --ablation full --pretrained_encoder <path>
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.ctg_crossformer import CTGCrossformerEncoder
from src.models.knowledge_guided_mil import KnowledgeGuidedMIL
from src.training.mil_dataset import MILBagDataset, collate_mil_bags, load_mil_splits
from src.training.train import create_patient_level_folds

ABLATIONS = {
    "plain":      dict(use_knowledge_attention=False, use_trajectory=False),
    "knowledge":  dict(use_knowledge_attention=True,  use_trajectory=False),
    "trajectory": dict(use_knowledge_attention=False, use_trajectory=True),
    "full":       dict(use_knowledge_attention=True,  use_trajectory=True),
}

BAG_LEAK_THRESHOLD = 0.60


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(ablation: str, feature_means, feature_stds, device,
                pretrained_encoder: Optional[str] = None) -> KnowledgeGuidedMIL:
    encoder = CTGCrossformerEncoder(
        in_channels=2, seq_len=4800, cnn_channels=128,
        n_heads_cross=4, n_heads_tf=8, n_tf_layers=4, d_ff=512,
        dropout=0.1, latent_dim=128,
    )
    if pretrained_encoder:
        state = torch.load(pretrained_encoder, map_location="cpu", weights_only=True)
        # Accept either a bare encoder state_dict or a full classification
        # checkpoint (train_ctg_crossformer.py saves the latter).
        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        encoder.load_state_dict(enc_state or state, strict=True)
        print(f"  Warm-started encoder from {pretrained_encoder}")

    return KnowledgeGuidedMIL(
        encoder=encoder,
        feature_means=feature_means, feature_stds=feature_stds,
        **ABLATIONS[ablation],
    ).to(device)


def run_epoch(model, loader, device, optimizer=None, scheduler=None,
              lambda_window: float = 0.3, scaler=None, grad_clip: float = 1.0):
    train = optimizer is not None
    model.train(train)
    total, n_batches = 0.0, 0
    p_logits, p_targets, w_logits, w_targets = [], [], [], []

    for batch in loader:
        X = batch["X"].to(device, non_blocking=True)
        mask = batch["mask"].to(device)
        yf = batch["y_features"].to(device)
        yp = batch["y_patient"].to(device)
        yw = batch["y_primary"].to(device)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                out = model(X, mask, yf)
                loss_patient = nn.functional.binary_cross_entropy_with_logits(
                    out["patient_logit"], yp
                )
                # Window loss averaged over real windows only
                wl = nn.functional.binary_cross_entropy_with_logits(
                    out["window_logits"], yw, reduction="none"
                )
                loss_window = (wl * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
                loss = loss_patient + lambda_window * loss_window

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total += float(loss.item())
        n_batches += 1
        p_logits.append(out["patient_logit"].detach().float().cpu().numpy())
        p_targets.append(yp.detach().cpu().numpy())
        m = mask.detach().cpu().numpy().astype(bool)
        w_logits.append(out["window_logits"].detach().float().cpu().numpy()[m])
        w_targets.append(yw.detach().cpu().numpy()[m])

    p_logits, p_targets = np.concatenate(p_logits), np.concatenate(p_targets)
    w_logits, w_targets = np.concatenate(w_logits), np.concatenate(w_targets)

    def safe_auc(y, s):
        return roc_auc_score(y, s) if len(set(y.tolist())) > 1 else float("nan")

    return {
        "loss": total / max(n_batches, 1),
        "patient_auroc": safe_auc(p_targets, p_logits),
        "patient_auprc": average_precision_score(p_targets, p_logits) if len(set(p_targets.tolist())) > 1 else float("nan"),
        "window_auroc": safe_auc(w_targets, w_logits),
        "patient_logits": p_logits,
        "patient_targets": p_targets,
    }


def main():
    ap = argparse.ArgumentParser(description="KG-MIL patient-level training (Model 9)")
    ap.add_argument("--data_dir", type=str, default="data/processed_mil/")
    ap.add_argument("--ablation", type=str, default="plain", choices=list(ABLATIONS) + ["all"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=4, help="bags per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--lambda_window", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained_encoder", type=str, default=None)
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints/model9_kgmil/")
    ap.add_argument("--max_folds", type=int, default=None, help="debug: stop after N folds")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    results_dir = os.path.join(args.checkpoint_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    X, y_patient, y_primary, y_figo, y_features, patient_ids, starts = load_mil_splits(args.data_dir)
    print(f"[MIL] {len(X)} windows | {len(set(patient_ids))} patients | "
          f"window prevalence {y_primary.float().mean():.3f}")

    scaler_path = os.path.join(args.data_dir, "feature_scaler.npz")
    if os.path.exists(scaler_path):
        s = np.load(scaler_path)
        f_means = torch.tensor(s["feature_means"], dtype=torch.float32)
        f_stds = torch.tensor(s["feature_stds"], dtype=torch.float32)
    else:
        f_means, f_stds = torch.zeros(8), torch.ones(8)

    folds = create_patient_level_folds(
        patient_ids, y_patient, k_folds=args.k_folds, secondary_labels=y_figo
    )

    ablations = list(ABLATIONS) if args.ablation == "all" else [args.ablation]

    for ablation in ablations:
        print(f"\n{'#'*66}\n# KG-MIL | ablation: {ablation.upper()} | device: {device}\n{'#'*66}")
        fold_metrics: List[Dict] = []
        oof_logits = np.full(len(set(patient_ids)), np.nan, dtype=np.float32)
        oof_targets = np.full(len(set(patient_ids)), np.nan, dtype=np.float32)
        pid_index = {p: i for i, p in enumerate(sorted(set(patient_ids)))}

        for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
            if args.max_folds and fold_idx > args.max_folds:
                break

            tr = MILBagDataset(X, y_patient, y_primary, y_figo, y_features, patient_ids, starts, train_idx)
            va = MILBagDataset(X, y_patient, y_primary, y_figo, y_features, patient_ids, starts, val_idx)

            # HARD GATE -- bag size must not predict the label
            leak = tr.bag_size_leak_auroc
            if not np.isnan(leak) and leak >= BAG_LEAK_THRESHOLD:
                raise RuntimeError(
                    f"Bag-size leak AUROC {leak:.4f} >= {BAG_LEAK_THRESHOLD} on fold {fold_idx} "
                    f"train bags. Patient-level results on this dataset would be an artifact of "
                    f"bag size. Use a uniform-stride dataset (src/preprocessing/pipeline_mil.py)."
                )

            print(f"\n--- Fold {fold_idx}/{args.k_folds} | train {len(tr)} bags "
                  f"(prev {tr.bag_labels.mean():.3f}) | val {len(va)} bags "
                  f"(prev {va.bag_labels.mean():.3f}) | bag-size leak {leak:.3f} ---")

            # Bag-level class balancing
            w = np.where(tr.bag_labels == 1,
                         1.0 / max(tr.bag_labels.sum(), 1),
                         1.0 / max((1 - tr.bag_labels).sum(), 1))
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)

            train_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                                      collate_fn=collate_mil_bags)
            val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=collate_mil_bags)

            model = build_model(ablation, f_means, f_stds, device, args.pretrained_encoder)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr,
                total_steps=max(args.epochs * max(len(train_loader), 1), 10), pct_start=0.1,
            )
            amp = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

            best_auc, best_val = -1.0, None
            for ep in range(1, args.epochs + 1):
                run_epoch(model, train_loader, device, optimizer, sched, args.lambda_window, amp)
                with torch.no_grad():
                    val = run_epoch(model, val_loader, device, None, None, args.lambda_window, None)
                if not np.isnan(val["patient_auroc"]) and val["patient_auroc"] > best_auc:
                    best_auc, best_val = val["patient_auroc"], val
                    torch.save(model.state_dict(),
                               os.path.join(args.checkpoint_dir, f"kgmil_{ablation}_fold{fold_idx}_best.pth"))
                if ep % 10 == 0 or ep == args.epochs:
                    lam = float(model.attention.knowledge_lambda.detach().cpu())
                    print(f"  ep {ep:>3}/{args.epochs} | val patient AUROC {val['patient_auroc']:.4f} "
                          f"| window AUROC {val['window_auroc']:.4f} | lambda {lam:+.3f}")

            print(f"  Fold {fold_idx} BEST patient AUROC: {best_auc:.4f} "
                  f"(window {best_val['window_auroc']:.4f}, AUPRC {best_val['patient_auprc']:.4f})")
            fold_metrics.append({
                "fold": fold_idx, "patient_auroc": best_auc,
                "window_auroc": best_val["window_auroc"], "patient_auprc": best_val["patient_auprc"],
            })
            for p, lg, tg in zip(va.patient_order, best_val["patient_logits"], best_val["patient_targets"]):
                oof_logits[pid_index[p]] = lg
                oof_targets[pid_index[p]] = tg

            del model
            torch.cuda.empty_cache()

        aucs = [m["patient_auroc"] for m in fold_metrics]
        waucs = [m["window_auroc"] for m in fold_metrics]
        print(f"\n{'='*66}\n RESULTS | KG-MIL {ablation.upper()}\n{'='*66}")
        print(f" PATIENT-level AUROC : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}   <-- primary")
        print(f" window-level AUROC  : {np.nanmean(waucs):.4f} +/- {np.nanstd(waucs):.4f}")
        print(f" per-fold            : {', '.join(f'{a:.4f}' for a in aucs)}")

        valid = ~np.isnan(oof_logits)
        if valid.sum() and len(set(oof_targets[valid].tolist())) > 1:
            print(f" pooled OOF patient AUROC: {roc_auc_score(oof_targets[valid], oof_logits[valid]):.4f}")

        with open(os.path.join(results_dir, f"kgmil_{ablation}_cv_results.json"), "w") as f:
            json.dump({"ablation": ablation, "folds": fold_metrics,
                       "patient_auroc_mean": float(np.mean(aucs)),
                       "patient_auroc_std": float(np.std(aucs)),
                       "seed": args.seed, "epochs": args.epochs}, f, indent=2)
        print(f"[Saved] {results_dir}/kgmil_{ablation}_cv_results.json")


if __name__ == "__main__":
    main()
