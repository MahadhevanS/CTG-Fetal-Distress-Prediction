"""
Phase 1: Minimal Predictive Sanity Check.

Evaluates P0 (Control), P1 (Minimal), and P2 (Quality-aware) using a compact
1D ResNet baseline across the frozen 5-fold patient partition.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.protocol import Protocol
from src.models.cnn1d_encoder import CNN1DEncoder

OUT_DIR = "results/phase1_predictive_sanity"
FOLDS_PATH = "data/processed_clinical/folds.json"
DATA_DIR = "data/phase1_candidates"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CompactClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.encoder = CNN1DEncoder(in_channels=in_channels, seq_len=4800, latent_dim=latent_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        logits = self.head(feat).squeeze(-1)
        return logits


def train_and_eval_fold(model, tr_loader, te_loader, epochs: int = 20, lr: float = 1e-3, pos_weight: float = 3.0):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))

    for ep in range(epochs):
        model.train()
        for x_b, y_b in tr_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for x_b, _ in te_loader:
            x_b = x_b.to(DEVICE)
            logits = model(x_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs.tolist())
    return np.array(preds)


def run_candidate_eval(tag: str, ds_file: str, in_channels: int, n_cont_channels: int = 2):
    print(f"\n--- Evaluating Candidate {tag} ({in_channels} channels) ---")
    data = torch.load(os.path.join(DATA_DIR, ds_file), weights_only=False)
    X = data["X"].numpy()
    y = data["y"].numpy()
    pid = data["pid"]

    # Load frozen protocol
    prot = Protocol.load_or_create(pid, y, path=FOLDS_PATH)

    oof_window_probs = np.zeros(len(y), dtype=float)
    fold_aurocs = []

    for fold, (tr_mask, te_patients) in enumerate(prot.folds(repeat=0), 1):
        te_mask = ~tr_mask
        te_idx = np.where(te_mask)[0]

        # Normalization fit on training fold only
        X_tr = X[tr_mask].copy()
        X_te = X[te_mask].copy()

        # Normalize continuous channels only (e.g. Ch 0, Ch 1)
        means = X_tr[:, :n_cont_channels, :].mean(axis=(0, 2), keepdims=True)
        stds = X_tr[:, :n_cont_channels, :].std(axis=(0, 2), keepdims=True) + 1e-6

        X_tr[:, :n_cont_channels, :] = (X_tr[:, :n_cont_channels, :] - means) / stds
        X_te[:, :n_cont_channels, :] = (X_te[:, :n_cont_channels, :] - means) / stds

        # Compute pos_weight based on train set
        n_pos = int(y[tr_mask].sum())
        n_neg = len(y[tr_mask]) - n_pos
        pos_weight = float(n_neg / max(1, n_pos))

        tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y[tr_mask], dtype=torch.float32))
        te_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y[te_mask], dtype=torch.float32))

        tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, drop_last=False)
        te_loader = DataLoader(te_ds, batch_size=64, shuffle=False)

        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)
        model = CompactClassifier(in_channels=in_channels, latent_dim=128)

        preds = train_and_eval_fold(model, tr_loader, te_loader, epochs=20, lr=1e-3, pos_weight=pos_weight)
        oof_window_probs[te_idx] = preds

        # Measure fold patient-level AUROC
        f_lab, f_sc = prot.to_patient(oof_window_probs, patients=te_patients, how="max")
        f_auc = roc_auc_score(f_lab, f_sc)
        fold_aurocs.append(f_auc)
        print(f"  Fold {fold}: AUROC = {f_auc:.4f} ({len(te_patients)} patients)")

    # Overall patient-level OOF metrics
    rep = prot.report(f"{tag}", oof_window_probs, how="max", verbose=True)
    rep["fold_aurocs"] = fold_aurocs
    rep["oof_window_probs"] = oof_window_probs
    return rep


def paired_bootstrap_test(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    diffs = []
    idx = np.arange(len(labels))
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(labels[b])) < 2:
            continue
        auc_a = roc_auc_score(labels[b], scores_a[b])
        auc_b = roc_auc_score(labels[b], scores_b[b])
        diffs.append(auc_b - auc_a)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_val = float(np.mean(diffs <= 0.0)) if np.mean(diffs) > 0 else float(np.mean(diffs >= 0.0))
    return float(np.mean(diffs)), (float(lo), float(hi)), p_val


def run_all_sanity_checks():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== RUNNING PREDICTIVE SANITY CHECK (P0 vs P1 vs P2) ===")

    rep_p0 = run_candidate_eval("P0_Control", "p0_dataset.pt", in_channels=3, n_cont_channels=2)
    rep_p1 = run_candidate_eval("P1_Minimal", "p1_dataset.pt", in_channels=3, n_cont_channels=2)
    rep_p2 = run_candidate_eval("P2_QualityAware", "p2_dataset.pt", in_channels=4, n_cont_channels=2)

    # Patient labels and scores
    p0_data = torch.load(os.path.join(DATA_DIR, "p0_dataset.pt"), weights_only=False)
    prot = Protocol.load_or_create(p0_data["pid"],
                                   p0_data["y"].numpy(),
                                   path=FOLDS_PATH)
    
    lab, sc_p0 = prot.to_patient(rep_p0["oof_window_probs"], how="max")
    _, sc_p1 = prot.to_patient(rep_p1["oof_window_probs"], how="max")
    _, sc_p2 = prot.to_patient(rep_p2["oof_window_probs"], how="max")

    # Paired tests
    diff_p1, ci_p1, pval_p1 = paired_bootstrap_test(lab, sc_p0, sc_p1)
    diff_p2, ci_p2, pval_p2 = paired_bootstrap_test(lab, sc_p0, sc_p2)

    results = {
        "P0_Control": {
            "auroc": rep_p0["auroc"],
            "auprc": rep_p0["auprc"],
            "ci": [rep_p0["ci_lo"], rep_p0["ci_hi"]],
            "fold_aurocs": rep_p0["fold_aurocs"]
        },
        "P1_Minimal": {
            "auroc": rep_p1["auroc"],
            "auprc": rep_p1["auprc"],
            "ci": [rep_p1["ci_lo"], rep_p1["ci_hi"]],
            "fold_aurocs": rep_p1["fold_aurocs"],
            "delta_auroc_vs_p0": diff_p1,
            "delta_ci_95": list(ci_p1),
            "p_val": pval_p1
        },
        "P2_QualityAware": {
            "auroc": rep_p2["auroc"],
            "auprc": rep_p2["auprc"],
            "ci": [rep_p2["ci_lo"], rep_p2["ci_hi"]],
            "fold_aurocs": rep_p2["fold_aurocs"],
            "delta_auroc_vs_p0": diff_p2,
            "delta_ci_95": list(ci_p2),
            "p_val": pval_p2
        }
    }

    print("\n=== FINAL COMPARISON SUMMARY ===")
    print(f"P0 (Control)       : AUROC {rep_p0['auroc']:.4f} [{rep_p0['ci_lo']:.3f}-{rep_p0['ci_hi']:.3f}] | AUPRC {rep_p0['auprc']:.4f}")
    print(f"P1 (Minimal)       : AUROC {rep_p1['auroc']:.4f} [{rep_p1['ci_lo']:.3f}-{rep_p1['ci_hi']:.3f}] | AUPRC {rep_p1['auprc']:.4f} | dAUROC {diff_p1:+.4f} [{ci_p1[0]:.3f}, {ci_p1[1]:.3f}] (p={pval_p1:.3f})")
    print(f"P2 (Quality-Aware) : AUROC {rep_p2['auroc']:.4f} [{rep_p2['ci_lo']:.3f}-{rep_p2['ci_hi']:.3f}] | AUPRC {rep_p2['auprc']:.4f} | dAUROC {diff_p2:+.4f} [{ci_p2[0]:.3f}, {ci_p2[1]:.3f}] (p={pval_p2:.3f})")

    with open(os.path.join(OUT_DIR, "predictive_sanity_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    return results


if __name__ == "__main__":
    run_all_sanity_checks()
