"""
Phase 2: CTU-UHB Multi-Representation Learning.

Evaluates Raw 1D, Multi-Channel 1D, CWT 2D, CWT+Mask 2D, and Recurrence Plot (RP) 2D
representations under the strict frozen patient-grouped 5-fold protocol.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.protocol import Protocol
from src.models.cnn1d_encoder import CNN1DEncoder
from src.cwt.wavelet import ContinuousWaveletTransform
from src.cwt.recurrence import RecurrencePlot
from src.cwt.models_2d import SmallCNN2D

FOLDS_PATH = "data/processed_clinical/folds.json"
DATA_DIR = "data/phase1_candidates"
OUT_DIR = "results/phase2_representation"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25
BATCH_SIZE = 64
LR = 1e-3


# ---------------------------------------------------------------------------
# 1D Baseline Model
# ---------------------------------------------------------------------------
class Model1D(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 128):
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
        return self.head(feat).squeeze(-1)


# ---------------------------------------------------------------------------
# 2D CWT Model (Waveform -> CWT -> 2D ResNet)
# ---------------------------------------------------------------------------
class ModelCWT2D(nn.Module):
    def __init__(self, in_channels: int = 1, num_scales: int = 64, time_pool: int = 8, width: int = 32):
        super().__init__()
        self.cwt = ContinuousWaveletTransform(num_scales=num_scales, time_pool=time_pool)
        self.cnn2d = SmallCNN2D(in_channels=in_channels, width=width, dropout=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, in_channels, 4800)
        # Scalogram: (Batch, in_channels, num_scales, 4800 // time_pool)
        scalogram = self.cwt(x)
        # Scale logarithmically for time-frequency contrast: log(1 + scalogram)
        scalogram_log = torch.log1p(scalogram)
        return self.cnn2d(scalogram_log).squeeze(-1)


# ---------------------------------------------------------------------------
# 2D CWT + Mask Model
# ---------------------------------------------------------------------------
class ModelCWTPlusMask2D(nn.Module):
    def __init__(self, num_scales: int = 64, time_pool: int = 8, width: int = 32):
        super().__init__()
        self.cwt = ContinuousWaveletTransform(num_scales=num_scales, time_pool=time_pool)
        self.time_pool = time_pool
        self.num_scales = num_scales
        self.cnn2d = SmallCNN2D(in_channels=2, width=width, dropout=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x[:, 0:1, :] is FHR signal, x[:, 1:2, :] is quality mask
        fhr = x[:, 0:1, :]
        mask = x[:, 1:2, :]

        scalogram = self.cwt(fhr) # (B, 1, 64, 600)
        scalogram_log = torch.log1p(scalogram)

        # Pool mask temporally and repeat across frequency scales
        mask_pooled = F.avg_pool1d(mask, kernel_size=self.time_pool, stride=self.time_pool) # (B, 1, 600)
        mask_2d = mask_pooled.unsqueeze(2).repeat(1, 1, self.num_scales, 1) # (B, 1, 64, 600)

        img = torch.cat([scalogram_log, mask_2d], dim=1) # (B, 2, 64, 600)
        return self.cnn2d(img).squeeze(-1)


# ---------------------------------------------------------------------------
# 2D Recurrence Plot Model (Waveform -> RP -> 2D ResNet)
# ---------------------------------------------------------------------------
class ModelRP2D(nn.Module):
    def __init__(self, dimension: int = 3, time_delay: int = 4, target_size: int = 128, width: int = 32):
        super().__init__()
        self.rp = RecurrencePlot(dimension=dimension, time_delay=time_delay, target_size=target_size)
        self.cnn2d = SmallCNN2D(in_channels=1, width=width, dropout=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 1, 4800) -> rp: (Batch, 1, 128, 128)
        rp_img = self.rp(x[:, 0:1, :])
        return self.cnn2d(rp_img).squeeze(-1)


# ---------------------------------------------------------------------------
# Training Harness
# ---------------------------------------------------------------------------
def train_and_eval_fold(model, tr_loader, te_loader, epochs: int = EPOCHS, lr: float = LR, pos_weight: float = 3.0):
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


def run_experiment(exp_name: str, model_builder, data_tensor: np.ndarray, y_arr: np.ndarray, pid_arr: np.ndarray,
                   cont_channel_indices: list):
    print(f"\n=======================================================")
    print(f"RUNNING: {exp_name}")
    print(f"=======================================================")
    prot = Protocol.load_or_create(pid_arr, y_arr, path=FOLDS_PATH)

    oof_window_probs = np.zeros(len(y_arr), dtype=float)
    fold_aurocs = []
    fold_auprcs = []
    fold_win_aurocs = []

    for fold, (tr_mask, te_patients) in enumerate(prot.folds(repeat=0), 1):
        te_mask = ~tr_mask
        te_idx = np.where(te_mask)[0]

        X_tr = data_tensor[tr_mask].copy()
        X_te = data_tensor[te_mask].copy()

        # Training-set normalization on continuous channels only
        if cont_channel_indices:
            means = X_tr[:, cont_channel_indices, :].mean(axis=(0, 2), keepdims=True)
            stds = X_tr[:, cont_channel_indices, :].std(axis=(0, 2), keepdims=True) + 1e-6

            X_tr[:, cont_channel_indices, :] = (X_tr[:, cont_channel_indices, :] - means) / stds
            X_te[:, cont_channel_indices, :] = (X_te[:, cont_channel_indices, :] - means) / stds

        n_pos = int(y_arr[tr_mask].sum())
        n_neg = len(y_arr[tr_mask]) - n_pos
        pos_weight = float(n_neg / max(1, n_pos))

        tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_arr[tr_mask], dtype=torch.float32))
        te_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_arr[te_mask], dtype=torch.float32))

        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)
        model = model_builder()

        preds = train_and_eval_fold(model, tr_loader, te_loader, epochs=EPOCHS, lr=LR, pos_weight=pos_weight)
        oof_window_probs[te_idx] = preds

        # Window-level and patient-level metrics per fold
        win_auc = roc_auc_score(y_arr[te_mask], preds)
        fold_win_aurocs.append(win_auc)

        f_lab, f_sc = prot.to_patient(oof_window_probs, patients=te_patients, how="max")
        f_auc = roc_auc_score(f_lab, f_sc)
        f_prc = average_precision_score(f_lab, f_sc)
        fold_aurocs.append(f_auc)
        fold_auprcs.append(f_prc)
        print(f"  Fold {fold}: Patient AUROC = {f_auc:.4f} | Window AUROC = {win_auc:.4f} | AUPRC = {f_prc:.4f}")

    # Parameter count
    param_count = sum(p.numel() for p in model_builder().parameters())

    # Overall patient-level report
    rep = prot.report(exp_name, oof_window_probs, how="max", verbose=True)
    overall_win_auc = roc_auc_score(y_arr, oof_window_probs)

    rep["param_count"] = param_count
    rep["window_auroc"] = float(overall_win_auc)
    rep["fold_aurocs"] = fold_aurocs
    rep["fold_auprcs"] = fold_auprcs
    rep["fold_win_aurocs"] = fold_win_aurocs
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


def run_phase2():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== STARTING PHASE 2: MULTI-REPRESENTATION LEARNING ===")

    # Load P2 (Quality-Aware) dataset from Phase 1
    # P2 Channels: Ch 0 = FHR(t), Ch 1 = Delta FHR(t), Ch 2 = Observed Mask, Ch 3 = Interp Mask
    p2_data = torch.load(os.path.join(DATA_DIR, "p2_dataset.pt"), weights_only=False)
    X_p2 = p2_data["X"].numpy()
    y_arr = p2_data["y"].numpy()
    pid_arr = p2_data["pid"]

    prot = Protocol.load_or_create(pid_arr, y_arr, path=FOLDS_PATH)

    results = {}

    # -------------------------------------------------------------
    # Exp 2.1: Raw FHR -> 1D CNN (1 Channel: FHR(t))
    # -------------------------------------------------------------
    X_exp1 = X_p2[:, 0:1, :] # (8517, 1, 4800)
    rep_2_1 = run_experiment(
        "Exp2.1_Raw_FHR_1D",
        lambda: Model1D(in_channels=1, latent_dim=128),
        X_exp1, y_arr, pid_arr, cont_channel_indices=[0]
    )
    results["Exp2.1_Raw_FHR_1D"] = rep_2_1

    # -------------------------------------------------------------
    # Exp 2.2: Raw + Delta FHR + Mask -> 1D CNN (3 Channels: FHR, Delta, Mask)
    # -------------------------------------------------------------
    X_exp2 = X_p2[:, :3, :] # (8517, 3, 4800)
    rep_2_2 = run_experiment(
        "Exp2.2_MultiChannel_1D",
        lambda: Model1D(in_channels=3, latent_dim=128),
        X_exp2, y_arr, pid_arr, cont_channel_indices=[0, 1]
    )
    results["Exp2.2_MultiChannel_1D"] = rep_2_2

    # -------------------------------------------------------------
    # Exp 2.3: CWT Scalogram -> 2D CNN (1 Channel: CWT of FHR)
    # -------------------------------------------------------------
    X_exp3 = X_p2[:, 0:1, :] # (8517, 1, 4800)
    rep_2_3 = run_experiment(
        "Exp2.3_CWT_2D",
        lambda: ModelCWT2D(in_channels=1, num_scales=64, time_pool=8, width=32),
        X_exp3, y_arr, pid_arr, cont_channel_indices=[0]
    )
    results["Exp2.3_CWT_2D"] = rep_2_3

    # -------------------------------------------------------------
    # Exp 2.4: CWT + Mask -> 2D CNN (2 Channels: CWT of Delta FHR + Mask)
    # -------------------------------------------------------------
    X_exp4 = np.concatenate([X_p2[:, 1:2, :], X_p2[:, 2:3, :]], axis=1) # (8517, 2, 4800)
    rep_2_4 = run_experiment(
        "Exp2.4_CWT_Plus_Mask_2D",
        lambda: ModelCWTPlusMask2D(num_scales=64, time_pool=8, width=32),
        X_exp4, y_arr, pid_arr, cont_channel_indices=[0]
    )
    results["Exp2.4_CWT_Plus_Mask_2D"] = rep_2_4

    # -------------------------------------------------------------
    # Exp 2.5: Recurrence Plot (RP) -> 2D CNN (1 Channel: RP)
    # -------------------------------------------------------------
    X_exp5 = X_p2[:, 1:2, :] # Delta FHR for RP
    rep_2_5 = run_experiment(
        "Exp2.5_RP_2D",
        lambda: ModelRP2D(dimension=3, time_delay=4, target_size=128, width=32),
        X_exp5, y_arr, pid_arr, cont_channel_indices=[0]
    )
    results["Exp2.5_RP_2D"] = rep_2_5

    # -------------------------------------------------------------
    # Secondary: Late Fusion (Multi-View 1D + 2D CWT)
    # -------------------------------------------------------------
    # Compute patient scores for each model
    lab, sc_raw = prot.to_patient(rep_2_1["oof_window_probs"], how="max")
    _, sc_multi = prot.to_patient(rep_2_2["oof_window_probs"], how="max")
    _, sc_cwt = prot.to_patient(rep_2_3["oof_window_probs"], how="max")
    _, sc_cwt_m = prot.to_patient(rep_2_4["oof_window_probs"], how="max")
    _, sc_rp = prot.to_patient(rep_2_5["oof_window_probs"], how="max")

    # Simple equal-weight late fusion of MultiChannel 1D and CWT+Mask 2D
    sc_fusion = 0.5 * sc_multi + 0.5 * sc_cwt_m
    auc_fusion = roc_auc_score(lab, sc_fusion)
    prc_fusion = average_precision_score(lab, sc_fusion)
    ci_lo_fus, ci_hi_fus = prot.bootstrap_ci(lab, sc_fusion)
    print(f"\n  Secondary Late Fusion (1D Multi + 2D CWT)  AUROC {auc_fusion:.4f} [{ci_lo_fus:.3f}-{ci_hi_fus:.3f}]  AUPRC {prc_fusion:.4f}")

    results["Secondary_Late_Fusion"] = {
        "name": "Secondary_Late_Fusion",
        "auroc": auc_fusion,
        "auprc": prc_fusion,
        "ci_lo": ci_lo_fus,
        "ci_hi": ci_hi_fus,
        "param_count": rep_2_2["param_count"] + rep_2_4["param_count"]
    }

    # -------------------------------------------------------------
    # Paired Statistical Tests against Raw FHR Control (Exp 2.1)
    # -------------------------------------------------------------
    models_to_test = [
        ("MultiChannel_1D", sc_multi, rep_2_2["fold_aurocs"]),
        ("CWT_2D", sc_cwt, rep_2_3["fold_aurocs"]),
        ("CWT_Plus_Mask_2D", sc_cwt_m, rep_2_4["fold_aurocs"]),
        ("RP_2D", sc_rp, rep_2_5["fold_aurocs"]),
        ("Late_Fusion", sc_fusion, None)
    ]

    print("\n=======================================================")
    print("PAIRED STATISTICAL COMPARISONS vs Raw FHR 1D Control")
    print("=======================================================")
    comparison_table = []
    for tag, sc_m, f_aucs in models_to_test:
        diff, ci, pval = paired_bootstrap_test(lab, sc_raw, sc_m)
        wins = int(sum([f_m > f_r for f_m, f_r in zip(f_aucs, rep_2_1["fold_aurocs"])])) if f_aucs else "N/A"
        print(f"  {tag:22s}: dAUROC = {diff:+.4f} [{ci[0]:.3f}, {ci[1]:.3f}] | p={pval:.3f} | 5-Fold Wins: {wins}/5")
        comparison_table.append({
            "model": tag,
            "delta_auroc": diff,
            "ci_95": list(ci),
            "p_val": pval,
            "wins_vs_raw": wins
        })

    # Save summary results
    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {
            "name": v["name"],
            "auroc": float(v["auroc"]),
            "auprc": float(v["auprc"]),
            "ci": [float(v["ci_lo"]), float(v["ci_hi"])],
            "param_count": int(v["param_count"]),
            "window_auroc": float(v.get("window_auroc", 0.0)),
            "fold_aurocs": [float(x) for x in v.get("fold_aurocs", [])]
        }

    summary_payload = {
        "benchmark_clinical_lr": {
            "auroc": 0.7271,
            "ci": [0.670, 0.779],
            "auprc": 0.409
        },
        "results": clean_results,
        "comparisons_vs_raw": comparison_table,
        "patient_scores": {
            "labels": [int(x) for x in lab],
            "raw_1d": [float(x) for x in sc_raw],
            "multi_1d": [float(x) for x in sc_multi],
            "cwt_2d": [float(x) for x in sc_cwt],
            "cwt_mask_2d": [float(x) for x in sc_cwt_m],
            "rp_2d": [float(x) for x in sc_rp],
            "fusion": [float(x) for x in sc_fusion]
        }
    }

    with open(os.path.join(OUT_DIR, "phase2_results.json"), "w") as fh:
        json.dump(summary_payload, fh, indent=2)

    print(f"\nSuccessfully saved Phase 2 evaluation results to {OUT_DIR}/phase2_results.json")
    return summary_payload


if __name__ == "__main__":
    run_phase2()
