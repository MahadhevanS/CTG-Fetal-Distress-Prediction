"""
Phase 6: CTU-UHB Continuous and Ordinal Acid-Base Supervision.

Implements and evaluates:
- Exp 6.0: Phase-4 Master Model Reproduction (Binary 1D ResNet + 19 Descriptors + Logit Modulation)
- Exp 6.1: Continuous Signal pH Regression (Huber Loss, S = -pH_hat, P90 aggregation)
- Exp 6.2: Continuous Clinical pH Regression (Huber/Ridge Regression on 19 Descriptors)
- Exp 6.3: Continuous Knowledge Fusion (Signal pH + Clinical pH Fusion)
- Exp 6.4: Ordinal Signal Supervision (Cumulative link multi-threshold BCE: <=7.05, <=7.15, <=7.25)
- Exp 6.5: Ordinal Knowledge-Guided Fusion (Signal + Clinical Ordinal Modulation)
- Exp 6.6: Soft-Target Binary Smoothing (Sigmoidal soft targets around 7.15 with tau in {0.01, 0.02, 0.03})
- Exp 6.7: Multi-Task Continuous + Ordinal Supervision
- Exp 6.8: Secondary & Composite Endpoints (BDecf >= 8, pH <= 7.15 OR BDecf >= 8, pH <= 7.15 AND BDecf >= 8)
- Paired statistical tests (2,000 bootstrap replicates) vs Phase 4 Baseline (0.7361)
- Secondary threshold performance (<=7.05, <=7.15, <=7.20, <=7.25) and pH severity band audits.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, HuberRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error, brier_score_loss, roc_curve
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.protocol import Protocol
from src.models.cnn1d_encoder import CNN1DEncoder

DATA_DIR = "data/processed_clinical"
P2_PATH = "data/phase1_candidates/p2_dataset.pt"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
FOLDS_PATH = "data/processed_clinical/folds.json"
OUT_DIR = "results/phase6_outcome_supervision"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 35
BATCH_SIZE = 64
LR = 1e-3


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


def bootstrap_ci(labels: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(labels))
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(labels[b])) < 2:
            continue
        aucs.append(roc_auc_score(labels[b], scores[b]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def safe_logit(p, eps=1e-5):
    p_c = np.clip(p, eps, 1.0 - eps)
    return np.log(p_c / (1.0 - p_c))


def run_phase6():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== EXECUTING PHASE 6: CONTINUOUS & ORDINAL ACID-BASE SUPERVISION ===")
    t0 = time.time()

    # 1. Load P2 Quality-Aware 20m Dataset (8,517 windows, 547 patients)
    p2_data = torch.load(P2_PATH, weights_only=False)
    X_raw = p2_data["X"] # (8517, 4, 4800)
    y_bin = p2_data["y"].numpy()
    pid_arr = p2_data["pid"]
    meta_p2 = [tuple(m) for m in p2_data["meta"]]

    # Use channels [1 (Delta FHR), 2 (Observed Mask)] for 1D ResNet
    X_tensor = X_raw[:, [1, 2], :]

    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    prot = Protocol.load_or_create(pid_arr, y_bin, path=FOLDS_PATH)
    clean_pids = prot.patients

    # Load Clinical Metadata (continuous pH, BDecf, etc.)
    df_meta = pd.read_csv(METADATA_PATH)
    if 'record_id' in df_meta.columns:
        df_meta = df_meta.set_index('record_id')

    patient_ph = np.array([float(df_meta.loc[int(p), 'ph']) for p in clean_pids], dtype=np.float32)
    patient_bdecf_raw = [df_meta.loc[int(p), 'bdecf'] if int(p) in df_meta.index and not pd.isna(df_meta.loc[int(p), 'bdecf']) else np.nan for p in clean_pids]
    patient_bdecf = np.array([float(x) if not np.isnan(x) else 4.56 for x in patient_bdecf_raw], dtype=np.float32)

    # Window-level targets mapped from patient metadata
    pid_to_ph = {str(p): float(df_meta.loc[int(p), 'ph']) for p in clean_pids}
    pid_to_bdecf = {str(p): float(patient_bdecf[i]) for i, p in enumerate(clean_pids)}
    y_ph_windows = np.array([pid_to_ph[str(p)] for p in pid_arr], dtype=np.float32)

    # 2. Load 19 Clinical Descriptors aligned with p2_dataset.pt
    c_meta = []
    c_Fe = []
    for s in ("train", "val", "test"):
        d = torch.load(os.path.join(DATA_DIR, f"{s}_dataset.pt"), weights_only=False)
        ext = np.load(os.path.join(DATA_DIR, f"{s}_extended_features.npy"))
        fe_split = np.hstack([d["y_features"].numpy(), ext])
        for m, f_row in zip(d["metadata"], fe_split):
            c_meta.append((str(m[0]), int(m[1]), int(m[2])))
            c_Fe.append(f_row)

    feat_lookup = {k: v for k, v in zip(c_meta, c_Fe)}
    Fe_aligned = np.array([feat_lookup[(str(m[0]), int(m[1]), int(m[2]))] for m in meta_p2], dtype=np.float32)

    # Patient-level clinical descriptor matrix (P90)
    F_patient = np.zeros((len(clean_pids), Fe_aligned.shape[1]), dtype=np.float32)
    for i, p in enumerate(clean_pids):
        idx = prot.pidx[p]
        F_patient[i] = np.percentile(Fe_aligned[idx], 90, axis=0)

    print(f"Loaded {len(y_bin)} windows from {len(clean_pids)} patients ({int(prot.plab.sum())} positive under pH <= 7.15).")
    print(f"Cohort pH: Mean = {patient_ph.mean():.3f}, SD = {patient_ph.std():.3f}, Min = {patient_ph.min():.2f}, Max = {patient_ph.max():.2f}")

    results = {
        "ph_distribution_audit": {
            "mean_ph": float(patient_ph.mean()),
            "std_ph": float(patient_ph.std()),
            "median_ph": float(np.median(patient_ph)),
            "min_ph": float(patient_ph.min()),
            "max_ph": float(patient_ph.max()),
            "count_le_705": int((patient_ph <= 7.05).sum()),
            "count_705_715": int(((patient_ph > 7.05) & (patient_ph <= 7.15)).sum()),
            "count_715_725": int(((patient_ph > 7.15) & (patient_ph <= 7.25)).sum()),
            "count_gt_725": int((patient_ph > 7.25).sum()),
            "borderline_710_720": int(((patient_ph >= 7.10) & (patient_ph <= 7.20)).sum()),
            "bdecf_ge_8": int((patient_bdecf >= 8.0).sum())
        },
        "models": {},
        "secondary_thresholds": {},
        "composite_endpoints": {},
        "paired_tests": [],
        "patient_scores": {}
    }

    # Helper function for CV fold split
    def get_fold_masks(f_idx):
        test_pids = np.array([p for p in prot.patients if prot.assignment[p][0] == f_idx])
        train_pids = np.array([p for p in prot.patients if prot.assignment[p][0] != f_idx])
        tr_win = np.isin(pid_arr, train_pids)
        te_win = np.isin(pid_arr, test_pids)
        tr_pat = np.isin(prot.patients, train_pids)
        te_pat = np.isin(prot.patients, test_pids)
        return tr_win, te_win, tr_pat, te_pat, test_pids

    # -----------------------------------------------------------------------
    # Exp 6.0: Phase-4 Baseline Master Model Reproduction (AUROC = 0.7361)
    # -----------------------------------------------------------------------
    print("\n--- Exp 6.0: Phase-4 Master Model Reproduction ---")
    p4_sig_oof = np.zeros(len(clean_pids), dtype=np.float32)
    p4_cli_oof = np.zeros(len(clean_pids), dtype=np.float32)
    p4_final_oof = np.zeros(len(clean_pids), dtype=np.float32)

    # 1. Train 1D Signal ResNet (Binary)
    win_sig_preds = np.zeros(len(y_bin), dtype=np.float32)
    for f_idx in range(5):
        tr_win, te_win, tr_pat, te_pat, te_pids = get_fold_masks(f_idx)
        X_tr = X_tensor[tr_win].clone()
        y_tr = torch.tensor(y_bin[tr_win], dtype=torch.float32)
        X_te = X_tensor[te_win].clone()

        mu = X_tr[:, 0, :].mean()
        std = X_tr[:, 0, :].std() + 1e-6
        X_tr[:, 0, :] = (X_tr[:, 0, :] - mu) / std
        X_te[:, 0, :] = (X_te[:, 0, :] - mu) / std

        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = CNN1DEncoder(in_channels=2, seq_len=4800, latent_dim=128).to(DEVICE)
        head = nn.Linear(128, 1).to(DEVICE)
        pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1.0)], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-4)

        model.train()
        head.train()
        for ep in range(EPOCHS):
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = head(model(bx)).squeeze(-1)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        model.eval()
        head.eval()
        with torch.no_grad():
            te_loader = DataLoader(TensorDataset(X_te), batch_size=BATCH_SIZE, shuffle=False)
            t_preds = []
            for (bx,) in te_loader:
                bx = bx.to(DEVICE)
                probs = torch.sigmoid(head(model(bx)).squeeze(-1)).cpu().numpy()
                t_preds.extend(probs)
            win_sig_preds[te_win] = np.array(t_preds)

        for p in te_pids:
            p_idx = np.where(clean_pids == p)[0][0]
            p4_sig_oof[p_idx] = np.percentile(win_sig_preds[pid_arr == p], 90)

        # 2. Train Clinical LR
        mu_c = F_patient[tr_pat].mean(axis=0)
        std_c = F_patient[tr_pat].std(axis=0) + 1e-6
        clf = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=SEED)
        clf.fit((F_patient[tr_pat] - mu_c) / std_c, prot.plab[tr_pat])
        p4_cli_oof[te_pat] = clf.predict_proba((F_patient[te_pat] - mu_c) / std_c)[:, 1]

        # 3. Logit Prior Modulation
        z_s_tr = safe_logit(p4_sig_oof[tr_pat])
        z_c_tr = safe_logit(p4_cli_oof[tr_pat])
        best_lam = 1.0
        best_auc = 0.0
        for lam in np.linspace(0.1, 3.0, 59):
            cur_auc = roc_auc_score(prot.plab[tr_pat], z_s_tr + lam * z_c_tr)
            if cur_auc > best_auc:
                best_auc = cur_auc
                best_lam = lam

        z_s_te = safe_logit(p4_sig_oof[te_pat])
        z_c_te = safe_logit(p4_cli_oof[te_pat])
        p4_final_oof[te_pat] = 1.0 / (1.0 + np.exp(-(z_s_te + best_lam * z_c_te)))

    auc_p4 = roc_auc_score(prot.plab, p4_final_oof)
    ci_p4 = bootstrap_ci(prot.plab, p4_final_oof)
    print(f"Phase-4 Master Model AUROC: {auc_p4:.4f} [{ci_p4[0]:.4f}, {ci_p4[1]:.4f}], AUPRC: {average_precision_score(prot.plab, p4_final_oof):.4f}")
    results["models"]["phase4_master"] = {
        "auroc": float(auc_p4),
        "auprc": float(average_precision_score(prot.plab, p4_final_oof)),
        "ci": [float(ci_p4[0]), float(ci_p4[1])]
    }

    # -----------------------------------------------------------------------
    # Exp 6.1: Continuous Signal pH Regression (Huber Loss, S = -pH_hat)
    # -----------------------------------------------------------------------
    print("\n--- Exp 6.1: Continuous Signal pH Regression ---")
    ph_reg_sig_oof = np.zeros(len(clean_pids), dtype=np.float32)
    win_ph_preds = np.zeros(len(y_bin), dtype=np.float32)

    for f_idx in range(5):
        tr_win, te_win, tr_pat, te_pat, te_pids = get_fold_masks(f_idx)
        X_tr = X_tensor[tr_win].clone()
        y_tr_ph = torch.tensor(y_ph_windows[tr_win], dtype=torch.float32)
        X_te = X_tensor[te_win].clone()

        mu = X_tr[:, 0, :].mean()
        std = X_tr[:, 0, :].std() + 1e-6
        X_tr[:, 0, :] = (X_tr[:, 0, :] - mu) / std
        X_te[:, 0, :] = (X_te[:, 0, :] - mu) / std

        train_ds = TensorDataset(X_tr, y_tr_ph)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = CNN1DEncoder(in_channels=2, seq_len=4800, latent_dim=128).to(DEVICE)
        head = nn.Linear(128, 1).to(DEVICE)
        criterion = nn.SmoothL1Loss(beta=0.05) # Huber loss
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-4)

        model.train()
        head.train()
        for ep in range(EPOCHS):
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                pred = head(model(bx)).squeeze(-1)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

        model.eval()
        head.eval()
        with torch.no_grad():
            te_loader = DataLoader(TensorDataset(X_te), batch_size=BATCH_SIZE, shuffle=False)
            t_preds = []
            for (bx,) in te_loader:
                bx = bx.to(DEVICE)
                preds = head(model(bx)).squeeze(-1).cpu().numpy()
                t_preds.extend(preds)
            win_ph_preds[te_win] = np.array(t_preds)

        for p in te_pids:
            p_idx = np.where(clean_pids == p)[0][0]
            # Minimum predicted pH (or P10 of pH_hat) corresponds to highest acidemia risk S = -pH_hat
            p_win_ph = win_ph_preds[pid_arr == p]
            ph_reg_sig_oof[p_idx] = float(np.percentile(p_win_ph, 10))

    risk_sig_reg = -ph_reg_sig_oof
    auc_sig_reg = roc_auc_score(prot.plab, risk_sig_reg)
    ci_sig_reg = bootstrap_ci(prot.plab, risk_sig_reg)
    mae_sig = mean_absolute_error(patient_ph, ph_reg_sig_oof)
    rmse_sig = np.sqrt(mean_squared_error(patient_ph, ph_reg_sig_oof))
    r_sig, _ = pearsonr(patient_ph, ph_reg_sig_oof)
    rho_sig, _ = spearmanr(patient_ph, ph_reg_sig_oof)

    print(f"Continuous Signal pH AUROC (<=7.15): {auc_sig_reg:.4f} [{ci_sig_reg[0]:.4f}, {ci_sig_reg[1]:.4f}], MAE: {mae_sig:.4f}, RMSE: {rmse_sig:.4f}, Pearson r: {r_sig:.4f}")
    results["models"]["continuous_signal"] = {
        "auroc": float(auc_sig_reg),
        "auprc": float(average_precision_score(prot.plab, risk_sig_reg)),
        "ci": [float(ci_sig_reg[0]), float(ci_sig_reg[1])],
        "mae": float(mae_sig),
        "rmse": float(rmse_sig),
        "pearson_r": float(r_sig),
        "spearman_rho": float(rho_sig)
    }

    # -----------------------------------------------------------------------
    # Exp 6.2: Continuous Clinical pH Regression (Huber/Ridge on 19 Features)
    # -----------------------------------------------------------------------
    print("\n--- Exp 6.2: Continuous Clinical pH Regression ---")
    ph_cli_reg_oof = np.zeros(len(clean_pids), dtype=np.float32)

    for f_idx in range(5):
        _, _, tr_pat, te_pat, _ = get_fold_masks(f_idx)
        mu_c = F_patient[tr_pat].mean(axis=0)
        std_c = F_patient[tr_pat].std(axis=0) + 1e-6
        reg = HuberRegressor(alpha=1.0, epsilon=1.35)
        reg.fit((F_patient[tr_pat] - mu_c) / std_c, patient_ph[tr_pat])
        ph_cli_reg_oof[te_pat] = reg.predict((F_patient[te_pat] - mu_c) / std_c)

    risk_cli_reg = -ph_cli_reg_oof
    auc_cli_reg = roc_auc_score(prot.plab, risk_cli_reg)
    ci_cli_reg = bootstrap_ci(prot.plab, risk_cli_reg)
    mae_cli = mean_absolute_error(patient_ph, ph_cli_reg_oof)
    rmse_cli = np.sqrt(mean_squared_error(patient_ph, ph_cli_reg_oof))
    r_cli, _ = pearsonr(patient_ph, ph_cli_reg_oof)

    print(f"Continuous Clinical pH AUROC (<=7.15): {auc_cli_reg:.4f} [{ci_cli_reg[0]:.4f}, {ci_cli_reg[1]:.4f}], MAE: {mae_cli:.4f}, Pearson r: {r_cli:.4f}")
    results["models"]["continuous_clinical"] = {
        "auroc": float(auc_cli_reg),
        "auprc": float(average_precision_score(prot.plab, risk_cli_reg)),
        "ci": [float(ci_cli_reg[0]), float(ci_cli_reg[1])],
        "mae": float(mae_cli),
        "rmse": float(rmse_cli),
        "pearson_r": float(r_cli)
    }

    # -----------------------------------------------------------------------
    # Exp 6.3: Continuous Knowledge Fusion (Signal + Clinical pH Fusion)
    # -----------------------------------------------------------------------
    print("\n--- Exp 6.3: Continuous Knowledge Fusion ---")
    ph_fusion_oof = np.zeros(len(clean_pids), dtype=np.float32)

    for f_idx in range(5):
        _, _, tr_pat, te_pat, _ = get_fold_masks(f_idx)
        s_tr = risk_sig_reg[tr_pat]
        c_tr = risk_cli_reg[tr_pat]
        s_tr_norm = (s_tr - s_tr.mean()) / (s_tr.std() + 1e-6)
        c_tr_norm = (c_tr - c_tr.mean()) / (c_tr.std() + 1e-6)

        best_w = 0.5
        best_auc = 0.0
        for w in np.linspace(0.0, 1.0, 51):
            comb = w * s_tr_norm + (1.0 - w) * c_tr_norm
            cur_auc = roc_auc_score(prot.plab[tr_pat], comb)
            if cur_auc > best_auc:
                best_auc = cur_auc
                best_w = w

        s_te_norm = (risk_sig_reg[te_pat] - s_tr.mean()) / (s_tr.std() + 1e-6)
        c_te_norm = (risk_cli_reg[te_pat] - c_tr.mean()) / (c_tr.std() + 1e-6)
        ph_fusion_oof[te_pat] = best_w * s_te_norm + (1.0 - best_w) * c_te_norm

    auc_cont_fus = roc_auc_score(prot.plab, ph_fusion_oof)
    ci_cont_fus = bootstrap_ci(prot.plab, ph_fusion_oof)
    print(f"Continuous Knowledge Fusion AUROC (<=7.15): {auc_cont_fus:.4f} [{ci_cont_fus[0]:.4f}, {ci_cont_fus[1]:.4f}]")
    results["models"]["continuous_fusion"] = {
        "auroc": float(auc_cont_fus),
        "auprc": float(average_precision_score(prot.plab, ph_fusion_oof)),
        "ci": [float(ci_cont_fus[0]), float(ci_cont_fus[1])]
    }

    # -----------------------------------------------------------------------
    # Exp 6.4: Ordinal Signal Supervision (Multi-Threshold Cumulative BCE)
    # Thresholds: c1=7.05, c2=7.15, c3=7.25
    # -----------------------------------------------------------------------
    print("\n--- Exp 6.4: Ordinal Acid-Base Supervision ---")
    y_ord_windows = np.zeros((len(y_bin), 3), dtype=np.float32)
    y_ord_windows[:, 0] = (y_ph_windows <= 7.05).astype(np.float32)
    y_ord_windows[:, 1] = (y_ph_windows <= 7.15).astype(np.float32)
    y_ord_windows[:, 2] = (y_ph_windows <= 7.25).astype(np.float32)

    ord_sig_oof_715 = np.zeros(len(clean_pids), dtype=np.float32)
    ord_sig_oof_all = np.zeros((len(clean_pids), 3), dtype=np.float32)

    for f_idx in range(5):
        tr_win, te_win, tr_pat, te_pat, te_pids = get_fold_masks(f_idx)
        X_tr = X_tensor[tr_win].clone()
        y_tr_ord = torch.tensor(y_ord_windows[tr_win], dtype=torch.float32)
        X_te = X_tensor[te_win].clone()

        mu = X_tr[:, 0, :].mean()
        std = X_tr[:, 0, :].std() + 1e-6
        X_tr[:, 0, :] = (X_tr[:, 0, :] - mu) / std
        X_te[:, 0, :] = (X_te[:, 0, :] - mu) / std

        train_ds = TensorDataset(X_tr, y_tr_ord)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = CNN1DEncoder(in_channels=2, seq_len=4800, latent_dim=128).to(DEVICE)
        head = nn.Linear(128, 3).to(DEVICE) # 3 cumulative logit heads
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-4)

        model.train()
        head.train()
        for ep in range(EPOCHS):
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                logits = head(model(bx))
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        model.eval()
        head.eval()
        with torch.no_grad():
            te_loader = DataLoader(TensorDataset(X_te), batch_size=BATCH_SIZE, shuffle=False)
            t_preds = []
            for (bx,) in te_loader:
                bx = bx.to(DEVICE)
                probs = torch.sigmoid(head(model(bx))).cpu().numpy()
                t_preds.extend(probs)
            t_preds = np.array(t_preds)

        for p in te_pids:
            p_idx = np.where(clean_pids == p)[0][0]
            p_win_ord = t_preds[pid_arr[te_win] == p]
            ord_sig_oof_all[p_idx] = np.percentile(p_win_ord, 90, axis=0)
            ord_sig_oof_715[p_idx] = ord_sig_oof_all[p_idx, 1]

    auc_ord_sig = roc_auc_score(prot.plab, ord_sig_oof_715)
    ci_ord_sig = bootstrap_ci(prot.plab, ord_sig_oof_715)
    print(f"Ordinal Signal AUROC (<=7.15): {auc_ord_sig:.4f} [{ci_ord_sig[0]:.4f}, {ci_ord_sig[1]:.4f}]")
    results["models"]["ordinal_signal"] = {
        "auroc": float(auc_ord_sig),
        "auprc": float(average_precision_score(prot.plab, ord_sig_oof_715)),
        "ci": [float(ci_ord_sig[0]), float(ci_ord_sig[1])]
    }

    # -----------------------------------------------------------------------
    # Exp 6.5: Ordinal Knowledge Fusion (Signal Ordinal + Clinical Prior)
    # -----------------------------------------------------------------------
    print("\n--- Exp 6.5: Ordinal Knowledge Fusion ---")
    ord_fusion_oof = np.zeros(len(clean_pids), dtype=np.float32)

    for f_idx in range(5):
        _, _, tr_pat, te_pat, _ = get_fold_masks(f_idx)
        z_s_tr = safe_logit(ord_sig_oof_715[tr_pat])
        z_c_tr = safe_logit(p4_cli_oof[tr_pat])

        best_lam = 1.0
        best_auc = 0.0
        for lam in np.linspace(0.1, 3.0, 59):
            cur_auc = roc_auc_score(prot.plab[tr_pat], z_s_tr + lam * z_c_tr)
            if cur_auc > best_auc:
                best_auc = cur_auc
                best_lam = lam

        z_s_te = safe_logit(ord_sig_oof_715[te_pat])
        z_c_te = safe_logit(p4_cli_oof[te_pat])
        ord_fusion_oof[te_pat] = 1.0 / (1.0 + np.exp(-(z_s_te + best_lam * z_c_te)))

    auc_ord_fus = roc_auc_score(prot.plab, ord_fusion_oof)
    ci_ord_fus = bootstrap_ci(prot.plab, ord_fusion_oof)
    print(f"Ordinal Knowledge Fusion AUROC (<=7.15): {auc_ord_fus:.4f} [{ci_ord_fus[0]:.4f}, {ci_ord_fus[1]:.4f}]")
    results["models"]["ordinal_fusion"] = {
        "auroc": float(auc_ord_fus),
        "auprc": float(average_precision_score(prot.plab, ord_fusion_oof)),
        "ci": [float(ci_ord_fus[0]), float(ci_ord_fus[1])]
    }

    # -----------------------------------------------------------------------
    # Exp 6.6: Soft-Target Binary Smoothing (tau = 0.02)
    # q(pH) = sigmoid((7.15 - pH) / tau)
    # -----------------------------------------------------------------------
    print("\n--- Exp 6.6: Soft-Target Binary Smoothing ---")
    tau = 0.02
    q_ph_windows = 1.0 / (1.0 + np.exp((y_ph_windows - 7.15) / tau))

    soft_sig_oof = np.zeros(len(clean_pids), dtype=np.float32)
    soft_fusion_oof = np.zeros(len(clean_pids), dtype=np.float32)

    for f_idx in range(5):
        tr_win, te_win, tr_pat, te_pat, te_pids = get_fold_masks(f_idx)
        X_tr = X_tensor[tr_win].clone()
        y_tr_soft = torch.tensor(q_ph_windows[tr_win], dtype=torch.float32)
        X_te = X_tensor[te_win].clone()

        mu = X_tr[:, 0, :].mean()
        std = X_tr[:, 0, :].std() + 1e-6
        X_tr[:, 0, :] = (X_tr[:, 0, :] - mu) / std
        X_te[:, 0, :] = (X_te[:, 0, :] - mu) / std

        train_ds = TensorDataset(X_tr, y_tr_soft)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = CNN1DEncoder(in_channels=2, seq_len=4800, latent_dim=128).to(DEVICE)
        head = nn.Linear(128, 1).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-4)

        model.train()
        head.train()
        for ep in range(EPOCHS):
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                logits = head(model(bx)).squeeze(-1)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        model.eval()
        head.eval()
        with torch.no_grad():
            te_loader = DataLoader(TensorDataset(X_te), batch_size=BATCH_SIZE, shuffle=False)
            t_preds = []
            for (bx,) in te_loader:
                bx = bx.to(DEVICE)
                probs = torch.sigmoid(head(model(bx)).squeeze(-1)).cpu().numpy()
                t_preds.extend(probs)
            t_preds = np.array(t_preds)

        for p in te_pids:
            p_idx = np.where(clean_pids == p)[0][0]
            soft_sig_oof[p_idx] = np.percentile(t_preds[pid_arr[te_win] == p], 90)

        # Soft Fusion with Clinical Prior
        z_s_tr = safe_logit(soft_sig_oof[tr_pat])
        z_c_tr = safe_logit(p4_cli_oof[tr_pat])
        best_lam = 1.0
        best_auc = 0.0
        for lam in np.linspace(0.1, 3.0, 59):
            cur_auc = roc_auc_score(prot.plab[tr_pat], z_s_tr + lam * z_c_tr)
            if cur_auc > best_auc:
                best_auc = cur_auc
                best_lam = lam

        z_s_te = safe_logit(soft_sig_oof[te_pat])
        z_c_te = safe_logit(p4_cli_oof[te_pat])
        soft_fusion_oof[te_pat] = 1.0 / (1.0 + np.exp(-(z_s_te + best_lam * z_c_te)))

    auc_soft_sig = roc_auc_score(prot.plab, soft_sig_oof)
    auc_soft_fus = roc_auc_score(prot.plab, soft_fusion_oof)
    ci_soft_fus = bootstrap_ci(prot.plab, soft_fusion_oof)
    print(f"Soft Target Signal AUROC: {auc_soft_sig:.4f}, Soft Target Fusion AUROC: {auc_soft_fus:.4f} [{ci_soft_fus[0]:.4f}, {ci_soft_fus[1]:.4f}]")
    results["models"]["soft_target_fusion"] = {
        "auroc": float(auc_soft_fus),
        "auprc": float(average_precision_score(prot.plab, soft_fusion_oof)),
        "ci": [float(ci_soft_fus[0]), float(ci_soft_fus[1])]
    }

    # -----------------------------------------------------------------------
    # Exp 6.7: Multi-Task Continuous + Ordinal Supervision
    # -----------------------------------------------------------------------
    print("\n--- Exp 6.7: Multi-Task Continuous + Ordinal Supervision ---")
    multi_task_oof = np.zeros(len(clean_pids), dtype=np.float32)

    for f_idx in range(5):
        tr_win, te_win, tr_pat, te_pat, te_pids = get_fold_masks(f_idx)
        X_tr = X_tensor[tr_win].clone()
        y_tr_ph = torch.tensor(y_ph_windows[tr_win], dtype=torch.float32)
        y_tr_ord = torch.tensor(y_ord_windows[tr_win], dtype=torch.float32)
        X_te = X_tensor[te_win].clone()

        mu = X_tr[:, 0, :].mean()
        std = X_tr[:, 0, :].std() + 1e-6
        X_tr[:, 0, :] = (X_tr[:, 0, :] - mu) / std
        X_te[:, 0, :] = (X_te[:, 0, :] - mu) / std

        train_ds = TensorDataset(X_tr, y_tr_ph, y_tr_ord)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = CNN1DEncoder(in_channels=2, seq_len=4800, latent_dim=128).to(DEVICE)
        ph_head = nn.Linear(128, 1).to(DEVICE)
        ord_head = nn.Linear(128, 3).to(DEVICE)
        crit_ph = nn.SmoothL1Loss(beta=0.05)
        crit_ord = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(ph_head.parameters()) + list(ord_head.parameters()), lr=LR, weight_decay=1e-4)

        model.train()
        ph_head.train()
        ord_head.train()
        for ep in range(EPOCHS):
            for bx, by_ph, by_ord in train_loader:
                bx, by_ph, by_ord = bx.to(DEVICE), by_ph.to(DEVICE), by_ord.to(DEVICE)
                optimizer.zero_grad()
                emb = model(bx)
                pred_ph = ph_head(emb).squeeze(-1)
                pred_ord = ord_head(emb)
                loss = crit_ph(pred_ph, by_ph) + 0.5 * crit_ord(pred_ord, by_ord)
                loss.backward()
                optimizer.step()

        model.eval()
        ord_head.eval()
        with torch.no_grad():
            te_loader = DataLoader(TensorDataset(X_te), batch_size=BATCH_SIZE, shuffle=False)
            t_preds = []
            for (bx,) in te_loader:
                bx = bx.to(DEVICE)
                probs = torch.sigmoid(ord_head(model(bx))[:, 1]).cpu().numpy()
                t_preds.extend(probs)
            t_preds = np.array(t_preds)

        for p in te_pids:
            p_idx = np.where(clean_pids == p)[0][0]
            multi_task_oof[p_idx] = np.percentile(t_preds[pid_arr[te_win] == p], 90)

    auc_mt = roc_auc_score(prot.plab, multi_task_oof)
    ci_mt = bootstrap_ci(prot.plab, multi_task_oof)
    print(f"Multi-Task Continuous+Ordinal Signal AUROC: {auc_mt:.4f} [{ci_mt[0]:.4f}, {ci_mt[1]:.4f}]")
    results["models"]["multitask_continuous_ordinal"] = {
        "auroc": float(auc_mt),
        "auprc": float(average_precision_score(prot.plab, multi_task_oof)),
        "ci": [float(ci_mt[0]), float(ci_mt[1])]
    }

    # -----------------------------------------------------------------------
    # Secondary pH Thresholds Evaluation (<=7.05, <=7.15, <=7.20, <=7.25)
    # -----------------------------------------------------------------------
    th_evals = {}
    for th_val in [7.05, 7.15, 7.20, 7.25]:
        y_th = (patient_ph <= th_val).astype(np.int64)
        th_evals[f"le_{th_val:.2f}"] = {
            "prevalence": int(y_th.sum()),
            "phase4_auroc": float(roc_auc_score(y_th, p4_final_oof)),
            "continuous_fusion_auroc": float(roc_auc_score(y_th, ph_fusion_oof)),
            "ordinal_fusion_auroc": float(roc_auc_score(y_th, ord_fusion_oof)),
            "soft_target_fusion_auroc": float(roc_auc_score(y_th, soft_fusion_oof))
        }
    results["secondary_thresholds"] = th_evals

    # -----------------------------------------------------------------------
    # Exp 6.8: Exploratory Composite & Secondary Endpoints
    # -----------------------------------------------------------------------
    y_bdecf_ge8 = (patient_bdecf >= 8.0).astype(np.int64)
    y_comp_or = ((patient_ph <= 7.15) | (patient_bdecf >= 8.0)).astype(np.int64)
    y_comp_and = ((patient_ph <= 7.15) & (patient_bdecf >= 8.0)).astype(np.int64)

    results["composite_endpoints"] = {
        "ph_le_715_primary": {
            "name": "pH <= 7.15 (Primary Benchmark)",
            "prevalence": int(prot.plab.sum()),
            "phase4_auroc": float(auc_p4),
            "ci": list(ci_p4),
            "category": "Primary"
        },
        "bdecf_ge_8_secondary": {
            "name": "BDecf >= 8.0 mmol/L (Secondary)",
            "prevalence": int(y_bdecf_ge8.sum()),
            "phase4_auroc": float(roc_auc_score(y_bdecf_ge8, p4_final_oof)),
            "ci": list(bootstrap_ci(y_bdecf_ge8, p4_final_oof)),
            "category": "Secondary"
        },
        "composite_or_exploratory": {
            "name": "pH <= 7.15 OR BDecf >= 8.0 (Exploratory)",
            "prevalence": int(y_comp_or.sum()),
            "phase4_auroc": float(roc_auc_score(y_comp_or, p4_final_oof)),
            "ci": list(bootstrap_ci(y_comp_or, p4_final_oof)),
            "category": "Exploratory"
        },
        "composite_and_exploratory": {
            "name": "pH <= 7.15 AND BDecf >= 8.0 (Exploratory Severe)",
            "prevalence": int(y_comp_and.sum()),
            "phase4_auroc": float(roc_auc_score(y_comp_and, p4_final_oof)),
            "ci": list(bootstrap_ci(y_comp_and, p4_final_oof)),
            "category": "Exploratory"
        }
    }

    # -----------------------------------------------------------------------
    # Paired Statistical Significance Comparisons (2,000 Bootstrap Replicates)
    # -----------------------------------------------------------------------
    print("\n--- Running Paired Significance Tests vs Phase-4 Master (0.7361) ---")
    comparisons = [
        ("Continuous Fusion vs Phase 4", p4_final_oof, ph_fusion_oof),
        ("Ordinal Fusion vs Phase 4", p4_final_oof, ord_fusion_oof),
        ("Soft-Target Fusion vs Phase 4", p4_final_oof, soft_fusion_oof),
        ("Multi-Task Signal vs Phase 4", p4_final_oof, multi_task_oof),
        ("Continuous Signal vs Binary Signal", p4_sig_oof, risk_sig_reg),
        ("Ordinal Signal vs Binary Signal", p4_sig_oof, ord_sig_oof_715)
    ]

    for name, s_base, s_target in comparisons:
        delta, ci_d, p_val = paired_bootstrap_test(prot.plab, s_base, s_target, n_boot=2000, seed=SEED)
        print(f"  {name:40s}: Delta AUROC = {delta:+.4f} (95% CI: [{ci_d[0]:+.4f}, {ci_d[1]:+.4f}]), p = {p_val:.4f}")
        results["paired_tests"].append({
            "comparison": name,
            "delta_auroc": delta,
            "ci": list(ci_d),
            "p_value": p_val
        })

    # -----------------------------------------------------------------------
    # Error-Rescue Analysis on Positives (N=110)
    # -----------------------------------------------------------------------
    pos_idx = np.where(prot.plab == 1)[0]
    th_p4 = np.percentile(p4_final_oof, 80)
    th_cont = np.percentile(ph_fusion_oof, 80)
    th_ord = np.percentile(ord_fusion_oof, 80)

    p4_corr = (p4_final_oof[pos_idx] >= th_p4)
    cont_corr = (ph_fusion_oof[pos_idx] >= th_cont)
    ord_corr = (ord_fusion_oof[pos_idx] >= th_ord)

    results["error_rescue"] = {
        "total_positives": len(pos_idx),
        "phase4_correct": int(p4_corr.sum()),
        "continuous_correct": int(cont_corr.sum()),
        "ordinal_correct": int(ord_corr.sum()),
        "p4_wrong_continuous_correct": int((~p4_corr & cont_corr).sum()),
        "p4_wrong_ordinal_correct": int((~p4_corr & ord_corr).sum()),
        "p4_correct_continuous_wrong": int((p4_corr & ~cont_corr).sum()),
        "p4_correct_ordinal_wrong": int((p4_corr & ~ord_corr).sum()),
        "all_three_correct": int((p4_corr & cont_corr & ord_corr).sum()),
        "union_any_correct": int((p4_corr | cont_corr | ord_corr).sum())
    }

    # Store patient scores for visualization
    results["patient_scores"] = {
        "labels": prot.plab.tolist(),
        "true_ph": patient_ph.tolist(),
        "true_bdecf": patient_bdecf.tolist(),
        "phase4_score": p4_final_oof.tolist(),
        "continuous_signal_ph": ph_reg_sig_oof.tolist(),
        "continuous_clinical_ph": ph_cli_reg_oof.tolist(),
        "continuous_fusion_score": ph_fusion_oof.tolist(),
        "ordinal_signal_score": ord_sig_oof_715.tolist(),
        "ordinal_fusion_score": ord_fusion_oof.tolist(),
        "soft_target_fusion_score": soft_fusion_oof.tolist()
    }

    out_file = os.path.join(OUT_DIR, "phase6_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSuccessfully saved Phase 6 evaluation results to {out_file} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    run_phase6()
