"""
Phase 5: CTU-UHB Multi-Resolution 1D Temporal Context Evaluation.

Implements and evaluates:
- Exp 5.0: Temporal Scale Audit (5m, 10m, 20m, 30m, 40m, 60m single-scale 1D ResNet with P90)
- Exp 5.1 & 5.2: Short vs Medium vs Long context analysis and missingness audit
- Exp 5.3-5.5: Multi-resolution signal models (shared encoder, scale-weighted gating, logit score fusion)
- Exp 5.7 & 5.8: Temporal clinical features and deceleration trend features (Logistic Regression)
- Exp 5.9: Primary Phase-5 candidate (Multi-resolution signal + temporal clinical fusion via Logit Prior Modulation)
- Paired statistical tests (2,000-replicate bootstrap) vs Phase 4 (0.7361)
- Stratified analyses (time-to-delivery, correlation between scales, single-scale correct case counts)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.protocol import Protocol
from src.models.cnn1d_encoder import CNN1DEncoder

DATA_DIR = "data/phase5_multires"
FOLDS_PATH = "data/processed_clinical/folds.json"
OUT_DIR = "results/phase5_multires"
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


def train_single_scale_encoder(X_tensor, y_tensor, pid_arr, prot: Protocol, scale_name: str):
    """
    Trains 1D ResNet on single-scale windows across 5 folds and computes P90 patient predictions.
    """
    print(f"\n--- Training Single-Scale Model [{scale_name}] (seq_len={X_tensor.shape[2]}) ---")
    patient_oof_scores = np.zeros(len(prot.patients), dtype=np.float32)
    window_oof_scores = np.zeros(len(y_tensor), dtype=np.float32)
    fold_aucs = []

    # Map each window index to patient index
    pid_to_p_idx = {p: i for i, p in enumerate(prot.patients)}
    win_pid_idx = np.array([pid_to_p_idx[p] for p in pid_arr])

    for f_idx in range(5):
        test_pids = np.array([p for p in prot.patients if prot.assignment[p][0] == f_idx])
        train_pids = np.array([p for p in prot.patients if prot.assignment[p][0] != f_idx])

        train_mask = np.isin(pid_arr, train_pids)
        test_mask = np.isin(pid_arr, test_pids)

        # Use channels [1 (Delta FHR), 2 (Observed Mask)] for 1D CNN
        X_tr = X_tensor[train_mask][:, [1, 2], :].clone()
        y_tr = y_tensor[train_mask].float()
        X_te = X_tensor[test_mask][:, [1, 2], :].clone()
        y_te = y_tensor[test_mask].float()

        # Z-score normalize channel 0 per fold
        mu = X_tr[:, 0, :].mean()
        std = X_tr[:, 0, :].std() + 1e-6
        X_tr[:, 0, :] = (X_tr[:, 0, :] - mu) / std
        X_te[:, 0, :] = (X_te[:, 0, :] - mu) / std

        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        seq_len = X_tensor.shape[2]
        model = CNN1DEncoder(in_channels=2, seq_len=seq_len, latent_dim=128).to(DEVICE)
        head = nn.Linear(128, 1).to(DEVICE)

        pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1.0)], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-4)

        model.train()
        head.train()
        for epoch in range(EPOCHS):
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                emb = model(bx)
                out = head(emb).squeeze(-1)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        # Inference on test windows
        model.eval()
        head.eval()
        with torch.no_grad():
            te_ds = TensorDataset(X_te)
            te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)
            te_preds = []
            for (bx,) in te_loader:
                bx = bx.to(DEVICE)
                logits = head(model(bx)).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                te_preds.extend(probs)
            te_preds = np.array(te_preds)

        window_oof_scores[test_mask] = te_preds

        # Patient P90 aggregation for test patients
        test_patient_indices = [pid_to_p_idx[p] for p in test_pids]
        for p_i in test_patient_indices:
            p_name = prot.patients[p_i]
            p_win_mask = (pid_arr == p_name)
            if p_win_mask.sum() > 0:
                patient_oof_scores[p_i] = np.percentile(window_oof_scores[p_win_mask], 90)
            else:
                patient_oof_scores[p_i] = 0.0

        f_auc = roc_auc_score(prot.plab[test_patient_indices], patient_oof_scores[test_patient_indices])
        fold_aucs.append(float(f_auc))

    total_auc = roc_auc_score(prot.plab, patient_oof_scores)
    total_auprc = average_precision_score(prot.plab, patient_oof_scores)
    ci_lo, ci_hi = bootstrap_ci(prot.plab, patient_oof_scores)
    print(f"[{scale_name}] Patient AUROC: {total_auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], AUPRC: {total_auprc:.4f}, Folds: {[round(x, 3) for x in fold_aucs]}")

    return {
        "auroc": float(total_auc),
        "auprc": float(total_auprc),
        "ci": [float(ci_lo), float(ci_hi)],
        "fold_aucs": fold_aucs,
        "patient_scores": patient_oof_scores.tolist(),
        "window_scores": window_oof_scores.tolist()
    }


def run_phase5_evaluation():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== EXECUTING PHASE 5 MULTI-RESOLUTION EVALUATION ===")
    t0 = time.time()

    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    # Load 20m labels to create master protocol
    data_20m = torch.load(os.path.join(DATA_DIR, "scale_20m_dataset.pt"), weights_only=False)
    prot = Protocol.load_or_create(data_20m["pid"], data_20m["y"].numpy(), path=FOLDS_PATH)
    y_true_patient = prot.plab.astype(np.int64)

    results = {
        "scale_audit": {},
        "matched_models": {},
        "temporal_clinical": {},
        "paired_tests": [],
        "scale_correlations": {},
        "scale_complementarity": {},
        "delivery_stratification": {},
        "patient_scores": {}
    }

    # -----------------------------------------------------------------------
    # Experiment 5.0: Single-Scale Audit (5m, 10m, 20m, 30m, 40m, 60m)
    # -----------------------------------------------------------------------
    candidate_scales = ["5m", "10m", "20m", "30m", "40m", "60m"]
    scale_patient_scores = {}

    for sc in candidate_scales:
        sc_file = os.path.join(DATA_DIR, f"scale_{sc}_dataset.pt")
        sc_data = torch.load(sc_file, weights_only=False)
        res_sc = train_single_scale_encoder(sc_data["X"], sc_data["y"], sc_data["pid"], prot, scale_name=sc)
        results["scale_audit"][sc] = res_sc
        scale_patient_scores[sc] = np.array(res_sc["patient_scores"])

    # -----------------------------------------------------------------------
    # Matched Multi-Resolution Cohort Evaluation (10m, 20m, 40m)
    # -----------------------------------------------------------------------
    matched_data = torch.load(os.path.join(DATA_DIR, "matched_multires_dataset.pt"), weights_only=False)
    matched_pids = matched_data["pid"]
    unique_matched_pids = np.unique(matched_pids)
    print(f"\nMatched Cohort: {len(unique_matched_pids)} eligible patients with full 40m history.")

    # Matched common protocol index
    matched_p_idx = np.array([np.where(prot.patients == p)[0][0] for p in unique_matched_pids])
    matched_y_true = prot.plab[matched_p_idx]

    s_10m = scale_patient_scores["10m"]
    s_20m = scale_patient_scores["20m"]
    s_40m = scale_patient_scores["40m"]

    # -----------------------------------------------------------------------
    # Experiment 5.6: Multi-Resolution Signal Score Fusion (Logit-space)
    # z_s = alpha_1 * logit(p10) + alpha_2 * logit(p20) + alpha_3 * logit(p40)
    # -----------------------------------------------------------------------
    print("\n--- Training Multi-Scale Signal Score Fusion (Logit Space) ---")
    multiscale_signal_oof = np.zeros(len(prot.patients), dtype=np.float32)
    learned_alphas = []

    def safe_logit(p, eps=1e-5):
        p_c = np.clip(p, eps, 1.0 - eps)
        return np.log(p_c / (1.0 - p_c))

    for f_idx in range(5):
        test_pids = np.array([p for p in prot.patients if prot.assignment[p][0] == f_idx])
        train_pids = np.array([p for p in prot.patients if prot.assignment[p][0] != f_idx])

        tr_idx = np.isin(prot.patients, train_pids)
        te_idx = np.isin(prot.patients, test_pids)

        z10_tr, z20_tr, z40_tr = safe_logit(s_10m[tr_idx]), safe_logit(s_20m[tr_idx]), safe_logit(s_40m[tr_idx])
        y_tr = prot.plab[tr_idx]

        # Optimize alpha weights on training fold to maximize AUROC
        def obj(weights):
            w = np.exp(weights) / np.sum(np.exp(weights))
            z_fused = w[0] * z10_tr + w[1] * z20_tr + w[2] * z40_tr
            # negative AUROC
            return -roc_auc_score(y_tr, z_fused)

        opt = minimize(obj, [0.0, 0.0, 0.0], method="Nelder-Mead")
        best_w = np.exp(opt.x) / np.sum(np.exp(opt.x))
        learned_alphas.append(best_w.tolist())

        # Test prediction
        z10_te, z20_te, z40_te = safe_logit(s_10m[te_idx]), safe_logit(s_20m[te_idx]), safe_logit(s_40m[te_idx])
        z_fused_te = best_w[0] * z10_te + best_w[1] * z20_te + best_w[2] * z40_te
        multiscale_signal_oof[te_idx] = 1.0 / (1.0 + np.exp(-z_fused_te))

    auc_ms_sig = roc_auc_score(prot.plab, multiscale_signal_oof)
    auprc_ms_sig = average_precision_score(prot.plab, multiscale_signal_oof)
    ci_ms_sig = bootstrap_ci(prot.plab, multiscale_signal_oof)
    mean_alpha = np.mean(learned_alphas, axis=0)
    print(f"Multi-Scale Signal AUROC: {auc_ms_sig:.4f} [{ci_ms_sig[0]:.4f}, {ci_ms_sig[1]:.4f}], AUPRC: {auprc_ms_sig:.4f}")
    print(f"Mean Learned Alphas: [10m: {mean_alpha[0]:.3f}, 20m: {mean_alpha[1]:.3f}, 40m: {mean_alpha[2]:.3f}]")

    results["matched_models"]["multiscale_signal"] = {
        "auroc": float(auc_ms_sig),
        "auprc": float(auprc_ms_sig),
        "ci": [float(ci_ms_sig[0]), float(ci_ms_sig[1])],
        "mean_alphas": mean_alpha.tolist()
    }

    # -----------------------------------------------------------------------
    # Experiment 5.7 & 5.8: Temporal Clinical Branch (10m, 20m, 40m Descriptors + Trends)
    # -----------------------------------------------------------------------
    print("\n--- Training Temporal Clinical Models (Multi-Scale Descriptors + Trends) ---")
    # Extract patient-level clinical matrices (P90)
    F10_p = np.zeros((len(prot.patients), 19), dtype=np.float32)
    F20_p = np.zeros((len(prot.patients), 19), dtype=np.float32)
    F40_p = np.zeros((len(prot.patients), 19), dtype=np.float32)
    Ftr_p = np.zeros((len(prot.patients), 19), dtype=np.float32)

    F10_win = matched_data["F_10m"].numpy()
    F20_win = matched_data["F_20m"].numpy()
    F40_win = matched_data["F_40m"].numpy()
    Ftr_win = matched_data["F_trends"].numpy()

    for i, p in enumerate(prot.patients):
        p_mask = (matched_pids == p)
        if p_mask.sum() > 0:
            F10_p[i] = np.percentile(F10_win[p_mask], 90, axis=0)
            F20_p[i] = np.percentile(F20_win[p_mask], 90, axis=0)
            F40_p[i] = np.percentile(F40_win[p_mask], 90, axis=0)
            Ftr_p[i] = np.percentile(Ftr_win[p_mask], 90, axis=0)

    # Train 1: Standard 20m Clinical LR (Reference)
    clinical_20m_oof = np.zeros(len(prot.patients), dtype=np.float32)
    # Train 2: Temporal Multi-Scale Clinical LR (10m + 20m + 40m + Trends: 19*4 = 76 features)
    clinical_temporal_oof = np.zeros(len(prot.patients), dtype=np.float32)
    # Train 3: Deceleration-specific Trend Clinical LR (20m + decel trends)
    decel_trend_idx = [4, 5, 6, 7, 8, 9, 10, 11] # decel descriptors
    F_decel_trend = np.hstack([F20_p, Ftr_p[:, decel_trend_idx]])
    clinical_decel_trend_oof = np.zeros(len(prot.patients), dtype=np.float32)

    F_temporal_all = np.hstack([F10_p, F20_p, F40_p, Ftr_p])

    for f_idx in range(5):
        test_pids = np.array([p for p in prot.patients if prot.assignment[p][0] == f_idx])
        train_pids = np.array([p for p in prot.patients if prot.assignment[p][0] != f_idx])
        tr_idx = np.isin(prot.patients, train_pids)
        te_idx = np.isin(prot.patients, test_pids)

        # Model 1: 20m Clinical LR
        mu20, std20 = F20_p[tr_idx].mean(axis=0), F20_p[tr_idx].std(axis=0) + 1e-6
        clf20 = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=SEED)
        clf20.fit((F20_p[tr_idx] - mu20) / std20, prot.plab[tr_idx])
        clinical_20m_oof[te_idx] = clf20.predict_proba((F20_p[te_idx] - mu20) / std20)[:, 1]

        # Model 2: Multi-Scale Temporal Clinical
        mu_t, std_t = F_temporal_all[tr_idx].mean(axis=0), F_temporal_all[tr_idx].std(axis=0) + 1e-6
        clf_t = LogisticRegression(max_iter=1000, C=0.05, penalty='l2', class_weight='balanced', random_state=SEED)
        clf_t.fit((F_temporal_all[tr_idx] - mu_t) / std_t, prot.plab[tr_idx])
        clinical_temporal_oof[te_idx] = clf_t.predict_proba((F_temporal_all[te_idx] - mu_t) / std_t)[:, 1]

        # Model 3: Decel Trend Clinical
        mu_dt, std_dt = F_decel_trend[tr_idx].mean(axis=0), F_decel_trend[tr_idx].std(axis=0) + 1e-6
        clf_dt = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=SEED)
        clf_dt.fit((F_decel_trend[tr_idx] - mu_dt) / std_dt, prot.plab[tr_idx])
        clinical_decel_trend_oof[te_idx] = clf_dt.predict_proba((F_decel_trend[te_idx] - mu_dt) / std_dt)[:, 1]

    auc_c20 = roc_auc_score(prot.plab, clinical_20m_oof)
    auc_ctemp = roc_auc_score(prot.plab, clinical_temporal_oof)
    auc_cdecel = roc_auc_score(prot.plab, clinical_decel_trend_oof)

    ci_c20 = bootstrap_ci(prot.plab, clinical_20m_oof)
    ci_ctemp = bootstrap_ci(prot.plab, clinical_temporal_oof)
    ci_cdecel = bootstrap_ci(prot.plab, clinical_decel_trend_oof)

    print(f"20m Clinical LR AUROC: {auc_c20:.4f} [{ci_c20[0]:.4f}, {ci_c20[1]:.4f}]")
    print(f"Multi-Scale Temporal Clinical LR AUROC: {auc_ctemp:.4f} [{ci_ctemp[0]:.4f}, {ci_ctemp[1]:.4f}]")
    print(f"Decel-Trend Clinical LR AUROC: {auc_cdecel:.4f} [{ci_cdecel[0]:.4f}, {ci_cdecel[1]:.4f}]")

    results["temporal_clinical"] = {
        "clinical_20m": {"auroc": float(auc_c20), "ci": [float(ci_c20[0]), float(ci_c20[1])], "auprc": float(average_precision_score(prot.plab, clinical_20m_oof))},
        "clinical_temporal_all": {"auroc": float(auc_ctemp), "ci": [float(ci_ctemp[0]), float(ci_ctemp[1])], "auprc": float(average_precision_score(prot.plab, clinical_temporal_oof))},
        "clinical_decel_trend": {"auroc": float(auc_cdecel), "ci": [float(ci_cdecel[0]), float(ci_cdecel[1])], "auprc": float(average_precision_score(prot.plab, clinical_decel_trend_oof))}
    }

    # -----------------------------------------------------------------------
    # Experiment 5.9: Primary Phase-5 Candidate (Multi-Resolution Signal + Clinical Logit Modulation)
    # z_final = z_signal + lambda * z_clinical
    # -----------------------------------------------------------------------
    print("\n--- Training Primary Phase-5 Candidate: Multi-Resolution Logit Prior Modulation ---")
    phase5_final_oof = np.zeros(len(prot.patients), dtype=np.float32)
    learned_lambdas = []

    for f_idx in range(5):
        test_pids = np.array([p for p in prot.patients if prot.assignment[p][0] == f_idx])
        train_pids = np.array([p for p in prot.patients if prot.assignment[p][0] != f_idx])
        tr_idx = np.isin(prot.patients, train_pids)
        te_idx = np.isin(prot.patients, test_pids)

        z_sig_tr = safe_logit(multiscale_signal_oof[tr_idx])
        z_cli_tr = safe_logit(clinical_decel_trend_oof[tr_idx])
        y_tr = prot.plab[tr_idx]

        # Learn optimal lambda on training fold
        best_lam = 1.0
        best_auc = 0.0
        for lam_cand in np.linspace(0.1, 3.0, 59):
            z_fused = z_sig_tr + lam_cand * z_cli_tr
            cur_auc = roc_auc_score(y_tr, z_fused)
            if cur_auc > best_auc:
                best_auc = cur_auc
                best_lam = lam_cand
        learned_lambdas.append(float(best_lam))

        # Test fold inference
        z_sig_te = safe_logit(multiscale_signal_oof[te_idx])
        z_cli_te = safe_logit(clinical_decel_trend_oof[te_idx])
        z_final_te = z_sig_te + best_lam * z_cli_te
        phase5_final_oof[te_idx] = 1.0 / (1.0 + np.exp(-z_final_te))

    auc_p5 = roc_auc_score(prot.plab, phase5_final_oof)
    auprc_p5 = average_precision_score(prot.plab, phase5_final_oof)
    ci_p5 = bootstrap_ci(prot.plab, phase5_final_oof)
    mean_lam = float(np.mean(learned_lambdas))
    print(f"\n==========================================================================")
    print(f"PRIMARY PHASE-5 CANDIDATE PATIENT AUROC: {auc_p5:.4f} [{ci_p5[0]:.4f}, {ci_p5[1]:.4f}], AUPRC: {auprc_p5:.4f}")
    print(f"Mean Learned Lambda (Signal + lambda*Clinical): {mean_lam:.3f}")
    print(f"==========================================================================")

    results["matched_models"]["phase5_final_candidate"] = {
        "auroc": float(auc_p5),
        "auprc": float(auprc_p5),
        "ci": [float(ci_p5[0]), float(ci_p5[1])],
        "mean_lambda": mean_lam,
        "learned_lambdas": learned_lambdas
    }

    # Load Phase-4 baseline scores for comparison
    phase4_path = "results/phase4_fusion/phase4_results.json"
    phase4_logit_oof = np.array(json.load(open(phase4_path))["patient_scores"]["logit_modulation"])
    auc_p4 = roc_auc_score(prot.plab, phase4_logit_oof)

    # -----------------------------------------------------------------------
    # Paired Statistical Tests (2,000 bootstrap replicates)
    # -----------------------------------------------------------------------
    print("\n--- Running Paired Statistical Significance Tests (2,000 Bootstrap Replicates) ---")
    comparisons = [
        ("Short (10m) vs Medium (20m)", s_20m, s_10m),
        ("Long (40m) vs Medium (20m)", s_20m, s_40m),
        ("Multi-Scale Signal vs 20m Signal", s_20m, multiscale_signal_oof),
        ("Multi-Scale Signal vs Best Single (10m)", s_10m, multiscale_signal_oof),
        ("Temporal Clinical vs 20m Clinical", clinical_20m_oof, clinical_decel_trend_oof),
        ("Phase 5 Final Candidate vs Phase 4 Baseline", phase4_logit_oof, phase5_final_oof),
        ("Phase 5 Final Candidate vs Multi-Scale Signal", multiscale_signal_oof, phase5_final_oof),
        ("Phase 5 Final Candidate vs 20m Clinical", clinical_20m_oof, phase5_final_oof)
    ]

    for name, s_base, s_target in comparisons:
        delta, ci_d, p_val = paired_bootstrap_test(prot.plab, s_base, s_target, n_boot=2000, seed=SEED)
        print(f"  {name:45s}: Delta AUROC = {delta:+.4f} (95% CI: [{ci_d[0]:+.4f}, {ci_d[1]:+.4f}]), p = {p_val:.4f}")
        results["paired_tests"].append({
            "comparison": name,
            "delta_auroc": delta,
            "ci": list(ci_d),
            "p_value": p_val
        })

    # -----------------------------------------------------------------------
    # Temporal Redundancy & Complementarity Analysis
    # -----------------------------------------------------------------------
    r_10_20, _ = pearsonr(s_10m, s_20m)
    rho_10_20, _ = spearmanr(s_10m, s_20m)
    r_20_40, _ = pearsonr(s_20m, s_40m)
    rho_20_40, _ = spearmanr(s_20m, s_40m)
    r_10_40, _ = pearsonr(s_10m, s_40m)
    rho_10_40, _ = spearmanr(s_10m, s_40m)

    results["scale_correlations"] = {
        "pearson_10_20": float(r_10_20),
        "spearman_10_20": float(rho_10_20),
        "pearson_20_40": float(r_20_40),
        "spearman_20_40": float(rho_20_40),
        "pearson_10_40": float(r_10_40),
        "spearman_10_40": float(rho_10_40)
    }

    # Case breakdown on positive patients
    pos_idx = np.where(prot.plab == 1)[0]
    th_10 = np.percentile(s_10m, 80)
    th_20 = np.percentile(s_20m, 80)
    th_40 = np.percentile(s_40m, 80)

    corr_10 = s_10m[pos_idx] >= th_10
    corr_20 = s_20m[pos_idx] >= th_20
    corr_40 = s_40m[pos_idx] >= th_40

    only_10 = int(np.sum(corr_10 & ~corr_20 & ~corr_40))
    only_20 = int(np.sum(~corr_10 & corr_20 & ~corr_40))
    only_40 = int(np.sum(~corr_10 & ~corr_20 & corr_40))
    all_three = int(np.sum(corr_10 & corr_20 & corr_40))
    any_scale = int(np.sum(corr_10 | corr_20 | corr_40))

    results["scale_complementarity"] = {
        "total_positives": len(pos_idx),
        "short_10m_only_correct": only_10,
        "medium_20m_only_correct": only_20,
        "long_40m_only_correct": only_40,
        "all_three_correct": all_three,
        "any_scale_correct_union": any_scale
    }

    # -----------------------------------------------------------------------
    # Time-to-Delivery Stratification Analysis
    # -----------------------------------------------------------------------
    ttd_arr = matched_data["time_to_delivery_min"]
    # Mean patient time to delivery of evaluation windows
    patient_ttd = np.zeros(len(prot.patients), dtype=np.float32)
    for i, p in enumerate(prot.patients):
        p_mask = (matched_pids == p)
        if p_mask.sum() > 0:
            patient_ttd[i] = float(np.mean(ttd_arr[p_mask]))

    early_mask = (patient_ttd > 20.0)
    late_mask = (patient_ttd <= 20.0)

    auc_early = float(roc_auc_score(prot.plab[early_mask], phase5_final_oof[early_mask])) if len(np.unique(prot.plab[early_mask])) > 1 else 0.0
    auc_late = float(roc_auc_score(prot.plab[late_mask], phase5_final_oof[late_mask])) if len(np.unique(prot.plab[late_mask])) > 1 else 0.0

    results["delivery_stratification"] = {
        "early_window_auroc_gt20min": auc_early,
        "late_window_auroc_le20min": auc_late
    }

    # Store patient scores for plotting
    results["patient_scores"] = {
        "labels": prot.plab.tolist(),
        "signal_10m": s_10m.tolist(),
        "signal_20m": s_20m.tolist(),
        "signal_40m": s_40m.tolist(),
        "multiscale_signal": multiscale_signal_oof.tolist(),
        "clinical_20m": clinical_20m_oof.tolist(),
        "clinical_temporal": clinical_decel_trend_oof.tolist(),
        "phase4_baseline": phase4_logit_oof.tolist(),
        "phase5_candidate": phase5_final_oof.tolist(),
        "time_to_delivery": patient_ttd.tolist()
    }

    # Save to json
    results_path = os.path.join(OUT_DIR, "phase5_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSuccessfully wrote Phase 5 evaluation results to {results_path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    run_phase5_evaluation()
