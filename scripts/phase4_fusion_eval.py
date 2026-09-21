"""
Phase 4: CTU-UHB Knowledge-Guided 1D Signal-Clinical Fusion.

Evaluates 19-feature Clinical LR, Feature Group Ablations (A-F), Signal-Only Control,
Naive Concatenation, Gated Fusion, Residual Fusion, Logit Modulation, and Residual Prediction
under the frozen 5-fold patient-grouped protocol.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.protocol import Protocol, load_clinical
from src.models.cnn1d_encoder import CNN1DEncoder

FOLDS_PATH = "data/processed_clinical/folds.json"
DATA_DIR = "data/processed_clinical"
OUT_DIR = "results/phase4_fusion"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_FUSION = 35
BATCH_SIZE = 64
LR_FUSION = 1e-3

# 19 Feature Group Index Map
FEATURE_GROUPS = {
    "Group_A_Baseline": [0, 12],                      # baseline_val, baseline_slope
    "Group_B_Variability": [1, 2, 13],                # stv, ltv, variability_slope
    "Group_C_Accelerations": [3],                     # accel_count
    "Group_D_Decelerations": [4, 5, 6, 7, 8, 9, 10, 11], # early, late, var, prol, max_depth, area, burden, longest
    "Group_E_Uterine_Activity": [14, 15, 16],         # contraction_count, tachysystole, mean_amplitude
    "Group_F_FHR_UC_Coupling": [17, 18],              # fhr_uc_lag, fhr_uc_coupling
}


# ---------------------------------------------------------------------------
# Fusion Architectures
# ---------------------------------------------------------------------------
class NaiveConcatFusion(nn.Module):
    def __init__(self, signal_dim: int = 128, clinical_dim: int = 19):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(signal_dim + clinical_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, h_s: torch.Tensor, x_c: torch.Tensor) -> torch.Tensor:
        z = torch.cat([h_s, x_c], dim=-1)
        return self.classifier(z).squeeze(-1)


class GatedKnowledgeFusion(nn.Module):
    def __init__(self, signal_dim: int = 128, clinical_dim: int = 19, hidden_dim: int = 64):
        super().__init__()
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, signal_dim)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(signal_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, signal_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(signal_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, h_s: torch.Tensor, x_c: torch.Tensor):
        h_c = self.clinical_proj(x_c) # (B, 128)
        gate_in = torch.cat([h_s, h_c], dim=-1) # (B, 256)
        g = self.gate_net(gate_in) # (B, 128)
        h_fused = g * h_s + (1.0 - g) * h_c # (B, 128)
        logits = self.classifier(h_fused).squeeze(-1)
        return logits, g


class ResidualKnowledgeFusion(nn.Module):
    def __init__(self, signal_dim: int = 128, clinical_dim: int = 19, hidden_dim: int = 64):
        super().__init__()
        self.residual_net = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, signal_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(signal_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, h_s: torch.Tensor, x_c: torch.Tensor):
        res = self.residual_net(x_c) # (B, 128)
        h_fused = h_s + res # (B, 128)
        logits = self.classifier(h_fused).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Evaluation Helper Functions
# ---------------------------------------------------------------------------
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


def run_phase4():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== STARTING PHASE 4: KNOWLEDGE-GUIDED 1D SIGNAL-CLINICAL FUSION ===")

    # 1. Load P2 Quality-Aware dataset (patient-by-patient order)
    p2_data = torch.load("data/phase1_candidates/p2_dataset.pt", weights_only=False)
    y_arr = p2_data["y"].numpy()
    pid_arr = p2_data["pid"]
    meta_p2 = [tuple(m) for m in p2_data["meta"]] # list of (str(pid), int(start), int(end))
    prot = Protocol.load_or_create(pid_arr, y_arr, path=FOLDS_PATH)
    print(f"Loaded {len(y_arr)} windows from {len(prot.patients)} patients ({int(prot.plab.sum())} positive).")

    # 2. Load 19 clinical features from data/processed_clinical/ and map to p2_dataset.pt order
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
    print(f"Aligned clinical features shape: {Fe_aligned.shape}")

    # 3. Extract Patient-Level Clinical Feature Matrix (P90 of window descriptors)
    p_features = np.zeros((len(prot.patients), Fe_aligned.shape[1]), dtype=np.float32)
    for i, p in enumerate(prot.patients):
        idx = prot.pidx[p]
        p_features[i] = np.percentile(Fe_aligned[idx], 90, axis=0) # P90 aggregation

    y_patients = prot.plab.copy()

    # 4. Load pre-extracted 1D Signal Window Embeddings and Window Probs from Phase 3
    emb_cache_path = "results/phase3_mil/extracted_embeddings.npz"
    cached = np.load(emb_cache_path)
    window_embeddings = cached["embeddings"] # (8517, 128)
    window_signal_probs = cached["window_probs"] # (8517,)

    # Patient-level signal representations (Max / P90)
    _, sc_signal_max = prot.to_patient(window_signal_probs, how="max")
    _, sc_signal_p90 = prot.to_patient(window_signal_probs, how="p90")

    # Patient-level top-3 window embedding
    p_signal_embs = np.zeros((len(prot.patients), 128), dtype=np.float32)
    for i, p in enumerate(prot.patients):
        idx = prot.pidx[p]
        top_idx = idx[np.argsort(window_signal_probs[idx])[-min(3, len(idx)):]]
        p_signal_embs[i] = window_embeddings[top_idx].mean(axis=0)

    results = {}

    # -------------------------------------------------------------
    # Exp 4.0: Clinical Baseline (19-feature Logistic Regression)
    # -------------------------------------------------------------
    print("\n--- Exp 4.0: Clinical Baseline (19-Feature LR) ---")
    oof_clinical_probs = np.zeros(len(prot.patients), dtype=float)

    for fold, (tr_mask, te_patients) in enumerate(prot.folds(repeat=0), 1):
        tr_p_mask = np.array([p not in te_patients for p in prot.patients])
        te_p_mask = ~tr_p_mask

        X_tr = p_features[tr_p_mask].copy()
        X_te = p_features[te_p_mask].copy()
        y_tr = y_patients[tr_p_mask]

        # Standardize strictly on training patients
        mu = X_tr.mean(axis=0, keepdims=True)
        sig = X_tr.std(axis=0, keepdims=True) + 1e-6
        X_tr = (X_tr - mu) / sig
        X_te = (X_te - mu) / sig

        lr = LogisticRegression(C=0.1, class_weight='balanced', random_state=SEED + fold)
        lr.fit(X_tr, y_tr)
        oof_clinical_probs[te_p_mask] = lr.predict_proba(X_te)[:, 1]

    rep_clin = prot.report_patient_scores("Clinical_19_LR", y_patients, oof_clinical_probs)
    results["Clinical_19_LR"] = rep_clin

    # -------------------------------------------------------------
    # Exp 4.1: Feature Group Ablations (Groups A - F)
    # -------------------------------------------------------------
    print("\n--- Exp 4.1: Clinical Feature Group Ablations ---")
    group_results = {}
    for g_name, col_indices in FEATURE_GROUPS.items():
        oof_group_probs = np.zeros(len(prot.patients), dtype=float)
        for fold, (tr_mask, te_patients) in enumerate(prot.folds(repeat=0), 1):
            tr_p_mask = np.array([p not in te_patients for p in prot.patients])
            te_p_mask = ~tr_p_mask

            X_tr = p_features[tr_p_mask][:, col_indices].copy()
            X_te = p_features[te_p_mask][:, col_indices].copy()
            y_tr = y_patients[tr_p_mask]

            mu = X_tr.mean(axis=0, keepdims=True)
            sig = X_tr.std(axis=0, keepdims=True) + 1e-6
            X_tr = (X_tr - mu) / sig
            X_te = (X_te - mu) / sig

            lr = LogisticRegression(C=0.1, class_weight='balanced', random_state=SEED + fold)
            lr.fit(X_tr, y_tr)
            oof_group_probs[te_p_mask] = lr.predict_proba(X_te)[:, 1]

        rep_g = prot.report_patient_scores(f"Group_{g_name}", y_patients, oof_group_probs)
        group_results[g_name] = rep_g
    results["Feature_Group_Ablations"] = group_results

    # -------------------------------------------------------------
    # Exp 4.2: Signal-Only Control
    # -------------------------------------------------------------
    print("\n--- Exp 4.2: Signal-Only Control ---")
    rep_sig_max = prot.report_patient_scores("Signal_P90_Control", y_patients, sc_signal_p90)
    results["Signal_P90_Control"] = rep_sig_max

    # -------------------------------------------------------------
    # Exp 4.3: Naive Concatenation Fusion ([h_s, x_c] -> MLP)
    # -------------------------------------------------------------
    print("\n--- Exp 4.3: Naive Concatenation Fusion ---")
    oof_concat_probs = np.zeros(len(prot.patients), dtype=float)

    for fold, (tr_mask, te_patients) in enumerate(prot.folds(repeat=0), 1):
        tr_p_mask = np.array([p not in te_patients for p in prot.patients])
        te_p_mask = ~tr_p_mask

        X_s_tr = p_signal_embs[tr_p_mask].copy()
        X_s_te = p_signal_embs[te_p_mask].copy()
        X_c_tr = p_features[tr_p_mask].copy()
        X_c_te = p_features[te_p_mask].copy()
        y_tr = y_patients[tr_p_mask]

        mu_c = X_c_tr.mean(axis=0, keepdims=True)
        sig_c = X_c_tr.std(axis=0, keepdims=True) + 1e-6
        X_c_tr = (X_c_tr - mu_c) / sig_c
        X_c_te = (X_c_te - mu_c) / sig_c

        # Split val patients from train
        fit_idx, val_idx = np.where(tr_p_mask)[0][:int(len(X_tr)*0.8)], np.where(tr_p_mask)[0][int(len(X_tr)*0.8):]

        torch.manual_seed(SEED + fold)
        model = NaiveConcatFusion(signal_dim=128, clinical_dim=19).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FUSION, weight_decay=1e-3)
        pos_weight = float((len(y_tr) - sum(y_tr)) / max(1, sum(y_tr)))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))

        tr_ds = TensorDataset(torch.tensor(X_s_tr, dtype=torch.float32),
                              torch.tensor(X_c_tr, dtype=torch.float32),
                              torch.tensor(y_tr, dtype=torch.float32))
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)

        for ep in range(EPOCHS_FUSION):
            model.train()
            for s_b, c_b, y_b in tr_loader:
                s_b, c_b, y_b = s_b.to(DEVICE), c_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                logits = model(s_b, c_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            s_te_t = torch.tensor(X_s_te, dtype=torch.float32).to(DEVICE)
            c_te_t = torch.tensor(X_c_te, dtype=torch.float32).to(DEVICE)
            logits = model(s_te_t, c_te_t)
            oof_concat_probs[te_p_mask] = torch.sigmoid(logits).cpu().numpy()

    rep_concat = prot.report_patient_scores("Naive_Concat_Fusion", y_patients, oof_concat_probs)
    results["Naive_Concat_Fusion"] = rep_concat

    # -------------------------------------------------------------
    # Exp 4.4: Gated Knowledge Fusion (Primary)
    # -------------------------------------------------------------
    print("\n--- Exp 4.4: Gated Knowledge Fusion (Primary) ---")
    oof_gated_probs = np.zeros(len(prot.patients), dtype=float)
    gate_values = np.zeros(len(prot.patients), dtype=float)

    for fold, (tr_mask, te_patients) in enumerate(prot.folds(repeat=0), 1):
        tr_p_mask = np.array([p not in te_patients for p in prot.patients])
        te_p_mask = ~tr_p_mask

        X_s_tr = p_signal_embs[tr_p_mask].copy()
        X_s_te = p_signal_embs[te_p_mask].copy()
        X_c_tr = p_features[tr_p_mask].copy()
        X_c_te = p_features[te_p_mask].copy()
        y_tr = y_patients[tr_p_mask]

        mu_c = X_c_tr.mean(axis=0, keepdims=True)
        sig_c = X_c_tr.std(axis=0, keepdims=True) + 1e-6
        X_c_tr = (X_c_tr - mu_c) / sig_c
        X_c_te = (X_c_te - mu_c) / sig_c

        torch.manual_seed(SEED + fold)
        model = GatedKnowledgeFusion(signal_dim=128, clinical_dim=19).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FUSION, weight_decay=1e-3)
        pos_weight = float((len(y_tr) - sum(y_tr)) / max(1, sum(y_tr)))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))

        tr_ds = TensorDataset(torch.tensor(X_s_tr, dtype=torch.float32),
                              torch.tensor(X_c_tr, dtype=torch.float32),
                              torch.tensor(y_tr, dtype=torch.float32))
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)

        for ep in range(EPOCHS_FUSION):
            model.train()
            for s_b, c_b, y_b in tr_loader:
                s_b, c_b, y_b = s_b.to(DEVICE), c_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                logits, _ = model(s_b, c_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            s_te_t = torch.tensor(X_s_te, dtype=torch.float32).to(DEVICE)
            c_te_t = torch.tensor(X_c_te, dtype=torch.float32).to(DEVICE)
            logits, g_val = model(s_te_t, c_te_t)
            oof_gated_probs[te_p_mask] = torch.sigmoid(logits).cpu().numpy()
            gate_values[te_p_mask] = g_val.mean(dim=-1).cpu().numpy()

    rep_gated = prot.report_patient_scores("Gated_Knowledge_Fusion", y_patients, oof_gated_probs)
    results["Gated_Knowledge_Fusion"] = rep_gated

    # -------------------------------------------------------------
    # Exp 4.5: Residual Knowledge Fusion (h_f = h_s + g(h_c))
    # -------------------------------------------------------------
    print("\n--- Exp 4.5: Residual Knowledge Fusion ---")
    oof_resid_probs = np.zeros(len(prot.patients), dtype=float)

    for fold, (tr_mask, te_patients) in enumerate(prot.folds(repeat=0), 1):
        tr_p_mask = np.array([p not in te_patients for p in prot.patients])
        te_p_mask = ~tr_p_mask

        X_s_tr = p_signal_embs[tr_p_mask].copy()
        X_s_te = p_signal_embs[te_p_mask].copy()
        X_c_tr = p_features[tr_p_mask].copy()
        X_c_te = p_features[te_p_mask].copy()
        y_tr = y_patients[tr_p_mask]

        mu_c = X_c_tr.mean(axis=0, keepdims=True)
        sig_c = X_c_tr.std(axis=0, keepdims=True) + 1e-6
        X_c_tr = (X_c_tr - mu_c) / sig_c
        X_c_te = (X_c_te - mu_c) / sig_c

        torch.manual_seed(SEED + fold)
        model = ResidualKnowledgeFusion(signal_dim=128, clinical_dim=19).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FUSION, weight_decay=1e-3)
        pos_weight = float((len(y_tr) - sum(y_tr)) / max(1, sum(y_tr)))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))

        tr_ds = TensorDataset(torch.tensor(X_s_tr, dtype=torch.float32),
                              torch.tensor(X_c_tr, dtype=torch.float32),
                              torch.tensor(y_tr, dtype=torch.float32))
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)

        for ep in range(EPOCHS_FUSION):
            model.train()
            for s_b, c_b, y_b in tr_loader:
                s_b, c_b, y_b = s_b.to(DEVICE), c_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                logits = model(s_b, c_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            s_te_t = torch.tensor(X_s_te, dtype=torch.float32).to(DEVICE)
            c_te_t = torch.tensor(X_c_te, dtype=torch.float32).to(DEVICE)
            logits = model(s_te_t, c_te_t)
            oof_resid_probs[te_p_mask] = torch.sigmoid(logits).cpu().numpy()

    rep_resid = prot.report_patient_scores("Residual_Knowledge_Fusion", y_patients, oof_resid_probs)
    results["Residual_Knowledge_Fusion"] = rep_resid

    # -------------------------------------------------------------
    # Exp 4.6: Clinical Prior Logit Modulation (logit(p_f) = logit(p_s) + lambda * logit(p_c))
    # -------------------------------------------------------------
    print("\n--- Exp 4.6: Clinical Prior Logit Modulation ---")
    oof_logit_probs = np.zeros(len(prot.patients), dtype=float)

    # Convert probs to logits with epsilon
    eps = 1e-6
    logit_s = np.log(np.clip(sc_signal_p90, eps, 1.0 - eps) / (1.0 - np.clip(sc_signal_p90, eps, 1.0 - eps)))
    logit_c = np.log(np.clip(oof_clinical_probs, eps, 1.0 - eps) / (1.0 - np.clip(oof_clinical_probs, eps, 1.0 - eps)))

    for fold, (tr_mask, te_patients) in enumerate(prot.folds(repeat=0), 1):
        tr_p_mask = np.array([p not in te_patients for p in prot.patients])
        te_p_mask = ~tr_p_mask

        # Learn optimal weight lambda on training fold
        best_lam = 1.0
        best_auc = -1.0
        for lam_cand in np.linspace(0.1, 3.0, 30):
            comb = logit_s[tr_p_mask] + lam_cand * logit_c[tr_p_mask]
            auc = roc_auc_score(y_patients[tr_p_mask], comb)
            if auc > best_auc:
                best_auc = auc
                best_lam = lam_cand

        comb_te = logit_s[te_p_mask] + best_lam * logit_c[te_p_mask]
        oof_logit_probs[te_p_mask] = 1.0 / (1.0 + np.exp(-comb_te))

    rep_logit = prot.report_patient_scores("Logit_Prior_Modulation", y_patients, oof_logit_probs)
    results["Logit_Prior_Modulation"] = rep_logit

    # -------------------------------------------------------------
    # Exp 4.7: Residual Clinical Correction (Signal predicting clinical residuals)
    # -------------------------------------------------------------
    print("\n--- Exp 4.7: Residual Clinical Prediction ---")
    # Clinical residuals: r_i = y_i - p_clinical
    residuals = y_patients - oof_clinical_probs
    oof_resid_pred_probs = np.zeros(len(prot.patients), dtype=float)

    for fold, (tr_mask, te_patients) in enumerate(prot.folds(repeat=0), 1):
        tr_p_mask = np.array([p not in te_patients for p in prot.patients])
        te_p_mask = ~tr_p_mask

        X_s_tr = p_signal_embs[tr_p_mask].copy()
        X_s_te = p_signal_embs[te_p_mask].copy()
        r_tr = residuals[tr_p_mask]

        # Fit ridge/linear regression to predict residuals
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=10.0)
        reg.fit(X_s_tr, r_tr)
        r_pred = reg.predict(X_s_te)

        # Corrected prediction: p_corr = p_clinical + r_pred
        p_corr = np.clip(oof_clinical_probs[te_p_mask] + r_pred, 0.0, 1.0)
        oof_resid_pred_probs[te_p_mask] = p_corr

    rep_resid_pred = prot.report_patient_scores("Residual_Error_Correction", y_patients, oof_resid_pred_probs)
    results["Residual_Error_Correction"] = rep_resid_pred

    # -------------------------------------------------------------
    # Paired Statistical Tests against Clinical LR & Signal-Only Control
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("PAIRED STATISTICAL COMPARISONS")
    print("=======================================================")
    paired_comparisons = []
    models_to_test = [
        ("Signal_P90_Control", sc_signal_p90),
        ("Naive_Concat_Fusion", oof_concat_probs),
        ("Gated_Knowledge_Fusion", oof_gated_probs),
        ("Residual_Knowledge_Fusion", oof_resid_probs),
        ("Logit_Prior_Modulation", oof_logit_probs),
        ("Residual_Error_Correction", oof_resid_pred_probs),
    ]

    for tag, sc_m in models_to_test:
        d_vs_clin, ci_clin, p_clin = paired_bootstrap_test(y_patients, oof_clinical_probs, sc_m)
        d_vs_sig, ci_sig, p_sig = paired_bootstrap_test(y_patients, sc_signal_p90, sc_m)
        print(f"  {tag:26s} vs Clinical LR: dAUROC = {d_vs_clin:+.4f} [{ci_clin[0]:.3f}, {ci_clin[1]:.3f}] (p={p_clin:.3f}) | vs Signal: dAUROC = {d_vs_sig:+.4f} [{ci_sig[0]:.3f}, {ci_sig[1]:.3f}] (p={p_sig:.3f})")
        paired_comparisons.append({
            "model": tag,
            "delta_vs_clinical": d_vs_clin,
            "ci_vs_clinical": list(ci_clin),
            "p_val_vs_clinical": p_clin,
            "delta_vs_signal": d_vs_sig,
            "ci_vs_signal": list(ci_sig),
            "p_val_vs_signal": p_sig
        })

    # -------------------------------------------------------------
    # Complementarity & Error Correlation Analysis
    # -------------------------------------------------------------
    r_p_sc, _ = pearsonr(sc_signal_p90, oof_clinical_probs)
    r_s_sc, _ = spearmanr(sc_signal_p90, oof_clinical_probs)

    e_s = y_patients - sc_signal_p90
    e_c = y_patients - oof_clinical_probs
    r_p_err, _ = pearsonr(e_s, e_c)
    r_s_err, _ = spearmanr(e_s, e_c)

    # Threshold for binary decision: top-110 ranked patients
    n_pos = int(np.sum(y_patients))
    thresh_s = np.sort(sc_signal_p90)[-n_pos]
    thresh_c = np.sort(oof_clinical_probs)[-n_pos]

    pred_s = (sc_signal_p90 >= thresh_s).astype(int)
    pred_c = (oof_clinical_probs >= thresh_c).astype(int)

    sig_correct = (pred_s == y_patients)
    clin_correct = (pred_c == y_patients)

    sig_only_pos = int(np.sum((pred_s == 1) & (pred_c == 0) & (y_patients == 1)))
    clin_only_pos = int(np.sum((pred_c == 1) & (pred_s == 0) & (y_patients == 1)))
    both_correct_pos = int(np.sum((pred_s == 1) & (pred_c == 1) & (y_patients == 1)))
    both_wrong_pos = int(np.sum((pred_s == 0) & (pred_c == 0) & (y_patients == 1)))

    # Calibration analysis for best fusion model (Logit Prior Modulation)
    brier_clin = brier_score_loss(y_patients, oof_clinical_probs)
    brier_sig = brier_score_loss(y_patients, sc_signal_p90)
    brier_logit = brier_score_loss(y_patients, oof_logit_probs)

    complementarity_payload = {
        "signal_vs_clinical_score_pearson": float(r_p_sc),
        "signal_vs_clinical_score_spearman": float(r_s_sc),
        "signal_error_vs_clinical_error_pearson": float(r_p_err),
        "signal_error_vs_clinical_error_spearman": float(r_s_err),
        "patient_breakdown_positives": {
            "total_positives": n_pos,
            "both_correct": both_correct_pos,
            "clinical_only_correct": clin_only_pos,
            "signal_only_correct": sig_only_pos,
            "both_incorrect": both_wrong_pos
        },
        "calibration": {
            "brier_clinical": float(brier_clin),
            "brier_signal": float(brier_sig),
            "brier_logit_fusion": float(brier_logit)
        },
        "mean_gate_activation": float(np.mean(gate_values))
    }

    # Save summary payload
    final_payload = {
        "benchmark_clinical_lr": {
            "auroc": rep_clin["auroc"],
            "ci": [rep_clin["ci_lo"], rep_clin["ci_hi"]],
            "auprc": rep_clin["auprc"]
        },
        "results": {k: {
            "name": v["name"],
            "auroc": float(v["auroc"]),
            "auprc": float(v["auprc"]),
            "ci": [float(v["ci_lo"]), float(v["ci_hi"])]
        } for k, v in results.items() if k != "Feature_Group_Ablations"},
        "feature_group_ablations": {k: {
            "name": v["name"],
            "auroc": float(v["auroc"]),
            "auprc": float(v["auprc"]),
            "ci": [float(v["ci_lo"]), float(v["ci_hi"])]
        } for k, v in group_results.items()},
        "paired_comparisons": paired_comparisons,
        "complementarity": complementarity_payload,
        "patient_scores": {
            "labels": [int(x) for x in y_patients],
            "clinical_lr": [float(x) for x in oof_clinical_probs],
            "signal_p90": [float(x) for x in sc_signal_p90],
            "naive_concat": [float(x) for x in oof_concat_probs],
            "gated_fusion": [float(x) for x in oof_gated_probs],
            "residual_fusion": [float(x) for x in oof_resid_probs],
            "logit_modulation": [float(x) for x in oof_logit_probs],
            "residual_correction": [float(x) for x in oof_resid_pred_probs]
        }
    }

    with open(os.path.join(OUT_DIR, "phase4_results.json"), "w") as fh:
        json.dump(final_payload, fh, indent=2)

    print(f"\nPhase 4 execution complete. Saved results to {OUT_DIR}/phase4_results.json")
    return final_payload


if __name__ == "__main__":
    run_phase4()
