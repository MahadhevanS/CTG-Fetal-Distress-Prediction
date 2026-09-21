"""
Export Final Deliverable Models for CTU-UHB Fetal Acidemia Prediction.

Exports:
1. Model B: Continuous Clinical Huber Regression (5-fold cross-validation models + full cohort model).
2. Scalers and feature statistics for 19 clinical descriptors.
3. Lightweight JSON parameters for zero-dependency inference.
4. Phase 4 Master Model metadata and weights.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.protocol import Protocol

FOLDS_PATH = "data/processed_clinical/folds.json"
P2_PATH = "data/phase1_candidates/p2_dataset.pt"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
MODEL_DIR = "models/continuous_clinical_huber"
PHASE4_DIR = "models/phase4_master"

FEATURE_NAMES = [
    "baseline", "stv", "ltv", "acc_count", "early_dec_count", "late_dec_count",
    "var_dec_count", "prolonged_dec_count", "dec_max_depth", "dec_area",
    "dec_burden", "longest_dec", "baseline_slope", "variability_slope",
    "uc_count", "tachysystole", "mean_uc_amp", "fhr_uc_lag", "fhr_uc_coupling"
]

def export_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PHASE4_DIR, exist_ok=True)
    print("=== EXPORTING FINAL DELIVERABLE MODELS ===")

    # 1. Load Folds & Data
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    p2_data = torch.load(P2_PATH, weights_only=False)
    y_bin = p2_data["y"].numpy()
    pid_arr = p2_data["pid"]

    prot = Protocol.load_or_create(pid_arr, y_bin, path=FOLDS_PATH)

    df_meta = pd.read_csv(METADATA_PATH)
    if 'record_id' in df_meta.columns:
        df_meta = df_meta.set_index('record_id')
    patient_ph = np.array([float(df_meta.loc[int(p), 'ph']) for p in clean_pids], dtype=np.float32)
    patient_labels = prot.plab.astype(np.int64)

    # 2. Extract 19 Clinical Features for 547 Patients
    DATA_DIR = "data/processed_clinical"
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
    meta_p2 = [tuple(m) for m in p2_data["meta"]]
    Fe_aligned = np.array([feat_lookup[(str(m[0]), int(m[1]), int(m[2]))] for m in meta_p2], dtype=np.float32)

    # Patient-level clinical descriptor matrix (P90)
    F_patient = np.zeros((len(clean_pids), Fe_aligned.shape[1]), dtype=np.float32)
    for i, p in enumerate(clean_pids):
        idx = prot.pidx[p]
        F_patient[i] = np.percentile(Fe_aligned[idx], 90, axis=0)

    print(f"Feature matrix shape: {F_patient.shape}")

    # 3. Fit and Export 5-Fold Huber Models
    ensemble_params = {"folds": [], "features": FEATURE_NAMES}
    ph_oof_preds = np.zeros(len(clean_pids), dtype=np.float32)

    for f_idx in range(5):
        te_pids = set([p for p in clean_pids if prot.assignment[p][0] == f_idx])
        tr_pat = np.array([i for i, p in enumerate(clean_pids) if p not in te_pids])
        te_pat = np.array([i for i, p in enumerate(clean_pids) if p in te_pids])

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(F_patient[tr_pat])
        X_te_scaled = scaler.transform(F_patient[te_pat])

        huber = HuberRegressor(alpha=1.0, epsilon=1.35)
        huber.fit(X_tr_scaled, patient_ph[tr_pat])
        preds_te = huber.predict(X_te_scaled)
        ph_oof_preds[te_pat] = preds_te

        # Save joblib
        joblib.dump(huber, os.path.join(MODEL_DIR, f"huber_fold_{f_idx}.joblib"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_fold_{f_idx}.joblib"))

        ensemble_params["folds"].append({
            "fold": f_idx,
            "coef": huber.coef_.tolist(),
            "intercept": float(huber.intercept_),
            "scale_mean": scaler.mean_.tolist(),
            "scale_std": scaler.scale_.tolist(),
            "train_patients": len(tr_pat),
            "val_patients": len(te_pat)
        })

    auc_b = roc_auc_score(patient_labels, -ph_oof_preds)
    mae_b = mean_absolute_error(patient_ph, ph_oof_preds)
    print(f"Exported 5-Fold Model B: OOF AUROC = {auc_b:.4f}, MAE = {mae_b:.4f}")

    # 4. Fit Full-Cohort Production Model
    full_scaler = StandardScaler()
    X_full_scaled = full_scaler.fit_transform(F_patient)
    full_huber = HuberRegressor(alpha=1.0, epsilon=1.35)
    full_huber.fit(X_full_scaled, patient_ph)

    joblib.dump(full_huber, os.path.join(MODEL_DIR, "huber_full_cohort.joblib"))
    joblib.dump(full_scaler, os.path.join(MODEL_DIR, "scaler_full_cohort.joblib"))

    ensemble_params["full_cohort_model"] = {
        "coef": full_huber.coef_.tolist(),
        "intercept": float(full_huber.intercept_),
        "scale_mean": full_scaler.mean_.tolist(),
        "scale_std": full_scaler.scale_.tolist(),
        "alpha": 1.0,
        "epsilon": 1.35
    }

    with open(os.path.join(MODEL_DIR, "model_ensemble_weights.json"), "w") as f:
        json.dump(ensemble_params, f, indent=2)

    # 5. Export Phase 4 Master Model Weights
    # Logistic regression on 19 descriptors + logit prior modulation
    p4_params = {"folds": [], "features": FEATURE_NAMES, "lambda_fusion": 0.60}
    for f_idx in range(5):
        te_pids = set([p for p in clean_pids if prot.assignment[p][0] == f_idx])
        tr_pat = np.array([i for i, p in enumerate(clean_pids) if p not in te_pids])
        te_pat = np.array([i for i, p in enumerate(clean_pids) if p in te_pids])

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(F_patient[tr_pat])
        clf = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=42)
        clf.fit(X_tr, patient_labels[tr_pat])

        joblib.dump(clf, os.path.join(PHASE4_DIR, f"clinical_lr_fold_{f_idx}.joblib"))
        joblib.dump(scaler, os.path.join(PHASE4_DIR, f"scaler_fold_{f_idx}.joblib"))

        p4_params["folds"].append({
            "fold": f_idx,
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
            "scale_mean": scaler.mean_.tolist(),
            "scale_std": scaler.scale_.tolist()
        })

    with open(os.path.join(PHASE4_DIR, "phase4_clinical_weights.json"), "w") as f:
        json.dump(p4_params, f, indent=2)

    print("All models successfully exported to models/continuous_clinical_huber/ and models/phase4_master/!")

if __name__ == "__main__":
    export_models()
