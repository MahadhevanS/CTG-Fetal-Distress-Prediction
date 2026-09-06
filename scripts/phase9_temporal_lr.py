"""
Phase 9 Model 1: Temporal Logistic Regression with ElasticNet / L1 Regularization.

Evaluates regularized linear model on 104-dimensional temporal + delta + trend feature matrix
under strictly patient-grouped 5-fold CV across warning horizons (>=60m, >=45m, >=30m, >=20m, >=10m).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.protocol import Protocol

OUT_DIR = "results/phase9_temporal"
FOLDS_PATH = "data/processed_clinical/folds.json"
P2_PATH = "data/phase1_candidates/p2_dataset.pt"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
FEATURES_PATH = os.path.join(OUT_DIR, "temporal_features.npz")

def run_temporal_lr():
    print("=== EXECUTING PHASE 9 MODEL 1: TEMPORAL LOGISTIC REGRESSION ===")

    # 1. Load Data
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p2_data = torch_data = np.load(FEATURES_PATH)
    X_temp = p2_data["X_temporal"] # (8517, 104)

    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    y_patient = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["primary_label_715"] for p in clean_pids])
    y_patient_705 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["severe_label_705"] for p in clean_pids])

    # 2. Evaluate across horizons
    horizons = [60, 45, 30, 20, 10, 0]
    results_by_horizon = {}

    for h in horizons:
        oof_scores = np.zeros(len(clean_pids), dtype=np.float32)

        # Select window index for each patient corresponding to horizon h
        pat_win_indices = []
        for pid in clean_pids:
            df_p = df_rolling[df_rolling["patient_id"] == pid].sort_values("time_before_delivery_min")
            df_eligible = df_p[df_p["time_before_delivery_min"] >= h]
            if not df_eligible.empty:
                chosen_idx = int(df_eligible.iloc[0]["window_index"])
            else:
                chosen_idx = int(df_p.iloc[-1]["window_index"])
            pat_win_indices.append(chosen_idx)

        X_h = X_temp[pat_win_indices] # (547, 104)

        for f_idx in range(5):
            te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
            tr_pat = np.array([i for i, p in enumerate(clean_pids) if p not in te_pids])
            te_pat = np.array([i for i, p in enumerate(clean_pids) if p in te_pids])

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_h[tr_pat])
            X_te = scaler.transform(X_h[te_pat])

            clf = LogisticRegression(C=0.05, penalty='l1', solver='liblinear', class_weight='balanced', random_state=42)
            clf.fit(X_tr, y_patient[tr_pat])
            oof_scores[te_pat] = clf.predict_proba(X_te)[:, 1]

        auc_715 = roc_auc_score(y_patient, oof_scores)
        auprc_715 = average_precision_score(y_patient, oof_scores)
        auc_705 = roc_auc_score(y_patient_705, oof_scores)

        results_by_horizon[f"horizon_{h}m"] = {
            "horizon_min": h,
            "auroc_715": round(float(auc_715), 4),
            "auprc_715": round(float(auprc_715), 4),
            "auroc_severe_705": round(float(auc_705), 4)
        }
        print(f"Temporal LR Horizon >={h:02d}m: AUROC (pH<=7.15) = {auc_715:.4f}, AUPRC = {auprc_715:.4f} | Severe (<=7.05) AUROC = {auc_705:.4f}")

    with open(os.path.join(OUT_DIR, "temporal_lr_metrics.json"), "w") as f:
        json.dump(results_by_horizon, f, indent=2)

if __name__ == "__main__":
    run_temporal_lr()
