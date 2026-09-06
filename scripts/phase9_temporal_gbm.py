"""
Phase 9 Model 2: Temporal Gradient-Boosted Decision Trees (HistGradientBoosting / LightGBM).

Captures non-linear feature interactions between current CTG state and rate-of-deterioration
under patient-grouped 5-fold CV across warning horizons.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

OUT_DIR = "results/phase9_temporal"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
FEATURES_PATH = os.path.join(OUT_DIR, "temporal_features.npz")

def run_temporal_gbm():
    print("=== EXECUTING PHASE 9 MODEL 2: TEMPORAL GRADIENT BOOSTING ===")

    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p2_data = np.load(FEATURES_PATH)
    X_temp = p2_data["X_temporal"]

    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    y_patient = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["primary_label_715"] for p in clean_pids])
    y_patient_705 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["severe_label_705"] for p in clean_pids])

    horizons = [60, 45, 30, 20, 10, 0]
    results_by_horizon = {}

    for h in horizons:
        oof_scores = np.zeros(len(clean_pids), dtype=np.float32)

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

            gbm = HistGradientBoostingClassifier(
                max_iter=50,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                learning_rate=0.05,
                class_weight='balanced',
                random_state=42
            )
            gbm.fit(X_h[tr_pat], y_patient[tr_pat])
            oof_scores[te_pat] = gbm.predict_proba(X_h[te_pat])[:, 1]

        auc_715 = roc_auc_score(y_patient, oof_scores)
        auprc_715 = average_precision_score(y_patient, oof_scores)
        auc_705 = roc_auc_score(y_patient_705, oof_scores)

        results_by_horizon[f"horizon_{h}m"] = {
            "horizon_min": h,
            "auroc_715": round(float(auc_715), 4),
            "auprc_715": round(float(auprc_715), 4),
            "auroc_severe_705": round(float(auc_705), 4)
        }
        print(f"Temporal GBM Horizon >={h:02d}m: AUROC (pH<=7.15) = {auc_715:.4f}, AUPRC = {auprc_715:.4f} | Severe (<=7.05) AUROC = {auc_705:.4f}")

    with open(os.path.join(OUT_DIR, "temporal_gbm_metrics.json"), "w") as f:
        json.dump(results_by_horizon, f, indent=2)

if __name__ == "__main__":
    run_temporal_gbm()
