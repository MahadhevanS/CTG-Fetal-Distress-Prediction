"""
Phase 9 Model 3: Prediction-Trajectory Dynamics & Exponential Moving Average (EWMA) Model.

Tracks the causal momentum, slope, and acceleration of the baseline Huber risk score:
1. S_t = -pH_hat(t)
2. Exponentially weighted moving average (EWMA) to filter momentary noise
3. Evaluates trajectory momentum across warning horizons.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

OUT_DIR = "results/phase9_temporal"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
FOLDS_PATH = "data/processed_clinical/folds.json"

def run_prediction_trajectory():
    print("=== EXECUTING PHASE 9 MODEL 3: PREDICTION TRAJECTORY & MOMENTUM ===")
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    y_patient = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["primary_label_715"] for p in clean_pids])
    y_patient_705 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["severe_label_705"] for p in clean_pids])

    # Compute EWMA risk trajectory causally for each patient (alpha = 0.4)
    df_rolling_sorted = df_rolling.sort_values(["patient_id", "start_sample"]).copy()
    ewma_scores = []
    
    for pid in clean_pids:
        df_p = df_rolling_sorted[df_rolling_sorted["patient_id"] == pid]
        raw_risks = df_p["acidemia_risk_score"].values
        # Causal EWMA: S_ewma(t) = alpha * S(t) + (1-alpha) * S_ewma(t-1)
        ewma = np.zeros(len(raw_risks), dtype=np.float32)
        alpha = 0.4
        cur = raw_risks[0]
        for k in range(len(raw_risks)):
            cur = alpha * raw_risks[k] + (1.0 - alpha) * cur
            ewma[k] = cur
        ewma_scores.extend(ewma)

    df_rolling_sorted["risk_ewma"] = ewma_scores

    horizons = [60, 45, 30, 20, 10, 0]
    results_by_horizon = {}

    for h in horizons:
        # Pick window for each patient at horizon h
        patient_ewma_scores = []
        for pid in clean_pids:
            df_p = df_rolling_sorted[df_rolling_sorted["patient_id"] == pid].sort_values("time_before_delivery_min")
            df_eligible = df_p[df_p["time_before_delivery_min"] >= h]
            if not df_eligible.empty:
                val = float(df_eligible.iloc[0]["risk_ewma"])
            else:
                val = float(df_p.iloc[-1]["risk_ewma"])
            patient_ewma_scores.append(val)

        patient_ewma_scores = np.array(patient_ewma_scores)

        auc_715 = roc_auc_score(y_patient, patient_ewma_scores)
        auprc_715 = average_precision_score(y_patient, patient_ewma_scores)
        auc_705 = roc_auc_score(y_patient_705, patient_ewma_scores)

        results_by_horizon[f"horizon_{h}m"] = {
            "horizon_min": h,
            "auroc_715": round(float(auc_715), 4),
            "auprc_715": round(float(auprc_715), 4),
            "auroc_severe_705": round(float(auc_705), 4)
        }
        print(f"Trajectory EWMA Horizon >={h:02d}m: AUROC (pH<=7.15) = {auc_715:.4f}, AUPRC = {auprc_715:.4f} | Severe (<=7.05) AUROC = {auc_705:.4f}")

    with open(os.path.join(OUT_DIR, "prediction_trajectory_metrics.json"), "w") as f:
        json.dump(results_by_horizon, f, indent=2)

if __name__ == "__main__":
    run_prediction_trajectory()
