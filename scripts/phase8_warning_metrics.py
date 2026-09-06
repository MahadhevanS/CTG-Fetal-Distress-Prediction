"""
Phase 8: Warning-Time Metrics & Detection-by-Horizon Analysis.

Calculates:
1. Patient-level warning time distribution for acidemic patients (pH <= 7.15 and pH <= 7.05).
2. Proportion of patients identified at horizons >=60m, >=45m, >=30m, >=20m, >=10m.
3. Summary metrics saved to results/phase8_rolling/warning_times.csv.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

OUT_DIR = "results/phase8_rolling"
ROLLING_PATH = os.path.join(OUT_DIR, "rolling_predictions.csv")

def compute_warning_times():
    print("=== EXECUTING PHASE 8: WARNING-TIME METRICS ANALYSIS ===")
    df = pd.read_csv(ROLLING_PATH)

    # Threshold for high risk (predicted pH <= 7.15)
    risk_threshold = -7.15 # corresponding to acidemia_risk_score >= -7.15 (pred_ph <= 7.15)

    acidemic_pids = df[df["primary_label_715"] == 1]["patient_id"].unique()
    severe_pids = df[df["severe_label_705"] == 1]["patient_id"].unique()

    warning_times_715 = []
    warning_records = []

    for pid in acidemic_pids:
        df_p = df[df["patient_id"] == pid].sort_values("time_before_delivery_min", ascending=False)
        # Find earliest window (max time_before_delivery) where risk score >= risk_threshold
        df_alerts = df_p[df_p["acidemia_risk_score"] >= risk_threshold]
        if not df_alerts.empty:
            earliest_alert_time = df_alerts.iloc[0]["time_before_delivery_min"]
            detected = True
        else:
            earliest_alert_time = 0.0
            detected = False

        warning_times_715.append(earliest_alert_time)
        warning_records.append({
            "patient_id": pid,
            "true_ph": df_p.iloc[0]["true_ph"],
            "primary_label_715": 1,
            "severe_label_705": df_p.iloc[0]["severe_label_705"],
            "warning_time_min": earliest_alert_time,
            "detected_flag": detected
        })

    wt_arr = np.array(warning_times_715)
    wt_detected = wt_arr[wt_arr > 0]

    wt_summary = {
        "total_acidemic_patients": len(acidemic_pids),
        "total_detected": len(wt_detected),
        "detection_rate_pct": round(len(wt_detected) / len(acidemic_pids) * 100, 2),
        "median_warning_time_min": round(float(np.median(wt_detected)), 2) if len(wt_detected) > 0 else 0.0,
        "mean_warning_time_min": round(float(np.mean(wt_detected)), 2) if len(wt_detected) > 0 else 0.0,
        "iqr_warning_time_min": round(float(np.percentile(wt_detected, 75) - np.percentile(wt_detected, 25)), 2) if len(wt_detected) > 0 else 0.0,
        "pct_detected_ge_10m": round(float((wt_arr >= 10.0).sum() / len(acidemic_pids) * 100), 2),
        "pct_detected_ge_20m": round(float((wt_arr >= 20.0).sum() / len(acidemic_pids) * 100), 2),
        "pct_detected_ge_30m": round(float((wt_arr >= 30.0).sum() / len(acidemic_pids) * 100), 2),
        "pct_detected_ge_45m": round(float((wt_arr >= 45.0).sum() / len(acidemic_pids) * 100), 2),
        "pct_detected_ge_60m": round(float((wt_arr >= 60.0).sum() / len(acidemic_pids) * 100), 2)
    }

    print("--- Primary Acidemia (pH <= 7.15) Warning Times ---")
    for k, v in wt_summary.items():
        print(f"  {k}: {v}")

    df_wt_records = pd.DataFrame(warning_records)
    df_wt_records.to_csv(os.path.join(OUT_DIR, "warning_times.csv"), index=False)

    with open(os.path.join(OUT_DIR, "warning_summary.json"), "w") as f:
        json.dump(wt_summary, f, indent=2)

if __name__ == "__main__":
    compute_warning_times()
