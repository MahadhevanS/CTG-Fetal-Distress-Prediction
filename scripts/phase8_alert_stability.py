"""
Phase 8: Alert Stability & False-Alarm Burden Evaluation.

Evaluates:
1. Alert persistence: Single-window vs 2-consecutive-window confirmation rule.
2. False alarm rate and burden on normal patients (pH > 7.15, N=437).
3. False alerts per monitoring hour and per patient.
4. Summary metrics saved to results/phase8_rolling/alert_metrics.csv.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

OUT_DIR = "results/phase8_rolling"
ROLLING_PATH = os.path.join(OUT_DIR, "rolling_predictions.csv")

def evaluate_alert_stability():
    print("=== EXECUTING PHASE 8: ALERT STABILITY & FALSE-ALARM BURDEN ===")
    df = pd.read_csv(ROLLING_PATH)

    risk_threshold = -7.15 # predicted pH <= 7.15
    normal_pids = df[df["primary_label_715"] == 0]["patient_id"].unique()
    acidemic_pids = df[df["primary_label_715"] == 1]["patient_id"].unique()

    # Calculate metrics for both single-alert and 2-consecutive-alert rules
    def run_rule(min_consecutive=1):
        total_false_alerts = 0
        patients_with_false_alerts = 0
        patient_false_counts = []
        total_monitoring_hours_normal = 0.0

        # False alarms on normal patients
        for pid in normal_pids:
            df_p = df[df["patient_id"] == pid].sort_values("time_before_delivery_min", ascending=False)
            is_alert = (df_p["acidemia_risk_score"] >= risk_threshold).values
            n_windows = len(df_p)
            # Duration in hours = (n_windows * 2.5 min stride + 17.5 min) / 60
            dur_hours = (n_windows * 2.5 + 17.5) / 60.0
            total_monitoring_hours_normal += dur_hours

            # Count alert events
            n_events = 0
            if min_consecutive == 1:
                n_events = int(np.sum(is_alert))
            else:
                # 2 consecutive windows
                for idx in range(len(is_alert) - 1):
                    if is_alert[idx] and is_alert[idx + 1]:
                        n_events += 1

            if n_events > 0:
                patients_with_false_alerts += 1
            total_false_alerts += n_events
            patient_false_counts.append(n_events)

        # True detection rate on acidemic patients
        true_detected = 0
        for pid in acidemic_pids:
            df_p = df[df["patient_id"] == pid].sort_values("time_before_delivery_min", ascending=False)
            is_alert = (df_p["acidemia_risk_score"] >= risk_threshold).values
            detected = False
            if min_consecutive == 1:
                detected = bool(np.any(is_alert))
            else:
                for idx in range(len(is_alert) - 1):
                    if is_alert[idx] and is_alert[idx + 1]:
                        detected = True
                        break
            if detected:
                true_detected += 1

        sens = true_detected / len(acidemic_pids)
        spec = 1.0 - (patients_with_false_alerts / len(normal_pids))
        fa_per_patient = total_false_alerts / len(normal_pids)
        fa_per_hour = total_false_alerts / total_monitoring_hours_normal if total_monitoring_hours_normal > 0 else 0.0

        return {
            "rule": f"{min_consecutive}-Window Alert Rule",
            "sensitivity": round(sens * 100, 2),
            "specificity": round(spec * 100, 2),
            "pct_normal_with_false_alert": round(patients_with_false_alerts / len(normal_pids) * 100, 2),
            "total_false_alerts": int(total_false_alerts),
            "false_alerts_per_patient": round(fa_per_patient, 2),
            "false_alerts_per_monitoring_hour": round(fa_per_hour, 2),
            "median_false_alerts_normal": float(np.median(patient_false_counts)),
            "total_normal_monitoring_hours": round(total_monitoring_hours_normal, 1)
        }

    res_single = run_rule(min_consecutive=1)
    res_persistent = run_rule(min_consecutive=2)

    df_alert_metrics = pd.DataFrame([res_single, res_persistent])
    df_alert_metrics.to_csv(os.path.join(OUT_DIR, "alert_metrics.csv"), index=False)

    print("--- Alert Stability & False Alarm Burden Results ---")
    print(df_alert_metrics.to_string(index=False))

    with open(os.path.join(OUT_DIR, "alert_summary.json"), "w") as f:
        json.dump({"single_alert": res_single, "persistent_alert_2win": res_persistent}, f, indent=2)

if __name__ == "__main__":
    evaluate_alert_stability()
