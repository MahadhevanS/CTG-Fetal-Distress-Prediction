"""
Phase 9: Alert Policy Optimization & False Alarm Mitigation.

Compares 5 clinical alerting policies:
1. Single-window threshold crossing
2. 2-window consecutive persistence
3. 3-window consecutive persistence
4. Sustained trend alert (elevated risk + positive slope)
5. Hysteresis dual-threshold rule (enter @ 7.15, leave @ 7.20)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

OUT_DIR = "results/phase9_temporal"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
FOLDS_PATH = "data/processed_clinical/folds.json"

def run_alert_policy_optimization():
    print("=== EXECUTING PHASE 9: ALERT POLICY OPTIMIZATION ===")
    df = pd.read_csv(ROLLING_PATH)
    normal_pids = df[df["primary_label_715"] == 0]["patient_id"].unique()
    acidemic_pids = df[df["primary_label_715"] == 1]["patient_id"].unique()

    # Pre-calculate total normal monitoring hours
    tot_normal_hours = 0.0
    for pid in normal_pids:
        n_win = len(df[df["patient_id"] == pid])
        tot_normal_hours += (n_win * 2.5 + 17.5) / 60.0

    policies = ["Single Window", "2-Window Persistence", "3-Window Persistence", "Sustained Trend", "Hysteresis Dual-Thresh"]
    policy_results = []

    for pol in policies:
        false_alarms = 0
        normal_with_alert = 0
        true_detected = 0
        warning_times = []

        for pid in normal_pids:
            df_p = df[df["patient_id"] == pid].sort_values("time_before_delivery_min", ascending=False)
            risks = df_p["acidemia_risk_score"].values
            n_events = 0

            if pol == "Single Window":
                flags = (risks >= -7.15)
                n_events = int(np.sum(flags))
            elif pol == "2-Window Persistence":
                flags = (risks >= -7.15)
                n_events = sum(1 for i in range(len(flags)-1) if flags[i] and flags[i+1])
            elif pol == "3-Window Persistence":
                flags = (risks >= -7.15)
                n_events = sum(1 for i in range(len(flags)-2) if flags[i] and flags[i+1] and flags[i+2])
            elif pol == "Sustained Trend":
                flags = (risks >= -7.18)
                # Risk >= -7.18 AND positive slope over 2 windows
                n_events = sum(1 for i in range(len(risks)-1) if risks[i] >= -7.18 and risks[i] > risks[i+1])
            elif pol == "Hysteresis Dual-Thresh":
                # Enter when risk >= -7.15, remain active until risk < -7.22
                in_alert = False
                for r in risks:
                    if not in_alert and r >= -7.15:
                        in_alert = True
                        n_events += 1
                    elif in_alert and r < -7.22:
                        in_alert = False

            if n_events > 0:
                normal_with_alert += 1
            false_alarms += n_events

        # Acidemic cases detection
        for pid in acidemic_pids:
            df_p = df[df["patient_id"] == pid].sort_values("time_before_delivery_min", ascending=False)
            risks = df_p["acidemia_risk_score"].values
            times = df_p["time_before_delivery_min"].values
            detected = False
            earliest_t = 0.0

            if pol == "Single Window":
                for r, t in zip(risks, times):
                    if r >= -7.15:
                        detected = True
                        earliest_t = t
                        break
            elif pol == "2-Window Persistence":
                for i in range(len(risks)-1):
                    if risks[i] >= -7.15 and risks[i+1] >= -7.15:
                        detected = True
                        earliest_t = times[i]
                        break
            elif pol == "3-Window Persistence":
                for i in range(len(risks)-2):
                    if risks[i] >= -7.15 and risks[i+1] >= -7.15 and risks[i+2] >= -7.15:
                        detected = True
                        earliest_t = times[i]
                        break
            elif pol == "Sustained Trend":
                for i in range(len(risks)-1):
                    if risks[i] >= -7.18 and risks[i] > risks[i+1]:
                        detected = True
                        earliest_t = times[i]
                        break
            elif pol == "Hysteresis Dual-Thresh":
                for r, t in zip(risks, times):
                    if r >= -7.15:
                        detected = True
                        earliest_t = t
                        break

            if detected:
                true_detected += 1
                warning_times.append(earliest_t)

        sens = true_detected / len(acidemic_pids)
        spec = 1.0 - (normal_with_alert / len(normal_pids))
        fa_per_pt = false_alarms / len(normal_pids)
        fa_per_hr = false_alarms / tot_normal_hours

        policy_results.append({
            "policy_name": pol,
            "sensitivity": round(sens * 100, 2),
            "specificity": round(spec * 100, 2),
            "false_alert_rate_per_hour": round(fa_per_hr, 3),
            "false_alerts_per_patient": round(fa_per_pt, 2),
            "total_false_alarms": int(false_alarms),
            "median_warning_time_min": round(float(np.median(warning_times)), 1) if warning_times else 0.0
        })

    df_pol = pd.DataFrame(policy_results)
    df_pol.to_csv(os.path.join(OUT_DIR, "alert_policy_metrics.csv"), index=False)
    print(df_pol.to_string(index=False))

if __name__ == "__main__":
    run_alert_policy_optimization()
