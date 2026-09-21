"""
Phase 11: Clinical Decision Operating Points & Warning Lead-Time Benchmark.

Evaluates:
- Clinical decision characteristics across all 6 benchmark models (P1 to P6):
  - High-Specificity operating threshold (anchored at 90th percentile)
  - Sensitivity, Specificity, PPV, NPV, FPR, False Alerts / Monitoring Hour, Alerts / Patient
- Warning lead-time distribution analysis:
  - Median, IQR, Mean, Min, Max
  - Proportions alerted >=10m, >=20m, >=30m

Outputs:
- results/phase11_priorart_benchmark/clinical_operating_points.csv
- results/phase11_priorart_benchmark/warning_time_results.csv
"""

import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "results/phase11_priorart_benchmark"
PRED_PATH = os.path.join(OUT_DIR, "model_predictions.csv")
FOLDS_PATH = "data/processed_clinical/folds.json"
os.makedirs(OUT_DIR, exist_ok=True)

def run_clinical_benchmark():
    print("=== EXECUTING PHASE 11: CLINICAL OPERATING & LEAD-TIME BENCHMARK ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_pred = pd.read_csv(PRED_PATH)
    df_pred["patient_id"] = df_pred["patient_id"].astype(str)
    
    patient_ids = df_pred["patient_id"].values
    t_del = df_pred["time_before_delivery_min"].values
    
    df_pat_labels = df_pred.groupby("patient_id")[["primary_label_715", "severe_label_705"]].first().reset_index()
    y_pat = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    
    models = [
        ("P1 (Compact CTG Baseline)", "P1_Compact_CTG"),
        ("P2 (Sequential / Event Baseline)", "P2_Sequential_Event"),
        ("P3 (Locked Snapshot Baseline)", "P3_Snapshot_Baseline"),
        ("P4 (Snapshot + State)", "P4_Snapshot_plus_State"),
        ("P5 (Snapshot + Trajectory)", "P5_Snapshot_plus_Trajectory"),
        ("P6 (Full Proposed Framework)", "P6_Full_Physiology_System")
    ]
    
    stride_hr = 2.5 / 60.0
    neg_pids = set([clean_pids[i] for i in range(len(clean_pids)) if y_pat[i] == 0])
    df_neg = df_pred[df_pred["patient_id"].isin(neg_pids)]
    total_neg_hours = len(df_neg) * stride_hr
    
    op_rows = []
    warn_rows = []
    
    for display_name, col_name in models:
        scores = df_pred[col_name].values
        tau = float(np.percentile(scores, 90.0))
        
        # Binary threshold alert flag per window
        alert_mask = scores >= tau
        df_pred["alert_flag"] = alert_mask
        
        pat_alerted = []
        lead_times = []
        alert_episodes_neg = 0
        
        for pid in clean_pids:
            df_p = df_pred[df_pred["patient_id"] == pid].sort_values("time_before_delivery_min", ascending=False)
            is_alerted = bool(df_p["alert_flag"].any())
            pat_alerted.append(is_alerted)
            
            if is_alerted:
                first_alert_row = df_p[df_p["alert_flag"]].iloc[0]
                lead_times.append(first_alert_row["time_before_delivery_min"])
                
            if pid in neg_pids:
                flags = df_p["alert_flag"].values
                diffs = np.diff(flags.astype(int))
                episodes = int(np.sum(diffs == 1)) + (1 if flags[0] else 0)
                alert_episodes_neg += episodes
                
        pat_alerted = np.array(pat_alerted)
        tp = int(np.sum((y_pat == 1) & pat_alerted))
        fn = int(np.sum((y_pat == 1) & ~pat_alerted))
        tn = int(np.sum((y_pat == 0) & ~pat_alerted))
        fp = int(np.sum((y_pat == 0) & pat_alerted))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        far_per_hour = alert_episodes_neg / total_neg_hours if total_neg_hours > 0 else 0.0
        alerts_per_patient = (tp + fp) / len(clean_pids)
        
        arr_leads = np.array(lead_times)
        n_alerted = len(arr_leads)
        med_lead = float(np.median(arr_leads)) if n_alerted > 0 else np.nan
        q25_lead = float(np.percentile(arr_leads, 25)) if n_alerted > 0 else np.nan
        q75_lead = float(np.percentile(arr_leads, 75)) if n_alerted > 0 else np.nan
        mean_lead = float(np.mean(arr_leads)) if n_alerted > 0 else np.nan
        min_lead = float(np.min(arr_leads)) if n_alerted > 0 else np.nan
        max_lead = float(np.max(arr_leads)) if n_alerted > 0 else np.nan
        
        pct_ge30 = float(np.mean(arr_leads >= 30) * 100) if n_alerted > 0 else 0.0
        pct_ge20 = float(np.mean(arr_leads >= 20) * 100) if n_alerted > 0 else 0.0
        pct_ge10 = float(np.mean(arr_leads >= 10) * 100) if n_alerted > 0 else 0.0
        
        op_rows.append({
            "model_code": col_name[:2],
            "model_name": display_name,
            "threshold_tau": round(tau, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "sensitivity_pct": round(sens * 100, 2),
            "specificity_pct": round(spec * 100, 2),
            "ppv_pct": round(ppv * 100, 2),
            "npv_pct": round(npv * 100, 2),
            "fpr_pct": round(fpr * 100, 2),
            "false_alerts_per_hour": round(far_per_hour, 4),
            "alerts_per_patient": round(alerts_per_patient, 3),
            "median_lead_time_min": round(med_lead, 1) if not np.isnan(med_lead) else "N/A"
        })
        
        warn_rows.append({
            "model_code": col_name[:2],
            "model_name": display_name,
            "patients_alerted": n_alerted,
            "median_lead_min": round(med_lead, 1) if not np.isnan(med_lead) else "N/A",
            "iqr_lead_min": f"{round(q25_lead, 1)} - {round(q75_lead, 1)}" if not np.isnan(q25_lead) else "N/A",
            "mean_lead_min": round(mean_lead, 1) if not np.isnan(mean_lead) else "N/A",
            "min_lead_min": round(min_lead, 1) if not np.isnan(min_lead) else "N/A",
            "max_lead_min": round(max_lead, 1) if not np.isnan(max_lead) else "N/A",
            "pct_warned_ge10m": round(pct_ge10, 1),
            "pct_warned_ge20m": round(pct_ge20, 1),
            "pct_warned_ge30m": round(pct_ge30, 1)
        })
        
    df_op = pd.DataFrame(op_rows)
    df_op.to_csv(os.path.join(OUT_DIR, "clinical_operating_points.csv"), index=False)
    
    df_warn = pd.DataFrame(warn_rows)
    df_warn.to_csv(os.path.join(OUT_DIR, "warning_time_results.csv"), index=False)
    
    print("\nSaved clinical_operating_points.csv and warning_time_results.csv")
    print(df_op[["model_code", "model_name", "sensitivity_pct", "specificity_pct", "ppv_pct", "false_alerts_per_hour", "median_lead_time_min"]].to_string(index=False))

if __name__ == "__main__":
    run_clinical_benchmark()
