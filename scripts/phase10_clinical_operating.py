"""
Phase 10 Work Package 10C: Clinical Operating Characteristics & Warning Time Analysis.

Evaluates:
- 5 Predefined Alert Policies:
  - Policy 1: Threshold (p_t >= tau)
  - Policy 2: Persistence (p_t >= tau and p_{t-1} >= tau)
  - Policy 3: State-based (Two consecutive S_t >= 3)
  - Policy 4: State progression (S_t >= 3 and V_t >= 0)
  - Policy 5: Hybrid (p_t >= 0.85*tau + S_t >= 2 + P_t >= 2)
- Clinical metrics: Sensitivity, Specificity, PPV, NPV, FPR, False Alerts / Monitoring Hour, Alerts / Patient.
- Warning-time distributions: Median, IQR, Mean, Min, Max, proportion warned >=10m, >=20m, >=30m.

Outputs:
- results/phase10_final/clinical_operating_points.csv
- results/phase10_final/warning_time_analysis.csv
"""

import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "results/phase10_final"
PRED_PATH = os.path.join(OUT_DIR, "final_predictions.csv")
FOLDS_PATH = "data/processed_clinical/folds.json"
os.makedirs(OUT_DIR, exist_ok=True)

def run_clinical_operating_analysis():
    print("=== EXECUTING WORK PACKAGE 10C: CLINICAL OPERATING ANALYSIS ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_pred = pd.read_csv(PRED_PATH)
    df_pred["patient_id"] = df_pred["patient_id"].astype(str)
    
    p8_prob = df_pred["Model_A_Snapshot_Baseline"].values
    state_arr = df_pred["research_state"].values
    vel_arr = df_pred["vel_1step"].values
    persist_arr = df_pred["state_persist"].values
    t_del = df_pred["time_before_delivery_min"].values
    
    tau = float(np.percentile(p8_prob, 90.0))
    print(f"Locked Operational Threshold tau (90th percentile): {tau:.4f}")
    
    n_win = len(df_pred)
    alert_masks = {
        "Policy 1 (Single Window Threshold: p >= tau)": np.zeros(n_win, dtype=bool),
        "Policy 2 (Two Consecutive Windows: p >= tau)": np.zeros(n_win, dtype=bool),
        "Policy 3 (State-Based: Two Consecutive S >= 3)": np.zeros(n_win, dtype=bool),
        "Policy 4 (State Progression: S >= 3 and V >= 0)": np.zeros(n_win, dtype=bool),
        "Policy 5 (Hybrid: p >= 0.85*tau + S >= 2 + P >= 2)": np.zeros(n_win, dtype=bool)
    }
    
    for pid in clean_pids:
        idx = np.where(df_pred["patient_id"].values == pid)[0]
        sort_order = np.argsort(-t_del[idx]) # chronological
        idx_sorted = idx[sort_order]
        
        p_sub = p8_prob[idx_sorted]
        s_sub = state_arr[idx_sorted]
        v_sub = vel_arr[idx_sorted]
        per_sub = persist_arr[idx_sorted]
        
        # Policy 1
        m1 = p_sub >= tau
        # Policy 2
        m2 = np.zeros(len(idx_sorted), dtype=bool)
        for i in range(1, len(idx_sorted)):
            if p_sub[i] >= tau and p_sub[i-1] >= tau:
                m2[i] = True
        # Policy 3
        m3 = np.zeros(len(idx_sorted), dtype=bool)
        for i in range(1, len(idx_sorted)):
            if s_sub[i] >= 3 and s_sub[i-1] >= 3:
                m3[i] = True
        # Policy 4
        m4 = (s_sub >= 3) & (v_sub >= 0)
        # Policy 5
        m5 = (p_sub >= (0.85 * tau)) & (s_sub >= 2) & (per_sub >= 2)
        
        alert_masks["Policy 1 (Single Window Threshold: p >= tau)"][idx_sorted] = m1
        alert_masks["Policy 2 (Two Consecutive Windows: p >= tau)"][idx_sorted] = m2
        alert_masks["Policy 3 (State-Based: Two Consecutive S >= 3)"][idx_sorted] = m3
        alert_masks["Policy 4 (State Progression: S >= 3 and V >= 0)"][idx_sorted] = m4
        alert_masks["Policy 5 (Hybrid: p >= 0.85*tau + S >= 2 + P >= 2)"][idx_sorted] = m5

    y_pat = np.array([df_pred[df_pred["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    n_pos_total = int((y_pat == 1).sum()) # 110
    n_neg_total = int((y_pat == 0).sum()) # 437
    
    stride_hr = 2.5 / 60.0
    neg_pids = set([clean_pids[i] for i in range(len(clean_pids)) if y_pat[i] == 0])
    df_neg = df_pred[df_pred["patient_id"].isin(neg_pids)]
    total_neg_hours = len(df_neg) * stride_hr
    
    op_rows = []
    warn_rows = []
    
    for pol_name, mask in alert_masks.items():
        df_pred["alert_flag"] = mask
        
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
            "policy": pol_name,
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
            "median_lead_time_min": round(med_lead, 1) if not np.isnan(med_lead) else "N/A",
            "pct_warned_ge30m": round(pct_ge30, 1) if not np.isnan(pct_ge30) else "N/A"
        })
        
        warn_rows.append({
            "policy": pol_name,
            "total_patients_alerted": n_alerted,
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
    df_warn.to_csv(os.path.join(OUT_DIR, "warning_time_analysis.csv"), index=False)
    
    print("\nSaved clinical_operating_points.csv and warning_time_analysis.csv")
    print(df_op.to_string(index=False))
    print("\nWork Package 10C Complete.")

if __name__ == "__main__":
    run_clinical_operating_analysis()
