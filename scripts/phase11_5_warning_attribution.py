"""
Phase 11.5 — Experiments 11.5-G & 11.5-H: Warning-Time Attribution & Threshold Robustness.

Objectives:
- Experiment 11.5-G: Attribute the longer P6 warning lead time (17.5 min vs 10-15 min in baselines)
  to examine whether it reflects genuine earlier discrimination or threshold behavior.
- Experiment 11.5-H: Test threshold robustness by evaluating all models (P1-P6) at matched
  operational criteria:
  1. Matched False Alert Rates (FAR = 0.25, 0.50, 0.75, 1.00 false alerts / hour).
  2. Matched Sensitivity levels (Sens = 50%, 60%, 70%, 80%).

Evaluation Protocol:
- Causal rolling predictions across all 547 CTU-UHB patients (8,517 windows, dt = 2.5 min).
- First alert lead time: T_lead = T_delivery - T_first_alert (minutes).
- False alerts: Alert windows occurring in true-negative patients (pH > 7.15). Total normal monitoring hours = sum of duration / 60.

Outputs:
- results/phase11_5_advantage_attribution/warning_time_attribution.csv
- results/phase11_5_advantage_attribution/threshold_robustness.csv
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

OUT_DIR = "results/phase11_5_advantage_attribution"
FOLDS_PATH = "data/processed_clinical/folds.json"
PRED_PATH = "results/phase11_priorart_benchmark/model_predictions.csv"
os.makedirs(OUT_DIR, exist_ok=True)

def run_warning_attribution():
    print("=== EXECUTING EXPERIMENTS 11.5-G & 11.5-H: WARNING ATTRIBUTION & THRESHOLD ROBUSTNESS ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_pred = pd.read_csv(PRED_PATH)
    df_pred["patient_id"] = df_pred["patient_id"].astype(str)
    
    patient_ids = df_pred["patient_id"].values
    t_del = df_pred["time_before_delivery_min"].values
    
    df_pat_labels = df_pred.groupby("patient_id")[["primary_label_715", "severe_label_705"]].first().reset_index()
    y_pat_715 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    
    pat_indices_map = {pid: np.where(patient_ids == str(pid))[0] for pid in clean_pids}
    
    # Calculate total normal monitoring hours across cohort
    # Each window represents 2.5 min of stride
    neg_pids = [p for p in clean_pids if df_pat_labels[df_pat_labels["patient_id"] == p]["primary_label_715"].iloc[0] == 0]
    total_neg_windows = sum([len(pat_indices_map[p]) for p in neg_pids])
    total_neg_hours = total_neg_windows * (2.5 / 60.0) # Monitoring hours in normal patients
    
    models = {
        "P1": ("P1_Compact_CTG", "P1 (Compact CTG Baseline)"),
        "P2": ("P2_Sequential_Event", "P2 (Sequential/Event Baseline)"),
        "P3": ("P3_Snapshot_Baseline", "P3 (Locked Snapshot Baseline)"),
        "P4": ("P4_Snapshot_plus_State", "P4 (Snapshot + State)"),
        "P5": ("P5_Snapshot_plus_Trajectory", "P5 (Snapshot + Trajectory)"),
        "P6": ("P6_Full_Physiology_System", "P6 (Full Proposed Framework)")
    }
    
    # -------------------------------------------------------------
    # Helper to compute operational metrics for a model at threshold th
    # -------------------------------------------------------------
    def compute_op_metrics(col_name, th):
        raw_scores = df_pred[col_name].values
        
        pat_alerted = []
        pat_lead_times = []
        total_false_alert_windows = 0
        
        for pid in clean_pids:
            idx = pat_indices_map[pid]
            p_scores = raw_scores[idx]
            p_tdel = t_del[idx]
            is_pos = df_pat_labels[df_pat_labels["patient_id"] == pid]["primary_label_715"].iloc[0] == 1
            
            alert_mask = (p_scores >= th)
            if np.any(alert_mask):
                pat_alerted.append(1)
                first_alert_idx = np.where(alert_mask)[0][0]
                lead_time = p_tdel[first_alert_idx]
                if is_pos:
                    pat_lead_times.append(lead_time)
            else:
                pat_alerted.append(0)
                
            if not is_pos:
                total_false_alert_windows += np.sum(alert_mask)
                
        pat_alerted = np.array(pat_alerted)
        cm = confusion_matrix(y_pat_715, pat_alerted, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        far_per_hour = total_false_alert_windows / total_neg_hours if total_neg_hours > 0 else 0.0
        
        if len(pat_lead_times) > 0:
            med_lead = float(np.median(pat_lead_times))
            iqr_lead = float(np.percentile(pat_lead_times, 75) - np.percentile(pat_lead_times, 25))
            mean_lead = float(np.mean(pat_lead_times))
            pct_10 = float(np.mean(np.array(pat_lead_times) >= 10.0) * 100.0)
            pct_20 = float(np.mean(np.array(pat_lead_times) >= 20.0) * 100.0)
            pct_30 = float(np.mean(np.array(pat_lead_times) >= 30.0) * 100.0)
        else:
            med_lead, iqr_lead, mean_lead, pct_10, pct_20, pct_30 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        return {
            "threshold": th,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "far_per_hour": far_per_hour,
            "n_warned_positives": len(pat_lead_times),
            "median_lead_min": med_lead,
            "iqr_lead_min": iqr_lead,
            "mean_lead_min": mean_lead,
            "pct_warned_ge10m": pct_10,
            "pct_warned_ge20m": pct_20,
            "pct_warned_ge30m": pct_30
        }

    # -------------------------------------------------------------
    # Experiment 11.5-G: Locked Reference Operating Points
    # -------------------------------------------------------------
    # Find operating point achieving standard sensitivity ~65-70% or locked FAR ~0.65
    exp_g_rows = []
    
    for m_code, (col_name, m_display) in models.items():
        # Sweep thresholds to find the threshold matching P6 locked FAR or standard sensitivity
        scores = df_pred[col_name].values
        th_grid = np.linspace(np.percentile(scores, 5), np.percentile(scores, 95), 100)
        
        best_op = None
        min_far_diff = 999.0
        
        # Match FAR to ~0.65/hr
        for th in th_grid:
            m_res = compute_op_metrics(col_name, th)
            far_diff = abs(m_res["far_per_hour"] - 0.654)
            if far_diff < min_far_diff:
                min_far_diff = far_diff
                best_op = m_res
                
        exp_g_rows.append({
            "model_code": m_code,
            "model_name": m_display,
            "operating_point_type": "Locked FAR Match (~0.65 / hr)",
            "operating_threshold": round(float(best_op["threshold"]), 4),
            "sensitivity": round(float(best_op["sensitivity"]), 4),
            "specificity": round(float(best_op["specificity"]), 4),
            "ppv": round(float(best_op["ppv"]), 4),
            "npv": round(float(best_op["npv"]), 4),
            "false_alerts_per_hour": round(float(best_op["far_per_hour"]), 4),
            "median_warning_min": round(float(best_op["median_lead_min"]), 2),
            "iqr_warning_min": round(float(best_op["iqr_lead_min"]), 2),
            "mean_warning_min": round(float(best_op["mean_lead_min"]), 2),
            "pct_warned_ge10m": round(float(best_op["pct_warned_ge10m"]), 2),
            "pct_warned_ge20m": round(float(best_op["pct_warned_ge20m"]), 2),
            "pct_warned_ge30m": round(float(best_op["pct_warned_ge30m"]), 2)
        })
        
    df_exp_g = pd.DataFrame(exp_g_rows)
    df_exp_g.to_csv(os.path.join(OUT_DIR, "warning_time_attribution.csv"), index=False)
    print("Saved warning_time_attribution.csv")
    
    # -------------------------------------------------------------
    # Experiment 11.5-H: Threshold-Robust Operational Comparison
    # -------------------------------------------------------------
    exp_h_rows = []
    
    far_targets = [0.25, 0.50, 0.75, 1.00]
    sens_targets = [0.50, 0.60, 0.70, 0.80]
    
    for m_code, (col_name, m_display) in models.items():
        scores = df_pred[col_name].values
        th_grid = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), 100)
        grid_metrics = [compute_op_metrics(col_name, th) for th in th_grid]
        
        # 1. Matched FAR targets
        for target_far in far_targets:
            best_m = min(grid_metrics, key=lambda x: abs(x["far_per_hour"] - target_far))
            exp_h_rows.append({
                "model_code": m_code,
                "model_name": m_display,
                "matching_criterion": f"Matched FAR = {target_far:.2f}/hr",
                "target_metric": "FAR",
                "target_value": target_far,
                "actual_far_per_hour": round(float(best_m["far_per_hour"]), 4),
                "actual_sensitivity": round(float(best_m["sensitivity"]), 4),
                "actual_specificity": round(float(best_m["specificity"]), 4),
                "ppv": round(float(best_m["ppv"]), 4),
                "npv": round(float(best_m["npv"]), 4),
                "threshold": round(float(best_m["threshold"]), 4),
                "median_warning_min": round(float(best_m["median_lead_min"]), 2),
                "iqr_warning_min": round(float(best_m["iqr_lead_min"]), 2),
                "pct_warned_ge20m": round(float(best_m["pct_warned_ge20m"]), 2)
            })
            
        # 2. Matched Sensitivity targets
        for target_sens in sens_targets:
            best_m = min(grid_metrics, key=lambda x: abs(x["sensitivity"] - target_sens))
            exp_h_rows.append({
                "model_code": m_code,
                "model_name": m_display,
                "matching_criterion": f"Matched Sensitivity = {int(target_sens*100)}%",
                "target_metric": "Sensitivity",
                "target_value": target_sens,
                "actual_far_per_hour": round(float(best_m["far_per_hour"]), 4),
                "actual_sensitivity": round(float(best_m["sensitivity"]), 4),
                "actual_specificity": round(float(best_m["specificity"]), 4),
                "ppv": round(float(best_m["ppv"]), 4),
                "npv": round(float(best_m["npv"]), 4),
                "threshold": round(float(best_m["threshold"]), 4),
                "median_warning_min": round(float(best_m["median_lead_min"]), 2),
                "iqr_warning_min": round(float(best_m["iqr_lead_min"]), 2),
                "pct_warned_ge20m": round(float(best_m["pct_warned_ge20m"]), 2)
            })
            
    df_exp_h = pd.DataFrame(exp_h_rows)
    df_exp_h.to_csv(os.path.join(OUT_DIR, "threshold_robustness.csv"), index=False)
    print("Saved threshold_robustness.csv")
    
    return df_exp_g, df_exp_h

if __name__ == "__main__":
    run_warning_attribution()
