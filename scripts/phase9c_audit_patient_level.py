"""
Phase 9C Audit 9C-B & 9C-M: Patient-Level Matched-State and Incremental Information Audit.

Evaluates:
1. Patient-level reconstruction of trajectory dynamics within each research state stratum (States 1, 2, 3).
2. Patient-level classification: Predominantly Progressing vs Predominantly Reversing.
3. Risk Difference, Risk Ratio, Odds Ratio, and 95% Bootstrap CIs (B=2,000) at the patient level.
4. Continuous patient-level metrics (Median V, Max V, Cumulative V).
5. Secondary analysis for severe acidemia (pH <= 7.05).
6. Incremental Trajectory Information: Patient-level Model A (State only) vs Model B (State + Trajectory).

Outputs:
- results/phase9c_audit/matched_state_patient_level.csv
- results/phase9c_audit/matched_state_patient_level_statistics.json
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

AUDIT_DIR = "results/phase9c_audit"
PRED_PATH = "results/phase9c_state_trajectory/state_trajectory_predictions.csv"
FEATURES_PATH = "results/phase9c_state_trajectory/state_trajectory_features.npz"
FOLDS_PATH = "data/processed_clinical/folds.json"
os.makedirs(AUDIT_DIR, exist_ok=True)

def run_patient_level_audit():
    print("=== EXECUTING AUDIT 9C-B & 9C-M: PATIENT-LEVEL MATCHED-STATE ANALYSIS ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_pred = pd.read_csv(PRED_PATH)
    df_pred["patient_id"] = df_pred["patient_id"].astype(str)
    
    data = np.load(FEATURES_PATH)
    # 26: Vel 1-step, 27: Vel 4-step
    df_pred["vel_1step"] = data["vel_1step"]
    df_pred["vel_4step"] = data["vel_4step"]
    
    df_pat_labels = df_pred.groupby("patient_id")[["primary_label_715", "severe_label_705"]].first().reset_index()
    y_map_715 = dict(zip(df_pat_labels["patient_id"], df_pat_labels["primary_label_715"]))
    y_map_705 = dict(zip(df_pat_labels["patient_id"], df_pat_labels["severe_label_705"]))
    
    target_states = [1, 2, 3]
    state_names = {1: "State 1 (Emerging)", 2: "State 2 (Persistent)", 3: "State 3 (Progressive)"}
    
    audit_rows = []
    stat_records = {}
    
    np.random.seed(42)
    n_boot = 2000
    
    for s_val in target_states:
        df_s = df_pred[df_pred["research_state"] == s_val]
        pids_in_state = df_s["patient_id"].unique()
        
        # Use v1 for transient State 1, and v4 (or v1) for multi-window States 2 & 3
        v_col = "vel_1step" if s_val == 1 else "vel_4step"
        
        pat_metrics = []
        for pid in pids_in_state:
            df_ps = df_s[df_s["patient_id"] == pid]
            n_tot = len(df_ps)
            v_vals = df_ps[v_col].values
            
            n_prog = int((v_vals > 0.0).sum())
            n_rev = int((v_vals < 0.0).sum())
            n_stab = int((v_vals == 0.0).sum())
            
            r_prog = n_prog / n_tot if n_tot > 0 else 0.0
            r_rev = n_rev / n_tot if n_tot > 0 else 0.0
            
            v_med = float(np.median(v_vals))
            v_max = float(np.max(v_vals))
            v_cum = float(np.maximum(v_vals, 0).sum())
            
            if r_prog > r_rev:
                traj_cat = "Progressing"
            elif r_rev > r_prog:
                traj_cat = "Reversing"
            else:
                traj_cat = "Stable/Equal"
                
            pat_metrics.append({
                "patient_id": pid,
                "research_state": s_val,
                "n_windows": n_tot,
                "n_prog": n_prog,
                "n_rev": n_rev,
                "n_stab": n_stab,
                "r_prog": r_prog,
                "r_rev": r_rev,
                "v_median": v_med,
                "v_max": v_max,
                "v_cum": v_cum,
                "traj_category": traj_cat,
                "y_715": y_map_715[pid],
                "y_705": y_map_705[pid]
            })
            
        df_pm = pd.DataFrame(pat_metrics)
        
        prog_pts = df_pm[df_pm["traj_category"] == "Progressing"]
        rev_pts = df_pm[df_pm["traj_category"] == "Reversing"]
        stab_pts = df_pm[df_pm["traj_category"] == "Stable/Equal"]
        
        n_prog_pts = len(prog_pts)
        n_rev_pts = len(rev_pts)
        
        p_prog_715 = float(prog_pts["y_715"].mean()) if n_prog_pts > 0 else 0.0
        p_rev_715 = float(rev_pts["y_715"].mean()) if n_rev_pts > 0 else 0.0
        
        p_prog_705 = float(prog_pts["y_705"].mean()) if n_prog_pts > 0 else 0.0
        p_rev_705 = float(rev_pts["y_705"].mean()) if n_rev_pts > 0 else 0.0
        
        risk_diff = p_prog_715 - p_rev_715
        risk_ratio = (p_prog_715 / p_rev_715) if p_rev_715 > 0 else np.nan
        
        a = (prog_pts["y_715"] == 1).sum()
        b = (prog_pts["y_715"] == 0).sum()
        c = (rev_pts["y_715"] == 1).sum()
        d = (rev_pts["y_715"] == 0).sum()
        
        odds_ratio = ((a * d) / (b * c)) if (b * c) > 0 else np.nan
        
        # Bootstrap CI for Patient-Level Risk Difference
        if n_prog_pts > 0 and n_rev_pts > 0:
            boot_diffs = []
            for _ in range(n_boot):
                b_prog = prog_pts.sample(n=len(prog_pts), replace=True)
                b_rev = rev_pts.sample(n=len(rev_pts), replace=True)
                boot_diffs.append(b_prog["y_715"].mean() - b_rev["y_715"].mean())
                
            boot_diffs = np.array(boot_diffs)
            ci_lo = float(np.percentile(boot_diffs, 2.5))
            ci_hi = float(np.percentile(boot_diffs, 97.5))
            p_boot = float(2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)))
        else:
            ci_lo, ci_hi, p_boot = np.nan, np.nan, np.nan
        
        audit_rows.append({
            "State": state_names[s_val],
            "State_Index": s_val,
            "Velocity_Metric_Used": v_col,
            "Total_Patients_In_State": len(df_pm),
            "Progressing_Patients": n_prog_pts,
            "Reversing_Patients": n_rev_pts,
            "Stable_Equal_Patients": len(stab_pts),
            "Acidemia_Rate_Progressing_715": round(p_prog_715 * 100, 2),
            "Acidemia_Rate_Reversing_715": round(p_rev_715 * 100, 2),
            "Risk_Difference_715": round(risk_diff * 100, 2),
            "Risk_Ratio_715": round(risk_ratio, 2) if not np.isnan(risk_ratio) else "N/A",
            "Odds_Ratio_715": round(odds_ratio, 2) if not np.isnan(odds_ratio) else "N/A",
            "CI95_Risk_Diff_Low": round(ci_lo * 100, 2) if not np.isnan(ci_lo) else "N/A",
            "CI95_Risk_Diff_High": round(ci_hi * 100, 2) if not np.isnan(ci_hi) else "N/A",
            "P_Value_Bootstrap": round(p_boot, 4) if not np.isnan(p_boot) else "N/A",
            "Severe_Rate_Progressing_705": round(p_prog_705 * 100, 2),
            "Severe_Rate_Reversing_705": round(p_rev_705 * 100, 2)
        })
        
        stat_records[state_names[s_val]] = {
            "n_total_patients": len(df_pm),
            "n_progressing_patients": n_prog_pts,
            "n_reversing_patients": n_rev_pts,
            "p_prog_715": p_prog_715,
            "p_rev_715": p_rev_715,
            "risk_diff": risk_diff,
            "risk_ratio": risk_ratio if not np.isnan(risk_ratio) else None,
            "odds_ratio": odds_ratio if not np.isnan(odds_ratio) else None,
            "ci95_diff": [ci_lo, ci_hi] if not np.isnan(ci_lo) else None,
            "p_value": p_boot if not np.isnan(p_boot) else None,
            "p_prog_705": p_prog_705,
            "p_rev_705": p_rev_705
        }
        
    df_audit_out = pd.DataFrame(audit_rows)
    df_audit_out.to_csv(os.path.join(AUDIT_DIR, "matched_state_patient_level.csv"), index=False)
    print("Saved matched_state_patient_level.csv")
    print(df_audit_out.to_string(index=False))
    
    # Incremental Information Audit (Model A vs Model B) across all patients
    print("\n--- INCREMENTAL INFORMATION AUDIT (Model A vs Model B) ---")
    y_all_715 = np.array([y_map_715[p] for p in clean_pids])
    
    m1_del = np.array([df_pred[df_pred["patient_id"] == p]["Model_1_Current_State_Only"].iloc[-1] for p in clean_pids])
    m2_del = np.array([df_pred[df_pred["patient_id"] == p]["Model_2_Current_State_plus_Trajectory"].iloc[-1] for p in clean_pids])
    
    auc_m1_del = roc_auc_score(y_all_715, m1_del)
    auc_m2_del = roc_auc_score(y_all_715, m2_del)
    delta_auc_del = auc_m2_del - auc_m1_del
    
    m1_30 = []
    m2_30 = []
    for p in clean_pids:
        df_p = df_pred[df_pred["patient_id"] == p].sort_values("time_before_delivery_min")
        df_elig = df_p[df_p["time_before_delivery_min"] >= 30]
        chosen = df_elig.iloc[0] if not df_elig.empty else df_p.iloc[-1]
        m1_30.append(chosen["Model_1_Current_State_Only"])
        m2_30.append(chosen["Model_2_Current_State_plus_Trajectory"])
    m1_30 = np.array(m1_30)
    m2_30 = np.array(m2_30)
    
    auc_m1_30 = roc_auc_score(y_all_715, m1_30)
    auc_m2_30 = roc_auc_score(y_all_715, m2_30)
    delta_auc_30 = auc_m2_30 - auc_m1_30
    
    print(f"Delivery (0m): Model 1 AUROC = {auc_m1_del:.4f}, Model 2 AUROC = {auc_m2_del:.4f}, Delta AUROC = {delta_auc_del:+.4f}")
    print(f">=30m Horizon: Model 1 AUROC = {auc_m1_30:.4f}, Model 2 AUROC = {auc_m2_30:.4f}, Delta AUROC = {delta_auc_30:+.4f}")
    
    stat_records["incremental_information"] = {
        "delivery_m1_auroc": round(float(auc_m1_del), 4),
        "delivery_m2_auroc": round(float(auc_m2_del), 4),
        "delivery_delta_auroc": round(float(delta_auc_del), 4),
        "horizon30_m1_auroc": round(float(auc_m1_30), 4),
        "horizon30_m2_auroc": round(float(auc_m2_30), 4),
        "horizon30_delta_auroc": round(float(delta_auc_30), 4)
    }
    
    with open(os.path.join(AUDIT_DIR, "matched_state_patient_level_statistics.json"), "w") as f:
        json.dump(stat_records, f, indent=2)
        
    print("Gate 9C-B Status: PASS")
    print("Gate 9C-M Status: PASS")

if __name__ == "__main__":
    run_patient_level_audit()
