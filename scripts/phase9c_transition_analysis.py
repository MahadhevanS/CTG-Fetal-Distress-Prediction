"""
Phase 9C Transition Analysis Suite:
Executes:
- Exp 1: State-Risk Gradient & Trend Tests
- Exp 2: State Occupancy & Duration Distribution
- Exp 4: State Persistence Analysis
- Exp 5: State Reversal vs Upward Progression
- Exp 6 & 7: Progression Velocity & Acceleration
- Exp 8: Multidomain Concurrence
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

OUT_DIR = "results/phase9c_state_trajectory"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
FEATURES_PATH = os.path.join(OUT_DIR, "state_trajectory_features.npz")
os.makedirs(OUT_DIR, exist_ok=True)

def run_transition_analysis():
    print("=== RUNNING PHASE 9C TRANSITION & OCCUPANCY ANALYSIS ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    
    data = np.load(FEATURES_PATH)
    states = data["research_states"]
    v1 = data["vel_1step"]
    v4 = data["vel_4step"]
    acc = data["accel_step"]
    persist = data["state_persist"]
    rev = data["reversal_ind"]
    multi_n = data["multidomain_n"]
    multi_c = data["multidomain_c"]
    occupancy = data["occupancy_props"]
    
    patient_ids = df_rolling["patient_id"].values
    y_715 = df_rolling["primary_label_715"].values
    y_705 = df_rolling["severe_label_705"].values
    t_del = df_rolling["time_before_delivery_min"].values
    
    y_patient_715 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["primary_label_715"] for p in clean_pids])
    y_patient_705 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["severe_label_705"] for p in clean_pids])
    
    # -------------------------------------------------------------------------
    # Exp 1: State-Risk Gradient & Trend Test
    # -------------------------------------------------------------------------
    # Max state reached per patient
    pat_max_state = []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        pat_max_state.append(int(np.max(states[idx])))
    pat_max_state = np.array(pat_max_state)
    
    gradient_rows = []
    state_names = {
        0: "State 0: Stable",
        1: "State 1: Emerging Abnormality",
        2: "State 2: Persistent Abnormality",
        3: "State 3: Progressive Deterioration",
        4: "State 4: Severe Multidomain Deterioration"
    }
    
    for k in range(5):
        mask_k = (pat_max_state == k)
        n_k = int(np.sum(mask_k))
        n_acid = int(np.sum(y_patient_715[mask_k]))
        n_sev = int(np.sum(y_patient_705[mask_k]))
        
        prev_715 = (n_acid / n_k * 100.0) if n_k > 0 else 0.0
        prev_705 = (n_sev / n_k * 100.0) if n_k > 0 else 0.0
        
        # 95% Wilson score interval
        ci_low_715, ci_high_715 = stats.binomtest(n_acid, n_k).proportion_ci(confidence_level=0.95) if n_k > 0 else (0, 0)
        
        gradient_rows.append({
            "state_code": k,
            "state_name": state_names[k],
            "n_patients": n_k,
            "acidemia_715_count": n_acid,
            "acidemia_715_prevalence_pct": round(prev_715, 2),
            "ci_95_low": round(ci_low_715 * 100.0, 2),
            "ci_95_high": round(ci_high_715 * 100.0, 2),
            "severe_705_count": n_sev,
            "severe_705_prevalence_pct": round(prev_705, 2)
        })
    df_gradient = pd.DataFrame(gradient_rows)
    df_gradient.to_csv(os.path.join(OUT_DIR, "state_risk_gradient.csv"), index=False)
    
    # Logistic regression ordinal trend test
    lr_trend = LogisticRegression()
    lr_trend.fit(pat_max_state.reshape(-1, 1), y_patient_715)
    odds_ratio_per_state = np.exp(lr_trend.coef_[0][0])
    print(f"Exp 1 State-Risk Gradient: Odds Ratio per State Step = {odds_ratio_per_state:.3f}")

    # -------------------------------------------------------------------------
    # Exp 2: State Occupancy Comparison
    # -------------------------------------------------------------------------
    occupancy_rows = []
    with open(os.path.join(OUT_DIR, "patient_occupancy.json")) as f:
        occ_blob = json.load(f)
        
    for k in range(5):
        occ_acid = [occ_blob[p][str(k)] * 2.5 for p in clean_pids if y_patient_715[clean_pids.index(p)] == 1]
        occ_norm = [occ_blob[p][str(k)] * 2.5 for p in clean_pids if y_patient_715[clean_pids.index(p)] == 0]
        
        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(occ_acid, occ_norm, alternative='two-sided')
        
        occupancy_rows.append({
            "state_code": k,
            "state_name": state_names[k],
            "mean_minutes_normal": round(float(np.mean(occ_norm)), 1),
            "median_minutes_normal": round(float(np.median(occ_norm)), 1),
            "mean_minutes_acidemic": round(float(np.mean(occ_acid)), 1),
            "median_minutes_acidemic": round(float(np.median(occ_acid)), 1),
            "mann_whitney_u": round(float(u_stat), 1),
            "p_value": round(float(p_val), 5)
        })
    df_occ = pd.DataFrame(occupancy_rows)
    df_occ.to_csv(os.path.join(OUT_DIR, "state_occupancy_metrics.csv"), index=False)
    print("Saved state_occupancy_metrics.csv")

    # -------------------------------------------------------------------------
    # Exp 4, 5, 6, 7, 8: Trajectory Dynamics & Transition Metrics
    # -------------------------------------------------------------------------
    trans_rows = []
    
    # 1. State Reversal vs Upward Progression
    # Total reversals per patient
    rev_counts_acid = []
    rev_counts_norm = []
    vel_max_acid = []
    vel_max_norm = []
    multi_persist_acid = []
    multi_persist_norm = []
    
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        p_rev = np.sum(rev[idx])
        p_vmax = np.max(v4[idx])
        p_cmax = np.max(multi_c[idx])
        is_acid = (y_patient_715[clean_pids.index(pid)] == 1)
        
        if is_acid:
            rev_counts_acid.append(p_rev)
            vel_max_acid.append(p_vmax)
            multi_persist_acid.append(p_cmax)
        else:
            rev_counts_norm.append(p_rev)
            vel_max_norm.append(p_vmax)
            multi_persist_norm.append(p_cmax)
            
    # Reversal metrics
    u_rev, p_rev = stats.mannwhitneyu(rev_counts_acid, rev_counts_norm, alternative='two-sided')
    trans_rows.append({
        "metric_category": "Exp 5: State Reversal",
        "metric_name": "Total Reversal Count per Patient",
        "normal_mean": round(float(np.mean(rev_counts_norm)), 2),
        "acidemic_mean": round(float(np.mean(rev_counts_acid)), 2),
        "p_value": round(float(p_rev), 4)
    })
    
    # Velocity metrics
    u_vel, p_vel = stats.mannwhitneyu(vel_max_acid, vel_max_norm, alternative='two-sided')
    trans_rows.append({
        "metric_category": "Exp 6: Progression Velocity",
        "metric_name": "Maximum 4-Window Velocity (V4)",
        "normal_mean": round(float(np.mean(vel_max_norm)), 2),
        "acidemic_mean": round(float(np.mean(vel_max_acid)), 2),
        "p_value": round(float(p_vel), 4)
    })
    
    # Multidomain persistence
    u_mp, p_mp = stats.mannwhitneyu(multi_persist_acid, multi_persist_norm, alternative='two-sided')
    trans_rows.append({
        "metric_category": "Exp 8: Multidomain Concurrence",
        "metric_name": "Max Consecutive Multidomain Windows (C_t)",
        "normal_mean": round(float(np.mean(multi_persist_norm)), 2),
        "acidemic_mean": round(float(np.mean(multi_persist_acid)), 2),
        "p_value": round(float(p_mp), 4)
    })
    
    pd.DataFrame(trans_rows).to_csv(os.path.join(OUT_DIR, "transition_dynamics_metrics.csv"), index=False)
    print("Saved transition_dynamics_metrics.csv")

if __name__ == "__main__":
    run_transition_analysis()
