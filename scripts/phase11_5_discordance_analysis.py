"""
Phase 11.5 — Experiment 11.5-F: Prediction Discordance Analysis (P6 vs P2).

Objective:
Identify and profile physiological regimes where the Full Proposed Framework (P6) succeeds while
the Vargas-Calixto Sequential Baseline (P2) fails, and vice versa.

Discordance Quadrants (at patient level):
- Group A: Concordant Correct (P6 Correct, P2 Correct)
- Group B: P6-Only Correct (P6 Correct, P2 Incorrect)
- Group C: P2-Only Correct (P2 Correct, P6 Incorrect)
- Group D: Concordant Incorrect (P6 Incorrect, P2 Incorrect)

Physiological Profiling Variables:
- Baseline FHR (bpm)
- STV (ms) & LTV (bpm)
- Acceleration count
- Late, Variable, Prolonged Decelerations
- Deceleration depth & total burden (min)
- Uterine Contraction count & Tachysystole
- FHR-UC coupling index & lag
- Deterioration State (0-4), Velocity, Persistence, Reversals

Outputs:
- results/phase11_5_advantage_attribution/prediction_discordance.csv
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

OUT_DIR = "results/phase11_5_advantage_attribution"
FOLDS_PATH = "data/processed_clinical/folds.json"
PRED_PATH = "results/phase11_priorart_benchmark/model_predictions.csv"
TRAJ_FEATURES_PATH = "results/phase9c_state_trajectory/state_trajectory_features.npz"
os.makedirs(OUT_DIR, exist_ok=True)

def run_discordance_analysis():
    print("=== EXECUTING EXPERIMENT 11.5-F: PREDICTION DISCORDANCE ANALYSIS ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_pred = pd.read_csv(PRED_PATH)
    df_pred["patient_id"] = df_pred["patient_id"].astype(str)
    patient_ids = df_pred["patient_id"].values
    t_del = df_pred["time_before_delivery_min"].values
    
    data_traj = np.load(TRAJ_FEATURES_PATH)
    X_traj = data_traj["X_state_trajectory"] # (8517, 40)
    
    df_pat_labels = df_pred.groupby("patient_id")[["primary_label_715", "severe_label_705"]].first().reset_index()
    y_pat_715 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    y_pat_705 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["severe_label_705"].iloc[0] for p in clean_pids])
    
    pat_window_indices = {pid: np.where(patient_ids == str(pid))[0] for pid in clean_pids}
    
    p2_preds = df_pred["P2_Sequential_Event"].values
    p6_preds = df_pred["P6_Full_Physiology_System"].values
    
    feature_names_dict = {
        "baseline_fhr": (0, "Baseline FHR (bpm)"),
        "stv": (1, "Short-Term Variability STV (ms)"),
        "ltv": (2, "Long-Term Variability LTV (bpm)"),
        "accel_count": (3, "Acceleration Count"),
        "late_decels": (5, "Late Decelerations Count"),
        "var_decels": (6, "Variable Decelerations Count"),
        "prolonged_decels": (7, "Prolonged Decelerations Count"),
        "decel_depth": (8, "Max Deceleration Depth (bpm)"),
        "decel_burden": (10, "Deceleration Burden (min)"),
        "uc_count": (14, "Contraction Count / 20min"),
        "tachysystole": (15, "Tachysystole Episode Indicator"),
        "fhr_uc_coupling": (18, "FHR-UC Coupling Index"),
        "research_state": (25, "Physiological State (0-4)"),
        "velocity_1step": (26, "Deterioration Velocity (1-step)"),
        "state_persist": (29, "State Persistence Duration"),
        "reversal_ind": (30, "Reversal Indicator (0/1)"),
        "multidomain_n": (31, "Multidomain Concurrence Count")
    }
    
    horizons = [30, 0]
    horizon_labels = {30: ">=30m (Early-Warning Horizon)", 0: "Delivery (0m Acute Horizon)"}
    
    rows = []
    
    for h in horizons:
        h_str = horizon_labels[h]
        
        # Get patient scores and chosen feature representations
        p2_pat_scores = []
        p6_pat_scores = []
        pat_chosen_feats = []
        
        for pid in clean_pids:
            idx = pat_window_indices[pid]
            t_pts = t_del[idx]
            if h > 0:
                eligible = np.where(t_pts >= h)[0]
                chosen = idx[eligible[0]] if len(eligible) > 0 else idx[-1]
            else:
                chosen = idx[-1]
            p2_pat_scores.append(p2_preds[chosen])
            p6_pat_scores.append(p6_preds[chosen])
            pat_chosen_feats.append(X_traj[chosen])
            
        p2_pat_scores = np.array(p2_pat_scores)
        p6_pat_scores = np.array(p6_pat_scores)
        pat_chosen_feats = np.array(pat_chosen_feats) # (547, 40)
        
        # Optimal Youden threshold for binary classification
        fpr2, tpr2, thr2 = roc_curve(y_pat_715, p2_pat_scores)
        opt_idx2 = np.argmax(tpr2 - fpr2)
        th_p2 = thr2[opt_idx2]
        
        fpr6, tpr6, thr6 = roc_curve(y_pat_715, p6_pat_scores)
        opt_idx6 = np.argmax(tpr6 - fpr6)
        th_p6 = thr6[opt_idx6]
        
        p2_bin = (p2_pat_scores >= th_p2).astype(int)
        p6_bin = (p6_pat_scores >= th_p6).astype(int)
        
        p2_correct = (p2_bin == y_pat_715)
        p6_correct = (p6_bin == y_pat_715)
        
        # Assign quadrants
        quadrants = []
        for c2, c6 in zip(p2_correct, p6_correct):
            if c6 and c2:
                quadrants.append("Group A: Concordant Correct")
            elif c6 and not c2:
                quadrants.append("Group B: P6-Only Correct")
            elif not c6 and c2:
                quadrants.append("Group C: P2-Only Correct")
            else:
                quadrants.append("Group D: Concordant Incorrect")
        quadrants = np.array(quadrants)
        
        # Calculate feature distributions across each quadrant
        for quad_name in ["Group A: Concordant Correct", "Group B: P6-Only Correct", "Group C: P2-Only Correct", "Group D: Concordant Incorrect"]:
            q_mask = (quadrants == quad_name)
            n_patients_q = int(np.sum(q_mask))
            n_pos_q = int(np.sum(y_pat_715[q_mask]))
            n_sev_q = int(np.sum(y_pat_705[q_mask]))
            prev_715 = float(n_pos_q / n_patients_q) if n_patients_q > 0 else 0.0
            prev_705 = float(n_sev_q / n_patients_q) if n_patients_q > 0 else 0.0
            
            row_dict = {
                "horizon": h_str,
                "horizon_min": h,
                "quadrant": quad_name,
                "n_patients": n_patients_q,
                "n_acidemia_positives": n_pos_q,
                "prevalence_715": round(prev_715, 4),
                "prevalence_705": round(prev_705, 4),
                "p2_threshold": round(float(th_p2), 4),
                "p6_threshold": round(float(th_p6), 4)
            }
            
            # Add mean and std for each physiological feature
            for feat_key, (col_idx, display_label) in feature_names_dict.items():
                if n_patients_q > 0:
                    vals = pat_chosen_feats[q_mask, col_idx]
                    row_dict[f"{feat_key}_mean"] = round(float(np.mean(vals)), 3)
                    row_dict[f"{feat_key}_std"] = round(float(np.std(vals)), 3)
                else:
                    row_dict[f"{feat_key}_mean"] = np.nan
                    row_dict[f"{feat_key}_std"] = np.nan
                    
            rows.append(row_dict)
            
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "prediction_discordance.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_out)} rows)")
    return df_out

if __name__ == "__main__":
    run_discordance_analysis()
