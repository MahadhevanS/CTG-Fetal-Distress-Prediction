"""
Phase 11.5 — Experiment 11.5-E: Full Framework Gain Decomposition.

Objective:
Explain the performance gap between Snapshot + Trajectory (P5) and the Full Multidomain Framework (P6)
(observed Phase 11 Delta AUROC = +0.0731, p = 0.006 at delivery).

Pre-specified Component Ladder:
- L1: Snapshot Risk Alone (R_t) [index 38]
- L2: Snapshot + State (R_t + S_t) [indices 38, 25]
- L3: Snapshot + State + Direction (R_t + S_t + D_t) [indices 38, 25, 26, 27, 28, 30]
- L4: Snapshot + State + Direction + Persistence (R_t + S_t + D_t + P_t) [indices 38, 25, 26, 27, 28, 29, 30]
- L5: Snapshot + Full Trajectory (P5) [indices 38, 26, 27, 28, 29, 30, 31, 32]
- L6: Snapshot + Trajectory + 6 Domain Severities [indices 38, 26-32, 19-24]
- L7: Snapshot + Trajectory + Domain Severities + State Occupancies [indices 38, 26-32, 19-24, 25, 33-37]
- L8: Full Multidomain Framework (P6: All 40 features) [indices 0-39]

Evaluates:
- Patient-stratified 5-fold CV (C=0.05, Logistic Regression, StandardScaler).
- Step-by-step incremental AUROC (Delta from step k-1 to step k).
- Total incremental AUROC (Delta from Snapshot L1 to step k).
- Vectorized paired patient-level bootstrap (B=2,000, seed=42).

Output:
- results/phase11_5_advantage_attribution/full_framework_decomposition.csv
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

OUT_DIR = "results/phase11_5_advantage_attribution"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
TRAJ_FEATURES_PATH = "results/phase9c_state_trajectory/state_trajectory_features.npz"
os.makedirs(OUT_DIR, exist_ok=True)

from phase11_bootstrap import paired_patient_bootstrap, fast_auc

def run_framework_decomposition():
    print("=== EXECUTING EXPERIMENT 11.5-E: FULL FRAMEWORK GAIN DECOMPOSITION ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    patient_ids = df_rolling["patient_id"].values
    t_del = df_rolling["time_before_delivery_min"].values
    y_715 = df_rolling["primary_label_715"].values
    y_705 = df_rolling["severe_label_705"].values
    p8_risk = df_rolling["risk_prob_proxy"].values
    
    df_pat_labels = df_rolling.groupby("patient_id")[["primary_label_715", "severe_label_705"]].first().reset_index()
    y_pat_715 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    y_pat_705 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["severe_label_705"].iloc[0] for p in clean_pids])
    
    pat_window_indices = {pid: np.where(patient_ids == str(pid))[0] for pid in clean_pids}
    
    data_traj = np.load(TRAJ_FEATURES_PATH)
    X_traj = data_traj["X_state_trajectory"] # (8517, 40)
    
    # Define ladder stages
    ladder = {
        "L1_Snapshot": {
            "step": 1,
            "name": "Step 1: Snapshot Risk Alone (R_t)",
            "indices": [38],
            "description": "Baseline continuous risk score"
        },
        "L2_Snapshot_State": {
            "step": 2,
            "name": "Step 2: Snapshot + State (R_t + S_t)",
            "indices": [38, 25],
            "description": "Adding physiological state categorization"
        },
        "L3_Snapshot_State_Direction": {
            "step": 3,
            "name": "Step 3: Snapshot + State + Direction (R_t + S_t + D_t)",
            "indices": [38, 25, 26, 27, 28, 30],
            "description": "Adding velocity, acceleration, and reversals"
        },
        "L4_Snapshot_State_Direction_Persist": {
            "step": 4,
            "name": "Step 4: Snapshot + State + Direction + Persistence",
            "indices": [38, 25, 26, 27, 28, 29, 30],
            "description": "Adding state duration persistence"
        },
        "L5_Snapshot_Trajectory_P5": {
            "step": 5,
            "name": "Step 5: Snapshot + Full Trajectory (P5)",
            "indices": [38, 26, 27, 28, 29, 30, 31, 32],
            "description": "P5: Snapshot + Trajectory dynamics"
        },
        "L6_Traj_plus_DomainSeverities": {
            "step": 6,
            "name": "Step 6: Trajectory + 6 Domain Severities",
            "indices": [38, 26, 27, 28, 29, 30, 31, 32, 19, 20, 21, 22, 23, 24],
            "description": "Adding multidomain physiological severity scores"
        },
        "L7_Traj_Domains_Occupancy": {
            "step": 7,
            "name": "Step 7: Trajectory + Domains + State Occupancies",
            "indices": [38, 25, 26, 27, 28, 29, 30, 31, 32, 19, 20, 21, 22, 23, 24, 33, 34, 35, 36, 37],
            "description": "Adding cumulative state occupancy history"
        },
        "L8_Full_Framework_P6": {
            "step": 8,
            "name": "Step 8: Full Multidomain Framework (P6)",
            "indices": list(range(0, 40)),
            "description": "Full 40-D physiological representation"
        }
    }
    
    n_samples = len(df_rolling)
    pred_dict = {k: np.zeros(n_samples, dtype=np.float32) for k in ladder}
    
    print("Training ladder stages across 5 folds...")
    for f_idx in range(5):
        te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
        tr_pids = set([p for p in clean_pids if p not in te_pids])
        
        train_mask = np.isin(patient_ids, list(tr_pids))
        val_mask = np.isin(patient_ids, list(te_pids))
        
        for k, l_info in ladder.items():
            feat_idx = l_info["indices"]
            X_sub = X_traj[:, feat_idx]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_sub[train_mask])
            X_va_s = scaler.transform(X_sub[val_mask])
            
            clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
            clf.fit(X_tr_s, y_715[train_mask])
            pred_dict[k][val_mask] = clf.predict_proba(X_va_s)[:, 1]
            
    def get_scores(pred_arr, h_val):
        p_scores = []
        for pid in clean_pids:
            idx = pat_window_indices[pid]
            t_pts = t_del[idx]
            if h_val > 0:
                eligible = np.where(t_pts >= h_val)[0]
                chosen = idx[eligible[0]] if len(eligible) > 0 else idx[-1]
            else:
                chosen = idx[-1]
            p_scores.append(pred_arr[chosen])
        return np.array(p_scores)
        
    horizons = [30, 0, 20, 10]
    horizon_labels = {30: ">=30m (Primary Horizon)", 0: "Delivery (0m)", 20: ">=20m", 10: ">=10m"}
    
    ladder_keys = list(ladder.keys())
    rows = []
    
    for h in horizons:
        h_str = horizon_labels[h]
        scores_by_step = {k: get_scores(pred_dict[k], h) for k in ladder_keys}
        aurocs = {k: roc_auc_score(y_pat_715, scores_by_step[k]) for k in ladder_keys}
        auprcs = {k: average_precision_score(y_pat_715, scores_by_step[k]) for k in ladder_keys}
        briers = {k: brier_score_loss(y_pat_715, scores_by_step[k]) for k in ladder_keys}
        
        base_scores = scores_by_step["L1_Snapshot"]
        base_auc = aurocs["L1_Snapshot"]
        
        for idx, k in enumerate(ladder_keys):
            step_info = ladder[k]
            step_num = step_info["step"]
            auc_k = aurocs[k]
            auprc_k = auprcs[k]
            brier_k = briers[k]
            scores_k = scores_by_step[k]
            
            # Step-wise delta (vs previous step in ladder)
            if idx == 0:
                step_delta = 0.0
                step_ci_lo = 0.0
                step_ci_hi = 0.0
                step_pval = 1.0
            else:
                prev_k = ladder_keys[idx - 1]
                prev_scores = scores_by_step[prev_k]
                boot_step = paired_patient_bootstrap(y_pat_715, prev_scores, scores_k, n_boot=2000, seed=42)
                step_delta = auc_k - aurocs[prev_k]
                step_ci_lo = boot_step["ci_95_low"]
                step_ci_hi = boot_step["ci_95_high"]
                step_pval = boot_step["p_value"]
                
            # Total delta (vs Step 1 Snapshot)
            if idx == 0:
                total_delta = 0.0
                total_ci_lo = 0.0
                total_ci_hi = 0.0
                total_pval = 1.0
            else:
                boot_tot = paired_patient_bootstrap(y_pat_715, base_scores, scores_k, n_boot=2000, seed=42)
                total_delta = auc_k - base_auc
                total_ci_lo = boot_tot["ci_95_low"]
                total_ci_hi = boot_tot["ci_95_high"]
                total_pval = boot_tot["p_value"]
                
            rows.append({
                "ladder_step": step_num,
                "step_code": k,
                "step_name": step_info["name"],
                "description": step_info["description"],
                "horizon": h_str,
                "horizon_min": h,
                "auroc": round(float(auc_k), 4),
                "auprc": round(float(auprc_k), 4),
                "brier_score": round(float(brier_k), 4),
                "stepwise_delta_auroc": round(float(step_delta), 4),
                "stepwise_ci_low": round(float(step_ci_lo), 4),
                "stepwise_ci_high": round(float(step_ci_hi), 4),
                "stepwise_p_value": round(float(step_pval), 4),
                "total_delta_from_snapshot": round(float(total_delta), 4),
                "total_ci_low": round(float(total_ci_lo), 4),
                "total_ci_high": round(float(total_ci_hi), 4),
                "total_p_value": round(float(total_pval), 4)
            })
            
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "full_framework_decomposition.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_out)} rows)")
    return df_out

if __name__ == "__main__":
    run_framework_decomposition()
