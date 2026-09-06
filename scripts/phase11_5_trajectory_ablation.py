"""
Phase 11.5 — Experiment 11.5-B: Trajectory Component Attribution (LOCO Ablation).

Objective:
Determine which temporal trajectory concepts contribute to the observed P5 > P3 improvement.

Candidate Trajectory Components (from frozen 40-D matrix):
- C1: Velocity / Risk Slope (indices [26, 27]: vel_1step, vel_4step)
- C2: Persistence (index [29]: state_persist)
- C3: Progression / Acceleration (index [28]: accel_step)
- C4: Direction Change / Reversals (index [30]: reversal_ind)
- C5: Multidomain Concurrence (indices [31, 32]: multidomain_n, multidomain_c)

Methodology:
- For each component C_i, train P5_{-C_i} under identical patient-stratified 5-fold CV (C=0.05, Logistic Regression, StandardScaler).
- Evaluate patient-level AUROC at primary early-warning horizon (>=30m) and delivery (0m).
- Compute Delta_i = AUROC(P5) - AUROC(P5_{-C_i}) using B=2,000 paired patient-level bootstrap.

Output:
- results/phase11_5_advantage_attribution/trajectory_component_ablation.csv
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

OUT_DIR = "results/phase11_5_advantage_attribution"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
TRAJ_FEATURES_PATH = "results/phase9c_state_trajectory/state_trajectory_features.npz"
os.makedirs(OUT_DIR, exist_ok=True)

from phase11_bootstrap import paired_patient_bootstrap, fast_auc

def run_trajectory_ablation():
    print("=== EXECUTING EXPERIMENT 11.5-B: TRAJECTORY COMPONENT ATTRIBUTION ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    patient_ids = df_rolling["patient_id"].values
    t_del = df_rolling["time_before_delivery_min"].values
    y_715 = df_rolling["primary_label_715"].values
    y_705 = df_rolling["severe_label_705"].values
    
    df_pat_labels = df_rolling.groupby("patient_id")[["primary_label_715", "severe_label_705"]].first().reset_index()
    y_pat_715 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    y_pat_705 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["severe_label_705"].iloc[0] for p in clean_pids])
    
    pat_window_indices = {pid: np.where(patient_ids == str(pid))[0] for pid in clean_pids}
    
    data_traj = np.load(TRAJ_FEATURES_PATH)
    X_traj = data_traj["X_state_trajectory"] # (8517, 40)
    
    # Full P5 feature set: Snapshot risk (38) + Trajectory (26, 27, 28, 29, 30, 31, 32)
    p5_full_indices = [38, 26, 27, 28, 29, 30, 31, 32]
    
    components = {
        "Full_P5": {
            "name": "Full P5 (Snapshot + All Trajectory Components)",
            "indices": p5_full_indices,
            "ablated_concept": "None (Full Model)"
        },
        "P5_minus_Velocity": {
            "name": "P5 without Velocity / Risk Slope",
            "indices": [i for i in p5_full_indices if i not in [26, 27]],
            "ablated_concept": "Velocity / Risk Slope (vel_1step, vel_4step)"
        },
        "P5_minus_Persistence": {
            "name": "P5 without State Persistence",
            "indices": [i for i in p5_full_indices if i not in [29]],
            "ablated_concept": "State Persistence (state_persist)"
        },
        "P5_minus_Acceleration": {
            "name": "P5 without Progression Acceleration",
            "indices": [i for i in p5_full_indices if i not in [28]],
            "ablated_concept": "Progression Acceleration (accel_step)"
        },
        "P5_minus_Reversal": {
            "name": "P5 without Reversal Indicator",
            "indices": [i for i in p5_full_indices if i not in [30]],
            "ablated_concept": "Direction Reversal (reversal_ind)"
        },
        "P5_minus_Multidomain": {
            "name": "P5 without Multidomain Concurrence",
            "indices": [i for i in p5_full_indices if i not in [31, 32]],
            "ablated_concept": "Multidomain Concurrence (multidomain_n, multidomain_c)"
        }
    }
    
    n_samples = len(df_rolling)
    pred_dict = {k: np.zeros(n_samples, dtype=np.float32) for k in components}
    
    print("Training LOCO ablation variants across 5 folds...")
    for f_idx in range(5):
        te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
        tr_pids = set([p for p in clean_pids if p not in te_pids])
        
        train_mask = np.isin(patient_ids, list(tr_pids))
        val_mask = np.isin(patient_ids, list(te_pids))
        
        for k, comp_info in components.items():
            feat_idx = comp_info["indices"]
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
    
    rows = []
    
    for h in horizons:
        h_str = horizon_labels[h]
        scores_full_p5 = get_scores(pred_dict["Full_P5"], h)
        auc_full_p5 = roc_auc_score(y_pat_715, scores_full_p5)
        auprc_full_p5 = average_precision_score(y_pat_715, scores_full_p5)
        
        for k, comp_info in components.items():
            scores_k = get_scores(pred_dict[k], h)
            auc_k = roc_auc_score(y_pat_715, scores_k)
            auprc_k = average_precision_score(y_pat_715, scores_k)
            
            if k == "Full_P5":
                delta_point = 0.0
                delta_mean = 0.0
                ci_lo = 0.0
                ci_hi = 0.0
                p_val = 1.0
                impact = "Reference Full Model"
            else:
                # Delta_i = AUROC(Full P5) - AUROC(P5 - C_i)
                boot_res = paired_patient_bootstrap(y_pat_715, scores_k, scores_full_p5, n_boot=2000, seed=42)
                delta_point = auc_full_p5 - auc_k
                delta_mean = boot_res["delta_mean"]
                ci_lo = boot_res["ci_95_low"]
                ci_hi = boot_res["ci_95_high"]
                p_val = boot_res["p_value"]
                
                if ci_lo > 0.0:
                    impact = "Significant Positive Contributor (Removal hurts performance)"
                elif ci_hi < 0.0:
                    impact = "Detrimental / Redundant (Removal improves performance)"
                else:
                    impact = "Minor / Distributed Contribution (CI overlaps 0)"
                    
            rows.append({
                "model_key": k,
                "model_name": comp_info["name"],
                "ablated_concept": comp_info["ablated_concept"],
                "horizon": h_str,
                "horizon_min": h,
                "auroc": round(float(auc_k), 4),
                "auprc": round(float(auprc_k), 4),
                "delta_from_full_p5": round(float(delta_point), 4),
                "delta_boot_mean": round(float(delta_mean), 4),
                "ci_95_low": round(float(ci_lo), 4),
                "ci_95_high": round(float(ci_hi), 4),
                "p_value": round(float(p_val), 4),
                "component_impact": impact
            })
            
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "trajectory_component_ablation.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_out)} rows)")
    return df_out

if __name__ == "__main__":
    run_trajectory_ablation()
