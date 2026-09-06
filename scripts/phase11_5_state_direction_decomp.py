"""
Phase 11.5 — Experiment 11.5-C: State, Direction and Persistence Decomposition.

Objective:
Determine whether physiological state (S_t) and temporal direction (D_t) provide complementary predictive information beyond snapshot risk (R_t).

Nested Model Architecture:
- Model A: R_t (Snapshot Risk Proxy, index [38])
- Model B: R_t + S_t (Snapshot + Physiological State, indices [38, 25])
- Model C: R_t + D_t (Snapshot + Direction/Velocity/Reversal, indices [38, 26, 27, 28, 30])
- Model D: R_t + S_t + D_t (Snapshot + State + Direction, indices [38, 25, 26, 27, 28, 30])
- Model E: R_t + S_t + D_t + P_t (Snapshot + State + Direction + Persistence, indices [38, 25, 26, 27, 28, 29, 30])

Evaluates:
- Patient-stratified 5-fold CV (C=0.05, Logistic Regression, StandardScaler).
- Primary horizons: >=30m and Delivery (0m), as well as >=20m, >=10m.
- Paired patient bootstrap (B=2,000) for nested comparisons:
  - Model B vs Model A (State contribution)
  - Model C vs Model A (Direction contribution)
  - Model D vs Model B (Direction added to State: test of complementarity)
  - Model D vs Model C (State added to Direction: test of complementarity)
  - Model E vs Model D (Persistence added to State + Direction)

Output:
- results/phase11_5_advantage_attribution/state_direction_decomposition.csv
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

def run_state_direction_decomposition():
    print("=== EXECUTING EXPERIMENT 11.5-C: STATE, DIRECTION & PERSISTENCE DECOMPOSITION ===")
    
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
    
    # Feature indices:
    # 38: R_t (Snapshot Risk)
    # 25: S_t (Current State)
    # 26, 27, 28, 30: D_t (vel_1step, vel_4step, accel_step, reversal_ind)
    # 29: P_t (state_persist)
    
    model_configs = {
        "Model_A_Rt": {
            "name": "Model A: R_t (Snapshot Risk Alone)",
            "indices": [38]
        },
        "Model_B_Rt_St": {
            "name": "Model B: R_t + S_t (Snapshot + State)",
            "indices": [38, 25]
        },
        "Model_C_Rt_Dt": {
            "name": "Model C: R_t + D_t (Snapshot + Direction)",
            "indices": [38, 26, 27, 28, 30]
        },
        "Model_D_Rt_St_Dt": {
            "name": "Model D: R_t + S_t + D_t (Snapshot + State + Direction)",
            "indices": [38, 25, 26, 27, 28, 30]
        },
        "Model_E_Rt_St_Dt_Pt": {
            "name": "Model E: R_t + S_t + D_t + P_t (Snapshot + State + Direction + Persistence)",
            "indices": [38, 25, 26, 27, 28, 29, 30]
        }
    }
    
    n_samples = len(df_rolling)
    pred_dict = {k: np.zeros(n_samples, dtype=np.float32) for k in model_configs}
    
    print("Training nested decomposition models across 5 folds...")
    for f_idx in range(5):
        te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
        tr_pids = set([p for p in clean_pids if p not in te_pids])
        
        train_mask = np.isin(patient_ids, list(tr_pids))
        val_mask = np.isin(patient_ids, list(te_pids))
        
        for k, cfg in model_configs.items():
            feat_idx = cfg["indices"]
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
    horizon_labels = {30: ">=30m (Primary Early-Warning)", 0: "Delivery (0m)", 20: ">=20m", 10: ">=10m"}
    
    nested_contrasts = [
        ("State Incremental (B vs A)", "Model_B_Rt_St", "Model_A_Rt", "Does State add to Snapshot Risk?"),
        ("Direction Incremental (C vs A)", "Model_C_Rt_Dt", "Model_A_Rt", "Does Direction add to Snapshot Risk?"),
        ("Direction added to State (D vs B)", "Model_D_Rt_St_Dt", "Model_B_Rt_St", "Does Direction add complementary info to State?"),
        ("State added to Direction (D vs C)", "Model_D_Rt_St_Dt", "Model_C_Rt_Dt", "Does State add complementary info to Direction?"),
        ("Persistence added (E vs D)", "Model_E_Rt_St_Dt_Pt", "Model_D_Rt_St_Dt", "Does Persistence add info beyond State + Direction?")
    ]
    
    rows = []
    
    for h in horizons:
        h_str = horizon_labels[h]
        scores_by_model = {k: get_scores(pred_dict[k], h) for k in model_configs}
        aurocs = {k: roc_auc_score(y_pat_715, scores_by_model[k]) for k in model_configs}
        auprcs = {k: average_precision_score(y_pat_715, scores_by_model[k]) for k in model_configs}
        
        for c_label, m_b, m_a, question in nested_contrasts:
            scores_b = scores_by_model[m_b]
            scores_a = scores_by_model[m_a]
            
            auc_b = aurocs[m_b]
            auc_a = aurocs[m_a]
            
            boot_res = paired_patient_bootstrap(y_pat_715, scores_a, scores_b, n_boot=2000, seed=42)
            
            if boot_res["ci_95_low"] > 0.0:
                inference = "Significant Complementary Information (CI > 0)"
            elif boot_res["ci_95_high"] < 0.0:
                inference = "Negative Increment (CI < 0)"
            else:
                inference = "No Significant Increment (CI overlaps 0)"
                
            rows.append({
                "contrast": c_label,
                "research_question": question,
                "horizon": h_str,
                "horizon_min": h,
                "model_b": model_configs[m_b]["name"],
                "auroc_b": round(float(auc_b), 4),
                "auprc_b": round(float(auprcs[m_b]), 4),
                "model_a": model_configs[m_a]["name"],
                "auroc_a": round(float(auc_a), 4),
                "auprc_a": round(float(auprcs[m_a]), 4),
                "delta_auroc_point": round(float(auc_b - auc_a), 4),
                "delta_auroc_boot_mean": round(float(boot_res["delta_mean"]), 4),
                "ci_95_low": round(float(boot_res["ci_95_low"]), 4),
                "ci_95_high": round(float(boot_res["ci_95_high"]), 4),
                "p_value": round(float(boot_res["p_value"]), 4),
                "complementarity_inference": inference
            })
            
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "state_direction_decomposition.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_out)} rows)")
    return df_out

if __name__ == "__main__":
    run_state_direction_decomposition()
