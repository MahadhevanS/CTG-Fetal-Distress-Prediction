"""
Phase 10 Work Package 10A: Final System Definition & Cross-Validation Pipeline.

Freezes and evaluates candidate systems under patient-stratified 5-fold CV:
- Model A (Snapshot Baseline): Continuous Clinical Huber risk score (R_t)
- Model B (Snapshot + State): Continuous risk combined with current physiological state (R_t + S_t)
- Model C (Snapshot + State + Trajectory): Complete framework (R_t + S_t + V_t + P_t + A_t + R_t + N_t)
- Model D (Snapshot + Trajectory): Continuous risk combined with trajectory (R_t + V_t + P_t + A_t + R_t + N_t)

Outputs:
- results/phase10_final/final_predictions.csv
- results/phase10_final/system_definition.json
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT_DIR = "results/phase10_final"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
TRAJ_FEATURES_PATH = "results/phase9c_state_trajectory/state_trajectory_features.npz"
os.makedirs(OUT_DIR, exist_ok=True)

def run_final_system_definition():
    print("=== EXECUTING WORK PACKAGE 10A: FINAL SYSTEM DEFINITION ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    
    data_traj = np.load(TRAJ_FEATURES_PATH)
    X_traj = data_traj["X_state_trajectory"] # (8517, 40)
    research_states = data_traj["research_states"]
    vel_1step = data_traj["vel_1step"]
    vel_4step = data_traj["vel_4step"]
    accel_step = data_traj["accel_step"]
    state_persist = data_traj["state_persist"]
    reversal_ind = data_traj["reversal_ind"]
    multidomain_n = data_traj["multidomain_n"]
    multidomain_c = data_traj["multidomain_c"]
    
    patient_ids = df_rolling["patient_id"].values
    y_715 = df_rolling["primary_label_715"].values
    y_705 = df_rolling["severe_label_705"].values
    t_del = df_rolling["time_before_delivery_min"].values
    p8_risk = df_rolling["risk_prob_proxy"].values # Base R_t
    
    # Feature indices from 40-D matrix:
    # 25: State S_t
    # 26: Vel 1-step
    # 27: Vel 4-step
    # 28: Accel
    # 29: Persist
    # 30: Reversal
    # 31: Multidomain N
    # 32: Multidomain C
    # 38: p8_risk
    # 39: ewma_risk
    
    # Define candidate feature sets
    feat_configs = {
        "Model_A_Snapshot_Baseline": None, # Direct P8 proxy
        "Model_B_Snapshot_plus_State": [38, 25],
        "Model_C_Snapshot_plus_State_plus_Trajectory": [38, 25, 26, 27, 28, 29, 30, 31, 32],
        "Model_D_Snapshot_plus_Trajectory": [38, 26, 27, 28, 29, 30, 31, 32],
        "Model_E_Full_Physiological_Fusion": list(range(0, 40))
    }
    
    n_samples = len(df_rolling)
    pred_dict = {k: np.zeros(n_samples, dtype=np.float32) for k in feat_configs}
    pred_dict["Model_A_Snapshot_Baseline"] = p8_risk.copy()
    
    print("Executing patient-grouped 5-fold CV for candidate systems...")
    for f_idx in range(5):
        te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
        tr_pids = set([p for p in clean_pids if p not in te_pids])
        
        train_mask = np.isin(patient_ids, list(tr_pids))
        val_mask = np.isin(patient_ids, list(te_pids))
        
        for name, feat_indices in feat_configs.items():
            if feat_indices is None:
                continue
                
            X_sub = X_traj[:, feat_indices]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_sub[train_mask])
            X_va_s = scaler.transform(X_sub[val_mask])
            
            clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
            clf.fit(X_tr_s, y_715[train_mask])
            pred_dict[name][val_mask] = clf.predict_proba(X_va_s)[:, 1]

    # Save final predictions table
    df_pred_out = pd.DataFrame({
        "patient_id": patient_ids,
        "window_index": df_rolling["window_index"].values,
        "time_before_delivery_min": t_del,
        "primary_label_715": y_715,
        "severe_label_705": y_705,
        "research_state": research_states,
        "vel_1step": vel_1step,
        "vel_4step": vel_4step,
        "accel_step": accel_step,
        "state_persist": state_persist,
        "reversal_ind": reversal_ind,
        "multidomain_n": multidomain_n,
        "multidomain_c": multidomain_c,
        **pred_dict
    })
    pred_path = os.path.join(OUT_DIR, "final_predictions.csv")
    df_pred_out.to_csv(pred_path, index=False)
    print(f"Saved {pred_path} ({len(df_pred_out)} rows)")
    
    sys_def = {
        "work_package": "10A — Final System Definition",
        "cohort_size": len(clean_pids),
        "total_rolling_windows": n_samples,
        "models_defined": list(feat_configs.keys()),
        "cross_validation": "Patient-stratified 5-fold CV",
        "causal_window_length_min": 20.0,
        "window_stride_min": 2.5
    }
    with open(os.path.join(OUT_DIR, "system_definition.json"), "w") as f:
        json.dump(sys_def, f, indent=2)
        
    print("Work Package 10A Complete.")

if __name__ == "__main__":
    run_final_system_definition()
