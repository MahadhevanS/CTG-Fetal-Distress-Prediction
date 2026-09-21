"""
Phase 9C Matched-Current-State Analysis Suite:
Executes:
- Exp 9: Stratify patients/windows in the SAME current state (e.g., State 1, 2, 3)
- Compare acidemia risk between Progressing (V > 0), Stable (V = 0), and Reversing (V < 0) trajectories.
- Tests Hypothesis H1: Does trajectory history distinguish risk when current state is held constant?
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = "results/phase9c_state_trajectory"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
FEATURES_PATH = os.path.join(OUT_DIR, "state_trajectory_features.npz")
os.makedirs(OUT_DIR, exist_ok=True)

def run_matched_state_analysis():
    print("=== RUNNING PHASE 9C MATCHED-CURRENT-STATE ANALYSIS ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    
    data = np.load(FEATURES_PATH)
    states = data["research_states"]
    v1 = data["vel_1step"]
    v4 = data["vel_4step"]
    
    patient_ids = df_rolling["patient_id"].values
    y_715 = df_rolling["primary_label_715"].values
    y_705 = df_rolling["severe_label_705"].values
    t_del = df_rolling["time_before_delivery_min"].values
    
    matched_rows = []
    
    # Stratify by Current State k in {1, 2, 3}
    for k in [1, 2, 3]:
        mask_k = (states == k)
        
        # Sub-divide by trajectory velocity V4
        # Progressing: V4 > 0
        # Stable: V4 == 0
        # Reversing: V4 < 0
        prog_mask = mask_k & (v4 > 0.05)
        stab_mask = mask_k & (np.abs(v4) <= 0.05)
        rev_mask = mask_k & (v4 < -0.05)
        
        groups = [
            ("Progressing (V4 > 0)", prog_mask),
            ("Stable / Persistent (V4 ~ 0)", stab_mask),
            ("Reversing / Recovering (V4 < 0)", rev_mask)
        ]
        
        for g_name, g_mask in groups:
            n_obs = int(np.sum(g_mask))
            if n_obs == 0:
                continue
            n_acid = int(np.sum(y_715[g_mask]))
            prev_715 = (n_acid / n_obs) * 100.0
            n_sev = int(np.sum(y_705[g_mask]))
            prev_705 = (n_sev / n_obs) * 100.0
            
            # Unique patients in this subgroup
            pts_in_group = np.unique(patient_ids[g_mask])
            n_pts = len(pts_in_group)
            
            matched_rows.append({
                "current_state_stratum": f"State {k}",
                "trajectory_dynamic": g_name,
                "n_observations": n_obs,
                "n_unique_patients": n_pts,
                "acidemia_715_obs": n_acid,
                "acidemia_prevalence_pct": round(prev_715, 2),
                "severe_705_obs": n_sev,
                "severe_prevalence_pct": round(prev_705, 2)
            })
            
    df_matched = pd.DataFrame(matched_rows)
    df_matched.to_csv(os.path.join(OUT_DIR, "matched_current_state.csv"), index=False)
    print("Saved matched_current_state.csv")

if __name__ == "__main__":
    run_matched_state_analysis()
