"""
Phase 9C Timing Analysis Suite:
Executes:
- Exp 3: First State-Entry Timing before Delivery (W_k = T_delivery - T_entry,k)
- Proportion of patients entering State 1, 2, 3, 4 at >=60m, >=45m, >=30m, >=20m, >=10m
"""

import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "results/phase9c_state_trajectory"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
ENTRY_TIMES_PATH = os.path.join(OUT_DIR, "patient_entry_times.json")
os.makedirs(OUT_DIR, exist_ok=True)

def run_timing_analysis():
    print("=== RUNNING PHASE 9C FIRST STATE-ENTRY TIMING ANALYSIS ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    
    y_patient_715 = {p: int(df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["primary_label_715"]) for p in clean_pids}
    y_patient_705 = {p: int(df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["severe_label_705"]) for p in clean_pids}
    
    with open(ENTRY_TIMES_PATH) as f:
        entry_blob = json.load(f)
        
    state_names = {
        1: "State 1: Emerging Abnormality",
        2: "State 2: Persistent Abnormality",
        3: "State 3: Progressive Deterioration",
        4: "State 4: Severe Multidomain Deterioration"
    }
    
    timing_rows = []
    
    for k in [1, 2, 3, 4]:
        # Collect entry times for acidemic vs normal patients who entered state k
        acid_times = [entry_blob[p][str(k)] for p in clean_pids if y_patient_715[p] == 1 and entry_blob[p][str(k)] is not None]
        norm_times = [entry_blob[p][str(k)] for p in clean_pids if y_patient_715[p] == 0 and entry_blob[p][str(k)] is not None]
        all_times = [entry_blob[p][str(k)] for p in clean_pids if entry_blob[p][str(k)] is not None]
        
        n_acid_total = sum([1 for p in clean_pids if y_patient_715[p] == 1])
        n_acid_entered = len(acid_times)
        
        # Calculate proportions of acidemic fetuses detected at horizons
        det_60 = sum([1 for t in acid_times if t >= 60])
        det_45 = sum([1 for t in acid_times if t >= 45])
        det_30 = sum([1 for t in acid_times if t >= 30])
        det_20 = sum([1 for t in acid_times if t >= 20])
        det_10 = sum([1 for t in acid_times if t >= 10])
        
        timing_rows.append({
            "state_code": k,
            "state_name": state_names[k],
            "n_acidemic_entered": n_acid_entered,
            "acidemic_entry_rate_pct": round(100.0 * n_acid_entered / n_acid_total, 2),
            "median_lead_time_min": round(float(np.median(acid_times)), 1) if n_acid_entered > 0 else 0.0,
            "iqr_lead_time_min": round(float(np.percentile(acid_times, 75) - np.percentile(acid_times, 25)), 1) if n_acid_entered > 0 else 0.0,
            "mean_lead_time_min": round(float(np.mean(acid_times)), 1) if n_acid_entered > 0 else 0.0,
            "detected_ge_60m_pct": round(100.0 * det_60 / n_acid_total, 2),
            "detected_ge_45m_pct": round(100.0 * det_45 / n_acid_total, 2),
            "detected_ge_30m_pct": round(100.0 * det_30 / n_acid_total, 2),
            "detected_ge_20m_pct": round(100.0 * det_20 / n_acid_total, 2),
            "detected_ge_10m_pct": round(100.0 * det_10 / n_acid_total, 2)
        })
        
    df_timing = pd.DataFrame(timing_rows)
    df_timing.to_csv(os.path.join(OUT_DIR, "state_entry_timing.csv"), index=False)
    print("Saved state_entry_timing.csv")

if __name__ == "__main__":
    run_timing_analysis()
