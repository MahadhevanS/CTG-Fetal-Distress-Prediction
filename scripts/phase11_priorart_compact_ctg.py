"""
Phase 11: P1 — DeepCTG-Inspired Compact CTG Baseline.

Methodological Concept:
Represents the compact engineered-feature approach used in DeepCTG-style systems.
Extracts 4 compact, clinically interpretable features per 20-minute causal window:
1. Min FHR Baseline (bpm)
2. Max FHR Baseline (bpm)
3. Acceleration Area (bpm * s)
4. Deceleration Area (bpm * s)

Evaluates under locked patient-stratified 5-fold cross-validation on CTU-UHB (547 patients, 8,517 windows).

Outputs:
- results/phase11_priorart_benchmark/p1_compact_ctg_predictions.npy
- results/phase11_priorart_benchmark/p1_compact_features.npz
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT_DIR = "results/phase11_priorart_benchmark"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
TRAJ_FEATURES_PATH = "results/phase9c_state_trajectory/state_trajectory_features.npz"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_compact_ctg_features():
    print("=== EXTRACTING P1: COMPACT CTG FEATURES (DeepCTG-Inspired) ===")
    
    # Load 19 raw clinical features from Phase 9C state trajectory features
    # Raw features in X_state_trajectory[:, 0:19]:
    # 0: baseline, 1: baseline_slope, 2: stv, 3: ltv, 4: var_slope,
    # 5: accel_count, 6: early_decel, 7: late_decel, 8: var_decel, 9: prolonged_decel,
    # 10: decel_max_depth, 11: decel_area, 12: decel_burden, 13: longest_decel,
    # 14: uc_count, 15: uc_tachy, 16: mean_uc_amp, 17: lag, 18: coupling
    
    data = np.load(TRAJ_FEATURES_PATH)
    X_raw = data["X_state_trajectory"][:, :19]
    
    baseline = X_raw[:, 0]
    baseline_slope = X_raw[:, 1]
    accel_count = X_raw[:, 5]
    decel_area = X_raw[:, 11]
    
    # Derive the 4 compact features:
    # 1. Min Baseline = Baseline - abs(slope) * 5.0
    # 2. Max Baseline = Baseline + abs(slope) * 5.0
    # 3. Acceleration Area Proxy = accel_count * 15.0 bpm * 15.0 s = accel_count * 225.0
    # 4. Deceleration Area = decel_area (bpm * s)
    min_baseline = baseline - np.abs(baseline_slope) * 5.0
    max_baseline = baseline + np.abs(baseline_slope) * 5.0
    accel_area = accel_count * 225.0
    
    X_compact = np.column_stack([
        min_baseline,
        max_baseline,
        accel_area,
        decel_area
    ]).astype(np.float32)
    
    np.savez_compressed(
        os.path.join(OUT_DIR, "p1_compact_features.npz"),
        X_compact=X_compact,
        feature_names=["min_baseline", "max_baseline", "accel_area", "decel_area"]
    )
    print(f"P1 Compact feature matrix shape: {X_compact.shape}")
    return X_compact

def run_p1_cross_validation():
    X_compact = extract_compact_ctg_features()
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    patient_ids = df_rolling["patient_id"].values
    y_715 = df_rolling["primary_label_715"].values
    
    p1_preds = np.zeros(len(df_rolling), dtype=np.float32)
    
    for f_idx in range(5):
        te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
        tr_pids = set([p for p in clean_pids if p not in te_pids])
        
        train_mask = np.isin(patient_ids, list(tr_pids))
        val_mask = np.isin(patient_ids, list(te_pids))
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_compact[train_mask])
        X_va = scaler.transform(X_compact[val_mask])
        
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        clf.fit(X_tr, y_715[train_mask])
        p1_preds[val_mask] = clf.predict_proba(X_va)[:, 1]
        
    np.save(os.path.join(OUT_DIR, "p1_compact_ctg_predictions.npy"), p1_preds)
    print("Saved p1_compact_ctg_predictions.npy")
    return p1_preds

if __name__ == "__main__":
    run_p1_cross_validation()
