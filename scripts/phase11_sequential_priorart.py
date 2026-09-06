"""
Phase 11: P2 — Vargas-Calixto-Inspired Sequential / Event Prior-Art Baseline.

Methodological Concept:
Represents the sequential CTG monitoring and event-based prediction approach:
1. Extracts event-based morphological features per 20-min epoch:
   - Baseline FHR
   - STV / LTV variability
   - Deceleration Frequency (count / 20 min)
   - Deceleration Max Depth (bpm)
   - Deceleration Total Burden (%)
   - UC Frequency (contractions / 20 min)
2. Trains an epoch-level classifier under patient-stratified 5-fold CV.
3. Applies a temporal persistence operator (2-consecutive qualifying window filter / EWMA sequence tracking)
   to produce patient-level sequential risk scores.

Outputs:
- results/phase11_priorart_benchmark/p2_sequential_predictions.npy
- results/phase11_priorart_benchmark/p2_sequential_features.npz
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

def extract_sequential_event_features():
    print("=== EXTRACTING P2: SEQUENTIAL / EVENT FEATURES (Vargas-Calixto-Inspired) ===")
    
    data = np.load(TRAJ_FEATURES_PATH)
    X_raw = data["X_state_trajectory"][:, :19]
    
    # Selected event features:
    # 0: baseline, 2: stv, 3: ltv, 8: var_decel, 10: decel_max_depth, 12: decel_burden, 14: uc_count
    idx_event = [0, 2, 3, 8, 10, 12, 14]
    X_event = X_raw[:, idx_event].astype(np.float32)
    
    np.savez_compressed(
        os.path.join(OUT_DIR, "p2_sequential_features.npz"),
        X_event=X_event,
        feature_names=["baseline", "stv", "ltv", "var_decel", "decel_max_depth", "decel_burden", "uc_count"]
    )
    print(f"P2 Sequential event feature matrix shape: {X_event.shape}")
    return X_event

def run_p2_cross_validation():
    X_event = extract_sequential_event_features()
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    patient_ids = df_rolling["patient_id"].values
    y_715 = df_rolling["primary_label_715"].values
    t_del = df_rolling["time_before_delivery_min"].values
    
    raw_epoch_preds = np.zeros(len(df_rolling), dtype=np.float32)
    
    for f_idx in range(5):
        te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
        tr_pids = set([p for p in clean_pids if p not in te_pids])
        
        train_mask = np.isin(patient_ids, list(tr_pids))
        val_mask = np.isin(patient_ids, list(te_pids))
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_event[train_mask])
        X_va = scaler.transform(X_event[val_mask])
        
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        clf.fit(X_tr, y_715[train_mask])
        raw_epoch_preds[val_mask] = clf.predict_proba(X_va)[:, 1]
        
    # Apply Sequential Persistence Operator (EWMA alpha=0.5 sequence filter respecting patient boundaries)
    p2_seq_preds = np.zeros(len(df_rolling), dtype=np.float32)
    for pid in clean_pids:
        idx = np.where(patient_ids == pid)[0]
        sort_order = np.argsort(-t_del[idx]) # chronological
        idx_sorted = idx[sort_order]
        
        scores = raw_epoch_preds[idx_sorted]
        ewma_scores = np.zeros(len(scores), dtype=np.float32)
        curr = scores[0]
        alpha = 0.5
        for i in range(len(scores)):
            curr = alpha * scores[i] + (1 - alpha) * curr
            ewma_scores[i] = curr
            
        p2_seq_preds[idx_sorted] = ewma_scores
        
    np.save(os.path.join(OUT_DIR, "p2_sequential_predictions.npy"), p2_seq_preds)
    print("Saved p2_sequential_predictions.npy")
    return p2_seq_preds

if __name__ == "__main__":
    run_p2_cross_validation()
