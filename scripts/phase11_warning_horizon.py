"""
Phase 11: Multi-Horizon Discrimination Benchmark.

Evaluates:
- Patient-level AUROC and AUPRC for all 6 models (P1 to P6) across horizons:
  >=60m, >=45m, >=30m, >=20m, >=10m, Delivery (0m)
- Primary outcome (pH <= 7.15) and severe acidemia (pH <= 7.05)

Outputs:
- results/phase11_priorart_benchmark/horizon_results.csv
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

OUT_DIR = "results/phase11_priorart_benchmark"
PRED_PATH = os.path.join(OUT_DIR, "model_predictions.csv")
FOLDS_PATH = "data/processed_clinical/folds.json"
os.makedirs(OUT_DIR, exist_ok=True)

def run_multihorizon_benchmark():
    print("=== EXECUTING PHASE 11: MULTI-HORIZON DISCRIMINATION BENCHMARK ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_pred = pd.read_csv(PRED_PATH)
    df_pred["patient_id"] = df_pred["patient_id"].astype(str)
    
    patient_ids = df_pred["patient_id"].values
    t_del = df_pred["time_before_delivery_min"].values
    
    df_pat_labels = df_pred.groupby("patient_id")[["primary_label_715", "severe_label_705"]].first().reset_index()
    y_pat_715 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    y_pat_705 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["severe_label_705"].iloc[0] for p in clean_pids])
    
    models = [
        ("P1_Compact_CTG", "P1 (Compact CTG Baseline - DeepCTG-Inspired)"),
        ("P2_Sequential_Event", "P2 (Sequential / Event Baseline - Vargas-Calixto-Inspired)"),
        ("P3_Snapshot_Baseline", "P3 (Locked Snapshot Baseline - Continuous Clinical Huber)"),
        ("P4_Snapshot_plus_State", "P4 (Snapshot + Physiological State)"),
        ("P5_Snapshot_plus_Trajectory", "P5 (Snapshot + Deterioration Trajectory)"),
        ("P6_Full_Physiology_System", "P6 (Full Physiology-Guided Proposed System)")
    ]
    
    horizons = [60, 45, 30, 20, 10, 0]
    pat_window_indices = {pid: np.where(patient_ids == str(pid))[0] for pid in clean_pids}
    
    def get_patient_scores(col_name, h_val):
        arr = df_pred[col_name].values
        p_scores = []
        for pid in clean_pids:
            idx = pat_window_indices[pid]
            t_pts = t_del[idx]
            if h_val > 0:
                eligible = np.where(t_pts >= h_val)[0]
                chosen = idx[eligible[0]] if len(eligible) > 0 else idx[-1]
            else:
                chosen = idx[-1]
            p_scores.append(arr[chosen])
        return np.array(p_scores)
        
    horizon_rows = []
    
    for h in horizons:
        h_label = f">={h}m" if h > 0 else "Delivery (0m)"
        
        for col_name, display_name in models:
            scores = get_patient_scores(col_name, h)
            
            auc_715 = roc_auc_score(y_pat_715, scores)
            auprc_715 = average_precision_score(y_pat_715, scores)
            auc_705 = roc_auc_score(y_pat_705, scores)
            auprc_705 = average_precision_score(y_pat_705, scores)
            
            horizon_rows.append({
                "model_code": col_name[:2],
                "model_name": display_name,
                "horizon": h_label,
                "horizon_min": h,
                "auroc_715": round(float(auc_715), 4),
                "auprc_715": round(float(auprc_715), 4),
                "auroc_705": round(float(auc_705), 4),
                "auprc_705": round(float(auprc_705), 4)
            })
            
    df_hor = pd.DataFrame(horizon_rows)
    df_hor.to_csv(os.path.join(OUT_DIR, "horizon_results.csv"), index=False)
    print("Saved horizon_results.csv")
    
    # Pivot table for summary display
    pivot_715 = df_hor.pivot(index="model_name", columns="horizon", values="auroc_715")
    print("\n--- MULTI-HORIZON AUROC TABLE (pH <= 7.15) ---")
    print(pivot_715.to_string())

if __name__ == "__main__":
    run_multihorizon_benchmark()
