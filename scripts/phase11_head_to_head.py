"""
Phase 11: Head-to-Head Comparative Benchmark Execution.

Evaluates the 6-Model Prior-Art Benchmark Suite:
- P1: DeepCTG-Inspired Compact CTG Baseline
- P2: Vargas-Calixto-Inspired Sequential / Event Baseline
- P3: Locked Snapshot Baseline (Continuous Clinical Huber)
- P4: Snapshot + State (R_t + S_t)
- P5: Snapshot + Trajectory (R_t + Dynamics)
- P6: Full Physiology-Guided System (Full Multi-domain Fusion)

Outputs:
- results/phase11_priorart_benchmark/model_predictions.csv
- results/phase11_priorart_benchmark/model_comparison.csv
- results/phase11_priorart_benchmark/auprc_results.csv
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

OUT_DIR = "results/phase11_priorart_benchmark"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P10_PRED_PATH = "results/phase10_final/final_predictions.csv"
os.makedirs(OUT_DIR, exist_ok=True)

from phase11_priorart_compact_ctg import run_p1_cross_validation
from phase11_sequential_priorart import run_p2_cross_validation

def run_head_to_head_benchmark():
    print("=== EXECUTING PHASE 11: HEAD-TO-HEAD BENCHMARK EVALUATION ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    
    # 1. Obtain P1 and P2 predictions
    p1_preds = run_p1_cross_validation()
    p2_preds = run_p2_cross_validation()
    
    # 2. Obtain P3, P4, P5, P6 predictions from Phase 10 unified predictions
    df_p10 = pd.read_csv(P10_PRED_PATH)
    p3_preds = df_p10["Model_A_Snapshot_Baseline"].values
    p4_preds = df_p10["Model_B_Snapshot_plus_State"].values
    p5_preds = df_p10["Model_D_Snapshot_plus_Trajectory"].values
    p6_preds = df_p10["Model_E_Full_Physiological_Fusion"].values
    
    patient_ids = df_rolling["patient_id"].values
    y_715 = df_rolling["primary_label_715"].values
    y_705 = df_rolling["severe_label_705"].values
    t_del = df_rolling["time_before_delivery_min"].values
    
    df_pat_labels = df_rolling.groupby("patient_id")[["primary_label_715", "severe_label_705"]].first().reset_index()
    y_pat_715 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    y_pat_705 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["severe_label_705"].iloc[0] for p in clean_pids])
    
    # Assemble unified prediction dataframe
    model_preds_dict = {
        "P1_Compact_CTG": p1_preds,
        "P2_Sequential_Event": p2_preds,
        "P3_Snapshot_Baseline": p3_preds,
        "P4_Snapshot_plus_State": p4_preds,
        "P5_Snapshot_plus_Trajectory": p5_preds,
        "P6_Full_Physiology_System": p6_preds
    }
    
    df_unified = pd.DataFrame({
        "patient_id": patient_ids,
        "window_index": df_rolling["window_index"].values,
        "time_before_delivery_min": t_del,
        "primary_label_715": y_715,
        "severe_label_705": y_705,
        **model_preds_dict
    })
    df_unified.to_csv(os.path.join(OUT_DIR, "model_predictions.csv"), index=False)
    print("Saved model_predictions.csv")
    
    # Multi-horizon evaluation
    horizons = [60, 45, 30, 20, 10, 0]
    pat_window_indices = {pid: np.where(patient_ids == str(pid))[0] for pid in clean_pids}
    
    def get_patient_horizon_scores(score_arr, h_val):
        p_scores = []
        for pid in clean_pids:
            idx = pat_window_indices[pid]
            t_pts = t_del[idx]
            if h_val > 0:
                eligible = np.where(t_pts >= h_val)[0]
                chosen = idx[eligible[0]] if len(eligible) > 0 else idx[-1]
            else:
                chosen = idx[-1]
            p_scores.append(score_arr[chosen])
        return np.array(p_scores)
        
    model_comp_rows = []
    auprc_rows = []
    
    model_names_map = {
        "P1_Compact_CTG": "P1 (Compact CTG Baseline - DeepCTG-Inspired)",
        "P2_Sequential_Event": "P2 (Sequential / Event Baseline - Vargas-Calixto-Inspired)",
        "P3_Snapshot_Baseline": "P3 (Locked Snapshot Baseline - Continuous Clinical Huber)",
        "P4_Snapshot_plus_State": "P4 (Snapshot + Physiological State)",
        "P5_Snapshot_plus_Trajectory": "P5 (Snapshot + Deterioration Trajectory)",
        "P6_Full_Physiology_System": "P6 (Full Physiology-Guided Proposed System)"
    }
    
    for h in horizons:
        h_label = f">={h}m" if h > 0 else "Delivery (0m)"
        
        for key, display_name in model_names_map.items():
            scores = get_patient_horizon_scores(model_preds_dict[key], h)
            
            auc_715 = roc_auc_score(y_pat_715, scores)
            auprc_715 = average_precision_score(y_pat_715, scores)
            auc_705 = roc_auc_score(y_pat_705, scores)
            auprc_705 = average_precision_score(y_pat_705, scores)
            brier_715 = brier_score_loss(y_pat_715, scores)
            
            model_comp_rows.append({
                "model_code": key[:2],
                "model_name": display_name,
                "horizon": h_label,
                "horizon_min": h,
                "auroc_715": round(float(auc_715), 4),
                "auprc_715": round(float(auprc_715), 4),
                "auroc_705": round(float(auc_705), 4),
                "auprc_705": round(float(auprc_705), 4),
                "brier_score_715": round(float(brier_715), 4)
            })
            
            auprc_rows.append({
                "model_code": key[:2],
                "model_name": display_name,
                "horizon": h_label,
                "auprc_primary_715": round(float(auprc_715), 4),
                "auprc_severe_705": round(float(auprc_705), 4)
            })
            
    df_comp = pd.DataFrame(model_comp_rows)
    df_comp.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
    
    df_auprc = pd.DataFrame(auprc_rows)
    df_auprc.to_csv(os.path.join(OUT_DIR, "auprc_results.csv"), index=False)
    
    print("\nSaved model_comparison.csv and auprc_results.csv")
    print("\n--- BENCHMARK PERFORMANCE AT PRIMARY >=30m HORIZON ---")
    df_30 = df_comp[df_comp["horizon_min"] == 30][["model_code", "model_name", "auroc_715", "auprc_715", "auroc_705"]]
    print(df_30.to_string(index=False))
    
    print("\n--- BENCHMARK PERFORMANCE AT DELIVERY (0m) ---")
    df_0 = df_comp[df_comp["horizon_min"] == 0][["model_code", "model_name", "auroc_715", "auprc_715", "auroc_705"]]
    print(df_0.to_string(index=False))

if __name__ == "__main__":
    run_head_to_head_benchmark()
