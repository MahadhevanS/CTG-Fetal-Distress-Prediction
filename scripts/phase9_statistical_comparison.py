"""
Phase 9 Statistical Comparison Script:
Performs paired patient-level bootstrap tests between Phase 9 candidate models and Phase 8 baseline.
Generates results/phase9_temporal/statistical_comparisons.json and horizon_metrics.csv.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

RESULTS_DIR = "results/phase9_temporal"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_statistical_comparison():
    print("=== RUNNING PHASE 9 STATISTICAL COMPARISON & BOOTSTRAP AUDIT ===")
    
    # Load patient-level predictions from Phase 8 baseline
    df_p8 = pd.read_csv("results/phase8_rolling/rolling_predictions.csv")
    
    # Load Phase 9 model predictions
    df_lr = pd.read_csv("results/phase9_temporal/temporal_lr_predictions.csv")
    df_gbm = pd.read_csv("results/phase9_temporal/temporal_gbm_predictions.csv")
    df_gru = pd.read_csv("results/phase9_temporal/sequential_gru_predictions.csv")
    df_traj = pd.read_csv("results/phase9_temporal/prediction_trajectory_predictions.csv")
    df_mh = pd.read_csv("results/phase9_temporal/multihorizon_predictions.csv")
    
    horizons = [60, 45, 30, 20, 10, 0]
    
    # Consolidate all metrics into horizon_metrics.csv
    rows = []
    models = {
        "Phase 8 Frozen Baseline": df_p8["p8_risk"].values if "p8_risk" in df_p8 else df_p8["risk_score"].values,
        "Model 1: Temporal LR": df_lr["risk_score"].values,
        "Model 2: Temporal GBM": df_gbm["risk_score"].values,
        "Model 3: Trajectory EWMA": df_traj["risk_score"].values,
        "Model 4: Sequential GRU": df_gru["risk_score"].values,
        "Model 5: Multi-Horizon": df_mh["risk_score"].values,
    }
    
    patient_ids = df_p8["patient_id"].values
    y_true_715 = df_p8["ph_le_715"].values
    y_true_705 = df_p8["ph_le_705"].values
    t_to_del = df_p8["time_to_delivery_min"].values
    
    # Compute patient-level aggregation for each horizon
    unique_pids = np.unique(patient_ids)
    
    comparison_results = {}
    
    for h in horizons:
        h_mask = (t_to_del >= h) if h > 0 else np.ones(len(patient_ids), dtype=bool)
        h_name = f">={h}m" if h > 0 else "Delivery (0m)"
        comparison_results[h_name] = {}
        
        # Filter patients who have at least one window in this horizon
        h_pids = np.unique(patient_ids[h_mask])
        
        # Max-pool per patient in horizon
        p_targets_715 = []
        p_targets_705 = []
        p_scores = {m: [] for m in models}
        
        for pid in h_pids:
            p_mask = (patient_ids == pid) & h_mask
            p_targets_715.append(y_true_715[p_mask][0])
            p_targets_705.append(y_true_705[p_mask][0])
            for m in models:
                p_scores[m].append(np.max(models[m][p_mask]))
                
        p_targets_715 = np.array(p_targets_715)
        p_targets_705 = np.array(p_targets_705)
        for m in models:
            p_scores[m] = np.array(p_scores[m])
            
        # Baseline scores
        base_scores = p_scores["Phase 8 Frozen Baseline"]
        base_auc_715 = roc_auc_score(p_targets_715, base_scores)
        
        # Paired bootstrap for each candidate model
        np.random.seed(42)
        n_boot = 2000
        n_patients = len(h_pids)
        
        for m in models:
            auc_715 = roc_auc_score(p_targets_715, p_scores[m])
            auprc_715 = average_precision_score(p_targets_715, p_scores[m])
            auc_705 = roc_auc_score(p_targets_705, p_scores[m])
            
            rows.append({
                "model": m,
                "horizon": h_name,
                "horizon_min": h,
                "n_patients": n_patients,
                "auroc_715": float(auc_715),
                "auprc_715": float(auprc_715),
                "auroc_705": float(auc_705)
            })
            
            # Bootstrap difference vs Phase 8
            deltas = []
            for _ in range(n_boot):
                idx = np.random.choice(n_patients, size=n_patients, replace=True)
                # Ensure both classes exist
                if len(np.unique(p_targets_715[idx])) < 2:
                    continue
                boot_base_auc = roc_auc_score(p_targets_715[idx], base_scores[idx])
                boot_m_auc = roc_auc_score(p_targets_715[idx], p_scores[m][idx])
                deltas.append(boot_m_auc - boot_base_auc)
                
            deltas = np.array(deltas)
            diff_mean = float(np.mean(deltas))
            ci_low = float(np.percentile(deltas, 2.5))
            ci_high = float(np.percentile(deltas, 97.5))
            p_val = float(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
            
            comparison_results[h_name][m] = {
                "auroc_715": float(auc_715),
                "auprc_715": float(auprc_715),
                "auroc_705": float(auc_705),
                "delta_vs_p8": diff_mean,
                "ci_95": [ci_low, ci_high],
                "p_value": p_val
            }
            
    # Save CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(RESULTS_DIR, "horizon_metrics.csv"), index=False)
    print("Saved horizon_metrics.csv")
    
    # Save JSON
    with open(os.path.join(RESULTS_DIR, "statistical_comparisons.json"), "w") as f:
        json.dump(comparison_results, f, indent=2)
    print("Saved statistical_comparisons.json")
    
if __name__ == "__main__":
    run_statistical_comparison()
