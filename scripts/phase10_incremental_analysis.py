"""
Phase 10 Work Package 10B: Incremental Contribution & Clustered Bootstrap Analysis.

Evaluates:
- Comparison 1: Snapshot vs Snapshot + State (R_t vs R_t + S_t)
- Comparison 2: Snapshot vs Snapshot + Trajectory (R_t vs R_t + V_t)
- Comparison 3: Snapshot vs Snapshot + State + Trajectory (R_t vs R_t + S_t + V_t)
- Comparison 4: Snapshot vs Full Physiological Fusion
- Multi-horizon evaluation: >=60m, >=45m, >=30m, >=20m, >=10m, Delivery (0m)
- Fast vectorized patient-level clustered bootstrap (B=2,000) for delta AUROC, 95% CIs, and empirical p-values.

Outputs:
- results/phase10_final/incremental_information.csv
- results/phase10_final/bootstrap_results.csv
- results/phase10_final/final_model_comparison.csv
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

OUT_DIR = "results/phase10_final"
PRED_PATH = os.path.join(OUT_DIR, "final_predictions.csv")
FOLDS_PATH = "data/processed_clinical/folds.json"
os.makedirs(OUT_DIR, exist_ok=True)

def fast_auc_vec(y_true, y_score):
    """Fast rank-sum AUROC calculation."""
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(y_score).rank().values
    pos_ranks = np.sum(ranks[y_true == 1])
    return (pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def run_incremental_analysis():
    print("=== EXECUTING WORK PACKAGE 10B: INCREMENTAL CONTRIBUTION ANALYSIS ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_pred = pd.read_csv(PRED_PATH)
    df_pred["patient_id"] = df_pred["patient_id"].astype(str)
    
    patient_ids = df_pred["patient_id"].values
    t_del = df_pred["time_before_delivery_min"].values
    
    df_pat_labels = df_pred.groupby("patient_id")[["primary_label_715", "severe_label_705"]].first().reset_index()
    y_patient_715 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    y_patient_705 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["severe_label_705"].iloc[0] for p in clean_pids])
    
    model_columns = {
        "Model A (Snapshot Baseline)": "Model_A_Snapshot_Baseline",
        "Model B (Snapshot + State)": "Model_B_Snapshot_plus_State",
        "Model C (Snapshot + State + Trajectory)": "Model_C_Snapshot_plus_State_plus_Trajectory",
        "Model D (Snapshot + Trajectory)": "Model_D_Snapshot_plus_Trajectory",
        "Model E (Full Physiological Fusion)": "Model_E_Full_Physiological_Fusion"
    }
    
    horizons = [60, 45, 30, 20, 10, 0]
    pat_window_indices = {pid: np.where(patient_ids == str(pid))[0] for pid in clean_pids}
    
    def get_horizon_patient_scores(col_name, h_val):
        score_arr = df_pred[col_name].values
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

    model_rows = []
    boot_rows = []
    incremental_rows = []
    
    np.random.seed(42)
    n_boot = 2000
    n_pts = len(clean_pids)
    
    boot_indices = [np.random.choice(n_pts, size=n_pts, replace=True) for _ in range(n_boot)]
    boot_indices = [idx for idx in boot_indices if len(np.unique(y_patient_715[idx])) >= 2]
    
    for h in horizons:
        h_label = f">={h}m" if h > 0 else "Delivery (0m)"
        
        base_scores = get_horizon_patient_scores(model_columns["Model A (Snapshot Baseline)"], h)
        base_auc_715 = roc_auc_score(y_patient_715, base_scores)
        base_boot_aucs = np.array([fast_auc_vec(y_patient_715[b_idx], base_scores[b_idx]) for b_idx in boot_indices])
        
        for m_name, col in model_columns.items():
            p_scores = get_horizon_patient_scores(col, h)
            
            auc_715 = roc_auc_score(y_patient_715, p_scores)
            auprc_715 = average_precision_score(y_patient_715, p_scores)
            auc_705 = roc_auc_score(y_patient_705, p_scores)
            auprc_705 = average_precision_score(y_patient_705, p_scores)
            brier = brier_score_loss(y_patient_715, p_scores)
            
            model_rows.append({
                "model": m_name,
                "horizon": h_label,
                "horizon_min": h,
                "auroc_715": round(float(auc_715), 4),
                "auprc_715": round(float(auprc_715), 4),
                "auroc_705": round(float(auc_705), 4),
                "auprc_705": round(float(auprc_705), 4),
                "brier_score": round(float(brier), 4)
            })
            
            # Bootstrap vs Baseline
            m_boot_aucs = np.array([fast_auc_vec(y_patient_715[b_idx], p_scores[b_idx]) for b_idx in boot_indices])
            deltas = m_boot_aucs - base_boot_aucs
            
            diff_mean = float(np.mean(deltas))
            ci_low = float(np.percentile(deltas, 2.5))
            ci_high = float(np.percentile(deltas, 97.5))
            p_val = float(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
            
            boot_rows.append({
                "model": m_name,
                "horizon": h_label,
                "horizon_min": h,
                "auroc_715": round(float(auc_715), 4),
                "delta_vs_snapshot": round(diff_mean, 4),
                "ci95_low": round(ci_low, 4),
                "ci95_high": round(ci_high, 4),
                "p_value": round(p_val, 4),
                "severe_auroc_705": round(float(auc_705), 4)
            })
            
            if m_name != "Model A (Snapshot Baseline)":
                # Evidence level
                if ci_low > 0:
                    evidence = "Strong Positive (CI excludes 0)"
                elif diff_mean > 0:
                    evidence = "Moderate Positive (CI overlaps 0)"
                elif ci_high < 0:
                    evidence = "Negative"
                else:
                    evidence = "No Meaningful Difference"
                    
                incremental_rows.append({
                    "comparison": f"{m_name} vs Snapshot Baseline",
                    "horizon": h_label,
                    "horizon_min": h,
                    "snapshot_auroc": round(float(base_auc_715), 4),
                    "model_auroc": round(float(auc_715), 4),
                    "delta_auroc": round(diff_mean, 4),
                    "ci95": f"[{round(ci_low, 4)}, {round(ci_high, 4)}]",
                    "p_value": round(p_val, 4),
                    "evidence_level": evidence
                })

    df_models = pd.DataFrame(model_rows)
    df_models.to_csv(os.path.join(OUT_DIR, "final_model_comparison.csv"), index=False)
    
    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(os.path.join(OUT_DIR, "bootstrap_results.csv"), index=False)
    
    df_inc = pd.DataFrame(incremental_rows)
    df_inc.to_csv(os.path.join(OUT_DIR, "incremental_information.csv"), index=False)
    
    print("\nSaved final_model_comparison.csv, bootstrap_results.csv, incremental_information.csv")
    print("\n--- INCREMENTAL VALUE AT >=30m HORIZON ---")
    df_inc_30 = df_inc[df_inc["horizon_min"] == 30]
    print(df_inc_30.to_string(index=False))
    
    print("\n--- INCREMENTAL VALUE AT DELIVERY (0m) ---")
    df_inc_0 = df_inc[df_inc["horizon_min"] == 0]
    print(df_inc_0.to_string(index=False))
    
    print("\nWork Package 10B Complete.")

if __name__ == "__main__":
    run_incremental_analysis()
