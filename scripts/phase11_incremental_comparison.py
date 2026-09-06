"""
Phase 11: Pairwise Superiority & Bootstrap Comparison Suite.

Executes:
- Clustered patient-level paired bootstrap (B=2,000) for all pre-specified pairwise comparisons:
  - P6 vs P1 (Full System vs Compact CTG Baseline)
  - P6 vs P2 (Full System vs Sequential / Event Baseline)
  - P6 vs P3 (Full System vs Locked Snapshot Baseline)
  - P5 vs P3 (Snapshot + Trajectory vs Snapshot Baseline)
  - P4 vs P3 (Snapshot + State vs Snapshot Baseline)
  - P6 vs P5 (Full System vs Snapshot + Trajectory)
- Evaluated across primary horizon (>=30m) and delivery (0m).
- Classifies evidence according to predefined Section 21 criteria:
  - Level 1: Statistically Significant Superiority (CI excludes 0)
  - Level 2: Higher Point Estimate Without Significance (CI overlaps 0)
  - Level 3: Operational Superiority / Comparable Performance

Outputs:
- results/phase11_priorart_benchmark/pairwise_bootstrap.csv
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT_DIR = "results/phase11_priorart_benchmark"
PRED_PATH = os.path.join(OUT_DIR, "model_predictions.csv")
FOLDS_PATH = "data/processed_clinical/folds.json"
os.makedirs(OUT_DIR, exist_ok=True)

from phase11_bootstrap import paired_patient_bootstrap

def run_pairwise_comparisons():
    print("=== EXECUTING PHASE 11: PAIRWISE SUPERIORITY BOOTSTRAP ANALYSIS ===")
    
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
        
    pairwise_specs = [
        ("P6 vs P1", "P6_Full_Physiology_System", "P1_Compact_CTG", "Full Proposed System vs DeepCTG-Inspired Compact CTG"),
        ("P6 vs P2", "P6_Full_Physiology_System", "P2_Sequential_Event", "Full Proposed System vs Vargas-Calixto-Inspired Sequential Baseline"),
        ("P6 vs P3", "P6_Full_Physiology_System", "P3_Snapshot_Baseline", "Full Proposed System vs Locked Snapshot Baseline"),
        ("P5 vs P3", "P5_Snapshot_plus_Trajectory", "P3_Snapshot_Baseline", "Trajectory Contribution vs Snapshot Baseline"),
        ("P4 vs P3", "P4_Snapshot_plus_State", "P3_Snapshot_Baseline", "Physiological State Contribution vs Snapshot Baseline"),
        ("P6 vs P5", "P6_Full_Physiology_System", "P5_Snapshot_plus_Trajectory", "Full Multi-Domain Context vs Trajectory-Only")
    ]
    
    horizons = [30, 0, 20, 10]
    horizon_labels = {30: ">=30m (Primary Early-Warning)", 0: "Delivery (0m Endpoint)", 20: ">=20m", 10: ">=10m"}
    
    comparison_rows = []
    
    for h in horizons:
        h_tag = horizon_labels[h]
        
        for comp_name, col_b, col_a, desc in pairwise_specs:
            scores_b = get_patient_scores(col_b, h)
            scores_a = get_patient_scores(col_a, h)
            
            auc_b = roc_auc_score(y_pat_715, scores_b)
            auc_a = roc_auc_score(y_pat_715, scores_a)
            
            boot_res = paired_patient_bootstrap(y_pat_715, scores_a, scores_b, n_boot=2000, seed=42)
            
            delta_mean = boot_res["delta_mean"]
            ci_lo = boot_res["ci_95_low"]
            ci_hi = boot_res["ci_95_high"]
            p_val = boot_res["p_value"]
            
            # Evidence level determination (Section 21)
            if ci_lo > 0.0:
                evidence = "Level 1: Statistically Significant Superiority (Outperformed)"
            elif delta_mean > 0.0:
                evidence = "Level 2: Higher Point Estimate without Significance"
            elif ci_hi < 0.0:
                evidence = "Inferior (Comparator significantly higher)"
            else:
                evidence = "Level 3: Comparable / No Statistically Meaningful Difference"
                
            comparison_rows.append({
                "comparison": comp_name,
                "description": desc,
                "horizon": h_tag,
                "horizon_min": h,
                "model_b_auroc": round(float(auc_b), 4),
                "model_a_auroc": round(float(auc_a), 4),
                "delta_auroc": round(float(delta_mean), 4),
                "ci95_low": round(float(ci_lo), 4),
                "ci95_high": round(float(ci_hi), 4),
                "p_value": round(float(p_val), 4),
                "evidence_level": evidence
            })
            
    df_pairwise = pd.DataFrame(comparison_rows)
    df_pairwise.to_csv(os.path.join(OUT_DIR, "pairwise_bootstrap.csv"), index=False)
    print("Saved pairwise_bootstrap.csv")
    
    print("\n--- PRIMARY >=30m SUPERIORITY COMPARISONS ---")
    df_p30 = df_pairwise[df_pairwise["horizon_min"] == 30][["comparison", "model_b_auroc", "model_a_auroc", "delta_auroc", "ci95_low", "ci95_high", "p_value", "evidence_level"]]
    print(df_p30.to_string(index=False))
    
    print("\n--- DELIVERY (0m) SUPERIORITY COMPARISONS ---")
    df_p0 = df_pairwise[df_pairwise["horizon_min"] == 0][["comparison", "model_b_auroc", "model_a_auroc", "delta_auroc", "ci95_low", "ci95_high", "p_value", "evidence_level"]]
    print(df_p0.to_string(index=False))

if __name__ == "__main__":
    run_pairwise_comparisons()
