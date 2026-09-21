"""
Phase 11.5 — Experiment 11.5-A: Horizon-Dependent Advantage Analysis.

Objective:
Determine whether the advantage of the proposed framework (P6) depends systematically on proximity to delivery.

Evaluates:
- Models: P1 (Compact CTG), P2 (Sequential/Event), P3 (Snapshot), P4 (Snapshot + State), P5 (Snapshot + Trajectory), P6 (Full System)
- Horizons: >=60m, >=45m, >=30m, >=20m, >=10m, Delivery (0m)
- Primary Contrasts:
  - Delta_6_2(h) = AUROC(P6, h) - AUROC(P2, h)
  - Delta_6_3(h) = AUROC(P6, h) - AUROC(P3, h)
  - Delta_6_5(h) = AUROC(P6, h) - AUROC(P5, h)
  - Delta_6_1(h) = AUROC(P6, h) - AUROC(P1, h)
  - Delta_5_3(h) = AUROC(P5, h) - AUROC(P3, h)

Statistical Engine:
- Vectorized paired patient-level bootstrap (B=2,000, seed=42)

Output:
- results/phase11_5_advantage_attribution/horizon_advantage.csv
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

OUT_DIR = "results/phase11_5_advantage_attribution"
FOLDS_PATH = "data/processed_clinical/folds.json"
PRED_PATH = "results/phase11_priorart_benchmark/model_predictions.csv"
os.makedirs(OUT_DIR, exist_ok=True)

from phase11_bootstrap import paired_patient_bootstrap, fast_auc

def run_horizon_advantage():
    print("=== EXECUTING EXPERIMENT 11.5-A: HORIZON-DEPENDENT ADVANTAGE ===")
    
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
    
    models = {
        "P1": "P1_Compact_CTG",
        "P2": "P2_Sequential_Event",
        "P3": "P3_Snapshot_Baseline",
        "P4": "P4_Snapshot_plus_State",
        "P5": "P5_Snapshot_plus_Trajectory",
        "P6": "P6_Full_Physiology_System"
    }
    
    horizons = [60, 45, 30, 20, 10, 0]
    horizon_labels = {
        60: ">=60m",
        45: ">=45m",
        30: ">=30m",
        20: ">=20m",
        10: ">=10m",
        0: "Delivery (0m)"
    }
    
    def get_scores(model_col, h_val):
        arr = df_pred[model_col].values
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
        
    contrasts = [
        ("P6 vs P2", "P6", "P2", "Full Framework vs Sequential/Event Prior-Art"),
        ("P6 vs P3", "P6", "P3", "Full Framework vs Locked Snapshot Baseline"),
        ("P6 vs P5", "P6", "P5", "Full Framework vs Snapshot + Trajectory"),
        ("P6 vs P1", "P6", "P1", "Full Framework vs Compact CTG Baseline"),
        ("P5 vs P3", "P5", "P3", "Snapshot + Trajectory vs Snapshot Baseline"),
        ("P4 vs P3", "P4", "P3", "Snapshot + State vs Snapshot Baseline")
    ]
    
    rows = []
    
    for h in horizons:
        h_str = horizon_labels[h]
        
        # Calculate individual model AUROCs & AUPRCs
        model_scores = {m_code: get_scores(col, h) for m_code, col in models.items()}
        model_aurocs = {m_code: roc_auc_score(y_pat_715, scores) for m_code, scores in model_scores.items()}
        model_auprcs = {m_code: average_precision_score(y_pat_715, scores) for m_code, scores in model_scores.items()}
        
        for contrast_name, m_b, m_a, desc in contrasts:
            scores_b = model_scores[m_b]
            scores_a = model_scores[m_a]
            
            auc_b = model_aurocs[m_b]
            auc_a = model_aurocs[m_a]
            auprc_b = model_auprcs[m_b]
            auprc_a = model_auprcs[m_a]
            
            boot_res = paired_patient_bootstrap(y_pat_715, scores_a, scores_b, n_boot=2000, seed=42)
            
            # Statistical interpretation
            if boot_res["ci_95_low"] > 0.0:
                signif = "Significant Superiority (CI > 0)"
            elif boot_res["ci_95_high"] < 0.0:
                signif = "Significant Inferiority (CI < 0)"
            else:
                signif = "Statistically Indistinguishable (CI overlaps 0)"
                
            rows.append({
                "contrast": contrast_name,
                "description": desc,
                "horizon": h_str,
                "horizon_min": h,
                "n_patients": len(clean_pids),
                "n_positives_715": int(np.sum(y_pat_715)),
                "model_b": m_b,
                "auroc_b": round(float(auc_b), 4),
                "auprc_b": round(float(auprc_b), 4),
                "model_a": m_a,
                "auroc_a": round(float(auc_a), 4),
                "auprc_a": round(float(auprc_a), 4),
                "delta_auroc_point": round(float(auc_b - auc_a), 4),
                "delta_auroc_boot_mean": round(float(boot_res["delta_mean"]), 4),
                "ci_95_low": round(float(boot_res["ci_95_low"]), 4),
                "ci_95_high": round(float(boot_res["ci_95_high"]), 4),
                "p_value": round(float(boot_res["p_value"]), 4),
                "statistical_inference": signif
            })
            
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "horizon_advantage.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_out)} rows)")
    return df_out

if __name__ == "__main__":
    run_horizon_advantage()
