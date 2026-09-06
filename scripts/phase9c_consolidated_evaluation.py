"""
Phase 9C Consolidated Evaluation Suite:
Executes:
- Exp 10: Snapshot vs State Trajectory Models (Models 1 to 5)
- Exp 11: Prediction Trajectory vs State Trajectory Comparison
- Exp 12: State-Based & Hybrid Alert Policies (Policies 1 to 5)
- Multi-Horizon Evaluation (>=60m, >=45m, >=30m, >=20m, >=10m, Delivery)
- Paired Patient-Level Bootstrap Statistical Significance Tests (B=2,000)

Outputs:
- results/phase9c_state_trajectory/state_trajectory_predictions.csv
- results/phase9c_state_trajectory/model_comparison_metrics.csv
- results/phase9c_state_trajectory/alert_policy_metrics.csv
- results/phase9c_state_trajectory/calibration_metrics.csv
- results/phase9c_state_trajectory/warning_times.csv
- results/phase9c_state_trajectory/statistical_comparisons.json
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

OUT_DIR = "results/phase9c_state_trajectory"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
FEATURES_PATH = os.path.join(OUT_DIR, "state_trajectory_features.npz")
os.makedirs(OUT_DIR, exist_ok=True)

def run_consolidated():
    print("=== RUNNING PHASE 9C CONSOLIDATED EVALUATION ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    
    data = np.load(FEATURES_PATH)
    X_all_40 = data["X_state_trajectory"] # (8517, 40)
    research_states = data["research_states"]
    vel_1step = data["vel_1step"]
    state_persist = data["state_persist"]
    
    patient_ids = df_rolling["patient_id"].values
    y_715 = df_rolling["primary_label_715"].values
    y_705 = df_rolling["severe_label_705"].values
    t_del = df_rolling["time_before_delivery_min"].values
    p8_risk = df_rolling["risk_prob_proxy"].values
    
    y_patient_715 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["primary_label_715"] for p in clean_pids])
    y_patient_705 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["severe_label_705"] for p in clean_pids])
    
    # Feature indices:
    # 0..18: Raw 19 features
    # 19..24: Domain severities (6)
    # 25: Current State S_t (1)
    # 26: Vel 1-step
    # 27: Vel 4-step
    # 28: Accel
    # 29: Persist
    # 30: Reversal
    # 31: Multi N
    # 32: Multi C
    # 33..37: Occupancy P0..P4 (5)
    # 38: p8_baseline_risk
    # 39: ewma_risk
    
    idx_state_only = [25]
    idx_state_traj = [25, 26, 27, 28, 29, 30]
    idx_feat_traj = list(range(0, 38)) # Raw 19 + Dom 6 + State + Traj 7 + Occ 5
    idx_pred_traj = [38, 39]
    idx_combined = list(range(0, 40))
    
    model_configs = {
        "Phase 8 Snapshot Baseline": None,
        "Model 1 (Current State Only)": idx_state_only,
        "Model 2 (Current State + Trajectory)": idx_state_traj,
        "Model 3 (Current Features + Trajectory)": idx_feat_traj,
        "Model 4 (Prediction Trajectory Only)": idx_pred_traj,
        "Model 5 (Combined Prediction + State Trajectory)": idx_combined
    }
    
    # Cross-Validation Predictions
    pred_dict = {k: np.zeros(len(df_rolling), dtype=np.float32) for k in model_configs}
    pred_dict["Phase 8 Snapshot Baseline"] = p8_risk.copy()
    
    print("Executing patient-grouped 5-fold CV for Phase 9C models...")
    for f_idx in range(5):
        te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
        tr_pids = set([p for p in clean_pids if p not in te_pids])
        
        train_mask = np.isin(patient_ids, list(tr_pids))
        val_mask = np.isin(patient_ids, list(te_pids))
        
        for name, feat_indices in model_configs.items():
            if feat_indices is None:
                continue
                
            X_sub = X_all_40[:, feat_indices]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_sub[train_mask])
            X_va_s = scaler.transform(X_sub[val_mask])
            
            clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
            clf.fit(X_tr_s, y_715[train_mask])
            pred_dict[name][val_mask] = clf.predict_proba(X_va_s)[:, 1]

    # Save state_trajectory_predictions.csv
    df_pred_out = pd.DataFrame({
        "patient_id": patient_ids,
        "window_idx": df_rolling["window_index"].values,
        "time_before_delivery_min": t_del,
        "primary_label_715": y_715,
        "severe_label_705": y_705,
        "research_state": research_states,
        "vel_1step": vel_1step,
        "state_persist": state_persist,
        **{k.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("+", "plus"): v for k, v in pred_dict.items()}
    })
    df_pred_out.to_csv(os.path.join(OUT_DIR, "state_trajectory_predictions.csv"), index=False)
    print("Saved state_trajectory_predictions.csv")

    # Horizon Metrics & Paired Bootstrap Tests
    horizons = [60, 45, 30, 20, 10, 0]
    pat_window_indices = {pid: np.where(patient_ids == str(pid))[0] for pid in clean_pids}
    
    def evaluate_horizon(score_array, h_val):
        p_scores = []
        for pid in clean_pids:
            idx = pat_window_indices[pid]
            t_pts = t_del[idx]
            if h_val > 0:
                eligible = np.where(t_pts >= h_val)[0]
                chosen = idx[eligible[0]] if len(eligible) > 0 else idx[-1]
            else:
                chosen = idx[-1]
            p_scores.append(score_array[chosen])
        return np.array(p_scores)

    model_rows = []
    calib_rows = []
    stat_comparisons = {}
    
    for h in horizons:
        h_label = f">={h}m" if h > 0 else "Delivery (0m)"
        stat_comparisons[h_label] = {}
        
        base_scores = evaluate_horizon(pred_dict["Phase 8 Snapshot Baseline"], h)
        base_auc = roc_auc_score(y_patient_715, base_scores)
        
        np.random.seed(42)
        n_boot = 2000
        n_pts = len(clean_pids)
        
        for m_name in model_configs.keys():
            p_scores = evaluate_horizon(pred_dict[m_name], h)
            auc_715 = roc_auc_score(y_patient_715, p_scores)
            auprc_715 = average_precision_score(y_patient_715, p_scores)
            auc_705 = roc_auc_score(y_patient_705, p_scores)
            brier = brier_score_loss(y_patient_715, p_scores)
            
            s_pts = np.clip(p_scores.copy(), 1e-6, 1 - 1e-6)
            logits = np.log(s_pts / (1 - s_pts)).reshape(-1, 1)
            cal_lr = LogisticRegression()
            cal_lr.fit(logits, y_patient_715)
            cal_slope = float(cal_lr.coef_[0][0])
            cal_intercept = float(cal_lr.intercept_[0])
            
            model_rows.append({
                "model": m_name,
                "horizon": h_label,
                "horizon_min": h,
                "auroc_715": round(float(auc_715), 4),
                "auprc_715": round(float(auprc_715), 4),
                "auroc_705": round(float(auc_705), 4),
                "brier_score": round(float(brier), 4)
            })
            
            calib_rows.append({
                "model": m_name,
                "horizon": h_label,
                "brier_score": round(float(brier), 4),
                "calibration_slope": round(cal_slope, 3),
                "calibration_intercept": round(cal_intercept, 3)
            })
            
            # Paired Bootstrap
            deltas = []
            for _ in range(n_boot):
                b_idx = np.random.choice(n_pts, size=n_pts, replace=True)
                if len(np.unique(y_patient_715[b_idx])) < 2:
                    continue
                b_base = roc_auc_score(y_patient_715[b_idx], base_scores[b_idx])
                b_m = roc_auc_score(y_patient_715[b_idx], p_scores[b_idx])
                deltas.append(b_m - b_base)
                
            deltas = np.array(deltas)
            diff_mean = float(np.mean(deltas))
            ci_low = float(np.percentile(deltas, 2.5))
            ci_high = float(np.percentile(deltas, 97.5))
            p_val = float(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
            
            stat_comparisons[h_label][m_name] = {
                "auroc_715": round(float(auc_715), 4),
                "auprc_715": round(float(auprc_715), 4),
                "auroc_705": round(float(auc_705), 4),
                "delta_vs_baseline": round(diff_mean, 4),
                "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                "p_value": round(p_val, 4)
            }
            
    pd.DataFrame(model_rows).to_csv(os.path.join(OUT_DIR, "model_comparison_metrics.csv"), index=False)
    pd.DataFrame(calib_rows).to_csv(os.path.join(OUT_DIR, "calibration_metrics.csv"), index=False)
    with open(os.path.join(OUT_DIR, "statistical_comparisons.json"), "w") as f:
        json.dump(stat_comparisons, f, indent=2)
    print("Saved model_comparison_metrics.csv, calibration_metrics.csv, statistical_comparisons.json")

    # Alert Policy Optimization (Policies 1 to 5)
    alert_rows = []
    ctrl_mask = (y_715 == 0)
    score_combined = pred_dict["Model 5 (Combined Prediction + State Trajectory)"]
    tau_comb = float(np.percentile(score_combined[ctrl_mask], 90.0))
    tau_base = float(np.percentile(p8_risk[ctrl_mask], 90.0))
    
    policies = {
        "Policy 1: Threshold-Only": lambda r, s, v, p, i: r[i] >= tau_comb,
        "Policy 2: 2-Window Persistence": lambda r, s, v, p, i: r[i] >= tau_comb and (i > 0 and r[i-1] >= tau_comb),
        "Policy 3: State-Based Alert": lambda r, s, v, p, i: s[i] >= 3 and (i > 0 and s[i-1] >= 3),
        "Policy 4: State Progression Alert": lambda r, s, v, p, i: s[i] >= 3 and v[i] >= 0,
        "Policy 5: Hybrid State-Prediction Alert": lambda r, s, v, p, i: r[i] >= (tau_comb * 0.85) and s[i] >= 2 and p[i] >= 2
    }
    
    total_monitoring_hours = (len(df_rolling) * 2.5) / 60.0
    pos_pids = set([p for p in clean_pids if df_rolling[df_rolling["patient_id"] == p].iloc[0]["primary_label_715"] == 1])
    neg_pids = set(clean_pids) - pos_pids
    
    for pol_name, pol_func in policies.items():
        patient_alerts = {}
        first_alert_times = {}
        total_false_alarms = 0
        
        for pid in clean_pids:
            idx = pat_window_indices[pid]
            p_r = score_combined[idx]
            p_s = research_states[idx]
            p_v = vel_1step[idx]
            p_p = state_persist[idx]
            p_tdel = t_del[idx]
            
            p_flagged = False
            first_t = None
            
            for i in range(len(idx)):
                is_alert = pol_func(p_r, p_s, p_v, p_p, i)
                if is_alert:
                    if not p_flagged:
                        p_flagged = True
                        first_t = p_tdel[i]
                    if pid in neg_pids:
                        total_false_alarms += 1
                        
            patient_alerts[pid] = p_flagged
            if p_flagged and pid in pos_pids:
                first_alert_times[pid] = first_t
                
        tp = sum([1 for p in pos_pids if patient_alerts[p]])
        fp = sum([1 for p in neg_pids if patient_alerts[p]])
        tn = len(neg_pids) - fp
        
        sens = (tp / len(pos_pids)) * 100.0
        spec = (tn / len(neg_pids)) * 100.0
        fa_rate_hr = total_false_alarms / total_monitoring_hours
        fa_per_pt = total_false_alarms / len(clean_pids)
        
        lead_times = list(first_alert_times.values())
        med_lead = float(np.median(lead_times)) if len(lead_times) > 0 else 0.0
        
        alert_rows.append({
            "policy_name": pol_name,
            "sensitivity_pct": round(sens, 2),
            "specificity_pct": round(spec, 2),
            "false_alert_rate_per_hour": round(fa_rate_hr, 3),
            "false_alerts_per_patient": round(fa_per_pt, 2),
            "total_false_alarms": total_false_alarms,
            "median_lead_time_min": round(med_lead, 1)
        })
    pd.DataFrame(alert_rows).to_csv(os.path.join(OUT_DIR, "alert_policy_metrics.csv"), index=False)
    print("Saved alert_policy_metrics.csv")

    # Warning Lead Times for Models
    warn_rows = []
    for m_name in model_configs.keys():
        p_sc = pred_dict[m_name]
        tau = np.percentile(p_sc[ctrl_mask], 90.0)
        
        warn_times = []
        det_60 = 0
        det_45 = 0
        det_30 = 0
        det_20 = 0
        det_10 = 0
        det_any = 0
        
        for pid in pos_pids:
            idx = pat_window_indices[pid]
            p_scores = p_sc[idx]
            p_tdel = t_del[idx]
            
            crossed = np.where(p_scores >= tau)[0]
            if len(crossed) > 0:
                det_any += 1
                earliest_t = np.max(p_tdel[crossed])
                warn_times.append(earliest_t)
                if earliest_t >= 60: det_60 += 1
                if earliest_t >= 45: det_45 += 1
                if earliest_t >= 30: det_30 += 1
                if earliest_t >= 20: det_20 += 1
                if earliest_t >= 10: det_10 += 1
                
        n_pos = len(pos_pids)
        warn_rows.append({
            "model": m_name,
            "sensitivity_overall_pct": round(100.0 * det_any / n_pos, 2),
            "median_warning_min": round(float(np.median(warn_times)), 1) if len(warn_times) > 0 else 0.0,
            "iqr_warning_min": round(float(np.percentile(warn_times, 75) - np.percentile(warn_times, 25)), 1) if len(warn_times) > 0 else 0.0,
            "mean_warning_min": round(float(np.mean(warn_times)), 1) if len(warn_times) > 0 else 0.0,
            "detected_ge_60m_pct": round(100.0 * det_60 / n_pos, 2),
            "detected_ge_45m_pct": round(100.0 * det_45 / n_pos, 2),
            "detected_ge_30m_pct": round(100.0 * det_30 / n_pos, 2),
            "detected_ge_20m_pct": round(100.0 * det_20 / n_pos, 2),
            "detected_ge_10m_pct": round(100.0 * det_10 / n_pos, 2)
        })
    pd.DataFrame(warn_rows).to_csv(os.path.join(OUT_DIR, "warning_times.csv"), index=False)
    print("Saved warning_times.csv")
    print("=== PHASE 9C CONSOLIDATED EVALUATION COMPLETE ===")

if __name__ == "__main__":
    run_consolidated()
