"""
Phase 9B Consolidated Evaluation Suite:
Executes:
1. Nested Model Comparison (Models A, B, C, D, E)
2. Deterioration Pattern Ablation Suite (Exps 1 to 7)
3. Transient vs Persistent vs Progressive State Analysis
4. Clinical Alert Policy Optimization (Policies 1 to 5)
5. Multi-Horizon Patient-Level Evaluation & Paired Bootstrap Tests

Outputs:
- results/phase9b_deterioration/deterioration_predictions.csv
- results/phase9b_deterioration/model_comparison_metrics.csv
- results/phase9b_deterioration/ablation_metrics.csv
- results/phase9b_deterioration/state_analysis.csv
- results/phase9b_deterioration/alert_policy_metrics.csv
- results/phase9b_deterioration/calibration_metrics.csv
- results/phase9b_deterioration/warning_times.csv
- results/phase9b_deterioration/statistical_comparisons.json
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

OUT_DIR = "results/phase9b_deterioration"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
FEATURES_PATH = os.path.join(OUT_DIR, "deterioration_features.npz")
os.makedirs(OUT_DIR, exist_ok=True)

def run_evaluation():
    print("=== RUNNING PHASE 9B CONSOLIDATED EVALUATION ===")
    
    # 1. Load folds & features
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    
    feat_data = np.load(FEATURES_PATH)
    X_full = feat_data["X_deterioration"] # (8517, 112)
    research_states = feat_data["research_state"]
    multidomain_n = feat_data["multidomain_n"]
    multidomain_persist = feat_data["multidomain_persist"]
    
    patient_ids = df_rolling["patient_id"].values
    y_715 = df_rolling["primary_label_715"].values
    y_705 = df_rolling["severe_label_705"].values
    t_del = df_rolling["time_before_delivery_min"].values
    p8_risk = df_rolling["risk_prob_proxy"].values
    
    y_patient_715 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["primary_label_715"] for p in clean_pids])
    y_patient_705 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["severe_label_705"] for p in clean_pids])
    
    # Pre-define feature slices for models & ablations:
    # Full 112 features:
    # 0..18: Raw 19
    # 19..37: Deltas 19
    # 38..56: Slopes 19
    # 57..75: Accels 19
    # 76..81: Dom Sev (6)
    # 82..87: Dom Delta (6)
    # 88..93: Dom Slope (6)
    # 94..99: Dom Persist (6)
    # 100..105: Dom Accel (6)
    # 106..107: Multidomain N, Persist (2)
    # 108: Research State (1)
    # 109..111: EWMA Risk, Delta Risk, Accel Risk (3)
    
    idx_raw_19 = np.arange(0, 19)
    idx_dom_deterioration = np.arange(76, 109) # 33 features: Dom Sev, Delta, Slope, Persist, Accel, Multi N/P, State
    idx_snapshot_deterioration = np.concatenate([idx_raw_19, idx_dom_deterioration]) # 52 features
    
    model_configs = {
        "Model A (Snapshot Baseline)": None, # Phase 8 risk directly
        "Model B (Generic Temporal Baseline)": np.arange(0, 76), # Raw + Deltas + Slopes + Accels
        "Model C (Physiological Deterioration)": idx_dom_deterioration,
        "Model D (Snapshot + Deterioration)": idx_snapshot_deterioration,
        "Model E (Full Trajectory + Domain)": np.arange(0, 112)
    }
    
    ablation_configs = {
        "Exp 1: Current State Only": idx_raw_19,
        "Exp 2: State + Change": np.arange(0, 38),
        "Exp 3: State + Persistence": np.concatenate([idx_raw_19, np.arange(94, 100)]),
        "Exp 4: State + Progression": np.arange(0, 57),
        "Exp 5: Full Deterioration Signature": np.arange(0, 76),
        "Exp 6: Domain-Level Deterioration": idx_snapshot_deterioration,
        "Exp 7: Full Trajectory + Multidomain": np.arange(0, 112)
    }
    
    # 2. Cross-Validation Predictions for all configurations
    all_configs = {**model_configs, **ablation_configs}
    pred_dict = {k: np.zeros(len(df_rolling), dtype=np.float32) for k in all_configs}
    pred_dict["Model A (Snapshot Baseline)"] = p8_risk.copy()
    
    print("Executing patient-grouped 5-fold CV for all models & ablations...")
    for f_idx in range(5):
        te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
        tr_pids = set([p for p in clean_pids if p not in te_pids])
        
        train_mask = np.isin(patient_ids, list(tr_pids))
        val_mask = np.isin(patient_ids, list(te_pids))
        
        for name, feat_indices in all_configs.items():
            if feat_indices is None:
                continue # Baseline already set
                
            X_sub = X_full[:, feat_indices]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_sub[train_mask])
            X_va_s = scaler.transform(X_sub[val_mask])
            
            clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
            clf.fit(X_tr_s, y_715[train_mask])
            pred_dict[name][val_mask] = clf.predict_proba(X_va_s)[:, 1]

    # Save deterioration_predictions.csv
    df_pred_out = pd.DataFrame({
        "patient_id": patient_ids,
        "window_idx": df_rolling["window_index"].values,
        "time_before_delivery_min": t_del,
        "primary_label_715": y_715,
        "severe_label_705": y_705,
        "research_state": research_states,
        "multidomain_n": multidomain_n,
        "multidomain_persist": multidomain_persist,
        **{k.replace(" ", "_").replace("(", "").replace(")", "").replace(":", ""): v for k, v in pred_dict.items()}
    })
    df_pred_out.to_csv(os.path.join(OUT_DIR, "deterioration_predictions.csv"), index=False)
    print("Saved deterioration_predictions.csv")

    # 3. Horizon Metrics & Paired Bootstrap Tests
    horizons = [60, 45, 30, 20, 10, 0]
    
    # Pre-index windows per patient for fast horizon queries
    pat_window_indices = {pid: np.where(patient_ids == str(pid))[0] for pid in clean_pids}
    
    def evaluate_horizon(score_array, h_val):
        p_scores = []
        for pid in clean_pids:
            idx = pat_window_indices[pid]
            t_pts = t_del[idx]
            if h_val > 0:
                eligible = np.where(t_pts >= h_val)[0]
                if len(eligible) > 0:
                    chosen = idx[eligible[0]]
                else:
                    chosen = idx[-1]
            else:
                chosen = idx[-1]
            p_scores.append(score_array[chosen])
        return np.array(p_scores)

    # 3A. Nested Models Benchmark
    model_rows = []
    calib_rows = []
    stat_comparisons = {}
    
    for h in horizons:
        h_label = f">={h}m" if h > 0 else "Delivery (0m)"
        stat_comparisons[h_label] = {}
        
        base_scores = evaluate_horizon(pred_dict["Model A (Snapshot Baseline)"], h)
        base_auc = roc_auc_score(y_patient_715, base_scores)
        
        # Bootstrap setup
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

    # 3B. Ablation Experiments Suite
    ablation_rows = []
    for exp_name in ablation_configs.keys():
        for h in horizons:
            h_label = f">={h}m" if h > 0 else "Delivery (0m)"
            p_scores = evaluate_horizon(pred_dict[exp_name], h)
            auc_715 = roc_auc_score(y_patient_715, p_scores)
            auprc_715 = average_precision_score(y_patient_715, p_scores)
            auc_705 = roc_auc_score(y_patient_705, p_scores)
            
            ablation_rows.append({
                "experiment": exp_name,
                "horizon": h_label,
                "horizon_min": h,
                "auroc_715": round(float(auc_715), 4),
                "auprc_715": round(float(auprc_715), 4),
                "auroc_705": round(float(auc_705), 4)
            })
    pd.DataFrame(ablation_rows).to_csv(os.path.join(OUT_DIR, "ablation_metrics.csv"), index=False)
    print("Saved ablation_metrics.csv")

    # 4. Transient vs Persistent vs Progressive State Analysis
    state_names = {
        0: "State 0: Stable",
        1: "State 1: Emerging Abnormality",
        2: "State 2: Persistent Abnormality",
        3: "State 3: Progressive Deterioration",
        4: "State 4: Severe Multidomain Deterioration"
    }
    
    state_rows = []
    # Patient level: Maximum research state reached by patient in monitoring
    for pid in clean_pids:
        idx = pat_window_indices[pid]
        p_states = research_states[idx]
        max_st = int(np.max(p_states))
        # state at >=30m vs delivery
        t_pts = t_del[idx]
        el_30 = np.where(t_pts >= 30)[0]
        st_30 = int(p_states[el_30[0]]) if len(el_30) > 0 else int(p_states[-1])
        st_del = int(p_states[-1])
        
        state_rows.append({
            "patient_id": pid,
            "true_ph_le_715": y_patient_715[clean_pids.index(pid)],
            "true_ph_le_705": y_patient_705[clean_pids.index(pid)],
            "max_state_overall": max_st,
            "state_at_30m": st_30,
            "state_at_delivery": st_del
        })
    df_states = pd.DataFrame(state_rows)
    
    state_summary = []
    for s_val, s_name in state_names.items():
        sub_overall = df_states[df_states["max_state_overall"] == s_val]
        n_pts = len(sub_overall)
        n_acid = int(np.sum(sub_overall["true_ph_le_715"]))
        acid_prev = (n_acid / n_pts * 100.0) if n_pts > 0 else 0.0
        n_sev = int(np.sum(sub_overall["true_ph_le_705"]))
        sev_prev = (n_sev / n_pts * 100.0) if n_pts > 0 else 0.0
        
        state_summary.append({
            "state_code": s_val,
            "state_name": s_name,
            "n_patients": n_pts,
            "acidemia_cases_715": n_acid,
            "acidemia_prevalence_pct": round(acid_prev, 2),
            "severe_cases_705": n_sev,
            "severe_prevalence_pct": round(sev_prev, 2)
        })
    pd.DataFrame(state_summary).to_csv(os.path.join(OUT_DIR, "state_analysis.csv"), index=False)
    print("Saved state_analysis.csv")

    # 5. Clinical Alert Policy Optimization
    # Evaluate Policies 1 to 5 on continuous rolling monitoring
    alert_rows = []
    
    # 90% Specificity threshold on control patients for Model D (Snapshot + Deterioration)
    ctrl_mask = (y_715 == 0)
    score_model_d = pred_dict["Model D (Snapshot + Deterioration)"]
    tau_d = float(np.percentile(score_model_d[ctrl_mask], 90.0))
    tau_base = float(np.percentile(p8_risk[ctrl_mask], 90.0))
    
    policies = {
        "Policy 1: Current Threshold": lambda r, s, p, i: r[i] >= tau_d,
        "Policy 2: 2-Window Confirmation": lambda r, s, p, i: r[i] >= tau_d and (i > 0 and r[i-1] >= tau_d),
        "Policy 3: Rising-Risk Policy": lambda r, s, p, i: (r[i] - r[max(0, i-2)]) > 0.05,
        "Policy 4: Sustained Rising-Risk": lambda r, s, p, i: r[i] >= tau_d and (r[i] - r[max(0, i-2)]) >= 0.0,
        "Policy 5: Physiological Deterioration Policy": lambda r, s, p, i: (s[i] >= 2 and r[i] >= (tau_d * 0.85))
    }
    
    total_monitoring_hours = (len(df_rolling) * 2.5) / 60.0 # hours
    pos_pids = set([p for p in clean_pids if df_rolling[df_rolling["patient_id"] == p].iloc[0]["primary_label_715"] == 1])
    neg_pids = set(clean_pids) - pos_pids
    
    for pol_name, pol_func in policies.items():
        patient_alerts = {}
        first_alert_times = {}
        total_false_alarms = 0
        
        for pid in clean_pids:
            idx = pat_window_indices[pid]
            p_r = score_model_d[idx]
            p_s = research_states[idx]
            p_p = multidomain_persist[idx]
            p_tdel = t_del[idx]
            
            p_flagged = False
            first_t = None
            
            for i in range(len(idx)):
                is_alert = pol_func(p_r, p_s, p_p, i)
                if is_alert:
                    if not p_flagged:
                        p_flagged = True
                        first_t = p_tdel[i]
                    if pid in neg_pids:
                        total_false_alarms += 1
                        
            patient_alerts[pid] = p_flagged
            if p_flagged and pid in pos_pids:
                first_alert_times[pid] = first_t
                
        # Calculate sensitivity, specificity, false alert rates
        tp = sum([1 for p in pos_pids if patient_alerts[p]])
        fn = len(pos_pids) - tp
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

    # 6. Warning Times Distribution
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
    print("=== PHASE 9B CONSOLIDATED EVALUATION COMPLETE ===")

if __name__ == "__main__":
    run_evaluation()
