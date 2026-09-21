"""
Phase 11.5 — Experiment 11.5-D: Progression versus Reversal Analysis.

Objective:
Determine whether the direction of physiological evolution differentiates subsequent acidemia risk
among patients who are in comparable current physiological states.

Hypothesis H3 (Association Hypothesis):
P(Y = 1 | S_t, Progressing) > P(Y = 1 | S_t, Reversing)

Methodology:
- For each research state (Stable=0, Emerging=1, Persistent=2, Progressive=3, Severe=4):
  - Stratify windows into Progressing (velocity > 0) vs Reversing (reversal_ind == 1 or velocity < 0).
  - Calculate patient-level clustered acidemia prevalence (pH <= 7.15) and severe acidemia (pH <= 7.05).
  - Estimate prevalence difference Delta_risk = Prev(Prog) - Prev(Rev) and Odds Ratio.
  - Compute patient-clustered bootstrap (B=2,000) 95% confidence intervals and empirical p-values.

Output:
- results/phase11_5_advantage_attribution/progression_reversal_analysis.csv
"""

import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "results/phase11_5_advantage_attribution"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
TRAJ_FEATURES_PATH = "results/phase9c_state_trajectory/state_trajectory_features.npz"
os.makedirs(OUT_DIR, exist_ok=True)

def run_progression_reversal():
    print("=== EXECUTING EXPERIMENT 11.5-D: PROGRESSION VS REVERSAL ANALYSIS ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    
    data_traj = np.load(TRAJ_FEATURES_PATH)
    research_states = data_traj["research_states"]
    vel_1step = data_traj["vel_1step"]
    reversal_ind = data_traj["reversal_ind"]
    
    df_rolling["research_state"] = research_states
    df_rolling["vel_1step"] = vel_1step
    df_rolling["reversal_ind"] = reversal_ind
    
    # State mapping
    state_names = {
        0: "State 0: Stable Autonomic",
        1: "State 1: Emerging Stress",
        2: "State 2: Persistent Decelerative",
        3: "State 3: Progressive Multidomain",
        4: "State 4: Severe Decompensation"
    }
    
    # Classify trajectory direction:
    # Progressing: vel_1step > 0
    # Reversing: reversal_ind == 1 or vel_1step < 0
    # Stationary: vel_1step == 0 and reversal_ind == 0
    directions = []
    for v, r in zip(vel_1step, reversal_ind):
        if v > 0.05:
            directions.append("Progressing")
        elif r == 1.0 or v < -0.05:
            directions.append("Reversing")
        else:
            directions.append("Stationary")
    df_rolling["trajectory_direction"] = directions
    
    pids_array = df_rolling["patient_id"].values
    y_715_array = df_rolling["primary_label_715"].values
    y_705_array = df_rolling["severe_label_705"].values
    risk_array = df_rolling["risk_prob_proxy"].values
    
    # Patient-clustered bootstrap function for state-stratified difference
    def bootstrap_state_comparison(df_sub, n_boot=2000, seed=42):
        rng = np.random.default_rng(seed)
        unique_pids = np.unique(df_sub["patient_id"].values)
        n_pts = len(unique_pids)
        
        prog_pids = set(df_sub[df_sub["trajectory_direction"] == "Progressing"]["patient_id"].unique())
        rev_pids = set(df_sub[df_sub["trajectory_direction"] == "Reversing"]["patient_id"].unique())
        
        # Patient-level outcome lookup
        pat_outcome_715 = df_sub.groupby("patient_id")["primary_label_715"].first().to_dict()
        pat_outcome_705 = df_sub.groupby("patient_id")["severe_label_705"].first().to_dict()
        
        boot_diffs_715 = []
        boot_diffs_705 = []
        boot_ors_715 = []
        
        for _ in range(n_boot):
            resamp_pids = rng.choice(unique_pids, size=n_pts, replace=True)
            resamp_set = set(resamp_pids)
            
            prog_in_boot = [p for p in resamp_pids if p in prog_pids]
            rev_in_boot = [p for p in resamp_pids if p in rev_pids]
            
            if len(prog_in_boot) < 2 or len(rev_in_boot) < 2:
                continue
                
            y_prog = np.array([pat_outcome_715[p] for p in prog_in_boot])
            y_rev = np.array([pat_outcome_715[p] for p in rev_in_boot])
            
            p_prog = np.mean(y_prog)
            p_rev = np.mean(y_rev)
            boot_diffs_715.append(p_prog - p_rev)
            
            # Odds ratio with Haldane correction (0.5)
            a = np.sum(y_prog == 1) + 0.5
            b = np.sum(y_prog == 0) + 0.5
            c = np.sum(y_rev == 1) + 0.5
            d = np.sum(y_rev == 0) + 0.5
            boot_ors_715.append((a * d) / (b * c))
            
            y_prog_705 = np.array([pat_outcome_705[p] for p in prog_in_boot])
            y_rev_705 = np.array([pat_outcome_705[p] for p in rev_in_boot])
            boot_diffs_705.append(np.mean(y_prog_705) - np.mean(y_rev_705))
            
        boot_diffs_715 = np.array(boot_diffs_715)
        boot_ors_715 = np.array(boot_ors_715)
        boot_diffs_705 = np.array(boot_diffs_705)
        
        if len(boot_diffs_715) == 0:
            return None
            
        p_val_diff = float(2.0 * min(np.mean(boot_diffs_715 <= 0.0), np.mean(boot_diffs_715 >= 0.0)))
        p_val_diff = min(1.0, max(0.0, p_val_diff))
        
        return {
            "diff_715_mean": float(np.mean(boot_diffs_715)),
            "diff_715_ci_lo": float(np.percentile(boot_diffs_715, 2.5)),
            "diff_715_ci_hi": float(np.percentile(boot_diffs_715, 97.5)),
            "diff_715_pval": p_val_diff,
            "or_715_mean": float(np.mean(boot_ors_715)),
            "or_715_ci_lo": float(np.percentile(boot_ors_715, 2.5)),
            "or_715_ci_hi": float(np.percentile(boot_ors_715, 97.5)),
            "diff_705_mean": float(np.mean(boot_diffs_705)),
            "diff_705_ci_lo": float(np.percentile(boot_diffs_705, 2.5)),
            "diff_705_ci_hi": float(np.percentile(boot_diffs_705, 97.5))
        }

    results = []
    
    # 1. State-by-state comparisons
    for s_val in range(5):
        s_name = state_names[s_val]
        df_s = df_rolling[df_rolling["research_state"] == s_val]
        
        df_prog = df_s[df_s["trajectory_direction"] == "Progressing"]
        df_rev = df_s[df_s["trajectory_direction"] == "Reversing"]
        df_stat = df_s[df_s["trajectory_direction"] == "Stationary"]
        
        n_win_prog = len(df_prog)
        n_win_rev = len(df_rev)
        n_win_stat = len(df_stat)
        
        pat_prog = df_prog["patient_id"].unique()
        pat_rev = df_rev["patient_id"].unique()
        
        n_pat_prog = len(pat_prog)
        n_pat_rev = len(pat_rev)
        
        # Calculate patient-level prevalence among patients exhibiting this direction in this state
        df_pat_labels = df_rolling.groupby("patient_id")[["primary_label_715", "severe_label_705"]].first()
        
        prev_715_prog = float(df_pat_labels.loc[pat_prog, "primary_label_715"].mean()) if n_pat_prog > 0 else np.nan
        prev_715_rev = float(df_pat_labels.loc[pat_rev, "primary_label_715"].mean()) if n_pat_rev > 0 else np.nan
        
        prev_705_prog = float(df_pat_labels.loc[pat_prog, "severe_label_705"].mean()) if n_pat_prog > 0 else np.nan
        prev_705_rev = float(df_pat_labels.loc[pat_rev, "severe_label_705"].mean()) if n_pat_rev > 0 else np.nan
        
        risk_prog_mean = float(df_prog["risk_prob_proxy"].mean()) if n_win_prog > 0 else np.nan
        risk_rev_mean = float(df_rev["risk_prob_proxy"].mean()) if n_win_rev > 0 else np.nan
        
        boot_metrics = bootstrap_state_comparison(df_s)
        
        if boot_metrics is not None and n_pat_prog >= 5 and n_pat_rev >= 5:
            ci_lo = boot_metrics["diff_715_ci_lo"]
            ci_hi = boot_metrics["diff_715_ci_hi"]
            p_val = boot_metrics["diff_715_pval"]
            or_est = boot_metrics["or_715_mean"]
            or_lo = boot_metrics["or_715_ci_lo"]
            or_hi = boot_metrics["or_715_ci_hi"]
            
            if ci_lo > 0.0:
                inference = "Significant Progression Risk Elevation (CI > 0)"
            elif ci_hi < 0.0:
                inference = "Reversal Associated with Higher Risk (CI < 0)"
            else:
                inference = "Directional Difference Indistinguishable (CI overlaps 0)"
        else:
            ci_lo = np.nan
            ci_hi = np.nan
            p_val = np.nan
            or_est = np.nan
            or_lo = np.nan
            or_hi = np.nan
            inference = "Insufficient Patient Sample Size"
            
        results.append({
            "state_id": s_val,
            "state_name": s_name,
            "n_windows_progressing": n_win_prog,
            "n_windows_reversing": n_win_rev,
            "n_windows_stationary": n_win_stat,
            "n_patients_progressing": n_pat_prog,
            "n_patients_reversing": n_pat_rev,
            "risk_proxy_mean_prog": round(risk_prog_mean, 4) if not np.isnan(risk_prog_mean) else np.nan,
            "risk_proxy_mean_rev": round(risk_rev_mean, 4) if not np.isnan(risk_rev_mean) else np.nan,
            "prevalence_715_prog": round(prev_715_prog, 4) if not np.isnan(prev_715_prog) else np.nan,
            "prevalence_715_rev": round(prev_715_rev, 4) if not np.isnan(prev_715_rev) else np.nan,
            "delta_prevalence_715": round(prev_715_prog - prev_715_rev, 4) if not np.isnan(prev_715_prog) and not np.isnan(prev_715_rev) else np.nan,
            "prevalence_705_prog": round(prev_705_prog, 4) if not np.isnan(prev_705_prog) else np.nan,
            "prevalence_705_rev": round(prev_705_rev, 4) if not np.isnan(prev_705_rev) else np.nan,
            "ci_95_low_diff": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
            "ci_95_high_diff": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
            "p_value_diff": round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "odds_ratio_715": round(or_est, 4) if not np.isnan(or_est) else np.nan,
            "odds_ratio_ci_low": round(or_lo, 4) if not np.isnan(or_lo) else np.nan,
            "odds_ratio_ci_high": round(or_hi, 4) if not np.isnan(or_hi) else np.nan,
            "scientific_interpretation": inference
        })
        
    # 2. Pooled cross-state comparison
    df_all_prog = df_rolling[df_rolling["trajectory_direction"] == "Progressing"]
    df_all_rev = df_rolling[df_rolling["trajectory_direction"] == "Reversing"]
    
    pat_all_prog = df_all_prog["patient_id"].unique()
    pat_all_rev = df_all_rev["patient_id"].unique()
    
    df_pat_labels = df_rolling.groupby("patient_id")[["primary_label_715", "severe_label_705"]].first()
    prev_all_prog_715 = float(df_pat_labels.loc[pat_all_prog, "primary_label_715"].mean())
    prev_all_rev_715 = float(df_pat_labels.loc[pat_all_rev, "primary_label_715"].mean())
    prev_all_prog_705 = float(df_pat_labels.loc[pat_all_prog, "severe_label_705"].mean())
    prev_all_rev_705 = float(df_pat_labels.loc[pat_all_rev, "severe_label_705"].mean())
    
    boot_pooled = bootstrap_state_comparison(df_rolling)
    
    results.append({
        "state_id": -1,
        "state_name": "Pooled All States",
        "n_windows_progressing": len(df_all_prog),
        "n_windows_reversing": len(df_all_rev),
        "n_windows_stationary": len(df_rolling[df_rolling["trajectory_direction"] == "Stationary"]),
        "n_patients_progressing": len(pat_all_prog),
        "n_patients_reversing": len(pat_all_rev),
        "risk_proxy_mean_prog": round(float(df_all_prog["risk_prob_proxy"].mean()), 4),
        "risk_proxy_mean_rev": round(float(df_all_rev["risk_prob_proxy"].mean()), 4),
        "prevalence_715_prog": round(prev_all_prog_715, 4),
        "prevalence_715_rev": round(prev_all_rev_715, 4),
        "delta_prevalence_715": round(prev_all_prog_715 - prev_all_rev_715, 4),
        "prevalence_705_prog": round(prev_all_prog_705, 4),
        "prevalence_705_rev": round(prev_all_rev_705, 4),
        "ci_95_low_diff": round(boot_pooled["diff_715_ci_lo"], 4),
        "ci_95_high_diff": round(boot_pooled["diff_715_ci_hi"], 4),
        "p_value_diff": round(boot_pooled["diff_715_pval"], 4),
        "odds_ratio_715": round(boot_pooled["or_715_mean"], 4),
        "odds_ratio_ci_low": round(boot_pooled["or_715_ci_lo"], 4),
        "odds_ratio_ci_high": round(boot_pooled["or_715_ci_hi"], 4),
        "scientific_interpretation": "Significant Global Trajectory Differentiation (CI > 0)" if boot_pooled["diff_715_ci_lo"] > 0 else "Indistinguishable"
    })
    
    df_out = pd.DataFrame(results)
    out_path = os.path.join(OUT_DIR, "progression_reversal_analysis.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_out)} rows)")
    return df_out

if __name__ == "__main__":
    run_progression_reversal()
