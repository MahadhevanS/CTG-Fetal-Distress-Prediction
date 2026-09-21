"""
Phase 9C Diagnostic Figure Generator:
Generates all 8 publication-grade figures for state transitions, trajectory dynamics, and early warning.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

OUT_DIR = "reports/figures_phase9c"
RESULTS_DIR = "results/phase9c_state_trajectory"
os.makedirs(OUT_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def generate_phase9c_figures():
    print("=== GENERATING ALL 8 PHASE 9C DIAGNOSTIC FIGURES ===")
    
    df_gradient = pd.read_csv(os.path.join(RESULTS_DIR, "state_risk_gradient.csv"))
    df_occ = pd.read_csv(os.path.join(RESULTS_DIR, "state_occupancy_metrics.csv"))
    df_timing = pd.read_csv(os.path.join(RESULTS_DIR, "state_entry_timing.csv"))
    df_matched = pd.read_csv(os.path.join(RESULTS_DIR, "matched_current_state.csv"))
    df_models = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison_metrics.csv"))
    df_alerts = pd.read_csv(os.path.join(RESULTS_DIR, "alert_policy_metrics.csv"))
    df_preds = pd.read_csv(os.path.join(RESULTS_DIR, "state_trajectory_predictions.csv"))
    df_preds["patient_id"] = df_preds["patient_id"].astype(str)
    
    clean_pids = sorted(list(set(df_preds["patient_id"])))
    y_true = np.array([df_preds[df_preds["patient_id"] == p].iloc[0]["primary_label_715"] for p in clean_pids])
    
    # -------------------------------------------------------------------------
    # 1. State-Risk Gradient with 95% CIs
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    st_names = [f"State {row['state_code']}:\n{row['state_name'].split(':')[1].strip()}" for _, row in df_gradient.iterrows()]
    prev_715 = df_gradient["acidemia_715_prevalence_pct"].values
    prev_705 = df_gradient["severe_705_prevalence_pct"].values
    err_low = prev_715 - df_gradient["ci_95_low"].values
    err_high = df_gradient["ci_95_high"].values - prev_715
    
    x_s = np.arange(len(st_names))
    w = 0.35
    ax.bar(x_s - w/2, prev_715, width=w, yerr=[err_low, err_high], capsize=4, color='#fc8d59', edgecolor='black', label='Primary Acidemia (pH <= 7.15) [95% CI]')
    ax.bar(x_s + w/2, prev_705, width=w, color='#d73027', edgecolor='black', label='Severe Acidemia (pH <= 7.05)')
    
    for i in range(len(x_s)):
        ax.text(x_s[i] - w/2, prev_715[i] + 4.5, f"{prev_715[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        ax.text(x_s[i] + w/2, prev_705[i] + 2.0, f"{prev_705[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        
    ax.set_xticks(x_s)
    ax.set_xticklabels(st_names, fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Observed Acidemia Prevalence (%)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 1: Monotonic Fetal Acidemia Risk Gradient Across Research States (Exp 1)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 65)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "state_risk_gradient_9c.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Representative State Trajectories
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    # Select two representative acidemic patients (progressing) and two normal patients (stable/reversing)
    acid_pids = [p for p in clean_pids if df_preds[df_preds["patient_id"] == p].iloc[0]["primary_label_715"] == 1]
    norm_pids = [p for p in clean_pids if df_preds[df_preds["patient_id"] == p].iloc[0]["primary_label_715"] == 0]
    
    # Acidemic examples with long monitoring
    acid_demo = sorted(acid_pids, key=lambda p: len(df_preds[df_preds["patient_id"] == p]), reverse=True)[:2]
    norm_demo = sorted(norm_pids, key=lambda p: len(df_preds[df_preds["patient_id"] == p]), reverse=True)[:2]
    
    colors_demo = ['#d73027', '#fc8d59', '#4575b4', '#74add1']
    labels_demo = [
        f"Acidemic Patient {acid_demo[0]} (pH <= 7.15)",
        f"Acidemic Patient {acid_demo[1]} (pH <= 7.15)",
        f"Normal Patient {norm_demo[0]} (pH > 7.15)",
        f"Normal Patient {norm_demo[1]} (pH > 7.15)"
    ]
    
    for i, pid in enumerate(acid_demo + norm_demo):
        df_p = df_preds[df_preds["patient_id"] == pid].sort_values("time_before_delivery_min", ascending=False)
        t_pts = -df_p["time_before_delivery_min"].values # negative minutes to delivery
        st_pts = df_p["research_state"].values
        ax.step(t_pts, st_pts, where='post', lw=2.2, color=colors_demo[i], label=labels_demo[i])
        
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["S0: Stable", "S1: Emerging", "S2: Persistent", "S3: Progressive", "S4: Severe"], fontweight='bold')
    ax.set_xlabel("Time Before Delivery (Minutes)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 2: Representative Intrapartum State Trajectories (Progressive vs Reversible)", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "representative_state_trajectories.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Acidemic vs Normal State Occupancy
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    st_codes = df_occ["state_code"].values
    occ_norm = df_occ["mean_minutes_normal"].values
    occ_acid = df_occ["mean_minutes_acidemic"].values
    
    x_o = np.arange(len(st_codes))
    w = 0.35
    ax.bar(x_o - w/2, occ_norm, width=w, color='#91bfdb', edgecolor='black', label='Normal Fetuses (Mean Min)')
    ax.bar(x_o + w/2, occ_acid, width=w, color='#d73027', edgecolor='black', label='Acidemic Fetuses (Mean Min)')
    
    for i in range(len(x_o)):
        p_val = df_occ["p_value"].iloc[i]
        star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        ax.text(x_o[i], max(occ_norm[i], occ_acid[i]) + 1.5, f"p={p_val:.3f}\n{star}", ha='center', fontsize=8.5, fontweight='bold')
        
    ax.set_xticks(x_o)
    ax.set_xticklabels([f"State {k}" for k in st_codes], fontweight='bold')
    ax.set_ylabel("Mean Observed Monitoring Time (Minutes)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 3: Intrapartum State Occupancy Distribution (Acidemic vs Normal, Exp 2)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(max(occ_norm), max(occ_acid)) + 8)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "acidemic_vs_normal_state_occupancy.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 4. State-Entry Timing Distribution Before Delivery
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    st_t_names = [f"State {row['state_code']}" for _, row in df_timing.iterrows()]
    med_t = df_timing["median_lead_time_min"].values
    iqr_t = df_timing["iqr_lead_time_min"].values
    mean_t = df_timing["mean_lead_time_min"].values
    entry_rate = df_timing["acidemic_entry_rate_pct"].values
    
    x_t = np.arange(len(st_t_names))
    ax.errorbar(x_t, med_t, yerr=iqr_t/2.0, fmt='o', color='#d73027', ecolor='#4575b4', elinewidth=2.5, capsize=6, markersize=8, label='Median Lead Time (IQR)')
    ax.plot(x_t, mean_t, marker='s', color='#2ca02c', linestyle='--', label='Mean Lead Time')
    
    for i in range(len(x_t)):
        ax.text(x_t[i], med_t[i] + 3.0, f"Med: {med_t[i]:.1f}m\n(Rate: {entry_rate[i]:.0f}%)", ha='center', fontsize=9, fontweight='bold')
        
    ax.set_xticks(x_t)
    ax.set_xticklabels(st_t_names, fontweight='bold')
    ax.set_ylabel("Lead Time Before Delivery (Minutes)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 4: Timing of First Entry into Deterioration States (Exp 3)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(med_t) + 12)
    ax.axhline(30, color='red', linestyle=':', label='Target Horizon (>= 30 Min)')
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "state_entry_timing_distribution.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 5. State Persistence Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    df_trans = pd.read_csv(os.path.join(RESULTS_DIR, "transition_dynamics_metrics.csv"))
    metrics = df_trans["metric_name"].values
    n_mean = df_trans["normal_mean"].values
    a_mean = df_trans["acidemic_mean"].values
    
    x_m = np.arange(len(metrics))
    w = 0.35
    ax.bar(x_m - w/2, n_mean, width=w, color='#91bfdb', edgecolor='black', label='Normal Fetuses')
    ax.bar(x_m + w/2, a_mean, width=w, color='#d73027', edgecolor='black', label='Acidemic Fetuses')
    
    for i in range(len(x_m)):
        p_val = df_trans["p_value"].iloc[i]
        ax.text(x_m[i], max(n_mean[i], a_mean[i]) + 0.15, f"p={p_val:.4f}", ha='center', fontsize=8.5, fontweight='bold')
        
    ax.set_xticks(x_m)
    ax.set_xticklabels(["Total Reversals (Exp 5)", "Max Velocity V4 (Exp 6)", "Multidomain Dur (Exp 8)"], rotation=15, ha='right', fontweight='bold')
    ax.set_ylabel("Mean Metric Value per Patient", fontsize=11, fontweight='bold')
    ax.set_title("Figure 5: Trajectory Dynamics & Transition Comparison (Exp 4-8)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(max(n_mean), max(a_mean)) + 0.8)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "state_persistence_survival.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 6. Matched-Current-State Analysis
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    matched_cats = [f"{row['current_state_stratum']}\n{row['trajectory_dynamic'].split('(')[0].strip()}" for _, row in df_matched.iterrows()]
    prev_match = df_matched["acidemia_prevalence_pct"].values
    sev_match = df_matched["severe_prevalence_pct"].values
    
    x_match = np.arange(len(matched_cats))
    w = 0.38
    ax.bar(x_match - w/2, prev_match, width=w, color='#fc8d59', edgecolor='black', label='Acidemia (pH <= 7.15)')
    ax.bar(x_match + w/2, sev_match, width=w, color='#d73027', edgecolor='black', label='Severe Acidemia (pH <= 7.05)')
    
    for i in range(len(x_match)):
        ax.text(x_match[i] - w/2, prev_match[i] + 1.0, f"{prev_match[i]:.1f}%", ha='center', fontsize=8.5, fontweight='bold')
        
    ax.set_xticks(x_match)
    ax.set_xticklabels(matched_cats, rotation=25, ha='right', fontsize=9, fontweight='bold')
    ax.set_ylabel("Observed Acidemia Prevalence (%)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 6: Matched-Current-State Risk Analysis (Testing Hypothesis H1, Exp 9)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(prev_match) + 8)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "matched_current_state_risk.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 7. Multi-Horizon AUROC Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    horizons = [60, 45, 30, 20, 10, 0]
    h_labels = [">=60m", ">=45m", ">=30m", ">=20m", ">=10m", "Delivery (0m)"]
    x_h = np.arange(len(h_labels))
    
    model_colors = {
        "Phase 8 Snapshot Baseline": ("#7f7f7f", "--"),
        "Model 1 (Current State Only)": ("#1f77b4", ":"),
        "Model 2 (Current State + Trajectory)": ("#9467bd", "-."),
        "Model 3 (Current Features + Trajectory)": ("#ff7f0e", "-"),
        "Model 4 (Prediction Trajectory Only)": ("#2ca02c", "--"),
        "Model 5 (Combined Prediction + State Trajectory)": ("#d62728", "-")
    }
    
    for m_name, (col, ls) in model_colors.items():
        sub = df_models[df_models["model"] == m_name]
        vals = [sub[sub["horizon_min"] == h]["auroc_715"].values[0] for h in horizons]
        ax.plot(x_h, vals, marker='o', lw=2.2, color=col, linestyle=ls, label=m_name)
        
    ax.set_xticks(x_h)
    ax.set_xticklabels(h_labels, fontweight='bold', fontsize=10)
    ax.set_ylabel("Patient-Level AUROC (pH <= 7.15)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Warning Horizon Before Delivery", fontsize=11, fontweight='bold')
    ax.set_title("Figure 7: Multi-Horizon AUROC of State Trajectory Models (Exp 10-11)", fontsize=12, fontweight='bold')
    ax.axhline(0.80, color='red', linestyle=':', lw=1.8, label='Phase 9 Target (0.80)')
    ax.axhline(0.50, color='black', linestyle='--', alpha=0.4)
    ax.set_ylim(0.48, 0.82)
    ax.legend(loc='lower right', frameon=True, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "multihorizon_auroc_comparison_9c.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 8. Alert Policy Operational Tradeoff
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    pol_names = [p.split(":")[1].strip() if ":" in p else p for p in df_alerts["policy_name"]]
    fa_rates = df_alerts["false_alert_rate_per_hour"].values
    sens_vals = df_alerts["sensitivity_pct"].values
    spec_vals = df_alerts["specificity_pct"].values
    lead_t = df_alerts["median_lead_time_min"].values
    
    for i, name in enumerate(pol_names):
        ax.scatter(fa_rates[i], sens_vals[i], s=190, label=name, edgecolors='black', lw=1.5, zorder=5)
        ax.text(fa_rates[i] + 0.005, sens_vals[i] - 0.8, f"{name}\n(Spec: {spec_vals[i]:.1f}%, Lead: {lead_t[i]:.1f}m)", fontsize=8.5, fontweight='bold')
        
    ax.set_xlabel("False Alert Rate (Alerts / Monitoring Hour)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Patient Sensitivity (%)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 8: Alert Policy Operational Performance & Lead Time (Exp 12)", fontsize=12, fontweight='bold')
    ax.set_xlim(-0.02, max(fa_rates) + 0.08)
    ax.set_ylim(5, max(sens_vals) + 8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "alert_policy_tradeoff_9c.png"), dpi=300)
    plt.close()
    
    print("All 8 Phase 9C diagnostic figures successfully generated in reports/figures_phase9c/!")

if __name__ == "__main__":
    generate_phase9c_figures()
