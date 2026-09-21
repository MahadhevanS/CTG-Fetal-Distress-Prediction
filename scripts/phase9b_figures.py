"""
Phase 9B Figure Generator:
Generates all 8 publication-grade diagnostic figures for physiology-guided deterioration modelling.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

OUT_DIR = "reports/figures_phase9b"
RESULTS_DIR = "results/phase9b_deterioration"
os.makedirs(OUT_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def generate_phase9b_figures():
    print("=== GENERATING ALL 8 PHASE 9B DIAGNOSTIC FIGURES ===")
    
    df_models = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison_metrics.csv"))
    df_ablation = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_metrics.csv"))
    df_states = pd.read_csv(os.path.join(RESULTS_DIR, "state_analysis.csv"))
    df_alerts = pd.read_csv(os.path.join(RESULTS_DIR, "alert_policy_metrics.csv"))
    df_preds = pd.read_csv(os.path.join(RESULTS_DIR, "deterioration_predictions.csv"))
    df_preds["patient_id"] = df_preds["patient_id"].astype(str)
    
    horizons = [60, 45, 30, 20, 10, 0]
    h_labels = [">=60m", ">=45m", ">=30m", ">=20m", ">=10m", "Delivery (0m)"]
    x = np.arange(len(h_labels))
    
    # -------------------------------------------------------------------------
    # 1. Model Comparison Across Horizons
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = {
        "Model A (Snapshot Baseline)": "#7f7f7f",
        "Model B (Generic Temporal Baseline)": "#1f77b4",
        "Model C (Physiological Deterioration)": "#9467bd",
        "Model D (Snapshot + Deterioration)": "#2ca02c",
        "Model E (Full Trajectory + Domain)": "#d62728"
    }
    styles = {
        "Model A (Snapshot Baseline)": "--",
        "Model B (Generic Temporal Baseline)": "-.",
        "Model C (Physiological Deterioration)": ":",
        "Model D (Snapshot + Deterioration)": "-",
        "Model E (Full Trajectory + Domain)": "-"
    }
    
    for m_name in colors.keys():
        sub = df_models[df_models["model"] == m_name]
        vals = [sub[sub["horizon_min"] == h]["auroc_715"].values[0] for h in horizons]
        ax.plot(x, vals, marker='o', lw=2.2, color=colors[m_name], linestyle=styles[m_name], label=m_name)
        
    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, fontweight='bold', fontsize=10)
    ax.set_ylabel("Patient-Level AUROC (pH <= 7.15)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Warning Horizon Before Delivery", fontsize=11, fontweight='bold')
    ax.set_title("Figure 1: Multi-Horizon Discrimination of Physiology-Guided Deterioration Models", fontsize=12, fontweight='bold')
    ax.axhline(0.80, color='red', linestyle=':', lw=1.8, label='Phase 9 Target (0.80)')
    ax.axhline(0.50, color='black', linestyle='--', alpha=0.4)
    ax.set_ylim(0.48, 0.82)
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_comparison_horizons.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Ablation Contributions (Exps 1 to 7)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    exp_names = list(df_ablation["experiment"].unique())
    short_exp_names = [e.split(":")[1].strip() if ":" in e else e for e in exp_names]
    
    auc_30m = [df_ablation[(df_ablation["experiment"] == e) & (df_ablation["horizon_min"] == 30)]["auroc_715"].values[0] for e in exp_names]
    auc_del = [df_ablation[(df_ablation["experiment"] == e) & (df_ablation["horizon_min"] == 0)]["auroc_715"].values[0] for e in exp_names]
    
    x_bar = np.arange(len(exp_names))
    w = 0.38
    ax.bar(x_bar - w/2, auc_30m, width=w, color='#4575b4', label='>= 30 Min Horizon', edgecolor='black', alpha=0.9)
    ax.bar(x_bar + w/2, auc_del, width=w, color='#d73027', label='Delivery (0 Min)', edgecolor='black', alpha=0.9)
    
    ax.set_xticks(x_bar)
    ax.set_xticklabels(short_exp_names, rotation=25, ha='right', fontsize=9, fontweight='bold')
    ax.set_ylabel("Patient-Level AUROC (pH <= 7.15)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 2: Component Ablation Progression from State to Multidomain Deterioration", fontsize=12, fontweight='bold')
    ax.set_ylim(0.45, 0.80)
    ax.axhline(0.5699, color='gray', linestyle='--', label='Phase 8 Baseline (>=30m: 0.5699)')
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ablation_contributions.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Multidomain Deterioration ROC Curves at >=30m
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6.5))
    clean_pids = sorted(list(set(df_preds["patient_id"].astype(str))))
    y_true = np.array([df_preds[df_preds["patient_id"] == p].iloc[0]["primary_label_715"] for p in clean_pids])
    
    models_to_plot = [
        ("Model A (Snapshot Baseline)", "Model_A_Snapshot_Baseline", "#7f7f7f", "--"),
        ("Model C (Physiological Deterioration)", "Model_C_Physiological_Deterioration", "#9467bd", "-."),
        ("Model D (Snapshot + Deterioration)", "Model_D_Snapshot_+_Deterioration", "#2ca02c", "-"),
        ("Model E (Full Trajectory + Domain)", "Model_E_Full_Trajectory_+_Domain", "#d62728", "-")
    ]
    
    for label, col, color, ls in models_to_plot:
        # Extract >=30m score
        scores = []
        for p in clean_pids:
            df_p = df_preds[df_preds["patient_id"] == p].sort_values("time_before_delivery_min")
            el = df_p[df_p["time_before_delivery_min"] >= 30]
            scores.append(el.iloc[0][col] if not el.empty else df_p.iloc[-1][col])
        scores = np.array(scores)
        fpr, tpr, _ = roc_curve(y_true, scores)
        score_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.2, color=color, linestyle=ls, label=f"{label} (AUC = {score_auc:.4f})")
        
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Chance (AUC = 0.50)')
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 3: ROC Discrimination at >=30 Minutes Horizon", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "multidomain_deterioration_roc.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 4. Research State Acidemia Prevalence Gradient
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    st_names = [f"S{row['state_code']}: {row['state_name'].split(':')[1].strip()}" for _, row in df_states.iterrows()]
    prev_715 = df_states["acidemia_prevalence_pct"].values
    prev_705 = df_states["severe_prevalence_pct"].values
    
    x_s = np.arange(len(st_names))
    w = 0.35
    ax.bar(x_s - w/2, prev_715, width=w, color='#fc8d59', edgecolor='black', label='Primary Acidemia (pH <= 7.15)')
    ax.bar(x_s + w/2, prev_705, width=w, color='#d73027', edgecolor='black', label='Severe Acidemia (pH <= 7.05)')
    
    for i in range(len(x_s)):
        ax.text(x_s[i] - w/2, prev_715[i] + 1.0, f"{prev_715[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        ax.text(x_s[i] + w/2, prev_705[i] + 1.0, f"{prev_705[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        
    ax.set_xticks(x_s)
    ax.set_xticklabels(st_names, rotation=15, ha='right', fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Observed Acidemia Prevalence (%)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 4: Acidemia Risk Gradient Across Physiological Deterioration Research States", fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(prev_715) + 12)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "research_state_acidemia_gradient.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 5. Alert Policy Tradeoff
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    pol_names = [p.split(":")[1].strip() if ":" in p else p for p in df_alerts["policy_name"]]
    fa_rates = df_alerts["false_alert_rate_per_hour"].values
    sens_vals = df_alerts["sensitivity_pct"].values
    spec_vals = df_alerts["specificity_pct"].values
    
    for i, name in enumerate(pol_names):
        ax.scatter(fa_rates[i], sens_vals[i], s=180, label=name, edgecolors='black', lw=1.5, zorder=5)
        ax.text(fa_rates[i] + 0.005, sens_vals[i] - 0.8, f"{name}\n(Spec {spec_vals[i]:.1f}%)", fontsize=8.5, fontweight='bold')
        
    ax.set_xlabel("False Alert Rate (Alerts / Monitoring Hour)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Patient Sensitivity (%)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 5: Operational Tradeoff Between Sensitivity and False Alarm Burden", fontsize=12, fontweight='bold')
    ax.set_xlim(-0.02, max(fa_rates) + 0.06)
    ax.set_ylim(5, max(sens_vals) + 8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "alert_policy_tradeoff_9b.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 6. Severe Acidemia Across Warning Horizons
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for m_name in colors.keys():
        sub = df_models[df_models["model"] == m_name]
        vals = [sub[sub["horizon_min"] == h]["auroc_705"].values[0] for h in horizons]
        ax.plot(x, vals, marker='s', lw=2.2, color=colors[m_name], linestyle=styles[m_name], label=m_name)
        
    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, fontweight='bold', fontsize=10)
    ax.set_ylabel("Severe Acidemia AUROC (pH <= 7.05)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Warning Horizon Before Delivery", fontsize=11, fontweight='bold')
    ax.set_title("Figure 6: Severe Acidemia (pH <= 7.05) Early Warning Across Horizons", fontsize=12, fontweight='bold')
    ax.axhline(0.50, color='black', linestyle='--', alpha=0.4)
    ax.set_ylim(0.48, 0.82)
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "severe_acidemia_trajectories.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 7. Domain Deterioration Radar / Bar Profile
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    feat_npz = np.load(os.path.join(RESULTS_DIR, "deterioration_features.npz"))
    dom_sev = feat_npz["domain_severities"] # (8517, 6)
    dom_labels = ["Baseline FHR", "Variability", "Accelerations", "Decelerations", "Uterine Activity", "FHR-UC Coupling"]
    
    y_all_715 = df_preds["primary_label_715"].values
    mean_sev_acid = np.mean(dom_sev[y_all_715 == 1], axis=0)
    mean_sev_norm = np.mean(dom_sev[y_all_715 == 0], axis=0)
    
    x_d = np.arange(len(dom_labels))
    w = 0.35
    ax.bar(x_d - w/2, mean_sev_norm, width=w, color='#91bfdb', edgecolor='black', label='Normal Fetuses (pH > 7.15)')
    ax.bar(x_d + w/2, mean_sev_acid, width=w, color='#fc8d59', edgecolor='black', label='Acidemic Fetuses (pH <= 7.15)')
    
    ax.set_xticks(x_d)
    ax.set_xticklabels(dom_labels, rotation=20, ha='right', fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Mean Standardized Domain Severity Score", fontsize=11, fontweight='bold')
    ax.set_title("Figure 7: Physiological Domain Severity Profile in Acidemic vs Normal Fetuses", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "domain_deterioration_radar.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 8. Decision Curve Analysis
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.2))
    thresh_probs = np.linspace(0.05, 0.40, 50)
    p_acid_prev = np.mean(y_true)
    
    # Model D net benefit at >=30m
    scores_d = []
    for p in clean_pids:
        df_p = df_preds[df_preds["patient_id"] == p].sort_values("time_before_delivery_min")
        el = df_p[df_p["time_before_delivery_min"] >= 30]
        scores_d.append(el.iloc[0]["Model_D_Snapshot_+_Deterioration"] if not el.empty else df_p.iloc[-1]["Model_D_Snapshot_+_Deterioration"])
    scores_d = np.array(scores_d)
    
    net_benefit_model = []
    net_benefit_all = []
    for pt in thresh_probs:
        tp = np.sum((scores_d >= pt) & (y_true == 1))
        fp = np.sum((scores_d >= pt) & (y_true == 0))
        nb = (tp / len(y_true)) - (fp / len(y_true)) * (pt / (1.0 - pt))
        net_benefit_model.append(nb)
        
        # Treat all
        nb_all = p_acid_prev - (1.0 - p_acid_prev) * (pt / (1.0 - pt))
        net_benefit_all.append(nb_all)
        
    ax.plot(thresh_probs, net_benefit_model, lw=2.5, color='#2ca02c', label='Model D (Snapshot + Deterioration at >=30m)')
    ax.plot(thresh_probs, net_benefit_all, lw=1.8, color='gray', linestyle='--', label='Treat All Strategy')
    ax.axhline(0, color='black', lw=1.5, linestyle=':', label='Treat None Strategy')
    
    ax.set_xlabel("Clinical Decision Threshold Probability (Pt)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Net Benefit", fontsize=11, fontweight='bold')
    ax.set_title("Figure 8: Decision Curve Analysis (Clinical Net Benefit at >=30m)", fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 0.22)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "decision_curve_analysis_9b.png"), dpi=300)
    plt.close()
    
    print("All 8 Phase 9B diagnostic figures successfully generated in reports/figures_phase9b/!")

if __name__ == "__main__":
    generate_phase9b_figures()
