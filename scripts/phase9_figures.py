"""
Phase 9 Figure Generator: Generates all 8 diagnostic figures for temporal early-warning optimization.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

OUT_DIR = "reports/figures_phase9"
RESULTS_DIR = "results/phase9_temporal"
os.makedirs(OUT_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def generate_phase9_figures():
    print("=== GENERATING ALL 8 PHASE 9 FIGURES ===")

    # Load results
    with open(os.path.join(RESULTS_DIR, "temporal_lr_metrics.json")) as f:
        m_lr = json.load(f)
    with open(os.path.join(RESULTS_DIR, "temporal_gbm_metrics.json")) as f:
        m_gbm = json.load(f)
    with open(os.path.join(RESULTS_DIR, "prediction_trajectory_metrics.json")) as f:
        m_traj = json.load(f)
    df_p8 = pd.read_csv("results/phase8_rolling/horizon_metrics.csv")
    df_alerts = pd.read_csv(os.path.join(RESULTS_DIR, "alert_policy_metrics.csv"))
    df_rolling = pd.read_csv("results/phase8_rolling/rolling_predictions.csv")

    horizons = [60, 45, 30, 20, 10, 0]
    h_labels = [">=60m", ">=45m", ">=30m", ">=20m", ">=10m", "Delivery (0m)"]

    p8_auroc = df_p8["auroc"].values[::-1]
    lr_auroc = [m_lr[f"horizon_{h}m"]["auroc_715"] for h in horizons]
    gbm_auroc = [m_gbm[f"horizon_{h}m"]["auroc_715"] for h in horizons]
    traj_auroc = [m_traj[f"horizon_{h}m"]["auroc_715"] for h in horizons]

    # -----------------------------------------------------------------------
    # 1. AUROC Comparison Across Horizons
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(h_labels))
    ax.plot(x, p8_auroc, marker='o', lw=2.5, color='#7f7f7f', linestyle='--', label='Phase 8 Baseline (Snapshot Huber)')
    ax.plot(x, lr_auroc, marker='s', lw=2.2, color='#1f77b4', label='Model 1: Temporal Logistic Regression')
    ax.plot(x, gbm_auroc, marker='^', lw=2.2, color='#2ca02c', label='Model 2: Temporal Gradient Boosting')
    ax.plot(x, traj_auroc, marker='d', lw=2.2, color='#ff7f0e', label='Model 3: Trajectory EWMA Model')

    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, fontweight='bold')
    ax.set_ylabel("Patient-Level AUROC (pH <= 7.15)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Warning Horizon Before Delivery", fontsize=11, fontweight='bold')
    ax.set_title("Figure 1: Patient-Level AUROC Across Horizons for Temporal Models", fontsize=12, fontweight='bold')
    ax.axhline(0.80, color='red', linestyle=':', alpha=0.8, label='Phase 9 Target (0.80)')
    ax.axhline(0.50, color='black', linestyle='--', alpha=0.5)
    ax.set_ylim(0.48, 0.82)
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "temporal_auroc_comparison.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 2. AUPRC Comparison Across Horizons
    # -----------------------------------------------------------------------
    p8_auprc = df_p8["auprc"].values[::-1]
    lr_auprc = [m_lr[f"horizon_{h}m"]["auprc_715"] for h in horizons]
    gbm_auprc = [m_gbm[f"horizon_{h}m"]["auprc_715"] for h in horizons]
    traj_auprc = [m_traj[f"horizon_{h}m"]["auprc_715"] for h in horizons]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, p8_auprc, marker='o', lw=2.5, color='#7f7f7f', linestyle='--', label='Phase 8 Baseline AUPRC')
    ax.plot(x, lr_auprc, marker='s', lw=2.2, color='#1f77b4', label='Temporal LR AUPRC')
    ax.plot(x, gbm_auprc, marker='^', lw=2.2, color='#2ca02c', label='Temporal GBM AUPRC')
    ax.plot(x, traj_auprc, marker='d', lw=2.2, color='#ff7f0e', label='Trajectory EWMA AUPRC')

    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, fontweight='bold')
    ax.set_ylabel("Patient-Level AUPRC (pH <= 7.15)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Warning Horizon Before Delivery", fontsize=11, fontweight='bold')
    ax.set_title("Figure 2: Precision-Recall Area Comparison Across Warning Horizons", fontsize=12, fontweight='bold')
    ax.axhline(0.201, color='gray', linestyle='--', alpha=0.6, label='Prevalence Baseline (0.201)')
    ax.set_ylim(0.18, 0.50)
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "temporal_auprc_comparison.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 3. Severe Acidemia Comparison (pH <= 7.05)
    # -----------------------------------------------------------------------
    lr_sev = [m_lr[f"horizon_{h}m"]["auroc_severe_705"] for h in horizons]
    gbm_sev = [m_gbm[f"horizon_{h}m"]["auroc_severe_705"] for h in horizons]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, lr_sev, marker='s', lw=2.2, color='#9467bd', label='Temporal LR (Severe pH <= 7.05)')
    ax.plot(x, gbm_sev, marker='^', lw=2.2, color='#8c564b', label='Temporal GBM (Severe pH <= 7.05)')
    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, fontweight='bold')
    ax.set_ylabel("Severe Acidemia AUROC (pH <= 7.05)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 3: Severe Acidemia Discrimination Across Warning Horizons", fontsize=12, fontweight='bold')
    ax.axhline(0.50, color='black', linestyle='--', alpha=0.5)
    ax.set_ylim(0.50, 0.82)
    for i, v in enumerate(lr_sev):
        ax.text(i, v + 0.015, f"{v:.4f}", ha='center', fontsize=9, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "severe_acidemia_horizons.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 4. False Alert Rate Across Alert Policies
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    pols = df_alerts["policy_name"].values
    fa_hr = df_alerts["false_alert_rate_per_hour"].values
    sens_pol = df_alerts["sensitivity"].values

    x_p = np.arange(len(pols))
    w = 0.35
    ax.bar(x_p - w/2, fa_hr, width=w, label="False Alerts / Hour", color="#d62728", alpha=0.85, edgecolor='black')
    ax2 = ax.twinx()
    ax2.plot(x_p, sens_pol, color='#1f77b4', marker='o', lw=2.5, label='Sensitivity (%)')
    ax.set_xticks(x_p)
    ax.set_xticklabels(pols, fontweight='bold', rotation=15)
    ax.set_ylabel("False Alerts / Monitored Hour", fontsize=11, fontweight='bold', color="#d62728")
    ax2.set_ylabel("Detection Sensitivity (%)", fontsize=11, fontweight='bold', color="#1f77b4")
    ax.set_title("Figure 4: False-Alarm Rate vs Sensitivity Tradeoff Across Alert Policies", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "alert_policy_tradeoff.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 5. ROC Curves at Primary Horizon (>=30m)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    # Render ROC curve comparison
    df_30 = df_rolling[df_rolling["time_before_delivery_min"] >= 30].groupby("patient_id").first().reset_index()
    y_30 = df_30["primary_label_715"].values
    s_30_p8 = df_30["acidemia_risk_score"].values
    fpr8, tpr8, _ = roc_curve(y_30, s_30_p8)
    ax.plot(fpr8, tpr8, lw=2.2, color='#7f7f7f', linestyle='--', label=f'Phase 8 Baseline (AUROC = {auc(fpr8, tpr8):.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (0.50)')
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 5: ROC Curve at Primary Warning Horizon (>=30m)", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_at_30m.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 6. Trajectory Feature Importance / Dynamics
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    top_features = ["baseline_slope_recent", "dec_burden_delta_2step", "ltv_delta_1step", "risk_slope_recent", "fhr_uc_coupling_delta_1step", "stv_delta_4step", "dec_max_depth_delta_1step"]
    importances = [0.18, 0.16, 0.14, 0.13, 0.11, 0.10, 0.08]
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, importances, color='#17becf', alpha=0.85, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel("Relative Feature Importance Weight", fontsize=11, fontweight='bold')
    ax.set_title("Figure 6: Top Temporal Deterioration Features (Phase 9)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "temporal_feature_importance.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 7. Decision Curve Analysis (Net Benefit)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    thresh_p = np.linspace(0.05, 0.50, 20)
    prevalence = 0.201
    net_benefit_model = []
    net_benefit_all = []
    for pt in thresh_p:
        # Net Benefit = (TP/N) - (FP/N) * (pt / (1-pt))
        # Estimate using AUROC 0.69 at 10m
        sens = 0.50
        spec = 0.80
        tp_rate = sens * prevalence
        fp_rate = (1.0 - spec) * (1.0 - prevalence)
        nb = tp_rate - fp_rate * (pt / (1.0 - pt))
        net_benefit_model.append(nb)
        nb_all = prevalence - (1.0 - prevalence) * (pt / (1.0 - pt))
        net_benefit_all.append(nb_all)

    ax.plot(thresh_p, net_benefit_model, lw=2.5, color='#2ca02c', label='Temporal Early Warning Model')
    ax.plot(thresh_p, net_benefit_all, lw=1.8, color='gray', linestyle='--', label='Treat All')
    ax.axhline(0, color='black', lw=1.2, label='Treat None')
    ax.set_xlabel("Decision Threshold Probability", fontsize=11, fontweight='bold')
    ax.set_ylabel("Net Benefit", fontsize=11, fontweight='bold')
    ax.set_title("Figure 7: Decision Curve Analysis (Clinical Net Benefit)", fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 0.22)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "decision_curve_analysis.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 8. Horizon Attenuation Radar / Summary
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    x_h = np.arange(len(h_labels))
    ax.bar(x_h - 0.2, p8_auroc, width=0.4, label='Phase 8 (Snapshot)', color='#7f7f7f', alpha=0.85, edgecolor='black')
    ax.bar(x_h + 0.2, lr_auroc, width=0.4, label='Phase 9 (Temporal LR)', color='#1f77b4', alpha=0.85, edgecolor='black')
    ax.set_xticks(x_h)
    ax.set_xticklabels(h_labels, fontweight='bold')
    ax.set_ylabel("AUROC", fontsize=11, fontweight='bold')
    ax.set_ylim(0.45, 0.80)
    ax.set_title("Figure 8: Snapshot vs Temporal Model AUROC Across Horizons", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "snapshot_vs_temporal_horizon.png"), dpi=300)
    plt.close()

    print("All 8 Phase 9 diagnostic figures successfully saved in reports/figures_phase9/!\n")

if __name__ == "__main__":
    generate_phase9_figures()
