"""
Phase 8 Figure Generator: Generates all 8 diagnostic figures for rolling 20-minute early-warning validation.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

OUT_DIR = "reports/figures_phase8"
RESULTS_DIR = "results/phase8_rolling"
os.makedirs(OUT_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def generate_phase8_figures():
    print("=== GENERATING ALL 8 PHASE 8 FIGURES ===")
    df_horizons = pd.read_csv(os.path.join(RESULTS_DIR, "horizon_metrics.csv"))
    df_rolling = pd.read_csv(os.path.join(RESULTS_DIR, "rolling_predictions.csv"))
    df_wt = pd.read_csv(os.path.join(RESULTS_DIR, "warning_times.csv"))
    df_alerts = pd.read_csv(os.path.join(RESULTS_DIR, "alert_metrics.csv"))

    # -----------------------------------------------------------------------
    # 1. AUROC vs Warning Horizon
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    h_labels = df_horizons["horizon"].values[::-1]
    auroc_vals = df_horizons["auroc"].values[::-1]
    ax.plot(range(len(h_labels)), auroc_vals, marker='o', lw=2.5, color='#1f77b4', label='Primary Endpoint (pH <= 7.15)')
    ax.set_xticks(range(len(h_labels)))
    ax.set_xticklabels(h_labels, fontweight='bold')
    ax.set_ylabel("Patient-Level AUROC", fontsize=11, fontweight='bold')
    ax.set_xlabel("Early-Warning Horizon Before Delivery", fontsize=11, fontweight='bold')
    ax.set_title("Figure 1: Patient-Level AUROC Across Early-Warning Horizons", fontsize=12, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Chance (0.50)')
    ax.set_ylim(0.45, 0.80)
    for i, v in enumerate(auroc_vals):
        ax.text(i, v + 0.015, f"{v:.4f}", ha='center', fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "auroc_vs_horizon.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 2. AUPRC vs Warning Horizon
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    auprc_vals = df_horizons["auprc"].values[::-1]
    ax.plot(range(len(h_labels)), auprc_vals, marker='s', lw=2.5, color='#ff7f0e', label='AUPRC (pH <= 7.15)')
    ax.set_xticks(range(len(h_labels)))
    ax.set_xticklabels(h_labels, fontweight='bold')
    ax.set_ylabel("Patient-Level AUPRC", fontsize=11, fontweight='bold')
    ax.set_xlabel("Early-Warning Horizon Before Delivery", fontsize=11, fontweight='bold')
    ax.set_title("Figure 2: Precision-Recall Area Across Warning Horizons", fontsize=12, fontweight='bold')
    ax.axhline(0.201, color='gray', linestyle='--', alpha=0.7, label='Prevalence Baseline (0.201)')
    ax.set_ylim(0.15, 0.50)
    for i, v in enumerate(auprc_vals):
        ax.text(i, v + 0.012, f"{v:.4f}", ha='center', fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "auprc_vs_horizon.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 3. Sensitivity vs Horizon (@ 90% Specificity)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sens_vals = df_horizons["sens_at_90_spec"].values[::-1]
    bars = ax.bar(range(len(h_labels)), sens_vals, color='#2ca02c', alpha=0.85, edgecolor='black', width=0.5)
    ax.set_xticks(range(len(h_labels)))
    ax.set_xticklabels(h_labels, fontweight='bold')
    ax.set_ylabel("Sensitivity (%) @ 90% Specificity", fontsize=11, fontweight='bold')
    ax.set_xlabel("Early-Warning Horizon Before Delivery", fontsize=11, fontweight='bold')
    ax.set_title("Figure 3: Clinically Actionable Sensitivity (at 90% Fixed Specificity)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 50)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0, f"{b.get_height():.1f}%", ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sensitivity_vs_horizon.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 4. Calibration Curves Across Horizons
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration')
    for h in [30, 20, 10]:
        df_sub = df_rolling[df_rolling["time_before_delivery_min"] >= h]
        if not df_sub.empty:
            df_h = df_sub.groupby("patient_id").first().reset_index()
            y_t = df_h["primary_label_715"].values
            p_t = df_h["risk_prob_proxy"].values
            # Decile calibration
            bins = np.linspace(0, 1, 6)
            b_idx = np.digitize(p_t, bins) - 1
            b_idx = np.clip(b_idx, 0, len(bins)-2)
            obs = [np.mean(y_t[b_idx == bi]) if np.sum(b_idx == bi) > 0 else np.nan for bi in range(len(bins)-1)]
            pred_m = [np.mean(p_t[b_idx == bi]) if np.sum(b_idx == bi) > 0 else np.nan for bi in range(len(bins)-1)]
            ax.plot(pred_m, obs, marker='o', lw=2, label=f"Horizon >={h}m")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight='bold')
    ax.set_ylabel("Observed Acidemia Fraction (pH <= 7.15)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 4: Reliability Calibration Across Warning Horizons", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "calibration_curves.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 5. Warning-Time Distribution
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    detected_wt = df_wt[df_wt["warning_time_min"] > 0]["warning_time_min"].values
    ax.hist(detected_wt, bins=15, color='#9467bd', alpha=0.8, edgecolor='black')
    ax.axvline(np.median(detected_wt), color='red', linestyle='--', lw=2, label=f"Median Warning Time = {np.median(detected_wt):.1f} min")
    ax.set_xlabel("Warning Lead Time Before Delivery (minutes)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Number of Acidemic Patients", fontsize=11, fontweight='bold')
    ax.set_title("Figure 5: Warning-Time Distribution for Detected Acidemic Cases", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "warning_time_distribution.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 6. Prediction Trajectories
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    # Aggregate mean risk trajectory by time before delivery
    df_rolling["time_bucket"] = (df_rolling["time_before_delivery_min"] // 5) * 5
    traj_acidemic = df_rolling[df_rolling["primary_label_715"] == 1].groupby("time_bucket")["acidemia_risk_score"].mean().sort_index(ascending=False)
    traj_normal = df_rolling[df_rolling["primary_label_715"] == 0].groupby("time_bucket")["acidemia_risk_score"].mean().sort_index(ascending=False)

    times = traj_acidemic.index.values
    ax.plot(times, traj_acidemic.values, marker='^', lw=2.5, color='#d62728', label='Acidemic Patients (pH <= 7.15)')
    ax.plot(times, traj_normal.values, marker='o', lw=2.5, color='#1f77b4', label='Normal Patients (pH > 7.15)')
    ax.invert_xaxis()
    ax.set_xlabel("Minutes Before Delivery", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mean Predicted Risk Score (S = -pH_hat)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 6: Longitudinal Risk Trajectory Approaching Delivery", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "prediction_trajectories.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 7. False-Alert Burden & Persistence Comparison
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    rules = df_alerts["rule"].values
    fa_per_pt = df_alerts["false_alerts_per_patient"].values
    fa_per_hr = df_alerts["false_alerts_per_monitoring_hour"].values

    x = np.arange(len(rules))
    w = 0.35
    ax.bar(x - w/2, fa_per_pt, width=w, label="False Alerts / Patient", color="#e377c2", alpha=0.85, edgecolor='black')
    ax.bar(x + w/2, fa_per_hr, width=w, label="False Alerts / Monitored Hour", color="#7f7f7f", alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(rules, fontweight='bold')
    ax.set_ylabel("Rate", fontsize=11, fontweight='bold')
    ax.set_title("Figure 7: False Alarm Burden: Single vs 2-Consecutive Alert Rules", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "false_alert_burden.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 8. ROC Curves at Selected Horizons
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for h, col in zip([30, 20, 10], ['#2ca02c', '#ff7f0e', '#1f77b4']):
        df_sub = df_rolling[df_rolling["time_before_delivery_min"] >= h]
        df_h = df_sub.groupby("patient_id").first().reset_index()
        y_t = df_h["primary_label_715"].values
        s_t = df_h["acidemia_risk_score"].values
        fpr, tpr, _ = roc_curve(y_t, s_t)
        a_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.2, color=col, label=f"Horizon >={h}m (AUROC = {a_val:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (0.50)')
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 8: ROC Curves at Primary Early-Warning Horizons", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curves_horizons.png"), dpi=300)
    plt.close()

    print("All 8 diagnostic figures successfully saved in reports/figures_phase8/!\n")

if __name__ == "__main__":
    generate_phase8_figures()
