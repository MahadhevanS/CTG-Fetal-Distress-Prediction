"""
Phase 11: Publication Figure Generation Suite.

Generates 5 publication-grade figures in reports/figures_phase11/:
1. priorart_comparison.png — Head-to-head AUROC comparison at >=30m and Delivery.
2. horizon_comparison.png — Multi-horizon trajectory from >=60m to delivery across all 6 models.
3. incremental_auroc.png — Forest plot of pairwise bootstrap delta AUROC and 95% CIs.
4. warning_time_comparison.png — Cumulative warning lead-time distributions.
5. alert_tradeoff.png — Sensitivity vs False Alert Rate operating points.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results/phase11_priorart_benchmark"
FIG_DIR = "reports/figures_phase11"
os.makedirs(FIG_DIR, exist_ok=True)

def generate_phase11_figures():
    print("=== GENERATING PHASE 11 PUBLICATION FIGURES ===")
    
    df_comp = pd.read_csv(os.path.join(OUT_DIR, "model_comparison.csv"))
    df_pair = pd.read_csv(os.path.join(OUT_DIR, "pairwise_bootstrap.csv"))
    df_op = pd.read_csv(os.path.join(OUT_DIR, "clinical_operating_points.csv"))
    df_warn = pd.read_csv(os.path.join(OUT_DIR, "warning_time_results.csv"))
    
    # 1. Figure 1: Head-to-head comparison (>=30m and Delivery)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    df_30 = df_comp[df_comp["horizon_min"] == 30].sort_values("model_code")
    df_0 = df_comp[df_comp["horizon_min"] == 0].sort_values("model_code")
    
    models = ["P1 (Compact CTG)", "P2 (Sequential)", "P3 (Snapshot)", "P4 (State)", "P5 (Trajectory)", "P6 (Full System)"]
    auc_30 = df_30["auroc_715"].values
    auc_0 = df_0["auroc_715"].values
    
    x = np.arange(len(models))
    w = 0.35
    
    rects1 = ax.bar(x - w/2, auc_30, width=w, label=">=30m Early-Warning Horizon", color="#4299e1", edgecolor="#2d3748")
    rects2 = ax.bar(x + w/2, auc_0, width=w, label="Delivery (0m) Endpoint", color="#2b6cb0", edgecolor="#2d3748")
    
    for bar in rects1:
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.008, f"{bar.get_height():.4f}", ha="center", fontsize=8, weight="bold")
    for bar in rects2:
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.008, f"{bar.get_height():.4f}", ha="center", fontsize=8, weight="bold")
        
    ax.axhline(0.80, color="#e53e3e", linestyle="--", linewidth=1.2, label="Aspiration Target (0.80)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, weight="bold")
    ax.set_ylim(0.40, 0.85)
    ax.set_ylabel("Patient-Level AUROC (pH <= 7.15)", fontsize=10, weight="bold")
    ax.set_title("Figure 1: Head-to-Head Benchmark Discrimination at >=30m and Delivery", fontsize=11, weight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "priorart_comparison.png"), dpi=300)
    plt.close(fig)
    
    # 2. Figure 2: Multi-Horizon Comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x_times = [60, 45, 30, 20, 10, 0]
    
    colors_map = {
        "P1": ("#a0aec0", "--", "o"),
        "P2": ("#718096", "-.", "s"),
        "P3": ("#3182ce", ":", "^"),
        "P4": ("#805ad5", "-", "v"),
        "P5": ("#d69e2e", "-", "D"),
        "P6": ("#38a169", "-", "P")
    }
    
    for code, (c, ls, m) in colors_map.items():
        df_m = df_comp[df_comp["model_code"] == code]
        y_vals = [df_m[df_m["horizon_min"] == h]["auroc_715"].iloc[0] for h in x_times]
        m_name = df_m["model_name"].iloc[0]
        ax.plot(x_times, y_vals, color=c, linestyle=ls, marker=m, lw=2, label=m_name)
        
    ax.axvline(40, color="#e53e3e", linestyle="--", alpha=0.7, label="Dataset Lead Limit (40m in 60m Trace)")
    ax.set_xlim(65, -5)
    ax.set_ylim(0.45, 0.78)
    ax.set_xlabel("Minutes Before Delivery", fontsize=10, weight="bold")
    ax.set_ylabel("Patient-Level AUROC (pH <= 7.15)", fontsize=10, weight="bold")
    ax.set_title("Figure 2: Multi-Horizon Discrimination Trajectory Across Prior-Art & Proposed Models", fontsize=11, weight="bold")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "horizon_comparison.png"), dpi=300)
    plt.close(fig)
    
    # 3. Figure 3: Forest Plot of Pairwise Bootstrap Delta AUROC
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    df_p0 = df_pair[df_pair["horizon_min"] == 0].reset_index(drop=True)
    
    y_pos = np.arange(len(df_p0))
    deltas = df_p0["delta_auroc"].values
    ci_los = df_p0["ci95_low"].values
    ci_his = df_p0["ci95_high"].values
    labels = df_p0["comparison"].values
    
    err_left = deltas - ci_los
    err_right = ci_his - deltas
    
    ax.errorbar(deltas, y_pos, xerr=[err_left, err_right], fmt="o", color="#2b6cb0", ecolor="#4a5568", elinewidth=2, capsize=5, capthick=1.5, markersize=7)
    ax.axvline(0.0, color="#e53e3e", linestyle="--", lw=1.2)
    
    for i in range(len(df_p0)):
        ax.text(deltas[i], y_pos[i] + 0.25, f"{deltas[i]:+.4f} [{ci_los[i]:+.4f}, {ci_his[i]:+.4f}]", ha="center", fontsize=8, weight="bold")
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, weight="bold")
    ax.set_xlabel("Paired Delta AUROC at Delivery (0m)", fontsize=10, weight="bold")
    ax.set_title("Figure 3: Pairwise Patient-Level Bootstrap Differences vs Baselines (B=2,000)", fontsize=11, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "incremental_auroc.png"), dpi=300)
    plt.close(fig)
    
    # 4. Figure 4: Warning Lead-Time Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    models_warn = ["P1 (Compact)", "P2 (Sequential)", "P3 (Snapshot)", "P4 (State)", "P5 (Trajectory)", "P6 (Full System)"]
    med_leads = df_warn["median_lead_min"].values
    pct_ge30 = df_warn["pct_warned_ge30m"].values
    
    x = np.arange(len(models_warn))
    ax.bar(x, med_leads, color="#319795", edgecolor="#2d3748", width=0.5, label="Median Lead Time (min)")
    for i in range(len(med_leads)):
        ax.text(x[i], med_leads[i] + 0.3, f"{med_leads[i]:.1f}m\n({pct_ge30[i]:.1f}% >=30m)", ha="center", fontsize=8, weight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels(models_warn, fontsize=8, weight="bold")
    ax.set_ylabel("Median Warning Lead Time (Minutes)", fontsize=10, weight="bold")
    ax.set_title("Figure 4: Warning Lead-Time Distribution Across Benchmark Models", fontsize=11, weight="bold")
    ax.set_ylim(0, 16)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "warning_time_comparison.png"), dpi=300)
    plt.close(fig)
    
    # 5. Figure 5: Sensitivity vs False Alert Rate Trade-off
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sens = df_op["sensitivity_pct"].values
    far = df_op["false_alerts_per_hour"].values
    codes = df_op["model_code"].values
    colors = ["#a0aec0", "#718096", "#3182ce", "#805ad5", "#d69e2e", "#38a169"]
    
    for i in range(len(codes)):
        ax.scatter(far[i], sens[i], color=colors[i], s=120, zorder=5, edgecolor="black")
        ax.annotate(f"{codes[i]}\n({df_op['model_name'].iloc[i].split('(')[0].strip()})", (far[i], sens[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8, weight="bold")
                    
    ax.set_xlabel("False Alerts per Monitoring Hour", fontsize=10, weight="bold")
    ax.set_ylabel("Clinical Sensitivity (%)", fontsize=10, weight="bold")
    ax.set_title("Figure 5: Sensitivity vs False Alarm Burden Trade-off Across Benchmark Suite", fontsize=11, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "alert_tradeoff.png"), dpi=300)
    plt.close(fig)
    
    print("Saved all 5 publication figures in reports/figures_phase11/")

if __name__ == "__main__":
    generate_phase11_figures()
