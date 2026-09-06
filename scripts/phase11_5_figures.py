"""
Phase 11.5 — Publication Figures Generation Engine.

Generates the 7 core figures for Phase 11.5 in reports/figures_phase11_5/:
1. fig1_horizon_dependent_delta_auroc.png
2. fig2_trajectory_component_contribution.png
3. fig3_state_versus_direction_incremental.png
4. fig4_progression_versus_reversal_risk.png
5. fig5_framework_waterfall_forest.png
6. fig6_p2_vs_p6_discordance_regimes.png
7. fig7_warning_time_vs_false_alert_tradeoff.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIG_DIR = "reports/figures_phase11_5"
RES_DIR = "results/phase11_5_advantage_attribution"
os.makedirs(FIG_DIR, exist_ok=True)

# Styling parameters
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

NAVY = "#1f4e78"
CRIMSON = "#9e1b32"
TEAL = "#008080"
AMBER = "#d97706"
SLATE = "#475569"
PURPLE = "#6b21a8"
GREEN = "#15803d"

def plot_fig1_horizon_dependent_delta():
    path = os.path.join(RES_DIR, "horizon_advantage.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    contrasts_to_plot = [
        ("P6 vs P2", "P6 vs P2 (Sequential / Event)", CRIMSON, "o-"),
        ("P6 vs P3", "P6 vs P3 (Locked Snapshot)", NAVY, "s-"),
        ("P6 vs P5", "P6 vs P5 (Snapshot + Trajectory)", TEAL, "^-"),
        ("P5 vs P3", "P5 vs P3 (Trajectory Increment)", AMBER, "d--")
    ]
    
    # Horizons from earliest (60m) to delivery (0m)
    h_order = [60, 45, 30, 20, 10, 0]
    h_x = np.array([60, 45, 30, 20, 10, 0])
    x_pos = np.arange(len(h_order))
    x_labels = [">=60m\n(T=40m)", ">=45m\n(T=40m)", ">=30m", ">=20m", ">=10m", "Delivery\n(0m)"]
    
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)
    
    for c_name, label, col, style in contrasts_to_plot:
        sub = df[df["contrast"] == c_name].set_index("horizon_min").loc[h_order]
        deltas = sub["delta_auroc_point"].values
        ci_lo = sub["ci_95_low"].values
        ci_hi = sub["ci_95_high"].values
        
        ax.plot(x_pos, deltas, style, color=col, label=label, linewidth=2.2, markersize=7)
        ax.fill_between(x_pos, ci_lo, ci_hi, color=col, alpha=0.15)
        
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel(r"$\Delta$ AUROC (Paired Difference) $\pm$ 95% CI", fontsize=11, fontweight="bold")
    ax.set_xlabel("Prediction Warning Horizon Before Delivery", fontsize=11, fontweight="bold")
    ax.set_title("Experiment 11.5-A: Horizon-Dependent Advantage Dynamics\n(Paired Patient-Level Bootstrap B=2,000)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9.5, loc="upper left")
    
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig1_horizon_dependent_delta_auroc.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def plot_fig2_trajectory_component_contribution():
    path = os.path.join(RES_DIR, "trajectory_component_ablation.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    
    # Filter delivery and >=30m
    df_del = df[df["horizon_min"] == 0].copy()
    df_del = df_del[df_del["model_key"] != "Full_P5"] # Ablations only
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    concepts = df_del["ablated_concept"].values[::-1]
    deltas = df_del["delta_from_full_p5"].values[::-1]
    ci_los = df_del["ci_95_low"].values[::-1]
    ci_his = df_del["ci_95_high"].values[::-1]
    
    y_pos = np.arange(len(concepts))
    
    ax.axvline(0, color="gray", linestyle=":", linewidth=1.2)
    
    colors = [CRIMSON if d > 0.01 else (NAVY if d > 0 else SLATE) for d in deltas]
    
    ax.barh(y_pos, deltas, xerr=[deltas - ci_los, ci_his - deltas], color=colors, alpha=0.85, capsize=4, height=0.55, edgecolor="none")
    
    for i, (d, p) in enumerate(zip(deltas, df_del["p_value"].values[::-1])):
        sig_str = f" (p={p:.3f})" if p < 0.05 else " (ns)"
        ax.text(max(0.001, d) + 0.003, i, f"+{d:.4f}{sig_str}", va="center", fontsize=9, fontweight="bold")
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(concepts, fontsize=10)
    ax.set_xlabel(r"Incremental Contribution $\Delta$ AUROC = AUROC(P5) - AUROC(P5 $_{-C_i}$)", fontsize=11, fontweight="bold")
    ax.set_title("Experiment 11.5-B: Trajectory Leave-One-Component-Out (LOCO) Ablation\n(Evaluation at Delivery Endpoint, B=2,000 Clustered Bootstrap)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")
    
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig2_trajectory_component_contribution.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def plot_fig3_state_direction_incremental():
    path = os.path.join(RES_DIR, "state_direction_decomposition.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, h_val, title_h in [(ax1, 30, ">=30 min Horizon"), (ax2, 0, "Delivery (0m) Endpoint")]:
        sub = df[df["horizon_min"] == h_val]
        
        # Get unique models and AUROCs
        models_order = [
            "Model A: R_t (Snapshot Risk Alone)",
            "Model B: R_t + S_t (Snapshot + State)",
            "Model C: R_t + D_t (Snapshot + Direction)",
            "Model D: R_t + S_t + D_t (Snapshot + State + Direction)",
            "Model E: R_t + S_t + D_t + P_t (Snapshot + State + Direction + Persistence)"
        ]
        short_labels = ["A: $R_t$\n(Snapshot)", "B: $R_t+S_t$\n(+State)", "C: $R_t+D_t$\n(+Direction)", "D: $R_t+S_t+D_t$\n(+Both)", "E: $R_t+S_t+D_t+P_t$\n(+Persist)"]
        
        aurocs = []
        for m in models_order:
            row_match = sub[sub["model_b"] == m]
            if len(row_match) > 0:
                aurocs.append(row_match["auroc_b"].iloc[0])
            else:
                row_match_a = sub[sub["model_a"] == m]
                aurocs.append(row_match_a["auroc_a"].iloc[0])
                
        x_pos = np.arange(len(models_order))
        cols = [SLATE, TEAL, AMBER, NAVY, PURPLE]
        
        bars = ax.bar(x_pos, aurocs, color=cols, alpha=0.85, width=0.55)
        for b, v in zip(bars, aurocs):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            
        ax.set_ylim(0.50, max(aurocs) + 0.05)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylabel("Patient-Level AUROC", fontsize=10, fontweight="bold")
        ax.set_title(title_h, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        
    plt.suptitle("Experiment 11.5-C: Nested Physiological State & Direction Complementarity", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig3_state_versus_direction_incremental.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def plot_fig4_progression_reversal():
    path = os.path.join(RES_DIR, "progression_reversal_analysis.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    
    # Filter state rows (0 to 4)
    df_states = df[df["state_id"] >= 0].copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    state_labels = ["State 0\n(Stable)", "State 1\n(Emerging)", "State 2\n(Persistent)", "State 3\n(Progressive)", "State 4\n(Severe)"]
    x = np.arange(len(state_labels))
    width = 0.35
    
    # Plot 1: Prevalence
    prev_prog = df_states["prevalence_715_prog"].values * 100.0
    prev_rev = df_states["prevalence_715_rev"].values * 100.0
    
    rects1 = ax1.bar(x - width/2, prev_prog, width, label="Progressing (v > 0)", color=CRIMSON, alpha=0.85)
    rects2 = ax1.bar(x + width/2, prev_rev, width, label="Reversing (v < 0 / rev)", color=TEAL, alpha=0.85)
    
    for r in rects1:
        h = r.get_height()
        if not np.isnan(h) and h > 0:
            ax1.text(r.get_x() + r.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        if not np.isnan(h) and h > 0:
            ax1.text(r.get_x() + r.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            
    ax1.set_xticks(x)
    ax1.set_xticklabels(state_labels, fontsize=9.5)
    ax1.set_ylabel("Acidemia Prevalence (pH <= 7.15) %", fontsize=10, fontweight="bold")
    ax1.set_title("Acidemia Prevalence within Matched States", fontsize=11, fontweight="bold")
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.set_ylim(0, max(np.nanmax(prev_prog), np.nanmax(prev_rev)) + 12)
    
    # Plot 2: Prevalence Difference with CIs
    diffs = df_states["delta_prevalence_715"].values * 100.0
    ci_lo = df_states["ci_95_low_diff"].values * 100.0
    ci_hi = df_states["ci_95_high_diff"].values * 100.0
    
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1.2)
    valid_mask = ~np.isnan(diffs)
    
    ax2.errorbar(x[valid_mask], diffs[valid_mask], yerr=[diffs[valid_mask] - ci_lo[valid_mask], ci_hi[valid_mask] - diffs[valid_mask]],
                 fmt="o-", color=NAVY, ecolor=NAVY, elinewidth=2, capsize=5, markersize=7, linewidth=2)
                 
    for i in np.where(valid_mask)[0]:
        ax2.text(x[i], diffs[i] + 2.5, f"+{diffs[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
        
    ax2.set_xticks(x)
    ax2.set_xticklabels(state_labels, fontsize=9.5)
    ax2.set_ylabel(r"Risk Difference $\Delta_{\text{prev}}$ (%) $\pm$ 95% CI", fontsize=10, fontweight="bold")
    ax2.set_title("Incremental Progression Risk Elevation", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    plt.suptitle("Experiment 11.5-D: Directional Risk Differentiation within Matched States\n(Hypothesis H3: P(Acidemia | State, Progressing) > P(Acidemia | State, Reversing))", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig4_progression_versus_reversal_risk.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def plot_fig5_framework_waterfall():
    path = os.path.join(RES_DIR, "full_framework_decomposition.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    
    df_del = df[df["horizon_min"] == 0].sort_values("ladder_step")
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    steps = df_del["step_name"].values
    short_steps = [
        "1. Snapshot (R_t)",
        "2. + State (S_t)",
        "3. + Direction (D_t)",
        "4. + Persistence",
        "5. + Full Traj (P5)",
        "6. + Domain Severities",
        "7. + State Occupancy",
        "8. Full Multi-Domain (P6)"
    ]
    aurocs = df_del["auroc"].values
    deltas = df_del["stepwise_delta_auroc"].values
    
    x_pos = np.arange(len(short_steps))
    
    colors = [SLATE] + [TEAL if d > 0.005 else (NAVY if d > 0 else CRIMSON) for d in deltas[1:-1]] + [PURPLE]
    
    bars = ax.bar(x_pos, aurocs, color=colors, alpha=0.85, width=0.55)
    
    for b, a, d, i in zip(bars, aurocs, deltas, range(len(short_steps))):
        if i == 0:
            ax.text(b.get_x() + b.get_width()/2, a + 0.004, f"{a:.4f}\n(Base)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        else:
            sign = "+" if d >= 0 else ""
            ax.text(b.get_x() + b.get_width()/2, a + 0.004, f"{a:.4f}\n({sign}{d:.4f})", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            
    ax.set_ylim(0.55, max(aurocs) + 0.04)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_steps, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Patient-Level AUROC at Delivery", fontsize=11, fontweight="bold")
    ax.set_title("Experiment 11.5-E: Full Framework Component Ladder Decomposition\n(Explaining the P6 - P5 Delta = +0.0731 Gain across Multidomain Layers)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig5_framework_waterfall_forest.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def plot_fig6_discordance_regimes():
    path = os.path.join(RES_DIR, "prediction_discordance.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    
    df_del = df[df["horizon_min"] == 0].set_index("quadrant")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))
    
    quad_labels = ["A: Concordant\nCorrect", "B: P6-Only\nCorrect", "C: P2-Only\nCorrect", "D: Concordant\nIncorrect"]
    quad_keys = ["Group A: Concordant Correct", "Group B: P6-Only Correct", "Group C: P2-Only Correct", "Group D: Concordant Incorrect"]
    x = np.arange(len(quad_labels))
    
    # 1. Deceleration Burden & Depth
    decel_burden = [df_del.loc[k, "decel_burden_mean"] for k in quad_keys]
    ax1.bar(x, decel_burden, color=[NAVY, CRIMSON, TEAL, SLATE], alpha=0.85, width=0.5)
    for i, v in enumerate(decel_burden):
        ax1.text(i, v + 0.1, f"{v:.1f}m", ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(quad_labels, fontsize=8.5)
    ax1.set_ylabel("Decel Burden (min)", fontsize=9.5, fontweight="bold")
    ax1.set_title("Deceleration Burden Profile", fontsize=10.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    # 2. Short-Term Variability (STV)
    stv = [df_del.loc[k, "stv_mean"] for k in quad_keys]
    ax2.bar(x, stv, color=[NAVY, CRIMSON, TEAL, SLATE], alpha=0.85, width=0.5)
    for i, v in enumerate(stv):
        ax2.text(i, v + 0.1, f"{v:.2f}ms", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(quad_labels, fontsize=8.5)
    ax2.set_ylabel("STV (ms)", fontsize=9.5, fontweight="bold")
    ax2.set_title("Autonomic Variability (STV) Profile", fontsize=10.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    # 3. Tachysystole & UC
    uc = [df_del.loc[k, "uc_count_mean"] for k in quad_keys]
    ax3.bar(x, uc, color=[NAVY, CRIMSON, TEAL, SLATE], alpha=0.85, width=0.5)
    for i, v in enumerate(uc):
        ax3.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(quad_labels, fontsize=8.5)
    ax3.set_ylabel("Contractions / 20min", fontsize=9.5, fontweight="bold")
    ax3.set_title("Uterine Activity Burden", fontsize=10.5, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    # 4. Deterioration Velocity & State
    state = [df_del.loc[k, "research_state_mean"] for k in quad_keys]
    ax4.bar(x, state, color=[NAVY, CRIMSON, TEAL, SLATE], alpha=0.85, width=0.5)
    for i, v in enumerate(state):
        ax4.text(i, v + 0.05, f"S={v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(quad_labels, fontsize=8.5)
    ax4.set_ylabel("Mean Research State (0-4)", fontsize=9.5, fontweight="bold")
    ax4.set_title("Physiological State Severity", fontsize=10.5, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    plt.suptitle("Experiment 11.5-F: Physiological Discordance Profiling (P6 vs P2 Operating Regimes)", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig6_p2_vs_p6_discordance_regimes.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def plot_fig7_warning_time_tradeoff():
    path = os.path.join(RES_DIR, "threshold_robustness.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    
    # Plot Warning lead time vs FAR across models
    df_far = df[df["target_metric"] == "FAR"].copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    models_plot = [
        ("P6", "P6 (Full Proposed System)", CRIMSON, "o-"),
        ("P2", "P2 (Sequential Baseline)", TEAL, "s-"),
        ("P3", "P3 (Snapshot Baseline)", NAVY, "^-"),
        ("P5", "P5 (Snapshot + Trajectory)", AMBER, "d--"),
        ("P1", "P1 (Compact Baseline)", SLATE, "x:")
    ]
    
    for m_code, label, col, style in models_plot:
        sub = df_far[df_far["model_code"] == m_code].sort_values("target_value")
        fars = sub["actual_far_per_hour"].values
        leads = sub["median_warning_min"].values
        ax1.plot(fars, leads, style, color=col, label=label, linewidth=2, markersize=6)
        
    ax1.set_xlabel("False Alerts per Monitoring Hour (FAR)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Median Warning Lead Time (min)", fontsize=10, fontweight="bold")
    ax1.set_title("Warning Lead Time vs. False Alert Rate", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=8.5, loc="lower right")
    
    # Plot 2: Matched Sensitivity
    df_sens = df[df["target_metric"] == "Sensitivity"].copy()
    
    for m_code, label, col, style in models_plot:
        sub = df_sens[df_sens["model_code"] == m_code].sort_values("target_value")
        sens = sub["actual_sensitivity"].values * 100.0
        leads = sub["median_warning_min"].values
        ax2.plot(sens, leads, style, color=col, label=label, linewidth=2, markersize=6)
        
    ax2.set_xlabel("Sensitivity Target (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Median Warning Lead Time (min)", fontsize=10, fontweight="bold")
    ax2.set_title("Warning Lead Time at Matched Sensitivity", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(frameon=True, fontsize=8.5, loc="lower right")
    
    plt.suptitle("Experiments 11.5-G & 11.5-H: Threshold-Robust Warning Time Attribution", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig7_warning_time_vs_false_alert_tradeoff.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def generate_all_figures():
    print("=== GENERATING PHASE 11.5 PUBLICATION FIGURES ===")
    plot_fig1_horizon_dependent_delta()
    plot_fig2_trajectory_component_contribution()
    plot_fig3_state_direction_incremental()
    plot_fig4_progression_reversal()
    plot_fig5_framework_waterfall()
    plot_fig6_discordance_regimes()
    plot_fig7_warning_time_tradeoff()
    print("All Phase 11.5 figures generated.")

if __name__ == "__main__":
    generate_all_figures()
