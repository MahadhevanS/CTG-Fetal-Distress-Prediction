"""
Phase 10 Work Package 10E: Final Evidence Package & Publication Figures.

Generates:
- Table 1: Master Experimental Progression (Phases 2 through 10)
- Publication Figures (in reports/figures_phase10/):
  - fig1_research_pipeline.png
  - fig2_phase_progression.png
  - fig3_horizon_auroc.png
  - fig4_state_risk_gradient.png
  - fig5_state_trajectory.png
  - fig6_alert_tradeoff.png

Outputs:
- results/phase10_final/table1_experimental_progression.csv
- reports/figures_phase10/*.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = "results/phase10_final"
FIG_DIR = "reports/figures_phase10"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def generate_evidence_package():
    print("=== EXECUTING WORK PACKAGE 10E: FINAL EVIDENCE PACKAGE & FIGURES ===")
    
    # 1. Table 1: Complete Experimental Progression
    table1_rows = [
        {"Phase": "Phase 2", "Experiment": "Raw FHR Representation (CNN1D / TCN)", "AUROC_Delivery": 0.6593, "Key_Finding": "Raw temporal sequence alone insufficient; patient context required."},
        {"Phase": "Phase 3", "Experiment": "Patient-Level Aggregation (P90 / Attention MIL)", "AUROC_Delivery": 0.6701, "Key_Finding": "Extreme-value pooling (P90) superior to complex learned bag attention."},
        {"Phase": "Phase 4", "Experiment": "Clinical Knowledge Fusion (19 Descriptors)", "AUROC_Delivery": 0.7361, "Key_Finding": "Infusion of FIGO morphological features yields largest single predictive gain."},
        {"Phase": "Phase 6", "Experiment": "Continuous Acid-Base Supervision (Huber pH)", "AUROC_Delivery": 0.7426, "Key_Finding": "Continuous pH target regression improves calibration and discrimination."},
        {"Phase": "Phase 7", "Experiment": "Locked Replication & Statistical Audit", "AUROC_Delivery": 0.7426, "Key_Finding": "Verified reproducible baseline; reconciled Phase 4 vs Continuous Huber."},
        {"Phase": "Phase 8", "Experiment": "Causal Rolling 20-Min Inference", "AUROC_Delivery": 0.7426, "Key_Finding": "Causal real-time feasibility established (11.4 ms latency; AUROC>=30m = 0.5461)."},
        {"Phase": "Phase 9A", "Experiment": "Generic Temporal Modelling (Seq/Multi-horizon)", "AUROC_Delivery": 0.7312, "Key_Finding": "Generic sequence architectures fail to solve >=30m early warning."},
        {"Phase": "Phase 9B", "Experiment": "Physiology-Guided Deterioration States", "AUROC_Delivery": 0.7390, "Key_Finding": "Monotonic state-risk gradient observed (8.6% -> 47.6% acidemia)."},
        {"Phase": "Phase 9C", "Experiment": "State Transitions, Trajectory & Timing", "AUROC_Delivery": 0.7303, "Key_Finding": "Trajectory informs acute distress; biological boundary confirmed at >=30m."},
        {"Phase": "Phase 10", "Experiment": "Final Clinical Decision & System Lock", "AUROC_Delivery": 0.7426, "Key_Finding": "Snapshot + Trajectory Alerting locked; theoretical max lead time = 40 min."}
    ]
    df_table1 = pd.DataFrame(table1_rows)
    df_table1.to_csv(os.path.join(OUT_DIR, "table1_experimental_progression.csv"), index=False)
    print("Saved table1_experimental_progression.csv")
    
    # 2. Figure 1: Overall Research Pipeline
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis("off")
    
    box_props = dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#2b6cb0", linewidth=1.5)
    arrow_props = dict(arrowstyle="->", color="#2d3748", lw=1.5)
    
    ax.text(0.1, 0.8, "Continuous CTG\n(4 Hz FHR + UC)", ha="center", va="center", bbox=box_props, fontsize=10, weight="bold")
    ax.text(0.35, 0.8, "Causal Rolling Window\n(20-min duration, 2.5m stride)", ha="center", va="center", bbox=box_props, fontsize=10, weight="bold")
    ax.text(0.65, 0.8, "19 Morphological Descriptors\n(Baseline, STV, Decels, UC, Lag)", ha="center", va="center", bbox=box_props, fontsize=10, weight="bold")
    
    ax.text(0.35, 0.4, "Snapshot Prediction R(t)\n(Continuous Huber pH Regression)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.5), fontsize=10, weight="bold")
    ax.text(0.65, 0.4, "Physiological State S(t) & Trajectory\n(Velocity V, Persistence P, Reversals R)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fefcbf", edgecolor="#d69e2e", linewidth=1.5), fontsize=10, weight="bold")
    
    ax.text(0.5, 0.1, "Final Clinical Decision & Hybrid Alerting Engine\n(p >= 0.85*tau + S >= 2 + P >= 2 -> 94.7% Spec, 0.075 FAR/hr)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.6", facecolor="#e6fffa", edgecolor="#319795", linewidth=2.0), fontsize=11, weight="bold")
    
    # Arrows
    ax.annotate("", xy=(0.23, 0.8), xytext=(0.17, 0.8), arrowprops=arrow_props)
    ax.annotate("", xy=(0.53, 0.8), xytext=(0.47, 0.8), arrowprops=arrow_props)
    ax.annotate("", xy=(0.35, 0.48), xytext=(0.35, 0.72), arrowprops=arrow_props)
    ax.annotate("", xy=(0.65, 0.48), xytext=(0.65, 0.72), arrowprops=arrow_props)
    ax.annotate("", xy=(0.45, 0.18), xytext=(0.35, 0.32), arrowprops=arrow_props)
    ax.annotate("", xy=(0.55, 0.18), xytext=(0.65, 0.32), arrowprops=arrow_props)
    
    ax.set_title("Figure 1: Proposed Causal Physiological Deterioration Early-Warning Architecture", fontsize=12, weight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_research_pipeline.png"), dpi=300)
    plt.close(fig)
    
    # 3. Figure 2: Phase Progression Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    phases = [r["Phase"] for r in table1_rows]
    aurocs = [r["AUROC_Delivery"] for r in table1_rows]
    colors = ["#cbd5e0", "#cbd5e0", "#63b3ed", "#3182ce", "#3182ce", "#3182ce", "#e2e8f0", "#e2e8f0", "#e2e8f0", "#2b6cb0"]
    
    bars = ax.bar(phases, aurocs, color=colors, edgecolor="#2d3748", width=0.6)
    ax.axhline(0.80, color="#e53e3e", linestyle="--", linewidth=1.5, label="Aspiration Target (AUROC = 0.80)")
    ax.axhline(0.7426, color="#2b6cb0", linestyle=":", linewidth=1.2, label="Locked Delivery Baseline (AUROC = 0.7426)")
    
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.008, f"{val:.4f}", ha="center", va="bottom", fontsize=8, weight="bold")
        
    ax.set_ylim(0.55, 0.85)
    ax.set_ylabel("Delivery AUROC (pH <= 7.15)", fontsize=10, weight="bold")
    ax.set_title("Figure 2: Methodological Evolution & Performance Trajectory Across Phases 2-10", fontsize=11, weight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_phase_progression.png"), dpi=300)
    plt.close(fig)
    
    # 4. Figure 3: AUROC vs Warning Horizon
    df_hor = pd.read_csv(os.path.join(OUT_DIR, "horizon_analysis.csv"))
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    x_times = [60, 45, 30, 20, 10, 0]
    y_snap = df_hor["snapshot_auroc_715"].values
    y_traj = df_hor["trajectory_auroc_715"].values
    y_full = df_hor["full_fusion_auroc_715"].values
    
    ax.plot(x_times, y_snap, marker="o", color="#3182ce", lw=2, label="Model A (Snapshot Baseline)")
    ax.plot(x_times, y_traj, marker="s", color="#d69e2e", lw=2, label="Model C (Snapshot + State + Trajectory)")
    ax.plot(x_times, y_full, marker="^", color="#38a169", lw=2, label="Model E (Full Physiological Fusion)")
    
    ax.axvline(40, color="#e53e3e", linestyle="--", alpha=0.7, label="Theoretical Lead Limit (40 min in 60m Trace)")
    ax.set_xlim(65, -5) # inverted time to delivery
    ax.set_ylim(0.50, 0.80)
    ax.set_xlabel("Warning Horizon: Minutes Before Delivery", fontsize=10, weight="bold")
    ax.set_ylabel("Patient-Level AUROC (pH <= 7.15)", fontsize=10, weight="bold")
    ax.set_title("Figure 3: Multi-Horizon Discrimination Approaching Delivery", fontsize=11, weight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_horizon_auroc.png"), dpi=300)
    plt.close(fig)
    
    # 5. Figure 4: State-Risk Gradient
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    states = ["State 0\n(Stable)", "State 1\n(Emerging)", "State 2\n(Persistent)", "State 3\n(Progressive)", "State 4\n(Severe)"]
    p_715 = [8.64, 14.19, 22.81, 35.80, 47.62]
    p_705 = [1.23, 3.38, 8.77, 16.05, 26.19]
    
    x = np.arange(len(states))
    w = 0.35
    
    ax.bar(x - w/2, p_715, width=w, label="Primary Acidemia (pH <= 7.15)", color="#3182ce", edgecolor="#2d3748")
    ax.bar(x + w/2, p_705, width=w, label="Severe Acidemia (pH <= 7.05)", color="#e53e3e", edgecolor="#2d3748")
    
    for i in range(len(states)):
        ax.text(x[i] - w/2, p_715[i] + 1.0, f"{p_715[i]:.1f}%", ha="center", fontsize=8, weight="bold")
        ax.text(x[i] + w/2, p_705[i] + 1.0, f"{p_705[i]:.1f}%", ha="center", fontsize=8, weight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels(states, fontsize=9, weight="bold")
    ax.set_ylabel("Observed Outcome Prevalence (%)", fontsize=10, weight="bold")
    ax.set_title("Figure 4: Monotonic Physiological State-Risk Gradient (Phase 9B/9C)", fontsize=11, weight="bold")
    ax.set_ylim(0, 55)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_state_risk_gradient.png"), dpi=300)
    plt.close(fig)
    
    # 6. Figure 5: State Trajectory (Progression vs Reversal)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    t = np.arange(7)
    traj_prog = [0, 1, 1, 2, 2, 3, 4]
    traj_rev = [0, 1, 2, 2, 1, 1, 0]
    traj_stab = [0, 0, 1, 1, 1, 1, 1]
    
    ax.plot(t, traj_prog, marker="o", color="#e53e3e", lw=2.5, label="Progressive Trajectory (Acidemia Risk ~ 39.9%)")
    ax.plot(t, traj_rev, marker="s", color="#38a169", lw=2.5, label="Reversing Trajectory (Acidemia Risk ~ 15.5%)")
    ax.plot(t, traj_stab, marker="^", color="#3182ce", lw=1.8, linestyle="--", label="Stable / Transient Trajectory")
    
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["State 0 (Stable)", "State 1 (Emerging)", "State 2 (Persistent)", "State 3 (Progressive)", "State 4 (Severe)"], fontsize=9, weight="bold")
    ax.set_xlabel("Consecutive 20-Min Monitoring Windows", fontsize=10, weight="bold")
    ax.set_title("Figure 5: Trajectory Dynamics Distinguish Acute Compromise in Matched Current States", fontsize=11, weight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_state_trajectory.png"), dpi=300)
    plt.close(fig)
    
    # 7. Figure 6: Clinical Alert Trade-off
    df_op = pd.read_csv(os.path.join(OUT_DIR, "clinical_operating_points.csv"))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    sens = df_op["sensitivity_pct"].values
    far = df_op["false_alerts_per_hour"].values
    labels = ["P1 (Single Threshold)", "P2 (2-Win Persistence)", "P3 (State S>=3)", "P4 (State Prog)", "P5 (Hybrid)"]
    colors = ["#e53e3e", "#dd6b20", "#805ad5", "#319795", "#2b6cb0"]
    
    for i in range(len(labels)):
        ax.scatter(far[i], sens[i], color=colors[i], s=120, zorder=5, edgecolor="black")
        ax.annotate(f"{labels[i]}\n(Lead: {df_op['median_lead_time_min'].iloc[i]}m)", (far[i], sens[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8, weight="bold")
                    
    ax.set_xlabel("False Alerts per Monitoring Hour", fontsize=10, weight="bold")
    ax.set_ylabel("Clinical Sensitivity (%)", fontsize=10, weight="bold")
    ax.set_title("Figure 6: Sensitivity vs False Alert Burden Trade-off Across Alert Policies", fontsize=11, weight="bold")
    ax.set_ylim(40, 105)
    ax.set_xlim(-0.2, 3.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_alert_tradeoff.png"), dpi=300)
    plt.close(fig)
    
    print("Saved all 6 publication figures in reports/figures_phase10/")
    print("Work Package 10E Complete.")

if __name__ == "__main__":
    generate_evidence_package()
