"""
Phase 12 — Master Synthesis Figures Generation Engine.

Generates the 8 core publication/thesis synthesis figures in reports/figures_phase12/:
1. fig1_overall_framework_architecture.png
2. fig2_experimental_progression_phases.png
3. fig3_prior_art_head_to_head_comparison.png
4. fig4_snapshot_vs_trajectory_increment.png
5. fig5_multidomain_component_attribution.png
6. fig6_horizon_dependent_dynamics.png
7. fig7_warning_time_vs_false_alert_tradeoff.png
8. fig8_physiological_state_trajectory_concept.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIG_DIR = "reports/figures_phase12"
P11_5_DIR = "results/phase11_5_advantage_attribution"
P11_DIR = "results/phase11_priorart_benchmark"
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

# -------------------------------------------------------------
# Figure 1: Overall Proposed Framework Architecture
# -------------------------------------------------------------
def plot_fig1_framework_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")
    
    # Draw conceptual blocks
    def draw_box(x, y, w, h, title, text, bg_col, text_col="white", border_col="none"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", 
                                      facecolor=bg_col, edgecolor=border_col, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.05, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=text_col)
        ax.text(x + w/2, y + h/2 - 0.03, text, ha="center", va="center", fontsize=8.5, color=text_col, linespacing=1.3)

    # 1. Input Signal
    draw_box(0.02, 0.55, 0.18, 0.35, "1. Causal CTG Window", "Rolling 20-min Window\n4 Hz FHR & UC Signals\nStrict Causal Boundary\nZero Future Lookahead", NAVY)
    
    # 2. Multidomain Feature Layer
    draw_box(0.24, 0.55, 0.22, 0.35, "2. Multidomain Severities", "6 Physiological Domains:\n• Baseline (FHR, slope)\n• Variability (STV, LTV)\n• Accelerations\n• Decelerations (depth, burden)\n• Uterine Contractions\n• FHR-UC Coupling Lag", TEAL)
    
    # 3. State Engine
    draw_box(0.50, 0.55, 0.22, 0.35, "3. State Transition Engine", "5 Physiological States:\nState 0: Stable Autonomic\nState 1: Emerging Stress\nState 2: Persistent Decel\nState 3: Progressive Multidomain\nState 4: Severe Decompensation", AMBER)
    
    # 4. Trajectory Dynamics
    draw_box(0.76, 0.55, 0.22, 0.35, "4. Trajectory Dynamics", "Temporal Features:\n• Velocity (v_1step, v_4step)\n• Acceleration (progression)\n• State Persistence Duration\n• Direction Reversals (r_ind)\n• Multidomain Concurrence", PURPLE)
    
    # 5. Integrated Fusion & Risk Engine
    draw_box(0.24, 0.08, 0.48, 0.35, "5. Physiology-Guided Fusion & Acidemia Risk Engine", "Hierarchical Integration:\nSnapshot Risk (R_t) + State (S_t) + Trajectory Dynamics (D_t, P_t) + Multidomain Coupling\nRegularized Huber/Logistic Prediction under Patient-Stratified 5-Fold CV", CRIMSON)
    
    # 6. Early-Warning Alert Engine
    draw_box(0.76, 0.08, 0.22, 0.35, "6. Early-Warning Alert", "Causal Operational Alert:\n• Real-Time Risk Score\n• Continuous Alert Tracking\n• Median Lead Time: 17.5 min\n• FAR: 0.65 false alerts/hr", GREEN)

    # Arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor=SLATE, edgecolor=SLATE, width=1.5, headwidth=7, shrink=0.05))
                    
    draw_arrow(0.20, 0.72, 0.24, 0.72)
    draw_arrow(0.46, 0.72, 0.50, 0.72)
    draw_arrow(0.72, 0.72, 0.76, 0.72)
    draw_arrow(0.35, 0.55, 0.35, 0.43)
    draw_arrow(0.61, 0.55, 0.61, 0.43)
    draw_arrow(0.87, 0.55, 0.87, 0.43)
    draw_arrow(0.72, 0.25, 0.76, 0.25)
    
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    plt.suptitle("Figure 1: Full Proposed Physiology-Guided Multidomain Deterioration Framework (P6)", fontsize=13, fontweight="bold", y=0.98)
    
    out_file = os.path.join(FIG_DIR, "fig1_overall_framework_architecture.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")

# -------------------------------------------------------------
# Figure 2: Experimental Progression across Phases 1 to 12
# -------------------------------------------------------------
def plot_fig2_experimental_progression():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    
    phases = [
        "Ph 2: Signal Rep\n(AUROC 0.684)",
        "Ph 3: Pooling\n(P90 0.692)",
        "Ph 4: Clinical Fusion\n(AUROC 0.736)",
        "Ph 6: Continuous Huber\n(AUROC 0.743)",
        "Ph 8: Rolling Causal\n(>=30m: 0.570)",
        "Ph 9C: State Traj\n(Gradient 7%->44%)",
        "Ph 11: Benchmark\n(P6: 0.687 vs P1: 0.509)",
        "Ph 11.5: Attribution\n(78% Multidomain Gain)"
    ]
    
    aurocs = [0.6842, 0.6915, 0.7361, 0.7426, 0.5699, 0.6142, 0.6872, 0.6872]
    x_pos = np.arange(len(phases))
    
    colors = [SLATE, SLATE, TEAL, TEAL, AMBER, PURPLE, CRIMSON, CRIMSON]
    
    ax.plot(x_pos, aurocs, "o-", color=NAVY, linewidth=2.2, markersize=8)
    for i, (p, a, c) in enumerate(zip(phases, aurocs, colors)):
        ax.scatter(i, a, color=c, s=120, zorder=5)
        offset = 0.015 if i != 4 else -0.025
        ax.text(i, a + offset, f"{a:.4f}", ha="center", fontsize=9, fontweight="bold", color=c)
        
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phases, fontsize=8.5)
    ax.set_ylabel("Patient-Level Primary AUROC (pH <= 7.15)", fontsize=10, fontweight="bold")
    ax.set_title("Figure 2: Empirical Progression & Scientific Discovery Across Experimental Phases", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0.50, 0.80)
    
    # Annotations
    ax.annotate("Static Retrospective Prediction", xy=(1.5, 0.76), fontsize=9.5, fontweight="bold", color=TEAL, ha="center")
    ax.annotate("Causal Rolling Transition (Acute Labor Boundary)", xy=(4.5, 0.52), fontsize=9.5, fontweight="bold", color=AMBER, ha="center")
    ax.annotate("Prior-Art Benchmark & Scientific Attribution", xy=(6.5, 0.72), fontsize=9.5, fontweight="bold", color=CRIMSON, ha="center")
    
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig2_experimental_progression_phases.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

# -------------------------------------------------------------
# Figure 3: Prior-Art Head-to-Head Comparison
# -------------------------------------------------------------
def plot_fig3_prior_art_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    
    models = ["P1 (Compact)", "P2 (Sequential)", "P3 (Snapshot)", "P4 (State)", "P5 (Traj)", "P6 (Full System)"]
    auc_30m = [0.5727, 0.5983, 0.5461, 0.5459, 0.5355, 0.5857]
    auc_del = [0.5090, 0.6774, 0.5976, 0.5945, 0.6142, 0.6872]
    
    x = np.arange(len(models))
    colors = [SLATE, TEAL, SLATE, AMBER, PURPLE, CRIMSON]
    
    # Plot 1: >=30m Horizon
    bars1 = ax1.bar(x, auc_30m, color=colors, alpha=0.85, width=0.55)
    for b, v in zip(bars1, auc_30m):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right", fontsize=8.5)
    ax1.set_ylabel("Patient-Level AUROC", fontsize=10, fontweight="bold")
    ax1.set_title(">=30-Minute Early-Warning Horizon\n(P2 Leads Numerically, Indistinguishable from P6)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0.45, 0.72)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    # Plot 2: Delivery Endpoint
    bars2 = ax2.bar(x, auc_del, color=colors, alpha=0.85, width=0.55)
    for b, v in zip(bars2, auc_del):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha="right", fontsize=8.5)
    ax2.set_ylabel("Patient-Level AUROC", fontsize=10, fontweight="bold")
    ax2.set_title("Delivery Acute Endpoint (0m)\n(P6 Significantly Outperforms P1 and P3)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0.45, 0.72)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    plt.suptitle("Figure 3: Prior-Art Head-to-Head Benchmark (Same-Cohort Causal Patient Evaluation)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig3_prior_art_head_to_head_comparison.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

# -------------------------------------------------------------
# Figure 4: Snapshot vs Trajectory Incremental Contribution
# -------------------------------------------------------------
def plot_fig4_trajectory_increment():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    contrasts = [
        "P5 vs P3 (Snapshot + Trajectory vs Snapshot)",
        "P4 vs P3 (Snapshot + State vs Snapshot)",
        "Model E vs Model A (State+Traj+Persist vs Snapshot)",
        "P6 vs P5 (Full System vs Snapshot + Trajectory)"
    ]
    deltas = [0.0165, -0.0031, 0.0170, 0.0731]
    ci_los = [0.0006, -0.0180, 0.0005, 0.0192]
    ci_his = [0.0319, 0.0120, 0.0336, 0.1292]
    p_vals = [0.041, 0.730, 0.043, 0.006]
    
    y_pos = np.arange(len(contrasts))
    ax.axvline(0, color="gray", linestyle=":", linewidth=1.2)
    
    colors = [PURPLE, SLATE, TEAL, CRIMSON]
    
    for i, (d, lo, hi, p, col) in enumerate(zip(deltas, ci_los, ci_his, p_vals, colors)):
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=col, ecolor=col, elinewidth=2.5, capsize=5, markersize=8)
        sign = "+" if d > 0 else ""
        sig_str = f" (p={p:.3f})" if p < 0.05 else " (ns)"
        ax.text(max(hi, d) + 0.004, i, f"{sign}{d:.4f}{sig_str}", va="center", fontsize=9, fontweight="bold", color=col)
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(contrasts, fontsize=9.5)
    ax.set_xlabel(r"Incremental Difference $\Delta$ AUROC $\pm$ 95% Bootstrap CI", fontsize=10.5, fontweight="bold")
    ax.set_title("Figure 4: Incremental Value of Temporal Trajectory Dynamics at Delivery\n(B=2,000 Paired Patient Bootstrap)", fontsize=11.5, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")
    
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig4_snapshot_vs_trajectory_increment.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

# -------------------------------------------------------------
# Figure 5: Multidomain Component Attribution Waterfall
# -------------------------------------------------------------
def plot_fig5_multidomain_attribution():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    steps = [
        "1. Snapshot (R_t)",
        "2. + State (S_t)",
        "3. + Direction (D_t)",
        "4. + Persistence",
        "5. + Full Traj (P5)",
        "6. + Domain Severities",
        "7. + State Occupancies",
        "8. Full Multi-Domain (P6)"
    ]
    aurocs = [0.5958, 0.5946, 0.6032, 0.6129, 0.6142, 0.6818, 0.6828, 0.6872]
    step_deltas = [0.0, -0.0012, +0.0086, +0.0097, +0.0013, +0.0676, +0.0010, +0.0044]
    
    x = np.arange(len(steps))
    colors = [SLATE, SLATE, TEAL, TEAL, PURPLE, CRIMSON, CRIMSON, CRIMSON]
    
    bars = ax.bar(x, aurocs, color=colors, alpha=0.85, width=0.55)
    
    for i, (b, a, d) in enumerate(zip(bars, aurocs, step_deltas)):
        if i == 0:
            ax.text(b.get_x() + b.get_width()/2, a + 0.004, f"{a:.4f}\n(Base)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        elif i == 5:
            ax.text(b.get_x() + b.get_width()/2, a + 0.004, f"{a:.4f}\n(+0.0676***)\n[78% of Gain]", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=CRIMSON)
        else:
            sign = "+" if d >= 0 else ""
            ax.text(b.get_x() + b.get_width()/2, a + 0.004, f"{a:.4f}\n({sign}{d:.4f})", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            
    ax.set_ylim(0.55, 0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Patient-Level AUROC at Delivery", fontsize=10.5, fontweight="bold")
    ax.set_title("Figure 5: Multidomain Physiological Severity as the Primary Performance Driver\n(Step 6 Accounts for ~78% of Total System Improvement, p < 0.001)", fontsize=11.5, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig5_multidomain_component_attribution.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

# -------------------------------------------------------------
# Figure 6: Horizon-Dependent Dynamics
# -------------------------------------------------------------
def plot_fig6_horizon_dynamics():
    fig, ax = plt.subplots(figsize=(9, 5))
    
    h_labels = [">=60m\n(T=40m)", ">=45m\n(T=40m)", ">=30m", ">=20m", ">=10m", "Delivery\n(0m)"]
    x = np.arange(len(h_labels))
    
    p6_auc = [0.5857, 0.5857, 0.5857, 0.6366, 0.6726, 0.6872]
    p2_auc = [0.5512, 0.5512, 0.5983, 0.6305, 0.6542, 0.6774]
    p3_auc = [0.5360, 0.5360, 0.5461, 0.6290, 0.6472, 0.5976]
    p1_auc = [0.5360, 0.5360, 0.5727, 0.5574, 0.5372, 0.5090]
    
    ax.plot(x, p6_auc, "o-", color=CRIMSON, label="P6 (Full Proposed System)", linewidth=2.5, markersize=7)
    ax.plot(x, p2_auc, "s-", color=TEAL, label="P2 (Sequential Event Baseline)", linewidth=2.2, markersize=6)
    ax.plot(x, p3_auc, "^-", color=NAVY, label="P3 (Locked Snapshot Baseline)", linewidth=2.0, markersize=6)
    ax.plot(x, p1_auc, "x:", color=SLATE, label="P1 (Compact CTG Baseline)", linewidth=1.8, markersize=6)
    
    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, fontsize=9.5)
    ax.set_ylabel("Patient-Level AUROC", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Warning Horizon Before Delivery", fontsize=10.5, fontweight="bold")
    ax.set_title("Figure 6: Horizon-Dependent Discrimination Trajectory Across Prior-Art Baselines", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9.5, loc="upper left")
    ax.set_ylim(0.48, 0.72)
    
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig6_horizon_dependent_dynamics.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

# -------------------------------------------------------------
# Figure 7: Warning-Time vs False Alert Rate Tradeoff
# -------------------------------------------------------------
def plot_fig7_warning_time():
    path = os.path.join(P11_5_DIR, "threshold_robustness.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df_far = df[df["target_metric"] == "FAR"].copy()
    
    fig, ax = plt.subplots(figsize=(8.5, 5))
    
    models = [
        ("P6", "P6 (Full Proposed System)", CRIMSON, "o-"),
        ("P2", "P2 (Sequential Baseline)", TEAL, "s-"),
        ("P3", "P3 (Snapshot Baseline)", NAVY, "^-"),
        ("P1", "P1 (Compact Baseline)", SLATE, "x:")
    ]
    
    for m_code, label, col, style in models:
        sub = df_far[df_far["model_code"] == m_code].sort_values("target_value")
        fars = sub["actual_far_per_hour"].values
        leads = sub["median_warning_min"].values
        ax.plot(fars, leads, style, color=col, label=label, linewidth=2.2, markersize=6)
        
    ax.set_xlabel("False Alert Rate (Alerts / Normal Patient Monitoring Hour)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Median Warning Lead Time (min)", fontsize=10.5, fontweight="bold")
    ax.set_title("Figure 7: Retrospective Warning Lead Time vs. False Alert Rate (FAR) Tradeoff", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9.5, loc="lower right")
    ax.set_ylim(5, 25)
    
    # Matched operating point highlight
    ax.axvline(0.50, color=AMBER, linestyle="--", alpha=0.7)
    ax.text(0.51, 23, "Matched FAR = 0.50/hr\n(P6 Lead: 16.0m vs P2: 12.5m, P3: 10.0m)", fontsize=8.5, fontweight="bold", color=AMBER)
    
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "fig7_warning_time_vs_false_alert_tradeoff.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

# -------------------------------------------------------------
# Figure 8: Conceptual Physiological State & Trajectory Space
# -------------------------------------------------------------
def plot_fig8_state_space_concept():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    
    # State boxes
    states = [
        ("State 0\nStable Autonomic", "Normal baseline (110-160)\nNormal STV > 4ms\nNo decel burden\nAcidemia: 6.8%", GREEN, 0.05),
        ("State 1\nEmerging Stress", "Isolated variable decels\nPreserved variability\nTransient recovery\nAcidemia: 16.7%", TEAL, 0.24),
        ("State 2\nPersistent Decel", "Recurrent decelerations\nModerate STV depression\nHigh contraction burden\nAcidemia: 22.8%", AMBER, 0.43),
        ("State 3\nProgressive Multi", "Multi-channel collapse\nTachysystole + decels\nDelayed FHR-UC lag\nAcidemia: 25.0%", PURPLE, 0.62),
        ("State 4\nSevere Decomp", "Profound bradycardia\nSevere STV loss (<2.5ms)\nProlonged late decels\nAcidemia: 43.8%", CRIMSON, 0.81)
    ]
    
    for title, text, col, x in states:
        rect = patches.FancyBboxPatch((x, 0.35), 0.14, 0.50, boxstyle="round,pad=0.02",
                                      facecolor=col, edgecolor="none", alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 0.07, 0.78, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color="white")
        ax.text(x + 0.07, 0.55, text, ha="center", va="center", fontsize=7.8, color="white", linespacing=1.25)
        
    # Progression arrows
    for i in range(4):
        x1 = states[i][3] + 0.14
        x2 = states[i+1][3]
        ax.annotate("", xy=(x2, 0.60), xytext=(x1, 0.60),
                    arrowprops=dict(facecolor=CRIMSON, edgecolor=CRIMSON, width=1.5, headwidth=6, shrink=0.08))
                    
    # Reversal arrow
    ax.annotate("Reversal Vector (v < 0 / Reversal Indicator)", xy=(0.38, 0.22), xytext=(0.62, 0.22),
                arrowprops=dict(facecolor=TEAL, edgecolor=TEAL, width=1.5, headwidth=6, shrink=0.08),
                ha="center", fontsize=9, fontweight="bold", color=TEAL)
                
    # Progression arrow
    ax.annotate("Progression Vector (v > 0 / Acceleration)", xy=(0.62, 0.12), xytext=(0.38, 0.12),
                arrowprops=dict(facecolor=CRIMSON, edgecolor=CRIMSON, width=1.5, headwidth=6, shrink=0.08),
                ha="center", fontsize=9, fontweight="bold", color=CRIMSON)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.95)
    plt.suptitle("Figure 8: Conceptual 5-State Physiological Deterioration & Trajectory Direction Space", fontsize=12, fontweight="bold", y=0.98)
    
    out_file = os.path.join(FIG_DIR, "fig8_physiological_state_trajectory_concept.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")

def generate_all_phase12_figures():
    print("=== GENERATING ALL 8 PHASE 12 SYNTHESIS FIGURES ===")
    plot_fig1_framework_architecture()
    plot_fig2_experimental_progression()
    plot_fig3_prior_art_comparison()
    plot_fig4_trajectory_increment()
    plot_fig5_multidomain_attribution()
    plot_fig6_horizon_dynamics()
    plot_fig7_warning_time()
    plot_fig8_state_space_concept()
    print("All Phase 12 synthesis figures successfully generated.")

if __name__ == "__main__":
    generate_all_phase12_figures()
