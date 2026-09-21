"""
Phase 5 Figure Generator: Generates all 7 diagnostic figures for Phase 5.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def generate_phase5_figures():
    results_path = "results/phase5_multires/phase5_results.json"
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        res = json.load(f)

    out_dir = "reports/figures_phase5"
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # -----------------------------------------------------------------------
    # Fig 1: AUROC vs Temporal Context Length
    # -----------------------------------------------------------------------
    scales = ["5m", "10m", "20m", "30m", "40m", "60m"]
    scale_mins = [5, 10, 20, 30, 40, 60]
    aurocs = [res["scale_audit"][s]["auroc"] for s in scales]
    ci_lows = [res["scale_audit"][s]["ci"][0] for s in scales]
    ci_highs = [res["scale_audit"][s]["ci"][1] for s in scales]
    yerr = [[a - l for a, l in zip(aurocs, ci_lows)], [h - a for a, h in zip(aurocs, ci_highs)]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(scale_mins, aurocs, yerr=yerr, fmt='-o', color='#1f77b4', ecolor='#1f77b4', elinewidth=2, capsize=5, lw=2.5, markersize=8, label='Single-Scale 1D ResNet (P90)')
    ax.axhline(res["scale_audit"]["20m"]["auroc"], color='gray', linestyle='--', alpha=0.7, label='20-min Reference (0.670)')
    ax.set_xlabel("Temporal Context Length (Minutes)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Patient-Level AUROC (95% CI)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 1: CTU-UHB 1D Waveform Discrimination vs Context Length", fontsize=12, fontweight='bold')
    ax.set_xticks(scale_mins)
    for x, y in zip(scale_mins, aurocs):
        ax.text(x, y + 0.012, f"{y:.4f}", ha='center', fontsize=9, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_context_length_auroc.png"), dpi=300)
    plt.close()

    # Targets & Patient Scores
    y_true = np.array(res["patient_scores"]["labels"])
    s_10 = np.array(res["patient_scores"]["signal_10m"])
    s_20 = np.array(res["patient_scores"]["signal_20m"])
    s_40 = np.array(res["patient_scores"]["signal_40m"])
    s_ms = np.array(res["patient_scores"]["multiscale_signal"])
    c_20 = np.array(res["patient_scores"]["clinical_20m"])
    p4_final = np.array(res["patient_scores"]["phase4_baseline"])
    p5_final = np.array(res["patient_scores"]["phase5_candidate"])

    # -----------------------------------------------------------------------
    # Fig 2: Correlation Between Temporal Scale Predictions
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [
        (s_10, s_20, "10-min ($p_{10}$)", "20-min ($p_{20}$)", res["scale_correlations"]["pearson_10_20"], axes[0]),
        (s_20, s_40, "20-min ($p_{20}$)", "40-min ($p_{40}$)", res["scale_correlations"]["pearson_20_40"], axes[1]),
        (s_10, s_40, "10-min ($p_{10}$)", "40-min ($p_{40}$)", res["scale_correlations"]["pearson_10_40"], axes[2])
    ]
    for x_s, y_s, x_lbl, y_lbl, r_val, ax in pairs:
        ax.scatter(x_s[y_true == 0], y_s[y_true == 0], color='#1f77b4', alpha=0.5, label='Normal (pH > 7.15)', s=20)
        ax.scatter(x_s[y_true == 1], y_s[y_true == 1], color='#d62728', alpha=0.8, label='Acidotic (pH <= 7.15)', s=30, marker='^')
        ax.set_xlabel(x_lbl, fontsize=10, fontweight='bold')
        ax.set_ylabel(y_lbl, fontsize=10, fontweight='bold')
        ax.set_title(f"r = {r_val:.3f}", fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', frameon=True, fontsize=8)
    plt.suptitle("Figure 2: Multi-Scale Score Concordance & Independent Variance", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_scale_correlations.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 3: Short / Medium / Long Score Distributions
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    scale_tuples = [(s_10, "10-min Short Score", axes[0]), (s_20, "20-min Medium Score", axes[1]), (s_40, "40-min Long Score", axes[2])]
    for s_arr, title, ax in scale_tuples:
        ax.hist(s_arr[y_true == 0], bins=20, alpha=0.6, color='#1f77b4', label='Normal (pH > 7.15)', density=True, edgecolor='black')
        ax.hist(s_arr[y_true == 1], bins=20, alpha=0.7, color='#d62728', label='Acidotic (pH <= 7.15)', density=True, edgecolor='black')
        ax.set_xlabel("Predicted Probability", fontsize=10, fontweight='bold')
        ax.set_ylabel("Density", fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, fontsize=8)
    plt.suptitle("Figure 3: Score Distributions Across Individual Temporal Scales", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_score_distributions.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 4: Patient-Level ROC Curves
    # -----------------------------------------------------------------------
    fpr_p5, tpr_p5, _ = roc_curve(y_true, p5_final)
    fpr_p4, tpr_p4, _ = roc_curve(y_true, p4_final)
    fpr_ms, tpr_ms, _ = roc_curve(y_true, s_ms)
    fpr_s20, tpr_s20, _ = roc_curve(y_true, s_20)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_p5, tpr_p5, color='#2ca02c', lw=2.5, label=f"Phase 5 Multi-Scale+Clinical (AUROC = {res['matched_models']['phase5_final_candidate']['auroc']:.4f})")
    ax.plot(fpr_p4, tpr_p4, color='#1f77b4', lw=2.0, linestyle='--', label=f"Phase 4 Baseline 20m (AUROC = {auc(fpr_p4, tpr_p4):.4f})")
    ax.plot(fpr_ms, tpr_ms, color='#9467bd', lw=2.0, linestyle='-.', label=f"Multi-Scale Signal Only (AUROC = {res['matched_models']['multiscale_signal']['auroc']:.4f})")
    ax.plot(fpr_s20, tpr_s20, color='#ff7f0e', lw=1.8, linestyle=':', label=f"20-min Signal Control (AUROC = {res['scale_audit']['20m']['auroc']:.4f})")
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.6, label='Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax.set_title('Figure 4: Patient-Level Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_roc_comparison.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 5: Model Benchmark Bar Comparison
    # -----------------------------------------------------------------------
    models_to_plot = [
        ("Signal 20m", res["scale_audit"]["20m"]["auroc"], res["scale_audit"]["20m"]["ci"]),
        ("Signal 10m", res["scale_audit"]["10m"]["auroc"], res["scale_audit"]["10m"]["ci"]),
        ("Multi-Scale Signal", res["matched_models"]["multiscale_signal"]["auroc"], res["matched_models"]["multiscale_signal"]["ci"]),
        ("Clinical 20m LR", res["temporal_clinical"]["clinical_20m"]["auroc"], res["temporal_clinical"]["clinical_20m"]["ci"]),
        ("Clinical Decel Trend", res["temporal_clinical"]["clinical_decel_trend"]["auroc"], res["temporal_clinical"]["clinical_decel_trend"]["ci"]),
        ("Phase 4 Baseline", auc(fpr_p4, tpr_p4), [0.679, 0.788]),
        ("Phase 5 Candidate", res["matched_models"]["phase5_final_candidate"]["auroc"], res["matched_models"]["phase5_final_candidate"]["ci"])
    ]
    labels = [m[0] for m in models_to_plot]
    scores = [m[1] for m in models_to_plot]
    ci_lo = [m[2][0] for m in models_to_plot]
    ci_hi = [m[2][1] for m in models_to_plot]
    err_lo = [s - l for s, l in zip(scores, ci_lo)]
    err_hi = [h - s for s, h in zip(scores, ci_hi)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    colors = ['#ff7f0e', '#ffbb78', '#9467bd', '#1f77b4', '#aec7e8', '#7f7f7f', '#2ca02c']
    bars = ax.bar(x, scores, yerr=[err_lo, err_hi], color=colors, alpha=0.85, capsize=4, edgecolor='black', width=0.55)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylim(0.5, 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontweight='bold')
    ax.set_ylabel('Patient AUROC (95% CI)', fontsize=11, fontweight='bold')
    ax.set_title('Figure 5: Phase 5 Multi-Resolution Architecture Benchmark', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.015, f"{val:.4f}", ha='center', fontsize=8.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_model_comparison.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 6: Temporal-Scale Contribution Weights (Alphas)
    # -----------------------------------------------------------------------
    alphas = res["matched_models"]["multiscale_signal"]["mean_alphas"]
    scale_names = ["Short (10-min)", "Medium (20-min)", "Long (40-min)"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(scale_names, alphas, color=['#ff7f0e', '#1f77b4', '#2ca02c'], alpha=0.85, edgecolor='black', width=0.5)
    ax.set_ylabel("Learned Logit Weight ($\\alpha_k$)", fontsize=11, fontweight='bold')
    ax.set_ylim(0.0, 0.7)
    ax.set_title("Figure 6: Multi-Scale Signal Contribution Weights", fontsize=12, fontweight='bold')
    for bar, val in zip(bars, alphas):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val*100:.1f}%", ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_scale_weights.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 7: Performance vs Time Before Delivery
    # -----------------------------------------------------------------------
    early_auc = res["delivery_stratification"]["early_window_auroc_gt20min"]
    late_auc = res["delivery_stratification"]["late_window_auroc_le20min"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cats = ["Early Windows (>20m to Delivery)", "Late Windows (<=20m to Delivery)"]
    strat_scores = [early_auc, late_auc]
    bars = ax.bar(cats, strat_scores, color=['#3470a3', '#d95f02'], alpha=0.85, edgecolor='black', width=0.45)
    ax.set_ylim(0.5, 0.85)
    ax.set_ylabel("Patient AUROC", fontsize=11, fontweight='bold')
    ax.set_title("Figure 7: Stratified Performance by Time Before Delivery", fontsize=12, fontweight='bold')
    for bar, val in zip(bars, strat_scores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.015, f"{val:.4f}", ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_delivery_stratification.png"), dpi=300)
    plt.close()

    print("Successfully generated all 7 Phase 5 figures in reports/figures_phase5/")


if __name__ == "__main__":
    generate_phase5_figures()
