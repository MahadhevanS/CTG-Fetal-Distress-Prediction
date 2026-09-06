"""
Phase 6 Figure Generator: Generates all 7 diagnostic figures for Phase 6.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr

def generate_phase6_figures():
    results_path = "results/phase6_outcome_supervision/phase6_results.json"
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        res = json.load(f)

    out_dir = "reports/figures_phase6"
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    y_true = np.array(res["patient_scores"]["labels"])
    true_ph = np.array(res["patient_scores"]["true_ph"])
    pred_ph = np.array(res["patient_scores"]["continuous_signal_ph"])
    p4_score = np.array(res["patient_scores"]["phase4_score"])
    cont_score = np.array(res["patient_scores"]["continuous_fusion_score"])
    ord_score = np.array(res["patient_scores"]["ordinal_fusion_score"])
    soft_score = np.array(res["patient_scores"]["soft_target_fusion_score"])

    # -----------------------------------------------------------------------
    # Fig 1: Cohort pH Distribution & Severity Bins
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    n, bins, patches = ax.hist(true_ph, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax.axvline(7.05, color='#d62728', linestyle='--', lw=2, label='Severe Acidemia (pH <= 7.05, N=41)')
    ax.axvline(7.15, color='#ff7f0e', linestyle='--', lw=2, label='Primary Acidemia Threshold (pH <= 7.15, N=110)')
    ax.axvline(7.25, color='#2ca02c', linestyle='--', lw=2, label='Normal Baseline Boundary (pH > 7.25, N=258)')
    ax.set_xlabel("Umbilical Artery Cord pH", fontsize=11, fontweight='bold')
    ax.set_ylabel("Patient Count", fontsize=11, fontweight='bold')
    ax.set_title("Figure 1: CTU-UHB Clean Cohort Umbilical Artery pH Distribution (N=547)", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_ph_distribution.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 2: Predicted vs Observed pH Scatter
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_ph[y_true == 0], pred_ph[y_true == 0], color='#1f77b4', alpha=0.5, label='Normal (pH > 7.15)', s=25)
    ax.scatter(true_ph[y_true == 1], pred_ph[y_true == 1], color='#d62728', alpha=0.8, label='Acidotic (pH <= 7.15)', s=35, marker='^')
    r_val = res["models"]["continuous_signal"]["pearson_r"]
    mae_val = res["models"]["continuous_signal"]["mae"]
    ax.plot([6.85, 7.45], [6.85, 7.45], color='black', linestyle='--', alpha=0.6, label='Ideal 1:1 Line')
    ax.set_xlabel("True Cord Blood pH", fontsize=11, fontweight='bold')
    ax.set_ylabel("Predicted 1D ResNet pH (P10 window)", fontsize=11, fontweight='bold')
    ax.set_title(f"Figure 2: Continuous pH Regression (Pearson r = {r_val:.3f}, MAE = {mae_val:.3f})", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_predicted_vs_observed_ph.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 3: pH Residual Distribution
    # -----------------------------------------------------------------------
    residuals = true_ph - pred_ph
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=25, color='#9467bd', alpha=0.75, edgecolor='black', density=True)
    ax.axvline(0.0, color='black', linestyle='--', lw=1.5)
    ax.set_xlabel("pH Residual (True pH - Predicted pH)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Density", fontsize=11, fontweight='bold')
    ax.set_title(f"Figure 3: Continuous pH Prediction Error Distribution (RMSE = {res['models']['continuous_signal']['rmse']:.3f})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_residual_distribution.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 4: AUROC Across Predefined pH Thresholds
    # -----------------------------------------------------------------------
    th_keys = ["le_7.05", "le_7.15", "le_7.20", "le_7.25"]
    th_labels = ["pH <= 7.05\n(Severe)", "pH <= 7.15\n(Primary)", "pH <= 7.20\n(Mild)", "pH <= 7.25\n(Pre-Acidemic)"]
    p4_th = [res["secondary_thresholds"][k]["phase4_auroc"] for k in th_keys]
    cont_th = [res["secondary_thresholds"][k]["continuous_fusion_auroc"] for k in th_keys]
    ord_th = [res["secondary_thresholds"][k]["ordinal_fusion_auroc"] for k in th_keys]

    x = np.arange(len(th_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - width, p4_th, width, label='Phase 4 Master (Binary)', color='#1f77b4', alpha=0.85, edgecolor='black')
    ax.bar(x, ord_th, width, label='Ordinal Fusion', color='#2ca02c', alpha=0.85, edgecolor='black')
    ax.bar(x + width, cont_th, width, label='Continuous Fusion', color='#ff7f0e', alpha=0.85, edgecolor='black')
    ax.set_ylabel("Patient-Level AUROC", fontsize=11, fontweight='bold')
    ax.set_title("Figure 4: Discriminative Performance Across Acid-Base Severity Thresholds", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(th_labels, fontweight='bold')
    ax.set_ylim(0.5, 0.85)
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_threshold_auroc.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 5: Patient-Level ROC Curves
    # -----------------------------------------------------------------------
    fpr_p4, tpr_p4, _ = roc_curve(y_true, p4_score)
    fpr_ord, tpr_ord, _ = roc_curve(y_true, ord_score)
    fpr_cont, tpr_cont, _ = roc_curve(y_true, cont_score)
    fpr_soft, tpr_soft, _ = roc_curve(y_true, soft_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_p4, tpr_p4, color='#1f77b4', lw=2.5, label=f"Phase 4 Master (AUROC = {auc(fpr_p4, tpr_p4):.4f})")
    ax.plot(fpr_ord, tpr_ord, color='#2ca02c', lw=2.0, linestyle='--', label=f"Ordinal Fusion (AUROC = {res['models']['ordinal_fusion']['auroc']:.4f})")
    ax.plot(fpr_soft, tpr_soft, color='#9467bd', lw=2.0, linestyle='-.', label=f"Soft Target Fusion (AUROC = {res['models']['soft_target_fusion']['auroc']:.4f})")
    ax.plot(fpr_cont, tpr_cont, color='#ff7f0e', lw=1.8, linestyle=':', label=f"Continuous Fusion (AUROC = {res['models']['continuous_fusion']['auroc']:.4f})")
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.6, label='Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax.set_title('Figure 5: Patient-Level ROC Curves (pH <= 7.15 Primary Endpoint)', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_roc_comparison.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 6: Phase 6 vs Phase 4 Patient-Level Score Concordance
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(p4_score[y_true == 0], ord_score[y_true == 0], color='#1f77b4', alpha=0.5, label='Normal (pH > 7.15)', s=25)
    ax.scatter(p4_score[y_true == 1], ord_score[y_true == 1], color='#d62728', alpha=0.8, label='Acidotic (pH <= 7.15)', s=35, marker='^')
    r_sc, _ = pearsonr(p4_score, ord_score)
    ax.set_xlabel("Phase 4 Master Predicted Risk", fontsize=11, fontweight='bold')
    ax.set_ylabel("Ordinal Fusion Predicted Risk", fontsize=11, fontweight='bold')
    ax.set_title(f"Figure 6: Phase 6 vs Phase 4 Score Concordance (Pearson r = {r_sc:.3f})", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_score_concordance.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 7: Performance Across pH Severity Bands
    # -----------------------------------------------------------------------
    bands = ["Severe (<=7.05)", "Moderate (7.05-7.15)", "Borderline (7.15-7.25)", "Normal (>7.25)"]
    band_masks = [
        (true_ph <= 7.05),
        ((true_ph > 7.05) & (true_ph <= 7.15)),
        ((true_ph > 7.15) & (true_ph <= 7.25)),
        (true_ph > 7.25)
    ]
    mean_risks_p4 = [float(p4_score[m].mean()) for m in band_masks]
    mean_risks_ord = [float(ord_score[m].mean()) for m in band_masks]

    x = np.arange(len(bands))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars1 = ax.bar(x - width/2, mean_risks_p4, width, label='Phase 4 Binary Risk', color='#1f77b4', alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + width/2, mean_risks_ord, width, label='Ordinal Fusion Risk', color='#2ca02c', alpha=0.85, edgecolor='black')
    ax.set_ylabel("Mean Predicted Acidemia Risk", fontsize=11, fontweight='bold')
    ax.set_title("Figure 7: Mean Risk Score Gradient Across Acid-Base Severity Tiers", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontweight='bold')
    ax.set_ylim(0.0, 0.7)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{bar.get_height():.3f}", ha='center', fontsize=9, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_severity_band_risks.png"), dpi=300)
    plt.close()

    print("Successfully generated all 7 Phase 6 figures in reports/figures_phase6/")


if __name__ == "__main__":
    generate_phase6_figures()
