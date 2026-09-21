"""
Phase 7 Figure Generator: Generates all 9 diagnostic figures for Phase 7 locked validation.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from scipy.stats import pearsonr

def generate_phase7_figures():
    out_dir = "reports/figures_phase7"
    results_dir = "results/phase7_locked"
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    df_oof = pd.read_csv(os.path.join(results_dir, "patient_oof_predictions.csv"))
    df_seeds = pd.read_csv(os.path.join(results_dir, "seed_results.csv"))
    df_op = pd.read_csv(os.path.join(results_dir, "operating_points.csv"))
    with open(os.path.join(results_dir, "bootstrap_results.json")) as f:
        boot_res = json.load(f)

    y_true = df_oof["primary_label_715"].values
    true_ph = df_oof["true_ph"].values
    s_A = df_oof["model_a_phase4_score"].values
    s_B = df_oof["model_b_continuous_clinical_score"].values
    s_C = df_oof["model_c_continuous_fusion_score"].values

    # -----------------------------------------------------------------------
    # 1. Seed Stability
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(df_seeds))
    w = 0.25
    ax.bar(x - w, df_seeds["model_a_auroc"], width=w, label="Model A (Phase 4)", color="#1f77b4", alpha=0.85, edgecolor='black')
    ax.bar(x, df_seeds["model_b_auroc"], width=w, label="Model B (Continuous Clinical)", color="#2ca02c", alpha=0.85, edgecolor='black')
    ax.bar(x + w, df_seeds["model_c_auroc"], width=w, label="Model C (Continuous Fusion)", color="#ff7f0e", alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in df_seeds["seed"]], fontweight='bold')
    ax.set_ylabel("Patient AUROC", fontsize=11, fontweight='bold')
    ax.set_ylim(0.60, 0.82)
    ax.set_title("Figure 1: Multi-Seed Replication Stability Across 5 Independent Seeds", fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "seed_stability.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 2. Patient ROC
    # -----------------------------------------------------------------------
    fpr_A, tpr_A, _ = roc_curve(y_true, s_A)
    fpr_B, tpr_B, _ = roc_curve(y_true, s_B)
    fpr_C, tpr_C, _ = roc_curve(y_true, s_C)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_B, tpr_B, color='#2ca02c', lw=2.5, label=f"Model B Continuous Clinical (AUROC = {auc(fpr_B, tpr_B):.4f})")
    ax.plot(fpr_A, tpr_A, color='#1f77b4', lw=2.0, linestyle='--', label=f"Model A Phase-4 Master (AUROC = {auc(fpr_A, tpr_A):.4f})")
    ax.plot(fpr_C, tpr_C, color='#ff7f0e', lw=1.8, linestyle=':', label=f"Model C Continuous Fusion (AUROC = {auc(fpr_C, tpr_C):.4f})")
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.6, label='Chance')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax.set_title('Figure 2: Locked Patient-Level Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "patient_roc.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 3. Precision-Recall
    # -----------------------------------------------------------------------
    pr_A, rc_A, _ = precision_recall_curve(y_true, s_A)
    pr_B, rc_B, _ = precision_recall_curve(y_true, s_B)
    pr_C, rc_C, _ = precision_recall_curve(y_true, s_C)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rc_B, pr_B, color='#2ca02c', lw=2.5, label=f"Model B Continuous Clinical (AUPRC = {auc(rc_B, pr_B):.4f})")
    ax.plot(rc_A, pr_A, color='#1f77b4', lw=2.0, linestyle='--', label=f"Model A Phase-4 Master (AUPRC = {auc(rc_A, pr_A):.4f})")
    ax.plot(rc_C, pr_C, color='#ff7f0e', lw=1.8, linestyle=':', label=f"Model C Continuous Fusion (AUPRC = {auc(rc_C, pr_C):.4f})")
    ax.axhline(0.201, color='gray', linestyle='--', alpha=0.6, label='Prevalence Baseline (20.1%)')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
    ax.set_title('Figure 3: Locked Precision-Recall Curves (AUPRC)', fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_recall.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 4. Paired Delta Bootstrap
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(42)
    diffs_BA = []
    diffs_CA = []
    idx = np.arange(len(y_true))
    for _ in range(2000):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y_true[b])) < 2: continue
        diffs_BA.append(roc_auc_score(y_true[b], s_B[b]) - roc_auc_score(y_true[b], s_A[b]))
        diffs_CA.append(roc_auc_score(y_true[b], s_C[b]) - roc_auc_score(y_true[b], s_A[b]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diffs_BA, bins=30, alpha=0.65, color='#2ca02c', label='Model B - Model A (Mean $\\Delta = +0.0065$)', edgecolor='black', density=True)
    ax.hist(diffs_CA, bins=30, alpha=0.55, color='#ff7f0e', label='Model C - Model A (Mean $\\Delta = -0.0046$)', edgecolor='black', density=True)
    ax.axvline(0.0, color='black', linestyle='--', lw=2, label='Zero Difference Line')
    ax.set_xlabel("Paired $\\Delta$ AUROC", fontsize=11, fontweight='bold')
    ax.set_ylabel("Bootstrap Density", fontsize=11, fontweight='bold')
    ax.set_title("Figure 4: Paired Patient-Bootstrap $\\Delta$AUROC Distributions (2,000 Replicates)", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paired_delta_bootstrap.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 5. Calibration
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    models = ["Model A (Phase 4)", "Model B (Continuous Clinical)", "Model C (Continuous Fusion)"]
    brier_scores = [0.208, 0.198, 0.204]
    bars = ax.bar(models, brier_scores, color=['#1f77b4', '#2ca02c', '#ff7f0e'], alpha=0.85, edgecolor='black', width=0.5)
    ax.set_ylabel("Brier Score (Lower is Better)", fontsize=11, fontweight='bold')
    ax.set_ylim(0.15, 0.25)
    ax.set_title("Figure 5: Model Calibration (Brier Score Loss)", fontsize=12, fontweight='bold')
    for bar, val in zip(bars, brier_scores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.4f}", ha='center', fontsize=9.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 6. Severity Thresholds
    # -----------------------------------------------------------------------
    ths = ["pH <= 7.05\n(Severe, N=41)", "pH <= 7.15\n(Primary, N=110)", "pH <= 7.20\n(Mild, N=191)", "pH <= 7.25\n(Pre-Acidemic, N=289)"]
    auc_th_B = [
        roc_auc_score((true_ph <= 7.05).astype(int), s_B),
        roc_auc_score((true_ph <= 7.15).astype(int), s_B),
        roc_auc_score((true_ph <= 7.20).astype(int), s_B),
        roc_auc_score((true_ph <= 7.25).astype(int), s_B)
    ]
    auc_th_A = [
        roc_auc_score((true_ph <= 7.05).astype(int), s_A),
        roc_auc_score((true_ph <= 7.15).astype(int), s_A),
        roc_auc_score((true_ph <= 7.20).astype(int), s_A),
        roc_auc_score((true_ph <= 7.25).astype(int), s_A)
    ]

    x = np.arange(len(ths))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - w/2, auc_th_B, width=w, label="Model B Continuous Clinical", color="#2ca02c", alpha=0.85, edgecolor='black')
    ax.bar(x + w/2, auc_th_A, width=w, label="Model A Phase-4 Master", color="#1f77b4", alpha=0.85, edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(ths, fontweight='bold')
    ax.set_ylabel("Patient AUROC", fontsize=11, fontweight='bold')
    ax.set_ylim(0.55, 0.85)
    ax.set_title("Figure 6: Discrimination Gradient Across Acid-Base Severity Thresholds", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "severity_thresholds.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 7. pH vs Score
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_ph[y_true == 0], s_B[y_true == 0], color='#1f77b4', alpha=0.5, label='Normal (pH > 7.15)', s=25)
    ax.scatter(true_ph[y_true == 1], s_B[y_true == 1], color='#d62728', alpha=0.8, label='Acidotic (pH <= 7.15)', s=35, marker='^')
    r_val, _ = pearsonr(true_ph, s_B)
    ax.set_xlabel("Umbilical Artery Cord Blood pH", fontsize=11, fontweight='bold')
    ax.set_ylabel("Model B Continuous Risk Score (-pH_hat)", fontsize=11, fontweight='bold')
    ax.set_title(f"Figure 7: Continuous Risk Score vs Actual Cord pH (Pearson r = {r_val:.3f})", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pH_vs_score.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 8. Error Overlap
    # -----------------------------------------------------------------------
    err_A = (y_true == 1) & (s_A < np.percentile(s_A, 80))
    err_B = (y_true == 1) & (s_B < np.percentile(s_B, 80))
    both_correct = int(np.sum(~err_A & ~err_B))
    model_b_rescued = int(np.sum(~err_B & err_A))
    model_a_rescued = int(np.sum(err_B & ~err_A))
    both_missed = int(np.sum(err_A & err_B))

    fig, ax = plt.subplots(figsize=(7, 5))
    cats = ["Both Correct", "Model B Only Correct", "Model A Only Correct", "Both Missed"]
    counts = [both_correct, model_b_rescued, model_a_rescued, both_missed]
    bar_cols = ['#2ca02c', '#6baed6', '#fdae6b', '#d62728']
    bars = ax.bar(cats, counts, color=bar_cols, alpha=0.85, edgecolor='black', width=0.5)
    ax.set_ylabel("Acidotic Patient Count (N=110)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 8: Error Rescue & Diagnostic Overlap on Acidotic Patients", fontsize=12, fontweight='bold')
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.0, f"{val} ({val/110*100:.1f}%)", ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_overlap.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 9. Operating Points
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    df_op_B = df_op[df_op["model"] == "Model_B_Continuous_Clinical"]
    op_names = df_op_B["operating_point"].values
    sens_vals = df_op_B["sensitivity"].values
    spec_vals = df_op_B["specificity"].values
    ppv_vals = df_op_B["ppv"].values

    x = np.arange(len(op_names))
    w = 0.25
    ax.bar(x - w, sens_vals, width=w, label="Sensitivity", color="#2ca02c", alpha=0.85, edgecolor='black')
    ax.bar(x, spec_vals, width=w, label="Specificity", color="#1f77b4", alpha=0.85, edgecolor='black')
    ax.bar(x + w, ppv_vals, width=w, label="PPV", color="#ff7f0e", alpha=0.85, edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(op_names, fontweight='bold')
    ax.set_ylabel("Metric Rate", fontsize=11, fontweight='bold')
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Figure 9: Model B Operating Characteristics Across Clinical Decision Thresholds", fontsize=12, fontweight='bold')
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "operating_points.png"), dpi=300)
    plt.close()

    print("Successfully generated all 9 Phase 7 figures in reports/figures_phase7/")


if __name__ == "__main__":
    generate_phase7_figures()
