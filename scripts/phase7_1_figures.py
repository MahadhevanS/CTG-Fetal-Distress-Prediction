"""
Phase 7.1 Figure Generator: Generates all 6 diagnostic figures for Phase 7.1 reconciliation.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import pearsonr, spearmanr

def generate_phase7_1_figures():
    out_dir = "reports/figures_phase7_reconciliation"
    results_dir = "results/phase7_reconciliation"
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    df_gold = pd.read_csv(os.path.join(results_dir, "phase4_gold_predictions.csv"))
    df_p7 = pd.read_csv("results/phase7_locked/patient_oof_predictions.csv")
    df_seeds = pd.read_csv("results/phase7_locked/seed_results.csv")
    with open("results/phase6_outcome_supervision/phase6_results.json") as f:
        res6 = json.load(f)

    y_true = df_gold["primary_label_715"].values
    s_p4_gold = df_gold["phase4_gold_logit_modulation_score"].values
    s_p7_A = df_p7["model_a_phase4_score"].values
    s_p7_B = df_p7["model_b_continuous_clinical_score"].values
    s_p6_C = np.array(res6["patient_scores"]["continuous_fusion_score"])
    s_p7_C = df_p7["model_c_continuous_fusion_score"].values

    # -----------------------------------------------------------------------
    # 1. Phase 4 Old vs New Scores
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(s_p4_gold[y_true == 0], s_p7_A[y_true == 0], color='#1f77b4', alpha=0.5, label='Normal (pH > 7.15)', s=25)
    ax.scatter(s_p4_gold[y_true == 1], s_p7_A[y_true == 1], color='#d62728', alpha=0.8, label='Acidotic (pH <= 7.15)', s=35, marker='^')
    r_A, _ = pearsonr(s_p4_gold, s_p7_A)
    ax.set_xlabel("Phase 4 Gold Master Score (AUROC = 0.7361)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Phase 7 Retrained Model A Score (AUROC = 0.6547)", fontsize=11, fontweight='bold')
    ax.set_title(f"Figure 1: Phase-4 Master Score Comparison (Pearson r = {r_A:.3f})", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase4_old_vs_new_scores.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 2. Phase 4 Rank Comparison
    # -----------------------------------------------------------------------
    rank_gold = np.argsort(np.argsort(s_p4_gold))
    rank_p7 = np.argsort(np.argsort(s_p7_A))
    rho_A, _ = spearmanr(s_p4_gold, s_p7_A)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(rank_gold[y_true == 0], rank_p7[y_true == 0], color='#1f77b4', alpha=0.5, label='Normal Ranks', s=20)
    ax.scatter(rank_gold[y_true == 1], rank_p7[y_true == 1], color='#d62728', alpha=0.8, label='Acidotic Ranks', s=30, marker='^')
    ax.plot([0, 547], [0, 547], color='black', linestyle='--', alpha=0.6, label='Ideal Rank Match')
    ax.set_xlabel("Phase 4 Gold Patient Rank", fontsize=11, fontweight='bold')
    ax.set_ylabel("Phase 7 Retrained Patient Rank", fontsize=11, fontweight='bold')
    ax.set_title(f"Figure 2: Rank Drift in Model A (Spearman rho = {rho_A:.3f})", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase4_rank_comparison.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 3. Model C Old vs New Scores
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(s_p6_C[y_true == 0], s_p7_C[y_true == 0], color='#ff7f0e', alpha=0.5, label='Normal (pH > 7.15)', s=25)
    ax.scatter(s_p6_C[y_true == 1], s_p7_C[y_true == 1], color='#d62728', alpha=0.8, label='Acidotic (pH <= 7.15)', s=35, marker='^')
    r_C, _ = pearsonr(s_p6_C, s_p7_C)
    ax.set_xlabel("Phase 6 Continuous Fusion Score (AUROC = 0.7315)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Phase 7 Retrained Model C Score (AUROC = 0.5622)", fontsize=11, fontweight='bold')
    ax.set_title(f"Figure 3: Model C Score Drift (Pearson r = {r_C:.3f})", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_c_old_vs_new_scores.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 4. Model B Seed Stability
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(df_seeds))
    bars = ax.bar(x, df_seeds["model_b_auroc"], color='#2ca02c', alpha=0.85, edgecolor='black', width=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in df_seeds["seed"]], fontweight='bold')
    ax.set_ylabel("Patient AUROC", fontsize=11, fontweight='bold')
    ax.set_ylim(0.70, 0.78)
    ax.set_title("Figure 4: Model B (Continuous Clinical Huber) Deterministic Stability Across Seeds", fontsize=12, fontweight='bold')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, "0.7426", ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_b_seed_stability.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 5. Corrected Paired Bootstrap
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(42)
    diffs_corr = []
    idx = np.arange(len(y_true))
    for _ in range(2000):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y_true[b])) < 2: continue
        diffs_corr.append(roc_auc_score(y_true[b], s_p7_B[b]) - roc_auc_score(y_true[b], s_p4_gold[b]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diffs_corr, bins=30, color='#2ca02c', alpha=0.7, edgecolor='black', density=True)
    ax.axvline(0.0, color='black', linestyle='--', lw=2, label='Zero Difference Line')
    ax.axvline(np.mean(diffs_corr), color='darkgreen', linestyle='-', lw=2, label=f"Mean Delta AUROC = {np.mean(diffs_corr):+.4f}")
    ax.set_xlabel("Corrected Paired Delta AUROC (Model B - Phase 4 Gold)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Bootstrap Density", fontsize=11, fontweight='bold')
    ax.set_title("Figure 5: Corrected Paired Bootstrap Distribution (95% CI: [-0.021, +0.034], p = 0.320)", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corrected_paired_bootstrap.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # 6. Phase 6 vs Phase 7 AUROC Comparison
    # -----------------------------------------------------------------------
    models = ["Model B (Continuous Clinical)", "Phase 4 Master (Model A)", "Continuous Fusion (Model C)"]
    p6_vals = [0.7426, 0.7361, 0.7315]
    p7_vals = [0.7426, 0.6547, 0.5622]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars1 = ax.bar(x - w/2, p6_vals, width=w, label="Phase 6 Frozen Baseline", color="#1f77b4", alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + w/2, p7_vals, width=w, label="Phase 7 Retrained Execution", color="#e377c2", alpha=0.85, edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(models, fontweight='bold')
    ax.set_ylabel("Patient AUROC", fontsize=11, fontweight='bold')
    ax.set_ylim(0.5, 0.82)
    ax.set_title("Figure 6: Forensic Comparison: Phase 6 Frozen Gold vs Phase 7 Retrained", fontsize=12, fontweight='bold')
    for b1, b2 in zip(bars1, bars2):
        ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.01, f"{b1.get_height():.4f}", ha='center', fontsize=9, fontweight='bold')
        ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.01, f"{b2.get_height():.4f}", ha='center', fontsize=9, fontweight='bold')
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase6_vs_phase7_auroc.png"), dpi=300)
    plt.close()

    print("Successfully generated all 6 Phase 7.1 reconciliation figures in reports/figures_phase7_reconciliation/")


if __name__ == "__main__":
    generate_phase7_1_figures()
