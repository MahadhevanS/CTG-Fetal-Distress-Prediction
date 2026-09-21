"""
Phase 3: Attention Dynamics, Entropy, Concentration, and Duration Stratification.

Computes attention concentration (C_max, C_top3), attention entropy H(a),
correlation r(a_i, s_i), and performance stratified by recording duration.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "results/phase3_mil"
OUT_DIR = "results/phase3_attention_analysis"


def run_attention_analysis():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== STARTING PHASE 3 ATTENTION DYNAMICS ANALYSIS ===")

    res_file = os.path.join(RESULTS_DIR, "phase3_results.json")
    if not os.path.exists(res_file):
        print(f"Results file {res_file} not found yet.")
        return

    with open(res_file) as fh:
        data = json.load(fh)

    att_weights = data["attention_weights"]
    labels = np.array(data["patient_scores"]["labels"])
    scores_att = np.array(data["patient_scores"]["attention_mil"])
    scores_max = np.array(data["patient_scores"]["fixed_max"])

    pids = list(att_weights.keys())

    # 1. Attention Concentration and Entropy
    c_max_list = []
    c_top3_list = []
    entropy_list = []
    bag_lens = []

    for p in pids:
        a = np.array(att_weights[p])
        n = len(a)
        bag_lens.append(n)
        c_max = float(np.max(a))
        c_top3 = float(np.sum(np.sort(a)[-min(3, n):]))
        # Entropy H(a) = - sum a * log(a + 1e-12)
        ent = float(-np.sum(a * np.log(a + 1e-12)))
        c_max_list.append(c_max)
        c_top3_list.append(c_top3)
        entropy_list.append(ent)

    c_max_arr = np.array(c_max_list)
    c_top3_arr = np.array(c_top3_list)
    ent_arr = np.array(entropy_list)
    bag_arr = np.array(bag_lens)

    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    # 2. Stratification by Recording Duration (Trunk / Full Hour)
    # Patients with < 17 windows (short/medium, <= 40 min recording) vs 17 windows (full 60 min horizon)
    short_mask = bag_arr < 17
    full_mask = bag_arr >= 17

    if np.sum(short_mask) > 0 and len(np.unique(labels[short_mask])) > 1:
        auc_short_att = float(roc_auc_score(labels[short_mask], scores_att[short_mask]))
        auc_short_max = float(roc_auc_score(labels[short_mask], scores_max[short_mask]))
    else:
        auc_short_att, auc_short_max = float("nan"), float("nan")

    auc_full_att = float(roc_auc_score(labels[full_mask], scores_att[full_mask]))
    auc_full_max = float(roc_auc_score(labels[full_mask], scores_max[full_mask]))

    # 3. Correlation between Attention and Max-Risk
    r_p, _ = pearsonr(scores_att, scores_max)
    r_s, _ = spearmanr(scores_att, scores_max)

    analysis_summary = {
        "attention_metrics": {
            "mean_c_max_overall": float(np.mean(c_max_arr)),
            "mean_c_max_positive": float(np.mean(c_max_arr[pos_mask])),
            "mean_c_max_negative": float(np.mean(c_max_arr[neg_mask])),
            "mean_c_top3_overall": float(np.mean(c_top3_arr)),
            "mean_c_top3_positive": float(np.mean(c_top3_arr[pos_mask])),
            "mean_c_top3_negative": float(np.mean(c_top3_arr[neg_mask])),
            "mean_entropy_overall": float(np.mean(ent_arr)),
            "mean_entropy_positive": float(np.mean(ent_arr[pos_mask])),
            "mean_entropy_negative": float(np.mean(ent_arr[neg_mask]))
        },
        "duration_stratification": {
            "short_records_n": int(np.sum(short_mask)),
            "short_records_att_auroc": auc_short_att,
            "short_records_max_auroc": auc_short_max,
            "full_records_n": int(np.sum(full_mask)),
            "full_records_att_auroc": auc_full_att,
            "full_records_max_auroc": auc_full_max,
        },
        "correlation_with_max_score": {
            "pearson_r": float(r_p),
            "spearman_rho": float(r_s)
        }
    }

    with open(os.path.join(OUT_DIR, "attention_dynamics_summary.json"), "w") as fh:
        json.dump(analysis_summary, fh, indent=2)

    print("\n--- Attention Dynamics Highlights ---")
    print(f"Mean C_max: Positives = {np.mean(c_max_arr[pos_mask]):.4f} | Negatives = {np.mean(c_max_arr[neg_mask]):.4f}")
    print(f"Mean C_top3: Positives = {np.mean(c_top3_arr[pos_mask]):.4f} | Negatives = {np.mean(c_top3_arr[neg_mask]):.4f}")
    print(f"Correlation between Attention Score and Fixed Max: r = {r_p:.4f} (rho = {r_s:.4f})")
    print(f"Duration Stratified AUROC (Short vs Full):")
    print(f"  Short (<60 min): Attention = {auc_short_att:.4f} vs Max = {auc_short_max:.4f}")
    print(f"  Full (60 min):   Attention = {auc_full_att:.4f} vs Max = {auc_full_max:.4f}")

    return analysis_summary


if __name__ == "__main__":
    run_attention_analysis()
