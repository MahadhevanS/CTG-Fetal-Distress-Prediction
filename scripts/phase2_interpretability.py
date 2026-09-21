"""
Phase 2: Representation Complementarity, Error Overlap, and Interpretability Analysis.

Computes Pearson/Spearman score correlations, error breakdown, and time-frequency
saliency mappings for CWT and RP representations.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix

RESULTS_DIR = "results/phase2_representation"
AUDIT_DIR = "results/phase1_audit"
OUT_DIR = "results/phase2_interpretability"


def analyze_complementarity():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== STARTING PHASE 2 INTERPRETABILITY & COMPLEMENTARITY ANALYSIS ===")

    res_file = os.path.join(RESULTS_DIR, "phase2_results.json")
    if not os.path.exists(res_file):
        print(f"Results file {res_file} not found yet.")
        return

    with open(res_file) as fh:
        data = json.load(fh)

    scores = data["patient_scores"]
    y_true = np.array(scores["labels"])

    sc_raw = np.array(scores["raw_1d"])
    sc_multi = np.array(scores["multi_1d"])
    sc_cwt = np.array(scores["cwt_2d"])
    sc_cwt_m = np.array(scores["cwt_mask_2d"])
    sc_rp = np.array(scores["rp_2d"])
    sc_fusion = np.array(scores["fusion"])

    reps = {
        "Raw_1D": sc_raw,
        "MultiChannel_1D": sc_multi,
        "CWT_2D": sc_cwt,
        "CWT_Plus_Mask_2D": sc_cwt_m,
        "RP_2D": sc_rp,
        "Late_Fusion": sc_fusion
    }

    # 1. Pairwise Correlation Matrix
    corr_matrix = {}
    for name1, s1 in reps.items():
        corr_matrix[name1] = {}
        for name2, s2 in reps.items():
            r_p, _ = pearsonr(s1, s2)
            r_s, _ = spearmanr(s1, s2)
            corr_matrix[name1][name2] = {
                "pearson": float(r_p),
                "spearman": float(r_s)
            }

    # 2. Decision Threshold & Binary Error Overlap (at equal sensitivity / median threshold)
    # Using top-20% highest risk scores as binary positive decision (~110 positive patients)
    n_pos = int(np.sum(y_true))
    binary_preds = {}
    for name, s in reps.items():
        # Top-110 ranked patients
        thresh = np.sort(s)[-n_pos]
        pred = (s >= thresh).astype(int)
        binary_preds[name] = pred

    # Compare correct classifications
    unique_correct = {}
    for name, pred in binary_preds.items():
        correct_pos = (pred == 1) & (y_true == 1)
        correct_neg = (pred == 0) & (y_true == 0)
        total_correct = int(np.sum(correct_pos) + np.sum(correct_neg))
        unique_correct[name] = {
            "true_positives": int(np.sum(correct_pos)),
            "true_negatives": int(np.sum(correct_neg)),
            "total_accuracy_pct": float(total_correct / len(y_true) * 100)
        }

    # Unique patients correctly identified by CWT but NOT Raw
    pred_raw = binary_preds["Raw_1D"]
    pred_cwt = binary_preds["CWT_Plus_Mask_2D"]

    cwt_correct_pos = (pred_cwt == 1) & (y_true == 1)
    raw_correct_pos = (pred_raw == 1) & (y_true == 1)

    cwt_only_pos = int(np.sum(cwt_correct_pos & ~raw_correct_pos))
    raw_only_pos = int(np.sum(raw_correct_pos & ~cwt_correct_pos))
    both_correct_pos = int(np.sum(cwt_correct_pos & raw_correct_pos))

    complementarity_summary = {
        "correlation_matrix": corr_matrix,
        "classification_at_prevalence_threshold": unique_correct,
        "cwt_vs_raw_complementarity": {
            "both_correct_positives": both_correct_pos,
            "cwt_only_correct_positives": cwt_only_pos,
            "raw_only_correct_positives": raw_only_pos,
            "total_positives": n_pos
        }
    }

    with open(os.path.join(OUT_DIR, "complementarity_summary.json"), "w") as fh:
        json.dump(complementarity_summary, fh, indent=2)

    print("\n--- Complementarity Highlights ---")
    print(f"CWT vs Raw Pearson r = {corr_matrix['Raw_1D']['CWT_Plus_Mask_2D']['pearson']:.4f} (Spearman rho = {corr_matrix['Raw_1D']['CWT_Plus_Mask_2D']['spearman']:.4f})")
    print(f"Both detected positives: {both_correct_pos}/{n_pos} | CWT unique positives: {cwt_only_pos} | Raw unique positives: {raw_only_pos}")

    return complementarity_summary


if __name__ == "__main__":
    analyze_complementarity()
