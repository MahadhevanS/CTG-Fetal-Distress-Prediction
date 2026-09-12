"""
Phase 13, Option E (docs/phase13_protocol.md Section 13) -- P90 pooling vs.
single-window selection.

Pre-registered confirmatory test: AUROC(P90_pooled) - AUROC(P6_single-window,
corrected convention), at delivery and >=30m, on the SAME frozen window-level
P6 predictions already cached in results/phase13/audit/p6_predictions.npz --
no new model fitting, no hyperparameter search (q=0.90 fixed by convention;
max/mean pooling included as exploratory context only, also parameter-free).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.aggregation.recency_weighted_p90 import patient_p90_scores
from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected, get_eligible_window_mask
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
DATA_DIR = "data/processed_clinical"
OUT_DIR = "results/phase13/p90_pooling"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [0, 10, 20, 30]
CONFIRMATORY_HORIZONS = {0, 30}


def patient_pool_scores(pred_arr, patient_ids, clean_pids, t_del, h_val, op):
    scores = []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        t_pts = t_del[idx]
        elig = get_eligible_window_mask(t_pts, h_val)
        if not np.any(elig):
            elig = np.ones_like(t_pts, dtype=bool)
        p_elig = pred_arr[idx[elig]]
        if op == "max":
            scores.append(float(np.max(p_elig)))
        elif op == "mean":
            scores.append(float(np.mean(p_elig)))
        else:
            raise ValueError(op)
    return np.array(scores)


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids = p6["patient_ids"]
    t_del = p6["t_del"]
    pred_cv = p6["pred_unweighted_cv"]
    pred_test = p6["pred_unweighted_test"]

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    y_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])

    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    y_test_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in test_pids])

    print("================================================================================")
    print("  PHASE 13, OPTION E -- P90 POOLING vs SINGLE-WINDOW SELECTION (pre-registered)   ")
    print("================================================================================")

    rows = []
    for h in HORIZONS:
        p90_cv = patient_p90_scores(pred_cv, patient_ids, clean_pids, t_del, h, np.inf)  # inf = uniform weights = plain P90
        p90_test = patient_p90_scores(pred_test, patient_ids, test_pids, t_del, h, np.inf)
        p6_cv = get_patient_scores_at_horizon_corrected(pred_cv, patient_ids, clean_pids, t_del, h)
        p6_test = get_patient_scores_at_horizon_corrected(pred_test, patient_ids, test_pids, t_del, h)

        max_cv = patient_pool_scores(pred_cv, patient_ids, clean_pids, t_del, h, "max")
        max_test = patient_pool_scores(pred_test, patient_ids, test_pids, t_del, h, "max")
        mean_cv = patient_pool_scores(pred_cv, patient_ids, clean_pids, t_del, h, "mean")
        mean_test = patient_pool_scores(pred_test, patient_ids, test_pids, t_del, h, "mean")

        auc_p6_cv, auc_p6_test = roc_auc_score(y_pat, p6_cv), roc_auc_score(y_test_pat, p6_test)
        auc_p90_cv, auc_p90_test = roc_auc_score(y_pat, p90_cv), roc_auc_score(y_test_pat, p90_test)
        auc_max_cv, auc_max_test = roc_auc_score(y_pat, max_cv), roc_auc_score(y_test_pat, max_test)
        auc_mean_cv, auc_mean_test = roc_auc_score(y_pat, mean_cv), roc_auc_score(y_test_pat, mean_test)

        boot_cv = paired_patient_bootstrap(y_pat, p6_cv, p90_cv, n_boot=2000, seed=42)
        pdel_cv, _, _ = delong_roc_test(y_pat, p90_cv, p6_cv)
        boot_test = paired_patient_bootstrap(y_test_pat, p6_test, p90_test, n_boot=2000, seed=42)
        pdel_test, _, _ = delong_roc_test(y_test_pat, p90_test, p6_test)

        is_confirmatory = h in CONFIRMATORY_HORIZONS
        row = {
            "horizon_min": h, "tier": "CONFIRMATORY" if is_confirmatory else "exploratory",
            "auroc_p6_single_window_cv": round(auc_p6_cv, 4), "auroc_p90_pooled_cv": round(auc_p90_cv, 4),
            "delta_p90_vs_p6_cv": round(auc_p90_cv - auc_p6_cv, 4),
            "boot_p_cv": round(boot_cv["p_value"], 4), "delong_p_cv": round(pdel_cv, 4),
            "ci_low_cv": round(boot_cv["ci_95_low"], 4), "ci_high_cv": round(boot_cv["ci_95_high"], 4),
            "auroc_p6_single_window_test": round(auc_p6_test, 4), "auroc_p90_pooled_test": round(auc_p90_test, 4),
            "delta_p90_vs_p6_test": round(auc_p90_test - auc_p6_test, 4),
            "boot_p_test": round(boot_test["p_value"], 4), "delong_p_test": round(pdel_test, 4),
            "ci_low_test": round(boot_test["ci_95_low"], 4), "ci_high_test": round(boot_test["ci_95_high"], 4),
            "auroc_max_pooled_cv": round(auc_max_cv, 4), "auroc_mean_pooled_cv": round(auc_mean_cv, 4),
            "auroc_max_pooled_test": round(auc_max_test, 4), "auroc_mean_pooled_test": round(auc_mean_test, 4),
        }
        rows.append(row)
        tag = "[CONFIRMATORY]" if is_confirmatory else "[exploratory] "
        print(f"{tag} h={h:>3}m  CV: P6={auc_p6_cv:.4f} P90={auc_p90_cv:.4f} max={auc_max_cv:.4f} mean={auc_mean_cv:.4f}  "
              f"(P90-P6 d={auc_p90_cv-auc_p6_cv:+.4f} p={boot_cv['p_value']:.3f} CI=[{boot_cv['ci_95_low']:+.4f},{boot_cv['ci_95_high']:+.4f}])")
        print(f"{'':14s} TEST: P6={auc_p6_test:.4f} P90={auc_p90_test:.4f} max={auc_max_test:.4f} mean={auc_mean_test:.4f}  "
              f"(P90-P6 d={auc_p90_test-auc_p6_test:+.4f} p={boot_test['p_value']:.3f} CI=[{boot_test['ci_95_low']:+.4f},{boot_test['ci_95_high']:+.4f}])")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, "p90_pooling_results.csv"), index=False)

    # ---- pre-registered survivor classification, confirmatory horizons only ----
    conf = df_out[df_out["tier"] == "CONFIRMATORY"]
    verdict = "CLOSED"
    if (conf["ci_low_cv"] > 0).all():
        verdict = "SURVIVOR (green)"
    elif (conf["delta_p90_vs_p6_cv"] > 0).any() or (conf["delta_p90_vs_p6_test"] > 0).any():
        verdict = "PROMISING (yellow) -- positive point estimate, CI crosses zero"
    print(f"\nPre-registered verdict (confirmatory horizons, CV CI rule): {verdict}")

    with open(os.path.join(OUT_DIR, "p90_pooling_summary.json"), "w") as fh:
        json.dump({"results": rows, "verdict": verdict}, fh, indent=2)
    print(f"Saved -> {OUT_DIR}/p90_pooling_results.csv")
    print(f"Saved -> {OUT_DIR}/p90_pooling_summary.json")


if __name__ == "__main__":
    main()
