"""
Phase 13, Hybrid 1 (docs/phase13_protocol.md Section 14) -- P90-pooled P6 +
Parity. Pre-registered exploratory hybrid (see Section 14's tiering note --
pre-registration fixes the spec, it does not make this confirmatory).

    logit(p_hybrid) = logit(P90_pooled_P6) + lambda * logit(p_parity)

Reuses, unmodified: the frozen window-level P6 scores (13.0A), Option E's
P90-pooling function (q=0.90 fixed), Option D's per-fold parity model and
Phase 4 lambda-selection rule. No new model is fit on raw features.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.aggregation.recency_weighted_p90 import patient_p90_scores
from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected, to_logit, from_logit, select_lambda_trainfold
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
DATA_DIR = "data/processed_clinical"
OUT_DIR = "results/phase13/hybrids"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [0, 10, 20, 30]
CONFIRMATORY_HORIZONS = {0, 30}  # "confirmatory" horizons within this exploratory-tier hybrid


def fit_parity_lr(parity_train, y_train, parity_apply):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(parity_train.reshape(-1, 1))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    return clf.predict_proba(scaler.transform(parity_apply.reshape(-1, 1)))[:, 1]


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids = p6["patient_ids"]
    t_del = p6["t_del"]
    pred_cv_window = p6["pred_unweighted_cv"]
    pred_test_window = p6["pred_unweighted_test"]

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    y_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])

    meta = pd.read_csv(METADATA_PATH)
    if "record_id" in meta.columns:
        meta = meta.set_index("record_id")
    parity_by_pid = {str(p): float(meta.loc[int(p), "parity"]) for p in clean_pids}
    parity_pat = np.array([parity_by_pid[p] for p in clean_pids])

    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = [p for p in clean_pids if p not in test_pids]
    y_test_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in test_pids])

    print("================================================================================")
    print("  PHASE 13 HYBRID 1 -- P90-POOLED P6 + PARITY  (pre-registered, exploratory tier)  ")
    print("================================================================================")

    rows = []
    for h in HORIZONS:
        # ---- base scores: P90-pooled P6 (Option E) and original single-window P6 ----
        p90_cv_all = patient_p90_scores(pred_cv_window, patient_ids, clean_pids, t_del, h, np.inf)
        p6_orig_cv_all = get_patient_scores_at_horizon_corrected(pred_cv_window, patient_ids, clean_pids, t_del, h)
        logit_p90_all = to_logit(p90_cv_all)

        # ---- 5-fold CV: parity model + lambda selected per fold on training patients only ----
        hybrid_cv = np.zeros(len(clean_pids))
        lambdas_cv = []
        for f_idx in range(5):
            te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
            tr_mask = np.array([p not in te_pids for p in clean_pids])
            te_mask = ~tr_mask

            p_parity = fit_parity_lr(parity_pat[tr_mask], y_pat[tr_mask], parity_pat)
            logit_parity = to_logit(p_parity)

            lam, _ = select_lambda_trainfold(logit_p90_all[tr_mask], logit_parity[tr_mask], y_pat[tr_mask])
            hybrid_cv[te_mask] = from_logit(logit_p90_all[te_mask] + lam * logit_parity[te_mask])
            lambdas_cv.append(lam)

        # ---- held-out test: lambda + parity model fit on train+val only ----
        trval_mask = np.array([p in train_val_pids for p in clean_pids])
        test_mask_pat = np.array([p in test_pids for p in clean_pids])

        p90_test_all = patient_p90_scores(pred_test_window, patient_ids, test_pids, t_del, h, np.inf)
        p6_orig_test_all = get_patient_scores_at_horizon_corrected(pred_test_window, patient_ids, test_pids, t_del, h)

        p90_trval = patient_p90_scores(pred_cv_window, patient_ids, train_val_pids, t_del, h, np.inf)
        p_parity_trval = fit_parity_lr(parity_pat[trval_mask], y_pat[trval_mask], parity_pat[trval_mask])
        lam_test, _ = select_lambda_trainfold(to_logit(p90_trval), to_logit(p_parity_trval), y_pat[trval_mask])

        p_parity_test = fit_parity_lr(parity_pat[trval_mask], y_pat[trval_mask], parity_pat[test_mask_pat])
        hybrid_test = from_logit(to_logit(p90_test_all) + lam_test * to_logit(p_parity_test))

        # ---- metrics ----
        auc_hybrid_cv = roc_auc_score(y_pat, hybrid_cv)
        auc_p90_cv = roc_auc_score(y_pat, p90_cv_all)
        auc_p6orig_cv = roc_auc_score(y_pat, p6_orig_cv_all)
        auc_hybrid_test = roc_auc_score(y_test_pat, hybrid_test)
        auc_p90_test = roc_auc_score(y_test_pat, p90_test_all)
        auc_p6orig_test = roc_auc_score(y_test_pat, p6_orig_test_all)

        # (a) attribution: hybrid vs P90-pooled-P6 alone
        boot_a_cv = paired_patient_bootstrap(y_pat, p90_cv_all, hybrid_cv, n_boot=2000, seed=42)
        pdel_a_cv, _, _ = delong_roc_test(y_pat, hybrid_cv, p90_cv_all)
        boot_a_test = paired_patient_bootstrap(y_test_pat, p90_test_all, hybrid_test, n_boot=2000, seed=42)
        pdel_a_test, _, _ = delong_roc_test(y_test_pat, hybrid_test, p90_test_all)

        # (b) total effect: hybrid vs original P6 single-window
        boot_b_cv = paired_patient_bootstrap(y_pat, p6_orig_cv_all, hybrid_cv, n_boot=2000, seed=42)
        pdel_b_cv, _, _ = delong_roc_test(y_pat, hybrid_cv, p6_orig_cv_all)
        boot_b_test = paired_patient_bootstrap(y_test_pat, p6_orig_test_all, hybrid_test, n_boot=2000, seed=42)
        pdel_b_test, _, _ = delong_roc_test(y_test_pat, hybrid_test, p6_orig_test_all)

        tier = "confirmatory-horizon" if h in CONFIRMATORY_HORIZONS else "exploratory-horizon"
        row = {
            "horizon_min": h, "horizon_tier": tier, "lambdas_cv_per_fold": lambdas_cv, "lambda_test": round(lam_test, 2),
            "auroc_p6_original_cv": round(auc_p6orig_cv, 4), "auroc_p90_pooled_cv": round(auc_p90_cv, 4), "auroc_hybrid_cv": round(auc_hybrid_cv, 4),
            "delta_a_hybrid_vs_p90_cv": round(auc_hybrid_cv - auc_p90_cv, 4), "boot_p_a_cv": round(boot_a_cv["p_value"], 4),
            "ci_low_a_cv": round(boot_a_cv["ci_95_low"], 4), "ci_high_a_cv": round(boot_a_cv["ci_95_high"], 4),
            "delta_b_hybrid_vs_p6orig_cv": round(auc_hybrid_cv - auc_p6orig_cv, 4), "boot_p_b_cv": round(boot_b_cv["p_value"], 4),
            "ci_low_b_cv": round(boot_b_cv["ci_95_low"], 4), "ci_high_b_cv": round(boot_b_cv["ci_95_high"], 4),
            "auroc_p6_original_test": round(auc_p6orig_test, 4), "auroc_p90_pooled_test": round(auc_p90_test, 4), "auroc_hybrid_test": round(auc_hybrid_test, 4),
            "delta_a_hybrid_vs_p90_test": round(auc_hybrid_test - auc_p90_test, 4), "boot_p_a_test": round(boot_a_test["p_value"], 4),
            "delta_b_hybrid_vs_p6orig_test": round(auc_hybrid_test - auc_p6orig_test, 4), "boot_p_b_test": round(boot_b_test["p_value"], 4),
        }
        rows.append(row)
        print(f"[{tier:20s}] h={h:>3}m  CV: P6orig={auc_p6orig_cv:.4f} P90={auc_p90_cv:.4f} Hybrid={auc_hybrid_cv:.4f}  "
              f"(a: hybrid-P90 d={auc_hybrid_cv-auc_p90_cv:+.4f} p={boot_a_cv['p_value']:.3f} | b: hybrid-P6orig d={auc_hybrid_cv-auc_p6orig_cv:+.4f} p={boot_b_cv['p_value']:.3f})")
        print(f"{'':24s} TEST: P6orig={auc_p6orig_test:.4f} P90={auc_p90_test:.4f} Hybrid={auc_hybrid_test:.4f}  "
              f"(a: d={auc_hybrid_test-auc_p90_test:+.4f} p={boot_a_test['p_value']:.3f} | b: d={auc_hybrid_test-auc_p6orig_test:+.4f} p={boot_b_test['p_value']:.3f})")

    df_out = pd.DataFrame(rows)
    df_out.drop(columns=["lambdas_cv_per_fold"]).to_csv(os.path.join(OUT_DIR, "hybrid_p90_parity_results.csv"), index=False)
    with open(os.path.join(OUT_DIR, "hybrid_p90_parity_results.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved -> {OUT_DIR}/hybrid_p90_parity_results.csv")


if __name__ == "__main__":
    main()
