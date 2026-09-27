"""
Independent reconciliation audit for the Model 3 + Parity Hybrid walkthrough
(results/model3_parity_hybrid/final_summary.md), requested after the
walkthrough's early-warning numbers for Model 3 (>=10m: 0.6637, >=20m:
0.5882, >=30m: 0.5811) did not match the previously locked canonical values
(0.6683, 0.6015, 0.5976).

ROOT CAUSE, established by direct inspection of scripts/model3_parity_hybrid/
hybrid_engine.py (lines ~373-401 and ~449-450), not by re-deriving from
scratch: for h>0, that script computes an `elig_mask` that DROPS any patient
who has no window with time_before_delivery_min >= h, before computing
AUROC/AUPRC/bootstrap. This is NOT the canonical convention used everywhere
else in this project. The canonical convention
(src.evaluation.phase13_common.get_patient_scores_at_horizon_corrected, and
src.models.phase16_causal_attention.eligible_prefix_length for Model 3)
NEVER drops a patient -- a patient with no eligible window at horizon h
falls back to their last available window's score and is KEPT in the
evaluation set, at every horizon, for every model (this fallback is exactly
what scripts/model3_direct_comparisons.py and
scripts/phase16_temporal_attention_model.py already do, and what reproduces
the locked canonical numbers). This is why N shrinks (547 -> 545 -> 534 ->
492) in the hybrid walkthrough's Table 2, and it affects EVERY model in
that table identically (P6, Max, P90, Model 3, Parity Fusion, Hybrid) --
not only Model 3.

This script does NOT retrain anything and does NOT change the hybrid
model in any way. It regenerates the identical per-patient score arrays
using hybrid_engine.py's own unmodified scoring procedure (same
checkpoints, same folds, same fusion fit, same seeds -- verbatim,
line-for-line reused) and then evaluates them TWICE at every horizon:
  (a) under the BUGGY convention (patient-dropping), to reproduce the
      walkthrough's own reported numbers and confirm the bug explains
      100% of the discrepancy -- not partially, not by coincidence.
  (b) under the CANONICAL convention (fallback, all patients kept), which
      is the one comparable to every other locked number in this project.

Per the audit brief: "Do not modify the model to improve its scores.
Perform an independent reconciliation audit." Nothing here re-fits, re-tunes,
or re-selects anything -- it is a pure re-evaluation of already-fixed,
already-fitted predictions under two different, precisely-specified patient
inclusion rules, reported side by side.
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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import (
    get_patient_scores_at_horizon_corrected, to_logit, from_logit, select_lambda_trainfold,
    get_eligible_window_mask,
)
from src.models.phase16_causal_attention import predict_at_horizon_for_patients
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = os.path.join(BASE_DIR, "data/processed_clinical/folds.json")
ROLLING_PATH = os.path.join(BASE_DIR, "results/phase8_rolling/rolling_predictions.csv")
P6_PRED_PATH = os.path.join(BASE_DIR, "results/phase13/audit/p6_predictions.npz")
METADATA_PATH = os.path.join(BASE_DIR, "data/raw/ctu-chb-intrapartum/clinical_metadata.csv")
DATA_DIR = os.path.join(BASE_DIR, "data/processed_clinical")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "results/phase16/checkpoints")
OUT_DIR = os.path.join(BASE_DIR, "results/model3_parity_hybrid")

HORIZONS = [0, 10, 20, 30]

# The walkthrough's own reported (buggy-convention) Model 3 numbers, for direct comparison.
WALKTHROUGH_MODEL3 = {0: 0.7216, 10: 0.6637, 20: 0.5882, 30: 0.5811}
# The previously locked canonical Model 3 numbers (results/phase16/Model_3_magnitude_position_auroc_results.csv).
LOCKED_MODEL3 = {0: 0.7216, 10: 0.6683, 20: 0.6015, 30: 0.5976}


def pool_windows(pred_arr, patient_ids_arr, pids_list, t_del, h_val, op):
    scores = []
    for pid in pids_list:
        idx = np.where(patient_ids_arr == str(pid))[0]
        t_pts = t_del[idx]
        elig = get_eligible_window_mask(t_pts, h_val)
        if not np.any(elig):
            elig = np.ones_like(t_pts, dtype=bool)
        p_elig = pred_arr[idx[elig]]
        if op == "max":
            scores.append(float(np.max(p_elig)))
        elif op == "p90":
            scores.append(float(np.percentile(p_elig, 90)))
        else:
            raise ValueError(op)
    return np.array(scores)


def fit_parity_model(parity_train, y_train, parity_apply):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(parity_train.reshape(-1, 1))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    X_ap = scaler.transform(parity_apply.reshape(-1, 1))
    return clf.predict_proba(X_ap)[:, 1]


def buggy_elig_mask(df_rolling, pids_list, h):
    """Reproduces hybrid_engine.py's patient-dropping filter EXACTLY, for
    comparison purposes only -- this is NOT the canonical convention."""
    if h == 0:
        return np.ones(len(pids_list), dtype=bool)
    return np.array([np.any(df_rolling[df_rolling["patient_id"] == p]["time_before_delivery_min"] >= h) for p in pids_list])


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    pids_arr = np.array(clean_pids)
    n_patients = len(clean_pids)

    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    y_lookup = {p: int(df_rolling[df_rolling["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    y_pat = np.array([y_lookup[p] for p in clean_pids])

    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = sorted(p for p in clean_pids if p not in test_pids)
    y_test_pat = np.array([y_lookup[p] for p in test_pids])
    trval_mask_pat = np.array([p in train_val_pids for p in clean_pids])
    test_mask_pat = np.array([p in test_pids for p in clean_pids])

    meta = pd.read_csv(METADATA_PATH).set_index("record_id")
    parity_by_pid = {str(p): float(meta.loc[int(p), "parity"]) for p in clean_pids}
    parity_pat = np.array([parity_by_pid[p] for p in clean_pids])

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids_arr = p6["patient_ids"]
    t_del = p6["t_del"]
    pred_cv_window = p6["pred_unweighted_cv"]
    pred_test_window = p6["pred_unweighted_test"]

    patient_data_cv, t_del_cv = build_patient_data(pred_cv_window, patient_ids_arr, df_rolling, clean_pids, y_lookup)
    patient_data_test, t_del_test = build_patient_data(pred_test_window, patient_ids_arr, df_rolling, test_pids, y_lookup)

    completed = load_completed_units(CHECKPOINT_DIR)
    fold_scorers = {f: load_scorer_checkpoint(completed[f"Model_3_magnitude_position_fold{f}"], in_dim=2, hidden=8) for f in range(5)}
    test_scorer = load_scorer_checkpoint(completed["Model_3_magnitude_position_testmodel"], in_dim=2, hidden=8)

    print("================================================================================")
    print("  CANONICAL HORIZON RECONCILIATION AUDIT -- Model 3 + Parity Hybrid                ")
    print("  Re-evaluating hybrid_engine.py's OWN (unmodified) scores under both conventions  ")
    print("================================================================================")

    # ---------------- regenerate identical per-patient scores (verbatim procedure) ----------------
    m3_cv = {h: np.zeros(n_patients) for h in HORIZONS}
    m3_test = {h: np.zeros(len(test_pids)) for h in HORIZONS}
    for f_idx in range(5):
        te_pids = [p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx]
        te_mask = np.isin(pids_arr, te_pids)
        scorer = fold_scorers[f_idx]
        for h in HORIZONS:
            out = predict_at_horizon_for_patients(scorer, te_pids, patient_data_cv, t_del_cv, True, h)
            m3_cv[h][te_mask] = [out[p][0] for p in pids_arr[te_mask]]
    for h in HORIZONS:
        out_t = predict_at_horizon_for_patients(test_scorer, test_pids, patient_data_test, t_del_test, True, h)
        m3_test[h] = np.array([out_t[p][0] for p in test_pids])

    p6_cv, p6_test, max_cv, max_test, p90_cv, p90_test = {}, {}, {}, {}, {}, {}
    for h in HORIZONS:
        p6_cv[h] = get_patient_scores_at_horizon_corrected(pred_cv_window, patient_ids_arr, clean_pids, t_del, h)
        p6_test[h] = get_patient_scores_at_horizon_corrected(pred_test_window, patient_ids_arr, test_pids, t_del, h)
        max_cv[h] = pool_windows(pred_cv_window, patient_ids_arr, clean_pids, t_del, h, "max")
        max_test[h] = pool_windows(pred_test_window, patient_ids_arr, test_pids, t_del, h, "max")
        p90_cv[h] = pool_windows(pred_cv_window, patient_ids_arr, clean_pids, t_del, h, "p90")
        p90_test[h] = pool_windows(pred_test_window, patient_ids_arr, test_pids, t_del, h, "p90")

    parity_fusion_cv = {h: np.zeros(n_patients) for h in HORIZONS}
    parity_fusion_test = {h: np.zeros(len(test_pids)) for h in HORIZONS}
    for f_idx in range(5):
        te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
        tr_mask = np.array([p not in te_pids for p in clean_pids])
        te_mask = ~tr_mask
        p_par_cv = fit_parity_model(parity_pat[tr_mask], y_pat[tr_mask], parity_pat)
        logit_par_cv = to_logit(p_par_cv)
        for h in HORIZONS:
            logit_p6_h = to_logit(p6_cv[h])
            lam, _ = select_lambda_trainfold(logit_p6_h[tr_mask], logit_par_cv[tr_mask], y_pat[tr_mask])
            parity_fusion_cv[h][te_mask] = from_logit(logit_p6_h[te_mask] + lam * logit_par_cv[te_mask])
    p_par_test_all = fit_parity_model(parity_pat[trval_mask_pat], y_pat[trval_mask_pat], parity_pat)
    logit_par_test_all = to_logit(p_par_test_all)
    for h in HORIZONS:
        logit_p6_te = to_logit(p6_test[h])
        logit_p6_trval = to_logit(p6_cv[h][trval_mask_pat])
        lam_te, _ = select_lambda_trainfold(logit_p6_trval, logit_par_test_all[trval_mask_pat], y_pat[trval_mask_pat])
        parity_fusion_test[h] = from_logit(logit_p6_te + lam_te * logit_par_test_all[test_mask_pat])

    hybrid_cv = {h: np.zeros(n_patients) for h in HORIZONS}
    hybrid_test = {h: np.zeros(len(test_pids)) for h in HORIZONS}
    for f_idx in range(5):
        te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
        tr_mask = np.array([p not in te_pids for p in clean_pids])
        te_mask = ~tr_mask
        p_par_all = fit_parity_model(parity_pat[tr_mask], y_pat[tr_mask], parity_pat)
        z_par_all = to_logit(p_par_all)
        for h in HORIZONS:
            z_m3_all = to_logit(m3_cv[h])
            X_tr = np.column_stack([z_m3_all[tr_mask], z_par_all[tr_mask]])
            X_te = np.column_stack([z_m3_all[te_mask], z_par_all[te_mask]])
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            clf.fit(X_tr_s, y_pat[tr_mask])
            hybrid_cv[h][te_mask] = clf.predict_proba(X_te_s)[:, 1]
    p_par_test_cohort = fit_parity_model(parity_pat[trval_mask_pat], y_pat[trval_mask_pat], parity_pat)
    z_par_test_cohort = to_logit(p_par_test_cohort)
    for h in HORIZONS:
        z_m3_trval = to_logit(m3_cv[h][trval_mask_pat])
        z_m3_test = to_logit(m3_test[h])
        X_trval = np.column_stack([z_m3_trval, z_par_test_cohort[trval_mask_pat]])
        X_test = np.column_stack([z_m3_test, z_par_test_cohort[test_mask_pat]])
        scaler_t = StandardScaler()
        X_trval_s = scaler_t.fit_transform(X_trval)
        X_test_s = scaler_t.transform(X_test)
        clf_t = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        clf_t.fit(X_trval_s, y_pat[trval_mask_pat])
        hybrid_test[h] = clf_t.predict_proba(X_test_s)[:, 1]

    models_cv = {"P6": p6_cv, "Max": max_cv, "P90": p90_cv, "Model 3": m3_cv, "Parity Fusion": parity_fusion_cv, "Hybrid": hybrid_cv}
    models_test = {"P6": p6_test, "Max": max_test, "P90": p90_test, "Model 3": m3_test, "Parity Fusion": parity_fusion_test, "Hybrid": hybrid_test}

    # ---------------- Step 1: confirm the bug reproduces the walkthrough's exact numbers ----------------
    print("\n--- Step 1: reproducing the walkthrough's OWN (buggy-convention) Model 3 numbers ---")
    confirm_rows = []
    for h in HORIZONS:
        elig = buggy_elig_mask(df_rolling, clean_pids, h)
        auc_buggy = roc_auc_score(y_pat[elig], m3_cv[h][elig])
        confirm_rows.append({
            "horizon": h, "n_patients_buggy_convention": int(np.sum(elig)),
            "model3_auroc_buggy_convention_reproduced": round(auc_buggy, 4),
            "model3_auroc_walkthrough_reported": WALKTHROUGH_MODEL3[h],
            "matches_walkthrough": abs(auc_buggy - WALKTHROUGH_MODEL3[h]) < 1e-3,
            "model3_auroc_locked_canonical": LOCKED_MODEL3[h],
        })
        print(f"  h={h:>3}m  N={int(np.sum(elig))}  reproduced(buggy)={auc_buggy:.4f}  "
              f"walkthrough_reported={WALKTHROUGH_MODEL3[h]:.4f}  locked_canonical={LOCKED_MODEL3[h]:.4f}")
    df_confirm = pd.DataFrame(confirm_rows)
    df_confirm.to_csv(os.path.join(OUT_DIR, "bug_reproduction_confirmation.csv"), index=False)
    bug_fully_explains = bool(df_confirm["matches_walkthrough"].all())
    print(f"\n  Bug fully explains the discrepancy (all 4 horizons match walkthrough exactly): {bug_fully_explains}")

    # ---------------- Step 2: the required canonical table, all models, all horizons, N always 547/83 ----------------
    print("\n--- Step 2: CANONICAL horizon table (fallback convention, N=547 CV / N=83 test at every horizon) ---")
    canonical_rows = []
    for h in HORIZONS:
        h_name = "Delivery" if h == 0 else f">={h}m"
        y_h_cv, y_h_test = y_pat, y_test_pat  # canonical convention: no patient dropped, ever
        hyb_cv_h, hyb_test_h = hybrid_cv[h], hybrid_test[h]
        auc_hyb_cv = roc_auc_score(y_h_cv, hyb_cv_h)

        for m_name in models_cv:
            score_cv = models_cv[m_name][h]
            score_test = models_test[m_name][h]
            auc_cv = roc_auc_score(y_h_cv, score_cv)
            auprc_cv = average_precision_score(y_h_cv, score_cv)
            auc_test = roc_auc_score(y_h_test, score_test)

            boot_cv = paired_patient_bootstrap(y_h_cv, score_cv, hyb_cv_h, n_boot=2000, seed=42)
            pdel_cv, _, _ = delong_roc_test(y_h_cv, hyb_cv_h, score_cv)
            boot_test = paired_patient_bootstrap(y_h_test, score_test, hyb_test_h, n_boot=2000, seed=42)

            canonical_rows.append({
                "horizon": h_name, "model": m_name, "n_patients_cv": len(y_h_cv), "n_patients_test": len(y_h_test),
                "cv_auroc": round(auc_cv, 4), "cv_auprc": round(auprc_cv, 4), "test_auroc": round(auc_test, 4),
                "cv_delta_hybrid_minus_model": round(auc_hyb_cv - auc_cv, 4),
                "cv_boot_p": round(boot_cv["p_value"], 4), "cv_delong_p": round(pdel_cv, 4),
                "cv_ci_low": round(boot_cv["ci_95_low"], 4), "cv_ci_high": round(boot_cv["ci_95_high"], 4),
                "test_delta_hybrid_minus_model": round(roc_auc_score(y_h_test, hyb_test_h) - auc_test, 4),
                "test_boot_p": round(boot_test["p_value"], 4),
            })
        print(f"  {h_name}: " + ", ".join(f"{m}={roc_auc_score(y_h_cv, models_cv[m][h]):.4f}" for m in models_cv))

    df_canon = pd.DataFrame(canonical_rows)
    df_canon.to_csv(os.path.join(OUT_DIR, "canonical_horizon_table.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/bug_reproduction_confirmation.csv")
    print(f"Saved -> {OUT_DIR}/canonical_horizon_table.csv")


if __name__ == "__main__":
    main()
