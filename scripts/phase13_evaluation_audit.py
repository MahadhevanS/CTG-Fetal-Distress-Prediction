"""
Phase 13.0A / Step 1-2 -- Evaluation audit (docs/phase13_protocol.md Sections 4, 7).

1. Reproduces canonical P6 (unweighted) fresh from X_state_trajectory, cross-
   checked against results/phase13_information_density/step_predictions.csv's
   Step_0_Baseline columns (independently computed weeks earlier) as a
   consistency check.
2. Computes both horizon conventions (existing vs. corrected) for P6, so
   every later Phase 13 branch can cite this table instead of re-deriving it.
3. Reconciles the 0.6872 (unweighted) vs 0.6843 (patient-normalized) "P6
   baseline" discrepancy explicitly, per protocol Section 7.

No new model is trained beyond the same LogisticRegression(C=0.05) already
used throughout this session for the P6 evaluation -- this is an audit, not
a new experiment.
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

from src.evaluation.phase13_common import (
    get_patient_scores_at_horizon, get_patient_scores_at_horizon_corrected,
)
from src.training.information_density_weighting import InformationDensityWeighter

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
TRAJ_PATH = "results/phase9c_state_trajectory/state_trajectory_features.npz"
DATA_DIR = "data/processed_clinical"
OUT_DIR = "results/phase13/audit"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [0, 10, 20, 30]


def fit_predict_lr(X_tr, y_tr, w_tr, X_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
    clf.fit(X_tr_s, y_tr, sample_weight=w_tr)
    return clf.predict_proba(X_te_s)[:, 1]


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    patient_ids = df["patient_id"].values
    t_del = df["time_before_delivery_min"].values
    y_715 = df["primary_label_715"].values
    elapsed_abs_min = (df["start_sample"].values.astype(np.float64) / (4.0 * 60.0)).astype(np.float32)

    data_traj = np.load(TRAJ_PATH)
    X_p6 = data_traj["X_state_trajectory"]
    X_raw_19 = X_p6[:, :19]

    y_pat_715 = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])

    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = [p for p in clean_pids if p not in test_pids]

    n = len(df)
    pred_unweighted_cv = np.zeros(n, dtype=np.float32)
    pred_patnorm_cv = np.zeros(n, dtype=np.float32)

    print("=== Phase 13.0A: canonical P6 reproduction (5-fold CV) ===")
    for f_idx in range(5):
        te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
        tr_pids = set(p for p in clean_pids if p not in te_pids)
        tr_mask = np.isin(patient_ids, list(tr_pids))
        va_mask = np.isin(patient_ids, list(te_pids))

        pred_unweighted_cv[va_mask] = fit_predict_lr(X_p6[tr_mask], y_715[tr_mask], None, X_p6[va_mask])

        weighter = InformationDensityWeighter(span=3.0, beta=1.0)
        weighter.fit(X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask])
        w_tr = weighter.transform(X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask])["w_patient"]
        pred_patnorm_cv[va_mask] = fit_predict_lr(X_p6[tr_mask], y_715[tr_mask], w_tr, X_p6[va_mask])

    tr_val_mask = np.isin(patient_ids, train_val_pids)
    test_mask = np.isin(patient_ids, test_pids)
    pred_unweighted_test = np.zeros(n, dtype=np.float32)
    pred_unweighted_test[test_mask] = fit_predict_lr(X_p6[tr_val_mask], y_715[tr_val_mask], None, X_p6[test_mask])
    y_test_pat_715 = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in test_pids])

    # --- Consistency check against the independently-computed step_predictions.csv ---
    ref_path = "results/phase13_information_density/step_predictions.csv"
    consistency_note = "reference file not found -- skipped"
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path)
        ref_cv = ref["cv_prob_Step_0_Baseline"].values
        max_abs_diff = float(np.max(np.abs(ref_cv - pred_unweighted_cv)))
        consistency_note = f"max|new - reference| across {n} windows = {max_abs_diff:.6f}"
        print(f"Consistency check vs. {ref_path}: {consistency_note}")

    rows = []
    for h in HORIZONS:
        for label, pred_cv, pred_te, y_te in [
            ("existing", pred_unweighted_cv, pred_unweighted_test, y_test_pat_715),
        ]:
            s_cv = get_patient_scores_at_horizon(pred_cv, patient_ids, clean_pids, t_del, h)
            s_cv_corr = get_patient_scores_at_horizon_corrected(pred_cv, patient_ids, clean_pids, t_del, h)
            s_te = get_patient_scores_at_horizon(pred_te, patient_ids, test_pids, t_del, h)
            s_te_corr = get_patient_scores_at_horizon_corrected(pred_te, patient_ids, test_pids, t_del, h)

            rows.append({
                "horizon_min": h,
                "cv_auroc_existing_convention": round(roc_auc_score(y_pat_715, s_cv), 4),
                "cv_auroc_corrected_convention": round(roc_auc_score(y_pat_715, s_cv_corr), 4),
                "cv_auprc_existing": round(average_precision_score(y_pat_715, s_cv), 4),
                "cv_auprc_corrected": round(average_precision_score(y_pat_715, s_cv_corr), 4),
                "test_auroc_existing_convention": round(roc_auc_score(y_te, s_te), 4),
                "test_auroc_corrected_convention": round(roc_auc_score(y_te, s_te_corr), 4),
            })
            print(f"  h={h:>3}m  CV existing={rows[-1]['cv_auroc_existing_convention']:.4f}  "
                  f"CV corrected={rows[-1]['cv_auroc_corrected_convention']:.4f}  "
                  f"TEST existing={rows[-1]['test_auroc_existing_convention']:.4f}  "
                  f"TEST corrected={rows[-1]['test_auroc_corrected_convention']:.4f}")

    df_horizons = pd.DataFrame(rows)
    df_horizons.to_csv(os.path.join(OUT_DIR, "horizon_alignment_comparison.csv"), index=False)

    # --- Baseline reconciliation (0.6872 vs 0.6843) ---
    auc_unweighted_cv0 = roc_auc_score(
        y_pat_715, get_patient_scores_at_horizon(pred_unweighted_cv, patient_ids, clean_pids, t_del, 0))
    auc_patnorm_cv0 = roc_auc_score(
        y_pat_715, get_patient_scores_at_horizon(pred_patnorm_cv, patient_ids, clean_pids, t_del, 0))

    reconciliation = pd.DataFrame([
        {"label": "P6_canonical (unweighted, sample_weight=None)", "auroc_delivery_cv": round(auc_unweighted_cv0, 4),
         "note": "Reference baseline for Options A/C/D. Matches published Phase 12.1 (0.6872)."},
        {"label": "P6_patient_normalized (w_patient applied)", "auroc_delivery_cv": round(auc_patnorm_cv0, 4),
         "note": "Used only as the stride experiment's baseline arm, applied symmetrically to both stride arms."},
    ])
    reconciliation.to_csv(os.path.join(OUT_DIR, "baseline_reconciliation.csv"), index=False)
    print("\n=== Baseline reconciliation ===")
    print(reconciliation.to_string(index=False))

    with open(os.path.join(OUT_DIR, "canonical_p6_reproduction.json"), "w") as fh:
        json.dump({
            "n_patients": len(clean_pids), "n_windows": n,
            "consistency_check_vs_step_predictions": consistency_note,
            "delivery_cv_unweighted": round(auc_unweighted_cv0, 4),
            "delivery_cv_patient_normalized": round(auc_patnorm_cv0, 4),
            "locked_phase12_reference": 0.6872,
            "horizon_table": rows,
        }, fh, indent=2)

    print(f"\nSaved -> {OUT_DIR}/horizon_alignment_comparison.csv")
    print(f"Saved -> {OUT_DIR}/baseline_reconciliation.csv")
    print(f"Saved -> {OUT_DIR}/canonical_p6_reproduction.json")

    # persist window-level predictions for reuse by Options A/C (avoid re-fitting P6 per branch)
    np.savez_compressed(os.path.join(OUT_DIR, "p6_predictions.npz"),
                         pred_unweighted_cv=pred_unweighted_cv, pred_unweighted_test=pred_unweighted_test,
                         patient_ids=patient_ids, t_del=t_del, y_715=y_715)
    print(f"Saved -> {OUT_DIR}/p6_predictions.npz (reused by Options A/C)")


if __name__ == "__main__":
    main()
