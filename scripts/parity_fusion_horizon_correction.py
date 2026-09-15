"""
Tier-2 hardening item 1 (docs/pre_external_validation_hardening_plan.md #1).

scripts/parity_fusion_test.py's get_patient_scores_at_horizon() uses the
ORIGINAL first-eligible-chronological-window convention -- not
get_patient_scores_at_horizon_corrected() (argmin-nearest, subject to
t_i >= h), which every Phase 13-audit, Phase 14, Phase 15, and Phase 16
comparison uses. Delivery (h=0) is identical either way (both select the
last window), so it is NOT recomputed here. Only >=10m/>=20m/>=30m are
recomputed, swapping the horizon-selection call and nothing else: same
frozen P6 window scores, same per-fold parity LogisticRegression objects,
same lambda-selection procedure (src.evaluation.phase13_common's shared
implementation, identical formula to the original script's local copy).
No refitting of anything -- a horizon-selection substitution only.

The original (inconsistent-convention) >=30m number is kept alongside the
corrected one, both explicitly labeled, per this project's standing rule
against silent reconciliation (see the P90-estimator handling, hardening
item #2).
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
    get_patient_scores_at_horizon_corrected, to_logit, from_logit, select_lambda_trainfold,
)
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
STEP_PRED_PATH = "results/phase13_information_density/step_predictions.csv"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
OUT_DIR = "results/parity_fusion"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [10, 20, 30]  # delivery (0) unaffected, not recomputed
ORIGINAL_30M = {  # from results/parity_fusion/parity_fusion_results.csv (original, uncorrected convention)
    "cv_auroc_p6": 0.5857, "cv_auroc_fused": 0.6351, "cv_delta": 0.0494, "cv_boot_p": 0.078, "cv_delong_p": 0.0736,
    "cv_ci_low": -0.0069, "cv_ci_high": 0.1026,
    "test_auroc_p6": 0.6845, "test_auroc_fused": 0.7629, "test_delta": 0.0784, "test_boot_p": 0.166, "test_delong_p": 0.1444,
    "test_ci_low": -0.0345, "test_ci_high": 0.1778,
}


def fit_parity_lr(parity_train, y_train, parity_apply):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(parity_train.reshape(-1, 1))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    X_ap = scaler.transform(parity_apply.reshape(-1, 1))
    return clf.predict_proba(X_ap)[:, 1]


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    df = pd.read_csv(STEP_PRED_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    patient_ids = df["patient_id"].values
    t_del = df["time_before_delivery_min"].values
    y_715 = df["primary_label_715"].values
    p6_cv_window = df["cv_prob_Step_0_Baseline"].values
    p6_test_window = df["test_prob_Step_0_Baseline"].values

    meta = pd.read_csv(METADATA_PATH)
    if "record_id" in meta.columns:
        meta = meta.set_index("record_id")
    parity_by_pid = {str(p): float(meta.loc[int(p), "parity"]) for p in clean_pids}

    y_pat_715 = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    parity_pat = np.array([parity_by_pid[p] for p in clean_pids])

    test_pt = torch.load("data/processed_clinical/test_dataset.pt", weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = [p for p in clean_pids if p not in test_pids]

    print("================================================================================")
    print("  PARITY FUSION -- HORIZON-CONVENTION CORRECTION (Tier-2 hardening item #1)        ")
    print("  Using get_patient_scores_at_horizon_corrected -- same convention as Model 3/4/P90")
    print("================================================================================")

    # ---------------- 5-fold CV ----------------
    p6_cv_patient_by_h = {h: get_patient_scores_at_horizon_corrected(p6_cv_window, patient_ids, clean_pids, t_del, h) for h in HORIZONS}
    fused_cv_patient_by_h = {h: np.zeros(len(clean_pids)) for h in HORIZONS}

    for f_idx in range(5):
        te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
        tr_pids = [p for p in clean_pids if p not in te_pids]
        tr_mask_pat = np.array([p in tr_pids for p in clean_pids])
        te_mask_pat = ~tr_mask_pat

        p_parity_pat = fit_parity_lr(parity_pat[tr_mask_pat], y_pat_715[tr_mask_pat], parity_pat)
        logit_parity_pat = to_logit(p_parity_pat)

        for h in HORIZONS:
            p6_h = p6_cv_patient_by_h[h]
            logit_p6_h = to_logit(p6_h)
            lam, _ = select_lambda_trainfold(logit_p6_h[tr_mask_pat], logit_parity_pat[tr_mask_pat], y_pat_715[tr_mask_pat])
            comb_te = logit_p6_h[te_mask_pat] + lam * logit_parity_pat[te_mask_pat]
            fused_cv_patient_by_h[h][te_mask_pat] = from_logit(comb_te)

    # ---------------- held-out internal test partition ----------------
    trval_mask_pat = np.array([p in train_val_pids for p in clean_pids])
    test_mask_pat = np.array([p in test_pids for p in clean_pids])

    p_parity_test = fit_parity_lr(parity_pat[trval_mask_pat], y_pat_715[trval_mask_pat], parity_pat)
    logit_parity_test_pat = to_logit(p_parity_test)

    y_test_pat = y_pat_715[test_mask_pat]
    fused_test_patient_by_h = {}
    p6_test_patient_by_h = {}
    for h in HORIZONS:
        p6_trval_h = get_patient_scores_at_horizon_corrected(p6_cv_window, patient_ids, train_val_pids, t_del, h)
        p6_testpat_h = get_patient_scores_at_horizon_corrected(p6_test_window, patient_ids, test_pids, t_del, h)

        logit_p6_trval = to_logit(p6_trval_h)
        logit_parity_trval = logit_parity_test_pat[trval_mask_pat]
        lam, _ = select_lambda_trainfold(logit_p6_trval, logit_parity_trval, y_pat_715[trval_mask_pat])

        logit_p6_test = to_logit(p6_testpat_h)
        logit_parity_testpat = logit_parity_test_pat[test_mask_pat]
        comb_test = logit_p6_test + lam * logit_parity_testpat
        fused_test_patient_by_h[h] = from_logit(comb_test)
        p6_test_patient_by_h[h] = p6_testpat_h

    # ---------------- report ----------------
    rows = []
    for h in HORIZONS:
        base_cv = p6_cv_patient_by_h[h]
        fused_cv = fused_cv_patient_by_h[h]
        auc_base_cv = roc_auc_score(y_pat_715, base_cv)
        auc_fused_cv = roc_auc_score(y_pat_715, fused_cv)
        boot_cv = paired_patient_bootstrap(y_pat_715, base_cv, fused_cv, n_boot=2000, seed=42)
        p_delong_cv, _, _ = delong_roc_test(y_pat_715, fused_cv, base_cv)

        base_test = p6_test_patient_by_h[h]
        fused_test = fused_test_patient_by_h[h]
        auc_base_test = roc_auc_score(y_test_pat, base_test)
        auc_fused_test = roc_auc_score(y_test_pat, fused_test)
        boot_test = paired_patient_bootstrap(y_test_pat, base_test, fused_test, n_boot=2000, seed=42)
        p_delong_test, _, _ = delong_roc_test(y_test_pat, fused_test, base_test)

        row = {
            "horizon": f">={h}m", "convention": "corrected",
            "cv_auroc_p6": round(auc_base_cv, 4), "cv_auroc_fused": round(auc_fused_cv, 4),
            "cv_delta": round(auc_fused_cv - auc_base_cv, 4), "cv_boot_p": round(boot_cv["p_value"], 4),
            "cv_delong_p": round(p_delong_cv, 4), "cv_ci_low": round(boot_cv["ci_95_low"], 4), "cv_ci_high": round(boot_cv["ci_95_high"], 4),
            "test_auroc_p6": round(auc_base_test, 4), "test_auroc_fused": round(auc_fused_test, 4),
            "test_delta": round(auc_fused_test - auc_base_test, 4), "test_boot_p": round(boot_test["p_value"], 4),
            "test_delong_p": round(p_delong_test, 4), "test_ci_low": round(boot_test["ci_95_low"], 4), "test_ci_high": round(boot_test["ci_95_high"], 4),
        }
        rows.append(row)
        print(f"\n--- Horizon: >={h}m (corrected convention) ---")
        print(f"  CV   : P6={auc_base_cv:.4f} -> P6+parity={auc_fused_cv:.4f}  Delta={row['cv_delta']:+.4f}  "
              f"boot_p={row['cv_boot_p']:.3f}  CI=[{row['cv_ci_low']:+.4f},{row['cv_ci_high']:+.4f}]")
        print(f"  TEST : P6={auc_base_test:.4f} -> P6+parity={auc_fused_test:.4f}  Delta={row['test_delta']:+.4f}  "
              f"boot_p={row['test_boot_p']:.3f}  CI=[{row['test_ci_low']:+.4f},{row['test_ci_high']:+.4f}]")

    # original (uncorrected-convention) >=30m row, kept alongside, explicitly labeled
    orig_row = {"horizon": ">=30m", "convention": "original (uncorrected, first-eligible-chronological-window)"}
    orig_row.update(ORIGINAL_30M)
    rows.append(orig_row)

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "parity_fusion_horizon_corrected_results.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\n{'='*90}")
    print("SUMMARY -- >=30m, both conventions, side by side:")
    r30_corrected = [r for r in rows if r["horizon"] == ">=30m" and r["convention"] == "corrected"][0]
    print(f"  original (uncorrected) : CV delta={ORIGINAL_30M['cv_delta']:+.4f} (p={ORIGINAL_30M['cv_boot_p']:.3f})   "
          f"TEST delta={ORIGINAL_30M['test_delta']:+.4f} (p={ORIGINAL_30M['test_boot_p']:.3f})")
    print(f"  corrected               : CV delta={r30_corrected['cv_delta']:+.4f} (p={r30_corrected['cv_boot_p']:.3f})   "
          f"TEST delta={r30_corrected['test_delta']:+.4f} (p={r30_corrected['test_boot_p']:.3f})")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
