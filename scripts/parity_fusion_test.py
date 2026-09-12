"""
Parity-only maternal covariate fusion test.

Stage 0 (univariate check) found that of 9 admission-safe maternal covariates,
only parity showed a statistically real association with y_primary
(AUROC 0.411, p=0.0004, survives Bonferroni across 9 tests) -- the joint
9-covariate model was near-chance (0.5132). This tests parity alone, fused
with the existing frozen P6 system via the exact mechanism that won in
Phase 4 (logit prior modulation, scripts/phase4_fusion_eval.py:416-444):

    logit(p_fused) = logit(p_P6) + lambda * logit(p_parity)

p_P6 is read directly from results/phase13_information_density/step_predictions.csv
(cv_prob_Step_0_Baseline / test_prob_Step_0_Baseline) -- the same frozen,
already-validated P6 baseline used throughout this session, not retrained.

p_parity is a per-patient univariate LogisticRegression(parity -> y_primary),
fit per fold on that fold's training patients only (mirrors Phase 4's own
clinical-model-as-fitted-probability convention, not raw feature value).

lambda is selected by the same procedure Phase 4 used: grid sweep over
[0.1, 3.0], maximizing TRAINING-FOLD AUROC on the combined logit, applied
unchanged to that fold's held-out patients (single scalar, fit-on-train,
no separate validation split needed -- this is the low-capacity mechanism
that has repeatedly beaten every more flexible fusion tried in this project).
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

from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
STEP_PRED_PATH = "results/phase13_information_density/step_predictions.csv"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
OUT_DIR = "results/parity_fusion"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-6
LAMBDA_GRID = np.linspace(0.1, 3.0, 30)


def to_logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def get_patient_scores_at_horizon(pred_arr, patient_ids, clean_pids, t_del, h_val):
    scores = []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        t_pts = t_del[idx]
        if h_val > 0:
            eligible = np.where(t_pts >= h_val)[0]
            chosen = idx[eligible[0]] if len(eligible) > 0 else idx[-1]
        else:
            chosen = idx[-1]
        scores.append(pred_arr[chosen])
    return np.array(scores)


def fit_parity_lr(parity_train, y_train, parity_apply):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(parity_train.reshape(-1, 1))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    X_ap = scaler.transform(parity_apply.reshape(-1, 1))
    return clf.predict_proba(X_ap)[:, 1]


def select_lambda(logit_p6_train, logit_parity_train, y_train):
    best_lam, best_auc = 1.0, -1.0
    for lam in LAMBDA_GRID:
        comb = logit_p6_train + lam * logit_parity_train
        auc = roc_auc_score(y_train, comb)
        if auc > best_auc:
            best_auc, best_lam = auc, lam
    return best_lam, best_auc


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

    horizons = [0, 30]
    print("================================================================================")
    print("  PARITY-ONLY FUSION TEST  (logit(p_P6) + lambda * logit(p_parity))              ")
    print("================================================================================")

    # ---------------- 5-fold CV ----------------
    p6_cv_patient_by_h = {h: get_patient_scores_at_horizon(p6_cv_window, patient_ids, clean_pids, t_del, h) for h in horizons}
    fused_cv_patient_by_h = {h: np.zeros(len(clean_pids)) for h in horizons}
    lambdas_used = []

    for f_idx in range(5):
        te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
        tr_pids = [p for p in clean_pids if p not in te_pids]
        tr_mask_pat = np.array([p in tr_pids for p in clean_pids])
        te_mask_pat = ~tr_mask_pat

        p_parity_pat = fit_parity_lr(parity_pat[tr_mask_pat], y_pat_715[tr_mask_pat], parity_pat)
        logit_parity_pat = to_logit(p_parity_pat)

        for h in horizons:
            p6_h = p6_cv_patient_by_h[h]
            logit_p6_h = to_logit(p6_h)
            lam, _ = select_lambda(logit_p6_h[tr_mask_pat], logit_parity_pat[tr_mask_pat], y_pat_715[tr_mask_pat])
            comb_te = logit_p6_h[te_mask_pat] + lam * logit_parity_pat[te_mask_pat]
            fused_cv_patient_by_h[h][te_mask_pat] = 1.0 / (1.0 + np.exp(-comb_te))
            if h == 0:
                lambdas_used.append(lam)

    print(f"\nLambda selected per fold (delivery horizon): {[round(l, 2) for l in lambdas_used]}")

    # ---------------- held-out test ----------------
    trval_mask_pat = np.array([p in train_val_pids for p in clean_pids])
    test_mask_pat = np.array([p in test_pids for p in clean_pids])

    p_parity_test = fit_parity_lr(parity_pat[trval_mask_pat], y_pat_715[trval_mask_pat], parity_pat)
    logit_parity_test_pat = to_logit(p_parity_test)

    y_test_pat = y_pat_715[test_mask_pat]
    p6_test_patient_by_h = {h: get_patient_scores_at_horizon(p6_test_window, patient_ids, test_pids, t_del, h) for h in horizons}
    fused_test_patient_by_h = {}
    test_lambdas = {}
    for h in horizons:
        # test_prob_Step_0_Baseline is only populated for the held-out test rows
        # (zero elsewhere), so train+val patients' own lambda-selection must use
        # their CV-fold P6 probability instead; the actual test-set evaluation
        # below uses test_prob for the held-out patients only.
        p6_trval_h = get_patient_scores_at_horizon(p6_cv_window, patient_ids, train_val_pids, t_del, h)
        p6_testpat_h = get_patient_scores_at_horizon(p6_test_window, patient_ids, test_pids, t_del, h)

        logit_p6_trval = to_logit(p6_trval_h)
        logit_parity_trval = logit_parity_test_pat[trval_mask_pat]
        lam, _ = select_lambda(logit_p6_trval, logit_parity_trval, y_pat_715[trval_mask_pat])
        test_lambdas[h] = lam

        logit_p6_test = to_logit(p6_testpat_h)
        logit_parity_testpat = logit_parity_test_pat[test_mask_pat]
        comb_test = logit_p6_test + lam * logit_parity_testpat
        fused_test_patient_by_h[h] = 1.0 / (1.0 + np.exp(-comb_test))
        p6_test_patient_by_h[h] = p6_testpat_h

    print(f"Lambda selected on train+val (delivery horizon): {test_lambdas[0]:.2f}")

    # ---------------- report ----------------
    rows = []
    print("\n" + "=" * 100)
    for h in horizons:
        base_cv = p6_cv_patient_by_h[h]
        fused_cv = fused_cv_patient_by_h[h]
        auc_base_cv = roc_auc_score(y_pat_715, base_cv)
        auc_fused_cv = roc_auc_score(y_pat_715, fused_cv)
        auprc_base_cv = average_precision_score(y_pat_715, base_cv)
        auprc_fused_cv = average_precision_score(y_pat_715, fused_cv)
        boot_cv = paired_patient_bootstrap(y_pat_715, base_cv, fused_cv, n_boot=2000, seed=42)
        p_delong_cv, _, _ = delong_roc_test(y_pat_715, fused_cv, base_cv)

        base_test = p6_test_patient_by_h[h]
        fused_test = fused_test_patient_by_h[h]
        auc_base_test = roc_auc_score(y_test_pat, base_test)
        auc_fused_test = roc_auc_score(y_test_pat, fused_test)
        auprc_base_test = average_precision_score(y_test_pat, base_test)
        auprc_fused_test = average_precision_score(y_test_pat, fused_test)
        boot_test = paired_patient_bootstrap(y_test_pat, base_test, fused_test, n_boot=2000, seed=42)
        p_delong_test, _, _ = delong_roc_test(y_test_pat, fused_test, base_test)

        h_name = "Delivery (0m)" if h == 0 else f">={h}m"
        print(f"\n--- Horizon: {h_name} ---")
        print(f"  CV   : P6={auc_base_cv:.4f} (AUPRC {auprc_base_cv:.4f})  ->  P6+parity={auc_fused_cv:.4f} (AUPRC {auprc_fused_cv:.4f})  "
              f"Delta={auc_fused_cv-auc_base_cv:+.4f}  boot_p={boot_cv['p_value']:.3f}  delong_p={p_delong_cv:.4f}  "
              f"95%CI=[{boot_cv['ci_95_low']:+.4f},{boot_cv['ci_95_high']:+.4f}]")
        print(f"  TEST : P6={auc_base_test:.4f} (AUPRC {auprc_base_test:.4f})  ->  P6+parity={auc_fused_test:.4f} (AUPRC {auprc_fused_test:.4f})  "
              f"Delta={auc_fused_test-auc_base_test:+.4f}  boot_p={boot_test['p_value']:.3f}  delong_p={p_delong_test:.4f}  "
              f"95%CI=[{boot_test['ci_95_low']:+.4f},{boot_test['ci_95_high']:+.4f}]")

        rows.append({
            "horizon": h_name, "cv_auroc_p6": round(auc_base_cv, 4), "cv_auroc_fused": round(auc_fused_cv, 4),
            "cv_delta": round(auc_fused_cv - auc_base_cv, 4), "cv_boot_p": round(boot_cv["p_value"], 4),
            "cv_delong_p": round(p_delong_cv, 4), "cv_ci_low": round(boot_cv["ci_95_low"], 4), "cv_ci_high": round(boot_cv["ci_95_high"], 4),
            "test_auroc_p6": round(auc_base_test, 4), "test_auroc_fused": round(auc_fused_test, 4),
            "test_delta": round(auc_fused_test - auc_base_test, 4), "test_boot_p": round(boot_test["p_value"], 4),
            "test_delong_p": round(p_delong_test, 4), "test_ci_low": round(boot_test["ci_95_low"], 4), "test_ci_high": round(boot_test["ci_95_high"], 4),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, "parity_fusion_results.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/parity_fusion_results.csv")


if __name__ == "__main__":
    main()
