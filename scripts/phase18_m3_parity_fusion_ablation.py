"""
Phase 18 -- M3 + Parity Fusion Ablation Study, Stage 1 (docs/phase18_m3_parity_fusion_ablation_protocol.md).

Answers, at the score level: (1) does parity add information beyond M3,
(2) which fusion mechanism combines the two best, (3) is any improvement
stable across horizons/resampling/the untouched test set. Every model here
consumes EXACTLY the same two frozen upstream scores per patient per
horizon (m3_score, parity_score) -- P6, Model 3, and the parity model
themselves are never retrained; only the fusion layer varies.

11 Stage-1 models (protocol Section 2): B0 prevalence-only, B1 parity-only,
B2 M3-only, F2 current-hybrid (L2, C=0.1, the reference), A3 probability-
space fusion, A4 logit-space fusion (4 weight-constraint variants), F1
unregularized logistic, F3 L1 logistic, F4 elastic-net (5 l1_ratios), F7
interaction logistic, P3 categorical-parity fusion.

Canonical patient-inclusion rule throughout (no patient ever dropped at any
horizon -- the exact bug found and fixed in the Model3+Parity Hybrid
investigation, results/model3_parity_hybrid/RECONCILIATION_AUDIT.md).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import to_logit, from_logit
from src.models.phase16_causal_attention import predict_at_horizon_for_patients
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
DATA_DIR = "data/processed_clinical"
CHECKPOINT_DIR = "results/phase16/checkpoints"
OUT_DIR = "results/phase18_fusion_ablation"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [0, 10, 20, 30]
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
L1_RATIO_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHA_GRID_FINE = np.linspace(0.0, 1.0, 21)


# ============================================================ fitters ====

def fit_logistic(X_tr, y_tr, X_ap, C=0.1, penalty="l2", l1_ratio=None):
    kwargs = dict(C=C, max_iter=2000, random_state=42)
    if penalty == "elasticnet":
        kwargs.update(penalty="elasticnet", solver="saga", l1_ratio=l1_ratio)
    elif penalty == "l1":
        kwargs.update(penalty="l1", solver="liblinear")
    else:
        kwargs.update(penalty="l2")
    clf = LogisticRegression(**kwargs)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_ap)[:, 1], clf


def fit_prob_avg(pm3_tr, ppar_tr, pm3_ap, ppar_ap, y_tr, alpha):
    if alpha == "selected":
        best_a, best_auc = 0.5, -1.0
        for a in ALPHA_GRID_FINE:
            s = a * pm3_tr + (1 - a) * ppar_tr
            auc = roc_auc_score(y_tr, s) if len(set(y_tr.tolist())) > 1 else 0.5
            if auc > best_auc:
                best_auc, best_a = auc, a
        alpha = best_a
    return alpha * pm3_ap + (1 - alpha) * ppar_ap, alpha


def _nll(params, ell_m3, ell_par, y):
    g0, g1, g2 = params
    z = g0 + g1 * ell_m3 + g2 * ell_par
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logit_fusion(ellm3_tr, ellpar_tr, y_tr, ellm3_ap, ellpar_ap, variant):
    if variant == "unconstrained":
        X_tr = np.column_stack([ellm3_tr, ellpar_tr])
        X_ap = np.column_stack([ellm3_ap, ellpar_ap])
        p_ap, clf = fit_logistic(X_tr, y_tr, X_ap, C=1e6)
        return p_ap, {"g0": float(clf.intercept_[0]), "g1": float(clf.coef_[0][0]), "g2": float(clf.coef_[0][1])}
    elif variant == "nonneg":
        res = minimize(_nll, x0=[0.0, 1.0, 1.0], args=(ellm3_tr, ellpar_tr, y_tr),
                        bounds=[(None, None), (0, None), (0, None)], method="L-BFGS-B")
        g0, g1, g2 = res.x
        z = g0 + g1 * ellm3_ap + g2 * ellpar_ap
        return from_logit(z), {"g0": float(g0), "g1": float(g1), "g2": float(g2)}
    elif variant == "sum_to_one":
        best_g1, best_auc = 0.5, -1.0
        for g1 in ALPHA_GRID_FINE:
            s = g1 * ellm3_tr + (1 - g1) * ellpar_tr
            auc = roc_auc_score(y_tr, s) if len(set(y_tr.tolist())) > 1 else 0.5
            if auc > best_auc:
                best_auc, best_g1 = auc, g1
        s_tr = best_g1 * ellm3_tr + (1 - best_g1) * ellpar_tr
        opt = minimize_scalar(lambda g0: _nll([g0, best_g1, 1 - best_g1], ellm3_tr, ellpar_tr, y_tr))
        g0 = opt.x
        z = g0 + best_g1 * ellm3_ap + (1 - best_g1) * ellpar_ap
        return from_logit(z), {"g0": float(g0), "g1": float(best_g1), "g2": float(1 - best_g1)}
    elif variant == "equal":
        s_tr = ellm3_tr + ellpar_tr
        opt = minimize_scalar(lambda g0: _nll([g0, 1.0, 1.0], ellm3_tr, ellpar_tr, y_tr))
        g0 = opt.x
        z = g0 + ellm3_ap + ellpar_ap
        return from_logit(z), {"g0": float(g0), "g1": 1.0, "g2": 1.0}
    else:
        raise ValueError(variant)


def parity_category_onehot(parity_raw):
    """Fixed, data-independent buckets: 0, 1, 2, >=3 -- defined before any evaluation."""
    cat = np.clip(parity_raw.astype(int), 0, 3)  # 3 absorbs 3+
    onehot = np.zeros((len(cat), 4))
    onehot[np.arange(len(cat)), cat] = 1.0
    return onehot[:, 1:]  # drop reference category (parity=0) to avoid collinearity with intercept


# ============================================================ main =======

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
    pred_cv_window = p6["pred_unweighted_cv"]
    pred_test_window = p6["pred_unweighted_test"]

    patient_data_cv, t_del_cv = build_patient_data(pred_cv_window, patient_ids_arr, df_rolling, clean_pids, y_lookup)
    patient_data_test, t_del_test = build_patient_data(pred_test_window, patient_ids_arr, df_rolling, test_pids, y_lookup)

    completed = load_completed_units(CHECKPOINT_DIR)
    fold_scorers = {f: load_scorer_checkpoint(completed[f"Model_3_magnitude_position_fold{f}"], in_dim=2, hidden=8) for f in range(5)}
    test_scorer = load_scorer_checkpoint(completed["Model_3_magnitude_position_testmodel"], in_dim=2, hidden=8)

    print("================================================================================")
    print("  PHASE 18 -- M3 + PARITY FUSION ABLATION (Stage 1)                                ")
    print("================================================================================")

    # ---------------- M3 scores, all horizons (identical procedure to every prior phase) ----------------
    m3_cv = {h: np.zeros(n_patients) for h in HORIZONS}
    m3_test = {h: np.zeros(len(test_pids)) for h in HORIZONS}
    for f_idx in range(5):
        te_pids = [p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx]
        te_mask = np.isin(pids_arr, te_pids)
        for h in HORIZONS:
            out = predict_at_horizon_for_patients(fold_scorers[f_idx], te_pids, patient_data_cv, t_del_cv, True, h)
            m3_cv[h][te_mask] = [out[p][0] for p in pids_arr[te_mask]]
    for h in HORIZONS:
        out_t = predict_at_horizon_for_patients(test_scorer, test_pids, patient_data_test, t_del_test, True, h)
        m3_test[h] = np.array([out_t[p][0] for p in test_pids])

    # ---------------- Parity model (horizon-independent) ----------------
    def fit_parity_lr(parity_train, y_train, parity_apply):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(parity_train.reshape(-1, 1))
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_tr, y_train)
        X_ap = scaler.transform(parity_apply.reshape(-1, 1))
        return clf.predict_proba(X_ap)[:, 1]

    p_parity_oof = np.zeros(n_patients)
    for f_idx in range(5):
        te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
        tr_mask = np.array([p not in te_pids for p in clean_pids])
        te_mask = ~tr_mask
        p_par_all = fit_parity_lr(parity_pat[tr_mask], y_pat[tr_mask], parity_pat)
        p_parity_oof[te_mask] = p_par_all[te_mask]
    p_parity_test = fit_parity_lr(parity_pat[trval_mask_pat], y_pat[trval_mask_pat], parity_pat)[test_mask_pat]

    # ================================================================
    # STAGE 1 MODEL FITTING -- per horizon, per fold, OOF + test-fold model
    # ================================================================
    MODEL_IDS = [
        "B0_prevalence", "B1_parity_only", "B2_m3_only",
        "F2_hybrid_C0.1",
        "A3_prob_avg_a0.25", "A3_prob_avg_a0.5", "A3_prob_avg_a0.75", "A3_prob_avg_selected",
        "A4_logit_unconstrained", "A4_logit_nonneg", "A4_logit_sum_to_one", "A4_logit_equal",
        "F1_unregularized", "F3_l1", "F7_interaction", "P3_categorical_parity",
    ]
    preds_cv = {mid: {h: np.zeros(n_patients) for h in HORIZONS} for mid in MODEL_IDS}
    preds_test = {mid: {h: np.zeros(len(test_pids)) for h in HORIZONS} for mid in MODEL_IDS}
    coef_rows = []

    for h in HORIZONS:
        for f_idx in range(5):
            te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
            tr_mask = np.array([p not in te_pids for p in clean_pids])
            te_mask = ~tr_mask
            y_tr, y_te = y_pat[tr_mask], y_pat[te_mask]

            pm3_tr, pm3_te = m3_cv[h][tr_mask], m3_cv[h][te_mask]
            ppar_tr, ppar_te = p_parity_oof[tr_mask], p_parity_oof[te_mask]
            zm3_tr = to_logit(pm3_tr); zm3_te = to_logit(pm3_te)
            zpar_tr = to_logit(ppar_tr); zpar_te = to_logit(ppar_te)

            preds_cv["B0_prevalence"][h][te_mask] = y_tr.mean()
            preds_cv["B1_parity_only"][h][te_mask] = ppar_te
            preds_cv["B2_m3_only"][h][te_mask] = pm3_te

            scaler = StandardScaler()
            Xz_tr = scaler.fit_transform(np.column_stack([zm3_tr, zpar_tr]))
            Xz_te = scaler.transform(np.column_stack([zm3_te, zpar_te]))

            p_ap, clf = fit_logistic(Xz_tr, y_tr, Xz_te, C=0.1)
            preds_cv["F2_hybrid_C0.1"][h][te_mask] = p_ap
            if h == 0:
                coef_rows.append({"model": "F2_hybrid_C0.1", "fold": f_idx, "intercept": float(clf.intercept_[0]),
                                   "coef_m3": float(clf.coef_[0][0]), "coef_parity": float(clf.coef_[0][1])})

            for a in [0.25, 0.5, 0.75, "selected"]:
                p_ap, a_used = fit_prob_avg(pm3_tr, ppar_tr, pm3_te, ppar_te, y_tr, a)
                key = f"A3_prob_avg_a{a}" if a != "selected" else "A3_prob_avg_selected"
                preds_cv[key][h][te_mask] = p_ap

            for variant in ["unconstrained", "nonneg", "sum_to_one", "equal"]:
                p_ap, params = fit_logit_fusion(zm3_tr, zpar_tr, y_tr, zm3_te, zpar_te, variant)
                preds_cv[f"A4_logit_{variant}"][h][te_mask] = p_ap
                if h == 0:
                    coef_rows.append({"model": f"A4_logit_{variant}", "fold": f_idx, **params})

            p_ap, clf = fit_logistic(Xz_tr, y_tr, Xz_te, C=1e6)
            preds_cv["F1_unregularized"][h][te_mask] = p_ap
            if h == 0:
                coef_rows.append({"model": "F1_unregularized", "fold": f_idx, "intercept": float(clf.intercept_[0]),
                                   "coef_m3": float(clf.coef_[0][0]), "coef_parity": float(clf.coef_[0][1])})

            p_ap, clf = fit_logistic(Xz_tr, y_tr, Xz_te, C=0.1, penalty="l1")
            preds_cv["F3_l1"][h][te_mask] = p_ap
            if h == 0:
                coef_rows.append({"model": "F3_l1", "fold": f_idx, "intercept": float(clf.intercept_[0]),
                                   "coef_m3": float(clf.coef_[0][0]), "coef_parity": float(clf.coef_[0][1])})

            inter_tr = zm3_tr * zpar_tr; inter_te = zm3_te * zpar_te
            Xi_tr = np.column_stack([Xz_tr, inter_tr]); Xi_te = np.column_stack([Xz_te, inter_te])
            p_ap, clf = fit_logistic(Xi_tr, y_tr, Xi_te, C=0.1)
            preds_cv["F7_interaction"][h][te_mask] = p_ap
            if h == 0:
                coef_rows.append({"model": "F7_interaction", "fold": f_idx, "intercept": float(clf.intercept_[0]),
                                   "coef_m3": float(clf.coef_[0][0]), "coef_parity": float(clf.coef_[0][1]),
                                   "coef_interaction": float(clf.coef_[0][2])})

            cat_tr = parity_category_onehot(parity_pat[tr_mask]); cat_te = parity_category_onehot(parity_pat[te_mask])
            zm3_only_tr = Xz_tr[:, [0]]; zm3_only_te = Xz_te[:, [0]]
            Xc_tr = np.column_stack([zm3_only_tr, cat_tr])
            Xc_te = np.column_stack([zm3_only_te, cat_te])
            p_ap, clf = fit_logistic(Xc_tr, y_tr, Xc_te, C=0.1)
            preds_cv["P3_categorical_parity"][h][te_mask] = p_ap
            if h == 0:
                coef_rows.append({"model": "P3_categorical_parity", "fold": f_idx, "intercept": float(clf.intercept_[0]),
                                   "coef_m3": float(clf.coef_[0][0]), "coef_parity_cat1": float(clf.coef_[0][1]),
                                   "coef_parity_cat2": float(clf.coef_[0][2]), "coef_parity_cat3plus": float(clf.coef_[0][3])})

        # ---------------- test-fold model (train_val -> test) ----------------
        y_trv = y_pat[trval_mask_pat]
        pm3_trv, pm3_te = m3_cv[h][trval_mask_pat], m3_test[h]
        ppar_trv, ppar_te = p_parity_oof[trval_mask_pat], p_parity_test
        zm3_trv = to_logit(pm3_trv); zm3_te = to_logit(pm3_te)
        zpar_trv = to_logit(ppar_trv); zpar_te = to_logit(ppar_te)

        preds_test["B0_prevalence"][h][:] = y_trv.mean()
        preds_test["B1_parity_only"][h][:] = ppar_te
        preds_test["B2_m3_only"][h][:] = pm3_te

        scaler = StandardScaler()
        Xz_trv = scaler.fit_transform(np.column_stack([zm3_trv, zpar_trv]))
        Xz_te = scaler.transform(np.column_stack([zm3_te, zpar_te]))
        p_ap, _ = fit_logistic(Xz_trv, y_trv, Xz_te, C=0.1)
        preds_test["F2_hybrid_C0.1"][h][:] = p_ap

        for a in [0.25, 0.5, 0.75, "selected"]:
            p_ap, _ = fit_prob_avg(pm3_trv, ppar_trv, pm3_te, ppar_te, y_trv, a)
            key = f"A3_prob_avg_a{a}" if a != "selected" else "A3_prob_avg_selected"
            preds_test[key][h][:] = p_ap

        for variant in ["unconstrained", "nonneg", "sum_to_one", "equal"]:
            p_ap, _ = fit_logit_fusion(zm3_trv, zpar_trv, y_trv, zm3_te, zpar_te, variant)
            preds_test[f"A4_logit_{variant}"][h][:] = p_ap

        p_ap, _ = fit_logistic(Xz_trv, y_trv, Xz_te, C=1e6)
        preds_test["F1_unregularized"][h][:] = p_ap
        p_ap, _ = fit_logistic(Xz_trv, y_trv, Xz_te, C=0.1, penalty="l1")
        preds_test["F3_l1"][h][:] = p_ap

        inter_trv = zm3_trv * zpar_trv; inter_te = zm3_te * zpar_te
        p_ap, _ = fit_logistic(np.column_stack([Xz_trv, inter_trv]), y_trv, np.column_stack([Xz_te, inter_te]), C=0.1)
        preds_test["F7_interaction"][h][:] = p_ap

        cat_trv = parity_category_onehot(parity_pat[trval_mask_pat]); cat_te = parity_category_onehot(parity_pat[test_mask_pat])
        Xc_trv = np.column_stack([scaler.transform(np.column_stack([zm3_trv, zpar_trv]))[:, [0]], cat_trv])
        Xc_te = np.column_stack([scaler.transform(np.column_stack([zm3_te, zpar_te]))[:, [0]], cat_te])
        p_ap, _ = fit_logistic(Xc_trv, y_trv, Xc_te, C=0.1)
        preds_test["P3_categorical_parity"][h][:] = p_ap

        print(f"  h={h:>3}m done -- {len(MODEL_IDS)} models fit (5-fold CV + test-fold model)")

    # ================================================================
    # C-GRID (F2 sensitivity) and L1_RATIO-GRID (F4 elastic net), delivery only
    # ================================================================
    print("\n--- F2 regularization sensitivity curve (delivery) ---")
    c_grid_rows = []
    for C in C_GRID:
        preds = np.zeros(n_patients)
        for f_idx in range(5):
            te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
            tr_mask = np.array([p not in te_pids for p in clean_pids])
            te_mask = ~tr_mask
            zm3_tr = to_logit(m3_cv[0][tr_mask]); zpar_tr = to_logit(p_parity_oof[tr_mask])
            zm3_te = to_logit(m3_cv[0][te_mask]); zpar_te = to_logit(p_parity_oof[te_mask])
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(np.column_stack([zm3_tr, zpar_tr]))
            Xte = scaler.transform(np.column_stack([zm3_te, zpar_te]))
            p_ap, clf = fit_logistic(Xtr, y_pat[tr_mask], Xte, C=C)
            preds[te_mask] = p_ap
            if C == 0.1:
                pass
        auc = roc_auc_score(y_pat, preds)
        c_grid_rows.append({"C": C, "cv_auroc": round(auc, 4)})
        print(f"  C={C:<8} CV AUROC={auc:.4f}")
    pd.DataFrame(c_grid_rows).to_csv(os.path.join(OUT_DIR, "f2_regularization_sensitivity.csv"), index=False)

    print("\n--- F4 elastic-net l1_ratio curve (delivery) ---")
    l1_grid_rows = []
    for rho in L1_RATIO_GRID:
        preds = np.zeros(n_patients)
        for f_idx in range(5):
            te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
            tr_mask = np.array([p not in te_pids for p in clean_pids])
            te_mask = ~tr_mask
            zm3_tr = to_logit(m3_cv[0][tr_mask]); zpar_tr = to_logit(p_parity_oof[tr_mask])
            zm3_te = to_logit(m3_cv[0][te_mask]); zpar_te = to_logit(p_parity_oof[te_mask])
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(np.column_stack([zm3_tr, zpar_tr]))
            Xte = scaler.transform(np.column_stack([zm3_te, zpar_te]))
            p_ap, _ = fit_logistic(Xtr, y_pat[tr_mask], Xte, C=0.1, penalty="elasticnet", l1_ratio=rho)
            preds[te_mask] = p_ap
        auc = roc_auc_score(y_pat, preds)
        l1_grid_rows.append({"l1_ratio": rho, "cv_auroc": round(auc, 4)})
        print(f"  l1_ratio={rho:<5} CV AUROC={auc:.4f}")
    pd.DataFrame(l1_grid_rows).to_csv(os.path.join(OUT_DIR, "f4_elasticnet_sensitivity.csv"), index=False)

    # ================================================================
    # MAIN RESULTS TABLE -- all 16 Stage-1 models x 4 horizons, CV + test, vs B2 and vs F2
    # ================================================================
    print("\n--- Main results table ---")
    rows = []
    for h in HORIZONS:
        h_name = "Delivery" if h == 0 else f">={h}m"
        base_m3_cv, base_m3_test = preds_cv["B2_m3_only"][h], preds_test["B2_m3_only"][h]
        base_hyb_cv, base_hyb_test = preds_cv["F2_hybrid_C0.1"][h], preds_test["F2_hybrid_C0.1"][h]
        for mid in MODEL_IDS:
            cv_arr, test_arr = preds_cv[mid][h], preds_test[mid][h]
            auc_cv = roc_auc_score(y_pat, cv_arr) if len(set(cv_arr.tolist())) > 1 else 0.5
            auprc_cv = average_precision_score(y_pat, cv_arr)
            auc_test = roc_auc_score(y_test_pat, test_arr) if len(set(test_arr.tolist())) > 1 else 0.5
            brier_cv = brier_score_loss(y_pat, np.clip(cv_arr, 0, 1))

            boot_vs_m3 = paired_patient_bootstrap(y_pat, base_m3_cv, cv_arr, n_boot=2000, seed=42)
            boot_vs_hyb = paired_patient_bootstrap(y_pat, base_hyb_cv, cv_arr, n_boot=2000, seed=42)
            boot_test_vs_m3 = paired_patient_bootstrap(y_test_pat, base_m3_test, test_arr, n_boot=2000, seed=42)

            rows.append({
                "horizon": h_name, "model": mid,
                "cv_auroc": round(auc_cv, 4), "cv_auprc": round(auprc_cv, 4), "cv_brier": round(brier_cv, 4),
                "test_auroc": round(auc_test, 4),
                "cv_delta_vs_m3": round(auc_cv - roc_auc_score(y_pat, base_m3_cv), 4),
                "cv_p_vs_m3": round(boot_vs_m3["p_value"], 4),
                "cv_ci_low_vs_m3": round(boot_vs_m3["ci_95_low"], 4), "cv_ci_high_vs_m3": round(boot_vs_m3["ci_95_high"], 4),
                "cv_delta_vs_hybrid": round(auc_cv - roc_auc_score(y_pat, base_hyb_cv), 4),
                "cv_p_vs_hybrid": round(boot_vs_hyb["p_value"], 4),
                "test_delta_vs_m3": round(auc_test - roc_auc_score(y_test_pat, base_m3_test), 4),
                "test_p_vs_m3": round(boot_test_vs_m3["p_value"], 4),
            })
        print(f"  {h_name}: " + ", ".join(f"{mid.split('_')[0]}={roc_auc_score(y_pat, preds_cv[mid][h]):.3f}" for mid in MODEL_IDS[:6]) + " ...")

    df_main = pd.DataFrame(rows)
    df_main.to_csv(os.path.join(OUT_DIR, "stage1_main_results.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/stage1_main_results.csv")

    # ================================================================
    # Coefficient stability table
    # ================================================================
    pd.DataFrame(coef_rows).to_csv(os.path.join(OUT_DIR, "stage1_coefficient_stability.csv"), index=False)
    print(f"Saved -> {OUT_DIR}/stage1_coefficient_stability.csv")

    # ================================================================
    # Prediction artifact export (long format, per protocol Section 1.3)
    # ================================================================
    export_rows = []
    for h in HORIZONS:
        for i, pid in enumerate(clean_pids):
            f_idx = folds_blob["assignment"][pid][0]
            row = {
                "patient_id": pid, "fold_id": f_idx, "split": "cv", "outcome": int(y_pat[i]),
                "parity_raw": float(parity_pat[i]), "parity_probability": round(float(p_parity_oof[i]), 5),
                "parity_logit": round(float(to_logit(p_parity_oof[i])), 5),
                "m3_score": round(float(m3_cv[h][i]), 5), "m3_logit": round(float(to_logit(m3_cv[h][i])), 5),
                "prediction_horizon": h,
            }
            for mid in MODEL_IDS:
                row[f"pred_{mid}"] = round(float(preds_cv[mid][h][i]), 5)
            export_rows.append(row)
        for i, pid in enumerate(test_pids):
            row = {
                "patient_id": pid, "fold_id": "test", "split": "test", "outcome": int(y_test_pat[i]),
                "parity_raw": float(parity_pat[clean_pids.index(pid)]), "parity_probability": round(float(p_parity_test[i]), 5),
                "parity_logit": round(float(to_logit(p_parity_test[i])), 5),
                "m3_score": round(float(m3_test[h][i]), 5), "m3_logit": round(float(to_logit(m3_test[h][i])), 5),
                "prediction_horizon": h,
            }
            for mid in MODEL_IDS:
                row[f"pred_{mid}"] = round(float(preds_test[mid][h][i]), 5)
            export_rows.append(row)
    pd.DataFrame(export_rows).to_csv(os.path.join(OUT_DIR, "stage1_patient_level_predictions.csv"), index=False)
    print(f"Saved -> {OUT_DIR}/stage1_patient_level_predictions.csv")

    print("\n--- Execution complete. ---")


if __name__ == "__main__":
    main()
