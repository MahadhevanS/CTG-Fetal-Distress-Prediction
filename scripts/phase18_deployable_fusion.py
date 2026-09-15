"""
Phase 18 Stage 1e -- DEPLOYABLE fusion (one model per fold, not one per
horizon).

Corrects a real design flaw in every fusion model built so far in this
study (Stage 1's 16 models, and -- it turns out -- the earlier Parity
Fusion and Model3+Parity Hybrid work from prior phases too): each was
fit SEPARATELY per retrospective horizon, using that horizon's own
truncated M3 score as a training feature. That is not deployable. At
inference time during actual labor, "minutes before delivery" is unknown
-- it is the very thing an early-warning system exists to help forecast.
A system cannot select which fusion coefficients to use based on
information that does not exist yet.

Model 3 itself never had this problem: it is ONE trained network, and
"horizon" is purely how far its already-fixed running score is truncated
for RETROSPECTIVE scoring (predict_all_prefixes) -- exactly the discipline
established in docs/phase16_protocol.md Section 4 ("every prefix length is
a separate training example sharing that patient's outcome label"). This
script applies that same discipline to the fusion layer: one fusion model
per fold, trained across every causal truncation of every training
patient's sequence (sample-weighted 1/T_i per patient so longer
recordings don't dominate, matching the same per-patient-normalization
principle Phase 13's information-density work and Phase 16's
patient_avg_bce_loss already use), then evaluated at each retrospective
horizon via the same eligible-prefix truncation used everywhere else.

Models covered: A3 fixed-alpha (0.25/0.5/0.75 -- genuinely need no fitting
at all beyond the parity model), A3 alpha-selected-once (a single alpha,
selected via pooled sample-weighted training AUROC across all truncations,
not per horizon), F2-deployable (one logistic hybrid), F7-deployable (one
interaction logistic), A4-sum-to-one-deployable. B2 (M3-only) is included
unchanged as the reference -- it was already deployable, being one trained
network truncated at eval time, same as always.
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

from src.evaluation.phase13_common import to_logit, from_logit, get_patient_scores_at_horizon_corrected
from src.models.phase16_causal_attention import predict_all_prefixes, eligible_prefix_length
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase11_bootstrap import paired_patient_bootstrap

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
DATA_DIR = "data/processed_clinical"
CHECKPOINT_DIR = "results/phase16/checkpoints"
OUT_DIR = "results/phase18_fusion_ablation"

HORIZONS = [0, 10, 20, 30]
ALPHA_GRID = np.linspace(0.0, 1.0, 21)


def fit_parity_lr(parity_train, y_train, parity_apply):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(parity_train.reshape(-1, 1))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    X_ap = scaler.transform(parity_apply.reshape(-1, 1))
    return clf.predict_proba(X_ap)[:, 1]


def build_pooled_training_set(train_pids, m3_seq_by_pid, parity_by_pid, y_lookup):
    """One row per (patient, causal truncation k=1..T_i). sample_weight =
    1/T_i so each patient contributes total weight 1, matching this
    project's established per-patient loss normalization."""
    rows_m3, rows_par, rows_y, rows_w = [], [], [], []
    for pid in train_pids:
        z_seq = m3_seq_by_pid[pid]  # length T_i, M3's running score at each truncation
        T_i = len(z_seq)
        y = y_lookup[pid]
        par = parity_by_pid[pid]
        for z_k in z_seq:
            rows_m3.append(z_k); rows_par.append(par); rows_y.append(y); rows_w.append(1.0 / T_i)
    return (np.array(rows_m3), np.array(rows_par), np.array(rows_y), np.array(rows_w))


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
    print("  PHASE 18 DEPLOYABLE FUSION -- one model per fold, ALL truncations pooled          ")
    print("================================================================================")

    # ---------------- M3's full running-score sequence per patient (real-time-valid) ----------------
    # Each patient scored by their OWN outer fold's scorer for CV; separately by test_scorer for test.
    m3_seq_cv = {}
    fold_of = {p: folds_blob["assignment"][p][0] for p in clean_pids}
    for pid in clean_pids:
        r_seq, elapsed_seq, y, T_i = patient_data_cv[pid]
        scorer = fold_scorers[fold_of[pid]]
        m3_seq_cv[pid] = predict_all_prefixes(scorer, r_seq, elapsed_seq, True, T_i)
    m3_seq_test = {}
    for pid in test_pids:
        r_seq, elapsed_seq, y, T_i = patient_data_test[pid]
        m3_seq_test[pid] = predict_all_prefixes(test_scorer, r_seq, elapsed_seq, True, T_i)

    # ---------------- parity model, OOF (unchanged from Stage 1) ----------------
    p_parity_oof = np.zeros(n_patients)
    for f_idx in range(5):
        te_pids = set(p for p in clean_pids if fold_of[p] == f_idx)
        tr_mask = np.array([p not in te_pids for p in clean_pids]); te_mask = ~tr_mask
        p_parity_oof[te_mask] = fit_parity_lr(parity_pat[tr_mask], y_pat[tr_mask], parity_pat)[te_mask]
    parity_prob_by_pid = {p: p_parity_oof[i] for i, p in enumerate(clean_pids)}
    p_parity_test_arr = fit_parity_lr(parity_pat[trval_mask_pat], y_pat[trval_mask_pat], parity_pat)[test_mask_pat]
    parity_prob_test_by_pid = {p: p_parity_test_arr[i] for i, p in enumerate(test_pids)}

    # M3-only reference at each horizon (already deployable -- one trained network, truncated at eval)
    m3_only_cv = {h: np.zeros(n_patients) for h in HORIZONS}
    m3_only_test = {h: np.zeros(len(test_pids)) for h in HORIZONS}
    for i, pid in enumerate(clean_pids):
        _, _, _, T_i = patient_data_cv[pid]
        for h in HORIZONS:
            k = eligible_prefix_length(t_del_cv[pid], h, T_i)
            m3_only_cv[h][i] = m3_seq_cv[pid][k - 1]
    for i, pid in enumerate(test_pids):
        _, _, _, T_i = patient_data_test[pid]
        for h in HORIZONS:
            k = eligible_prefix_length(t_del_test[pid], h, T_i)
            m3_only_test[h][i] = m3_seq_test[pid][k - 1]

    MODEL_IDS = ["A3_fixed_a0.25", "A3_fixed_a0.5", "A3_fixed_a0.75", "A3_selected_once",
                 "F2_deployable", "F7_deployable", "A4_sum_to_one_deployable"]
    preds_cv = {mid: {h: np.zeros(n_patients) for h in HORIZONS} for mid in MODEL_IDS}
    preds_test = {mid: {h: np.zeros(len(test_pids)) for h in HORIZONS} for mid in MODEL_IDS}
    fitted_params = []

    for f_idx in range(5):
        te_pids = [p for p in clean_pids if fold_of[p] == f_idx]
        tr_pids = [p for p in clean_pids if p not in te_pids]

        zm3_pool, par_pool, y_pool, w_pool = build_pooled_training_set(
            tr_pids, m3_seq_cv, parity_prob_by_pid, y_lookup)
        logit_m3_pool = to_logit(zm3_pool); logit_par_pool = to_logit(par_pool)

        # ---- A3 alpha selected once, sample-weighted, pooled across all truncations ----
        best_a, best_auc = 0.5, -1.0
        for a in ALPHA_GRID:
            s = a * zm3_pool + (1 - a) * par_pool
            auc = roc_auc_score(y_pool, s, sample_weight=w_pool)
            if auc > best_auc:
                best_auc, best_a = auc, a
        fitted_params.append({"model": "A3_selected_once", "fold": f_idx, "alpha": float(best_a)})

        # ---- F2-deployable: one logistic regression, sample-weighted, pooled ----
        scaler2 = StandardScaler()
        X2_pool = scaler2.fit_transform(np.column_stack([logit_m3_pool, logit_par_pool]))
        clf2 = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
        clf2.fit(X2_pool, y_pool, sample_weight=w_pool)
        fitted_params.append({"model": "F2_deployable", "fold": f_idx, "intercept": float(clf2.intercept_[0]),
                               "coef_m3": float(clf2.coef_[0][0]), "coef_parity": float(clf2.coef_[0][1])})

        # ---- F7-deployable: interaction, sample-weighted, pooled ----
        inter_pool = logit_m3_pool * logit_par_pool
        X7_pool = np.column_stack([X2_pool, inter_pool])
        clf7 = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
        clf7.fit(X7_pool, y_pool, sample_weight=w_pool)
        fitted_params.append({"model": "F7_deployable", "fold": f_idx, "intercept": float(clf7.intercept_[0]),
                               "coef_m3": float(clf7.coef_[0][0]), "coef_parity": float(clf7.coef_[0][1]),
                               "coef_interaction": float(clf7.coef_[0][2])})

        # ---- A4 sum-to-one deployable: single global weight, sample-weighted, pooled ----
        best_g1, best_auc4 = 0.5, -1.0
        for g1 in ALPHA_GRID:
            s = g1 * logit_m3_pool + (1 - g1) * logit_par_pool
            auc = roc_auc_score(y_pool, s, sample_weight=w_pool)
            if auc > best_auc4:
                best_auc4, best_g1 = auc, g1
        s_pool = best_g1 * logit_m3_pool + (1 - best_g1) * logit_par_pool
        cal = LogisticRegression(C=1e6, max_iter=2000)
        cal.fit(s_pool.reshape(-1, 1), y_pool, sample_weight=w_pool)
        g0 = float(cal.intercept_[0]); scale = float(cal.coef_[0][0])
        fitted_params.append({"model": "A4_sum_to_one_deployable", "fold": f_idx, "g1": float(best_g1), "g0": g0, "scale": scale})

        # ---- apply all fitted models to held-out patients, at every horizon ----
        for pid in te_pids:
            i = clean_pids.index(pid)
            _, _, _, T_i = patient_data_cv[pid]
            par_p = parity_prob_by_pid[pid]; logit_par_p = to_logit(par_p)
            for h in HORIZONS:
                k = eligible_prefix_length(t_del_cv[pid], h, T_i)
                z_k = m3_seq_cv[pid][k - 1]; logit_m3_k = to_logit(z_k)

                preds_cv["A3_fixed_a0.25"][h][i] = 0.25 * z_k + 0.75 * par_p
                preds_cv["A3_fixed_a0.5"][h][i] = 0.5 * z_k + 0.5 * par_p
                preds_cv["A3_fixed_a0.75"][h][i] = 0.75 * z_k + 0.25 * par_p
                preds_cv["A3_selected_once"][h][i] = best_a * z_k + (1 - best_a) * par_p

                x2 = scaler2.transform([[logit_m3_k, logit_par_p]])
                preds_cv["F2_deployable"][h][i] = clf2.predict_proba(x2)[0, 1]
                x7 = np.column_stack([x2, [[logit_m3_k * logit_par_p]]])
                preds_cv["F7_deployable"][h][i] = clf7.predict_proba(x7)[0, 1]
                s_val = best_g1 * logit_m3_k + (1 - best_g1) * logit_par_p
                preds_cv["A4_sum_to_one_deployable"][h][i] = from_logit(g0 + scale * s_val)

        print(f"  fold {f_idx}: alpha_selected={best_a:.2f}, F2 coef_m3={clf2.coef_[0][0]:.3f} coef_parity={clf2.coef_[0][1]:.3f}, "
              f"A4 g1={best_g1:.2f}")

    # ---------------- test-fold model (train_val -> test), same pooled-truncation discipline ----------------
    zm3_pool, par_pool, y_pool, w_pool = build_pooled_training_set(
        train_val_pids, m3_seq_cv, parity_prob_by_pid, y_lookup)
    logit_m3_pool = to_logit(zm3_pool); logit_par_pool = to_logit(par_pool)

    best_a, best_auc = 0.5, -1.0
    for a in ALPHA_GRID:
        s = a * zm3_pool + (1 - a) * par_pool
        auc = roc_auc_score(y_pool, s, sample_weight=w_pool)
        if auc > best_auc:
            best_auc, best_a = auc, a

    scaler2 = StandardScaler()
    X2_pool = scaler2.fit_transform(np.column_stack([logit_m3_pool, logit_par_pool]))
    clf2 = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
    clf2.fit(X2_pool, y_pool, sample_weight=w_pool)

    inter_pool = logit_m3_pool * logit_par_pool
    X7_pool = np.column_stack([X2_pool, inter_pool])
    clf7 = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
    clf7.fit(X7_pool, y_pool, sample_weight=w_pool)

    best_g1, best_auc4 = 0.5, -1.0
    for g1 in ALPHA_GRID:
        s = g1 * logit_m3_pool + (1 - g1) * logit_par_pool
        auc = roc_auc_score(y_pool, s, sample_weight=w_pool)
        if auc > best_auc4:
            best_auc4, best_g1 = auc, g1
    s_pool = best_g1 * logit_m3_pool + (1 - best_g1) * logit_par_pool
    cal = LogisticRegression(C=1e6, max_iter=2000)
    cal.fit(s_pool.reshape(-1, 1), y_pool, sample_weight=w_pool)
    g0 = float(cal.intercept_[0]); scale = float(cal.coef_[0][0])

    for i, pid in enumerate(test_pids):
        _, _, _, T_i = patient_data_test[pid]
        par_p = parity_prob_test_by_pid[pid]; logit_par_p = to_logit(par_p)
        for h in HORIZONS:
            k = eligible_prefix_length(t_del_test[pid], h, T_i)
            z_k = m3_seq_test[pid][k - 1]; logit_m3_k = to_logit(z_k)

            preds_test["A3_fixed_a0.25"][h][i] = 0.25 * z_k + 0.75 * par_p
            preds_test["A3_fixed_a0.5"][h][i] = 0.5 * z_k + 0.5 * par_p
            preds_test["A3_fixed_a0.75"][h][i] = 0.75 * z_k + 0.25 * par_p
            preds_test["A3_selected_once"][h][i] = best_a * z_k + (1 - best_a) * par_p

            x2 = scaler2.transform([[logit_m3_k, logit_par_p]])
            preds_test["F2_deployable"][h][i] = clf2.predict_proba(x2)[0, 1]
            x7 = np.column_stack([x2, [[logit_m3_k * logit_par_p]]])
            preds_test["F7_deployable"][h][i] = clf7.predict_proba(x7)[0, 1]
            s_val = best_g1 * logit_m3_k + (1 - best_g1) * logit_par_p
            preds_test["A4_sum_to_one_deployable"][h][i] = from_logit(g0 + scale * s_val)

    pd.DataFrame(fitted_params).to_csv(os.path.join(OUT_DIR, "deployable_fusion_fold_params.csv"), index=False)

    # ================================================================
    # RESULTS
    # ================================================================
    print("\n--- Deployable fusion results (one model per fold, evaluated at every horizon) ---")
    rows = []
    for h in HORIZONS:
        h_name = "Delivery" if h == 0 else f">={h}m"
        base_m3_cv, base_m3_test = m3_only_cv[h], m3_only_test[h]
        for mid in MODEL_IDS:
            cv_arr, test_arr = preds_cv[mid][h], preds_test[mid][h]
            auc_cv = roc_auc_score(y_pat, cv_arr)
            auprc_cv = average_precision_score(y_pat, cv_arr)
            auc_test = roc_auc_score(y_test_pat, test_arr)
            boot_cv = paired_patient_bootstrap(y_pat, base_m3_cv, cv_arr, n_boot=2000, seed=42)
            boot_test = paired_patient_bootstrap(y_test_pat, base_m3_test, test_arr, n_boot=2000, seed=42)
            rows.append({
                "horizon": h_name, "model": mid,
                "cv_auroc": round(auc_cv, 4), "cv_auprc": round(auprc_cv, 4),
                "cv_delta_vs_m3only": round(auc_cv - roc_auc_score(y_pat, base_m3_cv), 4),
                "cv_p_vs_m3only": round(boot_cv["p_value"], 4),
                "cv_ci_low": round(boot_cv["ci_95_low"], 4), "cv_ci_high": round(boot_cv["ci_95_high"], 4),
                "test_auroc": round(auc_test, 4),
                "test_delta_vs_m3only": round(auc_test - roc_auc_score(y_test_pat, base_m3_test), 4),
                "test_p_vs_m3only": round(boot_test["p_value"], 4),
            })
        m3_auc = roc_auc_score(y_pat, base_m3_cv)
        print(f"  {h_name}: M3-only={m3_auc:.4f} | " + ", ".join(f"{mid}={roc_auc_score(y_pat, preds_cv[mid][h]):.4f}" for mid in MODEL_IDS))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, "deployable_fusion_results.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/deployable_fusion_results.csv")
    print(f"Saved -> {OUT_DIR}/deployable_fusion_fold_params.csv")
    print("\n--- Execution complete. ---")


if __name__ == "__main__":
    main()
