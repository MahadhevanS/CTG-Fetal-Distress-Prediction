"""
Phase 16, Model 4 -- peak-aware fusion (docs/phase16_protocol.md Section 5).

Gated open per Section 9: Model 3 cleared all four Section 8 criteria (see
results/phase16/model3_verification.json -- epoch-cap concern resolved,
fold-resplit shows a stable +0.027 to +0.034 effect, no sign instability).

Feature vector, four scalars, deliberately low-capacity per the protocol's
explicit rejection of a deep fusion network:
    [max(prefix), p90(prefix), mean(prefix), z_from_model_3(prefix)]
-> LogisticRegression(C=0.1, fixed, not tuned)

Reuses Model 3's ALREADY-TRAINED, checkpointed per-fold scorers
(results/phase16/checkpoints/Model_3_magnitude_position_fold*.pt and
..._testmodel.pt) -- Model 3 is not retrained here, only its frozen output
is used as one of four fusion inputs, exactly as P6 itself is never
retrained anywhere in this project.
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

from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected
from src.models.phase16_causal_attention import eligible_prefix_length, score_full_sequence, pooled_prediction_from_logits
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
DATA_DIR = "data/processed_clinical"
OUT_DIR = "results/phase16"
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")

HORIZONS = [0, 10, 20, 30]
MODEL3_NAME = "Model_3_magnitude_position"


def scalar_pool_features(r_seq_np, k):
    r_k = r_seq_np[:k]
    return float(np.max(r_k)), float(np.percentile(r_k, 90)), float(np.mean(r_k))


def model3_z_at_k(scorer, r_seq, elapsed_seq, k):
    logits = score_full_sequence(scorer, r_seq, elapsed_seq, use_elapsed=True)
    z, _ = pooled_prediction_from_logits(logits, r_seq, k)
    return float(z.item())


def build_feature_matrix(scorer, patient_data, t_del_by_pid, pids, h_val):
    X = []
    for pid in pids:
        r_seq, elapsed_seq, y, T_i = patient_data[pid]
        t_del_seq = t_del_by_pid[pid]
        k = eligible_prefix_length(t_del_seq, h_val, T_i)
        r_np = r_seq.numpy()
        mx, p90, mean_ = scalar_pool_features(r_np, k)
        z3 = model3_z_at_k(scorer, r_seq, elapsed_seq, k)
        X.append([mx, p90, mean_, z3])
    return np.array(X, dtype=np.float64)


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids_arr = p6["patient_ids"]
    pred_cv_window = p6["pred_unweighted_cv"]
    pred_test_window = p6["pred_unweighted_test"]

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    y_lookup = {p: int(df[df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    y_pat = np.array([y_lookup[p] for p in clean_pids])

    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = [p for p in clean_pids if p not in test_pids]
    y_test_pat = np.array([y_lookup[p] for p in test_pids])

    patient_data_cv, t_del_cv = build_patient_data(pred_cv_window, patient_ids_arr, df, clean_pids, y_lookup)
    patient_data_test, t_del_test = build_patient_data(pred_test_window, patient_ids_arr, df, test_pids, y_lookup)

    completed = load_completed_units(CHECKPOINT_DIR)
    fold_scorers = {}
    for f_idx in range(5):
        unit_id = f"{MODEL3_NAME}_fold{f_idx}"
        assert unit_id in completed, f"Missing Model 3 checkpoint for fold {f_idx} -- run phase16_temporal_attention_model.py first"
        fold_scorers[f_idx] = load_scorer_checkpoint(completed[unit_id], in_dim=2, hidden=8)
    test_unit_id = f"{MODEL3_NAME}_testmodel"
    assert test_unit_id in completed, "Missing Model 3 test-model checkpoint"
    test_scorer = load_scorer_checkpoint(completed[test_unit_id], in_dim=2, hidden=8)

    print("================================================================================")
    print("  PHASE 16, MODEL 4 -- PEAK-AWARE FUSION  (reusing frozen Model 3 checkpoints)     ")
    print("================================================================================")

    sw_cv_by_h = {h: get_patient_scores_at_horizon_corrected(pred_cv_window, patient_ids_arr, clean_pids,
                                                              df["time_before_delivery_min"].values, h) for h in HORIZONS}
    sw_test_by_h = {h: get_patient_scores_at_horizon_corrected(pred_test_window, patient_ids_arr, test_pids,
                                                                df["time_before_delivery_min"].values, h) for h in HORIZONS}

    rows = []
    for h in HORIZONS:
        cv_pred = np.zeros(len(clean_pids))
        model3_cv = np.zeros(len(clean_pids))
        for f_idx in range(5):
            te_pids = [p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx]
            tr_pids = [p for p in clean_pids if p not in te_pids]
            scorer = fold_scorers[f_idx]

            X_tr = build_feature_matrix(scorer, patient_data_cv, t_del_cv, tr_pids, h)
            y_tr = np.array([y_lookup[p] for p in tr_pids])
            X_te = build_feature_matrix(scorer, patient_data_cv, t_del_cv, te_pids, h)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            clf.fit(X_tr_s, y_tr)
            probs = clf.predict_proba(X_te_s)[:, 1]

            te_mask = np.array([p in te_pids for p in clean_pids])
            cv_pred[te_mask] = probs
            model3_cv[te_mask] = X_te[:, 3]  # z_from_model_3 column, for attribution comparison

        X_trval = build_feature_matrix(test_scorer, patient_data_cv, t_del_cv, train_val_pids, h)
        y_trval = np.array([y_lookup[p] for p in train_val_pids])
        X_test = build_feature_matrix(test_scorer, patient_data_test, t_del_test, test_pids, h)
        scaler_t = StandardScaler()
        X_trval_s = scaler_t.fit_transform(X_trval)
        X_test_s = scaler_t.transform(X_test)
        clf_t = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        clf_t.fit(X_trval_s, y_trval)
        test_pred = clf_t.predict_proba(X_test_s)[:, 1]
        model3_test = X_test[:, 3]

        sw_cv = sw_cv_by_h[h]
        sw_test = sw_test_by_h[h]

        auc_cv = roc_auc_score(y_pat, cv_pred)
        auc_sw_cv = roc_auc_score(y_pat, sw_cv)
        auc_m3_cv = roc_auc_score(y_pat, model3_cv)
        auc_test = roc_auc_score(y_test_pat, test_pred)
        auc_sw_test = roc_auc_score(y_test_pat, sw_test)
        auc_m3_test = roc_auc_score(y_test_pat, model3_test)
        auprc_cv = average_precision_score(y_pat, cv_pred)

        boot_sw_cv = paired_patient_bootstrap(y_pat, sw_cv, cv_pred, n_boot=2000, seed=42)
        pdel_sw_cv, _, _ = delong_roc_test(y_pat, cv_pred, sw_cv)
        boot_sw_test = paired_patient_bootstrap(y_test_pat, sw_test, test_pred, n_boot=2000, seed=42)
        pdel_sw_test, _, _ = delong_roc_test(y_test_pat, test_pred, sw_test)

        boot_m3_cv = paired_patient_bootstrap(y_pat, model3_cv, cv_pred, n_boot=2000, seed=42)
        pdel_m3_cv, _, _ = delong_roc_test(y_pat, cv_pred, model3_cv)
        boot_m3_test = paired_patient_bootstrap(y_test_pat, model3_test, test_pred, n_boot=2000, seed=42)
        pdel_m3_test, _, _ = delong_roc_test(y_test_pat, test_pred, model3_test)

        row = {
            "horizon_min": h,
            "cv_auroc_sw": round(auc_sw_cv, 4), "cv_auroc_model3": round(auc_m3_cv, 4), "cv_auroc_model4": round(auc_cv, 4),
            "cv_delta_vs_sw": round(auc_cv - auc_sw_cv, 4), "cv_p_vs_sw": round(boot_sw_cv["p_value"], 4),
            "cv_ci_low_vs_sw": round(boot_sw_cv["ci_95_low"], 4), "cv_ci_high_vs_sw": round(boot_sw_cv["ci_95_high"], 4),
            "cv_delta_vs_model3": round(auc_cv - auc_m3_cv, 4), "cv_p_vs_model3": round(boot_m3_cv["p_value"], 4),
            "cv_ci_low_vs_model3": round(boot_m3_cv["ci_95_low"], 4), "cv_ci_high_vs_model3": round(boot_m3_cv["ci_95_high"], 4),
            "cv_auprc": round(auprc_cv, 4),
            "test_auroc_sw": round(auc_sw_test, 4), "test_auroc_model3": round(auc_m3_test, 4), "test_auroc_model4": round(auc_test, 4),
            "test_delta_vs_sw": round(auc_test - auc_sw_test, 4), "test_p_vs_sw": round(boot_sw_test["p_value"], 4),
            "test_delta_vs_model3": round(auc_test - auc_m3_test, 4), "test_p_vs_model3": round(boot_m3_test["p_value"], 4),
        }
        rows.append(row)
        print(f"h={h:>3}m  CV: sw={auc_sw_cv:.4f} model3={auc_m3_cv:.4f} model4={auc_cv:.4f}  "
              f"(vs sw: d{auc_cv-auc_sw_cv:+.4f} p={boot_sw_cv['p_value']:.3f} CI=[{boot_sw_cv['ci_95_low']:+.4f},{boot_sw_cv['ci_95_high']:+.4f}] | "
              f"vs model3: d{auc_cv-auc_m3_cv:+.4f} p={boot_m3_cv['p_value']:.3f})")
        print(f"{'':9s}TEST: sw={auc_sw_test:.4f} model3={auc_m3_test:.4f} model4={auc_test:.4f}  "
              f"(vs sw: d{auc_test-auc_sw_test:+.4f} p={boot_sw_test['p_value']:.3f} | vs model3: d{auc_test-auc_m3_test:+.4f} p={boot_m3_test['p_value']:.3f})")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "Model_4_peak_aware_fusion_results.csv"), index=False)
    with open(os.path.join(OUT_DIR, "phase16_model4_summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved -> {OUT_DIR}/Model_4_peak_aware_fusion_results.csv")


if __name__ == "__main__":
    main()
