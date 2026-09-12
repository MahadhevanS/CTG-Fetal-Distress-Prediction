"""
Phase 16 -- Trainable Patient-Level Temporal Aggregation (docs/phase16_protocol.md).

Trains Model 2 (magnitude-only causal attention) and Model 3 (magnitude +
temporal position) over the frozen P6 window-level score sequence, per the
frozen protocol: patient-grouped 5-fold CV (canonical folds.json), inner
validation carved from training patients only for early stopping, causal
chronological-truncation training, evaluation at delivery/10/20/30m via
truncation to the causally-eligible prefix at each horizon.

Model 4 (peak-aware fusion) is gated on Model 2/3 clearing protocol
Section 8/9's criteria -- decided programmatically after Models 2/3 finish,
not assumed in advance.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected
from src.models.phase16_causal_attention import (
    train_scorer, predict_at_horizon_for_patients, predict_all_prefixes,
)
from src.models.phase16_checkpoint_utils import get_or_train
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
DATA_DIR = "data/processed_clinical"
OUT_DIR = "results/phase16"
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [0, 10, 20, 30]
FS_HZ = 4.0


def build_patient_data(pred_arr, patient_ids_arr, rolling_df, pids_list, y_lookup):
    data = {}
    t_del_by_pid = {}
    for pid in pids_list:
        idx = np.where(patient_ids_arr == str(pid))[0]
        sub = rolling_df.iloc[idx]
        t_del_vals = sub["time_before_delivery_min"].values
        order = np.argsort(-t_del_vals)  # ascending chronological = descending time_before_delivery
        idx_ordered = idx[order]

        r_seq = pred_arr[idx_ordered].astype(np.float32)
        t_del_seq = rolling_df.iloc[idx_ordered]["time_before_delivery_min"].values.astype(np.float64)
        start_sample_seq = rolling_df.iloc[idx_ordered]["start_sample"].values.astype(np.float64)
        elapsed_seq = (start_sample_seq / (FS_HZ * 60.0))
        elapsed_seq = (elapsed_seq - elapsed_seq[0]).astype(np.float32)

        T_i = len(idx_ordered)
        data[pid] = (
            torch.tensor(r_seq, dtype=torch.float32),
            torch.tensor(elapsed_seq, dtype=torch.float32),
            torch.tensor(float(y_lookup[pid]), dtype=torch.float32),
            T_i,
        )
        t_del_by_pid[pid] = t_del_seq
    return data, t_del_by_pid


def carve_inner_validation(train_pids, y_lookup, frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    pos = [p for p in train_pids if y_lookup[p] == 1]
    neg = [p for p in train_pids if y_lookup[p] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_val_pos = max(1, int(round(len(pos) * frac)))
    n_val_neg = max(1, int(round(len(neg) * frac)))
    val_pids = pos[:n_val_pos] + neg[:n_val_neg]
    inner_train_pids = pos[n_val_pos:] + neg[n_val_neg:]
    return inner_train_pids, val_pids


def run_model(model_name, use_elapsed, clean_pids, folds_blob, train_val_pids, test_pids,
              patient_data_cv, t_del_cv, patient_data_test, t_del_test, y_lookup):
    print(f"\n{'='*90}\n  {model_name}  (use_elapsed={use_elapsed})\n{'='*90}")

    cv_preds = {h: {} for h in HORIZONS}
    per_fold_auroc_h0 = []
    interp_agree = []
    interp_argmax_alpha, interp_argmax_r = [], []
    epochs_per_fold = []

    in_dim = 2 if use_elapsed else 1
    for f_idx in range(5):
        te_pids = [p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx]
        tr_pids_all = [p for p in clean_pids if p not in te_pids]
        inner_tr, inner_val = carve_inner_validation(tr_pids_all, y_lookup, seed=42 + f_idx)

        unit_id = f"{model_name}_fold{f_idx}"
        scorer, val_loss, n_epochs, resumed = get_or_train(
            CHECKPOINT_DIR, unit_id, in_dim, 8,
            lambda: train_scorer(inner_tr, inner_val, patient_data_cv, use_elapsed, seed=42),
        )
        epochs_per_fold.append(n_epochs)
        tag = "[resumed]" if resumed else "[trained]"
        print(f"  fold {f_idx} {tag}: {n_epochs} epochs, best inner-val loss={val_loss:.4f} "
              f"({len(inner_tr)} inner-train / {len(inner_val)} inner-val / {len(te_pids)} outer-held-out)")

        for h in HORIZONS:
            out = predict_at_horizon_for_patients(scorer, te_pids, patient_data_cv, t_del_cv, use_elapsed, h)
            for pid, (z, alpha, amax_a, amax_r) in out.items():
                cv_preds[h][pid] = z
                if h == 0:
                    interp_agree.append(1 if amax_a == amax_r else 0)
                    interp_argmax_alpha.append(amax_a)
                    interp_argmax_r.append(amax_r)

        te_y = np.array([y_lookup[p] for p in te_pids])
        te_z = np.array([cv_preds[0][p] for p in te_pids])
        if len(set(te_y.tolist())) > 1:
            per_fold_auroc_h0.append(float(roc_auc_score(te_y, te_z)))

    inner_tr_tv, inner_val_tv = carve_inner_validation(train_val_pids, y_lookup, seed=123)
    # NOTE: trained on patient_data_cv (contains all 547 patients via the CV-source
    # P6 scores), NOT patient_data_test (which only holds the 83 held-out test
    # patients' test-source scores) -- train_val_pids only exist in patient_data_cv.
    test_unit_id = f"{model_name}_testmodel"
    scorer_test, val_loss_test, n_epochs_test, resumed_test = get_or_train(
        CHECKPOINT_DIR, test_unit_id, in_dim, 8,
        lambda: train_scorer(inner_tr_tv, inner_val_tv, patient_data_cv, use_elapsed, seed=42),
    )
    tag = "[resumed]" if resumed_test else "[trained]"
    print(f"  test-model {tag}: {n_epochs_test} epochs, best inner-val loss={val_loss_test:.4f}")

    test_preds = {h: {} for h in HORIZONS}
    for h in HORIZONS:
        out = predict_at_horizon_for_patients(scorer_test, test_pids, patient_data_test, t_del_test, use_elapsed, h)
        for pid, (z, alpha, amax_a, amax_r) in out.items():
            test_preds[h][pid] = z

    interp_corr = np.nan
    if len(interp_argmax_alpha) > 3:
        interp_corr, _ = stats.spearmanr(interp_argmax_alpha, interp_argmax_r)

    return {
        "cv_preds": cv_preds, "test_preds": test_preds,
        "per_fold_auroc_h0": per_fold_auroc_h0,
        "interp_agree_frac": float(np.mean(interp_agree)) if interp_agree else np.nan,
        "interp_argmax_spearman": float(interp_corr) if not np.isnan(interp_corr) else None,
        "epochs_per_fold": epochs_per_fold, "epochs_test": n_epochs_test,
        "scorer_test": scorer_test,
    }


def compute_lead_time_for_model(scorer, patient_data_cv, t_del_cv, use_elapsed, clean_pids, y_lookup, target_sens=0.80):
    delivery_scores_pos = []
    all_delivery = {}
    for pid in clean_pids:
        r_seq, elapsed_seq, y, T_i = patient_data_cv[pid]
        zs = predict_all_prefixes(scorer, r_seq, elapsed_seq, use_elapsed, T_i)
        all_delivery[pid] = zs
        if y_lookup[pid] == 1:
            delivery_scores_pos.append(zs[-1])
    threshold = float(np.percentile(delivery_scores_pos, (1.0 - target_sens) * 100)) if delivery_scores_pos else 0.5

    lead_times, false_alerts = [], []
    for pid in clean_pids:
        zs = all_delivery[pid]
        t_del_seq = t_del_cv[pid]
        alert_idx = np.where(zs >= threshold)[0]
        if y_lookup[pid] == 1:
            lead_times.append(float(t_del_seq[alert_idx[0]]) if len(alert_idx) > 0 else 0.0)
        else:
            false_alerts.append(1 if len(alert_idx) > 0 else 0)
    lead_times = np.array(lead_times)
    return {
        "threshold": round(threshold, 4),
        "median_lead_min": round(float(np.median(lead_times)), 2) if len(lead_times) else 0.0,
        "pct_detected_ge20min": round(float(np.mean(lead_times >= 20)), 4) if len(lead_times) else 0.0,
        "pct_detected_ge30min": round(float(np.mean(lead_times >= 30)), 4) if len(lead_times) else 0.0,
        "false_alert_rate": round(float(np.mean(false_alerts)), 4) if false_alerts else 0.0,
    }


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

    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = [p for p in clean_pids if p not in test_pids]

    print("Building per-patient causal sequences...")
    patient_data_cv, t_del_cv = build_patient_data(pred_cv_window, patient_ids_arr, df, clean_pids, y_lookup)
    patient_data_test, t_del_test = build_patient_data(pred_test_window, patient_ids_arr, df, test_pids, y_lookup)
    print(f"  {len(patient_data_cv)} CV patients, {len(patient_data_test)} test patients")

    # ---- Model 1 baseline (reused, not retrained) ----
    y_pat = np.array([y_lookup[p] for p in clean_pids])
    y_test_pat = np.array([y_lookup[p] for p in test_pids])
    sw_cv_by_h = {h: get_patient_scores_at_horizon_corrected(pred_cv_window, patient_ids_arr, clean_pids,
                                                              df["time_before_delivery_min"].values, h) for h in HORIZONS}
    sw_test_by_h = {h: get_patient_scores_at_horizon_corrected(pred_test_window, patient_ids_arr, test_pids,
                                                                df["time_before_delivery_min"].values, h) for h in HORIZONS}

    results_summary = {}
    model_runs = {}
    for model_name, use_elapsed in [("Model_2_magnitude_only", False), ("Model_3_magnitude_position", True)]:
        run = run_model(model_name, use_elapsed, clean_pids, folds_blob, train_val_pids, test_pids,
                         patient_data_cv, t_del_cv, patient_data_test, t_del_test, y_lookup)
        model_runs[model_name] = run

        rows = []
        for h in HORIZONS:
            cv_z = np.array([run["cv_preds"][h][p] for p in clean_pids])
            test_z = np.array([run["test_preds"][h][p] for p in test_pids])
            sw_cv = sw_cv_by_h[h]
            sw_test = sw_test_by_h[h]

            auc_cv = roc_auc_score(y_pat, cv_z)
            auc_sw_cv = roc_auc_score(y_pat, sw_cv)
            auc_test = roc_auc_score(y_test_pat, test_z)
            auc_sw_test = roc_auc_score(y_test_pat, sw_test)
            auprc_cv = average_precision_score(y_pat, cv_z)

            boot_cv = paired_patient_bootstrap(y_pat, sw_cv, cv_z, n_boot=2000, seed=42)
            pdel_cv, _, _ = delong_roc_test(y_pat, cv_z, sw_cv)
            boot_test = paired_patient_bootstrap(y_test_pat, sw_test, test_z, n_boot=2000, seed=42)
            pdel_test, _, _ = delong_roc_test(y_test_pat, test_z, sw_test)

            row = {
                "horizon_min": h, "cv_auroc": round(auc_cv, 4), "cv_auroc_sw": round(auc_sw_cv, 4),
                "cv_delta": round(auc_cv - auc_sw_cv, 4), "cv_p": round(boot_cv["p_value"], 4),
                "cv_ci_low": round(boot_cv["ci_95_low"], 4), "cv_ci_high": round(boot_cv["ci_95_high"], 4),
                "cv_delong_p": round(pdel_cv, 4), "cv_auprc": round(auprc_cv, 4),
                "test_auroc": round(auc_test, 4), "test_auroc_sw": round(auc_sw_test, 4),
                "test_delta": round(auc_test - auc_sw_test, 4), "test_p": round(boot_test["p_value"], 4),
                "test_delong_p": round(pdel_test, 4),
            }
            rows.append(row)
            print(f"  h={h:>3}m  CV: sw={auc_sw_cv:.4f} model={auc_cv:.4f} (d{auc_cv-auc_sw_cv:+.4f} p={boot_cv['p_value']:.3f} "
                  f"CI=[{boot_cv['ci_95_low']:+.4f},{boot_cv['ci_95_high']:+.4f}])  "
                  f"TEST: sw={auc_sw_test:.4f} model={auc_test:.4f} (d{auc_test-auc_sw_test:+.4f} p={boot_test['p_value']:.3f})")

        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"{model_name}_auroc_results.csv"), index=False)

        lead_metrics = compute_lead_time_for_model(model_runs[model_name]["scorer_test"], patient_data_cv, t_del_cv,
                                                    use_elapsed, clean_pids, y_lookup)
        print(f"  per-fold AUROC@delivery: {[round(a,4) for a in run['per_fold_auroc_h0']]}  "
              f"(min={min(run['per_fold_auroc_h0']):.4f} max={max(run['per_fold_auroc_h0']):.4f} std={np.std(run['per_fold_auroc_h0']):.4f})")
        print(f"  interpretability: argmax(alpha)==argmax(r) in {run['interp_agree_frac']:.1%} of patients "
              f"(Spearman rho={run['interp_argmax_spearman']})")
        print(f"  operational (delivery, 80% target sens): {lead_metrics}")

        results_summary[model_name] = {
            "auroc_table": rows, "per_fold_auroc_h0": run["per_fold_auroc_h0"],
            "per_fold_auroc_std": float(np.std(run["per_fold_auroc_h0"])),
            "interp_agree_frac": run["interp_agree_frac"], "interp_argmax_spearman": run["interp_argmax_spearman"],
            "epochs_per_fold": run["epochs_per_fold"], "operational_delivery": lead_metrics,
        }

    with open(os.path.join(OUT_DIR, "phase16_models_2_3_summary.json"), "w") as fh:
        json.dump(results_summary, fh, indent=2)
    print(f"\nSaved -> {OUT_DIR}/phase16_models_2_3_summary.json")

    return results_summary, model_runs, {
        "clean_pids": clean_pids, "test_pids": test_pids, "y_lookup": y_lookup,
        "sw_cv_by_h": sw_cv_by_h, "sw_test_by_h": sw_test_by_h,
        "patient_data_cv": patient_data_cv, "patient_data_test": patient_data_test,
        "t_del_cv": t_del_cv, "t_del_test": t_del_test,
        "folds_blob": folds_blob, "train_val_pids": train_val_pids,
    }


if __name__ == "__main__":
    main()
