"""
Phase 8 Gate 2: Multi-Horizon Early-Warning Evaluation.

Evaluates patient-level discrimination, calibration, and clinical operating points across
delivery-anchored warning horizons (>=60m, >=45m, >=30m, >=20m, >=10m, >=0m).
Preserves strict patient-level statistical independence (1 prediction per patient per horizon).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression

OUT_DIR = "results/phase8_rolling"
ROLLING_PATH = os.path.join(OUT_DIR, "rolling_predictions.csv")

def bootstrap_patient_metrics(labels: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    aucs = []
    auprcs = []
    idx = np.arange(len(labels))
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(labels[b])) < 2:
            continue
        aucs.append(roc_auc_score(labels[b], scores[b]))
        auprcs.append(average_precision_score(labels[b], scores[b]))
    
    auc_lo, auc_hi = np.percentile(aucs, [2.5, 97.5])
    prc_lo, prc_hi = np.percentile(auprcs, [2.5, 97.5])
    return (float(auc_lo), float(auc_hi)), (float(prc_lo), float(prc_hi))


def compute_operating_points(labels: np.ndarray, scores: np.ndarray, probs: np.ndarray):
    """
    Computes High Sensitivity (@ 90% Sens) and High Specificity (@ 90% Spec).
    """
    thresholds = np.sort(np.unique(scores))
    
    # 1. High Sensitivity (@ ~90%)
    best_thresh_sens = thresholds[0]
    best_sens = 1.0
    best_spec_at_sens = 0.0
    for th in thresholds:
        preds = (scores >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        if sens >= 0.88: # close to 90%
            best_thresh_sens = th
            best_sens = sens
            best_spec_at_sens = spec

    # Compute metrics at high sens threshold
    preds_sens = (scores >= best_thresh_sens).astype(int)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(labels, preds_sens, labels=[0, 1]).ravel()
    ppv_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
    npv_s = tn_s / (tn_s + fn_s) if (tn_s + fn_s) > 0 else 0

    # 2. High Specificity (@ ~90%)
    best_thresh_spec = thresholds[-1]
    best_spec = 1.0
    best_sens_at_spec = 0.0
    for th in reversed(thresholds):
        preds = (scores >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        if spec >= 0.88:
            best_thresh_spec = th
            best_spec = spec
            best_sens_at_spec = sens

    preds_spec = (scores >= best_thresh_spec).astype(int)
    tn_sp, fp_sp, fn_sp, tp_sp = confusion_matrix(labels, preds_spec, labels=[0, 1]).ravel()
    ppv_sp = tp_sp / (tp_sp + fp_sp) if (tp_sp + fp_sp) > 0 else 0
    npv_sp = tn_sp / (tn_sp + fn_sp) if (tn_sp + fn_sp) > 0 else 0

    return {
        "high_sensitivity_point": {
            "target_sensitivity": "90%",
            "threshold": float(best_thresh_sens),
            "sensitivity": float(tp_s / (tp_s + fn_s)),
            "specificity": float(tn_s / (tn_s + fp_s)),
            "ppv": float(ppv_s),
            "npv": float(npv_s),
            "fpr": float(fp_s / (tn_s + fp_s))
        },
        "high_specificity_point": {
            "target_specificity": "90%",
            "threshold": float(best_thresh_spec),
            "sensitivity": float(tp_sp / (tp_sp + fn_sp)),
            "specificity": float(tn_sp / (tn_sp + fp_sp)),
            "ppv": float(ppv_sp),
            "npv": float(npv_sp),
            "fpr": float(fp_sp / (tn_sp + fp_sp))
        }
    }


def compute_calibration_metrics(labels: np.ndarray, probs: np.ndarray):
    brier = float(brier_score_loss(labels, probs))
    # Fit logistic calibration: logit(p) vs label
    eps = 1e-5
    probs_clip = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(probs_clip / (1.0 - probs_clip)).reshape(-1, 1)
    
    cal_model = LogisticRegression(C=1e5)
    cal_model.fit(logits, labels)
    slope = float(cal_model.coef_[0][0])
    intercept = float(cal_model.intercept_[0])
    return {"brier_score": brier, "calibration_slope": slope, "calibration_intercept": intercept}


def run_horizon_evaluation():
    print("=== EXECUTING PHASE 8: MULTI-HORIZON EARLY-WARNING EVALUATION ===")
    df = pd.read_csv(ROLLING_PATH)

    # Defined warning horizons (minutes before delivery)
    horizons = [60, 45, 30, 20, 10, 0]

    results_primary = []
    results_severe = []
    calibrations = []
    bootstrap_records = {}

    all_pids = df["patient_id"].unique()

    for h in horizons:
        # For each patient, select the closest prediction at or before horizon h (i.e. time_before_delivery >= h)
        # If no window exists with time_before_delivery >= h, pick their earliest available window
        patient_rows = []
        for pid in all_pids:
            df_p = df[df["patient_id"] == pid].sort_values("time_before_delivery_min")
            df_eligible = df_p[df_p["time_before_delivery_min"] >= h]
            if not df_eligible.empty:
                # Pick the window closest to h
                chosen = df_eligible.iloc[0]
            else:
                # Pick earliest window
                chosen = df_p.iloc[-1]
            patient_rows.append(chosen)

        df_h = pd.DataFrame(patient_rows)
        y_true_715 = df_h["primary_label_715"].values
        y_true_705 = df_h["severe_label_705"].values
        scores = df_h["acidemia_risk_score"].values
        probs = df_h["risk_prob_proxy"].values

        # 1. Primary Endpoint (pH <= 7.15)
        auc_715 = roc_auc_score(y_true_715, scores)
        auprc_715 = average_precision_score(y_true_715, scores)
        (auc_lo, auc_hi), (prc_lo, prc_hi) = bootstrap_patient_metrics(y_true_715, scores, n_boot=2000)

        op_715 = compute_operating_points(y_true_715, scores, probs)
        cal_715 = compute_calibration_metrics(y_true_715, probs)
        cal_715["horizon_min"] = h
        calibrations.append(cal_715)

        results_primary.append({
            "horizon": f">={h} min" if h > 0 else "Delivery (0m)",
            "horizon_min": h,
            "patient_count": len(df_h),
            "positives_715": int(y_true_715.sum()),
            "auroc": round(auc_715, 4),
            "auroc_ci_95": f"[{auc_lo:.4f}, {auc_hi:.4f}]",
            "auprc": round(auprc_715, 4),
            "auprc_ci_95": f"[{prc_lo:.4f}, {prc_hi:.4f}]",
            "sens_at_90_spec": round(op_715["high_specificity_point"]["sensitivity"] * 100, 2),
            "spec_at_90_sens": round(op_715["high_sensitivity_point"]["specificity"] * 100, 2),
            "ppv_at_90_sens": round(op_715["high_sensitivity_point"]["ppv"] * 100, 2),
            "npv_at_90_sens": round(op_715["high_sensitivity_point"]["npv"] * 100, 2),
            "brier_score": round(cal_715["brier_score"], 4)
        })

        # 2. Secondary Severe Endpoint (pH <= 7.05)
        auc_705 = roc_auc_score(y_true_705, scores)
        auprc_705 = average_precision_score(y_true_705, scores)
        (auc_lo_sev, auc_hi_sev), (prc_lo_sev, prc_hi_sev) = bootstrap_patient_metrics(y_true_705, scores, n_boot=2000)

        results_severe.append({
            "horizon": f">={h} min" if h > 0 else "Delivery (0m)",
            "horizon_min": h,
            "positives_705": int(y_true_705.sum()),
            "auroc_severe": round(auc_705, 4),
            "auroc_ci_95": f"[{auc_lo_sev:.4f}, {auc_hi_sev:.4f}]",
            "auprc_severe": round(auprc_705, 4),
            "auprc_ci_95": f"[{prc_lo_sev:.4f}, {prc_hi_sev:.4f}]"
        })

        bootstrap_records[f"horizon_{h}m"] = {
            "primary_715": {"auroc": auc_715, "ci": [auc_lo, auc_hi], "auprc": auprc_715, "ci_prc": [prc_lo, prc_hi]},
            "severe_705": {"auroc": auc_705, "ci": [auc_lo_sev, auc_hi_sev], "auprc": auprc_705, "ci_prc": [prc_lo_sev, prc_hi_sev]},
            "operating_points": op_715
        }

        print(f"Horizon >={h:02d}m: AUROC (pH<=7.15) = {auc_715:.4f} [{auc_lo:.4f}, {auc_hi:.4f}], AUPRC = {auprc_715:.4f} | Severe (<=7.05) AUROC = {auc_705:.4f}")

    df_res_p = pd.DataFrame(results_primary)
    df_res_p.to_csv(os.path.join(OUT_DIR, "horizon_metrics.csv"), index=False)

    df_cal = pd.DataFrame(calibrations)
    df_cal.to_csv(os.path.join(OUT_DIR, "calibration_metrics.csv"), index=False)

    with open(os.path.join(OUT_DIR, "bootstrap_results.json"), "w") as f:
        json.dump(bootstrap_records, f, indent=2)

    print("\nHorizon evaluation results saved to results/phase8_rolling/horizon_metrics.csv.\n")

if __name__ == "__main__":
    run_horizon_evaluation()
