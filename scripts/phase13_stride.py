"""
Phase 13.2 (docs/phase13_protocol.md Section 8, Option B) -- Stride,
re-evaluated under both horizon conventions.

Reuses the 1.0-min and 2.5-min pipeline outputs already generated this
session (results/phase8_rolling{,_stride1p0}/, results/phase9c_state_trajectory{,_stride1p0}/)
-- no pipeline regeneration needed, only re-scoring under the corrected
horizon selector alongside the existing one, plus the operational metrics
(warning time, >=20/30-min detection, FAR) the protocol calls for as a
possible route to survivor status even without an AUROC gain.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import get_patient_scores_at_horizon, get_patient_scores_at_horizon_corrected
from src.training.information_density_weighting import InformationDensityWeighter
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
OUT_DIR = "results/phase13/stride"
os.makedirs(OUT_DIR, exist_ok=True)

ARMS = {
    "2.5min_baseline": {
        "rolling_path": "results/phase8_rolling/rolling_predictions.csv",
        "traj_path": "results/phase9c_state_trajectory/state_trajectory_features.npz",
    },
    "1.0min_pilot": {
        "rolling_path": "results/phase8_rolling_stride1p0/rolling_predictions.csv",
        "traj_path": "results/phase9c_state_trajectory_stride1p0/state_trajectory_features.npz",
    },
}
HORIZONS = [0, 10, 20, 30]
CONVENTIONS = {"existing": get_patient_scores_at_horizon, "corrected": get_patient_scores_at_horizon_corrected}


def compute_operational_metrics(pred_arr, df, clean_pids, target_sens=0.80):
    df = df.copy()
    df["pred_prob"] = pred_arr
    pat_delivery, pat_labels = {}, {}
    for pid in clean_pids:
        sub = df[df["patient_id"] == str(pid)]
        row = sub.sort_values("time_before_delivery_min").iloc[0]
        pat_delivery[pid] = row["pred_prob"]
        pat_labels[pid] = row["primary_label_715"]
    y_true = np.array([pat_labels[p] for p in clean_pids])
    scores = np.array([pat_delivery[p] for p in clean_pids])
    pos = scores[y_true == 1]
    threshold = float(np.percentile(pos, (1.0 - target_sens) * 100)) if len(pos) > 0 else 0.5

    lead_times, false_alerts = [], []
    for pid in clean_pids:
        sub = df[df["patient_id"] == str(pid)].sort_values("time_before_delivery_min", ascending=False)
        alerts = sub[sub["pred_prob"] >= threshold]
        if pat_labels[pid] == 1:
            lead_times.append(alerts.iloc[0]["time_before_delivery_min"] if len(alerts) > 0 else 0.0)
        else:
            false_alerts.append(1 if len(alerts) > 0 else 0)
    detect_20 = float(np.mean(np.array(lead_times) >= 20)) if lead_times else 0.0
    detect_30 = float(np.mean(np.array(lead_times) >= 30)) if lead_times else 0.0
    return {
        "threshold": round(threshold, 4), "median_lead_min": round(float(np.median(lead_times)), 2) if lead_times else 0.0,
        "pct_detected_ge20min": round(detect_20, 4), "pct_detected_ge30min": round(detect_30, 4),
        "false_alert_rate": round(float(np.mean(false_alerts)), 4) if false_alerts else 0.0,
    }


def evaluate_arm(name, rolling_path, traj_path, clean_pids, folds_blob):
    df = pd.read_csv(rolling_path)
    df["patient_id"] = df["patient_id"].astype(str)
    patient_ids = df["patient_id"].values
    t_del = df["time_before_delivery_min"].values
    y_715 = df["primary_label_715"].values
    elapsed_abs_min = (df["start_sample"].values.astype(np.float64) / (4.0 * 60.0)).astype(np.float32)
    X_p6 = np.load(traj_path)["X_state_trajectory"]
    X_raw_19 = X_p6[:, :19]

    y_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    n = len(df)
    pred_cv = np.zeros(n, dtype=np.float32)

    for f_idx in range(5):
        te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
        tr_mask = np.isin(patient_ids, list(p for p in clean_pids if p not in te_pids))
        va_mask = np.isin(patient_ids, list(te_pids))
        weighter = InformationDensityWeighter(span=3.0, beta=1.0)
        weighter.fit(X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask])
        w_tr = weighter.transform(X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask])["w_patient"]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_p6[tr_mask])
        X_va_s = scaler.transform(X_p6[va_mask])
        clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
        clf.fit(X_tr_s, y_715[tr_mask], sample_weight=w_tr)
        pred_cv[va_mask] = clf.predict_proba(X_va_s)[:, 1]

    return df, patient_ids, t_del, y_pat, pred_cv


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    arm_data = {}
    print("================================================================================")
    print("  PHASE 13.2 -- STRIDE, RE-EVALUATED UNDER BOTH HORIZON CONVENTIONS               ")
    print("================================================================================")
    for name, paths in ARMS.items():
        print(f"\n--- {name} ---")
        df, pids, t_del, y_pat, pred_cv = evaluate_arm(name, paths["rolling_path"], paths["traj_path"], clean_pids, folds_blob)
        arm_data[name] = {"df": df, "patient_ids": pids, "t_del": t_del, "y_pat": y_pat, "pred_cv": pred_cv}

    rows = []
    scores_by_conv_h = {"existing": {}, "corrected": {}}
    for conv_name, fn in CONVENTIONS.items():
        for h in HORIZONS:
            scores = {}
            for name, d in arm_data.items():
                s = fn(d["pred_cv"], d["patient_ids"], clean_pids, d["t_del"], h)
                scores[name] = s
            scores_by_conv_h[conv_name][h] = scores
            base = scores["2.5min_baseline"]
            pilot = scores["1.0min_pilot"]
            y_pat = arm_data["2.5min_baseline"]["y_pat"]
            auc_base = roc_auc_score(y_pat, base)
            auc_pilot = roc_auc_score(y_pat, pilot)
            boot = paired_patient_bootstrap(y_pat, base, pilot, n_boot=2000, seed=42)
            p_delong, _, _ = delong_roc_test(y_pat, pilot, base)
            rows.append({
                "convention": conv_name, "horizon_min": h,
                "auroc_2p5min": round(auc_base, 4), "auroc_1p0min": round(auc_pilot, 4),
                "delta": round(auc_pilot - auc_base, 4), "boot_p": round(boot["p_value"], 4),
                "delong_p": round(p_delong, 4), "ci_low": round(boot["ci_95_low"], 4), "ci_high": round(boot["ci_95_high"], 4),
            })
            print(f"[{conv_name:9s} h={h:>3}m] 2.5min={auc_base:.4f}  1.0min={auc_pilot:.4f}  "
                  f"d={auc_pilot-auc_base:+.4f}  p={boot['p_value']:.3f}")

    df_results = pd.DataFrame(rows)
    df_results.to_csv(os.path.join(OUT_DIR, "stride_corrected_horizon_results.csv"), index=False)

    print("\n--- Operational metrics (existing convention, both arms) ---")
    op_rows = []
    for name, d in arm_data.items():
        op = compute_operational_metrics(d["pred_cv"], d["df"], clean_pids)
        op["arm"] = name
        op_rows.append(op)
        print(f"  {name}: median_lead={op['median_lead_min']}min  >=20min={op['pct_detected_ge20min']:.1%}  "
              f">=30min={op['pct_detected_ge30min']:.1%}  FAR={op['false_alert_rate']:.1%}")
    pd.DataFrame(op_rows).to_csv(os.path.join(OUT_DIR, "stride_operational_metrics.csv"), index=False)

    print(f"\nSaved -> {OUT_DIR}/stride_corrected_horizon_results.csv")
    print(f"Saved -> {OUT_DIR}/stride_operational_metrics.csv")


if __name__ == "__main__":
    main()
