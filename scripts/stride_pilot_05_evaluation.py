"""
Stride experiment pilot -- Stage E (final comparison).

Compares the 1.0-minute-stride P6 state-trajectory system against the frozen
2.5-minute-stride baseline, on the SAME 547 patients and SAME 5-fold
patient-grouped CV assignment (data/processed_clinical/folds.json, untouched),
so the only thing differing between arms is window-extraction density.

Both arms get the SAME evaluation treatment:
  - per-patient weight normalization (K / n_windows(patient)) -- the
    structural fix established as a prerequisite for any stride comparison,
    since a finer stride mechanically produces more (correlated) windows per
    patient and would otherwise bias the comparison exactly like the bag-size
    leak audit_bag_size_leak.py exists to catch.
  - identical model: LogisticRegression(C=0.05) on the 40-D P6 feature set.
  - identical statistical protocol: paired patient bootstrap (B=2000) + DeLong.

Also reports "horizon-alignment slop" -- the mean absolute gap between the
target horizon (e.g. 30 min before delivery) and the actual
time_before_delivery of the window get_patient_scores_at_horizon() picks for
each patient. A finer stride mechanically reduces this gap; any AUROC
difference needs to be read alongside it so a real information gain isn't
confused with simply landing closer to the requested horizon.
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

from src.training.information_density_weighting import InformationDensityWeighter
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = os.path.join(BASE_DIR, "data", "processed_clinical", "folds.json")
OUT_DIR = os.path.join(BASE_DIR, "results", "stride_pilot")
os.makedirs(OUT_DIR, exist_ok=True)

ARMS = {
    "2.5min_baseline": {
        "rolling_path": os.path.join(BASE_DIR, "results", "phase8_rolling", "rolling_predictions.csv"),
        "traj_path": os.path.join(BASE_DIR, "results", "phase9c_state_trajectory", "state_trajectory_features.npz"),
    },
    "1.0min_pilot": {
        "rolling_path": os.path.join(BASE_DIR, "results", "phase8_rolling_stride1p0", "rolling_predictions.csv"),
        "traj_path": os.path.join(BASE_DIR, "results", "phase9c_state_trajectory_stride1p0", "state_trajectory_features.npz"),
    },
}


def get_patient_scores_at_horizon(pred_arr, patient_ids, clean_pids, t_del, h_val):
    p_scores, slop = [], []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        t_pts = t_del[idx]
        if h_val > 0:
            eligible = np.where(t_pts >= h_val)[0]
            if len(eligible) > 0:
                chosen = idx[eligible[0]]
                slop.append(abs(float(t_del[chosen]) - h_val))
            else:
                chosen = idx[-1]
                slop.append(np.nan)
        else:
            chosen = idx[-1]
            slop.append(abs(float(t_del[chosen])))
        p_scores.append(pred_arr[chosen])
    return np.array(p_scores), np.array(slop)


def load_arm(rolling_path, traj_path, elapsed_col_needed=True):
    df = pd.read_csv(rolling_path)
    df["patient_id"] = df["patient_id"].astype(str)
    data = np.load(traj_path)
    X_p6 = data["X_state_trajectory"]  # (N, 40)
    return {
        "df": df,
        "patient_ids": df["patient_id"].values,
        "t_del": df["time_before_delivery_min"].values,
        "y_715": df["primary_label_715"].values,
        "X_p6": X_p6,
        "X_raw_19": X_p6[:, :19],
        "elapsed_abs_min": (df["start_sample"].values.astype(np.float64) / (4.0 * 60.0)).astype(np.float32),
    }


def evaluate_arm(arm_name, arm_data, clean_pids, folds_blob, horizons):
    patient_ids = arm_data["patient_ids"]
    t_del = arm_data["t_del"]
    y_715 = arm_data["y_715"]
    X_p6 = arm_data["X_p6"]
    X_raw_19 = arm_data["X_raw_19"]
    elapsed_abs_min = arm_data["elapsed_abs_min"]

    y_pat_715 = np.array([
        arm_data["df"][arm_data["df"]["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids
    ])

    n = len(arm_data["df"])
    pred_cv = np.zeros(n, dtype=np.float32)

    windows_per_patient = pd.Series(patient_ids).value_counts()
    print(f"  [{arm_name}] {n} windows, {len(clean_pids)} patients, "
          f"windows/patient median={windows_per_patient.median():.0f} "
          f"(min={windows_per_patient.min()}, max={windows_per_patient.max()})")

    for f_idx in range(5):
        te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
        tr_pids = set(p for p in clean_pids if p not in te_pids)
        tr_mask = np.isin(patient_ids, list(tr_pids))
        va_mask = np.isin(patient_ids, list(te_pids))

        weighter = InformationDensityWeighter(span=3.0, beta=1.0)
        weighter.fit(X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask])
        w_tr = weighter.transform(
            X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask]
        )["w_patient"]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_p6[tr_mask])
        X_va_s = scaler.transform(X_p6[va_mask])

        clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
        clf.fit(X_tr_s, y_715[tr_mask], sample_weight=w_tr)
        pred_cv[va_mask] = clf.predict_proba(X_va_s)[:, 1]

    rows = []
    scores_by_h = {}
    for h in horizons:
        scores, slop = get_patient_scores_at_horizon(pred_cv, patient_ids, clean_pids, t_del, h)
        scores_by_h[h] = scores
        auc = roc_auc_score(y_pat_715, scores)
        auprc = average_precision_score(y_pat_715, scores)
        mean_slop = float(np.nanmean(slop))
        rows.append({
            "arm": arm_name, "horizon_min": h, "auroc": round(float(auc), 4),
            "auprc": round(float(auprc), 4), "mean_horizon_slop_min": round(mean_slop, 3),
            "n_windows": n, "windows_per_patient_median": float(windows_per_patient.median()),
        })
        print(f"    horizon={h:>3}m  AUROC={auc:.4f}  AUPRC={auprc:.4f}  mean_slop={mean_slop:.3f}min")

    return rows, scores_by_h, y_pat_715


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    horizons = [0, 10, 20, 30]

    print("================================================================================")
    print("  STRIDE PILOT -- STAGE E: 1.0-min vs 2.5-min stride, patient-normalized P6 model  ")
    print("================================================================================")

    all_rows = []
    scores_all = {}
    y_pat_ref = None
    for arm_name, paths in ARMS.items():
        print(f"\n--- Evaluating arm: {arm_name} ---")
        arm_data = load_arm(paths["rolling_path"], paths["traj_path"])
        rows, scores_by_h, y_pat = evaluate_arm(arm_name, arm_data, clean_pids, folds_blob, horizons)
        all_rows.extend(rows)
        scores_all[arm_name] = scores_by_h
        y_pat_ref = y_pat  # identical patient set/order across arms (both built from clean_pids)

    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(os.path.join(OUT_DIR, "stride_comparison_results.csv"), index=False)

    print("\n" + "=" * 90)
    print("  STATISTICAL COMPARISON: 1.0min pilot vs 2.5min baseline (paired, same 547 patients)")
    print("=" * 90)
    stat_rows = []
    for h in horizons:
        base_scores = scores_all["2.5min_baseline"][h]
        pilot_scores = scores_all["1.0min_pilot"][h]
        boot = paired_patient_bootstrap(y_pat_ref, base_scores, pilot_scores, n_boot=2000, seed=42)
        p_delong, auc_pilot, auc_base = delong_roc_test(y_pat_ref, pilot_scores, base_scores)
        print(f"  horizon={h:>3}m  baseline={auc_base:.4f}  pilot={auc_pilot:.4f}  "
              f"delta={auc_pilot - auc_base:+.4f}  boot_p={boot['p_value']:.3f}  delong_p={p_delong:.4f}  "
              f"95%CI=[{boot['ci_95_low']:+.4f}, {boot['ci_95_high']:+.4f}]")
        stat_rows.append({
            "horizon_min": h, "auroc_baseline_2p5min": round(auc_base, 4),
            "auroc_pilot_1p0min": round(auc_pilot, 4), "delta_auroc": round(auc_pilot - auc_base, 4),
            "boot_p_value": round(boot["p_value"], 4), "boot_ci_low": round(boot["ci_95_low"], 4),
            "boot_ci_high": round(boot["ci_95_high"], 4), "delong_p_value": round(p_delong, 4),
        })

    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(os.path.join(OUT_DIR, "stride_comparison_statistics.csv"), index=False)

    with open(os.path.join(OUT_DIR, "stride_pilot_summary.json"), "w") as fh:
        json.dump({
            "results": df_results.to_dict(orient="records"),
            "statistics": df_stats.to_dict(orient="records"),
        }, fh, indent=2)

    print(f"\nSaved -> {OUT_DIR}/stride_comparison_results.csv")
    print(f"Saved -> {OUT_DIR}/stride_comparison_statistics.csv")
    print(f"Saved -> {OUT_DIR}/stride_pilot_summary.json")


if __name__ == "__main__":
    main()
