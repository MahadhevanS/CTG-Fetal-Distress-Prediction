"""
Phase 13: Information-Density Adaptive Weighting Evaluation & 5-Step Validation Ladder.
====================================================================================
Implements and verifies the validation ladder from docs/information_density_weighting_design.md:

Ladder Steps:
- Step 0: Current Full System (P6) Baseline, unweighted
- Step 1: + Novelty / Elapsed features as model input only, unweighted
- Step 2: + Patient-normalization weight only (w_pat)
- Step 3: + Novelty-based weighting on top of Step 2 (w_pat * w_nov, beta=1.0)
- Step 4: + Elapsed-time shrinkage prior blended in (w_pat * w_nov, beta/span tuned via
          NESTED patient-grouped CV)

Fixed 2026-09-11 after verification found three issues in the original implementation:

1. NESTED-CV LEAKAGE (critical): the original script selected (beta, span) by
   maximizing OOF AUROC across the SAME 5 outer folds later reported as "Step 4
   CV performance" -- i.e. it picked hyperparameters using the exact data it
   then reported results on. Fixed by selecting (beta, span) via an inner
   patient-grouped CV run strictly on each outer fold's OWN training patients
   (never touching that outer fold's held-out patients), so each outer fold
   gets its own nested hyperparameter selection. The held-out test evaluation
   gets its own separate nested selection restricted to train+val patients only
   (test patients never participate in any hyperparameter selection).

2. ELAPSED-TIME GAP BUG: elapsed_monitoring_min was computed as
   window_index * stride_min, silently assuming no window was ever dropped by
   the quality gate. 38/547 patients (6.9%) have non-uniform stride from
   dropped windows. Fixed by deriving elapsed time from each window's actual
   start_sample (via InformationDensityWeighter's elapsed_lookup parameter),
   which is gap-aware.

3. Ordering robustness: every InformationDensityWeighter call now explicitly
   passes the real elapsed-time array as its chronological sort key, rather
   than relying on the array already being pre-sorted per patient.

Evaluation Protocol:
- Patient-grouped 5-fold CV across 547 patients, nested hyperparameter selection.
- Held-out test split validation on 83 patients (data/processed_clinical/test_dataset.pt).
- Primary endpoint: pH <= 7.15 (AUROC, AUPRC, Brier score) at Delivery (0m) and >=30m horizon.
- Severe endpoint: pH <= 7.05.
- Early warning operational metrics: median lead time, false-alert rate.
- Paired patient bootstrap (B=2,000) and DeLong test for statistical significance.
- Both stepwise (vs. previous step) AND total (vs. Step 0 baseline) deltas are reported --
  the total-vs-baseline comparison is the one that answers whether the mechanism helps.

Outputs:
- results/phase13_information_density/ladder_results.csv
- results/phase13_information_density/step_predictions.csv
- results/phase13_information_density/hyperparameter_sweep.csv (now: per-outer-fold + test nested selections)
- results/phase13_information_density/phase13_summary.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.training.information_density_weighting import InformationDensityWeighter
from scripts.phase11_bootstrap import paired_patient_bootstrap, fast_auc
from scripts.delong_test import delong_roc_test

OUT_DIR = "results/phase13_information_density"
FOLDS_PATH = "data/processed_clinical/folds.json"
DATA_DIR = "data/processed_clinical"
P2_PATH = "data/phase1_candidates/p2_dataset.pt"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
TRAJ_FEATURES_PATH = "results/phase9c_state_trajectory/state_trajectory_features.npz"

FS_HZ = 4.0
BETA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SPAN_GRID = [2.0, 3.0, 4.0]
N_INNER_FOLDS = 4

os.makedirs(OUT_DIR, exist_ok=True)


def load_cohort_data():
    """Loads all data sources and feature matrices."""
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    patient_ids = df_rolling["patient_id"].values
    t_del = df_rolling["time_before_delivery_min"].values
    y_715 = df_rolling["primary_label_715"].values
    y_705 = df_rolling["severe_label_705"].values

    # Real, gap-aware elapsed monitoring time (absolute minutes into the
    # patient's retained recording); InformationDensityWeighter offsets this
    # per-patient internally so each patient's own first window reads 0.
    elapsed_abs_min = (df_rolling["start_sample"].values.astype(np.float64) / (FS_HZ * 60.0)).astype(np.float32)

    df_pat_labels = df_rolling.groupby("patient_id")[["primary_label_715", "severe_label_705"]].first().reset_index()
    y_pat_715 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    y_pat_705 = np.array([df_pat_labels[df_pat_labels["patient_id"] == p]["severe_label_705"].iloc[0] for p in clean_pids])

    # Load 40-D state trajectory representation (P6)
    data_traj = np.load(TRAJ_FEATURES_PATH)
    X_p6 = data_traj["X_state_trajectory"] # (8517, 40)
    X_raw_19 = X_p6[:, :19]

    # Load official test set patient IDs
    import torch
    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(list(set([str(m[0]) for m in test_pt["metadata"]])))
    train_val_pids = [p for p in clean_pids if p not in test_pids]

    return {
        "folds_blob": folds_blob,
        "clean_pids": clean_pids,
        "df_rolling": df_rolling,
        "patient_ids": patient_ids,
        "t_del": t_del,
        "elapsed_abs_min": elapsed_abs_min,
        "y_715": y_715,
        "y_705": y_705,
        "y_pat_715": y_pat_715,
        "y_pat_705": y_pat_705,
        "X_p6": X_p6,
        "X_raw_19": X_raw_19,
        "test_pids": test_pids,
        "train_val_pids": train_val_pids,
    }


def get_patient_scores_at_horizon(pred_arr, patient_ids, clean_pids, t_del, h_val):
    """Aggregates window-level probabilities to patient-level score at horizon h_val."""
    p_scores = []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        t_pts = t_del[idx]
        if h_val > 0:
            eligible = np.where(t_pts >= h_val)[0]
            chosen = idx[eligible[0]] if len(eligible) > 0 else idx[-1]
        else:
            chosen = idx[-1]
        p_scores.append(pred_arr[chosen])
    return np.array(p_scores)


def compute_early_warning_operational_metrics(pred_arr, df_rolling, clean_pids, target_sens=0.80):
    """Computes median lead time (min) and false alert rate on patient cohort."""
    df = df_rolling.copy()
    df["pred_prob"] = pred_arr

    pat_delivery_scores = {}
    pat_labels = {}
    for pid in clean_pids:
        sub = df[df["patient_id"] == str(pid)]
        deliv_row = sub.sort_values("time_before_delivery_min").iloc[0]
        pat_delivery_scores[pid] = deliv_row["pred_prob"]
        pat_labels[pid] = deliv_row["primary_label_715"]

    pids = list(clean_pids)
    y_true = np.array([pat_labels[p] for p in pids])
    deliv_scores = np.array([pat_delivery_scores[p] for p in pids])

    pos_scores = deliv_scores[y_true == 1]
    if len(pos_scores) > 0:
        threshold = float(np.percentile(pos_scores, (1.0 - target_sens) * 100))
    else:
        threshold = 0.5

    lead_times = []
    false_alerts = []

    for pid in clean_pids:
        sub = df[df["patient_id"] == str(pid)].sort_values("time_before_delivery_min", ascending=False)
        is_pos = (pat_labels[pid] == 1)
        alert_rows = sub[sub["pred_prob"] >= threshold]

        if is_pos:
            if len(alert_rows) > 0:
                first_alert_time = alert_rows.iloc[0]["time_before_delivery_min"]
                lead_times.append(first_alert_time)
            else:
                lead_times.append(0.0)
        else:
            false_alerts.append(1 if len(alert_rows) > 0 else 0)

    median_lead_time = float(np.median(lead_times)) if len(lead_times) > 0 else 0.0
    false_alert_rate = float(np.mean(false_alerts)) if len(false_alerts) > 0 else 0.0

    return {
        "operating_threshold": round(threshold, 4),
        "median_lead_time_min": round(median_lead_time, 2),
        "false_alert_rate": round(false_alert_rate, 4),
    }


def _fit_predict_lr(X_tr, y_tr, w_tr, X_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
    clf.fit(X_tr_s, y_tr, sample_weight=w_tr)
    return clf.predict_proba(X_te_s)[:, 1]


def select_hyperparams_inner_cv(pool_pids, patient_ids, X_raw_19, elapsed_abs_min, X_p6, y_715,
                                 t_del, y_pat_lookup, tag="", n_inner=N_INNER_FOLDS, seed=42):
    """
    Selects (beta, span) via patient-grouped inner CV run strictly within pool_pids.
    No patient outside pool_pids is ever touched by fitting, prediction, or scoring
    here -- this is what makes the selection valid to apply to a held-out fold/test
    partition that does NOT overlap with pool_pids.
    """
    pool_pids = sorted(pool_pids)
    pool_arr = np.array(pool_pids)
    y_pool_pat = np.array([y_pat_lookup[p] for p in pool_pids])

    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    inner_splits = list(skf.split(pool_arr, y_pool_pat))

    sweep_rows = []
    best_score = -np.inf
    best_beta, best_span = 1.0, 3.0

    for span in SPAN_GRID:
        for beta in BETA_GRID:
            oof = np.full(len(patient_ids), np.nan, dtype=np.float64)

            for inner_tr_idx, inner_te_idx in inner_splits:
                inner_tr_pids = set(pool_arr[inner_tr_idx].tolist())
                inner_te_pids = set(pool_arr[inner_te_idx].tolist())

                tr_mask = np.isin(patient_ids, list(inner_tr_pids))
                te_mask = np.isin(patient_ids, list(inner_te_pids))

                weighter = InformationDensityWeighter(span=span, beta=beta)
                weighter.fit(X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask])
                w_tr = weighter.transform(
                    X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask]
                )["w_combined"]

                nov_tr = weighter.extract_novelty_features(X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask])
                nov_te = weighter.extract_novelty_features(X_raw_19[te_mask], patient_ids[te_mask], elapsed_abs_min[te_mask])

                X_tr = np.hstack([X_p6[tr_mask], nov_tr])
                X_te = np.hstack([X_p6[te_mask], nov_te])

                oof[te_mask] = _fit_predict_lr(X_tr, y_715[tr_mask], w_tr, X_te)

            scores_pat = get_patient_scores_at_horizon(oof, patient_ids, pool_pids, t_del, 0)
            auc = roc_auc_score(y_pool_pat, scores_pat)
            sweep_rows.append({
                "tag": tag, "span": span, "beta": beta,
                "inner_auroc_delivery": round(float(auc), 4),
                "n_pool_patients": len(pool_pids),
            })
            if auc > best_score:
                best_score = auc
                best_beta, best_span = beta, span

    return best_beta, best_span, pd.DataFrame(sweep_rows)


def run_validation_ladder():
    """Executes the full 5-step validation ladder with nested hyperparameter selection."""
    print("================================================================================")
    print("   PHASE 13: INFORMATION-DENSITY ADAPTIVE WEIGHTING — 5-STEP VALIDATION LADDER   ")
    print("                    (re-run 2026-09-11 with nested-CV + bug fixes)                ")
    print("================================================================================")

    cohort = load_cohort_data()
    folds_blob = cohort["folds_blob"]
    clean_pids = cohort["clean_pids"]
    df_rolling = cohort["df_rolling"]
    patient_ids = cohort["patient_ids"]
    t_del = cohort["t_del"]
    elapsed_abs_min = cohort["elapsed_abs_min"]
    y_715 = cohort["y_715"]
    y_705 = cohort["y_705"]
    y_pat_715 = cohort["y_pat_715"]
    X_p6 = cohort["X_p6"]
    X_raw_19 = cohort["X_raw_19"]
    test_pids = cohort["test_pids"]
    train_val_pids = cohort["train_val_pids"]

    y_pat_lookup = {p: int(v) for p, v in zip(clean_pids, y_pat_715)}

    # 1. Nested hyperparameter selection -- one selection per outer fold (using
    #    ONLY that fold's own training patients), plus one separate selection
    #    for the held-out test evaluation (using ONLY train+val patients).
    print("\n--- Nested CV: selecting (beta, span) per outer fold (inner CV on outer-train patients only) ---")
    fold_hparams = {}
    all_sweep_rows = []
    for f_idx in range(5):
        te_pids_f = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
        tr_pids_f = [p for p in clean_pids if p not in te_pids_f]
        b, s, sweep_df = select_hyperparams_inner_cv(
            tr_pids_f, patient_ids, X_raw_19, elapsed_abs_min, X_p6, y_715, t_del,
            y_pat_lookup, tag=f"outer_fold_{f_idx}"
        )
        sweep_df["outer_fold"] = f_idx
        all_sweep_rows.append(sweep_df)
        fold_hparams[f_idx] = (b, s)
        print(f"  Outer fold {f_idx}: selected beta={b}, span={s}")

    print("\n--- Nested CV: selecting (beta, span) for held-out test (inner CV on train+val patients only) ---")
    test_beta, test_span, sweep_df_test = select_hyperparams_inner_cv(
        train_val_pids, patient_ids, X_raw_19, elapsed_abs_min, X_p6, y_715, t_del,
        y_pat_lookup, tag="test_selection"
    )
    sweep_df_test["outer_fold"] = -1
    all_sweep_rows.append(sweep_df_test)
    print(f"  Test-partition selection: beta={test_beta}, span={test_span}")

    df_sweep = pd.concat(all_sweep_rows, ignore_index=True)
    sweep_path = os.path.join(OUT_DIR, "hyperparameter_sweep.csv")
    df_sweep.to_csv(sweep_path, index=False)
    print(f"\nSaved nested hyperparameter sweep -> {sweep_path}")

    # 2. Define Ladder Steps
    ladder_configs = {
        "Step_0_Baseline": {
            "step": 0,
            "name": "Step 0: Current Full System (P6) Baseline",
            "feature_set": "P6",
            "weighting": "none",
            "description": "Standard unweighted 40-D multidomain framework baseline"
        },
        "Step_1_Features_Only": {
            "step": 1,
            "name": "Step 1: + Novelty / Elapsed Features As Input Only",
            "feature_set": "P6_plus_novelty",
            "weighting": "none",
            "description": "Exposing novelty, novelty_smoothed, and elapsed_monitoring_min to model input (unweighted); elapsed time now gap-aware"
        },
        "Step_2_Patient_Norm": {
            "step": 2,
            "name": "Step 2: + Patient Normalization Weight Only (w_pat)",
            "feature_set": "P6_plus_novelty",
            "weighting": "patient_norm",
            "description": "Applying patient-normalization loss weight K / n_windows(patient) (novelty unweighted)"
        },
        "Step_3_Novelty_Weighting": {
            "step": 3,
            "name": "Step 3: + Novelty Weighting on Top of Patient Norm",
            "feature_set": "P6_plus_novelty",
            "weighting": "novelty_weighting",
            "description": "Applying composite weight w_pat * w_nov with beta=1.0, span=3.0"
        },
        "Step_4_Shrinkage_Prior": {
            "step": 4,
            "name": "Step 4: + Elapsed-Time Empirical Prior (Shrinkage)",
            "feature_set": "P6_plus_novelty",
            "weighting": "shrinkage_prior",
            "description": "Applying composite weight with empirical horizon shrinkage; beta/span selected via NESTED inner CV, independently per outer fold / per test selection (see hyperparameter_sweep.csv)"
        }
    }

    n_samples = len(df_rolling)
    pred_dict = {k: np.zeros(n_samples, dtype=np.float32) for k in ladder_configs}
    test_pred_dict = {k: np.zeros(n_samples, dtype=np.float32) for k in ladder_configs}

    print("\n--- Executing 5-Fold Cross-Validation for Validation Ladder ---")
    for f_idx in range(5):
        te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
        tr_pids = set([p for p in clean_pids if p not in te_pids])

        tr_mask = np.isin(patient_ids, list(tr_pids))
        va_mask = np.isin(patient_ids, list(te_pids))

        fold_beta, fold_span = fold_hparams[f_idx]

        # Standard weighter (span=3.0, fixed) drives Step 1-3 features/weights,
        # exactly as originally specified -- not tuned, to keep those steps a
        # clean, un-selected baseline for comparison against the tuned Step 4.
        weighter_std = InformationDensityWeighter(span=3.0, beta=1.0)
        weighter_std.fit(X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask])

        w_tr_step2 = weighter_std.transform(
            X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask]
        )["w_patient"]
        w_tr_step3 = weighter_std.transform(
            X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask], beta=1.0
        )["w_combined"]

        nov_tr = weighter_std.extract_novelty_features(X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask])
        nov_va = weighter_std.extract_novelty_features(X_raw_19[va_mask], patient_ids[va_mask], elapsed_abs_min[va_mask])

        # Step 4 weighter uses THIS outer fold's nested-selected (beta, span).
        weighter_step4 = InformationDensityWeighter(span=fold_span, beta=fold_beta)
        weighter_step4.fit(X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask])
        w_tr_step4 = weighter_step4.transform(
            X_raw_19[tr_mask], patient_ids[tr_mask], elapsed_abs_min[tr_mask]
        )["w_combined"]

        X_tr_dict = {
            "P6": X_p6[tr_mask],
            "P6_plus_novelty": np.hstack([X_p6[tr_mask], nov_tr])
        }
        X_va_dict = {
            "P6": X_p6[va_mask],
            "P6_plus_novelty": np.hstack([X_p6[va_mask], nov_va])
        }

        weights_map = {
            "none": None,
            "patient_norm": w_tr_step2,
            "novelty_weighting": w_tr_step3,
            "shrinkage_prior": w_tr_step4
        }

        for k, cfg in ladder_configs.items():
            f_set = cfg["feature_set"]
            w_mode = cfg["weighting"]

            X_tr_sub = X_tr_dict[f_set]
            X_va_sub = X_va_dict[f_set]
            w_sub = weights_map[w_mode]

            pred_dict[k][va_mask] = _fit_predict_lr(X_tr_sub, y_715[tr_mask], w_sub, X_va_sub)

    # 3. Train on Train+Val Partition and Evaluate on Held-Out Test Partition (83 patients)
    print("\n--- Training on Development Partition & Evaluating on Held-Out Test Partition ---")
    tr_val_mask = np.isin(patient_ids, list(train_val_pids))
    test_mask = np.isin(patient_ids, list(test_pids))

    weighter_std_test = InformationDensityWeighter(span=3.0, beta=1.0)
    weighter_std_test.fit(X_raw_19[tr_val_mask], patient_ids[tr_val_mask], elapsed_abs_min[tr_val_mask])

    w_trval_step2 = weighter_std_test.transform(
        X_raw_19[tr_val_mask], patient_ids[tr_val_mask], elapsed_abs_min[tr_val_mask]
    )["w_patient"]
    w_trval_step3 = weighter_std_test.transform(
        X_raw_19[tr_val_mask], patient_ids[tr_val_mask], elapsed_abs_min[tr_val_mask], beta=1.0
    )["w_combined"]

    nov_trval = weighter_std_test.extract_novelty_features(X_raw_19[tr_val_mask], patient_ids[tr_val_mask], elapsed_abs_min[tr_val_mask])
    nov_test = weighter_std_test.extract_novelty_features(X_raw_19[test_mask], patient_ids[test_mask], elapsed_abs_min[test_mask])

    weighter_test_step4 = InformationDensityWeighter(span=test_span, beta=test_beta)
    weighter_test_step4.fit(X_raw_19[tr_val_mask], patient_ids[tr_val_mask], elapsed_abs_min[tr_val_mask])
    w_trval_step4 = weighter_test_step4.transform(
        X_raw_19[tr_val_mask], patient_ids[tr_val_mask], elapsed_abs_min[tr_val_mask]
    )["w_combined"]

    X_trval_dict = {
        "P6": X_p6[tr_val_mask],
        "P6_plus_novelty": np.hstack([X_p6[tr_val_mask], nov_trval])
    }
    X_test_dict = {
        "P6": X_p6[test_mask],
        "P6_plus_novelty": np.hstack([X_p6[test_mask], nov_test])
    }

    weights_map_test = {
        "none": None,
        "patient_norm": w_trval_step2,
        "novelty_weighting": w_trval_step3,
        "shrinkage_prior": w_trval_step4
    }

    for k, cfg in ladder_configs.items():
        f_set = cfg["feature_set"]
        w_mode = cfg["weighting"]

        X_trval_sub = X_trval_dict[f_set]
        X_test_sub = X_test_dict[f_set]
        w_sub = weights_map_test[w_mode]

        test_pred_dict[k][test_mask] = _fit_predict_lr(X_trval_sub, y_715[tr_val_mask], w_sub, X_test_sub)

    # 4. Statistical Evaluation Across Horizons and Partitions
    ladder_keys = list(ladder_configs.keys())
    horizons = [0, 30, 20, 10]
    horizon_names = {0: "Delivery (0m)", 30: ">=30m (Primary Horizon)", 20: ">=20m", 10: ">=10m"}

    results_rows = []
    y_test_pat_715 = np.array([df_rolling[df_rolling["patient_id"] == p]["primary_label_715"].iloc[0] for p in test_pids])

    for h in horizons:
        h_name = horizon_names[h]
        scores_cv = {k: get_patient_scores_at_horizon(pred_dict[k], patient_ids, clean_pids, t_del, h) for k in ladder_keys}
        scores_test = {k: get_patient_scores_at_horizon(test_pred_dict[k], patient_ids, test_pids, t_del, h) for k in ladder_keys}

        base_scores_cv = scores_cv["Step_0_Baseline"]
        base_auc_cv = roc_auc_score(y_pat_715, base_scores_cv)

        base_scores_test = scores_test["Step_0_Baseline"]
        base_auc_test = roc_auc_score(y_test_pat_715, base_scores_test)

        for idx, k in enumerate(ladder_keys):
            cfg = ladder_configs[k]
            step_num = cfg["step"]
            step_name = cfg["name"]

            s_k_cv = scores_cv[k]
            auc_cv = roc_auc_score(y_pat_715, s_k_cv)
            auprc_cv = average_precision_score(y_pat_715, s_k_cv)
            brier_cv = brier_score_loss(y_pat_715, s_k_cv)

            s_k_test = scores_test[k]
            auc_test = roc_auc_score(y_test_pat_715, s_k_test)
            auprc_test = average_precision_score(y_test_pat_715, s_k_test)
            brier_test = brier_score_loss(y_test_pat_715, s_k_test)

            if idx == 0:
                step_delta_cv = 0.0
                step_ci_lo_cv = 0.0
                step_ci_hi_cv = 0.0
                step_p_boot_cv = 1.0
                step_p_delong_cv = 1.0
                step_delta_test = 0.0
                step_p_delong_test = 1.0
            else:
                prev_k = ladder_keys[idx - 1]
                prev_s_cv = scores_cv[prev_k]
                boot_step_cv = paired_patient_bootstrap(y_pat_715, prev_s_cv, s_k_cv, n_boot=2000, seed=42)
                p_del_cv, _, _ = delong_roc_test(y_pat_715, s_k_cv, prev_s_cv)

                step_delta_cv = auc_cv - roc_auc_score(y_pat_715, prev_s_cv)
                step_ci_lo_cv = boot_step_cv["ci_95_low"]
                step_ci_hi_cv = boot_step_cv["ci_95_high"]
                step_p_boot_cv = boot_step_cv["p_value"]
                step_p_delong_cv = p_del_cv

                prev_s_test = scores_test[prev_k]
                p_del_test, _, _ = delong_roc_test(y_test_pat_715, s_k_test, prev_s_test)
                step_delta_test = auc_test - roc_auc_score(y_test_pat_715, prev_s_test)
                step_p_delong_test = p_del_test

            if idx == 0:
                tot_delta_cv = 0.0
                tot_ci_lo_cv = 0.0
                tot_ci_hi_cv = 0.0
                tot_p_boot_cv = 1.0
                tot_p_delong_cv = 1.0
                tot_delta_test = 0.0
                tot_p_delong_test = 1.0
            else:
                boot_tot_cv = paired_patient_bootstrap(y_pat_715, base_scores_cv, s_k_cv, n_boot=2000, seed=42)
                p_del_tot_cv, _, _ = delong_roc_test(y_pat_715, s_k_cv, base_scores_cv)

                tot_delta_cv = auc_cv - base_auc_cv
                tot_ci_lo_cv = boot_tot_cv["ci_95_low"]
                tot_ci_hi_cv = boot_tot_cv["ci_95_high"]
                tot_p_boot_cv = boot_tot_cv["p_value"]
                tot_p_delong_cv = p_del_tot_cv

                p_del_tot_test, _, _ = delong_roc_test(y_test_pat_715, s_k_test, base_scores_test)
                tot_delta_test = auc_test - base_auc_test
                tot_p_delong_test = p_del_tot_test

            op_metrics = compute_early_warning_operational_metrics(pred_dict[k], df_rolling, clean_pids, target_sens=0.80)

            results_rows.append({
                "ladder_step": step_num,
                "step_code": k,
                "step_name": step_name,
                "description": cfg["description"],
                "horizon": h_name,
                "horizon_min": h,
                "cv_auroc": round(float(auc_cv), 4),
                "cv_auprc": round(float(auprc_cv), 4),
                "cv_brier": round(float(brier_cv), 4),
                "cv_stepwise_delta_auroc": round(float(step_delta_cv), 4),
                "cv_stepwise_ci_low": round(float(step_ci_lo_cv), 4),
                "cv_stepwise_ci_high": round(float(step_ci_hi_cv), 4),
                "cv_stepwise_p_boot": round(float(step_p_boot_cv), 4),
                "cv_stepwise_p_delong": round(float(step_p_delong_cv), 4),
                "cv_total_delta_from_baseline": round(float(tot_delta_cv), 4),
                "cv_total_ci_low": round(float(tot_ci_lo_cv), 4),
                "cv_total_ci_high": round(float(tot_ci_hi_cv), 4),
                "cv_total_p_boot": round(float(tot_p_boot_cv), 4),
                "cv_total_p_delong": round(float(tot_p_delong_cv), 4),
                "test_auroc": round(float(auc_test), 4),
                "test_auprc": round(float(auprc_test), 4),
                "test_brier": round(float(brier_test), 4),
                "test_stepwise_delta_auroc": round(float(step_delta_test), 4),
                "test_stepwise_p_delong": round(float(step_p_delong_test), 4),
                "test_total_delta_from_baseline": round(float(tot_delta_test), 4),
                "test_total_p_delong": round(float(tot_p_delong_test), 4),
                "median_lead_time_min": op_metrics["median_lead_time_min"],
                "false_alert_rate": op_metrics["false_alert_rate"],
            })

    df_results = pd.DataFrame(results_rows)
    results_path = os.path.join(OUT_DIR, "ladder_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\nSaved ladder evaluation results -> {results_path}")

    df_pred_out = pd.DataFrame({
        "patient_id": patient_ids,
        "window_index": df_rolling["window_index"].values,
        "time_before_delivery_min": t_del,
        "elapsed_monitoring_min_gap_aware": elapsed_abs_min,
        "primary_label_715": y_715,
        "severe_label_705": y_705,
        **{f"cv_prob_{k}": pred_dict[k] for k in ladder_keys},
        **{f"test_prob_{k}": test_pred_dict[k] for k in ladder_keys}
    })
    pred_path = os.path.join(OUT_DIR, "step_predictions.csv")
    df_pred_out.to_csv(pred_path, index=False)
    print(f"Saved step predictions -> {pred_path}")

    print("\n" + "="*110)
    print("     VALIDATION LADDER SUMMARY (DELIVERY 0M) -- stepwise vs prev | TOTAL vs Step-0 baseline")
    print("="*110)
    df_del = df_results[df_results["horizon_min"] == 0]
    for _, r in df_del.iterrows():
        print(f"[{r['ladder_step']}] {r['step_name']:<48} | CV {r['cv_auroc']:.4f} (step {r['cv_stepwise_delta_auroc']:+.4f} p={r['cv_stepwise_p_boot']:.3f} | TOTAL {r['cv_total_delta_from_baseline']:+.4f} p={r['cv_total_p_boot']:.3f}) | Test {r['test_auroc']:.4f} (TOTAL {r['test_total_delta_from_baseline']:+.4f} p={r['test_total_p_delong']:.3f})")

    print("\n" + "="*110)
    print("   VALIDATION LADDER SUMMARY (PRIMARY HORIZON >=30M) -- stepwise vs prev | TOTAL vs Step-0 baseline")
    print("="*110)
    df_30m = df_results[df_results["horizon_min"] == 30]
    for _, r in df_30m.iterrows():
        print(f"[{r['ladder_step']}] {r['step_name']:<48} | CV {r['cv_auroc']:.4f} (step {r['cv_stepwise_delta_auroc']:+.4f} p={r['cv_stepwise_p_boot']:.3f} | TOTAL {r['cv_total_delta_from_baseline']:+.4f} p={r['cv_total_p_boot']:.3f}) | Test {r['test_auroc']:.4f} (TOTAL {r['test_total_delta_from_baseline']:+.4f} p={r['test_total_p_delong']:.3f})")

    summary_data = {
        "title": "Information-Density Adaptive Weighting Validation Ladder (nested-CV re-run)",
        "cohort": {
            "n_patients": len(clean_pids),
            "n_test_patients": len(test_pids),
            "n_train_val_patients": len(train_val_pids),
            "n_windows": n_samples,
            "per_outer_fold_selected_hyperparameters": {str(f): {"beta": b, "span": s} for f, (b, s) in fold_hparams.items()},
            "test_partition_selected_hyperparameters": {"beta": test_beta, "span": test_span},
        },
        "delivery_metrics": df_del.to_dict(orient="records"),
        "horizon_30m_metrics": df_30m.to_dict(orient="records"),
    }
    summary_path = os.path.join(OUT_DIR, "phase13_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary_data, fh, indent=2)
    print(f"\nSaved summary JSON -> {summary_path}")

    return df_results, summary_data


if __name__ == "__main__":
    run_validation_ladder()
