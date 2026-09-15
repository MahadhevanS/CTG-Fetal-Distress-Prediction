"""
Phase 18 Stage 1g -- freeze the deployable A3 fusion mechanism and compute
operational metrics for the deployable candidate family.

Items 1-4 of the user-directed next-stage plan:
  1. Document/freeze the deployable A3 implementation (this script produces
     the frozen artifact + numbers; docs/phase18_deployable_fusion_protocol.md
     is the written spec built from its output).
  2. Verify the exact selected alpha and its fold-to-fold stability (across
     all 30 independent fold-fits already produced: 5 canonical +
     5 resplits x 5 folds), plus a FINAL alpha fit on all 547 patients
     pooled (no held-out fold) -- the actual value to deploy, matching this
     project's established "final artifact" convention
     (scripts/prepare_external_validation_artifacts.py fits P6-final,
     Model3-final, Parity-final the same way).
  3. Operational metrics (per-fold training-only threshold selection at 80%
     target sensitivity, patient-level "ever alerted" counting -- identical
     convention to every prior operational evaluation in this project,
     e.g. scripts/model3_parity_hybrid/hybrid_engine.py) for all 5
     deployable systems being compared.
  4. The 5-way comparison table: M3-only, F2-deployable, A3(train-selected
     alpha), A3(fixed alpha=0.75), A4(sum-to-one logit fusion) -- all
     evaluated as ONE continuous deployable score sequence per patient,
     never told the horizon, exactly matching the deployable discipline
     established in Stage 1e/1f.

Re-fits the canonical 5 folds' models fresh here (deterministic,
random_state=42 fixed throughout) rather than reloading saved coefficient
CSVs, to guarantee the operational-metric models are provably identical to
the already-verified CV/test models from
scripts/phase18_deployable_fusion.py -- confirmed by the sanity check
below before anything else is computed from them.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import to_logit, from_logit
from src.models.phase16_causal_attention import predict_all_prefixes, eligible_prefix_length
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
DATA_DIR = "data/processed_clinical"
CHECKPOINT_DIR = "results/phase16/checkpoints"
OUT_DIR = "results/phase18_fusion_ablation"

HORIZONS = [0, 10, 20, 30]
ALPHA_GRID = np.linspace(0.0, 1.0, 21)
TARGET_SENS = 0.80
SYSTEMS = ["M3_only", "F2_deployable", "A3_selected", "A3_fixed_0.75", "A4_sum_to_one"]


def fit_parity_lr(parity_train, y_train, parity_apply):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(parity_train.reshape(-1, 1))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    X_ap = scaler.transform(parity_apply.reshape(-1, 1))
    return clf.predict_proba(X_ap)[:, 1]


def build_pooled_training_set(train_pids, m3_seq_by_pid, parity_prob_by_pid, y_lookup):
    rows_m3, rows_par, rows_y, rows_w = [], [], [], []
    for pid in train_pids:
        z_seq = m3_seq_by_pid[pid]
        T_i = len(z_seq)
        y = y_lookup[pid]
        par = parity_prob_by_pid[pid]
        for z_k in z_seq:
            rows_m3.append(z_k); rows_par.append(par); rows_y.append(y); rows_w.append(1.0 / T_i)
    return (np.array(rows_m3), np.array(rows_par), np.array(rows_y), np.array(rows_w))


def fit_deployable_bundle(tr_pids, m3_seq_by_pid, parity_prob_by_pid, y_lookup):
    """Fits A3-selected-alpha, F2-deployable, A4-sum-to-one on the pooled
    (all-truncations) training set. Returns a dict of callables, each
    mapping (z_m3, p_parity) -> fused probability, plus the raw fitted
    parameters for reporting."""
    zm3_pool, par_pool, y_pool, w_pool = build_pooled_training_set(tr_pids, m3_seq_by_pid, parity_prob_by_pid, y_lookup)
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

    def score_fn(system, z_k, par_p):
        logit_m3_k = to_logit(np.array([z_k]))[0]; logit_par_p = to_logit(np.array([par_p]))[0]
        if system == "M3_only":
            return float(z_k)
        elif system == "A3_selected":
            return float(best_a * z_k + (1 - best_a) * par_p)
        elif system == "A3_fixed_0.75":
            return float(0.75 * z_k + 0.25 * par_p)
        elif system == "F2_deployable":
            x2 = scaler2.transform([[logit_m3_k, logit_par_p]])
            return float(clf2.predict_proba(x2)[0, 1])
        elif system == "A4_sum_to_one":
            s_val = best_g1 * logit_m3_k + (1 - best_g1) * logit_par_p
            return float(from_logit(g0 + scale * s_val))
        else:
            raise ValueError(system)

    params = {"alpha_selected": float(best_a), "F2_coef_m3": float(clf2.coef_[0][0]), "F2_coef_parity": float(clf2.coef_[0][1]),
              "F2_intercept": float(clf2.intercept_[0]), "A4_g1": float(best_g1), "A4_g0": g0, "A4_scale": scale}
    return score_fn, params


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    n_patients = len(clean_pids)

    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    y_lookup = {p: int(df_rolling[df_rolling["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    y_pat = np.array([y_lookup[p] for p in clean_pids])

    meta = pd.read_csv(METADATA_PATH).set_index("record_id")
    parity_by_pid = {str(p): float(meta.loc[int(p), "parity"]) for p in clean_pids}

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids_arr = p6["patient_ids"]
    pred_cv_window = p6["pred_unweighted_cv"]
    patient_data_cv, t_del_cv = build_patient_data(pred_cv_window, patient_ids_arr, df_rolling, clean_pids, y_lookup)

    completed = load_completed_units(CHECKPOINT_DIR)
    fold_of = {p: folds_blob["assignment"][p][0] for p in clean_pids}
    fold_scorers = {f: load_scorer_checkpoint(completed[f"Model_3_magnitude_position_fold{f}"], in_dim=2, hidden=8) for f in range(5)}

    print("================================================================================")
    print("  PHASE 18 -- FREEZE DEPLOYABLE FUSION + OPERATIONAL METRICS                       ")
    print("================================================================================")

    m3_seq_cv = {}
    for pid in clean_pids:
        r_seq, elapsed_seq, y, T_i = patient_data_cv[pid]
        m3_seq_cv[pid] = predict_all_prefixes(fold_scorers[fold_of[pid]], r_seq, elapsed_seq, True, T_i)

    # ================================================================
    # ITEM 2 -- alpha stability (full picture) + FINAL frozen fit (all 547 pooled)
    # ================================================================
    print("\n--- Item 2: alpha stability across all fold-fits, and the FINAL frozen fit ---")
    canon_params = pd.read_csv(os.path.join(OUT_DIR, "deployable_fusion_fold_params.csv"))
    resplit_params = pd.read_csv(os.path.join(OUT_DIR, "deployable_fusion_resplit_params.csv"))
    a_canon = canon_params[canon_params["model"] == "A3_selected_once"]["alpha"].astype(float)
    a_resplit = resplit_params[(resplit_params["split"] != "CANONICAL")]["alpha"].astype(float)
    all_alpha = pd.concat([a_canon, a_resplit])
    print(f"  30 independent fold-fits (5 canonical + 5 resplits x 5 folds):")
    print(f"  mean={all_alpha.mean():.4f}  std={all_alpha.std():.4f}  median={all_alpha.median():.4f}  "
          f"min={all_alpha.min():.4f}  max={all_alpha.max():.4f}")

    # parity OOF for the canonical folds (needed for canonical per-fold model fitting below)
    p_parity_oof = np.zeros(n_patients)
    for f_idx in range(5):
        te_pids = [p for p in clean_pids if fold_of[p] == f_idx]
        tr_pids = [p for p in clean_pids if fold_of[p] != f_idx]
        p_par = fit_parity_lr(np.array([parity_by_pid[p] for p in tr_pids]), np.array([y_lookup[p] for p in tr_pids]),
                               np.array([parity_by_pid[p] for p in clean_pids]))
        te_mask = np.array([fold_of[p] == f_idx for p in clean_pids])
        p_parity_oof[te_mask] = p_par[te_mask]
    parity_prob_by_pid = {p: p_parity_oof[i] for i, p in enumerate(clean_pids)}

    # FINAL frozen fit: parity model on ALL 547, fusion models on ALL 547 pooled (no held-out fold)
    p_parity_final = fit_parity_lr(np.array([parity_by_pid[p] for p in clean_pids]), y_pat,
                                    np.array([parity_by_pid[p] for p in clean_pids]))
    parity_prob_final_by_pid = {p: p_parity_final[i] for i, p in enumerate(clean_pids)}
    score_fn_final, params_final = fit_deployable_bundle(clean_pids, m3_seq_cv, parity_prob_final_by_pid, y_lookup)
    print(f"\n  FINAL frozen fit (all 547 patients, no held-out fold):")
    print(f"    alpha (A3-selected)  = {params_final['alpha_selected']:.4f}")
    print(f"    F2 coef_m3={params_final['F2_coef_m3']:.4f}  coef_parity={params_final['F2_coef_parity']:.4f}  intercept={params_final['F2_intercept']:.4f}")
    print(f"    A4 g1={params_final['A4_g1']:.4f}  g0={params_final['A4_g0']:.4f}  scale={params_final['A4_scale']:.4f}")

    with open(os.path.join(OUT_DIR, "deployable_fusion_FROZEN_params.json"), "w") as fh:
        json.dump({
            "alpha_stability": {"n_fold_fits": int(len(all_alpha)), "mean": float(all_alpha.mean()), "std": float(all_alpha.std()),
                                 "median": float(all_alpha.median()), "min": float(all_alpha.min()), "max": float(all_alpha.max()),
                                 "all_values": [float(v) for v in all_alpha]},
            "final_frozen_fit_all_547_patients": params_final,
            "mechanism": "p_fused = alpha * p_M3(t) + (1-alpha) * p_parity, alpha selected once via sample-weighted pooled training AUROC across every causal truncation",
        }, fh, indent=2)
    print(f"\nSaved -> {OUT_DIR}/deployable_fusion_FROZEN_params.json")

    # ================================================================
    # Sanity check -- refit canonical folds fresh here, confirm they match deployable_fusion_results.csv
    # ================================================================
    print("\n--- Sanity check: canonical per-fold refit here reproduces earlier CV numbers ---")
    canon_score_fns = {}
    for f_idx in range(5):
        te_pids = [p for p in clean_pids if fold_of[p] == f_idx]
        tr_pids = [p for p in clean_pids if fold_of[p] != f_idx]
        score_fn, params = fit_deployable_bundle(tr_pids, m3_seq_cv, parity_prob_by_pid, y_lookup)
        canon_score_fns[f_idx] = score_fn

    m3_only_cv = {h: np.zeros(n_patients) for h in HORIZONS}
    preds_cv = {sysname: {h: np.zeros(n_patients) for h in HORIZONS} for sysname in SYSTEMS}
    for i, pid in enumerate(clean_pids):
        _, _, _, T_i = patient_data_cv[pid]
        f_idx = fold_of[pid]
        par_p = parity_prob_by_pid[pid]
        for h in HORIZONS:
            k = eligible_prefix_length(t_del_cv[pid], h, T_i)
            z_k = m3_seq_cv[pid][k - 1]
            m3_only_cv[h][i] = z_k
            for sysname in SYSTEMS:
                preds_cv[sysname][h][i] = canon_score_fns[f_idx](sysname, z_k, par_p)

    ref = pd.read_csv(os.path.join(OUT_DIR, "deployable_fusion_results.csv"))
    h_name_map = {0: "Delivery", 10: ">=10m", 20: ">=20m", 30: ">=30m"}
    name_map = {"A3_selected": "A3_selected_once", "A3_fixed_0.75": "A3_fixed_a0.75", "F2_deployable": "F2_deployable", "A4_sum_to_one": "A4_sum_to_one_deployable"}
    mism = 0
    for h in HORIZONS:
        for sysname, refname in name_map.items():
            c = roc_auc_score(y_pat, preds_cv[sysname][h])
            r_row = ref[(ref["horizon"] == h_name_map[h]) & (ref["model"] == refname)]
            if len(r_row):
                r = r_row["cv_auroc"].iloc[0]
                if abs(c - r) > 1e-3:
                    print(f"  MISMATCH {sysname} h={h}: here={c:.4f} vs earlier={r:.4f}")
                    mism += 1
    print(f"  {'ALL MATCH' if mism == 0 else f'{mism} MISMATCHES'} ({len(HORIZONS)*len(name_map)} checks)")
    if mism > 0:
        print("  ABORTING -- do not trust operational metrics built on non-reproducing models.")
        return

    # ================================================================
    # ITEM 3 -- operational metrics (per-fold, training-only threshold, patient-level "ever alerted")
    # ================================================================
    print(f"\n--- Item 3: operational metrics ({TARGET_SENS:.0%} target sensitivity, patient-level ever-alerted) ---")
    op_rows = []
    for sysname in SYSTEMS:
        fold_thresholds = []
        alert_lead_times = []
        patient_alerted = np.zeros(n_patients, dtype=bool)

        for f_idx in range(5):
            te_pids = [p for p in clean_pids if fold_of[p] == f_idx]
            tr_pids = [p for p in clean_pids if fold_of[p] != f_idx]
            score_fn = canon_score_fns[f_idx]

            # threshold selected on TRAINING positive patients' delivery-horizon score
            tr_scores_delivery = []
            for pid in tr_pids:
                _, _, _, T_i = patient_data_cv[pid]
                z_k = m3_seq_cv[pid][T_i - 1]  # delivery = full prefix
                par_p = parity_prob_by_pid[pid]
                tr_scores_delivery.append(score_fn(sysname, z_k, par_p))
            tr_scores_delivery = np.array(tr_scores_delivery)
            tr_y = np.array([y_lookup[p] for p in tr_pids])
            pos_tr_scores = tr_scores_delivery[tr_y == 1]
            thresh_f = float(np.percentile(pos_tr_scores, (1 - TARGET_SENS) * 100))
            fold_thresholds.append(thresh_f)

            for pid in te_pids:
                i = clean_pids.index(pid)
                z_seq = m3_seq_cv[pid]; T_i = len(z_seq)
                par_p = parity_prob_by_pid[pid]
                seq_scores = np.array([score_fn(sysname, z_seq[k], par_p) for k in range(T_i)])
                alert_idx = np.where(seq_scores >= thresh_f)[0]
                if len(alert_idx) > 0:
                    patient_alerted[i] = True
                    if y_lookup[pid] == 1:
                        alert_lead_times.append(float(t_del_cv[pid][alert_idx[0]]))

        lead_arr = np.array(alert_lead_times)
        tp_alerted = int(np.sum(patient_alerted[y_pat == 1]))
        fp_alerted = int(np.sum(patient_alerted[y_pat == 0]))
        total_pos = int(np.sum(y_pat == 1)); total_neg = int(np.sum(y_pat == 0))
        far = fp_alerted / total_neg; sens = tp_alerted / total_pos
        row = {
            "system": sysname, "target_sensitivity": TARGET_SENS,
            "mean_threshold": round(float(np.mean(fold_thresholds)), 4),
            "achieved_sensitivity": round(sens, 4), "true_positives_alerted": tp_alerted, "false_positives_alerted": fp_alerted,
            "false_alert_rate": round(far, 4),
            "median_lead_time_min": round(float(np.median(lead_arr)), 2) if len(lead_arr) else 0.0,
            "iqr_lead_time_min": round(float(np.percentile(lead_arr, 75) - np.percentile(lead_arr, 25)), 2) if len(lead_arr) else 0.0,
            "pct_detected_ge10min": round(float(np.mean(lead_arr >= 10)), 4) if len(lead_arr) else 0.0,
            "pct_detected_ge20min": round(float(np.mean(lead_arr >= 20)), 4) if len(lead_arr) else 0.0,
            "pct_detected_ge30min": round(float(np.mean(lead_arr >= 30)), 4) if len(lead_arr) else 0.0,
        }
        op_rows.append(row)
        print(f"  {sysname:<16}: Sens={sens:.1%}, FAR={far:.1%}, MedianLead={row['median_lead_time_min']}m, "
              f"ge20m={row['pct_detected_ge20min']:.1%}, ge30m={row['pct_detected_ge30min']:.1%}")

    pd.DataFrame(op_rows).to_csv(os.path.join(OUT_DIR, "deployable_fusion_operational_metrics.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/deployable_fusion_operational_metrics.csv")

    # ================================================================
    # ITEM 4 -- 5-way comparison table (CV/test from existing files + operational from above)
    # ================================================================
    print("\n--- Item 4: 5-way comparison table ---")
    cv_test_ref = pd.read_csv(os.path.join(OUT_DIR, "deployable_fusion_results.csv"))
    stability_ref = pd.read_csv(os.path.join(OUT_DIR, "deployable_fusion_stability_ranking.csv"))
    op_ref = pd.DataFrame(op_rows).set_index("system")

    combined_rows = []
    for h in HORIZONS:
        for sysname, refname in name_map.items():
            cv_row = cv_test_ref[(cv_test_ref["horizon"] == h_name_map[h]) & (cv_test_ref["model"] == refname)]
            stab_row = stability_ref[(stability_ref["horizon_min"] == h) & (stability_ref["model"] == refname)]
            combined_rows.append({
                "horizon": h_name_map[h], "system": sysname,
                "cv_auroc": cv_row["cv_auroc"].iloc[0] if len(cv_row) else None,
                "cv_delta_vs_m3only": cv_row["cv_delta_vs_m3only"].iloc[0] if len(cv_row) else None,
                "cv_p_vs_m3only_canonical": cv_row["cv_p_vs_m3only"].iloc[0] if len(cv_row) else None,
                "n_splits_sig_of_6": stab_row["n_splits_sig_positive"].iloc[0] if len(stab_row) else None,
                "test_auroc": cv_row["test_auroc"].iloc[0] if len(cv_row) else None,
                "test_delta_vs_m3only": cv_row["test_delta_vs_m3only"].iloc[0] if len(cv_row) else None,
                "achieved_sensitivity": op_ref.loc[sysname, "achieved_sensitivity"] if sysname in op_ref.index else None,
                "false_alert_rate": op_ref.loc[sysname, "false_alert_rate"] if sysname in op_ref.index else None,
                "median_lead_min": op_ref.loc[sysname, "median_lead_time_min"] if sysname in op_ref.index else None,
            })
        # M3-only row (its own reference, delta=0 by definition)
        m3_auc = roc_auc_score(y_pat, m3_only_cv[h])
        combined_rows.append({
            "horizon": h_name_map[h], "system": "M3_only", "cv_auroc": round(m3_auc, 4), "cv_delta_vs_m3only": 0.0,
            "cv_p_vs_m3only_canonical": 1.0, "n_splits_sig_of_6": None, "test_auroc": None, "test_delta_vs_m3only": 0.0,
            "achieved_sensitivity": op_ref.loc["M3_only", "achieved_sensitivity"],
            "false_alert_rate": op_ref.loc["M3_only", "false_alert_rate"],
            "median_lead_min": op_ref.loc["M3_only", "median_lead_time_min"],
        })

    df_combined = pd.DataFrame(combined_rows)
    df_combined.to_csv(os.path.join(OUT_DIR, "deployable_fusion_5way_comparison.csv"), index=False)
    print(f"Saved -> {OUT_DIR}/deployable_fusion_5way_comparison.csv")
    for h in HORIZONS:
        print(f"\n  {h_name_map[h]}:")
        print(df_combined[df_combined["horizon"] == h_name_map[h]].to_string(index=False))

    print("\n--- Execution complete. ---")


if __name__ == "__main__":
    main()
