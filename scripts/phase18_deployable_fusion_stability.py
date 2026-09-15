"""
Phase 18 Stage 1f -- fold-resplit stability check for the DEPLOYABLE fusion
models (scripts/phase18_deployable_fusion.py): one fusion model per fold,
trained pooled across every causal truncation, evaluated at each
retrospective horizon via prefix truncation -- no horizon-specific
refitting anywhere.

Same discipline as every previous stability check in this study
(scripts/phase18_stability_checks.py, scripts/phase18_full_stability_sweep.py):
5 independent StratifiedKFold resplits (seeds 11/22/33/44/55). M3's own
per-patient running-score sequence is FROZEN (generated once from the
canonical-fold scorers, per this project's own established precedent that
M3 itself is not what this line of ablation is testing) and reused
unchanged across every resplit. Only the parity model and the fusion layer
are refit fresh per resplit, per fold -- the exact bug caught and fixed in
the previous (non-deployable) sweep is guarded against here by construction
(parity is always fit inside the per-fold function, never passed in
pre-computed).

A canonical-split pass is included and cross-checked against
deployable_fusion_results.csv before any resplit number is trusted.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import to_logit, from_logit
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
RESPLIT_SEEDS = [11, 22, 33, 44, 55]
MODEL_IDS = ["A3_fixed_a0.25", "A3_fixed_a0.5", "A3_fixed_a0.75", "A3_selected_once",
             "F2_deployable", "F7_deployable", "A4_sum_to_one_deployable"]


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


def fit_and_apply_one_fold(tr_pids, te_pids, m3_seq_by_pid, t_del_by_pid, parity_raw_by_pid, y_lookup):
    """Fits the parity model fresh (on tr_pids only) and all 7 deployable
    fusion models (pooled across every causal truncation, sample-weighted),
    then applies them to te_pids at every horizon. Returns
    {model_id: {h: {pid: pred}}}."""
    y_tr_pat = np.array([y_lookup[p] for p in tr_pids])
    parity_tr = np.array([parity_raw_by_pid[p] for p in tr_pids])
    parity_all_raw = np.array([parity_raw_by_pid[p] for p in tr_pids + te_pids])

    p_par_tr_and_te = fit_parity_lr(parity_tr, y_tr_pat, parity_all_raw)
    parity_prob_by_pid = {p: p_par_tr_and_te[i] for i, p in enumerate(tr_pids + te_pids)}

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

    out = {mid: {h: {} for h in HORIZONS} for mid in MODEL_IDS}
    for pid in te_pids:
        z_seq = m3_seq_by_pid[pid]; T_i = len(z_seq)
        par_p = parity_prob_by_pid[pid]; logit_par_p = to_logit(par_p)
        for h in HORIZONS:
            k = eligible_prefix_length(t_del_by_pid[pid], h, T_i)
            z_k = z_seq[k - 1]; logit_m3_k = to_logit(z_k)

            out["A3_fixed_a0.25"][h][pid] = 0.25 * z_k + 0.75 * par_p
            out["A3_fixed_a0.5"][h][pid] = 0.5 * z_k + 0.5 * par_p
            out["A3_fixed_a0.75"][h][pid] = 0.75 * z_k + 0.25 * par_p
            out["A3_selected_once"][h][pid] = best_a * z_k + (1 - best_a) * par_p

            x2 = scaler2.transform([[logit_m3_k, logit_par_p]])
            out["F2_deployable"][h][pid] = clf2.predict_proba(x2)[0, 1]
            x7 = np.column_stack([x2, [[logit_m3_k * logit_par_p]]])
            out["F7_deployable"][h][pid] = clf7.predict_proba(x7)[0, 1]
            s_val = best_g1 * logit_m3_k + (1 - best_g1) * logit_par_p
            out["A4_sum_to_one_deployable"][h][pid] = from_logit(g0 + scale * s_val)

    return out, {"alpha": best_a, "g1": best_g1, "coef_m3": float(clf2.coef_[0][0]), "coef_parity": float(clf2.coef_[0][1])}


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
    print("  PHASE 18 DEPLOYABLE-FUSION FOLD-RESPLIT STABILITY CHECK                          ")
    print("================================================================================")

    # ---------------- FROZEN M3 running-score sequences (canonical-fold scorers, never retrained) ----------------
    m3_seq_cv = {}
    for pid in clean_pids:
        r_seq, elapsed_seq, y, T_i = patient_data_cv[pid]
        m3_seq_cv[pid] = predict_all_prefixes(fold_scorers[fold_of[pid]], r_seq, elapsed_seq, True, T_i)

    m3_only_cv = {h: np.zeros(n_patients) for h in HORIZONS}
    for i, pid in enumerate(clean_pids):
        _, _, _, T_i = patient_data_cv[pid]
        for h in HORIZONS:
            k = eligible_prefix_length(t_del_cv[pid], h, T_i)
            m3_only_cv[h][i] = m3_seq_cv[pid][k - 1]

    # ---------------- splits: canonical + 5 resplits ----------------
    split_assignments = {"CANONICAL": fold_of}
    for seed in RESPLIT_SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        assign = {}
        for f_idx, (_, te_idx) in enumerate(skf.split(pids_arr, y_pat)):
            for p in pids_arr[te_idx]:
                assign[p] = f_idx
        split_assignments[f"resplit_{seed}"] = assign

    results_rows = []
    param_rows = []
    for split_name, assign in split_assignments.items():
        preds_oof = {mid: {h: np.zeros(n_patients) for h in HORIZONS} for mid in MODEL_IDS}
        for f_idx in range(5):
            te_pids = [p for p in clean_pids if assign[p] == f_idx]
            tr_pids = [p for p in clean_pids if assign[p] != f_idx]
            out, params = fit_and_apply_one_fold(tr_pids, te_pids, m3_seq_cv, t_del_cv, parity_by_pid, y_lookup)
            param_rows.append({"split": split_name, "fold": f_idx, **params})
            for mid in MODEL_IDS:
                for h in HORIZONS:
                    for pid, pred in out[mid][h].items():
                        preds_oof[mid][h][clean_pids.index(pid)] = pred

        for h in HORIZONS:
            base = m3_only_cv[h]
            auc_m3 = roc_auc_score(y_pat, base)
            for mid in MODEL_IDS:
                arr = preds_oof[mid][h]
                auc = roc_auc_score(y_pat, arr)
                boot = paired_patient_bootstrap(y_pat, base, arr, n_boot=2000, seed=42)
                results_rows.append({
                    "split": split_name, "horizon_min": h, "model": mid,
                    "auroc": round(auc, 4), "delta_vs_m3only": round(auc - auc_m3, 4),
                    "p_vs_m3only": round(boot["p_value"], 4),
                    "ci_low": round(boot["ci_95_low"], 4), "ci_high": round(boot["ci_95_high"], 4),
                })
        print(f"  {split_name:<14} done")

    df_out = pd.DataFrame(results_rows)
    df_out.to_csv(os.path.join(OUT_DIR, "deployable_fusion_resplit_sensitivity.csv"), index=False)
    pd.DataFrame(param_rows).to_csv(os.path.join(OUT_DIR, "deployable_fusion_resplit_params.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/deployable_fusion_resplit_sensitivity.csv")
    print(f"Saved -> {OUT_DIR}/deployable_fusion_resplit_params.csv")

    # ---------------- sanity check vs deployable_fusion_results.csv ----------------
    print("\n--- Sanity check: CANONICAL split reproduces deployable_fusion_results.csv? ---")
    try:
        ref = pd.read_csv(os.path.join(OUT_DIR, "deployable_fusion_results.csv"))
        canon = df_out[df_out["split"] == "CANONICAL"]
        mism = 0
        h_name_map = {0: "Delivery", 10: ">=10m", 20: ">=20m", 30: ">=30m"}
        for h in HORIZONS:
            for mid in MODEL_IDS:
                c = canon[(canon["horizon_min"] == h) & (canon["model"] == mid)]["auroc"].iloc[0]
                r_row = ref[(ref["horizon"] == h_name_map[h]) & (ref["model"] == mid)]
                if len(r_row):
                    r = r_row["cv_auroc"].iloc[0]
                    if abs(c - r) > 1e-3:
                        print(f"  MISMATCH {mid} h={h}: resplit-script={c} vs deployable-script={r}")
                        mism += 1
        print(f"  {'ALL MATCH' if mism == 0 else f'{mism} MISMATCHES'} ({len(HORIZONS)*len(MODEL_IDS)} checks)")
    except FileNotFoundError:
        print("  deployable_fusion_results.csv not found -- skipping cross-check")

    # ---------------- stability ranking ----------------
    print("\n--- Stability ranking: how many of 6 splits show significant, positive delta vs M3-only? ---")
    rank_rows = []
    for h in HORIZONS:
        for mid in MODEL_IDS:
            sub = df_out[(df_out["horizon_min"] == h) & (df_out["model"] == mid)]
            n_sig_pos = int(((sub["p_vs_m3only"] < 0.05) & (sub["delta_vs_m3only"] > 0)).sum())
            rank_rows.append({
                "horizon_min": h, "model": mid, "n_splits_sig_positive": n_sig_pos,
                "mean_delta": round(sub["delta_vs_m3only"].mean(), 4), "min_delta": round(sub["delta_vs_m3only"].min(), 4),
                "max_p": round(sub["p_vs_m3only"].max(), 4), "mean_auroc": round(sub["auroc"].mean(), 4),
            })
    df_rank = pd.DataFrame(rank_rows).sort_values(["horizon_min", "n_splits_sig_positive"], ascending=[True, False])
    df_rank.to_csv(os.path.join(OUT_DIR, "deployable_fusion_stability_ranking.csv"), index=False)
    print(f"Saved -> {OUT_DIR}/deployable_fusion_stability_ranking.csv")
    for h in HORIZONS:
        print(f"\n  h={h}m:")
        print(df_rank[df_rank["horizon_min"] == h].to_string(index=False))

    print("\n--- Execution complete. ---")


if __name__ == "__main__":
    main()
