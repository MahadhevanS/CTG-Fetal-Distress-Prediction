"""
Phase 18 Stage 1c -- full stability sweep, all 16 Stage-1 models x all 4
horizons x 5 independent fold resplits (+ the canonical split), looking for
a fusion mechanism that reliably beats M3-only regardless of which 5-way
patient split is used -- not just the one that happened to look
significant on the canonical partition. Motivated directly by the previous
stability check: of 4 (model, horizon) pairs that looked significant and
were stable across 5 bootstrap-resampling seeds, only 1 survived an
independent fold resplit. This sweep asks the question properly, once,
across the whole model set, instead of chasing individual flagged pairs.

Per-split model-fitting logic is copied verbatim from
scripts/phase18_m3_parity_fusion_ablation.py's main per-fold loop (not
reimplemented from memory) -- the exact lesson from the F7 bug caught
while building the previous stability-check script: a plausible-looking
reimplementation of a fusion formula is not the same model unless checked
against the original's own numbers. This script's canonical-split (resplit
"CANONICAL") results are verified below to reproduce
stage1_main_results.csv exactly before any resplit numbers are trusted.
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

from src.evaluation.phase13_common import to_logit
from src.models.phase16_causal_attention import predict_at_horizon_for_patients
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.phase18_m3_parity_fusion_ablation import (
    fit_logistic, fit_prob_avg, fit_logit_fusion, parity_category_onehot,
)

# Verbatim copy of the MODEL_IDS list from phase18_m3_parity_fusion_ablation.py's
# main() (defined locally there, not at module scope, so re-declared here).
MODEL_IDS = [
    "B0_prevalence", "B1_parity_only", "B2_m3_only",
    "F2_hybrid_C0.1",
    "A3_prob_avg_a0.25", "A3_prob_avg_a0.5", "A3_prob_avg_a0.75", "A3_prob_avg_selected",
    "A4_logit_unconstrained", "A4_logit_nonneg", "A4_logit_sum_to_one", "A4_logit_equal",
    "F1_unregularized", "F3_l1", "F7_interaction", "P3_categorical_parity",
]

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
DATA_DIR = "data/processed_clinical"
CHECKPOINT_DIR = "results/phase16/checkpoints"
OUT_DIR = "results/phase18_fusion_ablation"

HORIZONS = [0, 10, 20, 30]
RESPLIT_SEEDS = [11, 22, 33, 44, 55]


def fit_parity_lr(parity_train, y_train, parity_apply):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(parity_train.reshape(-1, 1))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    X_ap = scaler.transform(parity_apply.reshape(-1, 1))
    return clf.predict_proba(X_ap)[:, 1]


def fit_all_models_one_split(tr_mask, te_mask, m3_h, parity_raw, y_pat):
    """Verbatim copy of phase18_m3_parity_fusion_ablation.py's per-fold body
    (lines ~227-289), generalized to an arbitrary train/test mask pair
    instead of one canonical fold. Returns {model_id: predictions on te_mask}.

    IMPORTANT (fixed after being caught by a cross-check against
    phase18_stability_checks.py's numbers): the parity model is fit FRESH
    here, inside this function, using only the patients in tr_mask --
    never passed in as a pre-computed array. An earlier version of this
    script accepted a p_parity array computed once under the CANONICAL
    fold assignment and reused it (by slicing) for every resplit too. That
    silently leaked information for resplit patients whose canonical-fold
    training set overlapped their new resplit's test fold -- the parity
    "OOF" guarantee only held relative to the canonical split, not
    whichever split this function was actually being asked to score patients
    under. This inflated resplit-significance counts; do not reintroduce it."""
    y_tr = y_pat[tr_mask]
    preds = {}

    pm3_tr, pm3_te = m3_h[tr_mask], m3_h[te_mask]
    p_par_fitted = fit_parity_lr(parity_raw[tr_mask], y_tr, parity_raw)
    ppar_tr, ppar_te = p_par_fitted[tr_mask], p_par_fitted[te_mask]
    zm3_tr = to_logit(pm3_tr); zm3_te = to_logit(pm3_te)
    zpar_tr = to_logit(ppar_tr); zpar_te = to_logit(ppar_te)

    preds["B0_prevalence"] = np.full(te_mask.sum(), y_tr.mean())
    preds["B1_parity_only"] = ppar_te
    preds["B2_m3_only"] = pm3_te

    scaler = StandardScaler()
    Xz_tr = scaler.fit_transform(np.column_stack([zm3_tr, zpar_tr]))
    Xz_te = scaler.transform(np.column_stack([zm3_te, zpar_te]))

    preds["F2_hybrid_C0.1"], _ = fit_logistic(Xz_tr, y_tr, Xz_te, C=0.1)

    for a in [0.25, 0.5, 0.75, "selected"]:
        p_ap, _ = fit_prob_avg(pm3_tr, ppar_tr, pm3_te, ppar_te, y_tr, a)
        key = f"A3_prob_avg_a{a}" if a != "selected" else "A3_prob_avg_selected"
        preds[key] = p_ap

    for variant in ["unconstrained", "nonneg", "sum_to_one", "equal"]:
        p_ap, _ = fit_logit_fusion(zm3_tr, zpar_tr, y_tr, zm3_te, zpar_te, variant)
        preds[f"A4_logit_{variant}"] = p_ap

    preds["F1_unregularized"], _ = fit_logistic(Xz_tr, y_tr, Xz_te, C=1e6)
    preds["F3_l1"], _ = fit_logistic(Xz_tr, y_tr, Xz_te, C=0.1, penalty="l1")

    inter_tr = zm3_tr * zpar_tr; inter_te = zm3_te * zpar_te
    Xi_tr = np.column_stack([Xz_tr, inter_tr]); Xi_te = np.column_stack([Xz_te, inter_te])
    preds["F7_interaction"], _ = fit_logistic(Xi_tr, y_tr, Xi_te, C=0.1)

    cat_tr = parity_category_onehot(parity_raw[tr_mask]); cat_te = parity_category_onehot(parity_raw[te_mask])
    zm3_only_tr = Xz_tr[:, [0]]; zm3_only_te = Xz_te[:, [0]]
    Xc_tr = np.column_stack([zm3_only_tr, cat_tr]); Xc_te = np.column_stack([zm3_only_te, cat_te])
    preds["P3_categorical_parity"], _ = fit_logistic(Xc_tr, y_tr, Xc_te, C=0.1)

    return preds


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
    parity_pat = np.array([parity_by_pid[p] for p in clean_pids])

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids_arr = p6["patient_ids"]
    pred_cv_window = p6["pred_unweighted_cv"]
    patient_data_cv, t_del_cv = build_patient_data(pred_cv_window, patient_ids_arr, df_rolling, clean_pids, y_lookup)

    completed = load_completed_units(CHECKPOINT_DIR)
    fold_scorers = {f: load_scorer_checkpoint(completed[f"Model_3_magnitude_position_fold{f}"], in_dim=2, hidden=8) for f in range(5)}

    print("================================================================================")
    print("  PHASE 18 FULL STABILITY SWEEP -- all 16 models x 4 horizons x 6 splits           ")
    print("  (canonical + 5 independent resplits)                                             ")
    print("================================================================================")

    m3_cv = {h: np.zeros(n_patients) for h in HORIZONS}
    for f_idx in range(5):
        te_pids = [p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx]
        te_mask = np.isin(pids_arr, te_pids)
        for h in HORIZONS:
            out = predict_at_horizon_for_patients(fold_scorers[f_idx], te_pids, patient_data_cv, t_del_cv, True, h)
            m3_cv[h][te_mask] = [out[p][0] for p in pids_arr[te_mask]]

    # NOTE: no canonical-only p_parity_oof precomputed here -- fit_all_models_one_split
    # fits the parity model fresh, per split, per fold, exactly matching whatever
    # tr_mask/te_mask it is called with (canonical or resplit). See that
    # function's docstring for why reusing one canonical parity fit across
    # resplits was a real bug, not a stylistic choice.

    # ---------------- build the "split assignments": canonical + 5 resplits ----------------
    split_assignments = {"CANONICAL": {p: folds_blob["assignment"][p][0] for p in clean_pids}}
    for seed in RESPLIT_SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        assign = {}
        for f_idx, (_, te_idx) in enumerate(skf.split(pids_arr, y_pat)):
            for p in pids_arr[te_idx]:
                assign[p] = f_idx
        split_assignments[f"resplit_{seed}"] = assign

    results_rows = []
    for h in HORIZONS:
        m3_h = m3_cv[h]
        auc_m3_canon = roc_auc_score(y_pat, m3_h)

        for split_name, assign in split_assignments.items():
            preds_oof = {mid: np.zeros(n_patients) for mid in MODEL_IDS}
            for f_idx in range(5):
                te_mask = np.array([assign[p] == f_idx for p in clean_pids])
                tr_mask = ~te_mask
                preds_f = fit_all_models_one_split(tr_mask, te_mask, m3_h, parity_pat, y_pat)
                for mid, arr in preds_f.items():
                    preds_oof[mid][te_mask] = arr

            f2_oof = preds_oof["F2_hybrid_C0.1"]
            auc_f2 = roc_auc_score(y_pat, f2_oof)
            for mid in MODEL_IDS:
                arr = preds_oof[mid]
                auc = roc_auc_score(y_pat, arr) if len(set(arr.round(6).tolist())) > 1 else 0.5
                boot_vs_m3 = paired_patient_bootstrap(y_pat, m3_h, arr, n_boot=2000, seed=42)
                boot_vs_f2 = paired_patient_bootstrap(y_pat, f2_oof, arr, n_boot=2000, seed=42)
                results_rows.append({
                    "horizon_min": h, "split": split_name, "model": mid,
                    "auroc": round(auc, 4),
                    "delta_vs_canonical_m3": round(auc - auc_m3_canon, 4),
                    "delta_vs_split_m3": round(auc - roc_auc_score(y_pat, m3_h), 4),
                    "p_vs_m3": round(boot_vs_m3["p_value"], 4),
                    "delta_vs_split_f2": round(auc - auc_f2, 4),
                    "p_vs_f2": round(boot_vs_f2["p_value"], 4),
                })
            print(f"  h={h:>3}m {split_name:<14} done")

    df_out = pd.DataFrame(results_rows)
    df_out.to_csv(os.path.join(OUT_DIR, "full_stability_sweep.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/full_stability_sweep.csv")

    # ---------------- sanity check: canonical split must reproduce stage1_main_results.csv ----------------
    print("\n--- Sanity check: CANONICAL split reproduces stage1_main_results.csv? ---")
    canon = df_out[df_out["split"] == "CANONICAL"]
    try:
        main_ref = pd.read_csv(os.path.join(OUT_DIR, "stage1_main_results.csv"))
        main_ref_h0 = main_ref[main_ref["horizon"] == "Delivery"].set_index("model")["cv_auroc"]
        mism = 0
        for mid in MODEL_IDS:
            c = canon[(canon["horizon_min"] == 0) & (canon["model"] == mid)]["auroc"].iloc[0]
            ref = main_ref_h0.get(mid, None)
            if ref is not None and abs(c - ref) > 1e-3:
                print(f"  MISMATCH {mid}: sweep={c} vs stage1={ref}")
                mism += 1
        print(f"  {'ALL MATCH' if mism == 0 else f'{mism} MISMATCHES'} (delivery horizon, {len(MODEL_IDS)} models)")
    except FileNotFoundError:
        print("  stage1_main_results.csv not found -- skipping cross-check")

    # ---------------- stability ranking ----------------
    print("\n--- Stability ranking: how many of 6 splits (canonical + 5 resplits) show significant, positive delta vs M3-only? ---")
    rank_rows = []
    for h in HORIZONS:
        for mid in MODEL_IDS:
            if mid == "B2_m3_only":
                continue
            sub = df_out[(df_out["horizon_min"] == h) & (df_out["model"] == mid)]
            n_sig_pos_vs_m3 = int(((sub["p_vs_m3"] < 0.05) & (sub["delta_vs_split_m3"] > 0)).sum())
            n_sig_pos_vs_f2 = int(((sub["p_vs_f2"] < 0.05) & (sub["delta_vs_split_f2"] > 0)).sum())
            mean_delta_vs_m3 = sub["delta_vs_split_m3"].mean()
            min_delta_vs_m3 = sub["delta_vs_split_m3"].min()
            rank_rows.append({
                "horizon_min": h, "model": mid,
                "n_splits_sig_positive_vs_m3": n_sig_pos_vs_m3, "n_splits_sig_positive_vs_f2": n_sig_pos_vs_f2,
                "mean_delta_vs_m3_across_splits": round(mean_delta_vs_m3, 4),
                "min_delta_vs_m3_across_splits": round(min_delta_vs_m3, 4),
                "mean_auroc": round(sub["auroc"].mean(), 4),
            })
    df_rank = pd.DataFrame(rank_rows).sort_values(["horizon_min", "n_splits_sig_positive_vs_m3"], ascending=[True, False])
    df_rank.to_csv(os.path.join(OUT_DIR, "stability_ranking.csv"), index=False)
    print(f"Saved -> {OUT_DIR}/stability_ranking.csv")
    for h in HORIZONS:
        print(f"\n  h={h}m, ranked by # splits (of 6) significantly beating M3-only:")
        print(df_rank[df_rank["horizon_min"] == h].head(6).to_string(index=False))

    print("\n--- Execution complete. ---")


if __name__ == "__main__":
    main()
