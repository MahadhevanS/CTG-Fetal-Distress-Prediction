"""
Phase 18 Stage 1 -- stability checks for A3 (probability-space fusion) and
F7 (interaction logistic fusion), the two mechanisms that beat the current
hybrid (F2) at some horizons in the main ablation
(results/phase18_fusion_ablation/stage1_report.md). Flagged there as "not
yet checked in this pass" before either result should be read as more than
a promising lead.

Two checks, matching this project's own established conventions exactly
(scripts/model3_parity_hybrid/hybrid_engine.py Section 14,
scripts/phase16_model3_verification.py):

1. Bootstrap-seed sensitivity: the underlying predictions are fully
   deterministic (no randomness in fitting) -- this checks whether the
   reported p-values/CIs are themselves stable across 5 independent
   bootstrap resampling seeds, not an artifact of one resampling draw.

2. Fold-resplit sensitivity: 5 independent StratifiedKFold resplits
   (seeds 11/22/33/44/55). Per this project's own precedent
   (hybrid_engine.py's robustness section), the frozen canonical M3 OOF
   scores are NOT retrained per resplit -- only the parity model and the
   fusion layer (which are what this ablation is actually testing) are
   refit under each new fold structure. AUROC and delta are reported
   against the fixed canonical M3-only baseline, plus a resplit-matched F2
   refit under the identical resplit for a fair "does A3/F7 beat F2 under
   this resplit" comparison.
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
from src.models.phase16_causal_attention import predict_at_horizon_for_patients
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.phase18_m3_parity_fusion_ablation import fit_logistic, fit_prob_avg, ALPHA_GRID_FINE

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
DATA_DIR = "data/processed_clinical"
CHECKPOINT_DIR = "results/phase16/checkpoints"
OUT_DIR = "results/phase18_fusion_ablation"

HORIZONS = [0, 10, 20, 30]
BOOTSTRAP_SEEDS = [42, 1, 7, 123, 2024]
RESPLIT_SEEDS = [11, 22, 33, 44, 55]
# The specific (model, horizon) pairs flagged significant vs F2 in the main ablation.
FLAGGED_SIGNIFICANT = {
    ("A3_prob_avg_a0.5", 10), ("A3_prob_avg_selected", 20), ("A3_prob_avg_a0.25", 30),
    ("F7_interaction", 0),
}


def fit_parity_lr(parity_train, y_train, parity_apply):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(parity_train.reshape(-1, 1))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    X_ap = scaler.transform(parity_apply.reshape(-1, 1))
    return clf.predict_proba(X_ap)[:, 1]


def fit_f7(zm3_tr, zpar_tr, y_tr, zm3_ap, zpar_ap):
    """Must match scripts/phase18_m3_parity_fusion_ablation.py's F7 branch
    EXACTLY: the interaction term is the product of the RAW (unstandardized)
    logits, appended after the two main effects have been standardized --
    not the product of the standardized columns (a different, and strictly
    larger-scale, feature). Verified to reproduce the original script's
    canonical delivery AUROC (0.7357) before this check was trusted."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(np.column_stack([zm3_tr, zpar_tr]))
    Xap = scaler.transform(np.column_stack([zm3_ap, zpar_ap]))
    inter_tr = zm3_tr * zpar_tr; inter_ap = zm3_ap * zpar_ap
    Xtr_full = np.column_stack([Xtr, inter_tr]); Xap_full = np.column_stack([Xap, inter_ap])
    p_ap, clf = fit_logistic(Xtr_full, y_tr, Xap_full, C=0.1)
    return p_ap


def fit_f2(zm3_tr, zpar_tr, y_tr, zm3_ap, zpar_ap):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(np.column_stack([zm3_tr, zpar_tr]))
    Xap = scaler.transform(np.column_stack([zm3_ap, zpar_ap]))
    p_ap, _ = fit_logistic(Xtr, y_tr, Xap, C=0.1)
    return p_ap


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
    print("  PHASE 18 STABILITY CHECKS -- A3 and F7 (bootstrap-seed + fold-resplit)           ")
    print("================================================================================")

    # ---------------- canonical M3 OOF scores, all horizons (frozen, never retrained here) ----------------
    m3_cv = {h: np.zeros(n_patients) for h in HORIZONS}
    for f_idx in range(5):
        te_pids = [p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx]
        te_mask = np.isin(pids_arr, te_pids)
        for h in HORIZONS:
            out = predict_at_horizon_for_patients(fold_scorers[f_idx], te_pids, patient_data_cv, t_del_cv, True, h)
            m3_cv[h][te_mask] = [out[p][0] for p in pids_arr[te_mask]]

    # canonical parity OOF (needed for the canonical F2/A3/F7 baselines used as comparators)
    p_parity_oof = np.zeros(n_patients)
    for f_idx in range(5):
        te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
        tr_mask = np.array([p not in te_pids for p in clean_pids])
        te_mask = ~tr_mask
        p_parity_oof[te_mask] = fit_parity_lr(parity_pat[tr_mask], y_pat[tr_mask], parity_pat)[te_mask]

    # canonical model scores at every horizon (for the "vs F2 canonical" and "vs M3 canonical" comparators)
    canon = {"A3_prob_avg_a0.5": {}, "A3_prob_avg_selected": {}, "A3_prob_avg_a0.25": {}, "F7_interaction": {}, "F2_hybrid_C0.1": {}}
    for h in HORIZONS:
        pm3, ppar = m3_cv[h], p_parity_oof
        canon["A3_prob_avg_a0.5"][h] = 0.5 * pm3 + 0.5 * ppar
        canon["A3_prob_avg_a0.25"][h] = 0.25 * pm3 + 0.75 * ppar
        sel = np.zeros(n_patients)
        for f_idx in range(5):
            te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
            tr_mask = np.array([p not in te_pids for p in clean_pids]); te_mask = ~tr_mask
            p_ap, _ = fit_prob_avg(pm3[tr_mask], ppar[tr_mask], pm3[te_mask], ppar[te_mask], y_pat[tr_mask], "selected")
            sel[te_mask] = p_ap
        canon["A3_prob_avg_selected"][h] = sel
        f7 = np.zeros(n_patients); f2 = np.zeros(n_patients)
        for f_idx in range(5):
            te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
            tr_mask = np.array([p not in te_pids for p in clean_pids]); te_mask = ~tr_mask
            zm3_tr, zm3_te = to_logit(pm3[tr_mask]), to_logit(pm3[te_mask])
            zpar_tr, zpar_te = to_logit(ppar[tr_mask]), to_logit(ppar[te_mask])
            f7[te_mask] = fit_f7(zm3_tr, zpar_tr, y_pat[tr_mask], zm3_te, zpar_te)
            f2[te_mask] = fit_f2(zm3_tr, zpar_tr, y_pat[tr_mask], zm3_te, zpar_te)
        canon["F7_interaction"][h] = f7
        canon["F2_hybrid_C0.1"][h] = f2

    # ================================================================
    # CHECK 1 -- bootstrap-seed sensitivity
    # ================================================================
    print("\n--- Bootstrap-seed sensitivity (5 seeds: 42, 1, 7, 123, 2024) ---")
    boot_rows = []
    target_models = ["A3_prob_avg_a0.5", "A3_prob_avg_selected", "A3_prob_avg_a0.25", "F7_interaction"]
    for model in target_models:
        for h in HORIZONS:
            cand = canon[model][h]
            base_m3 = m3_cv[h]
            base_f2 = canon["F2_hybrid_C0.1"][h]
            for seed in BOOTSTRAP_SEEDS:
                b_m3 = paired_patient_bootstrap(y_pat, base_m3, cand, n_boot=2000, seed=seed)
                b_f2 = paired_patient_bootstrap(y_pat, base_f2, cand, n_boot=2000, seed=seed)
                flagged = (model, h) in FLAGGED_SIGNIFICANT
                boot_rows.append({
                    "model": model, "horizon_min": h, "seed": seed, "flagged_significant_in_main_ablation": flagged,
                    "auroc": round(roc_auc_score(y_pat, cand), 4),
                    "delta_vs_m3": round(b_m3["delta_mean"], 4), "p_vs_m3": round(b_m3["p_value"], 4),
                    "ci_low_vs_m3": round(b_m3["ci_95_low"], 4), "ci_high_vs_m3": round(b_m3["ci_95_high"], 4),
                    "delta_vs_f2": round(b_f2["delta_mean"], 4), "p_vs_f2": round(b_f2["p_value"], 4),
                    "ci_low_vs_f2": round(b_f2["ci_95_low"], 4), "ci_high_vs_f2": round(b_f2["ci_95_high"], 4),
                })
            p_range = [r["p_vs_f2"] for r in boot_rows if r["model"] == model and r["horizon_min"] == h]
            tag = " <-- FLAGGED SIGNIFICANT (main ablation, seed 42)" if (model, h) in FLAGGED_SIGNIFICANT else ""
            print(f"  {model:<24} h={h:>3}m  p_vs_f2 range=[{min(p_range):.3f},{max(p_range):.3f}]{tag}")

    pd.DataFrame(boot_rows).to_csv(os.path.join(OUT_DIR, "stability_bootstrap_seed_sensitivity.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/stability_bootstrap_seed_sensitivity.csv")

    # ================================================================
    # CHECK 2 -- fold-resplit sensitivity
    # ================================================================
    print("\n--- Fold-resplit sensitivity (5 resplits: seeds 11, 22, 33, 44, 55) ---")
    resplit_rows = []
    for h in HORIZONS:
        m3_h = m3_cv[h]  # frozen canonical M3, never retrained per resplit (matches hybrid_engine.py precedent)
        auc_m3_canon = roc_auc_score(y_pat, m3_h)
        for seed in RESPLIT_SEEDS:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            resplit_parity = np.zeros(n_patients)
            resplit_a3_05 = np.zeros(n_patients)
            resplit_a3_sel = np.zeros(n_patients)
            resplit_a3_025 = np.zeros(n_patients)
            resplit_f7 = np.zeros(n_patients)
            resplit_f2 = np.zeros(n_patients)

            for tr_idx, te_idx in skf.split(pids_arr, y_pat):
                tr_mask = np.isin(pids_arr, pids_arr[tr_idx])
                te_mask = np.isin(pids_arr, pids_arr[te_idx])
                y_tr = y_pat[tr_mask]

                p_par_te = fit_parity_lr(parity_pat[tr_mask], y_tr, parity_pat)[te_mask]
                resplit_parity[te_mask] = p_par_te

                pm3_tr, pm3_te = m3_h[tr_mask], m3_h[te_mask]
                p_par_tr = fit_parity_lr(parity_pat[tr_mask], y_tr, parity_pat)[tr_mask]

                resplit_a3_05[te_mask] = 0.5 * pm3_te + 0.5 * p_par_te
                resplit_a3_025[te_mask] = 0.25 * pm3_te + 0.75 * p_par_te
                p_sel, _ = fit_prob_avg(pm3_tr, p_par_tr, pm3_te, p_par_te, y_tr, "selected")
                resplit_a3_sel[te_mask] = p_sel

                zm3_tr, zm3_te = to_logit(pm3_tr), to_logit(pm3_te)
                zpar_tr, zpar_te = to_logit(p_par_tr), to_logit(p_par_te)
                resplit_f7[te_mask] = fit_f7(zm3_tr, zpar_tr, y_tr, zm3_te, zpar_te)
                resplit_f2[te_mask] = fit_f2(zm3_tr, zpar_tr, y_tr, zm3_te, zpar_te)

            auc_f2_resplit = roc_auc_score(y_pat, resplit_f2)
            for model, arr in [("A3_prob_avg_a0.5", resplit_a3_05), ("A3_prob_avg_selected", resplit_a3_sel),
                                ("A3_prob_avg_a0.25", resplit_a3_025), ("F7_interaction", resplit_f7),
                                ("F2_hybrid_C0.1_resplit_refit", resplit_f2)]:
                auc = roc_auc_score(y_pat, arr)
                boot_vs_f2r = paired_patient_bootstrap(y_pat, resplit_f2, arr, n_boot=2000, seed=42)
                resplit_rows.append({
                    "model": model, "horizon_min": h, "resplit_seed": seed,
                    "auroc": round(auc, 4), "delta_vs_canonical_m3": round(auc - auc_m3_canon, 4),
                    "delta_vs_resplit_f2": round(auc - auc_f2_resplit, 4),
                    "p_vs_resplit_f2": round(boot_vs_f2r["p_value"], 4),
                    "ci_low_vs_resplit_f2": round(boot_vs_f2r["ci_95_low"], 4), "ci_high_vs_resplit_f2": round(boot_vs_f2r["ci_95_high"], 4),
                })
        for model in ["A3_prob_avg_a0.5", "A3_prob_avg_selected", "A3_prob_avg_a0.25", "F7_interaction"]:
            aucs = [r["auroc"] for r in resplit_rows if r["model"] == model and r["horizon_min"] == h]
            print(f"  h={h:>3}m {model:<24} AUROC range=[{min(aucs):.4f},{max(aucs):.4f}]")

    pd.DataFrame(resplit_rows).to_csv(os.path.join(OUT_DIR, "stability_fold_resplit_sensitivity.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/stability_fold_resplit_sensitivity.csv")
    print("\n--- Execution complete. ---")


if __name__ == "__main__":
    main()
