"""
Phase 13.5 (docs/phase13_protocol.md Section 8, Option D) -- Parity fusion
robustness audit.

Extends results/parity_fusion/ (already run: CV +0.0223 p=0.172, TEST +0.0651
p=0.025, 30-shuffle permutation control 0/30 exceeding true delta) with the
four checks the protocol requires before parity can be called a survivor:

D1. Seed robustness -- (a) bootstrap-seed sensitivity of the CV p-value,
    (b) whether the result depends on the specific canonical fold partition
    by re-running the identical fusion procedure on independently-generated
    patient-grouped 5-fold splits.
D2. Recording-duration proxy check -- correlation between parity and
    windows-per-patient (a recording-length proxy), since a spurious
    "parity" effect could really be a disguised duration effect.
D3. Conditional analysis -- does parity remain associated with the outcome
    after accounting for the other 8 safe admission covariates jointly.
D4. Restates the existing bootstrap CI and permutation control for a single
    consolidated robustness report (no re-computation, just aggregation).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import to_logit, from_logit, select_lambda_trainfold
from scripts.phase11_bootstrap import paired_patient_bootstrap

FOLDS_PATH = "data/processed_clinical/folds.json"
STEP_PRED_PATH = "results/phase13_information_density/step_predictions.csv"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
OUT_DIR = "results/phase13/parity"
os.makedirs(OUT_DIR, exist_ok=True)


def get_patient_scores_h0(pred_arr, patient_ids, clean_pids, t_del):
    scores = []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        scores.append(pred_arr[idx[-1]])
    return np.array(scores)


def fit_parity_lr(parity_train, y_train, parity_apply):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(parity_train.reshape(-1, 1))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    return clf.predict_proba(scaler.transform(parity_apply.reshape(-1, 1)))[:, 1]


def run_parity_fusion_cv(fold_assignment, clean_pids, patient_ids, t_del, y_pat, p6_window, parity_pat):
    """One full parity-fusion CV pass under a given fold assignment dict {pid: fold}."""
    fused = np.zeros(len(clean_pids))
    p6_pat = get_patient_scores_h0(p6_window, patient_ids, clean_pids, t_del)
    logit_p6 = to_logit(p6_pat)
    for f_idx in sorted(set(fold_assignment.values())):
        te_pids = {p for p in clean_pids if fold_assignment[p] == f_idx}
        tr_mask = np.array([p not in te_pids for p in clean_pids])
        te_mask = ~tr_mask
        p_parity = fit_parity_lr(parity_pat[tr_mask], y_pat[tr_mask], parity_pat)
        logit_parity = to_logit(p_parity)
        lam, _ = select_lambda_trainfold(logit_p6[tr_mask], logit_parity[tr_mask], y_pat[tr_mask])
        fused[te_mask] = from_logit(logit_p6[te_mask] + lam * logit_parity[te_mask])
    return roc_auc_score(y_pat, p6_pat), roc_auc_score(y_pat, fused), fused


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    canonical_fold = {p: folds_blob["assignment"][p][0] for p in clean_pids}

    df = pd.read_csv(STEP_PRED_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    patient_ids = df["patient_id"].values
    t_del = df["time_before_delivery_min"].values
    p6_window = df["cv_prob_Step_0_Baseline"].values

    meta = pd.read_csv(METADATA_PATH)
    if "record_id" in meta.columns:
        meta = meta.set_index("record_id")
    parity_by_pid = {str(p): float(meta.loc[int(p), "parity"]) for p in clean_pids}

    y_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    parity_pat = np.array([parity_by_pid[p] for p in clean_pids])

    print("================================================================================")
    print("  PHASE 13.5 -- PARITY FUSION ROBUSTNESS AUDIT                                   ")
    print("================================================================================")

    # ---------------- D1a: bootstrap-seed sensitivity ----------------
    print("\n--- D1a: bootstrap-seed sensitivity (canonical folds, delivery horizon) ---")
    p6_auc_canon, fused_auc_canon, fused_scores_canon = run_parity_fusion_cv(
        canonical_fold, clean_pids, patient_ids, t_del, y_pat, p6_window, parity_pat)
    p6_scores_canon = get_patient_scores_h0(p6_window, patient_ids, clean_pids, t_del)

    boot_rows = []
    for seed in [42, 1, 7, 123, 2024]:
        boot = paired_patient_bootstrap(y_pat, p6_scores_canon, fused_scores_canon, n_boot=2000, seed=seed)
        boot_rows.append({"bootstrap_seed": seed, "delta": round(boot["delta_mean"], 4),
                           "ci_low": round(boot["ci_95_low"], 4), "ci_high": round(boot["ci_95_high"], 4),
                           "p_value": round(boot["p_value"], 4)})
        print(f"  seed={seed}: delta={boot['delta_mean']:+.4f}  CI=[{boot['ci_95_low']:+.4f},{boot['ci_95_high']:+.4f}]  p={boot['p_value']:.4f}")
    df_boot_seeds = pd.DataFrame(boot_rows)
    df_boot_seeds.to_csv(os.path.join(OUT_DIR, "bootstrap_seed_sensitivity.csv"), index=False)

    # ---------------- D1b: fold-assignment sensitivity ----------------
    print("\n--- D1b: fold-assignment sensitivity (independent re-splits, delivery horizon) ---")
    pids_arr = np.array(clean_pids)
    fold_rows = [{"fold_source": "canonical", "p6_auroc": round(p6_auc_canon, 4),
                  "fused_auroc": round(fused_auc_canon, 4), "delta": round(fused_auc_canon - p6_auc_canon, 4)}]
    print(f"  canonical: P6={p6_auc_canon:.4f} fused={fused_auc_canon:.4f} delta={fused_auc_canon-p6_auc_canon:+.4f}")
    for seed in [11, 22, 33, 44]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        alt_fold = {}
        for f_idx, (_, te_idx) in enumerate(skf.split(pids_arr, y_pat)):
            for p in pids_arr[te_idx]:
                alt_fold[p] = f_idx
        p6_auc, fused_auc, _ = run_parity_fusion_cv(alt_fold, clean_pids, patient_ids, t_del, y_pat, p6_window, parity_pat)
        fold_rows.append({"fold_source": f"resplit_seed_{seed}", "p6_auroc": round(p6_auc, 4),
                           "fused_auroc": round(fused_auc, 4), "delta": round(fused_auc - p6_auc, 4)})
        print(f"  resplit seed={seed}: P6={p6_auc:.4f} fused={fused_auc:.4f} delta={fused_auc-p6_auc:+.4f}")
    df_fold_sens = pd.DataFrame(fold_rows)
    df_fold_sens.to_csv(os.path.join(OUT_DIR, "fold_assignment_sensitivity.csv"), index=False)

    # ---------------- D2: recording-duration proxy check ----------------
    print("\n--- D2: parity vs. recording-duration proxy (windows per patient) ---")
    n_windows_per_pat = pd.Series(patient_ids).value_counts()
    n_windows_pat = np.array([n_windows_per_pat.get(p, 0) for p in clean_pids])
    r_pearson, p_pearson = stats.pearsonr(parity_pat, n_windows_pat)
    r_spearman, p_spearman = stats.spearmanr(parity_pat, n_windows_pat)
    print(f"  Pearson  r(parity, n_windows) = {r_pearson:+.4f}  p={p_pearson:.4f}")
    print(f"  Spearman r(parity, n_windows) = {r_spearman:+.4f}  p={p_spearman:.4f}")
    duration_check = pd.DataFrame([{
        "pearson_r": round(r_pearson, 4), "pearson_p": round(p_pearson, 4),
        "spearman_r": round(r_spearman, 4), "spearman_p": round(p_spearman, 4),
        "interpretation": "weak/no correlation -> parity effect not explained by recording length" if abs(r_pearson) < 0.15 else "non-trivial correlation -> needs further scrutiny"
    }])
    duration_check.to_csv(os.path.join(OUT_DIR, "parity_recording_duration.csv"), index=False)
    print(f"  -> {duration_check['interpretation'].iloc[0]}")

    # ---------------- D3: conditional analysis ----------------
    print("\n--- D3: does parity remain associated after accounting for other safe covariates? ---")
    safe_cols = ["age", "gest. weeks", "gravidity", "parity", "diabetes", "hypertension", "preeclampsia", "induced", "sex"]
    clean_int = [int(p) for p in clean_pids]
    sub = meta.loc[clean_int, safe_cols].apply(pd.to_numeric, errors="coerce")
    X_full = SimpleImputer(strategy="median").fit_transform(sub.values)
    X_full_s = StandardScaler().fit_transform(X_full)
    X_without_parity = np.delete(X_full_s, safe_cols.index("parity"), axis=1)

    clf_full = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(X_full_s, y_pat)
    parity_coef = clf_full.coef_[0][safe_cols.index("parity")]

    skf_cond = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_with = np.zeros(len(y_pat))
    oof_without = np.zeros(len(y_pat))
    for tr, te in skf_cond.split(X_full_s, y_pat):
        oof_with[te] = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(X_full_s[tr], y_pat[tr]).predict_proba(X_full_s[te])[:, 1]
        oof_without[te] = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(X_without_parity[tr], y_pat[tr]).predict_proba(X_without_parity[te])[:, 1]
    auc_with = roc_auc_score(y_pat, oof_with)
    auc_without = roc_auc_score(y_pat, oof_without)
    print(f"  Joint-LR coefficient on parity (all 9 safe covariates in model): {parity_coef:+.4f}")
    print(f"  5-fold CV AUROC, all 9 covariates:          {auc_with:.4f}")
    print(f"  5-fold CV AUROC, 8 covariates (no parity):  {auc_without:.4f}  (delta {auc_with-auc_without:+.4f})")
    conditional = pd.DataFrame([{
        "parity_coef_in_joint_model": round(parity_coef, 4),
        "joint_9cov_auroc": round(auc_with, 4), "joint_8cov_no_parity_auroc": round(auc_without, 4),
        "delta_from_dropping_parity": round(auc_with - auc_without, 4)
    }])
    conditional.to_csv(os.path.join(OUT_DIR, "parity_conditional_analysis.csv"), index=False)

    # ---------------- D4: consolidate existing bootstrap + permutation ----------------
    print("\n--- D4: consolidating existing bootstrap CI + 30-shuffle permutation control ---")
    existing_path = "results/parity_fusion/parity_fusion_results.csv"
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
        existing.to_csv(os.path.join(OUT_DIR, "parity_bootstrap.csv"), index=False)
        print(f"  Copied existing bootstrap results -> {OUT_DIR}/parity_bootstrap.csv")
    print("  Existing permutation control: 0/30 shuffled-parity permutations exceeded the true CV delta (+0.0223).")

    summary = {
        "bootstrap_seed_sensitivity": boot_rows,
        "fold_assignment_sensitivity": fold_rows,
        "recording_duration_proxy": duration_check.to_dict(orient="records")[0],
        "conditional_analysis": conditional.to_dict(orient="records")[0],
        "permutation_control_summary": "0/30 shuffled permutations exceeded true delta (+0.0223); mean shuffled delta -0.0467",
    }
    with open(os.path.join(OUT_DIR, "parity_robustness_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved -> {OUT_DIR}/parity_robustness_summary.json")


if __name__ == "__main__":
    main()
