"""
Phase 13.1 (docs/phase13_protocol.md Section 8, Option A) -- causal
recency-weighted P90 experiment.

Primary comparison (protocol A6, confirmatory): recency-weighted P90 vs.
plain P90, both built from the SAME window-level P6 predictions and the SAME
causal window pool -- isolates the aggregation strategy from any horizon-
selection confound.

Secondary comparison (protocol A7, exploratory): recency-P90 vs. P6's own
production score (single-window, corrected-horizon convention) -- tells us
whether recency-P90 is merely better than plain P90, or actually competitive
with the deployed system.

Half-life candidates are prespecified (protocol A5: inf/5/10/20 min), and
selected per outer fold via training-fold-only AUROC maximization (protocol
Section 6) -- never on the fold being reported, never on held-out test.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.aggregation.recency_weighted_p90 import patient_p90_scores, HALF_LIFE_CANDIDATES_MIN
from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
OUT_DIR = "results/phase13/recency_p90"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [0, 10, 20, 30]


def select_half_life_trainfold(pred_window, patient_ids, tr_pids, t_del, y_pat_lookup, h_val):
    best_hl, best_auc = np.inf, -1.0
    y_tr = np.array([y_pat_lookup[p] for p in tr_pids])
    for hl in HALF_LIFE_CANDIDATES_MIN:
        s = patient_p90_scores(pred_window, patient_ids, tr_pids, t_del, h_val, hl)
        auc = roc_auc_score(y_tr, s)
        if auc > best_auc:
            best_auc, best_hl = auc, hl
    return best_hl, best_auc


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids = p6["patient_ids"]
    t_del = p6["t_del"]
    pred_cv_window = p6["pred_unweighted_cv"]
    pred_test_window = p6["pred_unweighted_test"]

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    y_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])
    y_pat_lookup = {p: int(v) for p, v in zip(clean_pids, y_pat)}

    import torch
    test_pt = torch.load("data/processed_clinical/test_dataset.pt", weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = [p for p in clean_pids if p not in test_pids]
    y_test_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in test_pids])

    print("================================================================================")
    print("  PHASE 13.1 -- CAUSAL RECENCY-WEIGHTED P90                                       ")
    print("================================================================================")

    results_rows = []
    for h in HORIZONS:
        plain_cv = np.zeros(len(clean_pids))
        recency_cv = np.zeros(len(clean_pids))
        hl_selected = []

        for f_idx in range(5):
            te_pids = [p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx]
            tr_pids = [p for p in clean_pids if p not in te_pids]

            best_hl, _ = select_half_life_trainfold(pred_cv_window, patient_ids, tr_pids, t_del, y_pat_lookup, h)
            hl_selected.append(best_hl)

            te_mask = np.array([p in te_pids for p in clean_pids])
            plain_cv[te_mask] = patient_p90_scores(pred_cv_window, patient_ids, te_pids, t_del, h, np.inf)
            recency_cv[te_mask] = patient_p90_scores(pred_cv_window, patient_ids, te_pids, t_del, h, best_hl)

        # held-out test: half-life selected on train+val only
        best_hl_test, _ = select_half_life_trainfold(pred_cv_window, patient_ids, train_val_pids, t_del, y_pat_lookup, h)
        plain_test = patient_p90_scores(pred_test_window, patient_ids, test_pids, t_del, h, np.inf)
        recency_test = patient_p90_scores(pred_test_window, patient_ids, test_pids, t_del, h, best_hl_test)

        # secondary: P6 production score (corrected-horizon, single-window)
        p6_corrected_cv = get_patient_scores_at_horizon_corrected(pred_cv_window, patient_ids, clean_pids, t_del, h)
        p6_corrected_test = get_patient_scores_at_horizon_corrected(pred_test_window, patient_ids, test_pids, t_del, h)

        # ---- PRIMARY (confirmatory): recency P90 vs plain P90 ----
        auc_plain_cv = roc_auc_score(y_pat, plain_cv)
        auc_recency_cv = roc_auc_score(y_pat, recency_cv)
        boot_primary_cv = paired_patient_bootstrap(y_pat, plain_cv, recency_cv, n_boot=2000, seed=42)
        p_delong_primary_cv, _, _ = delong_roc_test(y_pat, recency_cv, plain_cv)

        auc_plain_test = roc_auc_score(y_test_pat, plain_test)
        auc_recency_test = roc_auc_score(y_test_pat, recency_test)
        boot_primary_test = paired_patient_bootstrap(y_test_pat, plain_test, recency_test, n_boot=2000, seed=42)
        p_delong_primary_test, _, _ = delong_roc_test(y_test_pat, recency_test, plain_test)

        # ---- SECONDARY (exploratory): recency P90 vs P6 production score ----
        auc_p6_cv = roc_auc_score(y_pat, p6_corrected_cv)
        auc_p6_test = roc_auc_score(y_test_pat, p6_corrected_test)

        row = {
            "horizon_min": h, "half_lives_selected_per_fold": hl_selected, "half_life_test": best_hl_test,
            "cv_auroc_plain_p90": round(auc_plain_cv, 4), "cv_auroc_recency_p90": round(auc_recency_cv, 4),
            "cv_delta_primary": round(auc_recency_cv - auc_plain_cv, 4),
            "cv_boot_p_primary": round(boot_primary_cv["p_value"], 4), "cv_delong_p_primary": round(p_delong_primary_cv, 4),
            "cv_ci_low_primary": round(boot_primary_cv["ci_95_low"], 4), "cv_ci_high_primary": round(boot_primary_cv["ci_95_high"], 4),
            "test_auroc_plain_p90": round(auc_plain_test, 4), "test_auroc_recency_p90": round(auc_recency_test, 4),
            "test_delta_primary": round(auc_recency_test - auc_plain_test, 4),
            "test_boot_p_primary": round(boot_primary_test["p_value"], 4), "test_delong_p_primary": round(p_delong_primary_test, 4),
            "cv_auroc_p6_production": round(auc_p6_cv, 4), "test_auroc_p6_production": round(auc_p6_test, 4),
            "cv_delta_recency_vs_p6": round(auc_recency_cv - auc_p6_cv, 4),
            "test_delta_recency_vs_p6": round(auc_recency_test - auc_p6_test, 4),
        }
        results_rows.append(row)
        print(f"h={h:>3}m  half-lives/fold={hl_selected}")
        print(f"  PRIMARY   CV: plain={auc_plain_cv:.4f} recency={auc_recency_cv:.4f} (d{auc_recency_cv-auc_plain_cv:+.4f} p={boot_primary_cv['p_value']:.3f})  "
              f"TEST: plain={auc_plain_test:.4f} recency={auc_recency_test:.4f} (d{auc_recency_test-auc_plain_test:+.4f} p={boot_primary_test['p_value']:.3f})")
        print(f"  SECONDARY CV: recency={auc_recency_cv:.4f} vs P6={auc_p6_cv:.4f} (d{auc_recency_cv-auc_p6_cv:+.4f})  "
              f"TEST: recency={auc_recency_test:.4f} vs P6={auc_p6_test:.4f} (d{auc_recency_test-auc_p6_test:+.4f})")

    with open(os.path.join(OUT_DIR, "recency_p90_results.json"), "w") as fh:
        json.dump(results_rows, fh, indent=2)
    pd.DataFrame(results_rows).drop(columns=["half_lives_selected_per_fold"]).to_csv(
        os.path.join(OUT_DIR, "recency_p90_results.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/recency_p90_results.csv")


if __name__ == "__main__":
    main()
