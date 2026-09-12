"""
Phase 16 -- Model 3 verification (docs/phase16_protocol.md Section 8.4).

Model 3 (magnitude + temporal position attention) was the first result in
the Phase 13-16 program to clear CV p<0.05 with a CI entirely above zero at
delivery (+0.0344, p=0.014, CI=[+0.0064,+0.0649]) -- but it also hit its
fixed 200-epoch training cap in 3 of 6 fits (vs. Model 2's 11-86), which
leaves it genuinely unclear whether it had converged. This script checks
that, two ways, and reports the result regardless of outcome (per protocol
Section 8: this is a diagnostic, not a search for a better number -- the
epoch cap is never widened based on which value looks best):

1. Epoch-budget extension: retrain the canonical 5-fold CV with max_epochs=400
   instead of 200. If the result changes materially, 200 was insufficient.
2. Fold-resplit sensitivity: retrain on 2 independently-generated 5-fold
   splits (same procedure used for the parity and P90+Parity hybrid checks
   in Phase 13) to check the result isn't an artifact of the canonical split.

Checkpointed the same way as scripts/phase16_temporal_attention_model.py --
each of the (up to) 15 individual fold-trainings this script runs is its
own resumable unit; interrupting and re-running this script skips whatever
has already completed.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected
from src.models.phase16_causal_attention import train_scorer, predict_at_horizon_for_patients
from src.models.phase16_checkpoint_utils import get_or_train
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.phase16_temporal_attention_model import build_patient_data, carve_inner_validation

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
OUT_DIR = "results/phase16"
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints_verification")

# Model 3's original headline result (max_epochs=200, canonical folds) -- for reference/comparison only.
ORIGINAL_DELIVERY_RESULT = {"auroc": 0.7216, "delta": 0.0344, "p": 0.014, "ci_low": 0.0064, "ci_high": 0.0649}


def run_5fold_delivery(tag, fold_assign, clean_pids, patient_data_cv, t_del_cv, y_lookup, max_epochs, seed_offset):
    """One 5-fold-CV pass at delivery, each fold checkpointed as its own unit."""
    preds = np.zeros(len(clean_pids))
    epochs_used = []
    for f_idx in sorted(set(fold_assign.values())):
        te_pids = [p for p in clean_pids if fold_assign[p] == f_idx]
        tr_pids_all = [p for p in clean_pids if p not in te_pids]
        inner_tr, inner_val = carve_inner_validation(tr_pids_all, y_lookup, seed=42 + f_idx + seed_offset)

        unit_id = f"{tag}_fold{f_idx}"
        scorer, val_loss, n_epochs, resumed = get_or_train(
            CHECKPOINT_DIR, unit_id, in_dim=2, hidden=8,
            train_fn=lambda: train_scorer(inner_tr, inner_val, patient_data_cv, use_elapsed=True,
                                           max_epochs=max_epochs, seed=42),
        )
        epochs_used.append(n_epochs)
        tag_str = "[resumed]" if resumed else "[trained]"
        print(f"    {unit_id} {tag_str}: {n_epochs} epochs, val_loss={val_loss:.4f}")

        out = predict_at_horizon_for_patients(scorer, te_pids, patient_data_cv, t_del_cv, True, 0)
        te_mask = np.array([p in te_pids for p in clean_pids])
        preds[te_mask] = [out[p][0] for p in np.array(clean_pids)[te_mask]]
    return preds, epochs_used


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids_arr = p6["patient_ids"]
    pred_cv_window = p6["pred_unweighted_cv"]

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    y_lookup = {p: int(df[df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    y_pat = np.array([y_lookup[p] for p in clean_pids])

    patient_data_cv, t_del_cv = build_patient_data(pred_cv_window, patient_ids_arr, df, clean_pids, y_lookup)
    sw_cv_h0 = get_patient_scores_at_horizon_corrected(
        pred_cv_window, patient_ids_arr, clean_pids, df["time_before_delivery_min"].values, 0)
    auc_sw = roc_auc_score(y_pat, sw_cv_h0)

    print("================================================================================")
    print("  PHASE 16 -- MODEL 3 VERIFICATION (checkpointed, resumable)                      ")
    print("================================================================================")
    print(f"Original headline result (max_epochs=200, canonical folds): {ORIGINAL_DELIVERY_RESULT}")

    results = {"original": ORIGINAL_DELIVERY_RESULT}

    print("\n--- Check 1: epoch-budget extension (400 vs original 200) ---")
    canonical_fold = {p: folds_blob["assignment"][p][0] for p in clean_pids}
    preds_400, epochs_400 = run_5fold_delivery("epoch_ext_400", canonical_fold, clean_pids,
                                                patient_data_cv, t_del_cv, y_lookup, max_epochs=400, seed_offset=0)
    auc_400 = roc_auc_score(y_pat, preds_400)
    boot_400 = paired_patient_bootstrap(y_pat, sw_cv_h0, preds_400, n_boot=2000, seed=42)
    print(f"  epochs used per fold (cap=400): {epochs_400}")
    print(f"  AUROC={auc_400:.4f}  delta_vs_sw={auc_400-auc_sw:+.4f}  p={boot_400['p_value']:.4f}  "
          f"CI=[{boot_400['ci_95_low']:+.4f},{boot_400['ci_95_high']:+.4f}]")
    results["epoch_extension_400"] = {
        "auroc": round(auc_400, 4), "delta": round(auc_400 - auc_sw, 4), "p": round(boot_400["p_value"], 4),
        "ci_low": round(boot_400["ci_95_low"], 4), "ci_high": round(boot_400["ci_95_high"], 4),
        "epochs_used": epochs_400,
    }

    print("\n--- Check 2: fold-resplit sensitivity (delivery, cap=200, matching original protocol) ---")
    pids_arr = np.array(clean_pids)
    resplit_results = []
    for seed in [11, 22]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        alt_fold = {}
        for f_idx, (_, te_idx) in enumerate(skf.split(pids_arr, y_pat)):
            for p in pids_arr[te_idx]:
                alt_fold[p] = f_idx
        preds, epochs = run_5fold_delivery(f"resplit_seed{seed}", alt_fold, clean_pids,
                                            patient_data_cv, t_del_cv, y_lookup, max_epochs=200, seed_offset=100 + seed)
        auc = roc_auc_score(y_pat, preds)
        boot = paired_patient_bootstrap(y_pat, sw_cv_h0, preds, n_boot=2000, seed=42)
        row = {"fold_source": f"resplit_seed_{seed}", "auroc": round(auc, 4), "delta": round(auc - auc_sw, 4),
               "p": round(boot["p_value"], 4), "ci_low": round(boot["ci_95_low"], 4),
               "ci_high": round(boot["ci_95_high"], 4), "epochs_used": epochs}
        resplit_results.append(row)
        print(f"  resplit seed={seed}: AUROC={auc:.4f}  delta={auc-auc_sw:+.4f}  p={boot['p_value']:.4f}  "
              f"CI=[{boot['ci_95_low']:+.4f},{boot['ci_95_high']:+.4f}]")

    results["fold_resplit_results"] = resplit_results

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "model3_verification.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved -> {OUT_DIR}/model3_verification.json")


if __name__ == "__main__":
    main()
