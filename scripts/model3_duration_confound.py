"""
Tier-2 hardening item 6 (docs/pre_external_validation_hardening_plan.md #6).

Phase 15 found that every FIXED aggregator's advantage over single-window
selection shrinks with longer recordings (r = -0.12 to -0.23, all p<.01 --
results/phase15/phase15_duration_analysis.csv). That check was never run
for Model 3, a TRAINABLE aggregator, which could plausibly behave
differently (it can, in principle, learn to discount very long sequences
rather than being mechanically diluted by them the way a fixed statistic is).

Pure inference -- no retraining. Reuses:
  - Phase 15's eligible_window_count() helper directly (delivery horizon).
  - Model 3's already-committed per-fold checkpoints (pure inference, same
    reload pattern as scripts/model3_direct_comparisons.py).

Correlates patient-level eligible-window count against Model 3's raw CV
delivery prediction (Pearson + Spearman, matching Phase 15's "value" column)
and against (Model 3 - single-window P6) delivery delta (Pearson + Spearman,
matching Phase 15's "delta" column) -- so the two are directly comparable in
one combined table.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected
from src.models.phase16_causal_attention import predict_at_horizon_for_patients
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase15_temporal_representation_study import eligible_window_count

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
CHECKPOINT_DIR = "results/phase16/checkpoints"
PHASE15_DURATION_PATH = "results/phase15/phase15_duration_analysis.csv"
OUT_PATH = "results/model3_duration_confound.csv"

MODEL_NAME = "Model_3_magnitude_position"
DELIVERY_H = 0


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids_arr = p6["patient_ids"]
    t_del = p6["t_del"]
    pred_cv_window = p6["pred_unweighted_cv"]

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    y_lookup = {p: int(df[df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}

    print("================================================================================")
    print("  MODEL 3 -- RECORDING-DURATION CONFOUND CHECK (Tier-2 hardening item #6)          ")
    print("  Does Model 3 inherit the confound Phase 15 found in every FIXED aggregator?      ")
    print("================================================================================")

    patient_data_cv, t_del_cv = build_patient_data(pred_cv_window, patient_ids_arr, df, clean_pids, y_lookup)

    completed = load_completed_units(CHECKPOINT_DIR)
    pids_arr = np.array(clean_pids)
    model3_cv_delivery = np.zeros(len(clean_pids))
    for f_idx in range(5):
        te_pids = [p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx]
        unit_id = f"{MODEL_NAME}_fold{f_idx}"
        rec = completed[unit_id]
        scorer = load_scorer_checkpoint(rec, in_dim=2, hidden=8)
        print(f"  [loaded] {unit_id} (trained {rec.get('n_epochs','?')} epochs)")
        out = predict_at_horizon_for_patients(scorer, te_pids, patient_data_cv, t_del_cv, True, DELIVERY_H)
        te_mask = np.isin(pids_arr, te_pids)
        model3_cv_delivery[te_mask] = [out[p][0] for p in pids_arr[te_mask]]

    sw_cv_delivery = get_patient_scores_at_horizon_corrected(pred_cv_window, patient_ids_arr, clean_pids, t_del, DELIVERY_H)
    n_windows_h0 = eligible_window_count(patient_ids_arr, clean_pids, t_del, DELIVERY_H)

    r_val, p_val = stats.pearsonr(n_windows_h0, model3_cv_delivery)
    rho_val, p_val_sp = stats.spearmanr(n_windows_h0, model3_cv_delivery)
    delta = model3_cv_delivery - sw_cv_delivery
    r_delta, p_delta = stats.pearsonr(n_windows_h0, delta)
    rho_delta, p_delta_sp = stats.spearmanr(n_windows_h0, delta)

    print(f"\n  n_windows range: [{n_windows_h0.min()}, {n_windows_h0.max()}], mean={n_windows_h0.mean():.1f}")
    print(f"  Model 3 (value)  : Pearson r={r_val:+.4f} (p={p_val:.4f})   Spearman rho={rho_val:+.4f} (p={p_val_sp:.4f})")
    print(f"  Model 3 - P6 (delta): Pearson r={r_delta:+.4f} (p={p_delta:.4f})   Spearman rho={rho_delta:+.4f} (p={p_delta_sp:.4f})")

    new_row = {
        "representation": "model3_trainable", "pearson_r_value_vs_nwindows": round(r_val, 4), "p_value": round(p_val, 4),
        "pearson_r_delta_vs_nwindows": round(r_delta, 4), "p_delta": round(p_delta, 4),
        "spearman_rho_value_vs_nwindows": round(rho_val, 4), "spearman_p_value": round(p_val_sp, 4),
        "spearman_rho_delta_vs_nwindows": round(rho_delta, 4), "spearman_p_delta": round(p_delta_sp, 4),
    }

    phase15_table = pd.read_csv(PHASE15_DURATION_PATH)
    combined = pd.concat([phase15_table, pd.DataFrame([new_row])], ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\n{'='*90}")
    print("COMBINED TABLE (Phase 15's fixed aggregators + Model 3):")
    print(combined.to_string(index=False))
    print(f"\nSaved -> {OUT_PATH}")

    print("\n--- Interpretation (mechanical, not cherry-picked) ---")
    fixed_r_delta = phase15_table["pearson_r_delta_vs_nwindows"].values
    if abs(r_delta) < np.min(np.abs(fixed_r_delta)) or p_delta >= 0.05:
        verdict = "Model 3's delta does NOT show the same confound pattern as the fixed aggregators (weaker or non-significant correlation with recording length)."
    else:
        verdict = "Model 3's delta DOES show a duration-dependence comparable to (or stronger than) the fixed aggregators -- the trainable model has not escaped the confound."
    print(f"  {verdict}")


if __name__ == "__main__":
    main()
