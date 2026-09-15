"""
Tier-2 hardening item 3 (docs/pre_external_validation_hardening_plan.md #3).

Every existing comparison for Model 3 is "Model 3 vs. single-window P6" (or,
for Model 4, "vs. Model 3"). A direct Model-3-vs-Max and Model-3-vs-P90
pairwise test has never been run -- results/phase16/phase16_models_2_3_summary.json
stores only aggregate metrics, no per-patient prediction vector.

This script is pure inference -- no retraining. It:
  1. Reloads Model 3's already-committed per-fold checkpoints
     (results/phase16/checkpoints/Model_3_magnitude_position_fold{0-4}.pt,
     ..._testmodel.pt) and reconstructs per-patient predictions at all four
     horizons via predict_at_horizon_for_patients -- the same function used
     to produce Model 3's original headline numbers.
  2. Reconstructs Max/P90 per-patient scores from the same frozen
     results/phase13/audit/p6_predictions.npz window scores, using the
     canonical P90 estimator (numpy.percentile, Tier-1 hardening item #2) --
     identical logic to scripts/phase14_temporal_aggregation.py's pool(),
     so the eligible-window set is apples-to-apples with Model 3's own
     causally-eligible prefix at each horizon.
  3. Runs paired bootstrap (B=2000) + DeLong for Model3-vs-Max and
     Model3-vs-P90 at every horizon, on both CV and the held-out internal
     test partition.

Model 3 vs. Model 4 is NOT recomputed here -- Model_4_peak_aware_fusion_results.csv
already has cv_delta_vs_model3 / cv_p_vs_model3 columns; this item is scoped
to the two genuinely missing pairs only.
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

from src.evaluation.phase13_common import get_eligible_window_mask
from src.models.phase16_causal_attention import predict_at_horizon_for_patients
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
DATA_DIR = "data/processed_clinical"
CHECKPOINT_DIR = "results/phase16/checkpoints"
OUT_PATH = "results/model3_direct_comparisons.csv"

HORIZONS = [0, 10, 20, 30]
MODEL_NAME = "Model_3_magnitude_position"


def pool(pred_arr, patient_ids, pids_list, t_del, h_val, op):
    """Identical to scripts/phase14_temporal_aggregation.py's pool() -- same
    eligible-window mask, same canonical numpy.percentile estimator."""
    scores = []
    for pid in pids_list:
        idx = np.where(patient_ids == str(pid))[0]
        t_pts = t_del[idx]
        elig = get_eligible_window_mask(t_pts, h_val)
        if not np.any(elig):
            elig = np.ones_like(t_pts, dtype=bool)
        p_elig = pred_arr[idx[elig]]
        if op == "p90":
            scores.append(float(np.percentile(p_elig, 90)))
        elif op == "max":
            scores.append(float(np.max(p_elig)))
        else:
            raise ValueError(op)
    return np.array(scores)


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids_arr = p6["patient_ids"]
    t_del = p6["t_del"]
    pred_cv_window = p6["pred_unweighted_cv"]
    pred_test_window = p6["pred_unweighted_test"]

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    y_lookup = {p: int(df[df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    y_pat = np.array([y_lookup[p] for p in clean_pids])

    import torch
    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    y_test_pat = np.array([y_lookup[p] for p in test_pids])

    print("================================================================================")
    print("  MODEL 3 -- DIRECT COMPARISONS vs. MAX AND P90 (Tier-2 hardening item #3)         ")
    print("  Pure inference on already-committed checkpoints -- no retraining.               ")
    print("================================================================================")

    patient_data_cv, t_del_cv = build_patient_data(pred_cv_window, patient_ids_arr, df, clean_pids, y_lookup)
    patient_data_test, t_del_test = build_patient_data(pred_test_window, patient_ids_arr, df, test_pids, y_lookup)

    completed = load_completed_units(CHECKPOINT_DIR)
    for f_idx in range(5):
        unit_id = f"{MODEL_NAME}_fold{f_idx}"
        if unit_id not in completed:
            raise RuntimeError(f"Missing committed checkpoint for {unit_id} -- expected pure inference, found nothing to load.")
    if f"{MODEL_NAME}_testmodel" not in completed:
        raise RuntimeError(f"Missing committed checkpoint for {MODEL_NAME}_testmodel.")

    # ---------------- Model 3 CV predictions (reload each fold's own checkpoint) ----------------
    model3_cv = {h: np.zeros(len(clean_pids)) for h in HORIZONS}
    pids_arr = np.array(clean_pids)
    for f_idx in range(5):
        te_pids = [p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx]
        rec = completed[f"{MODEL_NAME}_fold{f_idx}"]
        scorer = load_scorer_checkpoint(rec, in_dim=2, hidden=8)
        print(f"  [loaded] {MODEL_NAME}_fold{f_idx} (trained {rec.get('n_epochs','?')} epochs, val_loss={rec.get('val_loss','?')})")
        for h in HORIZONS:
            out = predict_at_horizon_for_patients(scorer, te_pids, patient_data_cv, t_del_cv, True, h)
            te_mask = np.isin(pids_arr, te_pids)
            model3_cv[h][te_mask] = [out[p][0] for p in pids_arr[te_mask]]

    # ---------------- Model 3 held-out internal test partition predictions ----------------
    rec_test = completed[f"{MODEL_NAME}_testmodel"]
    scorer_test = load_scorer_checkpoint(rec_test, in_dim=2, hidden=8)
    print(f"  [loaded] {MODEL_NAME}_testmodel (trained {rec_test.get('n_epochs','?')} epochs, val_loss={rec_test.get('val_loss','?')})")
    model3_test = {}
    for h in HORIZONS:
        out = predict_at_horizon_for_patients(scorer_test, test_pids, patient_data_test, t_del_test, True, h)
        model3_test[h] = np.array([out[p][0] for p in test_pids])

    # ---------------- Max / P90 per-patient scores (canonical estimator) ----------------
    max_cv = {h: pool(pred_cv_window, patient_ids_arr, clean_pids, t_del, h, "max") for h in HORIZONS}
    p90_cv = {h: pool(pred_cv_window, patient_ids_arr, clean_pids, t_del, h, "p90") for h in HORIZONS}
    max_test = {h: pool(pred_test_window, patient_ids_arr, test_pids, t_del, h, "max") for h in HORIZONS}
    p90_test = {h: pool(pred_test_window, patient_ids_arr, test_pids, t_del, h, "p90") for h in HORIZONS}

    # ---------------- paired comparisons ----------------
    rows = []
    for h in HORIZONS:
        h_name = "Delivery (0m)" if h == 0 else f">={h}m"
        m3_cv, m3_test = model3_cv[h], model3_test[h]

        for opp_name, opp_cv_arr, opp_test_arr in [
            ("Max", max_cv[h], max_test[h]),
            ("P90", p90_cv[h], p90_test[h]),
        ]:
            auc_m3_cv = roc_auc_score(y_pat, m3_cv)
            auc_opp_cv = roc_auc_score(y_pat, opp_cv_arr)
            boot_cv = paired_patient_bootstrap(y_pat, opp_cv_arr, m3_cv, n_boot=2000, seed=42)
            pdel_cv, _, _ = delong_roc_test(y_pat, m3_cv, opp_cv_arr)

            auc_m3_test = roc_auc_score(y_test_pat, m3_test)
            auc_opp_test = roc_auc_score(y_test_pat, opp_test_arr)
            boot_test = paired_patient_bootstrap(y_test_pat, opp_test_arr, m3_test, n_boot=2000, seed=42)
            pdel_test, _, _ = delong_roc_test(y_test_pat, m3_test, opp_test_arr)

            row = {
                "horizon": h_name, "comparison": f"Model3_vs_{opp_name}",
                "cv_auroc_model3": round(auc_m3_cv, 4), f"cv_auroc_{opp_name.lower()}": round(auc_opp_cv, 4),
                "cv_delta_model3_minus_opp": round(auc_m3_cv - auc_opp_cv, 4),
                "cv_boot_p": round(boot_cv["p_value"], 4), "cv_delong_p": round(pdel_cv, 4),
                "cv_ci_low": round(boot_cv["ci_95_low"], 4), "cv_ci_high": round(boot_cv["ci_95_high"], 4),
                "test_auroc_model3": round(auc_m3_test, 4), f"test_auroc_{opp_name.lower()}": round(auc_opp_test, 4),
                "test_delta_model3_minus_opp": round(auc_m3_test - auc_opp_test, 4),
                "test_boot_p": round(boot_test["p_value"], 4), "test_delong_p": round(pdel_test, 4),
                "test_ci_low": round(boot_test["ci_95_low"], 4), "test_ci_high": round(boot_test["ci_95_high"], 4),
            }
            rows.append(row)
            print(f"\n--- {h_name}: Model 3 vs. {opp_name} ---")
            print(f"  CV   : Model3={auc_m3_cv:.4f}  {opp_name}={auc_opp_cv:.4f}  "
                  f"Delta(M3-{opp_name})={row['cv_delta_model3_minus_opp']:+.4f}  p_boot={row['cv_boot_p']:.3f}  "
                  f"p_delong={row['cv_delong_p']:.4f}  CI=[{row['cv_ci_low']:+.4f},{row['cv_ci_high']:+.4f}]")
            print(f"  TEST : Model3={auc_m3_test:.4f}  {opp_name}={auc_opp_test:.4f}  "
                  f"Delta(M3-{opp_name})={row['test_delta_model3_minus_opp']:+.4f}  p_boot={row['test_boot_p']:.3f}  "
                  f"p_delong={row['test_delong_p']:.4f}  CI=[{row['test_ci_low']:+.4f},{row['test_ci_high']:+.4f}]")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
