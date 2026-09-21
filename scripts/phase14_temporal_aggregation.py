"""
Phase 14 -- Patient-Level Temporal Aggregation Study (docs/phase14_protocol.md).

Primary confirmatory test (Section 4, fixed before this script computed
anything): AUROC(P90) - AUROC(single-window), delivery horizon, 5-fold CV,
patient-level bootstrap p<0.05, with the held-out test point estimate
required only to agree in direction.

All four candidate aggregators (single-window, P90, max, mean) are
deterministic given the frozen window-level P6 scores -- no hyperparameter
selection anywhere in this script.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected, get_eligible_window_mask
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
OUT_DIR = "results/phase14"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [0, 10, 20, 30]
PRIMARY_HORIZON = 0
SECONDARY_CONFIRMATORY_HORIZON = 30


def pool(pred_arr, patient_ids, clean_pids, t_del, h_val, op):
    scores = []
    for pid in clean_pids:
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
        elif op == "mean":
            scores.append(float(np.mean(p_elig)))
        else:
            raise ValueError(op)
    return np.array(scores)


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids = p6["patient_ids"]
    t_del = p6["t_del"]
    pred_cv = p6["pred_unweighted_cv"]
    pred_test = p6["pred_unweighted_test"]

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    y_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])

    import torch
    test_pt = torch.load("data/processed_clinical/test_dataset.pt", weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    y_test_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in test_pids])

    print("================================================================================")
    print("  PHASE 14 -- PATIENT-LEVEL TEMPORAL AGGREGATION STUDY (pre-registered)           ")
    print("================================================================================")

    rows = []
    for h in HORIZONS:
        sw_cv = get_patient_scores_at_horizon_corrected(pred_cv, patient_ids, clean_pids, t_del, h)
        sw_test = get_patient_scores_at_horizon_corrected(pred_test, patient_ids, test_pids, t_del, h)

        cand_cv = {op: pool(pred_cv, patient_ids, clean_pids, t_del, h, op) for op in ["p90", "max", "mean"]}
        cand_test = {op: pool(pred_test, patient_ids, test_pids, t_del, h, op) for op in ["p90", "max", "mean"]}

        auc_sw_cv = roc_auc_score(y_pat, sw_cv)
        auc_sw_test = roc_auc_score(y_test_pat, sw_test)

        for op in ["p90", "max", "mean"]:
            auc_cv = roc_auc_score(y_pat, cand_cv[op])
            auc_test = roc_auc_score(y_test_pat, cand_test[op])
            auprc_cv = average_precision_score(y_pat, cand_cv[op])

            boot_cv = paired_patient_bootstrap(y_pat, sw_cv, cand_cv[op], n_boot=2000, seed=42)
            pdel_cv, _, _ = delong_roc_test(y_pat, cand_cv[op], sw_cv)
            boot_test = paired_patient_bootstrap(y_test_pat, sw_test, cand_test[op], n_boot=2000, seed=42)
            pdel_test, _, _ = delong_roc_test(y_test_pat, cand_test[op], sw_test)

            is_primary = (h == PRIMARY_HORIZON and op == "p90")
            tier = "PRIMARY CONFIRMATORY" if is_primary else (
                "secondary-confirmatory-horizon" if (h == SECONDARY_CONFIRMATORY_HORIZON and op == "p90") else "exploratory")

            row = {
                "horizon_min": h, "candidate": op, "tier": tier,
                "auroc_single_window_cv": round(auc_sw_cv, 4), "auroc_candidate_cv": round(auc_cv, 4),
                "delta_cv": round(auc_cv - auc_sw_cv, 4), "boot_p_cv": round(boot_cv["p_value"], 4),
                "delong_p_cv": round(pdel_cv, 4), "ci_low_cv": round(boot_cv["ci_95_low"], 4), "ci_high_cv": round(boot_cv["ci_95_high"], 4),
                "auroc_single_window_test": round(auc_sw_test, 4), "auroc_candidate_test": round(auc_test, 4),
                "delta_test": round(auc_test - auc_sw_test, 4), "boot_p_test": round(boot_test["p_value"], 4),
                "delong_p_test": round(pdel_test, 4), "auprc_candidate_cv": round(auprc_cv, 4),
            }
            rows.append(row)
            tag = {"PRIMARY CONFIRMATORY": "[PRIMARY]      ", "secondary-confirmatory-horizon": "[secondary-conf]", "exploratory": "[exploratory]  "}[tier]
            print(f"{tag} h={h:>3}m {op:>4s}  CV: sw={auc_sw_cv:.4f} cand={auc_cv:.4f} (d{auc_cv-auc_sw_cv:+.4f} p={boot_cv['p_value']:.3f} "
                  f"CI=[{boot_cv['ci_95_low']:+.4f},{boot_cv['ci_95_high']:+.4f}])  TEST: sw={auc_sw_test:.4f} cand={auc_test:.4f} "
                  f"(d{auc_test-auc_sw_test:+.4f} p={boot_test['p_value']:.3f})")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, "phase14_results.csv"), index=False)

    # ---- pre-registered verdict (Section 7), computed mechanically from the primary row ----
    primary = df_out[df_out["tier"] == "PRIMARY CONFIRMATORY"].iloc[0]
    cv_sig = primary["boot_p_cv"] < 0.05
    ci_all_positive = primary["ci_low_cv"] > 0
    test_agrees_direction = primary["delta_test"] > 0

    if cv_sig and ci_all_positive and test_agrees_direction:
        verdict = "INTERNALLY REPLICATED -- recommend promoting P90 pooling, pending external validation"
    elif cv_sig or (primary["delta_cv"] > 0 and test_agrees_direction):
        verdict = "NOT REPLICATED AT PRE-REGISTERED BAR -- still promising, not confirmed, no promotion"
    else:
        verdict = "CLOSED -- single-window selection remains production convention"

    print("\n" + "=" * 90)
    print(f"PRIMARY CONFIRMATORY TEST: P90 vs single-window, delivery, CV")
    print(f"  delta={primary['delta_cv']:+.4f}  p={primary['boot_p_cv']:.4f}  CI=[{primary['ci_low_cv']:+.4f},{primary['ci_high_cv']:+.4f}]")
    print(f"  test point estimate agrees in direction: {test_agrees_direction} (delta_test={primary['delta_test']:+.4f})")
    print(f"\nPRE-REGISTERED VERDICT (Section 7 decision rule, applied mechanically): {verdict}")
    print("=" * 90)

    with open(os.path.join(OUT_DIR, "phase14_summary.json"), "w") as fh:
        json.dump({"results": rows, "primary_test": primary.to_dict(), "verdict": verdict}, fh, indent=2)
    print(f"\nSaved -> {OUT_DIR}/phase14_results.csv")
    print(f"Saved -> {OUT_DIR}/phase14_summary.json")


if __name__ == "__main__":
    main()
