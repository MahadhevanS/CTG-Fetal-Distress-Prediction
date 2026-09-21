"""
Phase 13.3 (docs/phase13_protocol.md Section 8, Option C) -- Huber-derived
risk fusion.

Question: does the continuous pH-oriented Huber prediction (already sitting
in rolling_predictions.csv as `risk_prob_proxy`) contain incremental
information beyond P6's binary representation? No new inference, no shared
trunk, no retraining -- P6 and the Huber model are both frozen.

`risk_prob_proxy` is deliberately called the Huber-DERIVED RISK PROXY
throughout (a monotonic sigmoid transform of predicted pH), not "the Huber
probability" -- it was never separately calibrated as one.

Per protocol Section 8/C4: Huber's window-level score is aggregated to
patient level using the exact SAME horizon-selection call as P6 (both
conventions), so the fused pair always describes the same observed window --
not P6's last window fused against Huber's best-ever window.

Fusion and lambda selection reuse src/evaluation/phase13_common.py exactly
(Phase 4's own proven mechanism) -- no new fusion design.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import (
    get_patient_scores_at_horizon, get_patient_scores_at_horizon_corrected,
    to_logit, from_logit, select_lambda_trainfold,
)
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
DATA_DIR = "data/processed_clinical"
OUT_DIR = "results/phase13/huber"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [0, 10, 20, 30]
CONVENTIONS = {"existing": get_patient_scores_at_horizon, "corrected": get_patient_scores_at_horizon_corrected}


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    huber_risk_window = df["risk_prob_proxy"].values  # Huber-derived risk proxy, window-level

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids = p6["patient_ids"]
    t_del = p6["t_del"]
    y_715 = p6["y_715"]
    p6_cv_window = p6["pred_unweighted_cv"]
    p6_test_window = p6["pred_unweighted_test"]

    y_pat_715 = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])

    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = [p for p in clean_pids if p not in test_pids]
    y_test_pat_715 = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in test_pids])

    print("================================================================================")
    print("  PHASE 13.3 -- HUBER-DERIVED RISK FUSION  (logit(p_P6) + lambda*logit(p_Huber))  ")
    print("================================================================================")

    results_rows = []
    pred_dump = {"patient_id": clean_pids}

    for conv_name, horizon_fn in CONVENTIONS.items():
        for h in HORIZONS:
            # ---- patient-level scores, SAME selection function for P6 and Huber ----
            p6_cv_pat = horizon_fn(p6_cv_window, patient_ids, clean_pids, t_del, h)
            huber_cv_pat = horizon_fn(huber_risk_window, patient_ids, clean_pids, t_del, h)

            logit_p6_cv = to_logit(p6_cv_pat)
            logit_huber_cv = to_logit(huber_cv_pat)

            fused_cv = np.zeros(len(clean_pids))
            lambdas_this_h = []
            for f_idx in range(5):
                te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
                tr_mask = np.array([p not in te_pids for p in clean_pids])
                te_mask = ~tr_mask
                lam, _ = select_lambda_trainfold(logit_p6_cv[tr_mask], logit_huber_cv[tr_mask], y_pat_715[tr_mask])
                fused_cv[te_mask] = from_logit(logit_p6_cv[te_mask] + lam * logit_huber_cv[te_mask])
                lambdas_this_h.append(lam)

            # ---- held-out test: lambda selected on train+val only ----
            trval_mask = np.array([p in train_val_pids for p in clean_pids])
            test_mask = np.array([p in test_pids for p in clean_pids])

            p6_test_pat_trval = horizon_fn(p6_cv_window, patient_ids, train_val_pids, t_del, h)
            huber_test_pat_trval = horizon_fn(huber_risk_window, patient_ids, train_val_pids, t_del, h)
            lam_test, _ = select_lambda_trainfold(to_logit(p6_test_pat_trval), to_logit(huber_test_pat_trval),
                                                   y_pat_715[trval_mask])

            p6_test_pat = horizon_fn(p6_test_window, patient_ids, test_pids, t_del, h)
            huber_test_pat = horizon_fn(huber_risk_window, patient_ids, test_pids, t_del, h)
            fused_test = from_logit(to_logit(p6_test_pat) + lam_test * to_logit(huber_test_pat))

            # ---- metrics: P6 alone, Huber alone, fused -- CV and test ----
            auc_p6_cv = roc_auc_score(y_pat_715, p6_cv_pat)
            auc_huber_cv = roc_auc_score(y_pat_715, huber_cv_pat)
            auc_fused_cv = roc_auc_score(y_pat_715, fused_cv)
            auc_p6_test = roc_auc_score(y_test_pat_715, p6_test_pat)
            auc_huber_test = roc_auc_score(y_test_pat_715, huber_test_pat)
            auc_fused_test = roc_auc_score(y_test_pat_715, fused_test)

            boot_cv = paired_patient_bootstrap(y_pat_715, p6_cv_pat, fused_cv, n_boot=2000, seed=42)
            p_delong_cv, _, _ = delong_roc_test(y_pat_715, fused_cv, p6_cv_pat)
            boot_test = paired_patient_bootstrap(y_test_pat_715, p6_test_pat, fused_test, n_boot=2000, seed=42)
            p_delong_test, _, _ = delong_roc_test(y_test_pat_715, fused_test, p6_test_pat)

            row = {
                "convention": conv_name, "horizon_min": h,
                "cv_auroc_p6": round(auc_p6_cv, 4), "cv_auroc_huber_alone": round(auc_huber_cv, 4),
                "cv_auroc_fused": round(auc_fused_cv, 4), "cv_delta_fused_vs_p6": round(auc_fused_cv - auc_p6_cv, 4),
                "cv_boot_p": round(boot_cv["p_value"], 4), "cv_ci_low": round(boot_cv["ci_95_low"], 4),
                "cv_ci_high": round(boot_cv["ci_95_high"], 4), "cv_delong_p": round(p_delong_cv, 4),
                "test_auroc_p6": round(auc_p6_test, 4), "test_auroc_huber_alone": round(auc_huber_test, 4),
                "test_auroc_fused": round(auc_fused_test, 4), "test_delta_fused_vs_p6": round(auc_fused_test - auc_p6_test, 4),
                "test_boot_p": round(boot_test["p_value"], 4), "test_ci_low": round(boot_test["ci_95_low"], 4),
                "test_ci_high": round(boot_test["ci_95_high"], 4), "test_delong_p": round(p_delong_test, 4),
                "lambda_cv_folds": [round(l, 2) for l in lambdas_this_h], "lambda_test": round(lam_test, 2),
            }
            results_rows.append(row)
            print(f"[{conv_name:9s} h={h:>3}m] CV: P6={auc_p6_cv:.4f} Huber={auc_huber_cv:.4f} "
                  f"Fused={auc_fused_cv:.4f} (d{auc_fused_cv-auc_p6_cv:+.4f} p={boot_cv['p_value']:.3f})  |  "
                  f"TEST: P6={auc_p6_test:.4f} Huber={auc_huber_test:.4f} Fused={auc_fused_test:.4f} "
                  f"(d{auc_fused_test-auc_p6_test:+.4f} p={boot_test['p_value']:.3f})")

            if conv_name == "corrected":
                pred_dump[f"h{h}_p6_cv"] = get_patient_scores_at_horizon_corrected(p6_cv_window, patient_ids, clean_pids, t_del, h)
                pred_dump[f"h{h}_huber_fused_cv"] = fused_cv

    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(os.path.join(OUT_DIR, "huber_fusion_results.csv"), index=False)
    with open(os.path.join(OUT_DIR, "huber_fusion_results.json"), "w") as fh:
        json.dump(results_rows, fh, indent=2)
    pd.DataFrame(pred_dump).to_csv(os.path.join(OUT_DIR, "huber_patient_scores.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/huber_fusion_results.csv")


if __name__ == "__main__":
    main()
