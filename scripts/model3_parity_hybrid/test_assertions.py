"""
Automated Assertion Suite for Model 3 + Parity Hybrid (Section 16).

Validates:
1. Data integrity and causal boundaries
2. Leakage controls and feature matrix isolation
3. Model evaluation invariants, paired comparisons, and threshold selection
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, "results/model3_parity_hybrid")
FOLDS_PATH = os.path.join(BASE_DIR, "data/processed_clinical/folds.json")
ROLLING_PATH = os.path.join(BASE_DIR, "results/phase8_rolling/rolling_predictions.csv")
METADATA_PATH = os.path.join(BASE_DIR, "data/raw/ctu-chb-intrapartum/clinical_metadata.csv")
DATA_DIR = os.path.join(BASE_DIR, "data/processed_clinical")


def run_assertions():
    print("================================================================================")
    print("  RUNNING AUTOMATED ASSERTION SUITE (PHASE 9 / SECTION 16)                        ")
    print("================================================================================")
    
    passed_tests = []
    failed_tests = []

    def check(name, condition, details=""):
        if condition:
            passed_tests.append(name)
            print(f"  [PASS] {name}")
        else:
            failed_tests.append((name, details))
            print(f"  [FAIL] {name}: {details}")

    # -------------------------------------------------------------------------
    # 1. Data Assertions
    # -------------------------------------------------------------------------
    print("\n--- 1. Data Assertions ---")
    
    # 1.1 Load prediction table
    pred_path = os.path.join(RESULTS_DIR, "patient_level_predictions.csv")
    assert os.path.exists(pred_path), f"Missing predictions table at {pred_path}"
    df_pred = pd.read_csv(pred_path)
    
    check("No duplicate patient IDs in patient-level table",
          not df_pred["patient_id"].duplicated().any(),
          f"Found {df_pred['patient_id'].duplicated().sum()} duplicate IDs")
    
    check("Exactly 547 patient predictions",
          len(df_pred) == 547,
          f"Expected 547 patients, found {len(df_pred)}")
    
    # 1.2 Disjoint train and test partitions
    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = set(str(m[0]) for m in test_pt["metadata"])
    train_val_pids = set(df_pred[df_pred["outer_fold"] != "test"]["patient_id"].astype(str))
    
    check("No patient appears in both train and test partitions",
          train_val_pids.isdisjoint(test_pids),
          f"Overlap: {train_val_pids.intersection(test_pids)}")
    
    check("Held-out internal test partition count is 83",
          len(test_pids) == 83,
          f"Expected 83, found {len(test_pids)}")

    # 1.3 Rolling window mappings
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    
    check("Every window maps to exactly one valid patient",
          df_rolling["patient_id"].isin(df_pred["patient_id"].astype(str)).all(),
          "Windows contain unknown patient IDs")
    
    check("Every patient has at least one window",
          set(df_pred["patient_id"].astype(str)) == set(df_rolling["patient_id"]),
          "Mismatch between prediction patients and rolling window patients")
    
    # 1.4 Missing endpoints
    check("No missing primary endpoint after cohort construction",
          not df_pred["true_label"].isna().any(),
          "Found NaN in true_label")
    
    check("Primary endpoint binary 0/1 encoding",
          set(df_pred["true_label"].unique()).issubset({0, 1}),
          f"Unexpected values in true_label: {df_pred['true_label'].unique()}")

    # 1.5 Parity missingness and encoding
    check("Parity missingness is 0",
          not df_pred["parity_raw"].isna().any(),
          "Found NaN in parity_raw")
    
    meta = pd.read_csv(METADATA_PATH).set_index("record_id")
    raw_parity_from_meta = [int(meta.loc[int(p), "parity"]) for p in df_pred["patient_id"]]
    check("Parity raw values match clinical metadata exactly",
          list(df_pred["parity_raw"]) == raw_parity_from_meta,
          "Mismatch between pred table parity and metadata parity")

    # 1.6 Window timestamps and causality
    check("Window timestamps strictly increasing per patient",
          all(np.all(np.diff(df_rolling[df_rolling["patient_id"] == p]["start_sample"]) > 0) for p in df_pred["patient_id"]),
          "Non-increasing start samples found")
    
    check("No window uses post-delivery CTG data (t_del >= 0)",
          (df_rolling["time_before_delivery_min"] >= 0).all(),
          "Found negative time_before_delivery_min")
    
    check("All windows satisfy 4800-sample (20-min) length",
          ((df_rolling["end_sample"] - df_rolling["start_sample"]) == 4800).all(),
          "Window length mismatch")

    # -------------------------------------------------------------------------
    # 2. Model & Feature Matrix Assertions
    # -------------------------------------------------------------------------
    print("\n--- 2. Model & Feature Matrix Assertions ---")
    
    coef_path = os.path.join(RESULTS_DIR, "fusion_coefficients.csv")
    assert os.path.exists(coef_path), f"Missing coefficients file at {coef_path}"
    df_coef = pd.read_csv(coef_path)
    
    check("Fusion coefficients fitted for each of 5 folds plus test model",
          len(df_coef) == 6,
          f"Expected 6 rows, found {len(df_coef)}")
    
    check("Positive coefficient for Model 3 across all folds",
          (df_coef["coef_z_m3_scaled"] > 0).all(),
          f"Non-positive Model 3 coefficient: {df_coef['coef_z_m3_scaled'].tolist()}")
    
    check("Non-zero coefficient for Parity across all folds",
          (df_coef["coef_z_parity_scaled"] != 0).all(),
          f"Zero parity coefficient found: {df_coef['coef_z_parity_scaled'].tolist()}")

    # Feature matrix column check
    leakage_audit_path = os.path.join(RESULTS_DIR, "leakage_audit.json")
    assert os.path.exists(leakage_audit_path), f"Missing leakage audit at {leakage_audit_path}"
    with open(leakage_audit_path) as fh:
        leak_audit = json.load(fh)
    
    check("Leakage audit status is PASSED",
          leak_audit.get("status") == "PASSED",
          f"Leakage audit status: {leak_audit.get('status')}")
    
    check("Primary hybrid features strictly limited to Model 3 and Parity",
          leak_audit["verification_details"]["features_used"] == ["to_logit(model3_score)", "to_logit(p_parity)"],
          f"Unexpected features: {leak_audit['verification_details']['features_used']}")
    
    check("No duration or window count in primary hybrid features",
          leak_audit["leakage_invariants"]["no_duration_in_primary_hybrid"] and
          leak_audit["leakage_invariants"]["no_window_count_in_primary_hybrid"],
          "Duration or window count leaked into primary hybrid")

    # Check that predictions are valid probabilities in (0, 1)
    check("Hybrid scores strictly in range [0, 1]",
          ((df_pred["hybrid_score"] >= 0.0) & (df_pred["hybrid_score"] <= 1.0)).all(),
          f"Scores outside [0, 1]: min={df_pred['hybrid_score'].min()}, max={df_pred['hybrid_score'].max()}")

    # -------------------------------------------------------------------------
    # 3. Evaluation & Invariant Assertions
    # -------------------------------------------------------------------------
    print("\n--- 3. Evaluation & Invariant Assertions ---")
    
    deliv_path = os.path.join(RESULTS_DIR, "delivery_metrics.csv")
    ew_path = os.path.join(RESULTS_DIR, "early_warning_metrics.csv")
    op_path = os.path.join(RESULTS_DIR, "operational_metrics.csv")
    
    assert os.path.exists(deliv_path), "Missing delivery_metrics.csv"
    assert os.path.exists(ew_path), "Missing early_warning_metrics.csv"
    assert os.path.exists(op_path), "Missing operational_metrics.csv"
    
    df_deliv = pd.read_csv(deliv_path)
    df_ew = pd.read_csv(ew_path)
    df_op = pd.read_csv(op_path)
    
    # Identical patient counts across models at each horizon
    for h_label in ["Delivery"]:
        sub_h = df_deliv[df_deliv["horizon"] == h_label]
        check(f"Identical patient inclusion across all models at {h_label} (N=547)",
              (sub_h["n_patients"] == 547).all(),
              f"Patient counts: {sub_h['n_patients'].tolist()}")
        check(f"Identical positive count across all models at {h_label} (Pos=110)",
              (sub_h["n_positives"] == 110).all(),
              f"Positive counts: {sub_h['n_positives'].tolist()}")
        
    for h_label in ["10m", "20m", "30m"]:
        sub_h = df_ew[df_ew["horizon"] == h_label]
        n_pts_set = set(sub_h["n_patients"])
        n_pos_set = set(sub_h["n_positives"])
        check(f"Identical patient inclusion across all models at >={h_label}",
              len(n_pts_set) == 1 and len(n_pos_set) == 1,
              f"Mismatched counts at {h_label}: n_pts={n_pts_set}, n_pos={n_pos_set}")

    # Operational evaluation invariants
    check("Operational evaluation evaluated for target sensitivity = 0.80",
          (df_op["target_sensitivity"] == 0.80).all(),
          f"Unexpected target sensitivity: {df_op['target_sensitivity'].tolist()}")
    
    check("Operational false alert rates strictly in [0, 1]",
          ((df_op["false_alert_rate"] >= 0.0) & (df_op["false_alert_rate"] <= 1.0)).all(),
          f"Invalid FAR values: {df_op['false_alert_rate'].tolist()}")

    # -------------------------------------------------------------------------
    # 4. Summary & Exit
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print(f"  ASSERTION SUITE RESULTS: {len(passed_tests)} PASSED, {len(failed_tests)} FAILED")
    print("================================================================================")
    
    test_out_path = os.path.join(RESULTS_DIR, "test_results.txt")
    with open(test_out_path, "w") as fh:
        fh.write(f"Total assertions: {len(passed_tests) + len(failed_tests)}\n")
        fh.write(f"Passed: {len(passed_tests)}\n")
        fh.write(f"Failed: {len(failed_tests)}\n\n")
        fh.write("PASSED TESTS:\n")
        for t in passed_tests:
            fh.write(f"  [PASS] {t}\n")
        if failed_tests:
            fh.write("\nFAILED TESTS:\n")
            for t, det in failed_tests:
                fh.write(f"  [FAIL] {t}: {det}\n")
    print(f"Saved assertion test output -> {test_out_path}")
    
    assert len(failed_tests) == 0, f"Assertion Suite FAILED with {len(failed_tests)} failures"
    return True


if __name__ == "__main__":
    run_assertions()
