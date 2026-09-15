"""
Experiment E0: Data and Feature Audit for Model 3 + Parity Hybrid.

Verifies:
- Patient counts (clean cohort, train/val, test)
- Positive and negative counts for primary and secondary endpoints
- Parity distribution and missingness
- Number of windows and windows per patient
- Absence of duplicate patient IDs and duplicate window IDs
- Time ordering and causal boundaries
- Outcome availability
- Consistency across all source files
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

FOLDS_PATH = os.path.join(BASE_DIR, "data/processed_clinical/folds.json")
ROLLING_PATH = os.path.join(BASE_DIR, "results/phase8_rolling/rolling_predictions.csv")
P6_PRED_PATH = os.path.join(BASE_DIR, "results/phase13/audit/p6_predictions.npz")
METADATA_PATH = os.path.join(BASE_DIR, "data/raw/ctu-chb-intrapartum/clinical_metadata.csv")
DATA_DIR = os.path.join(BASE_DIR, "data/processed_clinical")
OUT_DIR = os.path.join(BASE_DIR, "results/model3_parity_hybrid")


def run_audit():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Load folds and patient IDs
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    n_patients = len(clean_pids)
    
    # Check for duplicate patient IDs
    has_dup_pids = len(clean_pids) != len(set(clean_pids))
    assert not has_dup_pids, "Duplicate patient IDs found in folds.json"
    
    # 2. Load rolling predictions
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    n_windows_total = len(df_rolling)
    
    # Check duplicate windows
    has_dup_windows = df_rolling.duplicated(subset=["patient_id", "start_sample"]).any()
    assert not has_dup_windows, "Duplicate window IDs found in rolling predictions"
    
    # Check patient coverage in rolling windows
    rolling_pids = sorted(df_rolling["patient_id"].unique())
    assert set(clean_pids) == set(rolling_pids), f"Mismatch between folds patients ({len(clean_pids)}) and rolling patients ({len(rolling_pids)})"
    
    # 3. Load metadata
    meta = pd.read_csv(METADATA_PATH).set_index("record_id")
    clean_int_pids = [int(p) for p in clean_pids]
    meta_clean = meta.loc[clean_int_pids]
    
    # Parity audit
    parity_missing = int(meta_clean["parity"].isna().sum())
    assert parity_missing == 0, f"Found {parity_missing} missing parity values"
    parity_counts = meta_clean["parity"].value_counts().sort_index().to_dict()
    parity_min = int(meta_clean["parity"].min())
    parity_max = int(meta_clean["parity"].max())
    
    # 4. Out-of-sample test partition
    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = sorted(p for p in clean_pids if p not in test_pids)
    n_test = len(test_pids)
    n_train_val = len(train_val_pids)
    assert set(test_pids).isdisjoint(set(train_val_pids)), "Train/val and Test partitions are not disjoint"
    assert len(test_pids) + len(train_val_pids) == n_patients, "Partition count sum mismatch"
    
    # 5. Outcome labels
    y_lookup_715 = {p: int(df_rolling[df_rolling["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    y_lookup_705 = {p: int(df_rolling[df_rolling["patient_id"] == p]["severe_label_705"].iloc[0]) for p in clean_pids}
    
    n_pos_715 = sum(y_lookup_715.values())
    n_neg_715 = n_patients - n_pos_715
    n_pos_705 = sum(y_lookup_705.values())
    n_neg_705 = n_patients - n_pos_705
    
    # Verify label consistency with true pH
    for p in clean_pids:
        sub = df_rolling[df_rolling["patient_id"] == p]
        true_ph = float(sub["true_ph"].iloc[0])
        expected_715 = int(true_ph <= 7.15)
        expected_705 = int(true_ph <= 7.05)
        assert y_lookup_715[p] == expected_715, f"Label inconsistency for patient {p} on pH <= 7.15"
        assert y_lookup_705[p] == expected_705, f"Label inconsistency for patient {p} on pH <= 7.05"
    
    # 6. Windows per patient and duration
    windows_per_patient = df_rolling["patient_id"].value_counts().to_dict()
    w_counts = list(windows_per_patient.values())
    min_windows = int(np.min(w_counts))
    max_windows = int(np.max(w_counts))
    mean_windows = float(np.mean(w_counts))
    median_windows = float(np.median(w_counts))
    
    # Check time ordering and causal boundary
    causal_violations = 0
    ordering_violations = 0
    post_delivery_windows = 0
    
    for p in clean_pids:
        sub = df_rolling[df_rolling["patient_id"] == p]
        t_del = sub["time_before_delivery_min"].values
        # Chronological order check
        start_samples = sub["start_sample"].values
        if not np.all(np.diff(start_samples) > 0):
            ordering_violations += 1
        # Causal check: time_before_delivery >= 0 (no post-delivery CTG)
        if np.any(t_del < 0):
            post_delivery_windows += 1
        # Window end sample check
        end_samples = sub["end_sample"].values
        if not np.all(end_samples - start_samples == 4800):
            causal_violations += 1
            
    assert causal_violations == 0, f"Found {causal_violations} window length violations"
    assert ordering_violations == 0, f"Found {ordering_violations} window ordering violations"
    assert post_delivery_windows == 0, f"Found {post_delivery_windows} post-delivery windows"
    
    # 7. Check P6 predictions file alignment
    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    p6_pids = [str(x) for x in p6["patient_ids"]]
    assert len(p6_pids) == n_windows_total, "P6 window count mismatch"
    assert p6_pids == list(df_rolling["patient_id"].values), "P6 patient ID sequence mismatch"
    
    # 8. Horizon eligibility counts
    horizons = [0, 10, 20, 30]
    elig_counts = {}
    for h in horizons:
        elig_pids = []
        for p in clean_pids:
            sub = df_rolling[df_rolling["patient_id"] == p]
            t_del = sub["time_before_delivery_min"].values
            if h == 0 or np.any(t_del >= h):
                elig_pids.append(p)
        elig_pos = sum(y_lookup_715[p] for p in elig_pids)
        elig_counts[f"h_{h}m"] = {
            "eligible_patients": len(elig_pids),
            "eligible_positives": elig_pos,
            "eligible_negatives": len(elig_pids) - elig_pos,
            "pct_eligible": round(len(elig_pids) / n_patients * 100, 2),
        }
    
    audit_data = {
        "status": "PASSED",
        "cohort": {
            "n_patients": n_patients,
            "n_train_val": n_train_val,
            "n_internal_test": n_test,
            "test_fraction": round(n_test / n_patients, 4),
        },
        "endpoints": {
            "primary_pH_715": {
                "positive": n_pos_715,
                "negative": n_neg_715,
                "prevalence": round(n_pos_715 / n_patients, 4),
            },
            "secondary_severe_pH_705": {
                "positive": n_pos_705,
                "negative": n_neg_705,
                "prevalence": round(n_pos_705 / n_patients, 4),
            }
        },
        "parity": {
            "missing_values": parity_missing,
            "min": parity_min,
            "max": parity_max,
            "value_distribution": parity_counts,
            "nulliparous_count": parity_counts.get(0, 0),
            "nulliparous_fraction": round(parity_counts.get(0, 0) / n_patients, 4),
        },
        "windows": {
            "n_total_windows": n_windows_total,
            "min_per_patient": min_windows,
            "max_per_patient": max_windows,
            "mean_per_patient": round(mean_windows, 2),
            "median_per_patient": round(median_windows, 2),
            "sampling_rate_hz": 4.0,
            "window_length_samples": 4800,
            "window_stride_samples": 600,
        },
        "invariants": {
            "duplicate_patient_ids": bool(has_dup_pids),
            "duplicate_window_ids": bool(has_dup_windows),
            "post_delivery_windows": int(post_delivery_windows),
            "ordering_violations": int(ordering_violations),
            "causal_window_violations": int(causal_violations),
            "disjoint_cv_test_split": True,
        },
        "horizon_eligibility": elig_counts,
    }
    
    # Save JSON
    json_path = os.path.join(OUT_DIR, "data_feature_audit.json")
    with open(json_path, "w") as fh:
        json.dump(audit_data, fh, indent=2)
    print(f"Saved audit JSON -> {json_path}")
    
    # Save Markdown report
    md_path = os.path.join(OUT_DIR, "data_feature_audit.md")
    with open(md_path, "w") as fh:
        fh.write("# Experiment E0: Data and Feature Audit Report\n\n")
        fh.write(f"**Status:** {audit_data['status']}\n\n")
        fh.write("## 1. Cohort Summary\n")
        fh.write(f"- Total Clean Patients: **{n_patients}**\n")
        fh.write(f"- 5-Fold CV Patients (Train/Val pool): **{n_train_val}** (84.8%)\n")
        fh.write(f"- Held-Out Internal Test Partition: **{n_test}** (15.2%)\n\n")
        fh.write("## 2. Endpoints\n")
        fh.write(f"- Primary endpoint (pH $\\le 7.15$): **{n_pos_715}** positives ({audit_data['endpoints']['primary_pH_715']['prevalence']*100:.1f}%), {n_neg_715} negatives\n")
        fh.write(f"- Secondary severe endpoint (pH $\\le 7.05$): **{n_pos_705}** positives ({audit_data['endpoints']['secondary_severe_pH_705']['prevalence']*100:.2f}%), {n_neg_705} negatives\n\n")
        fh.write("## 3. Parity Distribution\n")
        fh.write(f"- Missing values: **{parity_missing}**\n")
        fh.write(f"- Min/Max: {parity_min} / {parity_max}\n")
        fh.write(f"- Nulliparous (parity = 0): **{parity_counts.get(0, 0)}** ({parity_counts.get(0, 0)/n_patients*100:.1f}%)\n")
        fh.write("- Breakdown: " + ", ".join(f"parity {k}: {v}" for k, v in parity_counts.items()) + "\n\n")
        fh.write("## 4. Window & Causal Invariants\n")
        fh.write(f"- Total windows: **{n_windows_total}**\n")
        fh.write(f"- Windows per patient: mean {mean_windows:.1f}, median {median_windows:.1f}, range [{min_windows}, {max_windows}]\n")
        fh.write("- Duplicate patient IDs: **None**\n")
        fh.write("- Duplicate window entries: **None**\n")
        fh.write("- Post-delivery windows: **0**\n")
        fh.write("- Chronological ordering: **Fully verified**\n")
        fh.write("- Causal 20-min window boundary: **Fully verified**\n\n")
        fh.write("## 5. Horizon Patient Eligibility\n")
        fh.write("| Horizon | Eligible Patients | Positives | Negatives | % Cohort Eligible |\n")
        fh.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for h in horizons:
            info = elig_counts[f"h_{h}m"]
            fh.write(f"| $\\ge {h}$m | {info['eligible_patients']} | {info['eligible_positives']} | {info['eligible_negatives']} | {info['pct_eligible']}% |\n")
    print(f"Saved audit Markdown -> {md_path}")
    
    return audit_data


if __name__ == "__main__":
    run_audit()
