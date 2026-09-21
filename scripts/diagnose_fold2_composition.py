"""
Fold Composition Diagnostic (2026-08-16)
==========================================
Investigates why fold 2 (0-indexed fold[1], "Fold 2/5" in training logs) has
been the weakest or near-weakest fold across essentially every Model 8
experiment this session -- distress_only, plus_figo, full (every EMA/loss-
weighting/architecture variant tried) -- regardless of what changed about the
model. That consistency points at the fold's patient composition itself
rather than anything fixable by training changes.

Reconstructs the EXACT same patient-level fold split
train_knowledge_infused.py uses (same create_patient_level_folds() from
train.py, same StratifiedKFold(shuffle=True, random_state=42)) and compares
each fold's composition: distress prevalence, FIGO class distribution,
clinical feature averages (baseline/STV/LTV/decel counts), and window counts
per patient (a rough signal-quality/completeness proxy).

NOTE: raw pH values and other clinical_metadata.csv columns (gestational age,
delivery mode, Apgar, etc.) are NOT stored in the processed *_dataset.pt files
-- only derived features are. This script prints fold 2's patient (record_id)
list so it can be cross-referenced against the real clinical_metadata.csv
(only available where the raw CTU-CHB data lives, e.g. the GPU laptop / Drive)
for anything not captured here.
"""

import os
import sys

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.training.multi_task_dataset import load_all_multitask_splits
from src.training.train import create_patient_level_folds

FEATURE_NAMES = [
    "Baseline FHR (bpm)", "STV (bpm)", "LTV (bpm)", "Accel Count",
    "Early Decels", "Late Decels", "Var Decels", "Prolonged Decels",
]


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed/"
    use_joint = "--joint" in sys.argv
    print(f"Loading dataset from {data_dir} ...")
    dataset, patient_ids = load_all_multitask_splits(data_dir)

    y_all = dataset.y_primary
    if use_joint:
        print("Using JOINT (distress, FIGO) stratification (the new fix)")
        folds = create_patient_level_folds(patient_ids, y_all, k_folds=5, secondary_labels=dataset.y_figo)
    else:
        print("Using ORIGINAL distress-only stratification (pre-fix)")
        folds = create_patient_level_folds(patient_ids, y_all, k_folds=5)
    patient_ids_np = np.array(patient_ids)

    print(f"\nTotal: {len(dataset)} windows | {len(set(patient_ids))} unique patients")
    print("=" * 78)

    fold_summaries = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        y_primary_fold = dataset.y_primary[val_idx].numpy()
        y_figo_fold = dataset.y_figo[val_idx].numpy()
        y_features_fold = dataset.y_features[val_idx].numpy()
        val_patients = sorted(set(patient_ids_np[val_idx].tolist()))

        windows_per_patient = [
            int((patient_ids_np[val_idx] == pid).sum()) for pid in val_patients
        ]

        figo_counts = np.bincount(y_figo_fold, minlength=3)
        figo_props = figo_counts / max(len(y_figo_fold), 1)

        summary = {
            "fold": fold_idx,
            "n_windows": len(val_idx),
            "n_patients": len(val_patients),
            "distress_prevalence": float(y_primary_fold.mean()),
            "figo_normal_pct": float(figo_props[0] * 100),
            "figo_suspicious_pct": float(figo_props[1] * 100),
            "figo_pathological_pct": float(figo_props[2] * 100),
            "feature_means": y_features_fold.mean(axis=0),
            "windows_per_patient_mean": float(np.mean(windows_per_patient)),
            "windows_per_patient_std": float(np.std(windows_per_patient)),
            "windows_per_patient_min": int(np.min(windows_per_patient)),
            "windows_per_patient_max": int(np.max(windows_per_patient)),
            "patients": val_patients,
        }
        fold_summaries.append(summary)

        print(f"\n--- Fold {fold_idx}/5 ---")
        print(f" Windows: {summary['n_windows']} | Patients: {summary['n_patients']}")
        print(f" Distress prevalence (window-level): {summary['distress_prevalence']*100:.2f}%")
        print(f" FIGO: Normal={summary['figo_normal_pct']:.1f}% | "
              f"Suspicious={summary['figo_suspicious_pct']:.1f}% | "
              f"Pathological={summary['figo_pathological_pct']:.1f}%")
        print(f" Windows/patient: mean={summary['windows_per_patient_mean']:.1f} "
              f"std={summary['windows_per_patient_std']:.1f} "
              f"range=[{summary['windows_per_patient_min']}, {summary['windows_per_patient_max']}]")
        print(" Feature means:")
        for name, val in zip(FEATURE_NAMES, summary["feature_means"]):
            print(f"   {name:<20}: {val:.3f}")

    print("\n" + "=" * 78)
    print(" CROSS-FOLD COMPARISON (fold 2 flagged if it's an outlier on any axis)")
    print("=" * 78)

    def flag(fold2_val, other_vals, label, higher_is_notable=True):
        others_mean = np.mean(other_vals)
        others_std = np.std(other_vals) + 1e-6
        z = (fold2_val - others_mean) / others_std
        marker = "  <-- OUTLIER" if abs(z) > 1.5 else ""
        print(f" {label:<28}: fold2={fold2_val:8.3f} | other folds mean={others_mean:8.3f} "
              f"(std={others_std:.3f}) | z={z:+.2f}{marker}")

    f2 = fold_summaries[1]  # fold index 1 = "Fold 2"
    others = [s for i, s in enumerate(fold_summaries) if i != 1]

    flag(f2["distress_prevalence"], [s["distress_prevalence"] for s in others], "Distress prevalence")
    flag(f2["figo_pathological_pct"], [s["figo_pathological_pct"] for s in others], "FIGO Pathological %")
    flag(f2["figo_suspicious_pct"], [s["figo_suspicious_pct"] for s in others], "FIGO Suspicious %")
    flag(f2["windows_per_patient_mean"], [s["windows_per_patient_mean"] for s in others], "Windows/patient (mean)")
    flag(f2["windows_per_patient_std"], [s["windows_per_patient_std"] for s in others], "Windows/patient (std)")
    for i, name in enumerate(FEATURE_NAMES):
        flag(f2["feature_means"][i], [s["feature_means"][i] for s in others], name)

    print("\n" + "=" * 78)
    print(f" Fold 2 patient (record_id) list -- cross-reference against the real")
    print(f" clinical_metadata.csv (raw data location) for pH / gestational age /")
    print(f" delivery mode / Apgar / anything not captured in the processed tensors:")
    print("=" * 78)
    print(f2["patients"])


if __name__ == "__main__":
    main()
