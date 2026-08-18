"""
MIL substrate pipeline -- uniform-stride windowing with pipeline.py's full
signal-processing chain.

WHY THIS EXISTS (2026-08-18): patient-level (multiple-instance) modelling is
invalid on data/processed/ because pipeline.py selects its stride from the
patient's LABEL (30s for distress, 10min for normal, as minority oversampling).
That makes bag size a near-perfect label proxy -- measured AUROC 0.9947 using
window count alone as the predictor (distress median 80 windows vs normal 5).
Any max/top-k patient-level aggregation on that data measures bag size, not
fetal distress.

This pipeline keeps EVERYTHING that makes pipeline.py's signal quality good
(spike removal -> cubic interpolation -> lowpass -> iterative baseline ->
baseline-corrected channel 0 -> global train-fit z-score) and changes ONLY the
windowing/labelling:

  1. Uniform stride for every patient, independent of label.
  2. Adds y_patient -- the pH-based PATIENT outcome, broadcast to each of that
     patient's windows. This is the MIL target. y_primary (horizon-relabelled)
     is retained unchanged as the window-level auxiliary target.
  3. Patient-level quality gate via assess_patient_missing_ratio().

pipeline.py and data/processed/ are untouched -- every existing window-level
result in this project stays exactly reproducible.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Preprocessing modules (bare imports -- these modules are written to run with
# src/preprocessing/ on sys.path, same convention as pipeline.py)
from ingestion import load_ctu_chb_record, load_clinical_metadata, TARGET_FS
from signal_quality import get_valid_windows, assess_patient_missing_ratio
from filtering import (
    remove_spikes,
    interpolate_missing,
    apply_lowpass_filter,
    zscore_normalize_channels,
)
from baseline import calculate_iterative_baseline
from features import calculate_variability, detect_accelerations, detect_decelerations
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knowledge.figo import classify_figo


# ---------------------------------------------------------------------------
# Clinical Constants (identical to pipeline.py unless noted)
# ---------------------------------------------------------------------------

WINDOW_MINUTES = 20
LAST_HOUR_MINUTES = 60
PREDICTION_HORIZON_MINUTES = 30

WINDOW_SAMPLES = int(WINDOW_MINUTES * 60 * TARGET_FS)               # 4800
LAST_HOUR_SAMPLES = int(LAST_HOUR_MINUTES * 60 * TARGET_FS)         # 14400
PREDICTION_HORIZON_SAMPLES = int(PREDICTION_HORIZON_MINUTES * 60 * TARGET_FS)  # 7200

# CHANGED vs pipeline.py: one uniform stride for every patient in every split.
# pipeline.py uses DISTRESS_STRIDE_MINUTES=0.5 / NORMAL_STRIDE_MINUTES=10 chosen
# by label -- the source of the bag-size leak this pipeline exists to remove.
# 2.5 min over a 60-min truncated recording with a 20-min window yields ~17
# windows/patient, a workable MIL bag size, with no label dependence whatsoever.
STRIDE_MINUTES = 2.5
STRIDE_SAMPLES = int(STRIDE_MINUTES * 60 * TARGET_FS)               # 600

PATIENT_MAX_MISSING_RATIO = 0.5   # patient-level quality gate
WINDOW_MAX_MISSING_RATIO = 0.3    # unchanged from pipeline.py

FHR_MIN_BPM = 50.0
FHR_MAX_BPM = 240.0


def process_pipeline_mil(raw_data_dir: str, metadata_path: str, output_dir: str) -> None:
    """
    Generates the uniform-stride MIL substrate into output_dir as
    train/val/test_dataset.pt (6 keys: X, y_primary, y_patient, y_figo,
    y_features, metadata) plus ctu_signal_scaler.npz and feature_scaler.npz.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading metadata...")
    if not os.path.exists(metadata_path):
        print(f"  [ERROR] Metadata file not found: {metadata_path}")
        return

    metadata = load_clinical_metadata(metadata_path)
    metadata = (
        metadata.set_index('record_id')
        if 'record_id' in metadata.columns
        else metadata
    )

    # Patient-level stratified split BEFORE window extraction (zero leakage),
    # same 70/15/15 and random_state=42 as pipeline.py so the held-out test
    # cohort is directly comparable.
    records = list(metadata.index)
    y_patient_all = (metadata['ph'] <= 7.15).astype(int).values

    train_ids, test_val_ids, y_train, y_test_val = train_test_split(
        records, y_patient_all,
        test_size=0.3, stratify=y_patient_all, random_state=42
    )
    val_ids, test_ids, _, _ = train_test_split(
        test_val_ids, y_test_val,
        test_size=0.5, stratify=y_test_val, random_state=42
    )

    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}

    for split_name, ids in splits.items():
        print(f"\nProcessing {split_name} split ({len(ids)} patients)...")
        X_data, Y_primary, Y_patient, Y_figo, Y_features, Record_ids = [], [], [], [], [], []

        n_skipped_missing = 0
        n_skipped_no_data = 0
        n_excluded_quality = 0
        n_patients_kept = 0

        for record_id in ids:
            record_path = os.path.join(raw_data_dir, str(record_id))

            if not os.path.exists(record_path + '.dat'):
                n_skipped_missing += 1
                continue

            fhr, uc, fs = load_ctu_chb_record(record_path)
            if len(fhr) == 0:
                n_skipped_no_data += 1
                continue

            # Patient-level quality gate, on the raw untruncated signal
            if not assess_patient_missing_ratio(fhr, max_missing_ratio=PATIENT_MAX_MISSING_RATIO):
                n_excluded_quality += 1
                continue

            is_distress = bool(metadata.loc[record_id, 'ph'] <= 7.15)

            # Clinically relevant horizon: last 60 min before delivery
            if len(fhr) > LAST_HOUR_SAMPLES:
                fhr = fhr[-LAST_HOUR_SAMPLES:]
                uc = uc[-LAST_HOUR_SAMPLES:]

            signal_length = len(fhr)
            horizon_start_idx = max(0, signal_length - PREDICTION_HORIZON_SAMPLES)

            # CHANGED: uniform stride, no label branching, all splits alike.
            valid_starts = get_valid_windows(
                fhr, WINDOW_SAMPLES, STRIDE_SAMPLES,
                max_missing_ratio=WINDOW_MAX_MISSING_RATIO
            )
            if len(valid_starts) == 0:
                continue
            n_patients_kept += 1

            for start in valid_starts:
                end = start + WINDOW_SAMPLES
                fhr_win = fhr[start:end].copy()
                uc_win = uc[start:end].copy()

                # Full pipeline.py signal chain (unchanged)
                fhr_win = remove_spikes(fhr_win, fs=fs)
                fhr_win = interpolate_missing(fhr_win, clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
                fhr_win = apply_lowpass_filter(fhr_win, fs=fs)
                uc_win = apply_lowpass_filter(uc_win, fs=fs)

                baseline = calculate_iterative_baseline(fhr_win)
                fhr_norm = fhr_win - baseline

                stv, ltv = calculate_variability(fhr_win, fs=fs, baseline=baseline)
                accels = detect_accelerations(fhr_win, baseline, fs=fs)
                decels = detect_decelerations(fhr_win, baseline, uc_win, fs=fs)

                base_val = float(np.mean(baseline))
                figo_class = classify_figo(base_val, ltv, accels, decels)

                window_tensor = np.vstack((fhr_norm, uc_win))

                features_target = [
                    base_val, stv, ltv,
                    float(accels),
                    float(decels['early']), float(decels['late']),
                    float(decels['variable']), float(decels['prolonged']),
                ]

                # Window-level auxiliary label (horizon-relabelled, as pipeline.py)
                within_horizon = (start >= horizon_start_idx)
                window_label = int(is_distress and within_horizon)

                X_data.append(window_tensor)
                Y_primary.append(window_label)
                # NEW: patient-level MIL target -- the pH outcome itself, with no
                # horizon logic, broadcast to every window of this patient.
                Y_patient.append(int(is_distress))
                Y_figo.append(int(figo_class))
                Y_features.append(features_target)
                Record_ids.append((str(record_id), int(start), int(end)))

        if len(X_data) == 0:
            print(f"  [WARNING] No windows extracted for {split_name} split.")
            continue

        ds = {
            'X': torch.tensor(np.array(X_data), dtype=torch.float32),
            'y_primary': torch.tensor(np.array(Y_primary), dtype=torch.long),
            'y_patient': torch.tensor(np.array(Y_patient), dtype=torch.long),
            'y_figo': torch.tensor(np.array(Y_figo), dtype=torch.long),
            'y_features': torch.tensor(np.array(Y_features), dtype=torch.float32),
            'metadata': Record_ids,
        }
        out_path = os.path.join(output_dir, f'{split_name}_dataset.pt')
        torch.save(ds, out_path)

        n_pat_pos = int(torch.tensor(Y_patient).sum())
        print(f"  Saved {split_name}_dataset.pt - {len(X_data)} windows from {n_patients_kept} patients")
        print(f"    Skipped: {n_skipped_missing} missing .dat, {n_skipped_no_data} empty, "
              f"{n_excluded_quality} patient-quality")
        print(f"    Window labels : Normal={int((ds['y_primary']==0).sum())}  "
              f"Distress={int((ds['y_primary']==1).sum())}")
        print(f"    Patient labels: {n_pat_pos} distress windows-of-distress-patients "
              f"({100*n_pat_pos/len(X_data):.1f}% of windows)")

    # -----------------------------------------------------------------------
    # Global per-channel Z-score normalisation, fit on TRAIN only (as pipeline.py)
    # -----------------------------------------------------------------------
    print("\nApplying per-channel Z-score normalisation (fit on train split)...")
    train_path = os.path.join(output_dir, 'train_dataset.pt')
    if not os.path.exists(train_path):
        print("  [ERROR] train_dataset.pt not found -- cannot compute scaler.")
        return

    train_ds = torch.load(train_path, weights_only=False)
    _, train_mean, train_std = zscore_normalize_channels(train_ds['X'].numpy())

    scaler_path = os.path.join(output_dir, 'ctu_signal_scaler.npz')
    np.savez(scaler_path, mean=train_mean, std=train_std)
    print(f"  Scaler saved -> {scaler_path}")
    print(f"    Channel means : FHR={train_mean[0]:.4f}  UC={train_mean[1]:.4f}")
    print(f"    Channel stds  : FHR={train_std[0]:.4f}   UC={train_std[1]:.4f}")

    for split_name in ['train', 'val', 'test']:
        pt_path = os.path.join(output_dir, f'{split_name}_dataset.pt')
        if not os.path.exists(pt_path):
            continue
        ds = torch.load(pt_path, weights_only=False)
        X_norm, _, _ = zscore_normalize_channels(ds['X'].numpy(), mean=train_mean, std=train_std)
        ds['X'] = torch.tensor(X_norm, dtype=torch.float32)
        torch.save(ds, pt_path)
        print(f"  Normalised and saved {split_name}_dataset.pt")

    # -----------------------------------------------------------------------
    # Feature scaler for Model 8 / KG-MIL knowledge losses (train split only)
    # -----------------------------------------------------------------------
    train_ds = torch.load(train_path, weights_only=False)
    yf = train_ds['y_features'].numpy()
    feature_means = yf.mean(axis=0)
    feature_stds = np.clip(yf.std(axis=0), 1e-6, None)
    feat_scaler_path = os.path.join(output_dir, 'feature_scaler.npz')
    np.savez(feat_scaler_path, feature_means=feature_means, feature_stds=feature_stds)
    print(f"  Feature scaler saved -> {feat_scaler_path}")

    print("\nMIL substrate pipeline complete.")
    print("NEXT: run scripts/audit_bag_size_leak.py -- AUROC(window count -> patient")
    print("label) must be ~0.5 before any patient-level modelling proceeds.")


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'ctu-chb-intrapartum')
    METADATA_FILE = os.path.join(RAW_DIR, 'clinical_metadata.csv')
    OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed_mil')
    process_pipeline_mil(RAW_DIR, METADATA_FILE, OUT_DIR)
