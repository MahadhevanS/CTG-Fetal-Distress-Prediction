import os
import numpy as np
import pandas as pd
import torch

# Preprocessing modules
from ingestion import load_ctu_chb_record, load_clinical_metadata, TARGET_FS
from signal_quality import get_valid_windows, assess_patient_missing_ratio
from filtering import (
    interpolate_missing_linear,
    zscore_normalize_channels
)
from baseline import calculate_iterative_baseline
from features import calculate_variability, detect_accelerations, detect_decelerations
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knowledge.figo import classify_figo


# ---------------------------------------------------------------------------
# Paper-Matched Constants
# Reproduces Dang, Nguyen, Ho, "A Hybrid CNN-Transformer with Cross-Attention
# for Automated Fetal Distress Detection from Cardiotocography," E3S Web of
# Conferences 723, 01005 (2026), Section 3.1.
# ---------------------------------------------------------------------------

WINDOW_MINUTES = 20   # Same window length as pipeline.py -- not one of the
                      # identified differences, only stride/truncation differ.
WINDOW_SAMPLES = int(WINDOW_MINUTES * 60 * TARGET_FS)   # 4800

STRIDE_MINUTES = 5    # Paper: uniform 5-minute stride for every patient.
STRIDE_SAMPLES = int(STRIDE_MINUTES * 60 * TARGET_FS)   # 1200

PATIENT_MAX_MISSING_RATIO = 0.5   # Paper: exclude whole recordings >50% missing FHR.
WINDOW_MAX_MISSING_RATIO = 0.3    # Unchanged from pipeline.py's own per-window gate.

FHR_MIN_BPM = 50.0
FHR_MAX_BPM = 240.0


def process_pipeline_paper_match(raw_data_dir: str, metadata_path: str, output_dir: str) -> None:
    """
    Paper-matched preprocessing pipeline. Diverges from pipeline.py in exactly
    the ways identified against the paper's Methods section:
      - Patient-level exclusion (>50% missing FHR) before windowing.
      - No truncation to a "last hour" horizon.
      - Uniform 5-minute stride for every patient (no label-conditional stride).
      - No prediction-horizon relabeling -- every window from an Acidosis-
        outcome patient inherits that label directly.
      - Linear (not cubic-spline) interpolation for gaps <=15s.
      - Per-recording z-score normalization (not a single global scaler).
      - No train/val/test split -- a single pooled dataset, matching the
        paper's "5-fold CV over the whole cohort" evaluation protocol.
      - Channel 0 of X is gap-filled RAW FHR, not baseline-corrected (paper's
        preprocessing section does not mention baseline subtraction into the
        model input -- taken literally per user decision). baseline is still
        computed per window because y_figo/y_features (for Model 8
        compatibility) require it regardless of this choice.

    Args:
        raw_data_dir (str): Path to the CTU-CHB raw data directory
                            (contains .dat/.hea files).
        metadata_path (str): Path to clinical_metadata.csv.
        output_dir (str): Directory to write train_dataset.pt + feature_scaler.npz.
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

    records = list(metadata.index)

    X_data = []
    Y_primary = []
    Y_figo = []
    Y_features = []
    Record_ids = []

    n_total = len(records)
    n_skipped_missing = 0
    n_skipped_no_data = 0
    n_excluded_patient_quality = 0
    n_surviving_patients = 0

    print(f"\nProcessing {n_total} patients (paper-matched protocol)...")

    for record_id in records:
        record_path = os.path.join(raw_data_dir, str(record_id))

        if not os.path.exists(record_path + '.dat'):
            n_skipped_missing += 1
            continue

        fhr, uc, fs = load_ctu_chb_record(record_path)
        if len(fhr) == 0:
            n_skipped_no_data += 1
            continue

        # --- Patient-level quality gate (paper difference #1) ---
        # Evaluated on the raw, full-length, untruncated signal.
        if not assess_patient_missing_ratio(fhr, max_missing_ratio=PATIENT_MAX_MISSING_RATIO):
            n_excluded_patient_quality += 1
            continue

        is_distress = bool(metadata.loc[record_id, 'ph'] <= 7.15)

        # --- No truncation to a "last hour" horizon (paper difference #2) ---
        # Uses the recording's full available length.

        # --- Uniform stride for every patient (paper difference #2) ---
        valid_starts = get_valid_windows(fhr, WINDOW_SAMPLES, STRIDE_SAMPLES,
                                          max_missing_ratio=WINDOW_MAX_MISSING_RATIO)

        if len(valid_starts) == 0:
            continue

        n_surviving_patients += 1
        record_windows = 0

        for start in valid_starts:
            end = start + WINDOW_SAMPLES
            fhr_win = fhr[start:end].copy()
            uc_win = uc[start:end].copy()

            # --- Linear interpolation, gaps <=15s only (paper difference #5) ---
            fhr_win = interpolate_missing_linear(fhr_win, clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
            uc_win = interpolate_missing_linear(uc_win)

            # baseline always computed -- needed for y_figo/y_features
            # regardless of the channel-0 construction choice below.
            baseline = calculate_iterative_baseline(fhr_win)

            # --- Channel 0 = raw (gap-filled) FHR, not baseline-corrected
            # (paper difference #6, literal-match decision) ---
            fhr_channel = fhr_win

            stv, ltv = calculate_variability(fhr_win, fs=fs, baseline=baseline)
            accels = detect_accelerations(fhr_win, baseline, fs=fs)
            decels = detect_decelerations(fhr_win, baseline, uc_win, fs=fs)

            base_val = float(np.mean(baseline))
            figo_class = classify_figo(base_val, ltv, accels, decels)

            window_tensor = np.vstack((fhr_channel, uc_win))

            features_target = [
                base_val,
                stv, ltv,
                float(accels),
                float(decels['early']),
                float(decels['late']),
                float(decels['variable']),
                float(decels['prolonged']),
            ]

            # --- No horizon relabeling (paper difference #3) ---
            label = int(is_distress)

            X_data.append(window_tensor)
            Y_primary.append(label)
            Y_figo.append(int(figo_class))
            Y_features.append(features_target)
            Record_ids.append((str(record_id), int(start), int(end)))
            record_windows += 1

        # --- Per-recording z-score normalization (paper difference #4) ---
        # Applied to just this record's own windows, fit on those windows alone.
        if record_windows > 0:
            X_record = np.array(X_data[-record_windows:])
            X_record_norm, _, _ = zscore_normalize_channels(X_record, mean=None, std=None)
            X_data[-record_windows:] = list(X_record_norm)

    print(f"\nPatients scanned: {n_total}")
    print(f"  Skipped (missing .dat): {n_skipped_missing}")
    print(f"  Skipped (empty signal): {n_skipped_no_data}")
    print(f"  Excluded (patient-level >{int(PATIENT_MAX_MISSING_RATIO*100)}% missing): {n_excluded_patient_quality}")
    print(f"  Surviving patients (with >=1 valid window): {n_surviving_patients}")

    if len(X_data) == 0:
        print("  [WARNING] No windows extracted -- aborting.")
        return

    ds = {
        'X': torch.tensor(np.array(X_data), dtype=torch.float32),
        'y_primary': torch.tensor(np.array(Y_primary), dtype=torch.long),
        'y_figo': torch.tensor(np.array(Y_figo), dtype=torch.long),
        'y_features': torch.tensor(np.array(Y_features), dtype=torch.float32),
        'metadata': Record_ids,
    }
    out_path = os.path.join(output_dir, 'train_dataset.pt')
    torch.save(ds, out_path)

    n_normal = int((ds['y_primary'] == 0).sum())
    n_distress = int((ds['y_primary'] == 1).sum())
    n_total_windows = n_normal + n_distress
    print(f"\nSaved train_dataset.pt - {n_total_windows} windows")
    print(f"  Normal={n_normal} ({100*n_normal/n_total_windows:.1f}%)  "
          f"Distress={n_distress} ({100*n_distress/n_total_windows:.1f}%)")
    print(f"  (Paper reference: 404 patients, 1,753 windows, "
          f"83.4% Normal / 16.6% Distress)")

    # -----------------------------------------------------------------------
    # Feature scaler for Model 8's FIGO rule loss (inline, matches what
    # scripts/generate_feature_scaler.py computes, but that script hardcodes
    # data/processed/ with no CLI args)
    # -----------------------------------------------------------------------
    y_features_np = ds['y_features'].numpy()
    feature_means = y_features_np.mean(axis=0)
    feature_stds = np.clip(y_features_np.std(axis=0), 1e-6, None)
    scaler_path = os.path.join(output_dir, 'feature_scaler.npz')
    np.savez(scaler_path, feature_means=feature_means, feature_stds=feature_stds)
    print(f"  Feature scaler saved -> {scaler_path}")

    print("\nNOTE: no ctu_signal_scaler.npz-equivalent is saved -- normalization")
    print("was fit per-recording, so there is no single global (mean, std) to")
    print("persist for future single-recording inference. Known, accepted")
    print("limitation for this CV-only reproduction goal.")

    print("\nCTU-CHB paper-matched preprocessing pipeline complete.")


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'ctu-chb-intrapartum')
    METADATA_FILE = os.path.join(RAW_DIR, 'clinical_metadata.csv')
    OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed_paper_match')
    process_pipeline_paper_match(RAW_DIR, METADATA_FILE, OUT_DIR)
