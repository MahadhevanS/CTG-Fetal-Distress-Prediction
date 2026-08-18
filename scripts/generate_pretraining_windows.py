"""
Generates a dense, unlabeled window pool for self-supervised CTG-CrossFormer
pretraining -- see docs/model8_crossformer_run_history.md (2026-08-17 entry)
and the associated plan.

Reuses the exact per-window signal-processing chain
src/preprocessing/pipeline.py::process_pipeline() applies (spike removal,
interpolation, low-pass filter, baseline correction), but calls
get_valid_windows() directly with a small, uniform, label-agnostic stride
instead of pipeline.py's label-conditional stride/horizon-relabeling -- SSL
needs no labels, only clean windows. Excludes the 82 held-out test-set
patients (same split pipeline.py's own train_test_split already assigns,
replicated here) so the test set stays untouched by any part of pretraining.
Reuses the already-fit ctu_signal_scaler.npz for normalization -- does not refit.

Usage:
    python scripts/generate_pretraining_windows.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREPROCESSING_DIR = os.path.join(BASE_DIR, "src", "preprocessing")
if PREPROCESSING_DIR not in sys.path:
    # src/preprocessing/*.py use bare imports (`from ingestion import ...`),
    # designed to run with this directory itself on sys.path (Python does
    # this automatically for pipeline.py since it's normally invoked
    # directly as the entry point; this script is the entry point instead,
    # so it must add src/preprocessing/ explicitly).
    sys.path.insert(0, PREPROCESSING_DIR)
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from ingestion import load_ctu_chb_record, load_clinical_metadata
from filtering import remove_spikes, interpolate_missing, apply_lowpass_filter, zscore_normalize_channels
from baseline import calculate_iterative_baseline
from signal_quality import get_valid_windows

WINDOW_MINUTES = 20
LAST_HOUR_MINUTES = 60
TARGET_FS = 4.0
WINDOW_SAMPLES = int(WINDOW_MINUTES * 60 * TARGET_FS)          # 4800
LAST_HOUR_SAMPLES = int(LAST_HOUR_MINUTES * 60 * TARGET_FS)    # 14400
PRETRAIN_STRIDE_SAMPLES = int(30 * TARGET_FS)                  # 30s, 120 samples

FHR_MIN_BPM = 50.0
FHR_MAX_BPM = 240.0


def get_test_patient_ids(metadata: pd.DataFrame) -> set:
    """Replicates pipeline.py's own train/val/test split exactly (same
    random_state=42, same two-stage train_test_split) so pretraining can
    exclude precisely the same 82 patients pipeline.py reserves as the
    held-out test set."""
    records = list(metadata.index)
    y_patient = (metadata["ph"] <= 7.15).astype(int).values
    _, test_val_ids, _, y_test_val = train_test_split(
        records, y_patient, test_size=0.3, stratify=y_patient, random_state=42
    )
    _, test_ids, _, _ = train_test_split(
        test_val_ids, y_test_val, test_size=0.5, stratify=y_test_val, random_state=42
    )
    return set(test_ids)


def main():
    raw_dir = os.path.join(BASE_DIR, "data", "raw", "ctu-chb-intrapartum")
    metadata_path = os.path.join(raw_dir, "clinical_metadata.csv")
    output_path = os.path.join(BASE_DIR, "data", "processed", "pretrain_windows.pt")
    scaler_path = os.path.join(BASE_DIR, "data", "processed", "ctu_signal_scaler.npz")

    print("Loading metadata...")
    metadata = load_clinical_metadata(metadata_path)
    metadata = metadata.set_index("record_id") if "record_id" in metadata.columns else metadata

    test_ids = get_test_patient_ids(metadata)
    print(f"Excluding {len(test_ids)} held-out test-set patients from the pretraining corpus.")

    records = [r for r in metadata.index if r not in test_ids]
    print(f"Generating dense unlabeled windows for {len(records)} patients "
          f"(stride={PRETRAIN_STRIDE_SAMPLES} samples = 30s)...")

    X_data = []
    Record_ids = []
    n_skipped_missing = 0
    n_skipped_no_data = 0

    for record_id in records:
        record_path = os.path.join(raw_dir, str(record_id))
        if not os.path.exists(record_path + ".dat"):
            n_skipped_missing += 1
            continue

        fhr, uc, fs = load_ctu_chb_record(record_path)
        if len(fhr) == 0:
            n_skipped_no_data += 1
            continue

        if len(fhr) > LAST_HOUR_SAMPLES:
            fhr = fhr[-LAST_HOUR_SAMPLES:]
            uc = uc[-LAST_HOUR_SAMPLES:]

        valid_starts = get_valid_windows(fhr, WINDOW_SAMPLES, PRETRAIN_STRIDE_SAMPLES)

        for start in valid_starts:
            end = start + WINDOW_SAMPLES
            fhr_win = fhr[start:end].copy()
            uc_win = uc[start:end].copy()

            fhr_win = remove_spikes(fhr_win, fs=fs)
            fhr_win = interpolate_missing(fhr_win, clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
            fhr_win = apply_lowpass_filter(fhr_win, fs=fs)
            uc_win = apply_lowpass_filter(uc_win, fs=fs)

            baseline = calculate_iterative_baseline(fhr_win)
            fhr_norm = fhr_win - baseline

            window_tensor = np.vstack((fhr_norm, uc_win))
            X_data.append(window_tensor)
            Record_ids.append((str(record_id), int(start), int(end)))

    print(f"\nExtracted {len(X_data)} raw candidate windows "
          f"(skipped {n_skipped_missing} missing records, {n_skipped_no_data} empty records).")

    X = np.array(X_data, dtype=np.float32)

    print("Applying pre-fit signal scaler (not refitting)...")
    scaler = np.load(scaler_path)
    X_norm, _, _ = zscore_normalize_channels(X, mean=scaler["mean"], std=scaler["std"])

    ds = {
        "X": torch.tensor(X_norm, dtype=torch.float32),
        "metadata": Record_ids,
    }
    torch.save(ds, output_path)
    print(f"Saved {output_path} -- {len(X_data)} windows, {len(set(r[0] for r in Record_ids))} patients.")


if __name__ == "__main__":
    main()
