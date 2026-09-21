"""
Stride experiment pilot -- Stage B.

Rebuilds ONLY the P2 preprocessing candidate (the one phase8_rolling_inference.py
and phase9_temporal_features.py actually consume as P2_PATH) at the pilot's
1.0-minute stride, reusing the exact same per-window P2 signal chain as
scripts/phase1_candidates_builder.py (spike removal -> interpolation ->
lowpass -> rolling iterative baseline -> 4-channel stack). P0/P1 are skipped
since nothing downstream in this experiment reads them -- building them too
would triple the runtime for no reason here.

Uses the SAME cohort (data/processed_clinical/folds.json's clean_pids) as the
canonical 2.5-min pipeline, so both stride arms compare identical patients.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
_PREPROC_DIR = os.path.join(BASE_DIR, "src", "preprocessing")
if _PREPROC_DIR not in sys.path:
    sys.path.insert(0, _PREPROC_DIR)

from src.preprocessing.ingestion import load_ctu_chb_record, load_clinical_metadata, TARGET_FS
from src.preprocessing.signal_quality import get_valid_windows
from src.preprocessing.filtering import remove_spikes, interpolate_missing, apply_lowpass_filter
from src.preprocessing.baseline import rolling_iterative_baseline

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "ctu-chb-intrapartum")
METADATA_PATH = os.path.join(RAW_DIR, "clinical_metadata.csv")
FOLDS_PATH = os.path.join(BASE_DIR, "data", "processed_clinical", "folds.json")  # canonical, unchanged
OUT_DIR = os.path.join(BASE_DIR, "data", "phase1_candidates_stride1p0")

WINDOW_MINUTES = 20
LAST_HOUR_MINUTES = 60
STRIDE_MINUTES = 1.0
WINDOW_SAMPLES = int(WINDOW_MINUTES * 60 * TARGET_FS)
LAST_HOUR_SAMPLES = int(LAST_HOUR_MINUTES * 60 * TARGET_FS)
STRIDE_SAMPLES = int(STRIDE_MINUTES * 60 * TARGET_FS)
MAX_MISSING_RATIO = 0.5
FHR_MIN_BPM = 50.0
FHR_MAX_BPM = 240.0


def build_p2_only():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== STRIDE PILOT STAGE B: P2 candidate @ {STRIDE_MINUTES}-min stride "
          f"(window={WINDOW_MINUTES}min) ===")

    metadata = load_clinical_metadata(METADATA_PATH)
    if 'record_id' in metadata.columns:
        metadata = metadata.set_index('record_id')

    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    print(f"Cohort: {len(clean_pids)} patients from {FOLDS_PATH}")

    p2_windows = []
    y_labels = []
    patient_ids = []
    window_meta = []

    for n_done, pid in enumerate(clean_pids):
        rec_path = os.path.join(RAW_DIR, str(pid))
        fhr_full, uc_full, fs = load_ctu_chb_record(rec_path)
        ph_val = float(metadata.loc[int(pid), 'ph'])
        y_label = int(ph_val <= 7.15)

        full_len = len(fhr_full)
        if full_len > LAST_HOUR_SAMPLES:
            fhr = fhr_full[-LAST_HOUR_SAMPLES:]
            uc = uc_full[-LAST_HOUR_SAMPLES:]
        else:
            fhr = fhr_full.copy()
            uc = uc_full.copy()

        starts = get_valid_windows(fhr, WINDOW_SAMPLES, STRIDE_SAMPLES, max_missing_ratio=MAX_MISSING_RATIO)
        for start in starts:
            end = start + WINDOW_SAMPLES
            raw_fhr = fhr[start:end].copy()
            raw_uc = uc[start:end].copy()
            obs_mask = (raw_fhr > 0.0).astype(np.float32)

            f_p2_spikes = remove_spikes(raw_fhr.copy(), fs=fs)
            f_p2_interp = interpolate_missing(f_p2_spikes.copy(), clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
            f_p2 = apply_lowpass_filter(f_p2_interp, fs=fs)
            b_p2 = rolling_iterative_baseline(f_p2, fs=fs, win_sec=300.0, step_sec=30.0, round_to_5=False)

            interp_mask = ((raw_fhr > 0.0) & (f_p2_spikes > 0.0)).astype(np.float32)

            p2_ch0 = f_p2
            p2_ch1 = f_p2 - b_p2
            p2_ch2 = obs_mask
            p2_ch3 = interp_mask
            p2_arr = np.stack([p2_ch0, p2_ch1, p2_ch2, p2_ch3], axis=0).astype(np.float32)
            p2_windows.append(p2_arr)

            y_labels.append(y_label)
            patient_ids.append(str(pid))
            window_meta.append((str(pid), int(start), int(end)))

        if (n_done + 1) % 100 == 0:
            print(f"  ...{n_done + 1}/{len(clean_pids)} patients, {len(y_labels)} windows so far")

    p2_X = np.stack(p2_windows, axis=0)
    y_arr = np.array(y_labels, dtype=np.int64)
    pid_arr = np.array(patient_ids)

    print(f"Extracted {len(y_arr)} windows from {len(clean_pids)} patients.")
    print(f"P2 Shape: {p2_X.shape}")

    torch.save({"X": torch.tensor(p2_X), "y": torch.tensor(y_arr), "pid": pid_arr, "meta": window_meta},
               os.path.join(OUT_DIR, "p2_dataset.pt"))
    print(f"Successfully saved -> {OUT_DIR}/p2_dataset.pt")


if __name__ == "__main__":
    build_p2_only()
