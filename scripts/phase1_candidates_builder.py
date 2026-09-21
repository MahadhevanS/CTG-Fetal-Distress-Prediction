"""
Builds candidate preprocessed datasets P0, P1, and P2 for Phase 1.

P0: Current Pipeline (Control)
    - Spike removal -> cubic spline <=15s clamped [50, 240] -> 1.5Hz lowpass
    - Constant iterative baseline (±15 bpm exclusion, 5 bpm rounding)
    - Channels: [FHR - B, UC, missing_mask] (train-fit Z-score on ch 0, 1)

P1: Minimal Physiological Cleaning
    - Range bound [50, 240] & spike removal
    - Short-gap interpolation (<= 5s) only; gaps >5s kept missing
    - Rolling iterative baseline without aggressive lowpass filtering
    - Channels: [Raw FHR, Delta FHR, observed_mask] (train-fit Z-score on ch 0, 1)

P2: Quality-Aware Preprocessing
    - Spike removal & cubic spline <=15s clamped [50, 240]
    - 1.5Hz zero-phase lowpass
    - Rolling iterative baseline calculated within window
    - Explicit 4 Channels: [Raw FHR, Delta FHR, observed_mask, interp_mask]
    - Train-fit Z-score on ch 0, 1
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.ingestion import load_ctu_chb_record, load_clinical_metadata, TARGET_FS
from src.preprocessing.signal_quality import assess_patient_missing_ratio, get_valid_windows
from src.preprocessing.filtering import remove_spikes, interpolate_missing, apply_lowpass_filter, zscore_normalize_channels
from src.preprocessing.baseline import calculate_iterative_baseline, rolling_iterative_baseline
from src.training.protocol import Protocol

RAW_DIR = "data/raw/ctu-chb-intrapartum"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
FOLDS_PATH = "data/processed_clinical/folds.json"
OUT_DIR = "data/phase1_candidates"

WINDOW_MINUTES = 20
LAST_HOUR_MINUTES = 60
STRIDE_MINUTES = 2.5
WINDOW_SAMPLES = int(WINDOW_MINUTES * 60 * TARGET_FS)        # 4800
LAST_HOUR_SAMPLES = int(LAST_HOUR_MINUTES * 60 * TARGET_FS)  # 14400
STRIDE_SAMPLES = int(STRIDE_MINUTES * 60 * TARGET_FS)        # 600
MAX_MISSING_RATIO = 0.5
FHR_MIN_BPM = 50.0
FHR_MAX_BPM = 240.0


def build_candidates():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== BUILDING PREPROCESSING CANDIDATES P0, P1, P2 ===")

    metadata = load_clinical_metadata(METADATA_PATH)
    if 'record_id' in metadata.columns:
        metadata = metadata.set_index('record_id')

    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    print(f"Cohort: {len(clean_pids)} patients from {FOLDS_PATH}")

    # Containers for all windows
    p0_windows = []
    p1_windows = []
    p2_windows = []
    y_labels = []
    patient_ids = []
    window_meta = []

    for pid in clean_pids:
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

            # Observed mask (1 = observed, 0 = missing)
            obs_mask = (raw_fhr > 0.0).astype(np.float32)
            missing_mask = (raw_fhr == 0.0).astype(np.float32)

            # -------------------------------------------------------------
            # P0: Current Pipeline
            # -------------------------------------------------------------
            f_p0 = remove_spikes(raw_fhr.copy(), fs=fs)
            f_p0 = interpolate_missing(f_p0, clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
            f_p0 = apply_lowpass_filter(f_p0, fs=fs)
            u_p0 = apply_lowpass_filter(raw_uc.copy(), fs=fs)
            b_p0 = calculate_iterative_baseline(f_p0)
            
            p0_ch0 = f_p0 - b_p0
            p0_ch1 = u_p0
            p0_ch2 = missing_mask
            p0_arr = np.stack([p0_ch0, p0_ch1, p0_ch2], axis=0).astype(np.float32)
            p0_windows.append(p0_arr)

            # -------------------------------------------------------------
            # P1: Minimal Physiological Cleaning
            # -------------------------------------------------------------
            f_p1 = raw_fhr.copy()
            f_p1[(f_p1 < FHR_MIN_BPM) | (f_p1 > FHR_MAX_BPM)] = 0.0
            f_p1 = remove_spikes(f_p1, fs=fs)
            # Short gap interpolation <= 5s (20 samples)
            f_p1 = interpolate_missing(f_p1, max_gap_samples=20, clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
            b_p1 = rolling_iterative_baseline(f_p1, fs=fs, win_sec=300.0, step_sec=30.0, round_to_5=False)
            
            p1_ch0 = f_p1
            p1_ch1 = f_p1 - b_p1
            p1_ch2 = obs_mask
            p1_arr = np.stack([p1_ch0, p1_ch1, p1_ch2], axis=0).astype(np.float32)
            p1_windows.append(p1_arr)

            # -------------------------------------------------------------
            # P2: Quality-Aware Multi-Channel Preprocessing
            # -------------------------------------------------------------
            f_p2_spikes = remove_spikes(raw_fhr.copy(), fs=fs)
            f_p2_interp = interpolate_missing(f_p2_spikes.copy(), clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
            f_p2 = apply_lowpass_filter(f_p2_interp, fs=fs)
            b_p2 = rolling_iterative_baseline(f_p2, fs=fs, win_sec=300.0, step_sec=30.0, round_to_5=False)
            
            # Interpolation mask: 1 for observed valid sample, 0 for interpolated or missing
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

    p0_X = np.stack(p0_windows, axis=0)
    p1_X = np.stack(p1_windows, axis=0)
    p2_X = np.stack(p2_windows, axis=0)
    y_arr = np.array(y_labels, dtype=np.int64)
    pid_arr = np.array(patient_ids)

    print(f"Extracted {len(y_arr)} windows from {len(clean_pids)} patients.")
    print(f"P0 Shape: {p0_X.shape}")
    print(f"P1 Shape: {p1_X.shape}")
    print(f"P2 Shape: {p2_X.shape}")

    # Save candidate datasets
    torch.save({"X": torch.tensor(p0_X), "y": torch.tensor(y_arr), "pid": pid_arr, "meta": window_meta},
               os.path.join(OUT_DIR, "p0_dataset.pt"))
    torch.save({"X": torch.tensor(p1_X), "y": torch.tensor(y_arr), "pid": pid_arr, "meta": window_meta},
               os.path.join(OUT_DIR, "p1_dataset.pt"))
    torch.save({"X": torch.tensor(p2_X), "y": torch.tensor(y_arr), "pid": pid_arr, "meta": window_meta},
               os.path.join(OUT_DIR, "p2_dataset.pt"))

    print(f"Successfully saved candidate datasets to {OUT_DIR}/")


if __name__ == "__main__":
    build_candidates()
