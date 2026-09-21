"""
Phase 5 Dataset Builder: Multi-Resolution 1D Temporal Context & Clinical Descriptors.

Extracts:
1. Single-scale candidate datasets: 5m, 10m, 20m, 30m, 40m, 60m.
2. Causal matched-endpoint multi-resolution dataset (10m, 20m, 40m terminating at the exact same point t_end).
3. Multi-scale clinical descriptors (10m, 20m, 40m) and physiological trend features.
4. Comprehensive missingness and recording duration audit metadata.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.ingestion import load_ctu_chb_record, load_clinical_metadata, TARGET_FS
from src.preprocessing.signal_quality import assess_patient_missing_ratio, get_valid_windows
from src.preprocessing.filtering import remove_spikes, interpolate_missing, apply_lowpass_filter
from src.preprocessing.baseline import rolling_iterative_baseline
from src.preprocessing.features import calculate_variability, detect_accelerations, detect_decelerations
from src.knowledge.extended_features import extract_extended_features

RAW_DIR = "data/raw/ctu-chb-intrapartum"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
FOLDS_PATH = "data/processed_clinical/folds.json"
OUT_DIR = "data/phase5_multires"

FS = 4.0
LAST_HOUR_SAMPLES = int(60 * 60 * FS)  # 14400 samples
STRIDE_SAMPLES = int(2.5 * 60 * FS)    # 600 samples
MAX_MISSING_RATIO = 0.5
FHR_MIN_BPM = 50.0
FHR_MAX_BPM = 240.0

SCALES_MINUTES = {
    "5m": 5,
    "10m": 10,
    "20m": 20,
    "30m": 30,
    "40m": 40,
    "60m": 60
}


def preprocess_p2_window(raw_fhr: np.ndarray, raw_uc: np.ndarray, fs: float = FS):
    """
    Applies Phase 1 P2 Quality-Aware Preprocessing.
    Returns 4-channel array (4, T): [Raw FHR, Delta FHR, Observed Mask, Interp Mask]
    """
    obs_mask = (raw_fhr > 0.0).astype(np.float32)
    f_spikes = remove_spikes(raw_fhr.copy(), fs=fs)
    f_interp = interpolate_missing(f_spikes.copy(), clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
    f_clean = apply_lowpass_filter(f_interp, fs=fs)
    b_roll = rolling_iterative_baseline(f_clean, fs=fs, win_sec=300.0, step_sec=30.0, round_to_5=False)
    interp_mask = ((raw_fhr > 0.0) & (f_spikes > 0.0)).astype(np.float32)

    ch0 = f_clean
    ch1 = f_clean - b_roll
    ch2 = obs_mask
    ch3 = interp_mask
    return np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32), f_clean, b_roll


def extract_19_descriptors(f_clean: np.ndarray, b_roll: np.ndarray, raw_uc: np.ndarray, fs: float = FS):
    """
    Extracts 19 clinical physiological descriptors from a window.
    """
    u_clean = apply_lowpass_filter(raw_uc.copy(), fs=fs)
    stv, ltv = calculate_variability(f_clean, fs=fs, baseline=b_roll)
    accels = detect_accelerations(f_clean, b_roll, fs=fs)
    decels = detect_decelerations(f_clean, b_roll, u_clean, fs=fs)
    base_val = float(np.mean(b_roll))

    base8 = [
        base_val,
        float(stv),
        float(ltv),
        float(accels),
        float(decels["early"]),
        float(decels["late"]),
        float(decels["variable"]),
        float(decels["prolonged"])
    ]
    ext11 = extract_extended_features(f_clean - b_roll, u_clean, base_val, fs=fs)
    return np.concatenate([np.array(base8, dtype=np.float32), ext11.astype(np.float32)])


def build_phase5_datasets():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== STARTING PHASE 5 MULTI-RESOLUTION DATASET EXTRACTION ===")
    t0 = time.time()

    metadata = load_clinical_metadata(METADATA_PATH)
    if 'record_id' in metadata.columns:
        metadata = metadata.set_index('record_id')

    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    print(f"Loaded {len(clean_pids)} patients from {FOLDS_PATH}")

    # Scale containers
    scale_datasets = {k: {"X": [], "y": [], "pid": [], "meta": []} for k in SCALES_MINUTES}

    # Matched Multi-Resolution Containers (10m, 20m, 40m matched to exact same t_end)
    matched_data = {
        "X_10m": [],
        "X_20m": [],
        "X_40m": [],
        "F_10m": [],
        "F_20m": [],
        "F_40m": [],
        "F_trends": [],
        "y": [],
        "pid": [],
        "meta": [],
        "time_to_delivery_min": []
    }

    audit_stats = []

    for idx, pid in enumerate(clean_pids):
        rec_path = os.path.join(RAW_DIR, str(pid))
        fhr_full, uc_full, fs = load_ctu_chb_record(rec_path)
        ph_val = float(metadata.loc[int(pid), 'ph'])
        y_label = int(ph_val <= 7.15)

        full_len = len(fhr_full)
        if full_len > LAST_HOUR_SAMPLES:
            fhr = fhr_full[-LAST_HOUR_SAMPLES:].copy()
            uc = uc_full[-LAST_HOUR_SAMPLES:].copy()
        else:
            fhr = fhr_full.copy()
            uc = uc_full.copy()

        rec_dur_min = len(fhr) / (FS * 60.0)
        overall_missing = float(np.mean(fhr == 0.0))

        audit_stats.append({
            "pid": str(pid),
            "total_duration_min": len(fhr_full) / (FS * 60.0),
            "last_hour_duration_min": rec_dur_min,
            "missing_fraction": overall_missing,
            "y": y_label,
            "ph": ph_val
        })

        # 1. Single scale extractions
        for scale_name, dur_min in SCALES_MINUTES.items():
            win_samples = int(dur_min * 60 * FS)
            starts = get_valid_windows(fhr, win_samples, STRIDE_SAMPLES, max_missing_ratio=MAX_MISSING_RATIO)
            for st in starts:
                en = st + win_samples
                raw_f = fhr[st:en]
                raw_u = uc[st:en]
                x_p2, _, _ = preprocess_p2_window(raw_f, raw_u, fs=FS)
                scale_datasets[scale_name]["X"].append(x_p2)
                scale_datasets[scale_name]["y"].append(y_label)
                scale_datasets[scale_name]["pid"].append(str(pid))
                scale_datasets[scale_name]["meta"].append((str(pid), int(st), int(en)))

        # 2. Matched Endpoint Multi-Scale Extractions (10m = 2400, 20m = 4800, 40m = 9600)
        win_40m = int(40 * 60 * FS)  # 9600
        win_20m = int(20 * 60 * FS)  # 4800
        win_10m = int(10 * 60 * FS)  # 2400

        starts_40m = get_valid_windows(fhr, win_40m, STRIDE_SAMPLES, max_missing_ratio=MAX_MISSING_RATIO)
        for st40 in starts_40m:
            en = st40 + win_40m
            st20 = en - win_20m
            st10 = en - win_10m

            # Verify quality for all 3 sub-windows ending at en
            if np.mean(fhr[st20:en] == 0.0) > MAX_MISSING_RATIO or np.mean(fhr[st10:en] == 0.0) > MAX_MISSING_RATIO:
                continue

            # Process 10m
            x10, f10, b10 = preprocess_p2_window(fhr[st10:en], uc[st10:en], fs=FS)
            feat10 = extract_19_descriptors(f10, b10, uc[st10:en], fs=FS)

            # Process 20m
            x20, f20, b20 = preprocess_p2_window(fhr[st20:en], uc[st20:en], fs=FS)
            feat20 = extract_19_descriptors(f20, b20, uc[st20:en], fs=FS)

            # Process 40m
            x40, f40, b40 = preprocess_p2_window(fhr[st40:en], uc[st40:en], fs=FS)
            feat40 = extract_19_descriptors(f40, b40, uc[st40:en], fs=FS)

            # Compute trend features: slope = (feat40 - feat10) / 30 min
            delta_t_min = 30.0
            trends = (feat40 - feat10) / delta_t_min

            time_to_delivery = (len(fhr) - en) / (FS * 60.0)

            matched_data["X_10m"].append(x10)
            matched_data["X_20m"].append(x20)
            matched_data["X_40m"].append(x40)
            matched_data["F_10m"].append(feat10)
            matched_data["F_20m"].append(feat20)
            matched_data["F_40m"].append(feat40)
            matched_data["F_trends"].append(trends)
            matched_data["y"].append(y_label)
            matched_data["pid"].append(str(pid))
            matched_data["meta"].append((str(pid), int(st40), int(en)))
            matched_data["time_to_delivery_min"].append(time_to_delivery)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(clean_pids):
            print(f"Processed {idx + 1}/{len(clean_pids)} patients ({time.time() - t0:.1f}s)")

    # Save single-scale datasets
    for scale_name in SCALES_MINUTES:
        X_arr = np.stack(scale_datasets[scale_name]["X"], axis=0)
        y_arr = np.array(scale_datasets[scale_name]["y"], dtype=np.int64)
        pid_arr = np.array(scale_datasets[scale_name]["pid"])
        meta_arr = scale_datasets[scale_name]["meta"]

        out_file = os.path.join(OUT_DIR, f"scale_{scale_name}_dataset.pt")
        torch.save({"X": torch.tensor(X_arr), "y": torch.tensor(y_arr), "pid": pid_arr, "meta": meta_arr}, out_file)
        print(f"Saved single scale [{scale_name}]: {X_arr.shape} windows ({len(np.unique(pid_arr))} patients) -> {out_file}")

    # Save matched multi-resolution dataset
    matched_out = {
        "X_10m": torch.tensor(np.stack(matched_data["X_10m"], axis=0)),
        "X_20m": torch.tensor(np.stack(matched_data["X_20m"], axis=0)),
        "X_40m": torch.tensor(np.stack(matched_data["X_40m"], axis=0)),
        "F_10m": torch.tensor(np.stack(matched_data["F_10m"], axis=0)),
        "F_20m": torch.tensor(np.stack(matched_data["F_20m"], axis=0)),
        "F_40m": torch.tensor(np.stack(matched_data["F_40m"], axis=0)),
        "F_trends": torch.tensor(np.stack(matched_data["F_trends"], axis=0)),
        "y": torch.tensor(np.array(matched_data["y"], dtype=np.int64)),
        "pid": np.array(matched_data["pid"]),
        "meta": matched_data["meta"],
        "time_to_delivery_min": np.array(matched_data["time_to_delivery_min"], dtype=np.float32),
        "audit_df": audit_stats
    }

    matched_path = os.path.join(OUT_DIR, "matched_multires_dataset.pt")
    torch.save(matched_out, matched_path)
    print(f"Saved matched multi-resolution dataset: {matched_out['X_10m'].shape[0]} matched windows ({len(np.unique(matched_out['pid']))} patients) -> {matched_path}")

    # Save audit stats
    with open(os.path.join(OUT_DIR, "missingness_duration_audit.json"), "w") as f:
        json.dump(audit_stats, f, indent=2)

    print(f"=== PHASE 5 DATASET BUILDING COMPLETE ({time.time() - t0:.1f}s) ===")


if __name__ == "__main__":
    build_phase5_datasets()
