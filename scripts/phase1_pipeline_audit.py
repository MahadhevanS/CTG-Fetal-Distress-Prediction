"""
Phase 1: CTU-UHB Signal Quality & Preprocessing Re-engineering Audit.

Performs complete recording-level and window-level audits across all 547 clean cohort
patients (110 positive, 437 negative) and 8,517 20-minute windows.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.ingestion import load_ctu_chb_record, load_clinical_metadata, TARGET_FS
from src.preprocessing.signal_quality import assess_patient_missing_ratio, get_valid_windows
from src.preprocessing.filtering import remove_spikes, interpolate_missing, apply_lowpass_filter
from src.preprocessing.baseline import calculate_iterative_baseline, rolling_iterative_baseline
from src.preprocessing.features import calculate_variability, detect_accelerations, detect_decelerations


RAW_DIR = "data/raw/ctu-chb-intrapartum"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
FOLDS_PATH = "data/processed_clinical/folds.json"
OUT_DIR = "results/phase1_audit"

WINDOW_MINUTES = 20
LAST_HOUR_MINUTES = 60
STRIDE_MINUTES = 2.5
WINDOW_SAMPLES = int(WINDOW_MINUTES * 60 * TARGET_FS)        # 4800
LAST_HOUR_SAMPLES = int(LAST_HOUR_MINUTES * 60 * TARGET_FS)  # 14400
STRIDE_SAMPLES = int(STRIDE_MINUTES * 60 * TARGET_FS)        # 600
MAX_MISSING_RATIO = 0.5
FHR_MIN_BPM = 50.0
FHR_MAX_BPM = 240.0


def analyze_gap_durations(signal: np.ndarray, fs: float = 4.0) -> Dict[str, int]:
    """Classify continuous missing segments (0.0) into duration bins."""
    missing = (signal == 0.0).astype(int)
    if not np.any(missing):
        return {"0-5s": 0, "5-15s": 0, "15-30s": 0, "30-60s": 0, ">60s": 0, "total_gaps": 0, "max_gap_sec": 0.0}

    diffs = np.diff(np.pad(missing, (1, 1), 'constant'))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    lengths_sec = (ends - starts) / fs

    bins = {"0-5s": 0, "5-15s": 0, "15-30s": 0, "30-60s": 0, ">60s": 0,
            "total_gaps": len(lengths_sec), "max_gap_sec": float(np.max(lengths_sec)) if len(lengths_sec) > 0 else 0.0}

    for l in lengths_sec:
        if l <= 5.0:
            bins["0-5s"] += 1
        elif l <= 15.0:
            bins["5-15s"] += 1
        elif l <= 30.0:
            bins["15-30s"] += 1
        elif l <= 60.0:
            bins["30-60s"] += 1
        else:
            bins[">60s"] += 1
    return bins


def count_abrupt_jumps(signal: np.ndarray, fs: float = 4.0, max_rate: float = 25.0) -> int:
    """Count sample-to-sample transitions exceeding physiological threshold (>25 bpm/s)."""
    valid = signal > 0.0
    valid_pairs = valid[:-1] & valid[1:]
    if not np.any(valid_pairs):
        return 0
    deltas = np.abs(np.diff(signal))
    max_delta = max_rate / fs  # 6.25 bpm at 4 Hz
    return int(np.sum((deltas > max_delta) & valid_pairs))


def count_implausible_values(signal: np.ndarray, min_val: float = 50.0, max_val: float = 240.0) -> int:
    """Count non-zero samples outside [50, 240] bpm."""
    non_zero = signal[signal > 0.0]
    return int(np.sum((non_zero < min_val) | (non_zero > max_val)))


def count_flatline_samples(signal: np.ndarray, min_consecutive: int = 40) -> int:
    """Count non-zero samples that are part of flatlines (>10s identical values)."""
    if len(signal) == 0:
        return 0
    non_zero = signal > 0.0
    diffs = np.abs(np.diff(signal))
    flat_trans = (diffs == 0.0) & non_zero[:-1] & non_zero[1:]
    
    # Run-length encode flat segments
    padded = np.pad(flat_trans.astype(int), (1, 1), 'constant')
    c = np.diff(padded)
    s = np.where(c == 1)[0]
    e = np.where(c == -1)[0]
    lengths = e - s
    long_flats = lengths[lengths >= min_consecutive]
    return int(np.sum(long_flats + 1))


def run_full_audit():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== STARTING PHASE 1 AUDIT ===")

    # 1. Load clinical metadata
    metadata = load_clinical_metadata(METADATA_PATH)
    if 'record_id' in metadata.columns:
        metadata = metadata.set_index('record_id')
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    print(f"Loaded {len(clean_pids)} clean cohort patients from {FOLDS_PATH}")

    # Patient-level and window-level records
    patient_stats = []
    window_stats = []

    total_windows = 0
    total_pos_windows = 0

    # Gap distributions aggregated
    rec_gap_totals = {"0-5s": 0, "5-15s": 0, "15-30s": 0, "30-60s": 0, ">60s": 0, "total_gaps": 0}
    win_gap_totals = {"0-5s": 0, "5-15s": 0, "15-30s": 0, "30-60s": 0, ">60s": 0, "total_gaps": 0}

    # Morphology comparison tracking
    morph_records = []

    # Baseline comparison tracking
    baseline_records = []

    for idx, pid in enumerate(clean_pids):
        rec_path = os.path.join(RAW_DIR, str(pid))
        fhr_full, uc_full, fs = load_ctu_chb_record(rec_path)
        ph_val = float(metadata.loc[int(pid), 'ph'])
        is_pos = int(ph_val <= 7.15)

        full_len = len(fhr_full)
        full_dur_min = full_len / (fs * 60.0)

        # Full recording stats
        fhr_missing_frac = float(np.mean(fhr_full == 0.0))
        uc_missing_frac = float(np.mean(uc_full == 0.0))
        fhr_jumps = count_abrupt_jumps(fhr_full, fs)
        fhr_implausible = count_implausible_values(fhr_full, FHR_MIN_BPM, FHR_MAX_BPM)
        fhr_flatline = count_flatline_samples(fhr_full)
        rec_gaps = analyze_gap_durations(fhr_full, fs)
        for k in rec_gap_totals:
            if k in rec_gaps:
                rec_gap_totals[k] += rec_gaps[k]

        # Truncate to last hour
        if full_len > LAST_HOUR_SAMPLES:
            fhr = fhr_full[-LAST_HOUR_SAMPLES:]
            uc = uc_full[-LAST_HOUR_SAMPLES:]
        else:
            fhr = fhr_full.copy()
            uc = uc_full.copy()

        starts = get_valid_windows(fhr, WINDOW_SAMPLES, STRIDE_SAMPLES, max_missing_ratio=MAX_MISSING_RATIO)

        p_stat = {
            "record_id": pid,
            "is_distress": is_pos,
            "ph": ph_val,
            "full_duration_min": full_dur_min,
            "fhr_missing_frac": fhr_missing_frac,
            "uc_missing_frac": uc_missing_frac,
            "fhr_abrupt_jumps": fhr_jumps,
            "fhr_implausible": fhr_implausible,
            "fhr_flatline_samples": fhr_flatline,
            "max_missing_gap_sec": rec_gaps["max_gap_sec"],
            "total_gaps": rec_gaps["total_gaps"],
            "usable_windows": len(starts)
        }
        patient_stats.append(p_stat)

        # Window level audit
        for w_idx, start in enumerate(starts):
            end = start + WINDOW_SAMPLES
            w_fhr_raw = fhr[start:end].copy()
            w_uc_raw = uc[start:end].copy()

            total_windows += 1
            if is_pos:
                total_pos_windows += 1

            w_missing_frac = float(np.mean(w_fhr_raw == 0.0))
            w_uc_missing_frac = float(np.mean(w_uc_raw == 0.0))
            w_jumps = count_abrupt_jumps(w_fhr_raw, fs)
            w_implausible = count_implausible_values(w_fhr_raw, FHR_MIN_BPM, FHR_MAX_BPM)
            w_flat = count_flatline_samples(w_fhr_raw)
            w_gaps = analyze_gap_durations(w_fhr_raw, fs)
            for k in win_gap_totals:
                if k in w_gaps:
                    win_gap_totals[k] += w_gaps[k]

            # Compare preprocessing stages
            # 1. Raw valid samples
            # 2. P0 processing
            f_spikes = remove_spikes(w_fhr_raw.copy(), fs=fs)
            f_p0_interp = interpolate_missing(f_spikes.copy(), clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
            f_p0 = apply_lowpass_filter(f_p0_interp.copy(), fs=fs)
            u_p0 = apply_lowpass_filter(w_uc_raw.copy(), fs=fs)

            # 3. P1 processing (minimal: remove bounds/spikes, short gap <=5s linear interp, no lowpass)
            f_p1_clean = w_fhr_raw.copy()
            f_p1_clean[(f_p1_clean < FHR_MIN_BPM) | (f_p1_clean > FHR_MAX_BPM)] = 0.0
            f_p1_clean = remove_spikes(f_p1_clean, fs=fs)
            f_p1 = interpolate_missing(f_p1_clean, max_gap_samples=20, clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)

            # 4. Baseline Estimators
            b_const = calculate_iterative_baseline(f_p0)
            b_roll = rolling_iterative_baseline(f_p0, fs=fs, win_sec=300.0, step_sec=30.0, round_to_5=False)
            b_median = np.full_like(f_p0, np.median(f_p0[f_p0 > 0])) if np.any(f_p0 > 0) else np.full_like(f_p0, 140.0)

            # Whole-recording baseline vs window baseline (Leakage test)
            b_whole_rec = calculate_iterative_baseline(fhr)
            b_whole_window = b_whole_rec[start:end]

            baseline_diff = float(np.mean(np.abs(b_const - b_whole_window)))
            rolling_range = float(np.max(b_roll) - np.min(b_roll))

            baseline_records.append({
                "record_id": pid,
                "window_idx": w_idx,
                "b_const_mean": float(np.mean(b_const)),
                "b_roll_mean": float(np.mean(b_roll)),
                "b_roll_range": rolling_range,
                "b_diff_whole_rec": baseline_diff,
                "b_const_outside_normal": bool(np.mean(b_const) < 110.0 or np.mean(b_const) > 160.0)
            })

            # Morphology feature extraction (comparing P0 vs P1 vs Raw)
            stv_p0, ltv_p0 = calculate_variability(f_p0, fs=fs, baseline=b_const)
            acc_p0 = detect_accelerations(f_p0, b_const, fs=fs)
            dec_p0 = detect_decelerations(f_p0, b_const, u_p0, fs=fs)

            stv_p1, ltv_p1 = calculate_variability(f_p1, fs=fs, baseline=b_roll)
            acc_p1 = detect_accelerations(f_p1, b_roll, fs=fs)
            dec_p1 = detect_decelerations(f_p1, b_roll, w_uc_raw, fs=fs)

            # Deceleration total depth and area on P0 vs P1
            dec_total_p0 = sum(dec_p0.values())
            dec_total_p1 = sum(dec_p1.values())

            # Residual spikes & discontinuity after filtering
            res_spikes_p0 = count_abrupt_jumps(f_p0, fs)
            res_spikes_p1 = count_abrupt_jumps(f_p1, fs)

            interp_samples_p0 = int(np.sum((w_fhr_raw == 0.0) & (f_p0 > 0.0)))
            interp_frac_p0 = interp_samples_p0 / WINDOW_SAMPLES

            morph_records.append({
                "record_id": pid,
                "window_idx": w_idx,
                "is_distress": is_pos,
                "missing_frac": w_missing_frac,
                "interp_frac_p0": interp_frac_p0,
                "stv_p0": stv_p0, "ltv_p0": ltv_p0, "acc_p0": acc_p0, "dec_p0": dec_total_p0,
                "stv_p1": stv_p1, "ltv_p1": ltv_p1, "acc_p1": acc_p1, "dec_p1": dec_total_p1,
                "res_spikes_p0": res_spikes_p0, "res_spikes_p1": res_spikes_p1
            })

            window_stats.append({
                "record_id": pid,
                "window_idx": w_idx,
                "is_distress": is_pos,
                "missing_frac": w_missing_frac,
                "uc_missing_frac": w_uc_missing_frac,
                "abrupt_jumps": w_jumps,
                "implausible_count": w_implausible,
                "flatline_samples": w_flat,
                "max_gap_sec": w_gaps["max_gap_sec"],
                "interp_frac_p0": interp_frac_p0
            })

    print(f"\nAudit complete across {len(clean_pids)} patients and {total_windows} windows ({total_pos_windows} distress windows).")

    df_p = pd.DataFrame(patient_stats)
    df_w = pd.DataFrame(window_stats)
    df_m = pd.DataFrame(morph_records)
    df_b = pd.DataFrame(baseline_records)

    df_p.to_csv(os.path.join(OUT_DIR, "patient_quality_audit.csv"), index=False)
    df_w.to_csv(os.path.join(OUT_DIR, "window_quality_audit.csv"), index=False)
    df_m.to_csv(os.path.join(OUT_DIR, "morphology_audit.csv"), index=False)
    df_b.to_csv(os.path.join(OUT_DIR, "baseline_audit.csv"), index=False)

    # Summarize statistical findings
    summary = {
        "cohort": {
            "total_patients": len(df_p),
            "positive_patients": int(df_p["is_distress"].sum()),
            "negative_patients": int((df_p["is_distress"] == 0).sum()),
            "total_windows": len(df_w),
            "positive_windows": int(df_w["is_distress"].sum()),
            "negative_windows": int((df_w["is_distress"] == 0).sum()),
        },
        "recording_level_quality": {
            "fhr_missing_pct_mean": float(df_p["fhr_missing_frac"].mean() * 100),
            "fhr_missing_pct_median": float(df_p["fhr_missing_frac"].median() * 100),
            "fhr_missing_pct_p25_p75": [float(df_p["fhr_missing_frac"].quantile(0.25) * 100),
                                         float(df_p["fhr_missing_frac"].quantile(0.75) * 100)],
            "uc_missing_pct_mean": float(df_p["uc_missing_frac"].mean() * 100),
            "total_implausible_values": int(df_p["fhr_implausible"].sum()),
            "total_abrupt_jumps": int(df_p["fhr_abrupt_jumps"].sum()),
            "total_flatline_samples": int(df_p["fhr_flatline_samples"].sum()),
            "gap_duration_distribution": rec_gap_totals
        },
        "window_level_quality": {
            "window_missing_pct_mean": float(df_w["missing_frac"].mean() * 100),
            "window_missing_pct_median": float(df_w["missing_frac"].median() * 100),
            "window_interp_pct_mean": float(df_w["interp_frac_p0"].mean() * 100),
            "window_gap_distribution": win_gap_totals,
            "windows_with_zero_missing": int((df_w["missing_frac"] == 0).sum()),
            "windows_with_missing_gt_30pct": int((df_w["missing_frac"] > 0.30).sum()),
            "windows_with_missing_gt_40pct": int((df_w["missing_frac"] > 0.40).sum()),
        },
        "missingness_by_outcome": {
            "pos_fhr_missing_pct_mean": float(df_p[df_p["is_distress"] == 1]["fhr_missing_frac"].mean() * 100),
            "neg_fhr_missing_pct_mean": float(df_p[df_p["is_distress"] == 0]["fhr_missing_frac"].mean() * 100),
            "pos_window_missing_pct_mean": float(df_w[df_w["is_distress"] == 1]["missing_frac"].mean() * 100),
            "neg_window_missing_pct_mean": float(df_w[df_w["is_distress"] == 0]["missing_frac"].mean() * 100),
            "pos_windows_per_patient": float(df_p[df_p["is_distress"] == 1]["usable_windows"].mean()),
            "neg_windows_per_patient": float(df_p[df_p["is_distress"] == 0]["usable_windows"].mean()),
        },
        "baseline_audit": {
            "baseline_mean_bpm": float(df_b["b_const_mean"].mean()),
            "baseline_std_bpm": float(df_b["b_const_mean"].std()),
            "pct_outside_110_160_bpm": float(df_b["b_const_outside_normal"].mean() * 100),
            "mean_rolling_drift_within_window_bpm": float(df_b["b_roll_range"].mean()),
            "median_rolling_drift_within_window_bpm": float(df_b["b_roll_range"].median()),
            "leakage_diff_window_vs_whole_rec_bpm": float(df_b["b_diff_whole_rec"].mean())
        },
        "morphology_preservation": {
            "stv_p0_mean": float(df_m["stv_p0"].mean()),
            "stv_p1_mean": float(df_m["stv_p1"].mean()),
            "ltv_p0_mean": float(df_m["ltv_p0"].mean()),
            "ltv_p1_mean": float(df_m["ltv_p1"].mean()),
            "acc_p0_mean": float(df_m["acc_p0"].mean()),
            "acc_p1_mean": float(df_m["acc_p1"].mean()),
            "dec_p0_mean": float(df_m["dec_p0"].mean()),
            "dec_p1_mean": float(df_m["dec_p1"].mean()),
            "res_spikes_p0": int(df_m["res_spikes_p0"].sum()),
            "res_spikes_p1": int(df_m["res_spikes_p1"].sum())
        }
    }

    with open(os.path.join(OUT_DIR, "audit_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n--- Summary Highlights ---")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_full_audit()
