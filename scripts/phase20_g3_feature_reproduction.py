"""
Phase 20 Stage 0, gate G3(a): can the 19 raw window descriptors be recomputed from the raw CTU-CHB records?
Every C5 candidate (multi-scale / long-context / morphology) needs this. The window signal chain is per-window
(remove_spikes -> interpolate -> lowpass -> iterative baseline -> 8 clinical features + 11 extended features), so
if a full-window recomputation reproduces the stored X40[:, :19] then sub-window / 40-min variants are built on a
verified chain.  Protocol tolerance: 1e-5.  Sample: 40 patients (fixed seed), all of their windows.
"""
import os, sys, json
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src", "preprocessing"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from ingestion import load_ctu_chb_record
from filtering import remove_spikes, interpolate_missing, apply_lowpass_filter
from baseline import calculate_iterative_baseline
from features import calculate_variability, detect_accelerations, detect_decelerations
from knowledge.extended_features import extract_extended_features

RAW_DIR = "data/raw/ctu-chb-intrapartum"
ROLLING = "results/phase8_rolling/rolling_predictions.csv"
X40 = "results/phase9c_state_trajectory/state_trajectory_features.npz"
OUT = "results/phase20_redesign"
LAST_HOUR = 14400; WIN = 4800
FHR_MIN, FHR_MAX = 50.0, 240.0
TOL = 1e-5
_sc = np.load("data/processed_clinical/ctu_signal_scaler.npz")
SCALER_MEAN, SCALER_STD = _sc["mean"].astype(np.float32), _sc["std"].astype(np.float32)


def window_descriptors(raw_fhr, raw_uc, fs=4.0):
    """The exact per-window chain of pipeline_clinical.py -> (19,) in raw_0..raw_18 order."""
    f = remove_spikes(raw_fhr.copy(), fs=fs)
    f = interpolate_missing(f, clip_min=FHR_MIN, clip_max=FHR_MAX)
    f = apply_lowpass_filter(f, fs=fs)
    u = apply_lowpass_filter(raw_uc.copy(), fs=fs)
    baseline = calculate_iterative_baseline(f)
    stv, ltv = calculate_variability(f, fs=fs, baseline=baseline)
    accels = detect_accelerations(f, baseline, fs=fs)
    decels = detect_decelerations(f, baseline, u, fs=fs)
    base_val = float(np.mean(baseline))
    y8 = [base_val, stv, ltv, float(accels), float(decels["early"]), float(decels["late"]), float(decels["variable"]), float(decels["prolonged"])]
    # The stored extended features were computed by extract_batch() from the z-scored float32 tensors on disk
    # (verified: extract_batch(stored X) reproduces the stored npy exactly), so emulate that round trip.
    f32 = np.float32
    x0 = ((f - baseline).astype(f32) - SCALER_MEAN[0]) / SCALER_STD[0]
    x1 = (u.astype(f32) - SCALER_MEAN[1]) / SCALER_STD[1]
    ext = extract_extended_features(x0.astype(f32) * SCALER_STD[0] + SCALER_MEAN[0],
                                    x1.astype(f32) * SCALER_STD[1] + SCALER_MEAN[1], base_val, fs=fs)
    return np.array(y8 + [float(v) for v in ext], dtype=np.float64)


def main():
    df = pd.read_csv(ROLLING); df["patient_id"] = df["patient_id"].astype(str)
    X = np.load(X40)["X_state_trajectory"][:, :19]
    rng = np.random.default_rng(42)
    pids = sorted(df["patient_id"].unique()); pick = sorted(rng.choice(pids, 40, replace=False))
    diffs = []; n_win = 0; worst = {}
    for pid in pick:
        fhr, uc, fs = load_ctu_chb_record(os.path.join(RAW_DIR, str(pid)))
        full = len(fhr)
        if full > LAST_HOUR:
            fhr, uc = fhr[-LAST_HOUR:], uc[-LAST_HOUR:]
        sub = df[df["patient_id"] == pid]
        for i, s in zip(sub.index, sub["start_sample"].values):
            s = int(s)
            d = window_descriptors(fhr[s:s + WIN], uc[s:s + WIN], fs)
            err = np.abs(d - X[i].astype(np.float64)); diffs.append(err); n_win += 1
    diffs = np.array(diffs)
    max_by_feature = diffs.max(0)
    rel = diffs / (np.abs(X[:1]).max() + 1e-12)
    ok = bool(diffs.max() <= TOL)
    print(f"  G3(a): recomputed {n_win} windows of {len(pick)} patients; max |diff| per feature:")
    print("  ", np.round(max_by_feature, 8).tolist())
    print(f"  overall max |diff| = {diffs.max():.3e}   [{'PASS' if ok else 'FAIL'}] (tolerance {TOL})")
    json.dump({"n_patients": len(pick), "n_windows": n_win, "max_abs_diff_per_feature": [float(v) for v in max_by_feature],
               "overall_max_abs_diff": float(diffs.max()), "tolerance": TOL, "pass": ok}, open(os.path.join(OUT, "stage0_G3a.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
