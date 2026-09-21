"""
Phase 10, Experiment 1 -- minimally processed raw FHR.

HYPOTHESIS UNDER TEST
    Does minimally processed raw FHR contain predictive information that the 19
    clinical descriptors do not capture?

DESIGN PRINCIPLE
    Remove technical corruption. Destroy nothing physiological.

WHAT IS DELIBERATELY *NOT* DONE (each of these is an information bottleneck the
experiment exists to avoid):
    * no low-pass filter          -- pipeline_clinical applies 1.5 Hz Butterworth
    * no baseline subtraction     -- pipeline_clinical stores (fhr - baseline)
    * no baseline estimation      -- no calculate_iterative_baseline
    * no event detection          -- no accelerations / decelerations
    * no FIGO conversion
    * no feature extraction

WHAT IS DONE, AND WHY
    1. spike removal      technical corruption: >25 bpm/s is non-physiological
    2. bounded cubic interpolation of gaps <= 15 s (the existing convention)
    3. linear interpolation across LONGER gaps, PLUS a missingness channel

Point 3 deserves justification. pipeline_clinical leaves gaps > 15 s as 0.0,
which after scaling becomes a large negative excursion; 7.05 % of CTU samples
are in such gaps and they dominate window variance (measured in Phase 9B). A
model reading those as heart rate would be learning an artifact. Zero-filling
fabricates a bradycardia; interpolating fabricates a plausible trace. The
missingness channel is not physiological information -- it is metadata telling
the model which samples were measured, so it need not guess.

NORMALISATION IS FIT PER TRAINING FOLD, never on all data, so no test-fold
statistics leak (Phase 10 rule 5.5).

Channels: [0] FHR in bpm (absolute level preserved), [1] missingness mask.
"""
import os
from typing import Dict, List, Tuple

import numpy as np

from .filtering import interpolate_missing, remove_spikes
from .ingestion import load_ctu_chb_record

FS_RAW = 4.0
LAST_HOUR_SAMPLES = 14400          # 60 min at 4 Hz
WINDOW_MIN, STRIDE_MIN = 20.0, 2.5
MAX_MISSING_RATIO = 0.50           # identical to pipeline_clinical's gate
FHR_MIN_BPM, FHR_MAX_BPM = 50.0, 240.0
MAX_CUBIC_GAP_SAMPLES = 60         # 15 s

PREPROCESSING_AUDIT: Dict[str, str] = {
    "sampling_frequency_hz": "4.0 (as distributed); 1.0 variant by 4-sample mean of measured samples",
    "temporal_range": "final 60 minutes before delivery (14400 samples)",
    "artifact_criterion": "remove_spikes: |d(FHR)/dt| > 25 bpm/s -> sample zeroed, then interpolated",
    "interpolation_method": "cubic spline for gaps <= 15 s; linear across longer gaps",
    "max_interpolated_gap": "unbounded, but every interpolated sample is flagged in channel 1",
    "smoothing": "NONE (pipeline_clinical applies a 1.5 Hz low-pass; this does not)",
    "filtering": "NONE",
    "baseline_handling": "NONE - absolute bpm retained (pipeline_clinical subtracts a baseline)",
    "clipping": "[50, 240] bpm after interpolation",
    "normalization": "per-channel z-score, mean/std fit on TRAINING FOLD windows only",
    "window": "20 min (4800 samples @4Hz / 1200 @1Hz)",
    "stride": "2.5 min (600 samples @4Hz / 150 @1Hz)",
    "window_quality_gate": "drop window if >50% of raw FHR samples are missing",
    "channels": "[0] FHR bpm, [1] missingness mask (1 = sample was not measured)",
}


def minimal_fhr(fhr_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fhr_bpm, missing_mask). No filtering, no baseline removal."""
    missing = (fhr_raw <= 0).astype(np.float32)
    f = remove_spikes(fhr_raw.astype(np.float64).copy(), fs=FS_RAW)
    missing = np.maximum(missing, (f <= 0).astype(np.float32))
    f = interpolate_missing(f, max_gap_samples=MAX_CUBIC_GAP_SAMPLES,
                            clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
    still = f <= 0
    if still.any():
        good = ~still
        if good.sum() >= 2:
            idx = np.arange(len(f))
            f[still] = np.interp(idx[still], idx[good], f[good])
        else:
            f[still] = float(np.median(f[good])) if good.any() else 140.0
    f = np.clip(f, FHR_MIN_BPM, FHR_MAX_BPM)
    return f.astype(np.float32), missing


def to_1hz(fhr: np.ndarray, missing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decimate 4 Hz -> 1 Hz by averaging each group of 4 MEASURED samples.

    Averaging measured samples only (rather than plain striding) avoids both
    aliasing and letting interpolated values dominate a second.
    """
    n = (len(fhr) // 4) * 4
    f = fhr[:n].reshape(-1, 4)
    m = missing[:n].reshape(-1, 4)
    meas = m < 0.5
    num = (f * meas).sum(1)
    den = meas.sum(1)
    out = np.where(den > 0, num / np.maximum(den, 1), f.mean(1))
    return out.astype(np.float32), (1.0 - den / 4.0).astype(np.float32)


def build_patient_windows(record_path: str, fs_out: float
                          ) -> Tuple[np.ndarray, List[int]]:
    """(n_windows, 2, T) for one patient, plus the window start indices."""
    fhr_raw, _uc, fs = load_ctu_chb_record(record_path)
    if len(fhr_raw) == 0:
        return np.zeros((0, 2, 1), np.float32), []
    if len(fhr_raw) > LAST_HOUR_SAMPLES:
        fhr_raw = fhr_raw[-LAST_HOUR_SAMPLES:]

    f, m = minimal_fhr(fhr_raw)
    raw_missing_4hz = m.copy()
    if fs_out == 1.0:
        f, m = to_1hz(f, m)
        gate_src = raw_missing_4hz
        gate_factor = 4
    else:
        gate_src = raw_missing_4hz
        gate_factor = 1

    win = int(WINDOW_MIN * 60 * fs_out)
    stride = int(STRIDE_MIN * 60 * fs_out)
    out, starts = [], []
    for st in range(0, len(f) - win + 1, stride):
        g0, g1 = st * gate_factor, (st + win) * gate_factor
        if gate_src[g0:g1].mean() > MAX_MISSING_RATIO:
            continue
        out.append(np.vstack([f[st:st + win], m[st:st + win]]))
        starts.append(st)
    if not out:
        return np.zeros((0, 2, win), np.float32), []
    return np.asarray(out, np.float32), starts
