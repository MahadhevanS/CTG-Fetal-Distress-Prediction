"""
Extended clinical features -- knowledge the existing 8 features do NOT capture.

WHY. Eight knowledge-infusion mechanisms failed, and the explanation was that
the network already encodes what the FIGO rules encode. But that was only ever
demonstrated for the 8 features the pipeline computes (baseline, STV, LTV,
accel count, 4 deceleration COUNTS). It says nothing about clinical information
those 8 omit -- and they omit a great deal:

  * deceleration DEPTH and AREA -- FIGO grades severity by depth, we only count
  * FHR<->UC timing LAG -- the actual defining feature of a late deceleration;
    we carry only a binary "late" count produced by a threshold rule
  * uterine activity -- UC is half the input signal and NO feature is extracted
    from it at all. Tachysystole (>5 contractions/10 min) is an independent
    recognised risk factor.
  * within-window TRENDS -- a rising baseline or declining variability is a
    compensatory/ominous sign; single per-window values cannot express it.

There is also a data-efficiency argument: the network has only ~226 positive
windows to learn from. Hand-crafted features encode prior knowledge it does not
have to learn, which is the classic advantage in a low-data regime. The earlier
attempts re-supplied knowledge the network already had; these supply knowledge
it may never have extracted.

All features are computed from tensors already on disk (the signal scaler makes
raw units recoverable), so no pipeline rerun is required.

MISSING DATA: the pipeline stores missing FHR as 0 bpm, which becomes
-baseline after baseline subtraction. Samples implying an absolute FHR below
FHR_MIN_VALID are treated as missing and excluded from every statistic.
"""

from typing import List

import numpy as np

FHR_MIN_VALID = 50.0     # bpm; below this is missing/unphysiological
DECEL_THRESHOLD = 15.0   # bpm below baseline (FIGO)

EXTENDED_FEATURE_NAMES: List[str] = [
    "decel_max_depth",       # deepest excursion below baseline (bpm)
    "decel_area",            # mean depth below -15bpm over the window (bpm)
    "decel_burden",          # fraction of valid samples >15bpm below baseline
    "decel_longest_sec",     # longest continuous deceleration (seconds)
    "baseline_slope",        # FHR trend across the window (bpm/min)
    "variability_slope",     # trend in per-minute variability (bpm/min)
    "uc_contraction_count",  # contractions detected in the window
    "uc_tachysystole",       # 1 if >5 contractions per 10 min
    "uc_mean_amplitude",     # mean UC amplitude above its own baseline
    "fhr_uc_lag_sec",        # lag maximising UC vs FHR-drop correlation (s)
    "fhr_uc_coupling",       # strength of that coupling (correlation)
]


def _valid_mask(fhr_bc: np.ndarray, baseline: float) -> np.ndarray:
    return (fhr_bc + baseline) >= FHR_MIN_VALID


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def extract_extended_features(fhr_bc: np.ndarray, uc: np.ndarray,
                              baseline: float, fs: float = 4.0) -> np.ndarray:
    """
    Args:
        fhr_bc:   (T,) baseline-corrected FHR in bpm (negative = below baseline)
        uc:       (T,) uterine contraction trace, original units
        baseline: that window's baseline FHR in bpm
    Returns:
        (11,) float32 in EXTENDED_FEATURE_NAMES order.
    """
    out = np.zeros(len(EXTENDED_FEATURE_NAMES), dtype=np.float32)
    valid = _valid_mask(fhr_bc, baseline)
    if valid.sum() < fs * 30:          # under 30s of usable signal
        return out

    f = fhr_bc.copy()
    f[~valid] = 0.0

    # --- deceleration depth / area / burden -------------------------------
    below = np.where(valid, -f, 0.0)               # positive = bpm below baseline
    out[0] = float(np.clip(below.max(), 0, 200))
    deep = np.clip(below - DECEL_THRESHOLD, 0, None)
    out[1] = float(deep[valid].mean())
    decel_mask = valid & (below > DECEL_THRESHOLD)
    out[2] = float(decel_mask.sum() / max(valid.sum(), 1))
    out[3] = float(_longest_run(decel_mask) / fs)

    # --- within-window trends ---------------------------------------------
    idx = np.arange(len(f))[valid]
    if len(idx) > 10:
        t = (idx - idx.mean()) / fs / 60.0          # minutes, centred
        v = f[valid] - f[valid].mean()
        denom = float((t ** 2).sum())
        out[4] = float((t * v).sum() / denom) if denom > 0 else 0.0

    spm = int(60 * fs)
    n_min = len(f) // spm
    if n_min >= 3:
        var_per_min = []
        for i in range(n_min):
            sl = slice(i * spm, (i + 1) * spm)
            vm = valid[sl]
            var_per_min.append(float(f[sl][vm].std()) if vm.sum() > fs * 10 else np.nan)
        vpm = np.array(var_per_min, dtype=float)
        ok = ~np.isnan(vpm)
        if ok.sum() >= 3:
            x = np.arange(len(vpm))[ok].astype(float)
            x -= x.mean()
            yv = vpm[ok] - vpm[ok].mean()
            den = float((x ** 2).sum())
            out[5] = float((x * yv).sum() / den) if den > 0 else 0.0

    # --- uterine activity (never previously used) -------------------------
    uc_b = float(np.median(uc))
    uc_c = uc - uc_b
    amp = float(np.percentile(uc_c, 95))
    thr = max(amp * 0.5, 5.0)
    above = uc_c > thr
    # contractions last ~45-90s; require >=30s and merge gaps <30s
    min_len, gap = int(30 * fs), int(30 * fs)
    runs, i = [], 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    contractions = [r for r in merged if (r[1] - r[0]) >= min_len]
    n_c = len(contractions)
    minutes = len(uc) / fs / 60.0
    out[6] = float(n_c)
    out[7] = float(1.0 if (n_c / max(minutes, 1e-6)) * 10.0 > 5.0 else 0.0)
    out[8] = float(np.clip(amp, 0, 200))

    # --- FHR<->UC coupling: does FHR drop follow contractions, and how late? -
    if n_c > 0:
        dec = np.where(valid, np.clip(below, 0, None), 0.0)
        step = max(int(fs * 2), 1)                   # 2s resolution
        d_ds = dec[::step]
        u_ds = np.clip(uc_c, 0, None)[::step]
        if d_ds.std() > 1e-6 and u_ds.std() > 1e-6:
            d_z = (d_ds - d_ds.mean()) / d_ds.std()
            u_z = (u_ds - u_ds.mean()) / u_ds.std()
            max_lag = int(120 / 2)                   # search 0-120s
            best_c, best_l = 0.0, 0.0
            for lag in range(0, min(max_lag, len(d_z) - 5)):
                a = u_z[:len(u_z) - lag] if lag else u_z
                b = d_z[lag:]
                n = min(len(a), len(b))
                if n < 10:
                    break
                c = float(np.dot(a[:n], b[:n]) / n)
                if c > best_c:
                    best_c, best_l = c, lag * 2.0
            out[9] = float(best_l)
            out[10] = float(np.clip(best_c, -1, 1))
    return out


def extract_batch(X: np.ndarray, y_features: np.ndarray,
                  signal_mean: np.ndarray, signal_std: np.ndarray,
                  fs: float = 4.0) -> np.ndarray:
    """
    Args:
        X:          (N, 2, T) z-normalised tensors as stored on disk
        y_features: (N, 8) existing clinical features (column 0 = baseline bpm)
    Returns:
        (N, 11) extended features.
    """
    n = len(X)
    out = np.zeros((n, len(EXTENDED_FEATURE_NAMES)), dtype=np.float32)
    for i in range(n):
        fhr_bc = X[i, 0] * signal_std[0] + signal_mean[0]
        uc = X[i, 1] * signal_std[1] + signal_mean[1]
        out[i] = extract_extended_features(fhr_bc, uc, float(y_features[i, 0]), fs=fs)
    return out
