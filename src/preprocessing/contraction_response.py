"""
Contraction-response trajectories -- the contraction as the unit of analysis.

WHY THIS EXISTS (2026-09-04, phase 8)
-------------------------------------
Every representation this project has tried reduces a 20-minute window to
summary statistics and then aggregates windows by max. That representation can
say "there were 4 variable decelerations and the mean decel depth was 22 bpm".
It cannot say:

    "the 12th contraction produced a deeper, slower-recovering deceleration
     than the 3rd"

which is the physiological definition of fetal reserve: a healthy fetus recovers
fully from each contraction; a compromised one deteriorates across repeated
stress. docs/phase8_information_audit.md closed six information sources, all of
which were window-level feature additions. This module changes the UNIT rather
than adding columns.

    contraction -> FHR response -> response trajectory -> patient risk

WHAT IS DELIBERATELY UNCHANGED
  The signal chain (spike removal -> cubic interpolation -> lowpass -> iterative
  baseline) and the 60-minute truncation, so the cohort matches
  data/processed_clinical and the frozen protocol folds apply unchanged.

The one structural difference: the signal is processed as ONE continuous
60-minute segment per patient, because a contraction sequence does not respect
20-minute window boundaries.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from .baseline import calculate_iterative_baseline
from .filtering import apply_lowpass_filter, interpolate_missing, remove_spikes

FS = 4.0
LAST_HOUR_SAMPLES = 14400

# --- contraction detection -------------------------------------------------
UC_TONUS_WIN = int(10 * 60 * FS)      # 10 min rolling percentile = resting tone
UC_TONUS_PCT = 10
UC_MIN_PROMINENCE = 12.0              # UC units above tonus
UC_MIN_SEPARATION = int(60 * FS)      # contractions at least 60 s apart
UC_EDGE_FRAC = 0.30                   # onset/offset at 30 % of prominence
HALF_WIDTH_MIN_SEC = 22               # clamp for crowded contractions
HALF_WIDTH_MAX_SEC = 60

# --- response measurement --------------------------------------------------
PRE_REF_SEC = 60                      # pre-contraction FHR reference window
POST_SEARCH_SEC = 90                  # look this far past offset for the nadir
RECOVERY_TOL_BPM = 5.0                # "recovered" = within 5 bpm of pre-ref
MAX_MISSING_IN_RESPONSE = 0.50        # skip contractions with worse coverage
MIN_CONTRACTIONS = 4                  # fewer than this -> no trajectory

RESPONSE_NAMES: List[str] = [
    "depth",            # bpm below the pre-contraction reference
    "norm_depth",       # depth per unit of contraction amplitude
    "lag",              # seconds from UC peak to FHR nadir (+ = late)
    "recovery_sec",     # seconds from nadir back to within tolerance
    "recovered",        # 1 if it recovered before the next contraction
    "area",             # bpm.s below the reference
    "baseline_shift",   # pre-ref minus post-contraction reference
    "uc_amplitude",     # the stressor itself
]


def _rolling_percentile(x: np.ndarray, win: int, pct: float) -> np.ndarray:
    """Rolling percentile via strided views, edge-padded."""
    if win >= len(x):
        return np.full_like(x, np.percentile(x, pct))
    pad = win // 2
    xp = np.pad(x, (pad, win - pad - 1), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(xp, win)
    return np.percentile(view, pct, axis=1)


def detect_contractions(uc: np.ndarray, fs: float = FS) -> List[Dict[str, int]]:
    """Locate contractions in the UC channel.

    Returns one dict per contraction with sample indices for onset, peak and
    offset plus the peak amplitude above resting tone.
    """
    u = uc.astype(np.float64).copy()
    valid = u > 0
    if valid.sum() < fs * 300:                     # under 5 min of usable UC
        return []
    idx = np.arange(len(u))
    if (~valid).any():
        u[~valid] = np.interp(idx[~valid], idx[valid], u[valid])
    u = apply_lowpass_filter(u, fs=fs)
    tonus = _rolling_percentile(u, UC_TONUS_WIN, UC_TONUS_PCT)
    rel = u - tonus

    peaks, props = find_peaks(rel, prominence=UC_MIN_PROMINENCE,
                              distance=UC_MIN_SEPARATION)
    out: List[Dict[str, int]] = []
    half_lo, half_hi = int(HALF_WIDTH_MIN_SEC * fs), int(HALF_WIDTH_MAX_SEC * fs)
    for p, prom in zip(peaks, props["prominences"]):
        # Walk out to UC_EDGE_FRAC of the peak's OWN prominence, not of its
        # height above the rolling tonus: thresholding on the latter puts the
        # cut far up the flank when resting tone is low.
        thr = rel[p] - (1.0 - UC_EDGE_FRAC) * prom
        lo = p
        while lo > 0 and rel[lo] > thr:
            lo -= 1
        hi = p
        while hi < len(rel) - 1 and rel[hi] > thr:
            hi += 1
        # For closely-spaced contractions the prominence base is the SADDLE
        # between neighbours, so the walk terminates early and the contraction
        # looks 20 s long. Rejecting those would discard half of a normal
        # labour's contractions -- exactly the ones that matter, since crowding
        # is itself a stressor. Clamp the half-widths to a physiological range
        # instead of dropping the contraction.
        onset = p - int(np.clip(p - lo, half_lo, half_hi))
        offset = p + int(np.clip(hi - p, half_lo, half_hi))
        out.append({"onset": int(max(onset, 0)),
                    "peak": int(p),
                    "offset": int(min(offset, len(rel) - 1)),
                    "amplitude": float(prom)})
    return out


def measure_response(fhr: np.ndarray, bad: np.ndarray, c: Dict[str, int],
                     next_onset: Optional[int], fs: float = FS
                     ) -> Optional[np.ndarray]:
    """Characterise the FHR response to ONE contraction.

    `bad` marks samples that are NOT usable after the signal chain -- gaps
    longer than 15 s are deliberately left as 0.0 by interpolate_missing, and
    reading them as heart rate would fabricate a 140->0 deceleration. Every
    statistic below is therefore computed over usable samples only.

    Returns a vector in RESPONSE_NAMES order, or None if coverage is too poor.
    """
    n = len(fhr)
    pre_lo = max(0, c["onset"] - int(PRE_REF_SEC * fs))
    if c["onset"] - pre_lo < int(20 * fs):
        return None
    search_hi = min(n, c["offset"] + int(POST_SEARCH_SEC * fs))
    if search_hi - c["onset"] < int(30 * fs):
        return None
    if bad[pre_lo:search_hi].mean() > MAX_MISSING_IN_RESPONSE:
        return None

    pre_good = ~bad[pre_lo:c["onset"]]
    if pre_good.sum() < int(15 * fs):
        return None
    pre_ref = float(np.median(fhr[pre_lo:c["onset"]][pre_good]))

    seg = fhr[c["onset"]:search_hi]
    seg_good = ~bad[c["onset"]:search_hi]
    if seg_good.sum() < int(20 * fs):
        return None
    gidx = np.where(seg_good)[0]
    k = int(gidx[np.argmin(seg[gidx])])
    nadir_i = c["onset"] + k
    depth = pre_ref - float(seg[k])

    lag = (nadir_i - c["peak"]) / fs
    amp = max(c["amplitude"], 1e-6)

    # recovery: first usable sample after the nadir back within tolerance
    hi = min(n, nadir_i + int(300 * fs))
    tail, tail_good = fhr[nadir_i:hi], ~bad[nadir_i:hi]
    back = np.where(tail_good & (tail >= pre_ref - RECOVERY_TOL_BPM))[0]
    if len(back):
        recovery_sec = float(back[0] / fs)
        rec_idx = nadir_i + int(back[0])
        recovered = 1.0 if (next_onset is None or rec_idx < next_onset) else 0.0
    else:
        recovery_sec = float(len(tail) / fs)       # censored at 300 s
        recovered = 0.0

    below = np.clip(pre_ref - seg[seg_good], 0, None)
    # scale to the full response duration so partial coverage is comparable
    area = float(below.sum() / fs * (len(seg) / max(seg_good.sum(), 1)))

    post_lo = min(n, c["offset"] + int(60 * fs))
    post_hi = min(n, post_lo + int(60 * fs))
    post_good = ~bad[post_lo:post_hi]
    post_ref = (float(np.median(fhr[post_lo:post_hi][post_good]))
                if post_good.sum() > int(20 * fs) else pre_ref)
    baseline_shift = pre_ref - post_ref

    return np.array([depth, depth / amp, lag, recovery_sec, recovered,
                     area, baseline_shift, c["amplitude"]], dtype=np.float64)


# --- trajectory summarisation ----------------------------------------------
def _trend(v: np.ndarray) -> float:
    """Slope of v against contraction index, per contraction."""
    if len(v) < 3 or not np.isfinite(v).all():
        return np.nan
    t = np.arange(len(v), dtype=np.float64)
    t = (t - t.mean())
    denom = float((t * t).sum())
    return float((t * (v - v.mean())).sum() / denom) if denom > 0 else np.nan


def _third_contrast(v: np.ndarray) -> float:
    """Late-third mean minus early-third mean."""
    if len(v) < 6:
        return np.nan
    k = len(v) // 3
    return float(np.mean(v[-k:]) - np.mean(v[:k]))


TRAJ_STATS = ("mean", "sd", "worst", "trend", "late_minus_early")


def trajectory_features(R: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Summarise a patient's contraction-response sequence.

    R is (n_contractions, len(RESPONSE_NAMES)). The five statistics per response
    variable are what separate this from a window-level descriptor: `trend` and
    `late_minus_early` express DETERIORATION ACROSS CONTRACTIONS, which no
    existing feature can represent.
    """
    feats: List[float] = []
    for j in range(R.shape[1]):
        v = R[:, j]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            feats.extend([np.nan] * len(TRAJ_STATS))
            continue
        # "worst" is the max for depth-like variables, min for `recovered`
        worst = float(np.min(v)) if RESPONSE_NAMES[j] == "recovered" else float(np.max(v))
        feats.extend([float(np.mean(v)), float(np.std(v)), worst,
                      _trend(v), _third_contrast(v)])
    # labour-progression context: how the contractions themselves evolve
    feats.append(float(len(R)))
    if len(intervals) >= 2:
        feats.extend([float(np.mean(intervals)), _trend(intervals)])
    else:
        feats.extend([np.nan, np.nan])
    # deterioration summary: fraction of contractions failing to recover, and
    # the longest run of consecutive non-recoveries
    rec = R[:, RESPONSE_NAMES.index("recovered")]
    rec = rec[np.isfinite(rec)]
    if len(rec):
        feats.append(float(1.0 - rec.mean()))
        run = best = 0
        for r in rec:
            run = run + 1 if r < 0.5 else 0
            best = max(best, run)
        feats.append(float(best))
    else:
        feats.extend([np.nan, np.nan])
    return np.array(feats, dtype=np.float32)


def trajectory_feature_names() -> List[str]:
    names = [f"{r}_{s}" for r in RESPONSE_NAMES for s in TRAJ_STATS]
    names += ["n_contractions", "interval_mean", "interval_trend",
              "frac_unrecovered", "max_unrecovered_run"]
    return names


def patient_contraction_features(fhr_raw: np.ndarray, uc_raw: np.ndarray,
                                 fs: float = FS,
                                 second_start: float = -1.0
                                 ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Full per-patient pipeline: signal chain -> contractions -> trajectory.

    Returns (all_contraction_features, stage_split_features, n_contractions).
    `stage_split_features` repeats the trajectory summary separately over
    first-stage and second-stage contractions when the transition is known --
    this is the Route-1 x Route-2 combination.
    """
    f = remove_spikes(fhr_raw.astype(np.float64).copy(), fs=fs)
    f = interpolate_missing(f, clip_min=50.0, clip_max=240.0)
    # Gaps > 15 s survive interpolation as 0.0. Mark them unusable, then fill
    # them with the local level before filtering: a 0.0 step would make the
    # lowpass ring and smear a fake deceleration into the usable samples beside
    # the gap. The mask, not the fill, is what the statistics respect.
    bad = f <= 0.0
    if bad.any():
        if (~bad).sum() < int(60 * fs):
            n_names = len(trajectory_feature_names())
            return (np.full(n_names, np.nan, np.float32),
                    np.full(2 * n_names, np.nan, np.float32), 0)
        f[bad] = float(np.median(f[~bad]))
    f = apply_lowpass_filter(f, fs=fs)
    # the filter is zero-phase with a ~1 s impulse response; widen the mask so
    # samples adjacent to a gap are not trusted either
    if bad.any():
        w = int(4 * fs)
        bad = np.convolve(bad.astype(np.float32), np.ones(2 * w + 1), "same") > 0

    cons = detect_contractions(uc_raw, fs=fs)
    n_names = len(trajectory_feature_names())
    if len(cons) < MIN_CONTRACTIONS:
        return (np.full(n_names, np.nan, np.float32),
                np.full(2 * n_names, np.nan, np.float32), len(cons))

    rows, keep = [], []
    for i, c in enumerate(cons):
        nxt = cons[i + 1]["onset"] if i + 1 < len(cons) else None
        r = measure_response(f, bad, c, nxt, fs=fs)
        if r is not None:
            rows.append(r)
            keep.append(c)
    if len(rows) < MIN_CONTRACTIONS:
        return (np.full(n_names, np.nan, np.float32),
                np.full(2 * n_names, np.nan, np.float32), len(rows))

    R = np.vstack(rows)
    onsets = np.array([c["onset"] for c in keep], dtype=np.float64)
    intervals = np.diff(onsets) / fs / 60.0        # minutes between contractions
    allf = trajectory_features(R, intervals)

    # stage-aligned split
    if second_start is not None and second_start >= 0:
        m2 = onsets >= second_start
        parts = []
        for m in (~m2, m2):
            if m.sum() >= MIN_CONTRACTIONS:
                iv = np.diff(onsets[m]) / fs / 60.0
                parts.append(trajectory_features(R[m], iv))
            else:
                parts.append(np.full(n_names, np.nan, np.float32))
        stagef = np.concatenate(parts)
    else:
        stagef = np.full(2 * n_names, np.nan, np.float32)

    return allf, stagef, len(rows)
