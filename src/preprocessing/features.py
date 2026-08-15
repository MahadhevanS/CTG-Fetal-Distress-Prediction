import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Optional, Tuple


def _event_exclusion_mask(fhr: np.ndarray, baseline: np.ndarray, fs: float,
                           threshold_bpm: float = 15.0, min_duration_s: float = 15.0) -> np.ndarray:
    """
    Flags samples belonging to a qualifying acceleration or deceleration episode
    (>=15 bpm deviation from baseline, sustained >=15s -- the same FIGO threshold
    detect_accelerations()/detect_decelerations() use).

    LITERATURE FIX (2026-08-15): per NICHD/FIGO convention, baseline variability
    is assessed EXCLUDING accelerations and decelerations -- e.g. "Baseline FHR
    variability is determined in a 10 minute window, excluding accelerations and
    decelerations" and clinically read from "the stable period between FHR
    decelerations" (see calculate_variability()'s docstring for full citation).
    A deceleration is, by definition, a >=15 bpm excursion -- if left inside a
    1-minute window's range computation, a single deceleration alone can push
    that window's measured "variability" well past the 25 bpm threshold even
    when the actual baseline fluctuation around it is completely normal. Given
    ~85% of windows in this dataset contain at least one variable deceleration,
    this was very likely the dominant remaining contributor to the elevated LTV
    readings that the percentile-range and detrending fixes alone didn't resolve.

    Returns:
        Boolean array, same shape as fhr. True = sample is inside a qualifying
        accel/decel episode and should be excluded from variability estimation.
    """
    deviation = np.abs(fhr - baseline)
    is_event = deviation >= threshold_bpm
    duration_samples = int(min_duration_s * fs)

    changes = np.diff(is_event.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if len(is_event) > 0 and is_event[0]:
        starts = np.insert(starts, 0, 0)
    if len(is_event) > 0 and is_event[-1]:
        ends = np.append(ends, len(fhr))

    mask = np.zeros_like(is_event)
    for start, end in zip(starts, ends):
        if (end - start) >= duration_samples:
            mask[start:end] = True
    return mask


def calculate_variability(fhr: np.ndarray, fs: float = 4.0,
                           baseline: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Calculates Short-Term Variability (STV) and Long-Term Variability (LTV).

    Args:
        fhr (np.ndarray): Fetal Heart Rate signal (baseline corrected or raw).
        fs (float): Sampling frequency.
        baseline (np.ndarray, optional): Estimated baseline FHR, same shape as
            fhr (e.g. from calculate_iterative_baseline()). If given, samples
            belonging to a qualifying acceleration/deceleration episode are
            excluded from the LTV range computation -- see
            _event_exclusion_mask()'s docstring. If omitted, LTV is computed
            over the full window (previous behaviour), which will over-estimate
            variability on any window containing a deceleration.

    Returns:
        Tuple[float, float]: (STV, LTV)
    """
    if len(fhr) < 2:
        return 0.0, 0.0

    # STV: Mean absolute beat-to-beat difference
    # In clinical systems like Dawes-Redman, STV is calculated over 1/16th minute epochs.
    # Here we use a generalized mean absolute derivative as a proxy for the continuous signal.
    # Left unaffected by the event-exclusion fix below: STV averages over thousands of
    # beat-to-beat diffs across the whole window, so a handful of samples inside a
    # deceleration dilute out far less than they do in a 1-minute range statistic.
    stv = np.mean(np.abs(np.diff(fhr)))

    # LTV: Range over larger windows (e.g., 1 minute)
    # CALIBRATION FIX (2026-08-15): previously used np.ptp() (true max-min), which
    # is extremely sensitive to a single noisy/artifact sample within the window --
    # even after spike removal and the physiological interpolation clamp, one
    # residual outlier sample inflates the ENTIRE window's LTV. Verified against
    # the corrected CTU-CHB training split: 0% of windows showed LTV < 5 bpm
    # (FIGO's "reduced variability" band was structurally unreachable) while 81.7%
    # showed LTV > 25 bpm ("increased/saltatory") -- not a plausible clinical
    # distribution, a calibration artifact. Two changes, verified additive on real
    # reconstructed local data (mean LTV 46.6 -> 36.7 -> 30.9 bpm, >25bpm windows
    # 80.8% -> 67.8% -> 59.5% as each is added):
    #   1. Percentile range (p95-p5) instead of true min-max per 1-minute window --
    #      discards only the most extreme ~10% of samples (where residual noise
    #      concentrates) while still fully capturing genuine variability in the
    #      other 90%, unlike a tighter IQR-style range that would risk
    #      under-counting real physiological swings.
    #   2. Local linear detrending within each 1-minute window before measuring
    #      the range -- a genuine slow baseline drift *within* the minute (the
    #      pipeline only fits one baseline per 20-minute window, so intra-window
    #      drift was previously counted as "variability" rather than trend).
    # HONEST CAVEAT (superseded by the exclusion fix below, kept for history): even
    # combined, ~59.5% of windows in this (distress-oversampled) training split
    # still read above FIGO's 25 bpm band -- meaningfully better than 80.8%, but
    # not full resolution.
    #
    # THIRD FIX (2026-08-15, literature-driven): live search of the NICHD/FIGO
    # intrapartum monitoring literature (prompted directly by the user asking
    # to check prior art rather than keep guessing) surfaced that baseline
    # variability is clinically read EXCLUDING accelerations/decelerations, not
    # over the raw signal -- see _event_exclusion_mask()'s docstring for the
    # citations and full reasoning.
    #
    # VERIFIED END-TO-END on a real Colab pipeline regeneration (all 3 fixes
    # combined, true per-sample iterative baseline feeding this function
    # directly, n=6177 windows): mean LTV 46.26 -> 28.00 bpm, >25bpm prevalence
    # 81.7% -> 53.4%, 5-25bpm ("normal") coverage 18.3% -> 46.5%. A local
    # reconstruction test (constant-baseline approximation, run before this
    # real regeneration) had predicted the real fix would land a bit lower
    # (mean ~26.0, >25bpm ~48.9%) than it actually did (28.0 / 53.4%) --
    # i.e. the constant-baseline approximation UNDER-stated the true LTV,
    # the opposite of what was originally guessed; noted here so that
    # reconstruction-based estimates are trusted less in future work on this
    # function.
    #
    # HONEST CAVEAT: still not full resolution. >25bpm prevalence, while
    # roughly halved from the original bug, remains a majority of windows --
    # plausible remaining causes: (a) this split's stride-based oversampling
    # of distress patients skews toward genuinely more erratic intrapartum
    # traces than a general population sample; and/or (b) FIGO's 5-25 bpm
    # band may be implicitly calibrated against a different underlying
    # measurement convention (e.g. a coarser sampling/smoothing convention on
    # older clinical monitors) than this pipeline's raw 4 Hz signal
    # reproduces. <5bpm ("reduced") prevalence is genuinely ~0.1% even with
    # the true iterative baseline -- CONFIRMED not a reconstruction artifact
    # (it was ~0% in the local approximation too). This matches
    # derive_figo_criteria_flags()'s independent finding of 0% prevalence for
    # `variability_reduced`, which is why that flag is excluded from
    # FIGOCriteriaHead -- this is now cross-verified as a real property of
    # this dataset/pipeline rather than a bug. Treat variability-threshold-
    # based judgments (classify_figo, the rule loss, FIGOCriteriaHead's
    # remaining variability flags) as meaningfully improved and now verified
    # against a real end-to-end pipeline run, but the >25bpm elevation is a
    # known, documented residual gap rather than a solved problem.
    window_samples = int(60 * fs)

    # Event exclusion (2026-08-15): when a baseline is supplied, drop samples inside
    # a qualifying accel/decel episode before measuring each window's range -- see
    # _event_exclusion_mask()'s docstring. MIN_VALID_FRACTION guards against a
    # sub-window that's almost entirely inside one long episode (e.g. a prolonged
    # deceleration spanning most of the minute): with too few genuine "baseline"
    # samples left, the detrended range is not a meaningful variability estimate,
    # so that window falls back to using its full (unmasked) samples instead of a
    # near-empty, noise-dominated slice.
    MIN_VALID_FRACTION = 0.5
    event_mask = _event_exclusion_mask(fhr, baseline, fs) if baseline is not None else None

    def _detrended_range(window: np.ndarray) -> float:
        t = np.arange(len(window))
        slope, intercept = np.polyfit(t, window, 1)
        detrended = window - (slope * t + intercept) + window.mean()
        return float(np.percentile(detrended, 95) - np.percentile(detrended, 5))

    def _window_ltv(window: np.ndarray, window_event_mask: Optional[np.ndarray]) -> float:
        if window_event_mask is not None:
            valid = window[~window_event_mask]
            if len(valid) >= max(2, int(MIN_VALID_FRACTION * len(window))):
                return _detrended_range(valid)
        return _detrended_range(window)

    if len(fhr) < window_samples:
        ltv = _window_ltv(fhr, event_mask)
    else:
        # Calculate LTV as the mean of the detrended p95-p5 range in each 1-minute window
        n_windows = len(fhr) // window_samples
        ranges = [
            _window_ltv(
                fhr[i*window_samples : (i+1)*window_samples],
                event_mask[i*window_samples : (i+1)*window_samples] if event_mask is not None else None,
            )
            for i in range(n_windows)
        ]
        ltv = np.mean(ranges)

    return float(stv), float(ltv)

def detect_accelerations(fhr: np.ndarray, baseline: np.ndarray, fs: float = 4.0) -> int:
    """
    Detects FIGO accelerations: increase >= 15 bpm lasting >= 15 seconds.
    
    Args:
        fhr (np.ndarray): The FHR signal.
        baseline (np.ndarray): The extracted baseline FHR.
        fs (float): Sampling frequency.
        
    Returns:
        int: Number of accelerations detected in the window.
    """
    threshold_bpm = 15.0
    duration_samples = int(15 * fs)
    
    # Boolean mask where FHR is 15 bpm above baseline
    is_accel = (fhr - baseline) >= threshold_bpm
    
    # Find continuous regions of True
    changes = np.diff(is_accel.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    
    if is_accel[0]:
        starts = np.insert(starts, 0, 0)
    if is_accel[-1]:
        ends = np.append(ends, len(fhr))
        
    # Count regions that last at least duration_samples
    count = np.sum((ends - starts) >= duration_samples)
    return int(count)

def detect_decelerations(fhr: np.ndarray, baseline: np.ndarray, uc: np.ndarray, fs: float = 4.0) -> Dict[str, int]:
    """
    Detects and categorizes FIGO decelerations (Early, Late, Variable, Prolonged).
    
    Args:
        fhr (np.ndarray): The FHR signal.
        baseline (np.ndarray): The extracted baseline FHR.
        uc (np.ndarray): The Uterine Contraction signal.
        fs (float): Sampling frequency.
        
    Returns:
        Dict[str, int]: Counts of each deceleration type.
    """
    results = {'early': 0, 'late': 0, 'variable': 0, 'prolonged': 0}
    
    threshold_bpm = 15.0
    duration_samples = int(15 * fs)
    prolonged_samples = int(120 * fs) # 2 minutes
    
    is_decel = (baseline - fhr) >= threshold_bpm
    
    changes = np.diff(is_decel.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    
    if len(is_decel) > 0 and is_decel[0]:
        starts = np.insert(starts, 0, 0)
    if len(is_decel) > 0 and is_decel[-1]:
        ends = np.append(ends, len(fhr))
        
    # Find UC peaks to correlate with decelerations
    # Smoothing UC signal slightly to find true peaks
    uc_peaks, _ = find_peaks(uc, distance=int(30*fs), prominence=10)
    
    for start, end in zip(starts, ends):
        duration = end - start
        
        # Must be at least 15 seconds
        if duration < duration_samples:
            continue
            
        if duration >= prolonged_samples:
            results['prolonged'] += 1
            continue
            
        # Find nadir (lowest FHR point) of this deceleration
        decel_segment = fhr[start:end]
        nadir_idx = start + np.argmin(decel_segment)
        onset_duration = nadir_idx - start
        
        # Variable Deceleration: rapid descent (onset to nadir < 30 seconds)
        if onset_duration < (30 * fs):
            results['variable'] += 1
            continue
            
        # For Early/Late, correlate nadir with nearest UC peak
        if len(uc_peaks) > 0:
            # Find closest UC peak to the nadir
            closest_uc_idx = uc_peaks[np.argmin(np.abs(uc_peaks - nadir_idx))]
            time_diff = nadir_idx - closest_uc_idx
            
            # If nadir occurs after UC peak (e.g., > 15 seconds delay), it's Late
            if time_diff > (15 * fs):
                results['late'] += 1
            else:
                results['early'] += 1
        else:
            # If no UC peak found nearby but slow descent, default to early or unclassified
            results['early'] += 1
            
    return results
