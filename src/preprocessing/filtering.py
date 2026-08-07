import numpy as np
from scipy import interpolate
from scipy.signal import butter, filtfilt
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# GAP 1 FIX: Spike Removal
# ---------------------------------------------------------------------------

def remove_spikes(signal: np.ndarray, fs: float = 4.0,
                  max_rate_bpm_per_sec: float = 25.0) -> np.ndarray:
    """
    Removes unphysiological FHR spikes by detecting sample-to-sample rate
    changes that exceed the physiological limit of 25 bpm/sec.

    At 4 Hz, the maximum allowed delta per sample = 25 / 4 = 6.25 bpm.
    Spike samples are zeroed out (treated as missing) and subsequently
    handled by the cubic spline interpolation step that follows in the pipeline.

    Clinical Justification:
        Fetal heart rate changes do not occur at frequencies above 1.5 Hz
        (90 bpm per second). High-frequency spikes are transducer artifacts and
        must be removed to avoid misclassifying them as accelerations or variable
        decelerations. (preprocessing_technical_documentation.md §4.2)

    Args:
        signal (np.ndarray): The 1D FHR signal (bpm).
        fs (float): Sampling frequency in Hz. Default is 4.0 Hz (CTU-CHB standard).
        max_rate_bpm_per_sec (float): Maximum physiologically plausible rate-of-change
                                      in bpm/second. Default: 25 bpm/sec.

    Returns:
        np.ndarray: Signal copy with spike samples replaced by 0.0
                    (the missing value marker used throughout this pipeline).
    """
    if len(signal) < 2:
        return signal.copy()

    processed = signal.copy()
    max_delta_per_sample = max_rate_bpm_per_sec / fs  # e.g., 6.25 bpm at 4 Hz

    # Compute absolute sample-to-sample differences
    deltas = np.abs(np.diff(processed))

    # A spike at index i+1 is indicated by an excessive delta between i and i+1
    spike_mask = np.zeros(len(processed), dtype=bool)
    spike_positions = np.where(deltas > max_delta_per_sample)[0] + 1
    spike_mask[spike_positions] = True

    # Zero out spike samples — treated as missing data for the interpolation step
    processed[spike_mask] = 0.0

    return processed


# ---------------------------------------------------------------------------
# Existing: Missing Data Interpolation
# ---------------------------------------------------------------------------

def interpolate_missing(signal: np.ndarray, missing_value: float = 0.0,
                        max_gap_samples: int = 60,
                        clip_min: Optional[float] = None,
                        clip_max: Optional[float] = None) -> np.ndarray:
    """
    Interpolates short gaps in the signal using cubic spline interpolation.

    Gaps of <= 15 seconds (60 samples at 4 Hz) are interpolated. Longer gaps
    are left as 0.0 to prevent fabricating bradycardia or decelerations that
    did not actually occur.

    Clinical Justification:
        Fetal heart rate is controlled by the autonomic nervous system, which
        changes the rate smoothly — not in jagged, linear steps. Cubic spline
        interpolation preserves the first and second derivatives, mimicking the
        natural acceleration and deceleration limits of the fetal heart.
        (preprocessing_justification.md §2.2)

    BUG FIX (knowledge-infusion audit, 2026-08-07):
        CubicSpline is fit with extrapolate=True so that gap edges touching the
        signal boundary can still be filled. Unconstrained cubic extrapolation is
        unstable near sparse or boundary knots and can overshoot far beyond the
        valid signal range (observed: fitted values in the thousands of bpm on a
        signal whose real range is ~50-240 bpm), which then propagates into every
        downstream statistic (baseline, STV, LTV, Z-score scalers). clip_min/
        clip_max bound the *interpolated fill values only* — real, already-valid
        samples are never touched — so a repaired gap can never be more
        unphysiological than the artifact it was meant to replace.

    Args:
        signal (np.ndarray): The 1D signal array (e.g., FHR).
        missing_value (float): Value representing missing data (usually 0.0).
        max_gap_samples (int): Maximum continuous missing samples to interpolate.
                               (e.g., 15 seconds at 4Hz = 60 samples).
        clip_min (float, optional): Lower physiological bound to clamp interpolated
                                    fill values to (e.g., 50.0 bpm for FHR). None
                                    disables lower clamping (default; preserves
                                    prior behaviour for non-FHR callers).
        clip_max (float, optional): Upper physiological bound to clamp interpolated
                                    fill values to (e.g., 240.0 bpm for FHR). None
                                    disables upper clamping.

    Returns:
        np.ndarray: The interpolated signal.
    """
    # Create a copy to avoid modifying the original
    processed_signal = signal.copy()

    # Boolean mask of valid and missing data
    valid_mask = processed_signal != missing_value
    missing_mask = ~valid_mask

    # If no missing data, return original
    if not np.any(missing_mask):
        return processed_signal

    # If all missing data, cannot interpolate
    if not np.any(valid_mask):
        return processed_signal

    # Find continuous blocks of missing data
    changes = np.diff(missing_mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # Handle edge cases where signal starts or ends with missing data
    if missing_mask[0]:
        starts = np.insert(starts, 0, 0)
    if missing_mask[-1]:
        ends = np.append(ends, len(signal))

    # Calculate lengths of missing blocks
    gap_lengths = ends - starts

    # Create a mask for interpolation (only interpolate gaps <= max_gap_samples)
    interp_mask = np.zeros_like(missing_mask, dtype=bool)
    for start, end, length in zip(starts, ends, gap_lengths):
        if length <= max_gap_samples:
            interp_mask[start:end] = True

    # Interpolate only the selected gaps
    x = np.arange(len(signal))
    valid_x = x[valid_mask]
    valid_y = processed_signal[valid_mask]

    if len(valid_x) > 3:  # Cubic spline requires at least 4 points
        # extrapolate=True ensures boundary points do not become NaN
        spline = interpolate.CubicSpline(valid_x, valid_y, extrapolate=True)
        x_interp = x[interp_mask]
        interp_vals = spline(x_interp)
        if clip_min is not None or clip_max is not None:
            interp_vals = np.clip(interp_vals, clip_min, clip_max)
        processed_signal[interp_mask] = np.nan_to_num(interp_vals, nan=0.0)

    # Hard safety guard: ensure no NaN values remain
    return np.nan_to_num(processed_signal, nan=0.0)


# ---------------------------------------------------------------------------
# Existing: Low-Pass Butterworth Filter
# ---------------------------------------------------------------------------

def apply_lowpass_filter(signal: np.ndarray, fs: float,
                         cutoff: float = 1.5, order: int = 4) -> np.ndarray:
    """
    Applies a zero-phase low-pass Butterworth filter to smooth the signal.

    A 4th-order filter provides a steep frequency roll-off while minimising
    the Gibbs phenomenon (ringing artifacts). The forward-backward filtfilt
    implementation prevents phase shift, which is critical for preserving
    the temporal alignment between deceleration troughs and UC peaks used to
    distinguish early vs. late decelerations.

    Clinical Justification:
        FHR signals contain high-frequency noise from fetal movement, maternal
        heart rate crossover, and minor sensor displacements. A low-pass filter
        at 1.5 Hz removes these non-physiological fluctuations.
        (preprocessing_technical_documentation.md §4.2)

    Args:
        signal (np.ndarray): The 1D signal to filter.
        fs (float): Sampling frequency of the signal (Hz).
        cutoff (float): Cutoff frequency (Hz). Default: 1.5 Hz.
        order (int): Order of the Butterworth filter. Default: 4.

    Returns:
        np.ndarray: Filtered signal, or original if signal is too short.
    """
    # Clean input signal before filtering
    clean_signal = np.nan_to_num(signal, nan=0.0)

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # filtfilt requires len(signal) > 3 * max(len(a), len(b))
    if len(clean_signal) > 3 * max(len(a), len(b)):
        filtered_signal = filtfilt(b, a, clean_signal)
        return np.nan_to_num(filtered_signal, nan=0.0)
    else:
        return clean_signal


# ---------------------------------------------------------------------------
# GAP 5 FIX: Per-Channel Z-Score Normalization
# ---------------------------------------------------------------------------

def zscore_normalize_channels(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies per-channel Z-score normalization to a dataset of signal windows.

    The scaler (mean and std) must be fit on the TRAINING set ONLY and then
    applied identically to val/test sets. This prevents any data leakage from
    the evaluation sets into the normalization statistics.

    The normalization operates at the channel level:
      - Channel 0 (Baseline-Corrected FHR): Already centered near 0 by baseline
        subtraction. Z-score divides by the training std to enforce unit variance,
        stabilising gradient flow.
      - Channel 1 (UC): Full Z-score applied (subtract mean, divide by std).

    Formula per channel c:
        X_norm[:, c, :] = (X[:, c, :] - mean[c]) / std[c]

    Clinical Justification:
        Normalising to unit variance prevents the neural network from being
        biased by absolute amplitude offsets, which are patient-specific and
        sensor-dependent. This is especially important for the UC channel which
        can have highly variable amplitude across recordings.
        (detailed_preprocessing_plan.md §7.3)

    Args:
        X (np.ndarray): Array of shape (N, C, T) — N windows, C channels, T samples.
        mean (np.ndarray, optional): Pre-computed per-channel mean of shape (C,).
                                     If None, computed from X (use for training set).
        std (np.ndarray, optional): Pre-computed per-channel std of shape (C,).
                                    If None, computed from X (use for training set).

    Returns:
        Tuple:
            X_norm (np.ndarray): Normalised array of shape (N, C, T).
            mean (np.ndarray): Per-channel mean used, shape (C,).
            std (np.ndarray): Per-channel std used, shape (C,).
    """
    # Clean input array of any non-finite values
    X_clean = np.nan_to_num(X, nan=0.0)

    if mean is None:
        # Compute across N and T dimensions, independently per channel
        # X shape: (N, C, T) → mean over axes 0 and 2 → shape (C,)
        mean = X_clean.mean(axis=(0, 2))

    if std is None:
        std = X_clean.std(axis=(0, 2))
        # Guard against zero standard deviation (e.g., constant signal segments)
        std = np.where(std == 0.0, 1.0, std)

    assert not np.isnan(mean).any(), f"Z-score mean contains NaN values: {mean}"
    assert not np.isnan(std).any(), f"Z-score std contains NaN values: {std}"

    # Broadcast: (N, C, T) with (C,) shaped mean/std
    X_norm = (X_clean - mean[None, :, None]) / std[None, :, None]
    X_norm = np.nan_to_num(X_norm, nan=0.0)

    return X_norm, mean, std
