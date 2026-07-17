import numpy as np
from scipy import interpolate
from scipy.signal import butter, filtfilt

def interpolate_missing(signal: np.ndarray, missing_value: float = 0.0, max_gap_samples: int = 60) -> np.ndarray:
    """
    Interpolates short gaps in the signal using cubic spline interpolation.
    
    Args:
        signal (np.ndarray): The 1D signal array (e.g., FHR).
        missing_value (float): Value representing missing data (usually 0.0).
        max_gap_samples (int): Maximum continuous missing samples to interpolate.
                               (e.g., 15 seconds at 4Hz = 60 samples).
                               
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
    # np.diff on the mask finds the edges (transitions between valid and missing)
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
    # We fit the spline on all valid data points
    valid_x = x[valid_mask]
    valid_y = processed_signal[valid_mask]
    
    if len(valid_x) > 3: # Cubic spline requires at least 4 points
        spline = interpolate.CubicSpline(valid_x, valid_y, extrapolate=False)
        # Apply interpolation to the selected gaps
        x_interp = x[interp_mask]
        processed_signal[interp_mask] = spline(x_interp)
        
    return processed_signal

def apply_lowpass_filter(signal: np.ndarray, fs: float, cutoff: float = 1.5, order: int = 4) -> np.ndarray:
    """
    Applies a low-pass Butterworth filter to smooth the signal.
    
    Args:
        signal (np.ndarray): The 1D signal to filter.
        fs (float): Sampling frequency of the signal (Hz).
        cutoff (float): Cutoff frequency (Hz).
        order (int): Order of the Butterworth filter (2-4 recommended).
        
    Returns:
        np.ndarray: Filtered signal.
    """
    # Design the filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply forward-backward filter to prevent phase shift (filtfilt)
    # Note: filtfilt requires len(signal) > 3 * max(len(a), len(b))
    if len(signal) > 3 * max(len(a), len(b)):
        # To avoid filtering artifacts on the edges or across remaining 0-value gaps,
        # one might want to split the signal into continuous valid segments first.
        # For this base implementation, we apply it across the whole array.
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    else:
        return signal
