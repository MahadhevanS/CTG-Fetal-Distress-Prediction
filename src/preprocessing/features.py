import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Tuple

def calculate_variability(fhr: np.ndarray, fs: float = 4.0) -> Tuple[float, float]:
    """
    Calculates Short-Term Variability (STV) and Long-Term Variability (LTV).
    
    Args:
        fhr (np.ndarray): Fetal Heart Rate signal (baseline corrected or raw).
        fs (float): Sampling frequency.
        
    Returns:
        Tuple[float, float]: (STV, LTV)
    """
    if len(fhr) < 2:
        return 0.0, 0.0
        
    # STV: Mean absolute beat-to-beat difference
    # In clinical systems like Dawes-Redman, STV is calculated over 1/16th minute epochs.
    # Here we use a generalized mean absolute derivative as a proxy for the continuous signal.
    stv = np.mean(np.abs(np.diff(fhr)))
    
    # LTV: Range or variance over larger windows (e.g., 1 minute)
    window_samples = int(60 * fs)
    if len(fhr) < window_samples:
        ltv = np.ptp(fhr) # Peak-to-peak if less than 1 min
    else:
        # Calculate LTV as the mean of peak-to-peak amplitudes in 1-minute windows
        n_windows = len(fhr) // window_samples
        ranges = []
        for i in range(n_windows):
            window = fhr[i*window_samples : (i+1)*window_samples]
            ranges.append(np.ptp(window))
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
