import numpy as np

def assess_signal_quality(fhr: np.ndarray, missing_value: float = 0.0, max_missing_ratio: float = 0.3) -> bool:
    """
    Assesses if a signal window meets the quality threshold.
    
    Args:
        fhr (np.ndarray): Fetal heart rate signal window.
        missing_value (float): Value denoting missing signal (typically 0.0 in CTG).
        max_missing_ratio (float): Maximum allowed ratio of missing data.
        
    Returns:
        bool: True if the signal is valid (missing data <= threshold), False otherwise.
    """
    if len(fhr) == 0:
        return False
        
    missing_count = np.sum(fhr == missing_value)
    missing_ratio = missing_count / len(fhr)
    
    return missing_ratio <= max_missing_ratio

def get_valid_windows(fhr: np.ndarray, window_size: int, stride: int, max_missing_ratio: float = 0.3) -> list:
    """
    Extracts starting indices for all valid sliding windows within a continuous signal.
    
    Args:
        fhr: The continuous FHR signal.
        window_size: Size of the window in samples.
        stride: Stride in samples.
        max_missing_ratio: Maximum allowed missing signal ratio.
        
    Returns:
        list of int: Starting indices of valid windows.
    """
    valid_starts = []
    n_samples = len(fhr)
    
    for start_idx in range(0, n_samples - window_size + 1, stride):
        window = fhr[start_idx : start_idx + window_size]
        if assess_signal_quality(window, max_missing_ratio=max_missing_ratio):
            valid_starts.append(start_idx)
            
    return valid_starts
