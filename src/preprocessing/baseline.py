import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def asymmetric_least_squares_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.5, n_iter: int = 10) -> np.ndarray:
    """
    Estimates the baseline of a signal using Asymmetric Least Squares (ALS) smoothing.
    This robustly finds the baseline while ignoring transient spikes (like accelerations
    or decelerations) by weighting positive and negative deviations differently.
    
    Args:
        y (np.ndarray): The 1D input signal (e.g., FHR).
        lam (float): Smoothness parameter. Higher values make the baseline stiffer.
        p (float): Asymmetry parameter. For FHR, 0.5 balances peak/valley rejection.
        n_iter (int): Number of iterations for re-weighting.
        
    Returns:
        np.ndarray: The estimated baseline signal.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    
    for i in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
        
    return z

def calculate_iterative_baseline(fhr: np.ndarray, window_size: int = 600) -> np.ndarray:
    """
    A simpler iterative baseline commonly used in CTG analysis that excludes 
    large excursions (potential accelerations/decelerations) to find the true mean.
    
    Args:
        fhr (np.ndarray): The 1D FHR signal.
        window_size (int): Not strictly needed if processing pre-windowed 20-min chunks,
                           but kept for parameterization.
                           
    Returns:
        np.ndarray: The baseline value array (constant or slowly varying).
    """
    # 1. Start with the median of the entire window
    baseline = np.median(fhr)
    
    # 2. Iteratively refine by excluding values > 15bpm or < 15bpm from current baseline
    # (FIGO threshold for accels/decels)
    for _ in range(5):
        valid_mask = (fhr >= baseline - 15) & (fhr <= baseline + 15)
        if not np.any(valid_mask):
            break
        baseline = np.mean(fhr[valid_mask])
        
    # Round to nearest 5 bpm as per clinical standard
    baseline = np.round(baseline / 5.0) * 5.0
    
    return np.full_like(fhr, baseline)
