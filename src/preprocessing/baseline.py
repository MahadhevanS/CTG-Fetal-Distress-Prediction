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


def rolling_iterative_baseline(fhr: np.ndarray, fs: float = 4.0,
                               win_sec: float = 300.0, step_sec: float = 30.0,
                               round_to_5: bool = False) -> np.ndarray:
    """Drift-tracking variant of calculate_iterative_baseline.

    ADDED 2026-09-04 (Phase 9A). calculate_iterative_baseline returns
    `np.full_like(fhr, baseline)` -- ONE constant per window, rounded to 5 bpm.
    Validated against FHRMA expert consensus
    (scripts/validate_detectors_fhrma.py), the expert baseline moves within a
    20-minute window by a median of 14.2 bpm (IQR 6.8-29.7); 47% of windows move
    by more than the 15 bpm FIGO event threshold itself. A constant cannot
    represent any of that.

    This applies the SAME iterative 15-bpm exclusion rule, but in a centred
    rolling window evaluated on a coarse grid and linearly interpolated, so the
    baseline can follow drift.

    Measured on FHRMA (452 windows, 156 recordings), event detection with the
    unchanged FIGO rule:

        estimator            acc F1   dec F1
        constant (current)    0.297    0.433
        rolling 5 min         0.464    0.497
        expert (upper bound)  0.598    0.616

    Caveat, stated because it is a real trade-off: the rolling baseline is
    BETTER for event detection but WORSE point-wise against the expert baseline
    (median abs difference 9.23 vs 3.93 bpm). It is not a strict improvement,
    and it does not reach the expert bound.

    Args:
        fhr: 1D FHR signal (bpm), gaps already handled.
        fs: sampling frequency (Hz).
        win_sec: centred window length. 300 s was best on FHRMA event F1;
            selected against expert annotations only, never against any
            CTU-UHB outcome label.
        step_sec: grid spacing for evaluation before interpolation.
        round_to_5: match the legacy 5 bpm clinical rounding.

    Returns:
        np.ndarray: baseline of the same length as `fhr`.
    """
    n = len(fhr)
    if n == 0:
        return np.asarray(fhr, dtype=float)
    half = int(win_sec * fs / 2)
    step = max(int(step_sec * fs), 1)
    grid = np.arange(0, n, step)
    vals = np.empty(len(grid), dtype=float)
    for i, c in enumerate(grid):
        seg = fhr[max(0, c - half):min(n, c + half)]
        b = float(np.median(seg))
        for _ in range(5):
            m = (seg >= b - 15) & (seg <= b + 15)
            if not np.any(m):
                break
            b = float(np.mean(seg[m]))
        vals[i] = b
    out = np.interp(np.arange(n), grid, vals)
    return np.round(out / 5.0) * 5.0 if round_to_5 else out
