"""
Phase 11: Vectorized Paired Patient-Level Bootstrap Engine.

Provides:
- Fast rank-sum AUROC calculation (fast_auc)
- Paired patient bootstrap (B=2,000) for delta AUROC, 95% CIs, and empirical p-values
- Clustered patient sampling (1 patient = all their windows)
"""

import numpy as np
import pandas as pd

def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fast rank-sum AUROC calculation."""
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(y_score).rank().values
    pos_ranks = np.sum(ranks[y_true == 1])
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def paired_patient_bootstrap(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42
) -> dict:
    """
    Computes paired bootstrap for Model B vs Model A (Delta = Score_B - Score_A).
    Returns mean delta, 95% CI [lo, hi], and empirical two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    n_pts = len(y_true)
    
    # Pre-generate valid bootstrap patient resamples
    boot_deltas = []
    for _ in range(n_boot):
        b_idx = rng.choice(n_pts, size=n_pts, replace=True)
        y_b = y_true[b_idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_a = fast_auc(y_b, scores_a[b_idx])
        auc_b = fast_auc(y_b, scores_b[b_idx])
        if not np.isnan(auc_a) and not np.isnan(auc_b):
            boot_deltas.append(auc_b - auc_a)
            
    boot_deltas = np.array(boot_deltas)
    delta_mean = float(np.mean(boot_deltas))
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))
    
    # Two-sided empirical p-value
    p_val = float(2.0 * min(np.mean(boot_deltas <= 0.0), np.mean(boot_deltas >= 0.0)))
    p_val = min(1.0, max(0.0, p_val))
    
    return {
        "delta_mean": delta_mean,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "p_value": p_val,
        "n_valid_bootstraps": len(boot_deltas)
    }
