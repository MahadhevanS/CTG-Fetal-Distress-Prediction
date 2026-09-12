"""
Shared evaluation primitives for Phase 13 (docs/phase13_protocol.md).

Kept in one place deliberately: every Phase 13 branch (recency-P90, stride,
Huber fusion, parity) must use identical horizon selection and identical
logit-fusion/lambda-selection logic, or cross-branch comparisons stop being
apples-to-apples. This module is the single source of truth for both.
"""

from typing import Optional, Tuple
import numpy as np

EPS = 1e-6


def get_patient_scores_at_horizon(pred_arr, patient_ids, clean_pids, t_del, h_val):
    """
    EXISTING convention (matches every P6 number reported this session,
    including the locked 0.6872 / 0.5857). For h>0: first CHRONOLOGICAL
    window satisfying t_i >= h -- for most patients this is close to their
    earliest window, not the window nearest h. Retained for compatibility
    with historical numbers (Phase 13 protocol Section 4).
    """
    scores = []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        t_pts = t_del[idx]
        if h_val > 0:
            eligible = np.where(t_pts >= h_val)[0]
            chosen = idx[eligible[0]] if len(eligible) > 0 else idx[-1]
        else:
            chosen = idx[-1]
        scores.append(pred_arr[chosen])
    return np.array(scores)


def get_patient_scores_at_horizon_corrected(pred_arr, patient_ids, clean_pids, t_del, h_val):
    """
    CORRECTED convention (Phase 13 protocol Section 4): the window closest
    to h among those already observed at h (t_i >= h), i.e.
        i* = argmin_i |t_i - h|   subject to   t_i >= h
    Delivery (h=0) is identical to the existing convention (last window).
    Falls back to the patient's last available window if none satisfy
    t_i >= h (short recordings), matching the existing convention's fallback.
    """
    scores = []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        t_pts = t_del[idx]
        if h_val > 0:
            eligible_mask = t_pts >= h_val
            if np.any(eligible_mask):
                eligible_idx = idx[eligible_mask]
                eligible_t = t_pts[eligible_mask]
                chosen = eligible_idx[np.argmin(np.abs(eligible_t - h_val))]
            else:
                chosen = idx[-1]
        else:
            chosen = idx[-1]
        scores.append(pred_arr[chosen])
    return np.array(scores)


def get_eligible_window_mask(t_pts: np.ndarray, h_val: float) -> np.ndarray:
    """Windows already observable at horizon h (t_i >= h); all windows if h<=0."""
    if h_val > 0:
        return t_pts >= h_val
    return np.ones_like(t_pts, dtype=bool)


def to_logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def from_logit(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def select_lambda_trainfold(logit_base_train: np.ndarray, logit_aux_train: np.ndarray,
                             y_train: np.ndarray, lambda_grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Phase 4's exact fusion-lambda selection (scripts/phase4_fusion_eval.py:426-441):
    a single scalar maximizing TRAINING-FOLD AUROC on logit_base + lambda*logit_aux,
    applied unchanged to that fold's held-out patients. No separate inner
    validation split -- a 1-D, ~30-point grid on two already-computed logits
    is low enough capacity that fit-on-train/apply-to-held-out is safe (unlike
    the 2-D beta/span grid that required true nested CV in the information-
    density work). Returns (best_lambda, best_train_auroc).
    """
    from sklearn.metrics import roc_auc_score
    if lambda_grid is None:
        lambda_grid = np.linspace(0.1, 3.0, 30)
    best_lam, best_auc = 1.0, -1.0
    for lam in lambda_grid:
        comb = logit_base_train + lam * logit_aux_train
        auc = roc_auc_score(y_train, comb)
        if auc > best_auc:
            best_auc, best_lam = auc, lam
    return float(best_lam), float(best_auc)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    Weighted q-th quantile (q in [0,1]) via the standard weighted-percentile
    definition: sort by value, find the value where cumulative weight
    fraction crosses q. Used for both plain P90 (uniform weights) and
    recency-weighted P90 (exponential-decay weights) so both go through the
    identical estimator -- only the weights differ between the two.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    if total_w <= 0:
        return float(np.mean(values))
    cum_frac = (cum_w - 0.5 * w_sorted) / total_w
    return float(np.interp(q, cum_frac, v_sorted))
