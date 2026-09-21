"""
Phase 13.1 (docs/phase13_protocol.md Section 8, Option A) -- causal
recency-weighted P90 aggregation.

Tests a genuinely different hypothesis from every other Phase 13 branch:
does WHICH windows are pooled (and how heavily each counts) matter, holding
the underlying window-level P6 model fixed? This does not touch P6's own
horizon-selection scoring path -- it is a parallel aggregation pathway,
deliberately, so it can be compared against a plain-P90 control built the
exact same way rather than against the single-window horizon-selected
0.6872 baseline (see protocol Section 4/A4 -- comparing against a different
selection convention would confound aggregation strategy with horizon
strategy, the same trap the stride-vs-P6 baseline mismatch fell into
earlier this session).

Causal boundary (protocol A3-A4): at evaluation horizon h (h=0 is delivery),
a window i with time-before-delivery t_i is ELIGIBLE only if t_i >= h (it
was already observed by the time h is reached). For an eligible window,
elapsed time relative to the evaluation instant itself is
    dt_i = t_i - h      (>= 0 for every eligible window, by construction)
never t_delivery - t_i directly -- at h=0 those coincide (t_i - 0 = t_i),
which is fine because delivery IS the evaluation instant there; at h=30 they
do not, and using t_delivery - t_i instead of t_i - h would silently let a
window observed AFTER the h=30 evaluation point influence the score, which
this formulation forbids by restricting the pool to t_i >= h in the first
place.
"""

from typing import List
import numpy as np

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from src.evaluation.phase13_common import weighted_quantile, get_eligible_window_mask

HALF_LIFE_CANDIDATES_MIN: List[float] = [np.inf, 5.0, 10.0, 20.0]  # inf = A0, no weighting = plain P90


def half_life_to_lambda(half_life_min: float) -> float:
    if np.isinf(half_life_min):
        return 0.0
    return float(np.log(2.0) / half_life_min)


def compute_recency_weights(t_i: np.ndarray, h_val: float, half_life_min: float) -> np.ndarray:
    """w_i = exp(-lambda * dt_i), dt_i = t_i - h_val (>=0 for eligible windows only)."""
    lam = half_life_to_lambda(half_life_min)
    dt = t_i - h_val
    if lam == 0.0:
        return np.ones_like(t_i, dtype=np.float64)
    return np.exp(-lam * dt)


def patient_p90_scores(pred_arr: np.ndarray, patient_ids: np.ndarray, clean_pids: List[str],
                        t_del: np.ndarray, h_val: float, half_life_min: float, q: float = 0.90) -> np.ndarray:
    """
    Patient-level (recency-)weighted P90 of window-level predictions, over
    the causally-eligible window pool at horizon h_val. half_life_min=np.inf
    reduces exactly to plain P90 (uniform weights).
    """
    scores = []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        t_pts = t_del[idx]
        elig = get_eligible_window_mask(t_pts, h_val)
        if not np.any(elig):
            elig = np.ones_like(t_pts, dtype=bool)  # fallback: short recording, use everything available
        p_elig = pred_arr[idx[elig]]
        t_elig = t_pts[elig]
        w = compute_recency_weights(t_elig, h_val, half_life_min)
        scores.append(weighted_quantile(p_elig, w, q))
    return np.array(scores)
