"""
Continuous deceleration-burden candidates, and the machinery to compare them
against the binary `decel_repetitive` criterion.

WHY THIS MODULE EXISTS
----------------------
`decel_repetitive` is defined in descriptors.py as

    len(paired_contractions) / n_contractions > 0.50

and it drives 69.8% of early-warning deterioration events (63.7% alone) and
65.8% of Suspicious epochs. The median epoch carries 4 contractions, so the
test is decided by whether 2 or 3 of them carry a deceleration -- ONE
deceleration flips the FIGO state. The detector supplying that deceleration
runs at precision 0.509 against FHRMA expert consensus.

This module computes continuous alternatives. It does NOT define a new FIGO
rule: that is gated on FHRMA showing the continuous measures are more
faithful (Phase 3), and on a clinically defensible threshold existing
(Phase 5). Nothing here is fitted to any prediction target.

NOTHING HERE MODIFIES descriptors.py OR rules.py.
v3 labels stay exactly as built.
"""

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

import numpy as np

from .descriptors import (DECEL_UC_PAIRING_TOLERANCE_S, EVENT_MIN_DURATION_S,
                          delineate_events, detect_contractions)


@dataclass
class DecelBurden:
    """
    Continuous descriptions of deceleration burden in one epoch.

    Every quantity is defined in the docstring of the function that computes
    it. None of them is thresholded here.
    """
    # --- A. contraction-associated fraction (the >50% rule, un-thresholded)
    frac_contractions_with_decel: float   # in [0, 1]; the raw ratio
    n_contractions: int
    n_contractions_with_decel: int
    # --- B. severity-weighted burden, per minute of observation
    decel_area_per_min: float             # bpm*s of deficit per minute
    depth_x_duration_per_min: float       # bpm*s per minute, peak-depth form
    frac_time_in_decel: float             # unitless, in [0, 1]
    # --- C. contraction-normalised burden
    area_per_contraction: float           # bpm*s per contraction
    mean_decel_depth: float               # bpm
    mean_decel_duration_s: float          # s
    # --- D. temporal clustering / repetition
    max_consecutive_uc_with_decel: int    # longest run of affected contractions
    decel_interval_cv: float              # CV of inter-deceleration gaps
    decel_trend_per_min: float            # slope of per-minute decel count
    severity_trend: float                 # slope of per-deceleration depth
    # --- bookkeeping
    observation_min: float                # usable minutes the burden is over
    n_decels: int

    @staticmethod
    def names() -> List[str]:
        return [f.name for f in fields(DecelBurden)]

    def to_vector(self) -> np.ndarray:
        return np.array([float(getattr(self, n)) for n in self.names()],
                        dtype=np.float32)


def deceleration_events(fhr: np.ndarray, baseline: float, fs: float,
                        valid: np.ndarray) -> List[Dict]:
    """
    Every qualifying deceleration with its geometry.

    Uses the SAME delineation as descriptors.py (hysteresis edge + merge, the
    configuration validated against FHRMA at deceleration F1 0.603), so a
    difference between the binary and continuous measures cannot be explained
    by a different set of events.

    Returns one dict per deceleration with:
        start, end     sample indices
        nadir          sample index of the minimum
        depth_bpm      baseline - min(FHR)              [peak depth]
        duration_s     (end - start) / fs
        area_bpm_s     integral of max(baseline - FHR, 0) over the event,
                       in bpm*seconds -- the "deficit area", which is the
                       quantity a clinician means by a deep, long
                       deceleration being worse than a shallow, short one
        onset_s        time from start to nadir
    """
    out: List[Dict] = []
    need = int(EVENT_MIN_DURATION_S * fs)
    for s, e in delineate_events(fhr, baseline, fs, valid, -1):
        if (e - s) < need:
            continue
        seg = fhr[s:e]
        deficit = np.clip(baseline - seg, 0.0, None)
        nadir = s + int(np.argmin(seg))
        out.append(dict(start=int(s), end=int(e), nadir=nadir,
                        depth_bpm=float(baseline - seg.min()),
                        duration_s=float((e - s) / fs),
                        area_bpm_s=float(deficit.sum() / fs),
                        onset_s=float((nadir - s) / fs)))
    return out


def pair_to_contractions(decels: List[Dict], uc_peaks: np.ndarray,
                         fs: float) -> Dict[int, List[int]]:
    """
    Map contraction index -> indices of decelerations belonging to it.

    Same nearest-peak rule and same 2-minute tolerance as
    classify_decelerations, with ONE deliberate difference, documented
    because it changes the numbers:

    descriptors.py `continue`s out of the loop as soon as a deceleration is
    typed PROLONGED, so a prolonged deceleration is never added to
    `paired_contractions` and can never make an epoch "repetitive". That is a
    defect with respect to FIGO 2015, whose pathological criterion is
    "repetitive LATE OR PROLONGED decelerations" -- under the current code a
    prolonged deceleration is structurally excluded from the repetitiveness
    it is supposed to be able to satisfy. It is left unfixed in v3 (labels
    are frozen) and corrected here so the comparison is like-for-like on
    everything except what is being studied.
    """
    tol = int(DECEL_UC_PAIRING_TOLERANCE_S * fs)
    pairs: Dict[int, List[int]] = {}
    if uc_peaks.size == 0:
        return pairs
    for i, d in enumerate(decels):
        j = int(np.argmin(np.abs(uc_peaks - d["nadir"])))
        if abs(d["nadir"] - int(uc_peaks[j])) <= tol:
            pairs.setdefault(j, []).append(i)
    return pairs


def _slope(v: np.ndarray) -> float:
    if len(v) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(v), dtype=float), v, 1)[0])


def compute_burden(fhr: np.ndarray, uc: np.ndarray, fs: float,
                   baseline: float, valid: np.ndarray,
                   uc_valid: Optional[np.ndarray] = None) -> DecelBurden:
    """
    All burden candidates for one epoch.

    DEFINITIONS, in the order the study reports them.

    A. frac_contractions_with_decel
           |{contractions carrying >=1 deceleration}| / |contractions|
       The >50% rule's own ratio, kept continuous. Undefined with no
       contractions; reported as 0.0 with n_contractions = 0 so callers can
       exclude those epochs rather than silently treat them as benign.

    B. decel_area_per_min
           sum_e integral_e max(baseline - FHR, 0) dt  /  observation_minutes
       Total oxygen-deficit area in bpm*s per minute of usable signal.
       depth_x_duration_per_min is the cruder peak-depth surrogate,
           sum_e (depth_e * duration_e) / observation_minutes,
       included because it is the form most often written in the literature
       and the two can be compared.

    C. area_per_contraction
           sum_e area_e / |contractions|
       Normalises for contraction rate, so a trace with 6 contractions per
       10 min is comparable with one at 3.

    D. max_consecutive_uc_with_decel
           longest run of consecutive contraction indices each carrying >=1
           deceleration. This is what "repetitive" means physiologically --
           successive contractions each producing a response -- and unlike
           the >50% ratio it does not depend on the denominator.
       decel_interval_cv
           std/mean of gaps between consecutive deceleration nadirs;
           low CV = regular, contraction-locked; high CV = sporadic.
       decel_trend_per_min, severity_trend
           least-squares slopes of per-minute deceleration count and of
           per-deceleration depth, i.e. is the burden increasing.
    """
    obs_min = float(np.sum(valid) / fs / 60.0)
    decels = deceleration_events(fhr, baseline, fs, valid)
    uc_peaks = detect_contractions(uc, fs, uc_valid)
    pairs = pair_to_contractions(decels, uc_peaks, fs)

    n_uc = int(uc_peaks.size)
    n_hit = len(pairs)
    frac = (n_hit / n_uc) if n_uc > 0 else 0.0

    areas = np.array([d["area_bpm_s"] for d in decels]) if decels else np.zeros(0)
    depths = np.array([d["depth_bpm"] for d in decels]) if decels else np.zeros(0)
    durs = np.array([d["duration_s"] for d in decels]) if decels else np.zeros(0)

    total_area = float(areas.sum())
    per_min = (total_area / obs_min) if obs_min > 0 else 0.0
    dxd = float((depths * durs).sum() / obs_min) if obs_min > 0 else 0.0
    frac_time = float(durs.sum() * fs / max(np.sum(valid), 1))

    # D: longest run of consecutive contractions each carrying a deceleration
    run = best = 0
    for j in range(n_uc):
        run = run + 1 if j in pairs else 0
        best = max(best, run)

    if len(decels) >= 3:
        nad = np.array([d["nadir"] for d in decels], dtype=float) / fs
        gaps = np.diff(np.sort(nad))
        cv = float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else 0.0
    else:
        cv = 0.0

    # per-minute deceleration counts, for the trend
    n_min = max(int(round(obs_min)), 1)
    counts = np.zeros(n_min)
    for d in decels:
        b = min(int((d["nadir"] / fs) // 60), n_min - 1)
        counts[b] += 1

    return DecelBurden(
        frac_contractions_with_decel=float(frac),
        n_contractions=n_uc,
        n_contractions_with_decel=int(n_hit),
        decel_area_per_min=float(per_min),
        depth_x_duration_per_min=float(dxd),
        frac_time_in_decel=float(frac_time),
        area_per_contraction=float(total_area / n_uc) if n_uc > 0 else 0.0,
        mean_decel_depth=float(depths.mean()) if len(depths) else 0.0,
        mean_decel_duration_s=float(durs.mean()) if len(durs) else 0.0,
        max_consecutive_uc_with_decel=int(best),
        decel_interval_cv=cv,
        decel_trend_per_min=_slope(counts),
        severity_trend=_slope(depths) if len(depths) >= 2 else 0.0,
        observation_min=obs_min,
        n_decels=len(decels),
    )


def flip_sensitivity(n_uc: int, n_hit: int,
                     threshold: float = 0.50) -> Dict[str, object]:
    """
    Would the binary criterion change if ONE contraction's deceleration
    status were different?

    Returns the current verdict, the verdict with one deceleration added and
    with one removed, whether either flips it, and the absolute distance from
    the threshold in ratio units. This is the quantity that makes the rule
    brittle and it is computable in closed form.
    """
    if n_uc <= 0:
        return dict(rep=False, flips=False, margin=float("nan"),
                    margin_events=float("nan"))
    rep = (n_hit / n_uc) > threshold
    up = ((n_hit + 1) / n_uc) > threshold if n_hit < n_uc else rep
    dn = ((n_hit - 1) / n_uc) > threshold if n_hit > 0 else False
    # how many events away from the boundary
    crit = threshold * n_uc
    return dict(rep=bool(rep), flips=bool(up != rep or dn != rep),
                margin=float(abs(n_hit / n_uc - threshold)),
                margin_events=float(abs(n_hit - crit)))
