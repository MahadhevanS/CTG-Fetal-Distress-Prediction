"""
Non-overlapping epoch segmentation over WHOLE CTU-UHB recordings.

TWO DELIBERATE DEPARTURES FROM src/preprocessing/pipeline_clinical.py
----------------------------------------------------------------------

1. NON-OVERLAPPING epochs, not a 2.5-minute stride.

   The existing substrate uses 20-minute windows at a 2.5-minute stride, so
   adjacent windows share 87.5% of their signal. docs/auroc_ceiling_analysis.md
   showed what that does to a temporal label: for every acidotic patient the
   last negative and first positive window were 2.5 minutes apart, shared
   87.5% of their samples, and carried opposite labels. No model can separate
   such a pair, and cross-validation cannot separate them either -- two
   near-identical windows in train and test is leakage in all but name.

   The early-warning task is a temporal label, so it would inherit exactly
   that pathology. Non-overlapping epochs remove it by construction. This is
   the user's own Phase 2E requirement and it is applied to BOTH tasks, not
   only the prediction one, so that the state-detection and early-warning
   substrates are the same objects.

2. WHOLE recordings, not the last 60 minutes.

   pipeline_clinical.py truncates to the final hour. Measured over all 552
   records, duration is median 71.7 min (p25 70.0, p95 90.0, min 60.0), so
   truncation discards a median 11.7 minutes -- and it discards them from the
   START, which is where a deterioration trajectory begins. With a 30-minute
   horizon and 10-minute epochs, an anchor epoch needs 3 epochs of future
   inside the recording; on 60 minutes that leaves 3 anchors per patient, on
   the full recording it leaves a median of 4. Truncation costs roughly a
   quarter of the early-warning dataset for no benefit.

EPOCH LENGTH = 10 MINUTES
-------------------------
FIGO 2015 defines baseline as "estimated over 10 minutes" and variability is
read over the same span, so 10 minutes is the guideline's own unit of
assessment rather than a tuning choice. It also doubles the temporal
resolution available to the persistence rules in rules.py (a 30-minute
qualifier is 3 epochs, not 1.5) and doubles the number of prediction anchors
relative to 20-minute epochs. EPOCH_MINUTES is a module constant and the
audit reports the label distribution at 10 and 20 minutes so the choice is
checkable rather than asserted.

THE SIGNAL CHAIN IS UNCHANGED
-----------------------------
Spike removal -> cubic interpolation -> low-pass, exactly as in
pipeline_clinical.py, reusing the same functions. Descriptors are computed on
the cleaned signal in BPM (thresholds are clinical). The tensors saved for
the neural models are z-scored separately, fit on training patients only.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_PRE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing")
if _PRE not in sys.path:
    sys.path.insert(0, os.path.abspath(_PRE))

from filtering import (  # noqa: E402
    apply_lowpass_filter,
    interpolate_missing,
    remove_spikes,
)
from ingestion import TARGET_FS, load_ctu_chb_record  # noqa: E402

from .descriptors import EpochDescriptors, compute_descriptors  # noqa: E402
from .rules import StateDecision, classify_sequence  # noqa: E402

EPOCH_MINUTES = 10.0
EPOCH_SAMPLES = int(EPOCH_MINUTES * 60 * TARGET_FS)   # 2400 at 4 Hz

FHR_MIN_BPM = 50.0
FHR_MAX_BPM = 240.0

# A patient whose whole trace is mostly missing is dropped outright, matching
# pipeline_clinical.py. Individual epochs are NOT dropped -- they are marked
# UNREADABLE by rules.py and kept, so that a gap breaks a persistence run
# instead of being silently bridged.
PATIENT_MAX_MISSING_RATIO = 0.50

# A zero-phase Butterworth low-pass spreads a step over roughly its impulse
# response, so samples this close to a dropout edge are contaminated by it and
# are excluded from analysis along with the gap itself.
LOWPASS_SETTLE_S = 2.0


def _dilate(mask, k: int):
    """Widen a boolean mask by k samples on each side."""
    if k <= 0 or not mask.any():
        return mask
    import numpy as _np
    return _np.convolve(mask.astype(_np.int16), _np.ones(2 * k + 1, dtype=_np.int16),
                        mode="same") > 0


def segment_recording(fhr_raw: np.ndarray, uc_raw: np.ndarray, fs: float,
                      epoch_samples: int = EPOCH_SAMPLES
                      ) -> Tuple[List[Dict], np.ndarray]:
    """
    Cut one recording into consecutive non-overlapping epochs.

    Returns (epochs, clean_fhr_full) where each epoch dict carries its index,
    sample bounds, the cleaned FHR/UC slices in bpm, the pre-interpolation
    missingness mask, and the measured-sample fraction. A trailing remainder
    shorter than one epoch is discarded rather than zero-padded: a partial
    epoch would give the descriptors less signal than their thresholds assume
    and would silently read as low-variability.
    """
    n_epochs = len(fhr_raw) // epoch_samples
    epochs: List[Dict] = []

    fhr_clean_full = np.empty(0, dtype=np.float32)
    parts = []
    for i in range(n_epochs):
        s, e = i * epoch_samples, (i + 1) * epoch_samples
        raw_f = fhr_raw[s:e].astype(np.float64).copy()
        raw_u = uc_raw[s:e].astype(np.float64).copy()

        # missingness BEFORE any reconstruction (CTU-UHB encodes gaps as 0.0)
        mask = (raw_f == 0.0).astype(np.float32)
        quality = float(1.0 - mask.mean())

        f = remove_spikes(raw_f.copy(), fs=fs)
        f = interpolate_missing(f, clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)

        # WHAT SURVIVES INTERPOLATION IS NOT WHAT WAS MEASURED.
        # interpolate_missing() repairs gaps up to 15 s and deliberately
        # leaves longer ones at 0.0 rather than fabricate a deceleration
        # (see its docstring). Those 0.0 samples sit ~140 bpm below any real
        # baseline, so a rule engine reads each one as a deep, long
        # deceleration -- measured before this mask existed: 4.14 decels per
        # epoch at quality 0.50-0.70 versus 1.01 at quality 0.95-1.00, and a
        # mean "deepest deceleration" of 150 bpm. The validity mask is taken
        # HERE, after repair and before filtering, so it marks exactly the
        # samples that are still unusable.
        invalid = (f == 0.0)
        u_invalid = (raw_u == 0.0)

        f = apply_lowpass_filter(f, fs=fs)
        u = apply_lowpass_filter(raw_u.copy(), fs=fs)

        # The low-pass filter smears a gap into its neighbours, so widen the
        # invalid region by the filter's settling time before trusting the
        # samples next to one.
        invalid = _dilate(invalid, int(LOWPASS_SETTLE_S * fs))
        u_invalid = _dilate(u_invalid, int(LOWPASS_SETTLE_S * fs))

        parts.append(f.astype(np.float32))
        epochs.append(dict(index=i, start=s, end=e, fhr=f, uc=u,
                           mask=mask, quality=quality,
                           valid=~invalid, uc_valid=~u_invalid))

    if parts:
        fhr_clean_full = np.concatenate(parts)
    return epochs, fhr_clean_full


def describe_recording(fhr_raw: np.ndarray, uc_raw: np.ndarray, fs: float,
                       epoch_samples: int = EPOCH_SAMPLES,
                       epoch_minutes: float = EPOCH_MINUTES
                       ) -> Tuple[List[Dict], List[EpochDescriptors],
                                  List[StateDecision]]:
    """
    Segment, describe and classify one recording end to end.

    The baseline of each epoch is seeded from the previous epoch's baseline
    (see descriptors.estimate_baseline) which is why descriptors must be
    computed in chronological order and cannot be vectorised across epochs.
    That carry is causal -- previous epoch only.
    """
    epochs, _ = segment_recording(fhr_raw, uc_raw, fs, epoch_samples)

    descs: List[EpochDescriptors] = []
    prev_baseline: Optional[float] = None
    for ep in epochs:
        d = compute_descriptors(ep["fhr"], ep["uc"], fs, ep["quality"],
                                valid=ep["valid"], uc_valid=ep["uc_valid"],
                                prev_baseline=prev_baseline)
        descs.append(d)
        # Only carry a baseline that was itself readable, otherwise a bad
        # epoch propagates its level forward through the whole recording.
        if d.usable >= 0.5 and np.isfinite(d.baseline_bpm):
            prev_baseline = d.baseline_bpm

    decisions = classify_sequence(descs, epoch_minutes=epoch_minutes)
    return epochs, descs, decisions


def load_and_describe(record_path: str,
                      epoch_samples: int = EPOCH_SAMPLES,
                      epoch_minutes: float = EPOCH_MINUTES):
    """Convenience wrapper: path without extension -> (epochs, descs, decisions)."""
    fhr, uc, fs = load_ctu_chb_record(record_path)
    if len(fhr) == 0:
        return [], [], []
    if float(np.mean(fhr == 0.0)) > PATIENT_MAX_MISSING_RATIO:
        return [], [], []
    return describe_recording(fhr, uc, fs, epoch_samples, epoch_minutes)
