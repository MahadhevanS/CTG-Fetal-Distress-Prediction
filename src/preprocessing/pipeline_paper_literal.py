"""
Literal reconstruction of Dang et al. 2026 §3.1 preprocessing.

WHY A SECOND PAPER-MATCH PIPELINE
---------------------------------
`pipeline_paper_match.py` (2026-08-17) was the first attempt. It carries two
deviations from the paper's stated text that were inherited from `pipeline.py`
rather than chosen:

  1. `WINDOW_MAX_MISSING_RATIO = 0.3` -- a per-window quality gate. The paper
     specifies a per-RECORDING gate only; no per-window gate is mentioned.
  2. the label uses `ph <= 7.15`; the paper states `pH < 7.15`.

This module implements the paper's text literally and makes every remaining
interpretive choice explicit as a constructor argument, so the reproduction
ladder can move one thing at a time.

PAPER TEXT IMPLEMENTED (E3S Web Conf. 723, 01005 (2026), §3.1)
--------------------------------------------------------------
  "recordings with pH >= 7.15 are classified as Normal, while those with
   pH < 7.15 are classified as Acidosis"
  "After quality filtering (recordings with >50% missing FHR values are
   excluded), 404 patients remain in the analysis cohort."
  "Missing FHR samples (signal loss periods) are handled via linear
   interpolation for gaps not exceeding 15 seconds; longer gaps are preserved
   as-is to avoid introducing artificial patterns."
  "Both FHR and UC signals undergo per-recording z-score normalization to
   standardize the dynamic range across recordings."
  "Continuous recordings are segmented into 20-minute sliding windows (4,800
   samples at 4 Hz) with a 5-minute stride, yielding 1,753 analysis windows:
   1,462 Normal (83.4%) and 291 Acidosis (16.6%)."

NOT MENTIONED by the paper, and therefore NOT done here: spike removal,
low-pass filtering, baseline subtraction, truncation to a final hour, and any
per-window quality gate.

KNOWN IRRECONCILABLE FACTS (documented, not tuned)
--------------------------------------------------
  * The stated >50% missing-FHR rule removes 5 of 552 records and leaves 547.
    It cannot leave 404. Six candidate definitions of "missing" were searched
    (see scripts/reproduce_dang2026.py); none yields 404. The cohort is
    therefore an unresolved degree of freedom.
  * At a 5-minute stride over untruncated recordings, 404 patients would yield
    roughly 4,500 windows, not 1,753. 1,753/404 = 4.34 windows per patient
    implies a ~37-minute segment per recording, which the paper does not
    describe.
"""
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .filtering import interpolate_missing_linear
from .ingestion import TARGET_FS, load_ctu_chb_record

WINDOW_SAMPLES = int(20 * 60 * TARGET_FS)      # 4800
STRIDE_SAMPLES = int(5 * 60 * TARGET_FS)       # 1200
PH_THRESHOLD = 7.15                            # paper: pH < 7.15 => Acidosis
PATIENT_MAX_MISSING = 0.50
FHR_MIN_BPM, FHR_MAX_BPM = 50.0, 240.0


def zscore_per_recording(x: np.ndarray, valid_only: bool) -> np.ndarray:
    """Per-recording z-score.

    `valid_only` is an interpretive choice the paper does not settle. Computing
    statistics over ALL samples lets preserved zero-gaps drag the mean down;
    computing them over valid samples only leaves those gaps as large negative
    outliers after scaling. Literal reading = all samples.
    """
    ref = x[x > 0] if (valid_only and (x > 0).any()) else x
    mu, sd = float(np.mean(ref)), float(np.std(ref))
    return (x - mu) / (sd if sd > 1e-6 else 1.0)


def build(raw_dir: str, metadata_path: str, out_path: str,
          window_max_missing: Optional[float] = None,
          zscore_valid_only: bool = False,
          normalisation: str = "per_recording",
          fill_all_gaps: bool = False,
          truncate_minutes: Optional[int] = None,
          patient_max_missing: float = PATIENT_MAX_MISSING,
          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the paper-literal window set.

    Args:
        window_max_missing: None = no per-window gate (literal). A float adds
            one, which is what pipeline_paper_match.py did at 0.3.
        zscore_valid_only: see zscore_per_recording.
        truncate_minutes: None = no truncation (literal).
        patient_max_missing: the stated >50% recording-level gate.

    Returns (X, y, patient_ids).
    """
    meta = pd.read_csv(metadata_path)
    meta = meta.set_index("record_id") if "record_id" in meta.columns else meta

    X, Y, PID = [], [], []
    n_excluded = n_missing_file = 0
    for rec in meta.index:
        path = os.path.join(raw_dir, str(rec))
        if not os.path.exists(path + ".dat"):
            n_missing_file += 1
            continue
        fhr, uc, fs = load_ctu_chb_record(path)
        if len(fhr) == 0:
            n_missing_file += 1
            continue

        # recording-level quality gate, on the raw untruncated signal
        if float((fhr <= 0).mean()) > patient_max_missing:
            n_excluded += 1
            continue

        if truncate_minutes is not None:
            keep = int(truncate_minutes * 60 * fs)
            if len(fhr) > keep:
                fhr, uc = fhr[-keep:], uc[-keep:]

        ph = meta.loc[rec, "ph"]
        label = int(float(ph) < PH_THRESHOLD)          # paper: strictly less than

        # linear interpolation of gaps <= 15 s, applied to the whole recording;
        # longer gaps preserved as zeros
        gap = len(fhr) if fill_all_gaps else 60     # 60 samples = 15 s
        f = interpolate_missing_linear(fhr.copy(), max_gap_samples=gap,
                                       clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
        u = interpolate_missing_linear(uc.copy(), max_gap_samples=gap)
        if fill_all_gaps:
            # any still-zero sample (leading/trailing, or no valid anchor)
            for arr in (f, u):
                v = arr > 0
                if v.any() and not v.all():
                    idx = np.arange(len(arr))
                    arr[~v] = np.interp(idx[~v], idx[v], arr[v])
        if normalisation == "per_recording":       # the paper's stated choice
            f = zscore_per_recording(f, zscore_valid_only)
            u = zscore_per_recording(u, zscore_valid_only)
        elif normalisation == "none":
            pass                                    # scaled globally by caller

        raw_for_gate = fhr
        for start in range(0, len(f) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            end = start + WINDOW_SAMPLES
            if window_max_missing is not None:
                if float((raw_for_gate[start:end] <= 0).mean()) > window_max_missing:
                    continue
            X.append(np.vstack([f[start:end], u[start:end]]).astype(np.float32))
            Y.append(label)
            PID.append(str(rec))

    X = np.asarray(X, dtype=np.float32)
    if normalisation == "global":
        # One scaler for the whole cohort: preserves BETWEEN-recording
        # differences in level and variability amplitude, which per-recording
        # z-scoring removes. Not the paper's choice; used to isolate its cost.
        mu = X.mean(axis=(0, 2), keepdims=True)
        sd = np.maximum(X.std(axis=(0, 2), keepdims=True), 1e-6)
        X = ((X - mu) / sd).astype(np.float32)
    Y = np.asarray(Y, dtype=np.int64)
    PID = np.asarray(PID)
    if verbose:
        npat = len(set(PID.tolist()))
        print(f"[paper-literal] {npat} patients, {len(X)} windows, "
              f"{Y.sum()} acidosis ({100*Y.mean():.1f}%) | "
              f"{len(X)/max(npat,1):.2f} windows/patient")
        print(f"               excluded by >{patient_max_missing:.0%} rule: {n_excluded}"
              f" | missing files: {n_missing_file}")
        print(f"               paper states: 404 patients, 1753 windows, 16.6% acidosis, "
              f"4.34 windows/patient")
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        torch.save({"X": torch.from_numpy(X), "y": torch.from_numpy(Y),
                    "pid": PID}, out_path)
    return X, Y, PID
