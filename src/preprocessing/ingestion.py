import os
from math import gcd

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# GAP 4 FIX: Canonical sampling frequency constant
# ---------------------------------------------------------------------------

TARGET_FS: float = 4.0
"""
Standardised sampling frequency for all CTU-CHB signals (Hz).

All recordings in this pipeline must be at 4 Hz before feature extraction
and windowing. If a record is loaded at a different rate it is automatically
resampled using polyphase filtering (resample_poly) to preserve signal quality.

Reference: preprocessing_technical_documentation.md §2.1 & §3.1
"""


def load_ctu_chb_record(record_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Loads a CTU-CHB record using wfdb, validates the sampling frequency,
    and resamples to TARGET_FS (4 Hz) if necessary.

    CTU-CHB dataset signals:
        - Channel 0: Fetal Heart Rate (FHR) in bpm
        - Channel 1: Uterine Contraction (UC) in mmHg (arbitrary units)
        - Sampling Frequency: nominally 4 Hz

    GAP 4 FIX:
        Previously the pipeline assumed fs == 4 Hz and used it unchecked.
        Any recording at a different rate would silently produce incorrect
        window sizes and wrong feature durations.

        This function now:
        1. Reads the actual recorded fs from the WFDB header.
        2. If fs != TARGET_FS, resamples both FHR and UC channels using
           `scipy.signal.resample_poly` (polyphase, integer ratio) which
           minimises spectral distortion relative to simple decimation.
        3. If fs == TARGET_FS, asserts this explicitly so any future dataset
           change raises a clear error rather than silently misbehaving.

    Args:
        record_path: Path to the record without extension
                     (e.g., 'data/raw/ctu-chb-intrapartum/1001')

    Returns:
        fhr (np.ndarray): Fetal Heart Rate signal (bpm). Missing values == 0.0.
        uc (np.ndarray): Uterine Contraction signal. Missing values == 0.0.
        fs (float): Effective sampling frequency after resampling (always TARGET_FS
                    on success, or 0.0 on failure).
    """
    try:
        record = wfdb.rdrecord(record_path)

        # Signals are shaped (n_samples, n_channels)
        signals = record.p_signal
        fhr = signals[:, 0].copy()
        uc  = signals[:, 1].copy()
        fs  = float(record.fs)

        # Replace NaN with 0.0 (missing marker) — wfdb may return NaN for missing
        fhr = np.where(np.isnan(fhr), 0.0, fhr)
        uc  = np.where(np.isnan(uc),  0.0, uc)

        # ----------------------------------------------------------------
        # Sampling rate validation and resampling (GAP 4 core fix)
        # ----------------------------------------------------------------
        if fs != TARGET_FS:
            # Warn and resample — do not silently ignore the mismatch
            print(
                f"  [RESAMPLE] {os.path.basename(record_path)}: "
                f"fs={fs:.1f} Hz → {TARGET_FS:.1f} Hz"
            )

            # Use integer-ratio polyphase resampling for accuracy.
            # gcd reduces the ratio to lowest terms to minimise computation.
            fs_rounded  = int(round(fs))
            tgt_rounded = int(TARGET_FS)
            common      = gcd(tgt_rounded, fs_rounded)
            up          = tgt_rounded // common   # upsample factor
            down        = fs_rounded  // common   # downsample factor

            fhr = resample_poly(fhr, up, down).astype(np.float64)
            uc  = resample_poly(uc,  up, down).astype(np.float64)
            fs  = TARGET_FS

        else:
            # Strict assertion: if we expect 4 Hz, it must be exactly 4 Hz
            assert fs == TARGET_FS, (
                f"Unexpected sampling frequency fs={fs} Hz for record "
                f"'{record_path}'. Expected {TARGET_FS} Hz. "
                "This record cannot be processed safely."
            )

        return fhr, uc, fs

    except AssertionError:
        # Re-raise assertion errors so they are visible — do not swallow
        raise
    except Exception as e:
        print(f"  [ERROR] Failed to load record '{record_path}': {e}")
        return np.array([]), np.array([]), 0.0


def load_clinical_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Loads clinical metadata containing pH values, Apgar scores, etc.

    Supports CSV and Excel formats. Column names are normalised to lowercase
    with underscores to ensure consistent downstream access.

    Args:
        metadata_path: Path to the clinical metadata CSV or Excel file.

    Returns:
        pd.DataFrame: DataFrame with standardised column names.
    """
    if metadata_path.endswith('.csv'):
        df = pd.read_csv(metadata_path)
    elif metadata_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(metadata_path)
    else:
        raise ValueError(
            f"Unsupported metadata format: '{metadata_path}'. "
            "Use CSV (.csv) or Excel (.xls / .xlsx)."
        )

    # Standardise column names: strip whitespace, lowercase, spaces → underscores
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df
