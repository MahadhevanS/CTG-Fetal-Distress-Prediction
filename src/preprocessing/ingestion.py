import os
import wfdb
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional

def load_ctu_chb_record(record_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Loads a CTU-CHB record using wfdb.
    
    Args:
        record_path: Path to the record without extension (e.g., 'data/raw/ctu-chb-intrapartum/1001')
        
    Returns:
        fhr (np.ndarray): Fetal Heart Rate signal (bpm). Missing values are typically 0.
        uc (np.ndarray): Uterine Contraction signal.
        fs (float): Sampling frequency of the record (usually 4Hz for CTU-CHB).
    """
    try:
        # Load the record and header
        record = wfdb.rdrecord(record_path)
        
        # Signals are typically shaped (n_samples, n_channels)
        signals = record.p_signal
        
        # CTU-CHB typically has FHR on channel 0 and UC on channel 1
        fhr = signals[:, 0]
        uc = signals[:, 1]
        fs = record.fs
        
        return fhr, uc, fs
    except Exception as e:
        print(f"Error loading record {record_path}: {e}")
        return np.array([]), np.array([]), 0.0

def load_clinical_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Loads clinical metadata containing pH values, Apgar scores, etc.
    
    Args:
        metadata_path: Path to the clinical metadata CSV or Excel file.
        
    Returns:
        pd.DataFrame: DataFrame indexed by record ID with clinical features.
    """
    if metadata_path.endswith('.csv'):
        df = pd.read_csv(metadata_path)
    elif metadata_path.endswith('.xls') or metadata_path.endswith('.xlsx'):
        df = pd.read_excel(metadata_path)
    else:
        raise ValueError("Unsupported metadata format. Use CSV or Excel.")
        
    # Standardize column names if necessary
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df
