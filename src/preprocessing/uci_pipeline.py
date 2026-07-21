"""
UCI SisPorto Cardiotocography Dataset Preprocessor
===================================================

Processes the tabular UCI CTG dataset (2,126 records, 21 SisPorto features) for:
    1. Classical ML baseline experiments (Random Forest, XGBoost, SVM)
    2. Validating our automated feature extraction against SisPorto's computed features
    3. External validation benchmarks (cross-dataset generalisation assessment)

Project Role (workflow_and_plan.md §2, §3):
    - NOT used for primary deep learning training on time-series.
    - Does NOT contribute windows to train/val/test_dataset.pt.
    - The UCI dataset uses pre-extracted features from the SisPorto 2.0 system.

Label Mapping (from dataset documentation):
    - NSP == 1 → Normal   → y_binary = 0, y_figo_3class = 0
    - NSP == 2 → Suspect  → y_binary = excluded from binary task (ambiguous)
    - NSP == 3 → Pathological → y_binary = 1, y_figo_3class = 2

Outputs (saved to data/processed/):
    - uci_train_dataset.pt
    - uci_val_dataset.pt
    - uci_test_dataset.pt
    - uci_scaler.joblib  (StandardScaler fit on training data ONLY)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# SisPorto Feature Column Definitions
# ---------------------------------------------------------------------------

SISPORTO_FEATURE_COLS: List[str] = [
    'LB',       # Baseline FHR (bpm) — computed by SisPorto
    'AC',       # Accelerations (#/sec)
    'FM',       # Fetal movements (#/sec)
    'UC',       # Uterine contractions (#/sec)
    'DL',       # Light decelerations (#/sec)
    'DS',       # Severe decelerations (#/sec)
    'DP',       # Prolonged decelerations (#/sec)
    'ASTV',     # % time with abnormal short-term variability
    'MSTV',     # Mean value of short-term variability
    'ALTV',     # % time with abnormal long-term variability
    'MLTV',     # Mean value of long-term variability
    'Width',    # Width of FHR histogram
    'Min',      # Minimum of FHR histogram
    'Max',      # Maximum of FHR histogram
    'Nmax',     # # of histogram peaks
    'Nzeros',   # # of histogram zeros
    'Mode',     # Histogram mode
    'Mean',     # Histogram mean
    'Median',   # Histogram median
    'Variance', # Histogram variance
    'Tendency', # Histogram tendency (-1, 0, 1)
]

TARGET_NSP    = 'NSP'   # 3-class clinical label: 1=Normal, 2=Suspect, 3=Pathologic
TARGET_CLASS  = 'CLASS' # 10-class morphological label (optional, may not be present)


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_uci_dataset(uci_dir: str) -> pd.DataFrame:
    """
    Loads and parses the UCI SisPorto CTG dataset.

    The UCI CTG dataset is distributed as a multi-sheet Excel file (CTG.xls).
    The 'Raw Data' sheet contains the actual records. The first two rows of
    the header are merged in the original Excel file, so we skip row 0 and
    treat row 1 as the header.

    Also handles CSV variants if present.

    Args:
        uci_dir (str): Path to the extracted UCI dataset directory.

    Returns:
        pd.DataFrame: Raw dataset with original column names.

    Raises:
        FileNotFoundError: If no recognised dataset file is found.
    """
    candidates = [
        ('CTG.xls',   'excel', 'Raw Data'),
        ('CTG.xlsx',  'excel', 'Raw Data'),
        ('ctg.xls',   'excel', 'Raw Data'),
        ('ctg.xlsx',  'excel', 'Raw Data'),
        ('cardiotocography.csv', 'csv',   None),
        ('CTG.csv',   'csv',   None),
    ]

    for fname, ftype, sheet in candidates:
        fpath = os.path.join(uci_dir, fname)
        if not os.path.exists(fpath):
            continue

        print(f"  Loading UCI dataset from: {fpath}")
        if ftype == 'excel':
            # Try header=0 first (standard header), fallback to header=1 if needed
            df = pd.read_excel(fpath, sheet_name=sheet, header=0)
            df.columns = [str(c).strip() for c in df.columns]
            if TARGET_NSP not in df.columns:
                df = pd.read_excel(fpath, sheet_name=sheet, header=1)
                df.columns = [str(c).strip() for c in df.columns]
        else:
            df = pd.read_csv(fpath)
            df.columns = [str(c).strip() for c in df.columns]

        return df

    raise FileNotFoundError(
        f"UCI CTG dataset not found in: {uci_dir}\n"
        "Expected one of: CTG.xls, CTG.xlsx, CTG.csv, cardiotocography.csv\n"
        "Please ensure the dataset zip was extracted correctly."
    )


# ---------------------------------------------------------------------------
# Main Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess_uci(uci_dir: str, output_dir: str) -> None:
    """
    Full UCI SisPorto tabular preprocessing pipeline.

    Steps:
        1. Load raw data and validate required columns.
        2. Drop rows with missing NSP (target label) — records without a
           clinical classification are unusable.
        3. Median imputation for remaining feature NaNs (applied globally
           before splitting for simplicity; the fitted scaler handles
           distribution alignment).
        4. Map NSP → 0-indexed 3-class label and binary distress label.
        5. Stratified 70/15/15 split on the 3-class target.
        6. Fit Z-score StandardScaler on TRAINING features ONLY.
        7. Transform val/test features using the training scaler (no leakage).
        8. Save processed tensors and the scaler artifact.

    Args:
        uci_dir (str): Path to the extracted UCI dataset directory.
        output_dir (str): Directory to save processed outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load and validate
    # -----------------------------------------------------------------------
    print("\nLoading UCI SisPorto dataset...")
    try:
        df = load_uci_dataset(uci_dir)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        print("  Skipping UCI preprocessing.")
        return

    # Check for required feature columns — warn about any that are missing
    missing_feat_cols = [c for c in SISPORTO_FEATURE_COLS if c not in df.columns]
    if missing_feat_cols:
        print(f"  [WARNING] Missing SisPorto feature columns: {missing_feat_cols}")
        # Keep only the columns that exist
        feature_cols = [c for c in SISPORTO_FEATURE_COLS if c in df.columns]
        print(f"  Proceeding with {len(feature_cols)} available feature columns.")
    else:
        feature_cols = SISPORTO_FEATURE_COLS

    if TARGET_NSP not in df.columns:
        print(f"  [ERROR] Target column '{TARGET_NSP}' not found in dataset. "
              f"Available columns: {list(df.columns)}")
        print("  Skipping UCI preprocessing.")
        return

    print(f"  Total records loaded: {len(df)}")
    print(f"  Feature columns used: {len(feature_cols)}")

    # -----------------------------------------------------------------------
    # Step 2: Drop rows with missing target (NSP)
    # -----------------------------------------------------------------------
    df = df.dropna(subset=[TARGET_NSP])

    # Convert NSP to integer; some Excel files read it as float
    try:
        df[TARGET_NSP] = df[TARGET_NSP].astype(int)
    except Exception:
        # Filter to rows where NSP is numeric
        df = df[pd.to_numeric(df[TARGET_NSP], errors='coerce').notna()].copy()
        df[TARGET_NSP] = df[TARGET_NSP].astype(int)

    # Keep only valid NSP values (1, 2, 3)
    df = df[df[TARGET_NSP].isin([1, 2, 3])].copy()
    print(f"  Records after NSP validation: {len(df)}")
    print(f"  NSP distribution: {df[TARGET_NSP].value_counts().to_dict()}")

    # -----------------------------------------------------------------------
    # Step 3: Median imputation for feature NaNs
    # (Using global median; scaler's per-feature normalization handles the rest)
    # -----------------------------------------------------------------------
    feature_medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(feature_medians)

    # -----------------------------------------------------------------------
    # Step 4: Label mapping
    # -----------------------------------------------------------------------
    # 3-class label: NSP 1→0 (Normal), NSP 2→1 (Suspect), NSP 3→2 (Pathological)
    df['y_figo_3class'] = df[TARGET_NSP] - 1  # 0-indexed

    # Binary distress label: Pathological (NSP=3) → 1, Normal (NSP=1) → 0
    # Suspect (NSP=2) is EXCLUDED from binary classification — clinically ambiguous.
    # We train binary classifiers only on clearly Normal vs. clearly Pathological.
    df_binary = df[df[TARGET_NSP].isin([1, 3])].copy()
    df_binary['y_binary'] = (df_binary[TARGET_NSP] == 3).astype(int)

    print(f"\n  3-class records: {len(df)}  "
          f"(Normal={int((df['y_figo_3class']==0).sum())}, "
          f"Suspect={int((df['y_figo_3class']==1).sum())}, "
          f"Pathological={int((df['y_figo_3class']==2).sum())})")
    print(f"  Binary records (excl. Suspect): {len(df_binary)}  "
          f"(Normal={int((df_binary['y_binary']==0).sum())}, "
          f"Pathological={int((df_binary['y_binary']==1).sum())})")

    # -----------------------------------------------------------------------
    # Step 5: Stratified 70/15/15 split on 3-class target
    # -----------------------------------------------------------------------
    indices_3  = df.index.tolist()
    y_strat_3  = df['y_figo_3class'].values.astype(int)

    train_idx_3, test_val_idx_3 = train_test_split(
        indices_3, test_size=0.3, stratify=y_strat_3, random_state=42
    )
    # Build stratify array for the second split from the subset
    y_tv_3 = df.loc[test_val_idx_3, 'y_figo_3class'].values.astype(int)
    val_idx_3, test_idx_3 = train_test_split(
        test_val_idx_3, test_size=0.5, stratify=y_tv_3, random_state=42
    )

    def _extract(idx, df_src, feature_cols):
        X = df_src.loc[idx, feature_cols].values.astype(np.float32)
        y3 = df_src.loc[idx, 'y_figo_3class'].values.astype(np.int64)
        # Binary label — default to -1 for Suspect rows (excluded from binary head)
        yb = np.where(
            df_src.loc[idx, TARGET_NSP].values == 2,
            -1,
            (df_src.loc[idx, TARGET_NSP].values == 3).astype(np.int64)
        )
        return X, y3, yb

    X_train, y_train_3, y_train_b = _extract(train_idx_3, df, feature_cols)
    X_val,   y_val_3,   y_val_b   = _extract(val_idx_3,   df, feature_cols)
    X_test,  y_test_3,  y_test_b  = _extract(test_idx_3,  df, feature_cols)

    # -----------------------------------------------------------------------
    # Step 6: Fit Z-score scaler on TRAINING data ONLY
    # -----------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Persist scaler — must be loaded identically during inference
    scaler_path = os.path.join(output_dir, 'uci_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"\n  UCI feature scaler saved -> {scaler_path}")

    # -----------------------------------------------------------------------
    # Step 7: Save processed splits
    # -----------------------------------------------------------------------
    splits = {
        'uci_train': (X_train, y_train_3, y_train_b, list(train_idx_3)),
        'uci_val':   (X_val,   y_val_3,   y_val_b,   list(val_idx_3)),
        'uci_test':  (X_test,  y_test_3,  y_test_b,  list(test_idx_3)),
    }

    print()
    for split_name, (X, y3, yb, idx_list) in splits.items():
        out_dict = {
            'X':            torch.tensor(X,  dtype=torch.float32),
            'y_figo':       torch.tensor(y3, dtype=torch.long),
            'y_binary':     torch.tensor(yb, dtype=torch.long),
            'record_indices': idx_list,
            'feature_names':  feature_cols,
        }
        save_path = os.path.join(output_dir, f'{split_name}_dataset.pt')
        torch.save(out_dict, save_path)

        n_norm  = int((y3 == 0).sum())
        n_susp  = int((y3 == 1).sum())
        n_patho = int((y3 == 2).sum())
        print(f"  Saved {split_name}_dataset.pt - {len(X)} samples | "
              f"Normal={n_norm}  Suspect={n_susp}  Pathological={n_patho}")

    print("\nUCI SisPorto preprocessing complete.")


# ---------------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    UCI_DIR  = os.path.join(BASE_DIR, 'data', 'raw', 'cardiotocography')
    OUT_DIR  = os.path.join(BASE_DIR, 'data', 'processed')

    if not os.path.exists(UCI_DIR):
        print(f"[ERROR] UCI dataset directory not found: {UCI_DIR}")
        sys.exit(1)

    preprocess_uci(UCI_DIR, OUT_DIR)
