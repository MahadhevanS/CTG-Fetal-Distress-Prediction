import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Preprocessing modules
from ingestion import load_ctu_chb_record, load_clinical_metadata, TARGET_FS
from signal_quality import get_valid_windows
from filtering import (
    remove_spikes,            # GAP 1 FIX
    interpolate_missing,
    apply_lowpass_filter,
    zscore_normalize_channels # GAP 5 FIX
)
from baseline import calculate_iterative_baseline
from features import calculate_variability, detect_accelerations, detect_decelerations
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knowledge.figo import classify_figo


# ---------------------------------------------------------------------------
# Clinical Constants
# ---------------------------------------------------------------------------

WINDOW_MINUTES        = 20   # FIGO minimum CTG assessment window
LAST_HOUR_MINUTES     = 60   # Clinically relevant horizon — last hour before delivery
PREDICTION_HORIZON_MINUTES = 30  # GAP 2 FIX: terminal pH label assigned only within
                                 # the final 30 minutes before delivery

WINDOW_SAMPLES        = int(WINDOW_MINUTES * 60 * TARGET_FS)          # 4800
LAST_HOUR_SAMPLES     = int(LAST_HOUR_MINUTES * 60 * TARGET_FS)       # 14400
PREDICTION_HORIZON_SAMPLES = int(PREDICTION_HORIZON_MINUTES * 60 * TARGET_FS)  # 7200

# Training strides — dynamic to address class imbalance (training set ONLY)
DISTRESS_STRIDE_MINUTES = 0.5  # 30-sec stride for minority oversampling (~1,200 distress windows)
NORMAL_STRIDE_MINUTES   = 10   # Sparse overlap → majority undersampling (~2,000 normal windows)

# Fixed evaluation stride (val + test must reflect real-world distribution)
EVAL_STRIDE_MINUTES = 10


def process_pipeline(raw_data_dir: str, metadata_path: str, output_dir: str) -> None:
    """
    Master orchestration for CTG (CTU-CHB) preprocessing.

    Processing steps per window:
        1. Spike removal (GAP 1 FIX) — eliminates unphysiological transducer artifacts
        2. Missing data interpolation — cubic spline for gaps ≤ 15 seconds
        3. Low-pass filtering — 4th-order Butterworth, 1.5 Hz cutoff (zero-phase)
        4. Baseline estimation — iterative mean with ±15 bpm FIGO exclusion
        5. Baseline correction — FHR channel centred to zero
        6. Feature extraction — STV, LTV, accelerations, decelerations
        7. FIGO classification — programmatic rule engine (pseudo-labels)
        8. Prediction horizon labelling (GAP 2 FIX) — y_primary=1 only in final
           30 min before delivery, earlier windows labelled 0 even for distress patients

    Post-split:
        9. Per-channel Z-score normalisation (GAP 5 FIX) — fit on training set,
           applied to val/test; scaler persisted as ctu_signal_scaler.npz

    Args:
        raw_data_dir (str): Path to the CTU-CHB raw data directory
                            (contains .dat/.hea files).
        metadata_path (str): Path to clinical_metadata.csv.
        output_dir (str): Directory to write train/val/test_dataset.pt files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load clinical metadata
    # -----------------------------------------------------------------------
    print("Loading metadata...")
    if not os.path.exists(metadata_path):
        print(f"  [ERROR] Metadata file not found: {metadata_path}")
        return

    metadata = load_clinical_metadata(metadata_path)
    metadata = (
        metadata.set_index('record_id')
        if 'record_id' in metadata.columns
        else metadata
    )

    # -----------------------------------------------------------------------
    # Patient-level stratified split
    # Split BEFORE window extraction to guarantee zero data leakage.
    # -----------------------------------------------------------------------
    records   = list(metadata.index)
    y_patient = (metadata['ph'] <= 7.15).astype(int).values

    train_ids, test_val_ids, y_train, y_test_val = train_test_split(
        records, y_patient,
        test_size=0.3, stratify=y_patient, random_state=42
    )
    val_ids, test_ids, _, _ = train_test_split(
        test_val_ids, y_test_val,
        test_size=0.5, stratify=y_test_val, random_state=42
    )

    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}

    # -----------------------------------------------------------------------
    # Window extraction loop
    # -----------------------------------------------------------------------
    for split_name, ids in splits.items():
        print(f"\nProcessing {split_name} split ({len(ids)} patients)...")
        X_data     = []
        Y_primary  = []
        Y_figo     = []
        Y_features = []
        Record_ids = []

        # Stride selection: dynamic for training, fixed for evaluation
        distress_stride = int(DISTRESS_STRIDE_MINUTES * 60 * TARGET_FS)
        normal_stride   = int(NORMAL_STRIDE_MINUTES   * 60 * TARGET_FS)
        eval_stride     = int(EVAL_STRIDE_MINUTES      * 60 * TARGET_FS)

        n_skipped_missing  = 0
        n_skipped_no_data  = 0
        n_label_horizon    = 0  # windows relabelled to 0 (outside 30-min horizon)

        for record_id in ids:
            record_path = os.path.join(raw_data_dir, str(record_id))

            # Check raw file existence before loading
            if not os.path.exists(record_path + '.dat'):
                n_skipped_missing += 1
                continue

            fhr, uc, fs = load_ctu_chb_record(record_path)
            if len(fhr) == 0:
                n_skipped_no_data += 1
                continue

            # Patient-level distress flag (pH threshold)
            is_distress = bool(metadata.loc[record_id, 'ph'] <= 7.15)

            # --- Clinically relevant horizon: last 60 min before delivery ---
            # The CTU-CHB recording ends at delivery, so the last sample = delivery.
            if len(fhr) > LAST_HOUR_SAMPLES:
                fhr = fhr[-LAST_HOUR_SAMPLES:]
                uc  = uc[-LAST_HOUR_SAMPLES:]

            # Current truncated signal length (may be < LAST_HOUR_SAMPLES for
            # short recordings — use whatever is available)
            signal_length = len(fhr)

            # --- GAP 2 FIX: Prediction horizon boundary ---
            # Only windows that START within the final PREDICTION_HORIZON_SAMPLES
            # of the (truncated) signal are assigned the terminal pH label.
            # Earlier windows — even from distress patients — are not yet in distress
            # and receive y_primary = 0.
            # Ref: detailed_preprocessing_plan.md §6.2, preprocessing_justification.md §2.7
            horizon_start_idx = max(0, signal_length - PREDICTION_HORIZON_SAMPLES)

            # --- Stride selection per patient ---
            if split_name == 'train':
                stride = distress_stride if is_distress else normal_stride
            else:
                stride = eval_stride

            valid_starts = get_valid_windows(fhr, WINDOW_SAMPLES, stride)

            for start in valid_starts:
                end = start + WINDOW_SAMPLES
                fhr_win = fhr[start:end].copy()
                uc_win  = uc[start:end].copy()

                # -----------------------------------------------------------
                # Signal Processing Chain (order is clinically significant)
                # -----------------------------------------------------------

                # Step 1 — GAP 1 FIX: Remove unphysiological spikes before
                # interpolation so spikes become missing data, not artefacts
                fhr_win = remove_spikes(fhr_win, fs=fs)

                # Step 2 — Interpolate short gaps (≤ 15 sec / 60 samples)
                fhr_win = interpolate_missing(fhr_win)

                # Step 3 — Smooth residual high-frequency noise
                fhr_win = apply_lowpass_filter(fhr_win, fs=fs)
                uc_win  = apply_lowpass_filter(uc_win,  fs=fs)

                # -----------------------------------------------------------
                # Baseline Estimation & Normalisation
                # -----------------------------------------------------------
                baseline  = calculate_iterative_baseline(fhr_win)
                fhr_norm  = fhr_win - baseline   # Channel 0: baseline-corrected FHR

                # -----------------------------------------------------------
                # Clinical Feature Extraction
                # -----------------------------------------------------------
                stv, ltv = calculate_variability(fhr_win, fs=fs)
                accels   = detect_accelerations(fhr_win, baseline, fs=fs)
                decels   = detect_decelerations(fhr_win, baseline, uc_win, fs=fs)

                # -----------------------------------------------------------
                # FIGO Pseudo-Label (auxiliary target — not ground truth)
                # -----------------------------------------------------------
                base_val   = float(np.mean(baseline))
                figo_class = classify_figo(base_val, ltv, accels, decels)

                # -----------------------------------------------------------
                # Input Tensor Construction
                # Stack channels: (2, WINDOW_SAMPLES)
                # Ch 0 = baseline-corrected FHR, Ch 1 = filtered UC
                # -----------------------------------------------------------
                window_tensor = np.vstack((fhr_norm, uc_win))

                features_target = [
                    base_val,
                    stv, ltv,
                    float(accels),
                    float(decels['early']),
                    float(decels['late']),
                    float(decels['variable']),
                    float(decels['prolonged']),
                ]

                # -----------------------------------------------------------
                # GAP 2 FIX: Prediction Horizon Label Assignment
                # -----------------------------------------------------------
                # Within horizon (last 30 min): assign the patient's pH outcome
                # Before horizon: label = 0 regardless of patient pH outcome
                within_horizon = (start >= horizon_start_idx)
                if is_distress and not within_horizon:
                    label = 0   # Too early — not yet in distress
                    n_label_horizon += 1
                else:
                    label = int(is_distress and within_horizon)

                X_data.append(window_tensor)
                Y_primary.append(label)
                Y_figo.append(int(figo_class))
                Y_features.append(features_target)
                Record_ids.append((str(record_id), int(start), int(end)))

        # -------------------------------------------------------------------
        # Save raw (pre-normalisation) split file
        # -------------------------------------------------------------------
        if len(X_data) > 0:
            ds = {
                'X':         torch.tensor(np.array(X_data),     dtype=torch.float32),
                'y_primary': torch.tensor(np.array(Y_primary),  dtype=torch.long),
                'y_figo':    torch.tensor(np.array(Y_figo),     dtype=torch.long),
                'y_features':torch.tensor(np.array(Y_features), dtype=torch.float32),
                'metadata':  Record_ids,
            }
            out_path = os.path.join(output_dir, f'{split_name}_dataset.pt')
            torch.save(ds, out_path)
            print(f"  Saved {split_name}_dataset.pt - {len(X_data)} windows")
            print(f"    Normal={int((ds['y_primary']==0).sum())}  "
                  f"Distress={int((ds['y_primary']==1).sum())}")
            if n_label_horizon > 0:
                print(f"    Horizon-relabelled windows (distress->0): {n_label_horizon}")
        else:
            print(f"  [WARNING] No windows extracted for {split_name} split.")

        if n_skipped_missing:
            print(f"  Skipped {n_skipped_missing} records (missing .dat file)")
        if n_skipped_no_data:
            print(f"  Skipped {n_skipped_no_data} records (empty signal)")

    # -----------------------------------------------------------------------
    # GAP 5 FIX: Per-Channel Z-Score Normalisation
    # Fit on training set ONLY, apply identically to val + test.
    # Scaler persisted to output_dir for inference-time use.
    # -----------------------------------------------------------------------
    print("\nApplying per-channel Z-score normalisation...")

    train_path = os.path.join(output_dir, 'train_dataset.pt')
    if not os.path.exists(train_path):
        print("  [ERROR] train_dataset.pt not found — cannot compute scaler.")
        return

    # Load training set to compute scaler
    train_ds   = torch.load(train_path, weights_only=False)
    X_train_np = train_ds['X'].numpy()  # (N, 2, 4800)

    # Fit scaler on training data
    _, train_mean, train_std = zscore_normalize_channels(X_train_np)

    # Persist scaler (MUST be loaded identically during inference)
    scaler_path = os.path.join(output_dir, 'ctu_signal_scaler.npz')
    np.savez(scaler_path, mean=train_mean, std=train_std)
    print(f"  Scaler saved -> {scaler_path}")
    print(f"    Channel means : FHR={train_mean[0]:.4f}  UC={train_mean[1]:.4f}")
    print(f"    Channel stds  : FHR={train_std[0]:.4f}   UC={train_std[1]:.4f}")

    # Apply to all three splits
    for split_name in ['train', 'val', 'test']:
        pt_path = os.path.join(output_dir, f'{split_name}_dataset.pt')
        if not os.path.exists(pt_path):
            print(f"  [SKIP] {split_name}_dataset.pt not found")
            continue

        ds   = torch.load(pt_path, weights_only=False)
        X_np = ds['X'].numpy()
        X_norm, _, _ = zscore_normalize_channels(X_np, mean=train_mean, std=train_std)
        ds['X'] = torch.tensor(X_norm, dtype=torch.float32)
        torch.save(ds, pt_path)
        print(f"  Normalised and saved {split_name}_dataset.pt")

    print("\nCTU-CHB preprocessing pipeline complete.")


if __name__ == "__main__":
    BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    RAW_DIR        = os.path.join(BASE_DIR, 'data', 'raw', 'ctu-chb-intrapartum')
    METADATA_FILE  = os.path.join(RAW_DIR, 'clinical_metadata.csv')
    OUT_DIR        = os.path.join(BASE_DIR, 'data', 'processed')
    process_pipeline(RAW_DIR, METADATA_FILE, OUT_DIR)
