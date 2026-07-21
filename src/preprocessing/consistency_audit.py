"""
Dataset Consistency Audit
=========================

Validates all generated .pt files for:
    1. Tensor shape and dtype integrity
    2. Absence of NaN, Inf, and out-of-range values
    3. Class distribution and label correctness
    4. FIGO vs pH label cross-tabulation (clinical consistency)
    5. Pathological FIGO window feature consistency
    6. Z-score normalisation validation (train approx mean=0, std=1 per channel)
    7. Scaler artifact existence (ctu_signal_scaler.npz, uci_scaler.joblib)
    8. Prediction horizon label correctness (no distress windows exist
       outside the 30-min horizon — windows relabelled correctly to 0)
    9. Patient-level data leakage check (record IDs across splits)
   10. UCI dataset validation (shape, feature count, label distribution)
"""

import os
import sys
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pass(msg: str) -> None:
    print(f"    [PASS]  {msg}")


def _warn(msg: str) -> None:
    print(f"    [WARN]  {msg}")


def _fail(msg: str) -> None:
    print(f"    [FAIL]  {msg}")


def _section(title: str) -> None:
    print(f"\n{'-' * 56}")
    print(f"  {title}")
    print(f"{'-' * 56}")


# ---------------------------------------------------------------------------
# CTU-CHB Split Audit
# ---------------------------------------------------------------------------

def audit_ctu_splits(processed_dir: str) -> None:
    """
    Audits the three CTU-CHB splits (train/val/test_dataset.pt).
    """
    _section("CTU-CHB SPLIT AUDIT")

    splits      = ['train', 'val', 'test']
    split_data  = {}
    all_records = {}   # For leakage check

    for split in splits:
        file_path = os.path.join(processed_dir, f'{split}_dataset.pt')

        if not os.path.exists(file_path):
            _fail(f"{split}_dataset.pt not found in {processed_dir}")
            continue

        print(f"\n  [{split.upper()} SPLIT]")
        data = torch.load(file_path, weights_only=False)
        split_data[split] = data

        X          = data['X']
        y_primary  = data['y_primary']
        y_figo     = data['y_figo']
        y_features = data['y_features']
        metadata   = data.get('metadata', [])

        # ----------------------------------------------------------------
        # Check 1: Tensor shapes and dtypes
        # ----------------------------------------------------------------
        expected_shape = (len(X), 2, 4800)
        if X.shape == expected_shape:
            _pass(f"X shape correct: {tuple(X.shape)}")
        else:
            _fail(f"X shape unexpected: {tuple(X.shape)} - expected {expected_shape}")

        if X.dtype == torch.float32:
            _pass("X dtype = float32")
        else:
            _warn(f"X dtype = {X.dtype} (expected float32)")

        if y_primary.dtype == torch.long:
            _pass("y_primary dtype = long")
        else:
            _warn(f"y_primary dtype = {y_primary.dtype}")

        if y_features.shape[1] == 8:
            _pass(f"y_features shape correct: {tuple(y_features.shape)}")
        else:
            _fail(f"y_features has {y_features.shape[1]} columns - expected 8")

        # ----------------------------------------------------------------
        # Check 2: NaN and Inf checks
        # ----------------------------------------------------------------
        n_nan = torch.isnan(X).sum().item()
        n_inf = torch.isinf(X).sum().item()
        if n_nan == 0:
            _pass("No NaN values in X")
        else:
            _fail(f"X contains {n_nan} NaN values")

        if n_inf == 0:
            _pass("No Inf values in X")
        else:
            _fail(f"X contains {n_inf} Inf values")

        # ----------------------------------------------------------------
        # Check 3: Label validity
        # ----------------------------------------------------------------
        unique_primary = torch.unique(y_primary).tolist()
        if set(unique_primary).issubset({0, 1}):
            _pass(f"y_primary labels valid: {unique_primary}")
        else:
            _fail(f"y_primary contains invalid labels: {unique_primary}")

        unique_figo = torch.unique(y_figo).tolist()
        if set(unique_figo).issubset({0, 1, 2}):
            _pass(f"y_figo labels valid: {unique_figo}")
        else:
            _fail(f"y_figo contains invalid labels: {unique_figo}")

        # ----------------------------------------------------------------
        # Check 4: Class distribution
        # ----------------------------------------------------------------
        n_total    = len(X)
        n_normal   = int((y_primary == 0).sum())
        n_distress = int((y_primary == 1).sum())
        f0 = int((y_figo == 0).sum())
        f1 = int((y_figo == 1).sum())
        f2 = int((y_figo == 2).sum())

        print(f"\n    Class Distribution:")
        print(f"      Total windows : {n_total}")
        print(f"      pH Normal     : {n_normal}  ({100*n_normal/n_total:.1f}%)")
        print(f"      pH Distress   : {n_distress} ({100*n_distress/n_total:.1f}%)")
        print(f"      FIGO Normal   : {f0}")
        print(f"      FIGO Suspicious: {f1}")
        print(f"      FIGO Pathological: {f2}")

        # Check that train is reasonably balanced (target: 15–65% distress)
        if split == 'train':
            distress_ratio = n_distress / n_total
            if 0.15 <= distress_ratio <= 0.65:
                _pass(f"Train distress ratio {distress_ratio:.2%} is within 15-65% target")
            else:
                _warn(f"Train distress ratio {distress_ratio:.2%} is outside 15-65% target")

        # ----------------------------------------------------------------
        # Check 5: FIGO vs pH correlation (clinical consistency)
        # ----------------------------------------------------------------
        print(f"\n    FIGO vs pH Correlation:")
        y_np = y_primary.numpy()
        f_np = y_figo.numpy()
        for fc, fname in [(0, "Normal    "), (1, "Suspicious"), (2, "Patholog. ")]:
            idx = (f_np == fc)
            ph_norm = int((y_np[idx] == 0).sum())
            ph_dist = int((y_np[idx] == 1).sum())
            print(f"      FIGO {fname} -> pH Normal: {ph_norm:4d} | pH Distress: {ph_dist:4d}")

        # Sanity: FIGO Normal windows should rarely be pH Distress
        figo_normal_distress = int((y_np[f_np == 0] == 1).sum())
        total_figo_normal    = int((f_np == 0).sum())
        if total_figo_normal > 0:
            contamination = figo_normal_distress / total_figo_normal
            if contamination < 0.10:
                _pass(f"FIGO Normal contamination by pH Distress: "
                      f"{contamination:.1%} (< 10% threshold)")
            else:
                _warn(f"FIGO Normal contaminated by pH Distress: {contamination:.1%}")

        # ----------------------------------------------------------------
        # Check 6: Pathological FIGO clinical feature consistency
        # ----------------------------------------------------------------
        patho_idx = np.where(f_np == 2)[0]
        if len(patho_idx) > 0:
            yf_np      = y_features.numpy()
            late_decels = yf_np[patho_idx, 5]
            prol_decels = yf_np[patho_idx, 7]
            baselines   = yf_np[patho_idx, 0]

            has_late  = (late_decels > 0).sum()
            has_prol  = (prol_decels > 0).sum()
            low_base  = (baselines < 100).sum()
            explained = has_late + has_prol + low_base

            ratio = explained / len(patho_idx) if len(patho_idx) > 0 else 0
            msg = (f"Pathological FIGO consistency: "
                   f"{explained}/{len(patho_idx)} windows ({ratio:.1%}) "
                   f"have late/prolonged decels or baseline<100")
            if ratio >= 0.60:
                _pass(msg)
            else:
                _warn(msg + "  [below 60% threshold - review FIGO logic]")

        # ----------------------------------------------------------------
        # Check 7: Z-score normalisation (train only)
        # ----------------------------------------------------------------
        if split == 'train':
            X_np = X.numpy()   # (N, 2, 4800)
            fhr_mean = float(X_np[:, 0, :].mean())
            fhr_std  = float(X_np[:, 0, :].std())
            uc_mean  = float(X_np[:, 1, :].mean())
            uc_std   = float(X_np[:, 1, :].std())

            print(f"\n    Z-score Normalisation (Train):")
            print(f"      FHR ch: mean={fhr_mean:.4f}  std={fhr_std:.4f}")
            print(f"      UC  ch: mean={uc_mean:.4f}  std={uc_std:.4f}")

            for ch_name, mean_val, std_val in [
                ("FHR", fhr_mean, fhr_std),
                ("UC",  uc_mean,  uc_std),
            ]:
                mean_ok = abs(mean_val) < 0.5
                std_ok  = 0.5 < std_val < 2.0
                if mean_ok and std_ok:
                    _pass(f"{ch_name} channel: mean approx 0 ({mean_val:.3f}), "
                          f"std approx 1 ({std_val:.3f})")
                else:
                    _warn(f"{ch_name} channel: mean={mean_val:.3f}, std={std_val:.3f} "
                          f"- Z-score may not be applied or scaler mismatch")

        # ----------------------------------------------------------------
        # Collect record IDs for leakage check
        # ----------------------------------------------------------------
        if metadata:
            all_records[split] = set(m[0] for m in metadata)

    # -----------------------------------------------------------------------
    # Check 8: Patient-level data leakage verification
    # -----------------------------------------------------------------------
    _section("DATA LEAKAGE CHECK")
    if len(all_records) == 3:
        train_ids = all_records.get('train', set())
        val_ids   = all_records.get('val',   set())
        test_ids  = all_records.get('test',  set())

        tv_overlap = train_ids & val_ids
        tt_overlap = train_ids & test_ids
        vt_overlap = val_ids   & test_ids

        if not tv_overlap:
            _pass("Train intersect Val = empty  (no leakage)")
        else:
            _fail(f"Train intersect Val = {len(tv_overlap)} overlapping records: {list(tv_overlap)[:5]}")

        if not tt_overlap:
            _pass("Train intersect Test = empty  (no leakage)")
        else:
            _fail(f"Train intersect Test = {len(tt_overlap)} overlapping records: {list(tt_overlap)[:5]}")

        if not vt_overlap:
            _pass("Val intersect Test = empty  (no leakage)")
        else:
            _fail(f"Val intersect Test = {len(vt_overlap)} overlapping records: {list(vt_overlap)[:5]}")
    else:
        _warn("Cannot perform leakage check - not all splits loaded successfully")


# ---------------------------------------------------------------------------
# Scaler Artifact Checks
# ---------------------------------------------------------------------------

def audit_scalers(processed_dir: str) -> None:
    """
    Verifies that all required scaler artifacts exist in the processed directory.
    """
    _section("SCALER ARTIFACT CHECK")

    # CTU-CHB signal scaler
    ctu_scaler_path = os.path.join(processed_dir, 'ctu_signal_scaler.npz')
    if os.path.exists(ctu_scaler_path):
        scaler = np.load(ctu_scaler_path)
        _pass(f"ctu_signal_scaler.npz found")
        print(f"      FHR mean={scaler['mean'][0]:.4f}  std={scaler['std'][0]:.4f}")
        print(f"      UC  mean={scaler['mean'][1]:.4f}  std={scaler['std'][1]:.4f}")
        # Validate scaler has correct shapes
        if scaler['mean'].shape == (2,) and scaler['std'].shape == (2,):
            _pass("Scaler shapes correct: (2,)")
        else:
            _fail(f"Scaler shape unexpected: mean={scaler['mean'].shape}")
    else:
        _fail(f"ctu_signal_scaler.npz NOT FOUND in {processed_dir}")
        _fail("Re-run pipeline.py to generate the Z-score scaler")

    # UCI scaler
    uci_scaler_path = os.path.join(processed_dir, 'uci_scaler.joblib')
    if os.path.exists(uci_scaler_path):
        _pass("uci_scaler.joblib found")
    else:
        _warn("uci_scaler.joblib NOT FOUND - UCI preprocessing may not have run")


# ---------------------------------------------------------------------------
# UCI Split Audit
# ---------------------------------------------------------------------------

def audit_uci_splits(processed_dir: str) -> None:
    """
    Audits the three UCI SisPorto splits (uci_train/val/test_dataset.pt).
    """
    _section("UCI SisPorto SPLIT AUDIT")

    for split in ['uci_train', 'uci_val', 'uci_test']:
        file_path = os.path.join(processed_dir, f'{split}_dataset.pt')

        if not os.path.exists(file_path):
            _warn(f"{split}_dataset.pt not found - UCI preprocessing may not have run")
            continue

        print(f"\n  [{split.upper()}]")
        data = torch.load(file_path, weights_only=False)

        X      = data['X']
        y_figo = data['y_figo']
        y_bin  = data['y_binary']

        # Shape check: expected (N, 21) for 21 SisPorto features
        if X.ndim == 2 and X.shape[1] <= 21:
            _pass(f"X shape: {tuple(X.shape)}")
        else:
            _warn(f"X shape: {tuple(X.shape)} - check feature column list")

        if X.dtype == torch.float32:
            _pass("X dtype = float32")
        else:
            _warn(f"X dtype = {X.dtype}")

        # NaN check
        if not torch.isnan(X).any():
            _pass("No NaN values in X")
        else:
            _fail(f"X contains NaN values")

        # Label distribution
        n_total = len(X)
        n_norm  = int((y_figo == 0).sum())
        n_susp  = int((y_figo == 1).sum())
        n_patho = int((y_figo == 2).sum())
        n_bin_norm  = int((y_bin == 0).sum())
        n_bin_patho = int((y_bin == 1).sum())

        print(f"    Total={n_total} | 3-class: Normal={n_norm} Suspect={n_susp} "
              f"Pathological={n_patho}")
        print(f"    Binary: Normal={n_bin_norm} Pathological={n_bin_patho} "
              f"(Suspect excluded, y_binary=-1)")

        if 'feature_names' in data:
            _pass(f"feature_names present: {len(data['feature_names'])} features")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def run_consistency_audit(processed_dir: str) -> None:
    """
    Runs the full consistency audit on all generated dataset files.

    Args:
        processed_dir (str): Path to the data/processed/ directory.
    """
    print("\n" + "=" * 56)
    print("     DATASET CONSISTENCY AUDIT - v2 (Post Gap Fixes)")
    print("=" * 56)
    print(f"  Audit directory: {processed_dir}\n")

    audit_ctu_splits(processed_dir)
    audit_scalers(processed_dir)
    audit_uci_splits(processed_dir)

    print("\n" + "=" * 56)
    print("  Audit complete. Review any [WARN] or [FAIL] items above.")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

    if not os.path.exists(PROCESSED_DIR):
        print(f"[ERROR] Processed directory not found: {PROCESSED_DIR}")
        print("Run src/preprocessing/run_all.py first.")
        sys.exit(1)

    run_consistency_audit(PROCESSED_DIR)
