"""
Repair CTU-CHB Signal Corruption (Knowledge-Infusion Audit, 2026-08-07)
========================================================================
Root cause: interpolate_missing() filled short gaps using
scipy.interpolate.CubicSpline(..., extrapolate=True) with no physiological
bounds check. Near sparse/boundary gaps this can overshoot to thousands of
bpm, which then propagates into:
    - The channel-0 (FHR) entry of ctu_signal_scaler.npz (std inflated
      ~453 instead of a physiologically sane ~15-25), crushing the
      normalized dynamic range for the ~99% of clean windows.
    - The Baseline / STV / LTV / accel / decel clinical feature targets
      (y_features) and the FIGO pseudo-labels (y_figo) derived from them.

filtering.py / pipeline.py have already been patched so any *future* run of
process_pipeline() (which requires the raw CTU-CHB .dat/.hea files — not
present in this checkout, data/raw/ only has .gitkeep) will not reproduce
this. This script repairs the *already-generated* train/val/test_dataset.pt
+ ctu_signal_scaler.npz + feature_scaler.npz in data/processed/ without
needing the raw files, by exploiting two facts:

  1. Z-score normalization is exactly invertible: raw = X_norm*std + mean,
     regardless of whether the std/mean used were themselves good estimates.
  2. y_features[:, 0] (Baseline FHR) lets us reconstruct the exact
     post-filter, pre-baseline-correction fhr_win = raw_channel0 + baseline
     for every stored window — the same array baseline.py/features.py
     originally operated on, corruption and all.

For every window this script then:
  - Reconstructs fhr_win (raw_ch0 + baseline) and uc_win (raw_ch1).
  - Clamps fhr_win to [FHR_MIN_BPM, FHR_MAX_BPM] — the same physiological
    bound now applied inside interpolate_missing().
  - Re-derives baseline, STV, LTV, accel/decel counts, and the FIGO class
    from the clamped signal using the *actual production functions*
    (src/preprocessing/baseline.py, features.py, src/knowledge/figo.py) —
    not an approximation of them.
  - Rebuilds channel 0 as (clamped fhr_win - new baseline).
  - Refits the Z-score scaler on the corrected TRAIN split only, and
    applies it identically to val/test (same leakage-safe design as
    pipeline.py's own GAP-5 fix).

Usage:
    python scripts/repair_ctu_signal_corruption.py            # dry run (report only)
    python scripts/repair_ctu_signal_corruption.py --apply     # write corrected files
                                                                 # (originals backed up first)
"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing.baseline import calculate_iterative_baseline
from src.preprocessing.features import (
    calculate_variability,
    detect_accelerations,
    detect_decelerations,
)
from src.knowledge.figo import classify_figo
from src.preprocessing.filtering import zscore_normalize_channels

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FHR_MIN_BPM = 50.0
FHR_MAX_BPM = 240.0
FS = 4.0

SPLITS = ["train", "val", "test"]


def load_all():
    ds = {}
    for split in SPLITS:
        path = os.path.join(DATA_DIR, f"{split}_dataset.pt")
        if os.path.exists(path):
            ds[split] = torch.load(path, weights_only=False)
    return ds


def reprocess_window(fhr_win: np.ndarray, uc_win: np.ndarray):
    """Re-derives baseline/features/FIGO for one window from a clamped fhr_win,
    using the exact same functions the original pipeline calls."""
    fhr_clamped = np.clip(fhr_win, FHR_MIN_BPM, FHR_MAX_BPM)

    baseline = calculate_iterative_baseline(fhr_clamped)
    fhr_norm = fhr_clamped - baseline

    stv, ltv = calculate_variability(fhr_clamped, fs=FS)
    accels = detect_accelerations(fhr_clamped, baseline, fs=FS)
    decels = detect_decelerations(fhr_clamped, baseline, uc_win, fs=FS)

    base_val = float(np.mean(baseline))
    figo_class = classify_figo(base_val, ltv, accels, decels)

    features_target = [
        base_val, stv, ltv, float(accels),
        float(decels["early"]), float(decels["late"]),
        float(decels["variable"]), float(decels["prolonged"]),
    ]
    return fhr_norm, features_target, figo_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Write corrected files (default: dry-run report only)")
    args = parser.parse_args()

    scaler_path = os.path.join(DATA_DIR, "ctu_signal_scaler.npz")
    if not os.path.exists(scaler_path):
        print(f"[ERROR] {scaler_path} not found — cannot invert normalization.")
        return
    old_scaler = np.load(scaler_path)
    old_mean, old_std = old_scaler["mean"], old_scaler["std"]
    print(f"Existing scaler: FHR mean={old_mean[0]:.4f} std={old_std[0]:.4f} | "
          f"UC mean={old_mean[1]:.4f} std={old_std[1]:.4f}")

    datasets = load_all()
    if "train" not in datasets:
        print("[ERROR] train_dataset.pt not found.")
        return

    corrected = {}
    n_clamped_windows_total = 0
    n_clamped_samples_total = 0
    n_figo_changed_total = 0
    n_windows_total = 0

    for split, ds in datasets.items():
        X = ds["X"].numpy()  # (N, 2, 4800), currently Z-normalized
        y_features_old = ds["y_features"].numpy()
        y_figo_old = ds["y_figo"].numpy()
        N = X.shape[0]
        n_windows_total += N

        # Exact inverse of the Z-score transform applied by the original pipeline
        raw_ch0 = X[:, 0, :] * old_std[0] + old_mean[0]   # = fhr_win - baseline (corrupted)
        raw_ch1 = X[:, 1, :] * old_std[1] + old_mean[1]   # = uc_win

        baseline_old = y_features_old[:, 0:1]  # (N, 1), broadcast back to (N, 4800)
        fhr_win_recon = raw_ch0 + baseline_old  # exact reconstruction of pre-clamp fhr_win

        new_ch0 = np.zeros_like(raw_ch0)
        new_features = np.zeros_like(y_features_old)
        new_figo = np.zeros_like(y_figo_old)

        split_clamped_windows = 0
        split_clamped_samples = 0
        split_figo_changed = 0

        for i in range(N):
            fhr_win = fhr_win_recon[i]
            uc_win = raw_ch1[i]

            out_of_range = (fhr_win < FHR_MIN_BPM) | (fhr_win > FHR_MAX_BPM)
            n_bad = int(out_of_range.sum())
            if n_bad > 0:
                split_clamped_windows += 1
                split_clamped_samples += n_bad

            fhr_norm, feat, figo_cls = reprocess_window(fhr_win, uc_win)
            new_ch0[i] = fhr_norm
            new_features[i] = feat
            new_figo[i] = figo_cls
            if figo_cls != int(y_figo_old[i]):
                split_figo_changed += 1

        n_clamped_windows_total += split_clamped_windows
        n_clamped_samples_total += split_clamped_samples
        n_figo_changed_total += split_figo_changed

        print(f"\n[{split}] {N} windows | "
              f"{split_clamped_windows} windows had out-of-range samples "
              f"({split_clamped_samples} samples total) | "
              f"{split_figo_changed} FIGO labels changed after repair")

        ltv_old, ltv_new = y_features_old[:, 2], new_features[:, 2]
        print(f"  LTV  before: mean={ltv_old.mean():8.2f} std={ltv_old.std():8.2f} max={ltv_old.max():10.2f}")
        print(f"  LTV  after : mean={ltv_new.mean():8.2f} std={ltv_new.std():8.2f} max={ltv_new.max():10.2f}")
        stv_old, stv_new = y_features_old[:, 1], new_features[:, 1]
        print(f"  STV  before: mean={stv_old.mean():8.2f} std={stv_old.std():8.2f} max={stv_old.max():10.2f}")
        print(f"  STV  after : mean={stv_new.mean():8.2f} std={stv_new.std():8.2f} max={stv_new.max():10.2f}")

        corrected[split] = {
            "new_ch0": new_ch0,
            "raw_ch1": raw_ch1,
            "new_features": new_features,
            "new_figo": new_figo,
        }

    # Refit signal scaler on corrected TRAIN channel-0/1 only (leakage-safe, matches pipeline.py)
    train_raw_stack = np.stack(
        [corrected["train"]["new_ch0"], corrected["train"]["raw_ch1"]], axis=1
    )  # (N, 2, 4800)
    _, new_mean, new_std = zscore_normalize_channels(train_raw_stack)
    print(f"\nCorrected scaler: FHR mean={new_mean[0]:.4f} std={new_std[0]:.4f} "
          f"(was std={old_std[0]:.4f}) | UC mean={new_mean[1]:.4f} std={new_std[1]:.4f}")

    # Refit feature scaler on corrected TRAIN y_features
    feat_means = corrected["train"]["new_features"].mean(axis=0).astype(np.float32)
    feat_stds = np.clip(corrected["train"]["new_features"].std(axis=0), 1e-6, None).astype(np.float32)
    names = ["Baseline", "STV", "LTV", "Accels", "Early", "Late", "Variable", "Prolonged"]
    print("\nCorrected feature_scaler.npz:")
    for n, m, s in zip(names, feat_means, feat_stds):
        print(f"  {n:<10} mean={m:9.4f} std={s:9.4f}")

    print(f"\n=== SUMMARY ===")
    print(f"Total windows processed: {n_windows_total}")
    print(f"Windows with out-of-range samples: {n_clamped_windows_total} "
          f"({100*n_clamped_windows_total/n_windows_total:.2f}%)")
    print(f"Total out-of-range samples clamped: {n_clamped_samples_total}")
    print(f"FIGO pseudo-labels that changed after repair: {n_figo_changed_total} "
          f"({100*n_figo_changed_total/n_windows_total:.2f}%)")

    if not args.apply:
        print("\n[DRY RUN] No files written. Re-run with --apply to write corrected "
              "datasets (originals will be backed up to data/processed/pre_repair_backup/).")
        return

    backup_dir = os.path.join(DATA_DIR, "pre_repair_backup")
    os.makedirs(backup_dir, exist_ok=True)
    for fname in ["train_dataset.pt", "val_dataset.pt", "test_dataset.pt", "ctu_signal_scaler.npz"]:
        src = os.path.join(DATA_DIR, fname)
        if os.path.exists(src):
            dst = os.path.join(backup_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"[Backup] {fname} -> pre_repair_backup/")

    for split, ds in datasets.items():
        c = corrected[split]
        X_raw = np.stack([c["new_ch0"], c["raw_ch1"]], axis=1)
        X_norm, _, _ = zscore_normalize_channels(X_raw, mean=new_mean, std=new_std)
        ds["X"] = torch.tensor(X_norm, dtype=torch.float32)
        ds["y_features"] = torch.tensor(c["new_features"], dtype=torch.float32)
        ds["y_figo"] = torch.tensor(c["new_figo"], dtype=torch.long)
        out_path = os.path.join(DATA_DIR, f"{split}_dataset.pt")
        torch.save(ds, out_path)
        print(f"[Saved] {split}_dataset.pt (corrected)")

    np.savez(scaler_path, mean=new_mean, std=new_std)
    print(f"[Saved] ctu_signal_scaler.npz (corrected)")

    feature_scaler_path = os.path.join(DATA_DIR, "feature_scaler.npz")
    np.savez(feature_scaler_path, feature_means=feat_means, feature_stds=feat_stds)
    print(f"[Saved] feature_scaler.npz (corrected)")


if __name__ == "__main__":
    main()
