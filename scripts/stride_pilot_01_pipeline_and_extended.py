"""
Stride experiment pilot -- Stage A.

Regenerates the clinical-pipeline artifacts (windows + 19 clinical descriptors)
at an alternate window-extraction stride, without touching the canonical
2.5-minute data at data/processed_clinical/. Everything else (window length,
truncation, quality gates, split logic, random_state) is held fixed so stride
is the only thing that changes.

Reuses process_pipeline_clinical() from src/preprocessing/pipeline_clinical.py
unmodified -- overrides its module-level STRIDE_MINUTES/STRIDE_SAMPLES via
monkeypatch (only affects this process) rather than editing the frozen script.

Also reconstructs the 11 "extended" clinical descriptors (decel depth/area/
burden/longest, baseline/variability slope, UC count/tachysystole/amplitude,
FHR-UC lag/coupling) via src/knowledge/extended_features.py's
extract_extended_features(), which is not otherwise wired to a standalone
generator script in this repo (the one that produced the committed
data/processed_clinical/*_extended_features.npy was never committed).
"""

import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# pipeline_clinical.py imports "ingestion" etc. as bare (same-directory) modules,
# so it expects to be run with src/preprocessing/ itself on sys.path.
_PREPROC_DIR = os.path.join(BASE_DIR, "src", "preprocessing")
if _PREPROC_DIR not in sys.path:
    sys.path.insert(0, _PREPROC_DIR)

STRIDE_MINUTES_NEW = 1.0
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed_clinical_stride1p0")


def run_pipeline_at_new_stride():
    import src.preprocessing.pipeline_clinical as pc

    fs = pc.TARGET_FS if hasattr(pc, "TARGET_FS") else 4.0
    print(f"Overriding stride: {pc.STRIDE_MINUTES} min -> {STRIDE_MINUTES_NEW} min "
          f"(window={pc.WINDOW_MINUTES}min, truncation={pc.LAST_HOUR_MINUTES}min unchanged)")

    pc.STRIDE_MINUTES = STRIDE_MINUTES_NEW
    pc.STRIDE_SAMPLES = int(STRIDE_MINUTES_NEW * 60 * fs)

    pc.process_pipeline_clinical(
        raw_data_dir=os.path.join(BASE_DIR, "data", "raw", "ctu-chb-intrapartum"),
        metadata_path=os.path.join(BASE_DIR, "data", "raw", "ctu-chb-intrapartum", "clinical_metadata.csv"),
        output_dir=OUTPUT_DIR,
    )


def run_time_confound_audit():
    """Reuses the project's own mandatory gate: AUROC(window position -> y_primary) ~ 0.50."""
    audit_path = os.path.join(BASE_DIR, "scripts", "audit_time_confound.py")
    if not os.path.exists(audit_path):
        print("  [skip] scripts/audit_time_confound.py not found -- skipping gate")
        return
    import subprocess
    result = subprocess.run(
        [sys.executable, audit_path, "--data_dir", OUTPUT_DIR],
        cwd=BASE_DIR, capture_output=True, text=True
    )
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])


def build_extended_features():
    from src.knowledge.extended_features import extract_extended_features

    scaler = np.load(os.path.join(OUTPUT_DIR, "ctu_signal_scaler.npz"))
    mean = scaler["mean"]
    std = scaler["std"]

    for split in ("train", "val", "test"):
        path = os.path.join(OUTPUT_DIR, f"{split}_dataset.pt")
        if not os.path.exists(path):
            print(f"  [skip] no {split}_dataset.pt")
            continue
        d = torch.load(path, weights_only=False)
        X = d["X"].numpy()          # (N, 3, T) z-scored channels 0-1, raw mask channel 2
        y_features = d["y_features"].numpy()  # (N, 8): [baseline, stv, ltv, accels, early, late, var, prolonged]
        baseline_per_window = y_features[:, 0]

        n = X.shape[0]
        ext = np.zeros((n, 11), dtype=np.float32)
        for i in range(n):
            fhr_bc_raw = X[i, 0, :] * std[0] + mean[0]   # undo z-score -> physical bpm units
            uc_raw = X[i, 1, :] * std[1] + mean[1]
            ext[i] = extract_extended_features(fhr_bc_raw, uc_raw, float(baseline_per_window[i]), fs=4.0)

        out_path = os.path.join(OUTPUT_DIR, f"{split}_extended_features.npy")
        np.save(out_path, ext)
        print(f"  Saved {out_path}  shape={ext.shape}")


if __name__ == "__main__":
    print("=== STRIDE PILOT STAGE A: pipeline_clinical @ 1.0-min stride + extended features ===")
    run_pipeline_at_new_stride()
    print("\n--- Time-confound gate ---")
    run_time_confound_audit()
    print("\n--- Building extended clinical descriptors ---")
    build_extended_features()
    print("\nStage A complete.")
