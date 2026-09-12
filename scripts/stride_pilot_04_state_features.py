"""
Stride experiment pilot -- Stage D.

Rebuilds the Phase 9B (multidomain deterioration/severity) and Phase 9C (state
trajectory dynamics -- the 40-D "P6" feature matrix) artifacts against the new
1.0-min-stride rolling predictions. Neither script has any stride-specific
logic -- both operate generically on whatever rows are in rolling_predictions.csv
-- so this just monkeypatches their path constants and calls their existing
extraction functions unmodified.
"""

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

ROLLING_PATH = os.path.join(BASE_DIR, "results", "phase8_rolling_stride1p0", "rolling_predictions.csv")
P9B_OUT_DIR = os.path.join(BASE_DIR, "results", "phase9b_deterioration_stride1p0")
P9C_OUT_DIR = os.path.join(BASE_DIR, "results", "phase9c_state_trajectory_stride1p0")
FOLDS_PATH = os.path.join(BASE_DIR, "data", "processed_clinical", "folds.json")  # canonical, unchanged

if __name__ == "__main__":
    print("=== STRIDE PILOT STAGE D: Phase 9B + 9C state/trajectory features @ 1.0-min stride ===")
    os.makedirs(P9B_OUT_DIR, exist_ok=True)
    os.makedirs(P9C_OUT_DIR, exist_ok=True)

    # Phase 9B only ever reads p9a_data["X_temporal"][:, :19] from its FEATURES_P9A
    # dependency -- i.e. just the raw 19 descriptors, row-aligned to
    # rolling_predictions.csv via window_index. Build that minimal shim here
    # instead of re-running the full delta/slope/novelty feature builder for
    # columns Phase 9B never reads.
    import numpy as np
    ROLLING_STRIDE_DIR = os.path.join(BASE_DIR, "results", "phase8_rolling_stride1p0")
    P9A_SHIM_DIR = os.path.join(BASE_DIR, "results", "phase9_temporal_stride1p0")
    os.makedirs(P9A_SHIM_DIR, exist_ok=True)
    fe19plus = np.load(os.path.join(ROLLING_STRIDE_DIR, "fe_windows_19plus.npy"))
    np.savez_compressed(os.path.join(P9A_SHIM_DIR, "temporal_features.npz"), X_temporal=fe19plus)
    print(f"Built minimal Phase 9A shim -> {P9A_SHIM_DIR}/temporal_features.npz  shape={fe19plus.shape}")

    import scripts.phase9b_deterioration_features as p9b
    p9b.OUT_DIR = P9B_OUT_DIR
    p9b.ROLLING_PATH = ROLLING_PATH
    p9b.FEATURES_P9A = os.path.join(P9A_SHIM_DIR, "temporal_features.npz")
    if hasattr(p9b, "FOLDS_PATH"):
        p9b.FOLDS_PATH = FOLDS_PATH
    print("\n--- Phase 9B: multidomain severity / deterioration state ---")
    p9b.extract_deterioration_features()

    import scripts.phase9c_state_trajectory as p9c
    p9c.OUT_DIR = P9C_OUT_DIR
    p9c.ROLLING_PATH = ROLLING_PATH
    p9c.FEATURES_P9B = os.path.join(P9B_OUT_DIR, "deterioration_features.npz")
    if hasattr(p9c, "FOLDS_PATH"):
        p9c.FOLDS_PATH = FOLDS_PATH
    print("\n--- Phase 9C: state trajectory dynamics (40-D P6 features) ---")
    p9c.extract_state_trajectories()

    print("\nStage D complete.")
