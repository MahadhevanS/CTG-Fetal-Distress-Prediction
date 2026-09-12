"""
Stride experiment pilot -- Stage C.

Applies the FROZEN 5-fold Continuous Clinical Huber models (zero retraining --
same coefficients as the 2.5-min baseline) to the new 1.0-min-stride windows,
producing a rolling_predictions.csv in the same schema as
results/phase8_rolling/rolling_predictions.csv.

Reuses run_rolling_inference() from scripts/phase8_rolling_inference.py
unmodified, monkeypatching its module-level path constants to point at the
Stage A/B outputs and a new results directory.
"""

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OUT_DIR = os.path.join(BASE_DIR, "results", "phase8_rolling_stride1p0")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_clinical_stride1p0")
P2_PATH = os.path.join(BASE_DIR, "data", "phase1_candidates_stride1p0", "p2_dataset.pt")

if __name__ == "__main__":
    print("=== STRIDE PILOT STAGE C: rolling Huber inference @ 1.0-min stride (frozen models) ===")
    os.makedirs(OUT_DIR, exist_ok=True)

    import scripts.phase8_rolling_inference as ri
    ri.OUT_DIR = OUT_DIR
    ri.DATA_DIR = DATA_DIR
    ri.P2_PATH = P2_PATH
    # FOLDS_PATH and MODEL_DIR (frozen Huber models) stay canonical/unchanged on purpose.

    ri.run_rolling_inference()

    # Side artifact: dump the 19-D clinical descriptor matrix Fe_windows is
    # aligned 1:1 with rolling_predictions.csv rows via `window_index`
    # (i == enumerate(meta_p2) index used both places) -- Stage D's minimal
    # temporal_features.npz shim reuses this instead of recomputing it.
    import json
    import numpy as np
    import torch
    with open(ri.FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    p2_data = torch.load(P2_PATH, weights_only=False)
    meta_p2 = [tuple(m) for m in p2_data["meta"]]
    c_meta, c_Fe = [], []
    for s in ("train", "val", "test"):
        d = torch.load(os.path.join(DATA_DIR, f"{s}_dataset.pt"), weights_only=False)
        ext = np.load(os.path.join(DATA_DIR, f"{s}_extended_features.npy"))
        fe_split = np.hstack([d["y_features"].numpy(), ext])
        for m, f_row in zip(d["metadata"], fe_split):
            c_meta.append((str(m[0]), int(m[1]), int(m[2])))
            c_Fe.append(f_row)
    feat_lookup = {k: v for k, v in zip(c_meta, c_Fe)}
    Fe_windows = np.array([feat_lookup[(str(m[0]), int(m[1]), int(m[2]))] for m in meta_p2], dtype=np.float32)
    np.save(os.path.join(OUT_DIR, "fe_windows_19plus.npy"), Fe_windows)
    print(f"Saved side artifact -> {OUT_DIR}/fe_windows_19plus.npy  shape={Fe_windows.shape}")

    print("Stage C complete.")
