"""
Phase 8: Rolling 20-Minute Inference Pipeline.

Applies the frozen 5-Fold Continuous Clinical Huber model (Model B) across all 547 patients'
sequential 20-minute windows (8,517 windows), calculating delivery-anchored timestamps
and generating the master rolling predictions database.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.protocol import Protocol

OUT_DIR = "results/phase8_rolling"
FOLDS_PATH = "data/processed_clinical/folds.json"
P2_PATH = "data/phase1_candidates/p2_dataset.pt"
DATA_DIR = "data/processed_clinical"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
MODEL_DIR = "models/continuous_clinical_huber"

os.makedirs(OUT_DIR, exist_ok=True)

def run_rolling_inference():
    print("=== EXECUTING PHASE 8: ROLLING 20-MINUTE INFERENCE PIPELINE ===")

    # 1. Load Folds & Master Cohort
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p2_data = torch.load(P2_PATH, weights_only=False)
    y_bin = p2_data["y"].numpy()
    pid_arr = p2_data["pid"]
    meta_p2 = [tuple(m) for m in p2_data["meta"]] # (str(pid), int(start), int(end))

    prot = Protocol.load_or_create(pid_arr, y_bin, path=FOLDS_PATH)

    df_meta = pd.read_csv(METADATA_PATH)
    if 'record_id' in df_meta.columns:
        df_meta = df_meta.set_index('record_id')

    pid_to_ph = {str(p): float(df_meta.loc[int(p), 'ph']) for p in clean_pids}
    patient_labels_715 = {str(p): int(pid_to_ph[str(p)] <= 7.15) for p in clean_pids}
    patient_labels_705 = {str(p): int(pid_to_ph[str(p)] <= 7.05) for p in clean_pids}

    # 2. Load 19 Clinical Features for all 8,517 windows
    c_meta = []
    c_Fe = []
    for s in ("train", "val", "test"):
        d = torch.load(os.path.join(DATA_DIR, f"{s}_dataset.pt"), weights_only=False)
        ext = np.load(os.path.join(DATA_DIR, f"{s}_extended_features.npy"))
        fe_split = np.hstack([d["y_features"].numpy(), ext])
        for m, f_row in zip(d["metadata"], fe_split):
            c_meta.append((str(m[0]), int(m[1]), int(m[2])))
            c_Fe.append(f_row)

    feat_lookup = {k: v for k, v in zip(c_meta, c_Fe)}
    Fe_windows = np.array([feat_lookup[(str(m[0]), int(m[1]), int(m[2]))] for m in meta_p2], dtype=np.float32)
    print(f"Loaded {len(Fe_windows)} window feature vectors (19 dimensions each).")

    # 3. Determine max recording length per patient to anchor time to delivery
    patient_max_end = {}
    for (pid, start, end) in meta_p2:
        if pid not in patient_max_end or end > patient_max_end[pid]:
            patient_max_end[pid] = end

    # 4. Apply Frozen Out-Of-Fold Huber Models
    rolling_records = []
    
    # Load 5 fold models and scalers
    huber_models = [joblib.load(os.path.join(MODEL_DIR, f"huber_fold_{i}.joblib")) for i in range(5)]
    scalers = [joblib.load(os.path.join(MODEL_DIR, f"scaler_fold_{i}.joblib")) for i in range(5)]

    for i, (pid, start, end) in enumerate(meta_p2):
        fold = prot.assignment[pid][0]
        model = huber_models[fold]
        scaler = scalers[fold]

        x_feat = Fe_windows[i:i+1] # (1, 19)
        x_sc = scaler.transform(x_feat)
        pred_ph = float(model.predict(x_sc)[0])

        delta_t_sec = (patient_max_end[pid] - end) / 4.0
        delta_t_min = delta_t_sec / 60.0

        risk_score = -pred_ph
        risk_prob = 1.0 / (1.0 + np.exp(15.0 * (pred_ph - 7.15)))

        rolling_records.append({
            "window_index": i,
            "patient_id": str(pid),
            "fold": fold,
            "start_sample": start,
            "end_sample": end,
            "time_before_delivery_min": round(delta_t_min, 2),
            "true_ph": pid_to_ph[str(pid)],
            "primary_label_715": patient_labels_715[str(pid)],
            "severe_label_705": patient_labels_705[str(pid)],
            "predicted_ph_huber": round(pred_ph, 4),
            "acidemia_risk_score": round(risk_score, 4),
            "risk_prob_proxy": round(risk_prob, 4)
        })

    df_rolling = pd.DataFrame(rolling_records)
    df_rolling.to_csv(os.path.join(OUT_DIR, "rolling_predictions.csv"), index=False)

    print(f"Generated {len(df_rolling)} rolling predictions across {len(clean_pids)} patients.")
    print(f"Time before delivery range: {df_rolling['time_before_delivery_min'].min():.1f} min to {df_rolling['time_before_delivery_min'].max():.1f} min.")
    print("Master rolling predictions database saved to results/phase8_rolling/rolling_predictions.csv.\n")

if __name__ == "__main__":
    run_rolling_inference()
