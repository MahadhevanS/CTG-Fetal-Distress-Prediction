"""
Phase 9 Model 4: Sequential Neural Model (Compact GRU on Feature Sequences).

Processes the sequence of consecutive 20-minute clinical feature vectors (up to L=6 steps)
under patient-grouped 5-fold CV to capture non-linear temporal dynamics.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

OUT_DIR = "results/phase9_temporal"
FOLDS_PATH = "data/processed_clinical/folds.json"
DATA_DIR = "data/processed_clinical"
P2_PATH = "data/phase1_candidates/p2_dataset.pt"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CompactGRU(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: (B, seq_len, 19)
        out, h_n = self.gru(x) # h_n: (1, B, hidden_dim)
        logits = self.fc(h_n[-1]).squeeze(-1)
        return logits

def run_sequential_model():
    print("=== EXECUTING PHASE 9 MODEL 4: COMPACT SEQUENTIAL GRU ===")
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p2_data = torch.load(P2_PATH, weights_only=False)
    meta_p2 = [tuple(m) for m in p2_data["meta"]]
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)

    y_patient = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["primary_label_715"] for p in clean_pids], dtype=np.float32)
    y_patient_705 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["severe_label_705"] for p in clean_pids], dtype=np.float32)

    # Load 19 Clinical Features
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

    # For each patient, construct historical sequences of length L=6
    SEQ_LEN = 6
    horizons = [60, 45, 30, 20, 10, 0]
    results_by_horizon = {}

    for h in horizons:
        # Build sequence tensor for each patient at horizon h
        seq_list = []
        for pid in clean_pids:
            p_win_indices = [idx for idx, (p, start, end) in enumerate(meta_p2) if str(p) == str(pid)]
            p_win_indices.sort(key=lambda idx: meta_p2[idx][2])
            
            # Find eligible windows up to horizon h
            df_p = df_rolling.iloc[p_win_indices]
            df_eligible = df_p[df_p["time_before_delivery_min"] >= h]
            if not df_eligible.empty:
                cutoff_k = len(df_eligible)
            else:
                cutoff_k = 1

            sub_indices = p_win_indices[:cutoff_k]
            p_feats = Fe_windows[sub_indices]
            # Pad or truncate to SEQ_LEN
            if len(p_feats) >= SEQ_LEN:
                seq = p_feats[-SEQ_LEN:]
            else:
                pad_len = SEQ_LEN - len(p_feats)
                pad = np.tile(p_feats[0:1], (pad_len, 1))
                seq = np.vstack([pad, p_feats])
            seq_list.append(seq)

        X_seq_all = np.array(seq_list, dtype=np.float32) # (547, 6, 19)

        oof_scores = np.zeros(len(clean_pids), dtype=np.float32)

        for f_idx in range(5):
            te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
            tr_pat = np.array([i for i, p in enumerate(clean_pids) if p not in te_pids])
            te_pat = np.array([i for i, p in enumerate(clean_pids) if p in te_pids])

            # Scale features
            scaler = StandardScaler()
            X_tr_flat = scaler.fit_transform(X_seq_all[tr_pat].reshape(-1, 19)).reshape(-1, SEQ_LEN, 19)
            X_te_flat = scaler.transform(X_seq_all[te_pat].reshape(-1, 19)).reshape(-1, SEQ_LEN, 19)

            X_tr_t = torch.tensor(X_tr_flat, dtype=torch.float32)
            y_tr_t = torch.tensor(y_patient[tr_pat], dtype=torch.float32)
            X_te_t = torch.tensor(X_te_flat, dtype=torch.float32)

            train_ds = TensorDataset(X_tr_t, y_tr_t)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

            model = CompactGRU(input_dim=19, hidden_dim=32).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(DEVICE))

            model.train()
            for ep in range(25):
                for bx, by in train_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                te_out = torch.sigmoid(model(X_te_t.to(DEVICE))).cpu().numpy()
                oof_scores[te_pat] = te_out

        auc_715 = roc_auc_score(y_patient, oof_scores)
        auprc_715 = average_precision_score(y_patient, oof_scores)
        auc_705 = roc_auc_score(y_patient_705, oof_scores)

        results_by_horizon[f"horizon_{h}m"] = {
            "horizon_min": h,
            "auroc_715": round(float(auc_715), 4),
            "auprc_715": round(float(auprc_715), 4),
            "auroc_severe_705": round(float(auc_705), 4)
        }
        print(f"Sequential GRU Horizon >={h:02d}m: AUROC (pH<=7.15) = {auc_715:.4f}, AUPRC = {auprc_715:.4f} | Severe (<=7.05) AUROC = {auc_705:.4f}")

    with open(os.path.join(OUT_DIR, "sequential_gru_metrics.json"), "w") as f:
        json.dump(results_by_horizon, f, indent=2)

if __name__ == "__main__":
    run_sequential_model()
