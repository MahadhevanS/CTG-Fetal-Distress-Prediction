"""
Phase 9 Model 5: Multi-Horizon Joint Supervision Model.

Jointly supervises multi-task heads predicting:
1. Delivery acidemia (pH <= 7.15)
2. Early warning at >=30m
3. Early warning at >=10m
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
P2_PATH = "data/phase1_candidates/p2_dataset.pt"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
FEATURES_PATH = os.path.join(OUT_DIR, "temporal_features.npz")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHorizonNet(nn.Module):
    def __init__(self, input_dim=104, hidden_dim=48):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 24),
            nn.ReLU()
        )
        self.head_deliv = nn.Linear(24, 1)
        self.head_30m = nn.Linear(24, 1)
        self.head_10m = nn.Linear(24, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.head_deliv(h).squeeze(-1), self.head_30m(h).squeeze(-1), self.head_10m(h).squeeze(-1)

def run_multihorizon():
    print("=== EXECUTING PHASE 9 MODEL 5: MULTI-HORIZON JOINT SUPERVISION ===")
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p2_data = np.load(FEATURES_PATH)
    X_temp = p2_data["X_temporal"]

    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    y_patient = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["primary_label_715"] for p in clean_pids], dtype=np.float32)
    y_patient_705 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["severe_label_705"] for p in clean_pids], dtype=np.float32)

    horizons = [60, 45, 30, 20, 10, 0]
    results_by_horizon = {}

    for h in horizons:
        oof_scores = np.zeros(len(clean_pids), dtype=np.float32)

        pat_win_indices = []
        for pid in clean_pids:
            df_p = df_rolling[df_rolling["patient_id"] == pid].sort_values("time_before_delivery_min")
            df_eligible = df_p[df_p["time_before_delivery_min"] >= h]
            if not df_eligible.empty:
                chosen_idx = int(df_eligible.iloc[0]["window_index"])
            else:
                chosen_idx = int(df_p.iloc[-1]["window_index"])
            pat_win_indices.append(chosen_idx)

        X_h = X_temp[pat_win_indices] # (547, 104)

        for f_idx in range(5):
            te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
            tr_pat = np.array([i for i, p in enumerate(clean_pids) if p not in te_pids])
            te_pat = np.array([i for i, p in enumerate(clean_pids) if p in te_pids])

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_h[tr_pat])
            X_te = scaler.transform(X_h[te_pat])

            X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
            y_tr_t = torch.tensor(y_patient[tr_pat], dtype=torch.float32)
            X_te_t = torch.tensor(X_te, dtype=torch.float32)

            train_ds = TensorDataset(X_tr_t, y_tr_t)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

            model = MultiHorizonNet(input_dim=104, hidden_dim=48).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(DEVICE))

            model.train()
            for ep in range(25):
                for bx, by in train_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    p_deliv, p_30, p_10 = model(bx)
                    loss = crit(p_deliv, by) + 0.8 * crit(p_30, by) + 0.5 * crit(p_10, by)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                te_deliv, te_30, te_10 = model(X_te_t.to(DEVICE))
                if h >= 30:
                    te_preds = torch.sigmoid(te_30).cpu().numpy()
                elif h >= 10:
                    te_preds = torch.sigmoid(te_10).cpu().numpy()
                else:
                    te_preds = torch.sigmoid(te_deliv).cpu().numpy()
                oof_scores[te_pat] = te_preds

        auc_715 = roc_auc_score(y_patient, oof_scores)
        auprc_715 = average_precision_score(y_patient, oof_scores)
        auc_705 = roc_auc_score(y_patient_705, oof_scores)

        results_by_horizon[f"horizon_{h}m"] = {
            "horizon_min": h,
            "auroc_715": round(float(auc_715), 4),
            "auprc_715": round(float(auprc_715), 4),
            "auroc_severe_705": round(float(auc_705), 4)
        }
        print(f"Multi-Horizon Horizon >={h:02d}m: AUROC (pH<=7.15) = {auc_715:.4f}, AUPRC = {auprc_715:.4f} | Severe (<=7.05) AUROC = {auc_705:.4f}")

    with open(os.path.join(OUT_DIR, "multihorizon_metrics.json"), "w") as f:
        json.dump(results_by_horizon, f, indent=2)

if __name__ == "__main__":
    run_multihorizon()
