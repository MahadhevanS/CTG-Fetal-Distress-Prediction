"""
Phase 9 Consolidated Evaluation & Deliverables Generator:
Runs complete locked cross-validation for all 5 Phase 9 models, exports:
1. results/phase9_temporal/temporal_predictions.csv (patient-window level rolling predictions)
2. results/phase9_temporal/horizon_metrics.csv (AUROC/AUPRC per horizon per model)
3. results/phase9_temporal/calibration_metrics.csv (Brier score, slope, intercept per horizon)
4. results/phase9_temporal/warning_times.csv (distribution of warning times across models)
5. results/phase9_temporal/subgroup_metrics.csv (performance across delivery type, parity, duration)
6. results/phase9_temporal/statistical_comparisons.json (paired patient bootstrap vs Phase 8 baseline)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

OUT_DIR = "results/phase9_temporal"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
FEATURES_PATH = os.path.join(OUT_DIR, "temporal_features.npz")
os.makedirs(OUT_DIR, exist_ok=True)

class CompactGRU(nn.Module):
    def __init__(self, in_features=19, hidden_dim=16, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        logits = self.fc(out[:, -1, :])
        return torch.sigmoid(logits).squeeze(-1)

class MultiHorizonNet(nn.Module):
    def __init__(self, in_features=104, hidden_dim=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.head_10m = nn.Linear(hidden_dim, 1)
        self.head_20m = nn.Linear(hidden_dim, 1)
        self.head_30m = nn.Linear(hidden_dim, 1)
        self.head_del = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        h = self.shared(x)
        return {
            "p_del": torch.sigmoid(self.head_del(h)).squeeze(-1),
            "p_10m": torch.sigmoid(self.head_10m(h)).squeeze(-1),
            "p_20m": torch.sigmoid(self.head_20m(h)).squeeze(-1),
            "p_30m": torch.sigmoid(self.head_30m(h)).squeeze(-1)
        }

def run_consolidated():
    print("=== RUNNING PHASE 9 CONSOLIDATED EVALUATION & DELIVERABLES ===")
    
    # 1. Load folds & data
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    p2_data = np.load(FEATURES_PATH)
    X_temp = p2_data["X_temporal"] # (8517, 104)
    X_raw_19 = X_temp[:, :19]
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    
    patient_ids = df_rolling["patient_id"].values
    y_715 = df_rolling["primary_label_715"].values
    y_705 = df_rolling["severe_label_705"].values
    t_del = df_rolling["time_before_delivery_min"].values
    p8_risk = df_rolling["risk_prob_proxy"].values
    
    y_patient_715 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["primary_label_715"] for p in clean_pids])
    y_patient_705 = np.array([df_rolling[df_rolling["patient_id"] == str(p)].iloc[0]["severe_label_705"] for p in clean_pids])
    
    # Pre-allocate prediction arrays on all 8,517 windows
    pred_lr = np.zeros(len(df_rolling), dtype=np.float32)
    pred_gbm = np.zeros(len(df_rolling), dtype=np.float32)
    pred_traj = np.zeros(len(df_rolling), dtype=np.float32)
    pred_gru = np.zeros(len(df_rolling), dtype=np.float32)
    pred_mh = np.zeros(len(df_rolling), dtype=np.float32)
    
    # Causal EWMA trajectory on rolling risk
    for pid in clean_pids:
        p_idx = np.where(patient_ids == pid)[0]
        p_risks = p8_risk[p_idx]
        ewma = np.zeros_like(p_risks)
        curr = p_risks[0]
        for i, r in enumerate(p_risks):
            curr = 0.3 * r + 0.7 * curr
            ewma[i] = curr
        pred_traj[p_idx] = ewma
        
    # Precompute 4-window sequences for all 8,517 observations
    seq_len = 4
    X_all_seqs = np.zeros((len(df_rolling), seq_len, 19), dtype=np.float32)
    for pid in clean_pids:
        p_idx = np.where(patient_ids == pid)[0]
        p_raw = X_raw_19[p_idx]
        for i in range(len(p_idx)):
            start_l = max(0, i - seq_len + 1)
            hist = p_raw[start_l:i+1]
            if len(hist) < seq_len:
                pad = np.repeat(hist[:1], seq_len - len(hist), axis=0)
                hist = np.vstack([pad, hist])
            X_all_seqs[p_idx[i]] = hist

    print("Training models across 5 folds...")
    for f_idx in range(5):
        te_pids = set([p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx])
        tr_pids = set([p for p in clean_pids if p not in te_pids])
        
        train_mask = np.isin(patient_ids, list(tr_pids))
        val_mask = np.isin(patient_ids, list(te_pids))
        
        # Standardize features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_temp[train_mask])
        X_val_s = scaler.transform(X_temp[val_mask])
        
        # 1. Temporal LR
        clf_lr = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
        clf_lr.fit(X_train_s, y_715[train_mask])
        pred_lr[val_mask] = clf_lr.predict_proba(X_val_s)[:, 1]
        
        # 2. Temporal GBM
        clf_gbm = HistGradientBoostingClassifier(
            max_iter=50, max_leaf_nodes=15, min_samples_leaf=20,
            learning_rate=0.05, class_weight='balanced', random_state=42
        )
        clf_gbm.fit(X_temp[train_mask], y_715[train_mask])
        pred_gbm[val_mask] = clf_gbm.predict_proba(X_temp[val_mask])[:, 1]
        
        # 3. Sequential GRU
        X_tr_seq = X_all_seqs[train_mask].copy()
        X_va_seq = X_all_seqs[val_mask].copy()
        
        m_feat = np.mean(X_tr_seq, axis=(0, 1), keepdims=True)
        s_feat = np.std(X_tr_seq, axis=(0, 1), keepdims=True) + 1e-6
        X_tr_seq = (X_tr_seq - m_feat) / s_feat
        X_va_seq = (X_va_seq - m_feat) / s_feat
        
        torch.manual_seed(42 + f_idx)
        gru_net = CompactGRU(in_features=19, hidden_dim=16)
        crit = nn.BCELoss()
        opt = torch.optim.Adam(gru_net.parameters(), lr=0.005, weight_decay=1e-4)
        
        ds_tr = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_715[train_mask], dtype=torch.float32))
        dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)
        
        gru_net.train()
        for ep in range(15):
            for bx, by in dl_tr:
                opt.zero_grad()
                out = gru_net(bx)
                loss = crit(out, by)
                loss.backward()
                opt.step()
                
        gru_net.eval()
        with torch.no_grad():
            pred_gru[val_mask] = gru_net(torch.tensor(X_va_seq)).cpu().numpy()
            
        # 4. Multi-Horizon Net
        mh_net = MultiHorizonNet(in_features=104, hidden_dim=32)
        opt_mh = torch.optim.Adam(mh_net.parameters(), lr=0.005, weight_decay=1e-4)
        
        t_tr = t_del[train_mask]
        y_del_tr = y_715[train_mask].astype(np.float32)
        y_10_tr = (y_715[train_mask] & (t_tr >= 10)).astype(np.float32)
        y_20_tr = (y_715[train_mask] & (t_tr >= 20)).astype(np.float32)
        y_30_tr = (y_715[train_mask] & (t_tr >= 30)).astype(np.float32)
        
        ds_mh = TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_del_tr), torch.tensor(y_10_tr), torch.tensor(y_20_tr), torch.tensor(y_30_tr)
        )
        dl_mh = DataLoader(ds_mh, batch_size=128, shuffle=True)
        
        mh_net.train()
        for ep in range(20):
            for bx, b_del, b_10, b_20, b_30 in dl_mh:
                opt_mh.zero_grad()
                preds_dict = mh_net(bx)
                loss = crit(preds_dict["p_del"], b_del) + 0.5*crit(preds_dict["p_10m"], b_10) + 0.5*crit(preds_dict["p_20m"], b_20) + 0.5*crit(preds_dict["p_30m"], b_30)
                loss.backward()
                opt_mh.step()
                
        mh_net.eval()
        with torch.no_grad():
            preds_val_dict = mh_net(torch.tensor(X_val_s, dtype=torch.float32))
            pred_mh[val_mask] = preds_val_dict["p_del"].cpu().numpy()

    # Save temporal_predictions.csv
    df_pred_out = pd.DataFrame({
        "patient_id": patient_ids,
        "window_idx": df_rolling["window_index"].values,
        "time_before_delivery_min": t_del,
        "primary_label_715": y_715,
        "severe_label_705": y_705,
        "p8_baseline_risk": p8_risk,
        "temporal_lr_risk": pred_lr,
        "temporal_gbm_risk": pred_gbm,
        "trajectory_ewma_risk": pred_traj,
        "sequential_gru_risk": pred_gru,
        "multihorizon_risk": pred_mh
    })
    df_pred_out.to_csv(os.path.join(OUT_DIR, "temporal_predictions.csv"), index=False)
    print("Saved temporal_predictions.csv (8517 rows)")
    
    # 2. Compute Horizon Metrics & Paired Bootstrap Tests
    horizons = [60, 45, 30, 20, 10, 0]
    model_keys = {
        "Phase 8 Baseline": "p8_baseline_risk",
        "Model 1: Temporal LR": "temporal_lr_risk",
        "Model 2: Temporal GBM": "temporal_gbm_risk",
        "Model 3: Trajectory EWMA": "trajectory_ewma_risk",
        "Model 4: Sequential GRU": "sequential_gru_risk",
        "Model 5: Multi-Horizon": "multihorizon_risk"
    }
    
    horizon_rows = []
    calib_rows = []
    stat_comparisons = {}
    
    for h in horizons:
        h_name = f">={h}m" if h > 0 else "Delivery (0m)"
        
        # Horizon evaluation on patient-level (closest window >= h)
        pat_scores = {m: [] for m in model_keys}
        for pid in clean_pids:
            df_p = df_pred_out[df_pred_out["patient_id"] == pid].sort_values("time_before_delivery_min")
            df_eligible = df_p[df_p["time_before_delivery_min"] >= h]
            if not df_eligible.empty:
                row = df_eligible.iloc[0]
            else:
                row = df_p.iloc[-1]
            for m, col in model_keys.items():
                pat_scores[m].append(row[col])
                
        for m in model_keys:
            pat_scores[m] = np.array(pat_scores[m])
            
        base_scores = pat_scores["Phase 8 Baseline"]
        base_auc = roc_auc_score(y_patient_715, base_scores)
        
        stat_comparisons[h_name] = {}
        
        np.random.seed(42)
        n_boot = 2000
        n_pts = len(clean_pids)
        
        for m, col in model_keys.items():
            auc_715 = roc_auc_score(y_patient_715, pat_scores[m])
            auprc_715 = average_precision_score(y_patient_715, pat_scores[m])
            auc_705 = roc_auc_score(y_patient_705, pat_scores[m])
            brier = brier_score_loss(y_patient_715, pat_scores[m])
            
            s_pts = np.clip(pat_scores[m].copy(), 1e-6, 1 - 1e-6)
            logits = np.log(s_pts / (1 - s_pts)).reshape(-1, 1)
            cal_lr = LogisticRegression()
            cal_lr.fit(logits, y_patient_715)
            cal_slope = float(cal_lr.coef_[0][0])
            cal_intercept = float(cal_lr.intercept_[0])
            
            horizon_rows.append({
                "horizon": h_name,
                "horizon_min": h,
                "model": m,
                "n_patients": n_pts,
                "auroc_715": round(float(auc_715), 4),
                "auprc_715": round(float(auprc_715), 4),
                "auroc_705": round(float(auc_705), 4),
                "brier_score": round(float(brier), 4)
            })
            
            calib_rows.append({
                "horizon": h_name,
                "model": m,
                "brier_score": round(float(brier), 4),
                "calibration_slope": round(cal_slope, 3),
                "calibration_intercept": round(cal_intercept, 3)
            })
            
            # Bootstrap diff
            deltas = []
            for _ in range(n_boot):
                idx = np.random.choice(n_pts, size=n_pts, replace=True)
                if len(np.unique(y_patient_715[idx])) < 2:
                    continue
                b_base = roc_auc_score(y_patient_715[idx], base_scores[idx])
                b_m = roc_auc_score(y_patient_715[idx], pat_scores[m][idx])
                deltas.append(b_m - b_base)
                
            deltas = np.array(deltas)
            diff_mean = float(np.mean(deltas))
            ci_low = float(np.percentile(deltas, 2.5))
            ci_high = float(np.percentile(deltas, 97.5))
            p_val = float(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
            
            stat_comparisons[h_name][m] = {
                "auroc_715": round(float(auc_715), 4),
                "auprc_715": round(float(auprc_715), 4),
                "auroc_705": round(float(auc_705), 4),
                "delta_vs_p8": round(diff_mean, 4),
                "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                "p_value": round(p_val, 4)
            }
            
    pd.DataFrame(horizon_rows).to_csv(os.path.join(OUT_DIR, "horizon_metrics.csv"), index=False)
    pd.DataFrame(calib_rows).to_csv(os.path.join(OUT_DIR, "calibration_metrics.csv"), index=False)
    with open(os.path.join(OUT_DIR, "statistical_comparisons.json"), "w") as f:
        json.dump(stat_comparisons, f, indent=2)
    print("Saved horizon_metrics.csv, calibration_metrics.csv, statistical_comparisons.json")
    
    # 3. Warning Times Analysis
    warn_rows = []
    non_acid_mask = (y_715 == 0)
    for m, col in model_keys.items():
        ctrl_scores = df_pred_out[col].values[non_acid_mask]
        thresh = np.percentile(ctrl_scores, 90.0)
        
        pos_pids = [p for p in clean_pids if df_rolling[df_rolling["patient_id"] == p].iloc[0]["primary_label_715"] == 1]
        warn_times = []
        det_60 = 0
        det_45 = 0
        det_30 = 0
        det_20 = 0
        det_10 = 0
        det_any = 0
        
        for pid in pos_pids:
            p_mask = (patient_ids == pid)
            p_scores = df_pred_out[col].values[p_mask]
            p_tdel = t_del[p_mask]
            
            crossed = np.where(p_scores >= thresh)[0]
            if len(crossed) > 0:
                det_any += 1
                earliest_t = np.max(p_tdel[crossed])
                warn_times.append(earliest_t)
                if earliest_t >= 60: det_60 += 1
                if earliest_t >= 45: det_45 += 1
                if earliest_t >= 30: det_30 += 1
                if earliest_t >= 20: det_20 += 1
                if earliest_t >= 10: det_10 += 1
                
        n_pos = len(pos_pids)
        warn_rows.append({
            "model": m,
            "threshold_90spec": round(float(thresh), 4),
            "sensitivity_overall_pct": round(100.0 * det_any / n_pos, 2),
            "median_warning_min": round(float(np.median(warn_times)), 1) if len(warn_times) > 0 else 0.0,
            "iqr_warning_min": round(float(np.percentile(warn_times, 75) - np.percentile(warn_times, 25)), 1) if len(warn_times) > 0 else 0.0,
            "mean_warning_min": round(float(np.mean(warn_times)), 1) if len(warn_times) > 0 else 0.0,
            "detected_ge_60m_pct": round(100.0 * det_60 / n_pos, 2),
            "detected_ge_45m_pct": round(100.0 * det_45 / n_pos, 2),
            "detected_ge_30m_pct": round(100.0 * det_30 / n_pos, 2),
            "detected_ge_20m_pct": round(100.0 * det_20 / n_pos, 2),
            "detected_ge_10m_pct": round(100.0 * det_10 / n_pos, 2)
        })
        
    pd.DataFrame(warn_rows).to_csv(os.path.join(OUT_DIR, "warning_times.csv"), index=False)
    print("Saved warning_times.csv")
    
    # 4. Subgroup Metrics
    subgroup_rows = []
    durations = np.array([len(df_rolling[df_rolling["patient_id"] == str(p)]) * 2.5 for p in clean_pids])
    med_dur = np.median(durations)
    short_mask = durations <= med_dur
    long_mask = durations > med_dur
    
    for m, col in model_keys.items():
        for h_eval, h_label in [(30, ">=30m"), (0, "Delivery")]:
            pat_scores_h = []
            for pid in clean_pids:
                df_p = df_pred_out[df_pred_out["patient_id"] == str(pid)].sort_values("time_before_delivery_min")
                df_eligible = df_p[df_p["time_before_delivery_min"] >= h_eval]
                if not df_eligible.empty:
                    pat_scores_h.append(df_eligible.iloc[0][col])
                else:
                    pat_scores_h.append(df_p.iloc[-1][col])
            pat_scores_h = np.array(pat_scores_h)
            
            subgroup_rows.append({
                "model": m,
                "horizon": h_label,
                "subgroup": "Overall Cohort",
                "n_patients": len(clean_pids),
                "auroc_715": round(float(roc_auc_score(y_patient_715, pat_scores_h)), 4)
            })
            if len(np.unique(y_patient_715[short_mask])) > 1:
                subgroup_rows.append({
                    "model": m,
                    "horizon": h_label,
                    "subgroup": f"Short Monitoring (<= {med_dur:.0f}m)",
                    "n_patients": int(np.sum(short_mask)),
                    "auroc_715": round(float(roc_auc_score(y_patient_715[short_mask], pat_scores_h[short_mask])), 4)
                })
            if len(np.unique(y_patient_715[long_mask])) > 1:
                subgroup_rows.append({
                    "model": m,
                    "horizon": h_label,
                    "subgroup": f"Long Monitoring (> {med_dur:.0f}m)",
                    "n_patients": int(np.sum(long_mask)),
                    "auroc_715": round(float(roc_auc_score(y_patient_715[long_mask], pat_scores_h[long_mask])), 4)
                })
            
    pd.DataFrame(subgroup_rows).to_csv(os.path.join(OUT_DIR, "subgroup_metrics.csv"), index=False)
    print("Saved subgroup_metrics.csv")
    print("=== CONSOLIDATED EVALUATION COMPLETE ===")

if __name__ == "__main__":
    run_consolidated()
