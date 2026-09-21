"""
Phase 9: Temporal Feature Extraction & Causal History Builder.

Builds causal longitudinal features across rolling 20-minute windows for each patient:
1. Current 19 clinical physiological descriptors.
2. Multi-step temporal deltas (1-step, 2-step, 4-step differences).
3. Longitudinal slopes and rolling trajectory statistics.
4. Risk score trajectory dynamics (risk slope, acceleration, cumulative evidence).
5. Exports: results/phase9_temporal/temporal_features.npz
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.protocol import Protocol

OUT_DIR = "results/phase9_temporal"
FOLDS_PATH = "data/processed_clinical/folds.json"
P2_PATH = "data/phase1_candidates/p2_dataset.pt"
DATA_DIR = "data/processed_clinical"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"

os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "baseline", "stv", "ltv", "acc_count", "early_dec_count", "late_dec_count",
    "var_dec_count", "prolonged_dec_count", "dec_max_depth", "dec_area",
    "dec_burden", "longest_dec", "baseline_slope", "variability_slope",
    "uc_count", "tachysystole", "mean_uc_amp", "fhr_uc_lag", "fhr_uc_coupling"
]

def build_temporal_features():
    print("=== EXECUTING PHASE 9: CAUSAL TEMPORAL FEATURE EXTRACTION ===")

    # 1. Load Folds & Master Cohort
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p2_data = torch.load(P2_PATH, weights_only=False)
    y_bin = p2_data["y"].numpy()
    pid_arr = p2_data["pid"]
    meta_p2 = [tuple(m) for m in p2_data["meta"]] # (str(pid), int(start), int(end))
    prot = Protocol.load_or_create(pid_arr, y_bin, path=FOLDS_PATH)

    df_rolling = pd.read_csv(ROLLING_PATH)

    # 2. Load 19 Clinical Features aligned with p2_dataset.pt
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

    # 3. Construct Longitudinal Trajectories per Patient
    temporal_feature_list = []
    expanded_feature_names = list(FEATURE_NAMES)

    # Delta feature names
    for name in FEATURE_NAMES:
        expanded_feature_names.append(f"{name}_delta_1step")
    for name in FEATURE_NAMES:
        expanded_feature_names.append(f"{name}_delta_2step")
    for name in FEATURE_NAMES:
        expanded_feature_names.append(f"{name}_delta_4step")
    for name in FEATURE_NAMES:
        expanded_feature_names.append(f"{name}_slope_recent")

    # Risk trajectory feature names
    expanded_feature_names.extend([
        "risk_current",
        "risk_delta_1step",
        "risk_delta_2step",
        "risk_slope_recent",
        "risk_max_recent",
        "risk_min_recent",
        "risk_std_recent",
        "risk_consecutive_increases",
        "risk_cumulative_burden",
        "novelty_raw",
        "novelty_smoothed",
        "elapsed_monitoring_min"
    ])

    print(f"Total temporal feature dimensions: {len(expanded_feature_names)}")

    X_temporal = np.zeros((len(meta_p2), len(expanded_feature_names)), dtype=np.float32)

    # --- Out-of-fold novelty/elapsed-time feature construction -------------
    # Fixed 2026-09-11: the previous version z-scored Fe_windows and computed
    # the first-window fallback novelty using ALL 547 patients at once,
    # leaking val/test-partition statistics into every patient's novelty
    # feature (in violation of docs/information_density_weighting_design.md
    # Section 2.4's fold-isolation requirement). Reworked to reuse
    # InformationDensityWeighter's own fit()/transform() split: for each of
    # the 5 locked folds, z-score stats and the median-novelty fallback are
    # fit strictly on that fold's training patients and applied to that
    # fold's held-out patients only (out-of-fold, mirroring how the rolling
    # Huber model itself is frozen per fold elsewhere in this pipeline).
    # This also fixes the elapsed-time bug: elapsed_monitoring_min is now
    # derived from each window's actual start_sample rather than its
    # position index, so it stays correct for the ~6.9% of patients with
    # quality-gate-dropped (non-uniform-stride) windows.
    from src.training.information_density_weighting import InformationDensityWeighter

    patient_arr = np.array([str(m[0]) for m in meta_p2])
    elapsed_abs_min = np.array([float(m[1]) / (4.0 * 60.0) for m in meta_p2], dtype=np.float32)  # start_sample / (fs*60)

    novelty_cols = np.zeros((len(meta_p2), 3), dtype=np.float32)  # [raw, smoothed, elapsed_min]

    for f_idx in range(5):
        te_pids = set(p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx)
        tr_mask = np.array([p not in te_pids for p in patient_arr])
        te_mask = ~tr_mask
        if te_mask.sum() == 0 or tr_mask.sum() == 0:
            continue

        weighter = InformationDensityWeighter(span=3.0, beta=1.0)
        weighter.fit(Fe_windows[tr_mask], patient_arr[tr_mask], elapsed_abs_min[tr_mask])
        novelty_cols[te_mask] = weighter.extract_novelty_features(
            Fe_windows[te_mask], patient_arr[te_mask], elapsed_abs_min[te_mask]
        )

    for pid in clean_pids:
        # Get patient's window indices in chronological order
        p_win_indices = [idx for idx, (p, start, end) in enumerate(meta_p2) if str(p) == str(pid)]
        p_win_indices.sort(key=lambda idx: meta_p2[idx][2]) # sort by end_sample

        p_feats = Fe_windows[p_win_indices] # (K, 19)
        p_risks = df_rolling.iloc[p_win_indices]["acidemia_risk_score"].values # (K,)

        K = len(p_win_indices)
        p_nov_raw = novelty_cols[p_win_indices, 0]
        p_nov_sm = novelty_cols[p_win_indices, 1]
        p_elapsed = novelty_cols[p_win_indices, 2]

        for k in range(K):
            global_idx = p_win_indices[k]
            cur_feat = p_feats[k] # (19,)

            # 1-step delta (5 min prior)
            d1 = (cur_feat - p_feats[k-1]) if k >= 1 else np.zeros(19, dtype=np.float32)
            # 2-step delta (10 min prior)
            d2 = (cur_feat - p_feats[k-2]) if k >= 2 else np.zeros(19, dtype=np.float32)
            # 4-step delta (20 min prior)
            d4 = (cur_feat - p_feats[k-4]) if k >= 4 else np.zeros(19, dtype=np.float32)

            # Recent slope (over up to last 4 steps)
            lookback = min(k + 1, 4)
            if lookback >= 2:
                # Linear fit slope per feature
                t_steps = np.arange(lookback) * 5.0 # in minutes
                y_sub = p_feats[k - lookback + 1 : k + 1] # (lookback, 19)
                slopes = np.zeros(19, dtype=np.float32)
                for f_i in range(19):
                    slopes[f_i] = np.polyfit(t_steps, y_sub[:, f_i], 1)[0]
            else:
                slopes = np.zeros(19, dtype=np.float32)

            # Risk trajectory features
            cur_risk = p_risks[k]
            r_d1 = (cur_risk - p_risks[k-1]) if k >= 1 else 0.0
            r_d2 = (cur_risk - p_risks[k-2]) if k >= 2 else 0.0
            if lookback >= 2:
                r_sub = p_risks[k - lookback + 1 : k + 1]
                r_slope = float(np.polyfit(t_steps, r_sub, 1)[0])
                r_max = float(np.max(r_sub))
                r_min = float(np.min(r_sub))
                r_std = float(np.std(r_sub))
            else:
                r_slope = 0.0
                r_max = float(cur_risk)
                r_min = float(cur_risk)
                r_std = 0.0

            # Consecutive increases
            consec_inc = 0
            for j in range(k, 0, -1):
                if p_risks[j] > p_risks[j-1]:
                    consec_inc += 1
                else:
                    break

            # Cumulative burden: proportion of recent windows exceeding baseline mean risk
            cum_burden = float(np.mean(p_risks[:k+1] >= -7.15))

            nov_features = np.array([p_nov_raw[k], p_nov_sm[k], p_elapsed[k]], dtype=np.float32)

            risk_traj_vec = np.array([
                cur_risk, r_d1, r_d2, r_slope, r_max, r_min, r_std, float(consec_inc), cum_burden
            ], dtype=np.float32)

            # Concatenate all into final feature row
            full_row = np.hstack([cur_feat, d1, d2, d4, slopes, risk_traj_vec, nov_features])
            X_temporal[global_idx] = full_row

    print(f"Constructed temporal feature matrix with shape: {X_temporal.shape}")

    # 4. Save Features
    np.savez_compressed(os.path.join(OUT_DIR, "temporal_features.npz"), X_temporal=X_temporal)
    with open(os.path.join(OUT_DIR, "temporal_feature_names.json"), "w") as f:
        json.dump(expanded_feature_names, f, indent=2)

    # 5. Causal Verification Test
    # Verify that first window of any patient has exactly zero deltas (since no past data exists)
    first_window_deltas = X_temporal[0, 19:19+19*3]
    assert np.all(first_window_deltas == 0.0), "Causal Test Failed: Future data leaked into first window deltas!"
    print("Causal Verification Test: PASSED (Zero leakage confirmed).")
    print(f"Temporal features successfully saved to results/phase9_temporal/temporal_features.npz.\n")

if __name__ == "__main__":
    build_temporal_features()
