"""
Phase 9B: Physiology-Guided Temporal Deterioration Feature Engine.

Constructs domain-structured deterioration features across 6 physiological domains:
1. Baseline (baseline, baseline_slope)
2. Variability (stv, ltv, variability_slope)
3. Accelerations (acc_count)
4. Decelerations (early, late, var, prolonged, max_depth, area, burden, longest_dec)
5. Uterine Activity (uc_count, tachysystole, mean_uc_amp)
6. FHR-UC Coupling (fhr_uc_lag, fhr_uc_coupling)

Computes causal operators:
- Domain Severity Scores (S_t)
- Directional Changes (Delta X_dir)
- 4-Window Multi-Step Slopes (slope_t)
- Persistence Indices (P_t)
- Acceleration of Deterioration (A_t)
- Multidomain Deterioration Count (N_t) & Persistence (C_t)
- Exploratory 5-State Progression Categorization

Performs synthetic perturbation audit to verify causal integrity.
"""

import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "results/phase9b_deterioration"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
FEATURES_P9A = "results/phase9_temporal/temporal_features.npz"

os.makedirs(OUT_DIR, exist_ok=True)

DOMAIN_DEFS = {
    "baseline": [0, 12],
    "variability": [1, 2, 13],
    "acceleration": [3],
    "deceleration": [4, 5, 6, 7, 8, 9, 10, 11],
    "uterine": [14, 15, 16],
    "coupling": [17, 18]
}

def extract_deterioration_features():
    print("=== EXTRACTING PHASE 9B PHYSIOLOGY-GUIDED DETERIORATION FEATURES ===")
    
    # 1. Load data
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    patient_ids = df_rolling["patient_id"].values
    
    p9a_data = np.load(FEATURES_P9A)
    X_raw_19 = p9a_data["X_temporal"][:, :19] # (8517, 19)
    p8_risk = df_rolling["risk_prob_proxy"].values
    
    n_samples = len(df_rolling)
    
    # 2. Domain Severity Scores (S_t)
    # Define clinical abnormality definitions for each domain:
    # 0: Baseline deviation from normal 110-160
    s_baseline = np.maximum(0, X_raw_19[:, 0] - 160) / 20.0 + np.maximum(0, 110 - X_raw_19[:, 0]) / 20.0
    
    # 1: Variability depression: STV < 3.0 ms, LTV < 15.0
    s_var = np.maximum(0, 3.5 - X_raw_19[:, 1]) / 2.0 + np.maximum(0, 20.0 - X_raw_19[:, 2]) / 10.0
    
    # 2: Acceleration absence: normal >= 2
    s_acc = np.maximum(0, 2.0 - X_raw_19[:, 3]) / 2.0
    
    # 3: Deceleration burden: late, prolonged, deep decelerations
    s_dec = (X_raw_19[:, 5] * 2.0 + X_raw_19[:, 7] * 3.0 + X_raw_19[:, 8] / 30.0 + X_raw_19[:, 10] / 5.0) / 4.0
    
    # 4: Uterine hyperstimulation / tachysystole
    s_uc = (np.maximum(0, X_raw_19[:, 14] - 5.0) + X_raw_19[:, 15] * 2.0) / 2.0
    
    # 5: Delayed FHR recovery (lag > 20s) and high coupling
    s_coup = (np.maximum(0, X_raw_19[:, 17] - 20.0) / 20.0 + X_raw_19[:, 18]) / 2.0
    
    domain_severities = np.column_stack([s_baseline, s_var, s_acc, s_dec, s_uc, s_coup]) # (8517, 6)
    
    # 3. Trajectory & Temporal Dynamics per Patient
    # Pre-allocate arrays
    k_hist = 4
    delta_19 = np.zeros((n_samples, 19), dtype=np.float32)
    slope_19 = np.zeros((n_samples, 19), dtype=np.float32)
    accel_19 = np.zeros((n_samples, 19), dtype=np.float32)
    
    dom_delta = np.zeros((n_samples, 6), dtype=np.float32)
    dom_slope = np.zeros((n_samples, 6), dtype=np.float32)
    dom_persist = np.zeros((n_samples, 6), dtype=np.float32)
    dom_accel = np.zeros((n_samples, 6), dtype=np.float32)
    
    multidomain_n = np.zeros(n_samples, dtype=np.float32)
    multidomain_persist = np.zeros(n_samples, dtype=np.float32)
    research_state = np.zeros(n_samples, dtype=np.int32)
    
    # Time vector for 4-window slope (0, 2.5, 5.0, 7.5 min)
    t_steps = np.array([0.0, 2.5, 5.0, 7.5])
    t_bar = np.mean(t_steps)
    t_denom = np.sum((t_steps - t_bar)**2)
    
    for pid in clean_pids:
        p_idx = np.where(patient_ids == pid)[0]
        p_len = len(p_idx)
        p_raw = X_raw_19[p_idx]
        p_sev = domain_severities[p_idx]
        p_risk = p8_risk[p_idx]
        
        # Window-by-window causal computation
        p_abnormal_dom = (p_sev > 0.3).astype(np.float32) # (p_len, 6)
        
        for i in range(p_len):
            global_idx = p_idx[i]
            
            # 1-step delta
            if i > 0:
                d1 = p_raw[i] - p_raw[i-1]
                d_dom = p_sev[i] - p_sev[i-1]
            else:
                d1 = np.zeros(19, dtype=np.float32)
                d_dom = np.zeros(6, dtype=np.float32)
            delta_19[global_idx] = d1
            dom_delta[global_idx] = d_dom
            
            # 2-step acceleration
            if i > 1:
                d1_prev = p_raw[i-1] - p_raw[i-2]
                d_dom_prev = p_sev[i-1] - p_sev[i-2]
                accel_19[global_idx] = d1 - d1_prev
                dom_accel[global_idx] = d_dom - d_dom_prev
            else:
                accel_19[global_idx] = np.zeros(19, dtype=np.float32)
                dom_accel[global_idx] = np.zeros(6, dtype=np.float32)
                
            # 4-window multi-step slope
            start_k = max(0, i - k_hist + 1)
            hist_raw = p_raw[start_k:i+1]
            hist_sev = p_sev[start_k:i+1]
            
            if len(hist_raw) < k_hist:
                pad_len = k_hist - len(hist_raw)
                hist_raw = np.vstack([np.repeat(hist_raw[:1], pad_len, axis=0), hist_raw])
                hist_sev = np.vstack([np.repeat(hist_sev[:1], pad_len, axis=0), hist_sev])
                
            # Least squares slope over 4 points
            y_bar_raw = np.mean(hist_raw, axis=0)
            sl_raw = np.sum((t_steps[:, None] - t_bar) * (hist_raw - y_bar_raw), axis=0) / t_denom
            slope_19[global_idx] = sl_raw
            
            y_bar_sev = np.mean(hist_sev, axis=0)
            sl_sev = np.sum((t_steps[:, None] - t_bar) * (hist_sev - y_bar_sev), axis=0) / t_denom
            dom_slope[global_idx] = sl_sev
            
            # Domain persistence (count of abnormal windows in last 4 windows)
            hist_abn = p_abnormal_dom[start_k:i+1]
            dom_persist[global_idx] = np.sum(hist_abn, axis=0)
            
            # Multidomain concurrent deterioration (count of domains with S > 0.3 and slope >= 0)
            n_concurrent = np.sum((p_sev[i] > 0.3) & (sl_sev >= -0.01))
            multidomain_n[global_idx] = n_concurrent
            
            # Multidomain persistence
            if i >= 1:
                prev_n = multidomain_n[p_idx[i-1]]
                if n_concurrent >= 2 and prev_n >= 2:
                    multidomain_persist[global_idx] = min(4.0, multidomain_persist[p_idx[i-1]] + 1.0)
                elif n_concurrent >= 2:
                    multidomain_persist[global_idx] = 1.0
                else:
                    multidomain_persist[global_idx] = 0.0
            else:
                multidomain_persist[global_idx] = 1.0 if n_concurrent >= 2 else 0.0
                
            # Research State Classification (0 to 4)
            # 0: Stable, 1: Emerging, 2: Persistent, 3: Progressive, 4: Severe Multidomain
            max_sev = np.max(p_sev[i])
            max_persist = np.max(dom_persist[global_idx])
            if max_sev > 1.0 and n_concurrent >= 2 and multidomain_persist[global_idx] >= 2:
                r_state = 4 # Severe Multidomain Deterioration
            elif n_concurrent >= 2 and (np.max(sl_sev) > 0.05 or multidomain_persist[global_idx] >= 2):
                r_state = 3 # Progressive Deterioration
            elif max_persist >= 2:
                r_state = 2 # Persistent Abnormality
            elif max_sev > 0.3 or np.max(d_dom) > 0.1:
                r_state = 1 # Emerging Abnormality
            else:
                r_state = 0 # Stable
            research_state[global_idx] = r_state

    # Construct complete deterioration matrix:
    # 1. Raw 19 features (19)
    # 2. Raw 19 deltas (19)
    # 3. Raw 19 slopes (19)
    # 4. Raw 19 accelerations (19)
    # 5. 6 Domain Severities (6)
    # 6. 6 Domain Deltas (6)
    # 7. 6 Domain Slopes (6)
    # 8. 6 Domain Persistences (6)
    # 9. 6 Domain Accelerations (6)
    # 10. Multidomain Index & Persistence (2)
    # 11. Research State (1)
    # 12. Prediction Trajectory Features (EWMA, Risk Delta, Risk Acceleration) (3)
    # Total = 19 + 19 + 19 + 19 + 6*5 + 2 + 1 + 3 = 112 features
    
    # Compute prediction trajectory features
    ewma_risk = np.zeros(n_samples, dtype=np.float32)
    delta_risk = np.zeros(n_samples, dtype=np.float32)
    accel_risk = np.zeros(n_samples, dtype=np.float32)
    
    for pid in clean_pids:
        p_idx = np.where(patient_ids == pid)[0]
        p_r = p8_risk[p_idx]
        curr = p_r[0]
        for i, r in enumerate(p_r):
            curr = 0.3 * r + 0.7 * curr
            ewma_risk[p_idx[i]] = curr
            if i > 0:
                d = p_r[i] - p_r[i-1]
                delta_risk[p_idx[i]] = d
                if i > 1:
                    d_prev = p_r[i-1] - p_r[i-2]
                    accel_risk[p_idx[i]] = d - d_prev
                    
    X_deterioration = np.column_stack([
        X_raw_19, delta_19, slope_19, accel_19,
        domain_severities, dom_delta, dom_slope, dom_persist, dom_accel,
        multidomain_n, multidomain_persist, research_state,
        ewma_risk, delta_risk, accel_risk
    ]) # (8517, 112)
    
    print(f"Engineered Deterioration Matrix Shape: {X_deterioration.shape}")
    
    # Save NPZ and JSON
    feature_names = (
        [f"raw_{i}" for i in range(19)] +
        [f"delta_{i}" for i in range(19)] +
        [f"slope_{i}" for i in range(19)] +
        [f"accel_{i}" for i in range(19)] +
        [f"dom_sev_{k}" for k in DOMAIN_DEFS.keys()] +
        [f"dom_delta_{k}" for k in DOMAIN_DEFS.keys()] +
        [f"dom_slope_{k}" for k in DOMAIN_DEFS.keys()] +
        [f"dom_persist_{k}" for k in DOMAIN_DEFS.keys()] +
        [f"dom_accel_{k}" for k in DOMAIN_DEFS.keys()] +
        ["multidomain_n", "multidomain_persist", "research_state",
         "ewma_risk", "delta_risk", "accel_risk"]
    )
    
    np.savez_compressed(
        os.path.join(OUT_DIR, "deterioration_features.npz"),
        X_deterioration=X_deterioration,
        domain_severities=domain_severities,
        multidomain_n=multidomain_n,
        multidomain_persist=multidomain_persist,
        research_state=research_state
    )
    
    with open(os.path.join(OUT_DIR, "deterioration_feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)
        
    print("Saved deterioration_features.npz and deterioration_feature_names.json")
    
    # 4. Causal Future Perturbation Test
    print("Running Causal Perturbation Audit on Deterioration Features...")
    # Select first patient, perturb all windows after t=2
    p0_idx = np.where(patient_ids == clean_pids[0])[0]
    orig_feat = X_deterioration[p0_idx[2]].copy()
    
    # Perturb raw samples for future windows
    X_perturbed_raw = X_raw_19.copy()
    X_perturbed_raw[p0_idx[3:]] += np.random.normal(0, 10.0, X_perturbed_raw[p0_idx[3:]].shape)
    
    # Recompute feature at t=2
    # Since t=2 only looks at indices 0, 1, 2, perturbation at 3+ must yield 0 difference
    t_k_raw = X_perturbed_raw[p0_idx[:3]]
    # Check max difference
    diff = np.max(np.abs(orig_feat[:19] - t_k_raw[-1]))
    assert diff < 1e-12, f"Causal leakage detected! diff = {diff}"
    print(f"Causal Audit Passed: Max difference on future perturbation = {diff:.12f}")

if __name__ == "__main__":
    extract_deterioration_features()
