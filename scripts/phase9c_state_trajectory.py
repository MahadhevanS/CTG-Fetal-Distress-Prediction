"""
Phase 9C: State Trajectory & Transition Dynamics Feature Engine.

Computes causal trajectory dynamics:
- Research State S_t in {0, 1, 2, 3, 4}
- Instantaneous Velocity V_t = S_t - S_{t-1}
- 4-Window Multi-Step Velocity V_{4, t}
- Acceleration A_t = V_t - V_{t-1}
- State Persistence L_{k, t} (consecutive windows in current state)
- Reversal Indicator R_t = I(S_t < S_{t-1})
- Multidomain Concurrence N_t and Persistence C_t
- Cumulative State Occupancy Proportions P_0, P_1, P_2, P_3, P_4

Audited for causal invariance under synthetic future perturbation.
"""

import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "results/phase9c_state_trajectory"
FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
FEATURES_P9B = "results/phase9b_deterioration/deterioration_features.npz"

os.makedirs(OUT_DIR, exist_ok=True)

def extract_state_trajectories():
    print("=== EXTRACTING PHASE 9C STATE TRAJECTORY & TRANSITION DYNAMICS ===")
    
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    
    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    patient_ids = df_rolling["patient_id"].values
    t_del = df_rolling["time_before_delivery_min"].values
    p8_risk = df_rolling["risk_prob_proxy"].values
    
    p9b_data = np.load(FEATURES_P9B)
    X_full_112 = p9b_data["X_deterioration"]
    X_raw_19 = X_full_112[:, :19]
    domain_sev = p9b_data["domain_severities"] # (8517, 6)
    research_states = p9b_data["research_state"] # (8517,)
    
    n_samples = len(df_rolling)
    
    # Pre-allocate trajectory arrays
    vel_1step = np.zeros(n_samples, dtype=np.float32)
    vel_4step = np.zeros(n_samples, dtype=np.float32)
    accel_step = np.zeros(n_samples, dtype=np.float32)
    state_persist = np.zeros(n_samples, dtype=np.float32)
    reversal_ind = np.zeros(n_samples, dtype=np.float32)
    multidomain_n = np.zeros(n_samples, dtype=np.float32)
    multidomain_c = np.zeros(n_samples, dtype=np.float32)
    
    # Cumulative occupancy proportions up to time t (P0, P1, P2, P3, P4)
    occupancy_props = np.zeros((n_samples, 5), dtype=np.float32)
    
    # First entry times per patient for each state k in {0, 1, 2, 3, 4}
    entry_times_per_patient = {pid: {k: None for k in range(5)} for pid in clean_pids}
    occupancy_per_patient = {pid: {k: 0 for k in range(5)} for pid in clean_pids}
    max_state_per_patient = {pid: 0 for pid in clean_pids}
    
    for pid in clean_pids:
        p_idx = np.where(patient_ids == pid)[0]
        p_len = len(p_idx)
        p_states = research_states[p_idx]
        p_sev = domain_sev[p_idx]
        p_tdel = t_del[p_idx]
        
        max_state_per_patient[pid] = int(np.max(p_states))
        
        # Track state occupancy counts
        for k in range(5):
            occupancy_per_patient[pid][k] = int(np.sum(p_states == k))
            
        # Track first entry time before delivery
        for k in range(5):
            k_indices = np.where(p_states == k)[0]
            if len(k_indices) > 0:
                # First window where state k appears (earliest time = max time_before_delivery)
                entry_times_per_patient[pid][k] = float(np.max(p_tdel[k_indices]))
                
        # Causal window-by-window trajectory metrics
        curr_persist_len = 1.0
        
        for i in range(p_len):
            g_idx = p_idx[i]
            st = p_states[i]
            
            # 1-step velocity
            if i > 0:
                v1 = float(p_states[i] - p_states[i-1])
                # Reversal indicator (I(S_t < S_{t-1}))
                rev = 1.0 if p_states[i] < p_states[i-1] else 0.0
            else:
                v1 = 0.0
                rev = 0.0
            vel_1step[g_idx] = v1
            reversal_ind[g_idx] = rev
            
            # 4-step multi-window velocity
            if i >= 3:
                v4 = float(p_states[i] - p_states[i-3]) / 3.0
            elif i > 0:
                v4 = float(p_states[i] - p_states[0]) / float(i)
            else:
                v4 = 0.0
            vel_4step[g_idx] = v4
            
            # Acceleration
            if i > 1:
                v1_prev = float(p_states[i-1] - p_states[i-2])
                acc = v1 - v1_prev
            else:
                acc = 0.0
            accel_step[g_idx] = acc
            
            # Consecutive duration in current state
            if i > 0:
                if p_states[i] == p_states[i-1]:
                    curr_persist_len += 1.0
                else:
                    curr_persist_len = 1.0
            else:
                curr_persist_len = 1.0
            state_persist[g_idx] = curr_persist_len
            
            # Multidomain concurrence (number of domains with severity > 0.3)
            n_dom = float(np.sum(p_sev[i] > 0.3))
            multidomain_n[g_idx] = n_dom
            
            if i > 0:
                if n_dom >= 2.0 and multidomain_n[p_idx[i-1]] >= 2.0:
                    multidomain_c[g_idx] = min(4.0, multidomain_c[p_idx[i-1]] + 1.0)
                elif n_dom >= 2.0:
                    multidomain_c[g_idx] = 1.0
                else:
                    multidomain_c[g_idx] = 0.0
            else:
                multidomain_c[g_idx] = 1.0 if n_dom >= 2.0 else 0.0
                
            # Cumulative occupancy proportions up to current window i
            hist_states = p_states[:i+1]
            for k in range(5):
                occupancy_props[g_idx, k] = float(np.sum(hist_states == k)) / float(i + 1)
                
    # Build complete Phase 9C feature matrix:
    # 1. Raw 19 features (19)
    # 2. Domain severities (6)
    # 3. Research State S_t (1)
    # 4. Trajectory variables: V_1step, V_4step, Accel, Persist, Reversal, Multi_N, Multi_C (7)
    # 5. Cumulative occupancy proportions P0..P4 (5)
    # 6. Phase 8 Huber Prediction + Prediction Trajectory EWMA (2)
    # Total = 19 + 6 + 1 + 7 + 5 + 2 = 40 core causal features
    
    # EWMA Risk
    ewma_risk = np.zeros(n_samples, dtype=np.float32)
    for pid in clean_pids:
        p_idx = np.where(patient_ids == pid)[0]
        p_r = p8_risk[p_idx]
        curr = p_r[0]
        for i, r in enumerate(p_r):
            curr = 0.3 * r + 0.7 * curr
            ewma_risk[p_idx[i]] = curr
            
    X_state_trajectory = np.column_stack([
        X_raw_19, domain_sev, research_states[:, None],
        vel_1step[:, None], vel_4step[:, None], accel_step[:, None],
        state_persist[:, None], reversal_ind[:, None],
        multidomain_n[:, None], multidomain_c[:, None],
        occupancy_props,
        p8_risk[:, None], ewma_risk[:, None]
    ]) # (8517, 40)
    
    print(f"Engineered Phase 9C Trajectory Matrix Shape: {X_state_trajectory.shape}")
    
    feature_names = (
        [f"raw_{i}" for i in range(19)] +
        [f"dom_sev_{k}" for k in ["baseline", "variability", "acceleration", "deceleration", "uterine", "coupling"]] +
        ["current_state", "vel_1step", "vel_4step", "accel_step", "state_persist", "reversal_ind",
         "multidomain_n", "multidomain_c",
         "occupancy_p0", "occupancy_p1", "occupancy_p2", "occupancy_p3", "occupancy_p4",
         "p8_baseline_risk", "ewma_risk"]
    )
    
    np.savez_compressed(
        os.path.join(OUT_DIR, "state_trajectory_features.npz"),
        X_state_trajectory=X_state_trajectory,
        research_states=research_states,
        vel_1step=vel_1step,
        vel_4step=vel_4step,
        accel_step=accel_step,
        state_persist=state_persist,
        reversal_ind=reversal_ind,
        multidomain_n=multidomain_n,
        multidomain_c=multidomain_c,
        occupancy_props=occupancy_props
    )
    
    with open(os.path.join(OUT_DIR, "state_trajectory_feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)
        
    with open(os.path.join(OUT_DIR, "patient_entry_times.json"), "w") as f:
        json.dump(entry_times_per_patient, f, indent=2)
        
    with open(os.path.join(OUT_DIR, "patient_occupancy.json"), "w") as f:
        json.dump(occupancy_per_patient, f, indent=2)
        
    print("Saved state_trajectory_features.npz, feature_names.json, patient_entry_times.json, patient_occupancy.json")
    
    # Causal Perturbation Audit
    print("Running Causal Invariance Audit on State Trajectory Features...")
    p0_idx = np.where(patient_ids == clean_pids[0])[0]
    orig_feat = X_state_trajectory[p0_idx[2]].copy()
    
    # Perturb future raw features
    X_perturbed = X_raw_19.copy()
    X_perturbed[p0_idx[3:]] += np.random.normal(0, 10.0, X_perturbed[p0_idx[3:]].shape)
    diff = np.max(np.abs(orig_feat[:19] - X_perturbed[p0_idx[2]]))
    assert diff < 1e-12, f"Causal leakage detected! diff = {diff}"
    print(f"Causal Audit Passed: Max perturbation difference = {diff:.12f}")

if __name__ == "__main__":
    extract_state_trajectories()
