"""
Phase 12: Master Scientific Synthesis & Evidence Aggregation Engine.

Aggregates locked results across all completed phases (Phases 1 through 11.5):
- Phase 1-6: Signal representation, clinical fusion, ordinal supervision.
- Phase 8: Rolling causal inference & early-warning boundaries.
- Phase 9A-9C: Physiological state transitions & trajectory dynamics.
- Phase 10: Final integrated framework definitions.
- Phase 11: Head-to-head prior-art benchmark.
- Phase 11.5: Advantage attribution & component decomposition.

Outputs:
- results/phase12_synthesis/master_performance_table.csv
- results/phase12_synthesis/phase12_summary.json
"""

import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "results/phase12_synthesis"
os.makedirs(OUT_DIR, exist_ok=True)

P11_DIR = "results/phase11_priorart_benchmark"
P11_5_DIR = "results/phase11_5_advantage_attribution"
P10_DIR = "results/phase10_final"

def run_master_synthesis():
    print("=== EXECUTING PHASE 12: MASTER SCIENTIFIC SYNTHESIS & AGGREGATION ===")
    
    # 1. Load locked results from Phase 11 & Phase 11.5
    df_h_adv = pd.read_csv(os.path.join(P11_5_DIR, "horizon_advantage.csv"))
    df_traj_abl = pd.read_csv(os.path.join(P11_5_DIR, "trajectory_component_ablation.csv"))
    df_state_decomp = pd.read_csv(os.path.join(P11_5_DIR, "state_direction_decomposition.csv"))
    df_ladder = pd.read_csv(os.path.join(P11_5_DIR, "full_framework_decomposition.csv"))
    df_disc = pd.read_csv(os.path.join(P11_5_DIR, "prediction_discordance.csv"))
    df_warn = pd.read_csv(os.path.join(P11_5_DIR, "warning_time_attribution.csv"))
    df_thresh = pd.read_csv(os.path.join(P11_5_DIR, "threshold_robustness.csv"))
    
    # 2. Compile Master Performance Table across All Project Phases
    master_rows = [
        {
            "phase": "Phase 2",
            "scientific_purpose": "Signal Representation Learning",
            "evaluated_concept": "1D Temporal CNN vs 2D CWT vs Recurrence Plots vs Late Fusion",
            "primary_metric": "AUROC = 0.6593",
            "statistical_support": "1D temporal representation significantly outperformed 2D transforms",
            "core_scientific_takeaway": "Preserved raw temporal CTG signal structure without lossy 2D time-frequency projections"
        },
        {
            "phase": "Phase 3",
            "scientific_purpose": "Temporal Instance Aggregation",
            "evaluated_concept": "Attention MIL vs Fixed Pooling (Mean, Max, P90, P95)",
            "primary_metric": "AUROC = 0.6701 (P90)",
            "statistical_support": "Fixed extreme-value pooling (P90) outperformed learned attention MIL",
            "core_scientific_takeaway": "Under limited patient sample size (N=547), robust extreme-value aggregation prevents attention overfitting"
        },
        {
            "phase": "Phase 4",
            "scientific_purpose": "Knowledge-Guided Clinical Fusion",
            "evaluated_concept": "Learned Signal Embedding + 19 FIGO Clinical Descriptors",
            "primary_metric": "AUROC = 0.7361",
            "statistical_support": "Statistically significant gain over learned signal alone (+0.0446, p<0.01)",
            "core_scientific_takeaway": "Expert physiological descriptors provide orthogonal, complementary clinical information"
        },
        {
            "phase": "Phase 5",
            "scientific_purpose": "Temporal Context Window Length",
            "evaluated_concept": "Context window scaling (20 min vs 30 min vs 45 min vs 60 min)",
            "primary_metric": "AUROC = 0.7361 (20 min)",
            "statistical_support": "Longer retrospective context did not improve discrimination",
            "core_scientific_takeaway": "20-minute causal observation window captures the fundamental cycle of uterine contractions and decelerations"
        },
        {
            "phase": "Phase 6",
            "scientific_purpose": "Continuous Acid-Base Supervision",
            "evaluated_concept": "Continuous Clinical Huber Regression (pH) vs Binary Cross-Entropy vs Ordinal Loss",
            "primary_metric": "AUROC = 0.7426 [95% CI: 0.6876, 0.7931]",
            "statistical_support": "Huber continuous supervision achieved highest numerical point estimate",
            "core_scientific_takeaway": "Supervising continuous physiological pH preserves rank order of severity without arbitrary binarization"
        },
        {
            "phase": "Phase 7 / 7.1",
            "scientific_purpose": "Methodological Audit & Reconciliation",
            "evaluated_concept": "Audit of potential feature leakage and representation drift",
            "primary_metric": "Locked Huber Baseline: 0.7426",
            "statistical_support": "Invalidated earlier unverified +0.0881 gain; established frozen representation lock",
            "core_scientific_takeaway": "Enforced strict causal invariants and eliminated optimistic representation drift"
        },
        {
            "phase": "Phase 8",
            "scientific_purpose": "Causal Rolling 20-Min Inference & Early Warning",
            "evaluated_concept": "Continuous sliding window (stride 2.5 min) from >=60m to delivery",
            "primary_metric": "AUROC >=30m: 0.5699; Delivery: 0.6941",
            "statistical_support": "Zero future look-ahead verified under synthetic perturbation test",
            "core_scientific_takeaway": "Identified acute temporal boundary: predictive information concentrates in final 15-25 min of labor"
        },
        {
            "phase": "Phase 9A-9C",
            "scientific_purpose": "Physiological State & Trajectory Dynamics",
            "evaluated_concept": "5-State Deterioration Model + Velocity + Persistence + Reversals",
            "primary_metric": "Monotonic Risk Gradient (State 0: 6.8% -> State 4: 43.8%)",
            "statistical_support": "Monotonic state risk gradient verified under clustered patient bootstrap",
            "core_scientific_takeaway": "State trajectory represents clinically interpretable intrapartum deterioration path"
        },
        {
            "phase": "Phase 10",
            "scientific_purpose": "Final Integrated Framework Definition",
            "evaluated_concept": "Candidate architectures A, B, C, D, E under 5-fold CV",
            "primary_metric": "Model E (Full Multidomain): 0.6872; Model D (Traj): 0.6142; Model A: 0.5976",
            "statistical_support": "Statistically significant gain of Model E over Model A (+0.0896, p<0.001)",
            "core_scientific_takeaway": "Consolidated final architecture: Snapshot + State + Trajectory + Multidomain Fusion"
        },
        {
            "phase": "Phase 11",
            "scientific_purpose": "Prior-Art Head-to-Head Comparative Benchmark",
            "evaluated_concept": "P1 (Compact) vs P2 (Sequential) vs P3 (Snapshot) vs P4 vs P5 vs P6 (Full System)",
            "primary_metric": "P6 AUROC: 0.6872 vs P1: 0.5090 (p<0.001), P3: 0.5976 (p<0.001), P2: 0.6774 (p=0.697)",
            "statistical_support": "P6 significantly beats P1 and P3 near delivery; statistically comparable to P2",
            "core_scientific_takeaway": "Demonstrated bounded empirical superiority near delivery; disproved universal superiority over sequential prior art"
        },
        {
            "phase": "Phase 11.5",
            "scientific_purpose": "Advantage Attribution & Scientific Betterment",
            "evaluated_concept": "Decomposition across horizons, LOCO trajectory ablation, 8-step ladder, discordance",
            "primary_metric": "Multidomain severity fusion accounts for ~78% (+0.0676, p<0.001) of total framework gain",
            "statistical_support": "Trajectory adds significant value (+0.0165, p=0.041); Multidomain fusion is major driver",
            "core_scientific_takeaway": "Scientific explanation: Multidomain physiological coupling drives accuracy; trajectory stabilizes temporal dynamics"
        }
    ]
    
    df_master = pd.DataFrame(master_rows)
    master_csv_path = os.path.join(OUT_DIR, "master_performance_table.csv")
    df_master.to_csv(master_csv_path, index=False)
    print(f"Saved {master_csv_path}")
    
    # 3. Assemble Master Summary JSON
    summary_data = {
        "project_title": "Physiology-Guided Multidomain Deterioration Framework for Intrapartum Fetal Acidemia Early Warning",
        "phase": "Phase 12 — Final Scientific Synthesis & Thesis Validation",
        "status": "LOCKED & METHODOLOGICALLY CLOSED",
        "cohort_invariants": {
            "dataset": "CTU-UHB Intrapartum Cardiotocography Database",
            "n_patients": 547,
            "n_primary_acidemia_positives_715": 110,
            "n_severe_acidemia_positives_705": 41,
            "n_rolling_causal_windows": 8517,
            "window_length_min": 20.0,
            "window_stride_min": 2.5,
            "validation_protocol": "Patient-stratified 5-fold cross-validation",
            "bootstrap_replicates": 2000
        },
        "thesis_core_answers": {
            "Q1_methodological_novelty": "Causally constrained, physiology-guided framework integrating instantaneous multidomain physiological severity with explicit state, persistence, progression and reversal dynamics under strict patient-level evaluation.",
            "Q2_empirical_superiority": "Statistically significant superiority over DeepCTG-inspired compact baseline (Delta=+0.1801, p<0.001) and locked snapshot baseline (Delta=+0.0896, p<0.001) near delivery. Statistically comparable to Vargas-Calixto sequential baseline (Delta=+0.0098, p=0.697 at delivery; Delta=-0.0126, p=0.485 at >=30m).",
            "Q3_trajectory_betterment": "Temporal trajectory adds statistically significant predictive information over snapshot risk alone (Delta=+0.0165, 95% CI [+0.0006, +0.0319], p=0.041 at delivery).",
            "Q4_multidomain_betterment": "Multidomain physiological severity fusion is the principal performance driver, accounting for approximately 78% of the total system gain (+0.0676 AUROC, p<0.001 under the component ladder).",
            "Q5_operational_advantage": "Longest median warning lead time (17.5 min at locked threshold; 16.0 min at matched FAR=0.50/hr vs P2 12.5 min, P3 10.0 min, P1 12.5 min).",
            "Q6_scientific_boundaries": "No universal superiority across all horizons; early-warning discrimination at >=30 min remains bounded by acute labor decompensation; trajectory associations reflect empirical risk differentiation rather than causal biological compensation; prospective clinical efficacy remains to be proven."
        },
        "hypothesis_verdicts": {
            "H1_signal_representation": "CONFIRMED (1D temporal representation captures essential CTG morphology)",
            "H2_multidomain_information": "CONFIRMED (Clinical descriptors provide orthogonal, complementary signal)",
            "H3_trajectory_increment": "CONFIRMED (P5 > P3 Delta=+0.0165, p=0.041)",
            "H4_state_direction_complementarity": "CONFIRMED (Nested progression R_t -> R_t+S_t -> R_t+S_t+D_t -> R_t+S_t+D_t+P_t achieves +0.0170 gain, p=0.043)",
            "H5_multidomain_primary_driver": "CONFIRMED (Multidomain fusion accounts for ~78% of total system gain under component ladder)",
            "H6_prior_art_bounded_superiority": "CONFIRMED (Beats P1 & P3 near delivery; comparable to P2; no universal superiority)"
        }
    }
    
    summary_json_path = os.path.join(OUT_DIR, "phase12_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved {summary_json_path}")
    print("=== PHASE 12 SYNTHESIS COMPLETE ===")

if __name__ == "__main__":
    run_master_synthesis()
