# Phase 10 — Final Synthesis & Master Project Lock Report

## CTU-UHB Fetal Acidemia Prediction & Early-Warning Project

---

## 1. Executive Summary

This report delivers the final scientific synthesis, clinical decision analysis, and master project lock for the **CTU-UHB Fetal Acidemia Prediction & Early-Warning Program**. Over ten systematic research phases, the project investigated the predictive boundaries of intrapartum cardiotocography (CTG), progressing from raw representation learning to patient aggregation, clinical knowledge fusion, continuous acid-base supervision, causal rolling-window inference, and temporal physiological deterioration modeling.

The research program concludes with a definitive, defensible, and locked scientific finding:

$$\boxed{
\begin{gathered}
\textbf{Final Scientific Finding:} \\
\text{Representing intrapartum CTG as an evolving physiological deterioration trajectory} \\
\text{provides statistically significant incremental value near delivery } (\Delta\text{AUROC} = +0.02 \text{ to } +0.09, p < 0.05) \\
\text{and enables persistence-aware clinical alerting } (51.5\%\text{--}72.3\%\text{ Spec, } 0.48\text{--}1.05\text{ FAR/hr}), \\
\text{while the intrinsic biological compensation of the mammalian fetus restricts reliable} \\
\text{morphological CTG early warning at } \ge 30\text{ minutes before delivery to } \text{AUROC} \approx 0.55\text{--}0.59.
\end{gathered}
}$$

---

## 2. Definitive Research Hierarchy & Master Progression

```text
Phase 2: Raw CTG Signal Representation
  │  • Finding: Raw temporal models (CNN1D/TCN) reach AUROC = 0.6593.
  ▼
Phase 3: Patient-Level Temporal Aggregation
  │  • Finding: Extreme-value pooling (P90) achieves AUROC = 0.6701; superior to learned MIL.
  ▼
Phase 4: Clinical Knowledge Infusion
  │  • Finding: Infusion of 19 FIGO morphological features achieves AUROC = 0.7361 (+0.0660 gain).
  ▼
Phase 6: Continuous Acid-Base Supervision
  │  • Finding: Continuous Huber regression on umbilical pH achieves locked AUROC = 0.7426.
  ▼
Phase 7 & 7.1: Locked Validation & Replication Audit
  │  • Finding: Baseline reconciled and proven reproducible across 5 random seeds (p < 0.001).
  ▼
Phase 8: Causal Rolling 20-Min Early Warning
  │  • Finding: Causal real-time feasibility established (11.4 ms latency, AUROC = 0.7426 at delivery;
  │            AUROC collapses to 0.5461 at ≥30 min).
  ▼
Phase 9A: Generic Temporal & Sequential Benchmarking
  │  • Finding: Generic temporal models (multi-horizon RNN/GBM) fail to solve ≥30-min horizon.
  ▼
Phase 9B: Physiology-Guided Deterioration States
  │  • Finding: Multi-domain severity reveals strong monotonic risk gradient (8.6% → 47.6%).
  ▼
Phase 9C: State Transitions, Trajectory Dynamics & Timing
  │  • Finding: Trajectory history differentiates acute distress (e.g. State 3 progressing vs reversing).
  │            Identifies the 40-minute theoretical dataset lead boundary in 60-min recordings.
  ▼
Phase 10: Final Clinical Decision & Contribution Analysis
  │  • Finding: Trajectory incremental value confirmed at delivery (p = 0.026).
  │            Hybrid alerting achieves 10.0m median lead time with 1.05 FAR/hr.
  ▼
FINAL THESIS & REPOSITORY LOCK
```

---

## 3. Master Evidence Summary (Work Packages 10A–10F)

### 3.1 Incremental Value of Trajectory ($B=2,000$ Clustered Patient Bootstrap)
* **At Delivery ($0$m)**:
  * Model C (Snapshot + State + Trajectory) vs Snapshot Baseline: $\mathbf{\Delta \text{AUROC} = +0.0204}$ ($95\%\text{ CI}: [+0.0031, +0.0378], \mathbf{p = 0.026}$).
  * Model D (Snapshot + Trajectory) vs Snapshot Baseline: $\mathbf{\Delta \text{AUROC} = +0.0166}$ ($95\%\text{ CI}: [+0.0012, +0.0320], \mathbf{p = 0.034}$).
  * Model E (Full Physiological Fusion) vs Snapshot Baseline: $\mathbf{\Delta \text{AUROC} = +0.0900}$ ($95\%\text{ CI}: [+0.0351, +0.1492], \mathbf{p = 0.002}$).
* **At Primary Early-Warning Horizon ($\ge 30$ min)**:
  * Model E vs Baseline: $\mathbf{\Delta \text{AUROC} = +0.0392}$ ($95\%\text{ CI}: [-0.0133, +0.0910], p = 0.135$).

### 3.2 Predefined Clinical Alert Policies
* **Policy 2 (Two Consecutive Windows $p \ge \tau$)**:
  * Sensitivity: **$55.45\%$**, Specificity: **$72.31\%$**, False Alert Rate: **$0.476$ false alerts/hr**, Median Lead Time: **$7.5$ min**.
* **Policy 5 (Hybrid Decision Policy: $p \ge 0.85\tau + S \ge 2 + P \ge 2$)**:
  * Sensitivity: **$75.45\%$**, Specificity: **$51.49\%$**, False Alert Rate: **$1.046$ false alerts/hr**, Median Lead Time: **$10.0$ min**.

---

## 4. Final Scientific Contribution

The core scientific contribution of this thesis is:

> **A causally constrained, physiology-guided framework for modelling intrapartum CTG as evolving multidomain physiological states, with explicit representation of persistence, progression and reversal, and rigorous patient-level evaluation of their relationship with subsequent fetal acidemia.**

### What the Project Does NOT Claim:
1. The project does **not** claim to achieve an $\text{AUROC} \ge 0.80$ at $\ge 30$ minutes prior to delivery from CTG morphology alone.
2. The project does **not** claim that physiological state transitions prove direct biological causality.
3. The project does **not** claim to evaluate horizons $>40.0$ minutes on CTU-UHB due to the 60-minute duration constraint.

---

## 5. Master Deliverables Directory

### 5.1 Final Code Suite (`scripts/`)
* `phase10_final_system.py` — WP 10A Final System Definition.
* `phase10_incremental_analysis.py` — WP 10B Incremental Contribution Evaluator.
* `phase10_clinical_operating.py` — WP 10C Clinical Operating Point & Lead Time Engine.
* `phase10_horizon_analysis.py` — WP 10D Prediction-Horizon Boundary Analyzer.
* `phase10_final_figures.py` — WP 10E Evidence Package & Publication Figures.
* `phase10_reproducibility.py` — WP 10F Master Reproducibility Pipeline.

### 5.2 Results Data Package (`results/phase10_final/`)
* `final_predictions.csv` (8,517 rolling window predictions).
* `final_model_comparison.csv` (5 candidate models across 6 warning horizons).
* `bootstrap_results.csv` ($B=2,000$ paired patient bootstrap statistics).
* `incremental_information.csv` (Incremental $\Delta \text{AUROC}$ and evidence levels).
* `clinical_operating_points.csv` (Operating points for all 5 alerting policies).
* `warning_time_analysis.csv` (Lead-time distribution metrics).
* `horizon_analysis.csv` & `horizon_boundary_summary.json` (Horizon boundary analysis).
* `table1_experimental_progression.csv` (Master progression table).
* `phase10_summary.json` (Consolidated execution summary).

### 5.3 Publication Figures (`reports/figures_phase10/`)
1. `fig1_research_pipeline.png` — Final causal early-warning system architecture.
2. `fig2_phase_progression.png` — Methodological evolution across Phases 2 through 10.
3. `fig3_horizon_auroc.png` — Multi-horizon discrimination trajectory approaching delivery.
4. `fig4_state_risk_gradient.png` — Physiological state-risk gradient.
5. `fig5_state_trajectory.png` — Progression vs reversal trajectory dynamics.
6. `fig6_alert_tradeoff.png` — Sensitivity vs false-alarm burden trade-off.

---

## 6. Final Disposition & Thesis Declaration

### Status: **LOCKED (Phase 10 & Thesis Lock Complete)**
* **Repository Branch**: [`final_synthesis_models`](https://github.com/MahadhevanS/CTG-Fetal-Distress-Prediction/tree/final_synthesis_models)
* **Integrity**: Causal invariance verified ($\max |\Delta X| = 0.0$), patient-level independence preserved, all 15 audit gates confirmed, complete evidence package generated and locked.

# End of Phase 10 Master Report
