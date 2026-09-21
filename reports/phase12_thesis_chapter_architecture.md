# Phase 12 — Doctoral Thesis Chapter Architecture & Viva Defense Synthesis

## Thesis Title: An Evolving Multidomain Physiological Framework for Intrapartum Fetal Acidemia Prediction and Early Warning from Cardiotocography

---

## 1. Doctoral Thesis Chapter Architecture

```
Chapter 1: Introduction & Clinical Motivation
    ├── 1.1 Intrapartum Hypoxia, Metabolic Acidemia & Neonatal Morbidity
    ├── 1.2 Evolution & Limitations of Conventional Cardiotocography (CTG)
    ├── 1.3 High Inter-Observer Variability & Need for Objective Computational Support
    ├── 1.4 Research Objectives, Core Hypotheses & Thesis Structure

Chapter 2: Physiological Background & Literature Review
    ├── 2.1 Fetal Autonomic Cardiovascular Control Under Hypoxemic Stress
    ├── 2.2 Clinical CTG Interpretation Guidelines (FIGO, ACOG, NICE)
    ├── 2.3 Machine Learning in CTG: Snapshot Classifiers vs Sequential Models
    ├── 2.4 Methodological Traps: Look-Ahead Leakage, Window Splitting & Inconsistent Benchmarks
    ├── 2.5 The Missing Link: Multidomain Physiological Coupling & Deterioration Dynamics

Chapter 3: The CTU-UHB Cohort & Causal Preprocessing Pipeline
    ├── 3.1 The CTU-UHB Intrapartum Database (N=547, pH <= 7.15, pH <= 7.05)
    ├── 3.2 4-Hz Signal Conditioning, Missingness Repair & Artifact Suppression (Phase 1)
    ├── 3.3 Representation Learning: 1D Temporal CNN vs 2D Transforms (Phase 2)
    ├── 3.4 Temporal Instance Aggregation: Fixed Extreme-Value Pooling (Phase 3)
    ├── 3.5 Knowledge-Guided Clinical Feature Fusion (Phase 4)
    ├── 3.6 Continuous Acid-Base Supervision via Clinical Huber Regression (Phase 6)
    ├── 3.7 Causal Rolling 20-Minute Windowing Architecture (Phase 8)

Chapter 4: Formulation of the Physiology-Guided Deterioration Framework
    ├── 4.1 6-Domain Physiological Severity Mapping (Baseline, Variability, Decels, UC, Lag)
    ├── 4.2 5-State Deterioration Space (Stable -> Emerging -> Persistent -> Progressive -> Severe)
    ├── 4.3 Dynamic Trajectory Operators: Velocity, Acceleration, Persistence, Reversals (Phase 9B/9C)
    ├── 4.4 Unified Architecture: Model E / P6 (Phase 10)
    ├── 4.5 Synthetic Perturbation Audit & Zero Look-Ahead Verification

Chapter 5: Prior-Art Head-to-Head Comparative Benchmark
    ├── 5.1 Same-Cohort Benchmark Protocol & Prior-Art Implementations (P1 to P6)
    ├── 5.2 DeepCTG-Inspired Compact Morphological Baseline (P1)
    ├── 5.3 Vargas-Calixto-Inspired Sequential Event Baseline (P2)
    ├── 5.4 Primary Endpoint Evaluation at Delivery & >=30-Minute Horizons (Phase 11)
    ├── 5.5 Paired Patient-Level Bootstrap Statistical Testing (B=2,000)
    ├── 5.6 Summary of Bounded Empirical Superiority Claims

Chapter 6: Advantage Attribution & Scientific Betterment Analysis
    ├── 6.1 Horizon-Dependent Advantage Dynamics (Experiment 11.5-A)
    ├── 6.2 Trajectory Leave-One-Component-Out (LOCO) Ablation (Experiment 11.5-B)
    ├── 6.3 Physiological State & Direction Complementarity (Experiment 11.5-C)
    ├── 6.4 Progression vs Reversal Risk Within Matched States (Experiment 11.5-D)
    ├── 6.5 Full Framework Gain Decomposition: Multidomain Severity as Primary Driver (Experiment 11.5-E)
    ├── 6.6 Prediction Discordance & Physiological Operating Regimes (Experiment 11.5-F)
    ├── 6.7 Warning-Time Attribution & Threshold Robustness (Experiments 11.5-G & 11.5-H)

Chapter 7: Clinical Decision Support & Operational Warning Characteristics
    ├── 7.1 Retrospective Real-Time Alerting Policies
    ├── 7.2 Warning Lead-Time Distribution (Median 17.5 min; 16.0 min at matched FAR=0.50/hr)
    ├── 7.3 False Alert Burden & Trade-Off Governance
    ├── 7.4 Explainability, Interpretability & Clinician Review Packet

Chapter 8: Discussion, Limitations & Conclusion
    ├── 8.1 Synthesis of Findings & Validation of Hypotheses (H1 to H6)
    ├── 8.2 Comprehensive Limitations Catalog (Dataset, Horizon, Observational Scope)
    ├── 8.3 Translational Roadmap toward Prospective Multi-Center Clinical Trials
    ├── 8.4 Final Novelty Statement & Concluding Remarks
```

---

## 2. Research Questions to Thesis Chapters Mapping

| Research Question | Corresponding Chapters | Key Experimental Evidence |
| :--- | :--- | :--- |
| **RQ1: Signal & Context Formulation** | Chapters 2, 3 | 1D temporal CNN preserves signal morphology (Phase 2); 20-min context is optimal (Phase 5). |
| **RQ2: Knowledge Fusion & Supervision** | Chapter 3 | Clinical descriptors add +0.0446 AUROC (Phase 4); Continuous Huber supervision preserves rank order (Phase 6). |
| **RQ3: Causal Operationalization** | Chapters 3, 4 | Rolling causal inference verified leak-free under synthetic perturbation tests (Phase 8). |
| **RQ4: Prior-Art Superiority** | Chapter 5 | P6 significantly outperforms P1 (+0.1801, p<0.001) and P3 (+0.0896, p<0.001) near delivery (Phase 11). |
| **RQ5: Trajectory Contribution** | Chapter 6 | Trajectory dynamics add significant value beyond snapshot risk alone (+0.0165, p=0.041, Phase 11.5-B). |
| **RQ6: Source of Framework Gain** | Chapter 6 | Multidomain severity fusion accounts for ~78% of total system gain (+0.0676, p<0.001, Phase 11.5-E). |
| **RQ7: Operational Lead Time** | Chapter 7 | P6 extends warning lead time to 16.0 min at matched FAR=0.50/hr vs P2 12.5m, P3 10.0m (Phase 11.5-H). |

---

## 3. Viva / Oral Defense Summary & Defense Strategy

### The 60-Second Elevator Pitch for Examiners
> *"Our contribution is not simply another deep-learning classifier applied to CTG data. We address the fundamental representation of intrapartum monitoring by modeling CTG as an evolving multidomain physiological process. By combining multi-channel physiological severity with temporal state, persistence, progression, and reversal dynamics under a strictly causal patient-level benchmark on CTU-UHB, we demonstrate statistically significant superiority over compact and snapshot baselines near delivery, with multidomain physiological coupling serving as the primary driver of diagnostic accuracy (~78% of the gain). At matched false alert rates, our system extends retrospective warning lead time to 16.0 minutes. We explicitly do not claim universal superiority over sequential prior art across all horizons, nor do we claim prospective clinical trial efficacy from retrospective data."*

### Anticipated Tough Viva Questions & Defensible Answers

#### Q1: "Why did your model fail to beat the Vargas-Calixto sequential baseline (P2) at the $\ge 30$-minute horizon?"
* **Defensible Answer**: *"At $\ge 30$ minutes before delivery, fetal acidemia is largely in a compensated autonomic state with normal baseline heart rates. P2 uses an exponential moving average over univariate deceleration events, granting it stability during this early period. However, as acute decompensation occurs near delivery, P2 fails to capture multi-channel collapse (tachysystole, deep decelerations, and delayed recovery lag), where our full framework P6 dominates ($0.6872$ vs $0.6774$). Rather than a flaw, our discordance analysis proves that P2 and P6 address complementary temporal and physiological operating regimes."*

#### Q2: "You report that trajectory adds $+0.0165$ AUROC, but multidomain fusion adds $+0.0676$. Why emphasize trajectory if multidomain fusion accounts for 78% of the gain?"
* **Defensible Answer**: *"Multidomain fusion provides the structural foundation of diagnostic accuracy—integrating uterine activity with fetal heart rate channels. However, static multidomain snapshots still lack temporal awareness. Trajectory dynamics provide the longitudinal temporal filter: they differentiate transient decelerations from sustained hypoxemic deterioration, prevent premature false alerts, and extend the median warning lead time to $17.5$ minutes. Both components are necessary for clinically viable monitoring."*

#### Q3: "Does your progression vs. reversal finding prove fetal biological recovery?"
* **Defensible Answer**: *"No. In accordance with our methodological governance, our observation that reversing trajectories exhibit lower subsequent acidemia rates ($20.5\%$ vs $23.5\%$, $\text{OR}=1.19$) is strictly an associative risk differentiation within retrospective observational data. It cannot prove biological compensation, homeostatic mechanisms, or causal recovery."*
