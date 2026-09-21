# Phase 12 — Formal Novelty Claim & Methodological Boundaries

## CTU-UHB Fetal Acidemia Prediction & Early-Warning Project

### 1. Executive Novelty Formulation

The central novelty of this thesis is **not** simply "applying another deep-learning architecture to fetal distress prediction."

The core scientific contribution is formulated as follows:

$$\boxed{
\begin{gathered}
\textbf{Core Thesis Novelty Statement} \\
\text{"This work develops and validates a causally constrained, physiology-guided framework for representing} \\
\text{intrapartum cardiotocography as an evolving multidomain physiological process, integrating instantaneous} \\
\text{physiological severity with explicit state, persistence, progression, and reversal dynamics for patient-level} \\
\text{fetal acidemia early warning under a common, leak-free evaluation protocol."}
\end{gathered}
}$$

```
Figure Reference: reports/figures_phase12/fig1_overall_framework_architecture.png
```

---

## 2. The Four Contribution Layers

The contribution is structured into four integrated, mutually supportive layers:

### Layer 1: Causal Evaluation Architecture & Leak-Free Horizon Auditing
* **Problem**: Prior machine-learning literature in CTG frequently suffers from subtle retrospective look-ahead leakage (e.g., using post-delivery trace segments, whole-trace summary statistics, or non-causal filtering).
* **Novelty**: Implementation of a strictly causal rolling-window evaluation protocol ($20$-minute observation window, $2.5$-minute step size, $11.4$ ms inference latency) where $T_{\text{window}} < T_{\text{delivery}}$ is strictly enforced. Synthetic perturbation testing mathematically confirmed zero lookahead across all 19 engineered features and 6 domain severities.

### Layer 2: Multidomain Physiological Severity Representation
* **Problem**: Black-box neural networks or compact feature vectors conflate distinct physiological stress channels into uninterpretable embeddings.
* **Novelty**: Explicit multidomain representation encoding 6 physiological channels:
  1. Baseline fetal heart rate and slope
  2. Autonomic variability depression (STV, LTV)
  3. Acceleration capacity loss
  4. Deceleration burden, depth, and duration
  5. Uterine contraction frequency and tachysystole hyperstimulation
  6. FHR-UC coupling delay and autonomic recovery lag.
* **Empirical Validation**: Under the Phase 11.5 component ladder, this layer is the **principal driver of system performance**, accounting for $\approx 78\%$ of the total framework gain ($\Delta \text{AUROC} = +0.0676, p < 0.001$).

### Layer 3: Physiological State & Trajectory Dynamics
* **Problem**: Snapshot models treat each time window in isolation, failing to distinguish transient decelerations from persistent hypoxemic deterioration.
* **Novelty**: Formulation of a 5-state physiological space (Stable $\to$ Emerging $\to$ Persistent $\to$ Progressive $\to$ Severe) coupled with dynamic trajectory operators (1-step and 4-step velocity, progression acceleration, state persistence duration, direction reversal indicator, and multidomain concurrence).
* **Empirical Validation**: Demonstrates statistically significant incremental discrimination over snapshot risk alone ($\Delta \text{AUROC}_{\text{P5}-\text{P3}} = +0.0165, p = 0.041$).

### Layer 4: Rigorous Same-Cohort Benchmark Against Representative Prior-Art Approaches
* **Problem**: Published CTG models report disparate AUROCs evaluated across inconsistent cohorts, endpoints, and filtering protocols.
* **Novelty**: First direct, head-to-head, same-cohort comparative validation evaluating the proposed multidomain trajectory framework against faithful implementations of compact morphological prior art (DeepCTG-inspired P1), sequential event prior art (Vargas-Calixto-inspired P2), and locked snapshot Huber regression (P3) under identical $B=2,000$ patient-level bootstrap conditions.

---

## 3. Explicit Boundaries: What Is Novel vs. What Is NOT Claimed as Novel

To maintain complete publication integrity, the thesis explicitly delineates its novelty boundaries:

| Methodological Concept | Claimed as Individually Novel? | Thesis Position & Scope |
| :--- | :---: | :--- |
| **Predicting Fetal Acidemia from CTG** | **NO** | Standard intrapartum clinical objective explored in obstetrics for decades. |
| **20-Minute CTG Windows** | **NO** | Established FIGO clinical guideline standard for basal CTG assessment. |
| **Rolling / Sliding Window Inference** | **NO** | Standard sequential signal processing technique. |
| **FIGO Morphological Descriptors** | **NO** | Classical clinical feature definitions (STV, LTV, decelerations, contractions). |
| **Moving Average Persistence Rules** | **NO** | Classical temporal filtering technique (evaluated in P2 baseline). |
| **Multi-Domain Physiological Severity Integration** | **YES** | Novel structured formulation mapping 6 clinical channels into continuous severity scores. |
| **5-State Deterioration Space with Dynamic Trajectory Operators** | **YES** | Novel integration of velocity, acceleration, persistence, and reversals on CTG state space. |
| **Unified Multidomain + State + Trajectory Framework (P6)** | **YES** | Novel hierarchical combination achieving validated empirical gains. |
| **Attribution of Performance Gains to Multidomain Coupling vs Trajectory** | **YES** | First systematic ladder decomposition isolating the ~78% multidomain vs ~22% trajectory split. |

---

## 4. Final Novelty Summary

The scientific novelty resides in the **principled integration, physiological framing, causal operationalization, and rigorous component-level attribution** demonstrating that intrapartum CTG is best modeled as an evolving multidomain physiological process.
