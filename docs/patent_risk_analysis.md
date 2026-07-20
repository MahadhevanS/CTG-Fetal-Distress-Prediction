# Patent Risk & Technical Differentiation Analysis: US12094611B2

**Subject Patent**: US12094611B2 — "Deep Learning Based Fetal Heart Rate Analytics"  
**Assignee**: GE Precision Healthcare LLC  
**Priority Date**: November 2, 2021  
**Granted**: September 17, 2024  

> [!WARNING]
> **Disclaimer**: This document is a technical risk assessment and architectural differentiation analysis. It does not constitute formal legal counsel or a binding legal opinion. For official freedom-to-operate (FTO) clearance, consult a certified patent attorney.

---

## 1. Executive Summary

The granted GE patent **US12094611B2** is a significant piece of prior art in automated cardiotocography (CTG) analysis. It claims a system that identifies Fetal Heart Rate (FHR) graphical patterns, correlates them longitudinally over time to confirm physiological events, and transmits derived parameter values to a clinician.

For our **academic/research project (Project Work I at PSG College of Technology)**:
* **Direct Legal Risk**: **Negligible**. Academic and research-oriented implementations fall under experimental/academic use exceptions in most jurisdictions, and non-commercial academic research rarely attracts patent litigation as there are no commercial damages.
* **Academic Novelty/Patentability Risk**: **Moderate**. While it does not restrict publishing research papers, it limits our ability to claim broad novelty for general "temporal modeling of CTG events."

For any **future commercialization**:
* **Infringement Risk**: **Moderate**. A structured differentiation strategy is required. We must highlight our multi-task loss formulation, direct biochemical outcome supervision (pH), and model explainability features, which are absent from GE's patent claims.

---

## 2. Claim Breakdown vs. Our Framework

A patent is infringed only if a system implements **every single element** of at least one independent claim (literal infringement). Below is a comparison of GE's claimed workflow against our framework's architecture:

| GE Patent Claim Element (US12094611B2) | Our Preprocessing & Model Architecture | Technical Differentiation Status |
| :--- | :--- | :--- |
| **1. Pattern Recognition**: A supervised model is trained on annotated CTG tracings to recognize graphical patterns (accelerations, decelerations, etc.). | We train a shared temporal encoder to represent raw signals, but we do **not** use manual graphical pattern bounding boxes or annotations for primary training. We use mathematical algorithms to extract targets. | **Differentiated**: We use standard signal processing (e.g., cubic splines, Butterworth filtering) for signal extraction and multi-task heads for learning representation. |
| **2. Longitudinal Correlation**: The model learns longitudinal correlations between patterns occurring in different time periods of the same recording. | We employ a temporal deep learning model (e.g., BiLSTM, TCN, PatchCTG) to process continuous 20-minute windows. We do **not** run cross-temporal matching algorithms to compare a pattern at hour 1 with a pattern at hour 2 to confirm occurrence. | **Differentiated**: Our temporal modeling is continuous and representation-based, rather than correlation-based confirmation of localized graphical shapes. |
| **3. Correlation-Based Event Confirmation**: The system uses correlation to a previously or subsequently identified pattern to confirm if a physiological event actually occurred. | Our features (decelerations, STV, LTV) are calculated mathematically or regressed. We do **not** have a secondary confirmation loop based on subsequent pattern correlations. | **Highly Differentiated**: The patent specifically relies on correlation-based verification of events. Our model classifies states directly. |
| **4. Derived Clinical Parameters**: Upon confirmation, the model generates derived parameter values and reports the event plus parameters in real time. | Our model outputs auxiliary continuous metrics (baseline, early/late/variable/prolonged decelerations) and maps them to standard FIGO classes via multi-task heads. | **Partially Overlapping**: Both systems report derived parameters, but our output is tied to multi-task loss and FIGO guidelines. |

---

## 3. Key Areas of Legal & Intellectual Property Defense

Should our project transition from research to commercial development, our defense against US12094611B2 rests on three pillars:

### Pillar 1: Supervised Target Disparity (Biochemical vs. Graphical)
* **GE Patent**: Focuses on recognizing *graphical patterns* corresponding to physiological events (e.g., visually identifying a deceleration shape in the waveform).
* **Our Framework**: Trains the model with a primary multi-task head targeting a terminal biochemical outcome—**umbilical artery pH <= 7.15 (Acidemia)**. The network learns to predict the metabolic state of the fetus at birth, which is not a graphical waveform pattern.

### Pillar 2: Clinical Guideline Infusion (FIGO/NICE)
* **GE Patent**: Does not claim or mention the integration of clinical rules, expert-guided loss constraints, or programmatic mapping to international frameworks (such as FIGO 2015).
* **Our Framework**: Utilizes a [classify_figo](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/src/knowledge/figo.py#L3) module to map predicted parameters against medical rules, computing a customized knowledge loss:
  $$L_{total} = L_{\text{distress}} + \lambda_1 L_{\text{clinical}} + \lambda_2 L_{\text{FIGO}} + \lambda_3 L_{\text{knowledge}}$$
  This loss function regularizes the network's training path, ensuring clinical consistency.

### Pillar 3: Post-Hoc Model Explainability (XAI)
* **GE Patent**: Claims a black-box model that alerts the clinician to an event and its parameters. It has no mechanism for showing how the model arrived at its conclusion.
* **Our Framework**: Integrates explainability tools (Integrated Gradients, SHAP, attention weights) to map attention scores back to the exact physical timestamps of the raw CTG. This provides clinical interpretability.

---

## 4. Academic & Research Context

Because this project is conducted within an **academic institution (PSG College of Technology)**:
1. **The Experimental Use Exception**: In patent law, using a patented invention solely for research, academic experimentation, or non-commercial testing is generally protected under the experimental use exemption (subject to jurisdiction-specific definitions).
2. **Lack of Damages**: Patent holders (like GE Healthcare) typically seek injunctions or financial damages. Since an academic project does not generate revenue or market competition, there is no financial basis for litigation.
3. **Publication Freedom**: Patents restrict the commercial manufacture, sale, and use of an invention; they do **not** restrict the publication of scientific research papers, theses, or conference proceedings.

---

## 5. Mitigation & Architectural Recommendations

To ensure absolute safety and highlight our academic contribution:
1. **Emphasize Multi-Task Synergy**: Position the model not as a "pattern detector" (like GE's patent), but as a joint **Multi-Task Biochemical Predictor** that simultaneously optimizes regression, FIGO classification, and terminal pH prediction.
2. **Document Differentiators**: Maintain our codebase layout separating the temporal encoder from the clinical feature regressor and the FIGO logic. This modular design highlights that our system relies on multi-task learning rather than "longitudinal pattern correlation."
3. **Cite Prior Art Transparently**: If writing a thesis or paper, openly cite GE's patent as state-of-the-art prior art and explain how our multi-task, knowledge-infused framework builds upon and differentiates from it. This establishes academic integrity.
4. **Model Selection & Rerun Protocol Alignment**: Ensure all Phase 1 baseline temporal encoders (1D CNN, BiLSTM, GRU, TCN, Multi-Scale LSTM, PatchCTG, PatchTST) and the Phase 2 Multi-Task Framework process inputs as continuous sequence latent encodings, logging patent differentiation compliance in [model_inferences_log.md](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/docs/model_inferences_log.md) as part of [model_evaluation_plan.md](file:///c:/Users/ELCOT/Desktop/CTG-Fetal-Distress-Prediction/docs/model_evaluation_plan.md).
