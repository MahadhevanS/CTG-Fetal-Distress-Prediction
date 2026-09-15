# Limitations: Model 3 + Parity Multimodal Hybrid

This document provides an explicit, unvarnished accounting of the methodological, statistical, and clinical limitations of the Model 3 + Parity late-fusion hybrid investigation, in accordance with Section 20.5 of the execution specification.

---

## 1. Small Cohort Size and Event Scarcity
- **Total Patients:** The evaluation is conducted on $N=547$ intrapartum recordings from the single-center CTU-UHB cohort.
- **Limited Positive Events:** Only 110 primary distress cases ($\text{pH} \le 7.15$, 20.1% prevalence) and 41 severe cases ($\text{pH} \le 7.05$, 7.5% prevalence) are available across the entire dataset.
- **Statistical Power Limitations:** In a 5-fold cross-validation scheme, each validation fold contains only ~22 positive cases. On the 83-patient held-out internal test partition, only 17 positive cases exist. Consequently, performance metric differences of $\Delta \text{AUROC} \approx 0.01\text{--}0.02$ lack sufficient statistical power to clear two-sided significance thresholds ($p < 0.05$), and confidence intervals are broad.

---

## 2. Internal-Only Validation & Lack of External Cohort
- **Internal Partitions Only:** All results reflect 5-fold cross-validation and an 83-patient held-out internal test partition drawn from the same clinical site, clinical protocols, and acquisition hardware.
- **No External Generalization Evidence:** No external validation has been performed on independent clinical cohorts (such as the Oxford or STAN datasets). True clinical utility and generalizability remain unproven until the external validation study scoped in `docs/external_validation_handoff.md` is executed.
- **Status:** The hybrid is an internal research candidate only and must not be described as deployable or clinically validated.

---

## 3. Potential Observation-Duration and Opportunity Confounds
- **Duration Association:** Prior research in Phase 15 and Tier-2 hardening established that Model 3's incremental advantage over single-window P6 exhibits a negative correlation with recording duration ($r = -0.213, p < 0.0001$).
- **Observation Window Inequality:** Patients with longer active labour monitoring contribute up to 17 rolling windows, providing more opportunity for transient decelerations to register elevated risk scores compared to patients monitored for only 3–5 windows.
- **Residual Confounding:** Although duration and window counts were strictly excluded from the primary hybrid feature matrix, the hybrid inherits the underlying duration characteristics of Model 3.

---

## 4. Correlated Rolling Windows
- **High Autocorrelation:** The CTG rolling windows overlap by 17.5 minutes (87.5% signal overlap with a 2.5-minute stride).
- **Sequence Dependence:** Adjacent risk estimates are highly autocorrelated. While patient-level clustered bootstrap and patient-level pooling mitigate window-level independence violations, temporal risk dynamics remain constrained by sequence autocorrelation.

---

## 5. Architectural Dependence on Frozen P6
- **Single-Model Upstream Dependence:** The hybrid depends entirely on the window-level risk representation produced by the frozen Phase 12.1 P6 model (`LogisticRegression(C=0.05)` on 27-dimensional state trajectory features).
- **Error Propagation:** Any miscalibration, signal artifact susceptibility, or blind spots present in P6 directly propagate into Model 3 and subsequently into the hybrid. P6 itself was not retrained or adapted to multimodal inputs.

---

## 6. Information Redundancy Between Modalities
- **Partial Covariate Alignment:** Parity provides admission-level prior risk modulation (nulliparous mothers exhibit higher baseline distress rates). However, to the extent that prolonged labour and contraction fatigue in nulliparous labours manifest as progressive decelerations on CTG, Model 3 and Parity capture overlapping physiological vulnerability.

---

## 7. Uncertainty in the Temporal Mechanism
- **Shuffled-Time Control Ambiguity:** As documented in Tier-3 hardening (`scripts/model3_shuffled_time_control.py`), Model 3 with randomly permuted temporal ordering collapsed toward Model 2 ($0.7003$ vs $0.7006$), and real Model 3's delivery edge over shuffled time ($+0.0213$) did not reach significance ($p = 0.217$).
- **Capacity vs. Temporal Dynamics:** It remains scientifically unresolved at this cohort size whether Model 3's gain over fixed percentiles (Max/P90) reflects genuine temporal position dynamics or parameter capacity flexibility.

---

## 8. Absence of Clinician Benchmarking & Prospective Evidence
- **No Head-to-Head Clinician Comparison:** Model predictions have not been evaluated against real-time obstetrician or midwife cardiotocography interpretations or FIGO 2015 consensus panel classifications.
- **No Prospective or Interventional Trials:** There is no evidence that presenting hybrid risk scores to clinicians during labour alters decision-making, reduces unnecessary cesarean sections, or prevents neonatal acidemia.
- **Strict Non-Clinical Designation:** The hybrid model is not a medical device, is not approved for clinical decision support, and must not be used in patient care.
