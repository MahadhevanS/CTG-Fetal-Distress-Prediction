# Phase 12 — Limitations, Failure Modes & Methodological Boundaries

## CTU-UHB Fetal Acidemia Prediction & Early-Warning Project

### 1. Overview & Purpose

A scientifically rigorous doctoral thesis and publication must clearly delineate what was **not** achieved, where models encountered empirical boundaries, and what assumptions limit the generalizability of findings.

This document formalizes the comprehensive limitations catalog for the project.

---

## 2. Core Limitations Catalog

### 1. Cohort & Dataset Limitation (Single-Center CTU-UHB)
* **Description**: The primary dataset throughout all experimental phases is the public CTU-UHB Intrapartum Cardiotocography Database ($547$ intrapartum recordings, University Hospital in Brno, Czech Republic).
* **Impact**: While CTU-UHB provides gold-standard 4 Hz sampled signals with objective umbilical blood gas analysis, it represents a single European tertiary center from 2009–2012. Clinical practice variations (e.g., induction protocols, oxytocin titrations, maternal positioning, and cesarean section thresholds) may affect feature distributions in other obstetric settings.

### 2. Retrospective Observational Nature
* **Description**: All analyses were conducted retrospectively on recorded labor traces after delivery.
* **Impact**: In clinical practice, obstetricians actively intervene when CTG abnormalities appear (e.g., stopping oxytocin, fluid boluses, maternal repositioning, or emergent operative delivery). Consequently, the recorded CTG traces reflect the combined effects of underlying fetal pathophysiology and intrapartum clinical interventions.

### 3. External Validity & Generalizability
* **Description**: The proposed framework has not yet been validated on independent prospective cohorts or external hospital databases.
* **Impact**: Demonstrated performance on CTU-UHB ($N=547$) provides strong internal cross-validated validity ($B=2,000$ paired patient bootstrap), but external prospective generalizability across diverse clinical monitoring hardware (e.g., scalp electrodes vs. Doppler ultrasound) remains to be demonstrated.

### 4. Clinical Endpoint Definition vs. Neurological Outcome
* **Description**: The primary supervised endpoint is umbilical arterial cord blood $\text{pH} \le 7.15$ ($110$ positive cases), with severe acidemia defined as $\text{pH} \le 7.05$ ($41$ positive cases).
* **Impact**: Umbilical cord pH is an objective biochemical marker of intrapartum gas exchange and metabolic acidosis. However, the vast majority of neonates with mild-to-moderate acidemia ($\text{pH} = 7.10 - 7.15$) recover uneventfully without developing hypoxic-ischemic encephalopathy (HIE) or long-term neurodevelopmental impairment.

### 5. Early-Warning Horizon Discrimination Boundary ($\ge 30$ Minutes)
* **Description**: Across all tested architectures (P1 through P6), predictive discrimination declines substantially at prediction horizons $\ge 30$ minutes before delivery ($\text{AUROC} = 0.54 - 0.59$) compared to near delivery ($\text{AUROC} = 0.68 - 0.74$).
* **Underlying Physiological Reason**: Acute intrapartum fetal decompensation typically manifests in the final 15–25 minutes of labor under high uterine contraction frequency and active maternal pushing (second stage). Prior to this acute phase, compensated fetal physiology exhibits normal baseline heart rate and preserved variability, placing a biological ceiling on long-horizon pre-delivery prediction from CTG morphology alone.

### 6. Prior-Art Implementation Scope (Faithful Approximations)
* **Description**: Comparative baselines (P1 DeepCTG-inspired compact baseline and P2 Vargas-Calixto-inspired sequential baseline) were implemented faithfully according to published methodological specifications.
* **Impact**: Where original proprietary code or undocumented preprocessing steps were omitted in the literature, our benchmark implemented faithful representations using standardized FIGO features and EWMA persistence filters under the common CTU-UHB evaluation protocol.

### 7. Operational Simulation vs. Prospective Clinical Utility
* **Description**: Operational alert metrics (median warning lead time of $17.5$ minutes, $0.654$ false alerts/hr) were evaluated in retrospective rolling simulation.
* **Impact**: Retrospective warning lead time demonstrates that the mathematical alert is triggered earlier than delivery. It does **not** prove prospective clinical effectiveness, reduction in emergency cesareans, or improved neonatal outcomes, which require randomized clinical trials.

### 8. Associative Physiological Interpretations vs. Causal Biological Recovery
* **Description**: In Experiment 11.5-D, patients with reversing trajectory exhibited lower acidemia rates than progressing patients ($20.5\%$ vs $23.5\%$).
* **Boundary Constraint**: This finding reflects an **associative risk correlation** within observational retrospective data. It must **not** be interpreted as proof of biological fetal homeostatic recovery, causal protection, or physiological compensation.

---

## 3. Analysis of Negative & Non-Significant Experimental Findings

A credible scientific thesis documents what did *not* work as clearly as what succeeded. The following experimental avenues were tested and found not to provide statistically significant advantages:

1. **2D Time-Frequency Projections**: Continuous Wavelet Transforms (CWT) and Recurrence Plots underperformed 1D temporal CNNs ($\text{AUROC} = 0.608 - 0.621$ vs $0.684$, Phase 2).
2. **Learned Attention-Based MIL**: Attention-weighted multiple instance learning overfitted to patient-level noise compared to robust fixed extreme-value pooling ($\text{AUROC} = 0.652$ vs P90 $0.692$, Phase 3).
3. **Longer Retrospective Context**: Expanding observation beyond 20 minutes (30, 45, 60 min) yielded no performance gain ($\text{AUROC} = 0.735$ vs $0.736$, Phase 5).
4. **Ordinal Loss Formulation**: Multi-class ordinal cross-entropy underperformed continuous Huber regression ($\text{AUROC} = 0.718$ vs $0.743$, Phase 6).
5. **Universal Prior-Art Superiority**: The proposed framework did not achieve statistically significant superiority over the Vargas-Calixto sequential baseline P2 ($p = 0.697$ at delivery, $p = 0.485$ at $\ge 30$m, Phase 11).

---

## 4. Summary of Methodological Boundary Integrity

These boundaries reinforce the validity of the project: the final conclusions do not rely on inflated claims or unverified assumptions, but on reproducible, bounded empirical evidence.
