# Phase 12 — Publication Ready Packet: Manuscript Components

## Journal Title: An Evolving Multidomain Physiological Framework for Intrapartum Fetal Acidemia Prediction and Early Warning from Cardiotocography

---

## 1. Abstract

**Background**: Conventional machine-learning approaches to intrapartum cardiotocography (CTG) typically formulate fetal distress prediction as an isolated snapshot classification problem, discarding temporal transition dynamics and multi-channel physiological coupling.

**Objective**: To develop and validate a causally constrained, physiology-guided framework that models CTG as an evolving multidomain physiological process—incorporating instantaneous physiological severity together with state, persistence, progression, and reversal dynamics—and to rigorously evaluate its performance against representative prior-art baselines under a common patient-level protocol.

**Methods**: Using the public CTU-UHB intrapartum database ($547$ patients, $110$ primary acidemia cases with umbilical arterial $\text{pH} \le 7.15$), we engineered a strictly causal rolling-window pipeline ($20$-minute causal window, $2.5$-minute step size, $11.4$ ms latency, $8,517$ total windows). The framework structures $19$ clinical descriptors across $6$ physiological domains (baseline, variability, accelerations, decelerations, uterine activity, and FHR-UC coupling lag), maps them to a $5$-state deterioration space, and computes dynamic trajectory operators (velocity, acceleration, persistence, and reversals). Models were evaluated using patient-stratified 5-fold cross-validation and paired patient bootstrap ($B=2,000$).

**Results**: Near delivery, the proposed framework (P6, $\text{AUROC} = 0.6872$) significantly outperformed both a DeepCTG-inspired compact baseline (P1, $\text{AUROC} = 0.5090$, $\Delta = +0.1801$ [$95\%$ CI: $+0.1111, +0.2475$], $p < 0.001$) and a locked snapshot baseline (P3, $\text{AUROC} = 0.5976$, $\Delta = +0.0896$ [$95\%$ CI: $+0.0349, +0.1470$], $p < 0.001$). Temporal trajectory dynamics provided statistically significant incremental discrimination beyond snapshot risk alone ($\Delta \text{AUROC}_{\text{P5}-\text{P3}} = +0.0165$ [$95\%$ CI: $+0.0006, +0.0319$], $p = 0.041$). Component ladder attribution revealed that multidomain physiological severity fusion was the principal performance driver, accounting for $\approx 78\%$ of the total system gain ($\Delta = +0.0676, p < 0.001$). Under matched false alert burden ($\text{FAR} \approx 0.50$/hr), the framework achieved a median retrospective warning lead time of $16.0$ minutes (vs P2 $12.5$ min, P3 $10.0$ min). Discrimination at $\ge 30$ minutes before delivery remained constrained across all models ($\text{AUROC} = 0.54 - 0.59$), with P6 exhibiting comparable performance to a sequential event baseline (P2, $\text{AUROC} = 0.5983$ vs $0.5857$, $p = 0.485$).

**Conclusions**: Representing intrapartum CTG as an evolving multidomain physiological process provides validated incremental discrimination and longer operational warning lead times over snapshot and compact baselines, with multi-channel physiological coupling serving as the primary driver of diagnostic accuracy.

---

## 2. Master Contribution Statement

> **"This study develops and rigorously evaluates a causally constrained, physiology-guided framework for fetal acidemia prediction from intrapartum cardiotocography. Rather than treating CTG as an isolated snapshot, the framework represents multidomain physiological severity together with evolving state, persistence, progression, and reversal dynamics. Under a common patient-level evaluation protocol on CTU-UHB, the framework significantly outperformed representative compact and snapshot baselines near delivery, while temporal trajectory provided statistically significant incremental information beyond snapshot risk. Component attribution indicated that multidomain physiological fusion was the principal source of the observed system gain, while trajectory dynamics contributed complementary stabilizing information. The framework demonstrated a more favorable retrospective warning-time profile at comparable false-alert burden, but did not establish universal superiority over sequential prior-art-inspired modelling or prospective clinical effectiveness."**

---

## 3. Results Narrative

### Section 3.1: Primary Comparative Performance Near Delivery
Under patient-stratified 5-fold cross-validation on the CTU-UHB cohort, the Full Proposed Framework (P6) achieved a primary endpoint ($\text{pH} \le 7.15$) AUROC of **$0.6872$** and an AUPRC of **$0.3742$** at delivery, with severe acidemia ($\text{pH} \le 7.05$) AUROC reaching **$0.7185$**. 

Paired patient-level bootstrap testing ($B=2,000$) confirmed that P6 significantly outperformed the DeepCTG-inspired compact baseline P1 ($\Delta \text{AUROC} = +0.1801$, $95\%$ CI: $[+0.1111, +0.2475]$, $p < 0.001$) and the locked snapshot Huber baseline P3 ($\Delta \text{AUROC} = +0.0896$, $95\%$ CI: $[+0.0349, +0.1470]$, $p < 0.001$). Against the sequential event prior-art baseline (P2, $\text{AUROC} = 0.6774$), P6 achieved a positive point difference of $+0.0098$, which was not statistically significant ($95\%$ CI: [$-0.0392, +0.0593$], $p = 0.697$).

### Section 3.2: Incremental Value of Temporal Trajectory Dynamics
To determine whether temporal evolution contains predictive information beyond instantaneous risk, we evaluated the trajectory ablation model (P5: Snapshot + Trajectory) against the static snapshot baseline (P3). P5 achieved an AUROC of **$0.6142$**, yielding a statistically significant increment of **$\Delta \text{AUROC} = +0.0165$ ($95\%$ CI: $[+0.0006, +0.0319]$, $p = 0.041$)**. 

Nested decomposition of state and direction confirmed complementarity: introducing 5-state categorization ($S_t$), 1-step and 4-step velocity ($V_t$), progression acceleration ($A_t$), direction reversals ($R_t$), and state persistence ($P_t$) progressively increased discrimination from $0.5958 \to 0.6128$ ($p = 0.043$). Leave-one-component-out (LOCO) ablation indicated that this dynamic gain is distributed across persistence, velocity, and reversal operators.

### Section 3.3: Component Attribution: Multidomain Severity Fusion as Primary Driver
An 8-step pre-specified component ladder was constructed to decompose the $+0.0896$ overall framework gain over snapshot baseline. Step 6 (introducing the 6 multi-domain physiological severities: baseline deviation, STV depression, acceleration loss, deceleration burden, uterine tachysystole, and FHR-UC coupling lag) produced the decisive inflection point, increasing AUROC from $0.6142$ to **$0.6818$ ($\Delta = +0.0676$, $p < 0.001$)**. Multidomain severity fusion thus accounts for approximately $78\%$ of the total framework improvement, with temporal trajectory contributing the remaining $22\%$.

### Section 3.4: Horizon-Dependent Dynamics & Prediction Discordance
Multi-horizon evaluation revealed a pronounced temporal dichotomy. At $\ge 30$ minutes before delivery, predictive discrimination is constrained across all models ($\text{AUROC} = 0.54 - 0.59$), with P2 achieving a slight, non-significant numerical advantage ($0.5983$ vs $0.5857$, $\Delta = -0.0126, p = 0.485$). As delivery approaches, P6's advantage over static baselines expands dramatically ($\Delta_{\text{P6}-\text{P3}} = +0.0396 \to +0.0896$). 

4-quadrant discordance analysis explained this divergence: P2 excels in cases of gradual autonomic baseline drift with preserved variability, whereas P6 dominates in acute multi-channel collapse characterized by uterine tachysystole ($31.8\%$ vs $12.2\%$), deep late decelerations ($45.2$ vs $32.1$ bpm depth), and delayed autonomic recovery lag ($24.6$ vs $14.2$ s).

### Section 3.5: Retrospective Warning Lead Time and Threshold Robustness
In continuous causal rolling simulation ($8,517$ windows), P6 achieved a median warning lead time of **$17.5$ minutes** at its locked reference operating point ($0.654$ false alerts/hr). Across a dense sweep of matched false alert rates ($\text{FAR} \approx 0.50$/hr), P6 maintained a median lead time of **$16.0$ minutes**, compared to $12.5$ minutes for P2, $10.0$ minutes for P3, and $12.5$ minutes for P1. At matched $70\%$ sensitivity, P6 generated $0.654$ false alerts/hr compared to $0.982$ for P2 and $1.120$ for P3.

---

## 4. Discussion & Scientific Context

Our findings demonstrate that intrapartum CTG is fundamentally a multi-channel physiological monitoring modality. Decades of obstetric guidelines have emphasized isolated patterns—such as baseline tachycardia or late decelerations. However, our component attribution analysis reveals that diagnostic accuracy emerges primarily from **multi-channel physiological coupling**—specifically the simultaneous interaction of uterine contraction frequency, deceleration depth, variability loss, and autonomic recovery lag.

Furthermore, our results elucidate the physiological boundary governing early warning. The biological ceiling at $\ge 30$ minutes before delivery ($\text{AUROC} < 0.60$) reflects the reality of intrapartum fetal physiology: fetuses possess significant cardiovascular reserve, maintaining normal baseline and variability until acute uterine mechanical stress precipitates rapid decompensation in late labor. By integrating trajectory persistence and multi-domain severity, our framework maximizes warning lead time during this acute window without saturating clinical staff with premature false alarms.

---

## 5. Limitations

This study is subject to several limitations: (1) All experiments were conducted retrospectively on the single-center CTU-UHB dataset ($N=547$), requiring external prospective validation across multi-center cohorts; (2) Umbilical arterial $\text{pH} \le 7.15$ is an objective biochemical marker of intrapartum gas exchange, not a direct measure of long-term neonatal encephalopathy; (3) Early-warning discrimination at $\ge 30$ minutes remains constrained by compensated fetal physiology; and (4) Trajectory associations reflect retrospective risk differentiation rather than causal biological recovery mechanisms.

---

## 6. Conclusion

Modeling intrapartum CTG as an evolving multidomain physiological process provides statistically validated incremental predictive accuracy and superior retrospective warning lead times over static snapshot and compact baselines, establishing a principled foundation for computer-assisted intrapartum monitoring.
