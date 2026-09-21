# Final Project Synthesis: Results and Discussion

## 1. Overview

This project investigated machine-learning approaches for predicting fetal acidemia from intrapartum cardiotocography (CTG), with the CTU-UHB dataset serving as the principal experimental cohort. The primary clinical endpoint was defined as **umbilical arterial pH ≤ 7.15**, and an **AUROC of at least 0.85** was established as the minimum target for a successful predictive model.

Rather than evaluating a single architecture, the study progressively investigated the major sources from which predictive improvement could plausibly arise: signal preprocessing, signal representation, patient-level aggregation, clinical knowledge infusion, temporal context, and alternative supervision using the continuous acid-base outcome.

The final clean cohort consisted of:

* **547 patients**
* **110 acidemia-positive patients** at pH ≤ 7.15
* **8517 CTG windows**
* 20-minute FHR windows sampled at 4 Hz
* 2.5-minute window stride

All primary experiments were evaluated using **patient-grouped five-fold cross-validation**, ensuring that windows from the same patient could not occur in both training and evaluation partitions. Patient-level predictions, rather than individual-window predictions, were used for the principal evaluation.

Across the complete modelling programme, the strongest locked models achieved AUROC values of approximately **0.736–0.743**, below the predefined target of 0.85.

The final selected research model was the **Continuous Clinical Huber Regression model**, which achieved:

**AUROC = 0.7426 [95% CI: 0.6876–0.7931]**

with an AUPRC of **0.4773** for the primary pH ≤ 7.15 endpoint.

Importantly, this model was not statistically superior to the previous Phase 4 knowledge-guided fusion model. Instead, the two models demonstrated statistically comparable discrimination.

---

# 2. Experimental Strategy

The modelling programme was organised as a sequence of controlled experimental phases. Each phase addressed a specific hypothesis concerning the limitations of CTG-based acidemia prediction.

The experimental progression was:

**Phase 1 — Signal preprocessing and data integrity**

↓

**Phase 2 — Signal representation learning**

↓

**Phase 3 — Patient-level aggregation and multiple-instance learning**

↓

**Phase 4 — Knowledge-guided clinical feature fusion**

↓

**Phase 5 — Temporal and multi-resolution modelling**

↓

**Phase 6 — Continuous and ordinal acid-base supervision**

↓

**Phase 7/7.1 — Locked validation and baseline reconciliation**

This structure allowed unsuccessful modelling directions to be explicitly closed instead of repeatedly modifying the same architectures.

---

# 3. Phase 1 — Signal Preprocessing and Experimental Integrity

The first phase established the preprocessing and evaluation framework used throughout the subsequent experiments.

The preprocessing audit examined signal quality, missing samples, interpolation behaviour, physiological cleaning, normalisation, window boundaries, and patient coverage.

Particular attention was given to preventing information leakage between patients. Because each CTG recording generated multiple overlapping windows, a random window-level split would allow highly correlated segments from the same patient to appear in both training and evaluation data.

Therefore, all primary experiments used **patient-grouped splitting**.

The final preprocessing contract preserved the morphology of the FHR signal while avoiding aggressive transformations that could remove clinically meaningful variability or deceleration patterns.

This phase established an important methodological principle for the remainder of the project:

> Model performance must reflect generalisation to unseen patients rather than recognition of overlapping windows from previously observed recordings.

---

# 4. Phase 2 — Signal Representation Learning

Phase 2 investigated whether fetal acidemia could be predicted directly from CTG morphology using deep representation learning.

Several representations were evaluated:

* raw FHR using a one-dimensional CNN/ResNet,
* raw FHR combined with FHR derivatives and signal-quality masks,
* continuous wavelet transform scalograms,
* CWT combined with signal-quality masks,
* recurrence plots,
* late fusion between signal representations.

The strongest signal-only result was obtained from the raw one-dimensional FHR representation.

The patient-level AUROC was approximately:

**Raw 1D FHR model: 0.6593**

The alternative representations did not improve discrimination:

* multi-channel 1D representation: **0.6164**
* CWT: **0.5672**
* CWT + quality mask: **0.5355**
* recurrence plot: **0.5566**
* late representation fusion: **0.5893**

The CWT and recurrence-plot representations therefore provided no evidence of extracting additional acidemia-related information from the signal.

### Interpretation

Transforming CTG into visually richer two-dimensional representations did not automatically make the underlying physiological relationship easier to learn.

In particular, the CWT and recurrence-plot branches increased representational complexity without producing corresponding improvements in patient-level discrimination.

Consequently, the two-dimensional representation branch was closed and subsequent signal modelling retained the one-dimensional temporal representation.

---

# 5. Phase 3 — Patient-Level Aggregation and Multiple-Instance Learning

A CTG recording contains many windows, while the acid-base outcome is available at the patient level. Phase 3 therefore investigated whether the principal limitation was not window representation itself, but the method used to combine window predictions into a patient prediction.

Several fixed aggregation strategies were evaluated:

* maximum probability,
* 90th percentile probability,
* mean probability,
* top-three window averaging.

Learned aggregation methods were also evaluated using latent embeddings from the one-dimensional signal network:

* mean embedding pooling,
* maximum embedding pooling,
* attention-based multiple-instance learning,
* fixed-K attention pooling.

Among the tested approaches, **extreme-value aggregation** performed best.

The 90th-percentile strategy achieved:

**AUROC = 0.6701**

with similar performance from:

* top-three pooling: **0.6680**
* maximum pooling: **0.6664**

In contrast, learned multiple-instance approaches deteriorated substantially:

* attention MIL: **0.5337**
* fixed-K attention: **0.4975**
* mean embedding pooling: **0.4894**
* maximum embedding pooling: **0.4683**

Control experiments using shuffled patient associations and label permutation produced approximately chance-level discrimination, supporting the integrity of the experimental implementation.

### Interpretation

The results suggest that acidemia-related information, when present in the signal model, is concentrated in a relatively small subset of high-risk windows.

However, the patient cohort was not sufficiently large for flexible attention-based aggregation to reliably learn which windows should dominate the patient prediction.

The simpler P90/extreme-value strategy was therefore retained.

---

# 6. Phase 4 — Knowledge-Guided Clinical Fusion

Phase 4 tested whether explicitly engineered physiological descriptors could provide information complementary to the deep FHR representation.

Nineteen continuous CTG descriptors were extracted, covering:

* baseline FHR,
* short-term variability,
* long-term variability,
* accelerations,
* early decelerations,
* late decelerations,
* variable decelerations,
* prolonged decelerations,
* maximum deceleration depth,
* deceleration area,
* deceleration burden,
* longest deceleration,
* baseline trend,
* variability trend,
* uterine contraction frequency,
* tachysystole,
* mean uterine activity amplitude,
* FHR–UC lag,
* FHR–UC coupling.

A clinical logistic regression model using these descriptors achieved:

**AUROC = 0.7198**

The strongest individual feature family was the deceleration group, with AUROC **0.6968**, followed by accelerations and variability.

Several methods for combining clinical and signal information were evaluated.

The strongest method was **logit prior modulation**, which achieved:

**AUROC = 0.7361 [95% CI: 0.679–0.788]**

and:

**AUPRC = 0.4514**

The improvement over the standalone signal model was approximately **+0.066 AUROC**, demonstrating that the clinical descriptors contributed meaningful complementary information.

More flexible fusion methods did not improve performance:

* gated fusion: **0.6166**
* naïve concatenation: **0.6344**
* residual fusion: **0.5803**
* residual error correction: **0.6348**

### Complementarity analysis

The signal and clinical models were only moderately correlated:

* Pearson correlation ≈ **0.40**
* Spearman correlation ≈ **0.42**

Among the 110 acidemia-positive patients:

* 19 were identified predominantly by the signal model,
* 26 predominantly by the clinical model,
* 21 by both models.

This demonstrated genuine, although limited, complementarity between learned FHR morphology and engineered physiological descriptors.

### Interpretation

Phase 4 provided one of the most important findings of the project.

Adding physiological knowledge improved the signal-based model, but **low-capacity fusion was substantially more reliable than flexible neural fusion**.

The resulting Phase 4 model became the principal knowledge-guided model:

**20-minute 1D ResNet + 19 clinical descriptors + logit prior modulation**

with patient-level AUROC **0.7361**.

---

# 7. Phase 5 — Temporal and Multi-Resolution Modelling

Phase 5 investigated whether the limited performance of the previous models resulted from using a fixed 20-minute observation window.

Signal models were evaluated at multiple temporal scales:

* 5 minutes,
* 10 minutes,
* 20 minutes,
* 30 minutes,
* 40 minutes,
* 60 minutes.

No individual temporal scale produced a meaningful improvement.

Single-scale AUROC values ranged approximately from **0.54 to 0.57** within the Phase 5 temporal experiments.

Multi-scale signal modelling achieved approximately:

**AUROC = 0.5808**

while the complete Phase 5 candidate achieved:

**AUROC = 0.6258**

This remained substantially below the Phase 4 model at **0.7361**.

Predictions from neighbouring temporal scales were strongly correlated. For example:

* 10-minute vs 20-minute Spearman correlation ≈ **0.85**
* 20-minute vs 40-minute Spearman correlation ≈ **0.80**

### Interpretation

Increasing or combining temporal context did not reveal a previously inaccessible predictive signal.

Instead, the different temporal scales largely produced redundant patient rankings.

The 20-minute context was therefore retained, and further temporal expansion was discontinued.

---

# 8. Phase 6 — Continuous and Ordinal Acid-Base Supervision

Previous phases treated acidemia primarily as a binary classification problem using pH ≤ 7.15.

Phase 6 tested a different hypothesis:

> Does converting arterial pH into a binary label discard useful information about the continuous severity of fetal acid-base disturbance?

Several alternative objectives were evaluated:

* continuous Huber regression,
* continuous knowledge-guided fusion,
* soft-target supervision,
* ordinal supervision,
* continuous signal-only prediction,
* multi-task signal prediction.

The strongest result was obtained using **continuous clinical Huber regression**.

For the primary pH ≤ 7.15 endpoint:

**AUROC = 0.7426 [95% CI: 0.688–0.793]**

**AUPRC = 0.4773**

The continuous model also achieved:

**MAE = 0.0727**

and:

**Pearson correlation with measured pH = 0.3981**

Other approaches produced:

* continuous knowledge fusion: **0.7315**
* soft-target model: **0.7055**
* ordinal knowledge fusion: **0.6502**
* continuous signal model: **0.5519**
* multi-task signal model: **0.5495**
* ordinal signal model: **0.5478**

### Severe acidemia

An important secondary analysis considered the more severe endpoint:

**pH ≤ 7.05**

For this endpoint, continuous clinical regression achieved:

**AUROC = 0.7621**

This represented the strongest observed discrimination for a clinically defined pH threshold within the final modelling programme.

However, the severe-acidemia endpoint was secondary and could not replace the predefined primary endpoint.

### Comparison with Phase 4

The observed difference between continuous clinical regression and the Phase 4 model was small.

Initial comparison:

**ΔAUROC ≈ +0.0065**

with no statistically significant superiority.

Thus, continuous supervision produced a modest numerical improvement but did not fundamentally alter the achievable discrimination.

---

# 9. Phase 7 — Discovery of Baseline Reproduction Drift

During the initial locked Phase 7 evaluation, an apparent large improvement was observed.

The continuous clinical model achieved approximately **0.7426**, while a newly generated Phase 4-style baseline achieved only approximately **0.6547**.

This appeared to imply an improvement of approximately:

**ΔAUROC = +0.088**

with statistically significant bootstrap and DeLong tests.

However, this result was inconsistent with the historical Phase 4 result of **0.7361**.

Rather than accepting the apparent improvement, a forensic reconciliation was performed.

This identified an important experimental error:

> The Phase 7 pipeline had retrained the one-dimensional ResNet from random initialisation instead of reproducing the frozen Phase 3 signal representation used by the original Phase 4 model.

The retrained signal branch was substantially weaker than the historical frozen representation and consequently degraded the reconstructed Phase 4 fusion model.

Therefore, the apparent +0.088 AUROC improvement was an artefact of **baseline reproduction drift** and was formally discarded.

---

# 10. Phase 7.1 — Baseline Reconciliation

A complete reconciliation audit was performed.

It confirmed that the experimental data themselves were unchanged:

* **547/547 patients identical**
* **110/110 positive labels identical**
* **8517 windows identical**
* all **19 clinical descriptors identical**
* folds identical
* outcome labels identical

The historical Phase 4 signal representations were recovered from the frozen Phase 3 artefacts and used to construct an immutable Phase 4 Gold reference.

The corrected results were:

| Model                        |      AUROC |        95% CI |      AUPRC |
| ---------------------------- | ---------: | ------------: | ---------: |
| Continuous Clinical Huber    | **0.7426** | 0.6876–0.7931 | **0.4773** |
| Phase 4 Master Gold          | **0.7361** | 0.6829–0.7875 |     0.4514 |
| Clinical Logistic Regression |     0.7198 | 0.6645–0.7721 |     0.4350 |
| Standalone 1D ResNet P90     |     0.6701 | 0.6120–0.7254 |     0.3542 |

The corrected difference between the two strongest models was:

**Continuous Clinical Huber − Phase 4 Gold = +0.0061 AUROC**

Paired bootstrap:

**p = 0.3380**

DeLong test:

**p = 0.6457**

Therefore, there was **no evidence that the continuous clinical model was statistically superior to the Phase 4 model**.

The original +0.0881 improvement was invalid and was removed from the final interpretation.

---

# 11. Final Model Selection

The **Continuous Clinical Huber Regression model** was selected as the final research model.

This selection was not based on a claim of statistical superiority over Phase 4.

Instead, it was selected because it combined:

* the highest numerical primary-endpoint AUROC,
* the highest primary AUPRC among the final candidates,
* strong performance for severe acidemia,
* deterministic inference,
* low computational complexity,
* direct use of interpretable physiological descriptors,
* no requirement for GPU inference,
* substantially lower implementation complexity than the deep signal-fusion pipeline.

Its final primary performance was:

**AUROC = 0.7426 [95% CI: 0.6876–0.7931]**

**AUPRC = 0.4773**

For severe acidemia at pH ≤ 7.05:

**AUROC = 0.7621**

The model should therefore be described as the **final selected research model**, rather than as a clinically deployable system.

Clinical deployment would require substantially stronger validation, including external validation on independent cohorts.

---

# 12. Comparison Against the Project Target

The project established a minimum target of:

**AUROC ≥ 0.85**

The final primary-endpoint result was:

**AUROC = 0.7426**

Therefore:

**The predefined performance target was not achieved.**

This result should not be obscured by selecting a more favourable secondary endpoint or by reporting performance from an inconsistent evaluation protocol.

Instead, the contribution of the project lies in systematically establishing which modelling strategies improved discrimination and which did not under a rigorous patient-level evaluation framework.

The strongest evaluated models consistently occupied an approximate AUROC range of **0.72–0.76**, depending on the model and endpoint.

This should be interpreted as the **empirical performance range observed within the evaluated CTU-UHB design space**, rather than as a mathematical or information-theoretic upper bound on CTG-based acidemia prediction.

---

# 13. Major Findings

The complete experimental programme produced several important findings.

### 13.1 Clinical descriptors were stronger than deep signal representations

The clinical feature models consistently outperformed the standalone deep FHR models.

The final continuous clinical model achieved **0.7426**, compared with approximately **0.6701** for the strongest frozen standalone signal representation.

This suggests that explicit physiological descriptors such as deceleration characteristics and variability contain highly useful information that is difficult for a deep model to learn reliably from the available sample size.

### 13.2 Signal information remained complementary

Although weaker independently, the signal model was not completely redundant.

Phase 4 improved from clinical LR performance of **0.7198** to **0.7361** through knowledge-guided signal fusion.

Thus, raw signal morphology contained additional information, but the magnitude of this contribution was limited.

### 13.3 Increasing model complexity did not guarantee better performance

Several more sophisticated approaches performed worse:

* CWT models,
* recurrence plots,
* attention MIL,
* embedding pooling,
* gated fusion,
* residual fusion,
* multi-resolution modelling,
* ordinal deep supervision.

The dataset therefore favoured constrained models with strong inductive structure over highly flexible architectures.

### 13.4 Learned patient-level attention was ineffective

Fixed P90/max-type aggregation consistently outperformed attention-based MIL.

The likely explanation is the limited number of independent patient bags available for learning flexible patient-level aggregation.

### 13.5 Longer temporal context was largely redundant

Changing the observation duration and combining multiple temporal resolutions did not materially improve prediction.

High correlation between temporal-scale predictions suggested that the models were extracting largely overlapping information.

### 13.6 Continuous pH supervision was useful but not transformative

Continuous Huber regression produced the strongest numerical model and improved severe-acidemia discrimination.

However, the improvement over the Phase 4 binary/fusion model was only approximately **0.006 AUROC** and was not statistically significant.

Therefore, binary thresholding was not the sole explanation for the remaining performance gap.

### 13.7 Reproducibility was critical to model comparison

The Phase 7 baseline discrepancy demonstrated that seemingly minor implementation differences can create apparently large performance improvements.

Without reconciliation, the project could have incorrectly reported a statistically significant +0.088 AUROC gain.

Recovering the frozen Phase 4 representations showed that the true difference was approximately +0.006.

This became an important methodological result of the project itself.

---

# 14. Why the AUROC ≥ 0.85 Target Was Not Achieved

The experiments progressively tested several plausible explanations for the performance limitation.

If signal representation were the dominant problem, CWT, recurrence plots, or richer signal channels should have improved performance. They did not.

If patient aggregation were the dominant problem, learned MIL should have substantially improved over fixed pooling. It did not.

If missing physiological knowledge were the dominant problem, knowledge-guided fusion should have produced a large improvement. It produced only a modest improvement.

If insufficient temporal context were the dominant problem, longer and multi-resolution models should have improved discrimination. They did not.

If binary thresholding were the dominant problem, continuous or ordinal supervision should have produced a large improvement. Continuous supervision produced only a small numerical gain.

The accumulated evidence therefore indicates that no single modelling modification tested in this study was sufficient to bridge the gap from approximately **0.74 to the target of 0.85**.

A plausible interpretation is that the limitation arises from a combination of factors including:

* limited cohort size,
* class imbalance,
* imperfect correspondence between CTG morphology and arterial pH,
* physiological heterogeneity,
* measurement noise,
* timing differences between intrapartum CTG abnormalities and delivery blood-gas measurement,
* and incomplete clinical context available to the models.

These factors cannot be resolved solely by increasing neural-network complexity.

---

# 15. Additional FIGO and Early-Warning Investigation

A separate experimental branch investigated whether clinically defined fetal-state deterioration might provide a more learnable target than delivery pH.

Corrected FIGO-state detection improved progressively through knowledge infusion, with the strongest knowledge-infused model reaching approximately:

**AUROC = 0.776**

However, this remained below the desired discrimination target.

The subsequent early-warning task attempted to predict transition from Normal to Abnormal FIGO state within 30 minutes.

The corrected cohort contained:

* 1122 eligible anchors,
* 703 positive deterioration events,
* 454 patients.

A clock-based baseline achieved approximately **0.5681 AUROC**.

Among models using temporal history, the strongest non-clock model achieved only approximately:

**AUROC = 0.5842**

Further analysis demonstrated substantial instability in the reconstructed FIGO target. Many abnormal transitions were driven by repetitive deceleration rules, and approximately half of Normal-to-Abnormal transitions reverted in the following epoch.

Consequently, the FIGO reconstruction and early-warning branches were closed rather than being used to replace the primary pH prediction problem.

---

# 16. Methodological Contributions

Although the predefined AUROC target was not reached, the project produced several methodological contributions.

First, it established a reproducible patient-level CTG evaluation pipeline that prevents overlapping-window leakage.

Second, it systematically compared raw temporal learning, transformed representations, clinical feature engineering, multiple-instance learning, knowledge-guided fusion, temporal modelling, and continuous supervision under a common experimental framework.

Third, it demonstrated that simple physiological models can outperform considerably more complex deep architectures in limited clinical datasets.

Fourth, it showed that knowledge-guided fusion can extract modest complementary value from otherwise weaker deep signal representations.

Fifth, the Phase 7.1 reconciliation demonstrated the importance of immutable predictions, frozen representations, deterministic baselines, and exact provenance tracking when comparing successive machine-learning experiments.

---

# 17. Limitations

Several limitations must be considered when interpreting the results.

The principal limitation is the relatively small number of independent patients, particularly the 110 patients satisfying the primary acidemia definition.

Although thousands of overlapping windows were available, these windows do not constitute thousands of independent clinical examples.

Second, the primary outcome is based on arterial pH measured at delivery. CTG reflects evolving fetal physiology over time, whereas pH represents a later biochemical measurement. The relationship between the two is therefore indirect.

Third, the models primarily use CTG-derived information. Important clinical variables that may modify fetal risk were not comprehensively incorporated.

Fourth, the final models require external validation. Cross-validation within CTU-UHB provides evidence of internal generalisation but cannot establish generalisation across hospitals, populations, devices, clinical protocols, or acquisition conditions.

Finally, the empirical performance range observed in this study should not be interpreted as proving that higher CTU-UHB performance is impossible. It only demonstrates that the evaluated modelling families did not reliably exceed this range under the adopted patient-level protocol.

---

# 18. Future Work

Future work should prioritise **new information and stronger validation rather than additional architectural complexity**.

The most valuable directions are:

1. **External validation** of the final clinical and Phase 4 models on independent CTG cohorts.

2. **Larger multi-centre training cohorts** to increase the number of independent acidemia-positive patients.

3. Integration of additional maternal, fetal, labour, and delivery variables alongside CTG.

4. Investigation of clinically meaningful composite outcomes where justified by obstetric evidence.

5. Self-supervised or foundation-model pretraining using large CTG datasets that are strictly independent of the final evaluation cohort.

6. Prospective evaluation to determine whether model predictions provide useful information beyond standard clinical interpretation.

Future experiments should preserve patient-level separation and explicitly document any dataset exposure during pretraining.

---

# 19. Final Conclusion

This project conducted a systematic investigation of fetal acidemia prediction from intrapartum CTG using the CTU-UHB dataset.

The original target was **AUROC ≥ 0.85** for prediction of arterial **pH ≤ 7.15**.

That target was **not achieved**.

The final selected Continuous Clinical Huber Regression model achieved:

**AUROC = 0.7426 [95% CI: 0.6876–0.7931]**

with:

**AUPRC = 0.4773**

The previous Phase 4 knowledge-guided fusion model achieved:

**AUROC = 0.7361 [95% CI: 0.6829–0.7875]**

and the difference between these models was not statistically significant.

The experimental evidence showed that physiological feature engineering and constrained knowledge-guided modelling were more reliable than increasingly complex deep architectures for this cohort. Raw signal morphology contributed complementary information, but not enough to produce the large improvement required to reach the predefined target.

The project therefore does not claim that CTG-based acidemia prediction has been solved. Instead, it demonstrates through controlled negative and positive experiments where predictive information was obtained, where additional modelling complexity failed to help, and how rigorous patient-level evaluation substantially changes the interpretation of apparent machine-learning performance.

The Phase 7.1 reconciliation further demonstrated why reproducibility is essential: an apparent improvement of approximately +0.088 AUROC disappeared after restoring the correct historical baseline, leaving a true difference of only approximately +0.006.

The final outcome is therefore a reproducible and evidence-based assessment of CTG-based fetal acidemia prediction within the evaluated CTU-UHB setting. The strongest observed discrimination remained approximately **0.74 for the primary endpoint and 0.76 for severe acidemia**, indicating that future progress is more likely to require larger independent cohorts, additional clinical information, carefully controlled pretraining, and external validation than further increases in model complexity.
