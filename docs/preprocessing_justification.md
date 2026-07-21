# Preprocessing Algorithms & Techniques: Scientific & Clinical Justification ("The Why")

This document details the scientific, signal processing, and clinical justifications for the preprocessing choices made in the CTG Fetal Distress Prediction framework. It correlates our pipeline decisions with the peer-reviewed methodologies of the reference papers cited in our abstract.

---

## 1. Summary of Reference Papers from Abstract

To ground the justifications, we refer to the following papers listed in the project abstract:
* **[Ref 1] Mendis et al. (2025)**: *Cross-database evaluation of deep learning methods for intrapartum cardiotocography classification.* (IEEE J. Transl. Eng. Health Med.)
* **[Ref 2] Khan et al. (2025)**: *PatchCTG: A patch cardiotocography transformer for antepartum fetal health monitoring.* (Sensors)
* **[Ref 3] Asfaw et al. (2023)**: *Multimodal deep learning for predicting adverse birth outcomes based on early labour data.* (Bioengineering)
* **[Ref 4] Cao et al. (2023)**: *Intelligent antepartum fetal monitoring via deep learning and fusion of cardiotocographic signals and clinical data.* (Health Info. Sci. Syst.)
* **[Ref 5] Zhang et al. (2022)**: *Multimodal learning for fetal distress diagnosis using a multimodal medical information fusion framework.* (Frontiers in Physiology)
* **[Ref 6] McCoy et al. (2024)**: *Intrapartum electronic fetal heart rate monitoring to predict acidemia at birth with the use of deep learning.* (AJOG)

---

## 2. Preprocessing Techniques: Justifications & References

### 2.1 Signal Quality Assessment (SQA) & 30% Missing Ratio Rejection
* **The "Why" (Signal & ML Justification)**:
  CTG signals (especially FHR) are prone to dropouts when the ultrasound transducer loses contact with the maternal abdomen during movement or contractions. Training a neural network on windows with extensive missing data (represented as flat zero lines) forces the model to learn unphysiological features or flatline representations, leading to high false-alarm rates.
* **Clinical Justification**:
  In clinical practice, a tracing with >30% signal loss is deemed uninterpretable, prompting the clinician to adjust the monitor or switch to an internal fetal scalp electrode.
* **References**:
  * **[Ref 1] Mendis et al. (2025)** emphasizes strict data quality filters, discarding segments with prolonged signal dropouts to prevent cross-database model degradation.
  * **[Ref 6] McCoy et al. (2024)** utilizes SQA thresholding to discard noisy signals that do not reflect true cardiac cycles, ensuring high fidelity of clinical features.

---

### 2.2 Cubic Spline Interpolation for Short Gaps (<= 15 Seconds)
* **The "Why" (Signal & ML Justification)**:
  Linear interpolation creates sharp corners (first-derivative discontinuities), introducing artificial high-frequency noise into the frequency spectrum. Cubic spline interpolation fits a smooth polynomial curve that preserves the first and second derivatives, mimicking the natural physical acceleration and deceleration limits of the fetal heart. Gaps longer than 15 seconds (60 samples at 4 Hz) are left un-interpolated, because filling longer gaps risks fabricating bradycardia or decelerations that did not occur.
* **Clinical Justification**:
  Fetal heart rate is controlled by the autonomic nervous system, which changes the rate smoothly, not in jagged, linear steps.
* **References**:
  * **[Ref 4] Cao et al. (2023)** details the use of spline interpolation to reconstruct short signal losses to keep the integrity of continuous temporal signals.
  * **[Ref 1] Mendis et al. (2025)** validates that interpolating short dropouts keeps temporal sequence models stable.

---

### 2.3 4th-Order Low-Pass Butterworth Filter (Cutoff: 1.5 Hz)
* **The "Why" (Signal & ML Justification)**:
  FHR signals contain high-frequency noise from fetal movement, maternal heart rate crossover, and minor sensor displacements. A low-pass filter removes these non-physiological fluctuations. A 4th-order filter provides a steep frequency roll-off while minimizing the Gibbs phenomenon (ringing artifacts) at transition points. Crucially, a **zero-phase** forward-backward implementation (`filtfilt`) is used; causal filters introduce phase delay, which shifts deceleration troughs relative to contraction peaks, corrupting the phase alignment needed to distinguish early vs. late decelerations.
* **Clinical Justification**:
  Fetal heart rate changes do not occur at frequencies above 1.5 Hz (90 bpm per second). High-frequency spikes are artifacts and must be removed to avoid misclassifying them as accelerations or variable decelerations.
* **References**:
  * **[Ref 4] Cao et al. (2023)** utilizes low-pass filtering to smooth raw signals before extracting morphological peaks.
  * **[Ref 2] Khan et al. (2025)** employs low-pass Butterworth filtering to clean input channels before sending them to the temporal transformer.

---

### 2.4 FHR Baseline Estimation (Iterative Mean vs. ALS)
* **The "Why" (Signal & ML Justification)**:
  The clinical definition of accelerations and decelerations is relative to the "quiet state" baseline. A simple rolling average is heavily skewed by long decelerations or frequent accelerations. 
  * The **Iterative Mean Baseline** rejects samples outside a +/- 15 bpm window from the current mean, ensuring that transient excursions do not distort the estimate.
  * The **Asymmetric Least Squares (ALS)** baseline solves a regularized optimization problem penalizing baseline roughness (smoothness $\lambda = 10^5$) and uses asymmetry weights ($p = 0.5$) to reject transient peaks and valleys.
* **Clinical Justification**:
  Obstetricians visually determine the baseline by looking at the quiet segments of the tracing, ignoring decelerations and accelerations. Our algorithms programmatically emulate this clinical reasoning.
* **References**:
  * **[Ref 4] Cao et al. (2023)** and **[Ref 2] Khan et al. (2025)** establish baseline estimation as a critical first step, as all subsequent FIGO morphological classifications depend on baseline accuracy.

---

### 2.5 Baseline-Corrected FHR Input (FHR - Baseline)
* **The "Why" (Signal & ML Justification)**:
  A raw FHR signal ranges from 110 to 160 bpm. A temporal neural network trained on raw values will focus on absolute amplitude offsets, which are highly patient-specific. Subtracting the baseline centers the signal at 0.0. This normalization step isolates the temporal variations (deviations from baseline), stabilizes gradient flow, and prevents the network from learning absolute value biases.
* **Clinical Justification**:
  Fetal distress is diagnosed by the *dynamics* of the heart rate (e.g., late decelerations relative to baseline, loss of variability), not the absolute heart rate value alone.
* **References**:
  * **[Ref 2] Khan et al. (2025)** (PatchCTG) subtracts local offsets to align FHR patches.
  * **[Ref 4] Cao et al. (2023)** utilizes baseline-relative representations to improve deep network generalization.

---

### 2.6 20-Minute Windowing
* **The "Why" (Signal & ML Justification)**:
  A 20-minute window length contains 4,800 samples at 4 Hz, which is long enough to capture low-frequency patterns (like long-term variability) and multiple uterine contractions (which occur every 2 to 5 minutes during labor). Shorter windows (e.g., 5 minutes) cannot capture enough contraction-deceleration pairs to identify late or variable decelerations.
* **Clinical Justification**:
  All major guidelines (FIGO, NICE, ACOG) state that a minimum of 20 minutes of continuous monitoring is required to classify a CTG tracing as Normal, Suspicious, or Pathological.
* **References**:
  * **[Ref 2] Khan et al. (2025)** uses partitioned temporal window patches to capture context.
  * **[Ref 1] Mendis et al. (2025)** benchmarks window lengths and finds 20-minute windows optimal for clinical outcomes.

---

### 2.7 Last 60-Minute Prediction Horizon (Terminal pH Outcome Mapping)
* **The "Why" (Signal & ML Justification)**:
  Fetal distress and acidemia are progressive conditions that develop during active labor. If a baby is born with acidemia (pH <= 7.15), it does not mean they were in distress 6 hours before delivery. Labeling early, healthy windows with the terminal pathological class introduces severe label noise (weak supervision). Limiting the dataset to the final 60 minutes of labor focuses the model on the period when distress is physiologically present and developing.
* **Clinical Justification**:
  Intrapartum fetal monitoring is designed to detect acute distress emerging during the active/second stages of labor, when uterine contractions are strongest and fetal reserve is most challenged.
* **References**:
  * **[Ref 6] McCoy et al. (2024)** limits analysis strictly to the intrapartum tracing leading up to delivery to predict neonatal acidemia.
  * **[Ref 3] Asfaw et al. (2023)** justifies modeling the progression of labor characteristics closer to birth.

---

### 2.8 Patient-Level Stratified Splitting
* **The "Why" (Signal & ML Justification)**:
  CTG recordings are continuous, and consecutive sliding windows from the same patient are highly correlated. If windows from the same patient are split across train and test sets, the model will memorize patient-specific signatures (such as baseline, unique noise patterns, or heart rate characteristics) rather than learning generalized clinical rules. This leads to severe data leakage and inflated, ungeneralizable performance.
* **Clinical Justification**:
  A clinical diagnostic model must perform accurately on *new, unseen patients* in the delivery room, not just on historical data segments of previously seen patients.
* **References**:
  * **[Ref 1] Mendis et al. (2025)** and **[Ref 6] McCoy et al. (2024)** enforce strict patient-level splits to guarantee generalization across validation boundaries.

---

### 2.9 Dynamic Stride (Imbalance Mitigation in Training Only)
* **The "Why" (Signal & ML Justification)**:
  Pathological fetal distress is a minority class (typically <15% of patients). Direct training on such imbalanced data causes the model to converge to the majority class.
  * We apply a 2-minute stride (high overlap) to the distress class, and a 10-minute stride (low overlap) to the normal class during training. This balances the batch distribution naturally.
  * Standard synthetic generation methods (like SMOTE) generate unphysiological waveforms when applied to time-series. Our dynamic stride approach ensures that all training samples represent real, physiological recordings.
  * Using a fixed 10-minute stride for validation and test splits guarantees that the evaluation set reflects the true, unbiased clinical population distribution.
* **Clinical Justification**:
  Ensures the AI system is trained to identify rare pathological signals without introducing artificial, clinically implausible heart rate waveforms.
* **References**:
  * **[Ref 6] McCoy et al. (2024)** and **[Ref 1] Mendis et al. (2025)** discuss balancing techniques (such as weighted loss functions and oversampling) to tackle clinical minority imbalance.

---

### 2.10 Multi-Task & FIGO Guidance (Supervised Auxiliary Learning)
* **The "Why" (Signal & ML Justification)**:
  Mapping raw 4800-sample signals directly to a single binary pH label is a highly complex, underdetermined optimization problem. By forcing the shared temporal encoder to solve auxiliary regression tasks (predicting Baseline FHR, STV, LTV, and deceleration counts) and auxiliary classification tasks (predicting FIGO classes), we regularize the shared representations. The encoder is forced to learn features that map to real physiological components, preventing overfitting to spurious correlations.
* **Clinical Justification**:
  Aligning neural networks with clinical guidelines (FIGO) ensures that the model's intermediate representations are clinically interpretable and explainable to clinicians.
* **References**:
  * **[Ref 5] Zhang et al. (2022)** demonstrates that multimodal information fusion and predicting intermediate clinical parameters alongside outcomes leads to superior predictive performance.
  * **[Ref 4] Cao et al. (2023)** utilizes a combination of raw signals and clinical features to guide diagnostic output.

---

### 2.11 Spike Removal (> 25 bpm/sec Limit)
* **The "Why" (Signal & ML Justification)**:
  Unphysiological single-sample spikes caused by ultrasound transducer loss-of-lock or movement artifact introduce steep impulse spikes in the first derivative. If left uncleaned, these impulse artifacts trigger false acceleration and deceleration detection routines. `remove_spikes()` zero-out deltas $> 6.25 \text{ bpm/sample}$ (25 bpm/sec at 4 Hz) so they are reconstructed smoothly by cubic spline interpolation.
* **Clinical Justification**:
  Human cardiac autonomic regulation cannot physically accelerate or decelerate fetal heart rate instantaneously by $>25 \text{ bpm}$ in a single second. Spikes are non-biological artifacts.
* **References**:
  * **[Ref 1] Mendis et al. (2025)** and **[Ref 4] Cao et al. (2023)** emphasize artifact rejection prior to peak-detection routines.

---

### 2.12 Strict 30-Minute Prediction Horizon (GAP 2 FIX)
* **The "Why" (Signal & ML Justification)**:
  Assigning `y_primary = 1` to windows extracted 45–60 minutes before delivery in acidemic pregnancies causes weak supervision label noise. Fetal reserve depletes progressively; early windows may reflect fully compensated, physiologically normal states.
* **Clinical Justification**:
  Intrapartum fetal acidemia ($\text{pH} \le 7.15$) manifests in the terminal stage of labor during acute uterine stress. Enforcing `PREDICTION_HORIZON_MINUTES = 30` ensures the binary target represents true uncompensated hypoxia.
* **References**:
  * **[Ref 6] McCoy et al. (2024)** models acidemia strictly within terminal delivery horizons.

---

### 2.13 Sampling Frequency Validation & Polyphase Resampling
* **The "Why" (Signal & ML Justification)**:
  Fixed-kernel 1D CNNs and Temporal Transformers require identical temporal sampling resolution across all input vectors. If input recordings differ in sampling rate, fixed-length windows span different physical durations, distorting feature filters. Polyphase integer-ratio resampling (`resample_poly`) resamples signals cleanly to 4.0 Hz without aliasing.
* **Clinical Justification**:
  Clinical parameters (STV, LTV, 15-second acceleration windows) are defined in physical time units. Standardizing sampling rates guarantees exact clinical metric extraction.
* **References**:
  * **[Ref 2] Khan et al. (2025)** and **[Ref 1] Mendis et al. (2025)** mandate uniform sampling frequency across CTG datasets.

---

### 2.14 Leak-Free Per-Channel Z-Score Normalization
* **The "Why" (Signal & ML Justification)**:
  Channel amplitudes (Baseline-corrected FHR diff and raw UC mmHg) have different scales. Normalizing both channels to $\mu=0, \sigma=1$ prevents gradient domination by higher-magnitude channels. Computing scaling parameters $\mu, \sigma$ exclusively on the training split prevents data leakage.
* **Clinical Justification**:
  Uterine contraction transducers (toco) use relative pressure units that vary across clinical hardware. Z-score normalization standardizes contraction amplitude dynamics across different clinics.
* **References**:
  * **[Ref 2] Khan et al. (2025)** applies channel-wise standardization for temporal deep learning.

---

### 2.15 UCI SisPorto Dataset Classical ML Baseline Role
* **The "Why" (Signal & ML Justification)**:
  The UCI SisPorto dataset provides 2,126 pre-extracted 21-feature tabular samples. Preprocessing it with `StandardScaler` (fit on train only) enables benchmarking classical tabular models (Random Forest, XGBoost) as baseline comparators against our temporal deep learning encoder.
* **Clinical Justification**:
  Demonstrates whether end-to-end temporal deep learning on raw continuous signals outperforms traditional system-extracted morphological features (SisPorto 2.0).
* **References**:
  * **[Ref 4] Cao et al. (2023)** benchmarks deep learning architectures against tabular ML baselines.
