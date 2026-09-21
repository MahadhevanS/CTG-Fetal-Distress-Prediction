# Trained Models & Inference Guide: CTU-UHB Fetal Acidemia Prediction

This directory contains the serialized model checkpoints, scalers, and standalone prediction modules developed across the 7-phase research program.

---

## 1. Directory Structure

```
models/
├── predict.py                        # Standalone Python predictor module
├── continuous_clinical_huber/        # Model B: Continuous Clinical Huber Model (AUROC = 0.7426)
│   ├── huber_fold_0.joblib           # Fold 0 HuberRegressor
│   ├── huber_fold_1.joblib           # Fold 1 HuberRegressor
│   ├── huber_fold_2.joblib           # Fold 2 HuberRegressor
│   ├── huber_fold_3.joblib           # Fold 3 HuberRegressor
│   ├── huber_fold_4.joblib           # Fold 4 HuberRegressor
│   ├── huber_full_cohort.joblib      # Production model fit on full 547 patient cohort
│   ├── scaler_fold_0.joblib          # Scalers for each fold
│   ├── ...
│   └── model_ensemble_weights.json   # Portable JSON coefficients & scalers
└── phase4_master/                    # Phase 4 Master Model (AUROC = 0.7361)
    ├── clinical_lr_fold_0.joblib     # Fold 0 Clinical Logistic Regression
    ├── ...
    └── phase4_clinical_weights.json  # Phase 4 fusion coefficients & weights
```

---

## 2. Quickstart: 3-Line Python Inference

```python
from models.predict import CTUAcidemiaPredictor

# Initialize the 5-fold ensemble predictor (or mode="full" for full cohort fit)
predictor = CTUAcidemiaPredictor(mode="ensemble")

# Input 19 clinical features for a patient
patient_features = {
    "baseline": 140.0,
    "stv": 5.2,
    "ltv": 12.0,
    "acc_count": 2.0,
    "early_dec_count": 0.0,
    "late_dec_count": 1.0,
    "var_dec_count": 2.0,
    "prolonged_dec_count": 0.0,
    "dec_max_depth": 35.0,
    "dec_area": 450.0,
    "dec_burden": 0.12,
    "longest_dec": 65.0,
    "baseline_slope": -0.05,
    "variability_slope": -0.02,
    "uc_count": 5.0,
    "tachysystole": 0.0,
    "mean_uc_amp": 55.0,
    "fhr_uc_lag": 22.0,
    "fhr_uc_coupling": 0.65
}

# Run inference
result = predictor.predict_patient(patient_features)
print(result)
# Output:
# {
#   "predicted_ph": 7.182,
#   "acidemia_risk_score": -7.182,
#   "acidemia_probability_proxy": 0.382,
#   "acidemia_flag_715": false,
#   "severe_acidemia_flag_705": false
# }
```

---

## 3. Supported 19 Clinical Descriptors

| # | Feature Name | Description | Units / Scale |
| :-: | :--- | :--- | :--- |
| 1 | `baseline` | Estimated baseline fetal heart rate | bpm (110–160 normal) |
| 2 | `stv` | Short-Term Variability (epoch-to-epoch delta) | ms |
| 3 | `ltv` | Long-Term Variability (peak-to-trough in 1-min epochs) | bpm |
| 4 | `acc_count` | Number of accelerations ($\ge 15$ bpm for $\ge 15$ s) | count |
| 5 | `early_dec_count` | Decelerations synchronous with contraction peaks | count |
| 6 | `late_dec_count` | Decelerations delayed after contraction peaks (hypoxia marker) | count |
| 7 | `var_dec_count` | Variable decelerations (cord compression) | count |
| 8 | `prolonged_dec_count` | Decelerations lasting $> 2$ minutes | count |
| 9 | `dec_max_depth` | Maximum heart rate drop below baseline | bpm |
| 10 | `dec_area` | Total integrated deceleration area | $\text{bpm} \times \text{s}$ |
| 11 | `dec_burden` | Fraction of recording time spent decelerating | $[0, 1]$ |
| 12 | `longest_dec` | Duration of the single longest deceleration | seconds |
| 13 | `baseline_slope` | Linear drift of baseline FHR over time | $\text{bpm} / \text{hour}$ |
| 14 | `variability_slope` | Rate of loss of variability over time | $\text{bpm} / \text{hour}$ |
| 15 | `uc_count` | Number of uterine contractions | count |
| 16 | `tachysystole` | Contraction frequency $> 5$ per 10 minutes | binary $\{0, 1\}$ |
| 17 | `mean_uc_amp` | Average amplitude of uterine contractions | a.u. |
| 18 | `fhr_uc_lag` | Average time delay from contraction peak to FHR trough | seconds |
| 19 | `fhr_uc_coupling` | Cross-correlation between uterine activity and FHR | $[0, 1]$ |
