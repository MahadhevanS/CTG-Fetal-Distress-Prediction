# Experiment E0: Data and Feature Audit Report

**Status:** PASSED

## 1. Cohort Summary
- Total Clean Patients: **547**
- 5-Fold CV Patients (Train/Val pool): **464** (84.8%)
- Held-Out Internal Test Partition: **83** (15.2%)

## 2. Endpoints
- Primary endpoint (pH $\le 7.15$): **110** positives (20.1%), 437 negatives
- Secondary severe endpoint (pH $\le 7.05$): **41** positives (7.50%), 506 negatives

## 3. Parity Distribution
- Missing values: **0**
- Min/Max: 0 / 7
- Nulliparous (parity = 0): **373** (68.2%)
- Breakdown: parity 0: 373, parity 1: 138, parity 2: 29, parity 3: 4, parity 4: 1, parity 5: 1, parity 7: 1

## 4. Window & Causal Invariants
- Total windows: **8517**
- Windows per patient: mean 15.6, median 17.0, range [3, 17]
- Duplicate patient IDs: **None**
- Duplicate window entries: **None**
- Post-delivery windows: **0**
- Chronological ordering: **Fully verified**
- Causal 20-min window boundary: **Fully verified**

## 5. Horizon Patient Eligibility
| Horizon | Eligible Patients | Positives | Negatives | % Cohort Eligible |
| :--- | :--- | :--- | :--- | :--- |
| $\ge 0$m | 547 | 110 | 437 | 100.0% |
| $\ge 10$m | 545 | 108 | 437 | 99.63% |
| $\ge 20$m | 534 | 103 | 431 | 97.62% |
| $\ge 30$m | 492 | 91 | 401 | 89.95% |
