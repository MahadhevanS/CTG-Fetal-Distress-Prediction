# Knowledge-Infused Physiological Trajectory Model for CTG Fetal Distress Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Abstract & Clinical Motivation

Intrapartum fetal distress due to hypoxia and fetal acidemia (umbilical arterial $\text{pH} \le 7.15$) is a leading cause of preventable neonatal morbidity and mortality. Cardiotocography (CTG), which records continuous Fetal Heart Rate (FHR) and Uterine Contractions (UC), is the global clinical standard for intrapartum monitoring, but conventional visual interpretation suffers from high inter-observer variability and high false-alarm rates.

This project's **locked, production model (Phase 12.1, internally referred to as "P6")** predicts fetal acidemia risk from 20-minute causal CTG observation windows using an interpretable, knowledge-infused pipeline: multidomain physiological severity scoring (6 domains derived from FIGO-consistent clinical descriptors) → a 5-state physiological deterioration engine → temporal trajectory dynamics (velocity, persistence, reversals) → a single logistic-regression risk classifier over a 40-dimensional state-trajectory feature vector. It is **not** an end-to-end deep temporal encoder — that direction was explored extensively in earlier project phases and set aside; see [§ Superseded exploratory work](#-superseded-exploratory-work-not-locked-not-delivered) below for why.

---

## 🔒 What is locked, and what "locked" means

**Phase 12.1 is the sole authoritative production model.** It is frozen: not retrained, not modified, regardless of any later exploratory result. Promotion of any newer candidate requires a successful external-validation study (see [`docs/external_validation_handoff.md`](docs/external_validation_handoff.md)) — internal re-evaluation alone, however promising, is never sufficient.

| | |
|---|---|
| **Cohort** | 547 cleaned CTU-UHB intrapartum recordings (PhysioNet, patient-grouped 5-fold CV + an 83-patient held-out internal test partition) |
| **Primary endpoint** | Umbilical arterial pH ≤ 7.15 (110 positive, 20.1% prevalence) |
| **Architecture** | `LogisticRegression(C=0.05)` on a 40-D state-trajectory feature vector — domain severities, FIGO state, trajectory operators (velocity/persistence/reversal), occupancy proportions, a frozen Huber-ensemble risk feature, and EWMA risk. **No raw-signal deep learning; fully interpretable, linear in its final layer.** |
| **Input window** | 20-minute causal window (4800 samples @ 4 Hz), 2.5-minute stride, strictly historical (no post-delivery data) |
| **Preprocessing** | Spike removal → cubic-spline gap interpolation → lowpass filter → iterative baseline estimation (`src/preprocessing/pipeline_clinical.py`) |
| **Locked CV AUROC (delivery)** | **0.6872** |
| **Locked held-out test AUROC (delivery)** | **0.6497** |
| **Locked CV AUROC (≥30 min before delivery)** | **0.5828** |

Every number above is reproducible from committed artifacts — see `results/phase13/audit/p6_predictions.npz` (frozen window-level scores this whole project's downstream evaluation is built on) and `models/external_validation_handoff/` (portable frozen artifacts: `p6_final_classifier.joblib` + `p6_final_scaler.joblib`, fit once on all 547 patients for external deployment, plus a `manifest.json` with exact application instructions and provenance). The full raw-signal-to-prediction pipeline is documented step by step in [`docs/external_validation_handoff.md`](docs/external_validation_handoff.md) § 3.

Accuracy is not a meaningful metric at this prevalence (a majority-class baseline scores ~80%) — AUROC/AUPRC and patient-level bootstrap confidence intervals are used throughout instead.

---

## 🔬 Post-lock investigation: candidates under evaluation, none promoted

After Phase 12.1 was locked, a further, strictly non-destructive investigation (Phases 13–17, all committed on the `final_synthesis_models` branch) asked whether *aggregating* P6's own frozen window-level scores differently — never retraining P6 itself — could add information. Four candidates emerged, each tested with patient-level bootstrap significance, leakage audits, and (where relevant) permutation/shuffled-time controls:

| Candidate | Mechanism | Status |
|---|---|---|
| **Model 3** | Trainable causal-attention pooling over P6's window scores (magnitude + temporal position) | CV-significant vs. P6 at delivery, but not shown to beat cheap fixed aggregators or resolved as using genuine temporal information — see below |
| **Parity Fusion** | Logit-fusion of P6 with a univariate maternal-parity model (admission-time, independent covariate) | Significant on the held-out internal test partition at every horizon; the most robustness-checked candidate |
| **P90 pooling** | Fixed 90th-percentile pooling of P6's window scores (no training) | Directionally positive, not confirmed at a pre-registered bar |
| **Model 3 + Parity Hybrid** | Logistic fusion of Model 3 and Parity Fusion | Promising but inconclusive after an independent reconciliation audit corrected a patient-inclusion bug in its first report |

**None of these is promoted, deployed, or clinically usable.** All four remain "worth external validation" — the full current status, exact numbers, and every robustness check is in [`reports/candidate_models_metrics_reference.md`](reports/candidate_models_metrics_reference.md), with the pre-registered hardening protocol in [`docs/pre_external_validation_hardening_plan.md`](docs/pre_external_validation_hardening_plan.md) and the external-validation study design in [`docs/external_validation_handoff.md`](docs/external_validation_handoff.md).

**Model 3 specifically is not yet confirmed to work via genuine temporal-position learning** — three independent hardening checks (`reports/candidate_models_metrics_reference.md` §1.2a–§1.2c) found it statistically indistinguishable from cheap fixed aggregators, sharing their recording-duration confound, and not clearing its shuffled-time control at the pre-registered bar. Its original result against the P6 baseline stands unretracted; the mechanism claim above it does not yet.

---

## 🗂 Superseded exploratory work (not locked, not delivered)

Early project phases (roughly Phase 1–4) benchmarked seven raw-signal temporal deep-learning encoders (1D CNN, BiLSTM, GRU, TCN, Multi-Scale LSTM, PatchCTG, PatchTST) and, later, a CNN-transformer cross-attention architecture ("CrossFormer" / "Model 8", multi-task with FIGO-consistency losses). Checkpoints, run logs, and technical documentation for this track are kept for reproducibility (`checkpoints/ctg_crossformer_crp/`, `docs/model8_crossformer_run_history.md`, `docs/model8_technical_documentation.md`) but **this track is not what is locked or delivered**:

- A patient-outcome-conditioned window-stride leak was found in the data substrate this track was originally benchmarked on (window count alone predicted the label at AUROC 0.9947) — see `docs/model8_crossformer_run_history.md` Part C.
- The track's own headline number (CV 0.8565 / test 0.8653) never reproduced across 10 independent reruns (mean ≈0.78) and the underlying published paper's number also failed independent reproduction.
- Evaluated honestly at the patient level (the level the clinical outcome is actually defined at) on leak-free data, this architecture family scored **below** the interpretable clinical-feature baseline in a strict, leakage-controlled head-to-head (`reports/cwt/prior_art_reconciliation.md`, Table B: CTG-CrossFormer 0.6167 vs. clinical logistic regression 0.7268) — a data-scarcity / model-capacity mismatch on this cohort's size (547 patients, ~110 positives), not a tuning failure.

Entry points from this earlier track (`scripts/run_clinical_review.py`, `scripts/demo_inference.py`, `scripts/eval_crp_metrics.py`, `scripts/calibrate_crp.py`) still run against `checkpoints/ctg_crossformer_crp/` and are kept for archival reproducibility of that historical benchmark — **they do not reflect the locked production model** described above.

---

## 🚀 Getting started

```bash
git clone https://github.com/MahadhevanS/CTG-Fetal-Distress-Prediction.git
cd CTG-Fetal-Distress-Prediction
python -m venv venv
# Windows:  .\venv\Scripts\Activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

**Reproduce the locked model's preprocessing pipeline:**
```bash
python src/preprocessing/run_all.py            # end-to-end signal processing, patient-stratified windowing
python src/preprocessing/consistency_audit.py  # 10-point mathematical consistency audit on generated tensors
```

**Apply the frozen, portable Phase 12.1 artifacts** (fit once on all 547 patients, for external-cohort deployment): load `models/external_validation_handoff/p6_final_scaler.joblib` + `p6_final_classifier.joblib` on a 40-D state-trajectory feature vector built per `docs/external_validation_handoff.md` § 3 — exact application instructions and provenance (SHA of training data, exact feature construction scripts) are in `models/external_validation_handoff/manifest.json`.

**Reproduce the post-lock candidate investigation** (Model 3 / Parity Fusion / P90 / Hybrid): scripts under `scripts/phase13_*.py`, `scripts/phase14_*.py`, `scripts/phase15_*.py`, `scripts/phase16_*.py`, and `scripts/model3_parity_hybrid/`; see [`docs/phase16_protocol.md`](docs/phase16_protocol.md) for the frozen pre-registration each was run against.

---

## 📊 Dataset Citations & Links

1. **PhysioNet CTU-CHB Intrapartum CTG Database**:
   - 552 intrapartum CTG recordings (sampling rate 4.0 Hz) paired with clinical outcomes (umbilical artery pH, Apgar scores, delivery mode).
   - **DOI**: [10.13026/C2188R](https://doi.org/10.13026/C2188R)
   - **Citation**: Chudáček V. et al., *Open access intrapartum CTG database*, BMC Pregnancy and Childbirth, 2014.

2. **UCI Machine Learning Repository — Cardiotocography Dataset**:
   - 2,126 pre-extracted 21-feature SisPorto 2.0 records — evaluated as a candidate external-validation source and found unusable (no umbilical pH or acid-base outcome field); see `docs/external_validation_handoff.md` § 4.
   - **Link**: [UCI Cardiotocography Repository](https://archive.ics.uci.edu/dataset/193/cardiotocography)

---

## 📂 Project Structure

```text
CTG-Fetal-Distress-Prediction/
│
├── configs/          # YAML configs, including configs/model3_parity_hybrid.yaml
├── checkpoints/       # Saved weights, incl. the archival CrossFormer/CRP track
├── models/
│   ├── continuous_clinical_huber/       # Frozen 5-fold Huber risk ensemble (one P6 input feature)
│   └── external_validation_handoff/     # Frozen, portable P6/Model3/Parity artifacts + manifest
├── data/
│   ├── raw/                  # Raw PhysioNet CTU-CHB & UCI SisPorto data
│   └── processed_clinical/   # Locked patient folds (folds.json) & tensors for the clinical pipeline
├── docs/              # Protocols, pre-registrations, handoff/hardening plans (51 files)
├── reports/            # Phase-by-phase findings, closure summaries, candidate metrics reference (53 files)
├── results/            # Per-phase result CSVs/JSONs (regenerable from committed scripts)
├── scripts/            # phase1–phase17 pipeline scripts, incl. scripts/model3_parity_hybrid/
├── src/
│   ├── preprocessing/   # Filtering, baseline extraction, SQA, splitting
│   ├── knowledge/       # FIGO rule engine & clinical feature/descriptor extraction
│   ├── evaluation/       # Shared horizon-selection & fusion primitives (src/evaluation/phase13_common.py)
│   ├── models/           # Phase 16 causal-attention aggregator + checkpoint utilities
│   └── training/         # Cross-validation protocol
├── AI_AGENT_RULES.md   # Contributor guidelines & patent boundaries
└── SANITY_CHECK_REVIEW.md
```

---

## 📜 License & Patent Boundaries

This repository is licensed under the MIT License. Signal-processing and feature-extraction implementations maintain non-infringement boundary conditions relative to GE Patent US12094611B2 — see `AI_AGENT_RULES.md` for the specific constraints this project designs under.
