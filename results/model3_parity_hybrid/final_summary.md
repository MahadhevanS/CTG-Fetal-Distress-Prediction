# Final Summary Report: Model 3 + Parity Multimodal Hybrid Investigation

> ## ✅ CORRECTED AND RERUN — 2026-09-13
> This report was regenerated after an independent reconciliation audit
> ([`RECONCILIATION_AUDIT.md`](RECONCILIATION_AUDIT.md)) found that
> `scripts/model3_parity_hybrid/hybrid_engine.py` contained a
> patient-dropping bug in its horizon-eligibility filter (silently
> excluded patients lacking an eligible window at each horizon >0, instead
> of falling back to their last window per this project's canonical
> convention), a hardcoded fusion λ=1.4 in the Parity Fusion operational
> sequence instead of each fold's own tuned value, and a degenerate
> duration-tertile split (70% of patients tied at the window-count
> ceiling collapsed into a single stratum under a plain percentile cut).
> All three are fixed in the script (see inline comments at the fix
> sites), and the **entire pipeline was rerun end to end** — nothing
> below is hand-edited; every number is a direct read of the regenerated
> CSVs. Table 1 (delivery) and the ablation ladder (Table 4) were
> confirmed unaffected by the bugs and are numerically identical to the
> original run, as expected. Table 2, Table 3's Parity Fusion row, the
> duration-sensitivity table, and several interpretive claims in §20.4
> changed materially and are corrected below. The classification
> (**Promising but Inconclusive**) is unchanged, but the evidence
> supporting it is now narrower than originally claimed — see §20.4 and
> §20.6.

**Date:** 2026-09-13 (original run), corrected and rerun 2026-09-13
**Status:** Completed — reconciled
**Repository:** `MahadhevanS/CTG-Fetal-Distress-Prediction`
**Git Commit:** `72a5f60fb0d932d2624d5c55e2869f81087adad5`
**Configuration Hash:** `b94425d0c0005da3`
**Investigation Classification:** **Promising but Inconclusive (Promote to External-Validation Candidate alongside Model 3 and Parity Fusion)**

---

## 20.1 Executive Summary

1. **What was tested:** A leakage-controlled multimodal late-fusion hybrid model combining CTG-derived temporal trajectory risk scores from Phase 16 Model 3 ($z_{\text{M3}}$) with admission maternal parity ($z_{\text{parity}}$) via a small, regularized logistic fusion layer ($\sigma(\beta_0 + \beta_1 z_{\text{M3}} + \beta_2 z_{\text{parity}})$ with L2 penalty $C=0.1$).
2. **Implementation correctness:** Full pipeline implemented strictly from locked project contracts. 27 automated assertions passed with 0 failures — but note (per the reconciliation audit) that the assertions checking "identical patient inclusion across all models" at each horizon validated internal self-consistency of the (now-fixed) patient-dropping bug, not correctness against the canonical/locked convention. The test suite's coverage gap, not its execution, is why the bug was not caught before this correction.
3. **Leakage audit:** Passed with zero violations. Out-of-fold predictions were used for fusion training; parity scaling and logistic models were fitted exclusively on outer-training patients; no outcome, patient ID, window count, or duration entered feature matrices.
4. **Improvement over Model 3** (canonical patient set: N=547 CV / N=83 test at every horizon):
   - *Delivery (Primary):* Hybrid achieves CV AUROC **0.7324** vs Model 3 **0.7216** ($\Delta = +0.0108, p = 0.172$, 95% CI: $[-0.0049, +0.0262]$) — **not significant on CV.** On the held-out internal test partition, Hybrid achieves **0.6916** vs Model 3 **0.6729** ($\Delta = +0.0187, p = 0.030$) — significant.
   - *Early Warning (Exploratory):* **Two of three pre-delivery horizons reach CV significance, not three.**
     - $\ge 10$m: 0.6896 vs 0.6683 ($\Delta = +0.0213, p = 0.097$, CI: $[-0.0034, +0.0452]$) — **not significant.**
     - $\ge 20$m: 0.6481 vs 0.6015 ($\Delta = +0.0467, p = 0.005$, CI: $[+0.0119, +0.0810]$) — significant.
     - $\ge 30$m: 0.6451 vs 0.5976 ($\Delta = +0.0475, p = 0.013$, CI: $[+0.0109, +0.0827]$) — significant.
5. **Improvement over Parity Fusion alone:** At delivery, Hybrid achieves 0.7324 vs Parity Fusion 0.7094 ($\Delta = +0.0229, p = 0.093$, not significant). **At every early-warning horizon, the Hybrid is not significantly better than Parity Fusion alone** ($\ge$10m: $\Delta=-0.0112, p=.357$; $\ge$20m: $\Delta=+0.0047, p=.852$; $\ge$30m: $\Delta=+0.0083, p=.668$) — Parity Fusion alone already captures most of the early-warning signal; Model 3's marginal contribution on top of it is not established at any horizon.
6. **Improvement over Baseline (P6):** Statistically significant at delivery: 0.7324 vs 0.6872 ($\Delta = +0.0452, p = 0.004$, CI: $[+0.0138, +0.0782]$).
7. **Operational Behavior:** At 80% delivery target sensitivity, the Hybrid achieves the lowest reported false-alert rate **among the four models evaluated in this same table, under this project's Phase-16 patient-level "ever alerted" convention** (**65.2%** vs Model 3's 69.3%, Parity Fusion's 71.2%, and P6's 83.1%) — but achieved sensitivity differs across models (88.2%/88.2%/90.9%/92.7% respectively), so this is not a sensitivity-controlled comparison, and it is **not** comparable to Phase 15's Max/P90 operational figures, which use a different FAR definition.
8. **Statistical Conclusiveness:** The primary delivery gain over Model 3 ($\Delta = +0.0108$) is numerically positive but not significant on CV ($p = 0.172$). The internal-test delivery gain and the ≥20m/≥30m CV gains are significant. The ≥10m gain is not. 0/30 parity permutation controls reached the true delta, supporting a non-random parity-associated signal — this does not establish biological causation.
9. **External Validation Requirement:** External validation remains strictly mandatory. No claim of clinical readiness or deployment readiness is made.

---

## 20.2 Methods

- **Cohort & Endpoint:** 547 cleaned CTU-UHB intrapartum recordings. Primary endpoint: umbilical arterial $\text{pH} \le 7.15$ ($N=110, 20.1\%$). Secondary severe endpoint: $\text{pH} \le 7.05$ ($N=41, 7.5\%$).
- **Patient-Level Evaluation:** Clustered patient grouping; all windows from a patient reside in the same fold. **Every model is evaluated over the full canonical patient set (N=547 CV / N=83 test) at every horizon — no patient is ever excluded**, matching `src.evaluation.phase13_common.get_patient_scores_at_horizon_corrected`'s and `eligible_prefix_length`'s fallback-to-last-window convention used throughout this project.
- **CTG Preprocessing & Representation:** 4 Hz sampling; 4800-sample (20-min) rolling windows; 600-sample (2.5-min) stride; causal filtering (spike removal, cubic interpolation, bandpass, baseline correction).
- **Model 3 Inputs:** Attention logit $e_t = f(r_t, \text{elapsed}_t/60)$ with a 2→8→1 MLP (33 parameters) over frozen P6 window scores $r_t$. Elapsed time is measured causally from that patient's own first retained window.
- **Parity Encoding:** Univariate $\text{LogisticRegression}(C=1.0)$ fit on training patients' standardized parity: $p_{\text{parity}} = \sigma(\theta_0 + \theta_1 \cdot \text{StandardScaler}(\text{parity}))$, transformed via $\text{to\_logit}(p_{\text{parity}})$.
- **Fusion Architecture:** Regularized logistic regression: $\text{logit}(p_H) = \beta_0 + \beta_1 z_{\text{M3}} + \beta_2 z_{\text{parity}}$ where features are standardized on training folds prior to fitting with $C=0.1$.
- **Validation Protocols:**
  - Primary: 5-fold patient-grouped cross-validation ($N=547$, canonical `folds.json`).
  - Held-out internal test partition: $N=83$ patients (evaluated using `Model_3_magnitude_position_testmodel.pt`).
- **Statistical Inference:** Paired patient bootstrap ($B=2000$ replicates, seed 42) for $\Delta \text{AUROC}$, 95% percentile CIs, and two-sided empirical $p$-values; DeLong test where applicable.
- **Horizon Selection:** Canonical `get_patient_scores_at_horizon_corrected` (nearest window satisfying $t_i \ge h$). Horizons: Delivery ($h=0$), $\ge 10$m ($h=10$), $\ge 20$m ($h=20$), $\ge 30$m ($h=30$).
- **Operational Metrics:** Rolling prefix evaluation; threshold selected per fold on training positive cases to achieve 80% sensitivity at delivery; per-fold Parity Fusion sequences now use that fold's own trained λ (previously hardcoded to 1.4); reports FAR, lead time, and detection percentages. Patient-level "ever alerted" convention (Phase 16), not comparable to Phase 15's single-evaluation-point convention.

---

## 20.3 Results

### Table 1: Primary Delivery Discrimination Metrics (unaffected by the bug fixes — identical to the original run)

| Model | CV AUROC | CV AUPRC | Prevalence Lift | $\Delta$ vs. Baseline P6 | CV $p$ (boot / DeLong) | CV 95% CI | Test AUROC | Test $\Delta$ | Test $p$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P6 (Baseline)** | 0.6872 | 0.4067 | 2.02× | — | — | — | 0.6497 | — | — |
| **Max Pooling** | 0.7210 | 0.4115 | 2.05× | +0.0338 | .140 / .145 | [−0.0102, +0.0778] | 0.7130 | +0.0633 | .156 |
| **P90 Pooling** | 0.7178 | 0.4264 | 2.12× | +0.0307 | .225 / .231 | [−0.0189, +0.0822] | 0.6622 | +0.0125 | .797 |
| **Parity Fusion** | 0.7094 | 0.3935 | 1.96× | +0.0223 | .172 / .159 | [−0.0089, +0.0516] | 0.7148 | +0.0651 | .025 |
| **Model 3** | 0.7216 | 0.4335 | 2.16× | +0.0344 | .014 / .018 | [+0.0064, +0.0649] | 0.6729 | +0.0232 | .157 |
| **Hybrid (M3+Parity)**| **0.7324** | **0.4356** | **2.17×** | **+0.0452** | **.004 / .006** | **[+0.0138, +0.0782]** | **0.6916** | **+0.0419** | **.030** |

Per-fold delivery AUROC for Hybrid: `[0.7438, 0.7639, 0.7868, 0.6850, 0.6949]` (Mean: 0.7349, Std: 0.0393). On the internal test partition, the ranking is Parity Fusion (0.7148) > Max (0.7130) > **Hybrid (0.6916)** > Model 3 (0.6729) > P90 (0.6622) > P6 (0.6497) — the Hybrid leads on CV but is third on the internal test partition; the two rankings disagree.

### Table 2: Early-Warning Performance (AUROC Across Horizons) — CORRECTED, N=547/83 at every horizon

| Horizon | $N$ (Pos / Neg) | Hybrid AUROC | Model 3 AUROC | $\Delta$ (Hyb − M3) | CV $p$ | Parity Fusion AUROC | P6 AUROC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Delivery** | 547 (110 / 437) | **0.7324** | 0.7216 | +0.0108 | .172 | 0.7094 | 0.6872 |
| **$\ge 10$m** | 547 (110 / 437) | 0.6896 | 0.6683 | +0.0213 | .097 (n.s.) | 0.7008 | 0.6859 |
| **$\ge 20$m** | 547 (110 / 437) | **0.6481** | 0.6015 | **+0.0467** | **.005** | 0.6435 | 0.6238 |
| **$\ge 30$m** | 547 (110 / 437) | **0.6451** | 0.5976 | **+0.0475** | **.013** | 0.6368 | 0.5828 |

Model 3's own canonical early-warning numbers (0.6683 / 0.6015 / 0.5976) now match the previously locked project record exactly. Model 3 was never below P6 at ≥30m (0.5976 vs. 0.5828) — the earlier appearance of a "collapse" was an artifact of the patient-dropping bug depressing Model 3's own denominator differently from the other models. The Hybrid's early-warning gains over Model 3 are real at ≥20m/≥30m but roughly half the magnitude previously claimed, and the ≥10m claim does not survive at all.

### Table 3: Operational Behavior (80% Delivery Sensitivity Operating Point) — CORRECTED, Parity Fusion row (λ fix)

| Model | Achieved Sens | FAR | Median Lead (min) | IQR (min) | $\% \ge 20$m Detected | $\% \ge 30$m Detected |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P6** | 92.7% | 83.1% | 35.0 | 22.5 | 71.6% | 59.8% |
| **Model 3** | 88.2% | 69.3% | 32.5 | 27.5 | 66.0% | 54.6% |
| **Parity Fusion** | 90.9% | 71.2% | 40.0 | 15.0 | 80.0% | 68.0% |
| **Hybrid** | **88.2%** | **65.2%** | **35.0** | **25.0** | **70.1%** | **59.8%** |

Only the Parity Fusion row changed from the original run (FAR 72.1%→71.2%, IQR 37.5→15.0, ≥20m 84.0%→80.0%, ≥30m 70.0%→68.0%) after replacing the hardcoded λ=1.4 with each fold's own tuned value. All four rows use the identical operational evaluator (per-fold training-only threshold selection, patient-level "ever alerted" false-alert counting) — comparisons **within this table** are apples-to-apples; comparisons to Phase 15's Max/P90 figures are not, and are not made here.

### Table 4: Systematic Ablation Ladder (Delivery CV) — unaffected by the bug fixes

| ID | Description | AUROC | $\Delta$ vs. Model 3 | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | P6 Baseline | 0.6872 | −0.0344 | Locked single-window baseline |
| **A2** | Max Pooling | 0.7210 | −0.0006 | Zero-training magnitude benchmark |
| **A3** | P90 Pooling | 0.7178 | −0.0038 | Zero-training quantile benchmark |
| **A4** | Model 3 | 0.7216 | 0.0000 | CTG temporal trajectory candidate |
| **A5** | Parity Fusion | 0.7094 | −0.0122 | Baseline multimodal model |
| **A6** | **Primary Hybrid** | **0.7324** | **+0.0108** | Full late-fusion model |
| **A7** | Hybrid: Model 3 only | 0.7216 | 0.0000 | Confirms fusion layer does not inflate single-input |
| **A8** | Hybrid: Parity only | 0.5898 | −0.1318 | Independent admission prior |
| **A9** | Raw Probabilities (untransformed) | 0.7319 | +0.0103 | Logit transformation provides small numerical advantage |
| **A10**| Interaction Model ($z_{\text{M3}} \times z_{\text{parity}}$) | 0.7348 | +0.0132 | Marginal $+0.0024$ gain over A6 ($p=.070$); non-essential |

### Table 5: Robustness Controls Summary — corrected transcription, calibration numbers now read directly from `calibration_metrics.csv`

- **Bootstrap Seed Invariance:** $\Delta \text{AUROC} = +0.0107\text{--}+0.0108$ across 5 seeds ($p = .162\text{--}.183$). Unaffected by the fixes.
- **Fold-Resplit Stability:** Evaluated on 5 independent 5-fold stratified resplits (seeds 11, 22, 33, 44, 55). Hybrid AUROC remained within **$[0.7286, 0.7318]$** (corrected from a previously mis-transcribed $[0.7289, 0.7341]$, which did not match the underlying CSV). $\Delta$ vs. Model 3 was consistently positive ($+0.0070$ to $+0.0102$) but **not statistically significant in any of the 5 resplits** (all $p \ge .231$) — a stable point estimate is not evidence of a stable improvement; the CIs are the relevant evidence and they are uniformly inconclusive.
- **Parity Permutation Control:** Across 30 patient-level permutations of parity, **0 of 30** exceeded the real delta ($+0.0108$). Mean permuted delta was $-0.0039$ (corrected from a previously mis-transcribed $-0.0012$). This supports a non-random parity-associated signal under this permutation procedure; it does not establish biological causation, and cannot rule out cohort-specific confounding or other structural explanations.
- **Calibration** (read directly from `calibration_metrics.csv`; the previous version of this report quoted numbers that did not match this file): Out-of-fold Brier score — Model 3 **0.1424**, Parity Fusion **0.1708**, Hybrid **0.1403** (lowest of the three, as previously claimed, but not at the previously claimed values). Calibration slope/intercept — Model 3: 1.288 / 0.1051; Parity Fusion: 0.7922 / 1.0009; Hybrid: **1.1453 / 0.1749**. The Hybrid's slope is closer to 1 than Parity Fusion's but further from 1 than Model 3's; none of the three is "near-ideal" (slope 1.0, intercept 0.0) as the original report claimed for the Hybrid.
- **Duration Sensitivity** (tertile stratification bug fixed — see Table 6 below; the original 3-row tertile table did not exist in the saved artifact and has been replaced).

### Table 6: Duration/Observation-Opportunity Stratification — CORRECTED (was degenerate: 1 row, n=547, no stratification)

70% of patients (381/547) sit at the window-count ceiling (17 windows), which collapses a plain percentile-tertile split into one bin. Restratified as below-median / above-median-sub-ceiling / ceiling:

| Stratum | $N$ | Positives | Model 3 AUROC | Hybrid AUROC | $\Delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Low (≤13 windows) | 103 | 25 | 0.6887 | 0.7041 | +0.0154 |
| Mid (14–16 windows) | 63 | 13 | 0.7185 | 0.7200 | +0.0015 |
| High (ceiling, 17 windows) | 381 | 72 | 0.7312 | 0.7486 | +0.0174 |

The Hybrid maintains a small positive edge over Model 3 in all three strata (this part of the original claim's *direction* holds, even though the original numbers were fabricated relative to their own source file) — the edge is smallest in the middle stratum and largest in the two extremes, with no clear monotonic duration trend either way. This is a much weaker and more heterogeneous finding than a clean "three-stratum win," and should be reported as such.

---

## 20.4 Scientific Interpretation — revised throughout to match the corrected tables

### 1. Does parity add information beyond Model 3?
**Modestly, and only clearly established at ≥20m/≥30m, not at delivery or ≥10m.**
At delivery, parity adds $+0.0108$ AUROC over Model 3 ($p=0.172$ on CV — not significant; $p=0.030$ on internal test — significant, but a single 83-patient partition). At ≥20m and ≥30m, adding parity to Model 3 produces significant CV gains of $+0.0467$ ($p=.005$) and $+0.0475$ ($p=.013$) — real, but roughly half the magnitude originally claimed, and Model 3's own early-horizon numbers were never as weak as originally reported. At ≥10m, the gain ($+0.0213$) is not significant ($p=.097$). Permutation testing (0/30) supports the addition being a non-random, parity-associated signal — it does not, by itself, establish that the mechanism is causal or biological.

### 2. Does Model 3 add information beyond parity fusion?
**Only clearly at delivery; not established at any early-warning horizon.**
At delivery, adding Model 3 to Parity Fusion raises AUROC from $0.7094$ to $0.7324$ ($\Delta = +0.0229, p = 0.093$ on CV — not significant at conventional thresholds, though closer than the reverse direction). **At every early-warning horizon, adding Model 3 to Parity Fusion does not produce a significant improvement** (≥10m: $\Delta=-0.0112, p=.357$; ≥20m: $\Delta=+0.0047, p=.852$; ≥30m: $\Delta=+0.0083, p=.668$). Parity Fusion alone already captures nearly all of the Hybrid's early-warning performance.

### 3. Does the hybrid improve early warning?
**Partially. This is a real but more modest finding than originally reported, and it is better described as "the fused model inherits Parity Fusion's early-warning strength" than "the hybrid synthesizes and exceeds both components."**
Model 3 standalone was never as catastrophically weak at early horizons as originally reported — its canonical ≥20m/≥30m AUROCs (0.6015/0.5976) are comparable to Max and P90 at those same horizons, and ≥30m is actually slightly above P6. Parity Fusion alone is strong at early horizons (0.6435/0.6368). The Hybrid essentially matches Parity Fusion alone at ≥20m/≥30m (0.6481/0.6451, deltas over Parity Fusion of only +0.0047/+0.0083, neither significant) while significantly exceeding Model 3 alone at those same horizons. The correct reading: **parity carries the early-warning performance; Model 3 does not add to it detectably at this sample size.**

### 4. Is the hybrid better than Max or P90?
**Not established at conventional significance on CV; operational comparisons must stay within the single evaluator used in this table.**
At delivery, Hybrid (0.7324) numerically exceeds Max (0.7210, $\Delta=+0.0114, p=.449$) and P90 (0.7178, $\Delta=+0.0146, p=.369$) — neither difference is significant. On the internal test partition, the Hybrid actually trails both Max (0.7130) and Parity Fusion (0.7148). **The claim that the Hybrid "dominates" operationally by comparison to "Max's 51.5% FAR in Phase 15" is withdrawn** — that figure comes from a different operational-metric definition (Phase 15's single-evaluation-point convention vs. this table's Phase-16 patient-level "ever alerted" convention) and the two are not comparable under any convention established in this project. Within this table's own consistent evaluator, the Hybrid does have the lowest FAR of the four rows shown (65.2%), at 88.2% achieved sensitivity — a genuine but sensitivity-uncontrolled observation, not a proof of dominance.

### 5. Is the gain stable?
**The point estimate is stable; the improvement's statistical significance is not.**
Across 5 bootstrap seeds and 5 independent fold resplits, the Hybrid's raw AUROC is tightly clustered (0.7286–0.7324) and its coefficients keep the same sign in all 6 fits (Model 3: $[+0.658,+0.758]$, parity: $[+0.178,+0.301]$) — both of these hold up under audit. But the delta vs. Model 3 is not significant in any of the 5 independent resplits (all $p \ge .231$). A tight AUROC range and a consistently-signed coefficient are evidence the fitting procedure is stable — they are not, by themselves, evidence that the improvement over Model 3 is real rather than noise at this sample size.

### 6. Is there evidence of duration dependence?
**Parity is duration-independent ($r = -0.049, p = 0.25$); Model 3 possesses duration dependence ($r = -0.213, p < 0.0001$), consistent with the Tier-2 hardening finding already on record.**
The hybrid score itself shows a correlation with window count of $r = +0.016$ ($p = 0.71$) — not concerning. The corrected stratified analysis (Table 6) shows the Hybrid maintains a small positive edge over Model 3 in all three duration strata (Low +0.0154, Mid +0.0015, High +0.0174), but the effect is small and non-monotonic across strata, not the clean "wins in every stratum by a consistent margin" picture originally suggested by a table that did not actually exist in the saved output.

### 7. Is the temporal mechanism resolved?
**No.**
Fusing parity with Model 3 improves performance at some horizons, but it does not resolve the open scientific question from Tier-3 hardening regarding whether Model 3's attention mechanism genuinely decodes chronological progression versus acting as a non-linear percentile estimator (the shuffled-time control there was itself inconclusive — see `reports/candidate_models_metrics_reference.md` §1.2c). That mechanistic question remains open and is not addressed by this investigation.

### 8. Is the model ready for external validation?
**Ready to be *considered* for external validation, alongside — not instead of — Model 3 and Parity Fusion individually; not "ready" in the sense of a settled, decisive result.**
The pipeline is technically and procedurally sound (leakage audit passes on direct code inspection, coefficients are stable and consistently signed), and all preprocessing/causal boundaries are locked and documented. But the internal evidence for the Hybrid's incremental value over its own two components is weaker than originally reported: not significant over Model 3 at delivery or ≥10m, not significant over Parity Fusion at any early-warning horizon, and third (not first) on the internal test ranking. External validation should test the Hybrid as a genuine open question, not as a pre-confirmed improvement.

---

## 20.5 Limitations Summary

As detailed in [`limitations.md`](limitations.md), with one addition from this correction pass:
1. **Sample Size:** $N=547$ single-center recordings with only 110 primary events limit statistical power to prove small delivery and early-warning gains (multiple horizons now fail to reach $p<0.05$ under the corrected evaluation that passed under the buggy one).
2. **Internal-Only Cohort:** No validation on external clinical sites or differing monitoring equipment.
3. **Upstream P6 Dependence:** Inherits potential limitations of the frozen Phase 12.1 feature extractor.
4. **Non-Clinical Status:** The model is an investigative research candidate and must not be used in clinical practice.
5. **Report-generation reliability:** This investigation's first write-up contained a patient-inclusion bug plus at least three independent numeric transcription errors (calibration figures, duration-tertile table, resplit AUROC range) not caught by its own 27-assertion test suite, because none of those assertions checked narrative-vs-artifact consistency or cross-referenced the project's locked canonical values. Future reports from this pipeline should be diffed against their source CSVs before being treated as final.

---

## 20.6 Final Recommendation

In accordance with the pre-declared Decision Gates (Section 19):

### **Classification: Promising but Inconclusive (Promote to External-Validation Candidate)**

- **Rationale, updated to the corrected evidence:**
  1. The Hybrid clears Gate 1 (Data Integrity) and Gate 3 (Hybrid Validity — leakage audit) on direct code inspection. Gate 2 (Reference Reproducibility) passes for the base models it was checked against (delivery only); it did not previously check early-warning reproducibility, which is exactly where the bug was.
  2. At the primary delivery endpoint on 5-fold CV, the Hybrid achieves the highest AUROC in the repository (0.7324), but the difference over Model 3 ($\Delta = +0.0108$) does not clear $p < 0.05$ on cross-validation, and the Hybrid does not lead on the internal test partition (third, behind Parity Fusion and Max).
  3. The Hybrid significantly outperforms Model 3 at the held-out internal test delivery endpoint ($p = 0.030$), and at ≥20m ($p=.005$) and ≥30m ($p=.013$) on CV — genuine findings, at roughly half the magnitude originally reported, and not accompanied by a significant edge over Parity Fusion alone at those same horizons.
  4. It has the lowest FAR among the four models evaluated in Table 3's single, consistent operational evaluator (65.2%), at 88.2% achieved sensitivity — not a project-wide "lowest FAR" claim.
  5. 0/30 parity permutation controls reached the real delta, supporting a non-random parity-associated signal (not proof of biological causation).

**Actionable Next Step:**
Promote the **Model 3 + Parity Hybrid** alongside **Model 3** and **Parity Fusion** as three of the primary candidate systems to be tested in the external validation study scoped in `docs/external_validation_handoff.md` — carrying forward the corrected framing above (a promising but not yet internally-decisive addition, not a proven synthesis exceeding both of its components) rather than the original overstated claims. Max pooling and P90 pooling remain mandatory zero-training baselines per that document's existing scope.
