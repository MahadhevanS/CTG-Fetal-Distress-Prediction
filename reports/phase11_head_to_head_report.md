# Phase 11 — Head-to-Head Comparative Benchmark & Superiority Report

## 1. Executive Summary

This report documents the direct head-to-head comparative benchmark between the proposed physiology-guided trajectory framework (**P6**) and representative published prior-art baselines (**P1: DeepCTG-inspired compact baseline**, **P2: Vargas-Calixto-inspired sequential event baseline**, **P3: Locked snapshot baseline**, **P4: State ablation**, **P5: Trajectory ablation**).

All models were evaluated under identical patient-stratified 5-fold cross-validation on the CTU-UHB cohort ($N=547$ patients, $N_{\text{acidemia}}=110$ at $\text{pH} \le 7.15$, $8,517$ rolling windows).

---

## 2. Multi-Horizon Discrimination Benchmark (pH $\le 7.15$)

| Model Code | Model Description | $\ge 60$m | $\ge 45$m | $\ge 30$m | $\ge 20$m | $\ge 10$m | Delivery ($0$m) | Delivery AUPRC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P1** | DeepCTG-Inspired Compact CTG Baseline | $0.5090$ | $0.5090$ | $0.5727$ | $0.5693$ | $0.5595$ | $0.5090$ | $0.2333$ |
| **P2** | Vargas-Calixto-Inspired Sequential Baseline | $0.6774$ | $0.6774$ | **$0.5983$** | $0.5841$ | $0.5726$ | $0.6774$ | $0.3582$ |
| **P3** | Locked Snapshot Baseline (Continuous Huber) | $0.5976$ | $0.5976$ | $0.5461$ | $0.5500$ | $0.5360$ | $0.5976$ | $0.2967$ |
| **P4** | Snapshot + Physiological State ($R_t + S_t$) | $0.5945$ | $0.5945$ | $0.5459$ | $0.5505$ | $0.5360$ | $0.5945$ | $0.2893$ |
| **P5** | Snapshot + Deterioration Trajectory | $0.6142$ | $0.6142$ | $0.5355$ | $0.5489$ | $0.5327$ | $0.6142$ | $0.2982$ |
| **P6** | **Full Proposed Physiology-Guided System** | **$0.6872$** | **$0.6872$** | **$0.5857$** | **$0.5869$** | **$0.5814$** | **$0.6872$** | **$0.4067$** |

---

## 3. Pairwise Superiority Bootstrap Analysis ($B=2,000$ Clustered Patient Resamples)

$$\Delta \text{AUROC} = \text{AUROC}_{\text{Model B}} - \text{AUROC}_{\text{Model A}}$$

### 3.1 Delivery Endpoint (0m) Comparisons

| Comparison | Model B vs Model A | Model B AUROC | Model A AUROC | $\Delta \text{AUROC}$ | $95\%$ Bootstrap CI | Empirical $p$-value | Predefined Evidence Level |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **P6 vs P1** | Proposed Full System vs Compact CTG | $0.6872$ | $0.5090$ | $\mathbf{+0.1801}$ | $[+0.1111, +0.2475]$ | $\mathbf{p = 0.000}$ | **Level 1: Statistically Significant Superiority (Outperformed)** |
| **P6 vs P2** | Proposed Full System vs Sequential Event | $0.6872$ | $0.6774$ | $\mathbf{+0.0103}$ | $[-0.0343, +0.0523]$ | $p = 0.633$ | Level 2: Higher Point Estimate Without Significance |
| **P6 vs P3** | Proposed Full System vs Snapshot Baseline | $0.6872$ | $0.5976$ | $\mathbf{+0.0896}$ | $[+0.0349, +0.1470]$ | $\mathbf{p = 0.000}$ | **Level 1: Statistically Significant Superiority (Outperformed)** |
| **P5 vs P3** | Snapshot + Trajectory vs Snapshot Baseline | $0.6142$ | $0.5976$ | $\mathbf{+0.0165}$ | $[+0.0006, +0.0319]$ | $\mathbf{p = 0.041}$ | **Level 1: Statistically Significant Superiority (Outperformed)** |
| **P4 vs P3** | Snapshot + State vs Snapshot Baseline | $0.5945$ | $0.5976$ | $-0.0031$ | $[-0.0080, +0.0018]$ | $p = 0.219$ | Level 3: Comparable / No Significant Difference |
| **P6 vs P5** | Proposed Full System vs Snapshot + Traj | $0.6872$ | $0.6142$ | $\mathbf{+0.0731}$ | $[+0.0192, +0.1292]$ | $\mathbf{p = 0.006}$ | **Level 1: Statistically Significant Superiority (Outperformed)** |

### 3.2 Primary Early-Warning Horizon ($\ge 30$m) Comparisons

| Comparison | Model B vs Model A | Model B AUROC | Model A AUROC | $\Delta \text{AUROC}$ | $95\%$ Bootstrap CI | Empirical $p$-value | Predefined Evidence Level |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **P6 vs P1** | Proposed Full System vs Compact CTG | $0.5857$ | $0.5727$ | $\mathbf{+0.0127}$ | $[-0.0481, +0.0775]$ | $p = 0.699$ | Level 2: Higher Point Estimate Without Significance |
| **P6 vs P2** | Proposed Full System vs Sequential Event | $0.5857$ | $0.5983$ | $-0.0127$ | $[-0.0542, +0.0278]$ | $p = 0.560$ | Level 3: Comparable Performance |
| **P6 vs P3** | Proposed Full System vs Snapshot Baseline | $0.5857$ | $0.5461$ | $\mathbf{+0.0390}$ | $[-0.0138, +0.0912]$ | $p = 0.150$ | Level 2: Higher Point Estimate Without Significance |
| **P5 vs P3** | Snapshot + Trajectory vs Snapshot Baseline | $0.5355$ | $0.5461$ | $-0.0108$ | $[-0.0428, +0.0185]$ | $p = 0.507$ | Level 3: Comparable Performance |
| **P4 vs P3** | Snapshot + State vs Snapshot Baseline | $0.5459$ | $0.5461$ | $-0.0000$ | $[-0.0151, +0.0144]$ | $p = 0.998$ | Level 3: Comparable Performance |
| **P6 vs P5** | Proposed Full System vs Snapshot + Traj | $0.5857$ | $0.5355$ | $\mathbf{+0.0498}$ | $[+0.0020, +0.0981]$ | $\mathbf{p = 0.040}$ | **Level 1: Statistically Significant Superiority (Outperformed)** |

---

## 4. Key Findings

1. **Delivery Endpoint Superiority**:
   The full physiology-guided framework (**P6**) achieves **statistically significant superiority** over the DeepCTG-inspired compact baseline (**P1**, $\Delta \text{AUROC} = +0.1801, p < 0.001$) and the locked snapshot baseline (**P3**, $\Delta \text{AUROC} = +0.0896, p < 0.001$).
2. **Trajectory Value Added**:
   Adding trajectory dynamics to snapshot prediction (**P5 vs P3**) provides statistically significant improvement ($\Delta \text{AUROC} = +0.0165, p = 0.041$).
3. **Early-Warning Horizon ($\ge 30$m)**:
   At $\ge 30$ minutes before delivery, P6 achieves higher point-estimate discrimination than P1 and P3 ($\Delta = +0.0127$ to $+0.0390$), while remaining comparable to P2 ($\Delta = -0.0127, p = 0.560$). Furthermore, full multi-domain context significantly outperforms trajectory-alone at $\ge 30$m (**P6 vs P5**: $\Delta = +0.0498, p = 0.040$).

# End of Head-to-Head Report
