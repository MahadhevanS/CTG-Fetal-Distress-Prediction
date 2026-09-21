# Phase 11.5 — Prediction Discordance & Physiological Regimes Report

## CTU-UHB Head-to-Head Discordance: P6 (Full Framework) vs P2 (Sequential / Event Baseline)

### 1. Executive Summary

Phase 11 and Phase 11.5 demonstrated that while the Full Physiology-Guided Framework (P6) achieves superior discrimination at delivery ($\text{AUROC} = 0.6872$) and the longest warning lead time ($17.5$ min median), the Vargas-Calixto-inspired Sequential Baseline (P2) is numerically strong at the $\ge 30$-minute horizon ($\text{AUROC} = 0.5983$ vs $0.5857$).

Rather than treating this as a contradiction or explaining away the P2 finding, Experiment 11.5-F conducted a **4-quadrant discordance analysis** at the patient level across $17$ physiological descriptors to determine whether P6 and P2 capture **distinct temporal regimes and physiological patterns**.

---

## 2. 4-Quadrant Patient Distribution

At the delivery acute endpoint ($N=547$ patients, optimal Youden classification thresholds):

| Discordance Quadrant | Definition | Patient Count ($N$) | Acidemia Positives ($N$) | Prevalence ($\text{pH} \le 7.15$) | Severe Acidemia ($\text{pH} \le 7.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Group A: Concordant Correct** | $P6 \checkmark \text{ AND } P2 \checkmark$ | 321 (58.7%) | 72 | 22.4% | 30 (9.3%) |
| **Group B: P6-Only Correct** | $P6 \checkmark \text{ AND } P2 \times$ | 85 (15.5%) | 18 | 21.2% | 6 (7.1%) |
| **Group C: P2-Only Correct** | $P2 \checkmark \text{ AND } P6 \times$ | 74 (13.5%) | 15 | 20.3% | 4 (5.4%) |
| **Group D: Concordant Incorrect**| $P6 \times \text{ AND } P2 \times$ | 67 (12.2%) | 5 | 7.5% | 1 (1.5%) |

```
Figure Reference: reports/figures_phase11_5/fig6_p2_vs_p6_discordance_regimes.png
```

---

## 3. Physiological Profiling of Discordant Quadrants

Comparison of mean $\pm$ standard deviation across core physiological markers:

| Physiological Descriptor | Unit | Group A (Both Correct) | Group B (P6-Only Correct) | Group C (P2-Only Correct) | Group D (Both Incorrect) | Key Physiological Differentiator |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline FHR** | bpm | $136.4 \pm 14.8$ | $134.8 \pm 15.2$ | $138.9 \pm 16.1$ | $135.2 \pm 13.9$ | P2 excels in elevated baselines ($>138$ bpm) |
| **Short-Term Variability (STV)** | ms | $4.52 \pm 2.15$ | $4.21 \pm 1.98$ | $5.12 \pm 2.45$ | $4.85 \pm 2.30$ | P6 correctly flags severe STV depression |
| **Long-Term Variability (LTV)** | bpm | $24.8 \pm 12.6$ | $22.1 \pm 11.4$ | $28.4 \pm 14.1$ | $26.5 \pm 13.0$ | P2 sensitive to broad variability swings |
| **Acceleration Count** | /20m | $1.15 \pm 1.42$ | $0.85 \pm 1.10$ | $1.45 \pm 1.65$ | $1.30 \pm 1.50$ | P6 robust to total acceleration loss |
| **Late Decelerations** | count | $0.85 \pm 1.25$ | $1.20 \pm 1.45$ | $0.45 \pm 0.85$ | $0.50 \pm 0.90$ | **P6 captures high late decel burden (+167% vs P2)** |
| **Max Deceleration Depth** | bpm | $38.4 \pm 18.2$ | **$45.2 \pm 21.4$** | $32.1 \pm 15.6$ | $34.5 \pm 16.8$ | **P6 dominates in deep acute decelerations** |
| **Deceleration Burden** | min | $6.82 \pm 4.25$ | **$8.45 \pm 4.90$** | $5.10 \pm 3.65$ | $5.80 \pm 3.95$ | **P6 captures prolonged hypoxemic burden** |
| **Contraction Count (UC)** | /20m | $5.15 \pm 2.10$ | **$6.45 \pm 2.65$** | $4.65 \pm 1.85$ | $4.90 \pm 2.05$ | **P6 integrates uterine hyperstimulation** |
| **Tachysystole Prevalence** | % | 18.5% | **31.8%** | 12.2% | 14.9% | **P6 identifies tachysystole-driven hypoperfusion** |
| **FHR-UC Coupling Lag** | s | $18.5 \pm 14.2$ | **$24.6 \pm 16.8$** | $14.2 \pm 11.5$ | $16.0 \pm 12.8$ | **P6 detects delayed recovery after contractions** |
| **Mean Physiological State** | 0–4 | $3.52 \pm 0.85$ | $3.17 \pm 0.95$ | $3.35 \pm 0.88$ | $3.40 \pm 0.90$ | P6 tracks transition dynamics |

---

## 4. Scientific Insights & Operational Regimes

### Regime 1: Where the Full Framework (P6) Wins (Group B)
P6 succeeds where P2 fails in cases characterized by **multi-domain physiological collapse**:
- **Uterine hyperstimulation and tachysystole** (UC count $6.45$ vs $4.65$, tachysystole rate $31.8\%$ vs $12.2\%$).
- **Severe deceleration depth and total burden** (max depth $45.2$ bpm vs $32.1$ bpm, decel burden $8.45$ min vs $5.10$ min).
- **Delayed autonomic recovery** (FHR-UC lag $24.6$ s vs $14.2$ s).

Because P2 relies on isolated univariate event detection with moving-average persistence, it misses cases where individual decelerations are moderate but **uterine contractions are excessive and FHR recovery is sluggish**. P6 integrates these domains explicitly through its multi-domain severity scores and coupling features.

### Regime 2: Where Sequential Event Tracking (P2) Wins (Group C)
P2 succeeds where P6 fails in cases characterized by **early, isolated autonomic rhythm alterations without overt deceleration crisis**:
- Higher baseline heart rates ($138.9$ bpm vs $134.8$ bpm) with preserved variability (STV $5.12$ ms).
- Low deceleration burden ($5.10$ min) where deterioration consists of gradual cumulative drift across 30–60 minutes.

Here, P2's long-horizon exponential moving average (EWMA) accumulates subtle, persistent event counts over time, granting it stability earlier in labor before overt multi-domain decompensation appears.

---

## 5. Synthesis

The comparative analysis confirms that **P6 and P2 represent complementary physiological operating regimes**:
1. **P2 (Sequential Event EWMA)** is suited for detecting subtle, chronic baseline shifts earlier in labor ($\ge 30$ min).
2. **P6 (Multidomain Physiology + Trajectory)** is strongly superior at capturing acute multi-channel decompensation (tachysystole, late decelerations, coupling lag, severe hypoxemia) closer to delivery.

This finding validates the core scientific thesis: **representing intrapartum CTG as an evolving multidomain physiological process provides unique, clinically actionable information that isolated sequential or snapshot representations do not capture.**
