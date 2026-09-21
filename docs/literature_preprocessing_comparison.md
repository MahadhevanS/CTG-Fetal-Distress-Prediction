# How prior work preprocesses CTU-UHB — and where this project diverges

Reviewed 2026-09-02, before committing to the redesign in
[preprocessing_redesign.md](preprocessing_redesign.md). Sources at the bottom.

## Comparison

| aspect | **this project** | Dang et al. 2026 (the benchmark) | Foundation-model paper 2026 | DeepCTG 2.0 / multicentre |
|---|---|---|---|---|
| pH threshold | ≤ 7.15 | < 7.15 | < 7.15 → 113 cases (20.5%) | graded: severe < 7.05, moderate 7.05–7.20 |
| composite outcome | no | no | no | **yes** — Apgar ≤ 7 **or** pH grades |
| **window label** | **positive only in last 30 min** | patient label on every window | patient label on every window | patient level |
| window / stride | 20 min / 2.5 min | 20 min / 5 min | 7.5 min sliding | 30-min segments |
| **quality gate** | **>30% missing → drop the WINDOW** | >50% missing → drop the RECORDING | >50% quality per 30-min | — |
| interpolation | cubic | linear | linear | linear |
| positive windows | **4.3%** | 16.6% | ~20% | — |
| reported AUROC | 0.812 (CrossFormer) / 0.821 (CNN1D) | 0.822 | 0.83 | — |

Published AUROC range on CTU-UHB: **0.68 – 0.83** (Petrozziello et al. ≈ 0.75).

## What this changes about the diagnosis

### 1. The horizon rule is this project's own invention

No paper reviewed labels windows by position within the recording. Every one
applies the patient's outcome to all its windows. The 30-minute horizon appears
in `pipeline.py` as a "GAP 2 FIX" and is not inherited from the benchmark.

This is the single largest divergence and it is what produces the time confound
(time alone → AUROC 0.84) and the 4.3% positive rate versus the field's ~17–20%.

### 2. But patient-level labelling is *also* acknowledged as flawed

The foundation-model paper states it directly:

> "labeling every segment from an acidemic delivery as positive assumes the
> abnormality is continuously detectable, which is clinically unrealistic."

So the horizon rule was attacking a real problem that the field openly concedes.
It just chose a remedy that introduces a worse one — a label computed from the
future, which cannot be reproduced at inference. **The right move is not simply
"do what everyone else does"**; it is to fix the label without reintroducing
future information. Multiple-instance learning is the principled answer, and the
`y_patient` target already in the tensors supports it.

### 3. 0.81 is not underperformance — it is the top of the published range

This matters for how the result is written up. The project is at 0.81–0.82
against a field range of 0.68–0.83. **The ceiling is field-wide, not a defect in
this model.**

One caveat on comparability, which cuts against the project: this project's
0.812 is measured on the *horizon* task, while the field's 0.75–0.83 is on the
*patient-broadcast* task. They are different targets and **should not be compared
directly**. The delivered model scores 0.6086 on the patient-level target, which
is the number actually comparable to published work — and it is below the range.
That comparison should be made honestly before claiming parity with the
benchmark.

### 4. pH < 7.15 is the field convention, so keep it — but not alone

My earlier recommendation to replace the target was too strong. pH < 7.15 giving
113 positives (20.5%) is exactly what the benchmark and the foundation-model
paper use, and abandoning it forfeits comparability. The clinical-validity
finding still stands — 47% of those babies had Apgar5 ≥ 9, and the
foundation-model paper concedes pH "does not fully capture long-term
neurological outcome" — but the fix is to report **both**, not to swap one for
the other. Graded and composite definitions have precedent: DeepCTG 2.0 uses
severe/moderate bands, and the multicentre model uses Apgar ≤ 7 **or** pH grades.

### 5. The quality gate is stricter than anyone else's, in the wrong way

Prior work drops **recordings** with >50% missing. This project drops **windows**
with >30% missing. Two differences compound:

- the threshold is stricter (30% vs 50%);
- it operates per window, so a patient can silently lose only their *late*
  windows — which is precisely what the audit found (acidotic patients keep
  62.6% of horizon windows vs 86.2% pre-horizon; 27/108 lose all of them).

Dang et al.'s recording-level filter is blunter but **unbiased with respect to
time within a patient**. This project's per-window filter is not.

## Revised recommendation

Unchanged and now better supported:

- **Delete the horizon rule** (§4.2 of the redesign). No precedent; sole cause of
  the time confound; produces a label unusable at inference.
- **Relax the quality gate to 50% and apply it at recording level**, or keep
  per-window but add the missingness mask as an input channel and weight rather
  than drop. Aligns with the field and removes the temporal bias.

Revised:

- **Keep pH ≤ 7.15 as the primary target** for benchmark comparability, and emit
  the composite (pH ≤ 7.05 **or** BDecf ≥ 12 **or** Apgar5 < 7) as a **secondary**
  target reported alongside. Do not replace one with the other.
- **Report the patient-level number prominently.** It is the figure comparable to
  published work, and at 0.6086 it is currently below the field range.

Added:

- **Consider MIL properly** rather than either extreme. Both the horizon rule and
  patient-broadcast are attempts to answer "which windows in a bad labour are
  actually abnormal?" MIL answers it without inventing a label.

## Sources

- [A Hybrid CNN-Transformer with Cross-Attention for Automated Fetal Distress Detection](https://www.e3s-conferences.org/) — Dang, Nguyen & Ho, E3S Web of Conferences 2026 (local copy: `e3sconf_aiei2026_01005.pdf`)
- [A Foundation Model Approach for Fetal Stress Prediction During Labor](https://arxiv.org/html/2601.06149)
- [DeepCTG 2.0: deep learning to detect neonatal acidemia from cardiotocography during labor](https://pubmed.ncbi.nlm.nih.gov/39608037/)
- [AI-based prediction of fetal hypoxia: multicentre model development](https://link.springer.com/article/10.1186/s12916-026-04794-z)
- [Rapid detection of fetal compromise using input length invariant deep learning](https://www.nature.com/articles/s41598-024-63108-6)
- [Cross-Database Evaluation of Deep Learning Methods for Intrapartum Cardiotocography Classification](https://pubmed.ncbi.nlm.nih.gov/40657532/)
- [The CTU-UHB Intrapartum Cardiotocography Database](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)
