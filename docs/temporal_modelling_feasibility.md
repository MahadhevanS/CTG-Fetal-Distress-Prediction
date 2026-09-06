# Temporal trajectory modelling — feasibility and design report

Written 2026-09-04, before any temporal model is trained. Every number here is
reproducible offline from `data/processed_clinical/` through the frozen protocol
in [../src/training/protocol.py](../src/training/protocol.py).

**Research question.** Does modelling the temporal evolution of CTG windows
across a recording provide information for patient-level fetal-distress
prediction that is lost when window predictions are generated independently and
reduced by max aggregation?

**Verdict up front: technically viable, statistically marginal, and the
pre-training evidence says the answer is probably no.** The recommended first
run is a cheap control, not the full ladder. Reasoning below.

---

## A. Dataset feasibility

### A.1 Metadata structure — order IS genuine

`metadata` is a tuple `(record_id, start_sample, end_sample)`, e.g.
`('1044', 0, 4800)`. Chronological order is therefore reconstructible exactly,
by sorting on `start_sample`. No timestamps were fabricated for this analysis.

| check | result |
|---|---|
| duplicate window starts | **0 patients** |
| rows stored out of chronological order | **0 patients** (already sorted) |
| uniform 600-sample (2.5 min) stride | **509 / 547 (93.1%)** |
| non-uniform spacing (gaps) | 38 / 547 (6.9%) |

Gaps come from the quality gate dropping individual windows; the commonest
pattern is `(600, 3600)` — one run of normal stride with a single 6-window hole.
Window length is 4800 samples = 20 min at 4 Hz. `w_minutes_before_end` is also
present, range [0.0, 40.0] min.

### A.2 Cohort and sequence statistics

| quantity | value |
|---|---|
| patients | 547 |
| positive patients | 110 (20.1%) |
| windows | 8,517 |
| windows per patient | min 3, p25 15, **median 17**, p75 17, max 17 |
| patients with exactly 17 windows | 381 (69.7%) |
| patients with < 10 windows | 26 (4.8%) |
| windows/patient, positive vs negative | median 17 vs 17 — **no bag-size leak** |

Sequences are **short and bounded: T ≤ 17**. No truncation is needed, and no
downsampling strategy is required. 166 patients (30.3%) are shorter than 17 and
need padding with masking.

The bag-size check matters: on `data/processed/` window count alone predicted the
label at AUROC 0.9947 (`scripts/audit_bag_size_leak.py`). On this substrate it is
0.4657, and positive/negative medians are identical. Sequence models here cannot
cheat on length.

---

## B. Current baseline (T0)

Established through the frozen protocol, repeat 0, 5 patient-grouped folds,
patient-level max aggregation, CI bootstrapped over patients.

| model | patient AUROC | 95% CI |
|---|---:|---|
| 19-feature LR → max-agg | 0.7268 | [0.670–0.779] |
| mslstm E1 (binary only) → max-agg | 0.6834 | [0.620–0.740] |

**Do not mix recipes.** The architecture sweep reported mslstm at 0.7178 using
nested per-epoch selection; the ladder's E1 uses a fixed 20-epoch schedule with
no selection and gives 0.6834. Both are honest; they are different recipes. The
temporal experiment must use E1's recipe, because that is what the temporal
model would share.

---

## C. Is temporal ordering informative? — answered before training

This is the decisive section. The window score sequence `p_1 … p_T` already
exists out-of-fold for two independent sources. If temporal ordering carries
information, an order-dependent statistic should beat an order-invariant one.

| statistic | order-dependent | 19-feat LR | mslstm E1 |
|---|---|---:|---:|
| **max** | no | **0.7268** | **0.6834** |
| top-3 mean | no | 0.7178 | 0.6768 |
| p90 | no | 0.7166 | 0.6758 |
| std | no | 0.6896 | 0.6450 |
| mean | no | 0.6744 | 0.6440 |
| max increase | **yes** | 0.6313 | 0.6049 |
| last window | **yes** | 0.6227 | 0.6034 |
| late − early | **yes** | 0.6121 | 0.6012 |
| slope | **yes** | 0.6115 | 0.5985 |
| **argmax position** | **yes** | **0.4752** | **0.5087** |
| learned f(p₁…p_T), 11 stats, OOF | — | 0.7088 | 0.6709 |

Three results, both sources agreeing:

1. **Every order-invariant statistic beats every order-dependent one.**
2. **Where the peak risk occurs is at chance** (0.4752 / 0.5087).
3. **A learned function of the whole score sequence loses to plain max**
   (−0.0180 / −0.0125), fit on training patients only through the frozen folds.

This is the shuffle control (§9 of the brief) answered analytically: order-based
summaries carry strictly less signal than order-free ones on the same data.

### C.1 Why — the trajectory has the same shape in both classes

Mean OOF window score by position tercile:

| tercile | positive patients | negative patients |
|---|---:|---:|
| early | 0.4745 | 0.4361 |
| middle | 0.5087 | 0.4557 |
| late | 0.5895 | 0.4980 |

Risk rises toward the end of labour in **both** groups (+0.115 positive, +0.062
negative). Peak-risk position does not differ between them (median 0.826 vs
0.875, Mann-Whitney **p = 0.412**).

This reproduces, on clean data, the finding in
[auroc_ceiling_analysis.md](auroc_ceiling_analysis.md) that normal patients'
risk scores roughly double from the start of the hour to the end (ρ = +0.351,
p = 1.8e-27). **The deterioration trend is a property of late labour, not of
distress.** A trajectory-aware model would be learning a pattern that occurs in
everyone — which is precisely the failure mode this project already diagnosed
once, under the horizon label.

### C.2 Independent corroboration

`scripts/audit_evaluation_protocol.py` (`representation` probe) reached the same
conclusion from features rather than scores: trajectory slopes alone score
0.6583 and early-vs-late contrast 0.6648, versus 0.7290 for per-window plus
max-aggregate. Adding slope and contrast to the patient design **lowered**
AUROC. Two independent routes, same answer.

---

## D. Statistical feasibility of a sequence model

| | window-level (current) | patient-level sequence |
|---|---:|---:|
| independent training examples per fold | 6,813 | **437** |
| positive examples per fold | ~1,367 windows | **~88 patients** |
| supervision ratio | 15.6 : 1 | 1 |

This is the exact arithmetic that killed Model 9 (KG-MIL): *"patient-level
training has ~435 bags / ~86 positives to fit a 2.5M-param model, versus 5,286
window labels — supervision collapses by an order of magnitude"*
([model8_crossformer_run_history.md](model8_crossformer_run_history.md), Part E).
The numbers here are 437 / ~88 — essentially identical.

Three independent measurements now agree that patient-level *training* loses to
window training plus aggregation: Model 9's failed gate, the `representation`
probe, and Milestone 1's baseline C (0.6977 vs 0.7268).

**Implication for design:** Strategy A (frozen encoder, tiny aggregator) is the
only statistically defensible option. A GRU with hidden size 16 over 128-d
embeddings is ~7k parameters and an attention pooler ~2k — tractable on 437
patients. Strategy B (end-to-end fine-tuning) reintroduces the collapse and must
not be run first.

---

## E. Recommended experiment ladder

Only if the go/no-go control in §G passes.

| rung | model | params | risk |
|---|---|---:|---|
| **T0** | window model → max aggregation | 0 extra | control, already measured |
| T1 | frozen embeddings → mean pooling | ~0.3k | order-invariant; isolates pooling |
| T2 | frozen embeddings → max pooling | ~0.3k | order-invariant |
| T3 | frozen embeddings → gated attention pooling | ~2k | order-invariant unless positions added |
| T4 | frozen embeddings → GRU(16) | ~7k | **first order-aware rung** |
| T5 | lightweight temporal Transformer | ~30k+ | only if T4 clears the bar |

T1–T3 are order-invariant by construction, so **T4 versus T3 is the actual test
of the hypothesis** — not T4 versus T0, which conflates temporal modelling with
learned pooling.

### E.1 Controls

1. **Chronological vs shuffled** (essential). Same windows, permuted order, same
   everything else. Run on T4 only. `chronological ≈ shuffled` means the gain is
   pooling, not dynamics.
2. **Max vs learned aggregation** — already answered in §C: learned loses.
3. **Frozen vs fine-tuned encoder** — frozen only, until something works.

### E.2 Sequence handling

- Pad to T = 17, right-padded, with an explicit boolean mask.
- Mask must exclude padded slots from attention (−inf pre-softmax), from mean
  (divide by true length), and from max (−inf fill). `mil_dataset.py` already
  implements masked variable-length bag collation and was smoke-verified to give
  padded slots exactly zero attention — reuse it rather than rewriting.
- No truncation, no downsampling: T ≤ 17 already.
- 166 patients (30.3%) require padding.

### E.3 Deployment caveat

`w_minutes_before_end` must **not** be a model input. It is unavailable at
inference — a labour-ward monitor does not know when labour will end — and it is
the exact quantity that produced this project's original time confound. Relative
index within the windows seen *so far* is legitimate; normalised position `i/T`
is not, because `T` is unknown until delivery.

This experiment is **retrospective** (whole recording → one label). A real-time
formulation is a separate study, not a variant of this one.

---

## F. Success criteria — pre-registered

Same discipline as the multi-task ladder, which produced a clean null because the
bar was set in advance.

- Primary: patient AUROC, max/learned aggregation, frozen protocol, repeat 0.
- Report per-fold AUROC, per-fold delta, folds improved, paired Wilcoxon,
  patient bootstrap CI.
- **Bar: +0.02 pooled AND ≥4/5 folds in the same direction, T4 vs T3.**
- Report the continuous effect regardless of whether the bar is cleared.
- No architecture selection on the reported folds. One predefined configuration
  per rung; no hyperparameter sweep.

---

## G. Go / no-go

**Recommendation: do NOT run the T1–T5 ladder yet.**

The evidence against temporal ordering carrying extra information is already
substantial and comes from three independent directions (score-sequence
statistics, feature-trajectory statistics, and the identical rise in both
classes). Spending ~2 GPU-hours to confirm it would be spending it on a
hypothesis that has already been answered cheaply.

**Run this single control first (~25 min):**

> Extract frozen mslstm E1 embeddings, then compare **T3 (attention pooling,
> order-invariant)** against **T4 (GRU-16, order-aware)** and **T4-shuffled**.

Three numbers. The decision rule:

| outcome | conclusion | next |
|---|---|---|
| T4 > T3 by ≥0.02 **and** T4 > T4-shuffled | temporal order carries information | proceed to T5, then Strategy B |
| T4 ≈ T3 ≈ T4-shuffled | order is not the missing information | **stop**; record the negative and move to a new hypothesis |
| T3 > T0 but T4 ≈ T3 | learned pooling helps, ordering does not | pursue pooling, drop the temporal framing |

The third outcome is worth flagging: it is entirely possible that attention
pooling beats max without any temporal component. That would be a real result,
and it belongs to the aggregation question rather than the trajectory question.

**Failure modes to watch**

1. Overfitting the aggregator — 437 patients, ~88 positive. Keep it under ~10k
   parameters and regularise hard.
2. Reading a +0.005 as success. The baseline CI is 0.12 wide.
3. Comparing T4 to T0 rather than to T3, which conflates ordering with pooling.
4. Padded slots leaking into pooling — verify masks numerically before trusting
   any number.
5. Accidentally supplying `i/T` or minutes-to-end and reintroducing the time
   confound.
