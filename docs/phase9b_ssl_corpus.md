# Phase 9B — does enlarging the pre-training corpus with external data help?

Completed 2026-09-04. Controlled information-diversity experiment: the SSL
objective, architecture, fine-tuning procedure, folds and evaluation are
identical across arms. The only variable is **which recordings the encoder is
pre-trained on**.

Code: [`scripts/probe_ssl_corpus.py`](../scripts/probe_ssl_corpus.py).
Representation frozen per [phase9a](phase9a_detector_validation.md) —
`calculate_iterative_baseline` unchanged.

---

## 1. Result

| arm | pre-training corpus | AUROC | 95 % CI | AUPRC | sens@80spec | spec@90sens | ECE | fold SD |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `randinit` | none | **0.6489** | [0.596–0.701] | 0.2895 | 0.373 | 0.245 | 0.406 | 0.0316 |
| `ssl_ctu` | CTU-UHB (8 517 win) | **0.6202** | [0.560–0.678] | 0.2832 | 0.300 | 0.236 | 0.429 | 0.0379 |
| `ssl_ctu_fhrma` | CTU-UHB + FHRMA (11 814 win) | **0.6494** | [0.591–0.704] | 0.3009 | 0.345 | 0.222 | 0.405 | 0.0275 |
| `ssl_fhrma` | FHRMA only (3 297 win) | **0.6412** | [0.583–0.697] | 0.2932 | 0.382 | 0.268 | 0.426 | 0.0375 |
| — | **19-descriptor LR (frozen benchmark)** | **0.7271** | [0.670–0.779] | 0.4094 | — | — | — | — |

### The two numbers that matter, and they say different things

| contrast | value | reading |
|---|---:|---|
| `ssl_ctu_fhrma` − `ssl_ctu` (**the designed contrast**) | **+0.0293** | FHRMA's contribution, below the +0.0642 bar |
| **`ssl_ctu_fhrma` − `randinit`** | **+0.0005** | **the whole SSL enterprise is worth nothing** |

> The FHRMA contrast is real but it is **recovery from a self-inflicted
> regression, not a gain over doing nothing.** CTU-only SSL costs −0.0287;
> adding FHRMA buys back +0.0293; the net against random initialisation is
> **+0.0005**.

Reading the +0.0293 in isolation would have been the error this experiment was
designed to avoid. The `randinit` arm is what prevents it.

---

## 2. CTU-only SSL regression independently replicated

| run | substrate | protocol | effect vs random-init |
|---|---|---|---:|
| 2026-08-17 (Model 8 wide-distress) | MIL | old split | **−0.0262** |
| **2026-09-04 (this run)** | clinical | frozen patient-grouped folds | **−0.0287** |

Different substrate, different protocol, different downstream head, and
better-converged pre-training (holdout 0.2527 vs 0.2692) — same sign, nearly the
same magnitude. **CTU-only masked-reconstruction SSL does not merely fail to
help; it reliably hurts.**

The operating-point damage replicates too: sensitivity at 80 % specificity falls
0.373 → 0.300 and ECE worsens 0.406 → 0.429, exactly the pattern the August run
showed and that an AUROC-only scoreboard would have missed.

---

## 3. External-only pre-training is neutral, and it is the leakage-free arm

`ssl_fhrma` contains **no CTU-UHB patient**, so its 0.6412 is free of the
transductive leakage that arms 2 and 3 carry (§5). Against `randinit` it is
**−0.0077** — indistinguishable from nothing.

So the only leakage-clean external-data result available is null.

---

## 4. SSL reconstruction loss is not a proxy for representation quality

| arm | best holdout reconstruction loss | downstream AUROC |
|---|---:|---:|
| `ssl_fhrma` | **0.0863** (best) | 0.6412 |
| `ssl_ctu_fhrma` | 0.1970 | **0.6494** (best) |
| `ssl_ctu` | 0.2527 (worst) | 0.6202 (worst) |

The arm with by far the best SSL objective has middling downstream performance.
FHRMA windows contain **0.12 %** gap-artifact samples against CTU's **7.05 %**,
so they are intrinsically easier to reconstruct; a lower loss reflects an easier
holdout set, not a better encoder.

This is the same lesson as [phase9a](phase9a_detector_validation.md) in a
different guise: **a physiologically or objectively "better" intermediate does
not imply a better endpoint predictor.** Two independent demonstrations now.

---

## 5. Limitations, stated

1. **Transductive leakage in arms 2 and 3.** Both pre-train on all 547 CTU
   patients and are then evaluated by 5-fold CV over those same patients; the
   encoder saw every test fold's signals, never their labels. The arm 3 − arm 2
   contrast is unaffected (identical exposure), and arm 4 is clean. Absolute
   values for arms 2–3 are optimistic. Per-fold pre-training would fix it at 5×
   cost.
2. **All SSL arms hit the 60-epoch cap while still improving** — none
   early-stopped. They are undertrained by their own convergence criterion.
   This applies equally to control and experiment, so the contrast holds, but
   "SSL fails" is not established; "SSL as trained here fails" is.
3. **Domain gap.** FHRMA is 60× cleaner than CTU in residual gap artifacts. An
   encoder pre-trained on FHRMA may transfer poorly for reasons unrelated to
   information content.
4. **Corpus is 34 % of Fridman's** (687 of 984 recordings; ~830 of 2 444 hours).
   SPaM'17, the missing two thirds, is access-blocked
   ([phase9_data_availability.md](phase9_data_availability.md)).

---

## 6. Gates

| gate | verdict |
|---|---|
| 1 discrimination ≥ 0.85 | **fail** — best arm 0.6494 |
| 2 fold robustness | **pass** — fold SD 0.027–0.038, inside the ~0.036–0.05 sampling band |
| 3 generalisation | pass — no arm shows a train/test blow-up |
| 4 threshold behaviour | **fail** — best sens@80spec 0.382; spec@90sens ≤ 0.268 |
| 5 calibration | **fail** — ECE 0.405–0.429 across all arms |
| 6 leakage | arm 4 clean; arms 2–3 transductive, disclosed |
| 7 ablation | **satisfied by construction** — `randinit` isolates SSL, `ssl_ctu` vs `ssl_ctu_fhrma` isolates the corpus |

Every arm is also far below the 19-descriptor logistic regression (0.7271) on
the same folds, as every CrossFormer configuration in this project has been.

---

## 7. Verdict against the pre-registered decision bands

The pre-registered reading was: +0.00–0.02 insufficient; +0.03–0.06 interesting
but short; +0.06–0.12 investigate; >0.85 stop and audit.

**The designed contrast lands at +0.0293 — the bottom of the "interesting but
short" band. The honest figure against no pre-training at all is +0.0005, which
is the bottom of the first band.**

FHRMA's 135 recordings (+24 % corpus) are not sufficient, exactly as
[phase9_data_availability.md](phase9_data_availability.md) predicted. The route
is not refuted — it is **bounded by corpus access**, and the missing two thirds
of the corpus is precisely the part that cannot currently be obtained.
