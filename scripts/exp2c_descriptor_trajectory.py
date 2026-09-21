"""
PHASE 10 / EXPERIMENT 2C -- Descriptor trajectory, cheap statistical test.

QUESTION
    Does the EVOLUTION of the 11 FHR-only clinical descriptors across the final
    hour contain information beyond their patient-level summary?

WHY THIS AND NOT RAW-FHR MIL
    Experiment 1 established that minimally processed raw FHR carries less usable
    information than the hand-computed descriptors (0.6117 vs 0.7300 matched).
    The descriptors are performing a physiologically informed compression the CNN
    failed to rediscover. So the next question is not "can a network find
    something in the raw signal" but "is there anything in how the PROVEN
    predictive quantities move over time".

THE CONTROL THAT MAKES THIS INTERPRETABLE
    The anchor (0.7300) is a WINDOW-level LR with patient max-aggregation. A
    trajectory model is PATIENT-level. Comparing them directly would confound
    "temporal information" with "different estimator". So the summaries are split:

        static    mean, sd, min, max          -- ordering never used
        temporal  final, trend, final-initial, late-third minus early-third

    static           = patient-level reformulation, no temporal content
    static+temporal  = the same reformulation plus temporal content
    difference       = the actual answer

PRE-REGISTERED GATE (on static+temporal, patient-level AUROC)
    < 0.73       KILL the temporal-descriptor route
    0.73 - 0.76  insufficient; do NOT neuralize it
    0.76 - 0.80  build the small temporal model
    >= 0.80      investigate aggressively toward 0.85

No tuning. One regulariser setting (C=0.1, L2, balanced), identical to every
other LR in this project. Frozen folds. No test-fold selection.

Run from the repo root:
    python scripts/exp2c_descriptor_trajectory.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.training.protocol import Protocol

PROC = "data/processed_clinical"
OUT = "results/phase10"

# indices into hstack([y_features(8), extended(11)]) that need NO UC to compute
FHR_ONLY = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13]
FHR_NAMES = ["baseline", "stv", "ltv", "accels", "dec_prolonged",
             "decel_max_depth", "decel_area", "decel_burden",
             "decel_longest_sec", "baseline_slope", "variability_slope"]
STATIC = ["mean", "sd", "min", "max"]
TEMPORAL = ["final", "trend", "final_minus_initial", "late_minus_early"]


def summarise(seq):
    """seq: (T, D) chronologically ordered. Returns (static, temporal) blocks."""
    T, D = seq.shape
    st = np.concatenate([seq.mean(0), seq.std(0), seq.min(0), seq.max(0)])
    final = seq[-1]
    if T >= 3:
        t = np.arange(T, dtype=float)
        t -= t.mean()
        trend = (t[:, None] * (seq - seq.mean(0))).sum(0) / max((t * t).sum(), 1e-9)
    else:
        trend = np.zeros(D)
    fmi = seq[-1] - seq[0]
    k = max(T // 3, 1)
    lme = seq[-k:].mean(0) - seq[:k].mean(0)
    return st, np.concatenate([final, trend, fmi, lme])


def main():
    F, Y, PID, START = [], [], [], []
    for sp in ("train", "val", "test"):
        d = torch.load(os.path.join(PROC, f"{sp}_dataset.pt"), weights_only=False)
        F.append(np.hstack([d["y_features"].numpy(),
                            np.load(os.path.join(PROC, f"{sp}_extended_features.npy"))]))
        Y.append(d["y_primary"].numpy())
        PID.append(np.array([m[0] for m in d["metadata"]]))
        START.append(np.array([int(m[1]) for m in d["metadata"]]))
        del d
    F = np.concatenate(F)[:, FHR_ONLY]
    yw = np.concatenate(Y)
    pid = np.concatenate(PID)
    start = np.concatenate(START)

    patients = np.array(sorted(set(pid.tolist())))
    ST, TE, lab, nwin = [], [], [], []
    for p in patients:
        m = pid == p
        seq = F[m][np.argsort(start[m])]          # chronological
        s, t = summarise(seq)
        ST.append(s)
        TE.append(t)
        lab.append(int(yw[m].max()))
        nwin.append(int(m.sum()))
    ST, TE = np.vstack(ST), np.vstack(TE)
    lab, nwin = np.array(lab), np.array(nwin)
    print(f"{len(patients)} patients, {lab.sum()} positive; "
          f"windows/patient median {np.median(nwin):.0f} (min {nwin.min()})")
    print(f"static block {ST.shape[1]} features, temporal block {TE.shape[1]}")

    blob = json.load(open(os.path.join(PROC, "folds.json")))
    asg = {k: v for k, v in blob["assignment"].items() if k in set(patients)}
    P = Protocol(patients, lab, asg)

    def run(name, M):
        oof = np.zeros(len(M))
        cnt = np.zeros(len(M))
        for tr_mask, te_p in P.folds(repeat=0):
            tr = np.where(tr_mask)[0]
            te = np.isin(patients, te_p)
            clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=5000, C=0.1,
                                                   class_weight="balanced"))
            clf.fit(M[tr], lab[tr])
            oof[te] += clf.predict_proba(M[te])[:, 1]
            cnt[te] += 1
        oof /= np.maximum(cnt, 1)
        return P.report_patient_scores(name, lab, oof), oof

    print("\n--- patient-level, frozen folds, C=0.1 L2 balanced ---")
    r_st, _ = run("STATIC only (mean/sd/min/max)          ", ST)
    r_te, _ = run("TEMPORAL only (final/trend/fmi/lme)    ", TE)
    r_all, oof = run("STATIC + TEMPORAL                      ",
                     np.hstack([ST, TE]))

    anchor = 0.7300
    d_tmp = r_all["auroc"] - r_st["auroc"]
    print(f"\n  window-LR + max aggregation anchor (11 FHR descriptors): {anchor:.4f}")
    print(f"  patient-level reformulation, no temporal content:        "
          f"{r_st['auroc']:.4f}  ({r_st['auroc']-anchor:+.4f})")
    print(f"  TEMPORAL CONTRIBUTION (static+temporal minus static):    "
          f"{d_tmp:+.4f}   [MDE 0.0642]")

    def gate(a):
        if a >= 0.80:
            return "INVESTIGATE AGGRESSIVELY toward 0.85"
        if a >= 0.76:
            return "BUILD the small temporal model"
        if a >= 0.73:
            return "INSUFFICIENT - do not neuralize"
        return "KILL the temporal-descriptor route"

    print(f"\n  GATE (on static+temporal = {r_all['auroc']:.4f}): {gate(r_all['auroc'])}")

    # which temporal summaries carry anything at all, univariately
    print("\n  top univariate temporal features (patient-level AUROC):")
    names = [f"{n}_{s}" for s in TEMPORAL for n in FHR_NAMES]
    sc = []
    for j, n in enumerate(names):
        v = TE[:, j]
        m = np.isfinite(v)
        if m.sum() < 200 or np.std(v[m]) < 1e-9:
            continue
        a = roc_auc_score(lab[m], v[m])
        sc.append((max(a, 1 - a), n))
    for a, n in sorted(sc, reverse=True)[:8]:
        print(f"    {n:36s} {a:.4f}")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"anchor_window_lr_maxagg": anchor,
               "static": r_st, "temporal": r_te, "static_plus_temporal": r_all,
               "temporal_contribution": float(d_tmp),
               "gate": gate(r_all["auroc"])},
              open(os.path.join(OUT, "exp2c_descriptor_trajectory.json"), "w"),
              indent=1, default=float)
    np.save(os.path.join(OUT, "oof_exp2c.npy"), oof)


if __name__ == "__main__":
    main()
