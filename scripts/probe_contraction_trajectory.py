"""
Phase 8, Route 1 -- does the contraction-response trajectory carry information
the 19 window-level descriptors do not?

Features only. No network. The question this answers is whether the
REPRESENTATION contains signal, which is a precondition for building a sequence
model on top of it.

Frozen protocol: the 547 patients and patient-grouped folds in
data/processed_clinical/folds.json, patient-level AUROC, patient-bootstrap CIs.
Pre-registered bar: +0.0642 (phase-7 MDE).

Run from the repo root:
    python scripts/probe_contraction_trajectory.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing.contraction_response import (
    LAST_HOUR_SAMPLES, patient_contraction_features, trajectory_feature_names)
from src.preprocessing.ingestion import load_ctu_chb_record
from src.training.protocol import Protocol

RAW = "data/raw/ctu-chb-intrapartum"
PROC = "data/processed_clinical"
OUT = "results/phase8"
CACHE = os.path.join(OUT, "contraction_features.npz")


def header_field(rec, key):
    for line in open(os.path.join(RAW, f"{rec}.hea")):
        s = line.strip()
        if s.startswith("#" + key):
            return s.split()[-1]
    return None


def build_cohort():
    """Patient ids, labels and the patient-level 19-descriptor comparator."""
    pid, y, f19 = [], [], {}
    for sp in ("train", "val", "test"):
        d = torch.load(os.path.join(PROC, f"{sp}_dataset.pt"), weights_only=False)
        F = np.hstack([d["y_features"].numpy(),
                       np.load(os.path.join(PROC, f"{sp}_extended_features.npy"))])
        ids = np.array([m[0] for m in d["metadata"]])
        yy = d["y_primary"].numpy()
        for p in np.unique(ids):
            m = ids == p
            # match the protocol's patient aggregation: max over windows, plus
            # the mean, so the comparator is not handicapped by using only one
            f19[p] = np.concatenate([F[m].max(0), F[m].mean(0)])
            pid.append(p)
            y.append(int(yy[m].max()))
        del d
    order = np.argsort(pid)
    pid = np.array(pid)[order]
    y = np.array(y)[order]
    return pid, y, np.vstack([f19[p] for p in pid])


def extract(pid):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z["pid"]) == list(pid):
            print(f"[cache] {CACHE}")
            return z["A"], z["S"], z["ncon"]
    names = trajectory_feature_names()
    A = np.full((len(pid), len(names)), np.nan, np.float32)
    S = np.full((len(pid), 2 * len(names)), np.nan, np.float32)
    ncon = np.zeros(len(pid), np.int32)
    for i, rec in enumerate(pid):
        fhr, uc, fs = load_ctu_chb_record(os.path.join(RAW, str(rec)))
        full = len(fhr)
        if full > LAST_HOUR_SAMPLES:
            fhr, uc = fhr[-LAST_HOUR_SAMPLES:], uc[-LAST_HOUR_SAMPLES:]
        offset = full - len(fhr)
        p2 = header_field(rec, "Pos. II.st")
        try:
            p2 = float(p2)
        except (TypeError, ValueError):
            p2 = -1.0
        second_start = (p2 - offset) if p2 >= 0 else -1.0
        A[i], S[i], ncon[i] = patient_contraction_features(
            fhr, uc, fs=fs, second_start=second_start)
        if i % 100 == 0:
            print(f"  {i}/{len(pid)}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(CACHE, pid=pid, A=A, S=S, ncon=ncon)
    return A, S, ncon


def main():
    pid, y, F19 = build_cohort()
    print(f"cohort {len(pid)} patients, {y.sum()} positive")
    A, S, ncon = extract(pid)
    names = trajectory_feature_names()
    ok = np.isfinite(A).any(1)
    print(f"contractions detected: median {np.median(ncon[ncon>0]):.0f} per patient; "
          f"{(~ok).sum()} patients with no usable trajectory")

    blob = json.load(open(os.path.join(PROC, "folds.json")))
    asg = {k: v for k, v in blob["assignment"].items() if k in set(pid)}
    P = Protocol(pid, y, asg)

    def run(name, M, model="lr"):
        oof = np.zeros(len(M))
        cnt = np.zeros(len(M))
        for tr, te_p in P.folds(repeat=0):
            te = np.isin(pid, te_p)
            if model == "lr":
                clf = make_pipeline(
                    SimpleImputer(strategy="median"), StandardScaler(),
                    LogisticRegression(max_iter=5000, C=0.1,
                                       class_weight="balanced"))
            else:
                clf = make_pipeline(
                    SimpleImputer(strategy="median"),
                    GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                               learning_rate=0.05,
                                               random_state=0))
            clf.fit(M[tr], y[tr])
            oof[te] += clf.predict_proba(M[te])[:, 1]
            cnt[te] += 1
        oof /= np.maximum(cnt, 1)
        return P.report_patient_scores(name, y, oof)

    print("\n--- patient-level AUROC, frozen folds, repeat 0 ---")
    base = run("19 descriptors, patient-level (comparator)", F19)["auroc"]
    res = {}
    res["traj"] = run("contraction trajectory ALONE", A)
    res["19+traj"] = run("19 + contraction trajectory", np.hstack([F19, A]))
    res["stage"] = run("stage-split trajectory ALONE", S)
    res["19+stage"] = run("19 + stage-split trajectory", np.hstack([F19, S]))
    res["19+both"] = run("19 + trajectory + stage-split",
                         np.hstack([F19, A, S]))
    res["19+traj_gb"] = run("19 + trajectory (gradient boosting)",
                            np.hstack([F19, A]), model="gb")

    print(f"\ncomparator {base:.4f} | pre-registered bar +0.0642")
    for k, v in res.items():
        print(f"  {k:14s} {v['auroc'] - base:+.4f}")

    # which trajectory features carry univariate signal, and are the
    # DETERIORATION statistics (trend / late_minus_early) among them?
    print("\ntop univariate trajectory features (patient-level AUROC):")
    sc = []
    for j, n in enumerate(names):
        v = A[:, j]
        m = np.isfinite(v)
        if m.sum() < 200 or len(np.unique(y[m])) < 2:
            continue
        a = roc_auc_score(y[m], v[m])
        sc.append((max(a, 1 - a), n, m.sum()))
    for a, n, k in sorted(sc, reverse=True)[:12]:
        tag = "  <-- deterioration" if ("trend" in n or "late_minus_early" in n) else ""
        print(f"  {n:34s} {a:.4f}  (n={k}){tag}")

    pd.DataFrame([dict(name=v["name"], auroc=v["auroc"], ci_lo=v["ci_lo"],
                       ci_hi=v["ci_hi"], auprc=v["auprc"]) for v in res.values()]
                 ).to_csv(os.path.join(OUT, "contraction_trajectory.csv"),
                          index=False)


if __name__ == "__main__":
    main()
