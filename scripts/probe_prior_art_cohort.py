"""
Prior-art reconciliation -- what explains 0.822 (Dang et al. 2026) vs 0.7271?

Dang et al. state: "recordings with >50% missing FHR values are excluded, 404
patients remain". Measured directly on CTU-UHB, the MAXIMUM FHR missing ratio
across all 552 records is 0.535, so a >50% rule removes 5 records and leaves
547 -- which is exactly this project's cohort. To leave 404 the threshold would
have to be ~26%.

Their cohort is therefore the cleanest ~73% of the database by signal quality,
however it was actually selected. This script measures what that is worth, and
separates it from the other two protocol differences (evaluation unit, and
epoch selection), so the reconciliation is quantified rather than asserted.

Grid:
             cohort:  all 547        cleanest 404 (their implied cohort)
  unit: window-level      x                    x
  unit: patient-level     x                    x

Run from the repo root:
    python scripts/probe_prior_art_cohort.py
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

RAW = "data/raw/ctu-chb-intrapartum"
PROC = "data/processed_clinical"


def missing_ratio(rec):
    a = np.fromfile(os.path.join(RAW, f"{rec}.dat"), dtype="<i2").reshape(-1, 2)
    return float((a[:, 0] <= 0).mean())


def main():
    F, Y, PID = [], [], []
    for sp in ("train", "val", "test"):
        d = torch.load(os.path.join(PROC, f"{sp}_dataset.pt"), weights_only=False)
        F.append(np.hstack([d["y_features"].numpy(),
                            np.load(os.path.join(PROC, f"{sp}_extended_features.npy"))]))
        Y.append(d["y_primary"].numpy())
        PID.append(np.array([m[0] for m in d["metadata"]]))
        del d
    F, y, pid = np.concatenate(F), np.concatenate(Y), np.concatenate(PID)
    patients = np.array(sorted(set(pid)))

    mr = np.array([missing_ratio(p) for p in patients])
    order = np.argsort(mr)
    clean404 = set(patients[order[:404]].tolist())
    print(f"cohort {len(patients)} patients | cleanest-404 cut at missing ratio "
          f"{mr[order[403]]:.3f}")

    blob = json.load(open(os.path.join(PROC, "folds.json")))

    def evaluate(name, keep_patients):
        m = np.isin(pid, list(keep_patients))
        Fm, ym, pm = F[m], y[m], pid[m]
        asg = {k: v for k, v in blob["assignment"].items() if k in keep_patients}
        P = Protocol(pm, ym, asg)
        oof = np.zeros(len(Fm))
        cnt = np.zeros(len(Fm))
        for tr, te_p in P.folds(repeat=0):
            te = np.isin(pm, te_p)
            clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=5000, C=0.1,
                                                   class_weight="balanced"))
            clf.fit(Fm[tr], ym[tr])
            oof[te] += clf.predict_proba(Fm[te])[:, 1]
            cnt[te] += 1
        oof /= np.maximum(cnt, 1)
        win = roc_auc_score(ym, oof)                       # their metric
        pat = P.report(f"{name} -- patient-level", oof, how="max")
        n_pos = int(np.isin(list(keep_patients), None).sum())
        print(f"  {name} -- window-level pooled          AUROC {win:.4f}")
        return win, pat["auroc"], P

    print("\n--- full cohort (this project) ---")
    w_all, p_all, P_all = evaluate("all 547", set(patients.tolist()))

    print("\n--- cleanest 404 (Dang et al.'s implied cohort) ---")
    w_404, p_404, P_404 = evaluate("clean 404", clean404)
    print(f"    positives in that cohort: {int(P_404.plab.sum())} of 404 "
          f"({100*P_404.plab.mean():.1f}%) vs {int(P_all.plab.sum())} of 547 "
          f"({100*P_all.plab.mean():.1f}%)")

    print("\n--- reconciliation grid (19-descriptor model, frozen folds) ---")
    print(f"{'':22s} {'window-level':>14s} {'patient-level':>14s}")
    print(f"{'all 547':22s} {w_all:14.4f} {p_all:14.4f}")
    print(f"{'cleanest 404':22s} {w_404:14.4f} {p_404:14.4f}")
    print(f"\nDang et al. report 0.822, window-level pooled, on their 404-patient cohort.")
    print(f"Cohort effect at window level: {w_404 - w_all:+.4f}")
    print(f"Unit effect (patient - window), all 547: {p_all - w_all:+.4f}")


if __name__ == "__main__":
    main()
