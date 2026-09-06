"""
Phase 8, Route 1 -- the decisive additive test.

probe_contraction_trajectory.py compared trajectory features against a
PATIENT-level 19-descriptor model (0.6784). That is the right unit but the wrong
baseline: the established number is 0.7271, produced by training at the WINDOW
level and max-aggregating the probabilities.

This script asks the question that matters:

    does the contraction-response trajectory add anything ON TOP OF the
    strongest existing model?

The window-level 19-descriptor out-of-fold score is max-aggregated into one
number per patient and used as a single feature. Trajectory features are then
added, with feature selection performed INSIDE each training fold.

Note on the stacking: the per-patient baseline score is out-of-fold, so when it
is used as a training feature for fold k it carries information from models that
saw fold k. That bias is OPTIMISTIC, which makes a null result here safe to
believe and a positive result something to re-run nested.

Run from the repo root:
    python scripts/probe_contraction_stacked.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing.contraction_response import trajectory_feature_names
from src.training.protocol import Protocol

PROC = "data/processed_clinical"
OUT = "results/phase8"


def load_window_level():
    F, Y, PID = [], [], []
    for sp in ("train", "val", "test"):
        d = torch.load(os.path.join(PROC, f"{sp}_dataset.pt"), weights_only=False)
        F.append(np.hstack([d["y_features"].numpy(),
                            np.load(os.path.join(PROC, f"{sp}_extended_features.npy"))]))
        Y.append(d["y_primary"].numpy())
        PID.append(np.array([m[0] for m in d["metadata"]]))
        del d
    return np.concatenate(F), np.concatenate(Y), np.concatenate(PID)


def main():
    F19w, yw, pidw = load_window_level()
    Pw = Protocol.load_or_create(pidw, yw)

    # ---- the established window-level model, out of fold -------------------
    oof = np.zeros(len(F19w))
    cnt = np.zeros(len(F19w))
    for tr, te_p in Pw.folds(repeat=0):
        te = np.isin(pidw, te_p)
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=5000, C=0.1,
                                               class_weight="balanced"))
        clf.fit(F19w[tr], yw[tr])
        oof[te] += clf.predict_proba(F19w[te])[:, 1]
        cnt[te] += 1
    oof /= np.maximum(cnt, 1)
    base = Pw.report("19 descriptors, window-level + max (established)", oof, how="max")

    # ---- per-patient view --------------------------------------------------
    z = np.load(os.path.join(OUT, "contraction_features.npz"), allow_pickle=True)
    pid, A, S = z["pid"], z["A"], z["S"]
    lab, score19 = Pw.to_patient(oof, patients=pid, how="max")
    _, score19_mean = Pw.to_patient(oof, patients=pid, how="mean")
    B = np.column_stack([score19, score19_mean])

    blob = json.load(open(os.path.join(PROC, "folds.json")))
    asg = {k: v for k, v in blob["assignment"].items() if k in set(pid)}
    P = Protocol(pid, lab, asg)
    names = trajectory_feature_names()

    def run(name, M, model="lr", k_select=0):
        oo = np.zeros(len(M))
        cc = np.zeros(len(M))
        for tr, te_p in P.folds(repeat=0):
            te = np.isin(pid, te_p)
            Mtr, Mte = M[tr], M[te]
            if k_select and M.shape[1] > k_select:
                # univariate selection INSIDE the training fold only
                sc = []
                for j in range(M.shape[1]):
                    v, t = Mtr[:, j], lab[tr]
                    m = np.isfinite(v)
                    if m.sum() < 50 or len(np.unique(t[m])) < 2:
                        sc.append(0.5)
                        continue
                    sc.append(abs(roc_auc_score(t[m], v[m]) - 0.5))
                keep = np.argsort(sc)[::-1][:k_select]
                Mtr, Mte = Mtr[:, keep], Mte[:, keep]
            if model == "lr":
                clf = make_pipeline(
                    SimpleImputer(strategy="median"), StandardScaler(),
                    LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced"))
            else:
                clf = make_pipeline(
                    SimpleImputer(strategy="median"),
                    GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                               learning_rate=0.05, random_state=0))
            clf.fit(Mtr, lab[tr])
            oo[te] += clf.predict_proba(Mte)[:, 1]
            cc[te] += 1
        return P.report_patient_scores(name, lab, oo / np.maximum(cc, 1))

    print("\n--- does the trajectory ADD to the strongest baseline? ---")
    b = base["auroc"]
    res = {}
    res["base2"] = run("baseline score only (max+mean)", B)
    res["+traj"] = run("baseline + 45 trajectory features", np.hstack([B, A]))
    res["+traj8"] = run("baseline + top-8 trajectory (selected in-fold)",
                        np.hstack([B, A]), k_select=10)
    res["+traj_gb"] = run("baseline + trajectory (gradient boosting)",
                          np.hstack([B, A]), model="gb")
    res["+stage"] = run("baseline + stage-split trajectory", np.hstack([B, S]))
    res["+all8"] = run("baseline + top-10 of (traj + stage-split)",
                       np.hstack([B, A, S]), k_select=12)

    print(f"\nestablished baseline {b:.4f} | pre-registered bar +0.0642")
    for k, v in res.items():
        print(f"  {k:10s} {v['auroc'] - b:+.4f}")

    # the single most promising new feature, examined honestly
    j = names.index("baseline_shift_sd")
    v = A[:, j]
    m = np.isfinite(v)
    print(f"\nbaseline_shift_sd alone: AUROC {roc_auc_score(lab[m], v[m]):.4f} "
          f"on {m.sum()} patients")
    res["+bss"] = run("baseline + baseline_shift_sd only",
                      np.column_stack([B, v]))
    print(f"  added to baseline: {res['+bss']['auroc'] - b:+.4f}")


if __name__ == "__main__":
    main()
