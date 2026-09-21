"""
Post-hoc analysis of the Dang et al. reproduction ladder.

Every slice below is computed from the SAME saved out-of-fold predictions
written by reproduce_dang2026.py. Nothing is retrained to obtain a slice, so a
cohort or aggregation change cannot quietly become a tuning knob.

Track A (paper) and Track B (frozen protocol) are reported in separate tables
and never combined into one number.

Run from the repo root:
    python scripts/reproduce_dang2026_report.py
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE)

REPRO = "results/phase8/repro"
RAW = "data/raw/ctu-chb-intrapartum"


def missing_ratio(rec):
    a = np.fromfile(os.path.join(RAW, f"{rec}.dat"), dtype="<i2").reshape(-1, 2)
    return float((a[:, 0] <= 0).mean())


def hanley_ci(auc, n1, n0):
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2)
                  + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
    return auc - 1.96 * se, auc + 1.96 * se


def platt_pooled(oof, y, fold):
    """Per-fold Platt scaling before pooling.

    Independently trained folds produce logits on different scales; pooling raw
    logits biases the pooled AUROC low. The paper says only "pooled predictions"
    so both are reported.
    """
    z = np.full(len(oof), np.nan)
    for k in np.unique(fold[fold >= 0]):
        m = fold == k
        lr = LogisticRegression(max_iter=1000)
        lr.fit(oof[m].reshape(-1, 1), y[m])
        z[m] = lr.predict_proba(oof[m].reshape(-1, 1))[:, 1]
    ok = ~np.isnan(z)
    return roc_auc_score(y[ok], z[ok])


def patient_level(oof, y, pid, how="max"):
    fn = {"max": np.max, "mean": np.mean}[how]
    pats = np.array(sorted(set(pid.tolist())))
    sc = np.array([fn(oof[pid == p]) for p in pats])
    lab = np.array([int(y[pid == p].max()) for p in pats])
    return roc_auc_score(lab, sc), lab, sc


def main():
    files = sorted(glob.glob(os.path.join(REPRO, "*.npz")))
    if not files:
        print("no runs found -- run scripts/reproduce_dang2026.py first")
        return

    # cleanest-404 slice, the paper's implied cohort
    allrec = sorted({os.path.basename(p)[:-4]
                     for p in glob.glob(os.path.join(RAW, "*.hea"))})
    mr = {r: missing_ratio(r) for r in allrec}

    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        tag = os.path.basename(f)[:-4]
        oof, y, pid, fold = z["oof"], z["y"], z["pid"], z["fold"]
        cfg = json.loads(str(z["config"]))
        ok = ~np.isnan(oof)
        win = roc_auc_score(y[ok], oof[ok])
        win_platt = platt_pooled(oof[ok], y[ok], fold[ok])
        mof = float(np.nanmean(z["per_fold"]))
        pat, lab, sc = patient_level(oof[ok], y[ok], pid[ok])

        pats = np.array(sorted(set(pid[ok].tolist())))
        clean = set(sorted(pats, key=lambda p: mr.get(p, 1.0))[:404])
        m404 = ok & np.isin(pid, list(clean))
        win404 = roc_auc_score(y[m404], oof[m404])
        pat404, lab404, _ = patient_level(oof[m404], y[m404], pid[m404])

        lo, hi = hanley_ci(pat, int(lab.sum()), int((1 - lab).sum()))
        rows.append(dict(tag=tag, substrate=cfg["substrate"], split=cfg["split"],
                         selection=cfg["selection"], n_win=int(ok.sum()),
                         n_pat=len(pats), win=win, win_platt=win_platt,
                         mean_of_folds=mof, pat=pat, pat_lo=lo, pat_hi=hi,
                         win404=win404, pat404=pat404,
                         per_fold=list(np.round(z["per_fold"], 4))))

    print("=" * 100)
    print("TRACK A -- PAPER'S OWN METRIC (window-level, pooled across folds)")
    print("=" * 100)
    print(f"{'run':34s} {'subst':14s} {'sel':11s} {'pooled':>8s} {'+platt':>8s} "
          f"{'meanfold':>9s} {'clean404':>9s}")
    for r in rows:
        print(f"{r['tag']:34s} {r['substrate']:14s} {r['selection']:11s} "
              f"{r['win']:8.4f} {r['win_platt']:8.4f} {r['mean_of_folds']:9.4f} "
              f"{r['win404']:9.4f}")
    print(f"\n{'Dang et al. 2026 (reported)':34s} {'404 patients':14s} "
          f"{'outer_best':11s} {0.822:8.4f} {'':8s} {0.825:9.4f}")

    print()
    print("=" * 100)
    print("TRACK B -- FROZEN PROJECT PROTOCOL (patient-level, max-aggregated)")
    print("=" * 100)
    print(f"{'run':34s} {'patients':>9s} {'AUROC':>8s} {'95% CI':>18s} {'clean404':>9s}")
    for r in rows:
        print(f"{r['tag']:34s} {r['n_pat']:9d} {r['pat']:8.4f} "
              f"  [{r['pat_lo']:.3f}-{r['pat_hi']:.3f}] {r['pat404']:9.4f}")
    print(f"{'19-descriptor LR (frozen baseline)':34s} {547:9d} {0.7271:8.4f} "
          f"  [0.670-0.779] {0.7530:9.4f}")

    print()
    print("per-fold test AUROC (window level):")
    for r in rows:
        print(f"  {r['tag']:34s} {r['per_fold']}")
    print(f"  {'Dang et al. reported per-fold':34s} "
          f"[0.903, 0.802, 0.808, 0.806, 0.808]")

    with open(os.path.join(REPRO, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    print(f"\nwrote {REPRO}/summary.json")


if __name__ == "__main__":
    main()
