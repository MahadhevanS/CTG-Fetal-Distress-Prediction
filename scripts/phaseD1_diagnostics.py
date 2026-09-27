"""
Phase D1 diagnostics (docs/phaseD_deepfhr_reproduction_protocol.md, results/phaseD_deepfhr/phaseD1_report.md).
Two questions, both needed to explain why the CNN (D1's official result) falls short of the published ceiling
while staying faithful to the paper's own image-level-random-split protocol:

  1. Does the leakage signal exist at all in the reconstructed images? (a trivial pixel-level classifier, under
     the identical split, decoupled from any CNN training difficulty)
  2. Is the CNN's shortfall a training-duration problem, or a hard capacity/architecture ceiling? (train-set AUC
     vs. epoch count, on one fold -- if train AUC itself plateaus early, more epochs cannot help)

    python scripts/phaseD1_diagnostics.py
"""
import os, sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import scripts.phaseD1_deepfhr_paperstyle as d1

OUT_DIR = d1.OUT_DIR


def pixel_lr_control(X, y):
    """Q1: a trivial linear model on flattened raw pixels, same image-random 10-fold split as D1."""
    Xflat = X.reshape(len(X), -1)
    kf = KFold(10, shuffle=True, random_state=42)
    aucs = []
    for tr, te in kf.split(Xflat):
        sc = StandardScaler().fit(Xflat[tr])
        clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(sc.transform(Xflat[tr]), y[tr])
        p = clf.predict_proba(sc.transform(Xflat[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return {"mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)), "per_fold_auc": [float(a) for a in aucs]}


def within_vs_cross_record_distance(X, rid, n_records=80, seed=0):
    """Q1b: are a record's own 6 images closer to each other (in raw pixel space) than to other records' images?"""
    Xflat = X.reshape(len(X), -1)
    recs = np.unique(rid)[:n_records]
    rng = np.random.default_rng(seed)
    same, diff = [], []
    for r in recs:
        idx = np.where(rid == r)[0]
        if len(idx) < 2:
            continue
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                same.append(float(np.linalg.norm(Xflat[idx[i]] - Xflat[idx[j]])))
        other = rng.choice(np.where(rid != r)[0], 6, replace=False)
        for o in other:
            diff.append(float(np.linalg.norm(Xflat[idx[0]] - Xflat[o])))
    return {"mean_within_record_L2": float(np.mean(same)), "mean_cross_record_L2": float(np.mean(diff)),
           "ratio": float(np.mean(diff) / np.mean(same))}


def epoch_convergence_check(X, y, epochs_grid=(20, 100, 300)):
    """Q2: train-set AUC vs. epoch count on one fold. If train AUC plateaus early, the shortfall is capacity/
    image-construction, not training duration -- more epochs cannot fix it."""
    cfg = d1.load_cfg()
    kf = KFold(10, shuffle=True, random_state=42)
    tr_idx, te_idx = next(iter(kf.split(X)))
    rows = []
    for ep in epochs_grid:
        cfg["training"]["epochs"] = ep
        probs_te, probs_tr = d1.train_fold(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], cfg)
        rows.append({"epochs": ep, "test_auc": float(roc_auc_score(y[te_idx], probs_te)),
                    "train_auc": float(roc_auc_score(y[tr_idx], probs_tr))})
        print(f"  epochs={ep:4d}  test AUC={rows[-1]['test_auc']:.4f}  train AUC={rows[-1]['train_auc']:.4f}")
    return rows


def main():
    d = np.load(d1.IMG_CACHE)
    X, y, rid = d["X"], d["y"], d["rid"]

    print("=== Q1: pixel-level control (does the leakage signal exist, independent of CNN training?) ===")
    q1 = pixel_lr_control(X, y)
    print(f"  pixel-LR image-random 10-fold AUC: {q1['mean_auc']:.4f} +/- {q1['std_auc']:.4f}")

    print("\n=== Q1b: within-record vs cross-record image distance ===")
    q1b = within_vs_cross_record_distance(X, rid)
    print(f"  mean L2 distance: within-record pairs {q1b['mean_within_record_L2']:.3f}  vs cross-record pairs "
          f"{q1b['mean_cross_record_L2']:.3f}  (ratio {q1b['ratio']:.2f}x)")

    print("\n=== Q2: is the CNN's shortfall a training-duration problem or a hard ceiling? ===")
    q2 = epoch_convergence_check(X, y)

    out = {"pixel_lr_control": q1, "within_vs_cross_record_distance": q1b, "epoch_convergence_check": q2}
    json.dump(out, open(os.path.join(OUT_DIR, "d1_diagnostics.json"), "w"), indent=2, default=float)
    print(f"\nSaved -> {OUT_DIR}/d1_diagnostics.json")


if __name__ == "__main__":
    main()
