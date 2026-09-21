"""
Fit per-fold Platt calibration for a CRP ensemble on the held-out validation
split, derive an operating point at a target sensitivity, and report test-set
performance once.

WHY: the five fold checkpoints discriminate comparably (test AUROC 0.765-0.813)
but emit probabilities on very different scales (mean 0.018 to 0.466), because
training applies BOTH sqrt-inverse oversampling (WeightedRandomSampler) AND a
full n_neg/n_pos pos_weight -- see the class-weighting comment in
src/models/train_ctg_crossformer.py. The raw ensemble over-predicts risk by
~3.5x and a fixed threshold means different things in different seeds.

Per-fold Platt scaling is a monotone affine correction in logit space, so it
leaves each fold's ranking (and therefore the explainability rho) untouched
while making the averaged probability mean what it says.

val_dataset.pt is never read by train_ctg_crossformer.py (it loads train + test
only), so fitting here does not leak.

Usage:
    python scripts/calibrate_crp.py
    python scripts/calibrate_crp.py --ckpt_dir checkpoints/repl_crp_s7 --target_sens 0.90
"""
import argparse, glob, json, os, sys
import numpy as np, torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             confusion_matrix, precision_recall_curve, auc)
from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification


def logit(p):
    c = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(c / (1 - c))


def ece(y, p, bins=10):
    e, n, edges = 0.0, len(p), np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e


def auprc(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    return auc(rc, pr)


def op(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
    return dict(sens=tp / (tp + fn) if tp + fn else 0.0,
                spec=tn / (tn + fp) if tn + fp else 0.0,
                ppv=tp / (tp + fp) if tp + fp else 0.0,
                alarm_rate=float((p >= t).mean()))


def fold_probs(ckpt_dir, X, dev):
    out = []
    for c in sorted(glob.glob(os.path.join(ckpt_dir, "ctg_crossformer_fold_*_best.pth"))):
        enc = CTGCrossformerEncoder(in_channels=2, seq_len=4800, cnn_channels=128, n_heads_cross=4,
                                    n_heads_tf=8, n_tf_layers=4, d_ff=512, dropout=0.1, latent_dim=128)
        m = CTGCrossformerForClassification(encoder=enc, hidden_dim=128, dropout=0.3).to(dev)
        m.load_state_dict(torch.load(c, map_location=dev, weights_only=True))
        m.eval()
        b = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                b.append(torch.sigmoid(m(X[i:i + 64].to(dev))).cpu().numpy().ravel())
        out.append(np.concatenate(b))
    if not out:
        raise SystemExit(f"no fold checkpoints found in {ckpt_dir}")
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", default="checkpoints/ctg_crossformer_crp")
    ap.add_argument("--data_dir", default="data/processed_mil")
    ap.add_argument("--target_sens", type=float, default=0.80)
    ap.add_argument("--out", default=None, help="where to write the calibrator (default: <ckpt_dir>/calibration.json)")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = {}
    for s in ("val", "test"):
        d = torch.load(os.path.join(a.data_dir, f"{s}_dataset.pt"), map_location="cpu", weights_only=False)
        ds[s] = (d["X"], d["y_primary"].numpy())

    P = {s: fold_probs(a.ckpt_dir, ds[s][0], dev) for s in ("val", "test")}
    yv, yt = ds["val"][1], ds["test"][1]

    print(f"model     : {a.ckpt_dir}")
    print(f"fold means: val {np.round(P['val'].mean(1), 4)}")
    print(f"            test {np.round(P['test'].mean(1), 4)}")
    print(f"prevalence: val {yv.mean():.4f}  test {yt.mean():.4f}\n")

    # --- fit per-fold Platt on validation -------------------------------------
    coefs, cv, ct = [], [], []
    for i in range(P["val"].shape[0]):
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(logit(P["val"][i]).reshape(-1, 1), yv)
        coefs.append((float(lr.coef_[0][0]), float(lr.intercept_[0])))
        cv.append(lr.predict_proba(logit(P["val"][i]).reshape(-1, 1))[:, 1])
        ct.append(lr.predict_proba(logit(P["test"][i]).reshape(-1, 1))[:, 1])
    ev, et = np.stack(cv).mean(0), np.stack(ct).mean(0)
    rv, rt = P["val"].mean(0), P["test"].mean(0)

    print("=" * 78)
    print(" CALIBRATION (test set)")
    print("=" * 78)
    print(f"  {'':<10}{'AUROC':>9}{'AUPRC':>9}{'Brier':>10}{'ECE':>9}{'mean p':>9}")
    for nm, e in (("raw", rt), ("platt", et)):
        print(f"  {nm:<10}{roc_auc_score(yt, e):>9.4f}{auprc(yt, e):>9.4f}"
              f"{brier_score_loss(yt, e):>10.4f}{ece(yt, e):>9.4f}{e.mean():>9.4f}")

    # --- operating point derived on validation --------------------------------
    fpr, tpr, thr = roc_curve(yv, ev)
    t = float(thr[int(np.argmax(tpr >= a.target_sens))])
    print("\n" + "=" * 78)
    print(f" OPERATING POINT @ {a.target_sens:.0%} SENSITIVITY (threshold derived on VALIDATION)")
    print("=" * 78)
    print(f"  threshold (val-derived): {t:.4f}")
    for split, yy, ee in (("val", yv, ev), ("TEST", yt, et)):
        m = op(yy, ee, t)
        print(f"  {split:<5} sens {m['sens']:.3f}  spec {m['spec']:.3f}  "
              f"PPV {m['ppv']:.3f}  alarm rate {m['alarm_rate']:.3f}")
    print(f"  sensitivity drift val->test: {op(yt, et, t)['sens'] - op(yv, ev, t)['sens']:+.3f}")

    out = a.out or os.path.join(a.ckpt_dir, "calibration.json")
    json.dump({"ckpt_dir": a.ckpt_dir, "method": "per-fold Platt on logit, fit on val_dataset.pt",
               "platt_coefficients": [{"slope": s, "intercept": b} for s, b in coefs],
               "target_sensitivity": a.target_sens, "threshold": t,
               "val": op(yv, ev, t), "test": op(yt, et, t),
               "test_auroc": float(roc_auc_score(yt, et)), "test_auprc": float(auprc(yt, et)),
               "test_brier": float(brier_score_loss(yt, et)), "test_ece": float(ece(yt, et))},
              open(out, "w"), indent=2)
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
