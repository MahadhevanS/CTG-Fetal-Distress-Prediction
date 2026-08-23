"""
Evaluate every stored CTG-Crossformer ensemble on the held-out val and test
splits, in one table, from the saved checkpoints -- no retraining.

Reports discrimination (AUROC/AUPRC), calibration (Brier/ECE, mean predicted
probability vs true prevalence), and the operating point at both the deployed
0.30 cut and a validation-derived cut targeting a fixed sensitivity.

WHY BOTH OPERATING POINTS: a fixed 0.30 is not comparable across these models --
they emit probabilities on different scales (see docs/calibration_test_plan.md).
The validation-derived column is the like-for-like comparison; the 0.30 column
is what each model would currently do if deployed as-is.

Usage:
    python scripts/eval_all_models.py
    python scripts/eval_all_models.py --models crp_s42 crp_s1 --target_sens 0.90
    python scripts/eval_all_models.py --json docs/model_metrics.json
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             confusion_matrix, precision_recall_curve, auc,
                             accuracy_score)
from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification

# name -> (checkpoint dir, one-line description)
MODELS = {
    "crp_s42":       ("checkpoints/ctg_crossformer_crp",             "DELIVERED -- clinical-relational pretrain, seed 42"),
    "crp_s1":        ("checkpoints/repl_crp_s1",                     "CRP replication, seed 1"),
    "crp_s7":        ("checkpoints/repl_crp_s7",                     "CRP replication, seed 7"),
    "base_s1":       ("checkpoints/repl_base_s1",                    "paired baseline (no CRP), seed 1"),
    "base_s7":       ("checkpoints/repl_base_s7",                    "paired baseline (no CRP), seed 7"),
    "invfreq":       ("checkpoints/ctg_crossformer_invfreq",         "pre-CRP inverse_freq baseline"),
    "sqrtinv":       ("checkpoints/ctg_crossformer_sqrtinv",         "sqrt-inverse class weighting"),
    "labelconf":     ("checkpoints/ctg_crossformer_labelconf",       "label-confidence variant"),
    "mil_nested":    ("checkpoints/ctg_crossformer_mil_nested",      "nested early stopping, no class weight"),
    "foldmatched":   ("checkpoints/ctg_crossformer_mil_foldmatched", "fold-matched vs Model 8"),
}


def ece(y, p, bins=10):
    e, n, edges = 0.0, len(p), np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


def auprc(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    return float(auc(rc, pr))


def op(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
    return dict(acc=float(accuracy_score(y, (p >= t).astype(int))),
                sens=float(tp / (tp + fn)) if tp + fn else 0.0,
                spec=float(tn / (tn + fp)) if tn + fp else 0.0,
                ppv=float(tp / (tp + fp)) if tp + fp else 0.0,
                alarm=float((p >= t).mean()))


def fold_probs(ckpt_dir, X, dev, cache=None):
    if cache and os.path.exists(cache):
        return np.load(cache)
    ck = sorted(glob.glob(os.path.join(ckpt_dir, "ctg_crossformer_fold_*_best.pth")))
    if not ck:
        return None
    rows = []
    for c in ck:
        enc = CTGCrossformerEncoder(in_channels=2, seq_len=4800, cnn_channels=128, n_heads_cross=4,
                                    n_heads_tf=8, n_tf_layers=4, d_ff=512, dropout=0.1, latent_dim=128)
        m = CTGCrossformerForClassification(encoder=enc, hidden_dim=128, dropout=0.3).to(dev)
        m.load_state_dict(torch.load(c, map_location=dev, weights_only=True))
        m.eval()
        b = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                b.append(torch.sigmoid(m(X[i:i + 64].to(dev))).cpu().numpy().ravel())
        rows.append(np.concatenate(b))
    P = np.stack(rows)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, P)
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--data_dir", default="data/processed_mil")
    ap.add_argument("--target_sens", type=float, default=0.80)
    ap.add_argument("--cache_dir", default=None, help="optional dir to cache per-fold probabilities")
    ap.add_argument("--json", default=None, help="also write the table as JSON")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds, ys = {}, {}
    for s in ("val", "test"):
        d = torch.load(os.path.join(a.data_dir, f"{s}_dataset.pt"), map_location="cpu", weights_only=False)
        ds[s], ys[s] = d["X"], d["y_primary"].numpy()
    print(f"device {dev} | val n={len(ys['val'])} pos={ys['val'].sum()} ({ys['val'].mean():.3f}) | "
          f"test n={len(ys['test'])} pos={ys['test'].sum()} ({ys['test'].mean():.3f})\n")

    rows = {}
    for name in a.models:
        if name not in MODELS:
            print(f"[skip] unknown model '{name}'"); continue
        ckpt_dir, desc = MODELS[name]
        if not os.path.isdir(ckpt_dir):
            print(f"[skip] {name}: {ckpt_dir} missing"); continue
        P = {}
        for s in ("val", "test"):
            cache = os.path.join(a.cache_dir, f"P_{name}_{s}.npy") if a.cache_dir else None
            P[s] = fold_probs(ckpt_dir, ds[s], dev, cache)
        if P["test"] is None:
            print(f"[skip] {name}: no fold checkpoints"); continue
        ev, et = P["val"].mean(0), P["test"].mean(0)
        fpr, tpr, thr = roc_curve(ys["val"], ev)
        tval = float(thr[int(np.argmax(tpr >= a.target_sens))])
        rows[name] = {
            "checkpoint_dir": ckpt_dir, "description": desc,
            "fold_mean_probs_test": [round(float(x), 4) for x in P["test"].mean(1)],
            "val_auroc": float(roc_auc_score(ys["val"], ev)), "val_auprc": auprc(ys["val"], ev),
            "test_auroc": float(roc_auc_score(ys["test"], et)), "test_auprc": auprc(ys["test"], et),
            "test_brier": float(brier_score_loss(ys["test"], et)), "test_ece": ece(ys["test"], et),
            "test_mean_prob": float(et.mean()),
            "at_0.30": op(ys["test"], et, 0.30),
            "val_derived_threshold": tval,
            f"at_val_thr_{a.target_sens:g}sens": op(ys["test"], et, tval),
        }
        print(f"  scored {name:<12} {desc}")

    ORD = list(rows)
    print("\n" + "=" * 104)
    print(" DISCRIMINATION + CALIBRATION (test set)")
    print("=" * 104)
    print(f"  {'model':<13}{'AUROC':>8}{'AUPRC':>8}{'Brier':>9}{'ECE':>8}{'mean p':>9}"
          f"{'  (prevalence 0.046)':<22}")
    for n in ORD:
        r = rows[n]
        print(f"  {n:<13}{r['test_auroc']:>8.4f}{r['test_auprc']:>8.4f}{r['test_brier']:>9.4f}"
              f"{r['test_ece']:>8.4f}{r['test_mean_prob']:>9.4f}   {r['description']}")

    print("\n" + "=" * 104)
    print(" OPERATING POINT -- deployed 0.30 vs threshold derived on VALIDATION")
    print("=" * 104)
    key = f"at_val_thr_{a.target_sens:g}sens"
    print(f"  {'model':<13}| {'@0.30 acc':>10}{'sens':>7}{'spec':>7}{'PPV':>7} | "
          f"{'thr(val)':>9}{'acc':>7}{'sens':>7}{'spec':>7}{'PPV':>7}")
    for n in ORD:
        r, b, v = rows[n], rows[n]["at_0.30"], rows[n][key]
        print(f"  {n:<13}| {b['acc']:>10.4f}{b['sens']:>7.3f}{b['spec']:>7.3f}{b['ppv']:>7.3f} | "
              f"{r['val_derived_threshold']:>9.4f}{v['acc']:>7.4f}{v['sens']:>7.3f}{v['spec']:>7.3f}{v['ppv']:>7.3f}")

    print("\n" + "=" * 104)
    print(" PER-FOLD MEAN PROBABILITY (test) -- the calibration spread")
    print("=" * 104)
    for n in ORD:
        print(f"  {n:<13}{rows[n]['fold_mean_probs_test']}")

    if a.json:
        json.dump({"prevalence": {s: float(ys[s].mean()) for s in ("val", "test")},
                   "target_sensitivity": a.target_sens, "models": rows},
                  open(a.json, "w"), indent=2)
        print(f"\n  written: {a.json}")


if __name__ == "__main__":
    main()
