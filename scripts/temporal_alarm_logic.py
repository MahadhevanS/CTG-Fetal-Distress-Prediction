"""
Temporal alarm logic for the deployed CTG device.

A monitor sees a CONTINUOUS trace, not isolated windows, so the deployment unit
is an alarm over time -- not a per-window label. Genuine fetal compromise is
sustained (hypoxia develops through repeated decelerations); spurious high
scores tend to be transient. Requiring risk to PERSIST for N consecutive
windows should therefore suppress false alarms while preserving true detections.

Costs no retraining -- it is a decision rule applied on top of per-window scores.

Reports, at MATCHED sensitivity so comparisons are fair:
  - window-level specificity / precision for each persistence level N
  - patient-level spurious-callout rate (fraction of NORMAL patients that ever
    alarm), which is what a clinician actually experiences as "false alarms"

NOTE: with a 2.5-min stride and 20-min windows, consecutive windows overlap by
87.5%, so they are far from independent. "N consecutive" means "risk sustained
~2.5*(N-1) minutes longer", not N independent confirmations.
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification
from src.training.train import create_patient_level_folds


def build(dev):
    e = CTGCrossformerEncoder(in_channels=2, seq_len=4800, cnn_channels=128, n_heads_cross=4,
                              n_heads_tf=8, n_tf_layers=4, d_ff=512, dropout=0.1, latent_dim=128)
    return CTGCrossformerForClassification(encoder=e, hidden_dim=128, dropout=0.3).to(dev)


@torch.no_grad()
def score(model, X, idx, dev, bs=64):
    out = np.zeros(len(idx), dtype=np.float32)
    for s in range(0, len(idx), bs):
        c = idx[s:s + bs]
        out[s:s + len(c)] = torch.sigmoid(model(X[c].to(dev)).squeeze(-1)).cpu().numpy()
    return out


def persist(flags, n):
    """True where flags have been continuously True for n windows."""
    if n <= 1:
        return flags.copy()
    out = np.zeros_like(flags)
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        out[i] = run >= n
    return out


def apply_rule(prob, pids, starts, thresh, n, order_cache):
    """Alarm mask over all windows, persistence applied within each patient."""
    alarm = np.zeros(len(prob), dtype=bool)
    for p, m in order_cache.items():
        alarm[m] = persist(prob[m] >= thresh, n)
    return alarm


def build_order_cache(pids, starts):
    cache = {}
    for p in np.unique(pids):
        m = np.where(pids == p)[0]
        cache[p] = m[np.argsort(starts[m])]
    return cache


def thresh_for_sens(prob, y, pids, starts, n, target, cache):
    """Highest threshold (best specificity) that still reaches target sensitivity."""
    best = None
    for t in np.round(np.linspace(0.005, 0.995, 199), 3):
        a = apply_rule(prob, pids, starts, t, n, cache)
        sens = a[y == 1].mean() if (y == 1).any() else 0.0
        if sens >= target:
            best = t
    return best


def report(prob, y, pids, starts, label, targets=(0.90, 0.80)):
    cache = build_order_cache(pids, starts)
    print(f"\n{'=' * 82}")
    print(f" {label}   ({len(y)} windows, {int(y.sum())} positive, "
          f"{len(np.unique(pids))} patients, AUROC {roc_auc_score(y, prob):.4f})")
    print(f"{'=' * 82}")
    plabel = {p: int(y[pids == p].max()) for p in np.unique(pids)}
    normals = [p for p in plabel if plabel[p] == 0]
    for target in targets:
        print(f"\n  --- matched sensitivity >= {target:.0%} ---")
        print(f"  {'rule':<10}{'thresh':<9}{'sens':<8}{'spec':<8}{'precision':<11}"
              f"{'windows alarmed':<17}{'normal patients alarmed'}")
        for n in (1, 2, 3, 4):
            t = thresh_for_sens(prob, y, pids, starts, n, target, cache)
            if t is None:
                print(f"  N={n:<8}unreachable at this sensitivity")
                continue
            a = apply_rule(prob, pids, starts, t, n, cache)
            sens = a[y == 1].mean()
            spec = (~a[y == 0]).mean()
            prec = a[y == 1].sum() / max(a.sum(), 1)
            fa = np.mean([a[cache[p]].any() for p in normals]) if normals else float("nan")
            print(f"  N={n:<8}{t:<9.3f}{sens:<8.3f}{spec:<8.3f}{prec:<11.3f}"
                  f"{a.mean():<17.3f}{fa:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Temporal alarm persistence evaluation")
    ap.add_argument("--checkpoint_dir", default="checkpoints/ctg_crossformer_invfreq/")
    ap.add_argument("--data_dir", default="data/processed_mil/")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = torch.load(os.path.join(a.data_dir, "train_dataset.pt"), map_location="cpu", weights_only=False)
    X, y = d["X"], d["y_primary"].numpy()
    pids = np.array([m[0] for m in d["metadata"]])
    starts = np.array([m[1] for m in d["metadata"]])
    folds = create_patient_level_folds(list(pids), torch.as_tensor(y), k_folds=5,
                                       secondary_labels=d["y_figo"])

    oof = np.full(len(y), np.nan, np.float32)
    for k, (_, vi) in enumerate(folds, 1):
        m = build(dev)
        m.load_state_dict(torch.load(
            os.path.join(a.checkpoint_dir, f"ctg_crossformer_fold_{k}_best.pth"),
            map_location=dev, weights_only=True))
        m.eval()
        oof[vi] = score(m, X, vi, dev)
        del m
        torch.cuda.empty_cache()
    assert not np.isnan(oof).any()
    report(oof, y, pids, starts, "CROSS-VALIDATION (out-of-fold)")

    tp = os.path.join(a.data_dir, "test_dataset.pt")
    if os.path.exists(tp):
        dt = torch.load(tp, map_location="cpu", weights_only=False)
        Xt, yt = dt["X"], dt["y_primary"].numpy()
        pt = np.array([m[0] for m in dt["metadata"]])
        st = np.array([m[1] for m in dt["metadata"]])
        acc = np.zeros(len(yt), dtype=np.float64)
        for k in range(1, 6):
            m = build(dev)
            m.load_state_dict(torch.load(
                os.path.join(a.checkpoint_dir, f"ctg_crossformer_fold_{k}_best.pth"),
                map_location=dev, weights_only=True))
            m.eval()
            acc += score(m, Xt, np.arange(len(yt)), dev)
            del m
            torch.cuda.empty_cache()
        report(acc / 5.0, yt, pt, st, "HELD-OUT TEST (5-fold ensemble)")


if __name__ == "__main__":
    main()
