"""
Operating-point-targeted knowledge fusion.

The previous combiner was a logistic regression fit to minimise log-loss. That
optimises the whole probability surface, not the metric a screening device is
judged on. It bought AUPRC (+0.027 on test, the only knowledge effect in this
project that has replicated out-of-sample) at the cost of specificity at fixed
sensitivity (-0.032 at 90%, -0.142 at 80%).

This instead combines in logit space with ONE tunable scalar:

    score = logit(p_model) + alpha * logit(p_knowledge)

and selects alpha to maximise specificity at 90% sensitivity. One parameter
keeps the overfitting risk low -- important given that four separate CV gains
have failed to replicate on test in this project -- and alpha is directly
interpretable as how much authority clinical knowledge is given in the final
decision. alpha = 0 recovers the model exactly, so the fusion can never be
forced on if it does not help.

Selection is NESTED: for each outer CV fold, alpha is chosen on the other four
folds and evaluated on the held-out one, so the reported CV number is not
tuned on the data it is measured on.
"""
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.knowledge.figo import derive_figo_criteria_flags_torch
from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification
from src.training.train import create_patient_level_folds

CK = "checkpoints/ctg_crossformer_invfreq"
ALPHAS = np.round(np.arange(0.0, 1.51, 0.05), 2)
TARGET_SENS = 0.90
EPS = 1e-6


def build(dev):
    e = CTGCrossformerEncoder(in_channels=2, seq_len=4800, cnn_channels=128, n_heads_cross=4,
                              n_heads_tf=8, n_tf_layers=4, d_ff=512, dropout=0.1, latent_dim=128)
    return CTGCrossformerForClassification(encoder=e, hidden_dim=128, dropout=0.3).to(dev)


@torch.no_grad()
def score_windows(m, X, idx, dev, bs=64):
    o = np.zeros(len(idx), np.float32)
    for s in range(0, len(idx), bs):
        c = idx[s:s + bs]
        o[s:s + len(c)] = torch.sigmoid(m(X[c].to(dev)).squeeze(-1)).cpu().numpy()
    return o


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def spec_at_sens(y, s, target=TARGET_SENS):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.where(tpr >= target)[0]
    return float(1.0 - fpr[i].min()) if len(i) else np.nan


def knowledge_probs(K, y, fit_idx, apply_idx):
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    lr.fit(K[fit_idx], y[fit_idx])
    return lr.predict_proba(K[apply_idx])[:, 1]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load("data/processed_mil/train_dataset.pt", map_location="cpu", weights_only=False)
    X, y, yf = d["X"], d["y_primary"].numpy(), d["y_features"]
    pids = np.array([m[0] for m in d["metadata"]])
    folds = create_patient_level_folds(list(pids), torch.as_tensor(y), k_folds=5,
                                       secondary_labels=d["y_figo"])

    prob = np.full(len(y), np.nan, np.float32)
    for k, (_, vi) in enumerate(folds, 1):
        m = build(dev)
        m.load_state_dict(torch.load(f"{CK}/ctg_crossformer_fold_{k}_best.pth",
                                     map_location=dev, weights_only=True))
        m.eval()
        prob[vi] = score_windows(m, X, vi, dev)
        del m
        torch.cuda.empty_cache()

    E = np.delete(np.load("data/processed_mil/train_extended_features.npy"), 7, axis=1)
    FL = derive_figo_criteria_flags_torch(yf.float()).numpy()
    K = np.column_stack([FL, E])

    # ---- NESTED alpha selection --------------------------------------- #
    rows_model, rows_fused, chosen = [], [], []
    for oi, (tr, va) in enumerate(folds):
        inner = [(a, b) for j, (a, b) in enumerate(folds) if j != oi]
        best_a, best_s = 0.0, -np.inf
        for a in ALPHAS:
            scores = []
            for itr, iva in inner:
                itr = np.intersect1d(itr, tr)
                iva = np.intersect1d(iva, tr)
                if len(iva) < 20 or y[iva].sum() < 3:
                    continue
                kp = knowledge_probs(K, y, itr, iva)
                s = logit(prob[iva]) + a * logit(kp)
                scores.append(spec_at_sens(y[iva], s))
            if scores and np.nanmean(scores) > best_s:
                best_s, best_a = np.nanmean(scores), a
        chosen.append(best_a)
        kp_va = knowledge_probs(K, y, tr, va)
        fused = logit(prob[va]) + best_a * logit(kp_va)
        rows_model.append((roc_auc_score(y[va], prob[va]),
                           average_precision_score(y[va], prob[va]),
                           spec_at_sens(y[va], prob[va]),
                           spec_at_sens(y[va], prob[va], 0.80)))
        rows_fused.append((roc_auc_score(y[va], fused),
                           average_precision_score(y[va], fused),
                           spec_at_sens(y[va], fused),
                           spec_at_sens(y[va], fused, 0.80)))

    M, F = np.array(rows_model), np.array(rows_fused)
    print("CROSS-VALIDATION (nested alpha selection, per-fold mean)\n")
    print(f"  alphas chosen per fold: {chosen}\n")
    print(f"  {'variant':<22}{'AUROC':<9}{'AUPRC':<9}{'spec@90s':<11}{'spec@80s':<10}")
    print("  " + "-" * 60)
    print(f"  {'model alone':<22}{M[:,0].mean():<9.4f}{M[:,1].mean():<9.4f}"
          f"{np.nanmean(M[:,2]):<11.4f}{np.nanmean(M[:,3]):<10.4f}")
    print(f"  {'+ knowledge (tuned)':<22}{F[:,0].mean():<9.4f}{F[:,1].mean():<9.4f}"
          f"{np.nanmean(F[:,2]):<11.4f}{np.nanmean(F[:,3]):<10.4f}")
    print(f"\n  deltas: AUROC {F[:,0].mean()-M[:,0].mean():+.4f}   "
          f"AUPRC {F[:,1].mean()-M[:,1].mean():+.4f}   "
          f"spec@90s {np.nanmean(F[:,2])-np.nanmean(M[:,2]):+.4f}   "
          f"spec@80s {np.nanmean(F[:,3])-np.nanmean(M[:,3]):+.4f}")

    # ---- HELD-OUT TEST: alpha from CV (mode), combiner fit on CV pool --- #
    alpha = float(np.median(chosen))
    dt = torch.load("data/processed_mil/test_dataset.pt", map_location="cpu", weights_only=False)
    Xt, yt, yft = dt["X"], dt["y_primary"].numpy(), dt["y_features"]
    acc = np.zeros(len(yt))
    for k in range(1, 6):
        m = build(dev)
        m.load_state_dict(torch.load(f"{CK}/ctg_crossformer_fold_{k}_best.pth",
                                     map_location=dev, weights_only=True))
        m.eval()
        acc += score_windows(m, Xt, np.arange(len(yt)), dev)
        del m
        torch.cuda.empty_cache()
    pt = acc / 5.0
    Et = np.delete(np.load("data/processed_mil/test_extended_features.npy"), 7, axis=1)
    Kt = np.column_stack([derive_figo_criteria_flags_torch(yft.float()).numpy(), Et])

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    lr.fit(K, y)
    kpt = lr.predict_proba(Kt)[:, 1]
    fused_t = logit(pt) + alpha * logit(kpt)

    print(f"\n\nHELD-OUT TEST (alpha = {alpha} from CV; combiner fit on CV pool only)\n")
    print(f"  {'variant':<22}{'AUROC':<9}{'AUPRC':<9}{'spec@90s':<11}{'spec@80s':<10}")
    print("  " + "-" * 60)
    for nm, s in [("model alone", pt), ("+ knowledge (tuned)", fused_t)]:
        print(f"  {nm:<22}{roc_auc_score(yt,s):<9.4f}{average_precision_score(yt,s):<9.4f}"
              f"{spec_at_sens(yt,s):<11.4f}{spec_at_sens(yt,s,0.80):<10.4f}")
    print(f"\n  deltas: AUROC {roc_auc_score(yt,fused_t)-roc_auc_score(yt,pt):+.4f}   "
          f"AUPRC {average_precision_score(yt,fused_t)-average_precision_score(yt,pt):+.4f}   "
          f"spec@90s {spec_at_sens(yt,fused_t)-spec_at_sens(yt,pt):+.4f}   "
          f"spec@80s {spec_at_sens(yt,fused_t,0.80)-spec_at_sens(yt,pt,0.80):+.4f}")


if __name__ == "__main__":
    main()
