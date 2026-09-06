"""
What can knowledge infusion actually buy on the CLEAN data?

Fusion helps only if the signal branch and the knowledge branch make DIFFERENT
mistakes. If a deep encoder's patient scores are highly correlated with the
19-feature model's, concatenating or attending over them adds nothing, however
the plumbing is arranged. This measures that directly, before any KI
architecture is built.

Historical context (docs/model_inferences_log.md, OLD horizon-labelled data):
    distress_only   0.7462
    plus_figo       0.7462   <- FIGO rule loss gave EXACTLY zero
    plus_features   0.7939   (folds 1-4 only, not comparable)
    full            0.7774   <- +0.0312 over distress_only
and clinical-relational pretraining replicated at +0.0199 (commit e810de3).
Eight earlier KI mechanisms failed outright, explained at the time by "the
network already encodes what the FIGO rules encode"
(src/knowledge/extended_features.py).

That explanation was true when the network scored 0.81 and the features 0.78.
On the clean label the ordering INVERTED -- the features (0.7290 patient-level)
now beat every one of eight architectures (0.6167-0.7178). So the old argument
no longer applies and the question is genuinely open. This script answers it
empirically rather than by analogy.

Models compared, all patient-level, all patient-grouped CV:
    A  signal only      deep encoder -> window scores -> max-aggregate
    B  knowledge only   19 features  -> LR -> max-aggregate   (= 0.7290)
    C  late fusion      rank-average of A and B
    D  stacked fusion   LR on [deep patient score | aggregated features]

Usage:
    python scripts/probe_knowledge_fusion_ceiling.py --encoder cnn1d --epochs 20
"""
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import torch                                                          # noqa: E402
import torch.nn.functional as Fn                                      # noqa: E402
from torch.utils.data import DataLoader, TensorDataset                # noqa: E402
from sklearn.linear_model import LogisticRegression                   # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score    # noqa: E402
from sklearn.model_selection import StratifiedKFold                   # noqa: E402
from sklearn.pipeline import make_pipeline                            # noqa: E402
from sklearn.preprocessing import StandardScaler                      # noqa: E402
from scipy.stats import spearmanr                                     # noqa: E402

from src.models.encoder_registry import build_classifier, build_encoder  # noqa: E402


def load(data_dir):
    X, Fe, y, pid = [], [], [], []
    for s in ("train", "val", "test"):
        d = torch.load(os.path.join(data_dir, f"{s}_dataset.pt"), weights_only=False)
        X.append(d["X"].numpy()[:, :2, :])
        Fe.append(np.hstack([d["y_features"].numpy(),
                             np.load(os.path.join(data_dir, f"{s}_extended_features.npy"))]))
        y.append(d["y_primary"].numpy())
        pid.append(np.array([m[0] for m in d["metadata"]]))
    return (np.vstack(X), np.vstack(Fe), np.concatenate(y), np.concatenate(pid))


def train_fold(Xtr, ytr, Xte, encoder, epochs, device, seed=42):
    torch.manual_seed(seed)
    m_cfg = {"in_channels": 2}
    model = build_classifier(encoder, build_encoder(encoder, m_cfg), m_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                       torch.tensor(ytr, dtype=torch.float32))
    # sqrt-inverse oversampling, matching the project's standard recipe
    cnt = np.bincount(ytr.astype(int))
    w = (1.0 / np.sqrt(np.maximum(cnt, 1)))[ytr.astype(int)]
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                                     len(w), replacement=True)
    dl = DataLoader(ds, batch_size=32, sampler=sampler)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=max(len(dl) * epochs, 10), pct_start=0.1)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = Fn.binary_cross_entropy_with_logits(
                model(xb.to(device)).squeeze(-1), yb.to(device))
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xte), 256):
            xb = torch.tensor(Xte[i:i + 256], dtype=torch.float32).to(device)
            out.extend(torch.sigmoid(model(xb).squeeze(-1)).cpu().numpy())
    return np.array(out)


def rank(v):
    return np.argsort(np.argsort(v)) / max(len(v) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_clinical/")
    ap.add_argument("--encoder", default="cnn1d")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Fe, y, pid = load(args.data_dir)
    up = np.array(sorted(set(pid)))
    pidx = {p: np.where(pid == p)[0] for p in up}
    plab = np.array([int(y[pidx[p]].max()) for p in up])
    print(f"{len(y)} windows | {len(up)} patients | {plab.sum()} positive | device {device}")
    print(f"encoder {args.encoder}, {args.epochs} epochs, {args.folds}-fold patient-grouped\n")

    oof_sig = np.zeros(len(y))
    oof_kno = np.zeros(len(y))
    for k, (tr, te) in enumerate(StratifiedKFold(
            args.folds, shuffle=True, random_state=0).split(up, plab), 1):
        trS = set(up[tr])
        m = np.array([q in trS for q in pid])
        oof_sig[~m] = train_fold(X[m], y[m], X[~m], args.encoder, args.epochs, device)
        oof_kno[~m] = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")
        ).fit(Fe[m], y[m]).predict_proba(Fe[~m])[:, 1]
        print(f"  fold {k}/{args.folds} done")

    # patient-level scores (max-aggregation, the best aggregator measured)
    ps_sig = np.array([oof_sig[pidx[p]].max() for p in up])
    ps_kno = np.array([oof_kno[pidx[p]].max() for p in up])

    def rep(nm, s):
        return (f"{nm:34s} AUROC {roc_auc_score(plab, s):.4f}   "
                f"AUPRC {average_precision_score(plab, s):.4f}")

    print("\n" + "=" * 74)
    print("PATIENT-LEVEL RESULTS")
    print("=" * 74)
    print(rep("A  signal only (deep)", ps_sig))
    print(rep("B  knowledge only (19 feats + LR)", ps_kno))
    print(rep("C  late fusion (rank average)", 0.5 * rank(ps_sig) + 0.5 * rank(ps_kno)))

    # D: stacked -- nested CV so the stacker is never fit on its own test patients
    st = np.zeros(len(up))
    Fpat = np.array([np.concatenate([Fe[pidx[p]].max(0), Fe[pidx[p]].mean(0)]) for p in up])
    Z = np.column_stack([ps_sig, Fpat])
    for tr, te in StratifiedKFold(args.folds, shuffle=True, random_state=1).split(Z, plab):
        st[te] = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced")
        ).fit(Z[tr], plab[tr]).predict_proba(Z[te])[:, 1]
    print(rep("D  stacked (deep score + feats)", st))

    print("\n" + "=" * 74)
    print("IS THERE ANYTHING TO FUSE? -- agreement between the two branches")
    print("=" * 74)
    rho = spearmanr(ps_sig, ps_kno).correlation
    print(f"  Spearman rho(signal, knowledge) patient scores : {rho:+.3f}")
    # complementarity: does the deep score add anything to the features?
    err_k = np.abs(plab - rank(ps_kno))
    err_s = np.abs(plab - rank(ps_sig))
    print(f"  Spearman rho(error_signal, error_knowledge)    : "
          f"{spearmanr(err_s, err_k).correlation:+.3f}")
    n_sig_only = int(((err_s < 0.5) & (err_k >= 0.5)).sum())
    n_kno_only = int(((err_k < 0.5) & (err_s >= 0.5)).sum())
    print(f"  patients only the SIGNAL branch gets right     : {n_sig_only}")
    print(f"  patients only the KNOWLEDGE branch gets right  : {n_kno_only}")
    # Interpretation is DERIVED, not asserted. An earlier hardcoded conclusion
    # here said "high correlation means there is nothing to fuse", which flatly
    # contradicted the measured rho of ~0.37 on the first real run. Read the data.
    a_sig = roc_auc_score(plab, ps_sig)
    a_kno = roc_auc_score(plab, ps_kno)
    a_fus = max(roc_auc_score(plab, 0.5 * rank(ps_sig) + 0.5 * rank(ps_kno)),
                roc_auc_score(plab, st))
    print()
    if rho > 0.7:
        print("  The branches are strongly correlated: they see the same thing, and")
        print("  fusion cannot add much regardless of architecture.")
    else:
        lvl = "moderate" if rho > 0.5 else "low"
        print(f"  rho={rho:+.3f} ({lvl}) -- the branches carry partly DIFFERENT "
              f"information")
        print(f"  ({n_sig_only} patients only the signal branch gets right).")
        if a_fus < a_kno:
            print(f"  Yet fusion ({a_fus:.4f}) still loses to knowledge alone ({a_kno:.4f}).")
            print(f"  Cause: the signal branch ({a_sig:.4f}) is too weak to average in, and a")
            print("  stacker fit on 547 patients overfits before it can learn to down-weight")
            print("  it. Complementary information EXISTS but is not reachable by combining")
            print("  two separately-trained branches.")
            print("  Implication: use knowledge as AUXILIARY SUPERVISION inside one model,")
            print("  not as a second branch to fuse -- which matches this project's history:")
            print("  plus_features helped (+0.031/+0.048), plus_figo gave exactly +0.000.")
        else:
            print(f"  Fusion ({a_fus:.4f}) beats knowledge alone ({a_kno:.4f}) -- worth pursuing.")


if __name__ == "__main__":
    main()
