"""
Milestone 1 + 2: knowledge baselines and fusion, on ONE frozen fold partition.

Every model here is trained and scored through src/training/protocol.py, so all
numbers are comparable by construction -- which was NOT true of any previous
comparison in this project (each script built its own folds, and per-fold AUROC
swings 0.72-0.91, far larger than the effects being compared).

Protocol: patient-grouped folds, repeat 0 (5 folds, every patient tested once),
patient-level max-aggregation, 95% CI bootstrapped over patients.

MODELS
------
A    19 window features -> LR        -> max-aggregate   (the clinical baseline)
A'   19 window features -> MLP       -> max-aggregate
C    patient-aggregated features -> MLP                 (native patient-level)
C'   patient-aggregated features -> LR                  (native patient-level)
B    raw CTG -> encoder -> window probs -> max-aggregate (--encoder, cached)
F1   late fusion: rank-average of A and B
F2   stacked: LR on [B patient score | aggregated features]

Plus leave-one-group-out ablations on A, which answer WHICH clinical knowledge
carries the signal -- the Step-5 question -- for free, since A costs seconds.

ALREADY KNOWN BEFORE RUNNING THIS (do not re-derive):
  * A  = 0.7290 +/- 0.045   (scripts/audit_evaluation_protocol.py, aggregation)
  * B  = 0.6167 (crossformer) .. 0.7178 (mslstm), all 8 encoders swept
  * F1 = 0.6816, F2 = 0.6930 with a cnn1d branch -- BOTH BELOW A
  * Historical KI: plus_figo +0.0000, plus_features +0.031/+0.048, CRP +0.0199
This script's job is to put them on identical folds with intervals, and to add
the two genuinely new rows (C, C') plus the ablations.

Usage:
    python scripts/run_knowledge_baselines.py
    python scripts/run_knowledge_baselines.py --encoder mslstm --epochs 20
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.neural_network import MLPClassifier             # noqa: E402
from sklearn.pipeline import make_pipeline                   # noqa: E402
from sklearn.preprocessing import StandardScaler             # noqa: E402

from src.training.protocol import (FEATURE_GROUPS, Protocol,  # noqa: E402
                                   load_clinical)

REPEAT = 0   # one pass = every patient tested exactly once; deep models cannot
             # afford 25 fold-trainings, so all models share this single pass.


def mk_lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def mk_mlp(seed=0):
    # Deliberately small. 547 patients / 110 positive does not support depth;
    # this project has measured that lesson repeatedly (Model 9, KG-MIL).
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0,
                                       max_iter=2000, random_state=seed,
                                       early_stopping=True, n_iter_no_change=20))


def oof_window(P, Fe, y, mk, cols=None):
    """Out-of-fold window probabilities from a window-level model."""
    Fu = Fe if cols is None else Fe[:, cols]
    out = np.zeros(len(y))
    for tr, te_p in P.folds(repeat=REPEAT):
        te = ~tr
        out[te] = mk().fit(Fu[tr], y[tr]).predict_proba(Fu[te])[:, 1]
    return out


def patient_design(P, Fe):
    """One row per patient: max and mean of each feature over their windows."""
    return np.array([np.concatenate([Fe[P.pidx[p]].max(0), Fe[P.pidx[p]].mean(0)])
                     for p in P.patients])


def oof_patient(P, A, mk):
    """Out-of-fold scores from a model trained directly on patient rows."""
    out = np.zeros(len(P.patients))
    pos = {p: i for i, p in enumerate(P.patients)}
    for tr, te_p in P.folds(repeat=REPEAT):
        tr_p = np.array([p for p in P.patients if p not in set(te_p.tolist())])
        itr = [pos[p] for p in tr_p]
        ite = [pos[p] for p in te_p]
        out[ite] = mk().fit(A[itr], P.plab[itr]).predict_proba(A[ite])[:, 1]
    return out


def deep_oof(P, X, y, encoder, epochs, device_str=None):
    """Train the encoder once per fold, cache out-of-fold window probabilities."""
    tag = f"{encoder}_e{epochs}_r{REPEAT}"
    cached = Protocol.load_oof(tag)
    if cached is not None and len(cached) == len(y):
        print(f"  [cache] reusing OOF for {tag}")
        return cached

    import torch
    import torch.nn.functional as Fn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from src.models.encoder_registry import build_classifier, build_encoder

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = np.zeros(len(y))
    for k, (tr, te_p) in enumerate(P.folds(repeat=REPEAT), 1):
        te = ~tr
        torch.manual_seed(42)
        cfg = {"in_channels": X.shape[1]}
        model = build_classifier(encoder, build_encoder(encoder, cfg), cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        ds = TensorDataset(torch.tensor(X[tr], dtype=torch.float32),
                           torch.tensor(y[tr], dtype=torch.float32))
        cnt = np.bincount(y[tr].astype(int))
        w = (1.0 / np.sqrt(np.maximum(cnt, 1)))[y[tr].astype(int)]
        dl = DataLoader(ds, batch_size=32,
                        sampler=WeightedRandomSampler(
                            torch.as_tensor(w, dtype=torch.double), len(w), True))
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-4, total_steps=max(len(dl) * epochs, 10), pct_start=0.1)
        model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                Fn.binary_cross_entropy_with_logits(
                    model(xb.to(device)).squeeze(-1), yb.to(device)).backward()
                opt.step()
                sch.step()
        model.eval()
        Xte = X[te]
        probs = []
        with torch.no_grad():
            for i in range(0, len(Xte), 256):
                xb = torch.tensor(Xte[i:i + 256], dtype=torch.float32).to(device)
                probs.extend(torch.sigmoid(model(xb).squeeze(-1)).cpu().numpy())
        out[te] = probs
        print(f"  fold {k}/{P.n_folds} trained")
    Protocol.save_oof(tag, out)
    return out


def rank(v):
    return np.argsort(np.argsort(v)) / max(len(v) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_clinical/")
    ap.add_argument("--encoder", default=None,
                    help="add baseline B and the fusion rows using this encoder")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--in_channels", type=int, default=2, choices=[2, 3])
    ap.add_argument("--out", default="results/knowledge_baselines.json")
    args = ap.parse_args()

    X, Fe, y, pid = load_clinical(args.data_dir)
    X = X[:, :args.in_channels, :]
    P = Protocol.load_or_create(pid, y)
    print(f"\n{len(y)} windows | {len(P.patients)} patients | {int(P.plab.sum())} positive")
    print(f"protocol: patient-grouped, repeat {REPEAT}, {P.n_folds} folds, "
          f"patient-level max-aggregation, CI bootstrapped over patients\n")

    rows = []
    print("=" * 78)
    print("BASELINES")
    print("=" * 78)
    oof_A = oof_window(P, Fe, y, mk_lr)
    rows.append(P.report("A   19 feats -> LR -> max-agg", oof_A))
    rows.append(P.report("A'  19 feats -> MLP -> max-agg",
                         oof_window(P, Fe, y, mk_mlp)))

    Apat = patient_design(P, Fe)
    rows.append(P.report_patient_scores("C   patient feats -> MLP",
                                        P.plab, oof_patient(P, Apat, mk_mlp)))
    rows.append(P.report_patient_scores("C'  patient feats -> LR",
                                        P.plab, oof_patient(P, Apat, mk_lr)))

    if args.encoder:
        print(f"\n  training {args.encoder} ({args.epochs} epochs x {P.n_folds} folds)...")
        oof_B = deep_oof(P, X, y, args.encoder, args.epochs)
        rows.append(P.report(f"B   {args.encoder} -> max-agg", oof_B))

        _, sA = P.to_patient(oof_A)
        _, sB = P.to_patient(oof_B)
        print("\n" + "=" * 78)
        print("FUSION")
        print("=" * 78)
        rows.append(P.report_patient_scores("F1  late fusion (rank average)",
                                            P.plab, 0.5 * rank(sA) + 0.5 * rank(sB)))
        Z = np.column_stack([sB, Apat])
        rows.append(P.report_patient_scores("F2  stacked (B score + feats)",
                                            P.plab, oof_patient(P, Z, mk_lr)))

    print("\n" + "=" * 78)
    print("LEAVE-ONE-GROUP-OUT ABLATION on baseline A -- which knowledge matters?")
    print("=" * 78)
    base = rows[0]["auroc"]
    for grp, idx in FEATURE_GROUPS.items():
        cols = [i for i in range(Fe.shape[1]) if i not in idx]
        r = P.report(f"A minus {grp:15s} ({len(idx)} feats)",
                     oof_window(P, Fe, y, mk_lr, cols=cols))
        r["delta_vs_A"] = r["auroc"] - base
        rows.append(r)
        print(f"  {'':44s} delta vs A: {r['delta_vs_A']:+.4f}")

    print()
    print("=" * 78)
    print("GROUP-ONLY -- is a group UNINFORMATIVE, or merely REDUNDANT?")
    print("=" * 78)
    print("  Leave-one-out alone cannot tell those apart: a group can carry real")
    print("  signal and still cost nothing to remove, if other features cover it.")
    for grp, idx in FEATURE_GROUPS.items():
        r = P.report(f"A using ONLY {grp:12s} ({len(idx)} feats)",
                     oof_window(P, Fe, y, mk_lr, cols=idx))
        r["group_only"] = True
        rows.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nWrote {args.out}")
    print("\nIntervals overlap heavily at this sample size (110 positive patients). "
          "Read differences against the CIs, not the point estimates.")


if __name__ == "__main__":
    main()
