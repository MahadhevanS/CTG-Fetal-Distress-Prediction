"""
Milestone 3: auxiliary physiological supervision, one target at a time.

Runs a controlled ladder through the FROZEN protocol (src/training/protocol.py),
so every rung is comparable to every other rung and to the Milestone-1 baselines.

    E1  binary only                          <- the clean deep baseline
    E2  + pH
    E3  + pH + BDecf
    E4  + pH + BDecf + Apgar5
    E5  + descriptors (19-d regression)      <- best historical evidence (+0.048)
    E6  binary + descriptors ONLY            <- isolates the descriptor head

Only the set of auxiliary heads changes between rungs. The encoder, sampler,
loss weight, schedule, seed and folds are identical throughout, so a difference
between rungs is attributable to the added supervision rather than to tuning.

REFERENCE NUMBERS ON THIS PROTOCOL (do not re-derive):
    A   19 feats -> LR -> max-agg        0.7268  [0.670-0.779]
    C   patient feats -> MLP             0.6977
    F2  best fusion (mslstm branch)      0.7081
    B   deep, 8 encoders swept           0.6167 - 0.7178

WHAT WOULD COUNT AS SUCCESS
    0.75  meaningful   0.78  very good   0.80+  striking
Anything inside [0.670-0.779] is not distinguishable from the LR baseline at
110 positive patients -- report the CI, not the point estimate.

HONESTY NOTE: y_primary is defined as pH <= 7.15, so the pH head supplies the
same label at higher resolution rather than new physiology. See the docstring of
src/models/multitask_physio.py before writing this up.

Usage:
    python scripts/run_multitask_ladder.py --encoder mslstm --epochs 20
    python scripts/run_multitask_ladder.py --rungs E1,E2 --epochs 10
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

import torch                                                        # noqa: E402
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler  # noqa: E402

from src.models.encoder_registry import build_encoder               # noqa: E402
from src.models.multitask_physio import (MultiTaskPhysioNet,        # noqa: E402
                                         multitask_loss)
from src.training.protocol import Protocol                          # noqa: E402

REPEAT = 0

RUNGS = {
    "E1": dict(use_ph=False, use_bdecf=False, use_apgar=False, use_descriptors=False),
    "E2": dict(use_ph=True,  use_bdecf=False, use_apgar=False, use_descriptors=False),
    "E3": dict(use_ph=True,  use_bdecf=True,  use_apgar=False, use_descriptors=False),
    "E4": dict(use_ph=True,  use_bdecf=True,  use_apgar=True,  use_descriptors=False),
    "E5": dict(use_ph=True,  use_bdecf=True,  use_apgar=True,  use_descriptors=True),
    "E6": dict(use_ph=False, use_bdecf=False, use_apgar=False, use_descriptors=True),
}
LABELS = {
    "E1": "E1  binary only",
    "E2": "E2  + pH",
    "E3": "E3  + pH + BDecf",
    "E4": "E4  + pH + BDecf + Apgar",
    "E5": "E5  + descriptors (full)",
    "E6": "E6  binary + descriptors only",
}


def load(data_dir, in_channels):
    X, D, y, pid, ph, bd, ap = [], [], [], [], [], [], []
    for s in ("train", "val", "test"):
        d = torch.load(os.path.join(data_dir, f"{s}_dataset.pt"), weights_only=False)
        X.append(d["X"].numpy()[:, :in_channels, :])
        D.append(np.hstack([d["y_features"].numpy(),
                            np.load(os.path.join(data_dir, f"{s}_extended_features.npy"))]))
        y.append(d["y_primary"].numpy())
        pid.append(np.array([m[0] for m in d["metadata"]]))
        ph.append(d["y_ph"].numpy())
        bd.append(d["y_bdecf"].numpy())
        ap.append(d["y_apgar5"].numpy())
    return (np.vstack(X), np.vstack(D), np.concatenate(y), np.concatenate(pid),
            np.concatenate(ph), np.concatenate(bd), np.concatenate(ap))


def zfit(v, mask):
    """Standardise a regression target using TRAINING rows only."""
    m = np.nanmean(v[mask])
    s = np.nanstd(v[mask])
    return m, (s if s > 1e-6 else 1.0)


def run_rung(P, X, Dsc, y, ph, bd, ap, rung, args, device):
    tag = f"mtl_{args.encoder}_{rung}_e{args.epochs}_lam{args.lam}_r{REPEAT}"
    cached = Protocol.load_oof(tag)
    if cached is not None and len(cached) == len(y) and not args.force:
        print(f"  [cache] {tag}")
        return cached

    cfg = {"in_channels": X.shape[1]}
    oof = np.zeros(len(y))
    for k, (tr, te_p) in enumerate(P.folds(repeat=REPEAT), 1):
        te = ~tr
        torch.manual_seed(args.seed)
        model = MultiTaskPhysioNet(build_encoder(args.encoder, cfg),
                                   n_descriptors=Dsc.shape[1], **RUNGS[rung]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

        # Standardise every regression target on the TRAINING rows of this fold.
        ph_m, ph_s = zfit(ph, tr)
        bd_m, bd_s = zfit(bd, tr)
        ap_m, ap_s = zfit(ap, tr)
        d_m, d_s = Dsc[tr].mean(0), np.clip(Dsc[tr].std(0), 1e-6, None)

        ds = TensorDataset(
            torch.tensor(X[tr], dtype=torch.float32),
            torch.tensor(y[tr], dtype=torch.float32),
            torch.tensor((ph[tr] - ph_m) / ph_s, dtype=torch.float32),
            torch.tensor((bd[tr] - bd_m) / bd_s, dtype=torch.float32),
            torch.tensor((ap[tr] - ap_m) / ap_s, dtype=torch.float32),
            torch.tensor((Dsc[tr] - d_m) / d_s, dtype=torch.float32))

        cnt = np.bincount(y[tr].astype(int))
        w = (1.0 / np.sqrt(np.maximum(cnt, 1)))[y[tr].astype(int)]
        dl = DataLoader(ds, batch_size=32,
                        sampler=WeightedRandomSampler(
                            torch.as_tensor(w, dtype=torch.double), len(w), True))
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-4, total_steps=max(len(dl) * args.epochs, 10), pct_start=0.1)

        model.train()
        for _ in range(args.epochs):
            for xb, yb, pb, bb, ab, db in dl:
                batch = {"y": yb.to(device), "ph": pb.to(device), "bdecf": bb.to(device),
                         "apgar": ab.to(device), "descriptors": db.to(device)}
                opt.zero_grad()
                multitask_loss(model(xb.to(device)), batch, lam=args.lam)["total"].backward()
                opt.step()
                sch.step()

        model.eval()
        Xte = X[te]
        probs = []
        with torch.no_grad():
            for i in range(0, len(Xte), 256):
                xb = torch.tensor(Xte[i:i + 256], dtype=torch.float32).to(device)
                probs.extend(torch.sigmoid(model(xb)["distress"]).cpu().numpy())
        oof[te] = probs
        print(f"    fold {k}/{P.n_folds}")
    Protocol.save_oof(tag, oof)
    return oof


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--data_dir", default="data/processed_clinical/")
    ap_.add_argument("--encoder", default="mslstm",
                     help="mslstm (default) was the best patient-level encoder in the "
                          "protocol sweep at 0.7178, and is ~2x faster than patchtst "
                          "which tied it at 0.7175")
    ap_.add_argument("--epochs", type=int, default=20)
    ap_.add_argument("--lam", type=float, default=0.3,
                     help="shared weight on every auxiliary term; held constant across "
                          "rungs so a rung cannot win on tuning")
    ap_.add_argument("--in_channels", type=int, default=2, choices=[2, 3])
    ap_.add_argument("--rungs", default=",".join(RUNGS))
    ap_.add_argument("--seed", type=int, default=42)
    ap_.add_argument("--force", action="store_true")
    ap_.add_argument("--out", default="results/multitask_ladder.json")
    args = ap_.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Dsc, y, pid, ph, bd, apg = load(args.data_dir, args.in_channels)
    P = Protocol.load_or_create(pid, y)
    print(f"\n{len(y)} windows | {len(P.patients)} patients | {int(P.plab.sum())} positive")
    print(f"encoder {args.encoder} | {args.epochs} epochs | lambda {args.lam} | "
          f"device {device}")
    print(f"BDecf missing for {int(np.isnan(bd).sum())} windows (masked in the loss)\n")

    rows = []
    print("=" * 78)
    print("LADDER -- patient-level AUROC, max-aggregated, CI over patients")
    print("=" * 78)
    base = None
    for rung in [r.strip() for r in args.rungs.split(",") if r.strip()]:
        if rung not in RUNGS:
            sys.exit(f"[ABORT] unknown rung {rung}; choose from {list(RUNGS)}")
        print(f"  {LABELS[rung]}")
        oof = run_rung(P, X, Dsc, y, ph, bd, apg, rung, args, device)
        r = P.report(LABELS[rung], oof)
        r["rung"] = rung
        if rung == "E1":
            base = r["auroc"]
        if base is not None:
            r["delta_vs_E1"] = r["auroc"] - base
            if rung != "E1":
                print(f"  {'':44s} delta vs E1: {r['delta_vs_E1']:+.4f}")
        rows.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nWrote {args.out}")
    print("\nReference on this protocol: A (19 feats + LR) = 0.7268 [0.670-0.779].")
    print("A rung inside that interval is not distinguishable from the clinical baseline.")


if __name__ == "__main__":
    main()
