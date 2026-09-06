"""
The detection ablation matrix. ONE narrow experiment, pre-registered.

    python scripts/figo_run_detection.py --arms A B C D E
    python scripts/figo_eval_detection.py          # then score them

THE MATRIX
----------
| arm |          FHR |          UC | cross-attention | asks                     |
|-----|--------------|-------------|-----------------|--------------------------|
|  A  |            Y |           N |               N | control                  |
|  B  |            Y |           Y |               N | does UC help at all      |
|  C  |            Y |           Y |               Y | does INTERACTION help    |
|  D  |            Y | shuffled    |               Y | is the pairing real      |
|  E  |     shuffled |           Y |               Y | reverse control          |

  A vs B   does the UC channel carry usable information under plain fusion
  B vs C   does an explicit interaction mechanism beat simple fusion
  C vs D   does the model need the CORRECT FHR-UC temporal relationship
  C vs E   the same control from the other side

D is the decisive one. If C beats B but D matches C, the cross-attention
gained nothing from physiological coupling -- it found extra capacity or an
epoch-level UC statistic, and the hypothesis is not supported however good
the number looks.

SHUFFLING IS DONE ACROSS EPOCHS, WITHIN A FOLD
----------------------------------------------
Arm D pairs each epoch's FHR with ANOTHER epoch's UC, drawn from the same
fold. Marginal distributions of both channels are preserved exactly; only the
correspondence between them is destroyed. The permutation is applied to train
AND test alike, because the question is whether the model can use a
relationship that is not there -- not whether it survives a distribution
shift at test time.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not compute a headline metric. It writes oof.npz and stops.
scripts/figo_eval_detection.py is the only thing that turns predictions into
numbers, so that two arms can never be scored by two slightly different
recipes -- the failure src/training/protocol.py was written to end.

THE THRESHOLD IS CHOSEN INSIDE THE TRAINING FOLD
------------------------------------------------
Per fold, the lowest threshold reaching 90% sensitivity on that fold's INNER
VALIDATION patients (never on its test patients). Folds are averaged into one
threshold stored in oof.npz. A threshold picked on the predictions it is then
scored against is a fitted parameter, and the sensitivity and specificity it
produces are training numbers.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))

from figo_state.models import build, n_params                    # noqa: E402
from figo_state.legacy_cnn import SmallCNN                       # noqa: E402
from figo_state.protocol import FigoProtocol                     # noqa: E402

sys.path.insert(0, os.path.join(BASE, "scripts"))
from figo_eval_detection import threshold_for_sensitivity        # noqa: E402

DATA = os.path.join(BASE, "data", "processed_figo")
OUT = os.path.join(BASE, "results", "figo_detection")

ARMS: Dict[str, Dict] = {
    "A_fhr":          dict(arch="fhr",    shuffle=None,
                           label="A  FHR only"),
    "B_concat":       dict(arch="concat", shuffle=None,
                           label="B  FHR+UC, input concatenation"),
    "C_cross":        dict(arch="cross",  shuffle=None,
                           label="C  FHR+UC, cross-attention"),
    "D_cross_ucshuf": dict(arch="cross",  shuffle="uc",
                           label="D  C with UC shuffled across epochs"),
    "E_cross_fhrshuf": dict(arch="cross", shuffle="fhr",
                            label="E  C with FHR shuffled across epochs"),
    # The Gate-2 SmallCNN, carried forward unchanged so the frozen 0.7518
    # control appears in the same table under the same scoring. Arm A is NOT
    # that model -- it shares the encoder family of B and C so that the A/B/C
    # comparison is clean, which makes it a different network.
    "Z_frozen_smallcnn": dict(arch="smallcnn", shuffle=None,
                              label="Z  frozen Gate-2 SmallCNN (FHR+mask)"),
}

EPOCHS = 40
BATCH = 48
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10


# --------------------------------------------------------------------------

def oof_name(repeat: int) -> str:
    """Repeat 0 keeps the bare name so every result already measured against
    it stays valid; further repeats are suffixed. Replication must never
    silently overwrite the run it is replicating."""
    return "oof.npz" if repeat == 0 else f"oof_r{repeat}.npz"


def load() -> Tuple[np.ndarray, ...]:
    z = np.load(os.path.join(DATA, "epochs.npz"), allow_pickle=True)
    meta = pd.DataFrame({c: z[c] for c in z.files
                         if c not in ("X", "F", "descriptor_names",
                                      "fhr_valid", "uc_valid")})
    keep = (meta.y_state >= 0).values
    X = z["X"][keep]
    return (X[:, 0].astype(np.float32),                 # fhr, bpm
            X[:, 1].astype(np.float32),                 # uc
            z["fhr_valid"][keep].astype(bool),
            z["uc_valid"][keep].astype(bool),
            meta[keep].reset_index(drop=True))


def zfit(a: np.ndarray, valid: np.ndarray, mask: np.ndarray):
    """Mean/std over TRAINING rows and VALID samples only."""
    sel = a[mask][valid[mask]]
    return float(sel.mean()), float(max(sel.std(), 1e-6))


def make_batches(n: int, bs: int, shuffle: bool, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    return [idx[i:i + bs] for i in range(0, n, bs)]


def run_fold(model, tr, va, te, tensors, y, device, seed):
    fhr, uc, vf, vu = tensors
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    pos = float(y[tr].sum())
    w = torch.tensor([(len(y[tr]) - pos) / max(pos, 1.0)], device=device)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=w)
    rng = np.random.default_rng(seed)

    def forward(idx):
        return model(fhr[idx].to(device), uc[idx].to(device),
                     vf[idx].to(device), vu[idx].to(device))

    def evaluate(idx):
        model.eval()
        outs = []
        with torch.no_grad():
            for b in make_batches(len(idx), 256, False, rng):
                outs.append(forward(idx[b]).squeeze(1).cpu())
        return torch.cat(outs)

    yt = torch.tensor(y, dtype=torch.float32)
    best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS):
        model.train()
        for b in make_batches(len(tr), BATCH, True, rng):
            idx = tr[b]
            opt.zero_grad()
            loss = lossfn(forward(idx).squeeze(1), yt[idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        vl = float(lossfn(evaluate(va).to(device), yt[va].to(device)))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    p_va = torch.sigmoid(evaluate(va)).numpy()
    p_te = torch.sigmoid(evaluate(te)).numpy()
    return p_va, p_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fhr_np, uc_np, vf_np, vu_np, meta = load()
    pid = meta.record_id.values.astype(str)
    y = (meta.y_state.values >= 1).astype(int)
    print(f"device {device} | {len(y)} epochs, {len(set(pid))} patients, "
          f"{int(y.sum())} abnormal ({100 * y.mean():.1f}%)")

    worst = meta.groupby("record_id").y_state.max().to_dict()
    P = FigoProtocol.load_or_create(pid, {str(k): int(v) for k, v in worst.items()},
                                    path=os.path.join(DATA, "folds.json"))

    for arm in args.arms:
        cfg = ARMS[arm]
        d = os.path.join(OUT, arm)
        os.makedirs(d, exist_ok=True)
        print(f"\n{'=' * 74}\n{cfg['label']}\n{'=' * 74}")
        t0 = time.time()

        oof = np.zeros(len(y))
        fold_id = np.full(len(y), -1, int)
        thresholds = []

        for k, (tr_mask, te_mask) in enumerate(P.folds(repeat=args.repeat), 1):
            fit_mask, val_mask = P.inner_split(tr_mask)
            tr = np.where(fit_mask)[0]
            va = np.where(val_mask)[0]
            te = np.where(te_mask)[0]

            # z-score fitted on the FITTING rows and valid samples only, so
            # neither the inner validation nor the test patients touch it
            fm, fs = zfit(fhr_np, vf_np, fit_mask)
            um, us = zfit(uc_np, vu_np, fit_mask)
            fhr = torch.tensor((fhr_np - fm) / fs)
            uc = torch.tensor((uc_np - um) / us)
            # invalid samples carry no information; zero them so the encoder
            # sees a constant rather than an interpolation artefact
            fhr[~torch.tensor(vf_np)] = 0.0
            uc[~torch.tensor(vu_np)] = 0.0

            if cfg["shuffle"] is not None:
                # Break the FHR-UC correspondence while preserving both
                # marginals exactly. Permuted within fold, applied to train
                # and test alike -- see the module docstring.
                rng = np.random.default_rng(args.seed + 1000 * k)
                perm = rng.permutation(len(y))
                if cfg["shuffle"] == "uc":
                    uc, vu = uc[perm], vu_np[perm]
                    vf = vf_np
                else:
                    fhr, vf = fhr[perm], vf_np[perm]
                    vu = vu_np
                vf_t = torch.tensor(vf)
                vu_t = torch.tensor(vu)
            else:
                vf_t = torch.tensor(vf_np)
                vu_t = torch.tensor(vu_np)

            model = (SmallCNN() if cfg["arch"] == "smallcnn"
                     else build(cfg["arch"])).to(device)
            p_va, p_te = run_fold(model, tr, va, te,
                                  (fhr, uc, vf_t, vu_t), y, device,
                                  seed=args.seed + k)
            oof[te] = p_te
            fold_id[te] = k
            thresholds.append(threshold_for_sensitivity(y[va], p_va))
            print(f"    fold {k}/5  thr {thresholds[-1]:.4f}  "
                  f"({time.time() - t0:.0f}s)")

        assert (fold_id >= 0).all(), "some epochs never appeared in a test fold"
        np.savez(os.path.join(d, oof_name(args.repeat)),
                 prob=oof, y=y, pid=pid, fold=fold_id,
                 name=arm,
                 n_params=n_params(SmallCNN() if cfg["arch"] == "smallcnn"
                                   else build(cfg["arch"])),
                 threshold=float(np.mean(thresholds)),
                 fold_thresholds=np.array(thresholds))
        json.dump(dict(arm=arm, label=cfg["label"], arch=cfg["arch"],
                       shuffle=cfg["shuffle"], repeat=args.repeat,
                       seed=args.seed, epochs=EPOCHS, batch=BATCH, lr=LR,
                       n_params=n_params(SmallCNN() if cfg["arch"] == "smallcnn"
                                         else build(cfg["arch"])),
                       runtime_s=round(time.time() - t0, 1)),
                  open(os.path.join(d, f"config{'' if args.repeat==0 else '_r%d'%args.repeat}.json"), "w"), indent=2)
        print(f"  wrote {d}/{oof_name(args.repeat)}  ({time.time() - t0:.0f}s)")

    print("\nNow score them:  python scripts/figo_eval_detection.py")


if __name__ == "__main__":
    main()
