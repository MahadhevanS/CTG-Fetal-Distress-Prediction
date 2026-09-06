"""
KI phase: does teaching the network clinical concepts, or FIGO reasoning over
them, recover the gap between raw-signal detection and rule-defined detection?

    python scripts/figo_run_ki.py --arms KI1 KI2 KI3
    python scripts/figo_eval_detection.py --baseline Z_frozen_smallcnn --gate_on KI3_combined

Writes oof.npz per arm. Computes no headline metric -- that is
scripts/figo_eval_detection.py's job and only its job.

THE ARMS
--------
  KI1  concept supervision      L = L_state + lambda * L_clinical
                                state from a normal classifier head
  KI2  differentiable reasoning state comes ONLY through the soft FIGO rule
  KI3  combined                 both routes, learned 2-input combination

All three share arm Z's convolutional backbone verbatim, so any difference
from Z's 0.7559 is attributable to the heads and the loss.

LAMBDA IS SELECTED ON INNER-VALIDATION PATIENTS, NOT FIXED BY GUESS
-------------------------------------------------------------------
A single hand-picked lambda risks a FALSE NULL: if 0.3 happens to be wrong,
KI fails for a reason that has nothing to do with the hypothesis. So lambda
is chosen per fold from a fixed grid on that fold's INNER VALIDATION split,
which is carved from TRAINING patients and never touches the reported fold.
That is protocol rule 3's sanctioned mechanism, the same one used for early
stopping. The grid is fixed in advance and is not extended after seeing
results.

CONCEPT QUALITY IS RECORDED, ALWAYS
------------------------------------
Every run stores how well the heads actually recovered each concept (R^2 for
continuous, AUROC for binary) out-of-fold. This is the diagnostic that makes
a null interpretable: if KI2 fails AND the concept heads cannot estimate
baseline or variability, the failure is localised to concept estimation
rather than to the reasoning layer -- and since the soft rule reproduces the
label exactly when fed true concepts, that distinction is the whole result.
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
import torch.nn.functional as F
from sklearn.metrics import r2_score, roc_auc_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "scripts"))

from figo_state.ki_models import (CONCEPTS_BIN, CONCEPTS_CONT,  # noqa: E402
                                  KIModel, concept_loss, n_params)
from figo_state.protocol import FigoProtocol                    # noqa: E402
from figo_eval_detection import threshold_for_sensitivity       # noqa: E402

DATA = os.path.join(BASE, "data", "processed_figo")
OUT = os.path.join(BASE, "results", "figo_detection")

ARMS = {
    "KI1_concept":  dict(mode="ki1", use_uc=False,
                         label="KI-1  concept supervision"),
    "KI2_rule":     dict(mode="ki2", use_uc=False,
                         label="KI-2  differentiable FIGO rule"),
    "KI3_combined": dict(mode="ki3", use_uc=False,
                         label="KI-3  concepts + rule + head"),
    # ---- the ONE permitted refinement, and its matched control ------------
    # KI2_UC is the refinement: UC added to the backbone as two extra input
    # channels so n_contractions -- and hence decel_repetitive, the concept
    # that decides most of the label and was recovered at only AUROC 0.755 --
    # becomes estimable. No attention, no extra depth, +960 parameters.
    #
    # KI2_FIX is NOT a competing configuration. It is the control that makes
    # the refinement attributable: it runs the same corrected concept set
    # (has_acute_hypoxia_decel dropped from L_clinical) WITHOUT UC, so that
    # KI2_UC - KI2_FIX isolates the effect of UC from the effect of the
    # concept-set fix. Both changes landed together; without this arm they
    # could not be told apart.
    "KI2_FIX":      dict(mode="ki2", use_uc=False,
                         label="KI-2 control  corrected concept set, no UC"),
    "KI2_UC":       dict(mode="ki2", use_uc=True,
                         label="KI-2 + UC     REFINEMENT"),
}
LAMBDA_GRID = (0.1, 0.3, 1.0)       # fixed in advance, not extended

EPOCHS = 40
BATCH = 48
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10



def oof_name(repeat: int) -> str:
    """Repeat 0 keeps the bare name so every result already measured against
    it stays valid; further repeats are suffixed. Replication must never
    silently overwrite the run it is replicating."""
    return "oof.npz" if repeat == 0 else f"oof_r{repeat}.npz"


def load():
    z = np.load(os.path.join(DATA, "epochs.npz"), allow_pickle=True)
    names = [str(s) for s in z["descriptor_names"]]
    meta = pd.DataFrame({c: z[c] for c in z.files
                         if c not in ("X", "F", "descriptor_names",
                                      "fhr_valid", "uc_valid")})
    Fd = pd.DataFrame(z["F"], columns=names)
    k = (meta.y_state >= 0).values
    X = z["X"][k]
    return (X[:, 0].astype(np.float32), X[:, 1].astype(np.float32),
            z["fhr_valid"][k].astype(bool), z["uc_valid"][k].astype(bool),
            meta[k].reset_index(drop=True), Fd[k].reset_index(drop=True))


def batches(n, bs, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    return [idx[i:i + bs] for i in range(0, n, bs)]


def train_one(mode, lam, tr, va, tensors, targets, device, seed,
              use_uc=False):
    fhr, vf, uc, vu = tensors
    y, t_cont, t_bin, cmean, cstd = targets

    torch.manual_seed(seed)
    model = KIModel(mode=mode, use_uc=use_uc).to(device)
    model.set_scaler(cmean.to(device), cstd.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    pos = float(y[tr].sum())
    w = torch.tensor([(len(tr) - pos) / max(pos, 1.0)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=w)
    bp = ((1 - t_bin[tr]).sum(0) / t_bin[tr].sum(0).clamp(min=1)).clamp(0.1, 10.0)
    bp = bp.to(device)
    yt = torch.tensor(y, dtype=torch.float32)
    rng = np.random.default_rng(seed)

    def fwd(idx):
        return model(fhr[idx].to(device), uc[idx].to(device),
                     vf[idx].to(device), vu[idx].to(device))

    def infer(idx):
        model.eval()
        L, C, Bn = [], [], []
        with torch.no_grad():
            for b in batches(len(idx), 256, False, rng):
                o = fwd(idx[b])
                L.append(o["logit"].cpu())
                C.append(o["cont"].cpu())
                Bn.append(o["bin"].cpu())
        return torch.cat(L), torch.cat(C), torch.cat(Bn)

    best, best_state, bad = np.inf, None, 0
    for _ in range(EPOCHS):
        model.train()
        for b in batches(len(tr), BATCH, True, rng):
            idx = tr[b]
            opt.zero_grad()
            o = fwd(idx)
            loss = (bce(o["logit"], yt[idx].to(device))
                    + lam * concept_loss(o, t_cont[idx].to(device),
                                         t_bin[idx].to(device), bp))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        lv, cv, bv = infer(va)
        # Early stopping on the STATE loss only. The auxiliary loss is a
        # means, not the objective; stopping on the combined loss would let a
        # large lambda choose the checkpoint by concept fit.
        vl = float(bce(lv.to(device), yt[va].to(device)))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, infer, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fhr_np, uc_np, vf_np, vu_np, meta, Fd = load()
    pid = meta.record_id.values.astype(str)
    y = (meta.y_state.values >= 1).astype(int)
    print(f"device {device} | {len(y)} epochs, {len(set(pid))} patients, "
          f"{int(y.sum())} abnormal ({100 * y.mean():.1f}%)")
    print(f"concepts: {len(CONCEPTS_CONT)} continuous + {len(CONCEPTS_BIN)} binary")

    cont_raw = Fd[CONCEPTS_CONT].values.astype(np.float32)
    bin_raw = (Fd[CONCEPTS_BIN].values > 0).astype(np.float32)

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
        oof_cont = np.zeros_like(cont_raw)
        oof_bin = np.zeros_like(bin_raw)
        fold_id = np.full(len(y), -1, int)
        thr, chosen_lam = [], []

        for k, (tr_mask, te_mask) in enumerate(P.folds(repeat=args.repeat), 1):
            fit_mask, val_mask = P.inner_split(tr_mask)
            tr = np.where(fit_mask)[0]
            va = np.where(val_mask)[0]
            te = np.where(te_mask)[0]

            fm = float(fhr_np[fit_mask][vf_np[fit_mask]].mean())
            fs = float(max(fhr_np[fit_mask][vf_np[fit_mask]].std(), 1e-6))
            um = float(uc_np[fit_mask][vu_np[fit_mask]].mean())
            us = float(max(uc_np[fit_mask][vu_np[fit_mask]].std(), 1e-6))
            fhr = torch.tensor((fhr_np - fm) / fs)
            uc = torch.tensor((uc_np - um) / us)
            vft = torch.tensor(vf_np)
            vut = torch.tensor(vu_np)
            fhr[~vft] = 0.0
            uc[~vut] = 0.0

            cmean = torch.tensor(cont_raw[fit_mask].mean(0))
            cstd = torch.tensor(cont_raw[fit_mask].std(0)).clamp(min=1e-6)
            t_cont = torch.tensor((cont_raw - cmean.numpy()) / cstd.numpy())
            t_bin = torch.tensor(bin_raw)

            tensors = (fhr, vft, uc, vut)
            targets = (y, t_cont, t_bin, cmean, cstd)

            # lambda chosen on INNER-VALIDATION patients only
            best_lam, best_vl, best_pack = None, np.inf, None
            for lam in LAMBDA_GRID:
                model, infer, vl = train_one(cfg["mode"], lam, tr, va, tensors,
                                             targets, device,
                                             seed=args.seed + 100 * k,
                                             use_uc=cfg["use_uc"])
                if vl < best_vl:
                    best_lam, best_vl, best_pack = lam, vl, (model, infer)
            model, infer = best_pack
            chosen_lam.append(best_lam)

            lv, _, _ = infer(va)
            lt, ct, bt = infer(te)
            oof[te] = torch.sigmoid(lt).numpy()
            oof_cont[te] = (ct.numpy() * cstd.numpy() + cmean.numpy())
            oof_bin[te] = torch.sigmoid(bt).numpy()
            fold_id[te] = k
            thr.append(threshold_for_sensitivity(y[va],
                                                 torch.sigmoid(lv).numpy()))
            print(f"    fold {k}/5  lambda {best_lam}  thr {thr[-1]:.4f}  "
                  f"({time.time() - t0:.0f}s)")

        assert (fold_id >= 0).all()

        # ---- concept recovery, out of fold -------------------------------
        quality = {}
        for i, c in enumerate(CONCEPTS_CONT):
            quality[c] = dict(kind="continuous",
                              r2=float(r2_score(cont_raw[:, i], oof_cont[:, i])),
                              corr=float(np.corrcoef(cont_raw[:, i],
                                                     oof_cont[:, i])[0, 1]))
        for i, c in enumerate(CONCEPTS_BIN):
            t = bin_raw[:, i]
            quality[c] = dict(kind="binary",
                              auroc=(float(roc_auc_score(t, oof_bin[:, i]))
                                     if len(np.unique(t)) > 1 else float("nan")),
                              prevalence=float(t.mean()))

        model_p = n_params(KIModel(mode=cfg["mode"], use_uc=cfg["use_uc"]))
        np.savez(os.path.join(d, oof_name(args.repeat)),
                 prob=oof, y=y, pid=pid, fold=fold_id, name=arm,
                 n_params=model_p, threshold=float(np.mean(thr)),
                 fold_thresholds=np.array(thr),
                 lambdas=np.array(chosen_lam),
                 concept_pred_cont=oof_cont, concept_true_cont=cont_raw,
                 concept_pred_bin=oof_bin, concept_true_bin=bin_raw,
                 concept_names_cont=np.array(CONCEPTS_CONT),
                 concept_names_bin=np.array(CONCEPTS_BIN))
        json.dump(dict(arm=arm, label=cfg["label"], mode=cfg["mode"],
                       use_uc=cfg["use_uc"], lambda_grid=list(LAMBDA_GRID),
                       lambda_per_fold=chosen_lam, n_params=model_p,
                       epochs=EPOCHS, lr=LR, seed=args.seed,
                       concept_quality=quality,
                       runtime_s=round(time.time() - t0, 1)),
                  open(os.path.join(d, f"config{'' if args.repeat==0 else '_r%d'%args.repeat}.json"), "w"), indent=2)

        print(f"\n  concept recovery (out-of-fold):")
        for c in CONCEPTS_CONT:
            print(f"    {c:22s} R2 {quality[c]['r2']:+.3f}  "
                  f"r {quality[c]['corr']:+.3f}")
        for c in CONCEPTS_BIN:
            print(f"    {c:22s} AUROC {quality[c]['auroc']:.3f}  "
                  f"(prev {100 * quality[c]['prevalence']:.1f}%)")
        print(f"  wrote {d}/{oof_name(args.repeat)}  ({time.time() - t0:.0f}s)")

    print("\nScore with:  python scripts/figo_eval_detection.py "
          "--baseline Z_frozen_smallcnn --gate_on KI3_combined")


if __name__ == "__main__":
    main()
