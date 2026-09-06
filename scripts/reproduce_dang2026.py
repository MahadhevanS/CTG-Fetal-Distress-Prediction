"""
Reproduction of Dang et al. 2026 (CTG-CrossFormer, AUC 0.822 on CTU-UHB).

TRACK A ONLY. This script never touches the frozen project protocol; it
implements the paper's stated procedure and reports the paper's stated metric.
Track B numbers come from src/training/protocol.py and are quoted, never
recomputed here with different rules.

Every configuration writes its out-of-fold predictions to
results/phase8/repro/<tag>.npz so that downstream slices (cohort subsets,
patient-level aggregation, alternative pooling) are computed post-hoc from the
SAME predictions rather than by retraining with a moved goalpost.

PAPER TRAINING STRATEGY IMPLEMENTED (§3.3)
  Focal Loss, gamma = 2.0, inverse-frequency class weights (pos_weight = n_neg/n_pos)
  sqrt-inverse frequency oversampling via WeightedRandomSampler
  AdamW, weight decay 1e-5
  OneCycleLR, 10% linear warmup, max_lr 3e-4, final_lr 1.2e-9
  batch size 32, AMP
  early stopping, patience 15, monitoring validation AUC-ROC
  5-fold StratifiedGroupKFold, groups = patient id
  "AUC computed on pooled predictions across all folds"

DOCUMENTED DEVIATIONS FROM THE PAPER TEXT (unavoidable, recorded not hidden)
  * classification head is 256->128->1 (single logit) in this repo's
    ctg_crossformer.py; the paper states 256->128->2 with dropout (0.3, 0.15).
    Equivalent for AUROC.
  * the paper's 404-patient cohort cannot be derived from its stated >50%
    missing-FHR rule (that rule leaves 547). Runs use the literal rule; the
    404-patient slice is reported post-hoc as a sensitivity.

Usage:
    python scripts/reproduce_dang2026.py --tag paper_literal_outerbest \
        --substrate paper_literal --split sgkf --selection outer_best
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE)

from src.models.ctg_crossformer import (CTGCrossformerEncoder,
                                        CTGCrossformerForClassification)

OUT = "results/phase8/repro"


# ----------------------------------------------------------------- data
def load_substrate(name):
    if name in ("paper_literal", "paper_literal_global", "paper_literal_filled"):
        fn = {"paper_literal": "literal.pt",
              "paper_literal_global": "literal_globalnorm.pt",
              "paper_literal_filled": "literal_filled.pt"}[name]
        d = torch.load(f"data/processed_paper_literal/{fn}", weights_only=False)
        return d["X"].numpy(), d["y"].numpy(), np.asarray(d["pid"])
    if name == "paper_match":
        d = torch.load("data/processed_paper_match/train_dataset.pt", weights_only=False)
        X = d["X"].numpy()
        y = d["y_primary"].numpy()
        pid = np.array([m[0] for m in d["metadata"]])
        return X, y, pid
    if name == "clinical":
        Xs, ys, ps = [], [], []
        for sp in ("train", "val", "test"):
            d = torch.load(f"data/processed_clinical/{sp}_dataset.pt", weights_only=False)
            Xs.append(d["X"].numpy()[:, :2])          # FHR(bc), UC -- drop mask
            ys.append(d["y_primary"].numpy())
            ps.append(np.array([m[0] for m in d["metadata"]]))
            del d
        X = np.concatenate(Xs)
        # This substrate is stored in physical units (bpm, UC). The paper
        # substrates are z-scored per recording. Standardise per channel so the
        # comparison isolates the preprocessing/windowing/fold differences
        # rather than an input-scaling artefact.
        mu = X.mean(axis=(0, 2), keepdims=True)
        sd = X.std(axis=(0, 2), keepdims=True)
        X = ((X - mu) / np.maximum(sd, 1e-6)).astype(np.float32)
        return X, np.concatenate(ys), np.concatenate(ps)
    raise ValueError(name)


def folds_for(split, X, y, pid):
    """Yield (train_idx, test_idx). Both options are patient-grouped."""
    if split == "sgkf":                                # the paper's
        sgkf = StratifiedGroupKFold(n_splits=5)
        yield from sgkf.split(X, y, groups=pid)
    elif split == "protocol":                          # this project's folds
        blob = json.load(open("data/processed_clinical/folds.json"))
        asg = blob["assignment"]
        for k in range(5):
            te_p = {p for p, v in asg.items() if v[0] == k}
            te = np.where(np.isin(pid, list(te_p)))[0]
            tr = np.where(~np.isin(pid, list(te_p)))[0]
            yield tr, te
    else:
        raise ValueError(split)


# ----------------------------------------------------------------- loss
class BinaryFocalLoss(nn.Module):
    """Focal loss, gamma=2, with inverse-frequency positive weighting."""

    def __init__(self, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logit, target):
        logit, target = logit.view(-1), target.view(-1).float()
        p = torch.sigmoid(logit)
        ce = nn.functional.binary_cross_entropy_with_logits(
            logit, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha = self.pos_weight * target + 1.0 * (1 - target)
        return (alpha * (1 - p_t) ** self.gamma * ce).mean()


# ----------------------------------------------------------------- train
def train_fold(Xtr, ytr, Xva, yva, Xte, device, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = CTGCrossformerEncoder(in_channels=2, seq_len=4800, cnn_channels=128,
                                n_heads_cross=4, n_heads_tf=8, n_tf_layers=4,
                                d_ff=512, dropout=0.1, latent_dim=128)
    model = CTGCrossformerForClassification(enc, hidden_dim=128, dropout=0.3).to(device)

    n_pos, n_neg = int(ytr.sum()), int((1 - ytr).sum())
    pos_weight = n_neg / max(n_pos, 1)                       # inverse frequency
    crit = BinaryFocalLoss(gamma=2.0, pos_weight=pos_weight)

    # sqrt-inverse frequency oversampling
    cls_count = np.array([n_neg, n_pos], dtype=np.float64)
    w_cls = 1.0 / np.sqrt(np.maximum(cls_count, 1))
    sample_w = w_cls[ytr]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                    num_samples=len(ytr), replacement=True)

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=32, sampler=sampler, num_workers=0,
                    drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=args.max_epochs * len(dl),
        pct_start=0.10, anneal_strategy="cos",
        final_div_factor=3e-4 / 1.2e-9)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    @torch.no_grad()
    def infer(Xa):
        model.eval()
        out = np.zeros(len(Xa), np.float32)
        for i in range(0, len(Xa), 128):
            xb = torch.from_numpy(Xa[i:i + 128]).to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out[i:i + 128] = model(xb).float().view(-1).cpu().numpy()
        return out

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(args.max_epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        va = infer(Xva)
        auc = roc_auc_score(yva, va) if len(np.unique(yva)) > 1 else 0.5
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return infer(Xte), best_auc, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--substrate", default="paper_literal",
                    choices=["paper_literal", "paper_literal_global",
                             "paper_literal_filled", "paper_match", "clinical"])
    ap.add_argument("--split", default="sgkf", choices=["sgkf", "protocol"])
    ap.add_argument("--selection", default="outer_best",
                    choices=["outer_best", "nested"],
                    help="outer_best = pick the epoch by AUROC on the fold being "
                         "reported, as the paper describes. nested = carve an "
                         "inner patient-level split from the training fold.")
    ap.add_argument("--max_epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, pid = load_substrate(args.substrate)
    print(f"[{args.tag}] substrate={args.substrate} X={X.shape} "
          f"patients={len(set(pid.tolist()))} pos={y.sum()} ({100*y.mean():.1f}%)")
    print(f"          split={args.split} selection={args.selection} device={device}")

    oof = np.full(len(X), np.nan, np.float32)
    fold_id = np.full(len(X), -1, np.int32)
    per_fold = []
    t0 = time.time()
    for k, (tr, te) in enumerate(folds_for(args.split, X, y, pid), 1):
        if args.selection == "nested":
            tr_p = np.array(sorted(set(pid[tr].tolist())))
            plab = np.array([int(y[tr][pid[tr] == p].max()) for p in tr_p])
            strat = plab if len(np.unique(plab)) > 1 and np.bincount(plab).min() >= 2 else None
            fit_p, val_p = train_test_split(tr_p, test_size=0.2, stratify=strat,
                                            random_state=42)
            is_val = np.isin(pid[tr], val_p)
            fit_idx, val_idx = tr[~is_val], tr[is_val]
        else:
            # the paper as described: the reported fold is also the early-stopping
            # monitor. This is selection on the evaluated fold.
            fit_idx, val_idx = tr, te

        p, best, eps = train_fold(X[fit_idx], y[fit_idx], X[val_idx], y[val_idx],
                                  X[te], device, args, args.seed + k)
        oof[te], fold_id[te] = p, k
        fa = roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else float("nan")
        per_fold.append(fa)
        print(f"  fold {k}: test AUROC {fa:.4f}  (best monitor {best:.4f}, "
              f"{eps} epochs, {time.time()-t0:.0f}s elapsed)", flush=True)

    ok = ~np.isnan(oof)
    pooled = roc_auc_score(y[ok], oof[ok])
    print(f"\n[{args.tag}] pooled OOF window AUROC = {pooled:.4f}   "
          f"mean-of-folds = {np.nanmean(per_fold):.4f} +/- {np.nanstd(per_fold):.4f}")

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{args.tag}.npz"), oof=oof, y=y, pid=pid,
             fold=fold_id, per_fold=np.array(per_fold, dtype=float),
             config=json.dumps(vars(args)))
    print(f"          saved {OUT}/{args.tag}.npz")


if __name__ == "__main__":
    main()
