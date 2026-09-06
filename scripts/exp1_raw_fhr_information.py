"""
PHASE 10 / EXPERIMENT 1 -- Raw FHR information test.

QUESTION
    Does minimally processed raw FHR contain predictive information that the 19
    clinical descriptors do not capture?

DESIGN
    The AGGREGATION IS HELD FIXED at the clinical baseline's: 20-minute windows
    -> per-window probability -> patient-level max. Only the REPRESENTATION
    changes (raw waveform instead of 19 descriptors). Changing aggregation is
    Experiment 2; doing both at once would confound the representation test.

    E1-A  FHR at 1 Hz
    E1-B  FHR at 4 Hz
    (E1-C multi-resolution is NOT run here -- gated on A or B being promising.)

MODEL
    One small residual 1D CNN, ~200k parameters. No architecture sweep. The
    question is whether the representation carries information, not which
    network is best.

PROTOCOL (frozen, Phase 10 section 5)
    * the existing five patient-grouped folds from folds.json
    * patient is the statistical unit; primary metric patient-level AUROC
    * epoch selection on an inner split carved from TRAINING patients only
    * normalisation statistics fit on training-fold windows only
    * patient bootstrap CI; no test-fold tuning of anything

DECISION GATE (pre-registered, section 11)
    < 0.72      KILL
    0.72-0.75   do not tune; likely KILL
    0.75-0.80   PROMOTE to patient MIL
    0.80-0.85   PRIMARY CANDIDATE
    >= 0.85     FREEZE + AUDIT

Run from the repo root:
    python scripts/exp1_raw_fhr_information.py
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.preprocessing.pipeline_raw_minimal import (PREPROCESSING_AUDIT,
                                                    build_patient_windows)
from src.training.protocol import Protocol

RAW = "data/raw/ctu-chb-intrapartum"
PROC = "data/processed_clinical"
OUT = "results/phase10"
SEED = 42
LR_BASELINE = 0.7271


# ------------------------------------------------------------------- model
class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential() if (cin == cout and stride == 1)
                     else nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride,
                                                  bias=False),
                                        nn.BatchNorm1d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.act(h + self.skip(x))


class SmallCNN(nn.Module):
    """~200k parameters. Deliberately not tuned."""

    def __init__(self, in_ch=2, width=32, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.blocks = nn.Sequential(
            ResBlock(width, width * 2, stride=2),
            ResBlock(width * 2, width * 4, stride=2),
            ResBlock(width * 4, width * 4, stride=2))
        self.head = nn.Sequential(
            nn.Linear(width * 8, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        h = self.blocks(self.stem(x))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(h)


# ------------------------------------------------------------------- data
def build(fs_out, tag):
    cache = os.path.join(OUT, f"raw_{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"[cache] {cache}  X={z['X'].shape}")
        return z["X"], z["pid"]
    lab = {}
    for sp in ("train", "val", "test"):
        d = torch.load(os.path.join(PROC, f"{sp}_dataset.pt"), weights_only=False)
        ids = np.array([m[0] for m in d["metadata"]])
        y = d["y_primary"].numpy()
        for p in np.unique(ids):
            lab[p] = int(y[ids == p].max())
        del d
    pids = sorted(lab)
    X, PID = [], []
    t0 = time.time()
    for i, rec in enumerate(pids):
        w, _ = build_patient_windows(os.path.join(RAW, str(rec)), fs_out)
        if len(w) == 0:
            continue
        X.append(w)
        PID += [str(rec)] * len(w)
        if i % 150 == 0:
            print(f"  {i}/{len(pids)}  {time.time()-t0:.0f}s", flush=True)
    X = np.concatenate(X).astype(np.float32)
    PID = np.array(PID)
    os.makedirs(OUT, exist_ok=True)
    np.savez(cache, X=X, pid=PID)
    print(f"built {tag}: X={X.shape}, {len(set(PID))} patients")
    return X, PID


# --------------------------------------------------------------- training
def run_fold(Xtr, ytr, Xva, yva, Xte, device, seed, epochs=40, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # normalisation fit on the TRAINING fold only
    mu = Xtr[:, 0].mean()
    sd = max(float(Xtr[:, 0].std()), 1e-6)

    def norm(A):
        B = A.copy()
        B[:, 0] = (B[:, 0] - mu) / sd
        return B

    Xtr, Xva, Xte = norm(Xtr), norm(Xva), norm(Xte)
    model = SmallCNN(in_ch=Xtr.shape[1]).to(device)
    npos, nneg = int(ytr.sum()), int((1 - ytr).sum())
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([nneg / max(npos, 1)], device=device))
    w = (1.0 / np.sqrt(np.array([nneg, npos], float)))[ytr]
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=64,
                    sampler=WeightedRandomSampler(
                        torch.as_tensor(w, dtype=torch.double), len(ytr), True))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gs = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    @torch.no_grad()
    def infer(A):
        model.eval()
        o = np.zeros(len(A), np.float32)
        for i in range(0, len(A), 256):
            xb = torch.from_numpy(A[i:i + 256]).to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                o[i:i + 256] = model(xb).float().view(-1).cpu().numpy()
        return o

    best, state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = crit(model(xb).view(-1), yb)
            gs.scale(loss).backward()
            gs.step(opt)
            gs.update()
        sch.step()
        a = roc_auc_score(yva, infer(Xva)) if len(np.unique(yva)) > 1 else 0.5
        if a > best:
            best, bad = a, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if state:
        model.load_state_dict(state)
    return infer(Xte), ep + 1


def gates(lab, sc):
    o = np.argsort(-sc)
    y = lab[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    sens = tp / max(y.sum(), 1)
    spec = 1 - fp / max((1 - y).sum(), 1)
    p = 1 / (1 + np.exp(-sc))
    b = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[b == k].mean() - lab[b == k].mean()) * (b == k).mean()
              for k in range(10) if (b == k).any())
    return dict(sens_at_80spec=float(sens[np.argmin(np.abs(spec - 0.80))]),
                spec_at_90sens=float(spec[np.argmin(np.abs(sens - 0.90))]),
                ece=float(ece))


def verdict(a):
    if a >= 0.85:
        return "FREEZE + AUDIT"
    if a >= 0.80:
        return "PRIMARY CANDIDATE"
    if a >= 0.75:
        return "PROMOTE to patient MIL"
    if a >= 0.72:
        return "DO NOT TUNE - likely KILL"
    return "KILL"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    print("PREPROCESSING AUDIT")
    for k, v in PREPROCESSING_AUDIT.items():
        print(f"  {k:24s} {v}")
    n_par = sum(p.numel() for p in SmallCNN().parameters())
    print(f"\nmodel parameters: {n_par:,}  (budget 100k-300k)")
    assert 100_000 <= n_par <= 300_000, f"parameter budget violated: {n_par}"

    lab = {}
    for sp in ("train", "val", "test"):
        d = torch.load(os.path.join(PROC, f"{sp}_dataset.pt"), weights_only=False)
        ids = np.array([m[0] for m in d["metadata"]])
        yy = d["y_primary"].numpy()
        for p in np.unique(ids):
            lab[p] = int(yy[ids == p].max())
        del d

    results = {}
    for tag, fs_out in (("E1A_1hz", 1.0), ("E1B_4hz", 4.0)):
        print(f"\n{'='*80}\n{tag}  (FHR at {fs_out:g} Hz)\n{'='*80}")
        X, pid = build(fs_out, tag)
        y = np.array([lab[p] for p in pid])
        print(f"windows {X.shape}, {len(set(pid))} patients, {y.sum()} positive windows")
        P = Protocol.load_or_create(pid, y)
        oof = np.zeros(len(X))
        cnt = np.zeros(len(X))
        per_fold = []
        t0 = time.time()
        for k, (tr_mask, te_p) in enumerate(P.folds(repeat=0), 1):
            tr = np.where(tr_mask)[0]
            te = np.isin(pid, te_p)
            tr_p = np.array(sorted(set(pid[tr].tolist())))
            pl = np.array([int(y[tr][pid[tr] == p].max()) for p in tr_p])
            st = pl if len(np.unique(pl)) > 1 and np.bincount(pl).min() >= 2 else None
            _, val_p = train_test_split(tr_p, test_size=0.2, stratify=st,
                                        random_state=SEED)
            isv = np.isin(pid[tr], val_p)
            fi, vi = tr[~isv], tr[isv]
            assert not (set(pid[fi]) & set(pid[vi])), "fit/val patient overlap"
            assert not (set(pid[fi]) & set(pid[te])), "fit/test patient overlap"
            p_te, eps = run_fold(X[fi], y[fi], X[vi], y[vi], X[te], device,
                                 SEED + k)
            oof[te] += p_te
            cnt[te] += 1
            fa = roc_auc_score(y[te], p_te) if len(np.unique(y[te])) > 1 else np.nan
            per_fold.append(fa)
            print(f"  fold {k}: window AUROC {fa:.4f}  ({eps} epochs, "
                  f"{time.time()-t0:.0f}s)", flush=True)
        oof /= np.maximum(cnt, 1)
        r = P.report(f"{tag} patient-level (max agg)", oof, how="max")
        plab, psc = P.to_patient(oof, how="max")
        r.update(gates(plab, psc))
        r.update(dict(experiment_id=tag, input_resolution_hz=fs_out,
                      input_duration_min=60, window_min=20, stride_min=2.5,
                      architecture="SmallCNN residual 1D", parameter_count=n_par,
                      optimizer="AdamW lr=1e-3 wd=1e-4", batch_size=64,
                      epoch_limit=40, early_stopping="inner-val AUROC, patience 8",
                      seed=SEED, n_windows=int(len(X)), n_patients=len(set(pid)),
                      per_fold_window=[float(v) for v in per_fold],
                      fold_sd=float(np.nanstd(per_fold, ddof=1)),
                      window_auroc=float(roc_auc_score(y, oof)),
                      delta_vs_LR=float(r["auroc"] - LR_BASELINE),
                      verdict=verdict(r["auroc"])))
        np.save(os.path.join(OUT, f"oof_{tag}.npy"), oof)
        results[tag] = r
        print(f"  window {r['window_auroc']:.4f} | fold sd {r['fold_sd']:.4f} | "
              f"sens@80spec {r['sens_at_80spec']:.3f} | ECE {r['ece']:.3f}")
        print(f"  >>> vs LR {LR_BASELINE}: {r['delta_vs_LR']:+.4f}   "
              f"VERDICT: {r['verdict']}")

    print(f"\n{'='*88}\nEXPERIMENT 1 SUMMARY\n{'='*88}")
    print(f"{'model':28s} {'AUROC':>7s} {'95% CI':>16s} {'AUPRC':>7s} "
          f"{'dLR':>8s}  verdict")
    print(f"{'Clinical LR (19 descriptors)':28s} {LR_BASELINE:7.4f} "
          f"[0.670-0.779] {0.4094:7.4f} {0.0:+8.4f}  BASELINE")
    for t, r in results.items():
        print(f"{t:28s} {r['auroc']:7.4f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}] "
              f"{r['auprc']:7.4f} {r['delta_vs_LR']:+8.4f}  {r['verdict']}")
    best = max(results.values(), key=lambda r: r["auroc"])
    print(f"\nGO/NO-GO: best raw-FHR AUROC {best['auroc']:.4f} -> {best['verdict']}")
    print("E1-C (multi-resolution) runs only if this is PROMOTE or better.")
    json.dump({"audit": PREPROCESSING_AUDIT, "results": results},
              open(os.path.join(OUT, "exp1_raw_fhr.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
