"""
Phase 9B -- does ENLARGING the pre-training corpus with external data help?

Controlled information-diversity experiment. The SSL objective, architecture,
fine-tuning procedure, folds and evaluation are identical across arms. The ONLY
thing that changes is which recordings the encoder is pre-trained on.

    arm 1  randinit        no pre-training                       (reference)
    arm 2  ssl_ctu         CTU-UHB only            = the control
    arm 3  ssl_ctu_fhrma   CTU-UHB + FHRMA         = the experiment
    arm 4  ssl_fhrma       FHRMA only              = zero-leakage external check

WHY ARM 4 EXISTS -- LEAKAGE DISCLOSURE, READ THIS
-------------------------------------------------
Arms 2 and 3 pre-train on all 547 CTU-UHB patients and are then evaluated by
5-fold CV over those same patients. The encoder has therefore seen every test
fold's *signals* (never their labels). That is transductive SSL: common in the
literature, but a leakage vector under this project's Gate 6.

  * arm 3 - arm 2 is a CLEAN difference: both arms carry identical CTU exposure,
    so the contrast isolates FHRMA's contribution.
  * arm 4 pre-trains on FHRMA alone, which contains no CTU-UHB patient at all,
    so its absolute number is leakage-free by construction.

Absolute values for arms 2 and 3 should be read as optimistic. This is stated
rather than corrected because per-fold pre-training costs 5x and the difference,
which is what the experiment asks about, is unaffected.

The 2026-08-17 CTU-only SSL run regressed Model 8 by -0.0262. That run used the
older MIL substrate and is not directly comparable, which is why arm 2 is re-run
here under the frozen protocol rather than quoted.

REPRESENTATION IS FROZEN. Phase 9A showed that "repairing" the constant baseline
toward expert behaviour costs -0.0972 AUROC, so calculate_iterative_baseline is
retained unchanged (docs/phase9a_detector_validation.md).

Run from the repo root:
    python scripts/probe_ssl_corpus.py --arm all
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.models.ctg_crossformer import (CTGCrossformerEncoder,
                                        CTGCrossformerForClassification)
from src.models.ctg_crossformer_pretraining import (CTGCrossformerSSLEncoder,
                                                    CTGReconstructionDecoder,
                                                    CTGCrossformerSSLPretrainer,
                                                    masked_reconstruction_loss)
from src.preprocessing.baseline import calculate_iterative_baseline
from src.preprocessing.filtering import (apply_lowpass_filter,
                                         interpolate_missing, remove_spikes)
from src.training.protocol import Protocol
from src.training.ssl_masking import mask_ctg_signal

PROC = "data/processed_clinical"
FHRMA_DIR = "data/raw/fhrma/CTGDL_FHRMA_ano_csv"
OUT = "results/phase9/ssl"
CKPT = "checkpoints/phase9_ssl"
FS, WIN, STRIDE = 4.0, 4800, 600


# ------------------------------------------------------------------ corpora
def load_ctu():
    """The frozen clinical substrate.

    NOTE: X on disk is ALREADY z-scored by pipeline_clinical.py using
    ctu_signal_scaler.npz (fitted on the train split only). It is NOT raw bpm.
    Re-scaling here would be wrong, and scaling FHRMA by anything other than
    that same stored scaler would put the two corpora in different spaces.
    """
    Xs, ys, ps = [], [], []
    for sp in ("train", "val", "test"):
        d = torch.load(os.path.join(PROC, f"{sp}_dataset.pt"), weights_only=False)
        Xs.append(d["X"].numpy()[:, :2])
        ys.append(d["y_primary"].numpy())
        ps.append(np.array([m[0] for m in d["metadata"]]))
        del d
    z = np.load(os.path.join(PROC, "ctu_signal_scaler.npz"))
    scaler = (z["mean"].reshape(1, 2, 1), z["std"].reshape(1, 2, 1))
    return (np.concatenate(Xs).astype(np.float32), np.concatenate(ys),
            np.concatenate(ps), scaler)


def build_fhrma(scaler):
    """FHRMA windows through the SAME chain and the SAME scaler as CTU."""
    cache = os.path.join(OUT, "fhrma_pool.npy")
    if os.path.exists(cache):
        P = np.load(cache)
        print(f"[cache] FHRMA pool {P.shape}")
        return P
    mu, sd = scaler
    files = [f for f in sorted(glob.glob(os.path.join(FHRMA_DIR, "*.csv")))
             if not f.endswith(".csv.csv")]
    out = []
    for p in files:
        d = pd.read_csv(p)
        fhr = d["fhr"].to_numpy(float)
        uc = d["toco"].to_numpy(float)
        f = remove_spikes(fhr.copy(), fs=FS)
        f = interpolate_missing(f, clip_min=50.0, clip_max=240.0)
        f = apply_lowpass_filter(f, fs=FS)
        u = apply_lowpass_filter(uc.copy(), fs=FS)
        for st in range(0, len(f) - WIN + 1, STRIDE):
            a, b = f[st:st + WIN], u[st:st + WIN]
            if (fhr[st:st + WIN] <= 0).mean() > 0.5:
                continue
            base = calculate_iterative_baseline(a)      # frozen representation
            out.append(np.vstack([a - base, b]).astype(np.float32))
    P = np.asarray(out, np.float32)
    P = ((P - mu) / sd).astype(np.float32)   # the CTU train-split scaler
    os.makedirs(OUT, exist_ok=True)
    np.save(cache, P)
    print(f"FHRMA pool: {P.shape[0]} windows from {len(files)} recordings")
    return P


# ------------------------------------------------------------------ pretrain
def pretrain(pool, tag, device, epochs=60, patience=10, bs=64):
    path = os.path.join(CKPT, f"{tag}.pth")
    if os.path.exists(path):
        print(f"[cache] pretrained encoder {path}")
        return path
    os.makedirs(CKPT, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    idx = np.arange(len(pool))
    np.random.shuffle(idx)
    n_val = max(int(0.1 * len(idx)), 1)
    va, tr = idx[:n_val], idx[n_val:]
    enc = CTGCrossformerSSLEncoder(in_channels=2, seq_len=WIN, cnn_channels=128,
                                   n_heads_cross=4, n_heads_tf=8, n_tf_layers=4,
                                   d_ff=512, dropout=0.1, latent_dim=128)
    model = CTGCrossformerSSLPretrainer(enc, CTGReconstructionDecoder()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    dl = DataLoader(TensorDataset(torch.from_numpy(pool[tr])), batch_size=bs,
                    shuffle=True)
    Xva = torch.from_numpy(pool[va])
    best, bad, t0 = np.inf, 0, time.time()
    for ep in range(epochs):
        model.train()
        for (xb,) in dl:
            xb = xb.to(device, non_blocking=True)
            xm, m = mask_ctg_signal(xb)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = masked_reconstruction_loss(model(xm), xb, m)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        vl, nb = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(Xva), bs):
                xb = Xva[i:i + bs].to(device)
                xm, m = mask_ctg_signal(xb)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    vl += float(masked_reconstruction_loss(model(xm), xb, m))
                nb += 1
        vl /= max(nb, 1)
        if vl < best - 1e-5:
            best, bad = vl, 0
            torch.save(model.encoder.state_dict(), path)
        else:
            bad += 1
            if bad >= patience:
                break
        if ep % 10 == 0:
            print(f"    ep {ep:3d} holdout {vl:.5f} best {best:.5f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"  [{tag}] best holdout {best:.5f}, {ep+1} epochs, "
          f"{time.time()-t0:.0f}s -> {path}")
    return path


# ------------------------------------------------------------------ finetune
class Focal(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.g, self.w = gamma, pos_weight

    def forward(self, logit, target):
        logit, target = logit.view(-1), target.view(-1).float()
        p = torch.sigmoid(logit)
        ce = nn.functional.binary_cross_entropy_with_logits(logit, target,
                                                            reduction="none")
        pt = p * target + (1 - p) * (1 - target)
        a = self.w * target + 1.0 * (1 - target)
        return (a * (1 - pt) ** self.g * ce).mean()


def finetune_fold(Xtr, ytr, Xva, yva, Xte, ckpt, device, seed, epochs=40,
                  patience=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = CTGCrossformerEncoder(in_channels=2, seq_len=WIN, cnn_channels=128,
                                n_heads_cross=4, n_heads_tf=8, n_tf_layers=4,
                                d_ff=512, dropout=0.1, latent_dim=128)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        if len(missing) > 4:
            raise RuntimeError(f"encoder load looks wrong: {len(missing)} missing")
    model = CTGCrossformerForClassification(enc, hidden_dim=128, dropout=0.3).to(device)
    npos, nneg = int(ytr.sum()), int((1 - ytr).sum())
    crit = Focal(2.0, nneg / max(npos, 1))
    w = (1.0 / np.sqrt(np.array([nneg, npos], float)))[ytr]
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=32,
                    sampler=WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                                  len(ytr), replacement=True))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4,
                                              total_steps=epochs * len(dl),
                                              pct_start=0.10,
                                              final_div_factor=3e-4 / 1.2e-9)
    gs = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    @torch.no_grad()
    def infer(Xa):
        model.eval()
        o = np.zeros(len(Xa), np.float32)
        for i in range(0, len(Xa), 128):
            xb = torch.from_numpy(Xa[i:i + 128]).to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                o[i:i + 128] = model(xb).float().view(-1).cpu().numpy()
        return o

    best, state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = crit(model(xb), yb)
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
    return infer(Xte)


def gates(lab, sc):
    """Gate 4 (threshold behaviour) and Gate 5 (calibration)."""
    o = np.argsort(-sc)
    y = lab[o]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    sens = tp / max(y.sum(), 1)
    spec = 1 - fp / max((1 - y).sum(), 1)
    i = np.argmin(np.abs(spec - 0.80))
    j = np.argmin(np.abs(sens - 0.90))
    p = 1 / (1 + np.exp(-sc))
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[bins == b].mean() - lab[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    return dict(sens_at_80spec=float(sens[i]), spec_at_90sens=float(spec[j]),
                ece=float(ece))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)

    X, y, pid, scaler = load_ctu()
    print(f"CTU pool {X.shape}, {len(set(pid))} patients, {y.sum()} positive windows")
    F = build_fhrma(scaler)
    print(f"FHRMA pool {F.shape}")

    P = Protocol.load_or_create(pid, y)
    arms = {"randinit": None, "ssl_ctu": X, "ssl_ctu_fhrma": np.concatenate([X, F]),
            "ssl_fhrma": F}
    if args.arm != "all":
        arms = {args.arm: arms[args.arm]}

    results = {}
    for tag, pool in arms.items():
        print(f"\n===== arm: {tag} =====", flush=True)
        ckpt = None if pool is None else pretrain(pool, tag, device)
        oof = np.zeros(len(X))
        cnt = np.zeros(len(X))
        per_fold = []
        for k, (tr_mask, te_p) in enumerate(P.folds(repeat=0), 1):
            # Protocol.folds() yields a BOOLEAN mask over windows, not indices
            tr = np.where(tr_mask)[0]
            te = np.isin(pid, te_p)
            tr_p = np.array(sorted(set(pid[tr].tolist())))
            pl = np.array([int(y[tr][pid[tr] == p].max()) for p in tr_p])
            st = pl if len(np.unique(pl)) > 1 and np.bincount(pl).min() >= 2 else None
            fit_p, val_p = train_test_split(tr_p, test_size=0.2, stratify=st,
                                            random_state=42)
            isv = np.isin(pid[tr], val_p)
            fi, vi = tr[~isv], tr[isv]
            p_te = finetune_fold(X[fi], y[fi], X[vi], y[vi], X[te], ckpt,
                                 device, 42 + k)
            oof[te] += p_te
            cnt[te] += 1
            fa = roc_auc_score(y[te], p_te) if len(np.unique(y[te])) > 1 else np.nan
            per_fold.append(fa)
            print(f"  fold {k}: window AUROC {fa:.4f}", flush=True)
        oof /= np.maximum(cnt, 1)
        r = P.report(f"{tag} (patient-level)", oof, how="max")
        lab, sc = P.to_patient(oof, how="max")
        r.update(gates(lab, sc))
        r["per_fold_window"] = [float(v) for v in per_fold]
        r["fold_sd"] = float(np.nanstd(per_fold, ddof=1))
        results[tag] = r
        np.save(os.path.join(OUT, f"oof_{tag}.npy"), oof)
        print(f"  sens@80spec {r['sens_at_80spec']:.3f}  "
              f"spec@90sens {r['spec_at_90sens']:.3f}  ECE {r['ece']:.3f}  "
              f"fold sd {r['fold_sd']:.4f}")

    if len(results) > 1:
        print("\n" + "=" * 92)
        print(f"{'arm':16s} {'AUROC':>7s} {'95% CI':>16s} {'AUPRC':>7s} "
              f"{'s@80sp':>7s} {'sp@90s':>7s} {'ECE':>6s} {'foldSD':>7s}")
        print("=" * 92)
        for t, r in results.items():
            print(f"{t:16s} {r['auroc']:7.4f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}] "
                  f"{r['auprc']:7.4f} {r['sens_at_80spec']:7.3f} "
                  f"{r['spec_at_90sens']:7.3f} {r['ece']:6.3f} {r['fold_sd']:7.4f}")
        if "ssl_ctu" in results and "ssl_ctu_fhrma" in results:
            d = results["ssl_ctu_fhrma"]["auroc"] - results["ssl_ctu"]["auroc"]
            print(f"\nFHRMA contribution (clean contrast, identical CTU exposure): "
                  f"{d:+.4f}   [bar +0.0642]")
    json.dump(results, open(os.path.join(OUT, "ssl_corpus_results.json"), "w"),
              indent=1, default=float)
    print(f"\nwrote {OUT}/ssl_corpus_results.json")


if __name__ == "__main__":
    main()
