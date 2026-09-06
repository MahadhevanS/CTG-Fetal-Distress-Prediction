"""
Temporal control: T3 (order-invariant) vs T4 (order-aware) vs T4-shuffled.

This is the single go/no-go control from docs/temporal_modelling_feasibility.md
section G. It is NOT the full T1-T5 ladder, there is no encoder fine-tuning, and
there is no hyperparameter search -- one predefined configuration per rung.

    T3          frozen embeddings -> gated attention pooling -> classifier
                ORDER-INVARIANT by construction (attention is a set operation
                and no positional information is supplied).
    T4          frozen embeddings -> GRU(16) -> classifier
                ORDER-AWARE. This is the only rung that can see sequence order.
    T4-shuffled identical to T4, but each patient's windows are permuted with a
                fixed seed for both training and evaluation.

THE COMPARISON THAT MATTERS IS T4 vs T3, not T4 vs T0. T1-T3 are order-invariant,
so measuring T4 against max-aggregation would credit temporal modelling for a
gain that came from learned pooling.

Decision rule, pre-registered:
    T4 > T3 by >= 0.02 AND T4 > T4-shuffled  -> order carries information
    T4 ~ T3 ~ T4-shuffled                     -> order is not the missing signal
    T3 > T0 but T4 ~ T3                       -> pooling helps, ordering does not

Protocol: src/training/protocol.py, repeat 0, 5 patient-grouped folds, patient
AUROC, CI bootstrapped over patients. Encoder recipe matches ladder rung E1
(mslstm, 20 epochs, sqrt-inverse sampler, focal loss, OneCycleLR) so the frozen
representation is the one whose max-aggregated score is 0.6834.

Usage:
    python scripts/run_temporal_control.py
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

import torch                                                       # noqa: E402
import torch.nn as nn                                              # noqa: E402
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler  # noqa: E402
from sklearn.metrics import roc_auc_score                          # noqa: E402
from scipy.stats import wilcoxon                                   # noqa: E402

from src.models.encoder_registry import build_encoder              # noqa: E402
from src.models.multitask_physio import MultiTaskPhysioNet, focal_bce  # noqa: E402
from src.training.protocol import Protocol                         # noqa: E402

REPEAT = 0
T_MAX = 17           # measured maximum windows per patient; no truncation needed
LATENT = 128
SEED = 42

# One predefined configuration. No sweep -- see the brief.
AGG_EPOCHS = 100
AGG_LR = 1e-3
AGG_WD = 1e-3
ATTN_DIM = 16
GRU_HIDDEN = 16


# ------------------------------------------------------------------ aggregators
class AttentionPool(nn.Module):
    """Gated attention pooling (ABMIL, Ilse et al. 2018). A SET operation:
    permuting the inputs permutes the attention weights identically and leaves
    the pooled vector unchanged. No positional encoding is added, deliberately --
    this rung is the order-invariant control."""

    def __init__(self, latent=LATENT, attn_dim=ATTN_DIM):
        super().__init__()
        self.V = nn.Linear(latent, attn_dim)
        self.U = nn.Linear(latent, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.head = nn.Linear(latent, 1)

    def forward(self, h, mask):
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))          # padded slots -> 0 weight
        a = torch.softmax(a, dim=1)
        pooled = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return self.head(pooled).squeeze(-1), a


class GRUPool(nn.Module):
    """GRU over the window sequence. ORDER-AWARE. Uses packed sequences so
    padded steps never enter the recurrence and the final state is taken at each
    patient's true length."""

    def __init__(self, latent=LATENT, hidden=GRU_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(latent, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, h, mask):
        lens = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lens, batch_first=True,
                                                   enforce_sorted=False)
        _, hn = self.gru(packed)
        return self.head(hn[-1]).squeeze(-1), None


# ------------------------------------------------------------------ mask checks
def verify_masks(device):
    """Numerically verify that padding cannot influence either aggregator."""
    torch.manual_seed(0)
    print("=" * 78)
    print("PADDING MASK VERIFICATION")
    print("=" * 78)
    B, T, L = 4, T_MAX, LATENT
    h = torch.randn(B, T, L, device=device)
    lens = torch.tensor([17, 9, 3, 12], device=device)
    mask = torch.arange(T, device=device)[None, :] < lens[:, None]

    ok = True
    for name, model in (("attention", AttentionPool().to(device)),
                        ("GRU", GRUPool().to(device))):
        model.eval()
        with torch.no_grad():
            out1, a1 = model(h, mask)
            # (1) corrupt the padded slots with huge values
            h2 = h.clone()
            for i, n in enumerate(lens.tolist()):
                h2[i, n:] = 1e4 * torch.randn(T - n, L, device=device)
            out2, _ = model(h2, mask)
        same = torch.allclose(out1, out2, atol=1e-5)
        print(f"  {name:10s} output unchanged when padded slots are corrupted : "
              f"{'PASS' if same else 'FAIL'}  (max diff {float((out1-out2).abs().max()):.2e})")
        ok &= same
        if a1 is not None:
            pad_w = float(a1[~mask].abs().max())
            rows = float((a1.sum(1) - 1).abs().max())
            print(f"  {'':10s} attention on padded slots = {pad_w:.2e} "
                  f"{'PASS' if pad_w < 1e-8 else 'FAIL'}")
            print(f"  {'':10s} attention rows sum to 1, max error {rows:.2e} "
                  f"{'PASS' if rows < 1e-5 else 'FAIL'}")
            ok &= (pad_w < 1e-8) and (rows < 1e-5)

    # (2) attention must be permutation-invariant; GRU must NOT be
    perm = torch.randperm(17, device=device)
    hf = torch.randn(1, T, L, device=device)
    mf = torch.ones(1, T, dtype=torch.bool, device=device)
    for name, model in (("attention", AttentionPool().to(device)),
                        ("GRU", GRUPool().to(device))):
        model.eval()
        with torch.no_grad():
            o1, _ = model(hf, mf)
            o2, _ = model(hf[:, perm], mf)
        inv = torch.allclose(o1, o2, atol=1e-5)
        expect = (name == "attention")
        print(f"  {name:10s} permutation-invariant: {inv}  (expected {expect}) "
              f"{'PASS' if inv == expect else 'FAIL'}")
        ok &= (inv == expect)
    print(f"\n  ALL MASK CHECKS {'PASSED' if ok else 'FAILED'}\n")
    if not ok:
        sys.exit("[ABORT] mask verification failed -- no number below would be trustworthy.")
    return ok


# ------------------------------------------------------------------ data
def load(data_dir, in_channels=2):
    X, y, pid, start = [], [], [], []
    for s in ("train", "val", "test"):
        d = torch.load(os.path.join(data_dir, f"{s}_dataset.pt"), weights_only=False)
        X.append(d["X"].numpy()[:, :in_channels, :])
        y.append(d["y_primary"].numpy())
        pid.append(np.array([m[0] for m in d["metadata"]]))
        start.append(np.array([m[1] for m in d["metadata"]]))
    return np.vstack(X), np.concatenate(y), np.concatenate(pid), np.concatenate(start)


def train_encoder(Xtr, ytr, device, epochs, seed=SEED):
    """Ladder rung E1 exactly: mslstm, binary head only, no auxiliary tasks."""
    torch.manual_seed(seed)
    cfg = {"in_channels": Xtr.shape[1]}
    model = MultiTaskPhysioNet(build_encoder("mslstm", cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                       torch.tensor(ytr, dtype=torch.float32))
    cnt = np.bincount(ytr.astype(int))
    w = (1.0 / np.sqrt(np.maximum(cnt, 1)))[ytr.astype(int)]
    dl = DataLoader(ds, batch_size=32,
                    sampler=WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                                  len(w), True))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4,
                                              total_steps=max(len(dl) * epochs, 10),
                                              pct_start=0.1)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            focal_bce(model(xb.to(device))["distress"], yb.to(device)).backward()
            opt.step()
            sch.step()
    model.eval()
    return model


@torch.no_grad()
def embed(model, X, device, bs=256):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32).to(device)
        z = model.encoder(xb)
        if isinstance(z, tuple):
            z = z[0]
        out.append(z.cpu().numpy())
    return np.vstack(out)


def build_bags(Z, P, patients, order, shuffle_rng=None):
    """(N, T_MAX, LATENT) padded bags + boolean mask, chronological unless shuffled."""
    B = np.zeros((len(patients), T_MAX, Z.shape[1]), dtype=np.float32)
    M = np.zeros((len(patients), T_MAX), dtype=bool)
    for i, p in enumerate(patients):
        idx = order[p]
        if shuffle_rng is not None:
            idx = idx[shuffle_rng.permutation(len(idx))]
        n = min(len(idx), T_MAX)
        B[i, :n] = Z[idx[:n]]
        M[i, :n] = True
    return B, M


def train_agg(kind, Btr, Mtr, ytr, Bte, Mte, device, seed=SEED):
    torch.manual_seed(seed)
    model = (AttentionPool() if kind == "attn" else GRUPool()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=AGG_LR, weight_decay=AGG_WD)
    n_pos = max(ytr.sum(), 1)
    pw = torch.tensor([(len(ytr) - n_pos) / n_pos], dtype=torch.float32, device=device)
    bt = torch.tensor(Btr, device=device)
    mt = torch.tensor(Mtr, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    model.train()
    for _ in range(AGG_EPOCHS):
        perm = torch.randperm(len(yt), device=device)
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            logits, _ = model(bt[b], mt[b])
            lossf(logits, yt[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.tensor(Bte, device=device), torch.tensor(Mte, device=device))
        return torch.sigmoid(logits).cpu().numpy(), sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_clinical/")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="results/temporal_control.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verify_masks(device)

    X, y, pid, start = load(args.data_dir)
    P = Protocol.load_or_create(pid, y)
    order = {p: P.pidx[p][np.argsort(start[P.pidx[p]])] for p in P.patients}
    pos = {p: i for i, p in enumerate(P.patients)}
    print(f"{len(y)} windows | {len(P.patients)} patients | {int(P.plab.sum())} positive")
    print(f"T_MAX {T_MAX} | padded patients "
          f"{int(sum(len(order[p]) < T_MAX for p in P.patients))} "
          f"({100*np.mean([len(order[p]) < T_MAX for p in P.patients]):.1f}%)\n")

    RUNGS = ["T3_attention", "T4_gru", "T4_gru_shuffled"]
    oof = {r: np.zeros(len(P.patients)) for r in RUNGS}
    nparams = {}
    for k, (tr_mask, te_p) in enumerate(P.folds(repeat=REPEAT), 1):
        cache = f"results/oof_cache/embed_mslstm_e{args.epochs}_r{REPEAT}_f{k}.npy"
        if os.path.exists(cache):
            Z = np.load(cache)
            print(f"  fold {k}/5  [cached embeddings]")
        else:
            enc = train_encoder(X[tr_mask], y[tr_mask], device, args.epochs)
            Z = embed(enc, X, device)
            os.makedirs("results/oof_cache", exist_ok=True)
            np.save(cache, Z)
            print(f"  fold {k}/5  encoder trained, embeddings {Z.shape}")
            del enc
            torch.cuda.empty_cache()

        tr_p = np.array([p for p in P.patients if p not in set(te_p.tolist())])
        ytr = np.array([P.plab[pos[p]] for p in tr_p], dtype=np.float32)
        for rung in RUNGS:
            sh = np.random.default_rng(SEED) if "shuffled" in rung else None
            Btr, Mtr = build_bags(Z, P, tr_p, order, sh)
            sh2 = np.random.default_rng(SEED + 1) if "shuffled" in rung else None
            Bte, Mte = build_bags(Z, P, te_p, order, sh2)
            kind = "attn" if "attention" in rung else "gru"
            s, npar = train_agg(kind, Btr, Mtr, ytr, Bte, Mte, device)
            nparams[rung] = npar
            for p, v in zip(te_p, s):
                oof[rung][pos[p]] = v

    print("\n" + "=" * 78)
    print("RESULTS -- patient-level, frozen protocol, repeat 0")
    print("=" * 78)
    res = {}
    for rung in RUNGS:
        res[rung] = P.report_patient_scores(f"{rung:18s} ({nparams[rung]:,} params)",
                                            P.plab, oof[rung])
        res[rung]["n_params"] = nparams[rung]

    # per-fold scores and paired deltas, T4 vs T3 (the pre-registered comparison)
    print("\n" + "=" * 78)
    print("PER-FOLD AUROC and PAIRED DELTAS")
    print("=" * 78)
    perfold = {r: [] for r in RUNGS}
    for tr_mask, te_p in P.folds(repeat=REPEAT):
        lab = np.array([P.plab[pos[p]] for p in te_p])
        if len(np.unique(lab)) < 2:
            continue
        for r in RUNGS:
            perfold[r].append(roc_auc_score(lab, [oof[r][pos[p]] for p in te_p]))
    for r in RUNGS:
        print(f"  {r:18s} " + " ".join(f"{v:.4f}" for v in perfold[r]) +
              f"   mean {np.mean(perfold[r]):.4f}")
    print()
    for a, b in (("T4_gru", "T3_attention"), ("T4_gru", "T4_gru_shuffled")):
        d = np.array(perfold[a]) - np.array(perfold[b])
        try:
            pv = wilcoxon(d).pvalue
        except Exception:
            pv = float("nan")
        pooled = res[a]["auroc"] - res[b]["auroc"]
        verdict = "PASS" if (pooled >= 0.02 and (d > 0).sum() >= 4) else "fail"
        print(f"  {a} vs {b:18s} pooled {pooled:+.4f} | per-fold "
              f"[{', '.join(f'{v:+.3f}' for v in d)}] | {int((d>0).sum())}/5 | "
              f"W p={pv:.3f} | {verdict}")

    print("\n  Pre-registered bar: +0.02 pooled AND >=4/5 folds, T4 vs T3.")
    print("  Reference (different rung, same protocol): T0 max-agg 0.6834; "
          "19-feature LR 0.7268 [0.670-0.779].")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"results": res, "per_fold": perfold}, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
