"""
Phase 20 Stage 2 -- C6a: one causal GRU replaces both the window model and the TAM pooler (protocol section 4).

Unidirectional GRU (hidden 16) over the chronological sequence of standardised locked 40-D window vectors + elapsed/60,
input dropout 0.3, linear head -> patient-risk score at every step; TAM objective (BCE at every causal truncation, per-patient
1/T normalisation); Adam lr 3e-3, wd 1e-3, <=150 epochs, patience 10 on inner-val BCE (15% stratified, seed 42+fold+offset);
seeds 42/43/44, mean of the three seeds' running scores. Nothing is searched. Its running score replaces TAM's in harness
steps 3-5 (MCM, alpha, horizon scoring); compared with B0 on D464 canonical + 5 resplits.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import scripts.phase20_harness as hz
from scripts.phase16_temporal_attention_model import carve_inner_validation
from scripts.phase20_stage1_candidates import compare, adopt_rule

OUT = hz.OUT_DIR
HIDDEN, DROPOUT, LR, WD, MAX_EP, PATIENCE = 16, 0.3, 3e-3, 1e-3, 150, 10


class GRUScorer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.drop = nn.Dropout(DROPOUT); self.gru = nn.GRU(d, HIDDEN, batch_first=True); self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        h, _ = self.gru(self.drop(x))
        return self.head(h).squeeze(-1)                       # logits (N,T)


def build_tensors(c, pids):
    """Chronological padded (N,Tmax,41) raw features (40-D + elapsed/60), lengths, labels."""
    ss = c.df["start_sample"].values
    seqs, els = [], []
    for p in pids:
        idx = c.win_idx[p]; o = idx[np.argsort(ss[idx])]
        seqs.append(c.X40[o].astype(np.float32)); e = ss[o] / (4.0 * 60.0); els.append((e - e[0]).astype(np.float32) / 60.0)
    L = np.array([len(s) for s in seqs]); T = L.max(); N = len(pids)
    X = np.zeros((N, T, seqs[0].shape[1] + 1), dtype=np.float32)
    for i, (s, e) in enumerate(zip(seqs, els)):
        X[i, :len(s), :-1] = s; X[i, :len(s), -1] = e
    return torch.tensor(X), torch.tensor(L), torch.tensor([float(c.y[p]) for p in pids])


def loss_fn(logits, Y, L, sub):
    valid = torch.arange(logits.shape[1])[None, :] < L[:, None]
    ll = F.binary_cross_entropy_with_logits(logits, Y[:, None].expand_as(logits), reduction="none")
    return ((ll * valid).sum(1) / L)[sub].mean()


def fit_gru(X, L, Y, tr_idx, val_idx, seed):
    torch.manual_seed(seed)
    m = GRUScorer(X.shape[-1]); opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    best, best_state, bad, n_ep = float("inf"), None, 0, 0
    for ep in range(MAX_EP):
        n_ep = ep + 1
        m.train(); opt.zero_grad(); loss_fn(m(X), Y, L, tr_idx).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = loss_fn(m(X), Y, L, val_idx).item()
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    m.load_state_dict(best_state); m.eval()
    return m, n_ep


def gru_oof_sequences(c, universe, assign, offset, log):
    pids = list(universe); row = {p: i for i, p in enumerate(pids)}
    Xr, L, Y = build_tensors(c, pids); Z = torch.zeros(Xr.shape[:2])
    for f in range(5):
        te = [p for p in pids if assign[p] == f]; tr_all = [p for p in pids if assign[p] != f]
        tr_rows = [row[p] for p in tr_all]
        valid = (torch.arange(Xr.shape[1])[None, :] < L[:, None])
        allw = Xr[tr_rows][valid[tr_rows]]                                   # training windows only
        mu, sd = allw.mean(0), allw.std(0).clamp_min(1e-6)
        X = torch.where(valid[..., None], (Xr - mu) / sd, torch.zeros_like(Xr))
        inner_tr, inner_val = carve_inner_validation(tr_all, c.y, seed=42 + f + offset)
        tr_idx = torch.tensor([row[p] for p in inner_tr]); val_idx = torch.tensor([row[p] for p in inner_val])
        te_rows = torch.tensor([row[p] for p in te])
        for s in hz.SEEDS:
            m, n_ep = fit_gru(X, L, Y, tr_idx, val_idx, s)
            with torch.no_grad():
                Z[te_rows] += torch.sigmoid(m(X))[te_rows] / len(hz.SEEDS)
            log.append({"fold": f, "seed": s, "epochs": n_ep})
    return {p: Z[row[p], :int(L[row[p]])].numpy().astype(np.float64) for p in pids}


def main():
    c = hz.load_ctx()
    y464 = np.array([c.y[p] for p in c.d464]); tdel464 = hz.tdel_by_pid(c, c.d464)
    splits = hz.split_assignments(c, c.d464)
    base = np.load(os.path.join(OUT, "stage0_baseline_D464_scores.npz"))
    rows = []; log = []
    for name, assign in splits.items():
        off = 0 if name == "CANONICAL" else 100 + int(name.split("_")[1])
        seq = gru_oof_sequences(c, c.d464, assign, off, log)
        r, al = hz.urm_cv(c, c.d464, assign, seq, tdel464)
        rec = {"cand": "C6a", "split": name, "alpha_mean": round(float(np.mean(al)), 3), "M_gru_alone": hz.m_of(y464, r["CTG"])}
        rec.update(compare(y464, base[name], r["URM"], r["CTG"], r["CTG"]))
        rows.append(rec)
        np.save(os.path.join(OUT, f"stage2_scores_C6a_{name}.npy"), r["URM"])
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "stage2_c6a_results.csv"), index=False)
        print(f"  C6a {name:<12} dM={rec['dM']:+.4f} (p={rec['dM_p']:.3f})  M {rec['M_base']:.4f}->{rec['M_cand']:.4f}  "
              f"GRU-alone M {rec['M_gru_alone']:.4f}  per-horizon " + " ".join(f"{h}m={rec[f'd{h}']:+.4f}" for h in hz.H), flush=True)
    df = pd.DataFrame(rows); can = df[df.split == "CANONICAL"].iloc[0].to_dict(); res = df[df.split != "CANONICAL"]
    v = adopt_rule(can, res, adj_p=can["dM_p"])           # Holm family K = 1 (C6b not run: no PRS-v2 feature candidate adopted)
    v.update({"canonical_dM": round(can["dM"], 4), "canonical_p": round(can["dM_p"], 4), "resplit_dM": [round(x, 4) for x in res.dM],
              "epoch_cap_hits": int(sum(1 for l in log if l["epochs"] >= MAX_EP)), "n_trainings": len(log)})
    json.dump(v, open(os.path.join(OUT, "stage2_c6a_verdict.json"), "w"), indent=2)
    print("  VERDICT C6a:", v)


if __name__ == "__main__":
    main()
