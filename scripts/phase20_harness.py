"""
Phase 20 shared evaluation harness (docs/phase20_redesign_protocol.md, section 3).

One pipeline, identical for every arm; only the CTG window/step-score source differs:
  1. per split, out-of-fold window scores from the arm's window model      (oof_window_scores)
  2. TAM-architecture pooler (seeds 42/43/44, mean of running scores)      (pooler_oof_sequences)
  3. parity model fit fresh on the split's training patients               (fit_fusion_split)
  4. alpha selected once per fold by sample-weighted pooled training AUROC (grid step 0.05)
  5. held-out patients scored at each horizon's eligible prefix

Built on phase19_pooler_experiment.py (trainer) and phase19b_fusion_tam_vs_prs.py (fusion). Nothing here
touches the locked PRS/TAM/URM artefacts; it only reads them.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.phase16_causal_attention import eligible_prefix_length
from scripts.phase16_temporal_attention_model import carve_inner_validation
from scripts.phase19_pooler_experiment import (build_seqs, make_inputs, pooled_all_prefix, fit_arm,
                                               fast_auc, boot_all, summarize_delta, CUE_COLS, X40_PATH, X40_NAMES_PATH)

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
DATA_DIR = "data/processed_clinical"
OUT_DIR = "results/phase20_redesign"
os.makedirs(OUT_DIR, exist_ok=True)

H = [0, 10, 20, 30]
ALPHA_GRID = np.linspace(0.0, 1.0, 21)
SEEDS = [42, 43, 44]
RESPLIT_SEEDS = [11, 22, 33, 44, 55]
FRESH_RESPLIT_SEEDS = [66, 77, 88, 99, 111]


# ------------------------------------------------------------------ context
class Ctx:
    pass


def load_ctx():
    c = Ctx()
    blob = json.load(open(FOLDS_PATH))
    c.all_pids = sorted(blob["assignment"].keys())
    c.canon = {p: blob["assignment"][p][0] for p in c.all_pids}
    c.df = pd.read_csv(ROLLING_PATH); c.df["patient_id"] = c.df["patient_id"].astype(str)
    c.pid_w = c.df["patient_id"].values
    c.tdel_w = c.df["time_before_delivery_min"].values
    c.y_w = c.df["primary_label_715"].values
    c.y = {p: int(c.df[c.df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in c.all_pids}
    c.X40 = np.load(X40_PATH)["X_state_trajectory"]
    names = json.load(open(X40_NAMES_PATH)); c.col_idx = [names.index(k) for k in CUE_COLS]
    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    c.test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    c.d464 = [p for p in c.all_pids if p not in set(c.test_pids)]
    meta = pd.read_csv(METADATA_PATH).set_index("record_id")
    c.parity = {p: float(meta.loc[int(p), "parity"]) for p in c.all_pids}
    c.win_idx = {p: np.where(c.pid_w == p)[0] for p in c.all_pids}
    return c


def split_assignments(c, universe, resplit_seeds=RESPLIT_SEEDS, canonical=True):
    """{name: {pid: fold}} on `universe`. Canonical = folds.json assignment restricted to the universe."""
    out = {}
    if canonical:
        out["CANONICAL"] = {p: c.canon[p] for p in universe}
    pids = np.array(sorted(universe)); yv = np.array([c.y[p] for p in pids])
    for rs in resplit_seeds:
        a = {}
        for f, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rs).split(pids, yv)):
            for p in pids[te]:
                a[p] = f
        out[f"resplit_{rs}"] = a
    return out


# ------------------------------------------------------------------ step 1: window model, out-of-fold
def fit_p6(Xtr, ytr, Xte, **kw):
    """The locked P6 recipe: StandardScaler + LogisticRegression(C=0.05), unweighted (phase13_evaluation_audit)."""
    sc = StandardScaler(); a = sc.fit_transform(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42).fit(a, ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def oof_window_scores(c, X, universe, assign, fit_fn=fit_p6):
    """Window-level out-of-fold scores for every window of `universe` patients (0 elsewhere)."""
    fold_w = np.full(len(c.pid_w), -1, dtype=int)
    for p in universe:
        fold_w[c.win_idx[p]] = assign[p]
    pred = np.zeros(len(c.pid_w), dtype=np.float64)
    for f in range(5):
        te = fold_w == f; tr = (fold_w >= 0) & (fold_w != f)
        pred[te] = fit_fn(X[tr], c.y_w[tr], X[te], tdel=c.tdel_w[tr])
    return pred


# ------------------------------------------------------------------ step 2: pooler
def pooler_oof_sequences(c, pred, universe, assign, seeds=SEEDS, offset=0, log=None):
    """Out-of-fold running scores of the TAM-architecture pooler on the window scores `pred`.
    Returns {pid: chronological array (T,)} (mean over seeds) and the Seqs object."""
    S = build_seqs(pred.astype(np.float32), c.pid_w, c.df, c.X40, c.col_idx, list(universe), c.y)
    X = make_inputs(S, False)
    Z = torch.zeros(S.R.shape)
    for f in range(5):
        te = [p for p in universe if assign[p] == f]; tr_all = [p for p in universe if assign[p] != f]
        inner_tr, inner_val = carve_inner_validation(tr_all, c.y, seed=42 + f + offset)
        tr_idx = torch.tensor([S.row[p] for p in inner_tr]); val_idx = torch.tensor([S.row[p] for p in inner_val])
        te_rows = torch.tensor([S.row[p] for p in te])
        for s in seeds:
            sc, n_ep, best = fit_arm(S, X, tr_idx, val_idx, "bce", s)
            with torch.no_grad():
                z = pooled_all_prefix(sc, X, S.R, S.L)
            Z[te_rows] += z[te_rows] / len(seeds)
            if log is not None:
                log.append({"fold": f, "seed": s, "epochs": n_ep, "best_val_bce": round(best, 5)})
    seq = {p: Z[S.row[p], :int(S.L[S.row[p]])].numpy().astype(np.float64) for p in universe}
    return seq, S


def window_sequences(c, pred, universe):
    """The raw chronological window-score sequences (the PRS 'latest window' running score)."""
    out = {}
    for p in universe:
        idx = c.win_idx[p]; o = np.argsort(-c.tdel_w[idx])
        out[p] = pred[idx[o]].astype(np.float64)
    return out


def tdel_by_pid(c, universe):
    return {p: np.sort(c.tdel_w[c.win_idx[p]])[::-1] for p in universe}


# ------------------------------------------------------------------ steps 3-5: parity + alpha + horizon scoring
def fit_parity(par_tr, y_tr, par_all):
    sc = StandardScaler(); a = sc.fit_transform(par_tr.reshape(-1, 1))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(a, y_tr)
    return clf.predict_proba(sc.transform(par_all.reshape(-1, 1)))[:, 1]


def select_alpha(pids, seq, par, y):
    z, p, yy, w = [], [], [], []
    for pid in pids:
        s = seq[pid]; T = len(s)
        z.extend(s.tolist()); p.extend([par[pid]] * T); yy.extend([y[pid]] * T); w.extend([1.0 / T] * T)
    z, p, yy, w = map(np.array, (z, p, yy, w))
    best, best_auc = 0.5, -1.0
    for a in ALPHA_GRID:
        auc = roc_auc_score(yy, a * z + (1 - a) * p, sample_weight=w)
        if auc > best_auc:
            best_auc, best = auc, float(a)
    return best


def horizon_scores(pids, seq, tdel, alpha, par):
    out = np.zeros((len(pids), 4))
    for j, pid in enumerate(pids):
        s = seq[pid]; T = len(s)
        for a, h in enumerate(H):
            k = eligible_prefix_length(tdel[pid], h, T) - 1
            out[j, a] = s[k] if alpha is None else alpha * s[k] + (1 - alpha) * par[pid]
    return out


def urm_cv(c, universe, assign, seq, tdel, parity_fn=None):
    """URM-form fusion evaluated with 5-fold assignment. Returns dict of (n,4) score arrays in `universe` order
    ('URM', 'CTG' = the CTG score alone, 'PARITY'), plus the selected alphas.
    parity_fn(tr_pids, all_pids) -> {pid: p} lets Stage 1 substitute a covariate model for parity."""
    order = list(universe); row = {p: i for i, p in enumerate(order)}; n = len(order)
    res = {k: np.zeros((n, 4)) for k in ("URM", "CTG", "PARITY")}; alphas = []
    for f in range(5):
        te = [p for p in order if assign[p] == f]; tr = [p for p in order if assign[p] != f]
        if parity_fn is None:
            fitted = fit_parity(np.array([c.parity[p] for p in tr]), np.array([c.y[p] for p in tr]),
                                np.array([c.parity[p] for p in order]))
            par = dict(zip(order, fitted))
        else:
            par = parity_fn(tr, order)
        a = select_alpha(tr, seq, par, c.y); alphas.append(a)
        rows = [row[p] for p in te]
        res["URM"][rows] = horizon_scores(te, seq, tdel, a, par)
        res["CTG"][rows] = horizon_scores(te, seq, tdel, None, par)
        res["PARITY"][rows] = np.tile(np.array([par[p] for p in te])[:, None], (1, 4))
    return res, alphas


def m_of(y, scores):
    return float(np.mean([fast_auc(y, scores[:, a]) for a in range(4)]))


def per_h(y, scores):
    return [float(fast_auc(y, scores[:, a])) for a in range(4)]
