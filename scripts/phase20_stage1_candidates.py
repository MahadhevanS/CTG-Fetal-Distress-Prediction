"""
Phase 20 Stage 1 (docs/phase20_redesign_protocol.md sections 4-5).

  --part window : C1, C2, C3, C5a, C5b, C5c  (Holm family K = 6) on the development set D464
  --part c4     : covariate sets S1 -> S2 -> S3 in fixed-sequence order (late fusion, B0-harness TAM sequences)

Per candidate and split: window model with its own hyper-parameters chosen by inner patient-grouped 4-fold CV inside each
outer training set (criterion: inner-OOF mean over horizons of single-window patient AUROC) -> out-of-fold window scores
-> common harness (pooler seeds 42-44, MCM, alpha, horizon scoring) -> compared with baseline B0 in the same harness.
"""
import os, sys, json, argparse, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import scripts.phase20_harness as hz

warnings.filterwarnings("ignore")
OUT = hz.OUT_DIR
HOLM_K = 6
SPLIT_NAMES = ["CANONICAL"] + [f"resplit_{s}" for s in hz.RESPLIT_SEEDS]


# ------------------------------------------------------------------ fitters
def _w(tdel, tau):
    return np.maximum(np.exp(-tdel / tau), 0.2)


def f_lr(C, tau=None):
    def fit(Xtr, ytr, Xte, tdel=None):
        sc = StandardScaler(); a = sc.fit_transform(Xtr)
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        clf.fit(a, ytr, sample_weight=None if tau is None else _w(tdel, tau))
        return clf.predict_proba(sc.transform(Xte))[:, 1]
    return fit


def f_en(l1, C):
    def fit(Xtr, ytr, Xte, tdel=None):
        sc = StandardScaler(); a = sc.fit_transform(Xtr)
        clf = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1, C=C, max_iter=3000, random_state=42)
        clf.fit(a, ytr)
        return clf.predict_proba(sc.transform(Xte))[:, 1]
    return fit


def _hgb(Xtr, ytr, Xte):
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100,
                                       l2_regularization=1.0, random_state=42)
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def f_hgb():
    return lambda Xtr, ytr, Xte, tdel=None: _hgb(Xtr, ytr, Xte)


def f_avg():
    return lambda Xtr, ytr, Xte, tdel=None: 0.5 * (hz.fit_p6(Xtr, ytr, Xte) + _hgb(Xtr, ytr, Xte))


# ------------------------------------------------------------------ candidates (frozen in the protocol)
def build_features(c):
    X40 = c.X40.astype(np.float64)
    # C2: causal patient-relative deltas of the 19 raw descriptors (expanding mean over the patient's own past, incl. current)
    raw = X40[:, :19]; delta = np.zeros_like(raw)
    for p in c.all_pids:
        idx = c.win_idx[p]; o = idx[np.argsort(c.df["start_sample"].values[idx])]
        cm = np.cumsum(raw[o], axis=0) / np.arange(1, len(o) + 1)[:, None]
        delta[o] = raw[o] - cm
    c5 = np.load(os.path.join(OUT, "c5_features.npz"), allow_pickle=True)
    F5, names = c5["F"], list(c5["names"])
    sel = lambda pre: [i for i, n in enumerate(names) if n.startswith(pre)]
    return {"X40": X40, "C2": np.hstack([X40, delta]), "C5a": np.hstack([X40, F5[:, sel("c5a_")]]),
            "C5b": np.hstack([X40, F5[:, sel("c5b_")]]), "C5c": np.hstack([X40, F5[:, sel("c5c_")]])}


def candidates(feats):
    lr_grid = lambda: [(f"C={C}", f_lr(C)) for C in (0.01, 0.05, 0.2)]
    return {
        "C1": (feats["X40"], [(f"tau={t}", f_lr(0.05, tau=t)) for t in (20, 40, 80)]),
        "C2": (feats["C2"], lr_grid()),
        "C3": (feats["X40"], [(f"EN l1={l} C={C}", f_en(l, C)) for l in (0.3, 0.7) for C in (0.05, 0.2)] + [("HGB", f_hgb()), ("LR+HGB", f_avg())]),
        "C5a": (feats["C5a"], lr_grid()), "C5b": (feats["C5b"], lr_grid()), "C5c": (feats["C5c"], lr_grid()),
    }


# ------------------------------------------------------------------ nested selection
def prs_criterion(c, pred, pids):
    y = np.array([c.y[p] for p in pids])
    sc = hz.horizon_scores(pids, hz.window_sequences(c, pred, pids), hz.tdel_by_pid(c, pids), None, None)
    return hz.m_of(y, sc)


def select_and_predict(c, X, universe, assign, configs, offset):
    """Outer 5-fold with inner 4-fold selection of the config; returns window OOF scores and chosen config per fold."""
    fold_w = np.full(len(c.pid_w), -1, dtype=int)
    for p in universe: fold_w[c.win_idx[p]] = assign[p]
    pred = np.zeros(len(c.pid_w)); chosen = []
    for f in range(5):
        te_p = [p for p in universe if assign[p] == f]; tr_p = np.array(sorted(p for p in universe if assign[p] != f))
        ytr = np.array([c.y[p] for p in tr_p])
        scores = []
        if len(configs) > 1:
            inner = StratifiedKFold(4, shuffle=True, random_state=42 + f + offset)
            inner_w = np.full(len(c.pid_w), -1, dtype=int)
            for k, (_, v) in enumerate(inner.split(tr_p, ytr)):
                for p in tr_p[v]: inner_w[c.win_idx[p]] = k
            for name, fit in configs:
                ip = np.zeros(len(c.pid_w))
                for k in range(4):
                    va = inner_w == k; tr = (inner_w >= 0) & (inner_w != k)
                    ip[va] = fit(X[tr], c.y_w[tr], X[va], tdel=c.tdel_w[tr])
                scores.append(prs_criterion(c, ip, list(tr_p)))
            best = int(np.argmax(scores))
        else:
            best = 0
        tr = (fold_w >= 0) & (fold_w != f); te = fold_w == f
        pred[te] = configs[best][1](X[tr], c.y_w[tr], X[te], tdel=c.tdel_w[tr])
        chosen.append(configs[best][0])
    return pred, chosen


# ------------------------------------------------------------------ baseline sequences (cached; also used by C4)
def b0_sequences(c, y464, tdel464, splits):
    path = os.path.join(OUT, "stage1_b0_seqs.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    base = np.load(os.path.join(OUT, "stage0_baseline_D464_scores.npz")); out = {}
    for name, assign in splits.items():
        off = 0 if name == "CANONICAL" else 100 + int(name.split("_")[1])
        pw = hz.oof_window_scores(c, c.X40, c.d464, assign)
        seq, _ = hz.pooler_oof_sequences(c, pw, c.d464, assign, offset=off)
        r, _ = hz.urm_cv(c, c.d464, assign, seq, tdel464)
        assert np.abs(r["URM"] - base[name]).max() < 1e-6, f"B0 cache does not reproduce Stage 0 for {name}"
        out[name] = seq
        print(f"  B0 sequences cached for {name} (reproduces Stage 0 exactly)", flush=True)
    pickle.dump(out, open(path, "wb"))
    return out


def compare(y, base_scores, cand_scores, prs_base, prs_cand):
    sc = {"base": base_scores, "cand": cand_scores}
    boot = hz.boot_all(y, sc)
    rec = hz.summarize_delta(boot, "cand", "base", y, sc)
    rec["M_cand"] = hz.m_of(y, cand_scores); rec["M_base"] = hz.m_of(y, base_scores)
    rec["M_PRS_cand"] = hz.m_of(y, prs_cand); rec["M_PRS_base"] = hz.m_of(y, prs_base)
    return rec


def holm(pvals):
    order = np.argsort(pvals); adj = np.zeros(len(pvals)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (HOLM_K - rank) * pvals[i])); adj[i] = run
    return adj


def adopt_rule(can, res, adj_p=None, p_thr=0.05):
    """can: canonical row dict; res: DataFrame of 5 resplit rows."""
    a1 = bool(can["dM"] >= 0.005 and (adj_p if adj_p is not None else can["dM_p"]) < p_thr)
    n_pos = int((res.dM > 0).sum()); a2 = n_pos == 5
    can_min = min(can[f"d{h}"] for h in hz.H); res_min = min(res[f"d{h}"].mean() for h in hz.H)
    a3 = bool(can_min >= -0.020 and res_min >= -0.020)
    if a1 and a2 and a3: tier = "ADOPT"
    elif can["dM"] > 0 and n_pos >= 4: tier = "SUGGESTIVE"
    else: tier = "NOT SUPPORTED"
    return {"tier": tier, "A1": a1, "A2": a2, "A3": a3, "resplits_positive": n_pos, "canonical_worst_horizon": round(can_min, 4),
            "resplit_mean_worst_horizon": round(float(res_min), 4)}


# ------------------------------------------------------------------ window candidates
def run_window(c, only=None):
    y464 = np.array([c.y[p] for p in c.d464]); tdel464 = hz.tdel_by_pid(c, c.d464)
    splits = hz.split_assignments(c, c.d464)
    base = np.load(os.path.join(OUT, "stage0_baseline_D464_scores.npz"))
    feats = build_features(c); cands = candidates(feats)
    path = os.path.join(OUT, "stage1_window_results.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {(r["cand"], r["split"]) for r in rows}
    for cand, (X, configs) in cands.items():
        if only and cand not in only: continue
        for name, assign in splits.items():
            if (cand, name) in done: continue
            off = 0 if name == "CANONICAL" else 100 + int(name.split("_")[1])
            pw, chosen = select_and_predict(c, X, c.d464, assign, configs, off)
            seq, _ = hz.pooler_oof_sequences(c, pw, c.d464, assign, offset=off)
            r, al = hz.urm_cv(c, c.d464, assign, seq, tdel464)
            prs_c = hz.horizon_scores(c.d464, hz.window_sequences(c, pw, c.d464), tdel464, None, None)
            pw_b = hz.oof_window_scores(c, c.X40, c.d464, assign)
            prs_b = hz.horizon_scores(c.d464, hz.window_sequences(c, pw_b, c.d464), tdel464, None, None)
            rec = {"cand": cand, "split": name, "chosen": "|".join(chosen), "alpha_mean": round(float(np.mean(al)), 3)}
            rec.update(compare(y464, base[name], r["URM"], prs_b, prs_c))
            rows.append(rec); pd.DataFrame(rows).to_csv(path, index=False)
            np.save(os.path.join(OUT, f"stage1_scores_{cand}_{name}.npy"), r["URM"])
            print(f"  {cand:<4}{name:<12} dM={rec['dM']:+.4f} (p={rec['dM_p']:.3f})  M {rec['M_base']:.4f}->{rec['M_cand']:.4f}  "
                  f"PRS-only M {rec['M_PRS_base']:.4f}->{rec['M_PRS_cand']:.4f}  chosen: {chosen}", flush=True)
    verdict_window(rows)


def verdict_window(rows):
    df = pd.DataFrame(rows); out = {}
    cands = [k for k in ["C1", "C2", "C3", "C5a", "C5b", "C5c"] if (df.cand == k).any()]
    can = {k: df[(df.cand == k) & (df.split == "CANONICAL")].iloc[0].to_dict() for k in cands}
    adj = dict(zip(cands, holm(np.array([can[k]["dM_p"] for k in cands]) )))
    # unrun candidates keep the family at K = 6 (Holm uses HOLM_K, not len(cands))
    for k in cands:
        res = df[(df.cand == k) & (df.split != "CANONICAL")]
        if len(res) < 5: continue
        out[k] = {**adopt_rule(can[k], res, adj_p=adj[k]), "canonical_dM": round(can[k]["dM"], 4), "canonical_p": round(can[k]["dM_p"], 4),
                  "holm_adjusted_p": round(float(adj[k]), 4), "resplit_dM": [round(v, 4) for v in res.dM]}
        print(f"  {k}: {out[k]['tier']}  dM={out[k]['canonical_dM']:+.4f} p={out[k]['canonical_p']} holm={out[k]['holm_adjusted_p']} "
              f"resplits+={out[k]['resplits_positive']}/5")
    json.dump(out, open(os.path.join(OUT, "stage1_window_verdict.json"), "w"), indent=2)


# ------------------------------------------------------------------ C4: late-fusion covariates
COVARIATE_SETS = {
    "S0": ["parity"],
    "S1": ["parity", "age", "gravidity", "gest. weeks"],
    "S2": ["parity", "age", "gravidity", "gest. weeks", "any_risk", "liq. praecox", "meconium"],
    "S3": ["parity", "age", "gravidity", "gest. weeks", "any_risk", "liq. praecox", "meconium", "induced", "presentation_nonvertex"],
}


def covariate_table(c):
    import pandas as pd
    m = pd.read_csv(hz.METADATA_PATH).set_index("record_id")
    t = pd.DataFrame(index=c.all_pids)
    for v in ["parity", "age", "gravidity", "gest. weeks", "liq. praecox", "meconium", "induced"]:
        t[v] = [float(m.loc[int(p), v]) for p in c.all_pids]
    t["any_risk"] = [float(max(m.loc[int(p), k] for k in ("diabetes", "hypertension", "preeclampsia", "pyrexia")) > 0) for p in c.all_pids]
    pres = [m.loc[int(p), "presentation"] for p in c.all_pids]
    t["presentation_nonvertex"] = [np.nan if pd.isna(v) else float(v != 1) for v in pres]
    return t


def make_parity_fn(c, tab, cols, fixed_C=None):
    Xall = tab[cols].values.astype(float)
    def fn(tr, order):
        ix = {p: i for i, p in enumerate(c.all_pids)}
        Xtr = Xall[[ix[p] for p in tr]]; ytr = np.array([c.y[p] for p in tr])
        med = np.nanmedian(Xtr, axis=0)
        fill = lambda A: np.where(np.isnan(A), med, A)
        Xtr = fill(Xtr); Xo = fill(Xall[[ix[p] for p in order]])
        if fixed_C is None and len(cols) > 1:
            best, bs = 1.0, -1
            skf = StratifiedKFold(4, shuffle=True, random_state=42)
            for C in (0.1, 1.0):
                oof = np.zeros(len(tr))
                for a, b in skf.split(Xtr, ytr):
                    sc = StandardScaler().fit(Xtr[a])
                    oof[b] = LogisticRegression(C=C, max_iter=1000, random_state=42).fit(sc.transform(Xtr[a]), ytr[a]).predict_proba(sc.transform(Xtr[b]))[:, 1]
                s = roc_auc_score(ytr, oof)
                if s > bs: bs, best = s, C
        else:
            best = 1.0 if fixed_C is None else fixed_C
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=best, max_iter=1000, random_state=42).fit(sc.transform(Xtr), ytr)
        return dict(zip(order, clf.predict_proba(sc.transform(Xo))[:, 1]))
    return fn


def run_c4(c):
    y464 = np.array([c.y[p] for p in c.d464]); tdel464 = hz.tdel_by_pid(c, c.d464)
    splits = hz.split_assignments(c, c.d464)
    b0seq = b0_sequences(c, y464, tdel464, splits)
    tab = covariate_table(c)
    prev = "S0"; results = {}; rows = []
    for cur in ["S1", "S2", "S3"]:
        sc_prev, sc_cur = {}, {}
        for name, assign in splits.items():
            sc_prev[name] = hz.urm_cv(c, c.d464, assign, b0seq[name], tdel464, parity_fn=make_parity_fn(c, tab, COVARIATE_SETS[prev], fixed_C=1.0 if prev == "S0" else None))[0]["URM"]
            r, al = hz.urm_cv(c, c.d464, assign, b0seq[name], tdel464, parity_fn=make_parity_fn(c, tab, COVARIATE_SETS[cur]))
            sc_cur[name] = r["URM"]
            rec = {"set": cur, "vs": prev, "split": name, "alpha_mean": round(float(np.mean(al)), 3)}
            rec.update(compare(y464, sc_prev[name], sc_cur[name], sc_prev[name], sc_cur[name]))
            rows.append(rec)
            print(f"  {cur} vs {prev} {name:<12} dM={rec['dM']:+.4f} (p={rec['dM_p']:.3f})  M {rec['M_base']:.4f}->{rec['M_cand']:.4f}", flush=True)
        df = pd.DataFrame(rows); d = df[(df.set == cur)]
        can = d[d.split == "CANONICAL"].iloc[0].to_dict(); res = d[d.split != "CANONICAL"]
        v = adopt_rule(can, res, adj_p=None); v.update({"canonical_dM": round(can["dM"], 4), "canonical_p": round(can["dM_p"], 4)})
        results[cur] = v; print(f"  => {cur} vs {prev}: {v['tier']}  {v}")
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "stage1_c4_results.csv"), index=False)
        if v["tier"] != "ADOPT":
            print(f"  fixed-sequence stops: {cur} not adopted, later sets not tested"); break
        prev = cur
    json.dump(results, open(os.path.join(OUT, "stage1_c4_verdict.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", choices=["window", "c4"], required=True); ap.add_argument("--only", default="")
    a = ap.parse_args()
    c = hz.load_ctx()
    if a.part == "window":
        run_window(c, only=[s for s in a.only.split(",") if s])
    else:
        run_c4(c)


if __name__ == "__main__":
    main()
