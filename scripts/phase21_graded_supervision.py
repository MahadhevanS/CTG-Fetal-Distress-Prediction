"""
Phase 21 -- graded-outcome supervision of the window model (docs/phase21_graded_supervision_protocol.md).
Candidates G-A (ridge on -pH), G-B (multi-threshold logistic mean), G-C (soft-label logistic) + control B0-P (binary label + same Platt step),
all on the locked 40-D features, evaluated on the unchanged pH<=7.15 endpoint through the Phase 20 harness on D464.

  --stage screen   canonical + 5 resplits, Holm K=3 decision rule (section 4)
  --stage confirm  fresh resplits 66/77/88/99/111 for candidates that passed screening (section 5)
"""
import os, sys, json, argparse, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import scripts.phase20_harness as hz
import scripts.phase20_stage1_candidates as s1

warnings.filterwarnings("ignore")
s1.HOLM_K = 3
OUT = "results/phase21_graded"; os.makedirs(OUT, exist_ok=True)
TAUS = (7.20, 7.15, 7.10, 7.05)
SOFT_SCALE = 0.03


def _split(X):                      # last column carries the (training-only) pH; never used on held-out rows
    return X[:, :-1], X[:, -1]


def platt(raw_tr, y_tr, raw_te):
    mu, sd = raw_tr.mean(), raw_tr.std() + 1e-12
    m = LogisticRegression(C=1e6, max_iter=1000).fit(((raw_tr - mu) / sd)[:, None], y_tr)
    return m.predict_proba(((raw_te - mu) / sd)[:, None])[:, 1]


def f_b0p():
    def fit(Xtr, ytr, Xte, tdel=None):
        F, _ = _split(Xtr); sc = StandardScaler().fit(F)
        m = LogisticRegression(C=0.05, max_iter=1000, random_state=42).fit(sc.transform(F), ytr)
        return platt(m.decision_function(sc.transform(F)), ytr, m.decision_function(sc.transform(Xte[:, :-1])))
    return fit


def f_ridge(alpha):
    def fit(Xtr, ytr, Xte, tdel=None):
        F, ph = _split(Xtr); sc = StandardScaler().fit(F); A = sc.transform(F)
        m = Ridge(alpha=alpha).fit(A, -(ph - ph.mean()) / ph.std())
        return platt(m.predict(A), ytr, m.predict(sc.transform(Xte[:, :-1])))
    return fit


def f_multi(C):
    def fit(Xtr, ytr, Xte, tdel=None):
        F, ph = _split(Xtr); sc = StandardScaler().fit(F); A = sc.transform(F); B = sc.transform(Xte[:, :-1])
        p_tr = np.zeros(len(A)); p_te = np.zeros(len(B))
        for t in TAUS:
            m = LogisticRegression(C=C, max_iter=1000, random_state=42).fit(A, (ph <= t).astype(int))
            p_tr += m.predict_proba(A)[:, 1] / len(TAUS); p_te += m.predict_proba(B)[:, 1] / len(TAUS)
        return platt(p_tr, ytr, p_te)
    return fit


def f_soft(C):
    def fit(Xtr, ytr, Xte, tdel=None):
        F, ph = _split(Xtr); sc = StandardScaler().fit(F); A = sc.transform(F)
        q = 1.0 / (1.0 + np.exp(-(7.15 - ph) / SOFT_SCALE)); n = len(A)
        m = LogisticRegression(C=C, max_iter=1000, random_state=42)
        m.fit(np.vstack([A, A]), np.r_[np.ones(n), np.zeros(n)], sample_weight=np.r_[q, 1 - q])
        return platt(m.decision_function(A), ytr, m.decision_function(sc.transform(Xte[:, :-1])))
    return fit


def candidates():
    grid = lambda f, name, vals: [(f"{name}={v}", f(v)) for v in vals]
    return {"B0P": [("fixed", f_b0p())], "G-A": grid(f_ridge, "alpha", (10, 100, 1000)),
            "G-B": grid(f_multi, "C", (0.01, 0.05, 0.2)), "G-C": grid(f_soft, "C", (0.01, 0.05, 0.2))}


def extra_candidates():
    return {"G-A-wide": [(f"alpha={a}", f_ridge(a)) for a in (100, 1000, 10000, 100000)]}


def aug_features(c):
    meta = pd.read_csv(hz.METADATA_PATH).set_index("record_id")
    ph_p = {p: float(meta.loc[int(p), "ph"]) for p in c.all_pids}
    ph_w = np.array([ph_p[p] for p in c.pid_w])
    return np.hstack([c.X40.astype(np.float64), ph_w[:, None]])


def run_cand(c, X, cand, configs, name, assign, tdel464, y464):
    off = 0 if name == "CANONICAL" else 100 + int(name.split("_")[1])
    pw, chosen = s1.select_and_predict(c, X, c.d464, assign, configs, off)
    seq, _ = hz.pooler_oof_sequences(c, pw, c.d464, assign, offset=off)
    r, al = hz.urm_cv(c, c.d464, assign, seq, tdel464)
    prs = hz.horizon_scores(c.d464, hz.window_sequences(c, pw, c.d464), tdel464, None, None)
    return r["URM"], prs, chosen, al


def screen(c):
    y464 = np.array([c.y[p] for p in c.d464]); tdel464 = hz.tdel_by_pid(c, c.d464)
    splits = hz.split_assignments(c, c.d464); base = np.load(os.path.join(hz.OUT_DIR, "stage0_baseline_D464_scores.npz"))
    X = aug_features(c); path = os.path.join(OUT, "screen_results.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {(r["cand"], r["split"]) for r in rows}
    for cand, configs in candidates().items():
        for name, assign in splits.items():
            fn = os.path.join(OUT, f"scores_{cand}_{name}.npy")
            if (cand, name) in done and os.path.exists(fn): continue
            urm, prs, chosen, al = run_cand(c, X, cand, configs, name, assign, tdel464, y464)
            np.save(fn, urm)
            pw_b = hz.oof_window_scores(c, c.X40, c.d464, assign)
            prs_b = hz.horizon_scores(c.d464, hz.window_sequences(c, pw_b, c.d464), tdel464, None, None)
            rec = {"cand": cand, "split": name, "chosen": "|".join(chosen), "alpha_mean": round(float(np.mean(al)), 3)}
            rec.update(s1.compare(y464, base[name], urm, prs_b, prs))
            rows.append(rec); pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  {cand:<4}{name:<12} dM vs B0 {rec['dM']:+.4f} (p={rec['dM_p']:.3f})  M {rec['M_base']:.4f}->{rec['M_cand']:.4f}  "
                  f"PRS-only M {rec['M_PRS_base']:.4f}->{rec['M_PRS_cand']:.4f}  chosen {chosen}", flush=True)
    verdict(c, y464, splits, rows)


def verdict(c, y464, splits, rows):
    df = pd.DataFrame(rows); out = {}; cands = ["G-A", "G-B", "G-C"]
    can = {k: df[(df.cand == k) & (df.split == "CANONICAL")].iloc[0].to_dict() for k in cands}
    adj = dict(zip(cands, s1.holm(np.array([can[k]["dM_p"] for k in cands]))))
    for k in cands:
        res = df[(df.cand == k) & (df.split != "CANONICAL")]
        v = s1.adopt_rule(can[k], res, adj_p=adj[k])
        # A4: vs control B0-P (same Platt step, binary label)
        d_ctrl = []
        for name in splits:
            a = np.load(os.path.join(OUT, f"scores_{k}_{name}.npy")); b = np.load(os.path.join(OUT, f"scores_B0P_{name}.npy"))
            d_ctrl.append(hz.m_of(y464, a) - hz.m_of(y464, b))
        a4 = bool(d_ctrl[0] > 0 and sum(x > 0 for x in d_ctrl[1:]) >= 4)
        if v["tier"] == "ADOPT" and not a4: v["tier"] = "NOT SUPPORTED (gain not attributable to graded supervision: fails A4)"
        v.update({"A4": a4, "dM_vs_B0P": [round(x, 4) for x in d_ctrl], "canonical_dM": round(can[k]["dM"], 4), "canonical_p": round(can[k]["dM_p"], 4),
                  "holm_adjusted_p": round(float(adj[k]), 4), "resplit_dM": [round(x, 4) for x in res.dM]})
        out[k] = v; print(f"  {k}: {v['tier']}  dM={v['canonical_dM']:+.4f} p={v['canonical_p']} holm={v['holm_adjusted_p']} resplits+={v['resplits_positive']}/5  "
                          f"vs B0-P {v['dM_vs_B0P']}")
    cb = df[(df.cand == "B0P")]
    out["B0P_control_dM_vs_B0"] = [round(x, 4) for x in cb.dM]
    json.dump(out, open(os.path.join(OUT, "screen_verdict.json"), "w"), indent=2)


def confirm(c, cand_list):
    y464 = np.array([c.y[p] for p in c.d464]); tdel464 = hz.tdel_by_pid(c, c.d464)
    splits = hz.split_assignments(c, c.d464, resplit_seeds=hz.FRESH_RESPLIT_SEEDS, canonical=False)
    X = aug_features(c); cands = {**candidates(), **extra_candidates()}; res = {}
    base_scores = {}
    for name, assign in splits.items():
        off = 100 + int(name.split("_")[1])
        pw = hz.oof_window_scores(c, c.X40, c.d464, assign); seq, _ = hz.pooler_oof_sequences(c, pw, c.d464, assign, offset=off)
        base_scores[name] = hz.urm_cv(c, c.d464, assign, seq, tdel464)[0]["URM"]
    for cand in cand_list:
        per = []; boots = []
        for name, assign in splits.items():
            urm, _, chosen, _ = run_cand(c, X, cand, cands[cand], name, assign, tdel464, y464)
            sc = {"base": base_scores[name], "cand": urm}; b = hz.boot_all(y464, sc); boots.append(b["cand"] - b["base"])
            d = [hz.fast_auc(y464, urm[:, a]) - hz.fast_auc(y464, base_scores[name][:, a]) for a in range(4)]
            per.append(d); print(f"  fresh {cand} {name}: dM={np.mean(d):+.4f}  per-horizon {np.round(d, 4).tolist()}", flush=True)
        per = np.array(per); dM = per.mean(1); pooled = np.mean([bb.mean(1) for bb in boots], axis=0)        # same resamples (seed 42) across resplits
        p = float(min(1.0, 2 * min(np.mean(pooled <= 0), np.mean(pooled >= 0))))
        ok = bool(dM.mean() >= 0.005 and p < 0.05 and (dM > 0).sum() >= 4 and per.mean(0).min() >= -0.020)
        res[cand] = {"fresh_dM": [round(x, 4) for x in dM], "mean_dM": round(float(dM.mean()), 4), "pooled_p": round(p, 4),
                     "resplits_positive": int((dM > 0).sum()), "worst_horizon_mean": round(float(per.mean(0).min()), 4), "CONFIRMED": ok}
        print("  ", cand, res[cand])
    json.dump(res, open(os.path.join(OUT, "confirm_verdict.json"), "w"), indent=2)


# ------------------------------------------------------------------ 21b exploratory follow-up
def final_window_scores(c, X, configs, test_pids, seed=777):
    """Config chosen by inner 4-fold CV on all of D464 (criterion as in screening), then fit on D464 windows, predict test windows."""
    from sklearn.model_selection import StratifiedKFold
    tr_p = np.array(sorted(c.d464)); ytr = np.array([c.y[p] for p in tr_p]); chosen = configs[0][0]
    if len(configs) > 1:
        inner_w = np.full(len(c.pid_w), -1, dtype=int)
        for k, (_, v) in enumerate(StratifiedKFold(4, shuffle=True, random_state=seed).split(tr_p, ytr)):
            for p in tr_p[v]: inner_w[c.win_idx[p]] = k
        sc = []
        for name, fit in configs:
            ip = np.zeros(len(c.pid_w))
            for k in range(4):
                va = inner_w == k; tr = (inner_w >= 0) & (inner_w != k)
                ip[va] = fit(X[tr], c.y_w[tr], X[va], tdel=c.tdel_w[tr])
            sc.append(s1.prs_criterion(c, ip, list(tr_p)))
        configs = [configs[int(np.argmax(sc))]]; chosen = configs[0][0]
    tr = np.isin(c.pid_w, c.d464); te = np.isin(c.pid_w, test_pids)
    pred = np.zeros(len(c.pid_w)); pred[te] = configs[0][1](X[tr], c.y_w[tr], X[te], tdel=c.tdel_w[tr])
    return pred, chosen


def test_arm(c, X, configs):
    """Train on all D464, score the 83 test patients through the full harness. configs=None -> baseline B0 (locked P6 recipe)."""
    import torch
    from scripts.phase16_temporal_attention_model import carve_inner_validation
    from scripts.phase19_pooler_experiment import build_seqs, make_inputs, pooled_all_prefix, fit_arm
    canon = hz.split_assignments(c, c.d464)["CANONICAL"]; tp = c.test_pids
    if configs is None:
        pw_cv = hz.oof_window_scores(c, c.X40, c.d464, canon)
        te = np.isin(c.pid_w, tp); tr = np.isin(c.pid_w, c.d464)
        pw_te = np.zeros(len(c.pid_w)); pw_te[te] = hz.fit_p6(c.X40[tr], c.y_w[tr], c.X40[te]); chosen = "P6"
    else:
        pw_cv, _ = s1.select_and_predict(c, X, c.d464, canon, configs, 0)
        pw_te, chosen = final_window_scores(c, X, configs, tp)
    seq_cv, S = hz.pooler_oof_sequences(c, pw_cv, c.d464, canon, offset=0)
    inner_tr, inner_val = carve_inner_validation(c.d464, c.y, seed=123)
    tr_idx = torch.tensor([S.row[p] for p in inner_tr]); val_idx = torch.tensor([S.row[p] for p in inner_val])
    Xtr = make_inputs(S, False)
    S_te = build_seqs(pw_te.astype(np.float32), c.pid_w, c.df, c.X40, c.col_idx, list(tp), c.y); Xte = make_inputs(S_te, False)
    Z = 0
    for seed in hz.SEEDS:
        sc, _, _ = fit_arm(S, Xtr, tr_idx, val_idx, "bce", seed)
        with torch.no_grad():
            Z = Z + pooled_all_prefix(sc, Xte, S_te.R, S_te.L) / len(hz.SEEDS)
    seq_te = {p: Z[S_te.row[p], :int(S_te.L[S_te.row[p]])].numpy().astype(np.float64) for p in tp}
    order = list(c.all_pids)
    fitted = hz.fit_parity(np.array([c.parity[p] for p in c.d464]), np.array([c.y[p] for p in c.d464]), np.array([c.parity[p] for p in order]))
    par = dict(zip(order, fitted)); alpha = hz.select_alpha(c.d464, seq_cv, par, c.y)
    return hz.horizon_scores(tp, seq_te, hz.tdel_by_pid(c, tp), alpha, par), alpha, chosen


def followup(c):
    print("=== 21b (1/2): fresh resplits ===", flush=True)
    confirm(c, ["G-A", "G-A-wide"])
    fr = json.load(open(os.path.join(OUT, "confirm_verdict.json")))
    print("\n=== 21b (2/2): test partition, opened once ===", flush=True)
    X = aug_features(c); ytest = np.array([c.y[p] for p in c.test_pids])
    arms = {"B0": test_arm(c, c.X40, None), "G-A": test_arm(c, X, candidates()["G-A"]), "G-A-wide": test_arm(c, X, extra_candidates()["G-A-wide"])}
    scores = {k: v[0] for k, v in arms.items()}; boot = hz.boot_all(ytest, scores)
    out = {"n_test": len(ytest), "n_pos_test": int(ytest.sum())}
    for k in arms:
        out[k] = {"M": round(hz.m_of(ytest, scores[k]), 4), "per_horizon": [round(v, 4) for v in hz.per_h(ytest, scores[k])], "alpha": arms[k][1], "window_config": arms[k][2]}
    for k in ("G-A", "G-A-wide"):
        d = hz.summarize_delta(boot, k, "B0", ytest, scores); out[k]["dM_vs_B0"] = round(d["dM"], 4); out[k]["dM_ci"] = [round(d["dM_ci_lo"], 4), round(d["dM_ci_hi"], 4)]; out[k]["dM_p"] = round(d["dM_p"], 4)
    try:
        ref = pd.read_csv("results/phase18_fusion_ablation/deployable_fusion_results.csv")
        r = ref[ref["model"] == "A3_selected_once"]; out["frozen_URM_test_auroc_reference"] = {str(row["horizon"]): row.get("test_auroc") for _, row in r.iterrows()}
    except Exception as e:
        out["frozen_URM_test_auroc_reference"] = f"unavailable ({e})"
    ga = fr["G-A"]
    label = bool(ga["mean_dM"] >= 0.005 and ga["resplits_positive"] >= 4 and ga["worst_horizon_mean"] >= -0.020 and out["G-A"]["dM_vs_B0"] > 0)
    out["label_primary_G-A"] = "EXPLORATORY SUPPORT" if label else "EXPLORATORY NO SUPPORT"
    json.dump(out, open(os.path.join(OUT, "followup_21b.json"), "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["screen", "confirm", "followup"], required=True); ap.add_argument("--cands", default="")
    a = ap.parse_args(); c = hz.load_ctx()
    if a.stage == "screen": screen(c)
    elif a.stage == "followup": followup(c)
    else: confirm(c, [s for s in a.cands.split(",") if s])


if __name__ == "__main__":
    main()
