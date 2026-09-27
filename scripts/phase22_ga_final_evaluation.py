"""
Phase 22 -- G-A final training and full evaluation (docs/phase22_ga_final_evaluation_spec.md, frozen before any metric here was computed).

Stages (each cached under results/phase22_ga_final/):
  cv547   canonical 5-fold CV over all 547 patients: G-A, baseline B0, control B0-P -> scores, sequences, metrics
  robust  10 resplits on 547, per-seed spread, fixed-ridge-alpha sensitivity
  test    frozen definition trained on D464, scored on the 83 test patients (must reproduce Phase 21b: G-A M 0.8073, B0 0.7582)
  freeze  final model trained on all 547 + standalone scorer integrity check + interpretability
  report  assemble the markdown report
"""
import os, sys, json, argparse, pickle, warnings
import numpy as np, pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import scripts.phase20_harness as hz
import scripts.phase20_stage1_candidates as s1
import scripts.phase21_graded_supervision as p21
from scripts.delong_test import delong_roc_test
from src.models.phase16_causal_attention import AttentionScorer, predict_all_prefixes, eligible_prefix_length
from scripts.phase16_temporal_attention_model import carve_inner_validation
from scripts.phase19_pooler_experiment import build_seqs, make_inputs, fit_arm

warnings.filterwarnings("ignore")
OUT = "results/phase22_ga_final"; os.makedirs(OUT, exist_ok=True)
H = hz.H; HGRID = list(range(0, 45, 5))
FROZEN_URM_CV = {0: 0.7335, 10: 0.6921, 20: 0.6638, 30: 0.6631}
FROZEN_URM_TEST = {0: 0.7433, 10: 0.6988, 20: 0.7558, 30: 0.7754}
FROZEN_URM_OPS = {"achieved_sensitivity": 0.8636, "false_alert_rate": 0.6247, "median_lead": 40.0, "ge20": 0.8105, "ge30": 0.6947}
B = 2000


# ------------------------------------------------------------------ metric helpers
def ap(y, s): return float(average_precision_score(y, s))
def brier(y, s): return float(brier_score_loss(y, np.clip(s, 0, 1)))


def calib(y, s):
    s = np.clip(s, 1e-6, 1 - 1e-6); lg = np.log(s / (1 - s))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lg[:, None], y)
    edges = np.linspace(0, 1, 11); b = np.clip(np.digitize(s, edges) - 1, 0, 9); ece = 0.0; rel = []
    for k in range(10):
        mk = b == k
        if mk.sum():
            ece += mk.mean() * abs(y[mk].mean() - s[mk].mean()); rel.append({"bin": f"{edges[k]:.1f}-{edges[k + 1]:.1f}", "n": int(mk.sum()), "mean_pred": round(float(s[mk].mean()), 3), "obs_rate": round(float(y[mk].mean()), 3)})
    return {"brier": round(brier(y, s), 4), "cal_intercept": round(float(m.intercept_[0]), 3), "cal_slope": round(float(m.coef_[0][0]), 3),
            "mean_pred": round(float(s.mean()), 3), "prevalence": round(float(y.mean()), 3), "ece10": round(float(ece), 4)}, rel


def sens_at_spec(y, s, spec):
    thr = np.quantile(s[y == 0], spec); return float(np.mean(s[y == 1] > thr))


def boot_metrics(y, scores, B=B, seed=42):
    """scores: name -> (n,4). Returns per name/horizon percentile CIs for AUROC, AUPRC, Brier, sens@80/90 spec + paired dAUROC vs first arm."""
    rng = np.random.default_rng(seed); n = len(y); names = list(scores); acc = {}
    for _ in range(B):
        i = rng.choice(n, n, replace=True); yb = y[i]
        if yb.sum() in (0, n): continue
        for nm in names:
            for a in range(4):
                sb = scores[nm][i, a]
                acc.setdefault((nm, a, "auroc"), []).append(hz.fast_auc(yb, sb)); acc.setdefault((nm, a, "auprc"), []).append(ap(yb, sb))
                acc.setdefault((nm, a, "brier"), []).append(brier(yb, sb))
                acc.setdefault((nm, a, "sens80"), []).append(sens_at_spec(yb, sb, 0.8)); acc.setdefault((nm, a, "sens90"), []).append(sens_at_spec(yb, sb, 0.9))
    ci = lambda v: (round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4))
    return {k: ci(v) for k, v in acc.items()}, acc


def metric_table(y, scores, arms_order):
    ci, acc = boot_metrics(y, scores); rows = []
    for nm in arms_order:
        for a, h in enumerate(H):
            s = scores[nm][:, a]
            r = {"arm": nm, "horizon": h, "auroc": round(hz.fast_auc(y, s), 4), "auroc_ci": ci[(nm, a, "auroc")], "auprc": round(ap(y, s), 4), "auprc_ci": ci[(nm, a, "auprc")],
                 "sens_at_80spec": round(sens_at_spec(y, s, 0.8), 3), "sens_at_90spec": round(sens_at_spec(y, s, 0.9), 3)}
            c_, _ = calib(y, s); r.update({k: c_[k] for k in ("brier", "cal_intercept", "cal_slope", "ece10")}); r["brier_ci"] = ci[(nm, a, "brier")]
            rows.append(r)
    return pd.DataFrame(rows), acc


# ------------------------------------------------------------------ arms
def run_arm(c, arm, universe, assign, offset, Xaug, seeds=hz.SEEDS, configs=None, pw=None):
    if pw is None:
        if arm == "B0": pw = hz.oof_window_scores(c, c.X40, universe, assign); chosen = None
        else:
            cf = configs or (p21.candidates()["G-A"] if arm == "GA" else p21.candidates()["B0P"])
            pw, chosen = s1.select_and_predict(c, Xaug, universe, assign, cf, offset)
    else:
        chosen = None
    seq, _ = hz.pooler_oof_sequences(c, pw, universe, assign, seeds=seeds, offset=offset)
    tdel = hz.tdel_by_pid(c, universe)
    r, al = hz.urm_cv(c, universe, assign, seq, tdel)
    prs = hz.horizon_scores(universe, hz.window_sequences(c, pw, universe), tdel, None, None)
    return dict(URM=r["URM"], TAM=r["CTG"], PARITY=r["PARITY"], PRS=prs, seq=seq, pw=pw, alphas=al, chosen=chosen)


def urm_cv_grid(c, universe, assign, seq, tdel, hgrid):
    order = list(universe); row = {p: i for i, p in enumerate(order)}; out = np.zeros((len(order), len(hgrid)))
    for f in range(5):
        te = [p for p in order if assign[p] == f]; tr = [p for p in order if assign[p] != f]
        par = dict(zip(order, hz.fit_parity(np.array([c.parity[p] for p in tr]), np.array([c.y[p] for p in tr]), np.array([c.parity[p] for p in order]))))
        a = hz.select_alpha(tr, seq, par, c.y)
        for p in te:
            for j, h in enumerate(hgrid):
                k = eligible_prefix_length(tdel[p], h, len(seq[p])) - 1; out[row[p], j] = a * seq[p][k] + (1 - a) * par[p]
    return out


def operational(c, universe, assign, seq, target=0.8):
    order = list(universe); n = len(order); y = np.array([c.y[p] for p in order]); tdel = hz.tdel_by_pid(c, order)
    alerted = np.zeros(n, bool); lead = np.full(n, np.nan)
    for f in range(5):
        te = [p for p in order if assign[p] == f]; tr = [p for p in order if assign[p] != f]
        par = dict(zip(order, hz.fit_parity(np.array([c.parity[p] for p in tr]), np.array([c.y[p] for p in tr]), np.array([c.parity[p] for p in order]))))
        a = hz.select_alpha(tr, seq, par, c.y)
        trs = np.array([a * seq[p][-1] + (1 - a) * par[p] for p in tr]); ytr = np.array([c.y[p] for p in tr])
        thr = float(np.percentile(trs[ytr == 1], (1 - target) * 100))
        for p in te:
            i = order.index(p); idx = np.where(a * seq[p] + (1 - a) * par[p] >= thr)[0]
            if len(idx):
                alerted[i] = True
                if y[i] == 1: lead[i] = tdel[p][idx[0]]
    def summ(ix):
        yy, al, ld = y[ix], alerted[ix], lead[ix]; tp = int((al & (yy == 1)).sum()); fp = int((al & (yy == 0)).sum()); l = ld[(yy == 1) & ~np.isnan(ld)]
        return {"sens": tp / max((yy == 1).sum(), 1), "far": fp / max((yy == 0).sum(), 1), "ppv": tp / max(tp + fp, 1), "median_lead": float(np.median(l)) if len(l) else np.nan,
                "iqr_lead": float(np.percentile(l, 75) - np.percentile(l, 25)) if len(l) else np.nan, "ge10": float(np.mean(l >= 10)) if len(l) else np.nan,
                "ge20": float(np.mean(l >= 20)) if len(l) else np.nan, "ge30": float(np.mean(l >= 30)) if len(l) else np.nan}
    point = summ(np.arange(n)); rng = np.random.default_rng(42); bs = {k: [] for k in point}
    for _ in range(B):
        ix = rng.choice(n, n, replace=True)
        if y[ix].sum() == 0: continue
        for k, v in summ(ix).items(): bs[k].append(v)
    return {k: {"value": round(float(point[k]), 4), "ci": (round(float(np.nanpercentile(bs[k], 2.5)), 4), round(float(np.nanpercentile(bs[k], 97.5)), 4))} for k in point}


def decision_curve(y, s):
    pts = np.round(np.arange(0.10, 0.51, 0.05), 2); prev = y.mean(); rows = []
    for pt in pts:
        pred = s >= pt; tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); w = pt / (1 - pt)
        rows.append({"threshold": float(pt), "net_benefit": round((tp - fp * w) / len(y), 4), "treat_all": round(prev - (1 - prev) * w, 4)})
    return rows


# ------------------------------------------------------------------ stage: cv547
def stage_cv547(c):
    path = os.path.join(OUT, "cv547.pkl")
    if os.path.exists(path): return pickle.load(open(path, "rb"))
    universe = c.all_pids; Xaug = p21.aug_features(c); res = {}
    for arm in ("B0", "GA", "B0P"):
        res[arm] = run_arm(c, arm, universe, c.canon, 0, Xaug); print(f"  cv547 {arm}: M_URM {hz.m_of(np.array([c.y[p] for p in universe]), res[arm]['URM']):.4f}", flush=True)
    pickle.dump(res, open(path, "wb")); return res


def analyse_cv547(c, res):
    universe = c.all_pids; y = np.array([c.y[p] for p in universe]); out = {}
    # 1/2 discrimination + paired
    scores = {"GA": res["GA"]["URM"], "B0": res["B0"]["URM"]}
    tab, acc = metric_table(y, scores, ["GA", "B0"]); tab.to_csv(os.path.join(OUT, "cv547_metrics.csv"), index=False); out["table"] = tab.to_dict("records")
    out["M"] = {k: round(hz.m_of(y, v), 4) for k, v in scores.items()}
    bt = hz.boot_all(y, scores); d = hz.summarize_delta(bt, "GA", "B0", y, scores); out["paired_vs_B0"] = {k: round(float(v), 4) for k, v in d.items()}
    out["delong_delivery"] = dict(zip(("p", "auc_B0", "auc_GA"), [float(v) for v in delong_roc_test(y, scores["B0"][:, 0], scores["GA"][:, 0])]))
    # time course
    tdel = hz.tdel_by_pid(c, universe); curves = {}
    for arm in ("GA", "B0"):
        g = urm_cv_grid(c, universe, c.canon, res[arm]["seq"], tdel, HGRID); curves[arm] = [round(hz.fast_auc(y, g[:, j]), 4) for j in range(len(HGRID))]
    out["auc_vs_lead_time"] = {"horizons": HGRID, **curves}
    # calibration reliability at delivery
    out["calibration"] = {arm: dict(zip(("summary", "reliability"), calib(y, res[arm]["URM"][:, 0]))) for arm in ("GA", "B0")}
    # 4 operating points
    out["operational"] = {arm: operational(c, universe, c.canon, res[arm]["seq"]) for arm in ("GA", "B0")}
    out["decision_curve"] = {arm: decision_curve(y, res[arm]["URM"][:, 0]) for arm in ("GA", "B0")}
    # ablation
    abl = []
    for arm, lab in (("GA", "G-A"), ("B0", "Baseline B0"), ("B0P", "Control B0-P")):
        for comp, nm in (("PARITY", "parity only (MCM)"), ("PRS", "window only (latest window)"), ("TAM", "TAM only (pooled window scores)"), ("URM", "fused (URM form)")):
            if comp == "PARITY" and arm != "B0": continue
            s = res[arm][comp]; abl.append({"model": lab, "component": nm, **{f"{h}m": round(hz.fast_auc(y, s[:, a]), 4) for a, h in enumerate(H)}, "M": round(hz.m_of(y, s), 4)})
    out["ablation"] = abl
    # 8 subgroups
    meta = pd.read_csv(hz.METADATA_PATH).set_index("record_id")
    ph = np.array([meta.loc[int(p), "ph"] for p in universe]); bd = np.array([meta.loc[int(p), "bdecf"] for p in universe], float)
    dt = np.array([meta.loc[int(p), "deliv. type"] for p in universe]); par = np.array([c.parity[p] for p in universe])
    ds = {}
    for sp in ("train", "val", "test"):
        dd = torch.load(os.path.join(hz.DATA_DIR, f"{sp}_dataset.pt"), weights_only=False)
        for m_, w in zip(dd["metadata"], dd["w_is_second_stage"].numpy()): ds[(str(m_[0]), int(m_[1]))] = float(w)
    ss = c.df["start_sample"].values
    last2 = np.array([ds[(p, int(ss[c.win_idx[p][np.argmax(ss[c.win_idx[p]])]]))] for p in universe]); nwin = np.array([len(c.win_idx[p]) for p in universe])
    rng = np.random.default_rng(3)
    def auc_ci(yy, s):
        v = []
        for _ in range(1000):
            i = rng.choice(len(yy), len(yy), replace=True)
            if 0 < yy[i].sum() < len(yy): v.append(hz.fast_auc(yy[i], s[i]))
        return round(hz.fast_auc(yy, s), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)
    rows = []
    def sub(name, mask, target=None, tmask=None):
        yy = (y if target is None else target)[mask]; row = {"subgroup": name, "n": int(mask.sum()), "n_pos": int(yy.sum())}
        for arm in ("GA", "B0"):
            for a, h in ((0, "delivery"), (3, "ge30m")):
                if yy.sum() < 8 or (len(yy) - yy.sum()) < 8: row[f"{arm}_{h}"] = None
                else: row[f"{arm}_{h}"] = auc_ci(yy, res[arm]["URM"][mask, a])
        rows.append(row)
    sub("all 547", np.ones(len(y), bool)); sub("vaginal delivery", dt == 1); sub("caesarean", dt == 2); sub("primiparous", par == 0); sub("multiparous", par >= 1)
    sub("last window in 1st stage", last2 == 0); sub("last window in 2nd stage", last2 == 1)
    sub("short recording (<=12 windows)", nwin <= 12); sub("long recording (>=13 windows)", nwin >= 13)
    for nm, tgt, mk in (("severe: pH<=7.05 vs pH>7.20", (ph <= 7.05).astype(int), (ph <= 7.05) | (ph > 7.20)), ("BDecf>=8 vs pH>7.20", (np.nan_to_num(bd) >= 8).astype(int), ~np.isnan(bd) & ((np.nan_to_num(bd) >= 8) | (ph > 7.20))),
                        ("borderline pH 7.10-7.15 vs pH>7.20", ((ph > 7.10) & (ph <= 7.15)).astype(int), ((ph > 7.10) & (ph <= 7.15)) | (ph > 7.20))):
        sub(nm, mk, target=tgt)
    out["subgroups"] = rows
    # 10 interpretability
    Xa = p21.aug_features(c); F, phw = Xa[:, :-1], Xa[:, -1]; sc = StandardScaler().fit(F); A = sc.transform(F)
    rid = Ridge(alpha=1000).fit(A, -(phw - phw.mean()) / phw.std()); lr = LogisticRegression(C=0.05, max_iter=1000, random_state=42).fit(A, c.y_w)
    names = json.load(open(hz.X40_NAMES_PATH)); order = np.argsort(-np.abs(rid.coef_))[:10]
    out["interpret"] = {"top_ridge": [{"feature": names[j], "ridge_coef": round(float(rid.coef_[j]), 4), "P6_logistic_coef": round(float(lr.coef_[0][j]), 4)} for j in order],
                        "spearman_window_scores_OOF": round(float(spearmanr(res["GA"]["pw"][np.isin(c.pid_w, universe)], res["B0"]["pw"][np.isin(c.pid_w, universe)])[0]), 4),
                        "sign_agreement_top10": int(sum(np.sign(rid.coef_[j]) == np.sign(lr.coef_[0][j]) for j in order))}
    json.dump(out, open(os.path.join(OUT, "cv547_analysis.json"), "w"), indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    return out


# ------------------------------------------------------------------ stage: robust
def stage_robust(c, res):
    universe = c.all_pids; y = np.array([c.y[p] for p in universe]); Xaug = p21.aug_features(c); out = {}
    path = os.path.join(OUT, "robust_resplits.csv"); rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {r["seed"] for r in rows}
    for seed in hz.RESPLIT_SEEDS + hz.FRESH_RESPLIT_SEEDS:
        if seed in done: continue
        assign = hz.split_assignments(c, universe, resplit_seeds=[seed], canonical=False)[f"resplit_{seed}"]
        a = run_arm(c, "GA", universe, assign, 100 + seed, Xaug); b = run_arm(c, "B0", universe, assign, 100 + seed, Xaug)
        sc = {"GA": a["URM"], "B0": b["URM"]}; d = hz.summarize_delta(hz.boot_all(y, sc), "GA", "B0", y, sc)
        rows.append({"seed": seed, "set": "screening" if seed in hz.RESPLIT_SEEDS else "fresh", "M_GA": round(hz.m_of(y, a["URM"]), 4), "M_B0": round(hz.m_of(y, b["URM"]), 4), "dM": round(d["dM"], 4), "p": round(d["dM_p"], 3),
                     **{f"d{h}": round(d[f"d{h}"], 4) for h in H}})
        pd.DataFrame(rows).to_csv(path, index=False); print("  robust", rows[-1], flush=True)
    out["resplits"] = rows
    # per-seed pooler spread (canonical) + fixed ridge alpha sensitivity
    seeds = []
    for s in hz.SEEDS:
        r = run_arm(c, "GA", universe, c.canon, 0, Xaug, seeds=[s], pw=res["GA"]["pw"]); rb = run_arm(c, "B0", universe, c.canon, 0, Xaug, seeds=[s], pw=res["B0"]["pw"])
        seeds.append({"pooler_seed": s, "M_GA": round(hz.m_of(y, r["URM"]), 4), "M_B0": round(hz.m_of(y, rb["URM"]), 4)})
    out["per_seed"] = seeds
    sens = []
    for al in (10, 100, 1000):
        r = run_arm(c, "GA", universe, c.canon, 0, Xaug, configs=[(f"alpha={al}", p21.f_ridge(al))])
        sens.append({"ridge_alpha_fixed": al, "M": round(hz.m_of(y, r["URM"]), 4), **{f"{h}m": round(hz.fast_auc(y, r["URM"][:, a]), 4) for a, h in enumerate(H)}}); print("  sens", sens[-1], flush=True)
    out["alpha_sensitivity"] = sens
    json.dump(out, open(os.path.join(OUT, "robust.json"), "w"), indent=2, default=float); return out


# ------------------------------------------------------------------ stage: test
def stage_test(c):
    Xaug = p21.aug_features(c); ytest = np.array([c.y[p] for p in c.test_pids]); out = {}
    sc = {}
    for arm, X, cf in (("GA", Xaug, p21.candidates()["G-A"]), ("B0", c.X40, None)):
        s, alpha, chosen = p21.test_arm(c, X, cf); sc[arm] = s; out[arm] = {"alpha_fusion": float(alpha), "window_config": chosen}
    gate = abs(hz.m_of(ytest, sc["GA"]) - 0.8073) <= 0.001 and abs(hz.m_of(ytest, sc["B0"]) - 0.7582) <= 0.001
    print(f"  test reproduction gate (G-A 0.8073 / B0 0.7582): G-A {hz.m_of(ytest, sc['GA']):.4f}  B0 {hz.m_of(ytest, sc['B0']):.4f}  [{'PASS' if gate else 'FAIL'}]")
    out["reproduction_gate_pass"] = bool(gate)
    if not gate: json.dump(out, open(os.path.join(OUT, "test.json"), "w"), indent=2, default=float); return out
    tab, _ = metric_table(ytest, {"GA": sc["GA"], "B0": sc["B0"]}, ["GA", "B0"]); tab.to_csv(os.path.join(OUT, "test_metrics.csv"), index=False); out["table"] = tab.to_dict("records")
    out["M"] = {k: round(hz.m_of(ytest, v), 4) for k, v in sc.items()}
    d = hz.summarize_delta(hz.boot_all(ytest, sc), "GA", "B0", ytest, sc); out["paired_vs_B0"] = {k: round(float(v), 4) for k, v in d.items()}
    out["calibration_delivery"] = {arm: calib(ytest, sc[arm][:, 0])[0] for arm in sc}
    json.dump(out, open(os.path.join(OUT, "test.json"), "w"), indent=2, default=float); return out


# ------------------------------------------------------------------ stage: freeze
def stage_freeze(c, res):
    universe = c.all_pids; y = np.array([c.y[p] for p in universe]); Xaug = p21.aug_features(c); FD = os.path.join(OUT, "frozen_model"); os.makedirs(FD, exist_ok=True)
    # 1 window model: alpha by inner 4-fold CV over all 547 windows-by-patient, then fit + Platt
    tr_p = np.array(sorted(universe)); ytr = np.array([c.y[p] for p in tr_p]); inner_w = np.full(len(c.pid_w), -1, dtype=int)
    for k, (_, v) in enumerate(StratifiedKFold(4, shuffle=True, random_state=777).split(tr_p, ytr)):
        for p in tr_p[v]: inner_w[c.win_idx[p]] = k
    crit = {}
    for al in (10, 100, 1000):
        ip = np.zeros(len(c.pid_w)); f = p21.f_ridge(al)
        for k in range(4):
            va = inner_w == k; tr = (inner_w >= 0) & (inner_w != k); ip[va] = f(Xaug[tr], c.y_w[tr], Xaug[va])
        crit[al] = s1.prs_criterion(c, ip, list(tr_p))
    alpha_r = max(crit, key=crit.get)
    F, phw = Xaug[:, :-1], Xaug[:, -1]; scl = StandardScaler().fit(F); A = scl.transform(F)
    rid = Ridge(alpha=alpha_r).fit(A, -(phw - phw.mean()) / phw.std()); raw = rid.predict(A); mu, sd = raw.mean(), raw.std() + 1e-12
    pl = LogisticRegression(C=1e6, max_iter=1000).fit(((raw - mu) / sd)[:, None], c.y_w)
    win_model = {"ridge_alpha": alpha_r, "inner_cv_criterion": {str(k): round(v, 4) for k, v in crit.items()}, "feature_mean": scl.mean_.tolist(), "feature_scale": scl.scale_.tolist(),
                 "ridge_coef": rid.coef_.tolist(), "ridge_intercept": float(rid.intercept_), "platt_mu": float(mu), "platt_sd": float(sd), "platt_coef": float(pl.coef_[0][0]), "platt_intercept": float(pl.intercept_[0])}
    # 2 pooler: 3 seeds on the canonical-547 out-of-fold window scores
    pw = res["GA"]["pw"]; S = build_seqs(pw.astype(np.float32), c.pid_w, c.df, c.X40, c.col_idx, list(universe), c.y); X = make_inputs(S, False)
    itr, iva = carve_inner_validation(list(universe), c.y, seed=123)
    tri = torch.tensor([S.row[p] for p in itr]); vai = torch.tensor([S.row[p] for p in iva]); scorers = []
    for s in hz.SEEDS:
        sc_, ep, _ = fit_arm(S, X, tri, vai, "bce", s); scorers.append(sc_); torch.save(sc_.state_dict(), os.path.join(FD, f"pooler_seed{s}.pt"))
    # 3 parity + alpha (alpha from canonical-547 OOF sequences, in-sample parity, as in the Phase 18 freeze)
    order = list(universe); parx = np.array([c.parity[p] for p in order]); psc = StandardScaler().fit(parx.reshape(-1, 1))
    pclf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(psc.transform(parx.reshape(-1, 1)), y)
    par = dict(zip(order, pclf.predict_proba(psc.transform(parx.reshape(-1, 1)))[:, 1])); alpha_f = hz.select_alpha(order, res["GA"]["seq"], par, c.y)
    model = {"name": "G-A final (exploratory)", "trained_on": "all 547 patients", "window_model": win_model, "pooler_seeds": hz.SEEDS, "pooler_in_dim": 2, "pooler_hidden": 8,
             "parity": {"mean": float(psc.mean_[0]), "scale": float(psc.scale_[0]), "coef": float(pclf.coef_[0][0]), "intercept": float(pclf.intercept_[0])}, "fusion_alpha": float(alpha_f),
             "inputs": "40-D state-trajectory window vectors (chronological, locked feature definition), elapsed minutes per window, maternal parity"}
    json.dump(model, open(os.path.join(FD, "model.json"), "w"), indent=2)
    # 4 integrity: standalone scorer vs in-memory objects on every patient, + in-sample (optimistic) AUROC
    import scripts.phase22_ga_predict as pred
    frozen = pred.load(FD); maxd = 0.0; insample = np.zeros((len(order), 4)); ss = c.df["start_sample"].values
    tdel = hz.tdel_by_pid(c, order)
    for i, p in enumerate(order):
        idx = c.win_idx[p]; o = idx[np.argsort(ss[idx])]; Xw = c.X40[o].astype(np.float64); el = (ss[o] - ss[o][0]) / (4.0 * 60.0)
        run = pred.score_patient(frozen, Xw, el, c.parity[p])
        raw_i = rid.predict(scl.transform(Xw)); r_i = pl.predict_proba(((raw_i - mu) / sd)[:, None])[:, 1]
        z = np.mean([predict_all_prefixes(m, torch.tensor(r_i, dtype=torch.float32), torch.tensor(el, dtype=torch.float32), True, len(r_i)) for m in scorers], axis=0)
        ref = alpha_f * z + (1 - alpha_f) * par[p]; maxd = max(maxd, float(np.max(np.abs(run["risk"] - ref))))
        for a, h in enumerate(H): insample[i, a] = ref[eligible_prefix_length(tdel[p], h, len(ref)) - 1]
    integ = {"max_abs_diff_standalone_vs_in_memory": maxd, "pass": bool(maxd < 1e-5), "in_sample_auroc_OPTIMISTIC_not_a_performance_estimate": {f"{h}m": round(hz.fast_auc(y, insample[:, a]), 4) for a, h in enumerate(H)}}
    print("  freeze integrity", integ, flush=True); json.dump({"model": model, "integrity": integ}, open(os.path.join(OUT, "freeze.json"), "w"), indent=2, default=float)
    return integ


def main():
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--stage", choices=["cv547", "robust", "test", "freeze", "all"], default="all"); a = ap_.parse_args()
    c = hz.load_ctx(); res = stage_cv547(c)
    if a.stage in ("cv547", "all"):
        an = analyse_cv547(c, res); print("  cv547 M:", an["M"], "paired:", {k: an["paired_vs_B0"][k] for k in ("dM", "dM_p", "dM_ci_lo", "dM_ci_hi")})
    if a.stage in ("robust", "all"): stage_robust(c, res)
    if a.stage in ("test", "all"): stage_test(c)
    if a.stage in ("freeze", "all"): stage_freeze(c, res)


if __name__ == "__main__":
    main()
