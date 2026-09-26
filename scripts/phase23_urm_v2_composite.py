"""
Phase 23 -- URM-v2 composite (docs/phase23_urm_v2_composite_protocol.md, frozen before this file was written).

Arms per split (all 547 patients, canonical + 10 fresh resplits):
  B0    locked P6 window + standard pooler                                   (baseline)
  C     G-A window (ridge alpha=1000 fixed) + standard pooler
  B     window = 1/2 (P6 + G-A fixed)  + standard pooler
  A     P6 window + soft-label pooler
  V2    window = 1/2 (P6 + G-A fixed) + soft-label pooler                     (B + C + A)
  V2E   V2 + per-fold recalibration of the fused score                        (final composite)
"""
import os, sys, json
import numpy as np, pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import scripts.phase20_harness as hz
import scripts.phase20_stage1_candidates as s1
import scripts.phase21_graded_supervision as p21
import scripts.phase22_ga_final_evaluation as p22
from scripts.phase16_temporal_attention_model import carve_inner_validation
from scripts.phase19_pooler_experiment import build_seqs, make_inputs, pooled_all_prefix, fit_arm

OUT = "results/phase23_urm_v2"; os.makedirs(OUT, exist_ok=True)
FRESH = [121, 132, 143, 154, 165, 176, 187, 198, 209, 220]
ARMS = ["B0", "C", "B", "A", "V2", "V2E"]
SOFT_SCALE = 0.03
H = hz.H


# ------------------------------------------------------------------ components
def soft_labels(c):
    meta = pd.read_csv(hz.METADATA_PATH).set_index("record_id")
    return {p: float(1.0 / (1.0 + np.exp(-(7.15 - float(meta.loc[int(p), "ph"])) / SOFT_SCALE))) for p in c.all_pids}


def pooler_seq_soft(c, pw, universe, assign, q, offset, seeds=hz.SEEDS):
    """Identical to hz.pooler_oof_sequences except the cross-entropy target is the soft label q (also in early stopping)."""
    S = build_seqs(pw.astype(np.float32), c.pid_w, c.df, c.X40, c.col_idx, list(universe), c.y)
    S.Y = torch.tensor([q[p] for p in S.pids], dtype=torch.float32)
    X = make_inputs(S, False); Z = torch.zeros(S.R.shape)
    for f in range(5):
        te = [p for p in universe if assign[p] == f]; tr_all = [p for p in universe if assign[p] != f]
        inner_tr, inner_val = carve_inner_validation(tr_all, c.y, seed=42 + f + offset)
        tr_idx = torch.tensor([S.row[p] for p in inner_tr]); val_idx = torch.tensor([S.row[p] for p in inner_val]); te_rows = torch.tensor([S.row[p] for p in te])
        for s in seeds:
            sc, _, _ = fit_arm(S, X, tr_idx, val_idx, "bce", s)
            with torch.no_grad():
                z = pooled_all_prefix(sc, X, S.R, S.L)
            Z[te_rows] += z[te_rows] / len(seeds)
    return {p: Z[S.row[p], :int(S.L[S.row[p]])].numpy().astype(np.float64) for p in universe}


def urm_cv_recal(c, universe, assign, seq, tdel):
    """Raw fused horizon scores and per-fold recalibrated ones (weighted pooled causal-prefix recalibration)."""
    order = list(universe); row = {p: i for i, p in enumerate(order)}; n = len(order)
    raw = np.zeros((n, 4)); cal = np.zeros((n, 4)); lg = lambda s: np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
    for f in range(5):
        te = [p for p in order if assign[p] == f]; tr = [p for p in order if assign[p] != f]
        par = dict(zip(order, hz.fit_parity(np.array([c.parity[p] for p in tr]), np.array([c.y[p] for p in tr]), np.array([c.parity[p] for p in order]))))
        a = hz.select_alpha(tr, seq, par, c.y)
        z, yy, w = [], [], []
        for p in tr:
            s = a * seq[p] + (1 - a) * par[p]; z.extend(s.tolist()); yy.extend([c.y[p]] * len(s)); w.extend([1.0 / len(s)] * len(s))
        m = LogisticRegression(C=1e6, max_iter=1000).fit(lg(np.array(z))[:, None], np.array(yy), sample_weight=np.array(w))
        for p in te:
            for j, h in enumerate(H):
                k = hz.eligible_prefix_length(tdel[p], h, len(seq[p])) - 1; v = a * seq[p][k] + (1 - a) * par[p]
                raw[row[p], j] = v; cal[row[p], j] = m.predict_proba(lg(np.array([v]))[:, None])[0, 1]
    return raw, cal


def operational_point(c, universe, assign, seq, target=0.8):
    order = list(universe); n = len(order); y = np.array([c.y[p] for p in order]); tdel = hz.tdel_by_pid(c, order)
    alerted = np.zeros(n, bool); lead = np.full(n, np.nan)
    for f in range(5):
        te = [p for p in order if assign[p] == f]; tr = [p for p in order if assign[p] != f]
        par = dict(zip(order, hz.fit_parity(np.array([c.parity[p] for p in tr]), np.array([c.y[p] for p in tr]), np.array([c.parity[p] for p in order]))))
        a = hz.select_alpha(tr, seq, par, c.y)
        trs = np.array([a * seq[p][-1] + (1 - a) * par[p] for p in tr]); ytr = np.array([c.y[p] for p in tr]); thr = float(np.percentile(trs[ytr == 1], (1 - target) * 100))
        for p in te:
            i = order.index(p); idx = np.where(a * seq[p] + (1 - a) * par[p] >= thr)[0]
            if len(idx):
                alerted[i] = True
                if y[i] == 1: lead[i] = tdel[p][idx[0]]
    tp = int((alerted & (y == 1)).sum()); fp = int((alerted & (y == 0)).sum()); l = lead[(y == 1) & ~np.isnan(lead)]
    return {"sens": tp / (y == 1).sum(), "far": fp / (y == 0).sum(), "ppv": tp / max(tp + fp, 1), "median_lead": float(np.median(l)), "ge10": float(np.mean(l >= 10)),
            "ge20": float(np.mean(l >= 20)), "ge30": float(np.mean(l >= 30))}


# ------------------------------------------------------------------ one split
def run_split(c, name, assign, off, q, Xaug):
    U = c.all_pids; y = np.array([c.y[p] for p in U]); tdel = hz.tdel_by_pid(c, U)
    pw_b0 = hz.oof_window_scores(c, c.X40, U, assign)
    pw_ga, _ = s1.select_and_predict(c, Xaug, U, assign, [("alpha=1000", p21.f_ridge(1000))], off)
    pw_ens = 0.5 * (pw_b0 + pw_ga)
    seqs = {"B0": hz.pooler_oof_sequences(c, pw_b0, U, assign, offset=off)[0], "C": hz.pooler_oof_sequences(c, pw_ga, U, assign, offset=off)[0],
            "B": hz.pooler_oof_sequences(c, pw_ens, U, assign, offset=off)[0], "A": pooler_seq_soft(c, pw_b0, U, assign, q, off),
            "V2": pooler_seq_soft(c, pw_ens, U, assign, q, off)}
    scores, rows = {}, []
    for arm in ["B0", "C", "B", "A", "V2"]:
        r, al = hz.urm_cv(c, U, assign, seqs[arm], tdel); scores[arm] = r["URM"]
    raw, cal = urm_cv_recal(c, U, assign, seqs["V2"], tdel); scores["V2E"] = cal
    for arm in ARMS:
        s = scores[arm]; cs, _ = p22.calib(y, s[:, 0]); seq_for_ops = seqs["V2" if arm == "V2E" else arm]; ops = operational_point(c, U, assign, seq_for_ops)
        rows.append({"split": name, "arm": arm, "M": round(hz.m_of(y, s), 4), **{f"auc_{h}m": round(hz.fast_auc(y, s[:, a]), 4) for a, h in enumerate(H)},
                     "brier": cs["brier"], "cal_slope": cs["cal_slope"], "cal_intercept": cs["cal_intercept"], "ece10": cs["ece10"], **{f"op_{k}": round(v, 4) for k, v in ops.items()}})
    np.savez(os.path.join(OUT, f"scores_{name}.npz"), **scores)
    return rows


# ------------------------------------------------------------------ verdict
def verdict(c):
    df = pd.read_csv(os.path.join(OUT, "split_metrics.csv")); y = np.array([c.y[p] for p in c.all_pids]); out = {}
    fresh = [f"resplit_{s}" for s in FRESH]
    piv = lambda col: df.pivot(index="split", columns="arm", values=col)
    M = piv("M"); dM = (M["V2E"] - M["B0"]).loc[fresh]
    d1 = bool(dM.mean() >= 0.005 and (dM > 0).sum() >= 8)
    per_h = {h: float((piv(f"auc_{h}m")["V2E"] - piv(f"auc_{h}m")["B0"]).loc[fresh].mean()) for h in H}; d2 = all(v >= -0.010 for v in per_h.values())
    def lead_ok(vals):
        v = vals; return {"d_ge20": v["ge20"], "d_median_lead": v["med"], "d_sens": v["sens"], "d_far": v["far"],
                          "ok": bool(v["ge20"] >= -0.03 and v["med"] >= -2.5 and v["sens"] >= -0.03 and v["far"] <= 0.03)}
    def dlt(split_sel):
        g = lambda col: (piv(col)["V2E"] - piv(col)["B0"]).loc[split_sel]
        return {"ge20": float(np.mean(g("op_ge20"))), "med": float(np.mean(g("op_median_lead"))), "sens": float(np.mean(g("op_sens"))), "far": float(np.mean(g("op_far")))}
    L_can = lead_ok(dlt(["CANONICAL"])); L_fresh = lead_ok(dlt(fresh)); L = L_can["ok"] and L_fresh["ok"]
    label = "EXPLORATORY SUPPORT" if (d1 and d2 and L) else "EXPLORATORY NO SUPPORT"
    # component E
    def e_ok(sel):
        sl = piv("cal_slope")["V2E"].loc[sel].mean(); br = (piv("brier")["V2E"] - piv("brier")["V2"]).loc[sel].mean()
        dauc = np.mean([(piv(f"auc_{h}m")["V2E"] - piv(f"auc_{h}m")["V2"]).loc[sel].mean() for h in H])
        return {"slope": float(sl), "d_brier_vs_V2": float(br), "d_auc_vs_V2": float(dauc), "ok": bool(0.8 <= sl <= 1.25 and br <= 0 and dauc >= -0.005)}
    E_can, E_fresh = e_ok(["CANONICAL"]), e_ok(fresh); E_works = E_can["ok"] and E_fresh["ok"]
    # statistics: canonical paired bootstrap + pooled fresh
    sc_can = np.load(os.path.join(OUT, "scores_CANONICAL.npz")); bt = hz.boot_all(y, {"V2E": sc_can["V2E"], "B0": sc_can["B0"]})
    dcan = hz.summarize_delta(bt, "V2E", "B0", y, {"V2E": sc_can["V2E"], "B0": sc_can["B0"]})
    boots = []
    for nm in fresh:
        s = np.load(os.path.join(OUT, f"scores_{nm}.npz")); b = hz.boot_all(y, {"V2E": s["V2E"], "B0": s["B0"]}); boots.append((b["V2E"] - b["B0"]).mean(1))
    pooled = np.mean(boots, axis=0); p_fresh = float(min(1.0, 2 * min(np.mean(pooled <= 0), np.mean(pooled >= 0))))
    out = {"label": label, "D1": d1, "D2": d2, "L": L, "fresh_dM": [round(float(v), 4) for v in dM], "fresh_mean_dM": round(float(dM.mean()), 4), "fresh_positive": int((dM > 0).sum()),
           "fresh_mean_per_horizon_delta": {f"{h}m": round(v, 4) for h, v in per_h.items()}, "lead_canonical": L_can, "lead_fresh_mean": L_fresh,
           "E_component_works": E_works, "E_canonical": E_can, "E_fresh_mean": E_fresh,
           "canonical": {k: round(float(v), 4) for k, v in dcan.items()}, "fresh_pooled_p": round(p_fresh, 4)}
    json.dump(out, open(os.path.join(OUT, "verdict.json"), "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float)); return out


def main():
    c = hz.load_ctx(); q = soft_labels(c); Xaug = p21.aug_features(c)
    splits = hz.split_assignments(c, c.all_pids, resplit_seeds=FRESH, canonical=True)
    path = os.path.join(OUT, "split_metrics.csv"); rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {r["split"] for r in rows}
    for name, assign in splits.items():
        if name in done: continue
        off = 0 if name == "CANONICAL" else 100 + int(name.split("_")[1])
        new = run_split(c, name, assign, off, q, Xaug); rows += new; pd.DataFrame(rows).to_csv(path, index=False)
        m = {r["arm"]: r["M"] for r in new}; print(f"  {name:<13} M: " + "  ".join(f"{a} {m[a]:.4f}" for a in ARMS), flush=True)
    verdict(c)


if __name__ == "__main__":
    main()
