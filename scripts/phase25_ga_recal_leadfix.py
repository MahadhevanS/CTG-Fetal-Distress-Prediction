"""
Phase 25 -- Candidate D2: Candidate D's window model + pooler + recalibration, but with fusion alpha chosen by a
lead-time-constrained selection instead of plain training-AUROC argmax (docs/phase25_ga_recal_leadfix_protocol.md).
"""
import os, sys, json
import numpy as np, pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import scripts.phase20_harness as hz
import scripts.phase20_stage1_candidates as s1
import scripts.phase21_graded_supervision as p21
import scripts.phase22_ga_final_evaluation as p22
import scripts.phase23_urm_v2_composite as p23
import scripts.phase24_ga_recal as p24

OUT = "results/phase25_ga_recal_leadfix"; os.makedirs(OUT, exist_ok=True)
FRESH = [501, 514, 528, 541, 557, 569, 583, 598, 612, 627]
H = hz.H
MIN_LEAD = 35.0


def select_alpha_lead_constrained(tr_pids, seq, par, y, tdel, min_lead=MIN_LEAD, grid=hz.ALPHA_GRID):
    """argmax training AUROC (in-sample, existing convention) subject to in-sample training median lead >= min_lead."""
    z, p, yy, w = [], [], [], []
    for pid in tr_pids:
        s = seq[pid]; T = len(s); z.extend(s.tolist()); p.extend([par[pid]] * T); yy.extend([y[pid]] * T); w.extend([1.0 / T] * T)
    z, p, yy, w = map(np.array, (z, p, yy, w))
    tr_y = np.array([y[pid] for pid in tr_pids])
    feasible = []
    for a in grid:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yy, a * z + (1 - a) * p, sample_weight=w)
        trs = np.array([a * seq[pid][-1] + (1 - a) * par[pid] for pid in tr_pids])
        thr = np.percentile(trs[tr_y == 1], 20)
        leads = []
        for pid in tr_pids:
            if y[pid] != 1: continue
            s = seq[pid]; sc = a * s + (1 - a) * par[pid]; idx = np.where(sc >= thr)[0]
            if len(idx): leads.append(tdel[pid][idx[0]])
        med_lead = float(np.median(leads)) if leads else 0.0
        feasible.append((a, auc, med_lead))
    ok = [f for f in feasible if f[2] >= min_lead]
    if ok:
        return max(ok, key=lambda f: f[1])[0]
    return max(feasible, key=lambda f: f[2])[0]   # fallback: none satisfy -> take the largest achievable lead


def urm_cv_recal_constrained(c, universe, assign, seq, tdel):
    order = list(universe); row = {p: i for i, p in enumerate(order)}; n = len(order)
    raw = np.zeros((n, 4)); cal = np.zeros((n, 4)); alphas = []
    lg = lambda s: np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
    from sklearn.linear_model import LogisticRegression
    for f in range(5):
        te = [p for p in order if assign[p] == f]; tr = [p for p in order if assign[p] != f]
        par = dict(zip(order, hz.fit_parity(np.array([c.parity[p] for p in tr]), np.array([c.y[p] for p in tr]), np.array([c.parity[p] for p in order]))))
        a = select_alpha_lead_constrained(tr, seq, par, c.y, tdel); alphas.append(a)
        z, yy, w = [], [], []
        for p in tr:
            s = a * seq[p] + (1 - a) * par[p]; z.extend(s.tolist()); yy.extend([c.y[p]] * len(s)); w.extend([1.0 / len(s)] * len(s))
        m = LogisticRegression(C=1e6, max_iter=1000).fit(lg(np.array(z))[:, None], np.array(yy), sample_weight=np.array(w))
        for p in te:
            for j, h in enumerate(H):
                k = hz.eligible_prefix_length(tdel[p], h, len(seq[p])) - 1; v = a * seq[p][k] + (1 - a) * par[p]
                raw[row[p], j] = v; cal[row[p], j] = m.predict_proba(lg(np.array([v]))[:, None])[0, 1]
    return raw, cal, alphas


def op_far_matched_D2(c, universe, assign, seq_D2, seq_B0, alphas_D2, target_far=p24.B0_TARGET_FAR):
    order = list(universe); n = len(order); y = np.array([c.y[p] for p in order]); tdel = hz.tdel_by_pid(c, order)
    out = {}
    for name in ("D2", "B0"):
        alerted = np.zeros(n, bool); lead = np.full(n, np.nan)
        for f in range(5):
            te = [p for p in order if assign[p] == f]; tr = [p for p in order if assign[p] != f]
            par = dict(zip(order, hz.fit_parity(np.array([c.parity[p] for p in tr]), np.array([c.y[p] for p in tr]), np.array([c.parity[p] for p in order]))))
            a_b0 = hz.select_alpha(tr, seq_B0, par, c.y)
            neg_max = np.array([np.max(a_b0 * seq_B0[p] + (1 - a_b0) * par[p]) for p in tr if c.y[p] == 0]); thr = float(np.quantile(neg_max, 1 - target_far))
            seq_use, a_use = (seq_D2, alphas_D2[f]) if name == "D2" else (seq_B0, a_b0)
            for p in te:
                i = order.index(p); idx = np.where(a_use * seq_use[p] + (1 - a_use) * par[p] >= thr)[0]
                if len(idx):
                    alerted[i] = True
                    if y[i] == 1: lead[i] = tdel[p][idx[0]]
        tp = int((alerted & (y == 1)).sum()); fp = int((alerted & (y == 0)).sum()); l = lead[(y == 1) & ~np.isnan(lead)]
        out[name] = {"sens": tp / (y == 1).sum(), "far": fp / (y == 0).sum(), "median_lead": float(np.median(l)) if len(l) else np.nan,
                     "ge10": float(np.mean(l >= 10)) if len(l) else np.nan, "ge20": float(np.mean(l >= 20)) if len(l) else np.nan, "ge30": float(np.mean(l >= 30)) if len(l) else np.nan}
    return out


def run_split(c, name, assign, off, Xaug):
    U = c.all_pids; y = np.array([c.y[p] for p in U]); tdel = hz.tdel_by_pid(c, U)
    pw_b0 = hz.oof_window_scores(c, c.X40, U, assign)
    pw_ga, _ = s1.select_and_predict(c, Xaug, U, assign, [("alpha=1000", p21.f_ridge(1000))], off)
    seq_D2, _ = hz.pooler_oof_sequences(c, pw_ga, U, assign, offset=off)
    seq_B0, _ = hz.pooler_oof_sequences(c, pw_b0, U, assign, offset=off)
    r_b0, _ = hz.urm_cv(c, U, assign, seq_B0, tdel)
    raw_D2, cal_D2, alphas = urm_cv_recal_constrained(c, U, assign, seq_D2, tdel)
    cs_D2, _ = p22.calib(y, cal_D2[:, 0]); cs_raw, _ = p22.calib(y, raw_D2[:, 0])
    ops_far = op_far_matched_D2(c, U, assign, seq_D2, seq_B0, alphas)
    ops_sens_D2 = p23.operational_point(c, U, assign, seq_D2); ops_sens_B0 = p23.operational_point(c, U, assign, seq_B0)
    row = {"split": name, "alphas": alphas, "M_D2": round(hz.m_of(y, cal_D2), 4), "M_B0": round(hz.m_of(y, r_b0["URM"]), 4),
           **{f"D2_{h}m": round(hz.fast_auc(y, cal_D2[:, a]), 4) for a, h in enumerate(H)}, **{f"B0_{h}m": round(hz.fast_auc(y, r_b0["URM"][:, a]), 4) for a, h in enumerate(H)},
           "D2_cal_slope": cs_D2["cal_slope"], "D2_brier": cs_D2["brier"], "D2_ece10": cs_D2["ece10"], "D2_raw_brier": cs_raw["brier"],
           **{f"opfar_D2_{k}": round(v, 4) for k, v in ops_far["D2"].items()}, **{f"opfar_B0_{k}": round(v, 4) for k, v in ops_far["B0"].items()},
           **{f"opsens_D2_{k}": round(v, 4) for k, v in ops_sens_D2.items()}, **{f"opsens_B0_{k}": round(v, 4) for k, v in ops_sens_B0.items()}}
    np.savez(os.path.join(OUT, f"scores_{name}.npz"), D2=cal_D2, D2_raw=raw_D2, B0=r_b0["URM"])
    return row


def verdict(c):
    df = pd.read_csv(os.path.join(OUT, "split_metrics.csv")); y = np.array([c.y[p] for p in c.all_pids])
    fresh = [f"resplit_{s}" for s in FRESH]; idx = df.set_index("split")
    dM = (idx["M_D2"] - idx["M_B0"]).loc[fresh]
    d1 = bool(dM.mean() >= 0.005 and (dM > 0).sum() >= 8)
    per_h = {h: float((idx[f"D2_{h}m"] - idx[f"B0_{h}m"]).loc[fresh].mean()) for h in H}; d2 = all(v >= -0.010 for v in per_h.values())
    def lead_row(sel):
        g = lambda k: (idx[f"opfar_D2_{k}"] - idx[f"opfar_B0_{k}"])
        return {"d_sens": float(g("sens").loc[sel].mean()), "d_median_lead": float(g("median_lead").loc[sel].mean()), "d_ge20": float(g("ge20").loc[sel].mean())}
    Lc = lead_row(["CANONICAL"]); Lf = lead_row(fresh)
    l_ok = lambda v: v["d_sens"] >= -0.02 and v["d_median_lead"] >= -2.5 and v["d_ge20"] >= -0.03
    L = l_ok(Lc) and l_ok(Lf)
    e_ok = lambda sel: bool(0.8 <= idx["D2_cal_slope"].loc[sel].mean() <= 1.25 and idx["D2_brier"].loc[sel].mean() <= idx["D2_raw_brier"].loc[sel].mean())
    E = e_ok(["CANONICAL"]) and e_ok(fresh)
    label = "EXPLORATORY SUPPORT" if (d1 and d2 and L and E) else "EXPLORATORY NO SUPPORT"
    sc_can = np.load(os.path.join(OUT, "scores_CANONICAL.npz")); bt = hz.boot_all(y, {"D2": sc_can["D2"], "B0": sc_can["B0"]})
    dcan = hz.summarize_delta(bt, "D2", "B0", y, {"D2": sc_can["D2"], "B0": sc_can["B0"]})
    boots = []
    for s in FRESH:
        sc = np.load(os.path.join(OUT, f"scores_resplit_{s}.npz")); b = hz.boot_all(y, {"D2": sc["D2"], "B0": sc["B0"]}); boots.append((b["D2"] - b["B0"]).mean(1))
    pooled = np.mean(boots, axis=0); p_fresh = float(min(1.0, 2 * min(np.mean(pooled <= 0), np.mean(pooled >= 0))))
    out = {"label": label, "D1": d1, "D2crit": d2, "L": L, "E": E, "fresh_mean_dM": round(float(dM.mean()), 4), "fresh_dM_all": [round(v, 4) for v in dM],
           "fresh_positive": int((dM > 0).sum()), "fresh_mean_per_horizon_delta": {f"{h}m": round(v, 4) for h, v in per_h.items()},
           "lead_canonical_opfar": {k: round(v, 4) for k, v in Lc.items()}, "lead_fresh_mean_opfar": {k: round(v, 4) for k, v in Lf.items()},
           "calib_slope_canonical": round(float(idx["D2_cal_slope"].loc["CANONICAL"]), 3), "calib_slope_fresh_mean": round(float(idx["D2_cal_slope"].loc[fresh].mean()), 3),
           "canonical": {k: round(float(v), 4) for k, v in dcan.items()}, "fresh_pooled_p": round(p_fresh, 4),
           "caveat": "No independent/external cohort available; all splits reuse the same 547 CTU-UHB patients."}
    json.dump(out, open(os.path.join(OUT, "verdict.json"), "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float)); return out


def main():
    c = hz.load_ctx(); Xaug = p21.aug_features(c)
    splits = hz.split_assignments(c, c.all_pids, resplit_seeds=FRESH, canonical=True)
    path = os.path.join(OUT, "split_metrics.csv"); rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {r["split"] for r in rows}
    for name, assign in splits.items():
        if name in done: continue
        off = 0 if name == "CANONICAL" else 100 + int(name.split("_")[1])
        row = run_split(c, name, assign, off, Xaug); rows.append(row); pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  {name:<13} M_D2 {row['M_D2']:.4f}  M_B0 {row['M_B0']:.4f}  dM {row['M_D2']-row['M_B0']:+.4f}  alphas {row['alphas']}  "
              f"OPfar sens D2/B0 {row['opfar_D2_sens']:.3f}/{row['opfar_B0_sens']:.3f}  lead D2/B0 {row['opfar_D2_median_lead']}/{row['opfar_B0_median_lead']}  "
              f"ge20 D2/B0 {row['opfar_D2_ge20']:.3f}/{row['opfar_B0_ge20']:.3f}  cal_slope {row['D2_cal_slope']:.2f}", flush=True)
    verdict(c)


if __name__ == "__main__":
    main()
