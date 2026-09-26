"""
Phase 24 -- Candidate D: G-A window (fixed ridge alpha=1000) + standard pooler + recalibration (docs/phase24_ga_recal_protocol.md).
No independent cohort exists in this environment; evaluated on canonical + 10 fresh resplits with seeds never used in Phases 20-23.
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

OUT = "results/phase24_ga_recal"; os.makedirs(OUT, exist_ok=True)
FRESH = [301, 313, 327, 341, 359, 372, 386, 401, 417, 433]
H = hz.H
B0_TARGET_FAR = 0.625  # canonical B0 patient-level false-alert rate at 80% target sensitivity (Phase 18/22)


def op_far_matched(c, universe, assign, seq_D, seq_B0, target_far=B0_TARGET_FAR):
    """Threshold fixed FROM B0 (so both systems share one operating point), applied to both D and B0."""
    order = list(universe); n = len(order); y = np.array([c.y[p] for p in order]); tdel = hz.tdel_by_pid(c, order)
    out = {}
    for name, seq in (("D", seq_D), ("B0", seq_B0)):
        alerted = np.zeros(n, bool); lead = np.full(n, np.nan)
        for f in range(5):
            te = [p for p in order if assign[p] == f]; tr = [p for p in order if assign[p] != f]
            par = dict(zip(order, hz.fit_parity(np.array([c.parity[p] for p in tr]), np.array([c.y[p] for p in tr]), np.array([c.parity[p] for p in order]))))
            a = hz.select_alpha(tr, seq_B0, par, c.y)   # threshold derived from B0's OWN fused score on B0's own training negatives
            neg_max = np.array([np.max(a * seq_B0[p] + (1 - a) * par[p]) for p in tr if c.y[p] == 0]); thr = float(np.quantile(neg_max, 1 - target_far))
            a_use = hz.select_alpha(tr, seq, par, c.y)
            for p in te:
                i = order.index(p); idx = np.where(a_use * seq[p] + (1 - a_use) * par[p] >= thr)[0]
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
    seq_D, _ = hz.pooler_oof_sequences(c, pw_ga, U, assign, offset=off)
    seq_B0, _ = hz.pooler_oof_sequences(c, pw_b0, U, assign, offset=off)
    r_b0, _ = hz.urm_cv(c, U, assign, seq_B0, tdel)
    raw_D, cal_D = p23.urm_cv_recal(c, U, assign, seq_D, tdel)
    cs_D, _ = p22.calib(y, cal_D[:, 0]); cs_Draw, _ = p22.calib(y, raw_D[:, 0])
    ops_far = op_far_matched(c, U, assign, seq_D, seq_B0)
    ops_sens_D = p23.operational_point(c, U, assign, seq_D); ops_sens_B0 = p23.operational_point(c, U, assign, seq_B0)
    row = {"split": name, "M_D": round(hz.m_of(y, cal_D), 4), "M_B0": round(hz.m_of(y, r_b0["URM"]), 4),
           **{f"D_{h}m": round(hz.fast_auc(y, cal_D[:, a]), 4) for a, h in enumerate(H)}, **{f"B0_{h}m": round(hz.fast_auc(y, r_b0["URM"][:, a]), 4) for a, h in enumerate(H)},
           "D_cal_slope": cs_D["cal_slope"], "D_brier": cs_D["brier"], "D_ece10": cs_D["ece10"], "D_raw_brier": cs_Draw["brier"],
           **{f"opfar_D_{k}": round(v, 4) for k, v in ops_far["D"].items()}, **{f"opfar_B0_{k}": round(v, 4) for k, v in ops_far["B0"].items()},
           **{f"opsens_D_{k}": round(v, 4) for k, v in ops_sens_D.items()}, **{f"opsens_B0_{k}": round(v, 4) for k, v in ops_sens_B0.items()}}
    np.savez(os.path.join(OUT, f"scores_{name}.npz"), D=cal_D, D_raw=raw_D, B0=r_b0["URM"])
    return row


def verdict(c):
    df = pd.read_csv(os.path.join(OUT, "split_metrics.csv")); y = np.array([c.y[p] for p in c.all_pids])
    fresh = [f"resplit_{s}" for s in FRESH]; idx = df.set_index("split")
    dM = (idx["M_D"] - idx["M_B0"]).loc[fresh]
    d1 = bool(dM.mean() >= 0.005 and (dM > 0).sum() >= 8)
    per_h = {h: float((idx[f"D_{h}m"] - idx[f"B0_{h}m"]).loc[fresh].mean()) for h in H}; d2 = all(v >= -0.010 for v in per_h.values())
    def lead_row(sel):
        g = lambda k: (idx[f"opfar_D_{k}"] - idx[f"opfar_B0_{k}"])
        return {"d_sens": float(g("sens").loc[sel].mean()), "d_median_lead": float(g("median_lead").loc[sel].mean()), "d_ge20": float(g("ge20").loc[sel].mean())}
    Lc = lead_row(["CANONICAL"]); Lf = lead_row(fresh)
    l_ok = lambda v: v["d_sens"] >= -0.02 and v["d_median_lead"] >= -2.5 and v["d_ge20"] >= -0.03
    L = l_ok(Lc) and l_ok(Lf)
    e_ok = lambda sel: bool(0.8 <= idx["D_cal_slope"].loc[sel].mean() <= 1.25 and idx["D_brier"].loc[sel].mean() <= idx["D_raw_brier"].loc[sel].mean())
    E = e_ok(["CANONICAL"]) and e_ok(fresh)
    label = "EXPLORATORY SUPPORT" if (d1 and d2 and L and E) else "EXPLORATORY NO SUPPORT"
    sc_can = np.load(os.path.join(OUT, "scores_CANONICAL.npz")); bt = hz.boot_all(y, {"D": sc_can["D"], "B0": sc_can["B0"]})
    dcan = hz.summarize_delta(bt, "D", "B0", y, {"D": sc_can["D"], "B0": sc_can["B0"]})
    boots = []
    for s in FRESH:
        sc = np.load(os.path.join(OUT, f"scores_resplit_{s}.npz")); b = hz.boot_all(y, {"D": sc["D"], "B0": sc["B0"]}); boots.append((b["D"] - b["B0"]).mean(1))
    pooled = np.mean(boots, axis=0); p_fresh = float(min(1.0, 2 * min(np.mean(pooled <= 0), np.mean(pooled >= 0))))
    out = {"label": label, "D1": d1, "D2": d2, "L": L, "E": E, "fresh_mean_dM": round(float(dM.mean()), 4), "fresh_dM_all": [round(v, 4) for v in dM],
           "fresh_positive": int((dM > 0).sum()), "fresh_mean_per_horizon_delta": {f"{h}m": round(v, 4) for h, v in per_h.items()},
           "lead_canonical_opfar": {k: round(v, 4) for k, v in Lc.items()}, "lead_fresh_mean_opfar": {k: round(v, 4) for k, v in Lf.items()},
           "calib_slope_canonical": round(float(idx["D_cal_slope"].loc["CANONICAL"]), 3), "calib_slope_fresh_mean": round(float(idx["D_cal_slope"].loc[fresh].mean()), 3),
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
        print(f"  {name:<13} M_D {row['M_D']:.4f}  M_B0 {row['M_B0']:.4f}  dM {row['M_D']-row['M_B0']:+.4f}  "
              f"OPfar sens D/B0 {row['opfar_D_sens']:.3f}/{row['opfar_B0_sens']:.3f}  lead D/B0 {row['opfar_D_median_lead']}/{row['opfar_B0_median_lead']}  "
              f"cal_slope {row['D_cal_slope']:.2f}", flush=True)
    verdict(c)


if __name__ == "__main__":
    main()
