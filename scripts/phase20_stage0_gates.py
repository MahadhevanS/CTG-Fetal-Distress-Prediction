"""
Phase 20 Stage 0 -- gates G1, G2 and the noise floor G4 (docs/phase20_redesign_protocol.md section 5).
(G3, the descriptor-reproduction / causality gate for the C5 features, is a separate script.)

G1  P6 refit per canonical fold (547) reproduces p6_predictions.npz (max|diff| < 1e-6), incl. the train+val -> test model
G2  harness with baseline B0 on 547 canonical folds reproduces frozen URM / TAM within +-0.006 per horizon
G4  baseline B0 on the development set D464: canonical + 5 resplits (M and per-horizon), seed spread, SD of M
"""
import os, sys, json
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import scripts.phase20_harness as hz

FROZEN_URM = {0: 0.7335, 10: 0.6957, 20: 0.6675, 30: 0.6646}
FROZEN_TAM = {0: 0.7216, 10: 0.6683, 20: 0.6015, 30: 0.5976}
FROZEN_PRS = {0: 0.6872, 10: 0.6859, 20: 0.6238, 30: 0.5828}
TOL = 0.006
OUT = hz.OUT_DIR


def main():
    c = hz.load_ctx()
    gates = {}
    print("=" * 88 + "\n  PHASE 20 STAGE 0\n" + "=" * 88)

    # ---------------- G1 ----------------
    p6 = np.load(hz.P6_PRED_PATH, allow_pickle=True)
    assert (p6["patient_ids"] == c.pid_w).all(), "window order mismatch with p6_predictions.npz"
    pred = hz.oof_window_scores(c, c.X40, c.all_pids, c.canon)
    d_cv = float(np.max(np.abs(pred.astype(np.float32) - p6["pred_unweighted_cv"])))
    tv_mask = np.isin(c.pid_w, c.d464); te_mask = np.isin(c.pid_w, c.test_pids)
    p_test = hz.fit_p6(c.X40[tv_mask], c.y_w[tv_mask], c.X40[te_mask])
    d_te = float(np.max(np.abs(p_test.astype(np.float32) - p6["pred_unweighted_test"][te_mask])))
    g1 = d_cv < 1e-6 and d_te < 1e-6
    gates["G1"] = {"max_abs_diff_cv": d_cv, "max_abs_diff_test": d_te, "pass": bool(g1)}
    print(f"  G1 P6 refit vs p6_predictions.npz: max|diff| CV={d_cv:.2e}  test={d_te:.2e}  [{'PASS' if g1 else 'FAIL'}]")
    if not g1:
        json.dump(gates, open(os.path.join(OUT, "stage0_gates.json"), "w"), indent=2); print("  STOP (G1)"); return

    # ---------------- G2 ----------------
    log = []
    tdel = hz.tdel_by_pid(c, c.all_pids)
    seq, S = hz.pooler_oof_sequences(c, pred, c.all_pids, c.canon, offset=0, log=log)
    res, alphas = hz.urm_cv(c, c.all_pids, c.canon, seq, tdel)
    y547 = np.array([c.y[p] for p in c.all_pids])
    prs_seq = hz.window_sequences(c, pred, c.all_pids)
    prs = hz.horizon_scores(c.all_pids, prs_seq, tdel, None, None)
    ok = True; rows = {}
    for a, h in enumerate(hz.H):
        u = hz.fast_auc(y547, res["URM"][:, a]); t = hz.fast_auc(y547, res["CTG"][:, a]); r = hz.fast_auc(y547, prs[:, a])
        g_u, g_t, g_r = abs(u - FROZEN_URM[h]) <= TOL, abs(t - FROZEN_TAM[h]) <= TOL, abs(r - FROZEN_PRS[h]) <= 0.001
        ok &= g_u and g_t and g_r
        rows[h] = {"URM_harness": round(u, 4), "URM_frozen": FROZEN_URM[h], "TAM_harness": round(t, 4), "TAM_frozen": FROZEN_TAM[h], "PRS": round(r, 4)}
        print(f"  G2 h={h:>2}m: URM {u:.4f} vs {FROZEN_URM[h]:.4f} [{'PASS' if g_u else 'FAIL'}] | "
              f"TAM {t:.4f} vs {FROZEN_TAM[h]:.4f} [{'PASS' if g_t else 'FAIL'}] | PRS {r:.4f} vs {FROZEN_PRS[h]:.4f} [{'PASS' if g_r else 'FAIL'}]")
    gates["G2"] = {"per_horizon": rows, "alphas": [round(a, 2) for a in alphas], "pass": bool(ok),
                   "epoch_cap_hits": int(sum(1 for l in log if l["epochs"] >= 200)), "n_trainings": len(log)}
    print(f"  selected alpha per fold: {[round(a, 2) for a in alphas]}   epoch-cap hits {gates['G2']['epoch_cap_hits']}/{len(log)}")
    if not ok:
        json.dump(gates, open(os.path.join(OUT, "stage0_gates.json"), "w"), indent=2); print("  STOP (G2)"); return
    print("  G1 and G2 PASSED")

    # ---------------- G4: baseline on D464 ----------------
    y464 = np.array([c.y[p] for p in c.d464]); tdel464 = hz.tdel_by_pid(c, c.d464)
    splits = hz.split_assignments(c, c.d464)
    rows = []; saved = {}
    for name, assign in splits.items():
        off = 0 if name == "CANONICAL" else 100 + int(name.split("_")[1])
        pw = hz.oof_window_scores(c, c.X40, c.d464, assign)
        slog = []
        if name == "CANONICAL":
            per_seed = {s: hz.pooler_oof_sequences(c, pw, c.d464, assign, seeds=[s], offset=off, log=slog)[0] for s in hz.SEEDS}
            seq464 = {p: np.mean([per_seed[s][p] for s in hz.SEEDS], axis=0) for p in c.d464}
            for s in hz.SEEDS:
                r_s, _ = hz.urm_cv(c, c.d464, assign, per_seed[s], tdel464)
                rows.append({"split": "CANONICAL_seed%d" % s, "M": round(hz.m_of(y464, r_s["URM"]), 4),
                             **{f"URM_{h}m": round(v, 4) for h, v in zip(hz.H, hz.per_h(y464, r_s["URM"]))}})
        else:
            seq464, _ = hz.pooler_oof_sequences(c, pw, c.d464, assign, offset=off, log=slog)
        r, al = hz.urm_cv(c, c.d464, assign, seq464, tdel464)
        prs_r = hz.horizon_scores(c.d464, hz.window_sequences(c, pw, c.d464), tdel464, None, None)
        rec = {"split": name, "M": round(hz.m_of(y464, r["URM"]), 4), **{f"URM_{h}m": round(v, 4) for h, v in zip(hz.H, hz.per_h(y464, r["URM"]))},
               "M_TAM": round(hz.m_of(y464, r["CTG"]), 4), "M_PRS": round(hz.m_of(y464, prs_r), 4), "M_parity": round(hz.m_of(y464, r["PARITY"]), 4),
               "alpha_mean": round(float(np.mean(al)), 3), "cap_hits": int(sum(1 for l in slog if l["epochs"] >= 200))}
        rows.append(rec); print("  D464", rec)
        saved[name] = r["URM"]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "stage0_baseline_D464.csv"), index=False)
    np.savez(os.path.join(OUT, "stage0_baseline_D464_scores.npz"), pids=np.array(c.d464), **saved)
    main_rows = df[df.split.isin(["CANONICAL"] + [f"resplit_{s}" for s in hz.RESPLIT_SEEDS])]
    seed_rows = df[df.split.str.startswith("CANONICAL_seed")]
    gates["G4"] = {"M_canonical": float(main_rows.iloc[0].M), "M_mean_6_splits": float(main_rows.M.mean()), "M_sd_6_splits": float(main_rows.M.std(ddof=1)),
                   "M_min": float(main_rows.M.min()), "M_max": float(main_rows.M.max()),
                   "seed_M_sd_canonical": float(seed_rows.M.std(ddof=1)), "seed_M_values": [float(v) for v in seed_rows.M]}
    print(f"  G4 noise floor: M mean {gates['G4']['M_mean_6_splits']:.4f}  SD across 6 splits {gates['G4']['M_sd_6_splits']:.4f} "
          f"(range {gates['G4']['M_min']:.4f}-{gates['G4']['M_max']:.4f});  seed SD {gates['G4']['seed_M_sd_canonical']:.4f}")
    json.dump(gates, open(os.path.join(OUT, "stage0_gates.json"), "w"), indent=2)
    print("Saved -> results/phase20_redesign/")


if __name__ == "__main__":
    main()
