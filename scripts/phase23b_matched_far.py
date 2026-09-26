"""
Phase 23b -- POST-HOC descriptive check (not pre-registered, no decision weight): are the later warnings of G-A-type models just a different
operating point?  Compare B0, C (G-A fixed alpha), V2 at a MATCHED false-alert rate instead of a matched sensitivity: the per-fold threshold is the
value at which 62.5% (B0's canonical patient-level FAR) of the TRAINING negatives are ever alerted (max running fused score over the recording).
"""
import os, sys, json
import numpy as np, pandas as pd
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
import scripts.phase20_harness as hz
import scripts.phase20_stage1_candidates as s1
import scripts.phase21_graded_supervision as p21
import scripts.phase23_urm_v2_composite as p23

OUT = p23.OUT; TARGET_FAR = 0.625


def matched_far(c, U, assign, seq):
    order = list(U); n = len(order); y = np.array([c.y[p] for p in order]); tdel = hz.tdel_by_pid(c, order)
    alerted = np.zeros(n, bool); lead = np.full(n, np.nan)
    for f in range(5):
        te = [p for p in order if assign[p] == f]; tr = [p for p in order if assign[p] != f]
        par = dict(zip(order, hz.fit_parity(np.array([c.parity[p] for p in tr]), np.array([c.y[p] for p in tr]), np.array([c.parity[p] for p in order]))))
        a = hz.select_alpha(tr, seq, par, c.y)
        neg_max = np.array([np.max(a * seq[p] + (1 - a) * par[p]) for p in tr if c.y[p] == 0]); thr = float(np.quantile(neg_max, 1 - TARGET_FAR))
        for p in te:
            i = order.index(p); idx = np.where(a * seq[p] + (1 - a) * par[p] >= thr)[0]
            if len(idx):
                alerted[i] = True
                if y[i] == 1: lead[i] = tdel[p][idx[0]]
    tp = int((alerted & (y == 1)).sum()); fp = int((alerted & (y == 0)).sum()); l = lead[(y == 1) & ~np.isnan(lead)]
    return {"sens": tp / (y == 1).sum(), "far": fp / (y == 0).sum(), "median_lead": float(np.median(l)), "ge20": float(np.mean(l >= 20)), "ge30": float(np.mean(l >= 30))}


def main():
    c = hz.load_ctx(); q = p23.soft_labels(c); Xaug = p21.aug_features(c); U = c.all_pids
    splits = hz.split_assignments(c, U, resplit_seeds=p23.FRESH, canonical=True); rows = []
    for name, assign in splits.items():
        off = 0 if name == "CANONICAL" else 100 + int(name.split("_")[1])
        pw_b0 = hz.oof_window_scores(c, c.X40, U, assign); pw_ga, _ = s1.select_and_predict(c, Xaug, U, assign, [("alpha=1000", p21.f_ridge(1000))], off)
        seqs = {"B0": hz.pooler_oof_sequences(c, pw_b0, U, assign, offset=off)[0], "C": hz.pooler_oof_sequences(c, pw_ga, U, assign, offset=off)[0],
                "V2": p23.pooler_seq_soft(c, 0.5 * (pw_b0 + pw_ga), U, assign, q, off)}
        for arm, sq in seqs.items(): rows.append({"split": name, "arm": arm, **{k: round(v, 4) for k, v in matched_far(c, U, assign, sq).items()}})
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "matched_far.csv"), index=False); print("  done", name, flush=True)
    df = pd.DataFrame(rows); fr = df[df.split != "CANONICAL"]
    print("CANONICAL:\n", df[df.split == "CANONICAL"].to_string(index=False)); print("FRESH MEAN:\n", fr.groupby("arm")[["sens", "far", "median_lead", "ge20", "ge30"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
