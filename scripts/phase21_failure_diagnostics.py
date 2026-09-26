"""
Phase 21 -- why did the Phase 20 redesign find nothing?  EXPLORATORY diagnostics on the frozen-structure baseline B0
(development set D464, canonical folds).  Post-hoc outcome variables (pH, BDecf, delivery type, second stage) are used
ONLY to analyse the scores, never as model inputs.  No claim is made from this script alone.

D1 label noise     : AUROC of the baseline score against the pH<=7.15 label vs stricter / continuous outcomes
D2 data limitation : learning curve of the single-window PRS (patient AUROC at delivery) vs number of training patients
D3 who is missed   : AUROC / positive recall inside subgroups (delivery type, second stage at end, recording length)
D4 where is signal : per-descriptor univariate AUROC at the delivery window
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import scripts.phase20_harness as hz

OUT = "results/phase21_diagnostics"; os.makedirs(OUT, exist_ok=True)


def auc_ci(y, s, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); v = []
    for _ in range(B):
        i = rng.choice(n, n, replace=True)
        if 0 < y[i].sum() < n: v.append(hz.fast_auc(y[i], s[i]))
    return hz.fast_auc(y, s), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    c = hz.load_ctx(); pids = c.d464; y = np.array([c.y[p] for p in pids])
    meta = pd.read_csv(hz.METADATA_PATH).set_index("record_id")
    ph = np.array([meta.loc[int(p), "ph"] for p in pids]); bd = np.array([meta.loc[int(p), "bdecf"] for p in pids], dtype=float)
    dt = np.array([meta.loc[int(p), "deliv. type"] for p in pids])
    base = np.load(os.path.join(hz.OUT_DIR, "stage0_baseline_D464_scores.npz"))["CANONICAL"]
    s0, sM = base[:, 0], base.mean(1)
    res = {}

    # ---------------- D1 label noise ----------------
    print("=== D1: how much does the label threshold matter? (baseline URM score at delivery, D464 canonical) ===")
    rows = []
    def add(name, mask_pos, mask_all=None):
        m = np.ones(len(y), bool) if mask_all is None else mask_all
        yy = mask_pos[m].astype(int); a, lo, hi = auc_ci(yy, s0[m])
        rows.append({"target": name, "n_pos": int(yy.sum()), "n": int(m.sum()), "auroc": round(a, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}); print("  ", rows[-1])
    add("pH<=7.15 (the trained label)", ph <= 7.15)
    add("pH<=7.10", ph <= 7.10)
    add("pH<=7.05 (severe)", ph <= 7.05)
    add("BDecf>=8", np.nan_to_num(bd) >= 8, ~np.isnan(bd))
    add("BDecf>=12 (metabolic acidemia)", np.nan_to_num(bd) >= 12, ~np.isnan(bd))
    add("borderline positives (7.10<pH<=7.15) vs normal (pH>7.20)", (ph > 7.10) & (ph <= 7.15), ((ph > 7.10) & (ph <= 7.15)) | (ph > 7.20))
    add("deep positives (pH<=7.05) vs normal (pH>7.20)", ph <= 7.05, (ph <= 7.05) | (ph > 7.20))
    add("intermediate pH (7.15<pH<=7.20) vs normal (pH>7.20)", (ph > 7.15) & (ph <= 7.20), (ph > 7.15))
    rho, p = spearmanr(s0, -ph)
    print(f"   Spearman(score, -pH) over all patients: {rho:+.3f} (p={p:.1e})")
    n_pos = int((ph <= 7.15).sum()); n_border = int(((ph > 7.10) & (ph <= 7.15)).sum())
    print(f"   {n_border}/{n_pos} positives ({n_border / n_pos:.0%}) lie within 0.05 pH of the 7.15 cut-off")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "d1_label_threshold.csv"), index=False)
    res["D1"] = {"rows": rows, "spearman_score_vs_minus_pH": round(float(rho), 4), "share_positives_within_0.05_of_cutoff": round(n_border / n_pos, 3)}

    # ---------------- D2 learning curve ----------------
    print("\n=== D2: learning curve (single-window PRS, delivery-time patient AUROC, canonical D464 folds) ===")
    tdel = hz.tdel_by_pid(c, pids); rng = np.random.default_rng(1)
    lc = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        aucs = []
        for rep in range(8 if frac < 1 else 1):
            pred = np.zeros(len(c.pid_w))
            for f in range(5):
                tr_p = [p for p in pids if c.canon[p] != f]; te_p = [p for p in pids if c.canon[p] == f]
                if frac < 1:
                    yt = np.array([c.y[p] for p in tr_p]); keep = []
                    for cls in (0, 1):
                        ids = [p for p, yy in zip(tr_p, yt) if yy == cls]; keep += list(rng.choice(ids, max(2, int(round(frac * len(ids)))), replace=False))
                    tr_p = keep
                tr = np.isin(c.pid_w, tr_p); te = np.isin(c.pid_w, te_p)
                pred[te] = hz.fit_p6(c.X40[tr], c.y_w[tr], c.X40[te])
            sc = hz.horizon_scores(pids, hz.window_sequences(c, pred, pids), tdel, None, None)
            aucs.append(hz.m_of(y, sc))
        lc.append({"train_fraction": frac, "n_train_patients_per_fold": int(frac * 371), "M_mean": round(float(np.mean(aucs)), 4), "M_sd": round(float(np.std(aucs)), 4)}); print("  ", lc[-1])
    pd.DataFrame(lc).to_csv(os.path.join(OUT, "d2_learning_curve.csv"), index=False); res["D2"] = lc

    # ---------------- D3 subgroups ----------------
    print("\n=== D3: where is the score informative? (baseline URM at delivery, D464 canonical) ===")
    ds = {}
    import torch
    for sp in ("train", "val", "test"):
        d = torch.load(os.path.join(hz.DATA_DIR, f"{sp}_dataset.pt"), weights_only=False)
        for m_, w in zip(d["metadata"], d["w_is_second_stage"].numpy()):
            ds[(str(m_[0]), int(m_[1]))] = float(w)
    ss = c.df["start_sample"].values
    last2 = np.array([ds[(p, int(ss[c.win_idx[p][np.argmax(ss[c.win_idx[p]])]]))] for p in pids])   # last window in 2nd stage? (-1 unknown)
    nwin = np.array([len(c.win_idx[p]) for p in pids])
    rows = []
    def sub(name, mask):
        yy = y[mask]
        if yy.sum() < 8 or (len(yy) - yy.sum()) < 8:
            rows.append({"subgroup": name, "n": int(mask.sum()), "n_pos": int(yy.sum()), "auroc": None}); print("  ", rows[-1]); return
        a, lo, hi = auc_ci(yy, s0[mask]); rows.append({"subgroup": name, "n": int(mask.sum()), "n_pos": int(yy.sum()), "prevalence": round(float(yy.mean()), 3),
                                                        "auroc": round(a, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}); print("  ", rows[-1])
    sub("all D464", np.ones(len(y), bool))
    sub("vaginal delivery", dt == 1); sub("caesarean", dt == 2)
    sub("last window in 2nd stage", last2 == 1); sub("last window in 1st stage", last2 == 0); sub("2nd-stage status unknown", last2 < 0)
    for lo_, hi_, nm in [(0, 8, "short recording (<=8 windows)"), (9, 12, "medium (9-12 windows)"), (13, 99, "long (>=13 windows)")]:
        sub(nm, (nwin >= lo_) & (nwin <= hi_))
    par = np.array([c.parity[p] for p in pids]); sub("primiparous (parity 0)", par == 0); sub("multiparous", par >= 1)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "d3_subgroups.csv"), index=False); res["D3"] = rows

    # ---------------- D4 univariate signal ----------------
    print("\n=== D4: univariate AUROC of each raw descriptor at the delivery window (D464) ===")
    names = ["baseline", "stv", "ltv", "accels", "early", "late", "variable", "prolonged", "decel_depth", "decel_area", "decel_burden", "decel_longest",
             "baseline_slope", "variability_slope", "uc_count", "tachysystole", "uc_amp", "fhr_uc_lag", "fhr_uc_coupling"]
    last = np.array([c.win_idx[p][np.argmax(ss[c.win_idx[p]])] for p in pids]); rows = []
    for j, n in enumerate(names):
        a = hz.fast_auc(y, c.X40[last, j].astype(float)); rows.append({"descriptor": n, "auroc": round(a, 4), "signed": round(a - 0.5, 4)})
    df = pd.DataFrame(rows).sort_values("signed", key=abs, ascending=False); print(df.head(8).to_string(index=False))
    df.to_csv(os.path.join(OUT, "d4_univariate.csv"), index=False); res["D4_top"] = df.head(8).to_dict("records")
    json.dump(res, open(os.path.join(OUT, "diagnostics.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
