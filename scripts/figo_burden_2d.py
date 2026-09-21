"""
FINAL FHRMA-ONLY MEASUREMENT STUDY: is persistence x severity a better
representation of expert deceleration burden than either alone?

    python scripts/figo_burden_2d.py

NOT a prediction study. No pH, no BDecf, no Apgar, no early-warning label, no
CTU-UHB target, no outcome-driven threshold search. FHRMA carries no outcome,
so nothing here CAN be tuned toward prediction.

THE REFERENCE PROBLEM, STATED BEFORE ANY NUMBER
------------------------------------------------
FHRMA annotates deceleration SPANS. It does not annotate "burden". So every
"expert burden" quantity in this study is DERIVED by applying one of our own
summaries to the expert's spans, and that choice is not neutral:

  * against an expert SEVERITY reference (deceleration area), our
    `area_per_contraction` is favoured by construction -- it is the same
    functional form computed on different events;
  * against an expert PERSISTENCE reference (longest run of consecutive
    contractions carrying a deceleration), `max_consecutive_uc_with_decel`
    is favoured for the same reason.

Reporting against only one would manufacture the winner. Both are therefore
reported side by side throughout, and any claim that survives both is the
only kind this study can support.

WHY NO SCALAR COMBINATION IS INVENTED
--------------------------------------
Phase 3 of the brief forbids it, correctly: an arbitrary weighting is a free
parameter and would make the 2D arm win by construction. Two combination-free
instruments are used instead:

  PARETO DOMINANCE -- for a pair of epochs where one is >= the other on BOTH
    dimensions, the 2D representation makes an unambiguous ordering claim
    with no weights at all. Concordance on those pairs is assumption-free.
    Its coverage (what fraction of pairs are comparable) is reported too,
    because a representation that orders 30% of pairs perfectly and abstains
    on the rest is not obviously better than one that orders all of them
    moderately.

  LEAVE-ONE-RECORD-OUT FIT -- a monotone combination whose weights are fitted
    on the OTHER recordings and applied to the held-out one. This gives the
    best case a weighted combination could achieve, without letting a record
    influence its own score.
"""

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "scripts"))

from figo_burden_ordering import collect                          # noqa: E402

OUT = os.path.join(BASE, "results", "figo_burden")
SEV = "area_per_contraction"          # severity dimension
PER = "max_consecutive_uc_with_decel"  # persistence dimension
REFS = [("severity ref (expert area/contraction)", f"exp_{SEV}"),
        ("persistence ref (expert max-run)", f"exp_{PER}")]
MIN_EPOCHS = 4                        # per record, for within-record ordering


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def rank(v):
    return pd.Series(v).rank(pct=True).values


# --------------------------------------------------------------------------
def loo_scores(R: pd.DataFrame, ref: str) -> np.ndarray:
    """
    Leave-one-record-out monotone combination of the two dimensions.

    Weights are fitted by linear regression on the PERCENTILE RANKS of the
    two measures against the percentile rank of the expert reference, using
    every recording except the one being scored. Ranks make it monotone and
    scale-free; LOO stops a recording contributing to its own score.
    """
    out = np.full(len(R), np.nan)
    recs = R.record.values
    A = rank(R[f"ours_{SEV}"].values)
    B = rank(R[f"ours_{PER}"].values)
    y = rank(R[ref].values)
    X = np.column_stack([A, B])
    for rec in pd.unique(recs):
        te = recs == rec
        tr = ~te
        if tr.sum() < 20:
            continue
        m = LinearRegression().fit(X[tr], y[tr])
        out[te] = m.predict(X[te])
    return out


def pareto_within_record(R: pd.DataFrame, ref: str) -> Tuple[float, float, int]:
    """
    Concordance and coverage of Pareto-dominant pairs, within records.

    Returns (median per-record concordance, median coverage, n records).
    No weighting is involved: epoch i dominates j only if it is >= on both
    dimensions and > on at least one.
    """
    concs, covs = [], []
    for rec, g in R.groupby("record"):
        if len(g) < MIN_EPOCHS:
            continue
        a = g[f"ours_{SEV}"].values
        b = g[f"ours_{PER}"].values
        e = g[ref].values
        tot = comp = ok = 0
        for i in range(len(g)):
            for j in range(len(g)):
                if i >= j:
                    continue
                tot += 1
                if e[i] == e[j]:
                    continue
                dom_i = (a[i] >= a[j] and b[i] >= b[j]) and (a[i] > a[j] or b[i] > b[j])
                dom_j = (a[j] >= a[i] and b[j] >= b[i]) and (a[j] > a[i] or b[j] > b[i])
                if not (dom_i or dom_j):
                    continue
                comp += 1
                if dom_i and e[i] > e[j]:
                    ok += 1
                elif dom_j and e[j] > e[i]:
                    ok += 1
        if comp >= 3:
            concs.append(ok / comp)
            covs.append(comp / max(tot, 1))
    return (float(np.median(concs)) if concs else np.nan,
            float(np.median(covs)) if covs else np.nan, len(concs))


def within_record_rho(R: pd.DataFrame, score: np.ndarray, ref: str) -> Dict:
    rhos = []
    for rec, g in R.groupby("record"):
        if len(g) < MIN_EPOCHS:
            continue
        s = score[g.index.values]
        e = g[ref].values
        if np.isnan(s).any() or len(np.unique(s)) < 2 or len(np.unique(e)) < 2:
            continue
        r = spearmanr(s, e).statistic
        if np.isfinite(r):
            rhos.append(float(r))
    rhos = np.array(rhos)
    if len(rhos) == 0:
        return dict(median=np.nan, q1=np.nan, q3=np.nan, n=0, nonpos=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan)
    rng = np.random.default_rng(0)
    bs = [np.median(rng.choice(rhos, len(rhos), replace=True)) for _ in range(2000)]
    return dict(median=float(np.median(rhos)),
                q1=float(np.percentile(rhos, 25)),
                q3=float(np.percentile(rhos, 75)),
                n=int(len(rhos)), nonpos=float((rhos <= 0).mean()),
                ci_lo=float(np.percentile(bs, 2.5)),
                ci_hi=float(np.percentile(bs, 97.5)))


# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    R = collect().reset_index(drop=True)
    res: Dict = {}
    print(f"{len(R)} epochs from {R.record.nunique()} FHRMA recordings")
    print(f"records with >= {MIN_EPOCHS} epochs: "
          f"{int((R.groupby('record').size() >= MIN_EPOCHS).sum())}")

    # ------------------------------------------------------------------ 1
    hdr("PHASE 1 -- REPRODUCE AND FREEZE")
    prev = {SEV: dict(rho=0.400, wkappa=0.545, auroc10=0.982),
            PER: dict(rho=0.667, wkappa=0.346, auroc10=0.778)}
    ok = True
    for m in (SEV, PER):
        o, x = R[f"ours_{m}"].values, R[f"exp_{m}"].values
        rhos = []
        for rec, g in R.groupby("record"):
            if len(g) < 3:
                continue
            a, b = g[f"ours_{m}"].values, g[f"exp_{m}"].values
            if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
                continue
            r = spearmanr(a, b).statistic
            if np.isfinite(r):
                rhos.append(r)
        med = float(np.median(rhos))
        lab = (x >= np.quantile(x, 0.90)).astype(int)
        au = float(roc_auc_score(lab, o))
        d1, d2 = abs(med - prev[m]["rho"]), abs(au - prev[m]["auroc10"])
        flag = "OK" if (d1 < 0.02 and d2 < 0.02) else "** DIFFERS **"
        if flag != "OK":
            ok = False
        print(f"  {m:34s} within-rho {med:.3f} (was {prev[m]['rho']:.3f})  "
              f"top-decile AUROC {au:.3f} (was {prev[m]['auroc10']:.3f})  {flag}")
    if not ok:
        print("\n  reproduction DIFFERS materially -- stopping per the brief.")
        return
    print("  reproduction matches; same epochs, same preprocessing, detector untouched.")

    # ------------------------------------------------------------------ 2
    hdr("PHASE 2 -- ARE THE TWO DIMENSIONS COMPLEMENTARY?")
    a, b = R[f"ours_{SEV}"].values, R[f"ours_{PER}"].values
    print(f"  Spearman(severity, persistence) among OUR measures : "
          f"{spearmanr(a, b).statistic:+.3f}")
    print(f"  Spearman(severity, persistence) among EXPERT values : "
          f"{spearmanr(R[f'exp_{SEV}'], R[f'exp_{PER}']).statistic:+.3f}")
    print("\n  marginal and PARTIAL rank association with each expert reference")
    print("  (partial = residualise both against the other dimension's ranks)\n")
    print(f"  {'reference':38s}{'sev':>8s}{'per':>8s}{'sev|per':>10s}{'per|sev':>10s}")
    comp = {}
    for lab, ref in REFS:
        y = rank(R[ref].values)
        ra, rb = rank(a), rank(b)
        s_m = spearmanr(ra, y).statistic
        p_m = spearmanr(rb, y).statistic

        def partial(u, v, y_):
            ru = u - LinearRegression().fit(v.reshape(-1, 1), u).predict(v.reshape(-1, 1))
            ry = y_ - LinearRegression().fit(v.reshape(-1, 1), y_).predict(v.reshape(-1, 1))
            return spearmanr(ru, ry).statistic
        s_p = partial(ra, rb, y)
        p_p = partial(rb, ra, y)
        comp[lab] = dict(sev=float(s_m), per=float(p_m),
                         sev_given_per=float(s_p), per_given_sev=float(p_p))
        print(f"  {lab:38s}{s_m:8.3f}{p_m:8.3f}{s_p:10.3f}{p_p:10.3f}")
    res["phase2_complementarity"] = comp
    print("\n  A dimension adds information iff its PARTIAL association stays")
    print("  clearly non-zero after the other is accounted for.")

    # ------------------------------------------------------------------ 3
    hdr("PHASE 3 -- WITHIN-RECORD ORDERING")
    res["phase3"] = {}
    for lab, ref in REFS:
        print(f"\n  --- {lab} ---")
        print(f"  {'representation':34s}{'median rho':>12s}{'IQR':>16s}"
              f"{'95% CI':>16s}{'recs':>6s}{'rho<=0':>8s}")
        for nm, sc in ((f"{SEV} alone", rank(a)),
                       (f"{PER} alone", rank(b)),
                       ("2D, LOO-fitted combination", loo_scores(R, ref))):
            st = within_record_rho(R, sc, ref)
            res["phase3"].setdefault(lab, {})[nm] = st
            print(f"  {nm:34s}{st['median']:12.3f}"
                  f"   [{st['q1']:+.2f},{st['q3']:+.2f}]"
                  f"   [{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}]"
                  f"{st['n']:6d}{100*st['nonpos']:7.0f}%")
        pc, cov, nrec = pareto_within_record(R, ref)
        res["phase3"][lab]["pareto"] = dict(concordance=pc, coverage=cov, n=nrec)
        print(f"  {'2D, Pareto dominance (no weights)':34s}"
              f"{pc:12.3f}   concordance on {100*cov:.0f}% of pairs, {nrec} records")

    # ------------------------------------------------------------------ 4
    hdr("PHASE 4 -- SEVERITY TIERS (expert quartiles: low/moderate/high/extreme)")
    res["phase4"] = {}
    for lab, ref in REFS:
        x = R[ref].values
        try:
            tx = pd.qcut(x, 4, labels=False, duplicates="drop").astype(int)
        except ValueError:
            continue
        print(f"\n  --- {lab} ---")
        print(f"  {'representation':34s}{'exact':>8s}{'+/-1':>8s}{'wkappa':>9s}"
              f"{'low':>7s}{'mod':>7s}{'high':>7s}{'extr':>7s}")
        for nm, sc in ((f"{SEV} alone", rank(a)),
                       (f"{PER} alone", rank(b)),
                       ("2D, LOO-fitted", loo_scores(R, ref))):
            good = ~np.isnan(sc)
            try:
                to = pd.qcut(sc[good], 4, labels=False, duplicates="drop").astype(int)
            except ValueError:
                continue
            t = tx[good]
            ex = float((t == to).mean())
            nr = float((np.abs(t - to) <= 1).mean())
            wk = float(cohen_kappa_score(t, to, weights="quadratic"))
            per_tier = [float((to[t == k] == k).mean()) if (t == k).any() else np.nan
                        for k in range(4)]
            res["phase4"].setdefault(lab, {})[nm] = dict(
                exact=ex, within1=nr, wkappa=wk, per_tier=per_tier)
            print(f"  {nm:34s}{ex:8.3f}{nr:8.3f}{wk:9.3f}" +
                  "".join(f"{v:7.2f}" for v in per_tier))
        print("    per-tier columns are recall within each expert quartile.")
        print("    MID-RANGE (mod/high) is the region the Suspicious criterion needs.")

    # ------------------------------------------------------------------ 5
    hdr("PHASE 5 -- EXTREME-BURDEN DISCRIMINATION")
    res["phase5"] = {}
    for lab, ref in REFS:
        x = R[ref].values
        print(f"\n  --- {lab} ---")
        print(f"  {'representation':34s}{'top 25%':>10s}{'top 10%':>10s}")
        for nm, sc in ((f"{SEV} alone", rank(a)),
                       (f"{PER} alone", rank(b)),
                       ("2D, LOO-fitted", loo_scores(R, ref))):
            good = ~np.isnan(sc)
            row = {}
            for q, tag in ((0.75, "top 25%"), (0.90, "top 10%")):
                y = (x[good] >= np.quantile(x[good], q)).astype(int)
                row[tag] = float(roc_auc_score(y, sc[good])) if len(set(y)) > 1 else np.nan
            res["phase5"].setdefault(lab, {})[nm] = row
            print(f"  {nm:34s}{row['top 25%']:10.3f}{row['top 10%']:10.3f}")

    # ------------------------------------------------------------------ 6
    hdr("PHASE 6 -- FAILURE-MODE STRUCTURE OF THE 2D SPACE")
    ref = f"exp_{SEV}"
    hi_a = a >= np.median(a[a > 0]) if (a > 0).any() else a > 0
    hi_b = b >= 2
    hi_e = R[ref].values >= np.quantile(R[ref].values, 0.75)
    quad = {
        "A  persistence HIGH, expert burden LOW": hi_b & ~hi_e,
        "B  severity HIGH, persistence LOW": hi_a & ~hi_b,
        "C  BOTH high": hi_a & hi_b,
        "D  BOTH low, expert burden HIGH": (~hi_a) & (~hi_b) & hi_e,
    }
    print(f"  thresholds: severity >= median of non-zero ({np.median(a[a>0]):.0f} bpm*s),")
    print(f"              persistence >= 2 consecutive contractions,")
    print(f"              expert burden HIGH = top quartile\n")
    print(f"  {'quadrant':40s}{'n':>6s}{'%':>7s}{'uc':>7s}{'exp burden':>12s}")
    res["phase6"] = {}
    for nm, msk in quad.items():
        n = int(msk.sum())
        res["phase6"][nm] = dict(n=n, frac=float(msk.mean()),
                                 mean_uc=float(R.n_uc[msk].mean()) if n else np.nan,
                                 mean_exp=float(R[ref][msk].mean()) if n else np.nan)
        print(f"  {nm:40s}{n:6d}{100*msk.mean():6.1f}%"
              f"{R.n_uc[msk].mean():7.2f}{R[ref][msk].mean():12.1f}")

    # ------------------------------------------------------------------ 7
    hdr("PHASE 7 -- CAN A RULE BE DEFINED WITHOUT OUTCOME TUNING?")
    print("  Candidate regions, with thresholds taken ONLY from the expert")
    print("  distribution (its own quartiles), never from any CTU-UHB target.\n")
    xr = R[ref].values
    t_sev = float(np.quantile(R[f"ours_{SEV}"].values, 0.75))
    print(f"  severity cut = our 75th percentile ({t_sev:.0f} bpm*s/contraction)")
    print(f"  persistence cut = 2 consecutive contractions (physiological:")
    print(f"    'repetitive' minimally means two successive contractions respond)\n")
    hi_e75 = xr >= np.quantile(xr, 0.75)
    rules = {
        "high persistence AND high severity": (b >= 2) & (a >= t_sev),
        "high persistence OR high severity": (b >= 2) | (a >= t_sev),
        "persistence >= 2 alone": b >= 2,
        "severity >= p75 alone": a >= t_sev,
        "current v3 rule (>50% of contractions)":
            R.ours_frac_contractions_with_decel.values > 0.5,
    }
    print(f"  {'rule':42s}{'fires':>8s}{'sens':>7s}{'prec':>7s}{'F1':>7s}{'kappa':>7s}")
    res["phase7"] = {}
    for nm, m_ in rules.items():
        tp = int((m_ & hi_e75).sum())
        fp = int((m_ & ~hi_e75).sum())
        fn = int((~m_ & hi_e75).sum())
        se = tp / max(tp + fn, 1)
        pr = tp / max(tp + fp, 1)
        f1 = 2 * se * pr / max(se + pr, 1e-9)
        kp = float(cohen_kappa_score(hi_e75.astype(int), m_.astype(int)))
        res["phase7"][nm] = dict(fires=float(m_.mean()), sens=se, prec=pr, f1=f1,
                                 kappa=kp)
        print(f"  {nm:42s}{100*m_.mean():7.1f}%{se:7.3f}{pr:7.3f}{f1:7.3f}{kp:7.3f}")
    print("\n  target here is the EXPERT's top-quartile burden -- a measurement")
    print("  reference, not a clinical state and not an outcome.")

    json.dump(res, open(os.path.join(OUT, "burden_2d.json"), "w"),
              indent=2, default=float)
    print(f"\nwrote {OUT}/burden_2d.json")


if __name__ == "__main__":
    main()
